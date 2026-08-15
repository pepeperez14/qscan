"""Detección de anomalías en las series de precios.

No predice nada: busca datos rotos. Es la capa con mejor relación valor/riesgo de
todo el sistema, porque un solo dato malo contamina los ~35 indicadores del
activo y lo puede colar en el top-40 con un +400% que nunca ocurrió.

Todos los detectores son verificables mirando el gráfico, que es exactamente lo
que los hace seguros: cuando marcan algo, se puede comprobar si tenían razón.

Detectores:
  stale            cotización congelada (mismo cierre N días seguidos)
  split_sin_ajustar salto que coincide con un ratio de split típico
  tick_malo        salto extremo que se revierte al día siguiente
  volumen_cero     sesiones sin negociación
  ohlc_plano       open=high=low=close (no hubo mercado real)
  serie_muerta     el feed dejó de actualizarse
  duplicada        otra serie con retornos idénticos (feed duplicado)
  ilíquida         volumen mediano por debajo del umbral
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Ratios de split habituales, en ambas direcciones. Sólo se buscan ratios grandes
# (2:1 en adelante) y esto es deliberado. Un split 5:4 deja un salto del 20%, que
# en una cripto o en un valor pequeño es un martes cualquiera: no hay forma fiable
# de distinguirlo mirando sólo el precio. Se prefiere no detectar esos antes que
# marcar como dato roto un movimiento real.
SPLIT_RATIOS = [2, 5 / 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
SEVERITY = {
    "split_sin_ajustar": 5.0, "tick_malo": 4.0, "ohlc_plano": 3.0,
    "stale": 2.5, "serie_muerta": 2.0, "duplicada": 2.0,
    "volumen_cero": 1.0, "iliquida": 1.0,
}
QUARANTINE_AT = 4.0     # a partir de aquí el activo no entra en el scoring


def _max_run(mask: pd.DataFrame) -> pd.Series:
    """Racha máxima de True por columna, sin bucles por símbolo."""
    m = mask.fillna(False).to_numpy()
    out = np.zeros(m.shape[1], dtype=int)
    cur = np.zeros(m.shape[1], dtype=int)
    for row in m:
        cur = np.where(row, cur + 1, 0)
        out = np.maximum(out, cur)
    return pd.Series(out, index=mask.columns)


def detect(px: dict[str, pd.DataFrame], min_dollar_vol: float = 1e6,
           stale_days: int = 8, lookback: int = 504) -> pd.DataFrame:
    """Devuelve un informe por símbolo con banderas y severidad."""
    c = px["close"].tail(lookback)
    o, h, l = (px[k].tail(lookback).reindex_like(c) for k in ("open", "high", "low"))
    v = px["volume"].tail(lookback).reindex_like(c)

    ret = c.pct_change()
    logret = np.log(c / c.shift(1))
    sigma = logret.rolling(120, min_periods=40).std()

    flags: dict[str, pd.Series] = {}

    # --- cotización congelada ---------------------------------------------
    unchanged = c.diff().abs().le(1e-12) & c.notna()
    runs = _max_run(unchanged)
    flags["stale"] = runs >= stale_days

    # --- split sin ajustar -------------------------------------------------
    # Un split deja un salto grande que NO se revierte. El ratio observado no es
    # exactamente 1/k: incorpora también el movimiento real de esa sesión, así
    # que la tolerancia tiene que escalar con la volatilidad del activo. Con un
    # margen fijo del 2% se escapan casi todos en cripto y en small caps.
    ratio = (c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    daily_sd = logret.std().fillna(0.02)
    tol = (2.5 * daily_sd).clip(0.02, 0.10)

    near_split = pd.DataFrame(False, index=c.index, columns=c.columns)
    for k in SPLIT_RATIOS:
        for r in (k, 1 / k):
            near_split |= ((ratio - r).abs() / r).lt(tol, axis=1)
    big = ratio.sub(1).abs() > 0.25
    # el salto sigue en pie una semana después: un tick malo se deshace, un split no
    persists = (c.shift(-5) / c).sub(1).abs() < 0.25
    flags["split_sin_ajustar"] = (near_split & big & persists).sum() > 0

    # --- tick malo ---------------------------------------------------------
    extreme = logret.abs() > 6 * sigma
    reversal = (logret.shift(-1) * logret) < 0
    magnitude = logret.shift(-1).abs() > 0.6 * logret.abs()
    flags["tick_malo"] = (extreme & reversal & magnitude & (logret.abs() > 0.2)).sum() > 0

    # --- microestructura ---------------------------------------------------
    flags["volumen_cero"] = v.le(0).mean() > 0.10
    flat = (o.eq(h) & h.eq(l) & l.eq(c) & c.notna())
    flags["ohlc_plano"] = flat.mean() > 0.20

    # --- feed muerto -------------------------------------------------------
    last = c.apply(lambda s: s.last_valid_index())
    ref = c.index.max()
    flags["serie_muerta"] = (ref - pd.to_datetime(last)).dt.days > 10

    # --- liquidez ----------------------------------------------------------
    dv = (c * v).median()
    flags["iliquida"] = dv.fillna(0) < min_dollar_vol

    # --- series duplicadas -------------------------------------------------
    flags["duplicada"] = _duplicates(ret)

    rep = pd.DataFrame(flags).fillna(False)
    rep["severidad"] = sum(rep[k].astype(float) * w for k, w in SEVERITY.items()
                           if k in rep.columns)
    rep["cuarentena"] = rep["severidad"] >= QUARANTINE_AT
    rep["dollar_vol_mediano"] = dv
    rep["ultima_cotizacion"] = pd.to_datetime(last)
    rep.index.name = "symbol"
    return rep.sort_values("severidad", ascending=False)


def _duplicates(ret: pd.DataFrame) -> pd.Series:
    """Feeds duplicados por hash de los retornos: O(n) en vez de O(n²)."""
    tail = ret.tail(250).round(6)
    digest = {}
    for col in tail.columns:
        s = tail[col].dropna()
        if len(s) < 100:
            continue
        digest.setdefault(hashlib.md5(s.to_numpy().tobytes()).hexdigest(), []).append(col)
    dup = {c: False for c in ret.columns}
    for cols in digest.values():
        if len(cols) > 1:
            # se marcan todas menos la primera alfabéticamente
            for c in sorted(cols)[1:]:
                dup[c] = True
    return pd.Series(dup).reindex(ret.columns).fillna(False)


def summary(rep: pd.DataFrame) -> pd.DataFrame:
    """Recuento por tipo de anomalía, para el informe."""
    cols = [c for c in SEVERITY if c in rep.columns]
    s = rep[cols].sum().sort_values(ascending=False)
    return pd.DataFrame({"anomalia": s.index, "activos": s.to_numpy().astype(int),
                         "severidad": [SEVERITY[k] for k in s.index]})


def apply_quarantine(panel: pd.DataFrame, rep: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Saca del panel los activos en cuarentena antes de puntuar."""
    bad = set(rep.index[rep["cuarentena"]])
    if not bad:
        return panel, 0
    out = panel[~panel.symbol.isin(bad)]
    return out.reset_index(drop=True), len(bad)
