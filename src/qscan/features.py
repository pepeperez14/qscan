"""Panel de características: el análisis completo de la evolución de cada cotización.

Se calculan las series históricas completas de cada característica y después se
muestrean sólo en las fechas pedidas. Esto es lo que permite validar con años de
historia sin que la memoria explote: 9.000 activos x 2.000 sesiones x 35 features
en float32 serían ~2,5 GB; muestreando en fechas de rebalanceo baja a decenas de MB.

Cada fila responde a la pregunta "¿cómo se ha comportado este activo hasta hoy?"
en cinco dimensiones: tendencia, momento, volatilidad, estructura de precio y
comportamiento relativo al mercado.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import indicators as ta

log = logging.getLogger(__name__)

# horizonte -> ventana en sesiones sobre la que se mide el retorno futuro
HORIZON_DAYS = {"corto": 10, "medio": 63, "largo": 252}


# Grupos sin volumen negociado publicado. Los pares de divisas y los índices
# cotizan en Yahoo con volumen 0 o vacío: no es que sean ilíquidos, es que ese
# dato no existe para ellos. Aplicarles un filtro de volumen en dólares los
# elimina a todos, que es exactamente lo que pasaba: fx entraba con 30 series y
# salía con 0. Es un error de categoría, no de umbral.
SIN_VOLUMEN = {"fx", "index"}


def _dollar_volume(close: pd.DataFrame, vol: pd.DataFrame, n: int = 63) -> pd.DataFrame:
    return (close * vol).rolling(n, min_periods=n // 3).median()


def build_feature_history(px: dict[str, pd.DataFrame], bench: pd.Series
                          ) -> dict[str, pd.DataFrame]:
    """Devuelve {nombre_feature: DataFrame ancho fecha x símbolo}."""
    c, h, l, v = px["close"], px["high"], px["low"], px["volume"]
    f: dict[str, pd.DataFrame] = {}

    # --- tendencia ---------------------------------------------------------
    f["ma20_dist"] = ta.dist_to_ma(c, 20)
    f["ma50_dist"] = ta.dist_to_ma(c, 50)
    f["ma200_dist"] = ta.dist_to_ma(c, 200)
    f["ma50_over_200"] = ta.sma(c, 50) / ta.sma(c, 200) - 1.0
    for n, tag in ((63, "3m"), (252, "12m")):
        s, r2 = ta.slope_r2(c, n)
        f[f"slope_{tag}"], f[f"trendfit_{tag}"] = s, r2
    f["adx14"] = ta.adx(h, l, c, 14)

    # --- momento -----------------------------------------------------------
    for n, tag in ((5, "1w"), (21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")):
        f[f"ret_{tag}"] = ta.roc(c, n)
    f["mom_12_1"] = ta.momentum_skip(c, 252, 21)
    f["rsi14"] = ta.rsi(c, 14)
    f["rsi2"] = ta.rsi(c, 2)          # sobreventa de muy corto plazo
    f["macd_hist"] = ta.macd_hist(c)

    # --- volatilidad y riesgo ---------------------------------------------
    f["vol_21"] = ta.realized_vol(c, 21)
    f["vol_252"] = ta.realized_vol(c, 252)
    f["vol_ratio"] = f["vol_21"] / f["vol_252"].replace(0, np.nan)
    f["atr_pct"] = ta.atr(h, l, c, 14) / c
    f["dd_from_high"] = ta.drawdown(c, 252)
    f["ulcer_6m"] = ta.ulcer(c, 126)
    f["sharpe_12m"] = f["ret_12m"] / f["vol_252"].replace(0, np.nan)
    f["sortino_6m"] = _sortino(c, 126)

    # --- estructura de precio ---------------------------------------------
    f["donch_20"] = ta.donchian_pos(c, 20)
    f["donch_252"] = ta.donchian_pos(c, 252)
    f["boll_z"] = ta.bollinger_z(c, 20)
    f["gap_ratio"] = _gap_ratio(px["open"], c)
    f["vol_surge"] = ta.volume_surge(v, 20)
    f["dollar_vol"] = _dollar_volume(c, v)

    # --- relativo al mercado ----------------------------------------------
    beta, alpha = ta.beta_alpha(c, bench, 126)
    f["beta_6m"], f["alpha_6m"] = beta, alpha
    f["rs_3m"] = ta.relative_strength(c, bench, 63)
    f["rs_12m"] = ta.relative_strength(c, bench, 252)

    return f


def _sortino(px: pd.DataFrame, n: int) -> pd.DataFrame:
    r = np.log(px / px.shift(1))
    downside = r.where(r < 0, 0.0).pow(2).rolling(n, min_periods=n // 2).mean().pow(0.5)
    return (r.rolling(n, min_periods=n // 2).mean() * np.sqrt(252)) / \
        downside.replace(0, np.nan)


def _gap_ratio(op: pd.DataFrame, c: pd.DataFrame, n: int = 63) -> pd.DataFrame:
    """Proporción de la variación que ocurre fuera de sesión: mide riesgo de hueco."""
    gap = (op / c.shift(1) - 1.0).abs()
    total = (c / c.shift(1) - 1.0).abs()
    return gap.rolling(n, min_periods=n // 3).sum() / \
        total.rolling(n, min_periods=n // 3).sum().replace(0, np.nan)


def sample_panel(fh: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                 close: pd.DataFrame, groups: pd.Series,
                 min_dollar_vol: float = 1e6) -> pd.DataFrame:
    """Extrae el panel largo (date, symbol, features...) en las fechas dadas."""
    frames = []
    for d in dates:
        if d not in close.index:
            continue
        row = pd.DataFrame({k: df.loc[d] for k, df in fh.items()})
        row["close"] = close.loc[d]
        row = row[row["close"].notna()]
        exento = row.index.map(lambda s: groups.get(s) in SIN_VOLUMEN)
        row = row[(row["dollar_vol"].fillna(0) >= min_dollar_vol) | exento]
        if row.empty:
            continue
        row.insert(0, "date", d)
        row.index.name = "symbol"
        frames.append(row.reset_index())
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["group"] = panel.symbol.map(groups)
    num = panel.select_dtypes("number").columns
    panel[num] = panel[num].astype("float32")
    return panel


def forward_returns(close: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Retornos futuros por horizonte. SÓLO para validación, nunca para puntuar."""
    out = panel[["date", "symbol"]].copy()
    for name, n in HORIZON_DAYS.items():
        fwd = close.shift(-n) / close - 1.0
        stacked = fwd.stack().rename(f"fwd_{name}")
        stacked.index.names = ["date", "symbol"]
        out = out.merge(stacked.reset_index(), on=["date", "symbol"], how="left")
    return out


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Última sesión real de cada periodo: evita pedir fechas que no cotizaron."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.resample(freq).last().dropna().values)
