"""Puntuación transversal en tres horizontes.

Principios de diseño:

1. La comparación es SIEMPRE transversal y dentro del mismo grupo de activo. Un
   Sharpe de 0,8 es excelente en materias primas y mediocre en cripto; puntuar
   todo contra la misma distribución sólo produce un ranking de clases de activo.

2. Se winsoriza antes de estandarizar. Con miles de activos siempre hay colas
   absurdas, y una sola z de 40 se come el peso de todo lo demás.

3. Se exige un mínimo de componentes disponibles. Un activo con 3 de 9 features
   calculadas puede salir el primero por accidente aritmético; se descarta.

4. El score es un ranking, no una probabilidad. Un percentil 99 significa "el
   mejor posicionado del universo hoy según estas reglas", no "va a subir".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (feature, peso). El signo del peso codifica la dirección deseada.
WEIGHTS: dict[str, dict[str, float]] = {
    # 1-3 semanas: ruptura con confirmación de volumen, corregida por reversión
    "corto": {
        "donch_20": 0.16, "macd_hist": 0.14, "ma20_dist": 0.10, "adx14": 0.10,
        "vol_surge": 0.08, "rs_3m": 0.08,
        "rsi2": -0.14, "ret_1w": -0.10, "boll_z": -0.06, "vol_ratio": -0.04,
    },
    # 1-6 meses: tendencia intermedia y fuerza relativa
    "medio": {
        "ret_3m": 0.18, "rs_3m": 0.18, "ma50_over_200": 0.14, "trendfit_3m": 0.10,
        "dd_from_high": 0.09, "adx14": 0.06, "macd_hist": 0.05, "alpha_6m": 0.05,
        "vol_ratio": -0.09, "ulcer_6m": -0.06,
    },
    # 6-24 meses: momento largo, calidad de la tendencia y riesgo asumido
    "largo": {
        "mom_12_1": 0.22, "trendfit_12m": 0.15, "slope_12m": 0.14, "rs_12m": 0.13,
        "sharpe_12m": 0.11, "ma200_dist": 0.05, "sortino_6m": 0.05,
        "vol_252": -0.10, "ulcer_6m": -0.05,
    },
}

MIN_COVERAGE = 0.7      # fracción de componentes que debe estar disponible
MIN_PEERS = 20          # mínimo de activos en el grupo para que el z-score signifique algo


def _winsorized_z(s: pd.Series, lo: float = 0.025, hi: float = 0.975) -> pd.Series:
    v = s.astype("float64")
    if v.notna().sum() < 5:
        return pd.Series(np.nan, index=s.index)
    ql, qh = v.quantile(lo), v.quantile(hi)
    v = v.clip(ql, qh)
    sd = v.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (v - v.mean()) / sd


def score_cross_section(df: pd.DataFrame, weights: dict[str, float]
                        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Puntúa una sección transversal ya filtrada a un grupo y una fecha.

    Devuelve (score bruto, cobertura, dispersión de componentes).
    """
    zs, ws = {}, {}
    for feat, w in weights.items():
        if feat not in df.columns:
            continue
        z = _winsorized_z(df[feat])
        if z.notna().sum() == 0:
            continue
        zs[feat], ws[feat] = z * np.sign(w), abs(w)
    if not zs:
        nan = pd.Series(np.nan, index=df.index)
        return nan, nan, nan

    Z = pd.DataFrame(zs)
    W = pd.Series(ws)
    avail = Z.notna()
    coverage = (avail * W).sum(axis=1) / W.sum()
    weighted = (Z.fillna(0) * W).sum(axis=1) / (avail * W).sum(axis=1).replace(0, np.nan)
    # desacuerdo entre componentes: alta dispersión = señal frágil
    spread = Z.std(axis=1)
    return weighted, coverage, spread


def score_panel(panel: pd.DataFrame, weights: dict[str, dict[str, float]] = None
                ) -> pd.DataFrame:
    """Aplica el scoring a todo el panel, por fecha, grupo y horizonte."""
    weights = weights or WEIGHTS
    out = panel[["date", "symbol", "group", "close"]].copy()
    for h in weights:
        out[f"score_{h}"] = np.nan
        out[f"cov_{h}"] = np.nan
        out[f"spread_{h}"] = np.nan

    for (d, g), idx in panel.groupby(["date", "group"], sort=False).groups.items():
        sub = panel.loc[idx]
        if len(sub) < MIN_PEERS:
            continue
        for h, w in weights.items():
            raw, cov, spr = score_cross_section(sub, w)
            raw = raw.where(cov >= MIN_COVERAGE)
            out.loc[idx, f"score_{h}"] = raw
            out.loc[idx, f"cov_{h}"] = cov
            out.loc[idx, f"spread_{h}"] = spr

    # percentil dentro de fecha+grupo: interpretable y comparable entre días
    for h in weights:
        out[f"pct_{h}"] = (out.groupby(["date", "group"])[f"score_{h}"]
                           .rank(pct=True) * 100).round(1)
    out["score_global"] = out[[f"score_{h}" for h in weights]].mean(axis=1)
    return out


def contributions(panel: pd.DataFrame, horizon: str, date=None,
                  weights: dict[str, dict[str, float]] = None) -> pd.DataFrame:
    """Descompone el score en la aportación de cada componente.

    Esta descomposición es aritmética, no interpretativa: aportación = peso x z.
    Es importante que salga de aquí y no de un modelo de lenguaje, porque así lo
    que se cuenta luego en el informe es lo que el sistema realmente calculó.
    """
    weights = (weights or WEIGHTS)[horizon]
    date = date if date is not None else panel.date.max()
    day = panel[panel.date == date]
    rows = []
    for g, sub in day.groupby("group"):
        if len(sub) < MIN_PEERS:
            continue
        for feat, w in weights.items():
            if feat not in sub.columns:
                continue
            z = _winsorized_z(sub[feat]) * np.sign(w)
            rows.append(pd.DataFrame({
                "symbol": sub.symbol.to_numpy(), "group": g, "feature": feat,
                "z": z.to_numpy(), "peso": abs(w),
                "aportacion": (z * abs(w)).to_numpy(),
                "valor": sub[feat].to_numpy()}))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def redundancy(symbols: list[str], close: pd.DataFrame, window: int = 126,
               max_corr: float = 0.80) -> dict[str, str]:
    """Marca los activos que son, en la práctica, la misma apuesta que otro mejor
    situado en el ranking.

    Veinte posiciones que suben y bajan juntas no son veinte ideas: son una idea
    con veinte veces el tamaño. El ranking no lo ve, porque puntúa cada activo
    por separado. Se recorre en orden y se marca el que correlaciona por encima
    del umbral con alguno ya aceptado, señalando con cuál.
    """
    cols = [s for s in symbols if s in close.columns]
    if len(cols) < 2:
        return {}
    r = np.log(close[cols].tail(window + 1) / close[cols].tail(window + 1).shift(1))
    corr = r.corr()
    kept: list[str] = []
    flags: dict[str, str] = {}
    for s in symbols:
        if s not in corr.columns:
            continue
        if kept:
            c = corr.loc[s, kept].abs()
            if c.notna().any() and c.max() >= max_corr:
                flags[s] = str(c.idxmax())
                continue
        kept.append(s)
    return flags


def top_picks(scored: pd.DataFrame, horizon: str, n: int = 25,
              groups: list[str] | None = None, max_spread: float | None = 1.6
              ) -> pd.DataFrame:
    """Mejores posicionados de la última fecha, con filtro de coherencia interna."""
    last = scored[scored.date == scored.date.max()].copy()
    if groups:
        last = last[last.group.isin(groups)]
    if max_spread is not None:
        last = last[last[f"spread_{horizon}"].fillna(9) <= max_spread]
    cols = ["symbol", "group", "close", f"score_{horizon}", f"pct_{horizon}",
            f"spread_{horizon}"]
    return last.nlargest(n, f"score_{horizon}")[cols].reset_index(drop=True)
