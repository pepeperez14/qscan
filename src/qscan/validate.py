"""Validación walk-forward. Es la parte que decide si el sistema sirve para algo.

Un escáner que puntúa 9.000 activos con 30 indicadores SIEMPRE produce un top-25
que parece brillante. La pregunta que importa no es "¿qué sale arriba?" sino
"¿el orden que produzco tiene alguna relación con lo que pasa después?".

Métricas:
  - IC de Spearman por fecha: correlación de rangos entre score y retorno futuro.
    Un IC medio de 0,02-0,05 sostenido ya es explotable; uno de 0,20 casi siempre
    significa que hay una fuga de información en el pipeline.
  - t-stat del IC: mide si el IC medio se distingue de cero dada su variabilidad.
  - Spread por deciles: retorno del decil superior menos el inferior, que es lo
    que realmente capturaría una cartera long-short.
  - Rotación y coste: un spread de 40 pb con 90% de rotación mensual es negativo
    después de comisiones y horquilla.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .features import HORIZON_DAYS


def information_coefficient(scored: pd.DataFrame, fwd: pd.DataFrame,
                            horizon: str) -> pd.DataFrame:
    """IC de Spearman por fecha y grupo."""
    df = scored.merge(fwd, on=["date", "symbol"], how="left")
    col_s, col_f = f"score_{horizon}", f"fwd_{horizon}"
    rows = []
    for (d, g), sub in df.groupby(["date", "group"], sort=True):
        sub = sub[[col_s, col_f]].dropna()
        if len(sub) < 30:
            continue
        ic = stats.spearmanr(sub[col_s], sub[col_f]).statistic
        rows.append({"date": d, "group": g, "n": len(sub), "ic": ic})
    return pd.DataFrame(rows)


def _newey_west_se(x: np.ndarray, lags: int) -> float:
    """Error estándar robusto a solapamiento de ventanas.

    Con rebalanceo mensual y un horizonte de 12 meses, dos observaciones
    consecutivas del IC comparten 11 meses de retorno futuro: no son
    independientes. Usar el error estándar clásico infla el t-stat por un factor
    cercano a la raíz del solapamiento, y es la razón número uno por la que un
    backtest parece significativo y luego no lo es.
    """
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan
    d = x - x.mean()
    s = float(d @ d) / n
    for k in range(1, min(lags, n - 1) + 1):
        cov = float(d[k:] @ d[:-k]) / n
        s += 2 * (1 - k / (lags + 1)) * cov
    s = max(s, 1e-12)
    return float(np.sqrt(s / n))


def ic_summary(ic: pd.DataFrame, overlap_periods: int = 1) -> pd.DataFrame:
    """Resumen por grupo con t-stat corregido por solapamiento."""
    if ic.empty:
        return pd.DataFrame()
    g = ic.groupby("group")["ic"]
    res = pd.DataFrame({
        "periodos": g.size(), "ic_medio": g.mean(), "ic_std": g.std(),
        "ic_positivo_pct": g.apply(lambda s: (s > 0).mean() * 100),
    })
    lags = max(overlap_periods - 1, 0)
    se = g.apply(lambda s: _newey_west_se(s.to_numpy(), lags))
    res["t_stat"] = res.ic_medio / se
    res["t_ingenuo"] = res.ic_medio / (res.ic_std / np.sqrt(res.periodos))
    return res.round(3).sort_values("t_stat", ascending=False)


def decile_spread(scored: pd.DataFrame, fwd: pd.DataFrame, horizon: str,
                  n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorno futuro medio por decil de score, y serie temporal del spread."""
    df = scored.merge(fwd, on=["date", "symbol"], how="left")
    col_s, col_f = f"score_{horizon}", f"fwd_{horizon}"
    df = df.dropna(subset=[col_s, col_f])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def _bin(s: pd.Series) -> pd.Series:
        if s.nunique() < n_bins:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), n_bins, labels=False) + 1

    df["decil"] = df.groupby(["date", "group"])[col_s].transform(_bin)
    df = df.dropna(subset=["decil"])
    by_decile = (df.groupby(["group", "decil"])[col_f]
                 .agg(["mean", "median", "std", "count"]).round(4))
    per_date = df.pivot_table(index=["group", "date"], columns="decil",
                              values=col_f, aggfunc="mean")
    if per_date.empty:
        return by_decile, pd.DataFrame()
    spread = (per_date[n_bins] - per_date[1]).rename("spread").reset_index()
    return by_decile, spread


def turnover(scored: pd.DataFrame, horizon: str, top_n: int = 50) -> pd.DataFrame:
    """Rotación de la cartera top-N entre fechas consecutivas."""
    col = f"score_{horizon}"
    rows = []
    for g, sub in scored.groupby("group"):
        prev: set[str] = set()
        for d, day in sub.groupby("date", sort=True):
            cur = set(day.nlargest(top_n, col).symbol)
            if prev:
                rows.append({"group": g, "date": d,
                             "turnover": 1 - len(cur & prev) / max(len(cur), 1)})
            prev = cur
    return pd.DataFrame(rows)


def net_of_costs(spread: pd.DataFrame, turn: pd.DataFrame, horizon: str,
                 cost_bps: float = 10.0) -> pd.DataFrame:
    """Spread neto asumiendo `cost_bps` de coste por lado en la parte rotada."""
    if spread.empty or turn.empty:
        return pd.DataFrame()
    m = spread.merge(turn, on=["group", "date"], how="left")
    m["coste"] = m.turnover.fillna(0) * 2 * (cost_bps / 1e4)
    m["spread_neto"] = m.spread - m.coste
    res = m.groupby("group")[["spread", "coste", "spread_neto"]].mean()
    res["periodos"] = m.groupby("group").size()
    res["horizonte_dias"] = HORIZON_DAYS[horizon]
    return res.round(4)


def run_all(scored: pd.DataFrame, fwd: pd.DataFrame, cost_bps: float = 10.0,
            rebalance_days: int = 21) -> dict:
    """Batería completa para los tres horizontes."""
    out: dict[str, dict] = {}
    for h, days in HORIZON_DAYS.items():
        ic = information_coefficient(scored, fwd, h)
        dec, spread = decile_spread(scored, fwd, h)
        turn = turnover(scored, h)
        overlap = max(int(np.ceil(days / max(rebalance_days, 1))), 1)
        out[h] = {
            "ic": ic,
            "ic_resumen": ic_summary(ic, overlap_periods=overlap),
            "deciles": dec,
            "spread": spread,
            "neto": net_of_costs(spread, turn, h, cost_bps),
            "rotacion_media": turn.groupby("group")["turnover"].mean().round(3)
            if not turn.empty else pd.Series(dtype=float),
        }
    return out


def verdict(results: dict) -> pd.DataFrame:
    """Traduce las métricas a un juicio legible. Deliberadamente severo."""
    rows = []
    for h, r in results.items():
        s = r["ic_resumen"]
        if s.empty:
            continue
        for grp, row in s.iterrows():
            t = row["t_stat"]
            if not np.isfinite(t):
                v = "sin datos"
            elif abs(row["ic_medio"]) > 0.15:
                v = "SOSPECHOSO: revisar fuga de datos"
            elif t > 2.5:
                v = "señal consistente"
            elif t > 1.5:
                v = "señal débil, no concluyente"
            else:
                v = "sin evidencia de señal"
            rows.append({"horizonte": h, "grupo": grp, "ic_medio": row["ic_medio"],
                         "t_stat": round(t, 2) if np.isfinite(t) else np.nan,
                         "periodos": int(row["periodos"]), "veredicto": v})
    return pd.DataFrame(rows)
