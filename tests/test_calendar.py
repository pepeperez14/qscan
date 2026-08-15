"""Regresión: mezcla de calendarios (cripto 24/7 con bolsas de 5 días).

Este fallo no lo cazó ninguna prueba anterior porque el generador sintético daba
a todos los grupos el mismo calendario hábil. Es el defecto clásico de una
batería de tests: comprueba el código contra el mundo que el propio test imagina.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qscan import data, features, scoring  # noqa: E402
from synthetic import make_market  # noqa: E402


def _mixed_market(n_bdays: int = 900):
    """Acciones en días hábiles + cripto todos los días, como en producción."""
    eq, uni_eq = make_market(n_days=n_bdays, seed=5,
                             groups={"equity_us": 90, "etf": 40})
    # la cripto necesita 7/5 de barras para cubrir el mismo periodo natural
    n_cal = int(n_bdays * 7 / 5)
    cr, uni_cr = make_market(n_days=n_cal, seed=6, groups={"crypto": 40})

    # reetiquetar la cripto sobre un calendario natural de 7 días
    cal = pd.date_range(eq.date.min(), periods=n_cal, freq="D")
    out = []
    for sym, g in cr.groupby("symbol"):
        g = g.sort_values("date").head(len(cal)).copy()
        g["date"] = cal[:len(g)]
        out.append(g)
    cr = pd.concat(out, ignore_index=True)
    prices = pd.concat([eq, cr[cr.symbol != "^GSPC"]], ignore_index=True)
    uni = pd.concat([uni_eq, uni_cr[uni_cr.symbol != "^GSPC"]], ignore_index=True)
    return prices, uni.drop_duplicates("symbol")


def main() -> int:
    fails = []
    prices, uni = _mixed_market()
    raw = {f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
           for f in ("open", "high", "low", "close", "volume")}

    n_raw = len(raw["close"])
    px = data.to_business_calendar(raw)
    n_fix = len(px["close"])
    print(f"índice sin alinear: {n_raw} filas · alineado: {n_fix} filas")
    if (px["close"].index.dayofweek >= 5).any():
        fails.append("quedan fines de semana en el índice alineado")

    # una ventana de 252 filas debe cubrir ~1 año natural
    span = (px["close"].index[-1] - px["close"].index[-252]).days / 365
    print(f"252 filas = {span:.2f} años (debe rondar 1,00)")
    if not 0.93 < span < 1.07:
        fails.append(f"la ventana de 252 filas cubre {span:.2f} años, no 1")

    # y las features de ventana completa no pueden salir NaN para las acciones
    fh = features.build_feature_history(px, px["close"]["^GSPC"])
    eq_syms = uni.loc[uni.group.isin(["equity_us", "etf"]), "symbol"]
    eq_syms = [s for s in eq_syms if s in px["close"].columns]
    for feat in ("slope_12m", "trendfit_12m", "mom_12_1", "vol_252"):
        cov = fh[feat].iloc[-1].reindex(eq_syms).notna().mean()
        print(f"  cobertura de {feat:<13} en acciones: {cov*100:5.1f}%")
        if cov < 0.90:
            fails.append(f"{feat} sólo cubre {cov:.0%} de las acciones")

    # y la renta variable debe seguir presente en el ranking de largo plazo
    dates = features.rebalance_dates(px["close"].index, "ME")
    dates = dates[dates >= px["close"].index.min() + pd.Timedelta(days=400)][-6:]
    panel = features.sample_panel(fh, dates, px["close"],
                                  uni.set_index("symbol")["group"], 0)
    scored = scoring.score_panel(panel)
    last = scored[scored.date == scored.date.max()]
    for g in ("equity_us", "etf", "crypto"):
        n = last[(last.group == g) & last.score_largo.notna()].shape[0]
        print(f"  {g:<10} con score de largo plazo: {n}")
        if n == 0:
            fails.append(f"{g} desaparece del ranking de largo plazo")

    print()
    if fails:
        print("FALLOS:")
        for f in fails:
            print(" -", f)
        return 1
    print("TODAS LAS COMPROBACIONES PASAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
