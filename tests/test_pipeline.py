"""Prueba de punta a punta con datos sintéticos.

Se ejecuta con:  python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import data, features, scoring, validate  # noqa: E402
from synthetic import make_market  # noqa: E402


def run(signal: float, label: str, n_days: int = 1500, null: bool = False) -> pd.DataFrame:
    t0 = time.time()
    prices, uni = make_market(n_days=n_days, signal=signal, null=null)
    prices = data.quality_filter(prices)

    px = {f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
          for f in ("open", "high", "low", "close", "volume")}
    bench = px["close"]["^GSPC"]
    fh = features.build_feature_history(px, bench)

    dates = features.rebalance_dates(px["close"].index, "ME")
    dates = dates[dates >= px["close"].index.min() + pd.Timedelta(days=400)]
    panel = features.sample_panel(fh, dates, px["close"],
                                  uni.set_index("symbol")["group"], min_dollar_vol=0)
    scored = scoring.score_panel(panel)
    fwd = features.forward_returns(px["close"], panel)
    res = validate.run_all(scored, fwd)
    v = validate.verdict(res)

    print(f"\n{'='*72}\n{label}  (señal inyectada={signal})  "
          f"[{time.time()-t0:.1f}s, {panel.symbol.nunique()} activos, "
          f"{panel.date.nunique()} fechas]\n{'='*72}")
    for h in ("corto", "medio", "largo"):
        s = res[h]["ic_resumen"]
        if s.empty:
            continue
        print(f"-- {h}: IC medio global {s.ic_medio.mean():+.4f} · "
              f"|t| máximo {s.t_stat.abs().max():.2f}")
    return v


def main() -> int:
    fails = []

    # --- 1. control negativo: paseo aleatorio -----------------------------
    v0 = run(0.0, "CONTROL NEGATIVO — no debe encontrar señal", null=True)
    t_max = v0.t_stat.abs().max()
    ic_max = v0.ic_medio.abs().max()
    print(f"\n|t| máximo = {t_max:.2f} · |IC| máximo = {ic_max:.4f}")
    # El criterio es el t-stat, no el IC bruto: en grupos de 25-30 activos el IC
    # de una sola fecha tiene un error estándar de ~0,20, así que una media de
    # 0,09 sobre ruido es perfectamente esperable. Con ~24 combinaciones
    # grupo x horizonte, algún |t| por encima de 2 es normal; por encima de 3
    # empieza a oler a fuga de información.
    if t_max > 3.0 or ic_max > 0.15:
        fails.append(f"FUGA DE DATOS: |t|={t_max:.2f}, IC={ic_max:.3f} sobre ruido puro")
    else:
        print("OK: no inventa señal donde no la hay")

    # --- 2. control positivo: momento inyectado ---------------------------
    v1 = run(0.55, "CONTROL POSITIVO — debe encontrar el momento inyectado")
    med = v1[v1.horizonte == "medio"]
    best = med.t_stat.max() if not med.empty else np.nan
    print(f"\nmejor t-stat a medio plazo = {best:.2f}")
    if not np.isfinite(best) or best < 1.5:
        fails.append("NO DETECTA una señal que sí existe: revisar cálculo")
    else:
        print("OK: detecta la señal inyectada")

    # --- 3. sin lookahead --------------------------------------------------
    prices, uni = make_market(n_days=900, signal=0.0, seed=99)
    px = {f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
          for f in ("open", "high", "low", "close", "volume")}
    fh_full = features.build_feature_history(px, px["close"]["^GSPC"])
    cut = px["close"].index[-120]
    px_cut = {k: v.loc[:cut] for k, v in px.items()}
    fh_cut = features.build_feature_history(px_cut, px_cut["close"]["^GSPC"])
    diffs = []
    for k in fh_full:
        a, b = fh_full[k].loc[cut], fh_cut[k].loc[cut]
        common = a.index.intersection(b.index)
        d = (a[common] - b[common]).abs()
        rel = d / a[common].abs().replace(0, np.nan)
        if rel.max() > 1e-6:
            diffs.append((k, float(rel.max())))
    if diffs:
        fails.append(f"LOOKAHEAD en: {diffs}")
    else:
        print("\nOK: ninguna feature cambia al añadir datos futuros (sin lookahead)")

    print("\n" + "=" * 72)
    if fails:
        print("FALLOS:")
        for f in fails:
            print(" -", f)
        return 1
    print("TODAS LAS COMPROBACIONES PASAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
