"""Regresión: los grupos sin volumen publicado deben llegar al ranking.

Yahoo no publica volumen para pares de divisas ni para índices: viene 0 o vacío.
El filtro de liquidez en dólares, pensado para acciones, los eliminaba a todos.
No era un umbral mal calibrado, era aplicar un criterio a instrumentos para los
que ese dato no existe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qscan import features, scoring  # noqa: E402
from synthetic import make_market  # noqa: E402


def main() -> int:
    prices, uni = make_market(n_days=900, seed=21,
                              groups={"equity_us": 60, "fx": 30, "index": 30})
    # como en la realidad: divisas e índices sin volumen
    sin_vol = uni.loc[uni.group.isin(["fx", "index"]), "symbol"]
    prices.loc[prices.symbol.isin(sin_vol), "volume"] = 0.0

    px = {f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
          for f in ("open", "high", "low", "close", "volume")}
    fh = features.build_feature_history(px, px["close"]["^GSPC"])
    dates = features.rebalance_dates(px["close"].index, "ME")
    dates = dates[dates >= px["close"].index.min() + pd.Timedelta(days=400)][-4:]
    groups = uni.set_index("symbol")["group"]

    # con el filtro de liquidez activo, como en producción
    panel = features.sample_panel(fh, dates, px["close"], groups, min_dollar_vol=1e6)
    scored = scoring.score_panel(panel)
    last = scored[scored.date == scored.date.max()]

    fails = []
    print(f"{'grupo':<10} {'en panel':>9} {'con score':>10}")
    for g in ("equity_us", "fx", "index"):
        n_panel = int((panel[panel.date == panel.date.max()].group == g).sum())
        n_score = int(last[(last.group == g) & last.score_medio.notna()].shape[0])
        print(f"{g:<10} {n_panel:>9} {n_score:>10}")
        if n_score == 0:
            fails.append(f"{g} desaparece del ranking pese a tener precios válidos")

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
