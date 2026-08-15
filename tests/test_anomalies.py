"""Prueba del detector de anomalías: se inyectan fallos conocidos y se comprueba
que los encuentra, y que no marca las series sanas.

Un detector que marca todo es tan inútil como uno que no marca nada, así que se
mide también la tasa de falsos positivos sobre series limpias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qscan import anomalies  # noqa: E402
from synthetic import make_market  # noqa: E402


def main() -> int:
    prices, uni = make_market(n_days=700, seed=11,
                              groups={"equity_us": 120, "etf": 40, "crypto": 30})
    px = {f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
          for f in ("open", "high", "low", "close", "volume")}
    clean = list(px["close"].columns)
    n = len(px["close"])
    expected: dict[str, str] = {}

    def victim(tag: str) -> str:
        s = clean.pop(5)
        expected[s] = tag
        return s

    # 1. cotización congelada 15 sesiones
    s = victim("stale")
    px["close"].iloc[-40:-25, px["close"].columns.get_loc(s)] = \
        px["close"][s].iloc[-41]

    # 2. split 4:1 sin ajustar
    s = victim("split_sin_ajustar")
    i = px["close"].columns.get_loc(s)
    px["close"].iloc[-100:, i] = px["close"].iloc[-100:, i] / 4.0

    # 3. tick malo que se revierte
    s = victim("tick_malo")
    i = px["close"].columns.get_loc(s)
    px["close"].iloc[-60, i] = px["close"].iloc[-60, i] * 2.4

    # 4. volumen cero en un tercio de las sesiones
    s = victim("volumen_cero")
    px["volume"].iloc[-200:, px["volume"].columns.get_loc(s)] = 0.0

    # 5. OHLC plano
    s = victim("ohlc_plano")
    i = px["close"].columns.get_loc(s)
    for k in ("open", "high", "low"):
        px[k].iloc[-300:, i] = px["close"].iloc[-300:, i]

    # 6. feed muerto
    s = victim("serie_muerta")
    px["close"].iloc[-30:, px["close"].columns.get_loc(s)] = np.nan

    # 7. serie duplicada
    s = victim("duplicada")
    src = clean[0]
    px["close"][s] = px["close"][src].to_numpy()

    rep = anomalies.detect(px, min_dollar_vol=0.0)

    fails, hits = [], 0
    print(f"{'activo':>16}  {'esperado':<20} {'detectado'}")
    for sym, tag in expected.items():
        got = [c for c in anomalies.SEVERITY if c in rep.columns and bool(rep.loc[sym, c])]
        ok = tag in got
        hits += ok
        print(f"{sym:>16}  {tag:<20} {', '.join(got) or '—'}  {'OK' if ok else 'FALLO'}")
        if not ok:
            fails.append(f"no detecta {tag} en {sym}")

    # falsos positivos sobre las series sanas
    sane = rep.loc[[c for c in clean if c in rep.index]]
    fp_cols = [c for c in anomalies.SEVERITY if c in sane.columns and c != "iliquida"]
    fp = sane[fp_cols].any(axis=1).mean()
    print(f"\naciertos {hits}/{len(expected)} · falsos positivos "
          f"{fp*100:.1f}% sobre {len(sane)} series limpias")
    if fp > 0.05:
        fails.append(f"demasiados falsos positivos: {fp:.1%}")

    q = int(rep["cuarentena"].sum())
    print(f"activos en cuarentena: {q}")

    if fails:
        print("\nFALLOS:")
        for f in fails:
            print(" -", f)
        return 1
    print("\nTODAS LAS COMPROBACIONES PASAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
