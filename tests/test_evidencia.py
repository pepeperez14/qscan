"""Regresión: una medición que el sistema declara no creíble no puede repartir capital.

El caso real, del informe del 19/08/2026: la cripto a largo plazo salió con
IC 0,584 y t-stat 26,8 sobre SIETE ventanas de un año. Con esa historia las
ventanas se solapan casi por completo, así que siete "periodos" son una sola
observación independiente y el t-stat no mide nada.

El veredicto lo marcaba correctamente como "SOSPECHOSO: revisar fuga de datos".
Pero `pesos_por_evidencia` promediaba los t-stats sin mirar el veredicto, así
que ese 26,8 arrastraba la media del horizonte largo hasta 9,3 y se llevaba el
83% del capital de la cartera combinada. La cifra que el sistema declaraba
mentira entraba por la puerta de atrás.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import portfolio, validate  # noqa: E402


def _resumen(filas) -> dict:
    """Construye la estructura que devuelve validate.run_all, por horizonte."""
    out = {}
    for h in ("corto", "medio", "largo"):
        sub = [f for f in filas if f[0] == h]
        if not sub:
            continue
        df = pd.DataFrame([{"ic_medio": ic, "t_stat": t, "periodos": n}
                           for _, g, ic, t, n in sub],
                          index=[g for _, g, _, _, _ in sub])
        out[h] = {"ic_resumen": df}
    return out


def main() -> int:
    fails = []

    # cifras exactas del informe del 19/08/2026
    reales = [
        ("corto", "equity_us", 0.032, 2.60, 81),
        ("corto", "etf", 0.066, 2.16, 81),
        ("corto", "crypto", 0.118, 1.57, 16),
        ("medio", "equity_us", 0.044, 2.12, 77),
        ("medio", "crypto", 0.136, 1.50, 16),
        ("medio", "etf", 0.052, 1.29, 77),
        ("largo", "crypto", 0.584, 26.81, 7),
        ("largo", "etf", 0.039, 0.62, 68),
        ("largo", "equity_us", 0.031, 0.58, 68),
    ]
    v = validate.verdict(_resumen(reales))
    print(v.to_string(index=False))

    fila = v[(v.horizonte == "largo") & (v.grupo == "crypto")].iloc[0]
    print(f"\ncripto/largo -> usable={fila.usable} · {fila.veredicto}")
    if bool(fila.usable):
        fails.append("cripto a largo con 7 ventanas se marca como usable")
    if not bool(v[(v.horizonte == "corto") & (v.grupo == "equity_us")].usable.iloc[0]):
        fails.append("equity_us a corto (81 ventanas, t 2,60) debería ser usable")

    pesos, nota = portfolio.pesos_por_evidencia(v)
    print(f"\nreparto de la cartera combinada: "
          + " · ".join(f"{h} {p*100:.0f}%" for h, p in pesos.items()))
    print(f"nota: {nota}")
    if pesos["largo"] > 0.50:
        fails.append(f"el horizonte largo se lleva el {pesos['largo']*100:.0f}% "
                     f"por una medición que el propio sistema declara no creíble")

    # y el reparto ANTES del arreglo, para que se vea que la prueba prueba algo
    t_medio = {h: np.mean([t for hh, _, _, t, _ in reales if hh == h])
               for h in ("corto", "medio", "largo")}
    f_viejo = {h: max(t - 1, 0) for h, t in t_medio.items()}
    tot = sum(f_viejo.values())
    print("\ncon la media sin filtrar (lo que hacía antes): "
          + " · ".join(f"{h} {v_/tot*100:.0f}%" for h, v_ in f_viejo.items()))
    if f_viejo["largo"] / tot < 0.7:
        fails.append("el escenario no reproduce el problema: revisa la prueba")

    # un horizonte legítimamente fuerte tampoco puede acaparar sin límite
    extremo = validate.verdict(_resumen([
        ("corto", "equity_us", 0.03, 2.0, 80),
        ("medio", "equity_us", 0.03, 2.0, 80),
        ("largo", "equity_us", 0.05, 40.0, 80),
    ]))
    p2, _ = portfolio.pesos_por_evidencia(extremo)
    print(f"\ncon un t de 40 creíble por periodos: largo {p2['largo']*100:.0f}%")
    if p2["largo"] > 0.85:
        fails.append("sin tope, un solo horizonte acapara la cartera entera")

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
