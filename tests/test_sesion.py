"""Regresión: una sesión a medio formar no es la foto de hoy.

El caso real, de la ejecución de las 05:24 UTC del 20/08/2026. La cripto cotiza
24 horas y las divisas casi, así que ya tenían barra del día 20. La bolsa
americana no abría hasta seis horas después. El almacén tenía por tanto una
última fila con 348 símbolos de 12.235, y como el ranking se calcula SIEMPRE
sobre la última fila del panel, el informe de ese día salió así:

    equity_us   6651 con precios ->    0 en panel
    equity_eu    277 con precios ->    0 en panel
    etf         4742 con precios ->    0 en panel
    crypto       300 con precios ->   23 en panel
    fx            30 con precios ->   26 en panel

Un informe entero sin una sola acción. Y no falló nada: el panel se construyó,
los z-scores se calcularon sobre los pocos que había y salió una página con
aspecto perfectamente normal sobre un mercado que aún no había abierto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import data  # noqa: E402


def _mercado(n_acciones=200, n_cripto=20, n_fx=10, dias=120):
    fechas = pd.bdate_range("2026-03-02", periods=dias)
    rng = np.random.default_rng(5)
    cols = ([f"AC{i:03d}" for i in range(n_acciones)]
            + [f"CR{i:02d}/USD" for i in range(n_cripto)]
            + [f"FX{i:02d}=X" for i in range(n_fx)])
    px = {}
    for campo in ("open", "high", "low", "close", "volume"):
        px[campo] = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (dias, len(cols))), axis=0)),
            index=fechas, columns=cols)
    return px, cols, fechas


def main() -> int:
    fails = []

    # --- 1. sesión a medio formar: sólo cripto y divisas ---------------------
    px, cols, fechas = _mercado()
    hoy = fechas[-1]
    acciones = [c for c in cols if c.startswith("AC")]
    for campo in px:
        px[campo].loc[hoy, acciones] = np.nan

    presentes = px["close"].notna().sum(axis=1)
    print(f"sesión {hoy.date()}: {int(presentes.iloc[-1])} símbolos de {len(cols)} "
          f"(la anterior tenía {int(presentes.iloc[-2])})")

    corte = data.ultima_sesion_util(px["close"])
    print(f"última sesión considerada útil: {corte.date()} "
          f"(esperada {fechas[-2].date()})")
    if corte != fechas[-2]:
        fails.append(f"se acepta la sesión a medias: {corte.date()}")

    recortado = data.recortar_a_sesion_util(px)
    if recortado["close"].index[-1] != fechas[-2]:
        fails.append("recortar_a_sesion_util no elimina la sesión incompleta")
    if len(recortado["close"]) != len(px["close"]) - 1:
        fails.append("se ha recortado más de una sesión")
    n_final = int(recortado["close"].iloc[-1].notna().sum())
    print(f"tras recortar, la última fila tiene {n_final} símbolos")
    if n_final < len(cols) * 0.9:
        fails.append("la última sesión tras recortar tampoco está completa")

    # --- 2. un día normal no se toca -----------------------------------------
    px2, cols2, fechas2 = _mercado()
    corte2 = data.ultima_sesion_util(px2["close"])
    print(f"\nmercado completo: se conserva {corte2.date()} "
          f"(última {fechas2[-1].date()})")
    if corte2 != fechas2[-1]:
        fails.append("se descarta una sesión completa perfectamente válida")

    # --- 3. varias sesiones a medias seguidas --------------------------------
    px3, cols3, fechas3 = _mercado()
    acc3 = [c for c in cols3 if c.startswith("AC")]
    for campo in px3:
        px3[campo].loc[fechas3[-2:], acc3] = np.nan
    corte3 = data.ultima_sesion_util(px3["close"])
    print(f"dos sesiones a medias: se retrocede hasta {corte3.date()} "
          f"(esperada {fechas3[-3].date()})")
    if corte3 != fechas3[-3]:
        fails.append("no retrocede más de una sesión cuando hace falta")

    # --- 4. huecos normales no disparan el recorte ---------------------------
    # que falte un 20% de los símbolos un día cualquiera es lo habitual
    px4, cols4, fechas4 = _mercado()
    rng = np.random.default_rng(9)
    faltan = rng.choice(cols4, size=int(len(cols4) * 0.2), replace=False)
    for campo in px4:
        px4[campo].loc[fechas4[-1], faltan] = np.nan
    corte4 = data.ultima_sesion_util(px4["close"])
    print(f"con un 20% de ausencias: se conserva {corte4.date()} "
          f"(última {fechas4[-1].date()})")
    if corte4 != fechas4[-1]:
        fails.append("un día con ausencias normales se descarta por error")

    # --- 5. la lista de muertos exige DOS fallos seguidos --------------------
    print()
    m = {}
    m = data.actualizar_muertos(m, ["A", "B", "C"], {"A"}, pd.Timestamp("2026-08-19"))
    print(f"tras 1 ronda: {len(m)} con fallos · {len(data.muertos_vigentes(m, pd.Timestamp('2026-08-19')))} dados por muertos")
    if data.muertos_vigentes(m, pd.Timestamp("2026-08-19")):
        fails.append("un solo fallo ya da un símbolo por muerto")
    m = data.actualizar_muertos(m, ["A", "B", "C"], {"A", "B"}, pd.Timestamp("2026-08-20"))
    vig = data.muertos_vigentes(m, pd.Timestamp("2026-08-20"))
    print(f"tras 2 rondas: dados por muertos {sorted(vig)} (esperado ['C'])")
    if vig != {"C"}:
        fails.append(f"se esperaba sólo C dado por muerto, hay {sorted(vig)}")
    if "B" in m:
        fails.append("B respondió en la segunda ronda: debería salir del registro")

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
