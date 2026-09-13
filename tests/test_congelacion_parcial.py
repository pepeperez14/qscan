"""Regresión: la cripto no puede tapar que la bolsa lleva días sin actualizarse.

El caso real, encontrado el 13/09/2026. El sistema se declaró correcto DIECISÉIS
días seguidos —`.last_success` marcaba el 12/09— publicando siempre el mismo
informe del 28/08:

    .last_success        2026-09-12   (el escáner termina bien cada día)
    informe publicado    28/08/2026
    portfolio/curva.csv  28/08/2026   (once sesiones sin avanzar)

La cripto cotiza 24 horas y las divisas casi, así que seguían trayendo barras
todos los días. La fecha máxima del almacén era la de ayer y `comprobar_frescura`
—que miraba ese máximo en bruto— pasaba tan contenta. La bolsa llevaba once
sesiones sin entrar una sola fila.

Lo que hace este fallo distinto de los anteriores es que fueron DOS DEFENSAS
CORRECTAS ANULÁNDOSE. `ultima_sesion_util` descartaba con toda la razón esas
sesiones de sólo cripto y se quedaba en el 28/08; el control de frescura miraba
el máximo y no veía nada raro. Cada uno hacía bien su trabajo y entre los dos
produjeron el peor resultado posible: un informe coherente, publicado cada día,
sobre precios de dos semanas atrás.

Se comprueban las dos correcciones:
  1. el control de frescura mide la última sesión COMPLETA, no la última fila;
  2. la referencia de "sesión completa" es el mejor día de los últimos seis
     meses, no la mediana de los últimos veinte — con una ventana corta, un
     bloqueo más largo que la ventana acaba redefiniendo la normalidad.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import data  # noqa: E402

TMP = Path("/tmp/qscan_congelacion")


def _almacen(dias_congelado: int, n_bolsa: int = 2000, n_cripto: int = 300):
    """Bolsa parada hace N sesiones, cripto y divisas al día."""
    fechas = pd.bdate_range("2026-02-02", "2026-09-11")
    corte = fechas[-1 - dias_congelado] if dias_congelado else fechas[-1]
    filas = []
    for i in range(n_bolsa):
        f = fechas[fechas <= corte]
        filas.append(pd.DataFrame({"symbol": f"AC{i:04d}", "date": f, "open": 10.0,
                                   "high": 10.0, "low": 10.0, "close": 10.0,
                                   "volume": 1e6}))
    for i in range(n_cripto):
        filas.append(pd.DataFrame({"symbol": f"CR{i:03d}/USD", "date": fechas,
                                   "open": 1.0, "high": 1.0, "low": 1.0,
                                   "close": 1.0, "volume": 1e6}))
    df = pd.concat(filas, ignore_index=True)
    grupos = pd.Series({**{f"AC{i:04d}": "equity_us" for i in range(n_bolsa)},
                        **{f"CR{i:03d}/USD": "crypto" for i in range(n_cripto)}})
    return df, fechas, corte, grupos


def main() -> int:
    fails = []
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    hoy = pd.Timestamp("2026-09-11")

    # --- 1. once sesiones congelado: el control TIENE que cortar -------------
    df, fechas, corte, grupos = _almacen(dias_congelado=11)
    store = data.PriceStore(TMP / "prices.parquet")
    store.save(df)
    bruta = pd.to_datetime(df["date"]).max()
    print(f"bolsa parada el {corte.date()} · última fila del almacén "
          f"{bruta.date()} (cripto al día)")

    try:
        data.comprobar_frescura(store, max_sesiones=2, hoy=hoy, grupos=grupos)
        fails.append("DATOS CONGELADOS no salta: la cripto sigue tapando el "
                     "bloqueo de la bolsa")
        print("  -> no corta (MAL)")
    except SystemExit as e:
        print(f"  -> corta: {str(e)[:96]}...")
        if "28" not in str(e) and str(corte.date()) not in str(e):
            fails.append("el mensaje no dice cuál es la última sesión completa")

    # --- 2. la ventana larga: un bloqueo de 40 sesiones sigue detectándose ---
    # con la mediana de las últimas 20, a partir de la sesión 21 "sólo cripto"
    # pasa a ser la normalidad y el control deja de saltar
    df2, fechas2, corte2, grupos2 = _almacen(dias_congelado=40)
    store2 = data.PriceStore(TMP / "prices2.parquet")
    store2.save(df2)
    print(f"\ncuarenta sesiones congelado (bolsa parada el {corte2.date()}):")
    try:
        data.comprobar_frescura(store2, max_sesiones=2, hoy=hoy, grupos=grupos2)
        fails.append("con 40 sesiones de bloqueo el control ya no salta: la "
                     "ventana de referencia es demasiado corta")
        print("  -> no corta (MAL)")
    except SystemExit:
        print("  -> corta igualmente")

    # --- 3. un almacén sano no puede dar falsos positivos -------------------
    df3, fechas3, _, grupos3 = _almacen(dias_congelado=0)
    store3 = data.PriceStore(TMP / "prices3.parquet")
    store3.save(df3)
    try:
        r = data.comprobar_frescura(store3, max_sesiones=2, hoy=hoy, grupos=grupos3)
        print(f"\nalmacén al día: pasa el control ({r} sesiones de retraso)")
    except SystemExit as e:
        fails.append(f"falso positivo con el almacén al día: {e}")
        print(f"\nalmacén al día: CORTA (MAL) -> {str(e)[:80]}")

    # --- 4. ultima_sesion_util con la ventana larga -------------------------
    cols = sorted(df2.symbol.unique())
    close = df2.pivot(index="date", columns="symbol", values="close").sort_index()
    util = data.ultima_sesion_util(close)
    print(f"\nsobre la matriz ancha con 40 sesiones de bloqueo, la última "
          f"sesión útil es {pd.Timestamp(util).date()} (esperada {corte2.date()})")
    if pd.Timestamp(util) != corte2:
        fails.append(f"ultima_sesion_util devuelve {pd.Timestamp(util).date()}: "
                     f"ha aceptado sesiones de sólo cripto como completas")

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
