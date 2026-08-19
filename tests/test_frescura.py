"""Regresión del fallo real: el sistema se declaraba correcto con datos congelados.

Durante tres sesiones seguidas las ejecuciones terminaron en verde mientras el
almacén seguía anclado en el 14 de agosto. No hubo ningún error: la descarga
dejó de traer datos, la fusión no cambió nada y todo lo demás siguió calculando
sobre precios viejos. Una ejecución verde con datos rancios es peor que una
roja, porque nadie la mira.

Se comprueba también el mapeo de símbolos de la fuente de reserva, cuyo modo de
fallo es especialmente traicionero: un símbolo mal traducido no da error, trae
la serie de OTRO activo.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import data  # noqa: E402

TMP = Path("/tmp/qscan_frescura")


def _store_hasta(fecha: str) -> data.PriceStore:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    fechas = pd.bdate_range(end=fecha, periods=300)
    px = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(fechas))))
    df = pd.DataFrame({"symbol": "ACME", "date": fechas, "open": px, "high": px,
                       "low": px, "close": px, "volume": 1e6})
    s = data.PriceStore(TMP / "prices.parquet")
    s.save(df)
    return s


def main() -> int:
    fails = []

    # --- 1. el caso exacto que ocurrió ------------------------------------
    # almacén parado el viernes 14; el sistema siguió corriendo hasta el 19
    store = _store_hasta("2026-08-14")
    hoy = pd.Timestamp("2026-08-19")
    retraso = data.sesiones_de_retraso(pd.Timestamp("2026-08-14"), hoy)
    print(f"almacén al 2026-08-14, hoy 2026-08-19 -> {retraso} sesiones de retraso")
    try:
        data.comprobar_frescura(store, hoy=hoy)
        fails.append("NO detecta el caso real: tres sesiones sin avanzar pasaron "
                     "como si nada")
    except SystemExit as e:
        print("OK: corta la ejecución ->", str(e)[:80] + "...")

    # --- 2. no debe dar falsos positivos ----------------------------------
    casos = [
        ("2026-08-18", "2026-08-19", "día normal: cierre de ayer", False),
        ("2026-08-14", "2026-08-18", "lunes festivo: cierre del viernes", False),
        ("2026-08-19", "2026-08-19", "mismo día", False),
    ]
    for ultima, hoy_s, desc, debe_fallar in casos:
        s = _store_hasta(ultima)
        try:
            r = data.comprobar_frescura(s, hoy=pd.Timestamp(hoy_s))
            ok = not debe_fallar
            print(f"  {desc:<38} retraso {r} · {'OK' if ok else 'FALLO'}")
            if debe_fallar:
                fails.append(f"{desc}: debería haber fallado")
        except SystemExit:
            print(f"  {desc:<38} FALLA · {'OK' if debe_fallar else 'FALSO POSITIVO'}")
            if not debe_fallar:
                fails.append(f"{desc}: falso positivo, corta una ejecución sana")

    # --- 3. almacén vacío --------------------------------------------------
    vacio = data.PriceStore(TMP / "vacio.parquet")
    try:
        data.comprobar_frescura(vacio)
        fails.append("un almacén vacío no debería pasar el control")
    except SystemExit:
        print("OK: almacén vacío detectado")

    # --- 4. mapeo de la fuente de reserva ---------------------------------
    # Lo peligroso no es no traducir, es traducir mal: eso devuelve otra serie.
    esperado = {"AAPL": "aapl.us", "MSFT": "msft.us", "BRK-B": "brk.b.us",
                "^GSPC": None, "EURUSD=X": None, "GC=F": None, "ZEC/USD": None,
                "SAN.MC": None, "": None}
    for sym, esp in esperado.items():
        got = data._a_stooq(sym)
        if got != esp:
            fails.append(f"mapeo incorrecto de {sym!r}: {got!r} en vez de {esp!r}")
    print(f"mapeo de símbolos: {len(esperado)} casos comprobados")

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
