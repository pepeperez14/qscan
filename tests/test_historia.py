"""Regresión: un símbolo nuevo en el universo necesita su historia entera.

El caso real, del log del 19/08/2026: se incorporaron 278 acciones europeas al
universo y se descargaron con la MISMA fecha de inicio que el resto —seis
semanas— porque sólo había un `start` para todo el mundo. Resultado: de 277 con
precios, sólo 99 llegaron al panel. Las features de doce meses no se pueden
calcular sobre mes y medio de cotizaciones, así que dos tercios de la renta
variable europea entraron al universo y se quedaron fuera del ranking.

Y el segundo efecto, más silencioso: el inicio incremental era el MÍNIMO de las
últimas fechas del almacén, así que un solo símbolo rezagado obligaba a bajar
esa ventana para los doce mil restantes.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import data  # noqa: E402

TMP = Path("/tmp/qscan_historia")


def main() -> int:
    fails = []
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)

    hoy = data._ahora()
    fechas = pd.bdate_range(hoy - pd.Timedelta(days=1000), hoy)
    viejas = pd.bdate_range(hoy - pd.Timedelta(days=1000), hoy - pd.Timedelta(days=90))

    filas = []
    for s in [f"US{i:03d}" for i in range(50)]:          # al día
        filas.append(pd.DataFrame({"symbol": s, "date": fechas, "open": 10.0,
                                   "high": 10.0, "low": 10.0, "close": 10.0,
                                   "volume": 1e6}))
    filas.append(pd.DataFrame({"symbol": "REZAGADO", "date": viejas, "open": 10.0,
                               "high": 10.0, "low": 10.0, "close": 10.0,
                               "volume": 1e6}))
    store = data.PriceStore(TMP / "prices.parquet")
    store.save(pd.concat(filas, ignore_index=True))

    # y los "al día pero cortos": actualizados cada mañana, pero con seis
    # semanas de historia. Es el caso exacto de las 278 acciones europeas.
    recientes = pd.bdate_range(hoy - pd.Timedelta(days=42), hoy)
    cortos = [f"EUCORTO{i:02d}.MC" for i in range(10)]
    filas2 = [pd.DataFrame({"symbol": s, "date": recientes, "open": 10.0,
                            "high": 10.0, "low": 10.0, "close": 10.0,
                            "volume": 1e6}) for s in cortos]
    store.save(pd.concat([store.load()] + filas2, ignore_index=True))

    # universo: los 50 conocidos + el rezagado + los cortos + 20 europeos nuevos
    europeos = [f"EU{i:03d}.MC" for i in range(20)]
    uni = pd.DataFrame({
        "symbol": [f"US{i:03d}" for i in range(50)] + ["REZAGADO"] + cortos + europeos,
        "name": "x",
        "group": ["equity_us"] * 51 + ["equity_eu"] * (len(cortos) + 20),
    })
    # una cripto corta: comprueba que su rama también anota la petición
    cripto = pd.bdate_range(hoy - pd.Timedelta(days=42), hoy)
    store.save(pd.concat([store.load(), pd.DataFrame({
        "symbol": "BTC/USD", "date": cripto, "open": 1.0, "high": 1.0,
        "low": 1.0, "close": 1.0, "volume": 1.0})], ignore_index=True))
    uni = pd.concat([uni, pd.DataFrame([{"symbol": "BTC/USD", "name": "BTC",
                                         "group": "crypto"}])], ignore_index=True)

    peticiones = []

    def fetch_falso(symbols, start, **kw):
        peticiones.append((sorted(symbols), pd.Timestamp(start)))
        f = pd.bdate_range(pd.Timestamp(start), hoy)
        if not len(f):
            return pd.DataFrame(columns=["symbol", "date"] + data.COLS)
        return pd.concat([pd.DataFrame({
            "symbol": s, "date": f, "open": 10.0, "high": 10.0, "low": 10.0,
            "close": 10.0, "volume": 1e6}) for s in symbols], ignore_index=True)

    cripto_pedidas = []

    def cripto_falso(symbols, start, **kw):
        # no pasa por `peticiones`: ahí se cuentan sólo las llamadas a Yahoo
        cripto_pedidas.append((sorted(symbols), pd.Timestamp(start)))
        f = pd.bdate_range(pd.Timestamp(start), hoy)
        return pd.concat([pd.DataFrame({
            "symbol": s, "date": f, "open": 1.0, "high": 1.0, "low": 1.0,
            "close": 1.0, "volume": 1.0}) for s in symbols], ignore_index=True)

    real, real_c = data.fetch_yahoo, data.fetch_crypto
    data.fetch_yahoo, data.fetch_crypto = fetch_falso, cripto_falso
    try:
        data.update(uni, store, years=8, recarga_completa=False, usar_reserva=False)
    finally:
        data.fetch_yahoo, data.fetch_crypto = real, real_c

    print(f"llamadas a la descarga: {len(peticiones)}")
    for syms, start in peticiones:
        etiqueta = "incremental" if (hoy - start).days < 60 else "historia entera"
        print(f"  {len(syms):3d} símbolos desde {start.date()}  ({etiqueta})")

    if len(peticiones) != 2:
        fails.append(f"deberían ser dos descargas (incremental y completa), "
                     f"no {len(peticiones)}")

    completas = [(s, d) for s, d in peticiones if (hoy - d).days > 300]
    if not completas:
        fails.append("nadie recibe historia entera: los nuevos entrarían con "
                     "seis semanas de precios")
    else:
        pedidos_completos = set(completas[0][0])
        faltan_eu = [s for s in europeos if s not in pedidos_completos]
        if faltan_eu:
            fails.append(f"{len(faltan_eu)} europeos nuevos sin historia entera")
        if "REZAGADO" not in pedidos_completos:
            fails.append("el símbolo rezagado no recibe historia entera")
        faltan_cortos = [s for s in cortos if s not in pedidos_completos]
        if faltan_cortos:
            fails.append(f"{len(faltan_cortos)} símbolos al día pero con seis "
                         f"semanas de historia no reciben historia entera: es "
                         f"el caso de las acciones europeas")
        colados = [s for s in pedidos_completos if s.startswith("US")]
        if colados:
            fails.append(f"{len(colados)} símbolos al día piden historia entera "
                         f"sin necesidad")

    incrementales = [(s, d) for s, d in peticiones if (hoy - d).days <= 60]
    if incrementales:
        syms, start = incrementales[0]
        print(f"\nel rezagado NO arrastra la ventana incremental: empieza en "
              f"{start.date()} ({(hoy - start).days} días)")
        if (hoy - start).days > 30:
            fails.append("la ventana incremental sigue arrastrada por el rezagado")

    # y tras la actualización, los nuevos tienen historia de verdad
    df = store.load()
    n_eu = df[df.symbol.isin(europeos + cortos)].groupby("symbol").size()
    print(f"filas por europeo tras actualizar: mínimo {int(n_eu.min())}")
    if int(n_eu.min()) < 250:
        fails.append(f"los europeos se quedan con {int(n_eu.min())} sesiones: "
                     f"no llegan para las features de doce meses")

    # y no se les vuelve a pedir al día siguiente: si siguen cortos es que son
    # jóvenes de verdad, y pedir ocho años cada día no los hará más viejos
    peticiones.clear(); cripto_pedidas.clear()
    data.fetch_yahoo, data.fetch_crypto = fetch_falso, cripto_falso
    try:
        data.update(uni, store, years=8, recarga_completa=False, usar_reserva=False)
    finally:
        data.fetch_yahoo, data.fetch_crypto = real, real_c
    cripto_completas = [s for s, d in cripto_pedidas if (hoy - d).days > 300]
    print(f"en la segunda vuelta, cripto pidiendo historia entera: "
          f"{len(cripto_completas)}")
    if cripto_completas:
        fails.append("la cripto corta vuelve a pedir ocho años cada día: la "
                     "marca no se escribe en su rama")
    completas2 = [(s, d) for s, d in peticiones if (hoy - d).days > 300]
    n2 = len(completas2[0][0]) if completas2 else 0
    print(f"en la segunda vuelta piden historia entera: {n2} símbolos")
    if n2 > 5:
        fails.append(f"{n2} símbolos vuelven a pedir ocho años al día siguiente")

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
