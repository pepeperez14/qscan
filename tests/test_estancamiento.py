"""Regresión: la cartera no puede quedarse quieta ni valorar a coste en silencio.

Dos averías reales encontradas en el repositorio el 30/08/2026, ambas invisibles
en el log y ambas con la misma forma: el sistema seguía dando resultados con
aspecto correcto.

1. **La cartera llevaba seis sesiones sin avanzar.** `.last_success` decía
   2026-08-29 —el escáner corría y publicaba todos los días— mientras
   `curva.csv` seguía anclado en el 20/08. Seis sesiones de operaciones que
   nunca existieron.

2. **Cinco posiciones valoradas a su precio de compra.** La cartera compró el
   14/08 los ETFs apalancados ASTY, CRWL, DLLL, LOFF y SNOU. Días después el
   filtro de derivados los sacó del universo y la poda del almacén se llevó sus
   precios. `_valorar` no encontraba cotización y los contaba por su coste:
   15.993 € de 163.708 —casi el 10% del capital simulado— congelados, inmunes a
   subidas y bajadas, inflando la estabilidad aparente de la curva.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import data, portfolio  # noqa: E402

TMP = Path("/tmp/qscan_estancamiento")


class _Captura(logging.Handler):
    def __init__(self):
        super().__init__()
        self.errores: list[str] = []

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.errores.append(record.getMessage())


def prueba_alarma_estancamiento() -> list[str]:
    fails = []
    cap = _Captura()
    log = logging.getLogger("qscan.portfolio")
    log.addHandler(cap)
    try:
        # las fechas reales: estado en el 20/08, escaneo en el 28/08
        atraso = portfolio._avisar_estancamiento("2026-08-20", pd.Timestamp("2026-08-28"))
        print(f"  estado 20/08 · escaneo 28/08 -> {atraso} sesiones de atraso")
        if atraso != 6:
            fails.append(f"el atraso calculado es {atraso}, deberían ser 6")
        if not cap.errores:
            fails.append("seis sesiones de atraso no producen ningún ERROR")
        else:
            print(f"  ERROR registrado: {cap.errores[0][:80]}...")

        cap.errores.clear()
        # un fin de semana normal no es un estancamiento
        portfolio._avisar_estancamiento("2026-08-28", pd.Timestamp("2026-08-31"))
        print(f"  viernes -> lunes: {len(cap.errores)} errores")
        if cap.errores:
            fails.append("un fin de semana normal dispara la alarma")
    finally:
        log.removeHandler(cap)
    return fails


def prueba_liquidar_sin_precio() -> list[str]:
    fails = []
    fechas = pd.bdate_range("2026-06-01", periods=60)
    rng = np.random.default_rng(3)
    cols = ["VIVO", "MUERTO"]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (len(fechas), 2)), axis=0)),
        index=fechas, columns=cols)
    # MUERTO deja de cotizar hace 20 sesiones: es el caso de ASTY y compañía
    ultimo_precio = float(close["MUERTO"].iloc[-21])
    close.loc[fechas[-20:], "MUERTO"] = np.nan
    px = {c: close.copy() for c in ("open", "high", "low", "close")}
    px["volume"] = pd.DataFrame(1e6, index=fechas, columns=cols)

    b = {"efectivo": 1000.0, "posiciones": {
            "VIVO": {"unidades": 10.0, "coste_eur": 1000.0},
            "MUERTO": {"unidades": 10.0, "coste_eur": 5000.0}},
         "costes_acumulados": 0.0, "ordenes_pendientes": [], "creadas_el": None,
         "descartadas": [], "ahorro_acumulado": 0.0}

    grupos = pd.Series({"VIVO": "equity_us", "MUERTO": "etf"})
    cambio = portfolio.Cambio(close, pd.Series({"VIVO": "EUR", "MUERTO": "EUR"}))
    hoy = fechas[-1]

    valor_antes = portfolio._valorar(b, px, cambio, hoy)
    print(f"  valor antes de liquidar: {valor_antes:,.2f} € "
          f"(MUERTO contaba por su coste, 5.000 €)")

    cap = _Captura()
    log = logging.getLogger("qscan.portfolio")
    log.addHandler(cap)
    try:
        ops = portfolio._liquidar_sin_precio(
            b, px, cambio, grupos, portfolio.BROKERS["trade_republic"],
            hoy, "corto", "trade_republic")
    finally:
        log.removeHandler(cap)

    print(f"  operaciones generadas: {len(ops)} · posiciones restantes: "
          f"{sorted(b['posiciones'])}")
    if len(ops) != 1 or ops[0]["symbol"] != "MUERTO":
        fails.append(f"debería cerrarse sólo MUERTO, se cerraron {[o['symbol'] for o in ops]}")
    if "VIVO" not in b["posiciones"]:
        fails.append("se ha cerrado una posición que sí cotiza")
    if not cap.errores:
        fails.append("cerrar una posición sin precio no deja rastro en el log")

    esperado = ultimo_precio * 10.0
    obtenido = ops[0]["importe_eur"] if ops else 0.0
    print(f"  se vende al último precio conocido: {obtenido:,.2f} € "
          f"(bruto teórico {esperado:,.2f} € menos costes)")
    if not (esperado * 0.95 < obtenido <= esperado):
        fails.append(f"el importe {obtenido:,.2f} no corresponde al último "
                     f"precio conocido ({esperado:,.2f})")
    if abs(obtenido - 5000.0) < 1.0:
        fails.append("se ha liquidado al coste, no al último precio de mercado")

    valor_despues = portfolio._valorar(b, px, cambio, hoy)
    print(f"  valor después: {valor_despues:,.2f} € (ya marcado a mercado)")
    if abs(valor_despues - valor_antes) < 1.0:
        fails.append("la valoración no cambia: el coste seguía mandando")
    return fails


def prueba_proteger_almacen() -> list[str]:
    fails = []
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    est = {"escenarios": {"corto": {"brokers": {"trade_republic": {
        "posiciones": {"ASTY": {"unidades": 1.0, "coste_eur": 10.0}},
        "ordenes_pendientes": [{"symbol": "CRWL", "lado": "compra"}]}}}}}
    (TMP / "estado.json").write_text(json.dumps(est), encoding="utf-8")
    prot = portfolio.simbolos_en_cartera(TMP)
    print(f"  símbolos protegidos leídos del estado: {sorted(prot)}")
    if prot != {"ASTY", "CRWL"}:
        fails.append(f"se esperaban ASTY y CRWL, se leyó {sorted(prot)}")

    # y la poda del almacén los respeta
    # el almacén necesita bastantes símbolos: la poda tiene un freno que la
    # cancela si se llevaría más del 20%, y con dos símbolos siempre saltaría
    fechas = pd.bdate_range("2026-06-01", periods=500)
    vivos = [f"VIVO{i:02d}" for i in range(30)]
    filas = [pd.DataFrame({"symbol": s, "date": fechas, "open": 10.0, "high": 10.0,
                           "low": 10.0, "close": 10.0, "volume": 1e6})
             for s in vivos + ["ASTY"]]
    store = data.PriceStore(TMP / "prices.parquet")
    store.save(pd.concat(filas, ignore_index=True))
    uni = pd.DataFrame({"symbol": vivos, "name": "v", "group": "equity_us"})

    data.fetch_yahoo = lambda symbols, start, **kw: pd.concat(
        [pd.DataFrame({"symbol": s, "date": fechas[-3:], "open": 10.0, "high": 10.0,
                       "low": 10.0, "close": 10.0, "volume": 1e6}) for s in symbols],
        ignore_index=True)
    data.time.sleep = lambda s: None
    m = data.update(uni, store, recarga_completa=False, usar_reserva=False,
                    proteger=prot)
    quedan = set(m.symbol.unique())
    print(f"  tras podar con ASTY protegido: {len(quedan)} símbolos, "
          f"¿está ASTY? {'sí' if 'ASTY' in quedan else 'NO'}")
    if "ASTY" not in quedan:
        fails.append("la poda ha borrado los precios de una posición abierta")

    m2 = data.update(uni, store, recarga_completa=False, usar_reserva=False,
                     proteger=set())
    print(f"  sin proteger: {len(set(m2.symbol.unique()))} símbolos, "
          f"¿está ASTY? {'sí' if 'ASTY' in set(m2.symbol.unique()) else 'NO'}")
    if "ASTY" in set(m2.symbol.unique()):
        fails.append("sin protección tampoco poda: la prueba no prueba nada")
    return fails


def main() -> int:
    print("1. alarma de cartera estancada")
    fails = prueba_alarma_estancamiento()
    print("\n2. posiciones sin precio: se cierran a mercado, no a coste")
    fails += prueba_liquidar_sin_precio()
    print("\n3. la poda del almacén respeta lo que la cartera tiene")
    fails += prueba_proteger_almacen()
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
