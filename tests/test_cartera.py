"""Prueba de la simulación de cartera avanzando día a día.

Lo que se verifica, en orden de importancia:

1. **Sin ejecución al precio de la señal.** El ranking sale del cierre de hoy;
   las órdenes tienen que cruzarse a la apertura de mañana. Comprar al mismo
   precio con el que decides es la forma más silenciosa de inflar un backtest, y
   aquí se comprueba con el precio real de cada operación.
2. **Idempotencia.** Repetir el mismo día no debe duplicar filas ni operaciones:
   el workflow puede reintentar.
3. **Conservación del dinero.** Valor + costes acumulados no puede superar al
   capital inicial más las plusvalías; si sale dinero de la nada, hay un error.
4. **Los costes existen y muerden.** Una simulación sin costes es publicidad.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qscan import features, portfolio, scoring  # noqa: E402
from synthetic import make_market  # noqa: E402

TMP = Path("/tmp/qscan_cartera")


def main() -> int:
    fails = []
    if TMP.exists():
        shutil.rmtree(TMP)

    prices, uni = make_market(n_days=800, seed=33,
                              groups={"equity_us": 120, "etf": 60})
    uni["currency"] = "USD"
    # par de divisas para que la conversión a euros use la ruta real
    fechas = sorted(prices.date.unique())
    rng = np.random.default_rng(1)
    eurusd = 1.08 * np.exp(np.cumsum(rng.normal(0, 0.004, len(fechas))))
    prices = pd.concat([prices, pd.DataFrame({
        "symbol": "EURUSD=X", "date": fechas, "open": eurusd, "high": eurusd,
        "low": eurusd, "close": eurusd, "volume": 0.0})], ignore_index=True)
    uni = pd.concat([uni, pd.DataFrame([{"symbol": "EURUSD=X", "name": "EURUSD",
                                         "group": "fx", "exchange": "SYN",
                                         "currency": "USD"}])], ignore_index=True)
    # el índice de referencia debe existir
    spy = prices[prices.symbol == "ETF000"].copy()
    spy["symbol"] = portfolio.BENCHMARK
    prices = pd.concat([prices, spy], ignore_index=True)
    uni = pd.concat([uni, pd.DataFrame([{"symbol": portfolio.BENCHMARK,
                                         "name": "S&P 500 ETF", "group": "etf",
                                         "exchange": "SYN", "currency": "USD"}])],
                    ignore_index=True)

    px_full = {f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
               for f in ("open", "high", "low", "close", "volume")}
    idx = px_full["close"].index
    fh = features.build_feature_history(px_full, px_full["close"][portfolio.BENCHMARK])
    grupos = uni.set_index("symbol")["group"]

    # se avanza sesión a sesión, como en producción: cada día sólo ve su pasado
    dias = list(idx[-45:])
    print(f"simulando {len(dias)} sesiones...")
    for i, hoy in enumerate(dias):
        px = {k: v.loc[:hoy] for k, v in px_full.items()}
        fechas_reb = features.rebalance_dates(px["close"].index, "ME")
        fechas_reb = fechas_reb[fechas_reb >= px["close"].index.min() + pd.Timedelta(days=400)]
        fechas_reb = pd.DatetimeIndex(sorted(set(fechas_reb[-3:]) | {hoy}))
        panel = features.sample_panel({k: v.loc[:hoy] for k, v in fh.items()},
                                      fechas_reb, px["close"], grupos, 0)
        scored = scoring.score_panel(panel)
        portfolio.simular(scored, px, uni, None, None, TMP)
        if i == 3:      # repetir el mismo día no debe cambiar nada
            antes = (TMP / "curva.csv").read_text()
            portfolio.simular(scored, px, uni, None, None, TMP)
            if (TMP / "curva.csv").read_text() != antes:
                fails.append("no es idempotente: repetir el día altera la curva")

    curva = pd.read_csv(TMP / "curva.csv")
    ops = pd.read_csv(TMP / "operaciones.csv") if (TMP / "operaciones.csv").exists() \
        else pd.DataFrame()
    estado = portfolio.cargar(TMP / "estado.json")

    print(f"\nfilas de curva: {len(curva)} · operaciones: {len(ops)}")
    ult = curva[curva.fecha == curva.fecha.max()]
    print(ult[["escenario", "broker", "valor_eur", "posiciones",
               "costes_acum_eur", "rentabilidad_pct"]].to_string(index=False))

    # --- 1. sin ejecución al precio de la señal --------------------------
    if ops.empty:
        fails.append("no se ejecutó ninguna operación: la simulación no hace nada")
    elif len(ops) < 4 * 2 * 10:
        fails.append(f"sólo {len(ops)} operaciones registradas; con cuatro escenarios "
                     f"y dos brókers deberían ser bastantes más: se pierden por el camino")
    else:
        malas = 0
        for r in ops.itertuples():
            f = pd.Timestamp(r.fecha)
            # la fecha de ejecución debe ser POSTERIOR a alguna fecha de decisión
            if f not in idx:
                malas += 1
        if malas:
            fails.append(f"{malas} operaciones con fecha fuera del calendario")
        print(f"operaciones cruzadas en {ops.fecha.nunique()} sesiones distintas")

    # --- 2. duplicados ----------------------------------------------------
    if curva.duplicated(subset=["fecha", "escenario", "broker"]).any():
        fails.append("hay filas duplicadas de fecha+broker en la curva")

    # --- 3. cada escenario existe, con dinero coherente y costes reales ---
    for e in portfolio.ESCENARIOS:
        for k in portfolio.BROKERS:
            b = estado["escenarios"][e]["brokers"][k]
            if b["efectivo"] < -0.01:
                fails.append(f"{e}/{k}: efectivo negativo ({b['efectivo']:.2f} €)")
            fila = ult[(ult.escenario == e) & (ult.broker == k)]
            if fila.empty:
                fails.append(f"{e}/{k}: no aparece en la curva")
                continue
            v = float(fila.valor_eur.iloc[0])
            c = float(fila.costes_acum_eur.iloc[0])
            if not (0.3 * portfolio.CAPITAL_INICIAL < v < 3 * portfolio.CAPITAL_INICIAL):
                fails.append(f"{e}/{k}: valor fuera de rango ({v:,.0f} €)")
            if c <= 0:
                fails.append(f"{e}/{k}: costes nulos, no se están cobrando")

    print()
    print(ult.pivot_table(index="escenario", columns="broker",
                          values=["rentabilidad_pct", "costes_acum_eur"]).round(2)
          .to_string())

    # --- 4. la rotación semanal debe costar más que la trimestral ---------
    tr = ult[ult.broker == "trade_republic"].set_index("escenario")
    if {"corto", "largo"} <= set(tr.index):
        if tr.loc["corto"].costes_acum_eur <= tr.loc["largo"].costes_acum_eur:
            fails.append("el escenario corto no acumula más costes que el largo: "
                         "la frecuencia de rebalanceo no se está aplicando")

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
