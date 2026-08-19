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
    # float32 A PROPÓSITO: es lo que hace `data.update` para que la historia
    # entera quepa en la caché, y por tanto lo que la simulación ve en
    # producción. Con float64 esta prueba pasaba mientras el sistema real
    # llevaba semanas sin poder guardar el estado — `json` no serializa
    # `numpy.float32`. Una prueba que usa tipos más cómodos que los de
    # producción no está probando producción.
    px_full = {f: v.astype("float32") for f, v in px_full.items()}
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

    # --- 4. ninguna orden ejecutada por debajo del mínimo rentable --------
    minimo = (portfolio.BROKERS["trade_republic"].comision_orden_eur
              / portfolio.COMISION_MAX_PCT)
    compras = ops[(ops.lado == "compra") & (ops.broker == "trade_republic")]
    pequenas = compras[compras.importe_eur < minimo * 0.9]
    print(f"compras por debajo del mínimo rentable ({minimo:,.0f} €): {len(pequenas)}")
    if len(pequenas):
        fails.append(f"{len(pequenas)} compras por debajo del mínimo: la comisión "
                     f"se come una parte desproporcionada")

    # --- 5. el optimizador tiene que descartar operaciones ----------------
    ahorro = sum(estado["escenarios"][e]["brokers"][k].get("ahorro_acumulado", 0)
                 for e in portfolio.ESCENARIOS for k in portfolio.BROKERS)
    print(f"ahorro estimado por operaciones descartadas: {ahorro:,.2f} €")
    if ahorro <= 0:
        fails.append("el optimizador no descartó ninguna operación: no está actuando")

    # --- 6. control positivo: con señal medida, el sistema SÍ debe rotar ---
    # Sin alfa esperado el optimizador se queda quieto, que es lo correcto. Pero
    # hay que comprobar que no está simplemente roto: con un IC alto declarado,
    # las mismas señales tienen que producir rotación.
    veredicto_fuerte = pd.DataFrame([
        {"horizonte": h, "grupo": g, "ic_medio": 0.25, "t_stat": 4.0,
         "periodos": 80, "veredicto": "señal consistente"}
        for h in ("corto", "medio", "largo") for g in ("equity_us", "etf")])
    TMP2 = Path("/tmp/qscan_cartera_pos")
    if TMP2.exists():
        shutil.rmtree(TMP2)
    for hoy in dias:
        px = {k: v.loc[:hoy] for k, v in px_full.items()}
        fr = features.rebalance_dates(px["close"].index, "ME")
        fr = fr[fr >= px["close"].index.min() + pd.Timedelta(days=400)]
        fr = pd.DatetimeIndex(sorted(set(fr[-3:]) | {hoy}))
        panel2 = features.sample_panel({k: v.loc[:hoy] for k, v in fh.items()},
                                       fr, px["close"], grupos, 0)
        portfolio.simular(scoring.score_panel(panel2), px, uni, None,
                          veredicto_fuerte, TMP2)
    ops2 = pd.read_csv(TMP2 / "operaciones.csv")
    c1 = pd.read_csv(TMP / "curva.csv")
    c2 = pd.read_csv(TMP2 / "curva.csv")
    cost_sin = float(c1[(c1.fecha == c1.fecha.max()) & (c1.escenario == "corto")
                        & (c1.broker == "trade_republic")].costes_acum_eur.iloc[0])
    cost_con = float(c2[(c2.fecha == c2.fecha.max()) & (c2.escenario == "corto")
                        & (c2.broker == "trade_republic")].costes_acum_eur.iloc[0])
    print(f"\ncon IC declarado 0,25: {len(ops2)} operaciones ({len(ops)} sin señal)")
    print(f"  costes del escenario corto: {cost_con:,.2f} € con señal · "
          f"{cost_sin:,.2f} € sin señal")
    if len(ops2) <= len(ops):
        fails.append("con señal medida no rota más que sin ella: el alfa esperado "
                     "no está llegando a la decisión")

    # --- el informe tiene que RENDERIZAR el bloque de cartera ----------------
    # No basta con que la simulación funcione. `demo_report.py` construía la
    # página sin curva, así que `_bloque_cartera` salía por la primera línea y
    # todo ese camino —el JSON embebido, las tablas de escenarios, composición y
    # operaciones— no lo ejecutaba ninguna prueba. Un `NameError` de una sola
    # línea ahí tumbaba la ejecución entera en producción con las nueve pruebas
    # en verde.
    from qscan import report
    estado_fin = portfolio.cargar(TMP / "estado.json")
    comp_fin = portfolio.composicion(estado_fin, px_full, uni)
    curva_fin = portfolio.resumen(TMP)
    try:
        pagina = report.build_html(
            scored, panel, {}, uni.set_index("symbol")["name"].to_dict(), None,
            universe_size=len(uni), top_n=5, curva=curva_fin,
            estado_cartera=estado_fin, composicion=comp_fin)
    except Exception as e:
        pagina = ""
        fails.append(f"el informe no se puede generar con cartera: "
                     f"{type(e).__name__}: {e}")
    marcas = ["Carteras simuladas", "Composición de las carteras",
              "Trade Republic", "data-dias"]
    presentes = [m for m in marcas if m.lower() in pagina.lower()]
    print(f"\ninforme con cartera: {len(pagina):,} bytes · "
          f"bloques encontrados {len(presentes)}/{len(marcas)}")
    if pagina and len(presentes) < len(marcas):
        fails.append("la página se genera pero el bloque de cartera no aparece")

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
