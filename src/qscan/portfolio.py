"""Simulación de cartera: 40.000 € por escenario, con costes reales.

Se simulan cuatro carteras independientes, cada una con su propio capital de
40.000 €, para que sean directamente comparables entre sí y contra el índice:

  corto       15 posiciones, rebalanceo semanal
  medio       20 posiciones, rebalanceo mensual
  largo       20 posiciones, rebalanceo trimestral
  combinada   20 posiciones repartidas entre los tres horizontes según la
              evidencia que la validación encuentre en cada uno

Reglas que hacen que esto sea una simulación y no un folleto:

1. **Las órdenes se ejecutan a la APERTURA DE LA SESIÓN SIGUIENTE.** El ranking
   sale del cierre de hoy; nadie puede comprar a ese cierre. Ejecutar al mismo
   precio con el que decides es la forma más común y más silenciosa de inflar un
   backtest.
2. **Se cobran comisión y horquilla.** La comisión se recuerda; la horquilla es
   la que se come el resultado cuando se rota mucho.
3. **La divisa cuenta.** El capital es en euros y casi todo cotiza en dólares.
4. **Hay índice de referencia.** Sin comparar contra comprar y esperar, un
   mercado alcista hace que cualquier estrategia parezca buena.
5. **El pasado no se reescribe.** Cada día añade filas; nunca recalcula.
"""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CAPITAL_INICIAL = 40_000.0
BENCHMARK = "SPY"

# escenario -> (horizontes y su peso, nº de posiciones, frecuencia de rebalanceo)
ESCENARIOS: dict[str, dict] = {
    "corto": {"pesos": {"corto": 1.0}, "n": 15, "freq": "W",
              "etiqueta": "Corto plazo", "detalle": "15 posiciones · semanal"},
    "medio": {"pesos": {"medio": 1.0}, "n": 20, "freq": "M",
              "etiqueta": "Medio plazo", "detalle": "20 posiciones · mensual"},
    "largo": {"pesos": {"largo": 1.0}, "n": 20, "freq": "Q",
              "etiqueta": "Largo plazo", "detalle": "20 posiciones · trimestral"},
    "combinada": {"pesos": None, "n": 20, "freq": "M",
                  "etiqueta": "Combinada", "detalle": "20 posiciones · mensual"},
}


@dataclass
class Broker:
    """Modelo de costes. Cifras de agosto de 2026: estimaciones razonables, no un
    contrato. Cada bróker las cambia y varían por país y nivel de cuenta."""
    nombre: str
    comision_orden_eur: float = 0.0
    comision_etf_eur: float | None = None
    fx_deposito_pct: float = 0.0
    horquilla_bps: dict[str, float] = field(default_factory=dict)

    def comision(self, grupo: str) -> float:
        if grupo == "etf" and self.comision_etf_eur is not None:
            return self.comision_etf_eur
        return self.comision_orden_eur

    def horquilla(self, grupo: str) -> float:
        """Media horquilla en tanto por uno: lo que se pierde al cruzar."""
        return self.horquilla_bps.get(grupo, self.horquilla_bps.get("default", 15.0)) / 2e4


BROKERS = {
    "trade_republic": Broker(
        "Trade Republic", comision_orden_eur=1.0, fx_deposito_pct=0.0,
        horquilla_bps={"equity_us": 15, "equity_eu": 10, "etf": 8, "crypto": 100,
                       "commodity": 20, "bond": 12, "index": 20, "fx": 10,
                       "default": 15}),
    "etoro": Broker(
        "eToro", comision_orden_eur=1.80, comision_etf_eur=0.0,
        fx_deposito_pct=0.005,
        horquilla_bps={"equity_us": 10, "equity_eu": 15, "etf": 10, "crypto": 190,
                       "commodity": 25, "bond": 15, "index": 25, "fx": 15,
                       "default": 15}),
}


# --------------------------------------------------------------------------- #
# estado
# --------------------------------------------------------------------------- #
def _broker_vacio() -> dict:
    return {"efectivo": 0.0, "posiciones": {}, "costes_acumulados": 0.0,
            "ordenes_pendientes": [], "creadas_el": None,
            "descartadas": [], "ahorro_acumulado": 0.0}


def _estado_vacio() -> dict:
    return {
        "capital_inicial": CAPITAL_INICIAL,
        "inicio": None,
        "ultima_fecha": None,
        "escenarios": {
            e: {"ultimo_rebalanceo": None, "pesos_evidencia": None,
                "brokers": {k: _broker_vacio() for k in BROKERS}}
            for e in ESCENARIOS
        },
        "benchmark": {"unidades": 0.0, "invertido": 0.0, "pendiente": None},
    }


def cargar(ruta: Path) -> dict:
    if ruta.exists():
        est = json.loads(ruta.read_text(encoding="utf-8"))
        for e in ESCENARIOS:                    # tolerar escenarios nuevos
            est.setdefault("escenarios", {}).setdefault(
                e, {"ultimo_rebalanceo": None, "pesos_evidencia": None,
                    "brokers": {k: _broker_vacio() for k in BROKERS}})
        return est
    return _estado_vacio()


def a_json(o):
    """Convierte a tipos nativos lo que `json` no sabe escribir.

    ESTE ERA EL FALLO. El almacén guarda los precios en `float32` para que la
    historia entera quepa en la caché, así que todo lo que sale de las matrices
    de precios es `numpy.float32` y no `float`. `json.dumps` no lo serializa y
    lanza `TypeError: Object of type float32 is not JSON serializable`.

    Lo traicionero es DÓNDE saltaba: al guardar, o sea después de haber simulado
    el día entero correctamente. La simulación se ejecutaba, calculaba sus
    órdenes, y luego el estado se perdía al escribirlo. Y como la excepción
    estaba dentro de un `except` que sólo registraba un aviso, el informe seguía
    generándose impecable mientras la cartera llevaba semanas congelada en la
    misma fecha con las mismas órdenes pendientes.

    Se recorre la estructura entera en vez de usar `default=`: así también se
    convierten los NaN e infinitos, que `json` sí escribe pero produciendo un
    fichero que no es JSON válido y que al releerse rompería en otro sitio.
    """
    if isinstance(o, dict):
        return {str(k): a_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [a_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return [a_json(v) for v in o.tolist()]
    if o is None or o is pd.NaT:
        return None
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (pd.Timestamp, datetime.date, datetime.datetime)):
        return str(pd.Timestamp(o).date())
    if isinstance(o, str):
        return o
    return str(o)


def guardar(estado: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(a_json(estado), indent=1, ensure_ascii=False),
                    encoding="utf-8")


# --------------------------------------------------------------------------- #
# divisa
# --------------------------------------------------------------------------- #
class Cambio:
    """Convierte a euros. Sin esto, una cartera de acciones americanas medida en
    euros puede subir un 10% y parecer plana, o al revés."""

    def __init__(self, close: pd.DataFrame, monedas: pd.Series):
        self.monedas = monedas
        self.series = {"USD": close.get("EURUSD=X"), "GBP": close.get("EURGBP=X")}
        self._cache: dict[tuple, float] = {}

    def a_euros(self, importe: float, symbol: str, fecha) -> float:
        div = str(self.monedas.get(symbol, "USD")).upper()
        if div == "EUR":
            return importe
        clave = (div, fecha)
        if clave not in self._cache:
            s = self.series.get(div, self.series.get("USD"))
            tipo = np.nan
            if s is not None:
                prev = s[s.index <= fecha].dropna()
                if len(prev):
                    tipo = float(prev.iloc[-1])
            self._cache[clave] = tipo
        tipo = self._cache[clave]
        if not np.isfinite(tipo) or tipo <= 0:
            return importe
        return importe / tipo      # el par es EUR/XXX


# --------------------------------------------------------------------------- #
# ejecución
# --------------------------------------------------------------------------- #
def _siguiente_sesion(idx: pd.DatetimeIndex, fecha) -> pd.Timestamp | None:
    post = idx[idx > pd.Timestamp(fecha)]
    return post[0] if len(post) else None


def _ejecutar(b: dict, broker: Broker, px: dict, cambio: Cambio,
              grupos: pd.Series, hoy: pd.Timestamp, escenario: str,
              clave_broker: str) -> list[dict]:
    """Cruza las órdenes pendientes a la apertura de la sesión siguiente a
    aquella en que se decidieron."""
    if not b["ordenes_pendientes"] or not b["creadas_el"]:
        return []
    ses = _siguiente_sesion(px["open"].index, b["creadas_el"])
    if ses is None or ses > hoy:
        return []

    apertura, cierre = px["open"].loc[ses], px["close"].loc[ses]
    hechas = []
    # primero las ventas: liberan el efectivo que necesitan las compras
    for o in sorted(b["ordenes_pendientes"], key=lambda o: o["lado"] != "venta"):
        sym = o["symbol"]
        precio = apertura.get(sym, np.nan)
        if not np.isfinite(precio):
            precio = cierre.get(sym, np.nan)
        if not np.isfinite(precio) or precio <= 0:
            continue
        grupo = str(grupos.get(sym, "default"))
        h, com = broker.horquilla(grupo), broker.comision(grupo)

        if o["lado"] == "venta":
            uds = b["posiciones"].get(sym, {}).get("unidades", 0.0)
            if uds <= 0:
                continue
            bruto = cambio.a_euros(uds * precio * (1 - h), sym, ses)
            b["efectivo"] += bruto - com
            b["costes_acumulados"] += com + cambio.a_euros(uds * precio * h, sym, ses)
            b["posiciones"].pop(sym, None)
            hechas.append({"fecha": str(ses.date()), "escenario": escenario,
                           "broker": clave_broker, "symbol": sym, "lado": "venta",
                           "importe_eur": round(bruto, 2), "coste_eur": round(com, 2)})
        else:
            importe = min(float(o["importe_eur"]), b["efectivo"])
            if importe <= com:
                continue
            neto = importe - com
            precio_eur = cambio.a_euros(precio * (1 + h), sym, ses)
            if precio_eur <= 0:
                continue
            uds = neto / precio_eur          # fraccionadas: ambos brókers las permiten
            b["efectivo"] -= importe
            b["costes_acumulados"] += com + cambio.a_euros(uds * precio * h, sym, ses)
            pos = b["posiciones"].setdefault(sym, {"unidades": 0.0, "coste_eur": 0.0})
            pos["unidades"] += uds
            pos["coste_eur"] += neto
            hechas.append({"fecha": str(ses.date()), "escenario": escenario,
                           "broker": clave_broker, "symbol": sym, "lado": "compra",
                           "importe_eur": round(importe, 2), "coste_eur": round(com, 2)})

    b["ordenes_pendientes"] = []
    b["creadas_el"] = None
    return hechas


def _valorar(b: dict, px: dict, cambio: Cambio, hoy: pd.Timestamp) -> float:
    cierre = px["close"].loc[hoy]
    total = b["efectivo"]
    for sym, pos in b["posiciones"].items():
        p = cierre.get(sym, np.nan)
        total += cambio.a_euros(pos["unidades"] * p, sym, hoy) if np.isfinite(p) \
            else pos["coste_eur"]
    return float(total)


# --------------------------------------------------------------------------- #
# selección
# --------------------------------------------------------------------------- #
def pesos_por_evidencia(verdict: pd.DataFrame | None) -> tuple[dict[str, float], str]:
    """Reparte el escenario combinado según lo que la validación encuentre.

    Se pondera cada horizonte por su t-stat medio por encima de 1: por debajo de
    ahí no hay nada que distinga la señal del azar y no merece capital. Si ningún
    horizonte llega, se reparte a partes iguales y se dice explícitamente.

    Caveat honesto: los pesos salen de la misma validación que mide el sistema,
    así que hay algo de ajuste a los propios datos. Se mitiga usando un umbral
    duro (t>1) en vez de optimizar libremente, pero no desaparece.
    """
    igual = {h: 1 / 3 for h in ("corto", "medio", "largo")}
    if verdict is None or verdict.empty or "t_stat" not in verdict.columns:
        return igual, "sin validación disponible: reparto a partes iguales"
    fuerza = {}
    for h in ("corto", "medio", "largo"):
        sub = verdict[verdict.horizonte == h]
        t = pd.to_numeric(sub.t_stat, errors="coerce").dropna()
        fuerza[h] = max(float(t.mean()) - 1.0, 0.0) if len(t) else 0.0
    total = sum(fuerza.values())
    if total <= 0:
        return igual, "ningún horizonte supera t=1: reparto a partes iguales"
    pesos = {h: v / total for h, v in fuerza.items()}
    detalle = " · ".join(f"{h} {p*100:.0f}%" for h, p in pesos.items() if p > 0)
    return pesos, f"por evidencia de la validación ({detalle})"


def seleccionar(scored: pd.DataFrame, anomalias: pd.DataFrame | None,
                close: pd.DataFrame, n: int, pesos: dict[str, float]) -> list[str]:
    """Los n mejores según los horizontes indicados, sin duplicar apuestas."""
    from . import scoring

    ultima = scored[scored.date == scored.date.max()]
    if anomalias is not None:
        malos = set(anomalias.index[anomalias.get("cuarentena", False)])
        ultima = ultima[~ultima.symbol.isin(malos)]
    tam = ultima.group.value_counts()
    ultima = ultima[ultima.group.isin(tam[tam >= scoring.MIN_PEERS].index)]

    elegidos: list[str] = []
    for h, peso in sorted(pesos.items(), key=lambda kv: -kv[1]):
        cupo = int(round(n * peso))
        if cupo <= 0 or f"score_{h}" not in ultima.columns:
            continue
        cand = (ultima.dropna(subset=[f"score_{h}"])
                      .sort_values([f"pct_{h}", f"score_{h}"], ascending=False)
                      .symbol.tolist())
        cand = [s for s in cand if s not in elegidos]
        red = scoring.redundancy(cand[:cupo * 3], close, max_corr=0.80)
        elegidos += [s for s in cand if s not in red][:cupo]
    if len(elegidos) < n:      # rellenar con lo mejor del horizonte dominante
        dom = max(pesos, key=pesos.get)
        resto = (ultima.dropna(subset=[f"score_{dom}"])
                       .sort_values([f"pct_{dom}", f"score_{dom}"], ascending=False)
                       .symbol.tolist())
        elegidos += [s for s in resto if s not in elegidos][:n - len(elegidos)]
    return elegidos[:n]


def _toca_rebalanceo(ultimo: str | None, hoy: pd.Timestamp, freq: str) -> bool:
    if ultimo is None:
        return True
    u = pd.Timestamp(ultimo)
    if freq == "W":
        return hoy.isocalendar()[:2] != u.isocalendar()[:2]
    if freq == "Q":
        return (hoy.year, (hoy.month - 1) // 3) != (u.year, (u.month - 1) // 3)
    return (hoy.year, hoy.month) != (u.year, u.month)


# --------------------------------------------------------------------------- #
# optimización: operar sólo cuando el beneficio esperado cubre el coste
# --------------------------------------------------------------------------- #
MARGEN_SEGURIDAD = 1.5     # el beneficio esperado debe superar 1,5x el coste
COMISION_MAX_PCT = 0.001   # una comisión no puede pasar del 0,1% de la operación
BANDA_NO_OPERAR = 0.25     # no se ajusta un peso que se desvía menos de un 25%
PESO_MIN, PESO_MAX = 0.5, 2.0   # límites del peso relativo a la equiponderación


def alfa_esperado(scored: pd.DataFrame, verdict: pd.DataFrame | None,
                  close: pd.DataFrame, pesos: dict[str, float]) -> pd.Series:
    """Rentabilidad esperada de cada activo en su horizonte, en tanto por uno.

    Se usa la aproximación de Grinold:  E[r] = IC x z x sigma

    donde `IC` es el coeficiente de información MEDIDO por la validación para ese
    grupo y horizonte, `z` es el score transversal del activo y `sigma` la
    dispersión de rentabilidades en ese horizonte, estimada con la volatilidad
    realizada.

    Que el IC salga de la validación y no de una corazonada es lo que hace útil
    todo esto: si un grupo no tiene señal, su IC ronda cero, el alfa esperado
    ronda cero, y el optimizador concluye que ninguna rotación compensa la
    comisión. Es decir, el sistema deja de operar solo cuando no sabe nada. Un IC
    negativo se trata como cero: invertir la señal porque el backtest salió al
    revés es una de las formas más fáciles de sobreajustar.
    """
    ultima = scored[scored.date == scored.date.max()]
    if ultima.empty:
        return pd.Series(dtype=float)

    # volatilidad anualizada por activo, para escalar al horizonte
    r = np.log(close / close.shift(1)).tail(252)
    vol_anual = r.std() * np.sqrt(252)

    ic = {}
    if verdict is not None and not verdict.empty and "ic_medio" in verdict.columns:
        for _, row in verdict.iterrows():
            ic[(str(row.horizonte), str(row.grupo))] = float(row.ic_medio)

    total = pd.Series(0.0, index=ultima.symbol)
    for h, peso in pesos.items():
        col = f"score_{h}"
        if peso <= 0 or col not in ultima.columns:
            continue
        dias = HORIZONTE_DIAS.get(h, 63)
        for grupo, sub in ultima.groupby("group"):
            z = sub[col]
            if z.notna().sum() < 5:
                continue
            z = (z - z.mean()) / (z.std(ddof=0) or 1.0)
            ic_gh = max(ic.get((h, str(grupo)), 0.0), 0.0)
            sigma = vol_anual.reindex(sub.symbol).fillna(0.25) * np.sqrt(dias / 252)
            aporte = ic_gh * z.to_numpy() * sigma.to_numpy() * peso
            total.loc[sub.symbol] = total.reindex(sub.symbol).fillna(0.0).to_numpy() + aporte
    return total.groupby(level=0).first()


HORIZONTE_DIAS = {"corto": 10, "medio": 63, "largo": 252}


def _coste_por_euro(broker: Broker, grupo: str, importe: float) -> float:
    """Coste de cruzar, en tanto por uno del importe: comisión + media horquilla."""
    if importe <= 0:
        return 1.0
    return broker.comision(grupo) / importe + broker.horquilla(grupo)


def plan_ordenes(b: dict, broker: Broker, objetivo: list[str], alfa: pd.Series,
                 valor: float, grupos: pd.Series, n: int) -> tuple[list[dict], list[dict]]:
    """Decide qué operar de verdad. Devuelve (órdenes, descartadas).

    Tres filtros, en este orden:

    1. **Tamaño mínimo.** Con 1 € de comisión, una compra de 200 € paga un 0,5%
       antes de empezar. Se exige que la comisión no pase del 0,1% del importe.
    2. **Banda de no operar.** Si el peso actual sólo se desvía un poco del
       objetivo, se deja quieto: reequilibrar por reequilibrar es pagar por nada.
    3. **El beneficio esperado tiene que cubrir el coste con margen.** Cambiar A
       por B sólo compensa si la diferencia de alfa supera el coste de vender A y
       comprar B, multiplicado por un margen de seguridad, porque el alfa es una
       estimación ruidosa y operar al filo es perder de media.
    """
    ordenes, descartadas = [], []
    posiciones = b.get("posiciones", {})
    inicial = not posiciones

    # --- pesos objetivo por alfa, acotados alrededor de la equiponderación ---
    a_obj = alfa.reindex(objetivo).fillna(0.0).clip(lower=0.0)
    base = 1.0 / max(len(objetivo), 1)
    if a_obj.sum() > 0:
        w = a_obj / a_obj.sum()
        w = w.clip(base * PESO_MIN, base * PESO_MAX)
        w = w / w.sum()
    else:
        w = pd.Series(base, index=objetivo)   # sin alfa medible: equiponderar

    minimo = broker.comision_orden_eur / COMISION_MAX_PCT if COMISION_MAX_PCT else 0.0

    # --- salidas -----------------------------------------------------------
    for sym in list(posiciones):
        if sym in objetivo:
            continue
        grupo = str(grupos.get(sym, "default"))
        importe = posiciones[sym].get("coste_eur", 0.0)
        a_fuera = float(alfa.get(sym, 0.0))
        # ¿merece la pena salir? se compara contra el mejor candidato disponible
        mejor = float(a_obj.max()) if len(a_obj) else 0.0
        coste = (_coste_por_euro(broker, grupo, importe)
                 + _coste_por_euro(broker, str(grupos.get(a_obj.idxmax(), "default"))
                                   if len(a_obj) else "default", importe))
        if not inicial and (mejor - a_fuera) < MARGEN_SEGURIDAD * coste:
            descartadas.append({"symbol": sym, "lado": "venta", "importe_eur": importe,
                                "motivo": "el candidato no supera el coste del cambio",
                                "coste_evitado_eur": round(coste * importe, 2)})
            continue
        ordenes.append({"symbol": sym, "lado": "venta"})

    # --- entradas y ajustes -------------------------------------------------
    for sym in objetivo:
        grupo = str(grupos.get(sym, "default"))
        actual = posiciones.get(sym, {}).get("coste_eur", 0.0)
        deseado = float(w.get(sym, base)) * valor
        delta = deseado - actual

        if actual > 0:                       # ya lo tenemos: ¿hace falta ajustar?
            if abs(delta) < BANDA_NO_OPERAR * max(actual, 1.0):
                continue
            if abs(delta) < minimo:
                descartadas.append({"symbol": sym, "lado": "ajuste",
                                    "importe_eur": round(abs(delta), 2),
                                    "motivo": "ajuste por debajo del mínimo rentable",
                                    "coste_evitado_eur": round(broker.comision(grupo), 2)})
                continue
            if delta < 0:
                continue                     # recortes: se dejan correr, no compensan
            ordenes.append({"symbol": sym, "lado": "compra",
                            "importe_eur": round(delta, 2)})
            continue

        if deseado < minimo:
            descartadas.append({"symbol": sym, "lado": "compra",
                                "importe_eur": round(deseado, 2),
                                "motivo": "posición demasiado pequeña para su comisión",
                                "coste_evitado_eur": round(broker.comision(grupo), 2)})
            continue
        coste = _coste_por_euro(broker, grupo, deseado)
        if not inicial and float(alfa.get(sym, 0.0)) < MARGEN_SEGURIDAD * coste:
            descartadas.append({"symbol": sym, "lado": "compra",
                                "importe_eur": round(deseado, 2),
                                "motivo": "alfa esperado por debajo del coste de entrada",
                                "coste_evitado_eur": round(coste * deseado, 2)})
            continue
        ordenes.append({"symbol": sym, "lado": "compra",
                        "importe_eur": round(deseado, 2)})
    return ordenes, descartadas


# --------------------------------------------------------------------------- #
# simulación
# --------------------------------------------------------------------------- #
def simular(scored: pd.DataFrame, px: dict, universo: pd.DataFrame,
            anomalias: pd.DataFrame | None, verdict: pd.DataFrame | None,
            dir_salida: Path) -> dict:
    """Avanza un día. Idempotente: repetir el mismo día no cambia nada."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    r_estado, r_curva, r_ops = (dir_salida / "estado.json",
                                dir_salida / "curva.csv",
                                dir_salida / "operaciones.csv")
    estado = cargar(r_estado)
    hoy = pd.Timestamp(px["close"].index.max())
    if estado.get("ultima_fecha") == str(hoy.date()):
        log.info("simulación ya al día (%s)", hoy.date())
        estado["_ops_hoy"] = []
        estado["_pendientes"] = _pendientes(estado)
        return estado

    grupos = universo.set_index("symbol")["group"]
    monedas = universo.set_index("symbol").get("currency", pd.Series(dtype=str))
    cambio = Cambio(px["close"], monedas)

    if estado["inicio"] is None:
        estado["inicio"] = str(hoy.date())
        for e in ESCENARIOS:
            for k, br in BROKERS.items():
                neto = CAPITAL_INICIAL * (1 - br.fx_deposito_pct)
                b = estado["escenarios"][e]["brokers"][k]
                b["efectivo"] = neto
                b["costes_acumulados"] = CAPITAL_INICIAL - neto

    # 1. cruzar lo pendiente
    ops = []
    for e in ESCENARIOS:
        for k, br in BROKERS.items():
            ops += _ejecutar(estado["escenarios"][e]["brokers"][k], br, px,
                             cambio, grupos, hoy, e, k)

    # 2. índice de referencia
    _benchmark(estado, px, cambio, hoy)

    # 3. decidir rebalanceos (se cruzan mañana)
    pesos_comb, nota = pesos_por_evidencia(verdict)
    descartes_hoy: list[dict] = []
    for e, cfg in ESCENARIOS.items():
        esc = estado["escenarios"][e]
        if not _toca_rebalanceo(esc["ultimo_rebalanceo"], hoy, cfg["freq"]):
            continue
        pesos = cfg["pesos"] or pesos_comb
        objetivo = seleccionar(scored, anomalias, px["close"], cfg["n"], pesos)
        if not objetivo:
            continue
        alfa = alfa_esperado(scored, verdict, px["close"], pesos)
        for k, br in BROKERS.items():
            b = esc["brokers"][k]
            valor = _valorar(b, px, cambio, hoy)
            ordenes, descartadas = plan_ordenes(b, br, objetivo, alfa, valor,
                                                grupos, cfg["n"])
            b["ordenes_pendientes"] = ordenes
            b["creadas_el"] = str(hoy.date()) if ordenes else None
            b["descartadas"] = descartadas
            ahorro = sum(d.get("coste_evitado_eur", 0) or 0 for d in descartadas)
            b["ahorro_acumulado"] = round(b.get("ahorro_acumulado", 0.0) + ahorro, 2)
            descartes_hoy.append({"escenario": e, "broker": k,
                                  "n": len(descartadas), "ahorro_eur": round(ahorro, 2)})
        esc["ultimo_rebalanceo"] = str(hoy.date())
        if e == "combinada":
            esc["pesos_evidencia"] = {"pesos": pesos_comb, "nota": nota}

    # 4. valorar y registrar
    filas = []
    for e in ESCENARIOS:
        for k in BROKERS:
            b = estado["escenarios"][e]["brokers"][k]
            v = _valorar(b, px, cambio, hoy)
            filas.append({"fecha": str(hoy.date()), "escenario": e, "broker": k,
                          "valor_eur": round(v, 2),
                          "efectivo_eur": round(b["efectivo"], 2),
                          "posiciones": len(b["posiciones"]),
                          "costes_acum_eur": round(b["costes_acumulados"], 2),
                          "rentabilidad_pct": round((v / CAPITAL_INICIAL - 1) * 100, 3)})
    bm = estado["benchmark"]
    if bm["unidades"] > 0:
        p = px["close"].loc[hoy].get(BENCHMARK, np.nan)
        v = cambio.a_euros(bm["unidades"] * p, BENCHMARK, hoy) if np.isfinite(p) \
            else CAPITAL_INICIAL
        filas.append({"fecha": str(hoy.date()), "escenario": "benchmark",
                      "broker": "benchmark", "valor_eur": round(v, 2),
                      "efectivo_eur": 0.0, "posiciones": 1, "costes_acum_eur": 1.0,
                      "rentabilidad_pct": round((v / CAPITAL_INICIAL - 1) * 100, 3)})

    _anexar(r_curva, pd.DataFrame(filas), clave=["fecha", "escenario", "broker"])
    if ops:
        _anexar(r_ops, pd.DataFrame(ops))

    estado["ultima_fecha"] = str(hoy.date())
    guardar(estado, r_estado)
    estado["_ops_hoy"] = ops
    estado["_pendientes"] = _pendientes(estado)
    estado["_descartes"] = descartes_hoy
    return estado


def _pendientes(estado: dict) -> list[dict]:
    """Órdenes decididas hoy que se cruzarán en la próxima apertura. Es el parte
    de operaciones que el informe tiene que enseñar: lo que habría que hacer."""
    out = []
    for e, esc in estado.get("escenarios", {}).items():
        for k, b in esc.get("brokers", {}).items():
            for o in b.get("ordenes_pendientes", []):
                out.append({"escenario": e, "broker": k, "symbol": o["symbol"],
                            "lado": o["lado"],
                            "importe_eur": o.get("importe_eur"),
                            "decidida_el": b.get("creadas_el")})
    return out


def _benchmark(estado: dict, px: dict, cambio: Cambio, hoy: pd.Timestamp) -> None:
    bm = estado["benchmark"]
    if bm["unidades"] > 0 or BENCHMARK not in px["close"].columns:
        return
    if bm.get("pendiente"):
        ses = _siguiente_sesion(px["open"].index, bm["pendiente"])
        if ses is not None and ses <= hoy:
            p = px["open"].loc[ses].get(BENCHMARK, np.nan)
            if np.isfinite(p):
                br = BROKERS["trade_republic"]
                neto = CAPITAL_INICIAL - br.comision("etf")
                peur = cambio.a_euros(p * (1 + br.horquilla("etf")), BENCHMARK, ses)
                bm["unidades"], bm["invertido"] = neto / peur, CAPITAL_INICIAL
                bm["pendiente"] = None
    else:
        bm["pendiente"] = str(hoy.date())


def _anexar(ruta: Path, df: pd.DataFrame, clave: list[str] | None = None) -> None:
    """Añade filas sin reescribir el pasado.

    `clave` sólo se usa en la curva, con una fila por día/escenario/bróker. El
    registro de operaciones NO lleva clave: en un rebalanceo hay veinte compras
    del mismo bróker el mismo día y deduplicar se las cargaría todas menos una.
    """
    if ruta.exists():
        viejo = pd.read_csv(ruta)
        df = pd.concat([viejo, df], ignore_index=True)
        df = df.drop_duplicates(subset=clave, keep="last") if clave \
            else df.drop_duplicates()
    df.to_csv(ruta, index=False)


def commitear(dir_salida: Path) -> bool:
    """Guarda el estado en el repositorio y DICE si lo consiguió.

    La versión anterior se tragaba cualquier fallo en silencio. Si el push no
    funcionaba, la simulación se recalculaba entera cada día y nunca persistía,
    con el agravante de que el log no lo mencionaba: otra vez el mismo patrón de
    fallo silencioso que este proyecto ya ha pagado dos veces.
    """
    def _git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True)

    try:
        _git("config", "user.name", "github-actions[bot]")
        _git("config", "user.email",
             "41898282+github-actions[bot]@users.noreply.github.com")
        _git("add", str(dir_salida))
        r = _git("commit", "-m", "Actualizar simulación de cartera")
        if r.returncode != 0:
            if "nothing to commit" in (r.stdout + r.stderr):
                log.info("simulación sin cambios que guardar")
            else:
                log.error("no se pudo crear el commit de la simulación: %s",
                          (r.stdout + r.stderr).strip()[:300])
            return False
        p = _git("push")
        if p.returncode != 0:
            log.error("EL COMMIT DE LA SIMULACIÓN NO SE PUDO SUBIR: %s. "
                      "El estado se perderá y mañana se empezará de cero.",
                      (p.stdout + p.stderr).strip()[:300])
            return False
        log.info("simulación guardada y subida al repositorio")
        return True
    except Exception as e:  # pragma: no cover
        log.error("fallo al guardar la simulación: %s", e, exc_info=True)
        return False


def resumen(dir_salida: Path) -> pd.DataFrame:
    ruta = dir_salida / "curva.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


def operaciones(dir_salida: Path) -> pd.DataFrame:
    ruta = dir_salida / "operaciones.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


def composicion(estado: dict, px: dict, universo: pd.DataFrame,
                broker: str = "trade_republic") -> pd.DataFrame:
    """Qué tiene dentro cada cartera hoy.

    Se muestra la de un solo bróker porque ambos compran exactamente los mismos
    activos: el objetivo lo fija la señal, no el bróker. Lo que cambia entre
    ellos es el coste, y eso ya está en la tabla de escenarios.
    """
    hoy = pd.Timestamp(px["close"].index.max())
    cierre = px["close"].loc[hoy]
    monedas = universo.set_index("symbol").get("currency", pd.Series(dtype=str))
    nombres = universo.set_index("symbol").get("name", pd.Series(dtype=str))
    grupos = universo.set_index("symbol").get("group", pd.Series(dtype=str))
    cambio = Cambio(px["close"], monedas)

    filas = []
    for e in ESCENARIOS:
        b = estado.get("escenarios", {}).get(e, {}).get("brokers", {}).get(broker)
        if not b:
            continue
        valor_total = _valorar(b, px, cambio, hoy) or 1.0
        for sym, pos in sorted(b.get("posiciones", {}).items()):
            p = cierre.get(sym, np.nan)
            valor = cambio.a_euros(pos["unidades"] * p, sym, hoy) \
                if np.isfinite(p) else pos["coste_eur"]
            coste = pos["coste_eur"] or np.nan
            filas.append({
                "escenario": e, "symbol": sym,
                "nombre": str(nombres.get(sym, ""))[:44],
                "grupo": str(grupos.get(sym, "")),
                "valor_eur": round(valor, 2),
                "coste_eur": round(pos["coste_eur"], 2),
                "peso_pct": round(valor / valor_total * 100, 2),
                "plusvalia_eur": round(valor - pos["coste_eur"], 2),
                "plusvalia_pct": round((valor / coste - 1) * 100, 2)
                if np.isfinite(coste) and coste else np.nan,
            })
        if b.get("efectivo", 0) > 1:
            filas.append({"escenario": e, "symbol": "· efectivo",
                          "nombre": "sin invertir", "grupo": "",
                          "valor_eur": round(b["efectivo"], 2), "coste_eur": None,
                          "peso_pct": round(b["efectivo"] / valor_total * 100, 2),
                          "plusvalia_eur": None, "plusvalia_pct": None})
    return pd.DataFrame(filas)
