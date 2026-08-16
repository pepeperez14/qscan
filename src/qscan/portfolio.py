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
            "ordenes_pendientes": [], "creadas_el": None}


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


def guardar(estado: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(estado, indent=1, ensure_ascii=False), encoding="utf-8")


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
    for e, cfg in ESCENARIOS.items():
        esc = estado["escenarios"][e]
        if not _toca_rebalanceo(esc["ultimo_rebalanceo"], hoy, cfg["freq"]):
            continue
        pesos = cfg["pesos"] or pesos_comb
        objetivo = seleccionar(scored, anomalias, px["close"], cfg["n"], pesos)
        if not objetivo:
            continue
        for k in BROKERS:
            b = esc["brokers"][k]
            valor = _valorar(b, px, cambio, hoy)
            fuera = [s for s in b["posiciones"] if s not in objetivo]
            dentro = [s for s in objetivo if s not in b["posiciones"]]
            por_pos = valor / max(len(objetivo), 1)
            b["ordenes_pendientes"] = (
                [{"symbol": s, "lado": "venta"} for s in fuera] +
                [{"symbol": s, "lado": "compra", "importe_eur": round(por_pos, 2)}
                 for s in dentro])
            b["creadas_el"] = str(hoy.date())
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


def commitear(dir_salida: Path) -> None:
    """Guarda el estado en el repositorio: sin esto la simulación se perdería con
    la caché y cada día empezaría de cero."""
    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)
        subprocess.run(["git", "config", "user.email",
                        "41898282+github-actions[bot]@users.noreply.github.com"],
                       check=False)
        subprocess.run(["git", "add", str(dir_salida)], check=False)
        r = subprocess.run(["git", "commit", "-m", "Actualizar simulación de cartera"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["git", "push"], capture_output=True, text=True)
            log.info("simulación guardada en el repositorio")
    except Exception as e:  # pragma: no cover
        log.warning("no se pudo guardar la simulación: %s", e)


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
