"""Regresión: un grupo entero no puede desaparecer del universo en silencio.

El caso real, del log del 19/08/2026: los SEIS índices europeos fallaron a la
vez con "HTTP Error 403: Forbidden". Motivo: `pd.read_html(url)` descarga por
dentro con urllib, que se presenta como "Python-urllib/3.12", y Wikimedia
rechaza los agentes genéricos. Resultado: el universo NUNCA ha tenido una sola
acción europea — ni IBEX, ni DAX, ni CAC, ni FTSE — y sólo se registraba como
aviso, entre otros muchos avisos.

Se comprueban las dos defensas:
  1. la descarga se hace con un agente descriptivo y con la API REST de reserva;
  2. si un grupo esperado no aparece en el universo, se registra como ERROR.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import universe  # noqa: E402


class _Captura(logging.Handler):
    def __init__(self):
        super().__init__()
        self.errores: list[str] = []

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.errores.append(record.getMessage())


def prueba_agente() -> list[str]:
    fails = []
    pedidos = {}

    def get_falso(url, timeout=30):
        # sólo responde a quien se identifica: igual que Wikimedia
        pedidos[url] = universe.AGENTE
        if "rest_v1" in url:
            return ("<table><tr><th>Ticker</th><th>Company</th></tr>"
                    + "".join(f"<tr><td>T{i:02d}</td><td>Empresa {i}</td></tr>"
                              for i in range(20)) + "</table>")
        raise RuntimeError("HTTP Error 403: Forbidden")

    real = universe._get
    universe._get = get_falso
    try:
        tablas = universe._tablas_wiki("https://en.wikipedia.org/wiki/IBEX_35")
    finally:
        universe._get = real

    print(f"  la página normal da 403 y la REST responde: "
          f"{len(tablas)} tabla(s), {len(tablas[0])} filas")
    if not tablas or len(tablas[0]) < 15:
        fails.append("la reserva por API REST no devuelve la tabla")
    if "python-urllib" in universe.AGENTE.lower() or len(universe.AGENTE) < 20:
        fails.append("el agente no es descriptivo: Wikimedia lo rechazará")
    if "http" not in universe.AGENTE:
        fails.append("el agente debería incluir una forma de contacto")
    return fails


def prueba_grupo_ausente() -> list[str]:
    fails = []
    cap = _Captura()
    log = logging.getLogger("qscan.universe")
    log.addHandler(cap)

    real_us, real_eu, real_cr = (universe.us_listed, universe.eu_indices,
                                 universe.crypto)
    universe.us_listed = lambda: [
        universe.Asset(f"US{i:03d}", f"Acción {i}", "equity_us", "NASDAQ")
        for i in range(50)] + [
        universe.Asset(f"E{i:03d}", f"Fondo {i}", "etf", "NASDAQ")
        for i in range(30)]
    universe.eu_indices = lambda: []          # exactamente lo que pasó
    universe.crypto = lambda: []
    try:
        df = universe.build(include_crypto=True, include_eu=True)
    finally:
        universe.us_listed, universe.eu_indices, universe.crypto = \
            real_us, real_eu, real_cr
        log.removeHandler(cap)

    print(f"  universo construido sin Europa ni cripto: {len(df)} activos")
    for e in cap.errores:
        print(f"  ERROR registrado: {e[:90]}")
    if not any("equity_eu" in e for e in cap.errores):
        fails.append("la ausencia de renta variable europea no se registra como ERROR")
    if not any("crypto" in e for e in cap.errores):
        fails.append("la ausencia de cripto no se registra como ERROR")

    # y si están todos, no debe quejarse
    cap2 = _Captura()
    log.addHandler(cap2)
    universe.us_listed = lambda: [
        universe.Asset(f"US{i:03d}", f"A{i}", "equity_us", "NASDAQ") for i in range(20)
    ] + [universe.Asset(f"E{i:03d}", f"F{i}", "etf", "NASDAQ") for i in range(20)]
    universe.eu_indices = lambda: [
        universe.Asset(f"X{i:03d}.MC", f"Eu {i}", "equity_eu", "IBEX35", "EUR")
        for i in range(20)]
    universe.crypto = lambda: [
        universe.Asset(f"C{i}/USD", f"Cripto {i}", "crypto", "kraken") for i in range(20)]
    try:
        universe.build()
    finally:
        universe.us_listed, universe.eu_indices, universe.crypto = \
            real_us, real_eu, real_cr
        log.removeHandler(cap2)
    if cap2.errores:
        fails.append(f"se queja con el universo completo: {cap2.errores}")
    print(f"  con todos los grupos presentes: {len(cap2.errores)} errores")
    return fails


def main() -> int:
    print("1. agente descriptivo y reserva por API REST")
    fails = prueba_agente()
    print("\n2. un grupo ausente se registra como ERROR")
    fails += prueba_grupo_ausente()
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
