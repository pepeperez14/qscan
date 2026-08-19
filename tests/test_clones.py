"""Regresión: ni derivados apalancados en el universo ni clones en la cartera.

Dos problemas distintos con la misma raíz —el universo se construye solo, a
partir del directorio de NASDAQ, y no todo lo que cotiza es analizable:

1. Los ETFs apalancados e inversos se reajustan a diario y arrastran decaimiento
   por volatilidad. Su serie de precios es formalmente la de un activo con
   momento excelente justo antes de evaporarse, y el ranking no tiene forma de
   distinguirla. Lo mismo con los productos sobre el VIX, en contango
   permanente. Se filtran por nombre, y sólo entre ETFs: sobre acciones el mismo
   patrón daría falsos positivos obvios (Ultra Clean Holdings, 10x Genomics).

2. GLD, IAU, GLDM y SGOL son el mismo lingote con distinta comisión: correlación
   0,999. Cuatro de ellos en la cartera no son cuatro ideas, son una idea con
   cuatro veces el tamaño y cuatro comisiones. El filtro anterior sólo miraba
   dentro de cada horizonte, así que `corto` podía comprar GLD y `medio` IAU sin
   enterarse, y el relleno final no filtraba nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import portfolio, universe  # noqa: E402

# (nombre tal cual lo publica NASDAQ, ticker, ¿debe quedar fuera?)
CASOS = [
    ("Direxion Daily Gold Miners Index Bull 2X Shares", "NUGT", True),
    ("Direxion Daily Junior Gold Miners Index Bear 2X Shares", "JDST", True),
    ("ProShares Ultra Gold", "UGL", True),
    ("ProShares UltraShort Gold", "GLL", True),
    ("ProShares UltraPro QQQ", "TQQQ", True),
    ("T-Rex 2X Long MSTR Daily Target ETF", "MSTU", True),
    ("GraniteShares 2x Long NVDA Daily ETF", "NVDL", True),
    ("MicroSectors Gold Miners 3X Leveraged ETN", "GDXU", True),
    ("iPath Series B S&P 500 VIX Short-Term Futures ETN", "VXX", True),
    ("ProShares Short S&P500", "SH", True),
    # y lo que NO puede caerse por el camino
    ("SPDR Gold Shares", "GLD", False),
    ("iShares Gold Trust", "IAU", False),
    ("SPDR Gold MiniShares Trust", "GLDM", False),
    ("abrdn Physical Gold Shares ETF", "SGOL", False),
    ("VanEck Gold Miners ETF", "GDX", False),
    ("iShares Short Treasury Bond ETF", "SHV", False),
    ("Vanguard Short-Term Bond ETF", "BSV", False),
    ("Schwab Long-Term U.S. Treasury ETF", "SCHQ", False),
    ("Invesco DB Precious Metals Fund", "DBP", False),
]

# nombres de ACCIONES que el patrón tocaría si se aplicara fuera de los ETFs
ACCIONES_TRAMPA = [("Ultra Clean Holdings, Inc.", "UCTT"),
                   ("Ultragenyx Pharmaceutical Inc.", "RARE"),
                   ("10x Genomics, Inc.", "TXG"),
                   ("Bear Creek Mining Corp", "BCEKF")]


def prueba_filtro_universo() -> list[str]:
    fails = []
    for nombre, sym, esperado in CASOS:
        r = universe.derivado_estructural(nombre, sym)
        marca = "OK " if r == esperado else "MAL"
        print(f"  {marca} {'fuera ' if r else 'dentro'} {sym:6} {nombre[:50]}")
        if r != esperado:
            fails.append(f"{sym} ({nombre}): esperado "
                         f"{'fuera' if esperado else 'dentro'}")
    print(f"  — y las acciones trampa, que no pasan por el filtro:")
    for nombre, sym in ACCIONES_TRAMPA:
        print(f"     {sym:6} {nombre}")
    return fails


def _panel_con_clones() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Cuatro clones de oro, uno de plata y ocho activos independientes."""
    fechas = pd.bdate_range("2025-01-01", periods=300)
    rng = np.random.default_rng(7)

    oro = np.cumsum(rng.normal(0.0005, 0.010, len(fechas)))
    plata = np.cumsum(rng.normal(0.0004, 0.014, len(fechas)))
    series, liquidez = {}, {}

    # los clones comparten el 99,5% del movimiento: es lo que pasa de verdad
    # entre GLD, IAU, GLDM y SGOL
    clones = {"GLD": 8e8, "IAU": 3e8, "GLDM": 6e8, "SGOL": 2e7}
    for i, (s, vol) in enumerate(clones.items()):
        ruido = rng.normal(0, 0.0006, len(fechas))
        series[s] = 100 * np.exp(oro + np.cumsum(ruido))
        liquidez[s] = vol
    for s, vol in (("SLV", 5e8), ("SIVR", 1e7)):
        ruido = rng.normal(0, 0.0007, len(fechas))
        series[s] = 40 * np.exp(plata + np.cumsum(ruido))
        liquidez[s] = vol
    for i in range(8):
        series[f"IND{i}"] = 50 * np.exp(np.cumsum(
            rng.normal(0.0003, 0.013, len(fechas))))
        liquidez[f"IND{i}"] = 2e8

    close = pd.DataFrame(series, index=fechas)

    # ranking: los clones de oro copan la cabeza, luego plata, luego el resto.
    # SGOL —el menos líquido— va primero a propósito, para comprobar que entre
    # clones se acaba prefiriendo el que se cruza más barato.
    orden = ["SGOL", "GLD", "IAU", "GLDM", "SIVR", "SLV"] + \
            [f"IND{i}" for i in range(8)]
    scored = pd.DataFrame({
        "date": fechas[-1], "symbol": orden, "group": "etf",
        "score_medio": np.linspace(3.0, 0.1, len(orden)),
        "pct_medio": np.linspace(99, 40, len(orden)),
    })
    return scored, close, pd.Series(liquidez)


def prueba_seleccion() -> list[str]:
    fails = []
    scored, close, liq = _panel_con_clones()
    # MIN_PEERS exige un grupo poblado; con 14 símbolos hay de sobra si se baja
    from qscan import scoring
    original, scoring.MIN_PEERS = scoring.MIN_PEERS, 10
    try:
        elegidos = portfolio.seleccionar(scored, None, close, 8,
                                         {"medio": 1.0}, liq)
    finally:
        scoring.MIN_PEERS = original

    print(f"  seleccionados: {elegidos}")
    oro = [s for s in elegidos if s in ("GLD", "IAU", "GLDM", "SGOL")]
    plata = [s for s in elegidos if s in ("SLV", "SIVR")]
    print(f"  clones de oro en cartera: {oro} · de plata: {plata}")
    if len(oro) != 1:
        fails.append(f"deberían quedar 1 clon de oro y quedan {len(oro)}: {oro}")
    if len(plata) != 1:
        fails.append(f"deberían quedar 1 clon de plata y quedan {len(plata)}")
    if oro and oro[0] != "GLD":
        fails.append(f"entre clones debería quedarse el más líquido (GLD), "
                     f"no {oro[0]}")
    if len(set(elegidos)) != len(elegidos):
        fails.append("hay símbolos repetidos en la selección")

    # sin el filtro, los cuatro clones entrarían: si no, la prueba no prueba nada
    crudo = scored.sort_values("pct_medio", ascending=False).symbol.tolist()[:8]
    if len([s for s in crudo if s in ("GLD", "IAU", "GLDM", "SGOL")]) < 3:
        fails.append("el escenario no reproduce el problema: revisa la prueba")
    print(f"  sin filtrar, el top 8 sería: {crudo}")
    return fails


def prueba_horizontes_cruzados() -> list[str]:
    """El caso concreto que el filtro anterior no veía: dos horizontes que
    eligen el mismo activo con distinto envoltorio."""
    fails = []
    scored, close, liq = _panel_con_clones()
    scored["score_corto"] = scored["score_medio"]
    # en corto manda GLD; en medio, IAU. Cada horizonte por su cuenta elegiría
    # el suyo y la cartera acabaría con los dos.
    scored["pct_corto"] = [99 if s == "GLD" else 50 for s in scored.symbol]
    scored["pct_medio"] = [99 if s == "IAU" else 50 for s in scored.symbol]

    from qscan import scoring
    original, scoring.MIN_PEERS = scoring.MIN_PEERS, 10
    try:
        elegidos = portfolio.seleccionar(scored, None, close, 6,
                                         {"corto": 0.5, "medio": 0.5}, liq)
    finally:
        scoring.MIN_PEERS = original

    oro = [s for s in elegidos if s in ("GLD", "IAU", "GLDM", "SGOL")]
    print(f"  corto prefiere GLD, medio prefiere IAU -> cartera: {elegidos}")
    print(f"  clones de oro finales: {oro}")
    if len(oro) > 1:
        fails.append(f"dos horizontes han comprado el mismo activo: {oro}")
    return fails


def main() -> int:
    print("1. filtro de derivados estructurales")
    fails = prueba_filtro_universo()
    print("\n2. clones dentro de un horizonte")
    fails += prueba_seleccion()
    print("\n3. clones entre horizontes distintos")
    fails += prueba_horizontes_cruzados()
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
