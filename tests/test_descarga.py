"""Regresión: la descarga no puede perder medio universo en silencio.

El caso real: con lotes de 200 símbolos, los primeros veintitantos lotes vuelven
enteros y a partir del cuarenta la fuente devuelve 20-30 símbolos de cada 200.
Ni una excepción — yfinance no lanza error cuando le limitan el ritmo, escribe
los fallos por consola y devuelve el lote a medias. Resultado: ~1.500 activos
perdidos por ejecución, ejecución en verde y un ranking calculado sobre los
símbolos que sobrevivieron, que son los que se pidieron primero.

Se comprueban las dos defensas nuevas:
  1. el ritmo se adapta a la respuesta REAL (símbolos devueltos), no a las
     excepciones, y lo que falta se reintenta al final tras enfriar;
  2. el tramo de la fuente de reserva se empalma al nivel del almacén en vez de
     concatenarse en bruto, que crearía un escalón de precio inventado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import data  # noqa: E402


def _falso_lote(simbolos, fechas, escala=1.0):
    filas = []
    for s in simbolos:
        filas.append(pd.DataFrame({
            "symbol": s, "date": fechas, "open": 100.0 * escala,
            "high": 101.0 * escala, "low": 99.0 * escala,
            "close": 100.0 * escala, "volume": 1e6}))
    return pd.concat(filas, ignore_index=True) if filas else \
        pd.DataFrame(columns=["symbol", "date"] + data.COLS)


def prueba_ritmo_y_rescate() -> list[str]:
    fails = []
    fechas = pd.bdate_range("2026-08-10", periods=3)
    simbolos = [f"S{i:04d}" for i in range(500)]

    dormidas, llamadas = [], []

    def sleep_falso(s):
        dormidas.append(s)

    def lote_falso(chunk, start, retries, hilos):
        llamadas.append((len(chunk), hilos))
        # los dos primeros lotes grandes pasan; a partir de ahí, limitación
        grandes = [c for c, _ in llamadas if c > 30]
        if len(chunk) > 30 and len(grandes) > 2:
            return _falso_lote(chunk[:len(chunk) // 5], fechas), True
        return _falso_lote(chunk, fechas), False

    real_sleep, real_lote = data.time.sleep, data._descargar_lote
    data.time.sleep = sleep_falso
    data._descargar_lote = lote_falso
    try:
        out = data.fetch_yahoo(simbolos, "2026-08-01", batch=100)
    finally:
        data.time.sleep, data._descargar_lote = real_sleep, real_lote

    obtenidos = set(out.symbol.unique())
    perdidos = [s for s in simbolos if s not in obtenidos]
    print(f"primera pasada limitada al 20% desde el tercer lote · "
          f"símbolos finales {len(obtenidos)}/{len(simbolos)}")
    if perdidos:
        fails.append(f"la pasada de rescate no recuperó {len(perdidos)} símbolos")

    if out.duplicated(subset=["symbol", "date"]).any():
        fails.append("la descarga devuelve filas duplicadas symbol+fecha")

    chicos = [c for c, _ in llamadas if c <= 30]
    print(f"lotes grandes {len([c for c, _ in llamadas if c > 30])} · "
          f"lotes de rescate {len(chicos)}")
    if not chicos:
        fails.append("no hubo pasada de rescate en lotes pequeños")

    pico = max([d for d in dormidas if d != data.ENFRIAMIENTO] or [0])
    print(f"pausa máxima alcanzada: {pico:.0f} s (base {data.PAUSA_YAHOO})")
    if pico <= data.PAUSA_YAHOO * 2:
        fails.append("la pausa no creció al detectar lotes incompletos")
    if pico > data.PAUSA_MAX:
        fails.append("la pausa creció por encima del tope")
    return fails


def prueba_empalme() -> list[str]:
    fails = []
    fechas = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(11)
    base = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(fechas))))

    almacen = pd.DataFrame({"symbol": "ACME", "date": fechas, "open": base,
                            "high": base * 1.01, "low": base * 0.99,
                            "close": base, "volume": 1e6})

    # la reserva no ajusta dividendos: misma forma, escala un 20% más alta,
    # y llega tres sesiones más lejos que el almacén
    ext = pd.bdate_range(fechas[-1], periods=4)[1:]
    fechas_r = fechas.append(ext)
    base_r = np.concatenate([base, base[-1] * np.array([1.004, 1.001, 0.997])]) / 0.8
    reserva = pd.DataFrame({"symbol": "ACME", "date": fechas_r, "open": base_r,
                            "high": base_r * 1.01, "low": base_r * 0.99,
                            "close": base_r, "volume": 1e6})

    emp = data.empalmar_reserva(reserva, almacen)
    print(f"reserva original {len(reserva)} filas · tras empalmar {len(emp)}")
    if len(emp) != 3:
        fails.append(f"debería quedarse sólo con las 3 fechas nuevas, no {len(emp)}")

    junta = pd.concat([almacen, emp], ignore_index=True).sort_values("date")
    salto = float(junta["close"].pct_change().abs().max())
    print(f"salto diario máximo en la serie empalmada: {100 * salto:.2f}%")
    if salto > 0.10:
        fails.append(f"el empalme crea un escalón del {100 * salto:.0f}%: "
                     "no se está reescalando la reserva")

    # sin empalmar, el escalón tiene que estar ahí (si no, la prueba no prueba nada)
    crudo = pd.concat([almacen, reserva[reserva.date > fechas[-1]]],
                      ignore_index=True).sort_values("date")
    salto_crudo = float(crudo["close"].pct_change().abs().max())
    print(f"salto máximo SIN empalmar:                  {100 * salto_crudo:.2f}%")
    if salto_crudo < 0.15:
        fails.append("el escenario no reproduce el escalón: revisa la prueba")

    # un ticker que no es el mismo activo no debe reescalarse a la fuerza
    absurdo = reserva.copy()
    absurdo["close"] = absurdo["close"] * 100
    for c in ("open", "high", "low"):
        absurdo[c] = absurdo[c] * 100
    emp2 = data.empalmar_reserva(absurdo, almacen)
    ratio = float(emp2["close"].iloc[0] / almacen["close"].iloc[-1]) if len(emp2) else 0
    print(f"empalme absurdo (x100): factor resultante {ratio:.1f} (se deja crudo)")
    if ratio < 50:
        fails.append("un empalme imposible se ha aplicado igualmente")
    return fails


def main() -> int:
    fails = prueba_ritmo_y_rescate()
    print()
    fails += prueba_empalme()
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
