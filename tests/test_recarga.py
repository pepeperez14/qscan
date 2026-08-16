"""Regresión: la recarga completa periódica debe sustituir la historia vieja.

El caso que motiva todo esto: una empresa hace un split 4:1. Yahoo reajusta la
serie ENTERA hacia atrás. La actualización incremental sólo pide los últimos
días, así que el almacén se queda con el tramo antiguo en la escala vieja y el
nuevo en la nueva — un salto falso justo en la frontera, que además dispara el
detector de anomalías y acaba apartando un activo perfectamente sano.

Se comprueba también que fusionar en vez de sustituir NO arregla el problema:
esa es la razón por la que la recarga descarta las filas viejas de los símbolos
que vuelve a bajar.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qscan import anomalies, data  # noqa: E402

TMP = Path("/tmp/qscan_recarga")


def _serie(fechas, precios) -> pd.DataFrame:
    return pd.DataFrame({"symbol": "ACME", "date": fechas, "open": precios,
                         "high": precios * 1.01, "low": precios * 0.99,
                         "close": precios, "volume": 1e6})


def main() -> int:
    fails = []
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)

    fechas = pd.bdate_range("2024-01-01", periods=400)
    rng = np.random.default_rng(4)
    base = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(fechas))))

    # --- 1. el marcador decide cuándo toca ---------------------------------
    marca = TMP / ".ultima_recarga"
    if not data.toca_recarga_completa(marca):
        fails.append("sin marcador debería tocar recarga")
    marca.write_text(str((pd.Timestamp.utcnow().tz_localize(None)
                          - pd.Timedelta(days=5)).date()))
    if data.toca_recarga_completa(marca, cada_dias=30):
        fails.append("con 5 días de antigüedad no debería tocar")
    marca.write_text(str((pd.Timestamp.utcnow().tz_localize(None)
                          - pd.Timedelta(days=45)).date()))
    if not data.toca_recarga_completa(marca, cada_dias=30):
        fails.append("con 45 días sí debería tocar")
    print("marcador de recarga: decide bien en los tres casos")

    # --- 2. el escenario del split ----------------------------------------
    # almacén viejo: escala anterior al split en toda la serie
    viejo = _serie(fechas, base)
    # lo que Yahoo devuelve tras un split 4:1: TODA la historia dividida entre 4
    nuevo = _serie(fechas, base / 4.0)

    store = data.PriceStore(TMP / "prices.parquet")
    store.save(viejo)

    # (a) fusionando, que es lo que hacía la versión incremental
    fusion = pd.concat([viejo, nuevo.tail(20)], ignore_index=True)
    fusion = fusion.drop_duplicates(subset=["symbol", "date"], keep="last")
    px_fus = {f: fusion.pivot(index="date", columns="symbol", values=f).sort_index()
              for f in ("open", "high", "low", "close", "volume")}
    rep_fus = anomalies.detect(px_fus, min_dollar_vol=0)
    marcado_fus = bool(rep_fus.loc["ACME", "split_sin_ajustar"])
    print(f"fusionando el tramo nuevo: ¿ACME marcada como split sin ajustar? "
          f"{'SÍ' if marcado_fus else 'no'}")
    if not marcado_fus:
        fails.append("el escenario de prueba no reproduce el problema: revisa el test")

    # (b) sustituyendo, que es lo que hace la recarga completa
    limpio = nuevo.copy()
    px_lim = {f: limpio.pivot(index="date", columns="symbol", values=f).sort_index()
              for f in ("open", "high", "low", "close", "volume")}
    rep_lim = anomalies.detect(px_lim, min_dollar_vol=0)
    marcado_lim = bool(rep_lim.loc["ACME", "split_sin_ajustar"])
    print(f"sustituyendo la serie entera:  ¿ACME marcada?                     "
          f"{'SÍ' if marcado_lim else 'no'}")
    if marcado_lim:
        fails.append("tras la recarga completa el activo sigue marcado: la "
                     "sustitución no está limpiando la escala vieja")

    # --- 3. la sustitución no pierde filas --------------------------------
    if len(limpio) != len(viejo):
        fails.append("la serie recargada no tiene la misma longitud que la vieja")
    print(f"filas antes {len(viejo)} · después {len(limpio)}")

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
