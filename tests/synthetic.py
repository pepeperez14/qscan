"""Generador de mercados sintéticos para probar el pipeline sin red.

Dos regímenes, a propósito:

  - `signal=0.0`  -> paseos aleatorios puros. El sistema DEBE informar de que no
    hay señal. Si aquí sale un t-stat alto, hay una fuga de información.
  - `signal>0`    -> se inyecta autocorrelación de momento conocida. El sistema
    DEBE detectarla. Si aquí no sale nada, el cálculo está roto.

Un backtest que sólo se prueba con datos reales no distingue "no hay señal" de
"mi código está mal".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = {"equity_us": 220, "equity_eu": 90, "etf": 70, "crypto": 60,
          "commodity": 30, "fx": 24, "index": 22, "bond": 24}


def make_market(n_days: int = 1600, signal: float = 0.0, seed: int = 7,
                groups: dict[str, int] | None = None, null: bool = False
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (precios en formato largo, universo).

    `null=True` construye un nulo verdadero: deriva cero en el mercado y cero
    dispersión de deriva entre activos. Es importante entender por qué hace
    falta. Si cada activo tiene su propia deriva persistente, su rentabilidad
    pasada predice de verdad la futura, y el escáner acierta con razón: eso no
    es un fallo del código, es señal real en los datos simulados. Para poder
    afirmar "el sistema no inventa señal" hay que quitar toda fuente legítima
    de predictibilidad, deriva y beta incluidas.
    """
    rng = np.random.default_rng(seed)
    groups = groups or GROUPS
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    rows, uni = [], []

    # factor de mercado común: sin él, las betas y la fuerza relativa no significan nada
    mkt_drift = 0.0 if null else 0.0003
    mkt = rng.normal(mkt_drift, 0.010, n_days)

    for grp, n in groups.items():
        vol_base = {"crypto": 0.045, "fx": 0.005, "bond": 0.004, "commodity": 0.016,
                    "index": 0.010, "etf": 0.011}.get(grp, 0.018)
        for i in range(n):
            sym = f"{grp.replace('_', '').upper()}{i:03d}"
            beta = rng.normal(1.0, 0.35) if grp.startswith("equity") else rng.normal(0.3, 0.5)
            vol = vol_base * rng.lognormal(0, 0.35)
            drift = 0.0 if null else rng.normal(0.0002, 0.0004)
            idio = rng.normal(0, vol, n_days)
            r = drift + beta * mkt + idio

            if signal:
                # momento persistente: el retorno de hoy depende del de los 60 días
                # previos. Es la señal que el validador tiene que encontrar.
                for t in range(80, n_days):
                    r[t] += signal * np.tanh(r[t - 80:t - 20].sum() * 3) * vol

            px = 100 * np.exp(np.cumsum(r))
            hl = np.abs(rng.normal(0, vol * 0.7, n_days))
            close = px
            open_ = px * (1 + rng.normal(0, vol * 0.4, n_days))
            high = np.maximum(open_, close) * (1 + hl)
            low = np.minimum(open_, close) * (1 - hl)
            volu = rng.lognormal(13.5, 1.1, n_days) * (1 + 3 * np.abs(r) / vol)

            rows.append(pd.DataFrame({
                "symbol": sym, "date": dates, "open": open_, "high": high,
                "low": low, "close": close, "volume": volu}))
            uni.append({"symbol": sym, "name": f"Sintético {sym}", "group": grp,
                        "exchange": "SYN", "currency": "USD"})

    prices = pd.concat(rows, ignore_index=True)
    # benchmark explícito, para que el pipeline use la misma ruta que en real
    bench_px = 100 * np.exp(np.cumsum(mkt))
    prices = pd.concat([prices, pd.DataFrame({
        "symbol": "^GSPC", "date": dates, "open": bench_px, "high": bench_px * 1.003,
        "low": bench_px * 0.997, "close": bench_px, "volume": 1e9})], ignore_index=True)
    uni.append({"symbol": "^GSPC", "name": "S&P 500", "group": "index",
                "exchange": "SYN", "currency": "USD"})
    return prices, pd.DataFrame(uni)
