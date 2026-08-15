"""Descarga y almacenamiento incremental de OHLCV.

El almacén es un único parquet en formato largo (symbol, date, o/h/l/c/v),
particionado por grupo para que las descargas incrementales sean baratas.
Con ~9.000 símbolos y 8 años de historia diaria ocupa del orden de 150-250 MB
comprimido: entra de sobra en la caché de GitHub Actions (límite 10 GB).

Reglas que el resto del sistema da por hechas y que se garantizan aquí:
  - precios ajustados por splits y dividendos (auto_adjust)
  - índice de fechas ordenado, sin duplicados, en UTC naive
  - ningún relleno hacia adelante que invente cotizaciones donde no las hubo
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLS = ["open", "high", "low", "close", "volume"]


class PriceStore:
    def __init__(self, path: str | Path = "data/prices.parquet"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["symbol", "date"] + COLS)
        return pd.read_parquet(self.path)

    def save(self, df: pd.DataFrame) -> None:
        df = (df.dropna(subset=["close"])
                .drop_duplicates(subset=["symbol", "date"], keep="last")
                .sort_values(["symbol", "date"]))
        df.to_parquet(self.path, index=False, compression="zstd")
        log.info("almacén: %d filas, %d símbolos", len(df), df.symbol.nunique())

    def last_dates(self) -> pd.Series:
        df = self.load()
        if df.empty:
            return pd.Series(dtype="datetime64[ns]")
        return df.groupby("symbol")["date"].max()

    def wide(self, field: str = "close", min_obs: int = 250) -> pd.DataFrame:
        """Matriz fecha x símbolo. Base para todo el cálculo vectorizado."""
        df = self.load()
        if df.empty:
            return pd.DataFrame()
        w = df.pivot(index="date", columns="symbol", values=field).sort_index()
        return w.loc[:, w.notna().sum() >= min_obs]


def _to_long(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Normaliza la salida de yfinance (multi-índice de columnas) a formato largo."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({symbols[0]: raw}, axis=1)
        raw.columns = raw.columns.swaplevel(0, 1)
    frames = []
    fields = {c.lower(): c for c in raw.columns.get_level_values(0).unique()}
    for sym in symbols:
        try:
            sub = pd.DataFrame({
                k: raw[(fields[k], sym)] for k in COLS if k in fields
            })
        except KeyError:
            continue
        sub = sub.dropna(subset=["close"])
        if sub.empty:
            continue
        sub.insert(0, "symbol", sym)
        sub.insert(1, "date", pd.to_datetime(sub.index).tz_localize(None).normalize())
        frames.append(sub.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
    return pd.concat(frames, ignore_index=True)


def fetch_yahoo(symbols: list[str], start: str | pd.Timestamp,
                batch: int = 200, pause: float = 1.0, retries: int = 2) -> pd.DataFrame:
    """Descarga por lotes. yfinance rate-limitea: lotes grandes + pausa es lo estable."""
    import yfinance as yf

    out = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        for attempt in range(retries + 1):
            try:
                raw = yf.download(chunk, start=start, auto_adjust=True, progress=False,
                                  threads=True, group_by="column")
                out.append(_to_long(raw, chunk))
                break
            except Exception as e:
                if attempt == retries:
                    log.warning("lote %d fallido definitivamente: %s", i // batch, e)
                else:
                    time.sleep(5 * (attempt + 1))
        log.info("descargado lote %d/%d", i // batch + 1, (len(symbols) - 1) // batch + 1)
        time.sleep(pause)
    if not out:
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
    return pd.concat(out, ignore_index=True)


def fetch_crypto(symbols: list[str], start: pd.Timestamp, timeframe: str = "1d") -> pd.DataFrame:
    """OHLCV de cripto vía ccxt. Yahoo cubre mal el universo alt."""
    try:
        import ccxt
    except ImportError:
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
    ex = ccxt.binance({"enableRateLimit": True})
    since = int(pd.Timestamp(start).timestamp() * 1000)
    frames = []
    for sym in symbols:
        try:
            rows, cursor = [], since
            while True:
                batch = ex.fetch_ohlcv(sym, timeframe=timeframe, since=cursor, limit=1000)
                if not batch:
                    break
                rows += batch
                cursor = batch[-1][0] + 1
                if len(batch) < 1000:
                    break
            if not rows:
                continue
            d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            d["date"] = pd.to_datetime(d.ts, unit="ms").dt.normalize()
            d.insert(0, "symbol", sym)
            frames.append(d[["symbol", "date"] + COLS])
        except Exception as e:
            log.warning("cripto %s: %s", sym, e)
    return pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["symbol", "date"] + COLS)


def update(universe: pd.DataFrame, store: PriceStore, years: int = 8,
           lookback_days: int = 7) -> pd.DataFrame:
    """Actualización incremental: sólo pide lo que falta desde la última fecha."""
    existing = store.load()
    last = store.last_dates()
    full_start = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.DateOffset(years=years)

    crypto_syms = universe.loc[universe.group == "crypto", "symbol"].tolist()
    other_syms = universe.loc[universe.group != "crypto", "symbol"].tolist()

    def _start_for(syms: list[str]) -> pd.Timestamp:
        known = [last[s] for s in syms if s in last.index]
        if len(known) < len(syms) * 0.9:
            return full_start
        return min(known) - pd.Timedelta(days=lookback_days)

    new = []
    if other_syms:
        new.append(fetch_yahoo(other_syms, _start_for(other_syms)))
    if crypto_syms:
        new.append(fetch_crypto(crypto_syms, _start_for(crypto_syms)))

    merged = pd.concat([existing] + [n for n in new if not n.empty], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    for c in COLS:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float32")
    merged = quality_filter(merged)
    store.save(merged)
    return merged


def to_business_calendar(px: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Alinea todas las series a un calendario común de días hábiles.

    ESTO NO ES COSMÉTICO. La cripto cotiza 365 días y las bolsas 252, así que al
    pivotar juntas el índice común pasa a tener ~726 filas por cada 2 años en vez
    de 520. Como todas las ventanas del sistema están expresadas en FILAS, una
    ventana de 252 deja de ser "12 meses" y pasa a ser 8,3 meses para todo el
    mundo: `ret_12m`, `mom_12_1`, `vol_252` y `donch_252` medirían un periodo
    distinto del que dice su nombre.

    Peor aún: para las acciones, esas 252 filas sólo contienen ~180 cotizaciones
    reales (el resto son fines de semana en blanco), y las features que exigen
    ventana completa —`slope_12m`, `trendfit_12m`— salen NaN para TODAS las
    acciones. Al caer su cobertura por debajo del mínimo, la renta variable
    desaparecía entera del ranking de largo plazo sin dar ningún error.

    Se muestrea la cripto en el calendario bursátil, que es la práctica estándar
    cuando se comparan clases de activo entre sí. Basta con quedarse con los días
    hábiles: la cripto también cotiza esos días, así que sólo se descartan sus
    barras de sábado y domingo. No se rellena nada (los festivos de bolsa siguen
    siendo huecos) porque inventar cierres crearía retornos cero falsos y
    dispararía el detector de series congeladas.
    """
    idx = px["close"].index
    bdays = idx[idx.dayofweek < 5]
    return {k: df.reindex(bdays) for k, df in px.items()}


def quality_filter(df: pd.DataFrame, max_gap_ratio: float = 0.35) -> pd.DataFrame:
    """Descarta series rotas: precios no positivos, saltos imposibles, huecos masivos.

    Un salto diario mayor de +900% / -90% sin split casi siempre es un error del
    proveedor, y un solo dato malo contamina todos los indicadores del activo.
    """
    df = df[(df.close > 0) & (df.high >= df.low)]
    df = (df.drop_duplicates(subset=["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"]))
    ret = df.groupby("symbol")["close"].pct_change()
    bad = (ret > 9) | (ret < -0.9)
    # marcar el símbolo entero sólo si acumula varios saltos imposibles
    offenders = bad.groupby(df.symbol).sum()
    df = df[~df.symbol.isin(offenders[offenders >= 3].index)]
    # exigir cobertura razonable del calendario del propio grupo
    span = df.groupby("symbol")["date"].agg(["min", "max", "count"])
    expected = (span["max"] - span["min"]).dt.days * (252 / 365)
    keep = span.index[(span["count"] >= expected * (1 - max_gap_ratio)) | (expected < 60)]
    return df[df.symbol.isin(keep)].reset_index(drop=True)
