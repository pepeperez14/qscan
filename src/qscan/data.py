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
    from .universe import pick_venue

    ex, _ = pick_venue()
    if ex is None:
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
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


STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"


def _a_stooq(symbol: str) -> str | None:
    """Traduce un ticker de Yahoo al formato de Stooq.

    Sólo se cubren acciones y ETFs de EE.UU., que son el 97% del universo. Los
    futuros, divisas e índices usan códigos distintos en cada fuente y traducir a
    ojo sería peor que no traducir: un símbolo mal mapeado no falla, devuelve la
    serie de OTRO activo.
    """
    s = symbol.strip()
    if not s or any(c in s for c in "^=/.") or len(s) > 6:
        return None
    return s.replace("-", ".").lower() + ".us"


def fetch_stooq(symbols: list[str], start: pd.Timestamp, hilos: int = 12,
                limite: int = 3000, timeout: int = 20) -> pd.DataFrame:
    """Fuente de reserva cuando la principal deja de responder.

    No sustituye a Yahoo: cubre menos activos y menos historia. Existe para que
    una caída de la fuente principal degrade el sistema en vez de congelarlo.
    """
    import concurrent.futures as cf
    import io

    import requests

    pares = [(s, _a_stooq(s)) for s in symbols]
    pares = [(y, t) for y, t in pares if t][:limite]
    if not pares:
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
    log.info("fuente de reserva (stooq): pidiendo %d símbolos", len(pares))

    ses = requests.Session()
    ses.headers.update({"User-Agent": "qscan/1.0"})

    def uno(par):
        yahoo, stooq = par
        try:
            r = ses.get(STOOQ_URL.format(sym=stooq), timeout=timeout)
            if r.status_code != 200 or "Date" not in r.text[:64]:
                return None
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = [c.strip().lower() for c in d.columns]
            if "close" not in d.columns:
                return None
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d[d["date"] >= pd.Timestamp(start)]
            if d.empty:
                return None
            d.insert(0, "symbol", yahoo)
            for c in COLS:
                if c not in d.columns:
                    d[c] = np.nan
            return d[["symbol", "date"] + COLS]
        except Exception:
            return None

    frames = []
    with cf.ThreadPoolExecutor(max_workers=hilos) as ex:
        for res in ex.map(uno, pares):
            if res is not None and not res.empty:
                frames.append(res)
    if not frames:
        log.error("la fuente de reserva tampoco devolvió nada")
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
    out = pd.concat(frames, ignore_index=True)
    log.info("fuente de reserva: %d símbolos · última fecha %s",
             out.symbol.nunique(), pd.to_datetime(out["date"]).max().date())
    return out


MAX_RETRASO_SESIONES = 2


def sesiones_de_retraso(ultima: pd.Timestamp, hoy: pd.Timestamp | None = None) -> int:
    """Días hábiles transcurridos desde la última cotización del almacén."""
    hoy = hoy or pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if pd.isna(ultima):
        return 10_000
    return int(len(pd.bdate_range(pd.Timestamp(ultima), hoy)) - 1)


def comprobar_frescura(store: PriceStore, max_sesiones: int = MAX_RETRASO_SESIONES,
                       hoy: pd.Timestamp | None = None) -> int:
    """Falla en rojo si el almacén ha dejado de avanzar.

    Esta comprobación existe porque el modo de fallo real observado no fue un
    error, fue SILENCIO: la descarga dejó de traer datos nuevos, `merged` quedó
    igual que `existing`, y el sistema siguió calculando, publicando y
    declarándose correcto sobre precios de hace días. Una ejecución verde con
    datos congelados es peor que una roja: la roja se ve.
    """
    df = store.load()
    if df.empty:
        raise SystemExit("ALMACÉN VACÍO: la descarga no ha traído ningún precio.")
    ultima = pd.to_datetime(df["date"]).max()
    retraso = sesiones_de_retraso(ultima, hoy)
    log.info("última cotización del almacén: %s (%d sesiones de retraso)",
             ultima.date(), retraso)
    if retraso > max_sesiones:
        raise SystemExit(
            f"DATOS CONGELADOS: la última cotización es del {ultima.date()}, "
            f"{retraso} sesiones por detrás. La descarga no está trayendo datos "
            f"nuevos. Revisa los avisos del paso de descarga: lo más probable es "
            f"que la fuente esté limitando el ritmo o rechazando las peticiones. "
            f"Se corta aquí a propósito: seguir produciría un informe con "
            f"apariencia de recién hecho y precios viejos.")
    return retraso


def resumen_descarga(nuevos: list[pd.DataFrame], pedidos: int,
                     max_previo: pd.Timestamp | None) -> None:
    """Deja constancia de lo que la descarga trajo de verdad.

    Sin esto, una descarga que falla en todos los lotes y una que funciona se
    parecen demasiado en el log.
    """
    if not nuevos:
        log.error("la descarga no devolvió NADA para %d símbolos pedidos", pedidos)
        return
    df = pd.concat(nuevos, ignore_index=True)
    if df.empty:
        log.error("la descarga devolvió 0 filas para %d símbolos pedidos", pedidos)
        return
    obtenidos = df.symbol.nunique()
    maxima = pd.to_datetime(df["date"]).max()
    log.info("descarga: %d/%d símbolos con datos · %d filas · última fecha %s",
             obtenidos, pedidos, len(df), maxima.date())
    if obtenidos < pedidos * 0.5:
        log.error("sólo respondió el %.0f%% de los símbolos: la fuente está "
                  "rechazando peticiones", 100 * obtenidos / max(pedidos, 1))
    if max_previo is not None and maxima <= max_previo:
        log.error("la descarga NO aporta fechas nuevas (máximo previo %s, "
                  "máximo descargado %s)", pd.Timestamp(max_previo).date(),
                  maxima.date())


DIAS_ENTRE_RECARGAS = 30


def toca_recarga_completa(marca: Path, cada_dias: int = DIAS_ENTRE_RECARGAS) -> bool:
    """¿Toca volver a bajar toda la historia?

    Hace falta porque la actualización incremental sólo pide los últimos días,
    pero Yahoo revisa la historia HACIA ATRÁS: cuando una empresa hace un split o
    paga dividendo, se reajusta la serie entera. Nuestro almacén se quedaría con
    la versión vieja y desincronizada, y el detector de anomalías acabaría
    marcando el activo como "split sin ajustar" y apartándolo — apagando el
    activo en lugar de arreglar el dato.
    """
    if not marca.exists():
        return True
    try:
        ultima = pd.Timestamp(marca.read_text().strip())
    except (ValueError, OSError):
        return True
    dias = (pd.Timestamp.utcnow().tz_localize(None).normalize() - ultima).days
    return dias >= cada_dias


def update(universe: pd.DataFrame, store: PriceStore, years: int = 8,
           lookback_days: int = 7, recarga_completa: bool | None = None,
           cada_dias: int = DIAS_ENTRE_RECARGAS,
           usar_reserva: bool = True) -> pd.DataFrame:
    """Actualización incremental, con recarga completa periódica."""
    existing = store.load()
    last = store.last_dates()
    full_start = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.DateOffset(years=years)
    marca = store.path.parent / ".ultima_recarga"
    if recarga_completa is None:
        recarga_completa = toca_recarga_completa(marca, cada_dias)
    if recarga_completa:
        log.info("recarga completa: se vuelve a bajar la historia entera para "
                 "recoger splits y dividendos aplicados retroactivamente")

    crypto_syms = universe.loc[universe.group == "crypto", "symbol"].tolist()
    other_syms = universe.loc[universe.group != "crypto", "symbol"].tolist()

    def _start_for(syms: list[str]) -> pd.Timestamp:
        if recarga_completa:
            return full_start
        known = [last[s] for s in syms if s in last.index]
        if len(known) < len(syms) * 0.9:
            return full_start
        return min(known) - pd.Timedelta(days=lookback_days)

    new = []
    if other_syms:
        new.append(fetch_yahoo(other_syms, _start_for(other_syms)))
    if crypto_syms:
        new.append(fetch_crypto(crypto_syms, _start_for(crypto_syms)))

    nuevos = [n for n in new if not n.empty]
    max_previo = pd.to_datetime(existing["date"]).max() if not existing.empty else None
    resumen_descarga(nuevos, len(other_syms) + len(crypto_syms), max_previo)

    # Reserva: si la fuente principal devolvió poco o nada nuevo, se intenta otra
    # antes de darse por vencido. Que una caída de Yahoo congele el sistema
    # entero durante días es justo lo que pasó, y esto lo degrada en vez de
    # pararlo.
    if usar_reserva and other_syms:
        obtenidos = (pd.concat(nuevos, ignore_index=True) if nuevos
                     else pd.DataFrame(columns=["symbol", "date"]))
        pocos = obtenidos.symbol.nunique() < len(other_syms) * 0.5 if len(obtenidos) \
            else True
        sin_avance = (max_previo is not None and (
            obtenidos.empty or pd.to_datetime(obtenidos["date"]).max() <= max_previo))
        if pocos or sin_avance:
            log.warning("la fuente principal no avanzó (pocos=%s, sin_avance=%s): "
                        "se prueba la fuente de reserva", pocos, sin_avance)
            reserva = fetch_stooq(other_syms, _start_for(other_syms))
            if not reserva.empty:
                nuevos.append(reserva)
    if recarga_completa and nuevos and not existing.empty:
        # Se DESCARTAN las filas viejas de los símbolos recargados en vez de
        # fusionarlas. Si se mezclaran, un split dejaría el tramo antiguo en la
        # escala vieja y el nuevo en la nueva, con un salto falso justo en la
        # frontera: exactamente el dato roto que se quería eliminar.
        refrescados = set(pd.concat(nuevos, ignore_index=True).symbol.unique())
        antes = len(existing)
        existing = existing[~existing.symbol.isin(refrescados)]
        log.info("recarga completa: sustituidas %d filas de %d símbolos",
                 antes - len(existing), len(refrescados))

    merged = pd.concat([existing] + nuevos, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    for c in COLS:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float32")
    merged = quality_filter(merged)
    store.save(merged)
    if recarga_completa and nuevos:
        marca.parent.mkdir(parents=True, exist_ok=True)
        marca.write_text(str(pd.Timestamp.utcnow().tz_localize(None).date()))
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
