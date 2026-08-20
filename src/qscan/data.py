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


LOTE_YAHOO = 100
HILOS_YAHOO = 8
PAUSA_YAHOO = 1.5
PAUSA_MAX = 30.0
ENFRIAMIENTO = 120.0
LIMITE_RESCATE = 2500


def _es_limite(e) -> bool:
    t = str(e).lower()
    return ("too many requests" in t or "rate limit" in t
            or "ratelimit" in t or "429" in t)


def _descargar_lote(chunk: list[str], start, retries: int,
                    hilos: int) -> tuple[pd.DataFrame, bool]:
    """Un lote. Devuelve (datos en formato largo, ¿hubo límite explícito?)."""
    import yfinance as yf

    vacio = pd.DataFrame(columns=["symbol", "date"] + COLS)
    limite = False
    for intento in range(retries + 1):
        try:
            raw = yf.download(chunk, start=start, auto_adjust=True, progress=False,
                              threads=hilos, group_by="column")
            return _to_long(raw, chunk), limite
        except Exception as e:
            if _es_limite(e):
                limite = True
            if intento == retries:
                log.warning("lote fallido definitivamente: %s", e)
                return vacio, limite
            # Espera exponencial y deliberadamente larga. Dormir 5 s y reintentar
            # frente a un "too many requests" sólo sirve para confirmar el
            # bloqueo: el contador de la fuente se libera en decenas de segundos,
            # no en cinco.
            time.sleep(min(15 * (2 ** intento), PAUSA_MAX))
    return vacio, limite


def fetch_yahoo(symbols: list[str], start: str | pd.Timestamp,
                batch: int = LOTE_YAHOO, pause: float = PAUSA_YAHOO,
                retries: int = 2, hilos: int = HILOS_YAHOO,
                rescate: bool = True, sin_rescate: set[str] | None = None) -> pd.DataFrame:
    """Descarga por lotes con ritmo adaptativo y una pasada de rescate.

    El modo de fallo observado en producción no es una excepción: yfinance NO
    lanza error cuando la fuente limita el ritmo, devuelve el lote A MEDIAS y
    escribe los fallos por consola. Con lotes de 200 símbolos y todos los hilos
    abiertos, las primeras decenas de lotes pasan y a partir de ahí se pierden
    170 símbolos de cada 200 — más de mil activos por ejecución, en silencio,
    con la ejecución en verde.

    De ahí las tres decisiones de aquí:
      - el control se basa en CUÁNTOS símbolos volvieron, no en excepciones;
      - la pausa crece cuando un lote vuelve incompleto y decae cuando vuelve
        entero, así el ritmo se ajusta solo al límite real del día;
      - lo que falte tras la primera pasada se reintenta al final, después de
        dejar enfriar el contador, en lotes pequeños y con pocos hilos.
    """
    out: list[pd.DataFrame] = []
    faltan: list[str] = []
    pausa = pause
    total = (len(symbols) - 1) // batch + 1

    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        df, limite = _descargar_lote(chunk, start, retries, hilos)
        if not df.empty:
            out.append(df)
        obtenidos = set(df.symbol.unique()) if not df.empty else set()
        faltan += [s for s in chunk if s not in obtenidos]
        tasa = len(obtenidos) / max(len(chunk), 1)
        if limite or tasa < 0.6:
            pausa = min(pausa * 1.8, PAUSA_MAX)
            log.warning("lote %d/%d: sólo %d/%d símbolos (%s) · pausa -> %.0f s",
                        i // batch + 1, total, len(obtenidos), len(chunk),
                        "límite explícito" if limite else "respuesta parcial", pausa)
        else:
            if pausa > pause:
                pausa = max(pause, pausa * 0.7)
            log.info("lote %d/%d: %d/%d símbolos", i // batch + 1, total,
                     len(obtenidos), len(chunk))
        time.sleep(pausa)

    if rescate and sin_rescate:
        # Un warrant caducado o un ticker retirado no van a volver mañana. Sin
        # esta lista, cada ejecución gastaba dos minutos de enfriamiento más
        # varios de reintentos para recuperar 1 de 136 símbolos, todos muertos.
        antes = len(faltan)
        faltan = [s for s in faltan if s not in sin_rescate]
        if antes != len(faltan):
            log.info("rescate: %d símbolos dados por muertos, no se reintentan",
                     antes - len(faltan))

    if rescate and faltan:
        log.warning("rescate: %d de %d símbolos sin datos en la primera pasada; "
                    "se enfría %.0f s y se reintenta en lotes pequeños",
                    len(faltan), len(symbols), ENFRIAMIENTO)
        time.sleep(ENFRIAMIENTO)
        if len(faltan) > LIMITE_RESCATE:
            # Tope explícito y anunciado. Sin él, un día en que la fuente falle
            # entera dejaría la ejecución reintentando durante horas.
            log.error("rescate limitado a %d símbolos: %d se quedan SIN reintentar "
                      "en esta ejecución", LIMITE_RESCATE, len(faltan) - LIMITE_RESCATE)
            faltan = faltan[:LIMITE_RESCATE]
        chico = max(10, batch // 4)
        pausa_r = min(max(pause * 2, pausa * 0.4), 12.0)
        recuperados = 0
        for i in range(0, len(faltan), chico):
            df, _ = _descargar_lote(faltan[i:i + chico], start, 1, max(2, hilos // 4))
            if not df.empty:
                out.append(df)
                recuperados += df.symbol.nunique()
            time.sleep(pausa_r)
        log.info("rescate: recuperados %d de %d símbolos", recuperados, len(faltan))

    if not out:
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
    res = pd.concat(out, ignore_index=True)
    return res.drop_duplicates(subset=["symbol", "date"], keep="last")


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
    # El pool por defecto de urllib3 son 10 conexiones. Con 12 hilos el pool se
    # desborda, urllib3 descarta conexiones ("Connection pool is full") y parte
    # de las peticiones se pierden — que es justo lo que se vio: la reserva se
    # llamó y devolvió cero. El pool tiene que ser al menos tan grande como el
    # número de hilos que lo usan.
    ada = requests.adapters.HTTPAdapter(pool_connections=hilos,
                                        pool_maxsize=hilos, max_retries=1)
    ses.mount("https://", ada)
    ses.mount("http://", ada)
    fallos: dict[str, int] = {}

    def _anota(motivo: str):
        fallos[motivo] = fallos.get(motivo, 0) + 1

    def uno(par):
        # Se lleva la cuenta del MOTIVO de cada fallo. Sin esto, "la fuente de
        # reserva tampoco devolvió nada" no distingue entre un bloqueo, un
        # símbolo inexistente y un CSV sin la historia pedida — y las tres cosas
        # se arreglan de manera distinta.
        yahoo, stooq = par
        try:
            r = ses.get(STOOQ_URL.format(sym=stooq), timeout=timeout)
            if r.status_code != 200:
                _anota(f"http {r.status_code}")
                return None
            if "Date" not in r.text[:64]:
                _anota("sin datos para el símbolo")
                return None
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = [c.strip().lower() for c in d.columns]
            if "close" not in d.columns:
                _anota("csv sin columna close")
                return None
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d[d["date"] >= pd.Timestamp(start)]
            if d.empty:
                _anota("sin fechas dentro del rango pedido")
                return None
            d.insert(0, "symbol", yahoo)
            for c in COLS:
                if c not in d.columns:
                    d[c] = np.nan
            return d[["symbol", "date"] + COLS]
        except Exception as e:
            _anota(type(e).__name__)
            return None

    frames = []
    with cf.ThreadPoolExecutor(max_workers=hilos) as ex:
        for res in ex.map(uno, pares):
            if res is not None and not res.empty:
                frames.append(res)
    detalle = " · ".join(f"{k}: {v}" for k, v in
                         sorted(fallos.items(), key=lambda kv: -kv[1])[:5])
    if not frames:
        log.error("la fuente de reserva tampoco devolvió nada (%s)",
                  detalle or "sin motivo registrado")
        return pd.DataFrame(columns=["symbol", "date"] + COLS)
    if fallos:
        log.info("reserva: %d símbolos sin datos (%s)", sum(fallos.values()), detalle)
    out = pd.concat(frames, ignore_index=True)
    log.info("fuente de reserva: %d símbolos · última fecha %s",
             out.symbol.nunique(), pd.to_datetime(out["date"]).max().date())
    return out


def empalmar_reserva(reserva: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Pega el tramo de la fuente de reserva al nivel del almacén.

    Las dos fuentes no ajustan igual. Yahoo devuelve precios ajustados por
    splits Y por dividendos; Stooq ajusta splits pero no dividendos. Para un
    valor con dividendo del 3% anual y ocho años de historia, la diferencia de
    escala entre ambas series ronda el 25%. Concatenarlas sin más crearía un
    escalón artificial justo en la juntura: un retorno enorme e inventado en un
    solo día, que además dispararía el detector de anomalías y acabaría
    apartando un activo perfectamente sano.

    Así que se hace lo que hace cualquiera al empalmar dos series de precios: se
    calcula el factor en la última fecha en que ambas coinciden y se reescala la
    serie nueva a la vieja. Sólo se conservan las filas POSTERIORES a lo que ya
    hay en el almacén, que es lo único que se está intentando rellenar; si la
    juntura sale absurda (factor fuera de 0,2-5) se deja el símbolo sin tocar,
    porque un empalme así significa que los dos tickers no son el mismo activo.
    """
    if reserva.empty or existing.empty:
        return reserva
    r = reserva.copy()
    r["date"] = pd.to_datetime(r["date"])
    e = existing[["symbol", "date", "close"]].copy()
    e["date"] = pd.to_datetime(e["date"])

    ult = e.groupby("symbol")["date"].max()
    conocidos = r.symbol.isin(ult.index)

    solape = r[["symbol", "date", "close"]].merge(
        e.rename(columns={"close": "close_almacen"}), on=["symbol", "date"])
    if not solape.empty:
        ref = solape.sort_values("date").groupby("symbol").tail(1)
        factor = (ref["close_almacen"] / ref["close"]).values
        f = pd.Series(factor, index=ref["symbol"].values)
        f = f[(f > 0.2) & (f < 5) & f.notna()]
        esc = r.symbol.map(f)
        aplica = esc.notna()
        for c in ("open", "high", "low", "close"):
            r.loc[aplica, c] = r.loc[aplica, c].astype(float) * esc[aplica]
        log.info("reserva: %d símbolos reescalados al nivel del almacén "
                 "(factor mediano %.3f)", len(f), float(f.median()) if len(f) else 1.0)

    # sólo el tramo que el almacén no tiene: rellenar, no reescribir
    limite_sym = r.symbol.map(ult)
    r = r[~conocidos | (r["date"] > limite_sym)]
    return r.reset_index(drop=True)


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
    if max_previo is not None and maxima < max_previo:
        log.error("la descarga RETROCEDE (máximo previo %s, máximo descargado "
                  "%s)", pd.Timestamp(max_previo).date(), maxima.date())
    elif max_previo is not None and maxima == max_previo:
        # NO es un fallo. Es lo que pasa al relanzar el mismo día: el almacén ya
        # tiene la última sesión y la descarga trae exactamente lo mismo. Antes
        # esto se registraba como ERROR y además disparaba la fuente de reserva
        # sin ninguna necesidad. Si además la fecha estuviera vieja, quien lo
        # detecta es `comprobar_frescura`, que es su trabajo.
        log.info("la descarga confirma la última sesión ya almacenada (%s): "
                 "reejecución del mismo día", maxima.date())


DIAS_REINTENTO_MUERTOS = 30


def _ahora() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


FALLOS_PARA_MUERTO = 2


def leer_muertos(ruta: Path) -> dict[str, dict]:
    """Historial de fallos por símbolo: {symbol: {"desde": fecha, "fallos": n}}."""
    if not ruta.exists():
        return {}
    try:
        d = pd.read_csv(ruta)
        fallos = d["fallos"] if "fallos" in d.columns else 1
        return {str(s): {"desde": str(f), "fallos": int(n)}
                for s, f, n in zip(d["symbol"], d["desde"],
                                   fallos if hasattr(fallos, "__iter__")
                                   else [fallos] * len(d))}
    except Exception as e:
        log.warning("no se pudo leer la lista de símbolos muertos: %s", e)
        return {}


def escribir_muertos(ruta: Path, muertos: dict[str, dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": s, "desde": v["desde"], "fallos": v["fallos"]}
                  for s, v in muertos.items()]).to_csv(ruta, index=False)


def actualizar_muertos(muertos: dict[str, dict], pedidos: list[str],
                       respondieron: set[str],
                       hoy: pd.Timestamp | None = None) -> dict[str, dict]:
    """Suma un fallo al que no respondió y borra al que sí.

    Se cuentan fallos CONSECUTIVOS en vez de reconstruir la lista con los fallos
    de hoy. Un símbolo perfectamente vivo puede no responder una vez por un
    límite de ritmo puntual, y darlo por muerto al primer intento haría que
    dejásemos de pedirlo durante un mes por un tropiezo. Con dos fallos seguidos
    ya no es mala suerte.
    """
    hoy_txt = str((hoy if hoy is not None else _ahora()).date())
    out = dict(muertos)
    for s in pedidos:
        if s in respondieron:
            out.pop(s, None)
            continue
        prev = out.get(s)
        out[s] = {"desde": prev["desde"] if prev else hoy_txt,
                  "fallos": (prev["fallos"] if prev else 0) + 1}
    return out


def muertos_vigentes(muertos: dict[str, dict], hoy: pd.Timestamp | None = None,
                     dias: int = DIAS_REINTENTO_MUERTOS,
                     minimo: int = FALLOS_PARA_MUERTO) -> set[str]:
    """Los que no merece la pena reintentar todavía.

    No es una lista negra permanente: pasado el plazo se vuelve a probar con
    todos, por si un ticker vuelve a listarse o la fuente lo recupera. Lo que se
    evita es pagar el enfriamiento y los reintentos TODOS los días por warrants
    caducados y unidades de SPAC que ya no existen.
    """
    hoy = hoy if hoy is not None else _ahora()
    vivos = set()
    for s, v in muertos.items():
        try:
            if v["fallos"] >= minimo and (hoy - pd.Timestamp(v["desde"])).days < dias:
                vivos.add(s)
        except (ValueError, TypeError, KeyError):
            continue
    return vivos


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
    dias = (_ahora() - ultima).days
    return dias >= cada_dias


def update(universe: pd.DataFrame, store: PriceStore, years: int = 8,
           lookback_days: int = 7, recarga_completa: bool | None = None,
           cada_dias: int = DIAS_ENTRE_RECARGAS,
           usar_reserva: bool = True) -> pd.DataFrame:
    """Actualización incremental, con recarga completa periódica."""
    existing = store.load()
    last = store.last_dates()
    full_start = _ahora() - pd.DateOffset(years=years)
    marca = store.path.parent / ".ultima_recarga"
    if recarga_completa is None:
        recarga_completa = toca_recarga_completa(marca, cada_dias)
    if recarga_completa:
        log.info("recarga completa: se vuelve a bajar la historia entera para "
                 "recoger splits y dividendos aplicados retroactivamente")

    crypto_syms = universe.loc[universe.group == "crypto", "symbol"].tolist()
    other_syms = universe.loc[universe.group != "crypto", "symbol"].tolist()

    def repartir(syms: list[str]) -> tuple[list[str], list[str], pd.Timestamp]:
        """Separa los que necesitan historia entera de los que sólo el último tramo.

        Antes había UNA sola fecha de inicio para todo el mundo, y eso dejó a las
        278 acciones europeas —recién incorporadas al universo— con seis semanas
        de historia: sólo 99 de 277 llegaron al panel, porque las features de
        doce meses no se pueden calcular sobre mes y medio de precios. Un símbolo
        que el almacén no conoce necesita su historia completa, no el tramo
        incremental que sirve para los demás.

        Se aplica lo mismo a los rezagados: un símbolo que lleva semanas sin
        actualizarse no se arregla pidiendo los últimos siete días. Antes el
        inicio incremental era el MÍNIMO de todas las últimas fechas, así que un
        solo rezagado obligaba a rebajar la ventana de los doce mil restantes.
        """
        if recarga_completa:
            return [], syms, full_start
        conocidos = [s for s in syms if s in last.index]
        if len(conocidos) < len(syms) * 0.5:
            return [], syms, full_start          # almacén casi vacío
        corte = _ahora() - pd.Timedelta(days=30)
        al_dia = [s for s in conocidos if pd.Timestamp(last[s]) >= corte]
        rezagados = [s for s in syms if s not in set(al_dia)]
        inicio = (min(pd.Timestamp(last[s]) for s in al_dia)
                  - pd.Timedelta(days=lookback_days)) if al_dia else full_start
        return al_dia, rezagados, inicio

    ruta_muertos = store.path.parent / ".simbolos_muertos.csv"
    muertos = leer_muertos(ruta_muertos)
    vigentes = muertos_vigentes(muertos)
    if vigentes:
        log.info("%d símbolos en la lista de muertos (se reintentan cada %d días)",
                 len(vigentes), DIAS_REINTENTO_MUERTOS)

    new = []
    if other_syms:
        al_dia, completos, inicio = repartir(other_syms)
        if al_dia:
            log.info("descarga incremental de %d símbolos desde %s",
                     len(al_dia), inicio.date())
            new.append(fetch_yahoo(al_dia, inicio, sin_rescate=vigentes))
        if completos:
            log.info("historia entera para %d símbolos (nuevos en el universo o "
                     "con el almacén rezagado)", len(completos))
            new.append(fetch_yahoo(completos, full_start, sin_rescate=vigentes))
    if crypto_syms:
        al_dia_c, completos_c, inicio_c = repartir(crypto_syms)
        if al_dia_c:
            new.append(fetch_crypto(al_dia_c, inicio_c))
        if completos_c:
            new.append(fetch_crypto(completos_c, full_start))

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
        conseguidos = set(obtenidos.symbol.unique()) if len(obtenidos) else set()
        faltantes = [s for s in other_syms if s not in conseguidos]
        # "sin avance" es que la descarga se quede POR DETRÁS de lo que ya hay,
        # no que traiga lo mismo: relanzar el mismo día es normal y no debe
        # arrastrar a la fuente de reserva a pedir símbolos que no faltan.
        sin_avance = (max_previo is not None and (
            obtenidos.empty or pd.to_datetime(obtenidos["date"]).max() < max_previo))
        # Antes esto era todo-o-nada: la reserva sólo entraba si la fuente
        # principal fallaba casi por completo. Pero el fallo real no es una
        # caída, es una MERMA: se pierde un tercio del universo, el sistema
        # sigue en verde y el ranking se calcula sobre los activos que
        # sobrevivieron — que no son una muestra aleatoria, son los que se
        # pidieron primero. Ahora se le pide a la reserva exactamente lo que
        # falta.
        if faltantes and (len(faltantes) > len(other_syms) * 0.05 or sin_avance):
            log.warning("faltan %d de %d símbolos tras la fuente principal "
                        "(sin_avance=%s): se pide a la reserva sólo lo que falta",
                        len(faltantes), len(other_syms), sin_avance)
            # a la reserva se le pide historia completa: los que faltan suelen
            # ser justo los que el almacén no conoce o tiene rezagados
            reserva = fetch_stooq(faltantes, full_start)
            # En una recarga completa no se empalma: la historia vieja de esos
            # símbolos se va a descartar entera unas líneas más abajo, así que
            # la serie de la reserva no se pega a nada — es la serie entera, en
            # su propia escala, y reescalarla o recortarla sólo la estropearía.
            if not recarga_completa:
                reserva = empalmar_reserva(reserva, existing)
            if not reserva.empty:
                log.info("reserva: %d símbolos rellenados", reserva.symbol.nunique())
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
    # Se actualiza la lista de muertos: entra el que no ha devuelto nada y sale
    # el que ha vuelto a responder. Guardar sólo los que fallan HOY hace que la
    # lista se limpie sola en vez de crecer para siempre.
    if other_syms:
        respondieron = set(pd.concat(nuevos, ignore_index=True).symbol.unique()) \
            if nuevos else set()
        muertos = actualizar_muertos(muertos, other_syms, respondieron)
        escribir_muertos(ruta_muertos, muertos)
        log.info("registro de fallos: %d símbolos, de los cuales %d ya no se "
                 "reintentarán", len(muertos), len(muertos_vigentes(muertos)))

    merged = quality_filter(merged)

    # El almacén se queda con lo que el universo ya no incluye. Al retirar 838
    # ETFs apalancados el conteo de cobertura salió del 102%: se contaban
    # símbolos vivos en el almacén contra un universo más pequeño. Se poda, pero
    # con freno de mano: si la poda se llevara más del 20% del almacén, lo más
    # probable no es que el mercado haya encogido sino que la construcción del
    # universo haya fallado a medias, y borrar años de historia por eso sería
    # mucho peor que dejar filas de sobra.
    conocidos = set(universe.symbol)
    sobra = sorted(set(merged.symbol.unique()) - conocidos)
    if sobra:
        frac = len(sobra) / max(merged.symbol.nunique(), 1)
        if frac > 0.20:
            log.error("el universo ya no contiene %d de los %d símbolos del "
                      "almacén (%.0f%%). NO se podan: una caída así apunta a un "
                      "universo mal construido, no a un mercado que encoge.",
                      len(sobra), merged.symbol.nunique(), 100 * frac)
        else:
            log.info("almacén: se retiran %d símbolos que ya no están en el "
                     "universo", len(sobra))
            merged = merged[merged.symbol.isin(conocidos)]

    store.save(merged)

    # Cobertura de la ÚLTIMA sesión, que es la que decide cuántos activos entran
    # hoy en el ranking. El número de filas del almacén no sirve para esto: sigue
    # creciendo aunque la mitad del universo se haya quedado sin la barra de hoy.
    fechas = pd.to_datetime(merged["date"])
    habiles = fechas[fechas.dt.dayofweek < 5]
    if len(habiles):
        # La última sesión COMPLETA, no la última fila. Midiéndola sobre una
        # sesión a medio formar la cobertura salía del 3% y el error gritaba que
        # se había perdido el 97% del universo, cuando lo único que pasaba es
        # que Nueva York todavía no había abierto.
        por_fecha = merged[merged.symbol.isin(conocidos)].groupby(
            fechas[merged.symbol.isin(conocidos)]).symbol.nunique()
        por_fecha = por_fecha[por_fecha.index.dayofweek < 5].sort_index()
        norma = float(por_fecha.tail(21).head(20).median()) if len(por_fecha) > 1 else 0.0
        utiles = por_fecha[por_fecha >= norma * UMBRAL_SESION] if norma > 0 else por_fecha
        ultima = utiles.index.max() if len(utiles) else habiles.max()
        if ultima != habiles.max():
            log.info("la sesión del %s todavía se está formando (%d símbolos): "
                     "la cobertura se mide sobre el %s",
                     habiles.max().date(), int(por_fecha.get(habiles.max(), 0)),
                     ultima.date())
        # sólo cuentan los símbolos DEL UNIVERSO: si no, el almacén arrastra
        # símbolos retirados y la cobertura puede pasar del 100%
        al_dia = merged.loc[(fechas == ultima)
                            & merged.symbol.isin(conocidos), "symbol"].nunique()
        pedidos = len(conocidos)
        pct = 100 * al_dia / max(pedidos, 1)
        log.info("cobertura de la última sesión (%s): %d/%d símbolos (%.0f%%)",
                 ultima.date(), al_dia, pedidos, pct)
        if pct < 75:
            log.error("SE HA PERDIDO EL %.0f%% DEL UNIVERSO en la última sesión. "
                      "El ranking de hoy no se calcula sobre el universo completo "
                      "sino sobre los símbolos que la fuente sí respondió, que no "
                      "son una muestra aleatoria.", 100 - pct)
    if recarga_completa and nuevos:
        marca.parent.mkdir(parents=True, exist_ok=True)
        marca.write_text(str(_ahora().date()))
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


UMBRAL_SESION = 0.6


def ultima_sesion_util(close: pd.DataFrame, ventana: int = 20,
                       umbral: float = UMBRAL_SESION) -> pd.Timestamp | None:
    """La última sesión que está formada del todo.

    Una sesión a medio hacer no es la foto de hoy: es la foto de los pocos
    mercados que ya han abierto. El caso real: la ejecución de las 05:24 UTC del
    20/08/2026. La cripto cotiza 24 horas y las divisas casi, así que ya tenían
    barra del día 20; la bolsa americana no abría hasta seis horas después y la
    europea acababa de empezar. El almacén tenía por tanto una última fila con
    348 símbolos de 12.235, y como el ranking se calcula SIEMPRE sobre la última
    fila del panel, el informe de ese día salió sin una sola acción: 23 criptos y
    25 divisas, y todos los demás grupos a cero.

    Lo peligroso es que no falla nada. El panel se construye, los z-scores se
    calculan sobre los pocos que hay, y sale un informe con aspecto normal sobre
    un mercado que todavía no ha abierto.

    Se recorre hacia atrás y se acepta la primera sesión cuya cobertura llegue a
    una fracción de lo habitual en las sesiones previas. Un festivo en EE.UU. con
    Europa abierta se descarta por el mismo motivo, y con razón: ese día no hay
    sección transversal que comparar.
    """
    if close is None or close.empty:
        return None
    presentes = close.notna().sum(axis=1)
    for i in range(len(presentes) - 1, -1, -1):
        previas = presentes.iloc[max(0, i - ventana):i]
        norma = float(previas.median()) if len(previas) else 0.0
        if norma <= 0 or presentes.iloc[i] >= norma * umbral:
            return presentes.index[i]
    return presentes.index[-1]


def recortar_a_sesion_util(px: dict[str, pd.DataFrame],
                           umbral: float = UMBRAL_SESION) -> dict[str, pd.DataFrame]:
    """Corta las matrices en la última sesión completa. Ver `ultima_sesion_util`."""
    corte = ultima_sesion_util(px.get("close"), umbral=umbral)
    if corte is None:
        return px
    idx = px["close"].index
    descartadas = int((idx > corte).sum())
    if descartadas:
        presentes = px["close"].notna().sum(axis=1)
        log.warning("se descartan %d sesión(es) a medio formar por delante de %s "
                    "(la última tenía %d símbolos de %d): el ranking se calcula "
                    "sobre la última sesión completa",
                    descartadas, corte.date(), int(presentes.iloc[-1]),
                    px["close"].shape[1])
    return {k: df.loc[:corte] for k, df in px.items()}


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
