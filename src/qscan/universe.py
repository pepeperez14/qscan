"""Construcción del universo de activos.

Fuentes (todas públicas y gratuitas, funcionan desde GitHub Actions):
  - NASDAQ Trader symbol directory  -> todas las acciones y ETFs listados en EE.UU.
  - Wikipedia                       -> constituyentes de índices europeos principales
  - ccxt / Binance                  -> pares de cripto contra USDT
  - Lista estática                  -> futuros continuos, divisas, índices, bonos (vía Yahoo)

El objetivo NO es maximizar el número de tickers sino maximizar el número de
tickers *analizables*: series con historia suficiente y liquidez real. Un ticker
que negocia 8.000 EUR al día no es una oportunidad, es una trampa de liquidez.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, asdict

import pandas as pd
import requests

log = logging.getLogger(__name__)

NASDAQ_TRADER = "https://www.nasdaqtrader.com/dynamic/SymDir/{file}"

# Índices europeos: (nombre, url wikipedia, columna con el ticker, sufijo Yahoo)
EU_INDICES = [
    ("IBEX35", "https://en.wikipedia.org/wiki/IBEX_35", "Ticker", ".MC"),
    ("DAX", "https://en.wikipedia.org/wiki/DAX", "Ticker", ".DE"),
    ("CAC40", "https://en.wikipedia.org/wiki/CAC_40", "Ticker", ".PA"),
    ("FTSE100", "https://en.wikipedia.org/wiki/FTSE_100_Index", "Ticker", ".L"),
    ("FTSEMIB", "https://en.wikipedia.org/wiki/FTSE_MIB", "Ticker", ".MI"),
    ("AEX", "https://en.wikipedia.org/wiki/AEX_index", "Ticker", ".AS"),
]

# Macro-activos con ticker directo en Yahoo. Cubren materias primas, divisas,
# renta fija e índices: el contexto sin el que las señales de acciones no se leen.
# Cada grupo debe superar scoring.MIN_PEERS (20) o su sección transversal se
# descarta entera y el grupo desaparece del ranking sin previo aviso. Con las
# listas cortas originales (12-15 símbolos) eso pasaba siempre: materias primas,
# divisas, índices y renta fija nunca llegaban a puntuarse. Se amplían con ETFs
# líquidos, que además tienen mejor calidad de dato que los futuros continuos.
MACRO = {
    "commodity": [
        "GC=F", "SI=F", "HG=F", "CL=F", "BZ=F", "NG=F", "ZC=F", "ZW=F", "ZS=F",
        "KC=F", "SB=F", "CT=F", "PL=F", "PA=F", "LE=F", "HO=F", "RB=F", "OJ=F",
        "GLD", "SLV", "IAU", "USO", "BNO", "UNG", "DBC", "DBA", "CORN", "WEAT",
        "SOYB", "CANE", "PALL", "PPLT", "CPER", "URA", "GSG", "PDBC",
    ],
    "fx": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
        "NZDUSD=X", "USDCNY=X", "USDMXN=X", "USDBRL=X", "EURGBP=X", "EURJPY=X",
        "EURCHF=X", "EURSEK=X", "EURNOK=X", "USDSEK=X", "USDNOK=X", "USDPLN=X",
        "USDTRY=X", "USDZAR=X", "USDINR=X", "USDKRW=X", "USDSGD=X", "USDHKD=X",
        "AUDJPY=X", "GBPJPY=X", "CADJPY=X", "NZDJPY=X", "EURAUD=X", "DX-Y.NYB",
    ],
    "index": [
        "^GSPC", "^NDX", "^DJI", "^RUT", "^VIX", "^IXIC", "^NYA", "^MID", "^SML",
        "^STOXX50E", "^STOXX", "^GDAXI", "^FCHI", "^IBEX", "^FTSE", "^AEX",
        "^SSMI", "^OMX", "^BFX", "^ATX", "^N225", "^HSI", "^KS11", "^TWII",
        "^AXJO", "^GSPTSE", "^MXX", "^BVSP", "^BSESN", "^JKSE",
    ],
    "bond": [
        "^TNX", "^TYX", "^FVX", "^IRX", "TLT", "IEF", "SHY", "LQD", "HYG",
        "EMB", "TIP", "BND", "AGG", "GOVT", "VGIT", "VGLT", "VCIT", "VCSH",
        "MBB", "BIL", "SHV", "IGSB", "SPTL", "SCHO", "BWX", "IGOV", "PCY",
        "FLOT", "SJNK", "ANGL", "JNK", "BSV",
    ],
}

# Binance bloquea por geolocalización las IPs de EE.UU., que es justo donde
# corren los runners de GitHub Actions: desde ahí devuelve 451 y el universo se
# queda sin cripto en silencio. Se prueban varios mercados por orden.
CRYPTO_VENUES = [("kraken", "USD"), ("coinbase", "USD"), ("binance", "USDT")]


@dataclass(frozen=True)
class Asset:
    symbol: str          # ticker tal cual lo entiende el proveedor de datos
    name: str
    group: str           # equity_us | equity_eu | etf | crypto | commodity | fx | index | bond
    exchange: str = ""
    currency: str = "USD"

    def as_dict(self) -> dict:
        return asdict(self)


def _get(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "qscan/1.0"})
    r.raise_for_status()
    return r.text


# --------------------------------------------------------------------------- #
# productos estructuralmente inanalizables
# --------------------------------------------------------------------------- #
# Un multiplicador (2X, 3x, -1x) SÓLO cuenta si viene acompañado de una palabra
# direccional. Así "10x Genomics" no se confunde con "T-Rex 2X Long".
_MULTIPLICADOR = re.compile(r"(?<![A-Za-z0-9])[-+]?[1-3](?:\.\d)?\s*x(?![A-Za-z])", re.I)
_DIRECCIONAL = re.compile(r"\b(bull|bear|long|short|ultra\w*|inverse|leveraged|"
                          r"daily|apalancad\w+)\b", re.I)
_SIEMPRE_FUERA = re.compile(r"\bultra(short|pro)?\b|\binverse\b|\bleveraged\b"
                            r"|\bvix\b|volatility\s+index", re.I)

# Los inversos 1x no llevan multiplicador en el nombre ("ProShares Short S&P500")
# y "short" a secas descartaría medio mercado de renta fija (iShares Short
# Treasury Bond). Son pocos y conocidos, así que se nombran uno a uno.
INVERSOS_1X = {
    "SH", "PSQ", "DOG", "RWM", "EUM", "EFZ", "SEF", "SBB", "MYY", "MZZ",
    "DDG", "SJB", "TBF", "TBX", "IGBH", "BITI", "ETHD", "REK", "SDP", "SZK",
}


def derivado_estructural(nombre: str, symbol: str) -> bool:
    """¿Es un producto cuyo precio no se puede analizar como el de un activo?

    Los ETFs apalancados e inversos se reajustan a diario, así que su serie
    arrastra decaimiento por volatilidad: pierden valor incluso cuando el
    subyacente acaba donde empezó. Y los productos sobre el VIX cotizan futuros
    en contango permanente, así que caen estructuralmente.

    Todos ellos rompen exactamente lo que este sistema mide. `mom_12_1`,
    `slope_12m` o la distancia al máximo de 52 semanas describen la mecánica del
    envoltorio, no el comportamiento del activo. Un NUGT puede puntuar altísimo
    justo antes de un contrasplit, y el ranking no tiene forma de distinguirlo:
    su serie de precios es, formalmente, la de un activo con momento excelente.

    Se aplica SÓLO a ETFs. Sobre acciones daría falsos positivos evidentes
    (Ultra Clean Holdings, Ultragenyx, 10x Genomics) y ninguna acción es un
    envoltorio apalancado.
    """
    if symbol.upper() in INVERSOS_1X:
        return True
    n = nombre or ""
    if _SIEMPRE_FUERA.search(n):
        return True
    return bool(_MULTIPLICADOR.search(n) and _DIRECCIONAL.search(n))


def us_listed() -> list[Asset]:
    """Acciones y ETFs de EE.UU. desde el directorio oficial de NASDAQ Trader."""
    out: list[Asset] = []
    for fname, exch_default in (("nasdaqlisted.txt", "NASDAQ"), ("otherlisted.txt", "NYSE")):
        try:
            txt = _get(NASDAQ_TRADER.format(file=fname))
        except Exception as e:  # pragma: no cover - red
            log.warning("no se pudo bajar %s: %s", fname, e)
            continue
        # el fichero termina con una línea "File Creation Time"
        body = "\n".join(l for l in txt.splitlines() if not l.startswith("File Creation"))
        df = pd.read_csv(io.StringIO(body), sep="|")
        sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        df = df[df[sym_col].notna()]
        # descartar test issues y valores en situación anómala
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] != "Y"]
        if "Financial Status" in df.columns:
            df = df[df["Financial Status"].isin(["N", "Normal"]) | df["Financial Status"].isna()]
        is_etf = df["ETF"].eq("Y") if "ETF" in df.columns else pd.Series(False, index=df.index)
        fuera = 0
        for sym, name, etf in zip(df[sym_col], df.get("Security Name", ""), is_etf):
            sym = str(sym).strip()
            # los símbolos con $ o . de clases especiales rompen en Yahoo
            if not sym or any(c in sym for c in "$ "):
                continue
            sym = sym.replace(".", "-")
            if etf and derivado_estructural(str(name), sym):
                fuera += 1
                continue
            out.append(Asset(sym, str(name)[:120],
                             "etf" if etf else "equity_us", exch_default))
        if fuera:
            log.info("%s: %d ETFs apalancados/inversos/VIX descartados", fname, fuera)
    return out


def eu_indices() -> list[Asset]:
    """Constituyentes de los grandes índices europeos vía Wikipedia."""
    out: list[Asset] = []
    for name, url, col, suffix in EU_INDICES:
        try:
            tables = pd.read_html(url)
        except Exception as e:  # pragma: no cover - red
            log.warning("no se pudo leer %s: %s", name, e)
            continue
        for t in tables:
            cols = {str(c).strip(): c for c in t.columns}
            key = next((cols[c] for c in cols if c.lower() in ("ticker", "symbol", "epic")), None)
            if key is None or len(t) < 15:
                continue
            namecol = next((cols[c] for c in cols
                            if c.lower() in ("company", "name", "company name", "constituent")), key)
            for sym, cname in zip(t[key], t[namecol]):
                sym = str(sym).strip().upper().replace(" ", "")
                if not sym or sym == "NAN" or len(sym) > 8:
                    continue
                ysym = sym if "." in sym else f"{sym}{suffix}"
                out.append(Asset(ysym, str(cname)[:120], "equity_eu", name,
                                 "GBP" if suffix == ".L" else "EUR"))
            break  # la primera tabla válida es la de constituyentes
    return out


def pick_venue():
    """Primer mercado de cripto accesible desde esta máquina. Devuelve (ex, quote)."""
    try:
        import ccxt
    except ImportError:  # pragma: no cover
        log.warning("ccxt no instalado, se omite cripto")
        return None, None
    for name, quote in CRYPTO_VENUES:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            log.info("cripto: usando %s con cotización en %s", name, quote)
            return ex, quote
        except Exception as e:  # pragma: no cover - red
            log.warning("cripto: %s no accesible (%s)", name, str(e)[:120])
    log.warning("cripto: ningún mercado accesible; el universo se queda sin cripto")
    return None, None


def crypto(min_pairs: int = 300) -> list[Asset]:
    """Pares de cripto con volumen real, probando varios mercados."""
    ex, quote = pick_venue()
    if ex is None:
        return []
    try:
        tickers = ex.fetch_tickers()
    except Exception as e:  # pragma: no cover - red
        log.warning("no se pudieron leer tickers de %s: %s", ex.id, e)
        tickers = {}
    rows = []
    for sym, m in ex.markets.items():
        if not m.get("spot") or m.get("active") is False or m.get("quote") != quote:
            continue
        t = tickers.get(sym) or {}
        rows.append((sym, m.get("base", ""), t.get("quoteVolume") or 0.0))
    rows.sort(key=lambda r: r[2], reverse=True)
    venue = ex.id.upper()
    return [Asset(s, f"{b}/{quote}", "crypto", venue, quote) for s, b, _ in rows[:min_pairs]]


def macro() -> list[Asset]:
    return [Asset(s, s, g, "YAHOO") for g, syms in MACRO.items() for s in syms]


def build(include_crypto: bool = True, include_eu: bool = True) -> pd.DataFrame:
    """Ensambla el universo completo y lo devuelve deduplicado."""
    assets = macro() + us_listed()
    if include_eu:
        assets += eu_indices()
    if include_crypto:
        assets += crypto()
    df = pd.DataFrame([a.as_dict() for a in assets])
    df = df.drop_duplicates(subset="symbol").reset_index(drop=True)
    log.info("universo: %d activos\n%s", len(df), df.group.value_counts().to_string())
    return df
