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
MACRO = {
    "commodity": ["GC=F", "SI=F", "HG=F", "CL=F", "BZ=F", "NG=F", "ZC=F", "ZW=F",
                  "ZS=F", "KC=F", "SB=F", "CT=F", "PL=F", "PA=F", "LE=F"],
    "fx": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
           "NZDUSD=X", "USDCNY=X", "USDMXN=X", "USDBRL=X", "EURGBP=X", "EURJPY=X",
           "DX-Y.NYB"],
    "index": ["^GSPC", "^NDX", "^DJI", "^RUT", "^VIX", "^STOXX50E", "^GDAXI",
              "^FCHI", "^IBEX", "^FTSE", "^N225", "^HSI", "^BSESN", "^BVSP"],
    "bond": ["^TNX", "^TYX", "^FVX", "^IRX", "TLT", "IEF", "SHY", "LQD", "HYG",
             "EMB", "TIP", "BND"],
}


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
        for sym, name, etf in zip(df[sym_col], df.get("Security Name", ""), is_etf):
            sym = str(sym).strip()
            # los símbolos con $ o . de clases especiales rompen en Yahoo
            if not sym or any(c in sym for c in "$ "):
                continue
            out.append(Asset(sym.replace(".", "-"), str(name)[:120],
                             "etf" if etf else "equity_us", exch_default))
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


def crypto(quote: str = "USDT", min_pairs: int = 400) -> list[Asset]:
    """Pares de cripto con volumen real, vía ccxt sobre Binance."""
    try:
        import ccxt
    except ImportError:  # pragma: no cover
        log.warning("ccxt no instalado, se omite cripto")
        return []
    try:
        ex = ccxt.binance()
        markets = ex.load_markets()
        tickers = ex.fetch_tickers()
    except Exception as e:  # pragma: no cover - red
        log.warning("binance no accesible: %s", e)
        return []
    rows = []
    for sym, m in markets.items():
        if not m.get("spot") or not m.get("active") or m.get("quote") != quote:
            continue
        t = tickers.get(sym) or {}
        rows.append((sym, m.get("base", ""), t.get("quoteVolume") or 0.0))
    rows.sort(key=lambda r: r[2], reverse=True)
    return [Asset(s, f"{b}/{quote}", "crypto", "BINANCE", quote) for s, b, _ in rows[:min_pairs]]


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
