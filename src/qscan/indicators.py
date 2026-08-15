"""Indicadores técnicos vectorizados sobre matrices fecha x símbolo.

Todo opera sobre DataFrames anchos (índice = fecha, columnas = símbolos) para que
9.000 activos se calculen en segundos en lugar de en horas. Ninguna función mira
al futuro: cada valor en la fila t usa exclusivamente datos hasta t inclusive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# tendencia
# --------------------------------------------------------------------------- #
def sma(px: pd.DataFrame, n: int) -> pd.DataFrame:
    return px.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(px: pd.DataFrame, n: int) -> pd.DataFrame:
    return px.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def dist_to_ma(px: pd.DataFrame, n: int) -> pd.DataFrame:
    """Distancia relativa a la media móvil: sitúa el precio dentro de su tendencia."""
    m = sma(px, n)
    return px / m - 1.0


def slope_r2(px: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regresión log-lineal móvil: pendiente anualizada y R2.

    La pendiente dice cuánto sube; el R2 dice si sube de forma ordenada o a
    trompicones. Una tendencia con R2 alto es mucho más explotable que una con la
    misma pendiente y R2 bajo, y casi ningún sistema mira esto.
    """
    y = np.log(px.where(px > 0))
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()

    ymean = y.rolling(n, min_periods=n).mean()
    # cov(x,y) móvil = sum((x-xm)*y)/n  (el término xm*sum(y) se cancela)
    cov = y.rolling(n, min_periods=n).apply(lambda v: np.dot(v - v.mean(), x - xm), raw=True)
    beta = cov / sxx
    yvar = y.rolling(n, min_periods=n).var(ddof=0) * n
    r2 = (beta ** 2 * sxx) / yvar.replace(0, np.nan)
    return beta * 252, r2.clip(0, 1)


# --------------------------------------------------------------------------- #
# momento
# --------------------------------------------------------------------------- #
def roc(px: pd.DataFrame, n: int) -> pd.DataFrame:
    return px / px.shift(n) - 1.0


def momentum_skip(px: pd.DataFrame, n: int, skip: int = 21) -> pd.DataFrame:
    """Momento clásico 12-1: se salta el último mes porque revierte a corto."""
    return px.shift(skip) / px.shift(n) - 1.0


def rsi(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd_hist(px: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    line = ema(px, fast) - ema(px, slow)
    return (line - line.ewm(span=sig, adjust=False).mean()) / px


# --------------------------------------------------------------------------- #
# volatilidad y riesgo
# --------------------------------------------------------------------------- #
def realized_vol(px: pd.DataFrame, n: int = 63) -> pd.DataFrame:
    return np.log(px / px.shift(1)).rolling(n, min_periods=n // 2).std() * np.sqrt(252)


def true_range(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    pc = c.shift(1)
    return pd.concat([(h - l).stack(), (h - pc).abs().stack(), (l - pc).abs().stack()],
                     axis=1).max(axis=1).unstack()


def atr(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    return true_range(h, l, c).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def drawdown(px: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """Caída desde el máximo (ventana móvil si n, histórico si no)."""
    peak = px.rolling(n, min_periods=1).max() if n else px.cummax()
    return px / peak - 1.0


def ulcer(px: pd.DataFrame, n: int = 126) -> pd.DataFrame:
    """Índice Ulcer: penaliza caídas profundas y largas, no la volatilidad al alza."""
    dd = drawdown(px, n)
    return np.sqrt((dd ** 2).rolling(n, min_periods=n // 2).mean())


# --------------------------------------------------------------------------- #
# estructura de precio
# --------------------------------------------------------------------------- #
def donchian_pos(px: pd.DataFrame, n: int = 252) -> pd.DataFrame:
    """Posición dentro del rango de n sesiones: 1 = máximo, 0 = mínimo."""
    hi = px.rolling(n, min_periods=n // 3).max()
    lo = px.rolling(n, min_periods=n // 3).min()
    return (px - lo) / (hi - lo).replace(0, np.nan)


def bollinger_z(px: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    m = px.rolling(n, min_periods=n // 2).mean()
    s = px.rolling(n, min_periods=n // 2).std()
    return (px - m) / s.replace(0, np.nan)


def adx(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Fuerza de la tendencia con independencia de su dirección."""
    up, dn = h.diff(), -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = true_range(h, l, c).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def volume_surge(v: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    base = v.rolling(n, min_periods=n // 2).median()
    return v / base.replace(0, np.nan)


# --------------------------------------------------------------------------- #
# relativos al mercado
# --------------------------------------------------------------------------- #
def beta_alpha(px: pd.DataFrame, bench: pd.Series, n: int = 126
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = np.log(px / px.shift(1))
    rb = np.log(bench / bench.shift(1)).reindex(r.index)
    cov = r.rolling(n, min_periods=n // 2).cov(rb)
    var = rb.rolling(n, min_periods=n // 2).var()
    beta = cov.div(var.replace(0, np.nan), axis=0)
    alpha = (r.rolling(n, min_periods=n // 2).mean()
             - beta.mul(rb.rolling(n, min_periods=n // 2).mean(), axis=0)) * 252
    return beta, alpha


def relative_strength(px: pd.DataFrame, bench: pd.Series, n: int = 126) -> pd.DataFrame:
    rel = px.div(bench.reindex(px.index), axis=0)
    return rel / rel.shift(n) - 1.0
