"""Capa de lectura: convierte el ranking en comentario legible.

Regla de diseño, y es la que hace que esta capa sea segura: **el modelo de
lenguaje no calcula nada y no predice nada**. Recibe cifras ya calculadas
(aportación de cada componente, estadísticos de la serie, anomalías detectadas,
veredicto de la validación) y las verbaliza. Todo número que aparece en el texto
existía antes de llamar al modelo.

Por eso hay un generador determinista de respaldo: si no hay clave de API, el
informe sigue saliendo con comentarios algo más secos, pero con exactamente la
misma información. La capa de lenguaje es una comodidad, no una dependencia.
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("QSCAN_MODEL", "claude-sonnet-4-5")

FEATURE_ES = {
    "donch_20": "posición en el rango de 20 sesiones", "macd_hist": "histograma MACD",
    "ma20_dist": "distancia a la media de 20", "ma50_dist": "distancia a la media de 50",
    "ma200_dist": "distancia a la media de 200", "ma50_over_200": "cruce de medias 50/200",
    "adx14": "fuerza de tendencia (ADX)", "vol_surge": "repunte de volumen",
    "rs_3m": "fuerza relativa a 3 meses", "rs_12m": "fuerza relativa a 12 meses",
    "rsi2": "RSI de 2 sesiones", "ret_1w": "retorno de la última semana",
    "ret_3m": "retorno a 3 meses", "boll_z": "desviación de la banda de Bollinger",
    "vol_ratio": "volatilidad reciente frente a la anual",
    "trendfit_3m": "regularidad de la tendencia a 3 meses",
    "trendfit_12m": "regularidad de la tendencia a 12 meses",
    "dd_from_high": "caída desde máximos de 12 meses", "ulcer_6m": "índice Ulcer a 6 meses",
    "alpha_6m": "alfa a 6 meses", "mom_12_1": "momento 12-1",
    "slope_12m": "pendiente anualizada a 12 meses", "sharpe_12m": "Sharpe a 12 meses",
    "sortino_6m": "Sortino a 6 meses", "vol_252": "volatilidad anualizada",
}

SYSTEM = """Eres un analista cuantitativo que redacta el comentario de un escáner
de mercado en español de España. Reglas estrictas:

- NO predices. No digas que algo "va a subir", "tiene recorrido" ni "es una
  oportunidad". Describes cómo se ha comportado la serie y por qué el sistema lo
  coloca donde lo coloca.
- NO inventas cifras. Usa exclusivamente los números que se te dan. Si algo no
  está en los datos, no lo menciones.
- NO opinas sobre la empresa, el sector ni las noticias: sólo ves precio y volumen.
- Si el veredicto de validación dice que no hay señal en ese grupo y horizonte,
  dilo de forma explícita en la primera frase.
- Si hay anomalías de datos marcadas, adviértelo antes que nada.
- Dos o tres frases por activo. Directo, sin adjetivos de folleto.

Devuelve SOLO un objeto JSON: {"comentarios": {"SIMBOLO": "texto", ...}}"""


def _brief(sym: str, row: pd.Series, contrib: pd.DataFrame,
           verdict_txt: str, anomalies: list[str]) -> dict:
    """Ficha compacta de un activo: sólo cifras ya calculadas."""
    c = contrib[contrib.symbol == sym].sort_values("aportacion")
    top = c.tail(3).iloc[::-1]
    bottom = c.head(2)

    def _fmt(d):
        return [{"componente": FEATURE_ES.get(r.feature, r.feature),
                 "z": round(float(r.z), 2), "aportacion": round(float(r.aportacion), 3)}
                for r in d.itertuples()]

    out = {
        "simbolo": sym,
        "grupo": str(row.get("group", "")),
        "percentil_en_su_grupo": _num(row.get("pct")),
        "retorno_1m_pct": _num(row.get("ret_1m"), 100),
        "retorno_3m_pct": _num(row.get("ret_3m"), 100),
        "retorno_12m_pct": _num(row.get("ret_12m"), 100),
        "volatilidad_anual_pct": _num(row.get("vol_252"), 100),
        "caida_desde_maximos_pct": _num(row.get("dd_from_high"), 100),
        "a_favor": _fmt(top),
        "en_contra": _fmt(bottom),
        "desacuerdo_entre_componentes": _num(row.get("spread")),
        "validacion": verdict_txt,
    }
    if anomalies:
        out["anomalias_de_datos"] = anomalies
    return out


def _num(x, mult: float = 1.0):
    try:
        v = float(x) * mult
        return round(v, 1) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _fallback(brief: dict) -> str:
    """Comentario determinista. Misma información, redacción mecánica."""
    parts = []
    if brief.get("anomalias_de_datos"):
        parts.append("Aviso de datos: " + ", ".join(brief["anomalias_de_datos"]) + ".")
    if "sin evidencia" in brief["validacion"] or "no concluyente" in brief["validacion"]:
        parts.append(f"La validación de este grupo y horizonte dice "
                     f"«{brief['validacion']}»: trata el puesto con escepticismo.")
    fav = ", ".join(f"{a['componente']} (z {a['z']:+.2f})" for a in brief["a_favor"])
    con = ", ".join(f"{a['componente']} (z {a['z']:+.2f})" for a in brief["en_contra"])
    parts.append(f"Sube en el ranking sobre todo por {fav}." if fav else "")
    if con:
        parts.append(f"En contra: {con}.")
    r12, vol = brief.get("retorno_12m_pct"), brief.get("volatilidad_anual_pct")
    if r12 is not None and vol is not None:
        parts.append(f"Lleva {r12:+.1f}% en 12 meses con una volatilidad anual "
                     f"del {vol:.0f}%.")
    return " ".join(p for p in parts if p)


def _call_api(briefs: list[dict], api_key: str, timeout: int = 120) -> dict[str, str]:
    import requests

    payload = {
        "model": MODEL,
        "max_tokens": 4000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": json.dumps(
            {"activos": briefs}, ensure_ascii=False)}],
    }
    r = requests.post(API_URL, timeout=timeout, json=payload, headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json().get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("respuesta sin JSON")
    return json.loads(m.group(0)).get("comentarios", {})


def annotate(scored: pd.DataFrame, panel: pd.DataFrame, horizon: str,
             symbols: list[str], verdict: pd.DataFrame | None,
             anomaly_report: pd.DataFrame | None = None,
             batch: int = 12) -> dict[str, str]:
    """Genera un comentario por activo. Nunca falla: cae al modo determinista."""
    from . import scoring

    contrib = scoring.contributions(panel, horizon)
    if contrib.empty:
        return {}
    last = scored[scored.date == scored.date.max()].set_index("symbol")
    feats = panel[panel.date == panel.date.max()].set_index("symbol")

    ver = {}
    if verdict is not None and not verdict.empty:
        sub = verdict[verdict.horizonte == horizon]
        ver = dict(zip(sub.grupo, sub.veredicto))

    briefs = []
    for s in symbols:
        if s not in last.index:
            continue
        row = last.loc[s]
        g = str(row.get("group", ""))
        merged = pd.Series({
            "group": g, "pct": row.get(f"pct_{horizon}"),
            "spread": row.get(f"spread_{horizon}"),
            **{k: feats.loc[s].get(k) for k in
               ("ret_1m", "ret_3m", "ret_12m", "vol_252", "dd_from_high")
               if s in feats.index}})
        anom = []
        if anomaly_report is not None and s in anomaly_report.index:
            arow = anomaly_report.loc[s]
            anom = [k for k in arow.index if arow.get(k) is True]
        briefs.append(_brief(s, merged, contrib, ver.get(g, "sin validar"), anom))

    out = {b["simbolo"]: _fallback(b) for b in briefs}

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.info("sin ANTHROPIC_API_KEY: comentarios en modo determinista")
        return out

    for i in range(0, len(briefs), batch):
        chunk = briefs[i:i + batch]
        try:
            got = _call_api(chunk, key)
            for k, v in got.items():
                if k in out and isinstance(v, str) and v.strip():
                    out[k] = v.strip()
        except Exception as e:
            log.warning("lote %d de comentarios falló (%s); se mantiene el "
                        "texto determinista", i // batch, e)
    return out
