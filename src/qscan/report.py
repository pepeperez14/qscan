"""Dashboard HTML autocontenido (un solo fichero, sin dependencias externas)."""

from __future__ import annotations

import html
import json
from datetime import datetime

import numpy as np
import pandas as pd

HORIZON_LABEL = {"corto": "Corto plazo", "medio": "Medio plazo", "largo": "Largo plazo"}
HORIZON_SUB = {"corto": "1-3 semanas", "medio": "1-6 meses", "largo": "6-24 meses"}

CSS = """
*{box-sizing:border-box}
.viz-root{color-scheme:light;
  --surface-1:#fcfcfb; --surface-2:#f4f4f2; --border:#e2e2dd;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#8a8981;
  --series-1:#2a78d6; --good:#008300; --bad:#e34948; --warn:#eda100;
  background:var(--surface-1); color:var(--text-primary);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  margin:0;padding:24px;max-width:1180px;margin-inline:auto}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark; --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
  --series-1:#3987e5; --good:#008300; --bad:#e66767; --warn:#c98500;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
  --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
  --series-1:#3987e5; --good:#008300; --bad:#e66767; --warn:#c98500;}
h1{font-size:22px;margin:0 0 2px} h2{font-size:15px;margin:32px 0 10px;font-weight:600}
.sub{color:var(--text-secondary);font-size:13px;margin:0 0 20px}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 4px}
.tile{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px;min-width:132px;flex:1}
.tile .k{font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--text-muted)}
.tile .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}
.tabs{display:flex;gap:6px;margin:22px 0 12px;flex-wrap:wrap}
.tab{padding:7px 14px;border:1px solid var(--border);border-radius:999px;cursor:pointer;
  background:var(--surface-1);color:var(--text-secondary);font-size:13px}
.tab[aria-selected="true"]{background:var(--series-1);border-color:var(--series-1);color:#fff}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-weight:600;color:var(--text-secondary);font-size:11px;
  text-transform:uppercase;letter-spacing:.04em;padding:8px 8px;border-bottom:1px solid var(--border)}
td{padding:8px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}
tr:hover td{background:var(--surface-2)}
tr.cmt td{color:var(--text-secondary);font-size:12.5px;padding-top:0;
  border-bottom:1px solid var(--border);line-height:1.55;max-width:0}
.num{text-align:right} .sym{font-weight:600}
.tag.dup{border-color:var(--warn);color:var(--warn)}
.tag{font-size:11px;color:var(--text-secondary);background:var(--surface-2);
  border:1px solid var(--border);border-radius:5px;padding:1px 6px}
.pos{color:var(--good)} .neg{color:var(--bad)}
.note{background:var(--surface-2);border:1px solid var(--border);border-left:3px solid var(--warn);
  border-radius:8px;padding:12px 14px;color:var(--text-secondary);font-size:13px;margin:18px 0}
.hidden{display:none}
figure{margin:0}
figcaption{font-size:12px;color:var(--text-secondary);margin-bottom:8px}
.bar{fill:var(--series-1)}
.bar:hover{opacity:.82}
.axis{fill:var(--text-muted);font-size:10px}
.grid{stroke:var(--border);stroke-width:1}
footer{margin-top:36px;color:var(--text-muted);font-size:12px;
  border-top:1px solid var(--border);padding-top:14px}
"""


def _spark(series: np.ndarray, w: int = 78, h: int = 22) -> str:
    """Sparkline de una sola serie: 2px, sin ejes, sin leyenda (el título la nombra)."""
    v = np.asarray([x for x in series if np.isfinite(x)], dtype=float)
    if len(v) < 3:
        return ""
    lo, hi = v.min(), v.max()
    rng = hi - lo or 1.0
    xs = np.linspace(1, w - 1, len(v))
    ys = h - 2 - (v - lo) / rng * (h - 4)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    color = "var(--good)" if v[-1] >= v[0] else "var(--bad)"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _histogram(values: pd.Series, title: str, bins: int = 24,
               w: int = 560, h: int = 150) -> str:
    v = values.dropna().astype(float)
    if len(v) < 20:
        return ""
    counts, edges = np.histogram(v, bins=bins)
    pad_l, pad_b, pad_t = 34, 22, 6
    iw, ih = w - pad_l - 8, h - pad_b - pad_t
    bw = iw / bins
    mx = counts.max() or 1
    bars = []
    for i, c in enumerate(counts):
        bh = c / mx * ih
        x, y = pad_l + i * bw + 1, pad_t + ih - bh
        bars.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{max(bw - 2, 1):.1f}" '
            f'height="{bh:.1f}" rx="4" ry="4">'
            f'<title>{edges[i]:.2f} a {edges[i+1]:.2f}: {c} activos</title></rect>')
    ticks = []
    for frac in (0, 0.5, 1):
        y = pad_t + ih - frac * ih
        ticks.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{w-8}" y2="{y:.1f}"/>'
                     f'<text class="axis" x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end">'
                     f'{int(mx*frac)}</text>')
    xl = (f'<text class="axis" x="{pad_l}" y="{h-6}">{edges[0]:.1f}</text>'
          f'<text class="axis" x="{w-8}" y="{h-6}" text-anchor="end">{edges[-1]:.1f}</text>')
    return (f'<figure><figcaption>{html.escape(title)}</figcaption>'
            f'<svg width="100%" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{html.escape(title)}">{"".join(ticks)}{"".join(bars)}{xl}</svg>'
            f'</figure>')


def _version() -> str:
    """Versión del código, visible en el informe publicado.

    Sirve para responder desde el navegador a "¿qué código generó esto?" sin
    entrar en el repositorio ni en los logs.
    """
    from pathlib import Path
    for p in (Path("VERSION"), Path(__file__).resolve().parents[2] / "VERSION"):
        try:
            return "v" + p.read_text().strip()
        except OSError:
            continue
    return "(versión desconocida)"


def _fmt(x, pct=False, dec=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if pct:
        cls = "pos" if x > 0 else ("neg" if x < 0 else "")
        return f'<span class="{cls}">{x*100:+.1f}%</span>'
    return f"{x:,.{dec}f}"


def _table(df: pd.DataFrame, horizon: str, sparks: dict[str, np.ndarray],
           names: dict[str, str], comments: dict[str, str] | None = None,
           redundant: dict[str, str] | None = None) -> str:
    comments = comments or {}
    redundant = redundant or {}
    head = ("<tr><th>#</th><th>Activo</th><th>Grupo</th><th class='num'>Precio</th>"
            "<th class='num'>1m</th><th class='num'>3m</th><th class='num'>12m</th>"
            "<th class='num'>Vol 12m</th><th class='num'>Percentil</th>"
            "<th>12 meses</th></tr>")
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        sym = html.escape(str(r.symbol))
        nm = html.escape(str(names.get(r.symbol, ""))[:44])
        dup = redundant.get(r.symbol)
        dup_tag = (f" <span class='tag dup' title='Correlación alta con un activo "
                   f"mejor situado'>≈ {html.escape(dup)}</span>") if dup else ""
        rows.append(
            f"<tr><td>{i}</td>"
            f"<td><span class='sym'>{sym}</span>{dup_tag}<br>"
            f"<span style='color:var(--text-muted);font-size:11px'>{nm}</span></td>"
            f"<td><span class='tag'>{html.escape(str(r.group))}</span></td>"
            f"<td class='num'>{_fmt(r.close)}</td>"
            f"<td class='num'>{_fmt(getattr(r,'ret_1m',None),pct=True)}</td>"
            f"<td class='num'>{_fmt(getattr(r,'ret_3m',None),pct=True)}</td>"
            f"<td class='num'>{_fmt(getattr(r,'ret_12m',None),pct=True)}</td>"
            f"<td class='num'>{_fmt(getattr(r,'vol_252',None)*100 if np.isfinite(getattr(r,'vol_252',np.nan)) else None,dec=0)}</td>"
            f"<td class='num'><b>{_fmt(getattr(r,f'pct_{horizon}',None),dec=1)}</b></td>"
            f"<td>{_spark(sparks.get(r.symbol, np.array([])))}</td></tr>")
        txt = comments.get(r.symbol)
        if txt:
            rows.append(f"<tr class='cmt'><td></td><td colspan='9'>"
                        f"{html.escape(txt)}</td></tr>")
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def build_html(scored: pd.DataFrame, panel: pd.DataFrame, sparks: dict[str, np.ndarray],
               names: dict[str, str], verdict: pd.DataFrame,
               universe_size: int, top_n: int = 30,
               comments: dict[str, dict[str, str]] | None = None,
               anomalies: pd.DataFrame | None = None,
               quarantined: int = 0,
               redundant: dict[str, dict[str, str]] | None = None) -> str:
    comments = comments or {}
    redundant = redundant or {}
    asof = pd.to_datetime(scored.date.max())
    last = scored[scored.date == scored.date.max()]
    feats = panel[panel.date == panel.date.max()][
        ["symbol", "ret_1m", "ret_3m", "ret_12m", "vol_252"]]
    last = last.merge(feats, on="symbol", how="left")

    tiles = [("Activos analizados", f"{len(last):,}"),
             ("Universo descargado", f"{universe_size:,}"),
             ("En cuarentena", f"{quarantined:,}"),
             ("Grupos", f"{last.group.nunique()}"),
             ("Fecha de corte", asof.strftime("%d/%m/%Y"))]
    tiles_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in tiles)

    tabs, panes = [], []
    for i, h in enumerate(("corto", "medio", "largo")):
        sel = "true" if i == 0 else "false"
        tabs.append(f'<button class="tab" role="tab" aria-selected="{sel}" '
                    f'data-pane="p-{h}">{HORIZON_LABEL[h]}</button>')
        # se ordena por percentil dentro del grupo, no por z bruto: comparar el
        # z de una cripto con el de un bono no significa nada aunque el número
        # exista, y ordenar por él llena el top de la clase de activo más volátil
        d = (last.dropna(subset=[f"score_{h}"])
                 .sort_values([f"pct_{h}", f"score_{h}"], ascending=False)
                 .head(top_n))
        hist = _histogram(last[f"score_{h}"],
                          f"Distribución del score — {HORIZON_LABEL[h]} "
                          f"({len(last.dropna(subset=[f'score_{h}'])):,} activos)")
        panes.append(
            f'<div id="p-{h}" class="pane{"" if i==0 else " hidden"}">'
            f'<h2>{HORIZON_LABEL[h]} · {HORIZON_SUB[h]} — mejor posicionados</h2>'
            f'{_table(d, h, sparks, names, comments.get(h), redundant.get(h))}'
            f'<h2>Dónde cae el resto</h2>{hist}</div>')

    ver = ""
    if verdict is not None and not verdict.empty:
        rows = "".join(
            f"<tr><td>{html.escape(str(r.horizonte))}</td><td>{html.escape(str(r.grupo))}</td>"
            f"<td class='num'>{r.ic_medio:+.3f}</td><td class='num'>{r.t_stat:+.2f}</td>"
            f"<td class='num'>{r.periodos}</td><td>{html.escape(str(r.veredicto))}</td></tr>"
            for r in verdict.itertuples())
        ver = ("<h2>Validación walk-forward</h2>"
               "<table><thead><tr><th>Horizonte</th><th>Grupo</th><th class='num'>IC medio</th>"
               "<th class='num'>t-stat</th><th class='num'>Periodos</th><th>Veredicto</th>"
               f"</tr></thead><tbody>{rows}</tbody></table>")

    anom = ""
    if anomalies is not None and not anomalies.empty:
        rows = "".join(
            f"<tr><td>{html.escape(str(r.anomalia).replace('_',' '))}</td>"
            f"<td class='num'>{int(r.activos):,}</td>"
            f"<td class='num'>{r.severidad:.1f}</td></tr>"
            for r in anomalies.itertuples())
        anom = ("<h2>Control de calidad de los datos</h2>"
                "<p class='sub'>Series con problemas detectados. Las que superan "
                "severidad 4 quedan en cuarentena y no entran en el ranking: un solo "
                "dato malo contamina los 35 indicadores del activo.</p>"
                "<table><thead><tr><th>Anomalía</th><th class='num'>Activos</th>"
                f"<th class='num'>Severidad</th></tr></thead><tbody>{rows}</tbody></table>")

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Escáner multiactivo — {asof:%d/%m/%Y}</title><style>{CSS}</style></head>
<body class="viz-root">
<h1>Escáner multiactivo</h1>
<p class="sub">Análisis de la evolución de cotizaciones en tres horizontes ·
generado {datetime.utcnow():%d/%m/%Y %H:%M} UTC</p>
<div class="tiles">{tiles_html}</div>
<div class="note"><b>Esto no es asesoramiento de inversión.</b> El score es un
ranking transversal derivado únicamente del historial de precios y volumen: dice
qué activos están mejor posicionados <i>según estas reglas</i>, no qué va a subir.
Antes de usarlo, mira la tabla de validación: si el t-stat no supera 2, el orden
del ranking no se distingue del azar.</div>
<div class="tabs" role="tablist">{"".join(tabs)}</div>
{"".join(panes)}
{ver}
{anom}
<footer>Generado por qscan {_version()} · datos ajustados por splits y dividendos ·
sin datos fundamentales ni de sentimiento: sólo precio y volumen.</footer>
<script>
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{{
  document.querySelectorAll('.tab').forEach(x=>x.setAttribute('aria-selected','false'));
  t.setAttribute('aria-selected','true');
  document.querySelectorAll('.pane').forEach(p=>p.classList.add('hidden'));
  document.getElementById(t.dataset.pane).classList.remove('hidden');
}}));
</script></body></html>"""
