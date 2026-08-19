"""Dashboard HTML autocontenido (un solo fichero, sin dependencias externas)."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# `portfolio` no importa `report`, así que no hay ciclo. Se necesita `a_json`:
# lo que se embebe en la página sale de las mismas matrices float32 que rompían
# al guardar el estado.
from .portfolio import a_json

HORIZON_LABEL = {"corto": "Corto plazo", "medio": "Medio plazo", "largo": "Largo plazo"}
HORIZON_SUB = {"corto": "1-3 semanas", "medio": "1-6 meses", "largo": "6-24 meses"}

CSS = """
*{box-sizing:border-box}
.viz-root{color-scheme:light;
  --surface-1:#fcfcfb; --surface-2:#f4f4f2; --border:#e2e2dd;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#8a8981;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --good:#008300; --bad:#e34948; --warn:#eda100;
  background:var(--surface-1); color:var(--text-primary);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  margin:0;padding:24px;max-width:1180px;margin-inline:auto}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark; --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --good:#008300; --bad:#e66767; --warn:#c98500;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
  --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --good:#008300; --bad:#e66767; --warn:#c98500;}
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
.leyenda{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:12px;
  color:var(--text-secondary)}
.lg{display:inline-flex;align-items:center;gap:6px}
.lg i{width:10px;height:10px;border-radius:2px;display:inline-block}
.lbl{font-size:11px;font-weight:600}
.paneles{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:12px;margin-top:14px}
.panel{background:var(--surface-2);border:1px solid var(--border);
  border-radius:10px;padding:10px 12px}
.panel .k{font-size:11px;letter-spacing:.04em;text-transform:uppercase;
  color:var(--text-muted);margin-bottom:4px}
.comp details{border:1px solid var(--border);border-radius:10px;padding:10px 14px;
  margin-bottom:10px;background:var(--surface-2)}
.comp summary{cursor:pointer;font-size:13px;color:var(--text-secondary)}
.comp table{margin-top:8px}
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


SERIES = {  # slots 1-3 de la paleta: el único trío que valida en todos los pares
    "trade_republic": ("Trade Republic", "var(--series-1)"),
    "etoro": ("eToro", "var(--series-2)"),
    "benchmark": ("Comprar y esperar", "var(--series-3)"),
}
ESC_ETIQUETA = {"corto": "Corto plazo", "medio": "Medio plazo",
                "largo": "Largo plazo", "combinada": "Combinada"}
ESC_DETALLE = {"corto": "15 posiciones · rebalanceo semanal",
               "medio": "20 posiciones · rebalanceo mensual",
               "largo": "20 posiciones · rebalanceo trimestral",
               "combinada": "20 posiciones · reparto por evidencia"}
PERIODOS = [("7", "1 semana"), ("30", "1 mes"), ("90", "3 meses"),
            ("180", "6 meses"), ("365", "1 año"), ("0", "Todo")]
MUTED = "color:var(--text-muted);font-size:11px"


def _celda(txt: str, cls: str = "") -> str:
    return '<td class="num %s">%s</td>' % (cls, txt)


def _tabla_composicion(comp: pd.DataFrame) -> str:
    """Qué hay dentro de cada cartera. Se enseña siempre, no bajo demanda: una
    cartera cuyo contenido no ves es un número, no una decisión."""
    if comp is None or comp.empty:
        return ("<h2>Composición de las carteras</h2><p class='sub'>Todavía sin "
                "posiciones: aparecerán tras el primer rebalanceo.</p>")
    bloques = []
    for e, etiqueta in ESC_ETIQUETA.items():
        sub = comp[comp.escenario == e]
        if sub.empty:
            continue
        sub = sub.sort_values("valor_eur", ascending=False)
        filas = []
        for r in sub.itertuples():
            pl = getattr(r, "plusvalia_pct", None)
            try:
                pl_ok = pl is not None and np.isfinite(float(pl))
            except (TypeError, ValueError):
                pl_ok = False
            if pl_ok:
                cls = "pos" if float(pl) >= 0 else "neg"
                pl_txt = "%+.2f%%" % float(pl)
            else:
                cls, pl_txt = "", "—"
            grupo = str(getattr(r, "grupo", "") or "")
            tag = "<span class='tag'>%s</span>" % html.escape(grupo) if grupo else ""
            filas.append(
                "<tr><td class='sym'>%s %s<br><span style='%s'>%s</span></td>%s%s%s</tr>"
                % (html.escape(str(r.symbol)), tag, MUTED,
                   html.escape(str(getattr(r, "nombre", "") or "")),
                   _celda(format(float(r.valor_eur), ",.0f") + " €"),
                   _celda("%.1f%%" % float(r.peso_pct)),
                   _celda(pl_txt, cls)))
        bloques.append(
            "<details open><summary><b>%s</b> · %d posiciones · %s €</summary>"
            "<table><thead><tr><th>Activo</th><th class='num'>Valor</th>"
            "<th class='num'>Peso</th><th class='num'>Plusvalía</th></tr></thead>"
            "<tbody>%s</tbody></table></details>"
            % (html.escape(etiqueta), len(sub[sub.symbol != "· efectivo"]),
               format(float(sub.valor_eur.sum()), ",.0f"), "".join(filas)))
    return ("<h2>Composición de las carteras</h2>"
            "<p class='sub'>Posiciones a cierre de hoy, valoradas en euros. Ambos "
            "brókers compran exactamente los mismos activos —el objetivo lo fija la "
            "señal, no el bróker—, así que se muestra una sola composición; lo que "
            "cambia entre ellos es el coste.</p>"
            "<div class='comp'>" + "".join(bloques) + "</div>")


def _tabla_ops(ops, titulo: str, nota: str) -> str:
    df = ops if isinstance(ops, pd.DataFrame) else pd.DataFrame(ops)
    if df is None or df.empty:
        return ("<h2>%s</h2><p class='sub'>%s Hoy no hay ninguna: cada cartera sólo "
                "rota en su fecha de rebalanceo y entre medias se mantiene. No operar "
                "es la postura por defecto, y es donde se ahorran las comisiones.</p>"
                % (html.escape(titulo), html.escape(nota)))
    df = df.sort_values(["escenario", "broker", "lado", "symbol"])
    filas = []
    for r in df.itertuples():
        lado = str(r.lado)
        cls = "pos" if lado == "compra" else "neg"
        imp = getattr(r, "importe_eur", None)
        try:
            imp_txt = format(float(imp), ",.0f") + " €" if imp is not None \
                and np.isfinite(float(imp)) else "toda la posición"
        except (TypeError, ValueError):
            imp_txt = "toda la posición"
        coste = getattr(r, "coste_eur", None)
        try:
            coste_txt = format(float(coste), ",.2f") + " €" if coste is not None \
                and np.isfinite(float(coste)) else "—"
        except (TypeError, ValueError):
            coste_txt = "—"
        filas.append(
            "<tr><td><span class='tag'>%s</span></td><td>%s</td>"
            "<td class='sym'>%s</td><td class='%s'><b>%s</b></td>%s%s</tr>"
            % (html.escape(ESC_ETIQUETA.get(str(r.escenario), str(r.escenario))),
               html.escape(SERIES.get(str(r.broker), (str(r.broker), ""))[0]),
               html.escape(str(r.symbol)), cls, lado.upper(),
               _celda(imp_txt), _celda(coste_txt)))
    return ("<h2>%s <span class='tag'>%d</span></h2><p class='sub'>%s</p>"
            "<table><thead><tr><th>Escenario</th><th>Bróker</th><th>Activo</th>"
            "<th>Operación</th><th class='num'>Importe</th>"
            "<th class='num'>Comisión</th></tr></thead><tbody>%s</tbody></table>"
            % (html.escape(titulo), len(df), html.escape(nota), "".join(filas)))


def _tabla_descartes(estado: dict | None) -> str:
    """Lo que el sistema decidió NO hacer, y por qué.

    Es la parte más útil de un optimizador de costes y la que nadie enseña: las
    operaciones que no se hacen no dejan rastro en ningún sitio, así que sin esta
    tabla es imposible saber si el filtro está trabajando o está apagado.
    """
    if not estado:
        return ""
    filas, ahorro_total, ahorro_acum = [], 0.0, 0.0
    for e, esc in (estado.get("escenarios") or {}).items():
        for k, b in (esc.get("brokers") or {}).items():
            ahorro_acum += float(b.get("ahorro_acumulado") or 0.0)
            if k != "trade_republic":
                continue
            for d in b.get("descartadas") or []:
                ahorro_total += float(d.get("coste_evitado_eur") or 0.0)
                filas.append(
                    "<tr><td><span class='tag'>%s</span></td><td class='sym'>%s</td>"
                    "<td>%s</td><td>%s</td>%s</tr>"
                    % (html.escape(ESC_ETIQUETA.get(e, e)),
                       html.escape(str(d.get("symbol", ""))),
                       html.escape(str(d.get("lado", ""))),
                       html.escape(str(d.get("motivo", ""))),
                       _celda(format(float(d.get("coste_evitado_eur") or 0), ",.2f") + " €")))
    cabecera = ("<h2>Operaciones descartadas por coste</h2>"
                "<p class='sub'>El sistema sólo opera cuando el beneficio esperado "
                "supera el coste con un margen de 1,5x. El beneficio esperado sale de "
                "la fórmula <b>IC × z × σ</b>, con el IC que mide la propia validación: "
                "si un grupo no tiene señal, su IC ronda cero, ninguna rotación "
                "compensa la comisión y la cartera se queda quieta. Dejar de operar "
                "cuando no sabes nada es una decisión, no una avería. "
                "Ahorro acumulado estimado: <b>%s €</b>.</p>"
                % format(ahorro_acum, ",.0f"))
    if not filas:
        return cabecera + ("<p class='sub'>Hoy no se descartó ninguna: o no tocaba "
                           "rebalanceo, o todas las candidatas cubrían su coste.</p>")
    return (cabecera + "<table><thead><tr><th>Escenario</th><th>Activo</th>"
            "<th>Operación</th><th>Motivo del descarte</th>"
            "<th class='num'>Coste evitado</th></tr></thead><tbody>"
            + "".join(filas) + "</tbody></table>")


def _bloque_cartera(curva: pd.DataFrame, estado: dict | None,
                    comp: pd.DataFrame | None) -> str:
    if curva is None or curva.empty or "escenario" not in curva.columns:
        return ""

    # Los datos van embebidos y el navegador recalcula al cambiar de periodo. Es
    # la única forma de tener un selector real en una página estática: si el
    # servidor precalculara los periodos, cada uno sería una página distinta.
    datos = (curva[["fecha", "escenario", "broker", "valor_eur", "costes_acum_eur"]]
             .to_dict(orient="records"))
    payload = json.dumps(a_json({"curva": datos,
                          "etiquetas": ESC_ETIQUETA,
                          "detalles": ESC_DETALLE,
                          "series": {k: v[1] for k, v in SERIES.items()},
                          "nombres": {k: v[0] for k, v in SERIES.items()}}),
                         ensure_ascii=False)

    botones = "".join(
        '<button class="tab per" data-dias="%s" aria-selected="%s">%s</button>'
        % (d, "true" if d == "0" else "false", html.escape(n)) for d, n in PERIODOS)

    paneles = "".join(
        '<div class="panel"><div class="k">%s</div>'
        '<svg id="g-%s" width="100%%" viewBox="0 0 300 128" role="img" '
        'aria-label="Evolución de %s"></svg></div>'
        % (html.escape(et), e, html.escape(et)) for e, et in ESC_ETIQUETA.items())

    n_dias = curva.fecha.nunique()
    aviso = ""
    if n_dias < 60:
        aviso = ("<div class='note'><b>Llevas %d sesión%s de simulación.</b> A este "
                 "plazo la diferencia con el índice es ruido. Y desconfía del "
                 "escenario que vaya ganando ahora: con cuatro carteras compitiendo, "
                 "que una destaque las primeras semanas es lo esperable aunque "
                 "ninguna tenga ventaja real.</div>"
                 % (n_dias, "es" if n_dias != 1 else ""))

    nota_comb = ""
    if estado:
        pe = (estado.get("escenarios", {}).get("combinada") or {}).get("pesos_evidencia")
        if pe and pe.get("nota"):
            nota_comb = ("<p class='sub'>La cartera combinada reparte %s. Se recalcula "
                         "en cada rebalanceo, así que sigue a la evidencia en lugar de "
                         "quedarse fija en una corazonada. Contrapartida honesta: los "
                         "pesos salen de la misma validación que mide el sistema, así "
                         "que algo de ajuste a los propios datos hay.</p>"
                         % html.escape(str(pe["nota"])))

    leyenda = "".join('<span class="lg"><i style="background:%s"></i>%s</span>'
                      % (c, n) for n, c in SERIES.values())
    ops_hoy = (estado or {}).get("_ops_hoy") or []
    pend = (estado or {}).get("_pendientes") or []

    return ("<h2>Carteras simuladas · 40.000 € en cada escenario</h2>"
            "<p class='sub'>Cuatro estrategias con el mismo capital, en paralelo con "
            "los costes de Trade Republic y de eToro, contra comprar y esperar un ETF "
            "del S&amp;P 500. Las órdenes se cruzan a la apertura del día siguiente al "
            "de la señal, con comisión y horquilla aplicadas.</p>"
            + aviso
            + "<div class='tabs' role='group' aria-label='Periodo'>" + botones + "</div>"
            + "<p class='sub' id='per-nota'></p>"
            + "<table id='tabla-esc'><thead><tr><th>Escenario</th>"
              "<th class='num'>TR</th><th class='num'>eToro</th>"
              "<th class='num'>vs índice</th><th class='num'>Costes TR / eToro</th>"
              "</tr></thead><tbody></tbody></table>"
            + nota_comb
            + "<div class='paneles'>" + paneles + "</div>"
            + "<div class='leyenda'>" + leyenda + "</div>"
            + _tabla_composicion(comp)
            + _tabla_ops(pend, "Órdenes para la próxima apertura",
                         "Esto es lo que habría que ejecutar mañana al abrir el mercado.")
            + _tabla_ops(ops_hoy, "Ejecutado hoy",
                         "Órdenes decididas ayer y cruzadas hoy a la apertura, ya con "
                         "su coste aplicado.")
            + _tabla_descartes(estado)
            + "<script id='datos-cartera' type='application/json'>" + payload
            + "</script>" + JS_CARTERA)


JS_CARTERA = """<script>
(function(){
  const D = JSON.parse(document.getElementById('datos-cartera').textContent);
  const fechas = [...new Set(D.curva.map(r=>r.fecha))].sort();
  const clave = (e,b)=>e+'|'+b;
  const mapa = {};
  D.curva.forEach(r=>{ (mapa[clave(r.escenario,r.broker)] ||= {})[r.fecha] = r; });
  const escenarios = Object.keys(D.etiquetas);
  const eur = v => v.toLocaleString('es-ES',{maximumFractionDigits:0});

  function ventana(dias){
    if(!dias) return fechas;
    const fin = new Date(fechas[fechas.length-1]);
    const ini = new Date(fin); ini.setDate(ini.getDate()-dias);
    const iso = ini.toISOString().slice(0,10);
    const w = fechas.filter(f=>f>=iso);
    return w.length>=2 ? w : fechas.slice(-2);
  }
  // Rentabilidad DEL PERIODO: primer valor de la ventana como base, no el capital
  // inicial. Si no, "3 meses" seguiría mostrando la rentabilidad desde el origen.
  function rent(e,b,w){
    const s = mapa[clave(e,b)]; if(!s) return null;
    const va = w.map(f=>s[f]).filter(Boolean);
    if(va.length<2) return null;
    const a = va[0].valor_eur, z = va[va.length-1].valor_eur;
    return {pct:(z/a-1)*100, valor:z,
            coste: va[va.length-1].costes_acum_eur - va[0].costes_acum_eur};
  }
  function dibujar(e,w){
    const svg = document.getElementById('g-'+e); if(!svg) return;
    const series = ['trade_republic','etoro'].map(b=>({b,s:mapa[clave(e,b)]}))
      .filter(x=>x.s);
    const bench = mapa[clave('benchmark','benchmark')];
    if(bench) series.push({b:'benchmark',s:bench});
    const W=300,H=128,pl=42,pr=8,pt=8,pb=16, iw=W-pl-pr, ih=H-pt-pb;
    let vals=[];
    series.forEach(x=>w.forEach(f=>{ if(x.s[f]) vals.push(x.s[f].valor_eur); }));
    if(vals.length<2){ svg.innerHTML=''; return; }
    let lo=Math.min(...vals), hi=Math.max(...vals);
    const m=(hi-lo)*0.12 || Math.max(Math.abs(hi)*0.01,1); lo-=m; hi+=m;
    const X=i=>pl+iw*i/Math.max(w.length-1,1);
    const Y=v=>pt+ih-(v-lo)/(hi-lo)*ih;
    let out='';
    [0,1].forEach(fr=>{ const v=lo+(hi-lo)*fr, y=Y(v);
      out+=`<line class="grid" x1="${pl}" y1="${y}" x2="${pl+iw}" y2="${y}"/>`+
           `<text class="axis" x="${pl-5}" y="${y+3}" text-anchor="end">${eur(v)}</text>`;});
    series.forEach(x=>{
      const pts=w.map((f,i)=>x.s[f]?`${X(i)},${Y(x.s[f].valor_eur)}`:null)
                 .filter(Boolean).join(' ');
      if(pts) out+=`<polyline points="${pts}" fill="none" stroke="${D.series[x.b]}" `+
                   `stroke-width="${x.b==='benchmark'?1.5:2}" stroke-linejoin="round"/>`;
    });
    const bw=iw/Math.max(w.length-1,1);
    w.forEach((f,i)=>{
      const det=series.filter(x=>x.s[f])
        .map(x=>`${D.nombres[x.b]}: ${eur(x.s[f].valor_eur)} €`).join(' · ');
      out+=`<rect x="${X(i)-bw/2}" y="${pt}" width="${bw}" height="${ih}" `+
           `fill="transparent"><title>${f} — ${det}</title></rect>`;
    });
    svg.innerHTML=out;
  }
  function pintar(dias){
    const w = ventana(dias);
    const bench = rent('benchmark','benchmark',w);
    const tb = document.querySelector('#tabla-esc tbody'); tb.innerHTML='';
    escenarios.forEach(e=>{
      const tr = rent(e,'trade_republic',w), et = rent(e,'etoro',w);
      if(!tr && !et) return;
      const cel = r => r ? `<td class="num"><b class="${r.pct>=0?'pos':'neg'}">`+
        `${r.pct>=0?'+':''}${r.pct.toFixed(2)}%</b><br>`+
        `<span style="color:var(--text-muted);font-size:11px">${eur(r.valor)} €</span></td>`
        : '<td class="num">—</td>';
      let dif='<td class="num">—</td>';
      if(tr && bench){ const d=tr.pct-bench.pct;
        dif=`<td class="num ${d>=0?'pos':'neg'}"><b>${d>=0?'+':''}${d.toFixed(2)}</b></td>`; }
      const costes = (tr&&et) ? `${eur(tr.coste)} € / ${eur(et.coste)} €` : '—';
      tb.insertAdjacentHTML('beforeend',
        `<tr><td><b>${D.etiquetas[e]}</b><br><span style="color:var(--text-muted);`+
        `font-size:11px">${D.detalles[e]||''}</span></td>${cel(tr)}${cel(et)}${dif}`+
        `<td class="num" style="font-size:12px">${costes}</td></tr>`);
      dibujar(e,w);
    });
    if(bench){
      tb.insertAdjacentHTML('beforeend',
        `<tr><td><b>Comprar y esperar (SPY)</b><br><span style="color:var(--text-muted);`+
        `font-size:11px">referencia</span></td><td class="num"><b class="`+
        `${bench.pct>=0?'pos':'neg'}">${bench.pct>=0?'+':''}${bench.pct.toFixed(2)}%</b>`+
        `<br><span style="color:var(--text-muted);font-size:11px">${eur(bench.valor)} €`+
        `</span></td><td class="num">—</td><td class="num">—</td>`+
        `<td class="num">—</td></tr>`);
    }
    document.getElementById('per-nota').textContent =
      `Rentabilidad acumulada del ${w[0]} al ${w[w.length-1]} (${w.length} sesiones). `+
      `La base es el valor al inicio del periodo, no el capital inicial.`;
  }
  document.querySelectorAll('.per').forEach(b=>b.addEventListener('click',()=>{
    document.querySelectorAll('.per').forEach(x=>x.setAttribute('aria-selected','false'));
    b.setAttribute('aria-selected','true');
    pintar(parseInt(b.dataset.dias,10));
  }));
  pintar(0);
})();
</script>"""


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
               redundant: dict[str, dict[str, str]] | None = None,
               curva: pd.DataFrame | None = None,
               estado_cartera: dict | None = None,
               composicion: pd.DataFrame | None = None) -> str:
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
generado {datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC</p>
<div class="tiles">{tiles_html}</div>
<div class="note"><b>Esto no es asesoramiento de inversión.</b> El score es un
ranking transversal derivado únicamente del historial de precios y volumen: dice
qué activos están mejor posicionados <i>según estas reglas</i>, no qué va a subir.
Antes de usarlo, mira la tabla de validación: si el t-stat no supera 2, el orden
del ranking no se distingue del azar.</div>
<div class="tabs" role="tablist">{"".join(tabs)}</div>
{"".join(panes)}
{_bloque_cartera(curva, estado_cartera, composicion) if curva is not None else ""}
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
