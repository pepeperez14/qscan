"""Genera un informe de ejemplo con datos sintéticos (sin red)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qscan import data, features, scoring, validate, report, anomalies, explain
from synthetic import make_market

prices, uni = make_market(n_days=1500, signal=0.25, seed=3)
prices = data.quality_filter(prices)
px = {f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
      for f in ("open","high","low","close","volume")}
fh = features.build_feature_history(px, px["close"]["^GSPC"])
dates = features.rebalance_dates(px["close"].index, "ME")
dates = dates[dates >= px["close"].index.min() + pd.Timedelta(days=400)]
panel = features.sample_panel(fh, dates, px["close"], uni.set_index("symbol")["group"], 0)
rep = anomalies.detect(px, min_dollar_vol=0.0)
panel, nq = anomalies.apply_quarantine(panel, rep)
print("cuarentena:", nq)
scored = scoring.score_panel(panel)
fwd = features.forward_returns(px["close"], panel)
v = validate.verdict(validate.run_all(scored, fwd))
last = scored[scored.date==scored.date.max()]
keep=set()
for h in ("corto","medio","largo"): keep |= set(last.sort_values([f"pct_{h}", f"score_{h}"], ascending=False).head(30).symbol)
tail = px["close"].tail(252)
sparks = {s: tail[s].to_numpy() for s in keep if s in tail.columns}
per_h = {h: list(last.sort_values([f"pct_{h}", f"score_{h}"], ascending=False).head(30).symbol)
         for h in ("corto","medio","largo")}
comments = {h: explain.annotate(scored, panel, h, s[:8], v, rep) for h, s in per_h.items()}
redundant = {h: scoring.redundancy(s, px["close"]) for h, s in per_h.items()}
print("redundantes:", {h: len(d) for h, d in redundant.items()})
html = report.build_html(scored, panel, sparks, uni.set_index("symbol")["name"].to_dict(),
                         v, universe_size=len(uni), top_n=30, comments=comments,
                         anomalies=anomalies.summary(rep),
                         quarantined=int(rep["cuarentena"].sum()), redundant=redundant)
Path("out").mkdir(exist_ok=True)
Path("out/index.html").write_text(html, encoding="utf-8")
print("ok", len(html), "bytes ·", len(last), "activos ·", len(v), "filas de veredicto")
