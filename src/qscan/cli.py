"""Orquestación del pipeline. Cada paso es independiente y cacheable."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import anomalies, data, explain, features, report, scoring, universe, validate

log = logging.getLogger("qscan")

BENCH = "^GSPC"


def _wide(store: data.PriceStore) -> dict[str, pd.DataFrame]:
    df = store.load()
    if df.empty:
        raise SystemExit("almacén vacío: ejecuta primero `qscan update`")
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for f in ("open", "high", "low", "close", "volume"):
        out[f] = df.pivot(index="date", columns="symbol", values=f).sort_index()
    # imprescindible en cuanto hay cripto en el universo: ver data.to_business_calendar
    return data.to_business_calendar(out)


def cmd_universe(a) -> None:
    u = universe.build(include_crypto=not a.no_crypto, include_eu=not a.no_eu)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    u.to_csv(a.out, index=False)
    # Instantánea fechada del universo. Es lo único que permite reconstruir más
    # adelante un universo "point-in-time" y medir el sesgo de supervivencia:
    # hoy la lista sólo contiene valores vivos, así que cualquier validación
    # histórica es optimista. No se puede recuperar hacia atrás, sólo empezar a
    # guardarlo, y por eso conviene que exista desde la primera ejecución.
    snap = Path(a.out).parent / "snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
    u.to_csv(snap / f"universe_{stamp}.csv.gz", index=False, compression="gzip")
    print(f"{len(u)} activos -> {a.out} (instantánea {stamp})")
    print(u.group.value_counts().to_string())


def cmd_update(a) -> None:
    u = pd.read_csv(a.universe)
    if a.limit:
        u = u.head(a.limit)
    data.update(u, data.PriceStore(a.store), years=a.years)


def cmd_scan(a) -> None:
    store = data.PriceStore(a.store)
    px = _wide(store)
    u = pd.read_csv(a.universe)
    groups = u.set_index("symbol")["group"]

    bench = px["close"][BENCH] if BENCH in px["close"].columns else \
        px["close"].mean(axis=1)
    log.info("calculando features sobre %d activos", px["close"].shape[1])
    fh = features.build_feature_history(px, bench)

    dates = features.rebalance_dates(px["close"].index, a.freq)
    dates = dates[dates >= px["close"].index.min() + pd.Timedelta(days=400)]
    if a.history_months:
        dates = dates[-a.history_months:]
    # la última sesión disponible siempre entra: es la foto de hoy
    dates = pd.DatetimeIndex(sorted(set(dates) | {px["close"].index[-1]}))

    panel = features.sample_panel(fh, dates, px["close"], groups,
                                  min_dollar_vol=a.min_dollar_vol)
    if panel.empty:
        raise SystemExit("panel vacío: revisa filtros de liquidez o historia")

    # control de calidad ANTES de puntuar: un dato roto contamina 35 indicadores
    rep = anomalies.detect(px, min_dollar_vol=a.min_dollar_vol)
    rep.to_parquet(Path(a.out_dir) / "anomalies.parquet")
    panel, n_q = anomalies.apply_quarantine(panel, rep)
    log.info("cuarentena: %d activos apartados por calidad de datos", n_q)
    panel.to_parquet(Path(a.out_dir) / "panel.parquet", index=False)

    scored = scoring.score_panel(panel)
    scored.to_parquet(Path(a.out_dir) / "scores.parquet", index=False)

    fwd = features.forward_returns(px["close"], panel)
    fwd.to_parquet(Path(a.out_dir) / "forward.parquet", index=False)
    print(f"panel: {len(panel):,} filas · {panel.symbol.nunique():,} activos · "
          f"{panel.date.nunique()} fechas")


def cmd_validate(a) -> None:
    out = Path(a.out_dir)
    scored = pd.read_parquet(out / "scores.parquet")
    fwd = pd.read_parquet(out / "forward.parquet")
    res = validate.run_all(scored, fwd, cost_bps=a.cost_bps)
    v = validate.verdict(res)
    v.to_csv(out / "verdict.csv", index=False)
    for h, r in res.items():
        print(f"\n=== {h.upper()} ===")
        print(r["ic_resumen"].to_string() if not r["ic_resumen"].empty else "sin datos")
        if not r["neto"].empty:
            print(r["neto"].to_string())
    print("\n", v.to_string(index=False) if not v.empty else "sin veredicto")


def cmd_report(a) -> None:
    out = Path(a.out_dir)
    scored = pd.read_parquet(out / "scores.parquet")
    panel = pd.read_parquet(out / "panel.parquet")
    u = pd.read_csv(a.universe)
    names = u.set_index("symbol")["name"].to_dict()
    verdict = pd.read_csv(out / "verdict.csv") if (out / "verdict.csv").exists() else None

    close = _wide(data.PriceStore(a.store))["close"]
    last = scored[scored.date == scored.date.max()]
    rep = None
    if (out / "anomalies.parquet").exists():
        rep = pd.read_parquet(out / "anomalies.parquet")

    keep, per_h = set(), {}
    for h in ("corto", "medio", "largo"):
        syms = list(last.sort_values([f"pct_{h}", f"score_{h}"], ascending=False)
                        .head(a.top).symbol)
        per_h[h] = syms
        keep |= set(syms)
    tail = close.tail(252)
    sparks = {s: tail[s].to_numpy() for s in keep if s in tail.columns}

    redundant = {h: scoring.redundancy(syms, close, max_corr=a.max_corr)
                 for h, syms in per_h.items()}

    comments = {}
    if not a.no_comments:
        for h, syms in per_h.items():
            comments[h] = explain.annotate(scored, panel, h, syms[:a.comment_top],
                                           verdict, rep)

    html = report.build_html(scored, panel, sparks, names, verdict,
                             universe_size=len(u), top_n=a.top,
                             comments=comments,
                             anomalies=anomalies.summary(rep) if rep is not None else None,
                             quarantined=int(rep["cuarentena"].sum()) if rep is not None else 0,
                             redundant=redundant)
    dest = out / "index.html"
    dest.write_text(html, encoding="utf-8")
    print(f"informe -> {dest}")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser("qscan", description="Escáner multiactivo en tres horizontes")
    p.add_argument("--store", default="data/prices.parquet")
    p.add_argument("--universe", default="data/universe.csv")
    p.add_argument("--out-dir", default="out")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("universe"); u.add_argument("--out", default="data/universe.csv")
    u.add_argument("--no-crypto", action="store_true")
    u.add_argument("--no-eu", action="store_true"); u.set_defaults(f=cmd_universe)

    d = sub.add_parser("update"); d.add_argument("--years", type=int, default=8)
    d.add_argument("--limit", type=int, default=0); d.set_defaults(f=cmd_update)

    s = sub.add_parser("scan"); s.add_argument("--freq", default="ME")
    s.add_argument("--history-months", type=int, default=96)
    s.add_argument("--min-dollar-vol", type=float, default=1e6)
    s.set_defaults(f=cmd_scan)

    v = sub.add_parser("validate"); v.add_argument("--cost-bps", type=float, default=10.0)
    v.set_defaults(f=cmd_validate)

    r = sub.add_parser("report"); r.add_argument("--top", type=int, default=30)
    r.add_argument("--no-comments", action="store_true",
                   help="omite la capa de comentario")
    r.add_argument("--max-corr", type=float, default=0.80,
                   help="umbral para marcar activos redundantes entre sí")
    r.add_argument("--comment-top", type=int, default=12,
                   help="cuántos activos por horizonte reciben comentario")
    r.set_defaults(f=cmd_report)

    a = p.parse_args(argv)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    a.f(a)


if __name__ == "__main__":
    main()
