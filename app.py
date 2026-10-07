"""
app.py
======
Flask web server for the Financial Portfolio & Risk-Return Analyzer.

Usage
-----
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from __future__ import annotations

import io
import logging
import sys
from dataclasses import asdict

# Re-wrap stdout as UTF-8 on Windows so Unicode characters print correctly.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from data_loader import fetch_closing_prices
from analytics_engine import AssetMetrics, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

app = Flask(__name__)

DEFAULT_TICKERS = "AAPL,MSFT,GOOGL,AMZN"
DEFAULT_YEARS = 3
DEFAULT_RISK_FREE = 0.04


def _fmt_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def _fmt_date(ts) -> str:
    if ts is None:
        return "—"
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _metrics_to_rows(metrics: list[AssetMetrics]) -> list[dict]:
    rows = []
    for m in metrics:
        dd = m.drawdown
        rows.append({
            "ticker": m.ticker,
            "ann_return": _fmt_pct(m.annualised_return),
            "ann_return_raw": round(m.annualised_return * 100, 2),
            "ann_volatility": _fmt_pct(m.annualised_volatility),
            "ann_volatility_raw": round(m.annualised_volatility * 100, 2),
            "sharpe": f"{m.sharpe_ratio:.3f}",
            "sharpe_raw": round(m.sharpe_ratio, 3),
            "max_drawdown": _fmt_pct(dd.max_drawdown),
            "max_drawdown_raw": round(dd.max_drawdown * 100, 2),
            "peak_date": _fmt_date(dd.peak_date),
            "trough_date": _fmt_date(dd.trough_date),
            "recovery_date": _fmt_date(dd.recovery_date),
            "is_portfolio": m.ticker == "Portfolio",
        })
    return rows


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    rows = []
    chart_data = {}
    tickers_str = DEFAULT_TICKERS
    years = DEFAULT_YEARS
    risk_free = DEFAULT_RISK_FREE
    weights_str = ""

    if request.method == "POST":
        tickers_str = request.form.get("tickers", DEFAULT_TICKERS).upper().strip()
        years_val = request.form.get("years", str(DEFAULT_YEARS))
        rf_val = request.form.get("risk_free", str(DEFAULT_RISK_FREE))
        weights_str = request.form.get("weights", "").strip()

        try:
            years = int(years_val)
            risk_free = float(rf_val)
            tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

            if not tickers:
                raise ValueError("Please enter at least one ticker symbol.")

            if weights_str:
                weights = np.array([float(w.strip()) for w in weights_str.split(",")], dtype=float)
                if len(weights) != len(tickers):
                    raise ValueError(f"Number of weights ({len(weights)}) must match number of tickers ({len(tickers)}).")
                if not np.isclose(weights.sum(), 1.0, atol=1e-4):
                    raise ValueError(f"Weights must sum to 1. Current sum: {weights.sum():.4f}.")
            else:
                weights = np.full(len(tickers), 1.0 / len(tickers))

            prices: pd.DataFrame = fetch_closing_prices(tickers=tickers, years=years)

            loaded_tickers = prices.columns.tolist()
            if loaded_tickers != tickers:
                weights = np.full(len(loaded_tickers), 1.0 / len(loaded_tickers))
                tickers_str = ",".join(loaded_tickers)

            metrics = compute_metrics(prices=prices, weights=weights, risk_free_rate=risk_free)
            rows = _metrics_to_rows(metrics)

            asset_rows = [r for r in rows if not r["is_portfolio"]]
            chart_data = {
                "labels": [r["ticker"] for r in asset_rows],
                "returns": [r["ann_return_raw"] for r in asset_rows],
                "volatilities": [r["ann_volatility_raw"] for r in asset_rows],
                "sharpes": [r["sharpe_raw"] for r in asset_rows],
            }

        except (RuntimeError, ValueError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        rows=rows,
        chart_data=chart_data,
        error=error,
        tickers_str=tickers_str,
        years=years,
        risk_free=risk_free,
        weights_str=weights_str,
    )


if __name__ == "__main__":
    app.run(debug=True)
