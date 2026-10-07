"""
main.py
=======
Execution entry-point for the Financial Portfolio & Risk-Return Analyzer.

Usage
-----
    python main.py

Configuration is done by editing the constants in the ``CONFIG`` block below.
No CLI arguments are required; the script prints a rich Markdown summary table
to stdout and writes it to ``results_summary.md`` in the project directory.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

# Re-wrap stdout as UTF-8 on Windows so Unicode characters print correctly.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from data_loader import fetch_closing_prices
from analytics_engine import AssetMetrics, compute_metrics

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit this block to customise the analysis
# ──────────────────────────────────────────────────────────────────────────────

TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Set to None to use today as end date, or supply "YYYY-MM-DD"
END_DATE: str | None = None

# Years of history to fetch
HISTORY_YEARS: int = 3

# Portfolio weights — must sum to 1 and align with TICKERS order.
# Set to None for equal weighting (default).
WEIGHTS: list[float] | None = None

# Annual risk-free rate (decimal), e.g. 0.04 = 4 %
RISK_FREE_RATE: float = 0.04

# Path where the Markdown summary is written
OUTPUT_FILE: Path = Path(__file__).parent / "results_summary.md"

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────


def _pct(value: float, decimals: int = 2) -> str:
    """Format a decimal fraction as a percentage string.

    Args:
        value: Decimal fraction, e.g. ``0.1234``.
        decimals: Number of decimal places.  Defaults to ``2``.

    Returns:
        Formatted string, e.g. ``"12.34 %"``.
    """
    return f"{value * 100:.{decimals}f} %"


def _fmt_date(ts: pd.Timestamp | None) -> str:
    """Format a Timestamp as ``YYYY-MM-DD``, or return ``"—"`` for ``None``.

    Args:
        ts: Timestamp or ``None``.

    Returns:
        ISO date string or an em-dash placeholder.
    """
    return ts.strftime("%Y-%m-%d") if ts is not None else "—"


def _build_markdown_table(metrics: list[AssetMetrics]) -> str:
    """Render the metrics list as a formatted Markdown table.

    Separates individual assets from the portfolio row with a visual
    divider line so the blended portfolio stands out.

    Args:
        metrics: List of :class:`~analytics_engine.AssetMetrics` dataclasses
            as returned by :func:`~analytics_engine.compute_metrics`.

    Returns:
        A multi-line string containing a valid GitHub-Flavoured Markdown table.
    """
    header = (
        "| Asset       | Ann. Return | Ann. Volatility | Sharpe Ratio "
        "| Max Drawdown | DD Peak Date | DD Trough Date | DD Recovery Date |\n"
        "|-------------|:-----------:|:---------------:|:------------:"
        "|:------------:|:------------:|:--------------:|:----------------:|"
    )
    rows: list[str] = [header]

    asset_metrics = [m for m in metrics if m.ticker != "Portfolio"]
    portfolio_metric = next(m for m in metrics if m.ticker == "Portfolio")

    def _row(m: AssetMetrics) -> str:
        dd = m.drawdown
        return (
            f"| {m.ticker:<11} "
            f"| {_pct(m.annualised_return):^11} "
            f"| {_pct(m.annualised_volatility):^15} "
            f"| {m.sharpe_ratio:^12.3f} "
            f"| {_pct(m.drawdown.max_drawdown):^12} "
            f"| {_fmt_date(dd.peak_date):^12} "
            f"| {_fmt_date(dd.trough_date):^14} "
            f"| {_fmt_date(dd.recovery_date):^16} |"
        )

    for m in asset_metrics:
        rows.append(_row(m))

    # Visual separator before portfolio row
    rows.append(
        "|-------------|:-----------:|:---------------:|:------------:"
        "|:------------:|:------------:|:--------------:|:----------------:|"
    )
    rows.append(_row(portfolio_metric))

    return "\n".join(rows)


def _build_full_report(
    metrics: list[AssetMetrics],
    tickers: list[str],
    weights: np.ndarray,
    risk_free_rate: float,
    history_years: int,
) -> str:
    """Assemble the complete Markdown report string.

    Args:
        metrics: Full list of asset + portfolio metrics.
        tickers: Original ticker list.
        weights: Weight vector used for the portfolio.
        risk_free_rate: Risk-free rate used for Sharpe calculations.
        history_years: Number of years of history fetched.

    Returns:
        A multi-line Markdown string ready to be printed or saved.
    """
    weight_str = ", ".join(
        f"{t}: {w * 100:.1f} %" for t, w in zip(tickers, weights)
    )

    table = _build_markdown_table(metrics)

    report = f"""# Financial Portfolio & Risk-Return Analyzer
## Results Summary

**Tickers analysed :** {", ".join(tickers)}
**History window   :** {history_years} years
**Portfolio weights:** {weight_str}
**Risk-free rate   :** {_pct(risk_free_rate)} p.a.

---

### Risk / Return Metrics

{table}

---

### Metric Definitions

| Metric | Definition |
|---|---|
| **Ann. Return** | Compound Annual Growth Rate (CAGR) = exp(mean daily log-return × 252) − 1 |
| **Ann. Volatility** | Annualised Std Dev of daily log-returns = σ_daily × √252 |
| **Sharpe Ratio** | (Ann. Return − Risk-Free Rate) ÷ Ann. Volatility |
| **Max Drawdown** | Maximum peak-to-trough equity decline over the sample window |
| **DD Peak / Trough** | Calendar dates bounding the worst drawdown episode |
| **DD Recovery** | First date equity recovered above the prior peak; "—" if still underwater |

> *All returns are based on split- and dividend-adjusted daily closing prices sourced from Yahoo Finance.*
"""
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full portfolio analysis pipeline.

    Steps:

    1. Resolve portfolio weights (equal-weight if not configured).
    2. Fetch adjusted closing prices via :mod:`data_loader`.
    3. Compute all risk/return metrics via :mod:`analytics_engine`.
    4. Print the Markdown summary to stdout.
    5. Write the same summary to :data:`OUTPUT_FILE`.
    """
    # ── Resolve weights ───────────────────────────────────────────────────────
    n = len(TICKERS)
    weights: np.ndarray = (
        np.array(WEIGHTS, dtype=float)
        if WEIGHTS is not None
        else np.full(n, 1.0 / n)
    )

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        logger.error(
            "Configured WEIGHTS sum to %.6f — they must sum to 1.  Aborting.",
            weights.sum(),
        )
        sys.exit(1)

    if len(weights) != n:
        logger.error(
            "WEIGHTS length (%d) does not match TICKERS length (%d).  Aborting.",
            len(weights),
            n,
        )
        sys.exit(1)

    # ── Fetch data ────────────────────────────────────────────────────────────
    try:
        prices: pd.DataFrame = fetch_closing_prices(
            tickers=TICKERS,
            years=HISTORY_YEARS,
            end_date=END_DATE,
        )
    except (RuntimeError, ValueError) as exc:
        logger.error("Data ingestion failed: %s", exc)
        sys.exit(1)

    # Align weights if any tickers were dropped during loading
    loaded_tickers: list[str] = prices.columns.tolist()
    if loaded_tickers != TICKERS:
        logger.warning(
            "Ticker mismatch after loading. Configured: %s | Loaded: %s. "
            "Falling back to equal weights for loaded tickers.",
            TICKERS,
            loaded_tickers,
        )
        weights = np.full(len(loaded_tickers), 1.0 / len(loaded_tickers))

    # ── Compute metrics ───────────────────────────────────────────────────────
    logger.info("Computing risk/return metrics …")
    metrics: list[AssetMetrics] = compute_metrics(
        prices=prices,
        weights=weights,
        risk_free_rate=RISK_FREE_RATE,
    )

    # ── Build & emit report ───────────────────────────────────────────────────
    report: str = _build_full_report(
        metrics=metrics,
        tickers=loaded_tickers,
        weights=weights,
        risk_free_rate=RISK_FREE_RATE,
        history_years=HISTORY_YEARS,
    )

    print("\n" + report)

    OUTPUT_FILE.write_text(report, encoding="utf-8")
    logger.info("Report written to → %s", OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
