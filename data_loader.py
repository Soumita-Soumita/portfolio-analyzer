"""
data_loader.py
==============
Responsible for fetching and cleaning historical price data from Yahoo Finance.

All network I/O is isolated in this module so the rest of the application
can be tested independently with any DataFrame of closing prices.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def fetch_closing_prices(
    tickers: list[str],
    years: int = 3,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download adjusted daily closing prices for a list of tickers.

    Fetches data from Yahoo Finance via ``yfinance``, drops tickers for which
    no data was returned, forward-fills then back-fills any remaining gaps
    (e.g. staggered trading holidays), and reports any anomalies via the
    module logger.

    Args:
        tickers: List of valid Yahoo Finance ticker symbols, e.g.
            ``["AAPL", "MSFT", "GOOGL", "AMZN"]``.
        years: Number of calendar years of history to request.  Defaults
            to ``3``.
        end_date: ISO-8601 date string (``"YYYY-MM-DD"``) to use as the
            inclusive end of the window.  Defaults to today.

    Returns:
        A ``pd.DataFrame`` with a ``DatetimeIndex`` (trading days) and one
        column per successfully loaded ticker containing adjusted closing
        prices.  The DataFrame is guaranteed to have no ``NaN`` values on
        return — any ticker that is *entirely* empty is silently dropped and
        logged as a warning.

    Raises:
        ValueError: If *all* requested tickers fail to load, making further
            analysis impossible.

    Example:
        >>> prices = fetch_closing_prices(["AAPL", "MSFT"], years=3)
        >>> prices.shape[1]  # at most 2 columns
        2
    """
    end_dt: datetime = (
        datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()
    )
    start_dt: datetime = end_dt - timedelta(days=years * 365 + 5)  # small buffer

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    logger.info(
        "Fetching data for %d tickers from %s to %s …", len(tickers), start_str, end_str
    )

    last_exc: Exception | None = None
    for attempt in range(1, 4):          # up to 3 attempts
        try:
            raw: pd.DataFrame = yf.download(
                tickers=tickers,
                start=start_str,
                end=end_str,
                auto_adjust=True,        # use split/dividend-adjusted prices
                progress=False,
                threads=True,
            )
            break
        except Exception as exc:         # network errors, lock errors, etc.
            last_exc = exc
            logger.warning("Download attempt %d/3 failed: %s — retrying …", attempt, exc)
            time.sleep(2 ** attempt)     # exponential back-off: 2s, 4s
    else:
        raise RuntimeError(f"yfinance download failed after 3 attempts: {last_exc}") from last_exc

    # ── Extract the 'Close' layer (MultiIndex when >1 ticker) ─────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        prices: pd.DataFrame = raw["Close"]
    else:
        # Single-ticker download returns a flat DataFrame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # ── Drop fully-empty tickers and warn ─────────────────────────────────────
    empty_cols = prices.columns[prices.isna().all()].tolist()
    if empty_cols:
        logger.warning("No data returned for ticker(s): %s — dropping them.", empty_cols)
        prices = prices.drop(columns=empty_cols)

    if prices.empty:
        raise ValueError(
            "All requested tickers failed to load.  "
            "Check ticker symbols and network connectivity."
        )

    # ── Fill sparse NaN gaps (holidays, late listings) ────────────────────────
    prices = prices.ffill().bfill()

    loaded = prices.columns.tolist()
    logger.info("Successfully loaded %d ticker(s): %s", len(loaded), loaded)

    return prices
