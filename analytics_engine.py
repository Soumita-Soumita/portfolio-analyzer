"""
analytics_engine.py
===================
Pure-function risk/return analytics.  Every function is stateless and
operates exclusively on pandas Series / DataFrames and NumPy arrays.
No I/O occurs in this module.

Conventions
-----------
- ``prices``  : ``pd.DataFrame`` — daily adjusted closing prices, one column per asset.
- ``returns`` : ``pd.DataFrame`` — simple daily log-returns derived from ``prices``.
- ``weights`` : ``np.ndarray``   — 1-D array of floats that must sum to 1.
- ``TRADING_DAYS`` : 252 — the standard annualisation factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DrawdownResult:
    """Encapsulates the worst drawdown event for a single return series.

    Attributes:
        max_drawdown: Peak-to-trough decline as a *negative* decimal fraction,
            e.g. ``-0.3412`` for a 34.12 % drawdown.
        peak_date: Date of the equity-curve peak that precedes the trough.
        trough_date: Date of the maximum drawdown trough.
        recovery_date: First date the equity curve fully recovers above the
            prior peak.  ``None`` if recovery had not yet occurred within
            the sample window.
    """

    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]


@dataclass(frozen=True)
class AssetMetrics:
    """Risk/return summary for a single asset or the blended portfolio.

    Attributes:
        ticker: Asset label (e.g. ``"AAPL"`` or ``"Portfolio"``).
        annualised_return: Compound annualised growth rate (CAGR) as a decimal.
        annualised_volatility: Annualised standard deviation of daily log-returns.
        sharpe_ratio: Excess-return-per-unit-risk ratio.
        drawdown: Full drawdown detail encapsulated in a :class:`DrawdownResult`.
    """

    ticker: str
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    drawdown: DrawdownResult


# ──────────────────────────────────────────────────────────────────────────────
# Core vectorised helpers
# ──────────────────────────────────────────────────────────────────────────────


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns from a DataFrame of closing prices.

    Using log-returns (rather than simple returns) ensures time-additivity
    and a more symmetric distribution, which is important for drawdown
    calculations on long horizons.

    Args:
        prices: Daily adjusted closing prices.  Must contain at least
            two rows.

    Returns:
        A DataFrame of the same shape as ``prices`` (minus one row) containing
        ``ln(P_t / P_{t-1})``.  The first row of ``prices`` is consumed by the
        diff, so the index starts one day later.

    Example:
        >>> returns = compute_log_returns(prices)
        >>> returns.shape[0] == prices.shape[0] - 1
        True
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualised_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Compute the compound annualised return from daily log-returns.

    The formula is::

        CAGR = exp(mean_daily_log_return * TRADING_DAYS) - 1

    which correctly compounds the geometric mean of daily returns.

    Args:
        returns: A Series (single asset) or DataFrame (multiple assets) of
            daily log-returns.

    Returns:
        A scalar ``float`` when ``returns`` is a Series, otherwise a
        ``pd.Series`` with one value per column.
    """
    return np.exp(returns.mean() * TRADING_DAYS) - 1


def annualised_volatility(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Compute annualised volatility (standard deviation) of daily log-returns.

    Annualises by multiplying the daily standard deviation by ``sqrt(252)``
    under the assumption that returns are i.i.d. (standard industry convention).

    Args:
        returns: A Series (single asset) or DataFrame (multiple assets) of
            daily log-returns.

    Returns:
        A scalar ``float`` when ``returns`` is a Series, otherwise a
        ``pd.Series`` with one value per column.
    """
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> float | pd.Series:
    """Compute the annualised Sharpe ratio.

    Defined as::

        Sharpe = (Annualised Return − Risk-Free Rate) / Annualised Volatility

    Args:
        returns: Daily log-returns (Series or DataFrame).
        risk_free_rate: Annual risk-free rate as a decimal fraction.
            Defaults to ``0.04`` (4 %).

    Returns:
        A scalar ``float`` when ``returns`` is a Series, otherwise a
        ``pd.Series`` with one value per column.
    """
    ann_ret = annualised_return(returns)
    ann_vol = annualised_volatility(returns)
    return (ann_ret - risk_free_rate) / ann_vol


def maximum_drawdown(cumulative_returns: pd.Series) -> DrawdownResult:
    """Compute the maximum drawdown and its bounding dates.

    The algorithm is fully vectorised — no Python-level loops are used.

    Steps:

    1. Compute the running maximum of the equity curve (``rolling_max``).
    2. Compute the drawdown series: ``drawdown = equity / rolling_max - 1``.
    3. The trough is the index of the global minimum of the drawdown series.
    4. The peak is the last time the rolling max equals its value at the trough,
       i.e. the most-recent prior high before the trough.
    5. Recovery is the first date *after* the trough where the equity curve
       exceeds the rolling max at the peak.

    Args:
        cumulative_returns: A ``pd.Series`` of cumulative (compounded) returns
            or an equity curve, indexed by ``pd.DatetimeIndex``.  The series
            should represent a *level* (e.g. starting at 1.0), not differences.

    Returns:
        A :class:`DrawdownResult` dataclass with ``max_drawdown``,
        ``peak_date``, ``trough_date``, and ``recovery_date``.

    Example:
        >>> equity = (1 + daily_returns).cumprod()
        >>> dd = maximum_drawdown(equity)
        >>> dd.max_drawdown  # e.g. -0.348
        -0.348
    """
    rolling_max: pd.Series = cumulative_returns.cummax()
    drawdown_series: pd.Series = cumulative_returns / rolling_max - 1

    trough_idx: int = int(drawdown_series.argmin())
    trough_date: pd.Timestamp = drawdown_series.index[trough_idx]
    max_dd: float = float(drawdown_series.iloc[trough_idx])

    # Peak: last date on or before the trough where the rolling max == its
    # value at the trough (i.e. the high-water mark just before the decline).
    peak_mask: pd.Series = rolling_max.iloc[: trough_idx + 1] >= rolling_max.iloc[trough_idx]
    peak_date: pd.Timestamp = peak_mask[peak_mask].index[-1]

    # Recovery: first date after trough where equity exceeds peak level
    peak_level: float = float(rolling_max.iloc[trough_idx])
    post_trough: pd.Series = cumulative_returns.iloc[trough_idx + 1 :]
    recovery_candidates = post_trough[post_trough >= peak_level]
    recovery_date: Optional[pd.Timestamp] = (
        recovery_candidates.index[0] if not recovery_candidates.empty else None
    )

    return DrawdownResult(
        max_drawdown=max_dd,
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio-level aggregation
# ──────────────────────────────────────────────────────────────────────────────


def portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Compute the blended daily log-return series for a weighted portfolio.

    Uses a single matrix-vector dot product — no per-asset loop.

    Args:
        returns: Daily log-returns for each constituent asset (columns).
        weights: 1-D NumPy array of portfolio weights that *must* sum to 1.
            Length must equal ``returns.shape[1]``.

    Returns:
        A ``pd.Series`` of blended daily portfolio log-returns with the same
        ``DatetimeIndex`` as ``returns``.

    Raises:
        ValueError: If ``len(weights) != returns.shape[1]`` or weights do not
            sum to approximately 1.
    """
    if len(weights) != returns.shape[1]:
        raise ValueError(
            f"weights length ({len(weights)}) must equal number of assets "
            f"({returns.shape[1]})."
        )
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"Portfolio weights must sum to 1.  Got {weights.sum():.6f}."
        )
    blended: pd.Series = returns.dot(weights)
    blended.name = "Portfolio"
    return blended


def build_equity_curve(returns: pd.Series) -> pd.Series:
    """Convert a series of daily log-returns into a normalised equity curve.

    The equity curve starts at 1.0 and compounds forward so that each value
    represents the growth of $1 invested at the start of the series.

    Args:
        returns: Daily log-returns for a single series (asset or portfolio).

    Returns:
        A ``pd.Series`` of compounded equity values starting at 1.0.
    """
    return np.exp(returns.cumsum())


# ──────────────────────────────────────────────────────────────────────────────
# High-level orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(
    prices: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.04,
) -> list[AssetMetrics]:
    """Compute the full suite of risk/return metrics for each asset and the portfolio.

    Orchestrates all lower-level helpers into a single convenient call.
    Returns metrics for every individual asset *plus* one entry for the
    blended ``"Portfolio"`` at the end of the list.

    Args:
        prices: Daily adjusted closing prices (one column per ticker).
        weights: Portfolio weight vector that sums to 1.  Must match the
            column order of ``prices``.
        risk_free_rate: Annual risk-free rate (decimal).  Defaults to ``0.04``.

    Returns:
        A list of :class:`AssetMetrics` dataclasses — one per asset column
        in ``prices``, followed by one for the blended portfolio.

    Example:
        >>> metrics = compute_metrics(prices, weights, risk_free_rate=0.04)
        >>> metrics[-1].ticker
        'Portfolio'
    """
    daily_returns: pd.DataFrame = compute_log_returns(prices)
    results: list[AssetMetrics] = []

    # ── Per-asset metrics ─────────────────────────────────────────────────────
    for ticker in daily_returns.columns:
        ret_series: pd.Series = daily_returns[ticker]
        equity: pd.Series = build_equity_curve(ret_series)
        dd: DrawdownResult = maximum_drawdown(equity)

        results.append(
            AssetMetrics(
                ticker=ticker,
                annualised_return=float(annualised_return(ret_series)),
                annualised_volatility=float(annualised_volatility(ret_series)),
                sharpe_ratio=float(sharpe_ratio(ret_series, risk_free_rate)),
                drawdown=dd,
            )
        )

    # ── Portfolio metrics ─────────────────────────────────────────────────────
    port_ret: pd.Series = portfolio_returns(daily_returns, weights)
    port_equity: pd.Series = build_equity_curve(port_ret)
    port_dd: DrawdownResult = maximum_drawdown(port_equity)

    results.append(
        AssetMetrics(
            ticker="Portfolio",
            annualised_return=float(annualised_return(port_ret)),
            annualised_volatility=float(annualised_volatility(port_ret)),
            sharpe_ratio=float(sharpe_ratio(port_ret, risk_free_rate)),
            drawdown=port_dd,
        )
    )

    return results
