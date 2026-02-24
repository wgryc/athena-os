"""
Options-implied forward price curve with confidence intervals.

Fetches the full options chain from Databento OPRA, derives implied forward
prices and probability distributions at each expiration via put-call parity
and Breeden-Litzenberger, and renders a standalone HTML report.

Usage:
    python -m athena.metrics.options --ticker AAPL [--output report.html]
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm

load_dotenv()

DEFAULT_RISK_FREE_RATE = 0.045
MAX_EXPIRY_MONTHS = 24
CACHE_DIR = Path.cwd() / ".cache" / "options"
TEMPLATES_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# OSI symbol parsing
# ---------------------------------------------------------------------------

def parse_osi_symbol(symbol: str) -> dict | None:
    """Parse an OSI (OPRA) option symbol into its components.

    OSI format: ``{root padded to 6}{YYMMDD}{C|P}{strike x 1000, 8 digits}``

    Example: ``'AAPL  260320C00220000'`` -> AAPL, 2026-03-20, C, 220.00

    Args:
        symbol: Raw OSI symbol string (at least 21 characters).

    Returns:
        Dictionary with keys ``underlying``, ``expiry``, ``option_type``,
        and ``strike``, or ``None`` if parsing fails.
    """
    s = str(symbol).strip()
    if len(s) < 21:
        return None
    # Take the last 21 characters (some feeds prefix extra data)
    tail = s[-21:] if len(s) > 21 else s
    root = tail[0:6].strip()
    if not root:
        return None
    try:
        expiry = datetime.strptime(tail[6:12], "%y%m%d").date()
    except ValueError:
        return None
    opt_type = tail[12]
    if opt_type not in ("C", "P"):
        return None
    try:
        strike = int(tail[13:21]) / 1000.0
    except ValueError:
        return None
    if strike <= 0:
        return None
    return {
        "underlying": root,
        "expiry": expiry,
        "option_type": opt_type,
        "strike": strike,
    }


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class OptionsChain:
    """Container for one underlying at one expiration."""

    underlying: str
    expiration: date
    days_to_expiry: int
    years_to_expiry: float

    strikes: np.ndarray
    call_bids: np.ndarray
    call_asks: np.ndarray
    put_bids: np.ndarray
    put_asks: np.ndarray
    call_bid_sizes: np.ndarray
    call_ask_sizes: np.ndarray
    put_bid_sizes: np.ndarray
    put_ask_sizes: np.ndarray

    @property
    def call_mids(self) -> np.ndarray:
        return (self.call_bids + self.call_asks) / 2

    @property
    def put_mids(self) -> np.ndarray:
        return (self.put_bids + self.put_asks) / 2

    @property
    def n_strikes(self) -> int:
        return len(self.strikes)


# ---------------------------------------------------------------------------
# Data manager — Databento OPRA
# ---------------------------------------------------------------------------

class OptionsDataManager:
    """Fetches options chain data from Databento OPRA with disk caching."""

    DATASET = "OPRA.PILLAR"
    SCHEMA = "cbbo-1m"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Databento API key required. "
                "Set DATABENTO_API_KEY env var or pass api_key."
            )

    def _get_cache_path(self, underlying: str, query_date: date) -> Path:
        return CACHE_DIR / underlying.upper() / f"{query_date.isoformat()}.csv"

    def _find_query_date(self) -> date:
        """Find the most recent trading day to query.

        Walks back up to 15 trading days to skip weekends, holidays,
        and Databento's historical data processing delay. Prefers a
        date that already has cached data; otherwise returns the most
        recent non-weekend date.
        """
        now = datetime.now(timezone.utc)
        query_date = now.date()
        # If before US market close window (21:00 UTC), use previous day
        if now.hour < 21:
            query_date -= timedelta(days=1)

        # Walk back up to 15 trading days, prefer a date with cached data
        best = None
        for _ in range(15):
            while query_date.weekday() >= 5:
                query_date -= timedelta(days=1)
            if best is None:
                best = query_date
            # If we have cached data for this date, use it immediately
            if any(CACHE_DIR.glob(f"*/{query_date.isoformat()}.csv")):
                return query_date
            query_date -= timedelta(days=1)

        return best

    def fetch_chains(
        self,
        underlying: str,
        max_expiry_months: int = MAX_EXPIRY_MONTHS,
        use_cache: bool = True,
    ) -> list[OptionsChain]:
        """Fetch all available options chains for an underlying.

        Args:
            underlying: Ticker symbol (e.g. ``"AAPL"``).
            max_expiry_months: Maximum months forward to include.
            use_cache: Whether to read from/write to the disk cache.

        Returns:
            List of ``OptionsChain`` sorted by expiration, one per
            expiry. Only expirations within ``max_expiry_months`` with
            at least 5 liquid strikes are included.
        """
        ticker = underlying.upper()
        query_date = self._find_query_date()
        cache_path = self._get_cache_path(ticker, query_date)

        df: pd.DataFrame | None = None

        # Try disk cache
        if use_cache and cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                print(f"  Loaded {len(df)} rows from cache for {ticker}", flush=True)
            except Exception:
                df = None

        # Fetch from API
        if df is None:
            df = self._fetch_from_api(ticker, query_date)
            if df is not None and not df.empty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
                print(f"  Fetched {len(df)} rows from Databento for {ticker}", flush=True)
            else:
                raise ValueError(f"No options data returned for {ticker}")

        return self._build_chains(df, ticker, query_date, max_expiry_months)

    def _fetch_from_api(self, ticker: str, query_date: date) -> pd.DataFrame | None:
        """Fetch cbbo-1m snapshot from Databento OPRA."""
        client = db.Historical(self._api_key)

        # 1-minute window near market close (15:55-15:56 ET = 20:55-20:56 UTC)
        query_start = f"{query_date.isoformat()}T20:55"
        query_end = f"{query_date.isoformat()}T20:56"

        try:
            data = client.timeseries.get_range(
                dataset=self.DATASET,
                symbols=f"{ticker}.OPT",
                schema=self.SCHEMA,
                stype_in="parent",
                start=query_start,
                end=query_end,
            )
            df = data.to_df().reset_index()
            if "symbol" not in df.columns:
                return None
            return df
        except Exception as e:
            print(f"  Databento OPRA fetch failed: {e}", file=sys.stderr, flush=True)
            return None

    def _build_chains(
        self,
        df: pd.DataFrame,
        ticker: str,
        query_date: date,
        max_expiry_months: int,
    ) -> list[OptionsChain]:
        """Parse raw DataFrame into list of OptionsChain."""
        # Parse OSI symbols
        parsed = df["symbol"].apply(parse_osi_symbol)
        valid_mask = parsed.notna()
        df = df[valid_mask].copy()
        parsed = parsed[valid_mask]

        df["underlying"] = parsed.apply(lambda p: p["underlying"])
        df["expiry"] = parsed.apply(lambda p: p["expiry"])
        df["option_type"] = parsed.apply(lambda p: p["option_type"])
        df["strike"] = parsed.apply(lambda p: p["strike"])

        # Filter to matching underlying
        df = df[df["underlying"] == ticker]
        if df.empty:
            return []

        # Date filters
        today = query_date
        max_date = today + timedelta(days=max_expiry_months * 30)
        min_date = today + timedelta(days=3)  # skip < 3 DTE
        df = df[(df["expiry"] > min_date) & (df["expiry"] <= max_date)]

        if df.empty:
            return []

        # For each symbol, take the last row (most recent snapshot in the minute)
        df = df.sort_values("ts_event" if "ts_event" in df.columns else df.columns[0])
        df = df.drop_duplicates(
            subset=["expiry", "option_type", "strike"], keep="last"
        )

        chains = []
        for expiry, group in df.groupby("expiry"):
            chain = self._build_single_chain(group, ticker, expiry, today)
            if chain is not None:
                chains.append(chain)

        chains.sort(key=lambda c: c.expiration)
        return chains

    def _build_single_chain(
        self,
        group: pd.DataFrame,
        ticker: str,
        expiry: date,
        today: date,
    ) -> OptionsChain | None:
        """Build an OptionsChain from a group of rows for one expiration."""
        calls = group[group["option_type"] == "C"].set_index("strike")
        puts = group[group["option_type"] == "P"].set_index("strike")

        # Intersect strikes that have both call and put
        common_strikes = sorted(set(calls.index) & set(puts.index))
        if len(common_strikes) < 5:
            return None

        strikes = np.array(common_strikes, dtype=float)
        call_bids = np.array([float(calls.loc[k, "bid_px_00"]) for k in common_strikes])
        call_asks = np.array([float(calls.loc[k, "ask_px_00"]) for k in common_strikes])
        put_bids = np.array([float(puts.loc[k, "bid_px_00"]) for k in common_strikes])
        put_asks = np.array([float(puts.loc[k, "ask_px_00"]) for k in common_strikes])

        call_bid_sz = np.array([
            float(calls.loc[k, "bid_sz_00"]) if "bid_sz_00" in calls.columns else 0
            for k in common_strikes
        ])
        call_ask_sz = np.array([
            float(calls.loc[k, "ask_sz_00"]) if "ask_sz_00" in calls.columns else 0
            for k in common_strikes
        ])
        put_bid_sz = np.array([
            float(puts.loc[k, "bid_sz_00"]) if "bid_sz_00" in puts.columns else 0
            for k in common_strikes
        ])
        put_ask_sz = np.array([
            float(puts.loc[k, "ask_sz_00"]) if "ask_sz_00" in puts.columns else 0
            for k in common_strikes
        ])

        # Liquidity filter: remove strikes where bid = 0 on either side
        liquid = (call_bids > 0) & (put_bids > 0)

        # Remove strikes where spread > 50% of mid
        call_mids = (call_bids + call_asks) / 2
        put_mids = (put_bids + put_asks) / 2
        call_spread_pct = np.where(
            call_mids > 0, (call_asks - call_bids) / call_mids, 999
        )
        put_spread_pct = np.where(
            put_mids > 0, (put_asks - put_bids) / put_mids, 999
        )
        liquid &= (call_spread_pct < 0.50) | (put_spread_pct < 0.50)

        if liquid.sum() < 5:
            return None

        days = (expiry - today).days
        return OptionsChain(
            underlying=ticker,
            expiration=expiry,
            days_to_expiry=days,
            years_to_expiry=days / 365.25,
            strikes=strikes[liquid],
            call_bids=call_bids[liquid],
            call_asks=call_asks[liquid],
            put_bids=put_bids[liquid],
            put_asks=put_asks[liquid],
            call_bid_sizes=call_bid_sz[liquid],
            call_ask_sizes=call_ask_sz[liquid],
            put_bid_sizes=put_bid_sz[liquid],
            put_ask_sizes=put_ask_sz[liquid],
        )


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute Black-Scholes European call price.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Annualized risk-free rate.
        sigma: Annualized volatility.

    Returns:
        Theoretical call option price.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute Black-Scholes European put price via put-call parity.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Annualized risk-free rate.
        sigma: Annualized volatility.

    Returns:
        Theoretical put option price.
    """
    return bs_call_price(S, K, T, r, sigma) - S + K * np.exp(-r * T)


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "C",
) -> float | None:
    """Solve for Black-Scholes implied volatility using Brent's method.

    Args:
        price: Observed market price of the option.
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Annualized risk-free rate.
        option_type: ``"C"`` for call, ``"P"`` for put.

    Returns:
        Implied volatility as a float, or ``None`` if the solver fails
        or the price is at or below intrinsic value.
    """
    if T <= 0 or price <= 0:
        return None

    intrinsic = max(S - K * np.exp(-r * T), 0.0) if option_type == "C" else max(K * np.exp(-r * T) - S, 0.0)
    if price <= intrinsic + 1e-8:
        return None

    def objective(sigma):
        if option_type == "C":
            return bs_call_price(S, K, T, r, sigma) - price
        return bs_put_price(S, K, T, r, sigma) - price

    try:
        return brentq(objective, 0.001, 10.0, xtol=1e-8)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def compute_implied_forward(
    chain: OptionsChain,
    risk_free_rate: float,
) -> tuple[float, float]:
    """Compute the implied forward price via put-call parity regression.

    Fits ``call_mid - put_mid = a + b * K`` by OLS and derives the
    self-consistent forward as ``F = -a / b``.

    Args:
        chain: Options chain for a single expiration.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Tuple of ``(forward_price, r_squared)`` from the regression.
    """
    y = chain.call_mids - chain.put_mids
    K = chain.strikes

    # OLS: y = a + b*K  =>  b, a = polyfit(K, y, 1)
    coeffs = np.polyfit(K, y, 1)
    b, a = coeffs[0], coeffs[1]

    # F = -a/b (self-consistent, doesn't depend on rfr accuracy)
    if abs(b) < 1e-12:
        # Fallback: use discount factor
        df = np.exp(-risk_free_rate * chain.years_to_expiry)
        forward = a / df
    else:
        forward = -a / b

    # R-squared
    y_hat = np.polyval(coeffs, K)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(forward), float(r2)


def compute_implied_distribution(
    chain: OptionsChain,
    forward: float,
    risk_free_rate: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute the risk-neutral PDF via Breeden-Litzenberger.

    Uses OTM options (puts below forward converted to call-equivalent,
    calls above forward), fits a smoothing spline to the call price
    curve, and takes the second derivative to obtain the density.

    Args:
        chain: Options chain for a single expiration.
        forward: Implied forward price of the underlying.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Tuple of ``(price_grid, density)`` arrays, or ``None`` if
        fitting fails or the density integral is too small.
    """
    T = chain.years_to_expiry
    r = risk_free_rate
    df = np.exp(-r * T)

    # Build OTM call price curve
    # For K < F: C(K) = P(K) + df*(F - K)  (put-call parity)
    # For K >= F: C(K) = call mid directly
    c_otm = np.empty_like(chain.strikes)
    spreads = np.empty_like(chain.strikes)

    for i, K in enumerate(chain.strikes):
        if K < forward:
            c_otm[i] = chain.put_mids[i] + df * (forward - K)
            spreads[i] = chain.put_asks[i] - chain.put_bids[i]
        else:
            c_otm[i] = chain.call_mids[i]
            spreads[i] = chain.call_asks[i] - chain.call_bids[i]

    # Weights: inverse spread (tighter = more reliable)
    weights = 1.0 / np.maximum(spreads, 0.01)

    # Call prices must be non-negative and monotonically decreasing
    c_otm = np.maximum(c_otm, 0.0)

    if len(chain.strikes) < 5:
        return None

    # Adaptive smoothing spline — use k=4 when enough data, k=3 for thin chains
    n = len(chain.strikes)
    spline_order = 4 if n >= 8 else 3
    for s_mult in [0.01, 0.02, 0.05, 0.1, 0.2]:
        smoothing = n * s_mult
        try:
            spline = UnivariateSpline(
                chain.strikes, c_otm, w=weights, k=spline_order, s=smoothing
            )
        except Exception:
            continue

        # Evaluate second derivative on a fine grid
        K_min = chain.strikes[0]
        K_max = chain.strikes[-1]
        price_grid = np.linspace(K_min, K_max, 500)
        d2C = spline.derivative(n=2)(price_grid)

        density = np.exp(r * T) * d2C

        # Check quality: fraction of negative density
        neg_frac = np.sum(density < 0) / len(density)
        if neg_frac < 0.05:
            break
    else:
        # Use the last attempt regardless
        pass

    # Clip negatives and normalize
    density = np.maximum(density, 0.0)
    total = np.trapezoid(density, price_grid)
    if total < 0.01:
        return None
    density = density / total

    return price_grid, density


def compute_implied_volatilities(
    chain: OptionsChain,
    spot: float,
    risk_free_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Black-Scholes implied volatility for each strike.

    Uses OTM options (calls for strikes >= spot, puts below) to avoid
    deep-ITM numerical issues.

    Args:
        chain: Options chain for a single expiration.
        spot: Current underlying spot price.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Tuple of ``(strikes, ivs)`` arrays filtered to strikes where
        IV was successfully computed and falls in (0.01, 5.0).
    """
    T = chain.years_to_expiry
    r = risk_free_rate

    strike_list = []
    iv_list = []

    for i, K in enumerate(chain.strikes):
        if K >= spot:
            price = chain.call_mids[i]
            opt_type = "C"
        else:
            price = chain.put_mids[i]
            opt_type = "P"

        iv = implied_volatility(price, spot, K, T, r, opt_type)
        if iv is not None and 0.01 < iv < 5.0:
            strike_list.append(K)
            iv_list.append(iv)

    return np.array(strike_list), np.array(iv_list)


def get_atm_iv(strikes: np.ndarray, ivs: np.ndarray, forward: float) -> float | None:
    """Interpolate to get ATM implied volatility at the forward price.

    Args:
        strikes: Array of strike prices with known IVs.
        ivs: Array of implied volatilities corresponding to strikes.
        forward: Implied forward price used as the ATM reference.

    Returns:
        Interpolated ATM IV, or ``None`` if fewer than 2 data points.
    """
    if len(strikes) < 2:
        return None
    return float(np.interp(forward, strikes, ivs))


def compute_expected_move(
    chain: OptionsChain,
    forward: float,
) -> tuple[float, float]:
    """Compute the expected move from the ATM straddle price.

    Selects the strike closest to the forward price and sums the call
    and put mid-prices to obtain the straddle cost.

    Args:
        chain: Options chain for a single expiration.
        forward: Implied forward price of the underlying.

    Returns:
        Tuple of ``(dollar_move, percent_move)`` where percent_move is
        relative to the forward price.
    """
    atm_idx = int(np.argmin(np.abs(chain.strikes - forward)))
    straddle = chain.call_mids[atm_idx] + chain.put_mids[atm_idx]
    return float(straddle), float(straddle / forward) if forward > 0 else 0.0


def compute_distribution_stats(
    price_grid: np.ndarray,
    density: np.ndarray,
) -> dict:
    """Compute summary statistics from the implied price distribution.

    Derives mean, standard deviation, skewness, excess kurtosis,
    and key percentiles (10, 25, 50, 75, 90) via cumulative trapezoidal
    integration of the density.

    Args:
        price_grid: Array of underlying prices at which the density
            is evaluated.
        density: Normalised probability density values over the grid.

    Returns:
        Dictionary with keys ``mean``, ``std``, ``skew``, ``kurtosis``,
        ``p10``, ``p25``, ``p50``, ``p75``, and ``p90``.
    """
    # CDF via cumulative trapezoidal integration
    dx = np.diff(price_grid)
    avg_density = (density[:-1] + density[1:]) / 2
    cdf = np.zeros(len(price_grid))
    cdf[1:] = np.cumsum(avg_density * dx)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    def percentile(p: float) -> float:
        idx = np.searchsorted(cdf, p / 100.0)
        idx = min(idx, len(price_grid) - 1)
        return float(price_grid[idx])

    mean = float(np.trapezoid(price_grid * density, price_grid))
    var = float(np.trapezoid((price_grid - mean) ** 2 * density, price_grid))
    std = float(np.sqrt(max(var, 0)))

    skew = 0.0
    kurt = 0.0
    if std > 0:
        skew = float(np.trapezoid(((price_grid - mean) / std) ** 3 * density, price_grid))
        kurt = float(np.trapezoid(((price_grid - mean) / std) ** 4 * density, price_grid) - 3.0)

    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurt,
        "p10": percentile(10),
        "p25": percentile(25),
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
    }


# ---------------------------------------------------------------------------
# Forward curve builder
# ---------------------------------------------------------------------------

@dataclass
class ExpirationResult:
    """Computed results for a single expiration."""

    expiration: date
    days_to_expiry: int
    years_to_expiry: float
    forward: float
    forward_pcp_r2: float
    atm_iv: float | None
    expected_move_dollar: float
    expected_move_pct: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    mean: float
    std: float
    skew: float
    kurtosis: float
    n_strikes: int


def build_forward_curve(
    underlying: str,
    spot: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_expiry_months: int = MAX_EXPIRY_MONTHS,
    api_key: str | None = None,
    use_cache: bool = True,
) -> list[ExpirationResult]:
    """Build the full implied forward curve with confidence intervals.

    For each available expiration (up to ``max_expiry_months``), computes
    the implied forward price, risk-neutral probability distribution,
    and summary statistics (percentiles, IV, expected move).

    Args:
        underlying: Ticker symbol (e.g. ``"AAPL"``).
        spot: Current spot price of the underlying.
        risk_free_rate: Annualized risk-free rate.
        max_expiry_months: Maximum months forward to include.
        api_key: Databento API key. Falls back to env var if ``None``.
        use_cache: Whether to use the disk cache for options data.

    Returns:
        List of ``ExpirationResult`` objects sorted by expiration date.
        Expirations that fail quality checks are excluded.
    """
    manager = OptionsDataManager(api_key)
    chains = manager.fetch_chains(underlying, max_expiry_months, use_cache)

    if not chains:
        return []

    results = []
    for chain in chains:
        try:
            forward, r2 = compute_implied_forward(chain, risk_free_rate)

            if r2 < 0.80:
                print(
                    f"  Warning: skipping {chain.expiration} — "
                    f"put-call parity R²={r2:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if forward <= 0 or forward > spot * 3:
                print(
                    f"  Warning: skipping {chain.expiration} — "
                    f"unreasonable forward={forward:.2f}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            # Distribution
            dist_result = compute_implied_distribution(
                chain, forward, risk_free_rate
            )
            if dist_result is None:
                print(
                    f"  Warning: skipping {chain.expiration} — "
                    f"distribution fitting failed",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            price_grid, density = dist_result

            stats = compute_distribution_stats(price_grid, density)

            # IV
            iv_strikes, ivs = compute_implied_volatilities(
                chain, spot, risk_free_rate
            )
            atm_iv = get_atm_iv(iv_strikes, ivs, forward)

            # Expected move
            dollar_move, pct_move = compute_expected_move(chain, forward)

            results.append(ExpirationResult(
                expiration=chain.expiration,
                days_to_expiry=chain.days_to_expiry,
                years_to_expiry=chain.years_to_expiry,
                forward=forward,
                forward_pcp_r2=r2,
                atm_iv=atm_iv,
                expected_move_dollar=dollar_move,
                expected_move_pct=pct_move,
                n_strikes=chain.n_strikes,
                **stats,
            ))

        except Exception as e:
            print(
                f"  Warning: skipping {chain.expiration} — {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_curve_data(
    underlying: str,
    spot: float,
    results: list[ExpirationResult],
) -> dict:
    """Format forward curve results for the HTML template.

    Builds the JSON-serialisable structure consumed by the Jinja2
    report template, including curve data, IV term structure, and a
    per-expiration summary table.

    Args:
        underlying: Ticker symbol (e.g. ``"AAPL"``).
        spot: Current spot price.
        results: List of ``ExpirationResult`` objects from
            ``build_forward_curve``.

    Returns:
        Dictionary ready for JSON serialisation with keys ``underlying``,
        ``spot``, ``generated_at``, ``curve``, ``term_structure``, and
        ``summary``.
    """
    curve = {
        "labels": [],
        "days": [],
        "forwards": [],
        "p10": [],
        "p25": [],
        "p50": [],
        "p75": [],
        "p90": [],
    }
    term_structure = {"labels": [], "atm_ivs": []}
    summary = []

    # Anchor at today: all series converge to spot at T=0
    spot_rounded = round(spot, 2)
    today_label = date.today().strftime("%Y-%m-%d")
    curve["labels"].append(today_label)
    curve["days"].append(0)
    curve["forwards"].append(spot_rounded)
    curve["p10"].append(spot_rounded)
    curve["p25"].append(spot_rounded)
    curve["p50"].append(spot_rounded)
    curve["p75"].append(spot_rounded)
    curve["p90"].append(spot_rounded)
    summary.append({
        "expiration": today_label,
        "days": 0,
        "forward": spot_rounded,
        "atm_iv": None,
        "expected_move": 0.0,
        "expected_move_pct": 0.0,
        "p10": spot_rounded,
        "p25": spot_rounded,
        "p50": spot_rounded,
        "p75": spot_rounded,
        "p90": spot_rounded,
        "n_strikes": 0,
        "r2": 1.0,
    })

    for r in results:
        label = r.expiration.strftime("%Y-%m-%d")
        curve["labels"].append(label)
        curve["days"].append(r.days_to_expiry)
        curve["forwards"].append(round(r.forward, 2))
        curve["p10"].append(round(r.p10, 2))
        curve["p25"].append(round(r.p25, 2))
        curve["p50"].append(round(r.p50, 2))
        curve["p75"].append(round(r.p75, 2))
        curve["p90"].append(round(r.p90, 2))

        if r.atm_iv is not None:
            term_structure["labels"].append(label)
            term_structure["atm_ivs"].append(round(r.atm_iv * 100, 1))

        summary.append({
            "expiration": label,
            "days": r.days_to_expiry,
            "forward": round(r.forward, 2),
            "atm_iv": round(r.atm_iv, 4) if r.atm_iv else None,
            "expected_move": round(r.expected_move_dollar, 2),
            "expected_move_pct": round(r.expected_move_pct, 4),
            "p10": round(r.p10, 2),
            "p25": round(r.p25, 2),
            "p50": round(r.p50, 2),
            "p75": round(r.p75, 2),
            "p90": round(r.p90, 2),
            "n_strikes": r.n_strikes,
            "r2": round(r.forward_pcp_r2, 3),
        })

    return {
        "underlying": underlying,
        "spot": round(spot, 2),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "curve": curve,
        "term_structure": term_structure,
        "summary": summary,
    }


def generate_report(
    underlying: str,
    spot: float,
    results: list[ExpirationResult],
    output_path: str | Path,
) -> Path:
    """Render the forward curve report as a standalone HTML file.

    Args:
        underlying: Ticker symbol (e.g. ``"AAPL"``).
        spot: Current spot price.
        results: List of ``ExpirationResult`` objects.
        output_path: File path for the generated HTML report.

    Returns:
        Resolved ``Path`` to the written report file.
    """
    data = format_curve_data(underlying, spot, results)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("options_template.html")
    html = template.render(
        data=json.dumps(data),
        underlying=underlying,
        spot=spot,
    )

    output = Path(output_path)
    output.write_text(html)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an options-implied forward price curve report."
    )
    parser.add_argument("--ticker", "-t", required=True, help="US stock ticker")
    parser.add_argument(
        "--risk-free-rate", "-r", type=float, default=DEFAULT_RISK_FREE_RATE,
        help=f"Annualized risk-free rate (default: {DEFAULT_RISK_FREE_RATE})",
    )
    parser.add_argument("--output", "-o", default=None, help="Output HTML file path")
    parser.add_argument(
        "--max-months", "-m", type=int, default=MAX_EXPIRY_MONTHS,
        help=f"Maximum months forward (default: {MAX_EXPIRY_MONTHS})",
    )
    parser.add_argument("--no-cache", action="store_true", help="Skip disk cache")
    parser.add_argument("--spot", type=float, default=None, help="Override spot price")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    output = args.output or f"{ticker}_forward_curve.html"

    # Get spot price
    if args.spot is not None:
        spot = args.spot
    else:
        try:
            from ..pricingdata import YFinancePricingDataManager
            mgr = YFinancePricingDataManager()
            pp = mgr.get_price_point(ticker, datetime.now())
            spot = float(pp.price)
        except Exception as e:
            print(f"ERROR: Could not fetch spot price for {ticker}: {e}", file=sys.stderr)
            print("Use --spot to provide it manually.", file=sys.stderr)
            sys.exit(1)

    print(f"Building forward curve for {ticker} (spot: ${spot:.2f}) ...")

    results = build_forward_curve(
        underlying=ticker,
        spot=spot,
        risk_free_rate=args.risk_free_rate,
        max_expiry_months=args.max_months,
        use_cache=not args.no_cache,
    )

    if not results:
        print("ERROR: No valid expirations found.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Computed forward curve for {len(results)} expirations "
        f"({results[0].expiration} to {results[-1].expiration})"
    )

    path = generate_report(ticker, spot, results, output)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
