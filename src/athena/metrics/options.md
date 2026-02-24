# Options-Implied Price Distribution: Strategy & Plan

## Goal

Given a stock ticker (e.g. AAPL, TSLA, NVDA), fetch the full options chain from
Databento's OPRA feed, and derive the market's implied expectations for the
stock's future price. Visualize the results as dashboard widgets.

---

## 1. Data Acquisition from Databento

### Dataset & Schemas

The existing Databento integration uses `GLBX.MDP3` (CME futures). For equity
options we need the **`OPRA.PILLAR`** dataset, which is the consolidated feed
across all 17 US equity options exchanges. Our current Databento plan includes
OPRA access (verified).

All available OPRA schemas with verified cost for a single AAPL query:

| Schema | $/GB | Query Cost | Data Size | What You Get |
|--------|------|-----------|-----------|--------------|
| **`cbbo-1m`** | **$2.00** | **$0.005** | **2.5 MB** | **1-min sampled NBBO bid/ask — OUR PRIMARY** |
| `ohlcv-1m` | $280 | $0.005 | 0.02 MB | 1-min OHLCV bars (trades only, gaps on illiquid strikes) |
| `cmbp-1` | $0.16 | $0.116 | 778 MB | Every NBBO tick + trades (massive overkill) |
| `cbbo-1s` | $2.00 | $0.149 | 80 MB | 1-sec sampled NBBO (unnecessary granularity) |
| `ohlcv-1d` | $600 | $0.331 | 0.6 MB | Daily OHLCV bars |
| `ohlcv-1h` | $600 | $0.867 | 1.5 MB | Hourly OHLCV bars |
| `definition` | $5.00 | $0.006 | 1.2 MB | Instrument definitions (strike, expiry, type) |
| `statistics` | $11.00 | — | — | Open interest, settlement prices |

**Primary schema: `cbbo-1m`** — A single 1-minute query returns ~3,000 contracts
across all expirations with NBBO bid/ask at each strike, for $0.005. This is
30x cheaper than `cbbo-1s` and provides identical fields:
`bid_px_00`, `ask_px_00`, `bid_sz_00`, `ask_sz_00`, `symbol`.

**Why not OHLCV?** Only captures trades — illiquid far-OTM options may not
trade in any given minute, leaving gaps. BBO captures standing quotes even
without trades, giving full coverage across all strikes.

**Why not `cmbp-1`?** Lowest $/GB but generates 778 MB of tick data — massive
overkill for a single snapshot.

**Estimated cost for 12-month forward curve:**

| Scenario | Cost |
|----------|------|
| 1 ticker, 1 refresh | ~$0.005 |
| 10 tickers, 1 refresh | ~$0.05-0.10 |
| 10 tickers, daily for 1 month | ~$1-2 |

### Symbol Discovery

OPRA option symbology follows OSI format, embedded in the `symbol` column
returned by `cbbo-1m`. No separate `definition` query is needed — the symbol
itself encodes everything:

```
AAPL  260320C00220000
│     │     ││       │
│     │     ││       └─ Strike × 1000 ($220.00)
│     │     │└─────── Call (C) or Put (P)
│     │     └──────── Expiration YYMMDD (2026-03-20)
│     └────────────── padding (spaces to 6 chars)
└──────────────────── Underlying root
```

**Verified from live data:** A single `cbbo-1m` query with `stype_in="parent"`
and `symbols="AAPL.OPT"` returns all ~3,000 contracts across all expirations
in one call. We parse the OSI symbol to extract strike, expiry, and type.

**Workflow (simplified):**
1. Fetch `cbbo-1m` for `{TICKER}.OPT` with `stype_in="parent"` (1 query)
2. Parse OSI symbols → strike, expiration, call/put
3. Filter to desired expiration(s)
4. Bid/ask already included — no second query needed

### Integration with Existing Code

The current `DatabentoPricingDataManager` in `pricingdata.py` is built around
single-symbol OHLCV/TBBO queries for futures. Options data is fundamentally
different (hundreds of symbols per chain, different dataset). Two options:

**Option A — Extend `DatabentoPricingDataManager`:**
Add methods like `get_options_chain(underlying, expiration)` that use `OPRA`
dataset internally. Pro: single class, shared API key handling and caching
infra. Con: class becomes overloaded with two very different responsibilities.

**Option B — New class `OptionsDataManager` (recommended):**
Separate class in `options.py` (or a new `options_data.py`) dedicated to
options chain retrieval. Reuses the same `DATABENTO_API_KEY` env var. Has its
own caching strategy. Pro: clean separation, testable independently. Con:
some duplication of API key / client setup (minimal).

---

## 2. Mathematical Framework

### 2a. Implied Forward Price (Put-Call Parity)

For European-style options at a given expiration T:

```
C(K) - P(K) = e^(-rT) * (F - K)
```

Where F is the implied forward price, r is the risk-free rate, C(K) and P(K)
are call and put mid-prices at strike K.

**Method:** Regress `C(K) - P(K)` against K across all strikes. The intercept
gives `e^(-rT) * F`, from which we extract F. This is robust because it uses
all strikes, not just one.

**Alternatively:** Find the strike where C(K) ~ P(K) (the ATM forward strike).
At that strike, F ~ K. Quick and dirty but less precise.

### 2b. Implied Probability Distribution (Breeden-Litzenberger)

The core result: the risk-neutral probability density function (PDF) of the
underlying price at expiration is proportional to the second derivative of call
prices with respect to strike:

```
f(K) = e^(rT) * d^2C/dK^2
```

**Practical implementation:**

1. Sort strikes K_1 < K_2 < ... < K_n
2. Compute mid-prices M_i = (bid_i + ask_i) / 2 for calls at each strike
3. Apply finite-difference second derivative:
   ```
   f(K_i) ~ e^(rT) * (M_{i+1} - 2*M_i + M_{i-1}) / (dK)^2
   ```
   where dK = K_{i+1} - K_i (assuming uniform spacing; adjust for non-uniform)
4. Normalize so the PDF integrates to 1

**Smoothing:** Raw option prices are noisy. Before differentiating:
- Fit a smooth curve (cubic spline or polynomial) to mid-prices vs strike
- Or use a kernel density approach on the finite-difference estimates
- `scipy.interpolate.UnivariateSpline` with smoothing parameter is a good
  pragmatic choice

**Edge handling:** The distribution tails beyond the range of traded strikes
are not directly observable. We can:
- Assume log-normal tails beyond the last liquid strikes
- Or simply truncate and note the limitation

### 2c. Implied Volatility Surface

For each option contract, back out the Black-Scholes implied volatility:

```
BS_price(S, K, T, r, sigma) = Market_price  =>  solve for sigma
```

Use `scipy.optimize.brentq` or Newton-Raphson on the BS formula. This gives
IV as a function of (K, T) — the volatility surface.

**Useful derived views:**
- **Smile/skew** at a single expiration: IV vs strike (or moneyness K/F)
- **Term structure**: ATM IV vs expiration
- **Surface**: 3D or heatmap of IV across strike and expiry

### 2d. Expected Move (ATM Straddle)

The at-the-money straddle price approximates the market's expected move:

```
Expected move ~ Straddle_price = C_ATM + P_ATM
```

This is the simplest measure and can be expressed as a percentage of the
current stock price. It corresponds roughly to a 1-standard-deviation move
under the implied distribution.

---

## 3. Implementation Plan

### Phase 1: Data Layer (`options.py` or `options_data.py`)

```
class OptionsChain:
    """Container for a single underlying + expiration."""
    underlying: str
    expiration: date
    strikes: list[float]
    call_bids: list[float]
    call_asks: list[float]
    put_bids: list[float]
    put_asks: list[float]
    open_interest_calls: list[int]  # optional
    open_interest_puts: list[int]   # optional

class OptionsDataManager:
    def __init__(self, api_key=None):
        ...

    def get_expirations(self, underlying: str) -> list[date]:
        """Available expiration dates for this underlying."""

    def get_chain(self, underlying: str, expiration: date) -> OptionsChain:
        """Full options chain (all strikes) for one expiration."""

    def get_chains(self, underlying: str, n_expirations: int = 4) -> list[OptionsChain]:
        """Multiple nearest expirations for surface construction."""
```

**Caching strategy:**
- Disk cache at `.cache/options/{UNDERLYING}/{EXPIRATION}.json`
- Intraday TTL (~15 min) since option prices change rapidly
- No separate definition cache needed — OSI symbols parsed from quote data

### Phase 2: Analytics (`options.py` metrics)

```python
def compute_implied_forward(chain: OptionsChain, risk_free_rate: float) -> float:
    """Put-call parity regression for implied forward price."""

def compute_implied_distribution(chain: OptionsChain, risk_free_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Breeden-Litzenberger. Returns (price_points, probability_density)."""

def compute_implied_volatilities(chain: OptionsChain, spot: float, risk_free_rate: float) -> dict:
    """BS IV for each strike. Returns {strike: iv} for calls and puts."""

def compute_expected_move(chain: OptionsChain) -> tuple[float, float]:
    """ATM straddle. Returns (dollar_move, percent_move)."""

def compute_distribution_stats(prices: np.ndarray, density: np.ndarray) -> dict:
    """Mean, median, mode, std, skew, kurtosis of implied distribution."""
```

### Phase 3: Dashboard Widget(s)

Two candidate widgets, following the existing `DashboardWidget` pattern:

**Widget 1: `implied_distribution_chart_widget`**
- Input: `{"ticker": "AAPL", "expiration": "nearest"}` (or specific date)
- Chart: Area/bar chart of the implied PDF
- Vertical line at current price, vertical line at implied forward
- Summary stats in `to_context()` for LLM consumption
- Chart type: `"bar"` or `"line"` with fill

**Widget 2: `implied_volatility_chart_widget`**
- Input: `{"ticker": "AAPL"}`
- Chart: IV smile/skew — line chart of IV vs strike
- Multiple series if showing multiple expirations
- Chart type: `"line"`

Both extend `DashboardWidget`, set `self.chart_config` with Chart.js config,
and return HTML with a `<canvas class="widget-chart">` element.

### Phase 4: Configuration & Integration

Add widgets to `config.json`:
```json
[
    {"tool": "implied_distribution_chart_widget", "kwargs": {"ticker": "AAPL"}},
    {"tool": "implied_volatility_chart_widget", "kwargs": {"ticker": "AAPL"}}
]
```

Register widgets in `app.py` visual tools list.

---

## 4. Dependencies

**Already available in the project:**
- `databento` — API client
- `numpy`, `scipy`, `pandas` — numerical computation

**May need to add:**
- `py_vollib` or similar — for BS implied vol solving (or we roll our own,
  it's a small function)
- No new major dependencies expected; `scipy.optimize.brentq` + `scipy.stats.norm`
  cover the BS formula and root-finding

---

## 5. Key Risks & Considerations

### Risk-Neutral vs Real-World
The implied distribution is *risk-neutral* — it embeds risk premia. The
volatility risk premium means implied vol typically overstates realized vol by
~2-4 vol points. The distribution will be more left-skewed (fatter left tail)
than the "true" market expectation. This is a feature, not a bug: it's what
the market is pricing.

We should note this clearly in widget tooltips / context output.

### Data Volume & Cost
OPRA has 1M+ active instruments. A single `cbbo-1m` query for one underlying
returns ~3,000 contracts (all expirations) for ~$0.005. Even 10 tickers
refreshed daily for a month costs ~$1-2. Cost is negligible.

### Liquidity Filtering
Not all strikes are liquid. We should filter by:
- Open interest > some threshold (e.g., 100 contracts)
- Bid-ask spread < some percentage of mid (e.g., 20%)
- Or simply use only strikes within some range of ATM (e.g., +/- 30%)

This prevents garbage-in-garbage-out in the distribution calculation.

### American vs European Options
US equity options are American-style (can be exercised early). The
Breeden-Litzenberger result is exact for Europeans. For Americans, early
exercise premium introduces bias, especially for deep ITM puts and
dividend-paying stocks. Mitigations:
- Use OTM options only (calls above forward, puts below forward) — these
  have negligible early exercise premium
- Or accept the small bias and note it

### Non-Uniform Strike Spacing
Real option chains have non-uniform strike spacing (e.g., $1 near ATM, $5
further out). The finite-difference formula needs to account for this:
```
f(K_i) ~ e^(rT) * 2 * [(M_{i+1} - M_i)/(K_{i+1} - K_i) - (M_i - M_{i-1})/(K_i - K_{i-1})] / (K_{i+1} - K_{i-1})
```
Or interpolate to uniform grid first, then differentiate.

### Risk-Free Rate
Need a risk-free rate for discounting. Options:
- Use the existing FRED integration in `sharpe_advanced.py` to get treasury rates
- Or hardcode/approximate (current ~4-5% range) — precision matters less
  for short-dated options

---

## 6. Suggested Phasing

| Phase | Deliverable | Complexity |
|-------|-------------|------------|
| 1 | `OptionsDataManager` — fetch chain from Databento OPRA | Medium |
| 2 | Implied forward + expected move calculations | Low |
| 3 | Breeden-Litzenberger implied distribution | Medium |
| 4 | `implied_distribution_chart_widget` | Low (follows existing pattern) |
| 5 | Implied vol surface + smile chart widget | Medium |
| 6 | Caching, liquidity filtering, polish | Low-Medium |

Phases 1-4 form the MVP: "show me the market's implied price distribution for
AAPL at the nearest expiration." Phases 5-6 add depth and robustness.

---

## 7. Example Output (What the User Sees)

### Implied Distribution Chart
- X-axis: stock price at expiration ($150 ... $200 ... $250)
- Y-axis: probability density
- Shaded area chart showing the bell-curve-like (but skewed) distribution
- Vertical dashed line: current stock price
- Vertical solid line: implied forward price
- Annotation: "Expected move: +/- $12.50 (6.8%)"
- Annotation: "Expiration: March 21, 2025 (34 days)"

### IV Smile Chart
- X-axis: strike price (or moneyness K/S)
- Y-axis: implied volatility (%)
- Line for calls, line for puts (or single OTM line)
- Multiple expirations as separate colored lines

### Context for LLM (to_context output)
```
AAPL Options-Implied Outlook (exp: 2025-03-21, 34 days):
  Current price: $183.50
  Implied forward: $184.20 (+0.4%)
  Expected move: +/- $12.50 (6.8%)
  Distribution skew: -0.32 (slight left skew)
  Implied vol (ATM): 28.5%
  Key percentiles:
    10th: $168.20
    25th: $175.40
    75th: $192.80
    90th: $201.30
```

---

## 8. CLI Usage

The module can be run directly from the command line to generate a standalone
HTML report for any US stock ticker.

### Basic Usage

```bash
python -m athena.metrics.options --ticker AAPL
```

This fetches the options chain from Databento OPRA, computes the implied forward
curve with confidence intervals across all available expirations (up to 24
months out), and writes the report to `AAPL_forward_curve.html` in the current
directory.

### Custom Output File

```bash
python -m athena.metrics.options --ticker TSLA --output my_report.html
```

### All CLI Arguments

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--ticker` | `-t` | string | *(required)* | US stock ticker symbol (e.g. AAPL, TSLA, NVDA) |
| `--output` | `-o` | string | `{TICKER}_forward_curve.html` | Output HTML file path |
| `--risk-free-rate` | `-r` | float | `0.045` | Annualized risk-free rate for discounting |
| `--max-months` | `-m` | int | `24` | Maximum months forward to include expirations |
| `--no-cache` | | flag | off | Skip disk cache and fetch fresh data from Databento |
| `--spot` | | float | *(auto-fetched)* | Override spot price instead of fetching from YFinance |

### Examples

Score a specific ticker with all defaults (auto-fetches spot price, uses cache):
```bash
python -m athena.metrics.options -t RKLB
```

Write output to a specific path, skip cache, and limit to 6-month horizon:
```bash
python -m athena.metrics.options -t NVDA -o nvda_6m.html --no-cache --max-months 6
```

Override spot price manually (useful if YFinance is unavailable):
```bash
python -m athena.metrics.options -t AAPL --spot 230.50 -o aapl_custom.html
```

Use a different risk-free rate:
```bash
python -m athena.metrics.options -t MSFT -r 0.05
```

### Requirements

- `DATABENTO_API_KEY` environment variable must be set (or in `.env`)
- Databento OPRA access is required on your plan
- Spot price is auto-fetched via YFinance unless `--spot` is provided
