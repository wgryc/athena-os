# Options Theta Decay Analysis

## Goal

Given a portfolio xlsx file containing options positions (using OSI/OPRA symbol
format consistent with Databento), generate a standalone HTML report showing:

1. **Per-option theta decay curve** (2D) — theoretical value over time from
   purchase date to expiry, holding the underlying at today's spot price
2. **Per-option 3D pricing surface** (rotatable) — Black-Scholes value across
   both time and underlying price dimensions
3. **Portfolio expected value timeline** — aggregate position value from first
   purchase to last expiry, with a vertical "today" marker

---

## Portfolio Format

Options positions are identified in the standard portfolio xlsx by their
OSI-format symbols in the SYMBOL column:

| SYMBOL | DATE AND TIME | TRANSACTION TYPE | PRICE | QUANTITY | CURRENCY |
|--------|---------------|-----------------|-------|----------|----------|
| `AAPL  260320C00220000` | 2025-11-15 10:30:00 | BUY | 15.50 | 10 | USD |
| `TSLA  260620P00180000` | 2025-12-01 14:15:00 | BUY | 8.25 | 5 | USD |

- **PRICE**: Option premium per share (standard options quote convention)
- **QUANTITY**: Number of contracts (positive for BUY, negated for SELL)
- Non-option rows (stocks, cash, etc.) are silently skipped

### OSI Symbol Format (Databento OPRA)

```
AAPL  260320C00220000
│     │     ││       │
│     │     ││       └─ Strike × 1000 ($220.00)
│     │     │└─────── Call (C) or Put (P)
│     │     └──────── Expiration YYMMDD (2026-03-20)
│     └────────────── padding (spaces to 6 chars)
└──────────────────── Underlying root
```

This is the same symbology used by Databento's OPRA feed and parsed by the
existing `parse_osi_symbol()` function in `options.py`.

---

## Computations

### 1. Implied Volatility from Purchase Price

For each option position, IV is backed out from the purchase price:

1. Fetch the historical spot price at the purchase date (via YFinance)
2. Compute time to expiry at purchase: T₀ = (expiry − purchase_date) / 365.25
3. Solve `BS(S_purchase, K, T₀, r, σ) = purchase_price` for σ using Brent's
   method (existing `implied_volatility()` from `options.py`)

If historical spot is unavailable, falls back to current spot. If IV solve
fails, defaults to 30%.

### 2. Theta Decay Curve (2D Chart)

Computes BS theoretical value at each calendar day from purchase to expiry,
holding the underlying at **today's spot price**:

```
V(t) = BS(S_now, K, T(t), r, σ)
```

The chart overlays:
- **Theoretical value** — decays non-linearly, accelerating near expiry
- **Intrinsic value** — constant horizontal line (since spot is held constant)
- **Purchase price** — horizontal dashed reference line
- **Today** — vertical marker dividing past from projected future

The gap between theoretical and intrinsic lines represents the remaining
time value (extrinsic value), which is exactly what theta erodes.

### 3. 3D Pricing Surface (Rotatable)

Computes the full Black-Scholes surface:

```
V(t, S) = BS(S, K, T(t), r, σ)
```

Over a grid of:
- **Time**: purchase date → expiry (40 points)
- **Price**: current spot ± 30% (40 points)

This produces a 40×40 surface rendered via Plotly.js with full mouse rotation,
zoom, and hover tooltips. The surface reveals:
- How time value concentrates near ATM and vanishes at deep ITM/OTM
- The non-linear acceleration of decay near expiry
- The intrinsic "hockey stick" payoff that emerges at T=0

### 4. Portfolio Expected Value Timeline

Aggregates all positions from the earliest purchase date to the latest expiry:

```
Portfolio(t) = Σ V_i(t) × quantity_i    for all active positions at date t
```

- Before a position's purchase date: contributes $0
- Between purchase and expiry: Black-Scholes theoretical value × quantity
- After expiry: intrinsic value × quantity (or $0 if OTM)

A vertical "today" line divides realized decay from projected future decay.
A cost basis line shows cumulative capital deployed.

---

## CLI Usage

```bash
python -m athena.metrics.options_theta portfolio.xlsx
```

### Arguments

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `portfolio` | | string | *(required)* | Path to portfolio xlsx (positional) |
| `--output` | `-o` | string | `options_theta_report.html` | Output HTML file |
| `--risk-free-rate` | `-r` | float | `0.045` | Annualized risk-free rate |

### Examples

```bash
# Basic usage
python -m athena.metrics.options_theta my_options.xlsx

# Custom output and risk-free rate
python -m athena.metrics.options_theta my_options.xlsx -o report.html -r 0.05
```

---

## Key Assumptions

1. **Spot held constant**: Theta decay and portfolio curves use today's spot
   price throughout. This isolates the pure time decay effect.
2. **Constant IV**: The implied vol derived from the purchase price is used
   for all projections. In reality, IV changes over time (vol surface moves).
3. **European-style BS**: We use European Black-Scholes. US equity options
   are American-style; deep ITM options may show slight bias from early
   exercise premium.
4. **Calendar time**: All days (including weekends) are included since BS
   uses calendar time. Weekend theta is priced into Friday's close.

---

## Dependencies

**Python** (all already in the project):
- `numpy`, `scipy` — Black-Scholes computation and IV solving
- `pandas`, `openpyxl` — xlsx reading
- `yfinance` — spot price fetching (current and historical)
- `jinja2` — HTML template rendering

**Frontend** (loaded from CDN):
- **Chart.js** + annotation plugin — 2D theta decay and portfolio timeline
- **Plotly.js** — 3D rotatable pricing surfaces
