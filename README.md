# Forecasting the VIX from Macro and Credit-Market Variables (SARIMAX + GARCH)

One-step-ahead forecasting of the CBOE Volatility Index (VIX) using exogenous
macro and credit-market regressors, with two complementary models: a SARIMAX
model for the conditional mean and a SARIMAX + GARCH(2,1) two-stage model that
adds a conditional-variance forecast.

## Problem

Predict the VIX from market fundamentals — credit spreads, equity returns, rates,
sentiment, and option flow — rather than from its own history alone, and quantify
forecast uncertainty in a way that respects the VIX's volatility clustering.

## Data

- **Target:** `VIXCLS.csv` (daily VIX close).
- **Exogenous:** `combined_data.csv` — IG/HY credit spreads (`BAMLC0A4CBBB`,
  `BAMLC0A4CBBBEY`), `SPX return`, `DJIA return`, `DGS10`, `UMCSENT`, `UNRATE`,
  `ICSA`, and CBOE SP500 call/put volumes.

## Method

1. **EDA** — VIX level/distribution; correlation of each regressor with the VIX
   (credit spreads co-move most strongly).
2. **Stationarity** — ACF/PACF and an Augmented Dickey–Fuller test.
3. **Model 1 — SARIMAX** — `auto_arima` order selection with exogenous
   regressors; forecast, confidence intervals, and residual diagnostics.
4. **Model 2 — SARIMAX + GARCH(2,1)** — SARIMAX for the mean, a zero-mean GARCH
   fit to the SARIMAX residuals for the conditional variance and interval band.
5. **Evaluation** — mean squared error **and** mean directional accuracy (MDA),
   on a strict chronological hold-out.

## Methodology notes

- **Chronological split.** Train/test is strictly time-ordered (first 80% / last
  20%). The earlier version used `sklearn.train_test_split` with default
  shuffling, which destroyed temporal order *and* shuffled X and y independently
  (mis-aligning them); both are fixed here.
- **Correct GARCH target.** GARCH is fit on the SARIMAX residuals with a
  zero-mean spec — the right object to model — rather than passing exogenous
  regressors into `arch_model`.
- **Directional accuracy.** MDA is reported alongside MSE because, for a
  volatility signal, the sign of the next move is the actionable output.

## Limitations / next steps

- Single hold-out — a **walk-forward** (re-fit at each step) evaluation is the
  natural next iteration and a far better estimate of live performance.
- Exogenous regressors should be **lagged** to forecast-time availability to rule
  out look-ahead; worth auditing each one.
- MDA is statistical, not economic; no trading-rule or transaction-cost layer.

## Repository contents

```
VIX_Forecasting_SARIMAX_GARCH.ipynb   # EDA, stationarity, SARIMAX, SARIMAX+GARCH
README.md
# add: VIXCLS.csv, combined_data.csv (or document their FRED/CBOE sources)
```

## Running

```bash
pip install pandas numpy matplotlib seaborn statsmodels pmdarima arch scikit-learn
# set VIX_PATH and EXOG_PATH in the notebook
jupyter notebook VIX_Forecasting_SARIMAX_GARCH.ipynb
```

EDA, ACF/PACF, ADF, and forecast plots retain their saved outputs; re-run on the
data to regenerate model summaries and metrics.
