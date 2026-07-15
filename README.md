# Real-Time Stock Price Predictor (AAPL)

A time-series machine learning project that fetches live stock data,
engineers technical indicators, and predicts next-day price movement —
built with an honest evaluation of how well (or poorly) this actually
works, rather than an inflated headline number.

## Why "Real-Time"

Unlike a static dataset, `01_fetch_data.py` pulls **live, current data**
directly from Yahoo Finance every time it's run, using the `yfinance`
library. There's no stale CSV snapshot — running this today vs. next
week will reflect actual new trading days.

## Important: Read This Before Presenting Results

Predicting next-day stock prices is genuinely very difficult — this
isn't a limitation of this particular project, it's a well-established
finding in financial research (the "efficient market hypothesis"
argues that publicly available information is already reflected in
current prices, making next-day moves close to statistically random).

This project is built and evaluated with that honesty in mind:

- The model's predictions are compared against a **naive baseline**
  ("tomorrow = today, no change") — if the model can't beat that, it's
  said explicitly, not hidden.
- **Directional accuracy** (did it correctly predict up vs. down) is
  reported alongside price error, since that's often the more
  meaningful metric for this kind of problem.
- In testing, the model's directional accuracy came out close to 50%
  (essentially a coin flip) — which is exactly what financial theory
  would predict, and is reported as-is rather than cherry-picked.

**Why build this if it "doesn't work" well:** the value here is
demonstrating time-series feature engineering, correct train/test
methodology for sequential data, and honest model evaluation — not
claiming to have solved stock prediction (nobody has, reliably).

## Pipeline

| Step | Script | What it does |
|------|--------|---------------|
| 1 | `src/01_fetch_data.py` | Pulls live daily price data for AAPL via yfinance |
| 2 | `src/02_train_predict.py` | Engineers features, trains a model, evaluates honestly, outputs a live next-day prediction |

## Key Technical Decision: Predicting Returns, Not Raw Prices

An early version of this project predicted the next day's raw closing
price directly, and performed very badly (negative R²) — because
tree-based models like Random Forest can't extrapolate beyond the price
range seen in training. On a trending stock, test-period prices are
higher than anything seen in training, so the model badly under-predicts.

**The fix:** predict the next-day **percentage return** instead of the
raw price, then reconstruct the predicted price as
`today's price × (1 + predicted return)`. Returns are far more stable
over time than raw prices, which avoids this extrapolation problem
entirely. This is standard practice in financial ML and is a good
example of catching and correctly fixing a real methodological issue.

## Features Used

- 5/10/20-day simple moving averages
- Daily return, 10-day rolling volatility
- 5/10-day price momentum
- RSI (Relative Strength Index) — a classic technical indicator
- Volume change

## How To Run

```bash
pip install yfinance pandas scikit-learn
python src/01_fetch_data.py
python src/02_train_predict.py
```

Re-run both scripts anytime to get an updated prediction using the
latest trading data.

## Tech Stack

Python, yfinance, pandas, scikit-learn