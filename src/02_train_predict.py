"""
STEP 2: FEATURE ENGINEERING AND NEXT-DAY PRICE PREDICTION
=============================================================
Builds technical indicators from raw price/volume data, trains a model
to predict the NEXT trading day's closing price, evaluates it honestly,
and outputs a live prediction using the most recent available data.

IMPORTANT - READ THIS BEFORE PRESENTING RESULTS:
Predicting exact next-day stock prices is extremely difficult - this is
well-established in financial research (the "efficient market
hypothesis" argues that public information is already reflected in
current prices, leaving next-day moves close to random). This project
is a genuine, honest technical exercise in time-series feature
engineering and regression modeling - NOT a claim that this model can
be used to actually trade profitably. The evaluation section below is
designed to surface this honestly rather than oversell the result.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/stock_data.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------------------------------------
# 1. FEATURE ENGINEERING - TECHNICAL INDICATORS
# ---------------------------------------------------------------
# Moving averages - smooth out day-to-day noise, capture trend direction
df["SMA_5"] = df["Close"].rolling(5).mean()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_20"] = df["Close"].rolling(20).mean()

# Daily return - the day's percentage price change
df["daily_return"] = df["Close"].pct_change()

# Rolling volatility - how much the price has been swinging recently
df["volatility_10"] = df["daily_return"].rolling(10).std()

# Momentum - price change over the last 5/10 days
df["momentum_5"] = df["Close"] - df["Close"].shift(5)
df["momentum_10"] = df["Close"] - df["Close"].shift(10)

# RSI (Relative Strength Index) - classic technical indicator measuring
# whether a stock has been overbought or oversold recently (0-100 scale)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# Volume change - unusual volume can signal unusual price moves
df["volume_change"] = df["Volume"].pct_change()

# ---------------------------------------------------------------
# 2. TARGET: NEXT DAY'S RETURN (not raw price)
# ---------------------------------------------------------------
# IMPORTANT METHODOLOGY NOTE: predicting the raw next-day PRICE directly
# causes a real problem for tree-based models like Random Forest - they
# can only predict values within the range they saw during training.
# If a stock trends upward over time, the test period's prices are
# HIGHER than anything the model saw in training, and it will badly
# under-predict (this is a genuine, well-known limitation of tree
# models on trending data, not a bug in this code).
#
# The standard fix: predict the next-day RETURN (% change) instead of
# the absolute price. Returns are much more stable/bounded over time
# than raw prices, which avoids the extrapolation problem. We then
# reconstruct the predicted price as: predicted_price = today's price
# * (1 + predicted_return).
df["target_next_return"] = df["Close"].shift(-1) / df["Close"] - 1

feature_cols = [
    "SMA_5", "SMA_10", "SMA_20", "daily_return",
    "volatility_10", "momentum_5", "momentum_10", "RSI_14", "volume_change"
]
# Note: raw "Close" is intentionally excluded from features now too -
# using it would reintroduce the same trending-price problem indirectly.

model_df = df.dropna(subset=feature_cols + ["target_next_return"]).reset_index(drop=True)

X = model_df[feature_cols]
y = model_df["target_next_return"]

# ---------------------------------------------------------------
# 3. TIME-BASED TRAIN/TEST SPLIT
# ---------------------------------------------------------------
# CRITICAL: for time series, you must NOT randomly shuffle train/test -
# that would let the model "see the future" during training. Instead,
# train on the earlier 80% chronologically, test on the most recent 20%.
split_idx = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 5. EVALUATE HONESTLY
# ---------------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Naive baseline: "tomorrow's return = 0%" (i.e. no change). If our
# model can't beat this simple baseline, it's not adding real predictive
# value - this is the single most important honesty check here.
naive_pred = np.zeros_like(y_test)
naive_mae = mean_absolute_error(y_test, naive_pred)

# Directional accuracy: did we correctly predict the SIGN of the return
# (up vs down)? Often the more meaningful metric for trading-adjacent
# use cases than exact return magnitude.
actual_direction = (y_test.values > 0).astype(int)
pred_direction = (y_pred > 0).astype(int)
directional_accuracy = (actual_direction == pred_direction).mean()

print("\n=== Model Evaluation (predicting % return, not raw price) ===")
print(f"Model MAE (avg return error):   {mae:.4f}  ({mae*100:.2f}%)")
print(f"Naive baseline MAE ('0% change'): {naive_mae:.4f}  ({naive_mae*100:.2f}%)")
print(f"R-squared:                       {r2:.3f}")
print(f"Directional accuracy:            {directional_accuracy:.1%}  (correctly predicted up/down)")

if mae < naive_mae:
    print("\n-> Model beats the naive baseline on average return error.")
else:
    print("\n-> Model does NOT beat the naive 'no change' baseline.")
    print("   This is common and expected for next-day stock prediction -")
    print("   see the note at the top of this script for why.")

print(f"\nFor reference: random guessing on direction alone would average ~50%.")
print(f"This model's directional accuracy: {directional_accuracy:.1%}")

# ---------------------------------------------------------------
# 6. LIVE PREDICTION - TOMORROW'S PRICE
# ---------------------------------------------------------------
latest_row = df.dropna(subset=feature_cols).iloc[[-1]][feature_cols]
latest_date = df.dropna(subset=feature_cols).iloc[-1]["Date"]
latest_close = df.dropna(subset=feature_cols).iloc[-1]["Close"]

predicted_return = model.predict(latest_row)[0]
next_day_prediction = latest_close * (1 + predicted_return)

print("\n=== Live Next-Day Prediction ===")
print(f"Most recent data: {latest_date.date()}, Close = ${latest_close:.2f}")
print(f"Predicted next-day return: {predicted_return*100:+.2f}%")
print(f"Predicted next trading day close: ${next_day_prediction:.2f}")
print("\nReminder: this is a technical exercise, not investment advice.")

# Save results
results_df = pd.DataFrame({"actual_return": y_test.values, "predicted_return": y_pred})
results_df.to_csv("outputs/test_predictions.csv", index=False)

with open("outputs/evaluation_summary.txt", "w") as f:
    f.write("STOCK PRICE PREDICTION - EVALUATION SUMMARY\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Model MAE (return): {mae:.4f} ({mae*100:.2f}%)\n")
    f.write(f"Naive baseline MAE: {naive_mae:.4f} ({naive_mae*100:.2f}%)\n")
    f.write(f"R-squared: {r2:.3f}\n")
    f.write(f"Directional accuracy: {directional_accuracy:.1%}\n\n")
    f.write(f"Latest prediction ({latest_date.date()}): ")
    f.write(f"${next_day_prediction:.2f} ({predicted_return*100:+.2f}%)\n")

print("\nSaved outputs/test_predictions.csv and outputs/evaluation_summary.txt")
