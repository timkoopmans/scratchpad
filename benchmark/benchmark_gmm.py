import numpy as np
import pandas as pd
import time
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import optuna
import yfinance as yf
import random

# Set all random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# check if csv file exists
try:
    df = pd.read_csv('btc_5min_data.csv', index_col=0, parse_dates=True)
    # Ensure only numeric columns are used
    df = df[['close', 'volume']]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
except FileNotFoundError:
    print("btc_5min_data.csv not found, downloading data...")
    # Download real BTC price and volume data
    btc = yf.download('BTC-USD', start='2025-07-01', end='2025-07-31', interval='5m')
    df = btc[['Close', 'Volume']].dropna().rename(columns={'Close': 'close', 'Volume': 'volume'})
    # save to CSV for reproducibility
    df.to_csv('btc_5min_data.csv')

# If volume is always zero, replace it with synthetic values
if (df['volume'] == 0).all():
    np.random.seed(42)
    df['volume'] = np.abs(np.random.randn(len(df)) * 1000 + 5000)
print("Using data: ", df.head())
timestamps = df.index
n_samples = len(df)

# True labels for synthetic clusters (for accuracy)
true_labels_path = 'true_labels.npy'
try:
    true_labels = np.load(true_labels_path)
    if len(true_labels) != n_samples:
        raise ValueError("true_labels length mismatch")
except (FileNotFoundError, ValueError):
    true_labels = np.random.randint(0, 5, size=n_samples)
    np.save(true_labels_path, true_labels)

def objective(trial):
    n_components = trial.suggest_int("n_components", 3, 10)
    max_iter = trial.suggest_int("max_iter", 50, 500)
    weight_concentration_prior = trial.suggest_float("weight_concentration_prior", 0.1, 10.0, log=True)
    covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
    n_init = trial.suggest_int("n_init", 1, 5)
    init_params = trial.suggest_categorical("init_params", ["k-means++", "random"])
    reg_covar = trial.suggest_float("reg_covar", 1e-6, 1e-3, log=True)
    fit_kwargs = {
        "n_components": n_components,
        "weight_concentration_prior": weight_concentration_prior,
        "covariance_type": covariance_type,
        "random_state": 42,
        "max_iter": max_iter,
        "n_init": n_init,
        "init_params": init_params,
        "reg_covar": reg_covar,
    }
    features = np.column_stack((df["close"].values, df["volume"].values))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    start = time.time()
    model = BayesianGaussianMixture(**fit_kwargs)
    model.fit(X_scaled)
    fit_time = time.time() - start
    # Predict each bar retrospectively and measure predict time
    pred_labels = np.zeros(n_samples, dtype=int)
    start_pred = time.time()
    for i in range(n_samples):
        pred_labels[i] = model.predict(X_scaled[i].reshape(1, -1))[0]
    predict_time = time.time() - start_pred
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ari, fit_time, predict_time

study = optuna.create_study(
    directions=["maximize", "minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=20)

print("Pareto-optimal trials:")
for t in study.best_trials:
    print(f"Params: {t.params}, ARI: {t.values[0]:.6f}, Fit time: {t.values[1]:.6f}s, Predict time: {t.values[2]:.6f}s")

# Find the best trial (highest ARI, then lowest fit and predict time)
best_trial = max(study.best_trials, key=lambda t: (t.values[0], -t.values[1], -t.values[2]))

print("\nBest combination:")
print(f"Params: {best_trial.params}")
print(f"ARI: {best_trial.values[0]:.6f}")
print(f"Fit time: {best_trial.values[1]:.6f} seconds")
print(f"Predict time: {best_trial.values[2]:.6f} seconds")

# Estimate real-life total time for 4 weeks of 5-min bars
n_bars = n_samples
fit_time = best_trial.values[1]
predict_time = best_trial.values[2]
total_time = fit_time + predict_time
print(f"Total time to fit and predict across {n_bars} bars (4w): {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# Estimate for 20 tickers over 5 years (5-min bars)
minutes_per_year = 365 * 24 * 60
bars_per_year = minutes_per_year // 5
bars_5y_20t = bars_per_year * 5 * 20
multiplier = bars_5y_20t / n_bars if n_bars > 0 else 0
est_total_time_5y_20t = total_time * multiplier
print(f"Estimated time to fit and predict for 20 tickers over 5 years: {est_total_time_5y_20t:.2f} seconds ({est_total_time_5y_20t/3600:.2f} hours, {est_total_time_5y_20t/86400:.2f} days)")

# Baseline parameters
baseline_params = {
    "n_components": 10,
    "weight_concentration_prior": 10.0,
    "covariance_type": "full",
    "random_state": 42,
    "max_iter": 500,
    "n_init": 3,
    "init_params": "k-means++",
    "reg_covar": 1e-5,
}
features = np.column_stack((df["close"].values, df["volume"].values))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
start_fit = time.time()
baseline_model = BayesianGaussianMixture(**baseline_params)
baseline_model.fit(X_scaled)
baseline_fit_time = time.time() - start_fit
# Predict each bar retrospectively and measure predict time
baseline_pred_labels = np.zeros(n_samples, dtype=int)
start_pred = time.time()
for i in range(n_samples):
    baseline_pred_labels[i] = baseline_model.predict(X_scaled[i].reshape(1, -1))[0]
baseline_predict_time = time.time() - start_pred
baseline_ari = adjusted_rand_score(true_labels, baseline_pred_labels)
baseline_total_time = baseline_fit_time + baseline_predict_time
# Estimate for 20 tickers over 5 years (5-min bars)
baseline_multiplier = bars_5y_20t / n_bars if n_bars > 0 else 0
baseline_est_total_time_5y_20t = baseline_total_time * baseline_multiplier

print("\n--- Baseline Parameters ---")
print(f"Params: {baseline_params}")
print(f"ARI: {baseline_ari:.6f}")
print(f"Fit time: {baseline_fit_time:.6f} seconds")
print(f"Predict time: {baseline_predict_time:.6f} seconds")
print(f"Total time to fit and predict across {n_bars} bars (4w): {baseline_total_time:.2f} seconds ({baseline_total_time/60:.2f} minutes)")
print(f"Estimated time to fit and predict for 20 tickers over 5 years: {baseline_est_total_time_5y_20t:.2f} seconds ({baseline_est_total_time_5y_20t/3600:.2f} hours, {baseline_est_total_time_5y_20t/86400:.2f} days)")

print("\n--- Comparison ---")
print(f"Best Optuna ARI: {best_trial.values[0]:.6f} vs Baseline ARI: {baseline_ari:.6f}")
print(f"Best Optuna total time (4w): {total_time:.2f}s vs Baseline: {baseline_total_time:.2f}s")
print(f"Best Optuna est. time (5y, 20 tickers): {est_total_time_5y_20t:.2f}s vs Baseline: {baseline_est_total_time_5y_20t:.2f}s")
