#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun June 7 15:03:00 2026

@author: Quantitative Risk Management Framework
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import yfinance as yf

# Set global seed for exact reproducibility across all random generations
np.random.seed(42)

# ==========================================
# 1. MARKET DATA ACQUISITION & PORTFOLIO CONSTRUCTION
# ==========================================
print("FETCHING REAL-WORLD MARKET DATA VIA YFINANCE...")

# Define tickers for the requested global indices
# ^GSPC: S&P 500, ^IXIC: NASDAQ, ^FCHI: CAC 40, ^FTSE: FTSE 100, ^GDAXI: DAX, ^HSI: Hang Seng
tickers = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'CAC 40': '^FCHI',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'Hang Seng': '^HSI'
}

# Fetch 5 years of historical data to capture full market cycles/regimes
start_date = "2021-06-01"
end_date = "2026-06-01"

data = yf.download(list(tickers.values()), start=start_date, end=end_date)['Close']
data = data.dropna() # Drop non-overlapping holidays/weekends across global exchanges

# Compute daily log returns for statistical symmetry
asset_returns = np.log(data / data.shift(1)).dropna()

# Construct an equally-weighted portfolio
num_assets = len(tickers)
weights = np.array([1.0 / num_assets] * num_assets)

# Generate the historical portfolio returns time series
historical_returns = asset_returns.dot(weights).to_numpy()
n_days = len(historical_returns)

# Extract empirical parameters for log printout
emp_drift = np.mean(historical_returns)
emp_volatility = np.std(historical_returns)

# ==========================================
# 2. USER-DEFINED SIMULATION INPUT PARAMETERS
# ==========================================
M = 50000                  # Number of Monte Carlo scenarios
alpha = 0.975              # Confidence level for baseline CDF plotting (Plot 2)

initial_price = 100        # Starting asset price for trajectory paths simulation
n_path_steps = 100         # Horizon length (in days) to project the sample paths
n_paths_to_plot = 15       # Number of random paths to visualize per method

print("\nINITIALIZING PORTFOLIO STRESS TESTING ENGINE...")
print(f"Historical Sample Size: {n_days} trading days")
print(f"Empirical Daily Drift (Mean): {emp_drift:.4%}")
print(f"Empirical Daily Volatility (Std Dev): {emp_volatility:.4%}\n" + "-"*60)


# ==========================================
# APPROACH 1: GEOMETRIC BROWNIAN MOTION (NORMAL)
# ==========================================
# GBM extracts parameters via Maximum Likelihood Estimation, assuming perfect normality
mu_norm, sigma_norm = stats.norm.fit(historical_returns)

# 1A. Returns for VaR/CVaR static horizon
sim_returns_gbm = np.random.normal(loc=mu_norm, scale=sigma_norm, size=M)

# 1B. Dynamic path generation matrix (steps x paths)
paths_returns_gbm = np.random.normal(loc=mu_norm, scale=sigma_norm, size=(n_path_steps, n_paths_to_plot))
price_paths_gbm = initial_price * np.exp(np.vstack([np.zeros(n_paths_to_plot), np.cumsum(paths_returns_gbm, axis=0)]))


# ==========================================
# APPROACH 2: STUDENT'S T-DISTRIBUTION
# ==========================================
# Fits parameters directly to map history onto a theoretical t-curve (captures fat tails)
df_est, loc_est, scale_est = stats.t.fit(historical_returns)

# 2A. Returns for VaR/CVaR static horizon
sim_returns_t = stats.t.rvs(df=df_est, loc=loc_est, scale=scale_est, size=M)

# 2B. Dynamic path generation matrix
paths_returns_t = stats.t.rvs(df=df_est, loc=loc_est, scale=scale_est, size=(n_path_steps, n_paths_to_plot))
price_paths_t = initial_price * np.exp(np.vstack([np.zeros(n_paths_to_plot), np.cumsum(paths_returns_t, axis=0)]))


# ==========================================
# APPROACH 3: KERNEL DENSITY ESTIMATION (KDE)
# ==========================================
# Fits a non-parametric Gaussian KDE over the historical sample returns
kde = stats.gaussian_kde(historical_returns, bw_method='scott')
h = kde.factor * np.std(historical_returns)

# 3A. Returns for VaR/CVaR static horizon (Hierarchical sampling)
random_indices = np.random.choice(n_days, size=M, replace=True)
x_I = historical_returns[random_indices]
Z = np.random.normal(0, 1, M)
sim_returns_kde = x_I + h * Z

# 3B. Dynamic path generation matrix via structured KDE drawing loops
paths_returns_kde = np.zeros((n_path_steps, n_paths_to_plot))
for step in range(n_path_steps):
    idx = np.random.choice(n_days, size=n_paths_to_plot, replace=True)
    paths_returns_kde[step, :] = historical_returns[idx] + h * np.random.normal(0, 1, n_paths_to_plot)
price_paths_kde = initial_price * np.exp(np.vstack([np.zeros(n_paths_to_plot), np.cumsum(paths_returns_kde, axis=0)]))


# ==========================================
# RISK METRICS EVALUATION FUNCTION
# ==========================================
def calculate_var_cvar(sim_returns, alpha=0.975):
    """
    Sorts returns and extracts Value at Risk and Conditional Value at Risk.
    Returns are transformed to positive numbers representing losses.
    """
    sorted_returns = np.sort(sim_returns)
    k = int(np.floor((1 - alpha) * len(sorted_returns)))
    
    var = -sorted_returns[k]
    cvar = -np.mean(sorted_returns[:k])
    return var, cvar


# ==========================================
# COMPUTE SYNCHRONIZED TARGET METRICS
# ==========================================
# Calculate specific cross-quantile targets for Plot 1 and Plot 3
var_gbm_99, _ = calculate_var_cvar(sim_returns_gbm, alpha=0.99)
var_t_99, _   = calculate_var_cvar(sim_returns_t, alpha=0.99)
var_kde_99, _ = calculate_var_cvar(sim_returns_kde, alpha=0.99)

_, cvar_gbm_975 = calculate_var_cvar(sim_returns_gbm, alpha=0.975)
_, cvar_t_975   = calculate_var_cvar(sim_returns_t, alpha=0.975)
_, cvar_kde_975 = calculate_var_cvar(sim_returns_kde, alpha=0.975)

# Calculate Skewness and Excess Kurtosis across frameworks
skew_gbm, kurt_gbm = stats.skew(sim_returns_gbm), stats.kurtosis(sim_returns_gbm)
skew_t, kurt_t     = stats.skew(sim_returns_t), stats.kurtosis(sim_returns_t)
skew_kde, kurt_kde = stats.skew(sim_returns_kde), stats.kurtosis(sim_returns_kde)


# ==========================================
# STATISTICAL TESTS FOR SKEWNESS AND KURTOSIS
# ==========================================
def test_skewness_kurtosis(data, name):
    """
    Perform statistical tests for significance of skewness and excess kurtosis.
    """
    n = len(data)
    sample_skew = stats.skew(data)
    sample_kurt = stats.kurtosis(data)  # Excess kurtosis
    
    if n < 4:
        return {
            'name': name, 'skew': sample_skew, 'kurt': sample_kurt,
            'skew_pvalue': np.nan, 'kurt_pvalue': np.nan, 'dagostino_pvalue': np.nan
        }
    
    se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    se_kurt = np.sqrt(24 * n * (n - 1)**2 / ((n - 3) * (n - 2) * (n + 1) * (n + 3)))
    
    z_skew = sample_skew / se_skew if not np.isnan(se_skew) else np.nan
    z_kurt = sample_kurt / se_kurt if not np.isnan(se_kurt) else np.nan
    
    p_skew = 2 * (1 - stats.norm.cdf(abs(z_skew))) if not np.isnan(z_skew) else np.nan
    p_kurt = 2 * (1 - stats.norm.cdf(abs(z_kurt))) if not np.isnan(z_kurt) else np.nan
    
    dagostino_stat, dagostino_p = stats.normaltest(data)
    
    return {
        'name': name, 'skew': sample_skew, 'kurt': sample_kurt,
        'z_skew': z_skew, 'z_kurt': z_kurt,
        'skew_pvalue': p_skew, 'kurt_pvalue': p_kurt,
        'dagostino_stat': dagostino_stat, 'dagostino_pvalue': dagostino_p
    }


# Perform tests for each distribution
results_gbm = test_skewness_kurtosis(sim_returns_gbm, 'GBM (Normal)')
results_t = test_skewness_kurtosis(sim_returns_t, "Student's t")
results_kde = test_skewness_kurtosis(sim_returns_kde, 'KDE')


# ==========================================
# RESULTS PRINT OUT
# ==========================================
print(f"{'Approach':<30} | {'99.0% VaR':<10} | {'97.5% CVaR':<10} | {'Skewness':<9} | {'Ex. Kurt':<9}")
print("-" * 78)
print(f"{'1. Geometric Brownian (Normal)':<30} | {var_gbm_99:.4%}   | {cvar_gbm_975:.4%}   | {skew_gbm:+0.4f}  | {kurt_gbm:+0.4f}")
print(f"{'2. Student\'s t-Distribution':<30} | {var_t_99:.4%}   | {cvar_t_975:.4%}   | {skew_t:+0.4f}  | {kurt_t:+0.4f}")
print(f"{'3. Kernel Density Estimator':<30} | {var_kde_99:.4%}   | {cvar_kde_975:.4%}   | {skew_kde:+0.4f}  | {kurt_kde:+0.4f}")
print("-" * 78 + "\n")


# ==========================================
# PRINT STATISTICAL TEST RESULTS
# ==========================================
print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE TESTS FOR SKEWNESS AND KURTOSIS")
print("=" * 80)

for results in [results_gbm, results_t, results_kde]:
    skew_sig = '***' if results['skew_pvalue'] < 0.001 else '**' if results['skew_pvalue'] < 0.01 else '*' if results['skew_pvalue'] < 0.05 else ''
    kurt_sig = '***' if results['kurt_pvalue'] < 0.001 else '**' if results['kurt_pvalue'] < 0.01 else '*' if results['kurt_pvalue'] < 0.05 else ''
    dag_sig = '***' if results['dagostino_pvalue'] < 0.001 else '**' if results['dagostino_pvalue'] < 0.01 else '*' if results['dagostino_pvalue'] < 0.05 else ''
    
    print(f"\n{results['name']}:")
    print(f"  Skewness: {results['skew']:+0.4f} (z = {results['z_skew']:+0.4f}, p-value = {results['skew_pvalue']:.6f} {skew_sig})")
    print(f"  Ex. Kurtosis: {results['kurt']:+0.4f} (z = {results['z_kurt']:+0.4f}, p-value = {results['kurt_pvalue']:.6f} {kurt_sig})")
    print(f"  D'Agostino K^2: stat = {results['dagostino_stat']:+0.4f}, p-value = {results['dagostino_pvalue']:.6f} {dag_sig}")

print("\n" + "-" * 80)
print("Significance codes: '***' p < 0.001, '**' p < 0.01, '*' p < 0.05")
print("=" * 80 + "\n")

global_min_tail = np.percentile(historical_returns, 0.1)
global_max_tail = np.percentile(historical_returns, 15)

# Global styling mapping for the comparative charts
colors = {'gbm': '#1f77b4', 't': '#ff7f0e', 'kde': '#2ca02c'}

# ==========================================
# PLOT 1: PROBABILITY DENSITY TAIL FOCUS
# ==========================================
plt.figure(figsize=(10, 6))
plt.hist(sim_returns_gbm, bins=250, density=True, alpha=0.3, color=colors['gbm'], label='GBM (Normal)')
plt.hist(sim_returns_t, bins=250, density=True, alpha=0.3, color=colors['t'], label="Student's t")
plt.hist(sim_returns_kde, bins=250, density=True, alpha=0.3, color=colors['kde'], label='KDE')

plt.axvline(-var_gbm_99, color=colors['gbm'], linestyle=':', linewidth=2, label=f'GBM 99% VaR ({var_gbm_99:.2%})')
plt.axvline(-var_t_99, color=colors['t'], linestyle=':', linewidth=2, label=f'Student t 99% VaR ({var_t_99:.2%})')
plt.axvline(-var_kde_99, color=colors['kde'], linestyle=':', linewidth=2, label=f'KDE 99% VaR ({var_kde_99:.2%})')

plt.axvline(-cvar_gbm_975, color=colors['gbm'], linestyle='-', linewidth=2, label=f'GBM 97.5% CVaR ({cvar_gbm_975:.2%})')
plt.axvline(-cvar_t_975, color=colors['t'], linestyle='-', linewidth=2, label=f'Student t 97.5% CVaR ({cvar_t_975:.2%})')
plt.axvline(-cvar_kde_975, color=colors['kde'], linestyle='-', linewidth=2, label=f'KDE 97.5% CVaR ({cvar_kde_975:.2%})')

plt.title("Probability Density Focus: 99% VaR vs 97.5% CVaR Loss Tail (Real Data)", fontsize=13, fontweight='bold')
plt.xlabel("Daily Portfolio Returns")
plt.ylabel("Density")
plt.xlim(global_min_tail, global_max_tail)
plt.legend(fontsize=9, loc='upper left')
plt.grid(True, alpha=0.2)
plt.show()

# ==========================================
# PLOT 2: CUMULATIVE DISTRIBUTION FUNCTION (CDF)
# ==========================================
plt.figure(figsize=(10, 6))
for data, name, col in zip([sim_returns_gbm, sim_returns_t, sim_returns_kde], 
                           ['GBM (Normal)', "Student's t", 'KDE'], 
                           [colors['gbm'], colors['t'], colors['kde']]):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=name, color=col, linewidth=2)

plt.axhline(1-0.975, color='red', linestyle='--', linewidth=1.2, label='Significance Level (0.025)')
plt.axhline(1-0.99, color='blue', linestyle='--', linewidth=1.2, label='Significance Level (0.01)')
plt.title("Empirical CDF Quantile Trajectory (Real Data)", fontsize=13, fontweight='bold')
plt.xlabel("Daily Portfolio Returns")
plt.ylabel("Cumulative Probability")
plt.xlim(np.percentile(historical_returns, 0.1), 0.01) 
plt.ylim(0, 0.10) 
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.2)
plt.show()

# ==========================================
# PLOT 3: DIRECT METRIC COMPARISON (BAR CHART)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['GBM (Normal)', "Student's t", 'KDE']
var_values = [var_gbm_99 * 100, var_t_99 * 100, var_kde_99 * 100]    
cvar_values = [cvar_gbm_975 * 100, cvar_t_975 * 100, cvar_kde_975 * 100]

x = np.arange(len(categories))
width = 0.35
rects1 = ax.bar(x - width/2, var_values, width, label='99% VaR', color='#bcbd22', alpha=0.85)
rects2 = ax.bar(x + width/2, cvar_values, width, label='97.5% CVaR', color='#d62728', alpha=0.85)

ax.set_title("Direct Value Comparison: 99% VaR vs 97.5% CVaR (%)", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Loss Scale (%)")
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, axis='y', alpha=0.3)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
plt.tight_layout()
plt.show()

# ==========================================
# PLOTS 4, 5, 6: SIMULATED PRICE TRAJECTORIES
# ==========================================
for paths, name, col in zip([price_paths_gbm, price_paths_t, price_paths_kde],
                            ['Geometric Brownian Motion (Normal)', "Student's t-Distribution", 'Kernel Density Estimator (KDE)'],
                            [colors['gbm'], colors['t'], colors['kde']]):
    plt.figure(figsize=(10, 4))
    plt.plot(paths, linewidth=1.5, color=col, alpha=0.6)
    plt.title(f"{name} Simulated Price Paths (Real Data)", fontsize=12, fontweight='bold')
    plt.xlabel("Time Horizon (Days)")
    plt.ylabel("Asset Price")
    plt.grid(True, alpha=0.3)
    plt.axhline(initial_price, color='black', linestyle='--', alpha=0.5)
    plt.show()

# ==========================================
# PREPARATION FOR DISTRIBUTION OVERLAYS (PLOTS 7-9)
# ==========================================
x_axis = np.linspace(np.min(historical_returns), np.max(historical_returns), 1000)
normal_curve = stats.norm.pdf(x_axis, loc=mu_norm, scale=sigma_norm)

# PLOT 7: HISTOGRAM - GBM
plt.figure(figsize=(10, 4))
plt.hist(sim_returns_gbm, bins=150, density=True, alpha=0.6, color=colors['gbm'], edgecolor='black', linewidth=0.5)
plt.plot(x_axis, normal_curve, color='red', linestyle='-', linewidth=2, label='Theoretical Normal Curve')
plt.title("Distribution of Returns - Geometric Brownian Motion (Normal)", fontsize=12, fontweight='bold')
plt.xlabel("Simulated Daily Returns")
plt.ylabel("Frequency Density")
plt.xlim(np.percentile(historical_returns, 0.1), np.percentile(historical_returns, 99.9))
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# PLOT 8: HISTOGRAM - STUDENT'S T
plt.figure(figsize=(10, 4))
plt.hist(sim_returns_t, bins=150, density=True, alpha=0.6, color=colors['t'], edgecolor='black', linewidth=0.5)
plt.plot(x_axis, normal_curve, color='red', linestyle='-', linewidth=2, label='Theoretical Normal Curve')
t_curve = stats.t.pdf(x_axis, df=df_est, loc=loc_est, scale=scale_est)
plt.plot(x_axis, t_curve, color='black', linestyle='--', linewidth=2, label="Fitted Student's t Curve")
plt.title("Distribution of Returns - Student's t-Distribution", fontsize=12, fontweight='bold')
plt.xlabel("Simulated Daily Returns")
plt.ylabel("Frequency Density")
plt.xlim(np.percentile(historical_returns, 0.1), np.percentile(historical_returns, 99.9))
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# PLOT 9: HISTOGRAM - KDE
plt.figure(figsize=(10, 4))
plt.hist(sim_returns_kde, bins=150, density=True, alpha=0.6, color=colors['kde'], edgecolor='black', linewidth=0.5)
plt.plot(x_axis, normal_curve, color='red', linestyle='-', linewidth=2, label='Theoretical Normal Curve')
plt.plot(x_axis, kde.pdf(x_axis), color='darkgreen', linestyle='-', linewidth=2, label='KDE Continuous Fit Line')
plt.title("Distribution of Returns - Kernel Density Estimator (KDE)", fontsize=12, fontweight='bold')
plt.xlabel("Simulated Daily Returns")
plt.ylabel("Frequency Density")
plt.xlim(np.percentile(historical_returns, 0.1), np.percentile(historical_returns, 99.9))
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()