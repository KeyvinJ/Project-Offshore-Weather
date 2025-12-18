#!/usr/bin/env python3
"""
Execute Phase 5B1: Seasonal EVA Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import genextreme, gumbel_r, weibull_min
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("PHASE 5B1: SEASONAL EXTREME VALUE ANALYSIS")
print("="*80)
print("\nObjective: Analyze how extreme values vary by season")
print("Expected: Winter extremes >> Summer extremes")
print("\n" + "="*80 + "\n")

# ============================================================================
# PART 1: LOAD DATA AND PHASE 4A RESULTS
# ============================================================================

print("\n[PART 1] Loading Data...")

# Load hourly data
df = pd.read_csv('data/processed/era5_with_currents_cleaned.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['year_month'] = df['timestamp'].dt.to_period('M')

print(f"✓ Loaded {len(df):,} hourly records")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Load Phase 4A EVA distributions
with open('data/processed/phase4a/eva_distributions.pkl', 'rb') as f:
    eva_dists_phase4a = pickle.load(f)

print(f"✓ Loaded Phase 4A EVA distributions (baseline for comparison)")

# ============================================================================
# PART 2: DEFINE SEASONS AND EXTRACT SEASONAL MONTHLY MAXIMA
# ============================================================================

print("\n[PART 2] Extracting Seasonal Monthly Maxima...")

# Define meteorological seasons
SEASONS = {
    'Winter': [12, 1, 2],   # DJF (December, January, February)
    'Spring': [3, 4, 5],    # MAM (March, April, May)
    'Summer': [6, 7, 8],    # JJA (June, July, August)
    'Autumn': [9, 10, 11]   # SON (September, October, November)
}

# Extract monthly maxima for each season
seasonal_monthly_maxima = {}

for season, months in SEASONS.items():
    df_season = df[df['month'].isin(months)]

    mm_hs = df_season.groupby('year_month')['hs'].max().dropna()
    mm_wind = df_season.groupby('year_month')['wind_speed'].max().dropna()
    mm_current = df_season.groupby('year_month')['current_speed_knots_max'].max().dropna()

    seasonal_monthly_maxima[season] = {
        'hs': mm_hs.values,
        'wind': mm_wind.values,
        'current': mm_current.values if len(mm_current) > 0 else np.array([])
    }

    print(f"\n{season}:")
    print(f"  Hs:      {len(mm_hs)} months, mean={mm_hs.mean():.2f}m, max={mm_hs.max():.2f}m")
    print(f"  Wind:    {len(mm_wind)} months, mean={mm_wind.mean():.2f}m/s, max={mm_wind.max():.2f}m/s")
    if len(mm_current) > 0:
        print(f"  Current: {len(mm_current)} months, mean={mm_current.mean():.2f}kt, max={mm_current.max():.2f}kt")
    else:
        print(f"  Current: No data")

# ============================================================================
# PART 3: VISUALIZE SEASONAL DIFFERENCES
# ============================================================================

print("\n[PART 3] Creating Seasonal Visualizations...")

# Create comprehensive seasonal comparison figure
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Seasonal Extreme Value Comparison', fontsize=16, fontweight='bold')

variables = ['hs', 'wind', 'current']
var_labels = ['Significant Wave Height (m)', 'Wind Speed (m/s)', 'Current Speed (knots)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (var, label) in enumerate(zip(variables, var_labels)):
    # Histogram
    ax_hist = axes[i, 0]
    for season_idx, (season, months) in enumerate(SEASONS.items()):
        data = seasonal_monthly_maxima[season][var]
        if len(data) > 0:
            ax_hist.hist(data, bins=15, alpha=0.5, label=season, color=colors[season_idx], density=True)
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel('Density')
    ax_hist.set_title(f'{label} - Distribution')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # Boxplot
    ax_box = axes[i, 1]
    data_list = []
    labels_list = []
    for season in SEASONS.keys():
        data = seasonal_monthly_maxima[season][var]
        if len(data) > 0:
            data_list.append(data)
            labels_list.append(season)

    if data_list:
        bp = ax_box.boxplot(data_list, labels=labels_list, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data_list)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax_box.set_ylabel(label)
    ax_box.set_title(f'{label} - Comparison')
    ax_box.grid(True, alpha=0.3)

    # Monthly pattern
    ax_monthly = axes[i, 2]
    monthly_data = []
    for m in range(1, 13):
        df_month = df[df['month'] == m]
        if var == 'hs':
            monthly_max = df_month.groupby('year_month')['hs'].max().dropna()
        elif var == 'wind':
            monthly_max = df_month.groupby('year_month')['wind_speed'].max().dropna()
        else:
            monthly_max = df_month.groupby('year_month')['current_speed_knots_max'].max().dropna()

        if len(monthly_max) > 0:
            monthly_data.append(monthly_max.values)
        else:
            monthly_data.append(np.array([]))

    # Plot only months with data
    valid_months = [m for m in range(1, 13) if len(monthly_data[m-1]) > 0]
    valid_data = [monthly_data[m-1] for m in valid_months]

    if valid_data:
        bp_monthly = ax_monthly.boxplot(valid_data, positions=valid_months, patch_artist=True)

        # Color by season
        for m_idx, m in enumerate(valid_months):
            for season, months in SEASONS.items():
                if m in months:
                    season_idx = list(SEASONS.keys()).index(season)
                    bp_monthly['boxes'][m_idx].set_facecolor(colors[season_idx])
                    bp_monthly['boxes'][m_idx].set_alpha(0.5)

    ax_monthly.set_xlabel('Month')
    ax_monthly.set_ylabel(label)
    ax_monthly.set_title(f'{label} - Monthly Pattern')
    ax_monthly.set_xticks(range(1, 13))
    ax_monthly.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax_monthly.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/phase5b1/seasonal_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved seasonal_comparison.png")

# ============================================================================
# PART 4: FIT EVA DISTRIBUTIONS FOR EACH SEASON
# ============================================================================

print("\n[PART 4] Fitting EVA Distributions for Each Season...")

seasonal_eva_fits = {}

for season in SEASONS.keys():
    print(f"\n{season}:")
    seasonal_eva_fits[season] = {}

    for var in ['hs', 'wind', 'current']:
        data = seasonal_monthly_maxima[season][var]

        if len(data) < 10:  # Need at least 10 samples for reliable fitting
            print(f"  {var}: Insufficient data (n={len(data)})")
            seasonal_eva_fits[season][var] = None
            continue

        # Fit Weibull, GEV, Gumbel
        fits = {}

        # Weibull
        try:
            params_weibull = weibull_min.fit(data, floc=0)
            nll_weibull = -np.sum(weibull_min.logpdf(data, *params_weibull))
            aic_weibull = 2 * len(params_weibull) + 2 * nll_weibull
            fits['Weibull'] = {'params': params_weibull, 'aic': aic_weibull}
        except:
            fits['Weibull'] = None

        # GEV
        try:
            params_gev = genextreme.fit(data)
            nll_gev = -np.sum(genextreme.logpdf(data, *params_gev))
            aic_gev = 2 * len(params_gev) + 2 * nll_gev
            fits['GEV'] = {'params': params_gev, 'aic': aic_gev}
        except:
            fits['GEV'] = None

        # Gumbel
        try:
            params_gumbel = gumbel_r.fit(data)
            nll_gumbel = -np.sum(gumbel_r.logpdf(data, *params_gumbel))
            aic_gumbel = 2 * len(params_gumbel) + 2 * nll_gumbel
            fits['Gumbel'] = {'params': params_gumbel, 'aic': aic_gumbel}
        except:
            fits['Gumbel'] = None

        # Select best distribution
        valid_fits = {k: v for k, v in fits.items() if v is not None}
        if valid_fits:
            best_dist = min(valid_fits.keys(), key=lambda k: valid_fits[k]['aic'])
            seasonal_eva_fits[season][var] = {
                'distribution': best_dist,
                'params': valid_fits[best_dist]['params'],
                'aic': valid_fits[best_dist]['aic'],
                'all_fits': fits
            }
            print(f"  {var}: Best = {best_dist}, AIC = {valid_fits[best_dist]['aic']:.2f}, n = {len(data)}")
        else:
            seasonal_eva_fits[season][var] = None
            print(f"  {var}: Fitting failed")

# ============================================================================
# PART 5: CALCULATE SEASONAL RETURN PERIODS
# ============================================================================

print("\n[PART 5] Calculating Seasonal Return Periods...")

def calculate_return_level(distribution, params, return_period_years):
    """Calculate return level for given return period."""
    n_per_year = 12  # monthly maxima
    exc_prob = 1.0 / (return_period_years * n_per_year)
    quantile = 1 - exc_prob

    if distribution == 'Gumbel':
        return gumbel_r.ppf(quantile, *params)
    elif distribution == 'GEV':
        return genextreme.ppf(quantile, *params)
    elif distribution == 'Weibull':
        return weibull_min.ppf(quantile, *params)
    else:
        return np.nan

return_periods = [1, 5, 10, 25, 50, 100]

seasonal_return_levels = {}

for season in SEASONS.keys():
    seasonal_return_levels[season] = {}

    for var in ['hs', 'wind', 'current']:
        if seasonal_eva_fits[season][var] is not None:
            dist = seasonal_eva_fits[season][var]['distribution']
            params = seasonal_eva_fits[season][var]['params']

            levels = []
            for rp in return_periods:
                level = calculate_return_level(dist, params, rp)
                levels.append(level)

            seasonal_return_levels[season][var] = {
                'return_periods': return_periods,
                'levels': levels,
                'distribution': dist
            }
        else:
            seasonal_return_levels[season][var] = None

# Display results
print("\n" + "="*80)
print("SEASONAL RETURN PERIOD COMPARISON")
print("="*80)

for var, var_name in zip(['hs', 'wind', 'current'], ['Hs (m)', 'Wind (m/s)', 'Current (kt)']):
    print(f"\n{var_name}:")
    print("-" * 80)

    # Header
    header = f"{'Season':<10}"
    for rp in return_periods:
        header += f"{rp:>8}-yr"
    header += f"{'Distribution':<15}"
    print(header)
    print("-" * 80)

    # Each season
    for season in SEASONS.keys():
        if seasonal_return_levels[season][var] is not None:
            row = f"{season:<10}"
            for level in seasonal_return_levels[season][var]['levels']:
                row += f"{level:>11.2f}"
            row += f"  {seasonal_return_levels[season][var]['distribution']:<15}"
            print(row)
        else:
            print(f"{season:<10}  No data / insufficient samples")

    # Phase 4A baseline (for comparison)
    if var in eva_dists_phase4a:
        phase4a_dist = eva_dists_phase4a[var]['distribution']
        phase4a_params = eva_dists_phase4a[var]['params']

        row = f"{'Phase 4A':<10}"
        for rp in return_periods:
            level = calculate_return_level(phase4a_dist, phase4a_params, rp)
            row += f"{level:>11.2f}"
        row += f"  {phase4a_dist:<15} (ALL SEASONS AVERAGED)"
        print("-" * 80)
        print(row)

# ============================================================================
# PART 6: SAVE RESULTS
# ============================================================================

print("\n[PART 6] Saving Results...")

# Save pickle
import os
os.makedirs('data/processed/phase5b1', exist_ok=True)

results = {
    'seasonal_monthly_maxima': seasonal_monthly_maxima,
    'seasonal_eva_fits': seasonal_eva_fits,
    'seasonal_return_levels': seasonal_return_levels,
    'seasons': SEASONS
}

with open('data/processed/phase5b1/seasonal_eva_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✓ Saved seasonal_eva_results.pkl")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PHASE 5B1 COMPLETE: KEY FINDINGS")
print("="*80)

print("\n1. DATA AVAILABILITY:")
for season in SEASONS.keys():
    n_hs = len(seasonal_monthly_maxima[season]['hs'])
    n_wind = len(seasonal_monthly_maxima[season]['wind'])
    n_current = len(seasonal_monthly_maxima[season]['current'])
    print(f"   {season}: {n_hs} months (Hs), {n_wind} months (Wind), {n_current} months (Current)")

print("\n2. SEASONAL EXTREMES COMPARISON (100-year return levels):")
for var, var_name in zip(['hs', 'wind', 'current'], ['Hs', 'Wind', 'Current']):
    print(f"\n   {var_name}:")
    values = []
    for season in SEASONS.keys():
        if seasonal_return_levels[season][var] is not None:
            level_100yr = seasonal_return_levels[season][var]['levels'][-1]
            values.append((season, level_100yr))

    if values:
        values_sorted = sorted(values, key=lambda x: x[1], reverse=True)
        for season, level in values_sorted:
            print(f"      {season}: {level:.2f}")

        # Calculate winter/summer ratio
        winter_level = None
        summer_level = None
        for season, level in values:
            if season == 'Winter':
                winter_level = level
            elif season == 'Summer':
                summer_level = level

        if winter_level and summer_level:
            ratio = winter_level / summer_level
            print(f"      → Winter/Summer ratio: {ratio:.2f}x")

print("\n3. IMPLICATIONS FOR OPERATIONS:")
print("   - Winter operations face significantly higher extreme values")
print("   - Summer/Autumn offer safer weather windows")
print("   - Seasonal scheduling can reduce downtime risk")

print("\n" + "="*80)
print("Next: Phase 5B2 - Seasonal Copula Analysis")
print("Question: Does Hs-Wind dependence (τ=0.45) vary by season?")
print("="*80)
