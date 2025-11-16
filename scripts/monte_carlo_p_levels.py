"""
Monte Carlo Simulation to Derive Site-Specific P-Level Multipliers
Using REAL UK Northeast Coast Weather Data (2015-2025)

No assumptions - pure data-driven analysis!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("🎲 MONTE CARLO SIMULATION - SITE-SPECIFIC P-LEVELS")
print("=" * 80)
print("\nUsing REAL UK Northeast Coast weather data (2015-2025)")
print("No assumptions - just pure statistics!\n")

# ============================================================================
# LOAD REAL DATA
# ============================================================================

df = pd.read_parquet('data/processed/timeseries/UK_NortheastCoast_2015_2025_complete.parquet')

print(f"📊 Loaded {len(df):,} records of real weather data")
print(f"   Period: {df['time'].min()} to {df['time'].max()}")
print(f"   Duration: {(df['time'].max() - df['time'].min()).days} days\n")

# ============================================================================
# DEFINE OPERATIONS TO ANALYZE
# ============================================================================

operations = {
    'Crane Operations (Heavy Lift)': {
        'max_hs': 2.5,
        'max_wind': 15.0,
        'max_current': 1.5,
    },
    'Diving Operations': {
        'max_hs': 1.5,
        'max_wind': 10.0,
        'max_current': 0.75,
    },
    'ROV Operations': {
        'max_hs': 2.5,
        'max_wind': 15.0,
        'max_current': 1.0,
    },
}

# ============================================================================
# CALCULATE WORKABILITY FOR EACH RECORD
# ============================================================================

def calculate_workability_per_record(df, max_hs, max_wind, max_current):
    """
    Check if each 6-hour record is workable
    Returns boolean array
    """
    hs_ok = df['hs'] < max_hs
    wind_ok = df['wind_speed'] < max_wind

    # Handle current data (only available 2022-2025)
    if max_current is not None:
        has_current = df['current_speed_knots_max'].notna()
        current_ok = df['current_speed_knots_max'] < max_current

        # Where we have current data, check all 3 criteria
        # Where we don't have current data, check wave+wind only
        workable = hs_ok & wind_ok & ((~has_current) | current_ok)
    else:
        workable = hs_ok & wind_ok

    return workable.values

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_project_simulation(df, workable_array, work_days, n_simulations=10000):
    """
    Simulate project starting at random dates in the dataset

    Args:
        df: DataFrame with time series
        workable_array: Boolean array - is each 6h period workable?
        work_days: Number of actual work days needed
        n_simulations: Number of Monte Carlo simulations

    Returns:
        Array of project durations (in days)
    """
    print(f"   🎲 Running {n_simulations:,} Monte Carlo simulations...")

    durations = []

    # Total records available
    total_records = len(df)

    # Convert work days to 6-hour periods
    work_periods_needed = work_days * 4  # 4 x 6-hour periods per day

    for sim in range(n_simulations):
        # Random start index (leave enough room for project to complete)
        max_start = total_records - work_periods_needed - 100  # Buffer
        if max_start < 0:
            print(f"   ⚠️ Project too long for dataset! Skipping...")
            continue

        start_idx = np.random.randint(0, max_start)

        # Simulate the project
        periods_completed = 0
        calendar_periods = 0
        current_idx = start_idx

        while periods_completed < work_periods_needed and current_idx < total_records:
            # Is this 6-hour period workable?
            if workable_array[current_idx]:
                periods_completed += 1

            calendar_periods += 1
            current_idx += 1

        # Convert periods to days
        calendar_days = calendar_periods / 4  # 4 periods per day
        durations.append(calendar_days)

    return np.array(durations)

# ============================================================================
# ANALYZE EACH OPERATION
# ============================================================================

all_results = {}

for op_name, limits in operations.items():
    print(f"\n{'='*80}")
    print(f"🚢 {op_name}")
    print(f"{'='*80}")
    print(f"   Limits: Hs<{limits['max_hs']}m, Wind<{limits['max_wind']}m/s, Current<{limits['max_current']}kt")

    # Calculate workability per record
    workable = calculate_workability_per_record(
        df,
        limits['max_hs'],
        limits['max_wind'],
        limits['max_current']
    )

    overall_workability = (workable.sum() / len(workable)) * 100
    print(f"   Overall workability: {overall_workability:.1f}%")

    # Run Monte Carlo for different project sizes
    project_sizes = [10, 30, 60, 90]  # days

    op_results = {}

    for work_days in project_sizes:
        print(f"\n   📊 {work_days}-day project:")

        # Run Monte Carlo
        durations = monte_carlo_project_simulation(df, workable, work_days, n_simulations=10000)

        if len(durations) == 0:
            print(f"      ⚠️ No simulations completed!")
            continue

        # Calculate percentiles
        p10 = np.percentile(durations, 10)
        p50 = np.percentile(durations, 50)
        p80 = np.percentile(durations, 80)
        p90 = np.percentile(durations, 90)
        p95 = np.percentile(durations, 95)

        # Calculate multipliers
        mult_p80 = p80 / p50
        mult_p90 = p90 / p50
        mult_p95 = p95 / p50

        # Theoretical (using workability only)
        theoretical = work_days / (overall_workability / 100)

        print(f"      Theoretical (P50): {theoretical:.0f} days")
        print(f"      Simulated P10: {p10:.0f} days (fast!)")
        print(f"      Simulated P50: {p50:.0f} days (median)")
        print(f"      Simulated P80: {p80:.0f} days (×{mult_p80:.2f})")
        print(f"      Simulated P90: {p90:.0f} days (×{mult_p90:.2f})")
        print(f"      Simulated P95: {p95:.0f} days (×{mult_p95:.2f})")

        op_results[work_days] = {
            'durations': durations,
            'p10': p10,
            'p50': p50,
            'p80': p80,
            'p90': p90,
            'p95': p95,
            'mult_p80': mult_p80,
            'mult_p90': mult_p90,
            'mult_p95': mult_p95,
            'theoretical': theoretical
        }

    all_results[op_name] = {
        'workability': overall_workability,
        'projects': op_results
    }

# ============================================================================
# SUMMARY - SITE-SPECIFIC MULTIPLIERS
# ============================================================================

print("\n" + "=" * 80)
print("📊 SITE-SPECIFIC P-LEVEL MULTIPLIERS FOR UK NORTHEAST COAST")
print("=" * 80)

print("\nDerived from 10,000 Monte Carlo simulations using REAL weather data:")
print("(Not generic industry standards - YOUR site-specific multipliers!)\n")

for op_name, results in all_results.items():
    print(f"\n🚢 {op_name} (Workability: {results['workability']:.1f}%)")
    print("-" * 80)

    # Average multipliers across project sizes
    mult_p80_avg = np.mean([p['mult_p80'] for p in results['projects'].values()])
    mult_p90_avg = np.mean([p['mult_p90'] for p in results['projects'].values()])
    mult_p95_avg = np.mean([p['mult_p95'] for p in results['projects'].values()])

    print(f"   P80 Multiplier: {mult_p80_avg:.3f}  (Industry standard: 1.200)")
    print(f"   P90 Multiplier: {mult_p90_avg:.3f}  (Industry standard: 1.350)")
    print(f"   P95 Multiplier: {mult_p95_avg:.3f}  (Industry standard: 1.500)")

    if mult_p80_avg > 1.25:
        print(f"   ⚠️  Your site has HIGHER variability than industry standard!")
    elif mult_p80_avg < 1.15:
        print(f"   ✅ Your site has LOWER variability than industry standard!")
    else:
        print(f"   ✅ Your site matches industry standard well!")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("📈 GENERATING VISUALIZATIONS...")
print("=" * 80)

# Create output directory
output_dir = Path('data/processed/monte_carlo')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot for each operation
for op_name, results in all_results.items():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Monte Carlo Analysis: {op_name}\\nUK Northeast Coast - 10,000 Simulations per Project Size',
                 fontsize=16, fontweight='bold')

    project_sizes = [10, 30, 60, 90]

    for idx, work_days in enumerate(project_sizes):
        ax = axes[idx // 2, idx % 2]

        if work_days in results['projects']:
            proj_data = results['projects'][work_days]
            durations = proj_data['durations']

            # Histogram
            ax.hist(durations, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

            # Add percentile lines
            ax.axvline(proj_data['p50'], color='blue', linestyle='-', linewidth=2,
                      label=f"P50: {proj_data['p50']:.0f} days (median)")
            ax.axvline(proj_data['p80'], color='orange', linestyle='--', linewidth=2,
                      label=f"P80: {proj_data['p80']:.0f} days (×{proj_data['mult_p80']:.2f})")
            ax.axvline(proj_data['p90'], color='red', linestyle='--', linewidth=2,
                      label=f"P90: {proj_data['p90']:.0f} days (×{proj_data['mult_p90']:.2f})")

            ax.set_xlabel('Project Duration (days)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{work_days}-Day Project\\nWorkability: {results["workability"]:.1f}%',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    safe_name = op_name.replace(' ', '_').replace('(', '').replace(')', '')
    fig_path = output_dir / f'monte_carlo_{safe_name}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {fig_path}")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("💾 SAVING RESULTS...")
print("=" * 80)

# Save summary table
summary_data = []
for op_name, results in all_results.items():
    for work_days, proj_data in results['projects'].items():
        summary_data.append({
            'Operation': op_name,
            'Work_Days': work_days,
            'Workability_%': results['workability'],
            'P50_Days': proj_data['p50'],
            'P80_Days': proj_data['p80'],
            'P90_Days': proj_data['p90'],
            'P95_Days': proj_data['p95'],
            'P80_Multiplier': proj_data['mult_p80'],
            'P90_Multiplier': proj_data['mult_p90'],
            'P95_Multiplier': proj_data['mult_p95'],
            'Theoretical_P50': proj_data['theoretical']
        })

summary_df = pd.DataFrame(summary_data)
summary_path = output_dir / 'monte_carlo_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"   ✅ Summary saved: {summary_path}")

# Save detailed results
results_path = output_dir / 'monte_carlo_results.pkl'
import pickle
with open(results_path, 'wb') as f:
    pickle.dump(all_results, f)
print(f"   ✅ Detailed results saved: {results_path}")

print("\n" + "=" * 80)
print("✅ MONTE CARLO SIMULATION COMPLETE!")
print("=" * 80)

print("\n🎯 Key Findings:")
print("   • Used 10,000 simulations per project size")
print("   • Based on REAL UK Northeast Coast weather (2015-2025)")
print("   • Site-specific P-level multipliers calculated")
print("   • Accounts for actual weather variability at your location")
print("\n💡 Next Steps:")
print("   • Review the generated charts in data/processed/monte_carlo/")
print("   • Compare your multipliers to industry standard (1.20, 1.35)")
print("   • Use YOUR site-specific multipliers for project planning!")
print("   • Update Phase 2 with your custom P-levels if desired")

print("\n" + "=" * 80)
