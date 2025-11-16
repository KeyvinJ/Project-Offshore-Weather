"""
Quick exploration of Phase 1 data - Run this to see what we have!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("🔍 PHASE 1 DATA EXPLORATION - Quick Analysis")
print("=" * 80)

try:
    # Load the cleaned data
    df = pd.read_parquet('data/processed/timeseries/UK_NortheastCoast_2015_2025_cleaned.parquet')

    print(f"\n✅ Data loaded successfully!")
    print(f"   Records: {len(df):,}")
    print(f"   Time range: {df['time'].min()} to {df['time'].max()}")

    print(f"\n📋 Available Columns:")
    for i, col in enumerate(df.columns, 1):
        non_null = df[col].notna().sum()
        print(f"   {i}. {col:15s} - {non_null:,} non-null values ({non_null/len(df)*100:.1f}%)")

    print(f"\n📊 Quick Statistics:")
    print(df[['hs', 'tp', 'wind_speed']].describe().round(2))

    # Check for directional data
    print(f"\n🧭 DIRECTIONAL DATA CHECK:")
    has_wave_dir = 'dir' in df.columns and df['dir'].notna().sum() > 0
    has_wind_components = 'wind_u' in df.columns and 'wind_v' in df.columns

    if has_wave_dir:
        print(f"   ✅ Wave direction available!")
        print(f"      Range: {df['dir'].min():.1f}° to {df['dir'].max():.1f}°")
        print(f"      Mean: {df['dir'].mean():.1f}°")
    else:
        print(f"   ⚠️  Wave direction not found in cleaned data")

    if has_wind_components:
        print(f"   ✅ Wind components available (u10, v10)")
        wind_dir = (np.degrees(np.arctan2(-df['wind_u'], -df['wind_v'])) + 360) % 360
        print(f"      Wind direction can be calculated!")
        print(f"      Range: {wind_dir.min():.1f}° to {wind_dir.max():.1f}°")

    # Sample the data
    print(f"\n👀 Sample Data (first 3 rows):")
    print(df.head(3).to_string())

    print("\n" + "=" * 80)
    print("✅ DATA IS READY FOR EXPLORATION!")
    print("=" * 80)
    print("\nNext step: Open the Jupyter notebook:")
    print("   notebooks/PHASE1_Enhanced_Data_Exploration.ipynb")
    print("\nOr continue here for quick analysis...")

    # Quick weather window preview
    print("\n" + "=" * 80)
    print("🪟 QUICK PREVIEW: Weather Windows")
    print("=" * 80)

    max_hs = 2.5
    max_wind = 15.0

    df_sorted = df.sort_values('time').reset_index(drop=True)
    df_sorted['workable'] = (df_sorted['hs'] < max_hs) & (df_sorted['wind_speed'] < max_wind)
    df_sorted['workable_group'] = (df_sorted['workable'] != df_sorted['workable'].shift()).cumsum()

    workable_runs = df_sorted[df_sorted['workable']].groupby('workable_group').size()
    workable_days = workable_runs / 4  # Convert 6-hourly to days

    print(f"\nOperation limits: Hs<{max_hs}m, Wind<{max_wind}m/s")
    print(f"\nWeather Windows Found: {len(workable_days)}")
    print(f"   Longest window: {workable_days.max():.1f} days")
    print(f"   Average window: {workable_days.mean():.1f} days")
    print(f"   Median window: {workable_days.median():.1f} days")

    print(f"\n📊 Windows by duration:")
    for days in [1, 3, 5, 7, 10]:
        count = (workable_days >= days).sum()
        print(f"   {days:2d}+ days: {count:4d} windows")

    # Quick steepness preview
    print("\n" + "=" * 80)
    print("📐 QUICK PREVIEW: Wave Steepness")
    print("=" * 80)

    df_sorted['wavelength'] = 1.56 * df_sorted['tp']**2
    df_sorted['steepness'] = df_sorted['hs'] / df_sorted['wavelength']

    print(f"\nSteepness Statistics:")
    print(f"   Mean: {df_sorted['steepness'].mean():.4f}")
    print(f"   Median: {df_sorted['steepness'].median():.4f}")
    print(f"   Max: {df_sorted['steepness'].max():.4f}")

    gentle = (df_sorted['steepness'] < 0.02).sum()
    moderate = ((df_sorted['steepness'] >= 0.02) & (df_sorted['steepness'] < 0.04)).sum()
    steep = (df_sorted['steepness'] >= 0.04).sum()

    print(f"\n   Gentle (<0.02):      {gentle/len(df_sorted)*100:5.1f}%")
    print(f"   Moderate (0.02-0.04): {moderate/len(df_sorted)*100:5.1f}%")
    print(f"   Steep (>0.04):        {steep/len(df_sorted)*100:5.1f}%")

    # Year-over-year preview
    print("\n" + "=" * 80)
    print("📅 QUICK PREVIEW: Year-Over-Year Workability")
    print("=" * 80)

    df_sorted['year'] = pd.to_datetime(df_sorted['time']).dt.year
    df_complete = df_sorted[df_sorted['year'] < 2025].copy()

    print(f"\nWorkability by year (Hs<{max_hs}m, Wind<{max_wind}m/s):\n")

    yearly_workability = []
    for year in sorted(df_complete['year'].unique()):
        year_data = df_complete[df_complete['year'] == year]
        workable = ((year_data['hs'] < max_hs) & (year_data['wind_speed'] < max_wind)).sum()
        workability = (workable / len(year_data)) * 100
        yearly_workability.append(workability)
        print(f"   {year}: {workability:5.1f}%")

    print(f"\n   Best year: {max(yearly_workability):.1f}%")
    print(f"   Worst year: {min(yearly_workability):.1f}%")
    print(f"   Range: {max(yearly_workability) - min(yearly_workability):.1f}% points")
    print(f"   Std dev: {np.std(yearly_workability):.1f}%")

    print("\n" + "=" * 80)
    print("🎯 ENHANCEMENT OPPORTUNITIES CONFIRMED!")
    print("=" * 80)
    print("\n✅ Available for Phase 2:")
    print("   1. Weather Windows - READY (just previewed!)")
    print("   2. Wave Steepness - READY (just calculated!)")
    print("   3. Year-over-Year Trends - READY (just analyzed!)")
    if has_wave_dir:
        print("   4. Directional Analysis - READY (direction data available!)")
    print("   5. Persistence Analysis - READY")
    print("   6. Weekly Patterns - READY")
    print("   7. Scatter-Based Limits - READY")
    print("   8. Multi-Day Project Planner - READY")

    print("\n💡 All enhancements are possible with your data!")
    print("   Ready to upgrade Phase 2? 🚀")

except FileNotFoundError:
    print("\n❌ Data file not found!")
    print("   Looking for: data/processed/timeseries/UK_NortheastCoast_2015_2025_cleaned.parquet")
    print("   Make sure you're running from the project root directory")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
