"""Quick check to understand current data NaN coverage"""
import pandas as pd

df = pd.read_parquet('data/processed/timeseries/UK_NortheastCoast_2015_2025_complete.parquet')

print("=" * 80)
print("CURRENT DATA TEMPORAL COVERAGE ANALYSIS")
print("=" * 80)

# Overall dataset
print(f"\n📊 Complete Dataset:")
print(f"  Total records: {len(df):,}")
print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
print(f"  Duration: {(df['time'].max() - df['time'].min()).days} days")

# Current data coverage
has_current = df['current_velocity_max'].notna()
print(f"\n🌊 Current Data:")
print(f"  Records with current: {has_current.sum():,} ({has_current.sum()/len(df)*100:.1f}%)")
print(f"  Records WITHOUT current: {(~has_current).sum():,} ({(~has_current).sum()/len(df)*100:.1f}%)")

# Find where current data starts and ends
if has_current.sum() > 0:
    first_idx = df['current_velocity_max'].first_valid_index()
    last_idx = df['current_velocity_max'].last_valid_index()

    first_time = df.loc[first_idx, 'time']
    last_time = df.loc[last_idx, 'time']

    print(f"\n📅 Current Data Time Range:")
    print(f"  First valid: {first_time} (record #{first_idx})")
    print(f"  Last valid: {last_time} (record #{last_idx})")
    print(f"  Current duration: {(last_time - first_time).days} days")

    # Calculate gaps
    gap_start = (first_time - df['time'].min()).days
    gap_end = (df['time'].max() - last_time).days

    print(f"\n⚠️ Data Gaps:")
    print(f"  Missing at START: {gap_start} days ({df['time'].min()} to {first_time})")
    print(f"  Missing at END: {gap_end} days ({last_time} to {df['time'].max()})")

    # Show where NaNs are
    print(f"\n📍 NaN Distribution:")
    # Check first 100 records
    first_100_nan = df.head(100)['current_velocity_max'].isna().sum()
    print(f"  First 100 records: {first_100_nan} are NaN")

    # Check last 100 records
    last_100_nan = df.tail(100)['current_velocity_max'].isna().sum()
    print(f"  Last 100 records: {last_100_nan} are NaN")

    # Check middle records
    middle_start = len(df)//2 - 50
    middle_end = len(df)//2 + 50
    middle_nan = df.iloc[middle_start:middle_end]['current_velocity_max'].isna().sum()
    print(f"  Middle 100 records: {middle_nan} are NaN")

print("\n" + "=" * 80)
print("🤔 WHY ARE THERE NaNs?")
print("=" * 80)
print("\nLikely reasons:")
print("  1. Open-Meteo API doesn't have data for full 2015-2025 period")
print("  2. Data might start from a later date (e.g., 2020 onwards)")
print("  3. Data might end before 2025-11-08")
print("\nLet's check the raw downloaded files...")

# Check raw files
import os
from pathlib import Path

raw_dir = Path('data/raw/currents')
if raw_dir.exists():
    files = sorted(raw_dir.glob('*.parquet'))
    print(f"\n📁 Raw current files found: {len(files)}")

    for file in files:
        df_year = pd.read_parquet(file)
        if len(df_year) > 0:
            print(f"  {file.name}: {len(df_year):,} records, {df_year['time'].min()} to {df_year['time'].max()}")

print("\n" + "=" * 80)
