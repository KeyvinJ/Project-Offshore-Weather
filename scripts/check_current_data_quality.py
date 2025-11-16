"""
Quick check of current data quality in merged dataset
"""

import pandas as pd

print("=" * 80)
print("CHECKING CURRENT DATA QUALITY")
print("=" * 80)

# Load complete dataset
df = pd.read_parquet('data/processed/timeseries/UK_NortheastCoast_2015_2025_complete.parquet')

print(f"\nTotal records: {len(df):,}")
print(f"Time range: {df['time'].min()} to {df['time'].max()}")

print(f"\n📊 Current Data Availability:")
print(f"\nColumn-by-column NaN check:")

for col in df.columns:
    total = len(df)
    non_null = df[col].notna().sum()
    null = df[col].isna().sum()
    print(f"  {col:35s}: {non_null:6,} valid ({non_null/total*100:5.1f}%) | {null:6,} NaN ({null/total*100:5.1f}%)")

# Check current-specific columns
if 'current_velocity_mean' in df.columns:
    print(f"\n🌊 Current Data Details:")

    current_cols = ['current_velocity_mean', 'current_velocity_max',
                   'current_speed_knots_mean', 'current_speed_knots_max']

    for col in current_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                print(f"\n  {col}:")
                print(f"    Valid: {non_null:,}")
                print(f"    Mean: {df[col].mean():.3f}")
                print(f"    Max: {df[col].max():.3f}")
                print(f"    Min: {df[col].min():.3f}")
            else:
                print(f"\n  {col}: ALL NaN!")

    # Check where currents start
    print(f"\n📍 Where does current data start?")
    first_valid_idx = df['current_velocity_max'].first_valid_index()
    if first_valid_idx is not None:
        first_valid = df.loc[first_valid_idx]
        print(f"  First valid current: Index {first_valid_idx}")
        print(f"  Time: {first_valid['time']}")
        print(f"  Current: {first_valid['current_velocity_max']:.3f} m/s")

    # Check where currents end
    last_valid_idx = df['current_velocity_max'].last_valid_index()
    if last_valid_idx is not None:
        last_valid = df.loc[last_valid_idx]
        print(f"\n  Last valid current: Index {last_valid_idx}")
        print(f"  Time: {last_valid['time']}")
        print(f"  Current: {last_valid['current_velocity_max']:.3f} m/s")

    # Count consecutive NaN periods
    print(f"\n🔍 NaN Analysis:")
    is_null = df['current_velocity_max'].isna()
    null_runs = (is_null != is_null.shift()).cumsum()
    null_periods = df[is_null].groupby(null_runs).size()

    if len(null_periods) > 0:
        print(f"  Number of NaN periods: {len(null_periods)}")
        print(f"  Largest NaN gap: {null_periods.max()} records")
        print(f"  Total NaN records: {is_null.sum():,}")
    else:
        print(f"  ✅ No NaN periods!")

else:
    print("\n❌ No current columns found!")
    print("   Available columns:", df.columns.tolist())

print("\n" + "=" * 80)
