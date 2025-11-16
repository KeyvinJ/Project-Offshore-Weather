"""Quick check of when current data starts/ends"""
import pandas as pd

# Load data
df = pd.read_parquet('data/processed/timeseries/UK_NortheastCoast_2015_2025_complete.parquet')

print("=" * 60)
print("CURRENT DATA TEMPORAL COVERAGE")
print("=" * 60)

print(f"\nTotal dataset: {len(df):,} records")
print(f"Range: {df['time'].min()} to {df['time'].max()}")

# Check current data
current_valid = df['current_velocity_max'].notna()
print(f"\nCurrent data: {current_valid.sum():,} records ({current_valid.sum()/len(df)*100:.1f}%)")

if current_valid.sum() > 0:
    # First valid
    first_idx = df['current_velocity_max'].first_valid_index()
    first_time = df.loc[first_idx, 'time']
    first_val = df.loc[first_idx, 'current_velocity_max']

    # Last valid
    last_idx = df['current_velocity_max'].last_valid_index()
    last_time = df.loc[last_idx, 'time']
    last_val = df.loc[last_idx, 'current_velocity_max']

    print(f"\nFirst valid current:")
    print(f"  Time: {first_time}")
    print(f"  Value: {first_val:.3f} m/s")
    print(f"  Record index: {first_idx}")

    print(f"\nLast valid current:")
    print(f"  Time: {last_time}")
    print(f"  Value: {last_val:.3f} m/s")
    print(f"  Record index: {last_idx}")

    # Duration
    duration = last_time - first_time
    print(f"\nCurrent data duration: {duration.days} days")
    print(f"Gap at start: {(first_time - df['time'].min()).days} days")
    print(f"Gap at end: {(df['time'].max() - last_time).days} days")

print("\n" + "=" * 60)
