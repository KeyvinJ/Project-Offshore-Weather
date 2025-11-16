"""
Quick analysis of Phase 1 data to identify enhancement opportunities for Phase 2
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the cleaned data
print("=" * 80)
print("PHASE 1 DATA ANALYSIS - Finding Enhancement Opportunities")
print("=" * 80)

df = pd.read_parquet('data/processed/timeseries/UK_NortheastCoast_2015_2025_cleaned.parquet')

print(f"\n📊 Dataset Overview:")
print(f"  Records: {len(df):,}")
print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
print(f"  Duration: {(df['time'].max() - df['time'].min()).days} days")

print(f"\n📋 Available Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\n🔍 Sample Data:")
print(df.head(3))

print(f"\n📈 Basic Statistics:")
print(df.describe())

# Check if we have directional data
print(f"\n🧭 DIRECTIONAL DATA CHECK:")
if 'dir' in df.columns:
    print(f"  ✅ Wave direction available!")
    print(f"     Range: {df['dir'].min():.1f}° to {df['dir'].max():.1f}°")
    print(f"     Mean: {df['dir'].mean():.1f}°")
else:
    print(f"  ❌ Wave direction NOT in cleaned data")

if 'wind_u' in df.columns and 'wind_v' in df.columns:
    print(f"  ✅ Wind components available (can calculate direction)")
    # Calculate wind direction
    wind_dir = np.degrees(np.arctan2(df['wind_v'], df['wind_u'])) % 360
    print(f"     Wind direction range: {wind_dir.min():.1f}° to {wind_dir.max():.1f}°")
else:
    print(f"  ⚠️  Wind direction not directly available")

# Check temporal resolution
print(f"\n⏱️  TEMPORAL ANALYSIS:")
df_sorted = df.sort_values('time')
time_diffs = df_sorted['time'].diff()
print(f"  Time step: {time_diffs.mode()[0]}")
print(f"  Data frequency: 6-hourly = 4 records per day")

# Calculate some advanced metrics
print(f"\n🌊 ADVANCED WAVE METRICS:")

# Wave steepness (Hs/wavelength approximation)
# Wavelength ≈ 1.56 * Tp^2 (deep water approximation)
df['wavelength'] = 1.56 * df['tp']**2
df['steepness'] = df['hs'] / df['wavelength']

print(f"  Wave steepness (Hs/L):")
print(f"    Mean: {df['steepness'].mean():.4f}")
print(f"    Max: {df['steepness'].max():.4f}")
print(f"    Steep waves (>0.04): {(df['steepness'] > 0.04).sum()} occurrences")

# Combined wave-wind correlation
print(f"\n🔗 WAVE-WIND CORRELATION:")
correlation = df['hs'].corr(df['wind_speed'])
print(f"  Hs vs Wind speed correlation: {correlation:.3f}")
print(f"  {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} correlation")

print("\n" + "=" * 80)
print("ENHANCEMENT OPPORTUNITIES FOR PHASE 2")
print("=" * 80)

print("\n✨ What we CAN add to Phase 2:")
opportunities = []

# 1. Directional analysis
if 'dir' in df.columns:
    opportunities.append("1. Directional workability (operations limited by wave direction)")

# 2. Weather windows
opportunities.append("2. Weather window analysis (consecutive workable days)")

# 3. Persistence
opportunities.append("3. Persistence analysis (how long do conditions last?)")

# 4. Steepness criteria
opportunities.append("4. Wave steepness criteria (steep waves = vessel motions)")

# 5. Combined Hs+Tp limits
opportunities.append("5. Combined Hs+Tp operational limits (use scatter diagram)")

# 6. Seasonal detail
opportunities.append("6. Weekly/bi-weekly workability (better than monthly)")

# 7. Year-over-year trends
opportunities.append("7. Year-over-year variability (is workability changing?)")

# 8. Multi-day project planning
opportunities.append("8. Multi-day project windows (need X consecutive days)")

# 9. Time-of-day patterns
opportunities.append("9. Time-of-day patterns (diurnal effects)")

# 10. Extreme event analysis
opportunities.append("10. Extreme event analysis (storm frequency, return periods)")

for opp in opportunities:
    print(f"  {opp}")

print("\n" + "=" * 80)
print("✅ Analysis complete!")
print("=" * 80)
