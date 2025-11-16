"""
Merge hourly ocean current data with 6-hourly wave/wind data
Creates a complete metocean dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("🔄 MERGING CURRENT DATA WITH WAVE/WIND DATA")
print("=" * 80)

# ============================================================================
# STEP 1: Load existing wave/wind data (6-hourly)
# ============================================================================

print("\n📂 Loading existing wave/wind data...")

wave_wind_file = Path('data/processed/timeseries/UK_NortheastCoast_2015_2025_cleaned.parquet')

if not wave_wind_file.exists():
    print(f"❌ ERROR: Wave/wind data not found!")
    print(f"   Expected: {wave_wind_file}")
    print(f"   Run Phase 1 notebook first!")
    exit(1)

df_metocean = pd.read_parquet(wave_wind_file)
print(f"✅ Loaded {len(df_metocean):,} records (6-hourly)")
print(f"   Time range: {df_metocean['time'].min()} to {df_metocean['time'].max()}")
print(f"   Columns: {df_metocean.columns.tolist()}")

# ============================================================================
# STEP 2: Load and combine all current files (hourly)
# ============================================================================

print("\n📂 Loading ocean current data...")

current_dir = Path('data/raw/currents')

if not current_dir.exists():
    print(f"❌ ERROR: Current data directory not found!")
    print(f"   Expected: {current_dir}")
    print(f"   Run download_currents_10_years.py first!")
    exit(1)

current_files = sorted(current_dir.glob('ocean_currents_*.parquet'))

if len(current_files) == 0:
    print(f"❌ ERROR: No current data files found!")
    print(f"   Run download_currents_10_years.py first!")
    exit(1)

print(f"Found {len(current_files)} current data files:")

# Load and combine all years
current_dfs = []
for file in current_files:
    year = file.stem.split('_')[-1]
    df_year = pd.read_parquet(file)
    print(f"  • {year}: {len(df_year):,} records")
    current_dfs.append(df_year)

df_current = pd.concat(current_dfs, ignore_index=True)
df_current = df_current.sort_values('time').reset_index(drop=True)

print(f"\n✅ Combined: {len(df_current):,} hourly records")
print(f"   Time range: {df_current['time'].min()} to {df_current['time'].max()}")

# ============================================================================
# STEP 3: Downsample currents from HOURLY to 6-HOURLY
# ============================================================================

print("\n🔄 Downsampling currents from hourly to 6-hourly...")
print("   Method: SMART downsampling - preserving BOTH mean AND max!")
print("   → Mean: For statistics and reporting")
print("   → Max: For safety-critical workability limits ⚠️")

# Ensure time is datetime
df_current['time'] = pd.to_datetime(df_current['time'])

# Set time as index for resampling
df_current_indexed = df_current.set_index('time')

# Resample to 6-hourly with BOTH mean and max for velocity
# Use '6h' frequency starting at midnight (lowercase 'h' is the new standard)
df_current_6h = df_current_indexed.resample('6h').agg({
    'current_velocity': ['mean', 'max'],  # BOTH! Mean for stats, max for safety
    'current_u': 'mean',
    'current_v': 'mean'
})

# Flatten multi-level columns properly
# After aggregation: ('current_velocity', 'mean'), ('current_velocity', 'max'), ('current_u', 'mean'), etc.
df_current_6h.columns = ['_'.join(col).strip('_') for col in df_current_6h.columns.values]

# Now columns are: 'current_velocity_mean', 'current_velocity_max', 'current_u_mean', 'current_v_mean'
# Rename to simpler names
df_current_6h = df_current_6h.rename(columns={
    'current_u_mean': 'current_u',
    'current_v_mean': 'current_v'
})

# Recalculate direction from averaged U/V components (more accurate than averaging angles)
df_current_6h['current_direction'] = (np.degrees(np.arctan2(
    -df_current_6h['current_u'],
    -df_current_6h['current_v']
)) + 360) % 360

# Reset index
df_current_6h = df_current_6h.reset_index()

print(f"✅ Downsampled to {len(df_current_6h):,} records (6-hourly)")
print(f"   Time range: {df_current_6h['time'].min()} to {df_current_6h['time'].max()}")

# Data quality check
valid_before = df_current['current_velocity'].notna().sum()
valid_after_mean = df_current_6h['current_velocity_mean'].notna().sum()
valid_after_max = df_current_6h['current_velocity_max'].notna().sum()

print(f"\n📊 Downsampling quality:")
print(f"   Hourly records: {len(df_current):,}")
print(f"   6-hourly records: {len(df_current_6h):,}")
print(f"   Reduction factor: {len(df_current)/len(df_current_6h):.1f}x")
print(f"   Valid data retained: {valid_after_mean:,} records")

# Statistics comparison - THIS IS THE KEY PART!
print(f"\n🌊 Current statistics (hourly → 6-hourly):")
print(f"   Mean velocity:")
print(f"     Hourly: {df_current['current_velocity'].mean():.3f} m/s")
print(f"     6-hourly (mean): {df_current_6h['current_velocity_mean'].mean():.3f} m/s")
print(f"     ✅ Difference: {abs(df_current['current_velocity'].mean() - df_current_6h['current_velocity_mean'].mean()):.3f} m/s (minimal!)")

print(f"\n   Max velocity (CRITICAL FOR SAFETY!):")
print(f"     Hourly max: {df_current['current_velocity'].max():.3f} m/s ({df_current['current_velocity'].max()*1.94384:.2f} knots)")
print(f"     6-hourly max: {df_current_6h['current_velocity_max'].max():.3f} m/s ({df_current_6h['current_velocity_max'].max()*1.94384:.2f} knots)")
print(f"     ✅ Peak PRESERVED! Difference: {abs(df_current['current_velocity'].max() - df_current_6h['current_velocity_max'].max()):.3f} m/s")

print(f"\n   💡 Using MEAN would have underestimated max by: {df_current['current_velocity'].max() - df_current_6h['current_velocity_mean'].mean():.3f} m/s")
print(f"      That's {((df_current['current_velocity'].max() - df_current_6h['current_velocity_mean'].mean())/df_current['current_velocity'].max())*100:.1f}% error - DANGEROUS! ⚠️")

# ============================================================================
# STEP 4: Merge with wave/wind data
# ============================================================================

print("\n🔗 Merging current data with wave/wind data...")

# Ensure both have datetime
df_metocean['time'] = pd.to_datetime(df_metocean['time'])
df_current_6h['time'] = pd.to_datetime(df_current_6h['time'])

# Fix timezone mismatch: convert current data to timezone-naive (remove UTC)
# This matches the wave/wind data format
df_current_6h['time'] = df_current_6h['time'].dt.tz_localize(None)

print(f"   Timezone alignment: current data converted to match wave/wind data")

# Merge on time - include BOTH mean and max velocity!
df_complete = df_metocean.merge(
    df_current_6h[['time', 'current_velocity_mean', 'current_velocity_max',
                   'current_direction', 'current_u', 'current_v']],
    on='time',
    how='left'  # Keep all wave/wind records, add currents where available
)

print(f"✅ Merged dataset:")
print(f"   Total records: {len(df_complete):,}")
print(f"   Time range: {df_complete['time'].min()} to {df_complete['time'].max()}")

# Check merge quality
currents_available = df_complete['current_velocity_mean'].notna().sum()
currents_missing = df_complete['current_velocity_mean'].isna().sum()

print(f"\n📊 Merge quality:")
print(f"   Records with currents: {currents_available:,} ({currents_available/len(df_complete)*100:.1f}%)")
print(f"   Records without currents: {currents_missing:,} ({currents_missing/len(df_complete)*100:.1f}%)")

if currents_missing > 0:
    print(f"   Note: Some wave/wind records don't have matching current data")
    print(f"         (This is normal if time ranges don't fully overlap)")

# ============================================================================
# STEP 5: Calculate current speed in knots (for reference)
# ============================================================================

print("\n📏 Adding current speed in knots...")

# Convert BOTH mean and max to knots
df_complete['current_speed_knots_mean'] = df_complete['current_velocity_mean'] * 1.94384
df_complete['current_speed_knots_max'] = df_complete['current_velocity_max'] * 1.94384

# ============================================================================
# STEP 6: Summary statistics
# ============================================================================

print("\n" + "=" * 80)
print("📊 COMPLETE METOCEAN DATASET SUMMARY")
print("=" * 80)

print(f"\n📋 Available variables:")
for col in df_complete.columns:
    non_null = df_complete[col].notna().sum()
    print(f"  • {col:25s}: {non_null:,} records ({non_null/len(df_complete)*100:.1f}%)")

print(f"\n🌊 Ocean current statistics:")
if currents_available > 0:
    print(f"\n  📊 MEAN velocity (for statistics):")
    print(f"    Mean: {df_complete['current_velocity_mean'].mean():.3f} m/s ({df_complete['current_speed_knots_mean'].mean():.2f} knots)")
    print(f"    Median: {df_complete['current_velocity_mean'].median():.3f} m/s ({df_complete['current_speed_knots_mean'].median():.2f} knots)")
    print(f"    95th percentile: {df_complete['current_velocity_mean'].quantile(0.95):.3f} m/s ({df_complete['current_speed_knots_mean'].quantile(0.95):.2f} knots)")

    print(f"\n  ⚠️  MAX velocity (for workability limits - USE THIS!):")
    print(f"    Max of all maxes: {df_complete['current_velocity_max'].max():.3f} m/s ({df_complete['current_speed_knots_max'].max():.2f} knots)")
    print(f"    Mean of maxes: {df_complete['current_velocity_max'].mean():.3f} m/s ({df_complete['current_speed_knots_max'].mean():.2f} knots)")
    print(f"    95th %ile of maxes: {df_complete['current_velocity_max'].quantile(0.95):.3f} m/s ({df_complete['current_speed_knots_max'].quantile(0.95):.2f} knots)")

    # Classify by current strength (using MAX for safety!)
    weak = (df_complete['current_speed_knots_max'] < 0.5).sum()
    moderate = ((df_complete['current_speed_knots_max'] >= 0.5) & (df_complete['current_speed_knots_max'] < 1.5)).sum()
    strong = (df_complete['current_speed_knots_max'] >= 1.5).sum()

    print(f"\n  Current strength classification (using MAX - conservative):")
    print(f"    Weak (<0.5 knots):     {weak:,} ({weak/currents_available*100:.1f}%)")
    print(f"    Moderate (0.5-1.5 kt): {moderate:,} ({moderate/currents_available*100:.1f}%)")
    print(f"    Strong (>1.5 knots):   {strong:,} ({strong/currents_available*100:.1f}%)")

# ============================================================================
# STEP 7: Save complete dataset
# ============================================================================

print("\n💾 Saving complete metocean dataset...")

output_file = Path('data/processed/timeseries/UK_NortheastCoast_2015_2025_complete.parquet')
df_complete.to_parquet(output_file, index=False)

file_size_mb = output_file.stat().st_size / (1024 * 1024)

print(f"✅ Saved: {output_file}")
print(f"   Size: {file_size_mb:.2f} MB")
print(f"   Records: {len(df_complete):,}")

# Also save as CSV for easy viewing
csv_file = output_file.with_suffix('.csv')
print(f"\n💾 Saving CSV version (for Excel)...")
df_complete.head(1000).to_csv(csv_file, index=False)  # First 1000 records only
print(f"✅ Saved sample: {csv_file}")

print("\n" + "=" * 80)
print("✅ MERGE COMPLETE!")
print("=" * 80)

print(f"\nYou now have a COMPLETE metocean dataset with:")
print(f"  ✅ Wave data (Hs, Tp, direction)")
print(f"  ✅ Wind data (speed, direction)")
print(f"  ✅ Ocean current data - SMART downsampling!")
print(f"     → current_velocity_mean & current_speed_knots_mean (for statistics)")
print(f"     → current_velocity_max & current_speed_knots_max (for safety limits! ⚠️)")
print(f"  ✅ All aligned to 6-hourly resolution")
print(f"  ✅ {len(df_complete):,} records from 2015-2025")

print(f"\n🎯 Ready for Phase 2 with current limits!")
print(f"   File: {output_file}")

print(f"\n⚠️  IMPORTANT - Which current column to use:")
print(f"   • For workability limits → USE current_speed_knots_MAX ✅")
print(f"   • For statistics/reporting → USE current_speed_knots_MEAN 📊")
print(f"   • WHY: Peaks matter for safety! Mean smooths them out.")

print(f"\n💡 Typical current limits for operations:")
print(f"   • Diving operations: <0.5-1.0 knots (USE MAX!)")
print(f"   • ROV operations: <1.0-1.5 knots")
print(f"   • Cable laying: <1.0-1.5 knots")
print(f"   • DP positioning: <2.0 knots")
