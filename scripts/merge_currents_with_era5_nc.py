"""
Merge ocean current data with already-merged ERA5 wave/wind NetCDF files
Adds current data to existing era5_UK_NortheastCoast_YYYY_merged.nc files
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔄 MERGING CURRENT DATA WITH ERA5 NETCDF FILES")
print("=" * 80)

# Get the project root directory (parent of scripts folder)
script_dir = Path(__file__).parent
project_dir = script_dir.parent

print(f"\n📁 Project directory: {project_dir}")

# ============================================================================
# STEP 1: Load all ocean current data (hourly)
# ============================================================================

print("\n📂 Loading ocean current data...")

current_dir = project_dir / 'data/raw/currents'

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
# STEP 2: Downsample currents from HOURLY to 6-HOURLY
# ============================================================================

print("\n🔄 Downsampling currents from hourly to 6-hourly...")
print("   Method: Preserving BOTH mean AND max velocity")

# Ensure time is datetime and remove timezone
df_current['time'] = pd.to_datetime(df_current['time']).dt.tz_localize(None)

# Set time as index for resampling
df_current_indexed = df_current.set_index('time')

# Resample to 6-hourly with BOTH mean and max for velocity
df_current_6h = df_current_indexed.resample('6h').agg({
    'current_velocity': ['mean', 'max'],
    'current_u': 'mean',
    'current_v': 'mean'
})

# Flatten multi-level columns
df_current_6h.columns = ['_'.join(col).strip('_') for col in df_current_6h.columns.values]

# Rename for clarity
df_current_6h = df_current_6h.rename(columns={
    'current_u_mean': 'current_u',
    'current_v_mean': 'current_v'
})

# Recalculate direction from averaged U/V components
df_current_6h['current_direction'] = (np.degrees(np.arctan2(
    -df_current_6h['current_u'],
    -df_current_6h['current_v']
)) + 360) % 360

# Reset index
df_current_6h = df_current_6h.reset_index()

print(f"✅ Downsampled to {len(df_current_6h):,} records (6-hourly)")
print(f"   Columns: {df_current_6h.columns.tolist()}")

# ============================================================================
# STEP 3: Process each ERA5 NetCDF file year by year
# ============================================================================

print("\n📁 Processing ERA5 NetCDF files...")

era5_dir = project_dir / 'data/raw/era5'
output_dir = project_dir / 'data/processed/era5_with_currents'
output_dir.mkdir(parents=True, exist_ok=True)

# Find all merged ERA5 files
era5_files = sorted(era5_dir.glob('era5_UK_NortheastCoast_*_merged.nc'))

if len(era5_files) == 0:
    print(f"❌ ERROR: No ERA5 merged files found in {era5_dir}")
    exit(1)

print(f"\nFound {len(era5_files)} ERA5 files to process:")

successful_merges = []
failed_merges = []

for nc_file in era5_files:
    year = nc_file.stem.split('_')[-2]  # Extract year from filename

    print(f"\n{'='*80}")
    print(f"📥 Processing {year}: {nc_file.name}")
    print(f"{'='*80}")

    try:
        # Load ERA5 NetCDF file
        print(f"  Loading ERA5 data...")
        ds_era5 = xr.open_dataset(nc_file)

        print(f"  Variables in file: {list(ds_era5.data_vars)}")
        print(f"  Coordinates in file: {list(ds_era5.coords)}")
        print(f"  Dimensions in file: {dict(ds_era5.dims)}")

        # Detect time coordinate (could be 'time', 'valid_time', 'datetime', etc.)
        time_coord = None
        for possible_time in ['time', 'valid_time', 'datetime', 't']:
            if possible_time in ds_era5.coords:
                time_coord = possible_time
                break

        if time_coord is None:
            # Check dimensions
            for possible_time in ['time', 'valid_time', 'datetime', 't']:
                if possible_time in ds_era5.dims:
                    time_coord = possible_time
                    break

        if time_coord is None:
            print(f"  ❌ ERROR: Cannot find time coordinate!")
            print(f"     Available coords: {list(ds_era5.coords)}")
            print(f"     Available dims: {list(ds_era5.dims)}")
            failed_merges.append(year)
            continue

        print(f"  Time coordinate: '{time_coord}'")
        print(f"  Time range: {pd.to_datetime(ds_era5[time_coord].values[0])} to {pd.to_datetime(ds_era5[time_coord].values[-1])}")
        print(f"  Records: {len(ds_era5[time_coord]):,}")

        # Convert to DataFrame for easier merging
        df_era5 = ds_era5.to_dataframe().reset_index()

        # Rename time coordinate to 'time' for consistency
        if time_coord != 'time':
            df_era5 = df_era5.rename(columns={time_coord: 'time'})

        # Get time range for this year
        year_start = pd.Timestamp(f'{year}-01-01')
        year_end = pd.Timestamp(f'{year}-12-31 23:59:59')

        # Filter current data for this year
        df_current_year = df_current_6h[
            (df_current_6h['time'] >= year_start) &
            (df_current_6h['time'] <= year_end)
        ].copy()

        print(f"\n  Current data for {year}: {len(df_current_year):,} records")

        if len(df_current_year) == 0:
            print(f"  ⚠️  No current data for {year}, skipping...")
            continue

        # Ensure time columns are compatible
        df_era5['time'] = pd.to_datetime(df_era5['time'])

        # Merge current data with ERA5 data
        print(f"  Merging current data with ERA5...")

        # Since ERA5 has multiple lat/lon points, we need to handle this carefully
        # Current data is single point, so we'll add it as new variables

        # First, get unique times from ERA5
        era5_times = df_era5[['time']].drop_duplicates()

        # Merge current data with times
        df_merged_times = era5_times.merge(
            df_current_year[['time', 'current_velocity_mean', 'current_velocity_max',
                           'current_direction', 'current_u', 'current_v']],
            on='time',
            how='left'
        )

        # Now merge back with full ERA5 data
        df_complete = df_era5.merge(df_merged_times, on='time', how='left')

        print(f"  ✅ Merged dataset: {len(df_complete):,} records")

        # Check merge quality
        currents_available = df_complete['current_velocity_mean'].notna().sum()
        print(f"  Records with currents: {currents_available:,} ({currents_available/len(df_complete)*100:.1f}%)")

        # Convert back to xarray Dataset
        print(f"  Converting back to NetCDF format...")

        # Set the index back to the original dimensions
        if 'latitude' in df_complete.columns and 'longitude' in df_complete.columns:
            # Multi-point grid data
            df_complete = df_complete.set_index(['time', 'latitude', 'longitude'])
        else:
            # Single point data
            df_complete = df_complete.set_index('time')

        # Convert to xarray
        ds_complete = df_complete.to_xarray()

        # Add metadata for new variables
        if 'current_velocity_mean' in ds_complete:
            ds_complete['current_velocity_mean'].attrs = {
                'long_name': 'Ocean current velocity (6-hour mean)',
                'units': 'm/s',
                'source': 'ERA5-Ocean, downsampled from hourly to 6-hourly'
            }

        if 'current_velocity_max' in ds_complete:
            ds_complete['current_velocity_max'].attrs = {
                'long_name': 'Ocean current velocity (6-hour maximum)',
                'units': 'm/s',
                'source': 'ERA5-Ocean, downsampled from hourly to 6-hourly',
                'note': 'Use this for safety-critical workability assessments'
            }

        if 'current_direction' in ds_complete:
            ds_complete['current_direction'].attrs = {
                'long_name': 'Ocean current direction (from)',
                'units': 'degrees',
                'convention': 'meteorological (direction FROM which current flows)'
            }

        # Save merged dataset
        output_file = output_dir / f'era5_UK_NortheastCoast_{year}_with_currents.nc'

        print(f"  Saving to: {output_file.name}")
        ds_complete.to_netcdf(output_file)

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ Saved: {file_size_mb:.2f} MB")

        # Also save as parquet for easier analysis
        parquet_file = output_dir / f'era5_UK_NortheastCoast_{year}_with_currents.parquet'
        df_complete.reset_index().to_parquet(parquet_file, index=False)
        print(f"  ✅ Also saved as parquet: {parquet_file.name}")

        successful_merges.append(year)

        # Clean up
        ds_era5.close()

    except Exception as e:
        print(f"\n  ❌ ERROR processing {year}:")
        print(f"     {str(e)}")
        failed_merges.append(year)
        continue

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n\n{'='*80}")
print(f"📊 MERGE SUMMARY")
print(f"{'='*80}")

print(f"\n✅ Successfully merged: {len(successful_merges)} years")
for year in successful_merges:
    nc_file = output_dir / f'era5_UK_NortheastCoast_{year}_with_currents.nc'
    pq_file = output_dir / f'era5_UK_NortheastCoast_{year}_with_currents.parquet'
    nc_size = nc_file.stat().st_size / (1024 * 1024)
    pq_size = pq_file.stat().st_size / (1024 * 1024)
    print(f"  • {year}: {nc_size:.2f} MB (NetCDF) + {pq_size:.2f} MB (Parquet)")

if failed_merges:
    print(f"\n❌ Failed merges: {len(failed_merges)} years")
    for year in failed_merges:
        print(f"  • {year}")

print(f"\n📁 Output location: {output_dir}")

print(f"\n{'='*80}")
print(f"✅ MERGE COMPLETE!")
print(f"{'='*80}")

print(f"\nNew variables added to each file:")
print(f"  • current_velocity_mean (m/s) - 6-hour average")
print(f"  • current_velocity_max (m/s) - 6-hour maximum ⚠️  USE FOR SAFETY!")
print(f"  • current_direction (degrees) - direction FROM which current flows")
print(f"  • current_u (m/s) - eastward component")
print(f"  • current_v (m/s) - northward component")

print(f"\n💡 Files saved in TWO formats:")
print(f"  • NetCDF (.nc) - for spatial analysis and archiving")
print(f"  • Parquet (.parquet) - for fast data analysis in Python/pandas")

print(f"\n🎯 Ready for workability analysis!")
