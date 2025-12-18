"""
Replace old current data (2022-2025 only) with new ERA5-Ocean data (2015-2025)
Updates existing ERA5 merged NetCDF files in place
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔄 UPDATING CURRENT DATA IN ERA5 FILES")
print("=" * 80)
print("\nReplacing old current data (MeteoFrance 2022-2025)")
print("with new ERA5-Ocean data (2015-2025)\n")

# Get the project root directory
script_dir = Path(__file__).parent
project_dir = script_dir.parent

print(f"📁 Project directory: {project_dir}")

# ============================================================================
# STEP 1: Load all ocean current data (hourly)
# ============================================================================

print("\n📂 Loading NEW ocean current data (ERA5-Ocean)...")

current_dir = project_dir / 'data/raw/currents'

if not current_dir.exists():
    print(f"❌ ERROR: Current data directory not found!")
    print(f"   Expected: {current_dir}")
    exit(1)

current_files = sorted(current_dir.glob('ocean_currents_*.parquet'))

if len(current_files) == 0:
    print(f"❌ ERROR: No current data files found!")
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
print("   Preserving BOTH mean AND max velocity")

# Ensure time is datetime and remove timezone to match ERA5
df_current['time'] = pd.to_datetime(df_current['time']).dt.tz_localize(None)

# Set time as index for resampling
df_current_indexed = df_current.set_index('time')

# Resample to 6-hourly
df_current_6h = df_current_indexed.resample('6h').agg({
    'current_velocity': ['mean', 'max'],
    'current_u': 'mean',
    'current_v': 'mean'
})

# Flatten columns
df_current_6h.columns = ['_'.join(col).strip('_') for col in df_current_6h.columns.values]

# Rename
df_current_6h = df_current_6h.rename(columns={
    'current_u_mean': 'current_u',
    'current_v_mean': 'current_v'
})

# Recalculate direction
df_current_6h['current_direction'] = (np.degrees(np.arctan2(
    -df_current_6h['current_u'],
    -df_current_6h['current_v']
)) + 360) % 360

# Reset index
df_current_6h = df_current_6h.reset_index()

print(f"✅ Downsampled to {len(df_current_6h):,} records (6-hourly)")

# ============================================================================
# STEP 3: Update each ERA5 NetCDF file
# ============================================================================

print("\n📁 Updating ERA5 NetCDF files...")

era5_dir = project_dir / 'data/raw/era5'
backup_dir = project_dir / 'data/raw/era5_backup_before_current_update'

# Create backup directory
backup_dir.mkdir(parents=True, exist_ok=True)

# Find all merged ERA5 files
era5_files = sorted(era5_dir.glob('era5_UK_NortheastCoast_*_merged.nc'))

if len(era5_files) == 0:
    print(f"❌ ERROR: No ERA5 merged files found in {era5_dir}")
    exit(1)

print(f"\nFound {len(era5_files)} ERA5 files to update:")
print(f"📦 Backups will be saved to: {backup_dir}")

successful_updates = []
failed_updates = []

for nc_file in era5_files:
    year = nc_file.stem.split('_')[-2]

    print(f"\n{'='*80}")
    print(f"📝 Updating {year}: {nc_file.name}")
    print(f"{'='*80}")

    try:
        # Create backup first
        backup_file = backup_dir / nc_file.name
        print(f"  💾 Creating backup: {backup_file.name}")
        shutil.copy2(nc_file, backup_file)

        # Load ERA5 NetCDF file
        print(f"  📂 Loading ERA5 file...")
        ds_era5 = xr.open_dataset(nc_file)

        print(f"  Current variables: {list(ds_era5.data_vars)}")

        # Check for existing current variables
        current_vars = [v for v in ds_era5.data_vars
                       if 'current' in v.lower() or v in ['current_velocity_mean', 'current_velocity_max',
                                                           'current_direction', 'current_u', 'current_v',
                                                           'current_speed_knots_mean', 'current_speed_knots_max']]

        if current_vars:
            print(f"  🗑️  Removing old current variables: {current_vars}")
            ds_era5 = ds_era5.drop_vars(current_vars)
        else:
            print(f"  ℹ️  No existing current variables found")

        # Get time range for this year
        year_start = pd.Timestamp(f'{year}-01-01')
        year_end = pd.Timestamp(f'{year}-12-31 23:59:59')

        # Filter current data for this year
        df_current_year = df_current_6h[
            (df_current_6h['time'] >= year_start) &
            (df_current_6h['time'] <= year_end)
        ].copy()

        print(f"  📊 New current data for {year}: {len(df_current_year):,} records")

        if len(df_current_year) == 0:
            print(f"  ⚠️  No current data available for {year}")
            print(f"     Saving file without current data...")
            ds_era5.to_netcdf(nc_file)
            ds_era5.close()
            successful_updates.append((year, "no_current_data"))
            continue

        # Create xarray dataset for current data
        # Match the time coordinate from ERA5
        ds_current = xr.Dataset(
            {
                'current_velocity_mean': ('time', df_current_year['current_velocity_mean'].values),
                'current_velocity_max': ('time', df_current_year['current_velocity_max'].values),
                'current_direction': ('time', df_current_year['current_direction'].values),
                'current_u': ('time', df_current_year['current_u'].values),
                'current_v': ('time', df_current_year['current_v'].values),
            },
            coords={'time': df_current_year['time'].values}
        )

        # Add metadata
        ds_current['current_velocity_mean'].attrs = {
            'long_name': 'Ocean current velocity (6-hour mean)',
            'units': 'm/s',
            'source': 'ERA5-Ocean',
            'note': 'Downsampled from hourly to 6-hourly'
        }
        ds_current['current_velocity_max'].attrs = {
            'long_name': 'Ocean current velocity (6-hour maximum)',
            'units': 'm/s',
            'source': 'ERA5-Ocean',
            'note': 'Use for safety-critical workability limits'
        }
        ds_current['current_direction'].attrs = {
            'long_name': 'Ocean current direction',
            'units': 'degrees',
            'convention': 'Direction FROM which current flows'
        }
        ds_current['current_u'].attrs = {
            'long_name': 'Eastward current velocity component',
            'units': 'm/s'
        }
        ds_current['current_v'].attrs = {
            'long_name': 'Northward current velocity component',
            'units': 'm/s'
        }

        # Merge current data with ERA5
        print(f"  🔗 Merging new current data...")
        ds_merged = xr.merge([ds_era5, ds_current], compat='override')

        # Check merge quality
        print(f"  ✅ Merged variables: {list(ds_merged.data_vars)}")

        # Count records with current data
        current_count = (~ds_merged['current_velocity_mean'].isnull()).sum().values
        total_count = len(ds_merged.time)
        print(f"  📊 Records with currents: {current_count} / {total_count} ({current_count/total_count*100:.1f}%)")

        # Save updated file
        print(f"  💾 Saving updated file...")
        ds_merged.to_netcdf(nc_file)

        file_size_mb = nc_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ Updated: {file_size_mb:.2f} MB")

        successful_updates.append((year, "updated"))

        # Clean up
        ds_era5.close()
        ds_merged.close()

    except Exception as e:
        print(f"\n  ❌ ERROR updating {year}:")
        print(f"     {str(e)}")
        import traceback
        traceback.print_exc()
        failed_updates.append(year)

        # Restore from backup if update failed
        if backup_file.exists():
            print(f"  🔄 Restoring from backup...")
            shutil.copy2(backup_file, nc_file)

        continue

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n\n{'='*80}")
print(f"📊 UPDATE SUMMARY")
print(f"{'='*80}")

print(f"\n✅ Successfully updated: {len(successful_updates)} files")
for year, status in successful_updates:
    if status == "updated":
        print(f"  • {year}: ✅ Current data added (2015-2025 coverage)")
    else:
        print(f"  • {year}: ⚠️  No current data available for this year")

if failed_updates:
    print(f"\n❌ Failed updates: {len(failed_updates)} files")
    for year in failed_updates:
        print(f"  • {year}")

print(f"\n💾 Backups saved to: {backup_dir}")
print(f"   (You can delete these once you verify the updates)")

print(f"\n{'='*80}")
print(f"✅ UPDATE COMPLETE!")
print(f"{'='*80}")

print(f"\n📊 Your ERA5 files now have:")
print(f"  ✅ Wave data (swh, mwd, pp1d)")
print(f"  ✅ Wind data (u10, v10)")
print(f"  ✅ Current data (ERA5-Ocean, 2015-2025)")
print(f"     • current_velocity_mean (6-hour average)")
print(f"     • current_velocity_max (6-hour maximum) ⚠️  USE FOR SAFETY!")
print(f"     • current_direction")
print(f"     • current_u, current_v (components)")

print(f"\n🎯 Ready for workability analysis with full 10-year current data!")
