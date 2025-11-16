"""Quick script to unzip and merge the 2015 file"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import zipfile
import xarray as xr
import shutil

zip_path = Path('data/raw/era5/era5_UK_NortheastCoast_2015.nc')
temp_dir = Path('data/raw/era5/temp_2015')

print("Unzipping 2015 file...")
temp_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Find NC files
nc_files = list(temp_dir.glob('*.nc'))
print(f"Found {len(nc_files)} NetCDF files")

# Merge
print("Merging datasets...")
datasets = [xr.open_dataset(f) for f in nc_files]
merged = xr.merge(datasets, join='outer')

# Save
output_path = Path('data/raw/era5/era5_UK_NortheastCoast_2015_merged.nc')
print("Saving merged file...")
merged.to_netcdf(output_path)

# Close datasets before cleanup
print("Closing datasets...")
for ds in datasets:
    ds.close()
merged.close()

# Cleanup
print("Cleaning up...")
shutil.rmtree(temp_dir)
zip_path.unlink()

print(f"✓ Done! Created {output_path}")
print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
