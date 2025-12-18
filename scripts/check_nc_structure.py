"""Quick check of NetCDF structure"""
import xarray as xr
from pathlib import Path

nc_file = Path('C:/Work/Project-Offshore-Weather/data/raw/era5/era5_UK_NortheastCoast_2015_merged.nc')

print(f"Inspecting: {nc_file.name}\n")

ds = xr.open_dataset(nc_file)

print("="*80)
print("DATASET INFO")
print("="*80)
print(ds)

print("\n" + "="*80)
print("DIMENSIONS")
print("="*80)
print(dict(ds.dims))

print("\n" + "="*80)
print("COORDINATES")
print("="*80)
print(list(ds.coords))

print("\n" + "="*80)
print("DATA VARIABLES")
print("="*80)
print(list(ds.data_vars))

print("\n" + "="*80)
print("FIRST FEW VALUES OF EACH COORDINATE")
print("="*80)
for coord in ds.coords:
    print(f"\n{coord}:")
    print(ds[coord].values[:5])
