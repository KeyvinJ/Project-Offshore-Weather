"""
Check if we have ocean current data and investigate options
"""

import xarray as xr
from pathlib import Path

print("=" * 80)
print("🌊 OCEAN CURRENT DATA INVESTIGATION")
print("=" * 80)

# Check existing files
data_dir = Path('data/raw/era5')
nc_files = sorted(data_dir.glob('*_merged.nc'))

if len(nc_files) > 0:
    print(f"\n📂 Checking existing files...")
    sample_file = nc_files[0]
    print(f"   File: {sample_file.name}")

    try:
        ds = xr.open_dataset(sample_file)

        print(f"\n📋 Variables currently in your data:")
        for var in ds.data_vars:
            long_name = ds[var].attrs.get('long_name', 'No description')
            units = ds[var].attrs.get('units', 'No units')
            print(f"  • {var:10s} - {long_name} ({units})")

        # Check for current-related variables
        current_vars = ['uo', 'vo', 'u_current', 'v_current', 'ucur', 'vcur',
                       'eastward_sea_water_velocity', 'northward_sea_water_velocity']

        has_currents = any(var in ds.data_vars for var in current_vars)

        if has_currents:
            print(f"\n✅ GOOD NEWS: Ocean current data found!")
            for var in current_vars:
                if var in ds.data_vars:
                    print(f"   Found: {var}")
        else:
            print(f"\n❌ Ocean current data NOT in current files")
            print(f"   Your files contain: atmosphere and surface wave data")
            print(f"   Currents require: ERA5 ocean reanalysis dataset")

        ds.close()

    except Exception as e:
        print(f"\n❌ Error reading file: {e}")

else:
    print(f"\n❌ No NetCDF files found in data/raw/era5/")

print("\n" + "=" * 80)
print("🔍 ERA5 OCEAN CURRENT OPTIONS")
print("=" * 80)

print("""
ERA5 has TWO separate datasets:

1. **ERA5 Single Levels** (what you downloaded)
   ✅ Variables: waves (swh, pp1d, mwd)
   ✅ Variables: wind (u10, v10)
   ❌ NO ocean currents

2. **ERA5 Ocean Reanalysis** (separate download)
   ✅ Variables: ocean currents (uo, vo)
   ✅ Variables: sea surface temperature
   ✅ Variables: mixed layer depth
   ⚠️  Requires separate CDS API request

To get ocean currents, you need to download from:
  Dataset: 'reanalysis-era5-single-levels' → ocean variables
  OR
  Dataset: 'reanalysis-era5-ocean' (if available for your region)
""")

print("=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

print("""
Option 1: ✅ PROCEED TO PHASE 2 NOW (Recommended)
  Why:
  • Waves + Wind cover 80% of marine operations
  • Current data download will take time
  • You can add currents later as "Phase 1B"
  • Most workability analyses focus on waves/wind

  Proceed if: Cable laying, ROV, diving NOT your main focus

Option 2: ⏸️  ADD CURRENTS FIRST
  Why:
  • Critical for: diving, ROV, DP vessels, cable laying
  • More complete analysis
  • Better for certain vessel types

  Do this if: Your operations are current-limited

Option 3: 🔄 HYBRID APPROACH
  Why:
  • Complete Phase 2 with waves/wind
  • Download currents in parallel
  • Add "Phase 2B - Current Enhanced" later

  Best of both worlds!
""")

print("=" * 80)
print("❓ QUESTIONS TO DECIDE")
print("=" * 80)

print("""
Ask yourself:

1. What operations are you analyzing?
   • Crane operations → Waves/Wind sufficient ✅
   • Jack-up operations → Waves/Wind sufficient ✅
   • Diving/ROV → NEED currents ⚠️
   • Cable laying → NEED currents ⚠️
   • DP vessels → Currents helpful but not critical 🟡

2. How urgent is your analysis?
   • Need results soon → Proceed without currents
   • Have time → Add currents first

3. Can you do it in stages?
   • Yes → Phase 2 now, currents later
   • No → Add currents now
""")

print("\n✅ Script complete! Decide based on your operation type.")
