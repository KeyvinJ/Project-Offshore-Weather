"""
Download ocean current data from Open-Meteo Marine API
10 years (2015-2025), UK Northeast Coast
"""

import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
from pathlib import Path
from datetime import datetime
import time

print("=" * 80)
print("🌊 OCEAN CURRENT DATA DOWNLOADER")
print("=" * 80)
print("\nDownloading from Open-Meteo Marine API")
print("Location: UK Northeast Coast (Hartlepool/Middlesbrough)")
print("Period: 2015-2025 (10+ years)")
print("Resolution: Hourly\n")

# Setup the Open-Meteo API client with cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ============================================================================
# YOUR LOCATION - UK Northeast Coast
# ============================================================================
# Center point: Hartlepool/Middlesbrough area
center_lat = 54.672745
center_lon = -1.03719  # Note: Negative for West

# Buffer (same as ERA5 downloads)
buffer = 0.5

# Calculate bounding box
# Format: [lat_north, lon_west, lat_south, lon_east]
bbox_north = center_lat + buffer  # 55.17
bbox_south = center_lat - buffer  # 54.17
bbox_west = center_lon - buffer   # -1.54
bbox_east = center_lon + buffer   # -0.54

print(f"📍 Location Settings:")
print(f"  Center: {center_lat}°N, {abs(center_lon)}°W")
print(f"  Bounding box:")
print(f"    North: {bbox_north}°N")
print(f"    South: {bbox_south}°N")
print(f"    West: {abs(bbox_west)}°W")
print(f"    East: {abs(bbox_east)}°W")
print(f"    (Approx {buffer*2}° × {buffer*2}° area)")

# ============================================================================
# DOWNLOAD YEAR BY YEAR
# ============================================================================

# Output directory
output_dir = Path('data/raw/currents')
output_dir.mkdir(parents=True, exist_ok=True)

# Years to download
start_year = 2015
end_year = 2025  # Will get partial 2025

print(f"\n📅 Downloading {end_year - start_year + 1} years of data...")
print(f"  Output directory: {output_dir}\n")

# API endpoint
url = "https://marine-api.open-meteo.com/v1/marine"

successful_downloads = []
failed_downloads = []

for year in range(start_year, end_year + 1):
    print(f"\n{'='*80}")
    print(f"📥 Downloading {year}...")
    print(f"{'='*80}")

    # Set dates for this year
    if year == 2025:
        # Partial year - up to today
        start_date = f"{year}-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        # Full year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    print(f"  Period: {start_date} to {end_date}")

    # Parameters for API request
    params = {
        "latitude": center_lat,
        "longitude": center_lon,
        "hourly": [
            "ocean_current_velocity",    # m/s
            "ocean_current_direction"    # degrees
        ],
        "start_date": start_date,
        "end_date": end_date,
    }

    try:
        # Make API request
        print(f"  Requesting data from Open-Meteo API...")
        start_time = time.time()

        responses = openmeteo.weather_api(url, params=params)

        # Process response (should be only one for single location)
        response = responses[0]

        elapsed = time.time() - start_time
        print(f"  ✅ Data received ({elapsed:.1f} seconds)")

        # Extract data
        print(f"  Processing data...")
        print(f"    Coordinates: {response.Latitude():.4f}°N {response.Longitude():.4f}°E")
        print(f"    Elevation: {response.Elevation()} m asl")

        # Get hourly data
        hourly = response.Hourly()
        current_velocity = hourly.Variables(0).ValuesAsNumpy()
        current_direction = hourly.Variables(1).ValuesAsNumpy()

        # Create time index
        time_range = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        # Create DataFrame
        df = pd.DataFrame({
            'time': time_range,
            'current_velocity': current_velocity,  # m/s
            'current_direction': current_direction  # degrees
        })

        # Calculate U and V components (useful for later)
        # Direction is "from" direction (meteorological convention)
        # Convert to radians and get components
        dir_rad = np.radians(df['current_direction'])
        df['current_u'] = -df['current_velocity'] * np.sin(dir_rad)  # Eastward
        df['current_v'] = -df['current_velocity'] * np.cos(dir_rad)  # Northward

        print(f"    Records: {len(df):,}")
        print(f"    Time range: {df['time'].min()} to {df['time'].max()}")

        # Data quality check
        valid_velocity = df['current_velocity'].notna().sum()
        valid_direction = df['current_direction'].notna().sum()

        print(f"\n  📊 Data quality:")
        print(f"    Valid velocity: {valid_velocity:,} / {len(df):,} ({valid_velocity/len(df)*100:.1f}%)")
        print(f"    Valid direction: {valid_direction:,} / {len(df):,} ({valid_direction/len(df)*100:.1f}%)")

        if valid_velocity > 0:
            print(f"\n  🌊 Current statistics:")
            print(f"    Mean velocity: {df['current_velocity'].mean():.3f} m/s")
            print(f"    Max velocity: {df['current_velocity'].max():.3f} m/s")
            print(f"    Median velocity: {df['current_velocity'].median():.3f} m/s")

            # Convert to knots for reference
            mean_knots = df['current_velocity'].mean() * 1.94384
            max_knots = df['current_velocity'].max() * 1.94384
            print(f"    Mean velocity: {mean_knots:.2f} knots")
            print(f"    Max velocity: {max_knots:.2f} knots")

        # Save to parquet (efficient format)
        output_file = output_dir / f'ocean_currents_{year}.parquet'
        df.to_parquet(output_file, index=False)

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n  💾 Saved: {output_file.name}")
        print(f"    Size: {file_size_mb:.2f} MB")

        successful_downloads.append(year)

        # Be nice to the API - small delay between years
        if year < end_year:
            print(f"\n  ⏸️  Waiting 2 seconds before next download...")
            time.sleep(2)

    except Exception as e:
        print(f"\n  ❌ ERROR downloading {year}:")
        print(f"    {str(e)}")
        failed_downloads.append(year)
        continue

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n\n{'='*80}")
print(f"📊 DOWNLOAD SUMMARY")
print(f"{'='*80}")

print(f"\n✅ Successfully downloaded: {len(successful_downloads)} years")
for year in successful_downloads:
    file = output_dir / f'ocean_currents_{year}.parquet'
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"  • {year}: {size_mb:.2f} MB")

if failed_downloads:
    print(f"\n❌ Failed downloads: {len(failed_downloads)} years")
    for year in failed_downloads:
        print(f"  • {year}")

# Calculate total data
total_size = sum((output_dir / f'ocean_currents_{year}.parquet').stat().st_size
                 for year in successful_downloads) / (1024 * 1024)

print(f"\n📦 Total data downloaded: {total_size:.2f} MB")
print(f"📁 Location: {output_dir}")

print(f"\n{'='*80}")
print(f"✅ DOWNLOAD COMPLETE!")
print(f"{'='*80}")

print(f"\nNext steps:")
print(f"  1. Check data quality in each file")
print(f"  2. Merge with wave/wind data (align to 6-hourly)")
print(f"  3. Add current criteria to Phase 2")

print(f"\n💡 Current data notes:")
print(f"  • Data is HOURLY (higher resolution than waves/wind)")
print(f"  • Will need to downsample to 6-hourly to align")
print(f"  • Direction is 'from' direction (meteorological convention)")
print(f"  • Velocity is in m/s (multiply by 1.94384 for knots)")
