"""
Download ocean current data from OFFSHORE location
Open-Meteo Marine API - 25km offshore from UK Northeast Coast
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
print("🌊 OFFSHORE OCEAN CURRENT DATA DOWNLOADER")
print("=" * 80)
print("\nDownloading from Open-Meteo Marine API")
print("Location: UK Northeast Coast - OFFSHORE (25km from coast)")
print("Period: 2022-2025 (current data availability)")
print("Resolution: Hourly\n")

# Setup the Open-Meteo API client with cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ============================================================================
# OFFSHORE LOCATION - 25km from shore
# ============================================================================
# Original nearshore: 54.672745°N, -1.03719°W (~5km from shore)
# New offshore: 54.85°N, -0.70°W (~25km offshore)
#
# Moving northeast from coast to deeper water where:
# - Tidal currents are weaker
# - More representative of offshore diving conditions
# - Typical offshore wind farm / platform locations

center_lat = 54.85   # Further north (offshore)
center_lon = -0.70   # Further east (offshore)

print(f"📍 Location Settings:")
print(f"  Offshore center: {center_lat}°N, {abs(center_lon)}°W")
print(f"  Distance from shore: ~25 km")
print(f"  Water depth: ~40-60m (typical offshore work depth)")
print(f"")
print(f"🔄 Comparison:")
print(f"  Old nearshore location: 54.67°N, -1.04°W (~5km from shore)")
print(f"  New offshore location: 54.85°N, -0.70°W (~25km offshore)")
print(f"  Expected: LOWER currents offshore (0.3-1.5 kt vs 1.5-4.0 kt)")

# ============================================================================
# DOWNLOAD YEAR BY YEAR (2022-2025 only - current data availability)
# ============================================================================

# Output directory
output_dir = Path('data/raw/currents_offshore')
output_dir.mkdir(parents=True, exist_ok=True)

# Years to download (Open-Meteo Marine API has data from 2022 onwards)
start_year = 2022
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
        "end_date": end_date
    }

    try:
        # Make the API request
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        print(f"  ✅ API request successful!")
        print(f"  Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
        print(f"  Elevation: {response.Elevation()} m")
        print(f"  Timezone: {response.Timezone()} {response.TimezoneAbbreviation()}")

        # Process hourly data
        hourly = response.Hourly()
        hourly_current_velocity = hourly.Variables(0).ValuesAsNumpy()
        hourly_current_direction = hourly.Variables(1).ValuesAsNumpy()

        # Create datetime index
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }

        # Add current variables
        hourly_data["current_velocity"] = hourly_current_velocity  # m/s
        hourly_data["current_direction"] = hourly_current_direction  # degrees

        # Create DataFrame
        df = pd.DataFrame(data=hourly_data)

        # Calculate U and V components
        # U = eastward (positive = east, negative = west)
        # V = northward (positive = north, negative = south)
        # Direction is "direction from" in meteorological convention
        dir_rad = np.radians(df['current_direction'])
        df['current_u'] = -df['current_velocity'] * np.sin(dir_rad)
        df['current_v'] = -df['current_velocity'] * np.cos(dir_rad)

        # Statistics
        print(f"\n  📊 Data statistics:")
        print(f"    Records: {len(df):,}")
        print(f"    Mean velocity: {df['current_velocity'].mean():.3f} m/s ({df['current_velocity'].mean()*1.94384:.2f} knots)")
        print(f"    Max velocity: {df['current_velocity'].max():.3f} m/s ({df['current_velocity'].max()*1.94384:.2f} knots)")
        print(f"    Min velocity: {df['current_velocity'].min():.3f} m/s ({df['current_velocity'].min()*1.94384:.2f} knots)")

        # Check records below diving limits
        below_05kt = (df['current_velocity'] * 1.94384 < 0.5).sum()
        below_075kt = (df['current_velocity'] * 1.94384 < 0.75).sum()
        below_10kt = (df['current_velocity'] * 1.94384 < 1.0).sum()

        print(f"\n  🎯 Workability check:")
        print(f"    Below 0.5 kt (strict diving): {below_05kt:,} records ({below_05kt/len(df)*100:.1f}%)")
        print(f"    Below 0.75 kt (diving): {below_075kt:,} records ({below_075kt/len(df)*100:.1f}%)")
        print(f"    Below 1.0 kt (ROV): {below_10kt:,} records ({below_10kt/len(df)*100:.1f}%)")

        # Save to CSV
        output_file = output_dir / f'currents_offshore_{year}.csv'
        df.to_csv(output_file, index=False)
        print(f"\n  💾 Saved to: {output_file}")

        successful_downloads.append(year)

        # Rate limiting - be nice to the API
        if year < end_year:
            print(f"\n  ⏳ Waiting 2 seconds before next request...")
            time.sleep(2)

    except Exception as e:
        print(f"  ❌ Failed to download {year}: {str(e)}")
        failed_downloads.append(year)
        continue

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("📊 DOWNLOAD SUMMARY")
print(f"{'='*80}")

print(f"\n✅ Successful downloads: {len(successful_downloads)}")
for year in successful_downloads:
    print(f"  • {year}")

if failed_downloads:
    print(f"\n❌ Failed downloads: {len(failed_downloads)}")
    for year in failed_downloads:
        print(f"  • {year}")

print(f"\n📂 Output directory: {output_dir.absolute()}")
print(f"\n🎉 OFFSHORE current data download complete!")
print(f"\n⚠️  Next step: Run merge script to combine with wave/wind data")
