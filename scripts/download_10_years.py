"""
Download ERA5 data for 10 years: 2015-2025
Downloads each year separately to avoid CDS limits
Using 6-hourly data (00:00, 06:00, 12:00, 18:00)
Location: UK Northeast Coast (54.67°N, 1.04°W)
"""

import cdsapi
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
import sys
import zipfile
import xarray as xr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.parsers import ERA5Parser
from src.data.scatter_generator import ScatterGenerator
from src.data.processors import DataProcessor


def unzip_and_merge_era5(zip_path):
    """
    Unzip ERA5 download and merge multiple NetCDF files
    CDS returns data as zip with separate files for different data streams
    """
    zip_path = Path(zip_path)

    # Create temp directory for extraction
    temp_dir = zip_path.parent / f"temp_{zip_path.stem}"
    temp_dir.mkdir(exist_ok=True)

    print(f"  Unzipping {zip_path.name}...")

    # Extract all files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find all .nc files
    nc_files = list(temp_dir.glob("*.nc"))
    print(f"  Found {len(nc_files)} NetCDF files")

    if len(nc_files) == 0:
        print("  ✗ No NetCDF files found in zip!")
        return None

    # Open and merge all datasets
    datasets = []
    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        datasets.append(ds)

    # Merge all datasets
    if len(datasets) > 1:
        print(f"  Merging {len(datasets)} datasets...")
        merged = xr.merge(datasets, join='outer')
    else:
        merged = datasets[0]

    # Save merged dataset
    output_path = zip_path.parent / f"{zip_path.stem}_merged.nc"
    print(f"  Saving merged file...")
    merged.to_netcdf(output_path)

    # IMPORTANT: Close all datasets before cleanup
    print(f"  Closing datasets...")
    for ds in datasets:
        ds.close()
    merged.close()

    # Cleanup
    print(f"  Cleaning up temp files...")
    for nc_file in nc_files:
        nc_file.unlink()
    temp_dir.rmdir()

    # Remove original zip
    zip_path.unlink()

    print(f"  ✓ Merged to {output_path.name}")
    return str(output_path)


def download_single_year(client, year, area, output_dir):
    """Download a single year of 6-hourly data"""

    print("\n" + "=" * 80)
    print(f"DOWNLOADING YEAR {year}")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    # Output file (check for merged version first)
    output_file_merged = output_dir / f"era5_UK_NortheastCoast_{year}_merged.nc"
    output_file = output_dir / f"era5_UK_NortheastCoast_{year}.nc"

    # Check if merged file already exists
    if output_file_merged.exists():
        file_size = output_file_merged.stat().st_size / (1024 * 1024)
        print(f"✓ Merged file already exists ({file_size:.1f} MB) - skipping download")
        return str(output_file_merged)

    # Check if original file exists (might need merging)
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)
        # If file is small (< 5 MB), it's probably a zip that needs unzipping
        if file_size < 5:
            print(f"  Found zip file ({file_size:.1f} MB) - needs unzipping")
            merged_path = unzip_and_merge_era5(output_file)
            if merged_path:
                return merged_path
        else:
            print(f"✓ File already exists ({file_size:.1f} MB) - skipping download")
            return str(output_file)

    # Months to download
    if year == 2025:
        # Current year - only Jan to Nov
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
        print(f"Months: Jan-Nov (11 months)")
    else:
        # Full year
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        print(f"Months: Jan-Dec (12 months)")

    # CDS Request
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_wave_direction",
            "peak_wave_period",
            "significant_height_of_combined_wind_waves_and_swell"
        ],
        "year": [str(year)],
        "month": months,
        "day": [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
        ],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area
    }

    print("Submitting request to CDS...")
    print("This will wait in queue, then download automatically...")
    print("(CDS API handles the waiting internally - be patient!)")

    start_time = time.time()

    try:
        # The retrieve() method will handle queuing and downloading automatically
        # It blocks until complete, so no need for manual monitoring
        client.retrieve(dataset, request).download(str(output_file))

        file_size = output_file.stat().st_size / (1024 * 1024)
        total_time = time.time() - start_time

        print(f"\n✓ Downloaded ({file_size:.1f} MB in {time.strftime('%H:%M:%S', time.gmtime(total_time))})")

        # Unzip and merge if needed
        if file_size < 5:  # Small file = zip archive
            merged_path = unzip_and_merge_era5(output_file)
            if merged_path:
                return merged_path
            else:
                return None
        else:
            return str(output_file)

    except Exception as e:
        print(f"\n✗ Error downloading {year}: {e}")
        print(f"   Error type: {type(e).__name__}")
        return None


def process_all_years(nc_files, years, center_lat, center_lon):
    """Process and combine all downloaded years"""

    print("\n" + "=" * 80)
    print("PROCESSING ALL YEARS")
    print("=" * 80)

    all_dataframes = []

    for nc_file, year in zip(nc_files, years):
        print(f"\nProcessing {year}...")

        parser = ERA5Parser(nc_file)
        parser.load()

        df = parser.extract_location(lat=center_lat, lon=center_lon, method='nearest')
        print(f"  ✓ {len(df):,} records")

        all_dataframes.append(df)
        parser.close()

    # Combine all years
    print("\nCombining all years...")
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    df_combined = df_combined.sort_values('time').reset_index(drop=True)

    print(f"✓ Combined: {len(df_combined):,} total records")
    print(f"  Date range: {df_combined['time'].min()} to {df_combined['time'].max()}")

    # Quality control
    print("\nQuality control...")
    processor = DataProcessor()
    df_clean = processor.validate_ranges(df_combined)
    df_clean = processor.fill_missing(df_clean)
    print(f"✓ Cleaned: {len(df_clean):,} valid records")

    # Generate scatter diagram
    print("\nGenerating scatter diagram...")
    scatter_gen = ScatterGenerator()
    scatter_df = scatter_gen.generate(df_clean, hs_col='hs', tp_col='tp')
    print(f"✓ Scatter: {len(scatter_df)} cells")

    return df_clean, scatter_df


def save_results(df_clean, scatter_df, years):
    """Save processed results"""

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Time series
    ts_dir = Path('data/processed/timeseries')
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_file = ts_dir / f"UK_NortheastCoast_{min(years)}_{max(years)}.parquet"
    df_clean.to_parquet(ts_file, index=False)
    print(f"✓ Time series: {ts_file}")

    # Scatter diagram
    sc_dir = Path('data/processed/scatter_diagrams')
    sc_dir.mkdir(parents=True, exist_ok=True)

    sc_parquet = sc_dir / f"UK_NortheastCoast_{min(years)}_{max(years)}_scatter.parquet"
    scatter_df.to_parquet(sc_parquet, index=False)
    print(f"✓ Scatter (parquet): {sc_parquet}")

    sc_csv = sc_dir / f"UK_NortheastCoast_{min(years)}_{max(years)}_scatter.csv"
    scatter_df.to_csv(sc_csv, index=False)
    print(f"✓ Scatter (CSV): {sc_csv}")

    return ts_file, sc_csv


def display_results(df_clean, scatter_df, years):
    """Display comprehensive results"""

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY: {min(years)}-{max(years)}")
    print("=" * 80)

    print(f"\nDataset Information:")
    print(f"  Years: {len(years)} years ({min(years)}-{max(years)})")
    print(f"  Total records: {len(df_clean):,} (6-hourly)")
    print(f"  Date range: {df_clean['time'].min()} to {df_clean['time'].max()}")
    print(f"  Equivalent hours: {len(df_clean) * 6:,}")

    print(f"\nWave Statistics:")
    print(f"  Mean Hs: {df_clean['hs'].mean():.2f} m")
    print(f"  Median Hs: {df_clean['hs'].median():.2f} m")
    print(f"  Max Hs: {df_clean['hs'].max():.2f} m")
    print(f"  Std Dev Hs: {df_clean['hs'].std():.2f} m")
    print(f"  90th percentile: {df_clean['hs'].quantile(0.90):.2f} m")
    print(f"  95th percentile: {df_clean['hs'].quantile(0.95):.2f} m")
    print(f"  99th percentile: {df_clean['hs'].quantile(0.99):.2f} m")

    print(f"\nWave Period Statistics:")
    print(f"  Mean Tp: {df_clean['tp'].mean():.2f} s")
    print(f"  Median Tp: {df_clean['tp'].median():.2f} s")
    print(f"  Max Tp: {df_clean['tp'].max():.2f} s")

    print(f"\nWind Statistics:")
    print(f"  Mean wind speed: {df_clean['wind_speed'].mean():.2f} m/s")
    print(f"  Max wind speed: {df_clean['wind_speed'].max():.2f} m/s")
    print(f"  95th percentile: {df_clean['wind_speed'].quantile(0.95):.2f} m/s")

    print("\n" + "-" * 80)
    print("Wave Height Distribution (% of time):")
    print("-" * 80)
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for threshold in thresholds:
        pct = (df_clean['hs'] < threshold).sum() / len(df_clean) * 100
        bar = '█' * int(pct / 2)
        print(f"  Hs < {threshold:>3.1f}m: {pct:>6.1f}% {bar}")

    print("\n" + "-" * 80)
    print("Top 20 Most Frequent Sea States:")
    print("-" * 80)
    top = scatter_df.nlargest(20, 'frequency')[['hs_bin', 'tp_bin', 'frequency', 'percentage']]
    print(top.to_string(index=False))

    # Yearly breakdown
    print("\n" + "-" * 80)
    print("Yearly Statistics:")
    print("-" * 80)
    df_clean['year'] = pd.to_datetime(df_clean['time']).dt.year
    print(f"{'Year':<8} {'Records':>10} {'Mean Hs':>10} {'Max Hs':>10} {'Mean Wind':>12}")
    print("-" * 80)
    for year in sorted(df_clean['year'].unique()):
        year_data = df_clean[df_clean['year'] == year]
        print(f"{year:<8} {len(year_data):>10,} {year_data['hs'].mean():>10.2f} "
              f"{year_data['hs'].max():>10.2f} {year_data['wind_speed'].mean():>12.2f}")


def main():
    print("=" * 80)
    print("ERA5 10-YEAR DOWNLOAD: 2015-2025")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Location configuration
    center_lat = 55.4  # Your location
    center_lon = 0
    buffer = 0.5

    # Bounding box: [North, West, South, East]
    north = center_lat + buffer
    south = center_lat - buffer
    west = center_lon - buffer
    east = center_lon + buffer
    area = [north, west, south, east]

    print(f"Location: {center_lat:.4f}°N, {abs(center_lon):.4f}°W")
    print(f"Bounding box: [N:{north:.2f}, W:{west:.2f}, S:{south:.2f}, E:{east:.2f}]")
    print(f"Coverage: ~55km x 55km")

    # Years to download
    YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    print(f"\nYears to download: {YEARS}")
    print(f"Total: {len(YEARS)} years")
    print(f"Data resolution: 6-hourly (00:00, 06:00, 12:00, 18:00)")
    print(f"Expected records: ~40,000 total")

    print("\n" + "-" * 80)
    print("IMPORTANT NOTES:")
    print("-" * 80)
    print("• This will take 1-3 HOURS total (depends on CDS queue)")
    print("• Each year takes ~5-15 minutes")
    print("• You can stop and resume - already downloaded years are skipped")
    print("• Files saved to: data/raw/era5/")
    print("-" * 80)

    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Setup
    output_dir = Path('data/raw/era5')
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    print("\n✓ CDS client initialized")

    # Download each year
    print("\n" + "=" * 80)
    print("DOWNLOADING YEARS")
    print("=" * 80)

    downloaded_files = []
    downloaded_years = []

    for i, year in enumerate(YEARS, 1):
        print(f"\n[{i}/{len(YEARS)}] Year {year}")

        nc_file = download_single_year(client, year, area, output_dir)

        if nc_file:
            downloaded_files.append(nc_file)
            downloaded_years.append(year)
            print(f"✓ {year} complete")
        else:
            print(f"✗ {year} failed - skipping")

        # Brief pause between downloads (except for last one)
        if i < len(YEARS) and nc_file:
            print("  Pausing 10 seconds before next year...")
            time.sleep(10)

    # Summary of downloads
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Successfully downloaded: {len(downloaded_files)}/{len(YEARS)} years")
    print(f"Years: {downloaded_years}")

    if not downloaded_files:
        print("\n✗ No files downloaded. Exiting.")
        return

    # Process all years
    df_clean, scatter_df = process_all_years(
        downloaded_files,
        downloaded_years,
        center_lat,
        center_lon
    )

    # Save results
    ts_file, sc_file = save_results(df_clean, scatter_df, downloaded_years)

    # Display results
    display_results(df_clean, scatter_df, downloaded_years)

    # Final summary
    print("\n" + "=" * 80)
    print("✓ ALL COMPLETE!")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nYou now have {len(downloaded_years)} years of sea state data!")
    print(f"Total records: {len(df_clean):,} (6-hourly)")
    print(f"\nData files:")
    print(f"  • Time series: {ts_file}")
    print(f"  • Scatter diagram: {sc_file}")
    print(f"\nReady for Phase 2: Workability Analysis!")


if __name__ == '__main__':
    main()
