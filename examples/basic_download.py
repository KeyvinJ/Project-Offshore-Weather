"""
Simple example: Download and process ERA5 data for a location
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.downloaders import ERA5Downloader
from src.data.parsers import ERA5Parser
from src.data.scatter_generator import ScatterGenerator
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()

# Define location (North Sea, Aberdeen area)
location = {
    'name': 'Aberdeen',
    'lat': 57.5,
    'lon': 0.5,
    'bbox': (58.0, -1.0, 57.0, 2.0)  # (north, west, south, east)
}

# Download data
print("Downloading ERA5 data...")
downloader = ERA5Downloader()
nc_file = downloader.download_waves(
    years=[2023],
    bbox=location['bbox']
)

# Parse and extract
print("Parsing data...")
parser = ERA5Parser(nc_file)
df = parser.extract_location(lat=location['lat'], lon=location['lon'])

print(f"\nExtracted {len(df)} records")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print("\nStatistics:")
print(df[['hs', 'tp', 'wind_speed']].describe())

# Generate scatter diagram
print("\nGenerating scatter diagram...")
scatter_gen = ScatterGenerator()
scatter_df = scatter_gen.generate(df)

print(f"\nScatter diagram has {len(scatter_df)} cells")
print("\nMost frequent sea states:")
print(scatter_df.nlargest(10, 'frequency')[['hs_bin', 'tp_bin', 'frequency']])

parser.close()
print("\nDone!")
