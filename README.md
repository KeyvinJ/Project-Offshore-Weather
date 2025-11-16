# Marine Workability Analysis Tool

A Python-based tool for analyzing offshore marine operations workability using metocean data.

## Project Status

**Phase 1: Data Acquisition & Processing** - ✅ COMPLETE

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure CDS API

Your CDS API credentials have been configured at: `~/.cdsapirc`

To verify or update:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_KEY_HERE
```

### 3. Test the Setup

Run the test script to download sample data:

```bash
python scripts/test_era5_download.py
```

This will:
- Download ERA5 wave data for Aberdeen, North Sea (2023)
- Parse the NetCDF file
- Extract time series
- Generate scatter diagram
- Save processed results

**Note:** First download may take 5-15 minutes depending on CDS queue.

## Project Structure

```
marine-workability/
├── config/                 # Configuration files (vessels, limits)
├── data/
│   ├── raw/era5/          # Downloaded ERA5 NetCDF files
│   └── processed/         # Processed time series and scatter diagrams
├── src/
│   ├── data/              # Data acquisition & processing
│   │   ├── downloaders.py      # ERA5 downloader
│   │   ├── parsers.py          # NetCDF parser
│   │   ├── processors.py       # Data QC and processing
│   │   └── scatter_generator.py # Scatter diagram generation
│   └── utils/
│       └── logging_config.py    # Logging setup
└── scripts/
    └── test_era5_download.py    # Test script
```

## Phase 1 Capabilities

### Download ERA5 Data
```python
from src.data.downloaders import ERA5Downloader

downloader = ERA5Downloader()
nc_file = downloader.download_waves(
    years=[2020, 2021, 2022],
    bbox=(58.0, -1.0, 57.0, 2.0)  # North Sea
)
```

### Parse and Extract Data
```python
from src.data.parsers import ERA5Parser

parser = ERA5Parser(nc_file)
df = parser.extract_location(lat=57.5, lon=0.5)
```

### Generate Scatter Diagram
```python
from src.data.scatter_generator import ScatterGenerator

scatter_gen = ScatterGenerator()
scatter_df = scatter_gen.generate(df)
```

## Next Phases

- **Phase 2:** Core Workability Engine (operational limits, basic workability)
- **Phase 3:** RAO Integration (vessel motion calculations)
- **Phase 4:** Extreme Value Analysis (return periods)
- **Phase 5:** Production Features (reporting, visualization)
- **Phase 6:** Advanced Analytics (ML forecasting, optimization)

## Documentation

Full project documentation is available in `workability_project.md`

## Support

Created by: Bobsky - Project Engineer, N-Sea Group
