"""
Metocean data downloaders from various sources
"""
import cdsapi
import xarray as xr
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ERA5Downloader:
    """Download wave and wind data from ECMWF ERA5"""

    def __init__(self, output_dir: str = 'data/raw/era5'):
        self.client = cdsapi.Client()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_waves(
        self,
        years: List[int],
        bbox: Tuple[float, float, float, float],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Download wave data from ERA5

        Args:
            years: List of years [2020, 2021, 2022]
            bbox: (north, west, south, east) in degrees
            output_filename: Output file name

        Returns:
            Path to downloaded file
        """
        if output_filename is None:
            year_str = f"{min(years)}_{max(years)}"
            output_filename = f"era5_waves_{year_str}.nc"

        output_path = self.output_dir / output_filename

        logger.info(f"Downloading ERA5 data for years {years}")
        logger.info(f"Bounding box: {bbox}")
        logger.info(f"Output: {output_path}")

        request = {
            'product_type': 'reanalysis',
            'variable': [
                'significant_height_of_combined_wind_waves_and_swell',
                'peak_wave_period',
                'mean_wave_direction',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
            ],
            'year': [str(y) for y in years],
            'month': [f'{m:02d}' for m in range(1, 13)],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24)],
            'area': bbox,
            'format': 'netcdf',
        }

        self.client.retrieve(
            'reanalysis-era5-single-levels',
            request,
            str(output_path)
        )

        logger.info(f"Download complete: {output_path}")
        return str(output_path)


class NOAAWaveWatchDownloader:
    """Download from NOAA WaveWatch III"""

    def __init__(self, output_dir: str = 'data/raw/wavewatch'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_waves(self, start_date: str, end_date: str, lat: float, lon: float) -> str:
        """Download WaveWatch III data for location"""
        # Implementation for NOAA THREDDS server
        logger.info("WaveWatch III download not yet implemented")
        raise NotImplementedError("Use ERA5Downloader for now")
