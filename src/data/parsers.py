"""Parse metocean data from various formats"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ERA5Parser:
    """Parse ERA5 NetCDF files to pandas DataFrames"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.ds = None

    def load(self):
        """Load NetCDF file"""
        logger.info(f"Loading {self.filepath}")
        self.ds = xr.open_dataset(self.filepath)
        logger.info(f"Variables: {list(self.ds.data_vars)}")

    def extract_location(
        self,
        lat: float,
        lon: float,
        method: str = 'nearest'
    ) -> pd.DataFrame:
        """
        Extract time series at specific location

        Args:
            lat: Latitude (degrees N)
            lon: Longitude (degrees E)
            method: 'nearest' or 'linear'

        Returns:
            DataFrame with columns: time, hs, tp, dir, wind_speed, etc.
        """
        if self.ds is None:
            self.load()

        loc = self.ds.sel(latitude=lat, longitude=lon, method=method)

        # Detect time dimension name (can be 'time' or 'valid_time')
        time_var = 'valid_time' if 'valid_time' in loc else 'time'

        # Check which variables are available
        data_dict = {'time': loc[time_var].values}

        # Wave height
        if 'swh' in loc:
            data_dict['hs'] = loc['swh'].values
        elif 'significant_height_of_combined_wind_waves_and_swell' in loc:
            data_dict['hs'] = loc['significant_height_of_combined_wind_waves_and_swell'].values

        # Wave period
        if 'pp1d' in loc:
            data_dict['tp'] = loc['pp1d'].values
        elif 'peak_wave_period' in loc:
            data_dict['tp'] = loc['peak_wave_period'].values

        # Wave direction
        if 'mwd' in loc:
            data_dict['dir'] = loc['mwd'].values
        elif 'mean_wave_direction' in loc:
            data_dict['dir'] = loc['mean_wave_direction'].values

        # Wind components
        if 'u10' in loc:
            data_dict['wind_u'] = loc['u10'].values
        if 'v10' in loc:
            data_dict['wind_v'] = loc['v10'].values

        df = pd.DataFrame(data_dict)

        # Calculate wind speed and direction if wind components exist
        if 'wind_u' in df.columns and 'wind_v' in df.columns:
            df['wind_speed'] = np.sqrt(df['wind_u']**2 + df['wind_v']**2)
            df['wind_dir'] = (270 - np.degrees(np.arctan2(df['wind_v'], df['wind_u']))) % 360

        logger.info(f"Extracted {len(df)} time steps")
        logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"Variables: {list(df.columns)}")

        return df

    def extract_region(self, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
        """Extract data for rectangular region"""
        if self.ds is None:
            self.load()

        lat_min, lon_min, lat_max, lon_max = bbox
        subset = self.ds.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
        return subset

    def close(self):
        if self.ds is not None:
            self.ds.close()
