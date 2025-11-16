"""Data processing and quality control"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean metocean data"""

    def __init__(self):
        pass

    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using standard deviation method

        Args:
            df: DataFrame
            column: Column to check for outliers
            n_std: Number of standard deviations for threshold

        Returns:
            DataFrame with outliers removed
        """
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std

        initial_count = len(df)
        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed_count = initial_count - len(df_clean)

        logger.info(f"Removed {removed_count} outliers from {column}")
        return df_clean

    def fill_missing(
        self,
        df: pd.DataFrame,
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        Fill missing values

        Args:
            df: DataFrame
            method: 'interpolate', 'forward', 'backward'

        Returns:
            DataFrame with missing values filled
        """
        df_filled = df.copy()

        missing_before = df_filled.isna().sum().sum()

        if method == 'interpolate':
            df_filled = df_filled.interpolate(method='linear')
        elif method == 'forward':
            df_filled = df_filled.fillna(method='ffill')
        elif method == 'backward':
            df_filled = df_filled.fillna(method='bfill')

        missing_after = df_filled.isna().sum().sum()
        logger.info(f"Filled {missing_before - missing_after} missing values using {method}")

        return df_filled

    def validate_ranges(
        self,
        df: pd.DataFrame,
        hs_max: float = 20.0,
        tp_max: float = 30.0,
        wind_max: float = 50.0
    ) -> pd.DataFrame:
        """
        Validate that data is within physically reasonable ranges

        Args:
            df: DataFrame
            hs_max: Maximum valid Hs (m)
            tp_max: Maximum valid Tp (s)
            wind_max: Maximum valid wind speed (m/s)

        Returns:
            DataFrame with invalid values removed
        """
        initial_count = len(df)

        if 'hs' in df.columns:
            df = df[(df['hs'] >= 0) & (df['hs'] <= hs_max)]

        if 'tp' in df.columns:
            df = df[(df['tp'] >= 0) & (df['tp'] <= tp_max)]

        if 'wind_speed' in df.columns:
            df = df[(df['wind_speed'] >= 0) & (df['wind_speed'] <= wind_max)]

        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} records with invalid ranges")

        return df

    def resample_timeseries(
        self,
        df: pd.DataFrame,
        freq: str = '1H',
        time_col: str = 'time'
    ) -> pd.DataFrame:
        """
        Resample time series to different frequency

        Args:
            df: DataFrame
            freq: Frequency string ('1H', '3H', '6H', '1D', etc.)
            time_col: Name of time column

        Returns:
            Resampled DataFrame
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

        df_resampled = df.resample(freq).mean()

        logger.info(f"Resampled from {len(df)} to {len(df_resampled)} records at {freq}")

        return df_resampled.reset_index()
