"""
Workability Calculator

Provides functionality to assess whether weather conditions meet operational
limits for offshore marine operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class WorkabilityCalculator:
    """
    Calculate workability based on operational weather limits.

    Workability is defined as the percentage of time when all weather
    conditions (Hs, Wind, Current) are within acceptable limits.
    """

    def __init__(self):
        """Initialize the workability calculator."""
        pass

    def check_workability(
        self,
        hs: float,
        wind: float,
        current: float,
        limits: Dict[str, float]
    ) -> bool:
        """
        Check if weather conditions meet operational limits.

        Parameters
        ----------
        hs : float
            Significant wave height in meters
        wind : float
            Wind speed in m/s
        current : float
            Current speed in knots
        limits : dict
            Dictionary with keys: 'hs_max', 'wind_max', 'current_max'

        Returns
        -------
        bool
            True if all conditions are within limits, False otherwise
        """
        # Handle NaN values - assume non-workable if any value is missing
        if np.isnan(hs) or np.isnan(wind) or np.isnan(current):
            return False

        # Check all limits
        hs_ok = hs <= limits.get('hs_max', np.inf)
        wind_ok = wind <= limits.get('wind_max', np.inf)
        current_ok = current <= limits.get('current_max', np.inf)

        return hs_ok and wind_ok and current_ok

    def calculate_daily_workability(
        self,
        data: pd.DataFrame,
        limits: Dict[str, float],
        date_col: str = 'time',
        hs_col: str = 'hs',
        wind_col: str = 'wind_speed',
        current_col: str = 'current'
    ) -> pd.DataFrame:
        """
        Calculate daily workability from historical data.

        Parameters
        ----------
        data : pd.DataFrame
            Historical weather data with datetime index or date column
        limits : dict
            Operational limits {'hs_max': X, 'wind_max': Y, 'current_max': Z}
        date_col : str
            Name of the datetime column
        hs_col : str
            Name of the significant wave height column
        wind_col : str
            Name of the wind speed column
        current_col : str
            Name of the current speed column

        Returns
        -------
        pd.DataFrame
            Daily workability statistics with columns:
            - date: Date
            - workability_pct: Percentage of 6-hourly periods that are workable
            - n_periods: Number of 6-hourly periods available
            - n_workable: Number of workable periods
        """
        # Ensure data has datetime index
        if date_col in data.columns:
            df = data.set_index(date_col) if date_col != data.index.name else data.copy()
        else:
            df = data.copy()

        # Check workability for each timestep
        df['workable'] = df.apply(
            lambda row: self.check_workability(
                row[hs_col],
                row[wind_col],
                row[current_col],
                limits
            ),
            axis=1
        )

        # Group by date and calculate daily statistics
        df['date'] = df.index.date
        daily_stats = df.groupby('date').agg({
            'workable': ['sum', 'count', 'mean']
        }).reset_index()

        # Flatten column names
        daily_stats.columns = ['date', 'n_workable', 'n_periods', 'workability_pct']
        daily_stats['workability_pct'] = daily_stats['workability_pct'] * 100

        return daily_stats

    def identify_weather_windows(
        self,
        data: pd.DataFrame,
        limits: Dict[str, float],
        min_duration_hours: int = 6,
        date_col: str = 'time',
        hs_col: str = 'hs',
        wind_col: str = 'wind_speed',
        current_col: str = 'current'
    ) -> List[Dict]:
        """
        Identify continuous weather windows that meet operational limits.

        Parameters
        ----------
        data : pd.DataFrame
            Historical weather data
        limits : dict
            Operational limits
        min_duration_hours : int
            Minimum duration (hours) for a valid weather window
        date_col : str
            Name of the datetime column
        hs_col : str
            Name of the significant wave height column
        wind_col : str
            Name of the wind speed column
        current_col : str
            Name of the current speed column

        Returns
        -------
        list of dict
            Each dict contains:
            - start: Start datetime
            - end: End datetime
            - duration_hours: Duration in hours
            - n_periods: Number of 6-hourly periods
        """
        # Ensure data has datetime index
        if date_col in data.columns:
            df = data.set_index(date_col) if date_col != data.index.name else data.copy()
        else:
            df = data.copy()

        # Check workability for each timestep
        df['workable'] = df.apply(
            lambda row: self.check_workability(
                row[hs_col],
                row[wind_col],
                row[current_col],
                limits
            ),
            axis=1
        )

        # Find continuous workable periods
        windows = []
        window_start = None

        for idx, row in df.iterrows():
            if row['workable']:
                if window_start is None:
                    window_start = idx
            else:
                if window_start is not None:
                    # Window ended, check if it meets minimum duration
                    duration = (idx - window_start).total_seconds() / 3600
                    if duration >= min_duration_hours:
                        windows.append({
                            'start': window_start,
                            'end': idx,
                            'duration_hours': duration,
                            'n_periods': int(duration / 6)
                        })
                    window_start = None

        # Handle case where window extends to end of data
        if window_start is not None:
            duration = (df.index[-1] - window_start).total_seconds() / 3600
            if duration >= min_duration_hours:
                windows.append({
                    'start': window_start,
                    'end': df.index[-1],
                    'duration_hours': duration,
                    'n_periods': int(duration / 6)
                })

        return windows

    def calculate_workability_stats(
        self,
        data: pd.DataFrame,
        limits: Dict[str, float],
        hs_col: str = 'hs',
        wind_col: str = 'wind_speed',
        current_col: str = 'current'
    ) -> Dict:
        """
        Calculate overall workability statistics.

        Parameters
        ----------
        data : pd.DataFrame
            Historical weather data
        limits : dict
            Operational limits
        hs_col : str
            Name of the significant wave height column
        wind_col : str
            Name of the wind speed column
        current_col : str
            Name of the current speed column

        Returns
        -------
        dict
            Statistics including:
            - overall_workability_pct: Overall workability percentage
            - limiting_factor_pct: Percentage each factor limits operations
            - n_total_periods: Total number of periods analyzed
            - n_workable_periods: Number of workable periods
        """
        df = data.copy()

        # Check each limit individually
        df['hs_ok'] = df[hs_col] <= limits.get('hs_max', np.inf)
        df['wind_ok'] = df[wind_col] <= limits.get('wind_max', np.inf)
        df['current_ok'] = df[current_col] <= limits.get('current_max', np.inf)
        df['all_ok'] = df['hs_ok'] & df['wind_ok'] & df['current_ok']

        # Calculate statistics
        n_total = len(df)
        n_workable = df['all_ok'].sum()
        workability_pct = (n_workable / n_total) * 100 if n_total > 0 else 0

        # Identify limiting factors (when not workable, which factor was the issue?)
        df_not_workable = df[~df['all_ok']]
        n_not_workable = len(df_not_workable)

        limiting_factors = {}
        if n_not_workable > 0:
            limiting_factors['hs'] = (~df_not_workable['hs_ok']).sum() / n_not_workable * 100
            limiting_factors['wind'] = (~df_not_workable['wind_ok']).sum() / n_not_workable * 100
            limiting_factors['current'] = (~df_not_workable['current_ok']).sum() / n_not_workable * 100
        else:
            limiting_factors = {'hs': 0, 'wind': 0, 'current': 0}

        return {
            'overall_workability_pct': workability_pct,
            'limiting_factor_pct': limiting_factors,
            'n_total_periods': n_total,
            'n_workable_periods': int(n_workable)
        }
