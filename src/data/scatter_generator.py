"""Generate wave scatter diagrams from time series"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ScatterGenerator:
    """Create Hs-Tp scatter diagrams from wave time series"""

    def __init__(
        self,
        hs_bins: Optional[np.ndarray] = None,
        tp_bins: Optional[np.ndarray] = None
    ):
        if hs_bins is None:
            self.hs_bins = np.arange(0, 10.5, 0.5)
        else:
            self.hs_bins = hs_bins

        if tp_bins is None:
            self.tp_bins = np.arange(0, 21, 1)
        else:
            self.tp_bins = tp_bins

        logger.info(f"Hs bins: {len(self.hs_bins)-1} intervals")
        logger.info(f"Tp bins: {len(self.tp_bins)-1} intervals")

    def generate(
        self,
        df: pd.DataFrame,
        hs_col: str = 'hs',
        tp_col: str = 'tp'
    ) -> pd.DataFrame:
        """
        Generate scatter diagram from time series

        Returns DataFrame with columns:
            hs_bin, tp_bin, frequency, percentage
        """
        logger.info(f"Generating scatter from {len(df)} records")

        hist, hs_edges, tp_edges = np.histogram2d(
            df[hs_col],
            df[tp_col],
            bins=[self.hs_bins, self.tp_bins]
        )

        scatter_data = []

        for i in range(len(self.hs_bins) - 1):
            for j in range(len(self.tp_bins) - 1):
                frequency = hist[i, j]

                if frequency > 0:
                    hs_center = (self.hs_bins[i] + self.hs_bins[i+1]) / 2
                    tp_center = (self.tp_bins[j] + self.tp_bins[j+1]) / 2

                    scatter_data.append({
                        'hs_bin': hs_center,
                        'tp_bin': tp_center,
                        'hs_lower': self.hs_bins[i],
                        'hs_upper': self.hs_bins[i+1],
                        'tp_lower': self.tp_bins[j],
                        'tp_upper': self.tp_bins[j+1],
                        'frequency': int(frequency),
                        'percentage': frequency / len(df) * 100
                    })

        scatter_df = pd.DataFrame(scatter_data)

        logger.info(f"Generated {len(scatter_df)} non-zero cells")
        logger.info(f"Total frequency: {scatter_df['frequency'].sum()}")

        return scatter_df

    def generate_monthly(
        self,
        df: pd.DataFrame,
        time_col: str = 'time',
        hs_col: str = 'hs',
        tp_col: str = 'tp'
    ) -> dict:
        """Generate scatter diagrams for each month"""
        df = df.copy()
        df['month'] = pd.to_datetime(df[time_col]).dt.month

        monthly_scatters = {}

        for month in range(1, 13):
            month_df = df[df['month'] == month]
            if len(month_df) > 0:
                scatter = self.generate(month_df, hs_col, tp_col)
                monthly_scatters[month] = scatter

        return monthly_scatters

    def save(self, scatter_df: pd.DataFrame, filepath: str):
        """Save scatter diagram"""
        scatter_df.to_parquet(filepath, index=False)
        logger.info(f"Saved to {filepath}")

    def load(self, filepath: str) -> pd.DataFrame:
        """Load scatter diagram"""
        return pd.read_parquet(filepath)
