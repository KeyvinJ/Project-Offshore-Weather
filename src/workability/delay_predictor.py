"""
Delay Predictor

Monte Carlo-based prediction of operational delays due to weather conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from .calculator import WorkabilityCalculator


class DelayPredictor:
    """
    Predict project delays using Monte Carlo simulation with copula-based
    or naive independence assumptions.
    """

    def __init__(self, calculator: Optional[WorkabilityCalculator] = None):
        """
        Initialize the delay predictor.

        Parameters
        ----------
        calculator : WorkabilityCalculator, optional
            Workability calculator instance. If None, creates a new one.
        """
        self.calculator = calculator if calculator is not None else WorkabilityCalculator()

    def predict_delays_historical(
        self,
        workscope: Dict,
        historical_data: pd.DataFrame,
        n_simulations: int = 1000,
        date_col: str = 'time',
        hs_col: str = 'hs',
        wind_col: str = 'wind_speed',
        current_col: str = 'current'
    ) -> Dict:
        """
        Predict delays using historical weather patterns (bootstrap resampling).

        This method uses actual historical weather sequences to predict delays.
        It randomly selects starting dates from historical data and simulates
        project execution.

        Parameters
        ----------
        workscope : dict
            Workscope definition with keys:
            - 'name': str, operation name
            - 'start_date': str or datetime, planned start date
            - 'mob_days': int, mobilization days (weather-independent)
            - 'pure_ops_days': int, pure operational days (weather-dependent)
            - 'demob_days': int, demobilization days (weather-independent)
            - 'limits': dict with 'hs_max', 'wind_max', 'current_max'
        historical_data : pd.DataFrame
            Historical weather data
        n_simulations : int
            Number of Monte Carlo simulations
        date_col : str
            Name of the datetime column
        hs_col : str
            Name of the Hs column
        wind_col : str
            Name of the wind speed column
        current_col : str
            Name of the current column

        Returns
        -------
        dict
            Prediction results with keys:
            - 'p10_delay': float, 10th percentile delay (optimistic)
            - 'p50_delay': float, 50th percentile delay (median)
            - 'p90_delay': float, 90th percentile delay (pessimistic)
            - 'mean_delay': float, mean delay
            - 'total_duration_p10': float, total duration at P10
            - 'total_duration_p50': float, total duration at P50
            - 'total_duration_p90': float, total duration at P90
            - 'workability_pct': float, average workability percentage
            - 'delays_distribution': np.ndarray, all simulated delays
            - 'workscope': dict, original workscope definition
        """
        # Parse workscope
        mob_days = workscope.get('mob_days', 0)
        pure_ops_days = workscope['pure_ops_days']
        demob_days = workscope.get('demob_days', 0)
        limits = workscope['limits']

        # Ensure data has datetime index
        if date_col in historical_data.columns:
            df = historical_data.set_index(date_col) if date_col != historical_data.index.name else historical_data.copy()
        else:
            df = historical_data.copy()

        # Calculate workability for each timestep
        df['workable'] = df.apply(
            lambda row: self.calculator.check_workability(
                row[hs_col],
                row[wind_col],
                row[current_col],
                limits
            ),
            axis=1
        )

        # Storage for simulation results
        delays = []
        total_durations = []
        workable_counts = []

        # Assuming 6-hourly data, convert days to periods
        periods_per_day = 4  # 24 hours / 6 hours

        # Monte Carlo simulation
        for sim in range(n_simulations):
            # Randomly select a start index (ensure we have enough data ahead)
            max_periods_needed = (mob_days + pure_ops_days + demob_days + 60) * periods_per_day  # +60 days buffer for delays
            max_start_idx = len(df) - max_periods_needed

            if max_start_idx < 0:
                raise ValueError(f"Insufficient historical data. Need ~{max_periods_needed} periods, have {len(df)}")

            start_idx = np.random.randint(0, max_start_idx)

            # Simulate project execution
            current_period = start_idx

            # Phase 1: Mobilization (weather-independent)
            current_period += mob_days * periods_per_day

            # Phase 2: Pure operations (weather-DEPENDENT)
            completed_ops_days = 0
            ops_start_period = current_period
            workable_period_count = 0

            while completed_ops_days < pure_ops_days:
                if current_period >= len(df):
                    # Ran out of data - use last known workability rate
                    raise ValueError(f"Simulation {sim} ran out of historical data")

                # Check if current period is workable
                if df.iloc[current_period]['workable']:
                    # Accumulate workable time (6-hourly periods)
                    workable_period_count += 1
                    # Check if we've completed a full day of operations (4 periods)
                    if workable_period_count % periods_per_day == 0:
                        completed_ops_days += 1

                current_period += 1

            ops_end_period = current_period

            # Phase 3: Demobilization (weather-independent)
            current_period += demob_days * periods_per_day

            # Calculate results
            total_periods = current_period - start_idx
            total_duration_days = total_periods / periods_per_day

            planned_duration = mob_days + pure_ops_days + demob_days
            delay_days = total_duration_days - planned_duration

            ops_periods = ops_end_period - ops_start_period
            ops_workability_pct = (workable_period_count / ops_periods * 100) if ops_periods > 0 else 0

            delays.append(delay_days)
            total_durations.append(total_duration_days)
            workable_counts.append(ops_workability_pct)

        # Calculate statistics
        delays = np.array(delays)
        total_durations = np.array(total_durations)

        return {
            'p10_delay': np.percentile(delays, 10),
            'p50_delay': np.percentile(delays, 50),
            'p90_delay': np.percentile(delays, 90),
            'mean_delay': np.mean(delays),
            'total_duration_p10': np.percentile(total_durations, 10),
            'total_duration_p50': np.percentile(total_durations, 50),
            'total_duration_p90': np.percentile(total_durations, 90),
            'workability_pct': np.mean(workable_counts),
            'delays_distribution': delays,
            'total_durations_distribution': total_durations,
            'workscope': workscope
        }

    def predict_delays_copula(
        self,
        workscope: Dict,
        copula_model: any,
        marginal_distributions: Dict,
        start_date: datetime,
        n_simulations: int = 1000
    ) -> Dict:
        """
        Predict delays using copula-based Monte Carlo simulation.

        This method generates synthetic weather scenarios using fitted copula
        models and marginal distributions, then simulates project execution.

        Parameters
        ----------
        workscope : dict
            Workscope definition (same as predict_delays_historical)
        copula_model : pyvinecopulib copula object
            Fitted copula model (e.g., vine copula or bivariate copula)
        marginal_distributions : dict
            Dictionary with fitted marginal distributions:
            - 'hs': {'func': scipy.stats distribution, 'params': tuple}
            - 'wind': {'func': scipy.stats distribution, 'params': tuple}
            - 'current': {'func': scipy.stats distribution, 'params': tuple}
        start_date : datetime
            Project start date
        n_simulations : int
            Number of Monte Carlo simulations

        Returns
        -------
        dict
            Same structure as predict_delays_historical
        """
        # Parse workscope
        mob_days = workscope.get('mob_days', 0)
        pure_ops_days = workscope['pure_ops_days']
        demob_days = workscope.get('demob_days', 0)
        limits = workscope['limits']

        # Assuming 6-hourly data
        periods_per_day = 4

        # Storage for simulation results
        delays = []
        total_durations = []
        workable_counts = []

        # Estimate maximum periods needed
        max_periods_needed = (mob_days + pure_ops_days + demob_days + 120) * periods_per_day  # +120 days buffer

        for sim in range(n_simulations):
            # Generate synthetic weather scenario from copula
            copula_samples = copula_model.simulate(max_periods_needed)

            # Transform to original scale using marginal distributions
            hs_samples = marginal_distributions['hs']['func'].ppf(
                copula_samples[:, 0],
                *marginal_distributions['hs']['params']
            )
            wind_samples = marginal_distributions['wind']['func'].ppf(
                copula_samples[:, 1],
                *marginal_distributions['wind']['params']
            )

            # Handle current - could be from copula or independent
            if copula_samples.shape[1] == 3:
                # 3D copula includes current
                current_samples = marginal_distributions['current']['func'].ppf(
                    copula_samples[:, 2],
                    *marginal_distributions['current']['params']
                )
            else:
                # 2D copula, current is independent
                current_samples = marginal_distributions['current']['func'].rvs(
                    *marginal_distributions['current']['params'],
                    size=max_periods_needed
                )

            # Simulate project execution
            current_period = 0

            # Phase 1: Mobilization
            current_period += mob_days * periods_per_day

            # Phase 2: Pure operations
            completed_ops_days = 0
            ops_start_period = current_period
            workable_period_count = 0

            while completed_ops_days < pure_ops_days:
                if current_period >= max_periods_needed:
                    raise ValueError(f"Simulation {sim} exceeded maximum periods")

                # Check if current period is workable
                is_workable = self.calculator.check_workability(
                    hs_samples[current_period],
                    wind_samples[current_period],
                    current_samples[current_period],
                    limits
                )

                if is_workable:
                    workable_period_count += 1
                    if workable_period_count % periods_per_day == 0:
                        completed_ops_days += 1

                current_period += 1

            ops_end_period = current_period

            # Phase 3: Demobilization
            current_period += demob_days * periods_per_day

            # Calculate results
            total_duration_days = current_period / periods_per_day
            planned_duration = mob_days + pure_ops_days + demob_days
            delay_days = total_duration_days - planned_duration

            ops_periods = ops_end_period - ops_start_period
            ops_workability_pct = (workable_period_count / ops_periods * 100) if ops_periods > 0 else 0

            delays.append(delay_days)
            total_durations.append(total_duration_days)
            workable_counts.append(ops_workability_pct)

        # Calculate statistics
        delays = np.array(delays)
        total_durations = np.array(total_durations)

        return {
            'p10_delay': np.percentile(delays, 10),
            'p50_delay': np.percentile(delays, 50),
            'p90_delay': np.percentile(delays, 90),
            'mean_delay': np.mean(delays),
            'total_duration_p10': np.percentile(total_durations, 10),
            'total_duration_p50': np.percentile(total_durations, 50),
            'total_duration_p90': np.percentile(total_durations, 90),
            'workability_pct': np.mean(workable_counts),
            'delays_distribution': delays,
            'total_durations_distribution': total_durations,
            'workscope': workscope
        }

    def predict_delays_naive(
        self,
        workscope: Dict,
        marginal_distributions: Dict,
        start_date: datetime,
        n_simulations: int = 1000
    ) -> Dict:
        """
        Predict delays using NAIVE independence assumption (WRONG but for comparison).

        This method generates weather scenarios assuming complete independence
        between Hs, Wind, and Current. This is the approach commonly used but
        is known to be incorrect.

        Parameters
        ----------
        workscope : dict
            Workscope definition
        marginal_distributions : dict
            Fitted marginal distributions for Hs, Wind, Current
        start_date : datetime
            Project start date
        n_simulations : int
            Number of Monte Carlo simulations

        Returns
        -------
        dict
            Same structure as other predict methods
        """
        # Parse workscope
        mob_days = workscope.get('mob_days', 0)
        pure_ops_days = workscope['pure_ops_days']
        demob_days = workscope.get('demob_days', 0)
        limits = workscope['limits']

        periods_per_day = 4
        max_periods_needed = (mob_days + pure_ops_days + demob_days + 120) * periods_per_day

        delays = []
        total_durations = []
        workable_counts = []

        for sim in range(n_simulations):
            # Generate INDEPENDENT samples (WRONG!)
            hs_samples = marginal_distributions['hs']['func'].rvs(
                *marginal_distributions['hs']['params'],
                size=max_periods_needed
            )
            wind_samples = marginal_distributions['wind']['func'].rvs(
                *marginal_distributions['wind']['params'],
                size=max_periods_needed
            )
            current_samples = marginal_distributions['current']['func'].rvs(
                *marginal_distributions['current']['params'],
                size=max_periods_needed
            )

            # Simulate project (same logic as copula method)
            current_period = 0
            current_period += mob_days * periods_per_day

            completed_ops_days = 0
            ops_start_period = current_period
            workable_period_count = 0

            while completed_ops_days < pure_ops_days:
                if current_period >= max_periods_needed:
                    raise ValueError(f"Simulation {sim} exceeded maximum periods")

                is_workable = self.calculator.check_workability(
                    hs_samples[current_period],
                    wind_samples[current_period],
                    current_samples[current_period],
                    limits
                )

                if is_workable:
                    workable_period_count += 1
                    if workable_period_count % periods_per_day == 0:
                        completed_ops_days += 1

                current_period += 1

            ops_end_period = current_period
            current_period += demob_days * periods_per_day

            total_duration_days = current_period / periods_per_day
            planned_duration = mob_days + pure_ops_days + demob_days
            delay_days = total_duration_days - planned_duration

            ops_periods = ops_end_period - ops_start_period
            ops_workability_pct = (workable_period_count / ops_periods * 100) if ops_periods > 0 else 0

            delays.append(delay_days)
            total_durations.append(total_duration_days)
            workable_counts.append(ops_workability_pct)

        delays = np.array(delays)
        total_durations = np.array(total_durations)

        return {
            'p10_delay': np.percentile(delays, 10),
            'p50_delay': np.percentile(delays, 50),
            'p90_delay': np.percentile(delays, 90),
            'mean_delay': np.mean(delays),
            'total_duration_p10': np.percentile(total_durations, 10),
            'total_duration_p50': np.percentile(total_durations, 50),
            'total_duration_p90': np.percentile(total_durations, 90),
            'workability_pct': np.mean(workable_counts),
            'delays_distribution': delays,
            'total_durations_distribution': total_durations,
            'workscope': workscope
        }

    def predict_multiple_workscopes(
        self,
        workscopes: List[Dict],
        method: str = 'historical',
        **kwargs
    ) -> List[Dict]:
        """
        Predict delays for multiple sequential workscopes.

        Parameters
        ----------
        workscopes : list of dict
            List of workscope definitions (executed sequentially)
        method : str
            Prediction method: 'historical', 'copula', or 'naive'
        **kwargs
            Additional arguments passed to the specific prediction method

        Returns
        -------
        list of dict
            Prediction results for each workscope
        """
        results = []

        for i, workscope in enumerate(workscopes):
            # For sequential workscopes, adjust start date based on previous completions
            if i > 0 and method != 'historical':
                prev_result = results[-1]
                # Use P50 completion date as start for next workscope
                # (This is simplified - could be more sophisticated)
                pass

            # Select prediction method
            if method == 'historical':
                result = self.predict_delays_historical(workscope, **kwargs)
            elif method == 'copula':
                result = self.predict_delays_copula(workscope, **kwargs)
            elif method == 'naive':
                result = self.predict_delays_naive(workscope, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            results.append(result)

        return results
