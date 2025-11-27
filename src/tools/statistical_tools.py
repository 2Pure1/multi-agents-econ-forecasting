"""Statistical and Econometric Tools for Economic Analysis

This module provides a comprehensive suite of statistical and econometric functions 
for analyzing economic time-series data. It includes tools for data preparation, 
trend analysis, indicator calculation, business cycle identification, anomaly 
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Suppress common warnings from statsmodels to keep output clean
warnings.filterwarnings('ignore')

@dataclass
class EconomicIndicator:
    """A data container for the results of an economic indicator calculation."""
    name: str
    value: float
    trend: str
    confidence: float
    interpretation: str
    metadata: Dict[str, Any]

@dataclass
class ForecastResult:
    """A data container for the results of a forecasting operation."""
    point_forecast: float
    confidence_interval: Tuple[float, float]
    model_type: str
    accuracy_metrics: Dict[str, float]
    forecast_periods: int

class StatisticalTools:
    """
    A class that encapsulates a wide range of statistical tools for economic analysis.
    
    The methods in this class are designed to be used by other agents, such as the
    EconomicAnalystAgent, to perform detailed analysis on economic data.
    """
    
    def __init__(self):
        """Initializes the StatisticalTools class."""
        self.scaler = StandardScaler()
        
    def prepare_time_series_data(self, data: pd.DataFrame, 
                               date_column: str = 'TimePeriod',
                               value_column: str = 'DataValue') -> pd.Series:
        """
        Prepares raw DataFrame data for time-series analysis.
        
        This involves setting a datetime index, sorting the data, and ensuring
        the value column is numeric.
        
        Args:
            data: The input DataFrame.
            date_column: The name of the column containing date information.
            value_column: The name of the column containing the series values.
            
        Returns:
            A pandas Series with a DatetimeIndex, ready for analysis.
            
        Raises:
            ValueError: If there is an error in processing the data.
        """
        try:
            df = data.copy()
            
            # Ensure the date column is of datetime type
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
            
            # Set the date column as the index and sort
            df = df.set_index(date_column).sort_index()
            
            # Convert the value column to a numeric type, coercing errors
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            
            # Remove any rows with missing values in the target column
            df = df.dropna(subset=[value_column])
            
            return df[value_column]
            
        except Exception as e:
            raise ValueError(f"Error preparing time series data: {e}")
    
    def analyze_growth_trends(self, data: pd.DataFrame, 
                            value_column: str = 'DataValue',
                            window: int = 4) -> Dict[str, Any]:
        """
        Analyzes growth trends, volatility, and seasonality in the data.
        
        Args:
            data: DataFrame containing the economic time-series data.
            value_column: The name of the column to analyze.
            window: The rolling window size for moving averages (default is 4 for quarterly data).
            
        Returns:
            A dictionary containing a comprehensive analysis of growth trends.
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if len(ts_data) < 2:
                return {"status": "error", "message": "Insufficient data for trend analysis"}
            
            # --- Growth Rate Calculation ---
            growth_rates = ts_data.pct_change() * 100
            
            # --- Trend Analysis using Linear Regression ---
            X = np.arange(len(ts_data))
            X = add_constant(X)
            y = ts_data.values
            model = OLS(y, X).fit()
            trend_slope = model.params[1] if len(model.params) > 1 else 0
            trend_strength = model.rsquared # R-squared as a measure of trend strength
            
            # Determine the overall trend direction
            if trend_slope > 0.05: # Using a small threshold to avoid noise
                trend_direction = "upward"
            elif trend_slope < -0.05:
                trend_direction = "downward"
            else:
                trend_direction = "flat"
            
            # --- Volatility Analysis ---
            volatility = growth_rates.std()
            recent_volatility = growth_rates.tail(window).std()
            
            # --- Seasonality Detection ---
            seasonal_strength = self._detect_seasonality(ts_data)
            
            return {
                "status": "success",
                "current_growth_rate": growth_rates.iloc[-1] if not growth_rates.empty else 0,
                "average_growth_rate": growth_rates.mean(),
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "trend_slope": trend_slope,
                "volatility": volatility,
                "recent_volatility": recent_volatility,
                "seasonal_strength": seasonal_strength,
                "data_points": len(ts_data),
                "latest_value": ts_data.iloc[-1],
                "period_range": f"{ts_data.index[0].strftime('%Y-%m-%d')} to {ts_data.index[-1].strftime('%Y-%m-%d')}",
                "growth_rates_summary": {
                    "mean": growth_rates.mean(),
                    "std": growth_rates.std(),
                    "min": growth_rates.min(),
                    "max": growth_rates.max()
                }
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"Growth trend analysis failed: {e}"}
    
    def calculate_indicators(self, data: pd.DataFrame, 
                           value_column: str = 'DataValue') -> Dict[str, Any]:
        """
        Calculates a dictionary of key economic indicators from the data.
        
        Args:
            data: DataFrame containing the economic data.
            value_column: The name of the column to analyze.
            
        Returns:
            A dictionary containing calculated indicators and an interpretation.
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if len(ts_data) < 4:
                return {"status": "error", "message": "Insufficient data for indicator calculation"}
            
            indicators = {}
            
            # --- Basic Descriptive Statistics ---
            indicators["current_value"] = ts_data.iloc[-1]
            indicators["mean"] = ts_data.mean()
            indicators["median"] = ts_data.median()
            indicators["std_dev"] = ts_data.std()
            indicators["cv"] = (ts_data.std() / ts_data.mean()) * 100  # Coefficient of Variation

            # --- Growth & Volatility Indicators ---
            growth_rates = ts_data.pct_change() * 100
            indicators["recent_growth"] = growth_rates.tail(4).mean()  # Avg growth over the last year
            indicators["long_term_growth"] = growth_rates.mean()
            indicators["volatility"] = growth_rates.std()
            indicators["max_drawdown"] = self._calculate_max_drawdown(ts_data)
            
            # --- Trend & Stationarity ---
            indicators["trend_strength"] = self._calculate_trend_strength(ts_data)
            adf_result = adfuller(ts_data.dropna())
            indicators["is_stationary"] = adf_result[1] < 0.05  # p-value < 0.05 suggests stationarity
            
            # --- Cyclical & Momentum Indicators ---
            indicators["business_cycle_position"] = self._assess_business_cycle_position(ts_data)
            indicators["momentum"] = self._calculate_momentum(ts_data)
            
            return {
                "status": "success",
                "indicators": indicators,
                "interpretation": self._interpret_indicators(indicators)
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"Indicator calculation failed: {e}"}
    
    def identify_business_cycles(self, data: pd.DataFrame,
                               value_column: str = 'DataValue') -> Dict[str, Any]:
        """
        Identifies business cycle phases using the Hodrick-Prescott filter.
        
        Args:
            data: DataFrame containing the economic data.
            value_column: The name of the column to analyze.
            
        Returns:
            A dictionary containing the business cycle analysis.
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if len(ts_data) < 8:  # Need sufficient data for cycle detection
                return {"status": "error", "message": "Insufficient data for business cycle analysis"}
            
            # Decompose the series into trend and cycle components using HP filter
            cycle, trend = self._hp_filter(ts_data)
            
            # Find the peaks and troughs in the cyclical component
            peaks, troughs = self._find_peaks_troughs(cycle)
            
            # Determine the current phase based on the last peak/trough
            current_phase = self._determine_current_phase(cycle)
            
            return {
                "status": "success",
                "current_phase": current_phase,
                "cycle_component": cycle.tolist(),
                "trend_component": trend.tolist(),
                "peaks": [ts_data.index[i].strftime('%Y-%m-%d') for i in peaks],
                "troughs": [ts_data.index[i].strftime('%Y-%m-%d') for i in troughs],
                "average_cycle_duration": self._calculate_cycle_duration(peaks, troughs),
                "cycle_amplitude": self._calculate_cycle_amplitude(cycle),
                "current_phase_duration": self._calculate_current_phase_duration(cycle, peaks, troughs)
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"Business cycle analysis failed: {e}"}
    
    def detect_anomalies(self, data: pd.DataFrame,
                        value_column: str = 'DataValue',
                        method: str = 'iqr',
                        threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detects anomalies (outliers) in the economic data.
        
        Args:
            data: DataFrame containing the economic data.
            value_column: The name of the column to analyze.
            method: The detection method to use ('zscore', 'iqr'). 'iqr' is often more robust.
            threshold: The threshold for detection (e.g., 1.5 for IQR, 2.5 or 3 for z-score).
            
        Returns:
            A dictionary containing the detected anomalies.
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            anomalies = []
            anomaly_indices = []
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(ts_data))
                anomaly_indices = np.where(z_scores > threshold)[0]
                
            elif method == 'iqr':
                Q1 = ts_data.quantile(0.25)
                Q3 = ts_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                anomaly_indices = ts_data[(ts_data < lower_bound) | (ts_data > upper_bound)].index
            
            else:
                return {"status": "error", "message": f"Unknown anomaly detection method: {method}"}

            # Format the list of anomalies
            for idx in anomaly_indices:
                date = ts_data.index[idx] if isinstance(idx, (int, np.integer)) else idx
                value = ts_data.loc[date]
                anomalies.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': value,
                    'deviation_score': float((value - ts_data.mean()) / ts_data.std()),
                    'type': 'high' if value > ts_data.mean() else 'low'
                })
            
            return {
                "status": "success",
                "anomalies": anomalies,
                "anomaly_count": len(anomalies),
                "anomaly_percentage": (len(anomalies) / len(ts_data)) * 100 if len(ts_data) > 0 else 0,
                "detection_method": method,
                "threshold": threshold
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"Anomaly detection failed: {e}"}
    
    def forecast_arima(self, data: pd.DataFrame,
                      value_column: str = 'DataValue',
                      periods: int = 8,
                      order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
        """
        Generates a forecast using a specified ARIMA model.
        
        Args:
            data: DataFrame containing the time-series data.
            value_column: The name of the column to forecast.
            periods: The number of future periods to forecast.
            order: The (p, d, q) order of the ARIMA model.
            
        Returns:
            A dictionary with the forecast, confidence intervals, and model diagnostics.
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if len(ts_data) < 10:
                return {"status": "error", "message": "Insufficient data for ARIMA modeling"}
            
            # Fit the ARIMA model
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            # Generate the forecast
            forecast = fitted_model.get_forecast(steps=periods)
            forecast_mean = forecast.predicted_mean
            confidence_int = forecast.conf_int()
            
            # --- Prepare Output ---
            last_date = ts_data.index[-1]
            freq = pd.infer_freq(ts_data.index) or 'Q' # Default to Quarterly
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            
            forecast_results = []
            for i, (date, point_fc, (lower, upper)) in enumerate(zip(
                forecast_dates, forecast_mean, confidence_int.values)):
                forecast_results.append({
                    'period': date.strftime('%Y-%m-%d'),
                    'point_forecast': float(point_fc),
                    'confidence_lower': float(lower),
                    'confidence_upper': float(upper),
                    'period_ahead': i + 1
                })
            
            return {
                "status": "success",
                "forecasts": forecast_results,
                "model_summary": {
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic,
                    "order": order,
                },
                "accuracy_metrics": self._calculate_model_accuracy(fitted_model, ts_data),
                "residuals_analysis": {
                    "mean_residual": fitted_model.resid.mean(),
                    "std_residual": fitted_model.resid.std(),
                    "normality_pvalue": stats.normaltest(fitted_model.resid.dropna())[1]
                }
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"ARIMA forecasting failed: {e}"}
    
    def build_arima_model(self, data: pd.DataFrame,
                         value_column: str = 'DataValue',
                         auto_select: bool = True,
                         max_order: int = 3) -> Dict[str, Any]:
        """
        Builds and validates an ARIMA model, with optional automatic order selection.
        
        Args:
            data: DataFrame containing the time-series data.
            value_column: The name of the column to model.
            auto_select: If True, automatically find the best ARIMA order.
            max_order: The maximum p and q to check during auto-selection.
            
        Returns:
            A dictionary with the best model's order, parameters, and diagnostics.
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            best_order = self._auto_arima(ts_data, max_order=max_order) if auto_select else (1, 1, 1)
            
            # Fit the final model with the best order
            model = ARIMA(ts_data, order=best_order)
            fitted_model = model.fit()
            
            residuals = fitted_model.resid.dropna()
            
            return {
                "status": "success",
                "best_order": best_order,
                "summary": {
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic,
                    "log_likelihood": fitted_model.llf
                },
                "parameters": fitted_model.params.to_dict(),
                "residuals_diagnostics": {
                    "mean": residuals.mean(),
                    "std": residuals.std(),
                    "jarque_bera_pvalue": stats.jarque_bera(residuals)[1],
                    "ljung_box_pvalue": sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
                },
                "in_sample_accuracy": self._calculate_model_accuracy(fitted_model, ts_data)
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"ARIMA model building failed: {e}"}
    
    # ======================================================================
    # Helper Methods (Private)
    # ======================================================================
    
    def _detect_seasonality(self, ts_data: pd.Series) -> float:
        """Helper to detect seasonality strength."""
        try:
            if len(ts_data) >= 8:  # Need at least 2 years of quarterly data
                # Decompose the series to isolate the seasonal component
                result = seasonal_decompose(ts_data, model='additive', period=4)
                # Calculate strength as the standard deviation of the seasonal component
                # relative to the standard deviation of the original series.
                seasonal_strength = result.seasonal.std() / ts_data.std()
                return float(seasonal_strength)
            return 0.0
        except Exception:
            return 0.0 # Return 0 if seasonality detection fails
    
    def _calculate_max_drawdown(self, ts_data: pd.Series) -> float:
        """Helper to calculate the maximum drawdown (a measure of risk)."""
        cumulative = (1 + ts_data.pct_change()).cumprod().fillna(1)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min()) if not drawdown.empty else 0.0
    
    def _calculate_trend_strength(self, ts_data: pd.Series) -> float:
        """Helper to calculate trend strength using R-squared from a linear regression."""
        X = np.arange(len(ts_data))
        X = add_constant(X)
        y = ts_data.values
        model = OLS(y, X).fit()
        return float(model.rsquared)
    
    def _assess_business_cycle_position(self, ts_data: pd.Series) -> str:
        """Helper to make a simple assessment of the current business cycle position."""
        if len(ts_data) < 4:
            return "insufficient_data"
        
        recent_growth = ts_data.pct_change().tail(4).mean()
        volatility = ts_data.pct_change().std()
        
        if recent_growth > (0.5 * volatility): # Growth is significantly positive
            return "expansion"
        elif recent_growth < (-0.5 * volatility): # Growth is significantly negative
            return "contraction"
        else:
            return "stable/transition"
    
    def _calculate_momentum(self, ts_data: pd.Series, periods: int = 4) -> float:
        """Helper to calculate a momentum indicator (year-over-year change)."""
        if len(ts_data) < periods + 1:
            return 0.0
        return float(ts_data.pct_change(periods=periods).iloc[-1])
    
    def _hp_filter(self, ts_data: pd.Series, lamb: float = 1600) -> Tuple[pd.Series, pd.Series]:
        """Applies the Hodrick-Prescott filter to decompose a series into trend and cycle."""
        # For quarterly data, a lambda of 1600 is standard.
        cycle, trend = sm.tsa.filters.hpfilter(ts_data, lamb=lamb)
        return cycle, trend
    
    def _find_peaks_troughs(self, series: pd.Series) -> Tuple[List[int], List[int]]:
        """Finds local peaks and troughs in a time series."""
        # Using scipy's find_peaks for a more robust implementation
        peaks, _ = stats.find_peaks(series.values)
        troughs, _ = stats.find_peaks(-series.values)
        return peaks.tolist(), troughs.tolist()
    
    def _determine_current_phase(self, cycle: pd.Series) -> str:
        """Determines the current phase (expansion/contraction) based on recent movement."""
        if len(cycle) < 2:
            return "undetermined"
        
        # Check if the last few points are generally rising or falling
        recent_slope = cycle.tail(3).diff().mean()
        
        if recent_slope > 0:
            return "expansion"
        elif recent_slope < 0:
            return "contraction"
        else:
            return "transition"
    
    def _calculate_cycle_duration(self, peaks: List[int], troughs: List[int]) -> float:
        """Calculates the average duration of full business cycles (trough to trough)."""
        if len(troughs) < 2:
            return 0.0
        # Calculate the time difference between consecutive troughs
        durations = np.diff(troughs)
        return float(np.mean(durations)) if len(durations) > 0 else 0.0

    def _calculate_cycle_amplitude(self, cycle: pd.Series) -> float:
        """Calculates the amplitude of the cycle as the difference between its max and min."""
        return float(cycle.max() - cycle.min()) if not cycle.empty else 0.0
    
    def _calculate_current_phase_duration(self, cycle: pd.Series, peaks: List[int], troughs: List[int]) -> int:
        """Calculates the duration of the current, ongoing phase."""
        if not peaks and not troughs:
            return 0
        
        last_peak = max(peaks) if peaks else -1
        last_trough = max(troughs) if troughs else -1
        last_turning_point = max(last_peak, last_trough)
        
        return len(cycle) - last_turning_point if last_turning_point != -1 else len(cycle)

    def _interpret_indicators(self, indicators: Dict[str, Any]) -> str:
        """Provides a brief, human-readable interpretation of a set of indicators."""
        interpretation = []
        
        if indicators.get("recent_growth", 0) > 1.5:
            interpretation.append("Strong recent growth.")
        elif indicators.get("recent_growth", 0) < -0.5:
            interpretation.append("Recent data shows contraction.")
        
        if indicators.get("volatility", 0) > 5:
            interpretation.append("High volatility environment detected.")
        
        if indicators.get("trend_strength", 0) > 0.75:
            interpretation.append("Data shows a strong, clear trend.")
        
        return " ".join(interpretation) if interpretation else "Economic conditions appear stable."
    
    def _calculate_mape(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculates Mean Absolute Percentage Error, handling potential zeros in actuals."""
        actual, predicted = np.array(actual), np.array(predicted)
        # Avoid division by zero
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    
    def _auto_arima(self, ts_data: pd.Series, max_order: int = 3) -> Tuple[int, int, int]:
        """
        A simplified grid-search for the best ARIMA order based on AIC.
        
        Note: For production use, a more sophisticated library like 'pmdarima' 
        (auto_arima) is highly recommended as it is more efficient and robust.
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # Grid search over a small range of p, d, q values
        for p in range(max_order + 1):
            for d in range(2):  # Differencing up to 1
                for q in range(max_order + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except Exception:
                        # Ignore orders that cause errors
                        continue
        return best_order
    
    def _calculate_model_accuracy(self, model, ts_data: pd.Series) -> Dict[str, float]:
        """Calculates a dictionary of common accuracy metrics for a fitted model."""
        try:
            # In-sample predictions
            predictions = model.predict(start=ts_data.index[1], end=ts_data.index[-1])
            actual = ts_data[1:]
            
            return {
                "mae": mean_absolute_error(actual, predictions),
                "rmse": np.sqrt(mean_squared_error(actual, predictions)),
                "mape": self._calculate_mape(actual, predictions),
                "r2": r2_score(actual, predictions)
            }
        except Exception:
            # Return empty metrics if calculation fails
            return {"mae": 0, "rmse": 0, "mape": 0, "r2": 0}

# ======================================================================
# Example Usage Block
# ======================================================================
if __name__ == "__main__":
    # This block demonstrates how to use the StatisticalTools class and is useful for testing.
    
    # --- 1. Create Sample Data ---
    # Simulate quarterly GDP data with a clear trend, seasonality, and some noise.
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='Q')
    np.random.seed(42)
    trend = np.linspace(100, 120, len(dates))
    seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 4)
    noise = np.random.normal(0, 1.5, len(dates))
    gdp_data = trend + seasonal + noise
    
    sample_data = pd.DataFrame({
        'TimePeriod': dates,
        'DataValue': gdp_data
    })
    
    # --- 2. Initialize the Toolset ---
    tools = StatisticalTools()
    
    print("="*50)
    print("Testing Statistical and Econometric Tools")
    print("="*50)
    
    # --- 3. Test Public Methods ---
    
    # Test growth trend analysis
    print("\n--- Analyzing Growth Trends ---")
    growth_analysis = tools.analyze_growth_trends(sample_data)
    print(growth_analysis)
    
    # Test indicator calculation
    print("\n--- Calculating Economic Indicators ---")
    indicators = tools.calculate_indicators(sample_data)
    print(indicators)
    
    # Test business cycle analysis
    print("\n--- Identifying Business Cycles ---")
    cycles = tools.identify_business_cycles(sample_data)
    print(cycles)
    
    # Test anomaly detection
    print("\n--- Detecting Anomalies ---")
    anomalies = tools.detect_anomalies(sample_data)
    print(anomalies)
    
    # Test ARIMA model building
    print("\n--- Building ARIMA Model ---")
    arima_model = tools.build_arima_model(sample_data)
    print(arima_model)
    
    # Test ARIMA forecasting
    if arima_model.get('status') == 'success':
        print("\n--- Forecasting with ARIMA ---")
        forecast = tools.forecast_arima(sample_data, periods=4, order=arima_model['best_order'])
        print(forecast)
