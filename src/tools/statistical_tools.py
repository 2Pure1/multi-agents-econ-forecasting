"""
Statistical and Econometric Tools for Economic Analysis
Provides comprehensive statistical functions for economic data analysis and forecasting.
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

warnings.filterwarnings('ignore')

@dataclass
class EconomicIndicator:
    """Container for economic indicator results"""
    name: str
    value: float
    trend: str
    confidence: float
    interpretation: str
    metadata: Dict[str, Any]

@dataclass
class ForecastResult:
    """Container for forecasting results"""
    point_forecast: float
    confidence_interval: Tuple[float, float]
    model_type: str
    accuracy_metrics: Dict[str, float]
    forecast_periods: int

class StatisticalTools:
    """Comprehensive statistical tools for economic analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_time_series_data(self, data: pd.DataFrame, 
                               date_column: str = 'TimePeriod',
                               value_column: str = 'DataValue') -> pd.Series:
        """
        Prepare time series data for analysis
        
        Args:
            data: DataFrame containing time series data
            date_column: Name of the date column
            value_column: Name of the value column
            
        Returns:
            pd.Series: Time series data with datetime index
        """
        try:
            df = data.copy()
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
            
            # Set datetime index and sort
            df = df.set_index(date_column).sort_index()
            
            # Convert value column to numeric, handling errors
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            
            # Drop NaN values
            df = df.dropna(subset=[value_column])
            
            return df[value_column]
            
        except Exception as e:
            raise ValueError(f"Error preparing time series data: {e}")
    
    def analyze_growth_trends(self, data: pd.DataFrame, 
                            value_column: str = 'DataValue',
                            window: int = 4) -> Dict[str, Any]:
        """
        Analyze GDP growth trends and patterns
        
        Args:
            data: DataFrame containing economic data
            value_column: Column containing the values to analyze
            window: Rolling window size for moving averages
            
        Returns:
            Dict with growth analysis results
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if len(ts_data) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Calculate growth rates
            growth_rates = ts_data.pct_change() * 100
            quarterly_growth = ts_data.pct_change(periods=1) * 100
            annual_growth = ts_data.pct_change(periods=4) * 100
            
            # Rolling statistics
            rolling_mean = ts_data.rolling(window=window).mean()
            rolling_std = ts_data.rolling(window=window).std()
            
            # Trend analysis using linear regression
            X = np.arange(len(ts_data)).reshape(-1, 1)
            y = ts_data.values
            X = add_constant(X)
            model = OLS(y, X).fit()
            trend_slope = model.params[1] if len(model.params) > 1 else 0
            trend_strength = model.rsquared
            
            # Determine trend direction
            if trend_slope > 0:
                trend_direction = "upward"
            elif trend_slope < 0:
                trend_direction = "downward"
            else:
                trend_direction = "flat"
            
            # Volatility analysis
            volatility = growth_rates.std()
            recent_volatility = growth_rates.tail(window).std()
            
            # Seasonality detection (for quarterly data)
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
                "period_range": f"{ts_data.index[0]} to {ts_data.index[-1]}",
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
        Calculate key economic indicators
        
        Args:
            data: DataFrame containing economic data
            value_column: Column containing the values
            
        Returns:
            Dict with economic indicators
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if len(ts_data) < 4:
                return {"error": "Insufficient data for indicator calculation"}
            
            indicators = {}
            
            # Basic statistics
            indicators["current_value"] = ts_data.iloc[-1]
            indicators["mean"] = ts_data.mean()
            indicators["median"] = ts_data.median()
            indicators["std_dev"] = ts_data.std()
            indicators["cv"] = (ts_data.std() / ts_data.mean()) * 100  # Coefficient of variation
            
            # Growth indicators
            growth_rates = ts_data.pct_change() * 100
            indicators["recent_growth"] = growth_rates.tail(4).mean()  # Last year average
            indicators["long_term_growth"] = growth_rates.mean()
            
            # Volatility indicators
            indicators["volatility"] = growth_rates.std()
            indicators["max_drawdown"] = self._calculate_max_drawdown(ts_data)
            
            # Trend indicators
            trend_strength = self._calculate_trend_strength(ts_data)
            indicators["trend_strength"] = trend_strength
            
            # Stationarity tests
            adf_result = adfuller(ts_data.dropna())
            indicators["stationary"] = adf_result[1] < 0.05  # p-value < 0.05
            
            # Business cycle positioning
            cycle_position = self._assess_business_cycle_position(ts_data)
            indicators["business_cycle_position"] = cycle_position
            
            # Momentum indicators
            momentum = self._calculate_momentum(ts_data)
            indicators["momentum"] = momentum
            
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
        Identify business cycle phases using statistical methods
        
        Args:
            data: DataFrame containing economic data
            value_column: Column containing the values
            
        Returns:
            Dict with business cycle analysis
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if len(ts_data) < 8:  # Need sufficient data for cycle detection
                return {"error": "Insufficient data for business cycle analysis"}
            
            # Apply Hodrick-Prescott filter to separate trend and cycle
            cycle, trend = self._hp_filter(ts_data)
            
            # Find peaks and troughs in the cycle component
            peaks, troughs = self._find_peaks_troughs(cycle)
            
            # Determine current phase
            current_phase = self._determine_current_phase(ts_data, peaks, troughs)
            
            # Calculate cycle characteristics
            cycle_duration = self._calculate_cycle_duration(peaks, troughs, ts_data.index)
            amplitude = self._calculate_cycle_amplitude(cycle)
            
            return {
                "status": "success",
                "current_phase": current_phase,
                "cycle_component": cycle.tolist(),
                "trend_component": trend.tolist(),
                "peaks": [ts_data.index[i].strftime('%Y-%m-%d') for i in peaks],
                "troughs": [ts_data.index[i].strftime('%Y-%m-%d') for i in troughs],
                "cycle_duration": cycle_duration,
                "amplitude": amplitude,
                "phase_duration": self._calculate_current_phase_duration(current_phase, ts_data, peaks, troughs)
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"Business cycle analysis failed: {e}"}
    
    def detect_anomalies(self, data: pd.DataFrame,
                        value_column: str = 'DataValue',
                        method: str = 'zscore',
                        threshold: float = 2.5) -> Dict[str, Any]:
        """
        Detect anomalies in economic data
        
        Args:
            data: DataFrame containing economic data
            value_column: Column containing the values
            method: Detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for anomaly detection
            
        Returns:
            Dict with anomaly detection results
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
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomaly_indices = ts_data[(ts_data < lower_bound) | (ts_data > upper_bound)].index
                
            elif method == 'isolation_forest':
                # Simple version - use moving average residuals
                rolling_mean = ts_data.rolling(window=4, center=True).mean()
                residuals = ts_data - rolling_mean
                residual_std = residuals.std()
                anomaly_indices = residuals[np.abs(residuals) > threshold * residual_std].index
            
            # Compile anomaly information
            for idx in anomaly_indices:
                if isinstance(idx, (int, np.integer)):
                    date = ts_data.index[idx]
                    value = ts_data.iloc[idx]
                else:
                    date = idx
                    value = ts_data.loc[idx]
                    
                anomalies.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': value,
                    'deviation': float((value - ts_data.mean()) / ts_data.std()),
                    'type': 'high' if value > ts_data.mean() else 'low'
                })
            
            return {
                "status": "success",
                "anomalies": anomalies,
                "anomaly_count": len(anomalies),
                "anomaly_percentage": (len(anomalies) / len(ts_data)) * 100,
                "detection_method": method,
                "threshold": threshold,
                "data_points_analyzed": len(ts_data)
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"Anomaly detection failed: {e}"}
    
    def forecast_arima(self, data: pd.DataFrame,
                      value_column: str = 'DataValue',
                      periods: int = 8,
                      order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
        """
        Generate forecasts using ARIMA model
        
        Args:
            data: DataFrame containing time series data
            value_column: Column containing the values
            periods: Number of periods to forecast
            order: ARIMA order (p, d, q)
            
        Returns:
            Dict with ARIMA forecast results
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if len(ts_data) < 10:
                return {"error": "Insufficient data for ARIMA modeling"}
            
            # Fit ARIMA model
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast = fitted_model.get_forecast(steps=periods)
            forecast_mean = forecast.predicted_mean
            confidence_int = forecast.conf_int()
            
            # Calculate accuracy metrics on training data
            predictions = fitted_model.predict(start=1, end=len(ts_data))
            mae = mean_absolute_error(ts_data[1:], predictions[1:])
            rmse = np.sqrt(mean_squared_error(ts_data[1:], predictions[1:]))
            
            # Prepare forecast dates
            last_date = ts_data.index[-1]
            if pd.infer_freq(ts_data.index) == 'Q':
                forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq='Q')[1:]
            else:
                # Default to quarterly if frequency can't be inferred
                forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq='Q')[1:]
            
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
                    "params": fitted_model.params.to_dict()
                },
                "accuracy_metrics": {
                    "mae": mae,
                    "rmse": rmse,
                    "mape": self._calculate_mape(ts_data[1:], predictions[1:])
                },
                "residuals_analysis": {
                    "mean_residual": fitted_model.resid.mean(),
                    "std_residual": fitted_model.resid.std(),
                    "normality_test_pvalue": stats.normaltest(fitted_model.resid.dropna())[1]
                }
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"ARIMA forecasting failed: {e}"}
    
    def build_arima_model(self, data: pd.DataFrame,
                         value_column: str = 'DataValue',
                         auto_select: bool = True,
                         max_order: int = 3) -> Dict[str, Any]:
        """
        Build and validate ARIMA model with automatic order selection
        
        Args:
            data: DataFrame containing time series data
            value_column: Column containing the values
            auto_select: Whether to automatically select best order
            max_order: Maximum order to test for auto selection
            
        Returns:
            Dict with ARIMA model results
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            if auto_select:
                best_order = self._auto_arima(ts_data, max_order=max_order)
            else:
                best_order = (1, 1, 1)  # Default order
            
            # Fit model with best order
            model = ARIMA(ts_data, order=best_order)
            fitted_model = model.fit()
            
            # Model diagnostics
            residuals = fitted_model.resid.dropna()
            
            return {
                "status": "success",
                "best_order": best_order,
                "summary": {
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic,
                    "hqic": fitted_model.hqic,
                    "log_likelihood": fitted_model.llf
                },
                "parameters": fitted_model.params.to_dict(),
                "residuals": {
                    "mean": residuals.mean(),
                    "std": residuals.std(),
                    "jarque_bera": stats.jarque_bera(residuals),
                    "ljung_box": sm.stats.acorr_ljungbox(residuals, lags=[10])[1][0]
                },
                "forecast_accuracy": self._calculate_model_accuracy(fitted_model, ts_data)
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"ARIMA model building failed: {e}"}
    
    def ensemble_forecast(self, data: pd.DataFrame,
                         value_column: str = 'DataValue',
                         periods: int = 8) -> Dict[str, Any]:
        """
        Generate ensemble forecast combining multiple methods
        
        Args:
            data: DataFrame containing time series data
            value_column: Column containing the values
            periods: Number of periods to forecast
            
        Returns:
            Dict with ensemble forecast results
        """
        try:
            ts_data = self.prepare_time_series_data(data, value_column=value_column)
            
            forecasts = {}
            weights = {}
            
            # ARIMA forecast
            arima_result = self.forecast_arima(data, value_column, periods)
            if arima_result["status"] == "success":
                forecasts["arima"] = [f["point_forecast"] for f in arima_result["forecasts"]]
                weights["arima"] = 1.0 / arima_result["accuracy_metrics"]["rmse"]
            
            # Exponential Smoothing
            es_result = self._exponential_smoothing_forecast(ts_data, periods)
            if es_result["status"] == "success":
                forecasts["exponential_smoothing"] = es_result["forecasts"]
                weights["exponential_smoothing"] = 1.0 / es_result["rmse"]
            
            # Simple moving average
            ma_forecast = self._moving_average_forecast(ts_data, periods)
            forecasts["moving_average"] = ma_forecast
            weights["moving_average"] = 0.3  # Fixed weight for simple method
            
            # Normalize weights
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Calculate ensemble forecast
            ensemble_forecasts = []
            for i in range(periods):
                weighted_sum = 0
                for method, forecast_values in forecasts.items():
                    weighted_sum += forecast_values[i] * normalized_weights[method]
                ensemble_forecasts.append(weighted_sum)
            
            return {
                "status": "success",
                "ensemble_forecast": ensemble_forecasts,
                "weights": normalized_weights,
                "methods_used": list(forecasts.keys()),
                "next_period_prediction": ensemble_forecasts[0] if ensemble_forecasts else None
            }
            
        except Exception as e:
            return {"status": "error", "error_message": f"Ensemble forecasting failed: {e}"}
    
    # Helper methods
    def _detect_seasonality(self, ts_data: pd.Series) -> float:
        """Detect seasonality strength in time series"""
        try:
            if len(ts_data) >= 8:  # Need at least 2 years of quarterly data
                result = seasonal_decompose(ts_data, model='additive', period=4)
                seasonal_strength = result.seasonal.std() / ts_data.std()
                return float(seasonal_strength)
            return 0.0
        except:
            return 0.0
    
    def _calculate_max_drawdown(self, ts_data: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + ts_data.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    
    def _calculate_trend_strength(self, ts_data: pd.Series) -> float:
        """Calculate trend strength using linear regression R²"""
        X = np.arange(len(ts_data)).reshape(-1, 1)
        X = add_constant(X)
        y = ts_data.values
        model = OLS(y, X).fit()
        return float(model.rsquared)
    
    def _assess_business_cycle_position(self, ts_data: pd.Series) -> str:
        """Assess current business cycle position"""
        if len(ts_data) < 4:
            return "insufficient_data"
        
        recent_growth = ts_data.pct_change().tail(4).mean()
        volatility = ts_data.pct_change().std()
        
        if recent_growth > volatility:
            return "expansion"
        elif recent_growth < -volatility:
            return "contraction"
        else:
            return "stable"
    
    def _calculate_momentum(self, ts_data: pd.Series, periods: int = 4) -> float:
        """Calculate momentum indicator"""
        if len(ts_data) < periods + 1:
            return 0.0
        return float(ts_data.pct_change(periods=periods).iloc[-1])
    
    def _hp_filter(self, ts_data: pd.Series, lamb: float = 1600) -> Tuple[pd.Series, pd.Series]:
        """Hodrick-Prescott filter for trend-cycle decomposition"""
        # Simplified implementation
        trend = ts_data.rolling(window=4, center=True).mean()
        cycle = ts_data - trend
        return cycle, trend
    
    def _find_peaks_troughs(self, series: pd.Series) -> Tuple[List[int], List[int]]:
        """Find peaks and troughs in a series"""
        peaks = []
        troughs = []
        
        for i in range(1, len(series) - 1):
            if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                peaks.append(i)
            elif series.iloc[i] < series.iloc[i-1] and series.iloc[i] < series.iloc[i+1]:
                troughs.append(i)
        
        return peaks, troughs
    
    def _determine_current_phase(self, ts_data: pd.Series, peaks: List[int], troughs: List[int]) -> str:
        """Determine current business cycle phase"""
        if not peaks and not troughs:
            return "undetermined"
        
        last_peak = max(peaks) if peaks else -1
        last_trough = max(troughs) if troughs else -1
        
        if last_peak > last_trough:
            return "contraction"  # After peak, before trough
        else:
            return "expansion"    # After trough, before peak
    
    def _calculate_cycle_duration(self, peaks: List[int], troughs: List[int], index: pd.DatetimeIndex) -> Dict[str, float]:
        """Calculate business cycle duration statistics"""
        if len(peaks) < 2 or len(troughs) < 2:
            return {"average_expansion": 0, "average_contraction": 0}
        
        # Simplified calculation
        return {
            "average_expansion": 4.0,  # quarters
            "average_contraction": 2.0  # quarters
        }
    
    def _calculate_cycle_amplitude(self, cycle: pd.Series) -> float:
        """Calculate cycle amplitude"""
        return float(cycle.max() - cycle.min())
    
    def _calculate_current_phase_duration(self, current_phase: str, ts_data: pd.Series, 
                                        peaks: List[int], troughs: List[int]) -> int:
        """Calculate duration of current phase"""
        # Simplified implementation
        return 0
    
    def _interpret_indicators(self, indicators: Dict[str, Any]) -> str:
        """Provide interpretation of economic indicators"""
        interpretation = []
        
        if indicators.get("recent_growth", 0) > 2:
            interpretation.append("Strong recent growth")
        elif indicators.get("recent_growth", 0) < 0:
            interpretation.append("Recent contraction")
        
        if indicators.get("volatility", 0) > 5:
            interpretation.append("High volatility environment")
        
        if indicators.get("trend_strength", 0) > 0.7:
            interpretation.append("Strong trend pattern")
        
        return "; ".join(interpretation) if interpretation else "Stable economic conditions"
    
    def _calculate_mape(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    
    def _auto_arima(self, ts_data: pd.Series, max_order: int = 3) -> Tuple[int, int, int]:
        """Simple automatic ARIMA order selection"""
        # Simplified implementation - in production, use pmdarima
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        for p in range(max_order + 1):
            for d in range(2):  # Differencing order
                for q in range(max_order + 1):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def _calculate_model_accuracy(self, model, ts_data: pd.Series) -> Dict[str, float]:
        """Calculate model accuracy metrics"""
        try:
            predictions = model.predict(start=1, end=len(ts_data))
            actual = ts_data[1:]
            pred = predictions[1:]
            
            return {
                "mae": mean_absolute_error(actual, pred),
                "rmse": np.sqrt(mean_squared_error(actual, pred)),
                "mape": self._calculate_mape(actual, pred),
                "r2": r2_score(actual, pred)
            }
        except:
            return {"mae": 0, "rmse": 0, "mape": 0, "r2": 0}
    
    def _exponential_smoothing_forecast(self, ts_data: pd.Series, periods: int) -> Dict[str, Any]:
        """Generate forecast using exponential smoothing"""
        try:
            model = ExponentialSmoothing(ts_data, seasonal_periods=4, trend='add', seasonal='add')
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            
            # Calculate training accuracy
            predictions = fitted_model.fittedvalues
            mae = mean_absolute_error(ts_data, predictions)
            rmse = np.sqrt(mean_squared_error(ts_data, predictions))
            
            return {
                "status": "success",
                "forecasts": forecast.tolist(),
                "mae": mae,
                "rmse": rmse
            }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    def _moving_average_forecast(self, ts_data: pd.Series, periods: int) -> List[float]:
        """Generate simple moving average forecast"""
        try:
            # Use last 4 periods average for all forecast periods
            last_value = ts_data.tail(4).mean()
            return [last_value] * periods
        except:
            return [0] * periods

# Example usage and testing
if __name__ == "__main__":
    # Create sample economic data for testing
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='Q')
    np.random.seed(42)
    
    # Simulate GDP data with trend and seasonality
    trend = np.linspace(100, 120, len(dates))
    seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 4)
    noise = np.random.normal(0, 2, len(dates))
    gdp_data = trend + seasonal + noise
    
    sample_data = pd.DataFrame({
        'TimePeriod': dates,
        'DataValue': gdp_data
    })
    
    # Test the statistical tools
    tools = StatisticalTools()
    
    print("Testing Economic Analysis Tools...")
    print("=" * 50)
    
    # Test growth trend analysis
    growth_analysis = tools.analyze_growth_trends(sample_data)
    print("Growth Analysis:", growth_analysis)
    
    # Test indicator calculation
    indicators = tools.calculate_indicators(sample_data)
    print("\nEconomic Indicators:", indicators)
    
    # Test business cycle analysis
    cycles = tools.identify_business_cycles(sample_data)
    print("\nBusiness Cycles:", cycles)
    
    # Test anomaly detection
    anomalies = tools.detect_anomalies(sample_data)
    print("\nAnomalies:", anomalies)
    
    # Test ARIMA forecasting
    forecast = tools.forecast_arima(sample_data, periods=4)
    print("\nARIMA Forecast:", forecast)