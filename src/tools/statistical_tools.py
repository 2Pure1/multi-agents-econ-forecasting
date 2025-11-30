import pandas as pd
import numpy as np
from scipy import stats, signal
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import logging
import warnings

# Suppress warnings for cleaner notebook output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalTools:
    """
    A collection of statistical methods for economic data analysis and forecasting.
    Includes robust date handling and error safety.
    """

    # --- PART 1: ANALYSIS METHODS ---
    def analyze_growth_trends(self, df):
        try:
            if 'DataValue' not in df.columns: return {'status': 'error', 'message': 'Missing DataValue'}
            data = df['DataValue'].dropna()
            if len(data) < 4: return {'status': 'error', 'message': 'Insufficient data'}
            
            cycle, trend = sm.tsa.filters.hpfilter(data, 1600)
            return {
                'status': 'success',
                'current_growth_rate': data.pct_change().iloc[-1] * 100,
                'average_growth_rate': data.pct_change().mean() * 100,
                'trend_direction': "Upward" if trend.iloc[-1] > trend.iloc[-2] else "Downward",
                'trend_strength': np.std(trend) / np.std(data),
                'volatility': data.pct_change().std() * 100,
                'seasonal_strength': np.std(cycle) / np.std(data)
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def calculate_indicators(self, df):
        try:
            data = df['DataValue'].dropna()
            stats_dict = {
                'current_value': data.iloc[-1],
                'mean': data.mean(),
                'std_dev': data.std(),
                'volatility': (data.pct_change().std() * 100) if len(data) > 1 else 0
            }
            if len(data) >= 4:
                stats_dict['momentum'] = data.diff(4).iloc[-1]
            return {'status': 'success', 'indicators': stats_dict}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def identify_business_cycles(self, df):
        try:
            data = df['DataValue'].dropna().values
            cycle, trend = sm.tsa.filters.hpfilter(data, 1600)
            peaks, _ = signal.find_peaks(cycle, distance=4)
            troughs, _ = signal.find_peaks(-cycle, distance=4)
            
            current_phase = "Stable"
            if len(peaks) > 0 and len(troughs) > 0:
                current_phase = "Contraction" if peaks[-1] > troughs[-1] else "Expansion"

            return {
                'status': 'success',
                'current_phase': current_phase,
                'peaks': peaks.tolist(),
                'troughs': troughs.tolist(),
                'cycle_component': cycle.tolist(),
                'amplitude': np.ptp(cycle)
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def detect_anomalies(self, df, method='zscore', threshold=2.0):
        try:
            data = df['DataValue'].dropna()
            z_scores = np.abs(stats.zscore(data))
            anomalies = []
            anomaly_indices = np.where(z_scores > threshold)[0]
            dates = df['TimePeriod'].dropna() if 'TimePeriod' in df.columns else pd.Series(range(len(data)))
            
            for idx in anomaly_indices:
                anomalies.append({
                    'date': dates.iloc[idx],
                    'value': data.iloc[idx],
                    'deviation': z_scores[idx],
                    'type': 'Statistical Outlier'
                })
            return {
                'status': 'success', 
                'anomaly_count': len(anomalies), 
                'anomaly_percentage': (len(anomalies)/len(data))*100,
                'anomalies': anomalies
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    # --- PART 2: FORECASTING METHODS ---
    
    def build_arima_model(self, df, auto_select=True, max_order=3):
        try:
            data = df['DataValue'].dropna()
            # Simple grid search 
            best_aic = float('inf')
            best_order = (1, 1, 1)
            best_model_res = None

            orders = [(1,1,1), (1,1,0), (0,1,1)] if auto_select else [(1,1,1)]

            for order in orders:
                try:
                    model = ARIMA(data, order=order)
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = order
                        best_model_res = res
                except:
                    continue
            
            if best_model_res is None:
                model = ARIMA(data, order=(1,1,0))
                best_model_res = model.fit()

            return {
                'status': 'success',
                'best_order': best_order,
                'summary': {
                    'aic': best_model_res.aic,
                    'bic': best_model_res.bic,
                    'log_likelihood': best_model_res.llf
                },
                'parameters': best_model_res.params.to_dict(),
                'forecast_accuracy': {
                    'mae': np.mean(np.abs(best_model_res.resid)),
                    'rmse': np.sqrt(np.mean(best_model_res.resid**2))
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def forecast_arima(self, df, periods=4):
        """
        Generates forecasts using ARIMA with SAFE date handling.
        """
        try:
            data = df['DataValue'].dropna()
            model = ARIMA(data, order=(1,1,1))
            model_fit = model.fit()
            
            forecast_res = model_fit.get_forecast(steps=periods)
            summary = forecast_res.summary_frame(alpha=0.05)
            
            forecasts = []
            last_date = df['TimePeriod'].max() if 'TimePeriod' in df.columns else None
            
            # --- DATE GENERATION FIX ---
            # Instead of adding offsets in a loop (which can overflow), we use date_range
            if last_date:
                # Generate future dates efficiently
                # freq='Q' matches standard GDP quarterly data
                future_dates = pd.date_range(start=last_date, periods=periods+1, freq='Q')[1:]
            else:
                future_dates = list(range(periods))

            for i, (_, row) in enumerate(summary.iterrows()):
                # Safety check: Ensure we don't go out of bounds of our date list
                date_val = future_dates[i] if i < len(future_dates) else f"Period {i+1}"
                
                forecasts.append({
                    'period_ahead': i + 1,
                    'period': date_val,
                    'point_forecast': row['mean'],
                    'confidence_lower': row['mean_ci_lower'],
                    'confidence_upper': row['mean_ci_upper']
                })

            return {'status': 'success', 'forecasts': forecasts}
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def ensemble_forecast(self, df, periods=4):
        try:
            data = df['DataValue'].dropna()
            
            # 1. ARIMA
            arima_res = self.forecast_arima(df, periods)
            if arima_res['status'] == 'success':
                arima_preds = [f['point_forecast'] for f in arima_res['forecasts']]
            else:
                arima_preds = [data.mean()] * periods

            # 2. Naive
            naive_preds = [data.iloc[-1]] * periods

            # 3. Moving Average
            ma_val = data.tail(4).mean()
            ma_preds = [ma_val] * periods

            # Weighted Ensemble
            ensemble_preds = []
            for i in range(periods):
                val = (arima_preds[i] * 0.5) + (ma_preds[i] * 0.3) + (naive_preds[i] * 0.2)
                ensemble_preds.append(val)

            return {
                'status': 'success',
                'methods_used': ['ARIMA', 'Moving Average', 'Naive'],
                'weights': {'ARIMA': 0.5, 'Moving Average': 0.3, 'Naive': 0.2},
                'ensemble_forecast': ensemble_preds
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}