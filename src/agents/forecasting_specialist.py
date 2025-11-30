import pandas as pd
import logging
from ..tools.statistical_tools import StatisticalTools

logger = logging.getLogger(__name__)

# Robust Tool Wrapper (Same as we used for Economic Analyst)
class ForecastingTool:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

class ForecastingSpecialistAgent:
    """
    Agent responsible for time series forecasting and model selection.
    """
    def __init__(self, model):
        self.model = model
        self.stat_tools = StatisticalTools()
        
        self.tools = [
            ForecastingTool(name="forecast_gdp", fn=self.forecast_gdp),
            ForecastingTool(name="build_arima_model", fn=self.build_arima_model),
            ForecastingTool(name="generate_ensemble_forecast", fn=self.generate_ensemble_forecast)
        ]
        
        self.agent = type('obj', (object,), {'tools': self.tools})
        logger.info("ForecastingSpecialistAgent initialized.")

    def forecast_gdp(self, data: list, horizon=4):
        """
        Generates GDP forecasts for a specified horizon using ARIMA.
        """
        try:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            result = self.stat_tools.forecast_arima(df, periods=horizon)
            
            if result['status'] == 'success':
                # Add simplified summary for the LLM/User
                next_val = result['forecasts'][0]['point_forecast']
                result['next_quarter_prediction'] = f"{next_val:,.2f}"
                result['horizon'] = horizon
                result['confidence'] = 0.95
            return result
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def build_arima_model(self, data: list):
        """
        Builds and evaluates an ARIMA model on the data.
        """
        try:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            return self.stat_tools.build_arima_model(df)
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def generate_ensemble_forecast(self, data: list):
        """
        Generates an ensemble forecast combining multiple methods.
        """
        try:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            return self.stat_tools.ensemble_forecast(df, periods=4)
        except Exception as e:
            return {'status': 'error', 'message': str(e)}