# Import necessary libraries and modules
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from typing import Dict, List
import pandas as pd
# Import the StatisticalTools class from the correct module
from ..tools.statistical_tools import StatisticalTools

class ForecastingSpecialistAgent:
    """
    An agent that specializes in generating and evaluating economic forecasts.
    
    This agent uses tools from the StatisticalTools class to perform tasks like
    building ARIMA models, generating forecasts, and evaluating their accuracy.
    """
    
    def __init__(self, model: Gemini):
        """
        Initializes the ForecastingSpecialistAgent.
        
        Args:
            model: An instance of a Gemini model to power the LlmAgent.
        """
        self.stat_tools = StatisticalTools()
        self.agent = LlmAgent(
            name="forecasting_specialist", 
            model=model,
            instruction="""You are an expert economic forecaster.
            
            Your responsibilities:
            1. Build and validate time-series models like ARIMA.
            2. Generate forecasts for economic data with confidence intervals.
            3. Create ensemble forecasts by combining multiple models.
            4. Evaluate the accuracy of forecasts against actual data.
            
            Always use robust statistical methods, validate your models, and clearly
            state the uncertainty of your predictions.""",
            tools=[
                FunctionTool(func=self.generate_arima_forecast),
                FunctionTool(func=self.build_and_validate_arima_model),
                FunctionTool(func=self.generate_ensemble_forecast),
                FunctionTool(func=self.evaluate_forecast_accuracy)
            ]
        )
    
    def generate_arima_forecast(self, data: List[Dict], periods: int = 8) -> Dict:
        """
        Generates a forecast using an ARIMA model.
        
        Args:
            data: A list of dictionaries representing time-series data.
            periods: The number of future periods to forecast.
            
        Returns:
            A dictionary containing the forecast results.
        """
        if not data:
            return {"status": "error", "message": "Input data cannot be empty."}
            
        try:
            df = pd.DataFrame(data)
            # Note: This uses a default ARIMA order. For better results, 
            # build_and_validate_arima_model should be called first to find the best order.
            forecast_result = self.stat_tools.forecast_arima(df, periods=periods)
            return forecast_result
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    def build_and_validate_arima_model(self, data: List[Dict]) -> Dict:
        """
        Builds and validates an ARIMA model, automatically selecting the best order.
        
        Args:
            data: A list of dictionaries representing time-series data.
            
        Returns:
            A dictionary containing the model summary and diagnostics.
        """
        if not data:
            return {"status": "error", "message": "Input data cannot be empty."}
            
        try:
            df = pd.DataFrame(data)
            model_results = self.stat_tools.build_arima_model(df)
            return model_results
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    def generate_ensemble_forecast(self, data: List[Dict], periods: int = 8) -> Dict:
        """
        Generates an ensemble forecast by combining multiple model types.
        
        Args:
            data: A list of dictionaries representing time-series data.
            periods: The number of future periods to forecast.

        Returns:
            A dictionary containing the ensemble forecast.
        """
        if not data:
            return {"status": "error", "message": "Input data cannot be empty."}
            
        try:
            df = pd.DataFrame(data)
            # The ensemble_forecast method is not yet fully implemented in StatisticalTools.
            # Returning a placeholder message.
            return {"status": "info", "message": "Ensemble forecast tool is not fully implemented yet."}
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    def evaluate_forecast_accuracy(self, historical_data: List[Dict], 
                                       forecast_data: List[Dict]) -> Dict:
        """
        Evaluates the accuracy of a forecast by comparing it to actual historical data.
        
        Args:
            historical_data: A list of dictionaries with the actual historical values.
            forecast_data: A list of dictionaries with the forecasted values.
            
        Returns:
            A dictionary containing accuracy metrics like MAE, RMSE, and MAPE.
        """
        if not historical_data or not forecast_data:
            return {"status": "error", "message": "Input data for both historical and forecast cannot be empty."}
            
        try:
            hist_df = pd.DataFrame(historical_data)
            forecast_df = pd.DataFrame(forecast_data)
            accuracy = self.stat_tools.evaluate_forecast_accuracy(hist_df, forecast_df)
            return accuracy
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}
