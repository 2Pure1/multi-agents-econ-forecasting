# Import necessary libraries and modules
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from typing import Dict, List
import pandas as pd
# Import the visualization tools
from ..tools.visualization_tools import VisualizationTools

class VisualizationAgent:
    """
    An agent responsible for creating and exporting visualizations of economic data.
    
    This agent uses tools from the VisualizationTools class to generate plots
    and then returns them as serializable dictionaries.
    """
    
    def __init__(self, model: Gemini):
        """
        Initializes the VisualizationAgent.
        
        Args:
            model: An instance of a Gemini model to power the LlmAgent.
        """
        self.viz_tools = VisualizationTools()
        self.agent = LlmAgent(
            name="visualization_agent",
            model=model,
            instruction="""You are a data visualization specialist for economic data.
            
            Your responsibilities:
            1. Create time series plots with trends and forecasts.
            2. Generate charts for growth analysis.
            3. Export visualizations to different file formats like HTML or PNG.
            
            Focus on creating clear, accurate, and professional-looking charts.
            Return plots as JSON data that can be exported.""",
            tools=[
                FunctionTool(func=self.plot_forecast),
                FunctionTool(func=self.plot_growth_analysis),
                FunctionTool(func=self.export_visualization)
            ]
        )
    
    def plot_forecast(self, historical_data: List[Dict], forecast_data: Dict) -> Dict:
        """
        Creates a plot of historical data with forecasted values.
        
        Args:
            historical_data: A list of dictionaries with the historical time-series data.
            forecast_data: A dictionary containing the forecast results from the forecasting agent.
            
        Returns:
            A dictionary containing the status and the plot's dictionary representation.
        """
        if not historical_data:
            return {"status": "error", "message": "Historical data cannot be empty."}
            
        try:
            hist_df = pd.DataFrame(historical_data)
            # Generate the plot using the visualization tool
            fig = self.viz_tools.plot_forecasts(hist_df, forecast_data)
            return {
                "status": "success",
                "figure": fig.to_dict(), # Return the figure as a serializable dictionary
                "message": "Forecast visualization created successfully."
            }
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    def plot_growth_analysis(self, data: List[Dict]) -> Dict:
        """
        Creates a bar chart analyzing the growth rate of the provided data.
        
        Args:
            data: A list of dictionaries representing time-series data.
            
        Returns:
            A dictionary containing the status and the plot's dictionary representation.
        """
        if not data:
            return {"status": "error", "message": "Input data cannot be empty."}
            
        try:
            df = pd.DataFrame(data)
            # Generate the chart using the visualization tool
            chart = self.viz_tools.plot_growth_chart(df)
            return {
                "status": "success",
                "figure": chart.to_dict(), # Return the figure as a serializable dictionary
                "message": "Growth analysis chart created successfully."
            }
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    def export_visualization(self, figure_dict: Dict, output_path: str) -> Dict:
        """
        Exports a visualization to a specified file format (e.g., HTML, PNG).
        
        Args:
            figure_dict: The dictionary representation of a Plotly figure.
            output_path: The path to save the file, e.g., 'forecast.html'.
            
        Returns:
            A dictionary containing the status and the path of the exported file.
        """
        if not figure_dict or not output_path:
            return {"status": "error", "message": "Figure data and output path cannot be empty."}
            
        try:
            # Export the figure using the visualization tool
            result = self.viz_tools.export_visualization(figure_dict, output_path)
            return result
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred during export: {e}"}
