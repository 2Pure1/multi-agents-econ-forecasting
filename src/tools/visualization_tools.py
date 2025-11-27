import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import os

class VisualizationTools:
    """
    A class for creating standard economic data visualizations using Plotly.
    
    The methods in this class generate Plotly figure objects.
    """

    def plot_forecasts(self, historical_df: pd.DataFrame, forecast_data: Dict) -> go.Figure:
        """
        Plots historical data along with forecasted values and confidence intervals.
        
        Args:
            historical_df: DataFrame containing the historical data. It should have 
                           'TimePeriod' and 'DataValue' columns.
            forecast_data: A dictionary containing forecast results, including a 
                           'forecasts' key with a list of forecast points.
                           
        Returns:
            A Plotly Figure object.
        """
        fig = go.Figure()
        
        # Add a trace for the historical data
        fig.add_trace(go.Scatter(
            x=historical_df['TimePeriod'], 
            y=historical_df['DataValue'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add a trace for the forecasted data
        if forecast_data.get("status") == "success" and forecast_data.get('forecasts'):
            forecast_df = pd.DataFrame(forecast_data['forecasts'])
            fig.add_trace(go.Scatter(
                x=forecast_df['period'],
                y=forecast_df['point_forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            # Add a shaded area for the confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['period'].tolist() + forecast_df['period'].tolist()[::-1],
                y=forecast_df['confidence_upper'].tolist() + forecast_df['confidence_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title="Economic Forecast",
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Legend"
        )
        return fig

    def plot_growth_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Creates a bar chart showing the percentage growth rate of the data.
        
        Args:
            df: DataFrame with 'TimePeriod' and 'DataValue' columns.
            
        Returns:
            A Plotly Figure object.
        """
        # Calculate percentage change for growth rate
        df['growth'] = df['DataValue'].pct_change() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['TimePeriod'],
            y=df['growth'],
            name='Quarterly Growth Rate'
        ))
        
        fig.update_layout(
            title="Growth Rate Analysis",
            xaxis_title="Date",
            yaxis_title="Growth Rate (%)"
        )
        return fig

    def export_visualization(self, fig_dict: Dict, file_path: str) -> Dict:
        """
        Exports a Plotly figure (from its dictionary representation) to a file.
        
        Args:
            fig_dict: The dictionary representation of a Plotly figure.
            file_path: The path to save the file to (e.g., 'figure.html', 'figure.png').
            
        Returns:
            A dictionary with the status and path of the exported file.
        """
        try:
            # Recreate the figure from the dictionary
            fig = go.Figure(fig_dict)
            
            # Ensure the output directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Write the file based on the extension
            if file_path.endswith(".html"):
                fig.write_html(file_path)
            elif file_path.endswith(".png"):
                fig.write_image(file_path)
            elif file_path.endswith(".json"):
                fig.write_json(file_path)
            else:
                return {"status": "error", "message": "Unsupported format. Use .html, .png, or .json"}
                
            return {"status": "success", "path": os.path.abspath(file_path)}
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to export visualization: {e}"}
