import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import os

class VisualizationTools:
    """
    A collection of Plotly visualization methods for economic data.
    """

    def create_trend_chart(self, data: list) -> go.Figure:
        """Creates a multi-line trend chart for GDP and Inflation."""
        try:
            df = pd.DataFrame(data)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # GDP Line
            if 'GDP' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['GDP'], name="GDP ($B)", line=dict(color='#1f77b4', width=3)),
                    secondary_y=False
                )
            
            # Inflation Line
            if 'Inflation' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Inflation'], name="Inflation (%)", line=dict(color='#d62728', dash='dot')),
                    secondary_y=True
                )
            
            fig.update_layout(title_text="Economic Trends (GDP vs Inflation)", height=400)
            return fig
        except Exception as e:
            return self._create_error_chart(str(e))

    def create_health_gauge(self, current_score: float) -> go.Figure:
        """Creates a gauge chart for the Economic Health Score."""
        try:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_score,
                title = {'text': "Economic Health"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}
                }
            ))
            fig.update_layout(height=300)
            return fig
        except Exception as e:
            return self._create_error_chart(str(e))

    def create_dashboard_layout(self, data: list) -> go.Figure:
        """Creates the full composite 4-panel dashboard."""
        try:
            df = pd.DataFrame(data)
            latest = df.iloc[-1]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("GDP Trend", "Unemployment", "Inflation", "Health Score"),
                specs=[[{}, {}], [{}, {"type": "indicator"}]]
            )

            # 1. GDP
            if 'GDP' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['GDP'], name="GDP", line=dict(color='blue')), row=1, col=1)
            
            # 2. Unemployment
            if 'Unemployment' in df.columns:
                fig.add_trace(go.Bar(x=df['Date'], y=df['Unemployment'], name="Unemployment", marker_color='orange'), row=1, col=2)
            
            # 3. Inflation
            if 'Inflation' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Inflation'], name="Inflation", fill='tozeroy'), row=2, col=1)
            
            # 4. Health Gauge
            if 'Health_Score' in df.columns:
                fig.add_trace(go.Indicator(
                    mode="gauge+number", value=latest['Health_Score'],
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
                ), row=2, col=2)

            fig.update_layout(height=800, title_text="Executive Economic Dashboard", showlegend=False)
            return fig
        except Exception as e:
            return self._create_error_chart(str(e))

    def _create_error_chart(self, error_msg):
        """Returns a blank chart with error text."""
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {error_msg}", x=0.5, y=0.5, showarrow=False)
        return fig

    def export_visualization(self, fig, file_path: str) -> Dict:
        """Exports a Plotly figure to a file."""
        try:
            # Check if fig is a dict (serialized) or Figure object
            if isinstance(fig, dict):
                fig = go.Figure(fig)
                
            if file_path.endswith(".html"):
                fig.write_html(file_path)
            elif file_path.endswith(".json"):
                fig.write_json(file_path)
            
            return {"status": "success", "path": file_path}
        except Exception as e:
            return {"status": "error", "message": str(e)}