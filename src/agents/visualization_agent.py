from ..tools.visualization_tools import VisualizationTools
import pandas as pd

# Wrapper class for tools (Robust approach)
class VizToolWrapper:
    def __init__(self, name, fn):
        self.name = name
        self.func = fn # ADK uses 'func', older versions might use 'fn'

class VisualizationAgent:
    """
    Agent responsible for creating charts and dashboards using VisualizationTools.
    """
    def __init__(self, model):
        self.model = model
        self.viz_tools = VisualizationTools()
        
        # Define tools using the wrapper to be compatible with ADK patterns
        self.tools = [
            VizToolWrapper(name="create_trend_chart", fn=self.create_trend_chart),
            VizToolWrapper(name="create_health_gauge", fn=self.create_health_gauge),
            VizToolWrapper(name="create_dashboard_layout", fn=self.create_dashboard_layout),
            VizToolWrapper(name="export_visualization", fn=self.export_visualization)
        ]
        
        # Expose tools list for the coordinator
        self.agent = type('obj', (object,), {'tools': self.tools})

    def create_trend_chart(self, data):
        # Convert dictionary list back to list if needed
        return self.viz_tools.create_trend_chart(data)

    def create_health_gauge(self, score):
        return self.viz_tools.create_health_gauge(score)

    def create_dashboard_layout(self, data):
        return self.viz_tools.create_dashboard_layout(data)

    def export_visualization(self, figure, filename):
        return self.viz_tools.export_visualization(figure, filename)