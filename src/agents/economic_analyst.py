import pandas as pd
import logging
from ..tools.statistical_tools import StatisticalTools

# Configure logging
logger = logging.getLogger(__name__)

# --- ROBUSTNESS FIX ---
# Instead of relying on 'google.genai.types.FunctionTool' which changes between versions,
# we define a local Tool wrapper. This guarantees the class never crashes on import.
class AnalysisTool:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

class EconomicAnalystAgent:
    """
    Agent responsible for performing deep economic analysis using statistical tools.
    Acts as the bridge between the LLM and the raw data processing logic.
    """
    
    def __init__(self, model):
        self.model = model
        self.stat_tools = StatisticalTools()
        
        # Define the tools available to the agent using our robust wrapper
        self.tools = [
            AnalysisTool(name="analyze_growth_trends", fn=self.analyze_growth_trends),
            AnalysisTool(name="calculate_economic_indicators", fn=self.calculate_economic_indicators),
            AnalysisTool(name="identify_business_cycles", fn=self.identify_business_cycles),
            AnalysisTool(name="detect_anomalies", fn=self.detect_anomalies)
        ]
        
        # 'agent' attribute used for inspection in the notebook
        self.agent = type('obj', (object,), {'tools': self.tools})
        
        logger.info("EconomicAnalystAgent initialized with robust AnalysisTools.")

    def analyze_growth_trends(self, data: list):
        """
        Analyzes GDP growth trends, volatility, and direction.
        Args:
            data: List of dictionaries containing economic data.
        """
        try:
            # Robustness: Ensure data is DataFrame
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            return self.stat_tools.analyze_growth_trends(df)
        except Exception as e:
            logger.error(f"Error in analyze_growth_trends: {e}")
            return {"status": "error", "message": str(e)}

    def calculate_economic_indicators(self, data: list):
        """
        Calculates key statistical indicators (mean, std dev, momentum, etc.).
        """
        try:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            return self.stat_tools.calculate_indicators(df)
        except Exception as e:
            logger.error(f"Error in calculate_economic_indicators: {e}")
            return {"status": "error", "message": str(e)}

    def identify_business_cycles(self, data: list):
        """
        Identifies business cycle phases (expansion, contraction) and peaks/troughs.
        """
        try:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            
            # Robustness: Handle differing column names from various API responses
            if 'DataValue' not in df.columns and 'value' in df.columns:
                df = df.rename(columns={'value': 'DataValue'})
                
            return self.stat_tools.identify_business_cycles(df)
        except Exception as e:
            logger.error(f"Error in identify_business_cycles: {e}")
            return {"status": "error", "message": str(e)}

    def detect_anomalies(self, data: list):
        """
        Detects statistical anomalies or outliers in the economic data series.
        """
        try:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            return self.stat_tools.detect_anomalies(df)
        except Exception as e:
            logger.error(f"Error in detect_anomalies: {e}")
            return {"status": "error", "message": str(e)}