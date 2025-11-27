# Import necessary libraries and modules
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from typing import Dict, List
import pandas as pd
import numpy as np
# Import the statistical tools used for analysis
# Note: The 'statistical_tools.py' file needs to be created and implemented.
from ..tools.statistical_tools import StatisticalTools

class EconomicAnalystAgent:
    """
    An agent that performs economic analysis on time-series data.
    
    This agent uses a suite of statistical tools to analyze trends, calculate
    indicators, identify business cycles, and detect anomalies in economic data.
    """
    
    def __init__(self, model: Gemini):
        """
        Initializes the EconomicAnalystAgent.
        
        Args:
            model: An instance of a Gemini model to power the LlmAgent.
        """
        self.stat_tools = StatisticalTools()
        self.agent = LlmAgent(
            name="economic_analyst",
            model=model,
            instruction="""You are an expert economic analyst specializing in macroeconomic trends.
            
            Your responsibilities:
            1. Analyze GDP growth trends and components.
            2. Identify business cycle patterns from time-series data.
            3. Calculate key economic indicators.
            4. Provide insights on economic health based on data.
            5. Detect anomalies and structural breaks in economic data.
            
            Use your statistical tools for rigorous analysis and provide data-driven insights.""",
            tools=[
                FunctionTool(func=self.analyze_growth_trends),
                FunctionTool(func=self.calculate_economic_indicators),
                FunctionTool(func=self.identify_business_cycles),
                FunctionTool(func=self.detect_anomalies)
            ]
        )
    
    def analyze_growth_trends(self, data: List[Dict]) -> Dict:
        """
        Analyzes growth trends and patterns in the provided time-series data.
        
        Args:
            data: A list of dictionaries representing time-series data, 
                  e.g., [{'date': '2023-01-01', 'value': 100}, ...].
                  
        Returns:
            A dictionary containing the analysis results.
        """
        # Input validation and error handling
        if not data:
            return {"status": "error", "message": "Input data cannot be empty."}
        
        try:
            df = pd.DataFrame(data)
            # Assuming 'date' and 'value' columns exist for analysis
            analysis = self.stat_tools.analyze_growth_trends(df)
            return {
                "status": "success",
                "analysis": analysis,
                "trend": analysis.get("trend_direction", "unknown"),
                "confidence": analysis.get("confidence", 0.0)
            }
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    def calculate_economic_indicators(self, data: List[Dict]) -> Dict:
        """
        Calculates key economic indicators from the provided data.
        
        Args:
            data: A list of dictionaries representing time-series data.
            
        Returns:
            A dictionary containing the calculated indicators.
        """
        if not data:
            return {"status": "error", "message": "Input data cannot be empty."}
            
        try:
            df = pd.DataFrame(data)
            indicators = self.stat_tools.calculate_indicators(df)
            return {
                "status": "success",
                "indicators": indicators,
                "message": "Calculated key economic indicators"
            }
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    def identify__business_cycles(self, data: List[Dict]) -> Dict:
        """
        Identifies business cycle phases (e.g., expansion, contraction) from the data.
        
        Args:
            data: A list of dictionaries representing time-series data.
            
        Returns:
            A dictionary containing the business cycle analysis.
        """
        if not data:
            return {"status": "error", "message": "Input data cannot be empty."}
            
        try:
            df = pd.DataFrame(data)
            cycles = self.stat_tools.identify_business_cycles(df)
            return {
                "status": "success",
                "business_cycles": cycles,
                "current_phase": cycles.get("current_phase", "unknown") if cycles else "unknown"
            }
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    def detect_anomalies(self, data: List[Dict]) -> Dict:
        """
        Detects anomalies or outliers in the economic data.
        
        Args:
            data: A list of dictionaries representing time-series data.
            
        Returns:
            A dictionary containing the detected anomalies.
        """
        if not data:
            return {"status": "error", "message": "Input data cannot be empty."}
            
        try:
            df = pd.DataFrame(data)
            anomalies = self.stat_tools.detect_anomalies(df)
            return {
                "status": "success",
                "anomalies": anomalies,
                "anomaly_count": len(anomalies.get("anomalies", []))
            }
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": f"Error processing data: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}
