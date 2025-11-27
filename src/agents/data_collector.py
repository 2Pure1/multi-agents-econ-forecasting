# Import necessary libraries and modules
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from typing import Dict, List
import pandas as pd
# Import the BEAClient and the response processing function from the bea_client module
from ..tools.bea_client import BEAClient, process_bea_response

class DataCollectorAgent:
    """
    An agent responsible for collecting economic data using the BEAClient.
    
    This agent is designed to be part of a multi-agent system. It exposes
    data-fetching capabilities as tools that can be called by a language model.
    """
    
    def __init__(self, bea_api_key: str, model: Gemini):
        """
        Initializes the DataCollectorAgent.
        
        Args:
            bea_api_key: The API key for the BEA API.
            model: An instance of a Gemini model to power the LlmAgent.
        """
        self.bea_client = BEAClient(bea_api_key)
        self.agent = LlmAgent(
            name="data_collector",
            model=model,
            instruction="""You are an economic data collector specializing in U.S. economic data from the BEA.
            
            Your responsibilities:
            1. Collect Gross Domestic Product (GDP) data.
            2. Gather Government Current Receipts and Expenditures data.
            3. Retrieve inflation and price index data.
            4. Return structured data for analysis.
            
            Always check data availability and handle API errors gracefully.
            Return data in a clean, analyzable format.""",
            tools=[
                FunctionTool(func=self.get_gdp_data),
                FunctionTool(func=self.get_gov_receipts_expenditures_data),
                FunctionTool(func=self.get_inflation_data)
            ]
        )
    
    def get_gdp_data(self) -> Dict:
        """
        Fetches GDP data using the BEAClient and formats it as a dictionary.
        
        Returns:
            A dictionary containing the status, data, and a message.
        """
        # Call the synchronous method from BEAClient
        response = self.bea_client.get_gdp_data()
        
        # Check if the API call was successful before processing the data
        if response.get("status") == "error":
            return response
            
        df = process_bea_response(response)
        return {
            "status": "success",
            "data": df.to_dict('records') if not df.empty else [],
            "message": f"Retrieved {len(df)} GDP data points"
        }
    
    def get_gov_receipts_expenditures_data(self) -> Dict:
        """
        Fetches Government Receipts and Expenditures data and formats it.
        
        Returns:
            A dictionary containing the status, data, and a message.
        """
        # Call the synchronous method from BEAClient
        response = self.bea_client.get_gov_receipts_expenditures_data()
        
        # Check for errors before processing
        if response.get("status") == "error":
            return response
            
        df = process_bea_response(response)
        return {
            "status": "success", 
            "data": df.to_dict('records') if not df.empty else [],
            "message": f"Retrieved {len(df)} government receipts and expenditures data points"
        }
    
    def get_inflation_data(self) -> Dict:
        """
        Fetches inflation data and formats it as a dictionary.
        
        Returns:
            A dictionary containing the status, data, and a message.
        """
        # Call the synchronous method from BEAClient
        response = self.bea_client.get_inflation_data()
        
        # Check for errors before processing
        if response.get("status") == "error":
            return response
            
        df = process_bea_response(response)
        return {
            "status": "success",
            "data": df.to_dict('records') if not df.empty else [],
            "message": f"Retrieved {len(df)} inflation data points"
        }
