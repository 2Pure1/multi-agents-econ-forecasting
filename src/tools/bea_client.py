import requests
import pandas as pd
from typing import Dict, List, Optional
import os

class BEAClient:
    """
    A client for interacting with the U.S. Bureau of Economic Analysis (BEA) API.
    
    This client simplifies fetching economic data from NIPA tables and processing 
    the responses into a structured format.
    """
    
    # Constants for NIPA table names to improve readability and maintainability
    NIPA_TABLE_GDP = "T10105"  # Table 1.1.5: Gross Domestic Product
    NIPA_TABLE_GOV_RECEIPTS_EXPENDITURES = "T10600"  # Table 1.6: Government Current Receipts and Expenditures
    NIPA_TABLE_INFLATION = "T11004" # Table 1.1.4: Price Indexes for Gross Domestic Product
    
    def __init__(self, api_key: str):
        """
        Initializes the BEAClient with the provided API key.
        
        Args:
            api_key: Your BEA API key. It is recommended to load this from 
                     an environment variable rather than hardcoding it.
        """
        self.api_key = api_key
        self.base_url = "https://apps.bea.gov/api/data"
        
    def get_nipa_data(self, table_name: str, frequency: str = "Q") -> Dict:
        """
        Fetches data for a specific NIPA (National Income and Product Accounts) table.
        
        Args:
            table_name: The name of the NIPA table to retrieve.
            frequency: The frequency of the data (e.g., "Q" for quarterly, "A" for annual).
            
        Returns:
            A dictionary containing the API response, or an error message if the request fails.
        """
        params = {
            'UserID': self.api_key,
            'method': 'GetData',
            'datasetname': 'NIPA',
            'TableName': table_name,
            'Frequency': frequency,
            'ResultFormat': 'JSON'
        }
        
        try:
            # Make the API request
            response = requests.get(self.base_url, params=params)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Handle network-related errors
            return {"status": "error", "error_message": f"BEA API request error: {e}"}
        except Exception as e:
            # Handle other potential errors, such as JSON decoding errors
            return {"status": "error", "error_message": f"An unexpected error occurred: {e}"}
    
    def get_gdp_data(self) -> Dict:
        """Fetches GDP data from NIPA Table 1.1.5."""
        return self.get_nipa_data(self.NIPA_TABLE_GDP)
    
    def get_gov_receipts_expenditures_data(self) -> Dict:
        """
        Fetches Government Current Receipts and Expenditures data from NIPA Table 1.6.
        
        Note: For unemployment data, consider using the Bureau of Labor Statistics (BLS) API,
        as it is the primary source for that information.
        """
        return self.get_nipa_data(self.NIPA_TABLE_GOV_RECEIPTS_EXPENDITURES)
    
    def get_inflation_data(self) -> Dict:
        """Fetches price index data for inflation analysis from NIPA Table 1.1.4."""
        return self.get_nipa_data(self.NIPA_TABLE_INFLATION)

def process_bea_response(response: Dict) -> pd.DataFrame:
    """
    Converts a raw BEA API JSON response into a pandas DataFrame.
    
    Args:
        response: The JSON response from the BEA API as a dictionary.
        
    Returns:
        A pandas DataFrame containing the data, or an empty DataFrame if no data is found.
    """
    # Safely access nested data to avoid KeyError
    if response.get("BEAAPI", {}).get("Results", {}).get("Data"):
        data = response["BEAAPI"]["Results"]["Data"]
        df = pd.DataFrame(data)
        return df
    return pd.DataFrame()
    