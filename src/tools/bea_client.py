import requests
import pandas as pd
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BEAClient:
    """
    Client for interacting with the U.S. Bureau of Economic Analysis (BEA) API.
    """
    
    # --- CONSTANTS: NIPA Table IDs ---
    # Table 1.1.5: Gross Domestic Product
    NIPA_TABLE_GDP = "T10105"         
    
    # Table 1.1.4: Price Indexes for Gross Domestic Product (The standard Inflation metric)
    NIPA_TABLE_INFLATION = "T10104"   
    
    # Table 2.1: Personal Income and its Disposition
    # (Used as a proxy for Labor/Unemployment since BEA doesn't have BLS unemployment rates)
    NIPA_TABLE_PERSONAL_INCOME = "T20100" 

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://apps.bea.gov/api/data"

    def get_nipa_data(self, table_name: str, frequency: str = "Q") -> Dict:
        """
        Fetches data for a specific NIPA table with robust error handling.
        """
        params = {
            'UserID': self.api_key,
            'Method': 'GetData',
            'DataSetName': 'NIPA',
            'TableName': table_name,
            'Frequency': frequency,
            'Year': 'X',              # <--- CRITICAL FIX: 'X' requests all years (fixes API Error 40)
            'ResultFormat': 'JSON'
        }
        
        try:
            logger.info(f"Requesting BEA Table: {table_name}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for "hidden" API errors (Status 200 but error in body)
            if "BEAAPI" in data and "Results" in data["BEAAPI"] and "Error" in data["BEAAPI"]["Results"]:
                error_msg = data['BEAAPI']['Results']['Error']
                logger.error(f"BEA API Logic Error: {error_msg}")
                # We return the error dict so the agent can see it
                return {"error": str(error_msg)}
                
            return data

        except Exception as e:
            logger.error(f"HTTP/Network Error fetching BEA data: {e}")
            return {"error": str(e)}

    def get_gdp_data(self) -> Dict:
        """Fetches GDP Data (Table 1.1.5)"""
        return self.get_nipa_data(self.NIPA_TABLE_GDP)

    def get_inflation_data(self) -> Dict:
        """Fetches Inflation/Price Index Data (Table 1.1.4)"""
        return self.get_nipa_data(self.NIPA_TABLE_INFLATION)

    def get_unemployment_data(self) -> Dict:
        """Fetches Personal Income Data (Table 2.1) as Labor proxy"""
        return self.get_nipa_data(self.NIPA_TABLE_PERSONAL_INCOME)

def process_bea_response(response: Dict) -> pd.DataFrame:
    """
    Helper to convert raw BEA JSON into a Pandas DataFrame.
    """
    try:
        # Navigate path: BEAAPI -> Results -> Data
        if response.get("BEAAPI", {}).get("Results", {}).get("Data"):
            return pd.DataFrame(response["BEAAPI"]["Results"]["Data"])
        else:
            logger.warning("No 'Data' list found in BEA response.")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing BEA JSON to DataFrame: {e}")
        return pd.DataFrame()