from ..tools.bea_client import BEAClient
import logging

logger = logging.getLogger(__name__)

class DataCollectorAgent:
    """
    Agent responsible for collecting economic data using available tools.
    """
    def __init__(self, bea_api_key, model=None):
        self.bea_client = BEAClient(bea_api_key)
        self.model = model 
        
        # Mocking the 'tools' attribute so the notebook can display available tools
        self.agent = type('obj', (object,), {'tools': []})
        self.agent.tools = [
            type('tool', (object,), {'name': 'get_gdp_data'}),
            type('tool', (object,), {'name': 'get_unemployment_data'}),
            type('tool', (object,), {'name': 'get_inflation_data'})
        ]

    def _format_response(self, raw_data, data_type):
        """Standardizes the response format for the notebook's checks."""
        try:
            # Check for API errors captured in the client
            if isinstance(raw_data, dict) and 'error' in raw_data:
                return {'status': 'error', 'message': raw_data['error'], 'data': []}

            # Check for successful data
            if raw_data.get('BEAAPI', {}).get('Results', {}).get('Data'):
                rows = raw_data['BEAAPI']['Results']['Data']
                return {
                    'status': 'success',
                    'data': rows,
                    'message': f"Retrieved {len(rows)} {data_type} data points"
                }
            else:
                return {'status': 'error', 'message': "No data returned from API", 'data': []}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'data': []}

    def get_gdp_data(self):
        """Fetch GDP data and format response."""
        raw_data = self.bea_client.get_gdp_data()
        return self._format_response(raw_data, "GDP")

    def get_unemployment_data(self):
        """Fetch Unemployment/Labor data and format response."""
        raw_data = self.bea_client.get_unemployment_data()
        return self._format_response(raw_data, "unemployment")

    def get_inflation_data(self):
        """Fetch Inflation data and format response."""
        raw_data = self.bea_client.get_inflation_data()
        return self._format_response(raw_data, "inflation")