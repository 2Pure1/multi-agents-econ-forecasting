# Import necessary libraries from the ADK and standard library
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from typing import Dict, List
import asyncio

# Import the specialist agents that this coordinator will manage
from .data_collector import DataCollectorAgent
from .economic_analyst import EconomicAnalystAgent
from .forecasting_specialist import ForecastingSpecialistAgent
from .visualization_agent import VisualizationAgent

class EconomicTeamCoordinator:
    """
    The main coordinator for the multi-agent economic forecasting team.
    
    This class initializes all the specialist agents and a primary coordinator agent
    that delegates tasks to the specialists based on a user's query.
    """
    
    def __init__(self, bea_api_key: str, model: Gemini):
        """
        Initializes the coordinator and all specialist agents.
        
        Args:
            bea_api_key: The API key for the BEA API, to be passed to the data collector.
            model: The Gemini model instance to be used by all agents.
        """
        self.model = model
        self.bea_api_key = bea_api_key
        
        # --- 1. Initialize all the specialized agents ---
        self.data_collector = DataCollectorAgent(bea_api_key, model)
        self.economic_analyst = EconomicAnalystAgent(model)
        self.forecasting_specialist = ForecastingSpecialistAgent(model)
        self.visualization_agent = VisualizationAgent(model)
        
        # --- 2. Create the main coordinator agent ---
        # This agent's role is to understand the user's high-level goal and
        # delegate sub-tasks to the appropriate specialist agent.
        self.coordinator_agent = LlmAgent(
            name="economic_team_coordinator",
            model=model,
            instruction="""You are the expert coordinator of a team of economic analysis agents.
            Your primary role is to delegate tasks to the correct specialist agent based on the user's request.
            
            Your Team:
            - Data Collector: Call this agent to fetch any required economic data.
            - Economic Analyst: Call this agent to perform statistical analysis, identify trends, or calculate indicators.
            - Forecasting Specialist: Call this agent to build models, generate forecasts, or evaluate forecast accuracy.
            - Visualization Agent: Call this agent to create plots, charts, or other visualizations.
            
            Your Workflow:
            Based on the user's query, you must devise a plan and call the agents in a logical sequence.
            For example, to generate a forecast, you must first collect data, then you can generate the forecast.
            
            Coordinate the team effectively to fulfill the user's request.""",
            # The AgentTool allows one agent to call another agent.
            tools=[
                AgentTool(agent=self.data_collector.agent),
                AgentTool(agent=self.economic_analyst.agent),
                AgentTool(agent=self.forecasting_specialist.agent),
                AgentTool(agent=self.visualization_agent.agent)
            ]
        )
        
        # --- 3. Setup the Runner and Session Service ---
        # The SessionService stores the history of the conversation.
        self.session_service = InMemorySessionService()
        # The Runner executes the agent logic, handling the flow of messages and tool calls.
        self.runner = Runner(
            agent=self.coordinator_agent,
            session_service=self.session_service
        )
    
    async def run_complete_analysis(self, user_query: str) -> Dict:
        """
        Runs the full multi-agent workflow for a given user query.
        
        Args:
            user_query: The high-level query from the user.
            
        Returns:
            A dictionary containing the final response from the agent.
        """
        session_id = f"analysis_session_{hash(user_query)}"
        
        try:
            # Create a new session for this analysis
            await self.session_service.create_session(
                app_name="economic_forecasting",
                user_id="user", # In a real app, this would be a unique user identifier
                session_id=session_id
            )
            
            # The runner executes the agent conversation asynchronously
            results = {}
            async for event in self.runner.run_async(
                user_id="user",
                session_id=session_id,
                new_message=user_query
            ):
                # The final response is the summary or answer from the coordinator agent
                if event.is_final_response() and event.content:
                    results['final_response'] = event.content
                # In a more advanced implementation, one could also capture and return
                # intermediate tool calls and outputs here for debugging or structured results.
                
            return {
                "status": "success",
                "session_id": session_id,
                "results": results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"An unexpected error occurred during the analysis: {e}"
            }
    
    async def get_gdp_forecast(self) -> Dict:
        """
        A convenience method to run a common, pre-defined analysis workflow.
        
        This method defines a specific, multi-step query and passes it to the
        coordinator to execute.
        
        Returns:
            The result of the full analysis.
        """
        query = """Please coordinate the team to perform a full economic analysis:
        1. Collect the latest GDP data.
        2. Analyze the current GDP growth trends and calculate key indicators.
        3. Build an ARIMA model and generate an 8-quarter forecast for GDP.
        4. Create a visualization of the historical data and the forecast.
        5. Provide a brief executive summary of the findings.
        """
        
        return await self.run_complete_analysis(query)
