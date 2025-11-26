# Economic Forecasting Multi-Agent System

A multi-agent system for economic data analysis and forecasting using Google ADK and BEA API.

## 🎯 Overview

This project demonstrates a multi-agent architecture where specialized AI agents collaborate as a data science team to:
- Collect economic data from Bureau of Economic Analysis (BEA) API
- Analyze macroeconomic trends and indicators
- Generate forecasts using statistical models
- Create interactive visualizations and dashboards

## 🏗️ Architecture

### Agent Team Structure

| Agent | Role | Responsibilities |
|-------|------|------------------|
| **Data Collector** | Data Engineer | Fetches data from BEA API, handles data quality |
| **Economic Analyst** | Data Analyst | Analyzes trends, calculates indicators, detects patterns |
| **Forecasting Specialist** | Data Scientist | Builds models, generates predictions, evaluates accuracy |
| **Visualization Agent** | BI Specialist | Creates dashboards, charts, and reports |
| **Team Coordinator** | Project Manager | Orchestrates workflow and coordinates team members |

## 🚀 Quick Start

### Prerequisites

1. BEA API Key from [BEA.gov](https://www.bea.gov/API/signup/)
2. Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### Installation

```bash
git clone https://github.com/2Pure1/multi-agents-econ-forecasting.git
cd multi-agents-econ-forecasting

pip install -r requirements.txt

# Set environment variables
export BEA_API_KEY="your_bea_api_key"
export GOOGLE_API_KEY="your_gemini_api_key"
```
### Basic Usage

```bash
from examples.multi_agent_workflow import main
import asyncio

asyncio.run(main())
```

### 📊 Features

### Data Collection
 * Real-time BEA API integration
 * Multiple economic indicators (GDP, Unemployment, Inflation)
 * Data validation and cleaning

### Analysis Capabilities
 * Trend analysis and pattern recognition
 * Business cycle identification
 * Anomaly detection
 * Economic health assessment

### Forecasting Models
 * ARIMA time series forecasting
 * Ensemble methods
 * Confidence interval estimation
 * Model performance evaluation

### Visualization
 * Interactive Plotly dashboards
 * Time series charts with forecasts
 * Economic indicator comparisons
 * Export to multiple formats

### 🛠️ Technical Stack
 * Multi-Agent Framework: Google ADK (Agent Development Kit)
 * LLM: Google Gemini 2.0 Flash
 * Data Source: BEA (Bureau of Economic Analysis) API
 * Analysis: pandas, statsmodels, scikit-learn
 * Visualization: Plotly, Matplotlib
 * Async Processing: asyncio

##  Project Structure

multi-agents-econ-forecasting/

├── src/agents/          # Specialized agent implementations

├── src/tools/           # Economic analysis tools

├── examples/            # Usage examples and workflows

├── config/              # Agent and API configuration

├── notebooks/           # Jupyter notebook demonstrations

└── evaluation/          # Test cases and evaluation metrics

## 🔧 Configuration

Edit config/agent_config.yaml to customize:
 * Agent behaviors and tools
 * Forecasting parameters
 * Data sources and frequencies
 * Model preferences

## 📈 Evaluation

The system includes comprehensive evaluation:
 * Test cases for each agent
 * Forecast accuracy metrics
 * Economic indicator validation
 * Performance benchmarking

## 🙏 Acknowledgments
 * Google ADK Team for the agent development framework
 * Bureau of Economic Analysis for economic data
 * Kaggle for the educational resources

