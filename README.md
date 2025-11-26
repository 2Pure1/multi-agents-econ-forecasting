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
git clone https://github.com/2Pure1/economic-forecasting-agents.git
cd economic-forecasting-agents

pip install -r requirements.txt

# Set environment variables
export BEA_API_KEY="your_bea_api_key"
export GOOGLE_API_KEY="your_gemini_api_key"