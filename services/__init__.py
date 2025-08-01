# services/__init__.py
"""
Services package for SMS Trading Bot
Contains all service modules and integrations
"""

from .database import DatabaseService
from .openai_service import OpenAIService
from .technical_analysis import TechnicalAnalysisService
from .twilio_service import TwilioService
from .llm_agent import TradingAgent, ToolExecutor
from .news_sentiment import NewsSentimentService
from .fundamental_analysis import FundamentalAnalysisEngine, FundamentalAnalysisTool
from .key_builder import KeyBuilder
from .gemini_service import GeminiPersonalityService  # ← ADD THIS for Gemini personality analysis

__all__ = [
    "DatabaseService",
    "OpenAIService", 
    "TechnicalAnalysisService", 
    "TwilioService",
    "TradingAgent",
    "ToolExecutor",
    "NewsSentimentService",
    "FundamentalAnalysisEngine",
    "FundamentalAnalysisTool",
    "KeyBuilder",
    "GeminiPersonalityService"  # ← ADD THIS
]
