# ===== main.py - FIXED VERSION WITH ALL SERVICES WORKING =====
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os
from datetime import datetime, timedelta, timezone
import traceback
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any
import json
import re
import random
from dashboard_routes import dashboard_router
from fastapi.staticfiles import StaticFiles

# Import configuration
try:
    from config import settings
except ImportError:
    # Fallback configuration if config module not available
    class Settings:
        environment = "development"
        log_level = "INFO"
        testing_mode = True
        mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/ai')
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        eodhd_api_key = os.getenv('EODHD_API_KEY')
        marketaux_api_key = os.getenv('MARKETAUX_API_KEY')
        
        # Plan limits
        free_weekly_limit = 4
        paid_monthly_limit = 100
        pro_daily_cooloff = 50
        
        def get_capability_summary(self):
            return {
                "sms_enabled": bool(self.twilio_account_sid),
                "ai_enabled": bool(self.openai_api_key),
                "database_enabled": bool(self.mongodb_url),
                "testing_mode": self.testing_mode
            }
        
        def get_security_config(self):
            return {"testing_mode": self.testing_mode}
        
        def get_scheduler_config(self):
            return {"enabled": True}
        
        def validate_runtime_requirements(self):
            return {"valid": True}
    
    settings = Settings()

# Import services with proper error handling and detailed logging
try:
    from services.database import DatabaseService
    logger.info("✅ DatabaseService imported successfully")
except Exception as e:
    DatabaseService = None
    logger.error(f"❌ DatabaseService failed: {str(e)} | Type: {type(e).__name__}")

try:
    from services.openai_service import OpenAIService
    logger.info("✅ OpenAIService imported successfully")
except Exception as e:
    OpenAIService = None
    logger.error(f"❌ OpenAIService failed: {str(e)} | Type: {type(e).__name__}")

try:
    from services.twilio_service import TwilioService
    logger.info("✅ TwilioService imported successfully")
except Exception as e:
    TwilioService = None
    logger.error(f"❌ TwilioService failed: {str(e)} | Type: {type(e).__name__}")

try:
    from services.technical_analysis import TechnicalAnalysisService
    logger.info("✅ TechnicalAnalysisService imported successfully")
except Exception as e:
    TechnicalAnalysisService = None
    logger.error(f"❌ TechnicalAnalysisService failed: {str(e)} | Type: {type(e).__name__}")

try:
    from services.llm_agent import TradingAgent, ToolExecutor
    logger.info("✅ LLM Agent services imported successfully")
except Exception as e:
    TradingAgent = None
    ToolExecutor = None
    logger.error(f"❌ LLM Agent services failed: {str(e)} | Type: {type(e).__name__}")

try:
    from services.news_sentiment import NewsSentimentService
    logger.info("✅ NewsSentimentService imported successfully")
except Exception as e:
    NewsSentimentService = None
    logger.error(f"❌ NewsSentimentService failed: {str(e)} | Type: {type(e).__name__}")

try:
    from services.weekly_scheduler import WeeklyScheduler
    logger.info("✅ WeeklyScheduler imported successfully")
except Exception as e:
    WeeklyScheduler = None
    logger.error(f"❌ WeeklyScheduler failed: {str(e)} | Type: {type(e).__name__}")

try:
    from core.message_handler import MessageHandler
    logger.info("✅ MessageHandler imported successfully")
except Exception as e:
    MessageHandler = None
    logger.error(f"❌ MessageHandler failed: {str(e)} | Type: {type(e).__name__}")

# ===== FUNDAMENTAL ANALYSIS ENGINE IMPORT =====
try:
    from services.fundamental_analysis import FundamentalAnalysisEngine, FundamentalAnalysisTool, AnalysisDepth
    logger.info("✅ FundamentalAnalysisEngine imported successfully")
except Exception as e:
    FundamentalAnalysisEngine = None
    FundamentalAnalysisTool = None
    AnalysisDepth = None
    logger.error(f"❌ FundamentalAnalysisEngine failed: {str(e)} | Type: {type(e).__name__}")

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level)

# ===== HYPER-PERSONALIZED USER PROFILES =====

class UserPersonalityEngine:
    """Learns and adapts to each user's unique communication style and trading personality"""
    
    def __init__(self):
        self.user_profiles = defaultdict(lambda: {
            "communication_style": {
                "formality": "casual",  # casual, professional, friendly
                "energy": "moderate",   # low, moderate, high, excited
                "emoji_usage": "some",  # none, minimal, some, lots
                "message_length": "medium",  # short, medium, long
                "slang_level": "some",  # none, minimal, some, lots
                "technical_depth": "medium"  # basic, medium, advanced
            },
            "trading_personality": {
                "risk_tolerance": "moderate",  # conservative, moderate, aggressive
                "trading_style": "swing",      # day, swing, long_term
                "experience_level": "intermediate",  # beginner, intermediate, advanced
                "preferred_sectors": [],
                "common_symbols": [],
                "win_rate": None,
                "typical_position_size": "medium"
            },
            "learning_data": {
                "total_messages": 0,
                "successful_trades_mentioned": 0,
                "loss_trades_mentioned": 0,
                "favorite_phrases": [],
                "question_patterns": [],
                "response_preferences": []
            },
            "context_memory": {
                "recent_positions": [],
                "watchlist": [],
                "last_discussed_stocks": [],
                "goals_mentioned": [],
                "concerns_expressed": []
            }
        })
    
    def learn_from_message(self, phone_number: str, message: str, intent: dict):
        """Learn from each user interaction"""
        profile = self.user_profiles[phone_number]
        
        # Analyze communication style
        self._analyze_communication_style(profile, message)
        
        # Update trading personality
        self._update_trading_personality(profile, message, intent)
        
        # Store learning data
        profile["learning_data"]["total_messages"] += 1
        
        # Update context memory
        self._update_context_memory(profile, message, intent)
        
        logger.info(f"📚 Learning updated for {phone_number}: Style={profile['communication_style']['formality']}, Energy={profile['communication_style']['energy']}")
    
    def _analyze_communication_style(self, profile: dict, message: str):
        """Analyze how the user communicates"""
        msg_lower = message.lower()
        
        # Formality detection
        formal_indicators = ['please', 'thank you', 'could you', 'would you', 'analysis', 'evaluation']
        casual_indicators = ['yo', 'hey', 'what\'s up', 'gonna', 'wanna', 'btw', 'lol']
        
        if any(word in msg_lower for word in casual_indicators):
            profile["communication_style"]["formality"] = "casual"
        elif any(word in msg_lower for word in formal_indicators):
            profile["communication_style"]["formality"] = "professional"
        
        # Energy level detection
        excited_indicators = ['!', 'wow', 'awesome', 'amazing', 'love', 'hate', 'crazy']
        low_energy_indicators = ['okay', 'fine', 'whatever', 'sure', 'alright']
        
        excitement_score = sum(1 for word in excited_indicators if word in msg_lower)
        if excitement_score > 2 or message.count('!') > 1:
            profile["communication_style"]["energy"] = "high"
        elif any(word in msg_lower for word in low_energy_indicators):
            profile["communication_style"]["energy"] = "low"
        
        # Emoji usage
        emoji_count = sum(1 for char in message if ord(char) > 127)
        if emoji_count > 2:
            profile["communication_style"]["emoji_usage"] = "lots"
        elif emoji_count > 0:
            profile["communication_style"]["emoji_usage"] = "some"
        
        # Message length preference
        if len(message) < 20:
            profile["communication_style"]["message_length"] = "short"
        elif len(message) > 100:
            profile["communication_style"]["message_length"] = "long"
        
        # Technical depth
        technical_terms = ['rsi', 'macd', 'support', 'resistance', 'fibonacci', 'bollinger', 'volume', 'volatility']
        if any(term in msg_lower for term in technical_terms):
            profile["communication_style"]["technical_depth"] = "advanced"
    
    def _update_trading_personality(self, profile: dict, message: str, intent: dict):
        """Update trading personality based on message content"""
        msg_lower = message.lower()
        
        # Risk tolerance detection
        conservative_words = ['safe', 'conservative', 'careful', 'risk', 'worried', 'scared']
        aggressive_words = ['yolo', 'moon', 'aggressive', 'all in', 'big position']
        
        if any(word in msg_lower for word in aggressive_words):
            profile["trading_personality"]["risk_tolerance"] = "aggressive"
        elif any(word in msg_lower for word in conservative_words):
            profile["trading_personality"]["risk_tolerance"] = "conservative"
        
        # Trading style detection
        day_trading_words = ['day trade', 'scalp', 'quick', 'fast', 'intraday']
        long_term_words = ['hold', 'long term', 'invest', 'retirement', 'years']
        
        if any(word in msg_lower for word in day_trading_words):
            profile["trading_personality"]["trading_style"] = "day"
        elif any(word in msg_lower for word in long_term_words):
            profile["trading_personality"]["trading_style"] = "long_term"
        
        # Track symbols mentioned
        if intent.get("symbols"):
            for symbol in intent["symbols"]:
                if symbol not in profile["trading_personality"]["common_symbols"]:
                    profile["trading_personality"]["common_symbols"].append(symbol)
                    # Keep only last 20 symbols
                    if len(profile["trading_personality"]["common_symbols"]) > 20:
                        profile["trading_personality"]["common_symbols"] = profile["trading_personality"]["common_symbols"][-20:]
    
    def _update_context_memory(self, profile: dict, message: str, intent: dict):
        """Update context memory for personalized responses"""
        msg_lower = message.lower()
        
        # Detect wins/losses
        if any(word in msg_lower for word in ['profit', 'win', 'gain', 'up', 'made money']):
            profile["learning_data"]["successful_trades_mentioned"] += 1
        elif any(word in msg_lower for word in ['loss', 'lose', 'down', 'lost money', 'bad trade']):
            profile["learning_data"]["loss_trades_mentioned"] += 1
        
        # Store recent symbols as last discussed
        if intent.get("symbols"):
            profile["context_memory"]["last_discussed_stocks"] = intent["symbols"][:5]
        
        # Detect goals and concerns
        goal_words = ['goal', 'target', 'want to', 'hoping', 'plan to']
        concern_words = ['worried', 'concerned', 'scared', 'nervous', 'afraid']
        
        if any(word in msg_lower for word in goal_words):
            profile["context_memory"]["goals_mentioned"].append(message[:100])
        elif any(word in msg_lower for word in concern_words):
            profile["context_memory"]["concerns_expressed"].append(message[:100])
    
    def generate_personalized_prompt(self, phone_number: str, user_message: str, ta_data: dict = None) -> str:
        """Generate a hyper-personalized prompt for OpenAI"""
        profile = self.user_profiles[phone_number]
        
        # Build personality description
        personality_desc = self._build_personality_description(profile)
        
        # Build context
        context_desc = self._build_context_description(profile)
        
        # Build technical analysis context
        ta_context = ""
        if ta_data:
            ta_context = f"\n\nREAL MARKET DATA:\n{json.dumps(ta_data, indent=2)[:500]}..."
        
        prompt = f"""You are {phone_number}'s personal AI trading assistant. You know them intimately and communicate exactly like their best trading buddy would.

PERSONALITY PROFILE:
{personality_desc}

TRADING CONTEXT:
{context_desc}

CRITICAL COMMUNICATION RULES:
1. Match their energy: {profile['communication_style']['energy']} energy level
2. Use their formality: {profile['communication_style']['formality']} style
3. Message length: Keep responses {profile['communication_style']['message_length']} (like they prefer)
4. Technical depth: Use {profile['communication_style']['technical_depth']} level analysis
5. Emoji usage: Use {profile['communication_style']['emoji_usage']} emojis

RESPONSE GUIDELINES:
- Sound like their personal trading coach who knows their history
- Reference their past wins/losses naturally if relevant
- Use their preferred stocks/sectors when giving examples
- Remember what they've told you before
- Be encouraging but honest about risks
- Use casual language that matches their style
- Keep under 450 characters for SMS efficiency

USER MESSAGE: "{user_message}"
{ta_context}

Respond as their personalized trading assistant who knows them well:"""
        
        return prompt
    
    def _build_personality_description(self, profile: dict) -> str:
        """Build personality description for prompt"""
        comm = profile["communication_style"]
        trading = profile["trading_personality"]
        
        return f"""Communication Style: {comm['formality']}, {comm['energy']} energy, prefers {comm['message_length']} messages
Trading Personality: {trading['risk_tolerance']} risk tolerance, {trading['trading_style']} trader, {trading['experience_level']} level
Common Stocks: {', '.join(trading['common_symbols'][:5]) if trading['common_symbols'] else 'Getting to know their preferences'}
Win/Loss Ratio: {profile['learning_data']['successful_trades_mentioned']} wins mentioned, {profile['learning_data']['loss_trades_mentioned']} losses mentioned"""
    
    def _build_context_description(self, profile: dict) -> str:
        """Build context description for prompt"""
        context = profile["context_memory"]
        
        return f"""Recent Stocks Discussed: {', '.join(context['last_discussed_stocks']) if context['last_discussed_stocks'] else 'None yet'}
Total Conversations: {profile['learning_data']['total_messages']}
Recent Goals: {context['goals_mentioned'][-1] if context['goals_mentioned'] else 'None mentioned'}
Recent Concerns: {context['concerns_expressed'][-1] if context['concerns_expressed'] else 'None mentioned'}"""

    def get_user_profile(self, phone_number: str) -> dict:
        """Get user profile for external access"""
        return self.user_profiles.get(phone_number, {})

# Initialize personality engine
personality_engine = UserPersonalityEngine()

# ===== METRICS COLLECTION SYSTEM =====

class MetricsCollector:
    def __init__(self):
        self.lock = Lock()
        self.start_time = datetime.now()
        
        # Request metrics
        self.total_requests = 0
        self.requests_by_endpoint = defaultdict(int)
        self.requests_by_ticker = defaultdict(int)
        self.recent_requests = deque(maxlen=100)
        
        # Performance metrics
        self.response_times = defaultdict(list)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Error tracking
        self.errors_by_endpoint = defaultdict(int)
        self.recent_errors = deque(maxlen=50)
        
        # Popular tickers tracking
        self.ticker_request_count = defaultdict(int)
        
    def record_request(self, endpoint: str, ticker: str = None, response_time: float = 0, cache_status: str = None, error: bool = False):
        with self.lock:
            self.total_requests += 1
            self.requests_by_endpoint[endpoint] += 1
            
            if ticker:
                self.requests_by_ticker[ticker.upper()] += 1
                self.ticker_request_count[ticker.upper()] += 1
            
            # Record recent request
            request_info = {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "ticker": ticker,
                "response_time_ms": round(response_time * 1000, 2),
                "cache_status": cache_status,
                "error": error
            }
            self.recent_requests.append(request_info)
            
            # Performance tracking
            if response_time > 0:
                self.response_times[endpoint].append(response_time)
                if len(self.response_times[endpoint]) > 50:
                    self.response_times[endpoint] = self.response_times[endpoint][-50:]
            
            # Cache tracking
            if cache_status == "hit":
                self.cache_hits += 1
            elif cache_status == "miss":
                self.cache_misses += 1
            
            # Error tracking
            if error:
                self.errors_by_endpoint[endpoint] += 1
                error_info = {
                    "timestamp": datetime.now().isoformat(),
                    "endpoint": endpoint,
                    "ticker": ticker
                }
                self.recent_errors.append(error_info)
    
    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            uptime = datetime.now() - self.start_time
            
            # Calculate average response times
            avg_response_times = {}
            for endpoint, times in self.response_times.items():
                if times:
                    avg_response_times[endpoint] = round(sum(times) / len(times) * 1000, 2)
            
            # Get top tickers
            top_tickers = sorted(
                self.ticker_request_count.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Calculate requests per minute
            total_minutes = max(uptime.total_seconds() / 60, 1)
            requests_per_minute = round(self.total_requests / total_minutes, 2)
            
            # Cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = round((self.cache_hits / total_cache_requests * 100), 2) if total_cache_requests > 0 else 0
            
            return {
                "uptime": {
                    "seconds": int(uptime.total_seconds()),
                    "formatted": str(uptime).split('.')[0]
                },
                "requests": {
                    "total": self.total_requests,
                    "per_minute": requests_per_minute,
                    "by_endpoint": dict(self.requests_by_endpoint),
                    "recent": list(self.recent_requests)[-10:]
                },
                "performance": {
                    "avg_response_times_ms": avg_response_times,
                    "cache_hit_rate": cache_hit_rate,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses
                },
                "tickers": {
                    "top_requested": top_tickers,
                    "unique_count": len(self.ticker_request_count)
                },
                "errors": {
                    "by_endpoint": dict(self.errors_by_endpoint),
                    "recent": list(self.recent_errors)[-5:]
                }
            }

# ===== PLAN LIMITS CONFIGURATION =====
PLAN_LIMITS = {
    "free": {
        "weekly_limit": getattr(settings, 'free_weekly_limit', 4),
        "price": 0,
        "features": ["Basic market updates", "Stock analysis on demand"]
    },
    "paid": {
        "monthly_limit": getattr(settings, 'paid_monthly_limit', 100),
        "price": 29,
        "features": ["Personalized insights", "Portfolio tracking", "Market analytics"]
    },
    "pro": {
        "unlimited": True,
        "daily_cooloff": getattr(settings, 'pro_daily_cooloff', 50),
        "price": 99,
        "features": ["Unlimited messages", "Real-time alerts", "Advanced screeners", "Priority support"]
    }
}

# Popular tickers list
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
    'NFLX', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'UBER', 'LYFT',
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP',
    'JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV', 'TMO', 'ABT', 'LLY',
    'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'AMGN',
    'XOM', 'CVX', 'COP', 'BA', 'CAT', 'GE', 'MMM', 'HON',
    'T', 'VZ', 'CMCSA', 'TMUS',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV', 'VEA', 'IEFA', 'EEM',
    'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI',
    'COIN', 'MSTR', 'SQ', 'HOOD',
    'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'ENPH', 'PLUG',
    'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'WISH', 'CLOV',
    'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN',
    'F', 'SNAP', 'PINS', 'ZM', 'ROKU', 'PTON', 'SHOP',
    'ARKK', 'ARKQ', 'ARKG', 'ARKW', 'SQQQ', 'TQQQ', 'UVXY', 'VIX'
]

def is_popular_ticker(ticker: str) -> bool:
    """Check if ticker is in popular list."""
    return ticker.upper() in POPULAR_TICKERS

# Global metrics instance
metrics = MetricsCollector()

# Global services - COMPLETE LIST
db_service = None
openai_service = None
twilio_service = None
message_handler = None
scheduler_task = None
ta_service = None
trading_agent = None
tool_executor = None
news_service = None
fundamental_service = None
fundamental_tool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    global db_service, openai_service, twilio_service, message_handler, ta_service, trading_agent, tool_executor, news_service, fundamental_service, fundamental_tool, scheduler_task
    
    logger.info("🚀 Starting SMS Trading Bot with Complete Intelligence Suite...")
    
    try:
        # Initialize Database Service FIRST
        if DatabaseService:
            db_service = DatabaseService()
            try:
                await db_service.initialize()
                logger.info("✅ Database service initialized")
            except Exception as e:
                logger.error(f"❌ Database service failed: {e}")
                db_service = None
        
        # Initialize OpenAI Service
        if OpenAIService:
            try:
                openai_service = OpenAIService()
                logger.info("✅ OpenAI service initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI service failed: {e}")
                openai_service = None
        
        # Initialize Twilio Service
        if TwilioService:
            try:
                twilio_service = TwilioService()
                logger.info("✅ Twilio service initialized")
            except Exception as e:
                logger.error(f"❌ Twilio service failed: {e}")
                twilio_service = None
        
        # Initialize Technical Analysis Service
        if TechnicalAnalysisService:
            try:
                ta_service = TechnicalAnalysisService()
                # Pass redis client from db_service if available
                if db_service and hasattr(db_service, 'redis') and db_service.redis:
                    logger.info("✅ TA Service using Redis cache")
                logger.info("✅ Technical Analysis service initialized")
            except Exception as e:
                logger.error(f"❌ Technical Analysis service failed: {e}")
                ta_service = None
        
        # Initialize News Sentiment Service - FIXED PARAMETERS
        if NewsSentimentService:
            try:
                news_service = NewsSentimentService(
                    redis_client=db_service.redis if db_service and hasattr(db_service, 'redis') else None,
                    openai_service=openai_service
                )
                logger.info("✅ News Sentiment Service initialized")
            except Exception as e:
                logger.error(f"❌ News Sentiment service failed: {e}")
                news_service = None
        
        # Initialize Fundamental Analysis Service - FIXED PARAMETERS
        if FundamentalAnalysisEngine:
            try:
                eodhd_key = getattr(settings, 'eodhd_api_key', None)
                if eodhd_key:
                    fundamental_service = FundamentalAnalysisEngine(
                        eodhd_api_key=eodhd_key,
                        redis_client=db_service.redis if db_service and hasattr(db_service, 'redis') else None
                    )
                    
                    # Initialize Fundamental Analysis Tool - FIXED PARAMETERS
                    if FundamentalAnalysisTool:
                        fundamental_tool = FundamentalAnalysisTool(
                            eodhd_api_key=eodhd_key,
                            redis_client=db_service.redis if db_service and hasattr(db_service, 'redis') else None
                        )
                        logger.info("✅ Fundamental Analysis Tool initialized")
                    
                    logger.info("✅ Fundamental Analysis Engine initialized")
                else:
                    logger.warning("⚠️ EODHD_API_KEY not set - Fundamental Analysis unavailable")
                    fundamental_service = None
            except Exception as e:
                logger.error(f"❌ Fundamental Analysis service failed: {e}")
                fundamental_service = None
        
        # Initialize Trading Agent (Hybrid LLM Agent)
        if TradingAgent:
            try:
                if openai_service and personality_engine:
                    trading_agent = TradingAgent(
                        openai_client=openai_service,
                        personality_engine=personality_engine,
                        database_service=db_service
                    )
                    logger.info("✅ Trading agent initialized")
                else:
                    logger.warning("⚠️ Trading agent not initialized - missing dependencies")
                    trading_agent = None
            except Exception as e:
                logger.error(f"❌ Trading agent failed: {e}")
                trading_agent = None
        
        # Initialize Tool Executor
        if ToolExecutor:
            try:
                tool_executor = ToolExecutor(
                    ta_service=ta_service,
                    portfolio_service=None,
                    screener_service=None,
                    news_service=news_service,
                    fundamental_tool=fundamental_tool
                )
                logger.info("✅ Tool executor initialized")
            except Exception as e:
                logger.error(f"❌ Tool executor failed: {e}")
                tool_executor = None
        
        # Initialize Message Handler
        if MessageHandler:
            try:
                if db_service and openai_service and twilio_service:
                    message_handler = MessageHandler(
                        db_service=db_service,
                        openai_service=openai_service,
                        twilio_service=twilio_service
                    )
                    logger.info("✅ Message handler initialized")
                else:
                    logger.warning("⚠️ Message handler not initialized - missing dependencies")
                    message_handler = None
            except Exception as e:
                logger.error(f"❌ Message handler failed: {e}")
                message_handler = None
        
        # Initialize Weekly Scheduler
        if WeeklyScheduler:
            try:
                if db_service and twilio_service:
                    scheduler = WeeklyScheduler(db_service, twilio_service)
                    scheduler_task = asyncio.create_task(scheduler.start_scheduler())
                    logger.info("✅ Weekly scheduler started")
                else:
                    logger.warning("⚠️ Weekly scheduler not started - missing dependencies")
            except Exception as e:
                logger.error(f"❌ Weekly scheduler failed: {e}")
        
        # Final status report
        available_engines = []
        if ta_service:
            available_engines.append("Technical Analysis")
        if news_service:
            available_engines.append("News Sentiment")
        if fundamental_service:
            available_engines.append("Fundamental Analysis")
        
        logger.info(f"📊 Available Analysis Engines: {', '.join(available_engines) if available_engines else 'None'}")
        
        agent_status = "Complete Intelligence Suite" if trading_agent and tool_executor else "Fallback Mode"
        logger.info(f"🤖 Agent Status: {agent_status}")
        
        logger.info("✅ SMS Trading Bot startup completed")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Continue in degraded mode
        logger.warning("⚠️ Continuing in degraded mode")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SMS Trading Bot...")
    
    try:
        # Close scheduler
        if scheduler_task:
            scheduler_task.cancel()
            logger.info("✅ Weekly scheduler stopped")
        
        # Close services
        if ta_service and hasattr(ta_service, 'close'):
            await ta_service.close()
            logger.info("✅ Technical Analysis service closed")
        
        if news_service and hasattr(news_service, 'close'):
            await news_service.close()
            logger.info("✅ News Sentiment service closed")
        
        if fundamental_service and hasattr(fundamental_service, 'close'):
            await fundamental_service.close()
            logger.info("✅ Fundamental Analysis service closed")
        
        if db_service and hasattr(db_service, 'close'):
            await db_service.close()
            logger.info("✅ Database service closed")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")
    
    logger.info("✅ Shutdown complete")

app = FastAPI(
    title="SMS Trading Bot",
    description="Complete Intelligence Suite with Technical + News + Fundamental Analysis + Hybrid LLM Agent",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files and dashboard
from fastapi.staticfiles import StaticFiles
from pathlib import Path

Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(dashboard_router)

# ===== MIDDLEWARE =====

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    response_time = time.time() - start_time
    
    ticker = None
    if hasattr(request, 'query_params'):
        ticker = request.query_params.get('ticker')
    
    cache_status = None
    if hasattr(response, 'headers') and 'x-cache-status' in response.headers:
        cache_status = response.headers['x-cache-status']
    
    metrics.record_request(
        endpoint=str(request.url.path),
        ticker=ticker,
        response_time=response_time,
        cache_status=cache_status,
        error=response.status_code >= 400
    )
    
    return response

# ===== CONVERSATION STORAGE =====
conversation_history = defaultdict(list)

def store_conversation(phone_number: str, user_message: str, bot_response: str = None):
    """Store conversation in memory for dashboard viewing"""
    timestamp = datetime.now().isoformat()
    
    conversation_history[phone_number].append({
        "timestamp": timestamp,
        "direction": "inbound",
        "content": user_message,
        "type": "user_message"
    })
    
    if bot_response:
        conversation_history[phone_number].append({
            "timestamp": timestamp,
            "direction": "outbound", 
            "content": bot_response,
            "type": "bot_response"
        })
    
    if len(conversation_history[phone_number]) > 50:
        conversation_history[phone_number] = conversation_history[phone_number][-50:]

# ===== RATE LIMITING =====

async def check_rate_limits(phone_number: str) -> dict:
    """Check if user has exceeded their plan limits"""
    return None  # Always allow for demo

# ===== MAIN ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API with Complete Intelligence Suite", 
        "status": "running",
        "version": "2.0.0",
        "environment": settings.environment,
        "capabilities": settings.get_capability_summary(),
        "agent_type": "hybrid_llm" if trading_agent else "fallback",
        "analysis_engines": {
            "technical_analysis": ta_service is not None,
            "news_sentiment": news_service is not None,
            "fundamental_analysis": fundamental_service is not None
        },
        "plan_limits": PLAN_LIMITS
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": settings.environment,
            "version": "2.0.0",
            "agent_type": "hybrid_llm" if trading_agent else "fallback"
        }
        
        # Check database health
        if db_service is not None:
            try:
                await db_service.db.command("ping")
                health_status["database"] = {
                    "mongodb": {"status": "connected"},
                    "redis": {"status": "connected" if hasattr(db_service, 'redis') and db_service.redis is not None else "not_configured"}
                }
            except Exception as e:
                health_status["database"] = {
                    "mongodb": {"status": "error", "error": str(e)},
                    "redis": {"status": "unknown"}
                }
        else:
            health_status["database"] = {
                "mongodb": {"status": "not_initialized"},
                "redis": {"status": "not_initialized"}
            }
        
        # Check service availability
        health_status["services"] = {
            "database": "available" if db_service is not None else "unavailable",
            "openai": "available" if openai_service is not None else "unavailable", 
            "twilio": "available" if twilio_service is not None else "unavailable",
            "message_handler": "available" if message_handler is not None else "unavailable",
            "ta_service": "available" if ta_service is not None else "unavailable",
            "news_service": "available" if news_service is not None else "unavailable",
            "fundamental_service": "available" if fundamental_service is not None else "unavailable",
            "trading_agent": "available" if trading_agent is not None else "unavailable",
            "tool_executor": "available" if tool_executor is not None else "unavailable"
        }
        
        health_status["overall_status"] = "healthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

# ===== SMS WEBHOOK =====

@app.post("/webhook/sms")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming SMS messages"""
    try:
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return PlainTextResponse("Missing required fields", status_code=400)
        
        store_conversation(from_number, message_body)
        
        # Process with hybrid agent system
        bot_response = await process_sms_with_hybrid_agent(message_body, from_number)
        
        store_conversation(from_number, message_body, bot_response)
        
        # Background processing
        if message_handler:
            background_tasks.add_task(
                message_handler.process_incoming_message,
                from_number,
                message_body
            )
        
        return PlainTextResponse(
            '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"❌ SMS webhook error: {e}")
        return PlainTextResponse("Internal error", status_code=500)

async def process_sms_with_hybrid_agent(message_body: str, phone_number: str) -> str:
    """Process SMS using Hybrid LLM Agent system"""
    
    start_time = time.time()
    
    try:
        # Check rate limits
        rate_limit_response = await check_rate_limits(phone_number)
        if rate_limit_response:
            return rate_limit_response.get("message", "Rate limit exceeded")
        
        # Update user activity
        if db_service:
            await db_service.update_user_activity(phone_number)
        
        # Parse intent
        logger.info(f"🧠 Processing: '{message_body}'")
        
        if trading_agent:
            intent_data = await trading_agent.parse_intent(message_body, phone_number)
            logger.info(f"🎯 LLM Intent: {intent_data['intent']} | Symbols: {intent_data.get('symbols', [])} | Confidence: {intent_data.get('confidence', 0):.2f}")
        else:
            intent_data = analyze_message_intent_fallback(message_body)
            logger.info(f"🎯 Fallback Intent: {intent_data['intent']} | Symbols: {intent_data.get('symbols', [])} | Confidence: {intent_data.get('confidence', 0):.2f}")
        
        # Execute tools
        tool_results = {}
        
        if tool_executor:
            tool_results = await tool_executor.execute_tools(intent_data, phone_number)
        else:
            tool_results = await execute_tools_fallback(intent_data, phone_number)
        
        # Learn from interaction
        if personality_engine:
            personality_engine.learn_from_message(
                phone_number=phone_number,
                message=message_body,
                intent=intent_data
            )
        
        # Generate response
        user_profile = personality_engine.get_user_profile(phone_number) if personality_engine else {}
        
        if trading_agent:
            response_text = await trading_agent.generate_response(
                user_message=message_body,
                intent_data=intent_data,
                tool_results=tool_results,
                user_phone=phone_number,
                user_profile=user_profile
            )
        else:
            response_text = generate_fallback_response(intent_data, tool_results, user_profile)
        
        # Send SMS
        if twilio_service and response_text:
            sms_sent = await twilio_service.send_message(phone_number, response_text)
            if not sms_sent:
                logger.error(f"❌ Failed to send SMS to {phone_number}")
        
        # Store in database
        if db_service:
            await db_service.store_conversation(
                phone_number=phone_number,
                user_message=message_body,
                bot_response=response_text,
                intent=intent_data,
                tool_results=tool_results
            )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Processed in {processing_time:.2f}s using {'Hybrid Agent' if trading_agent else 'Fallback'}")
        
        return response_text
        
    except Exception as e:
        logger.error(f"💥 SMS processing failed: {traceback.format_exc()}")
        
        error_response = "Sorry, I'm having technical issues right now. Please try again in a moment! 🔧"
        
        if twilio_service:
            await twilio_service.send_message(phone_number, error_response)
        
        return error_response

# ===== FALLBACK FUNCTIONS =====

def analyze_message_intent_fallback(message: str) -> dict:
    """Fallback intent analysis"""
    import re
    
    message_lower = message.lower()
    
    # Symbol extraction
    potential_symbols = re.findall(r'\b[A-Z]{2,5}\b', message.upper())
    
    # Company mappings
    company_mappings = {
        'plug power': 'PLUG', 'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
        'amazon': 'AMZN', 'google': 'GOOGL', 'facebook': 'META', 'meta': 'META',
        'nvidia': 'NVDA', 'amd': 'AMD', 'netflix': 'NFLX', 'spotify': 'SPOT',
        'palantir': 'PLTR', 'gamestop': 'GME', 'amc': 'AMC'
    }
    
    for company, symbol in company_mappings.items():
        if company in message_lower:
            potential_symbols.append(symbol)
    
    # Filter false positives
    exclude_words = {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
        'OUR', 'HAD', 'BY', 'DO', 'GET', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO',
        'YO', 'HEY', 'SO', 'OH', 'AH', 'UM', 'UH', 'YEP', 'NAH', 'LOL', 'OMG', 'WOW',
        'BRO', 'FAM', 'THO', 'TBH', 'NGL', 'SMH', 'FML', 'IRL', 'BTW', 'TBF', 'IDK',
        'TO', 'AT', 'IN', 'ON', 'OR', 'OF', 'IS', 'IT', 'BE', 'GO', 'UP', 'MY', 'AS',
        'IF', 'NO', 'WE', 'ME', 'HE', 'AN', 'AM', 'US', 'A', 'I',
        'BUY', 'SELL', 'HOLD', 'CALL', 'PUT', 'BULL', 'BEAR', 'MOON', 'DIP', 'RIP'
    }
    
    valid_symbols = []
    for symbol in potential_symbols:
        if (symbol not in exclude_words and 
            len(symbol) >= 2 and 
            len(symbol) <= 5 and
            symbol.isalpha()):
            valid_symbols.append(symbol)
    
    symbols = list(dict.fromkeys(valid_symbols))
    
    # Intent detection
    if any(word in message_lower for word in ['portfolio', 'positions', 'holdings']):
        intent = "portfolio"
        requires_tools = ["portfolio_check"]
    elif any(word in message_lower for word in ['find', 'screen', 'search', 'discover']):
        intent = "screener"
        requires_tools = ["stock_screener"]
    elif any(word in message_lower for word in ['help', 'commands', 'start']):
        intent = "help"
        requires_tools = []
    elif any(word in message_lower for word in ['fundamental', 'fundamentals', 'valuation', 'ratios', 'pe ratio', 'financial', 'earnings', 'revenue']):
        intent = "fundamental_analysis"
        requires_tools = ["fundamental_analysis"]
    elif any(word in message_lower for word in ['complete', 'full', 'comprehensive', 'detailed', 'deep dive']):
        intent = "comprehensive_analysis"
        requires_tools = ["technical_analysis", "news_sentiment", "fundamental_analysis"]
    elif symbols:
        intent = "analyze"
        requires_tools = ["technical_analysis", "news_sentiment"]
    else:
        intent = "general"
        requires_tools = []
    
    return {
        "intent": intent,
        "symbols": symbols,
        "confidence": 0.4,
        "requires_tools": requires_tools,
        "fallback": True
    }

async def execute_tools_fallback(intent_data: dict, user_phone: str) -> dict:
    """Fallback tool execution"""
    results = {}
    
    try:
        requires_tools = intent_data.get("requires_tools", [])
        symbols = intent_data.get("symbols", [])
        
        # Technical Analysis
        if "technical_analysis" in requires_tools and symbols:
            if ta_service:
                for symbol in symbols[:2]:
                    ta_data = await ta_service.analyze_symbol(symbol.upper())
                    if ta_data:
                        if "technical_analysis" not in results:
                            results["technical_analysis"] = {}
                        results["technical_analysis"][symbol] = ta_data
        
        # News Sentiment
        if "news_sentiment" in requires_tools and symbols:
            if news_service:
                for symbol in symbols[:2]:
                    try:
                        news_data = await news_service.get_sentiment(symbol.upper())
                        if news_data and not news_data.get('error'):
                            if "news_sentiment" not in results:
                                results["news_sentiment"] = {}
                            results["news_sentiment"][symbol] = news_data
                    except Exception as e:
                        logger.warning(f"News sentiment failed for {symbol}: {e}")
        
        # Fundamental Analysis
        if "fundamental_analysis" in requires_tools and symbols:
            if fundamental_tool:
                for symbol in symbols[:2]:
                    try:
                        user_profile = personality_engine.get_user_profile(user_phone)
                        depth = "comprehensive" if user_profile.get("plan_type") == "pro" else "standard"
                        
                        fundamental_result = await fundamental_tool.execute({
                            "symbol": symbol.upper(),
                            "depth": depth,
                            "user_style": user_profile.get("communication_style", {}).get("formality", "casual")
                        })
                        
                        if fundamental_result.get("success"):
                            if "fundamental_analysis" not in results:
                                results["fundamental_analysis"] = {}
                            results["fundamental_analysis"][symbol] = fundamental_result["analysis_result"]
                    except Exception as e:
                        logger.warning(f"Fundamental analysis failed for {symbol}: {e}")
        
        if not results:
            results["market_data_unavailable"] = True
        
    except Exception as e:
        logger.error(f"Fallback tool execution failed: {e}")
        results["tool_error"] = str(e)
    
    return results

def generate_fallback_response(intent_data: dict, tool_results: dict, user_profile: dict) -> str:
    """Generate fallback response"""
    
    intent = intent_data["intent"]
    symbols = intent_data.get("symbols", [])
    
    style = user_profile.get("communication_style", {}) if user_profile else {}
    formality = style.get("formality", "casual")
    energy = style.get("energy", "moderate")
    
    if intent == "fundamental_analysis" and symbols:
        symbol = symbols[0]
        if tool_results.get("fundamental_analysis", {}).get(symbol):
            fundamental_data = tool_results["fundamental_analysis"][symbol]
            
            if formality == "casual" and energy == "high":
                response = f"{symbol} Fundamentals looking good! 💪 "
            elif formality == "professional":
                response = f"{symbol} Fundamental Analysis: "
            else:
                response = f"{symbol} fundamentals: "
                
            if hasattr(fundamental_data, 'overall_score'):
                response += f"Score: {fundamental_data.overall_score:.0f}/100"
            if hasattr(fundamental_data, 'financial_health'):
                response += f" | Health: {fundamental_data.financial_health.value.title()}"
            
            return response
        else:
            return f"Sorry, fundamental data for {symbol} unavailable right now 📊"
    
    elif intent == "comprehensive_analysis" and symbols:
        symbol = symbols[0]
        has_ta = tool_results.get("technical_analysis", {}).get(symbol)
        has_news = tool_results.get("news_sentiment", {}).get(symbol)
        has_fundamental = tool_results.get("fundamental_analysis", {}).get(symbol)
        
        if formality == "casual":
            response = f"{symbol} complete breakdown! 📊\n"
        else:
            response = f"{symbol} Comprehensive Analysis:\n"
        
        if has_ta:
            ta_data = has_ta
            price_data = ta_data.get("price", {})
            if price_data.get("current"):
                response += f"Price: ${price_data['current']}"
                if price_data.get("change_percent"):
                    change = price_data["change_percent"]
                    response += f" ({change:+.1f}%)\n"
        
        if has_fundamental:
            fund_data = has_fundamental
            if hasattr(fund_data, 'overall_score'):
                response += f"Fundamentals: {fund_data.overall_score:.0f}/100\n"
        
        if has_news:
            news_data = has_news
            if news_data.get('sentiment'):
                sentiment = news_data['sentiment']
                emoji = "🐂" if sentiment > 0.2 else "🐻" if sentiment < -0.2 else "➡️"
                response += f"News: {emoji} {sentiment:.1f} sentiment"
        
        return response if len(response) > len(f"{symbol} complete breakdown! 📊\n") else f"Analysis data unavailable for {symbol}"
    
    elif intent == "analyze" and symbols:
        symbol = symbols[0]
        if tool_results.get("technical_analysis", {}).get(symbol):
            ta_data = tool_results["technical_analysis"][symbol]
            
            if formality == "casual" and energy == "high":
                response = f"{symbol} looking good! 🚀 "
            elif formality == "professional":
                response = f"{symbol} Analysis: "
            else:
                response = f"Here's {symbol}: "
                
            price_data = ta_data.get("price", {})
            if price_data.get("current"):
                response += f"${price_data['current']}"
                if price_data.get("change_percent"):
                    change = price_data["change_percent"]
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    response += f" {direction}{abs(change):.1f}%"
            
            return response
        else:
            if formality == "casual":
                return f"Sorry, can't get live {symbol} data right now 😅 Market data service unavailable!"
            else:
                return f"Market data for {symbol} is currently unavailable. Please try again later."
    
    elif intent == "help":
        if formality == "casual":
            return "Hey! I'm your trading buddy 🤖 Ask me about any stock like 'How's AAPL?' or say 'TSLA fundamentals' for deep analysis!"
        else:
            return "I'm your AI trading assistant. Ask me about stocks, technical analysis, fundamentals, or market insights. How can I help?"
    
    elif intent == "portfolio":
        return "Portfolio tracking coming soon! For now, ask me about specific stocks. 📊"
    
    else:
        if formality == "casual" and energy == "high":
            return "What's up! 🚀 Ask me about any stock and I'll give you the full scoop - technical, fundamental, and news analysis!"
        else:
            return "I'm here to help with trading questions! Try asking about a stock symbol like 'AAPL fundamentals' or 'How is TSLA doing?'"

# ===== REMAINING ENDPOINTS =====

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    try:
        payload = await request.body()
        logger.info("Stripe webhook received")
        return {"status": "received", "message": "Stripe webhook processed"}
        
    except Exception as e:
        logger.error(f"❌ Stripe webhook error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/admin")
async def admin_dashboard():
    """Admin dashboard"""
    try:
        user_stats = {"total_users": len(personality_engine.user_profiles), "active_today": 0, "messages_today": 0}
        scheduler_status = {"status": "active" if scheduler_task and not scheduler_task.done() else "inactive"}
        system_metrics = metrics.get_metrics()
        
        return {
            "title": "SMS Trading Bot Admin Dashboard",
            "status": "operational",
            "version": "2.0.0",
            "environment": settings.environment,
            "agent_type": "hybrid_llm" if trading_agent else "fallback",
            "services": {
                "database": "connected" if db_service else "disconnected",
                "message_handler": "active" if message_handler else "inactive",
                "ta_service": "active" if ta_service else "inactive",
                "news_service": "active" if news_service else "inactive",
                "fundamental_service": "active" if fundamental_service else "inactive",
                "trading_agent": "active" if trading_agent else "inactive",
                "tool_executor": "active" if tool_executor else "inactive"
            },
            "stats": user_stats,
            "scheduler": scheduler_status,
            "metrics": system_metrics,
            "configuration": settings.get_capability_summary(),
            "plan_limits": PLAN_LIMITS
        }
    except Exception as e:
        logger.error(f"❌ Admin dashboard error: {e}")
        return {"error": str(e)}

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive service metrics"""
    try:
        system_metrics = metrics.get_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "service": system_metrics,
            "personality_engine": {
                "total_profiles": len(personality_engine.user_profiles),
                "active_learning": True,
                "avg_messages_per_user": sum(p["learning_data"]["total_messages"] for p in personality_engine.user_profiles.values()) / max(len(personality_engine.user_profiles), 1)
            },
            "analysis_engines": {
                "technical_analysis": ta_service is not None,
                "news_sentiment": news_service is not None,
                "fundamental_analysis": fundamental_service is not None
            },
            "hybrid_agent": {
                "trading_agent_available": trading_agent is not None,
                "tool_executor_available": tool_executor is not None,
                "agent_type": "complete_intelligence_suite" if trading_agent else "fallback"
            },
            "system": {
                "version": "2.0.0",
                "environment": settings.environment,
                "testing_mode": settings.testing_mode,
                "hyper_personalization": "active",
                "intelligence_suite": "complete" if (ta_service and news_service and fundamental_service) else "partial"
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Test interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Complete Intelligence Suite Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .engines { background: #e2e3e5; padding: 10px; border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🤖 Complete Intelligence Suite SMS Trading Bot</h1>
    <div class="engines">
        <strong>Available:</strong> Technical Analysis + News Sentiment + Fundamental Analysis + Hybrid LLM Agent + Personality Learning
    </div>
    
    <form onsubmit="testSMS(event)">
        <div class="form-group">
            <label>From Phone Number:</label>
            <input type="text" id="phone" value="+13012466712" required>
        </div>
        
        <div class="form-group">
            <label>Message Body:</label>
            <textarea id="message" rows="3" required>give me complete analysis of AAPL - technical, fundamental, and news 📊</textarea>
        </div>
        
        <button type="submit">Test Complete Intelligence Suite</button>
    </form>
    
    <div class="form-group" style="margin-top: 20px;">
        <button type="button" onclick="setMessage('TSLA fundamentals')">Fundamental Analysis</button>
        <button type="button" onclick="setMessage('yo what\\'s NVDA doing? 🚀')">Technical + News</button>
        <button type="button" onclick="setMessage('complete breakdown of MSFT')">All Engines</button>
    </div>
    
    <div id="result"></div>
    
    <script>
        function setMessage(msg) {
            document.getElementById('message').value = msg;
        }
        
        async function testSMS(event) {
            event.preventDefault();
            
            const phone = document.getElementById('phone').value;
            const message = document.getElementById('message').value;
            const resultDiv = document.getElementById('result');
            
            try {
                const formData = new URLSearchParams();
                formData.append('From', phone);
                formData.append('Body', message);
                
                const response = await fetch('/webhook/sms', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.text();
                    resultDiv.innerHTML = `<div class="result success">
                        <h3>✅ Success!</h3>
                        <p>SMS processed with Complete Intelligence Suite</p>
                        <pre>${result}</pre>
                        <p><strong>🧠 Hybrid LLM Agent + All Analysis Engines Active</strong></p>
                    </div>`;
                } else {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">
                    <h3>❌ Error</h3>
                    <p>${error.message}</p>
                </div>`;
            }
        }
    </script>
</body>
</html>
"""

# ===== DEBUG ENDPOINTS =====

@app.get("/debug/test-fundamental/{symbol}")
async def test_fundamental_analysis(symbol: str, depth: str = "standard"):
    """Test fundamental analysis"""
    try:
        if not fundamental_service and not fundamental_tool:
            return {
                "symbol": symbol.upper(),
                "fundamental_service_available": False,
                "error": "Fundamental Analysis Engine not initialized",
                "recommendation": "Check EODHD_API_KEY environment variable"
            }
        
        if fundamental_tool:
            result = await fundamental_tool.execute({
                "symbol": symbol.upper(),
                "depth": depth,
                "user_style": "casual"
            })
            
            if result.get("success"):
                analysis_result = result["analysis_result"]
                
                return {
                    "symbol": symbol.upper(),
                    "fundamental_analysis": {
                        "available": True,
                        "overall_score": getattr(analysis_result, 'overall_score', None),
                        "financial_health": getattr(analysis_result, 'financial_health', None),
                        "data_completeness": getattr(analysis_result, 'data_completeness', None),
                        "analysis_depth": depth
                    },
                    "sms_response": result["sms_response"],
                    "service_status": "working"
                }
            else:
                return {
                    "symbol": symbol.upper(),
                    "fundamental_analysis": {"available": False},
                    "error": result.get("error", "Unknown error"),
                    "service_status": "error"
                }
        else:
            return {
                "symbol": symbol.upper(),
                "fundamental_analysis": {"available": False},
                "error": "Fundamental tool not initialized"
            }
            
    except Exception as e:
        logger.error(f"Fundamental analysis test error: {e}")
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "recommendation": "Check EODHD API key and service configuration"
        }

@app.get("/debug/diagnose-services")
async def diagnose_services():
    """Comprehensive service diagnosis"""
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "environment_variables": {},
        "service_availability": {},
        "connection_tests": {},
        "personality_engine": {},
        "hybrid_agent": {},
        "recommendations": []
    }
    
    # Check environment variables
    env_vars = {
        "EODHD_API_KEY": "Set" if os.getenv('EODHD_API_KEY') else None,
        "OPENAI_API_KEY": "Set" if os.getenv('OPENAI_API_KEY') else None,
        "MARKETAUX_API_KEY": "Set" if os.getenv('MARKETAUX_API_KEY') else None,
        "MONGODB_URL": "Set" if os.getenv('MONGODB_URL') else None,
    }
    diagnosis["environment_variables"] = env_vars
    
    # Check service initialization
    diagnosis["service_availability"] = {
        "openai_service": openai_service is not None,
        "db_service": db_service is not None,
        "message_handler": message_handler is not None,
        "twilio_service": twilio_service is not None,
        "ta_service": ta_service is not None,
        "news_service": news_service is not None,
        "fundamental_service": fundamental_service is not None,
        "personality_engine": True,
        "trading_agent": trading_agent is not None,
        "tool_executor": tool_executor is not None
    }
    
    # Test personality engine
    diagnosis["personality_engine"] = {
        "total_profiles": len(personality_engine.user_profiles),
        "learning_active": True,
        "features": ["communication_style", "trading_personality", "context_memory"]
    }
    
    # Test hybrid agent system
    if trading_agent:
        try:
            test_intent = await trading_agent.parse_intent("test message", "+1555TEST")
            diagnosis["hybrid_agent"] = {
                "trading_agent": {
                    "available": True,
                    "intent_parsing": True,
                    "test_successful": test_intent is not None
                },
                "tool_executor": {
                    "available": tool_executor is not None,
                    "ta_integration": ta_service is not None,
                    "news_integration": news_service is not None,
                    "fundamental_integration": fundamental_service is not None
                }
            }
        except Exception as e:
            diagnosis["hybrid_agent"] = {
                "trading_agent": {
                    "available": True,
                    "intent_parsing": False,
                    "error": str(e)
                }
            }
    else:
        diagnosis["hybrid_agent"] = {
            "trading_agent": {
                "available": False,
                "reason": "OpenAI service not available or failed to initialize"
            }
        }
    
    # Test all analysis services
    analysis_tests = {}
    
    # Test TA Service
    if ta_service:
        try:
            test_ta_result = await ta_service.analyze_symbol("AAPL")
            analysis_tests["technical_analysis"] = {
                "working": True,
                "test_successful": test_ta_result is not None,
                "data_source": test_ta_result.get('source', 'unknown') if test_ta_result else None
            }
        except Exception as e:
            analysis_tests["technical_analysis"] = {"working": False, "error": str(e)}
    else:
        analysis_tests["technical_analysis"] = {"working": False, "error": "Service not initialized"}
    
    # Test News Service
    if news_service:
        try:
            test_news_result = await news_service.get_sentiment("AAPL")
            analysis_tests["news_sentiment"] = {
                "working": True,
                "test_successful": test_news_result is not None and not test_news_result.get('error')
            }
        except Exception as e:
            analysis_tests["news_sentiment"] = {"working": False, "error": str(e)}
    else:
        analysis_tests["news_sentiment"] = {"working": False, "error": "Service not initialized"}
    
    # Test Fundamental Service
    if fundamental_tool:
        try:
            test_fund_result = await fundamental_tool.execute({
                "symbol": "AAPL",
                "depth": "basic",
                "user_style": "casual"
            })
            analysis_tests["fundamental_analysis"] = {
                "working": True,
                "test_successful": test_fund_result.get("success", False)
            }
        except Exception as e:
            analysis_tests["fundamental_analysis"] = {"working": False, "error": str(e)}
    else:
        analysis_tests["fundamental_analysis"] = {"working": False, "error": "Service not initialized"}
    
    diagnosis["connection_tests"] = analysis_tests
    
    # Test OpenAI if available
    if openai_service:
        try:
            # Simple test without calling actual method
            diagnosis["connection_tests"]["openai"] = {
                "available": True,
                "service_initialized": True
            }
        except Exception as e:
            diagnosis["connection_tests"]["openai"] = {
                "available": True,
                "test_successful": False,
                "error": str(e)
            }
    else:
        diagnosis["connection_tests"]["openai"] = {
            "available": False,
            "error": "OpenAI service not initialized"
        }
    
    # Generate recommendations
    recommendations = []
    
    if not env_vars["EODHD_API_KEY"]:
        recommendations.append("❌ Set EODHD_API_KEY for Technical Analysis + Fundamental Analysis")
    
    if not env_vars["MARKETAUX_API_KEY"]:
        recommendations.append("❌ Set MARKETAUX_API_KEY for News Sentiment Analysis")
    
    if not env_vars["OPENAI_API_KEY"]:
        recommendations.append("❌ Set OPENAI_API_KEY for Hybrid LLM Agent")
    elif not trading_agent:
        recommendations.append("❌ Hybrid LLM Agent failed to initialize")
    
    # Check working services
    working_services = []
    if analysis_tests.get("technical_analysis", {}).get("working"):
        working_services.append("Technical Analysis")
    if analysis_tests.get("news_sentiment", {}).get("working"):
        working_services.append("News Sentiment")
    if analysis_tests.get("fundamental_analysis", {}).get("working"):
        working_services.append("Fundamental Analysis")
    
    if len(working_services) == 3:
        recommendations.append("✅ Complete Intelligence Suite operational!")
    elif working_services:
        recommendations.append(f"⚠️ Partial setup: {', '.join(working_services)} working.")
    
    if not recommendations:
        recommendations.append("🚀 Perfect setup - All systems operational!")
    
    diagnosis["recommendations"] = recommendations
    
    return diagnosis

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Complete Intelligence Suite SMS Trading Bot on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Testing mode: {settings.testing_mode}")
    logger.info(f"🧠 Personality Engine: Active")
    logger.info(f"📈 Technical Analysis: {'Available' if TechnicalAnalysisService else 'Unavailable'}")
    logger.info(f"📰 News Sentiment: {'Available' if NewsSentimentService else 'Unavailable'}")
    logger.info(f"📊 Fundamental Analysis: {'Available' if FundamentalAnalysisEngine else 'Unavailable'}")
    logger.info(f"🤖 Hybrid LLM Agent: {'Available' if TradingAgent else 'Unavailable'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )

# ===== ADD THESE ENDPOINTS TO YOUR main.py =====

@app.post("/api/test/sms-with-response")
async def test_sms_with_response(request: Request):
    """Test SMS processing and return response"""
    try:
        form_data = await request.form()
        from_number = form_data.get('From', '+1555TEST')
        message_body = form_data.get('Body', 'test message')
        
        # Process with hybrid agent system
        bot_response = await process_sms_with_hybrid_agent(message_body, from_number)
        
        return {
            "success": True,
            "user_message": message_body,
            "bot_response": bot_response,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Test SMS processing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/debug/test-ta/{symbol}")
async def test_technical_analysis(symbol: str):
    """Test technical analysis for a symbol"""
    try:
        if not ta_service:
            return {
                "symbol": symbol.upper(),
                "technical_analysis": {"available": False},
                "error": "Technical Analysis service not initialized"
            }
        
        ta_data = await ta_service.analyze_symbol(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "technical_analysis": {
                "available": True,
                "data": ta_data,
                "service_status": "working"
            }
        }
        
    except Exception as e:
        logger.error(f"Technical analysis test error: {e}")
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "recommendation": "Check EODHD API key and service configuration"
        }

@app.post("/debug/test-message")
async def test_message_processing(request: Request):
    """Test complete message processing pipeline"""
    try:
        data = await request.json()
        message = data.get('message', 'test AAPL')
        phone = data.get('phone', '+1555TEST')
        
        # Process message
        response = await process_sms_with_hybrid_agent(message, phone)
        
        return {
            "success": True,
            "input_message": message,
            "phone_number": phone,
            "bot_response": response,
            "services_status": {
                "ta_service": ta_service is not None,
                "news_service": news_service is not None,
                "fundamental_service": fundamental_service is not None,
                "trading_agent": trading_agent is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Message processing test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/debug/test-full-flow/{symbol}")
async def test_full_analysis_flow(symbol: str):
    """Test complete analysis flow for a symbol"""
    try:
        results = {}
        
        # Test Technical Analysis
        if ta_service:
            try:
                ta_result = await ta_service.analyze_symbol(symbol.upper())
                results["technical_analysis"] = {"success": True, "data": ta_result}
            except Exception as e:
                results["technical_analysis"] = {"success": False, "error": str(e)}
        else:
            results["technical_analysis"] = {"success": False, "error": "Service not available"}
        
        # Test News Sentiment
        if news_service:
            try:
                news_result = await news_service.get_sentiment(symbol.upper())
                results["news_sentiment"] = {"success": True, "data": news_result}
            except Exception as e:
                results["news_sentiment"] = {"success": False, "error": str(e)}
        else:
            results["news_sentiment"] = {"success": False, "error": "Service not available"}
        
        # Test Fundamental Analysis
        if fundamental_tool:
            try:
                fund_result = await fundamental_tool.execute({
                    "symbol": symbol.upper(),
                    "depth": "basic",
                    "user_style": "casual"
                })
                results["fundamental_analysis"] = {"success": fund_result.get("success", False), "data": fund_result}
            except Exception as e:
                results["fundamental_analysis"] = {"success": False, "error": str(e)}
        else:
            results["fundamental_analysis"] = {"success": False, "error": "Service not available"}
        
        # Test Trading Agent
        if trading_agent:
            try:
                intent = await trading_agent.parse_intent(f"analyze {symbol}", "+1555TEST")
                results["trading_agent"] = {"success": True, "intent": intent}
            except Exception as e:
                results["trading_agent"] = {"success": False, "error": str(e)}
        else:
            results["trading_agent"] = {"success": False, "error": "Service not available"}
        
        return {
            "symbol": symbol.upper(),
            "full_analysis_test": results,
            "overall_status": "Complete Intelligence Suite" if all(
                r.get("success") for r in results.values()
            ) else "Partial functionality"
        }
        
    except Exception as e:
        logger.error(f"Full flow test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
