# ===== main.py - COMPLETE VERSION WITH HYBRID LLM AGENT + FUNDAMENTAL ANALYSIS =====
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

# Import services with fallbacks
try:
    from services.database import DatabaseService
    logger.info("✅ DatabaseService imported successfully")
except Exception as e:
    DatabaseService = None
    logger.error(f"❌ DatabaseService failed: {str(e)} | Type: {type(e).__name__}")
    import traceback
    logger.error(f"Full traceback: {traceback.format_exc()}")

try:
    from services.openai_service import OpenAIService
    logger.info("✅ OpenAIService imported successfully")
except Exception as e:
    OpenAIService = None
    logger.error(f"❌ OpenAIService failed: {str(e)} | Type: {type(e).__name__}")
    import traceback
    logger.error(f"Full traceback: {traceback.format_exc()}")

try:
    from services.twilio_service import TwilioService
    logger.info("✅ TwilioService imported successfully")
except Exception as e:
    TwilioService = None
    logger.error(f"❌ TwilioService failed: {str(e)} | Type: {type(e).__name__}")
    import traceback
    logger.error(f"Full traceback: {traceback.format_exc()}")

try:
    from services.technical_analysis import TechnicalAnalysisService
    logger.info("✅ TechnicalAnalysisService imported successfully")
except Exception as e:
    TechnicalAnalysisService = None
    logger.error(f"❌ TechnicalAnalysisService failed: {str(e)} | Type: {type(e).__name__}")
    import traceback
    logger.error(f"Full traceback: {traceback.format_exc()}")

try:
    from services.llm_agent import TradingAgent, ToolExecutor
    logger.info("✅ LLM Agent services imported successfully")
except Exception as e:
    TradingAgent = None
    ToolExecutor = None
    logger.error(f"❌ LLM Agent services failed: {str(e)} | Type: {type(e).__name__}")
    import traceback
    logger.error(f"Full traceback: {traceback.format_exc()}")

try:
    from services.news_sentiment import NewsSentimentService
except ImportError:
    NewsSentimentService = None
    logger.warning("NewsSentimentService not available")

try:
    from services.weekly_scheduler import WeeklyScheduler
except ImportError:
    WeeklyScheduler = None
    logger.warning("WeeklyScheduler not available")

try:
    from core.message_handler import MessageHandler
except ImportError:
    MessageHandler = None
    logger.warning("MessageHandler not available")

# ===== FUNDAMENTAL ANALYSIS ENGINE IMPORT =====
try:
    from services.fundamental_analysis import FundamentalAnalysisEngine, FundamentalAnalysisTool, AnalysisDepth
    logger.info("✅ FundamentalAnalysisEngine imported successfully")
except Exception as e:
    FundamentalAnalysisEngine = None
    FundamentalAnalysisTool = None
    AnalysisDepth = None
    logger.error(f"❌ FundamentalAnalysisEngine failed: {str(e)} | Type: {type(e).__name__}")
    import traceback
    logger.error(f"Full traceback: {traceback.format_exc()}")

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

# Global services
db_service = None
openai_service = None
twilio_service = None
message_handler = None
scheduler_task = None
ta_service = None  # Integrated TA service
trading_agent = None  # Hybrid LLM agent
tool_executor = None  # Tool execution engine
news_service = None
fundamental_service = None  # Fundamental analysis engine
fundamental_tool = None     # Fundamental analysis tool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    global db_service, openai_service, twilio_service, message_handler, ta_service, trading_agent, tool_executor, news_service, fundamental_service, fundamental_tool, scheduler_task
    
    logger.info("🚀 Starting SMS Trading Bot with Hybrid LLM Agent + Fundamental Analysis...")
    
    try:
        # Initialize Database Service
        if DatabaseService:
            db_service = DatabaseService()
            try:
                await db_service.initialize()
                logger.info("✅ Database service initialized")
            except Exception as e:
                logger.error(f"❌ Database service failed: {e}")
                logger.error(f"MongoDB URL: {settings.mongodb_url[:50]}...")
                logger.error(f"Redis URL: {settings.redis_url[:30]}...")
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
                logger.info("✅ TA Service initialized with API key: {'Set' if getattr(settings, 'eodhd_api_key', None) else 'Missing'}")
            except Exception as e:
                logger.error(f"❌ Technical Analysis service failed: {e}")
                ta_service = None
        
        # Initialize Fundamental Analysis Service
        if FundamentalAnalysisEngine:
            try:
                # Check if Redis is available for caching
                redis_available = db_service and hasattr(db_service, 'redis') and db_service.redis
                if redis_available:
                    logger.info("✅ Redis available for fundamental analysis caching")
                else:
                    logger.warning("⚠️ Redis unavailable for fundamental analysis: Using weekly caching fallback")
                
                fundamental_service = FundamentalAnalysisEngine(
                    eodhd_api_key=getattr(settings, 'eodhd_api_key', None),
                    redis_client=db_service.redis if redis_available else None,
                    cache_ttl=3600 if redis_available else 604800  # 1 hour with Redis, 1 week without
                )
                logger.info("✅ Fundamental Analysis Engine initialized with weekly caching")
            except Exception as e:
                logger.error(f"❌ Fundamental Analysis service failed: {e}")
                fundamental_service = None
        
        # Initialize News Sentiment Service
        if NewsSentimentService:
            try:
                news_service = NewsSentimentService(
                    marketaux_api_key=getattr(settings, 'marketaux_api_key', None),
                    redis_client=db_service.redis if db_service and hasattr(db_service, 'redis') else None,
                    openai_service=openai_service
                )
                logger.info("✅ News Sentiment Service initialized with MarketAux API: {'Set' if getattr(settings, 'marketaux_api_key', None) else 'Missing'}")
                logger.info(f"✅ Redis available: {db_service and hasattr(db_service, 'redis') and db_service.redis is not None}")
                logger.info(f"✅ OpenAI service available: {openai_service is not None}")
            except Exception as e:
                logger.error(f"❌ News Sentiment service failed: {e}")
                news_service = None
        
        # Initialize Trading Agent (Hybrid LLM Agent)
        if TradingAgent:
            try:
                if openai_service and personality_engine:
                    trading_agent = TradingAgent(
                        openai_client=openai_service,
                        personality_engine=personality_engine,
                        database_service=db_service  # Pass database service for context
                    )
                    logger.info("✅ Trading agent initialized")
                else:
                    logger.warning("⚠️ Trading agent not initialized - missing OpenAI service or personality engine")
                    trading_agent = None
            except Exception as e:
                logger.error(f"❌ Trading agent failed: {e}")
                trading_agent = None
        
        # Initialize Fundamental Analysis Tool
        if FundamentalAnalysisTool:
            try:
                if fundamental_service:
                    fundamental_tool = FundamentalAnalysisTool(fundamental_service)
                    logger.info("✅ Fundamental Analysis Tool initialized")
                else:
                    fundamental_tool = None
                    logger.warning("⚠️ Fundamental Analysis Tool not available")
            except Exception as e:
                logger.error(f"❌ Fundamental Analysis Tool failed: {e}")
                fundamental_tool = None
        
        # Initialize Tool Executor
        if ToolExecutor:
            try:
                tool_executor = ToolExecutor(
                    ta_service=ta_service,
                    portfolio_service=None,  # Not implemented yet
                    screener_service=None,   # Not implemented yet
                    news_service=news_service,
                    fundamental_tool=fundamental_tool
                )
                logger.info("✅ Tool executor initialized with TA + News + Fundamental Analysis")
            except Exception as e:
                logger.error(f"❌ Tool executor failed: {e}")
                tool_executor = None
        
        # Initialize Message Handler
        if MessageHandler:
            try:
                message_handler = MessageHandler()
                logger.info("✅ Enhanced message handler initialized with context support")
            except Exception as e:
                logger.error(f"❌ Message handler failed: {e}")
                message_handler = None
        
        # Final status report
        logger.info("🎯 Complete Intelligence Suite: TA + News + Fundamental Analysis + LLM Agent")
        
        # Log available analysis engines
        available_engines = []
        if ta_service:
            available_engines.append("Technical Analysis")
        if news_service:
            available_engines.append("News Sentiment")
        if fundamental_service:
            available_engines.append("Fundamental Analysis")
        
        logger.info(f"📊 Available Analysis Engines: {', '.join(available_engines) if available_engines else 'None'}")
        logger.info("✅ SMS Trading Bot startup completed")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if not settings.testing_mode:
            # Don't raise in development mode to allow partial functionality
            logger.warning("⚠️ Continuing in degraded mode")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SMS Trading Bot...")
    
    try:
        # Close services that need cleanup
        if ta_service and hasattr(ta_service, 'close'):
            await ta_service.close()
            logger.info("✅ Technical Analysis service closed")
        
        if news_service and hasattr(news_service, 'close'):
            await news_service.close()
            logger.info("✅ News Sentiment service closed")
        
        if db_service and hasattr(db_service, 'close'):
            await db_service.close()
            logger.info("✅ Database service closed")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")
    
    logger.info("✅ Shutdown complete")
    
    # Clear global variables (they are already declared as global at the top)
    trading_agent = None
    tool_executor = None

app = FastAPI(
    title="SMS Trading Bot",
    description="Hyper-personalized SMS trading insights with Hybrid LLM Agent + Fundamental Analysis",
    version="2.0.0",
    lifespan=lifespan
)

# After app = FastAPI(...) in main.py
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Create directories
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Mount static files directly in main.py
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include dashboard router
app.include_router(dashboard_router)

# ===== FASTAPI MIDDLEWARE FOR METRICS =====

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Extract ticker from request
    ticker = None
    if hasattr(request, 'query_params'):
        ticker = request.query_params.get('ticker')
    
    # Get cache status from response if possible
    cache_status = None
    if hasattr(response, 'headers') and 'x-cache-status' in response.headers:
        cache_status = response.headers['x-cache-status']
    
    # Record metrics
    metrics.record_request(
        endpoint=str(request.url.path),
        ticker=ticker,
        response_time=response_time,
        cache_status=cache_status,
        error=response.status_code >= 400
    )
    
    return response

# ===== CONVERSATION STORAGE (Simple In-Memory for Demo) =====
conversation_history = defaultdict(list)  # phone_number -> list of messages

def store_conversation(phone_number: str, user_message: str, bot_response: str = None):
    """Store conversation in memory for dashboard viewing"""
    timestamp = datetime.now().isoformat()
    
    # Store user message
    conversation_history[phone_number].append({
        "timestamp": timestamp,
        "direction": "inbound",
        "content": user_message,
        "type": "user_message"
    })
    
    # Store bot response if provided
    if bot_response:
        conversation_history[phone_number].append({
            "timestamp": timestamp,
            "direction": "outbound", 
            "content": bot_response,
            "type": "bot_response"
        })
    
    # Keep only last 50 messages per user
    if len(conversation_history[phone_number]) > 50:
        conversation_history[phone_number] = conversation_history[phone_number][-50:]

# ===== RATE LIMITING =====

async def check_rate_limits(phone_number: str) -> dict:
    """Check if user has exceeded their plan limits"""
    # For demo purposes, always allow messages
    # In production, implement proper rate limiting based on subscription
    return None

# ===== MAIN ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API with Hybrid LLM Agent + Fundamental Analysis", 
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
        "plan_limits": PLAN_LIMITS,
        "endpoints": {
            "health": "/health",
            "sms_webhook": "/webhook/sms",
            "stripe_webhook": "/webhook/stripe",
            "admin_dashboard": "/admin",
            "test_interface": "/dashboard",
            "metrics": "/metrics",
            "user_management": "/admin/users/*",
            "scheduler": "/admin/scheduler/*"
        }
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
        
        # Overall health determination
        critical_services_ok = True  # Always healthy in demo mode
        health_status["overall_status"] = "healthy" if critical_services_ok else "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "services": {
                    "database": "unknown",
                    "redis": "unknown", 
                    "message_handler": "unknown",
                    "weekly_scheduler": "unknown",
                    "trading_agent": "unknown"
                }
            }
        )

# ===== SMS WEBHOOK ENDPOINTS =====

@app.post("/webhook/sms")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming SMS messages from Twilio with Hybrid LLM Agent"""
    try:
        # Parse Twilio webhook data
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return PlainTextResponse("Missing required fields", status_code=400)
        
        # Store the incoming message immediately
        store_conversation(from_number, message_body)
        
        # Process with hybrid LLM agent system
        bot_response = await process_sms_with_hybrid_agent(message_body, from_number)
        
        # Store the bot response
        store_conversation(from_number, message_body, bot_response)
        
        # Process message in background if handler available
        if message_handler:
            background_tasks.add_task(
                message_handler.process_incoming_message,
                from_number,
                message_body
            )
        else:
            logger.info("Message handler not available - used hybrid agent response")
        
        # Return empty TwiML response
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
        # Check rate limits first
        rate_limit_response = await check_rate_limits(phone_number)
        if rate_limit_response:
            return rate_limit_response.get("message", "Rate limit exceeded")
        
        # Update user activity
        if db_service:
            await db_service.update_user_activity(phone_number)
        
        # STEP 1: Parse intent using Hybrid LLM Agent or fallback
        logger.info(f"🧠 Processing with Hybrid Agent: '{message_body}'")
        
        if trading_agent:
            # Use LLM-based intent parsing
            intent_data = await trading_agent.parse_intent(message_body, phone_number)
            logger.info(f"🎯 LLM Intent: {intent_data['intent']} | Symbols: {intent_data.get('symbols', [])} | Confidence: {intent_data.get('confidence', 0):.2f}")
        else:
            # Fallback to regex-based parsing
            intent_data = analyze_message_intent_fallback(message_body)
            logger.info(f"🎯 Fallback Intent: {intent_data['intent']} | Symbols: {intent_data.get('symbols', [])} | Confidence: {intent_data.get('confidence', 0):.2f}")
        
        # STEP 2: Execute required tools
        tool_results = {}
        
        if tool_executor:
            # Use hybrid tool executor
            tool_results = await tool_executor.execute_tools(intent_data, phone_number)
        else:
            # Fallback tool execution
            tool_results = await execute_tools_fallback(intent_data, phone_number)
        
        # STEP 3: Learn from interaction (personality engine)
        if personality_engine:
            personality_engine.learn_from_message(
                phone_number=phone_number,
                message=message_body,
                intent=intent_data
            )
        
        # STEP 4: Generate personalized response
        user_profile = personality_engine.get_user_profile(phone_number) if personality_engine else {}
        
        if trading_agent:
            # Use hybrid LLM agent for response generation
            response_text = await trading_agent.generate_response(
                user_message=message_body,
                intent_data=intent_data,
                tool_results=tool_results,
                user_phone=phone_number,
                user_profile=user_profile
            )
        else:
            # Fallback response generation
            response_text = generate_fallback_response(intent_data, tool_results, user_profile)
        
        # Send SMS response
        if twilio_service and response_text:
            sms_sent = await twilio_service.send_message(phone_number, response_text)
            if not sms_sent:
                logger.error(f"❌ Failed to send SMS to {phone_number}")
        
        # Store conversation in database
        if db_service:
            await db_service.store_conversation(
                phone_number=phone_number,
                user_message=message_body,
                bot_response=response_text,
                intent=intent_data,
                tool_results=tool_results
            )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Processed message in {processing_time:.2f}s using {'Hybrid Agent' if trading_agent else 'Fallback'}")
        
        return response_text
        
    except Exception as e:
        logger.error(f"💥 SMS processing failed for {phone_number}: {traceback.format_exc()}")
        
        # Send error response to user
        error_response = "Sorry, I'm having technical issues right now. Please try again in a moment! 🔧"
        
        if twilio_service:
            await twilio_service.send_message(phone_number, error_response)
        
        return error_response

# ===== FALLBACK FUNCTIONS (For when hybrid agent is not available) =====

def analyze_message_intent_fallback(message: str) -> dict:
    """Fallback regex-based intent analysis with improved symbol extraction"""
    import re
    
    message_lower = message.lower()
    
    # Enhanced symbol extraction with comprehensive filtering
    potential_symbols = re.findall(r'\b[A-Z]{2,5}\b', message.upper())
    
    # Company names to symbols mapping
    company_mappings = {
        'plug power': 'PLUG', 'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
        'amazon': 'AMZN', 'google': 'GOOGL', 'facebook': 'META', 'meta': 'META',
        'nvidia': 'NVDA', 'amd': 'AMD', 'netflix': 'NFLX', 'spotify': 'SPOT',
        'palantir': 'PLTR', 'gamestop': 'GME', 'amc': 'AMC'
    }
    
    # Check for company names in the message
    for company, symbol in company_mappings.items():
        if company in message_lower:
            potential_symbols.append(symbol)
    
    # COMPREHENSIVE exclude_words list to prevent false positives
    exclude_words = {
        # Basic words
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
        'OUR', 'HAD', 'BY', 'DO', 'GET', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO',
        'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'HOW', 'WHAT',
        'WHEN', 'WHERE', 'WHY', 'WILL', 'WITH', 'HAS', 'HIS', 'HIM',
        
        # CRITICAL: Casual words that were causing false positives
        'YO', 'HEY', 'SO', 'OH', 'AH', 'UM', 'UH', 'YEP', 'NAH', 'LOL', 'OMG', 'WOW',
        'BRO', 'FAM', 'THO', 'TBH', 'NGL', 'SMH', 'FML', 'IRL', 'BTW', 'TBF', 'IDK',
        
        # Basic prepositions and particles
        'TO', 'AT', 'IN', 'ON', 'OR', 'OF', 'IS', 'IT', 'BE', 'GO', 'UP', 'MY', 'AS',
        'IF', 'NO', 'WE', 'ME', 'HE', 'AN', 'AM', 'US', 'A', 'I',
        
        # Questions and responses
        'YES', 'YET', 'OUT', 'OFF', 'BAD',
        
        # Trading terms that aren't symbols
        'BUY', 'SELL', 'HOLD', 'CALL', 'PUT', 'BULL', 'BEAR', 'MOON', 'DIP', 'RIP',
        'YOLO', 'HODL', 'FOMO', 'ATH', 'RSI', 'MACD', 'EMA', 'SMA', 'PE', 'DD',
        
        # Time references
        'TODAY', 'THEN', 'SOON', 'LATER', 'WEEK',
        
        # Geographic/org abbreviations
        'AI', 'API', 'CEO', 'CFO', 'IPO', 'ETF', 'SEC', 'FDA', 'FBI', 'CIA', 'NYC',
        'LA', 'SF', 'DC', 'UK', 'US', 'EU', 'JP', 'CN', 'IN', 'CA', 'TX', 'FL',
        
        # Units and currencies
        'K', 'M', 'B', 'T', 'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF',
        
        # Internet slang
        'AF', 'FR', 'NM', 'WTF', 'LMAO', 'ROFL', 'TTYL', 'SWAG', 'LIT',
        
        # Common typos/variations
        'UR', 'CUZ', 'PLZ', 'THX', 'NP', 'YA', 'IM', 'ILL', 'WONT', 'CANT'
    }
    
    # Filter symbols with enhanced validation
    valid_symbols = []
    for symbol in potential_symbols:
        if (symbol not in exclude_words and 
            len(symbol) >= 2 and 
            len(symbol) <= 5 and
            symbol.isalpha() and  # Only letters, no numbers
            not symbol.lower() in ['yo', 'hey', 'so', 'oh']):  # Extra protection
            valid_symbols.append(symbol)
    
    # Remove duplicates while preserving order
    symbols = list(dict.fromkeys(valid_symbols))
    
    # Enhanced Intent detection with fundamental analysis support
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
        requires_tools = ["technical_analysis", "news_sentiment"]  # Default to TA + News
    else:
        intent = "general"
        requires_tools = []
    
    return {
        "intent": intent,
        "symbols": symbols,
        "confidence": 0.4,  # Lower confidence for fallback
        "requires_tools": requires_tools,
        "fallback": True
    }

async def execute_tools_fallback(intent_data: dict, user_phone: str) -> dict:
    """Fallback tool execution without ToolExecutor"""
    results = {}
    
    try:
        requires_tools = intent_data.get("requires_tools", [])
        symbols = intent_data.get("symbols", [])
        
        # Technical Analysis
        if "technical_analysis" in requires_tools and symbols:
            if ta_service:
                for symbol in symbols[:2]:  # Limit to 2 symbols
                    ta_data = await ta_service.analyze_symbol(symbol.upper())
                    if ta_data:
                        if "technical_analysis" not in results:
                            results["technical_analysis"] = {}
                        results["technical_analysis"][symbol] = ta_data
        
        # News Sentiment Analysis
        if "news_sentiment" in requires_tools and symbols:
            if news_service:
                for symbol in symbols[:2]:  # Limit to 2 symbols
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
                for symbol in symbols[:2]:  # Limit to 2 symbols
                    try:
                        # Get user tier for analysis depth
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
            
            if not results.get("fundamental_analysis"):
                results["fundamental_analysis_unavailable"] = True
        
        # If no tools succeeded, mark as unavailable
        if not results:
            results["market_data_unavailable"] = True
        
    except Exception as e:
        logger.error(f"Fallback tool execution failed: {e}")
        results["tool_error"] = str(e)
    
    return results

def generate_fallback_response(intent_data: dict, tool_results: dict, user_profile: dict) -> str:
    """Generate simple fallback response based on user personality"""
    
    intent = intent_data["intent"]
    symbols = intent_data.get("symbols", [])
    
    # Try to get user's communication style for personalization
    style = user_profile.get("communication_style", {}) if user_profile else {}
    formality = style.get("formality", "casual")
    energy = style.get("energy", "moderate")
    
    if intent == "fundamental_analysis" and symbols:
        symbol = symbols[0]
        if tool_results.get("fundamental_analysis", {}).get(symbol):
            fundamental_data = tool_results["fundamental_analysis"][symbol]
            
            # Format response based on user style
            if formality == "casual" and energy == "high":
                response = f"{symbol} Fundamentals looking good! 💪 "
            elif formality == "professional":
                response = f"{symbol} Fundamental Analysis: "
            else:
                response = f"{symbol} fundamentals: "
                
            # Add key fundamental metrics
            if hasattr(fundamental_data, 'overall_score'):
                response += f"Score: {fundamental_data.overall_score:.0f}/100"
            if hasattr(fundamental_data, 'financial_health'):
                response += f" | Health: {fundamental_data.financial_health.value.title()}"
            if hasattr(fundamental_data, 'ratios') and fundamental_data.ratios.pe_ratio:
                response += f" | P/E: {fundamental_data.ratios.pe_ratio:.1f}"
            
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
        
        # Add available analysis components
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
            
            # Format response based on user's style
            if formality == "casual" and energy == "high":
                response = f"{symbol} looking good! 🚀 "
            elif formality == "professional":
                response = f"{symbol} Analysis: "
            else:
                response = f"Here's {symbol}: "
                
            # Add price data if available
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

# ===== ENHANCED PERSONALITY RESPONSE GENERATION =====

async def generate_hyper_personalized_response(user_message: str, phone_number: str) -> str:
    """DEPRECATED: Use process_sms_with_hybrid_agent instead"""
    # This function is kept for backward compatibility but redirects to the new hybrid system
    return await process_sms_with_hybrid_agent(user_message, phone_number)

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

# ===== REMAINING ENDPOINTS =====

@app.get("/debug/test-fundamental/{symbol}")
async def test_fundamental_analysis(symbol: str, depth: str = "standard"):
    """Test the fundamental analysis engine"""
    try:
        if not fundamental_service and not fundamental_tool:
            return {
                "symbol": symbol.upper(),
                "fundamental_service_available": False,
                "error": "Fundamental Analysis Engine not initialized",
                "recommendation": "Check EODHD_API_KEY environment variable and restart"
            }
        
        # Test fundamental analysis
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
                        "cache_status": "weekly_cache",
                        "analysis_depth": depth
                    },
                    "sms_response": result["sms_response"],
                    "service_status": "working",
                    "weekly_caching": True
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

@app.get("/admin")
async def admin_dashboard():
    """Comprehensive admin dashboard"""
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
            "plan_limits": PLAN_LIMITS,
            "popular_tickers": {"count": len(POPULAR_TICKERS), "sample": POPULAR_TICKERS[:10]}
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
    """Simple test interface for SMS webhook testing"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Complete Intelligence Suite Test Interface</title>
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
        .agent-badge { background: #007bff; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
        .engines { background: #e2e3e5; padding: 10px; border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🤖 Complete Intelligence Suite SMS Trading Bot</h1>
    <div class="engines">
        <strong>Available Analysis Engines:</strong> Technical Analysis + News Sentiment + Fundamental Analysis + Hybrid LLM Agent + Personality Learning
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
        <label>Quick Test Messages:</label>
        <button type="button" onclick="setMessage('TSLA fundamentals')">Fundamental Analysis</button>
        <button type="button" onclick="setMessage('yo what\\'s NVDA doing? thinking about calls 🚀')">Technical + News</button>
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
                        <p>SMS webhook processed successfully with <span class="agent-badge">Complete Intelligence Suite</span></p>
                        <pre>${result}</pre>
                        <p><strong>🧠 The system intelligently parsed intent, executed multiple analysis engines, and generated a personalized response!</strong></p>
                        <p><strong>📊 Analysis Engines: Technical Analysis + News Sentiment + Fundamental Analysis</strong></p>
                        <p><strong>🤖 Hybrid LLM Agent coordinated everything with personality learning active.</strong></p>
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Complete Intelligence Suite SMS Trading Bot on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Testing mode: {settings.testing_mode}")
    logger.info(f"🧠 Personality Engine: Active")
    logger.info(f"📈 Technical Analysis: {'Available' if TechnicalAnalysisService else 'Unavailable'}")
    logger.info(f"📰 News Sentiment: {'Available' if NewsSentimentService else 'Unavailable'}")
    logger.info(f"📊 Fundamental Analysis: {'Available' if FundamentalAnalysisEngine else 'Unavailable'}")
    logger.info(f"🤖 Hybrid LLM Agent: {'Available' if TradingAgent else 'Unavailable - check OPENAI_API_KEY'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
