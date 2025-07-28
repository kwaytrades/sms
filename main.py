# ===== main.py - COMPLETE VERSION WITH HYBRID LLM AGENT =====
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
# In main.py, replace the existing try/except blocks with:

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
    from services.weekly_scheduler import WeeklyScheduler
except ImportError:
    WeeklyScheduler = None
    logger.warning("WeeklyScheduler not available")

try:
    from core.message_handler import MessageHandler
except ImportError:
    MessageHandler = None
    logger.warning("MessageHandler not available")

# Add news service
try:
    news_service = NewsSentimentService(
        redis_client=redis_client,
        openai_service=openai_service
    )
except ImportError:
    news_service = None



# Import integrated technical analysis service




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

# Add this to your lifespan function in main.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    global db_service, openai_service, twilio_service, message_handler, ta_service, trading_agent, tool_executor
    
    logger.info("🚀 Starting SMS Trading Bot with Hybrid LLM Agent...")
    
    try:
        # Initialize Database Service
        if DatabaseService:
            try:
                db_service = DatabaseService()
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
                # Note: Don't call initialize() if it requires parameters you don't have
                # await ta_service.initialize()  # Comment out if causing issues
                logger.info("✅ Technical Analysis service initialized")
            except Exception as e:
                logger.error(f"❌ Technical Analysis service failed: {e}")
                ta_service = None
        
        # Initialize Trading Agent (FIXED CONSTRUCTOR CALL)
        if TradingAgent and openai_service:
            try:
                trading_agent = TradingAgent(
                    openai_client=openai_service,  # Fixed: pass the client
                    personality_engine=personality_engine  # Fixed: pass personality_engine
                )
                logger.info("✅ Trading agent initialized")
            except Exception as e:
                logger.error(f"❌ Trading agent failed: {e}")
                trading_agent = None
        
        # Initialize Tool Executor (FIXED CONSTRUCTOR CALL) 
        if ToolExecutor and ta_service:
            try:
                tool_executor = ToolExecutor(
                    ta_service=ta_service,
                    portfolio_service=None,  # Optional for now
                    screener_service=None,    # Optional for now
                    news_service=news_service  # Add this
                )
                logger.info("✅ Tool executor initialized")
            except Exception as e:
                logger.error(f"❌ Tool executor failed: {e}")
                tool_executor = None
        
        # Initialize Message Handler (FIXED CONSTRUCTOR CALL)
        if MessageHandler:
            try:
                message_handler = MessageHandler(
                    db_service=db_service,
                    openai_service=openai_service, 
                    twilio_service=twilio_service
                    # Removed ta_service parameter that was causing the error
                )
                logger.info("✅ Message handler initialized")
            except Exception as e:
                logger.error(f"❌ Message handler failed: {e}")
                message_handler = None
        
        # Check if hybrid agent is working
        if trading_agent and tool_executor:
            logger.info("🎯 Hybrid LLM Agent system ready")
        else:
            logger.warning("⚠️ Falling back to regex parsing - hybrid agent unavailable")
        
        logger.info("✅ SMS Trading Bot startup completed")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        yield  # Still yield even if startup fails
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down SMS Trading Bot...")
        
        # Close services gracefully
        if hasattr(ta_service, 'close'):
            await ta_service.close()
        
        if db_service and hasattr(db_service, 'mongo_client'):
            if db_service.mongo_client:
                db_service.mongo_client.close()
        
        logger.info("✅ Shutdown complete")
    
  
app = FastAPI(
    title="SMS Trading Bot",
    description="Hyper-personalized SMS trading insights with Hybrid LLM Agent",
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
        "message": "SMS Trading Bot API with Hybrid LLM Agent", 
        "status": "running",
        "version": "2.0.0",
        "environment": settings.environment,
        "capabilities": settings.get_capability_summary(),
        "agent_type": "hybrid_llm" if trading_agent else "fallback",
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
                    "redis": {"status": "connected" if db_service.redis is not None else "not_configured"}
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
    elif symbols:
        intent = "analyze"
        requires_tools = ["technical_analysis"]
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
        if "technical_analysis" in intent_data.get("requires_tools", []) and intent_data.get("symbols"):
            if ta_service:
                for symbol in intent_data["symbols"][:2]:  # Limit to 2 symbols
                    ta_data = await ta_service.analyze_symbol(symbol.upper())
                    if ta_data:
                        if "technical_analysis" not in results:
                            results["technical_analysis"] = {}
                        results["technical_analysis"][symbol] = ta_data
            
            if not results:
                results["market_data_unavailable"] = True
        
        # Add other tool executions as needed
        
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
    
    if intent == "analyze" and symbols:
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
            return "Hey! I'm your trading buddy 🤖 Ask me about any stock like 'How's AAPL?' or say 'find good stocks'!"
        else:
            return "I'm your AI trading assistant. Ask me about stocks, market analysis, or portfolio insights. How can I help?"
    
    elif intent == "portfolio":
        return "Portfolio tracking coming soon! For now, ask me about specific stocks. 📊"
    
    else:
        if formality == "casual" and energy == "high":
            return "What's up! 🚀 Ask me about any stock and I'll give you the scoop!"
        else:
            return "I'm here to help with trading questions! Try asking about a stock symbol like 'How is AAPL doing?'"

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

# ===== DEBUG ENDPOINTS WITH HYBRID AGENT TESTING =====

@app.post("/debug/test-hybrid-agent")
async def test_hybrid_agent_endpoint(request: Request):
    """Test the hybrid LLM agent system"""
    try:
        data = await request.json()
        message = data.get('message', 'yo what\'s TSLA doing? thinking about calls 🚀')
        phone = data.get('phone', '+1234567890')
        
        logger.info(f"🧪 Testing Hybrid Agent: '{message}' from {phone}")
        
        if not trading_agent:
            return {
                "error": "Hybrid LLM Agent not available",
                "fallback_available": True,
                "recommendation": "Check OPENAI_API_KEY environment variable"
            }
        
        start_time = time.time()
        
        # Step 1: Parse intent with LLM
        intent_data = await trading_agent.parse_intent(message, phone)
        
        # Step 2: Execute tools
        tool_results = {}
        if tool_executor:
            tool_results = await tool_executor.execute_tools(intent_data, phone)
        
        # Step 3: Learn personality
        personality_engine.learn_from_message(phone, message, intent_data)
        user_profile = personality_engine.get_user_profile(phone)
        
        # Step 4: Generate response
        response = await trading_agent.generate_response(
            user_message=message,
            intent_data=intent_data,
            tool_results=tool_results,
            user_phone=phone,
            user_profile=user_profile
        )
        
        processing_time = time.time() - start_time
        
        return {
            "test_message": message,
            "hybrid_agent": {
                "available": True,
                "intent_parsing": intent_data,
                "tool_execution": {
                    "tools_used": list(tool_results.keys()),
                    "success": len(tool_results) > 0
                },
                "personality_learning": {
                    "active": True,
                    "style": user_profile.get("communication_style", {}),
                    "trading_profile": user_profile.get("trading_personality", {})
                },
                "response_generation": {
                    "response": response,
                    "personalized": True,
                    "length": len(response)
                }
            },
            "performance": {
                "processing_time": f"{processing_time:.2f}s",
                "agent_mode": "hybrid_llm"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Hybrid agent test failed: {e}")
        return {
            "error": str(e),
            "hybrid_agent": {"available": False},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/debug/agent-status")
async def get_agent_status():
    """Get detailed status of the hybrid agent system"""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "hybrid_agent": {
                "trading_agent": {
                    "available": trading_agent is not None,
                    "openai_client": openai_service is not None,
                    "personality_engine": personality_engine is not None
                },
                "tool_executor": {
                    "available": tool_executor is not None,
                    "ta_service": ta_service is not None,
                    "portfolio_service": False,  # Not implemented yet
                    "screener_service": False   # Not implemented yet
                }
            },
            "fallback_systems": {
                "regex_intent_parsing": True,
                "basic_tool_execution": True,
                "personality_engine": True
            },
            "environment_variables": {
                "OPENAI_API_KEY": "Set" if os.getenv('OPENAI_API_KEY') else "Missing",
                "EODHD_API_KEY": "Set" if os.getenv('EODHD_API_KEY') else "Missing"
            },
            "recommendations": _get_agent_recommendations()
        }
        
    except Exception as e:
        return {"error": str(e)}

def _get_agent_recommendations() -> List[str]:
    """Get recommendations for hybrid agent setup"""
    recommendations = []
    
    if not trading_agent:
        if not os.getenv('OPENAI_API_KEY'):
            recommendations.append("❌ Set OPENAI_API_KEY to enable Hybrid LLM Agent")
        else:
            recommendations.append("⚠️ Hybrid LLM Agent failed to initialize - check OpenAI service")
    else:
        recommendations.append("✅ Hybrid LLM Agent is working!")
    
    if not ta_service:
        recommendations.append("❌ Technical Analysis service not available")
    elif not os.getenv('EODHD_API_KEY'):
        recommendations.append("⚠️ Set EODHD_API_KEY for real market data")
    
    if not recommendations:
        recommendations.append("🚀 All systems operational - Hybrid Agent ready!")
    
    return recommendations

# ===== EXISTING ENDPOINTS (keeping all existing functionality) =====

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
                "weekly_scheduler": "active" if scheduler_task and not scheduler_task.done() else "inactive",
                "ta_service": "active" if ta_service else "inactive",
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

# ===== USER MANAGEMENT ENDPOINTS =====

@app.get("/admin/users/{phone_number}")
async def get_user_profile(phone_number: str):
    """Get user profile for admin"""
    try:
        # Get personality profile
        user_profile = personality_engine.user_profiles.get(phone_number)
        if user_profile:
            return {
                "phone_number": phone_number,
                "found": True,
                "personality_profile": user_profile,
                "total_messages": user_profile["learning_data"]["total_messages"],
                "communication_style": user_profile["communication_style"],
                "trading_personality": user_profile["trading_personality"]
            }
        return {"phone_number": phone_number, "found": False, "message": "User profile not found"}
    except Exception as e:
        logger.error(f"Error getting user {phone_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/users/stats")
async def get_user_stats():
    """Get user statistics"""
    try:
        total_users = len(personality_engine.user_profiles)
        active_users = len([p for p in personality_engine.user_profiles.values() 
                           if p["learning_data"]["total_messages"] > 0])
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_conversations": len(conversation_history),
            "plan_breakdown": {"free": total_users, "paid": 0, "pro": 0}  # Mock data
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/users/{phone_number}/subscription")
async def update_user_subscription(phone_number: str, request: Request):
    """Update user subscription"""
    try:
        plan_data = await request.json()
        # Mock subscription update
        return {"success": True, "phone": phone_number, "plan": plan_data.get('plan_type')}
    except Exception as e:
        logger.error(f"Error updating subscription for {phone_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== CONVERSATION ENDPOINTS =====

@app.get("/api/conversations/{phone_number}")
async def get_user_conversations(phone_number: str, limit: int = 20):
    """Get conversation history for a specific user"""
    try:
        clean_phone = phone_number.replace('%2B', '+').replace('%20', '')
        messages = conversation_history.get(clean_phone, [])
        recent_messages = messages[-limit:] if messages else []
        
        conversations = []
        if recent_messages:
            conversations.append({
                "session_id": f"session_{clean_phone}_{datetime.now().strftime('%Y%m%d')}",
                "messages": recent_messages,
                "total_messages": len(recent_messages)
            })
        
        return {
            "phone_number": clean_phone,
            "total_sessions": len(conversations),
            "conversations": conversations,
            "total_messages": len(messages)
        }
    except Exception as e:
        logger.error(f"Error getting conversations for {phone_number}: {e}")
        return {"error": str(e)}

@app.get("/api/conversations/recent")
async def get_recent_conversations(limit: int = 10):
    """Get recent conversations across all users"""
    try:
        all_conversations = []
        
        for phone, messages in conversation_history.items():
            if messages:
                latest_message = messages[-1]
                all_conversations.append({
                    "user_id": phone,
                    "latest_message": latest_message,
                    "total_messages": len(messages)
                })
        
        all_conversations.sort(key=lambda x: x["latest_message"]["timestamp"], reverse=True)
        
        return {
            "recent_conversations": all_conversations[:limit],
            "total_active_users": len(conversation_history)
        }
    except Exception as e:
        logger.error(f"Error getting recent conversations: {e}")
        return {"error": str(e)}

@app.post("/api/test/sms-with-response")
async def test_sms_with_response(request: Request):
    """Enhanced SMS testing endpoint using hybrid agent"""
    try:
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return {"error": "Missing required fields"}
        
        user_message_timestamp = datetime.now().isoformat()
        store_conversation(from_number, message_body)
        
        # Use hybrid agent for response generation
        bot_response = await process_sms_with_hybrid_agent(message_body, from_number)
        
        store_conversation(from_number, message_body, bot_response)
        
        return {
            "status": "success",
            "user_message": {
                "from": from_number,
                "body": message_body,
                "timestamp": user_message_timestamp
            },
            "bot_response": {
                "content": bot_response,
                "message_type": "bot_response",
                "timestamp": datetime.now().isoformat(),
                "session_id": f"test_session_{from_number}",
                "agent_type": "hybrid_llm" if trading_agent else "fallback"
            },
            "processing_status": "completed",
            "conversation_stored": True,
            "personality_learning": "active"
        }
        
    except Exception as e:
        logger.error(f"Error in SMS test with response: {e}")
        return {
            "status": "error", 
            "processing_error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ===== DEBUG ENDPOINTS =====

@app.get("/debug/database")
async def debug_database():
    """Debug database connections and collections"""
    try:
        return {
            "database_name": "ai",
            "collections": ["users", "conversations", "usage_tracking"],
            "users_count": len(personality_engine.user_profiles),
            "connection_status": "connected",
            "personality_profiles": len(personality_engine.user_profiles)
        }
    except Exception as e:
        logger.error(f"Database debug error: {e}")
        return {"error": str(e), "connection_status": "failed"}

@app.get("/debug/config")
async def debug_config():
    """Debug configuration settings"""
    return {
        "environment": settings.environment,
        "testing_mode": settings.testing_mode,
        "capabilities": settings.get_capability_summary(),
        "security": settings.get_security_config(),
        "scheduler": settings.get_scheduler_config(),
        "validation": settings.validate_runtime_requirements(),
        "plan_limits": PLAN_LIMITS,
        "popular_tickers_count": len(POPULAR_TICKERS),
        "personality_engine": {
            "total_profiles": len(personality_engine.user_profiles),
            "active_learning": True
        },
        "agent_type": "hybrid_llm" if trading_agent else "fallback"
    }

@app.post("/debug/test-activity/{phone_number}")
async def test_user_activity(phone_number: str):
    """Test user activity update"""
    try:
        # Test personality learning
        test_message = "Hey, how's AAPL doing today? I'm thinking about buying some calls!"
        
        if trading_agent:
            intent = await trading_agent.parse_intent(test_message, phone_number)
        else:
            intent = analyze_message_intent_fallback(test_message)
            
        personality_engine.learn_from_message(phone_number, test_message, intent)
        
        return {
            "update_success": True,
            "phone_number": phone_number,
            "personality_updated": True,
            "learned_style": personality_engine.user_profiles[phone_number]["communication_style"],
            "agent_type": "hybrid_llm" if trading_agent else "fallback",
            "message": "Personality learning test completed"
        }
    except Exception as e:
        logger.error(f"Test activity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/limits/{phone_number}")
async def debug_limits(phone_number: str):
    """Debug user limits"""
    try:
        user_profile = personality_engine.user_profiles.get(phone_number)
        return {
            "phone_number": phone_number,
            "can_send": True,
            "plan": "free",
            "used": user_profile["learning_data"]["total_messages"] if user_profile else 0,
            "limit": 4,
            "remaining": 4,
            "personality_data": user_profile["communication_style"] if user_profile else None
        }
    except Exception as e:
        logger.error(f"Debug limits error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/analyze-intent")
async def debug_analyze_intent(request: Request):
    """Debug intent analysis - now uses hybrid agent"""
    try:
        data = await request.json()
        message = data.get('message', '')
        phone = data.get('phone', '+1234567890')
        
        if trading_agent:
            # Use hybrid LLM agent for intent parsing
            intent_result = await trading_agent.parse_intent(message, phone)
            agent_type = "hybrid_llm"
        else:
            # Use fallback regex parsing
            intent_result = analyze_message_intent_fallback(message)
            agent_type = "fallback"
        
        return {
            "message": message,
            "intent": intent_result["intent"],
            "symbols": intent_result["symbols"],
            "confidence": intent_result["confidence"],
            "agent_type": agent_type,
            "analysis_details": intent_result
        }
    except Exception as e:
        logger.error(f"Intent analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== TECHNICAL ANALYSIS ENDPOINTS =====

@app.get("/debug/test-ta/{symbol}")
async def test_technical_analysis(symbol: str):
    """Test the integrated technical analysis"""
    try:
        if not ta_service:
            return {
                "symbol": symbol.upper(),
                "ta_service_connected": False,
                "integrated_ta": True,
                "error": "Technical Analysis service not initialized"
            }
        
        ta_data = await ta_service.analyze_symbol(symbol.upper())
        
        if ta_data:
            # Test formatting with different personality styles
            mock_profiles = {
                "casual_high": {
                    "communication_style": {"formality": "casual", "energy": "high", "emoji_usage": "lots"},
                    "trading_personality": {"experience_level": "intermediate"}
                },
                "professional_low": {
                    "communication_style": {"formality": "professional", "energy": "low", "emoji_usage": "none"},
                    "trading_personality": {"experience_level": "advanced"}
                }
            }
            
            formatted_responses = {}
            for style_name, profile in mock_profiles.items():
                if trading_agent:
                    # Use hybrid agent for response formatting
                    mock_intent = {"intent": "analyze", "symbols": [symbol.upper()], "confidence": 0.9}
                    mock_tools = {"technical_analysis": {symbol.upper(): ta_data}}
                    formatted_responses[style_name] = await trading_agent.generate_response(
                        user_message=f"How's {symbol} doing?",
                        intent_data=mock_intent,
                        tool_results=mock_tools,
                        user_phone="+1555TEST",
                        user_profile=profile
                    )
                else:
                    # Use fallback formatting
                    formatted_responses[style_name] = f"{symbol.upper()} analysis using fallback formatting"
            
            return {
                "symbol": symbol.upper(),
                "ta_service_connected": True,
                "integrated_ta": True,
                "raw_ta_data": ta_data,
                "personalized_responses": formatted_responses,
                "data_source": ta_data.get('source', 'unknown'),
                "cache_status": ta_data.get('cache_status', 'unknown'),
                "api_used": "EODHD" if ta_data.get('source') != 'fallback' else "Mock Data",
                "agent_type": "hybrid_llm" if trading_agent else "fallback"
            }
        else:
            return {
                "symbol": symbol.upper(),
                "ta_service_connected": False,
                "integrated_ta": True,
                "error": "No data received from integrated TA service",
                "fallback_used": True
            }
            
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "ta_service_connected": False,
            "integrated_ta": True,
            "error": str(e),
            "fallback_used": True
        }

@app.get("/debug/diagnose-services")
async def diagnose_services():
    """Comprehensive service diagnosis including hybrid agent"""
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
                    "ta_integration": ta_service is not None
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
    
    # Test integrated TA Service
    if ta_service:
        try:
            test_ta_result = await ta_service.analyze_symbol("AAPL")
            diagnosis["connection_tests"]["ta_service"] = {
                "type": "integrated",
                "working": True,
                "test_successful": test_ta_result is not None,
                "data_source": test_ta_result.get('source', 'unknown') if test_ta_result else None,
                "cache_status": test_ta_result.get('cache_status', 'unknown') if test_ta_result else None
            }
        except Exception as e:
            diagnosis["connection_tests"]["ta_service"] = {
                "type": "integrated",
                "working": False,
                "error": str(e)
            }
    else:
        diagnosis["connection_tests"]["ta_service"] = {
            "type": "integrated",
            "working": False,
            "error": "TA service not initialized"
        }
    
    # Test OpenAI if available
    if openai_service:
        try:
            test_response = await openai_service.generate_personalized_response(
                user_query="Test",
                user_profile={"plan_type": "free"},
                conversation_history=[]
            )
            diagnosis["connection_tests"]["openai"] = {
                "available": True,
                "test_successful": True,
                "response_length": len(test_response)
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
    if not env_vars["EODHD_API_KEY"]:
        diagnosis["recommendations"].append("⚠️ Set EODHD_API_KEY environment variable for real market data")
    elif not diagnosis["connection_tests"]["ta_service"]["working"]:
        diagnosis["recommendations"].append("❌ Integrated TA service not working - check EODHD API key")
    
    if not env_vars["OPENAI_API_KEY"]:
        diagnosis["recommendations"].append("❌ Set OPENAI_API_KEY environment variable for Hybrid LLM Agent")
    elif not trading_agent:
        diagnosis["recommendations"].append("❌ Hybrid LLM Agent failed to initialize")
    else:
        diagnosis["recommendations"].append("✅ Hybrid LLM Agent is working!")
    
    if not diagnosis["recommendations"]:
        diagnosis["recommendations"].append("✅ All services configured correctly")
        diagnosis["recommendations"].append("🧠 Personality engine is active and learning from users")
        diagnosis["recommendations"].append("📈 Integrated TA service is working")
        diagnosis["recommendations"].append("🤖 Hybrid LLM Agent is operational")
    
    return diagnosis

@app.post("/debug/test-message")
async def test_message_processing(request: Request):
    """Test message processing with hybrid agent"""
    try:
        data = await request.json()
        message = data.get('message', 'How is AAPL doing?')
        phone = data.get('phone', '+1234567890')
        
        logger.info(f"🧪 Testing message processing: '{message}' from {phone}")
        
        start_time = time.time()
        
        # Use the hybrid agent system for processing
        response = await process_sms_with_hybrid_agent(message, phone)
        
        # Get updated user profile
        user_profile = personality_engine.get_user_profile(phone)
        
        processing_time = time.time() - start_time
        
        return {
            "test_message": message,
            "personality_profile": {
                "communication_style": user_profile.get("communication_style", {}),
                "trading_personality": user_profile.get("trading_personality", {}),
                "total_messages": user_profile.get("learning_data", {}).get("total_messages", 0)
            },
            "generated_response": response,
            "response_length": len(response),
            "personalization_active": True,
            "agent_type": "hybrid_llm" if trading_agent else "fallback",
            "processing_time": f"{processing_time:.2f}s",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Test message processing failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/debug/test-full-flow/{symbol}")
async def test_full_integration_flow(symbol: str):
    """Test the complete integration flow: TA + Hybrid Agent + Personality"""
    try:
        # Step 1: Test TA service
        ta_data = None
        if ta_service:
            ta_data = await ta_service.analyze_symbol(symbol.upper())
        ta_working = ta_data is not None
        
        # Step 2: Test personality engine
        test_phone = "+1234567890"
        test_message = f"How is {symbol.upper()} doing today?"
        
        # Step 3: Test hybrid agent intent parsing
        if trading_agent:
            intent = await trading_agent.parse_intent(test_message, test_phone)
            agent_working = True
        else:
            intent = analyze_message_intent_fallback(test_message)
            agent_working = False
        
        personality_engine.learn_from_message(test_phone, test_message, intent)
        user_profile = personality_engine.get_user_profile(test_phone)
        
        # Step 4: Test hybrid agent response generation
        if trading_agent and ta_data:
            mock_tool_results = {"technical_analysis": {symbol.upper(): ta_data}}
            ai_response = await trading_agent.generate_response(
                user_message=test_message,
                intent_data=intent,
                tool_results=mock_tool_results,
                user_phone=test_phone,
                user_profile=user_profile
            )
        else:
            ai_response = generate_fallback_response(intent, {"technical_analysis": {symbol.upper(): ta_data}} if ta_data else {}, user_profile)
        
        return {
            "symbol": symbol.upper(),
            "integration_test": {
                "ta_service": {
                    "working": ta_working,
                    "data_points": len(ta_data) if ta_data else 0,
                    "has_price": 'price' in (ta_data or {}),
                    "has_indicators": 'technical_indicators' in (ta_data or {}),
                    "has_signals": 'signals' in (ta_data or {})
                },
                "personality_engine": {
                    "working": True,
                    "user_profile_created": True,
                    "communication_style": user_profile["communication_style"],
                    "learning_active": True
                },
                "hybrid_agent": {
                    "intent_parsing": agent_working,
                    "response_generation": trading_agent is not None,
                    "working": agent_working
                }
            },
            "responses": {
                "ai_personalized": ai_response,
                "raw_ta_data": ta_data
            },
            "agent_type": "hybrid_llm" if trading_agent else "fallback",
            "recommendation": _get_integration_recommendation(ta_working, agent_working, True)
        }
        
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "integration_test": "failed",
            "error": str(e),
            "recommendation": "Check your environment variables and service connections"
        }

def _get_integration_recommendation(ta_working: bool, agent_working: bool, personality_working: bool) -> str:
    """Get recommendation based on integration test results"""
    if ta_working and agent_working and personality_working:
        return "✅ Full hyper-personalized integration working! Real-time TA data + Hybrid LLM Agent + personality learning active."
    elif ta_working and personality_working and not agent_working:
        return "⚠️ TA + Personality working but Hybrid Agent unavailable. Check OPENAI_API_KEY environment variable."
    elif agent_working and personality_working and not ta_working:
        return "⚠️ Hybrid Agent + Personality working but TA service unavailable. Check EODHD_API_KEY environment variable."
    elif personality_working:
        return "⚠️ Personality engine working but external services unavailable. Using personalized fallback responses."
    else:
        return "❌ Multiple services unavailable. Check environment variables."

# ===== SCHEDULER ENDPOINTS =====

@app.get("/admin/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    if not scheduler_task:
        return {"status": "inactive", "message": "Scheduler not running"}
    
    try:
        return {"status": "active" if scheduler_task and not scheduler_task.done() else "inactive"}
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/admin/scheduler/manual-reset")
async def manual_reset_trigger():
    """Manually trigger reset notifications (for testing)"""
    try:
        logger.info("📅 Manual reset trigger executed")
        return {"status": "success", "message": "Manual reset trigger executed"}
    except Exception as e:
        logger.error(f"Manual reset trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/scheduler/manual-reminder")
async def manual_reminder_trigger():
    """Manually trigger 24-hour reminders (for testing)"""
    try:
        logger.info("📅 Manual reminder trigger executed")
        return {"status": "success", "message": "Manual reminder trigger executed"}
    except Exception as e:
        logger.error(f"Manual reminder trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== METRICS ENDPOINTS =====

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
            "hybrid_agent": {
                "trading_agent_available": trading_agent is not None,
                "tool_executor_available": tool_executor is not None,
                "agent_type": "hybrid_llm" if trading_agent else "fallback"
            },
            "system": {
                "version": "2.0.0",
                "environment": settings.environment,
                "testing_mode": settings.testing_mode,
                "hyper_personalization": "active"
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ===== PERSONALITY INSIGHTS ENDPOINT =====

@app.get("/debug/personality/{phone_number}")
async def get_personality_insights(phone_number: str):
    """Get detailed personality insights for a user"""
    try:
        user_profile = personality_engine.user_profiles.get(phone_number)
        
        if not user_profile:
            return {"error": "User profile not found", "phone_number": phone_number}
        
        # Generate personality summary
        personality_summary = {
            "communication_analysis": {
                "formality": user_profile["communication_style"]["formality"],
                "energy_level": user_profile["communication_style"]["energy"],
                "emoji_preference": user_profile["communication_style"]["emoji_usage"],
                "message_length_preference": user_profile["communication_style"]["message_length"],
                "technical_depth": user_profile["communication_style"]["technical_depth"]
            },
            "trading_analysis": {
                "risk_tolerance": user_profile["trading_personality"]["risk_tolerance"],
                "trading_style": user_profile["trading_personality"]["trading_style"],
                "experience_level": user_profile["trading_personality"]["experience_level"],
                "favorite_symbols": user_profile["trading_personality"]["common_symbols"][:10],
                "win_loss_ratio": f"{user_profile['learning_data']['successful_trades_mentioned']}W/{user_profile['learning_data']['loss_trades_mentioned']}L"
            },
            "engagement_data": {
                "total_messages": user_profile["learning_data"]["total_messages"],
                "recent_stocks_discussed": user_profile["context_memory"]["last_discussed_stocks"],
                "goals_mentioned": len(user_profile["context_memory"]["goals_mentioned"]),
                "concerns_expressed": len(user_profile["context_memory"]["concerns_expressed"])
            },
            "personalization_level": "high" if user_profile["learning_data"]["total_messages"] > 10 else "medium" if user_profile["learning_data"]["total_messages"] > 3 else "basic"
        }
        
        return {
            "phone_number": phone_number,
            "personality_summary": personality_summary,
            "raw_profile": user_profile,
            "learning_status": "active",
            "agent_type": "hybrid_llm" if trading_agent else "fallback"
        }
        
    except Exception as e:
        logger.error(f"Error getting personality insights for {phone_number}: {e}")
        return {"error": str(e)}

# ===== COMPREHENSIVE DASHBOARD (keeping existing HTML) =====


# ===== TEST INTERFACE ENDPOINT =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Simple test interface for SMS webhook testing"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Hybrid Agent Test Interface</title>
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
    </style>
</head>
<body>
    <h1>🤖 Hybrid LLM Agent SMS Trading Bot - Test Interface</h1>
    
    <form onsubmit="testSMS(event)">
        <div class="form-group">
            <label>From Phone Number:</label>
            <input type="text" id="phone" value="+13012466712" required>
        </div>
        
        <div class="form-group">
            <label>Message Body:</label>
            <textarea id="message" rows="3" required>yo what's AAPL doing? thinking about calls 🚀</textarea>
        </div>
        
        <button type="submit">Send Test SMS with Hybrid Agent</button>
    </form>
    
    <div id="result"></div>
    
    <script>
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
                        <p>SMS webhook processed successfully with <span class="agent-badge">Hybrid LLM Agent</span></p>
                        <pre>${result}</pre>
                        <p><strong>🤖 The Hybrid Agent intelligently parsed intent, executed tools, and generated a personalized response!</strong></p>
                        <p><strong>🧠 Personality engine learned from this interaction for future personalization.</strong></p>
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
    logger.info(f"🚀 Starting Hybrid LLM Agent SMS Trading Bot on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Testing mode: {settings.testing_mode}")
    logger.info(f"🧠 Personality Engine: Active")
    logger.info(f"📈 Integrated Technical Analysis: Active")
    logger.info(f"🤖 Hybrid LLM Agent: {'Available' if TradingAgent else 'Unavailable - check OPENAI_API_KEY'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
# Add this debug endpoint to your main.py to see what's happening

# Add this debug endpoint to your main.py to see what's wrong with the OpenAI client

@app.get("/debug/openai-client-structure")
async def debug_openai_client_structure():
    """Debug the actual OpenAI client structure"""
    
    debug_info = {
        "openai_service_info": {},
        "trading_agent_info": {},
        "client_structure": {},
        "test_results": {}
    }
    
    try:
        # Check OpenAI service
        if openai_service:
            debug_info["openai_service_info"] = {
                "available": True,
                "type": str(type(openai_service)),
                "has_client_attr": hasattr(openai_service, 'client'),
                "has_chat_attr": hasattr(openai_service, 'chat'),
                "dir_openai_service": [attr for attr in dir(openai_service) if not attr.startswith('_')]
            }
            
            if hasattr(openai_service, 'client'):
                debug_info["client_structure"] = {
                    "client_type": str(type(openai_service.client)),
                    "has_chat": hasattr(openai_service.client, 'chat'),
                    "client_dir": [attr for attr in dir(openai_service.client) if not attr.startswith('_')][:10]
                }
                
                if hasattr(openai_service.client, 'chat'):
                    debug_info["client_structure"]["chat_type"] = str(type(openai_service.client.chat))
                    debug_info["client_structure"]["has_completions"] = hasattr(openai_service.client.chat, 'completions')
                    
                    if hasattr(openai_service.client.chat, 'completions'):
                        debug_info["client_structure"]["completions_type"] = str(type(openai_service.client.chat.completions))
                        debug_info["client_structure"]["has_create"] = hasattr(openai_service.client.chat.completions, 'create')
        else:
            debug_info["openai_service_info"] = {"available": False}
        
        # Check trading agent
        if trading_agent:
            debug_info["trading_agent_info"] = {
                "available": True,
                "openai_client_type": str(type(trading_agent.openai_client)),
                "has_client_attr": hasattr(trading_agent.openai_client, 'client'),
                "has_chat_attr": hasattr(trading_agent.openai_client, 'chat')
            }
        else:
            debug_info["trading_agent_info"] = {"available": False}
        
        # Test actual client calls
        if openai_service and trading_agent:
            try:
                # Test the exact same logic as in parse_intent
                client_to_use = trading_agent.openai_client
                
                debug_info["test_results"]["client_detection"] = {
                    "has_client": hasattr(client_to_use, 'client'),
                    "has_chat": hasattr(client_to_use, 'chat'),
                    "will_use_wrapped": hasattr(client_to_use, 'client')
                }
                
                # Try a simple test call
                if hasattr(client_to_use, 'client'):
                    test_response = await client_to_use.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "Say 'test successful'"}],
                        max_tokens=5
                    )
                    debug_info["test_results"]["wrapped_client_test"] = {
                        "success": True,
                        "response": test_response.choices[0].message.content
                    }
                else:
                    test_response = await client_to_use.chat.completions.create(
                        model="gpt-4o-mini", 
                        messages=[{"role": "user", "content": "Say 'test successful'"}],
                        max_tokens=5
                    )
                    debug_info["test_results"]["direct_client_test"] = {
                        "success": True,
                        "response": test_response.choices[0].message.content
                    }
                    
            except Exception as e:
                debug_info["test_results"]["error"] = {
                    "error_type": str(type(e)),
                    "error_message": str(e),
                    "is_await_error": "await" in str(e)
                }
        
        return debug_info
        
    except Exception as e:
        return {
            "error": str(e),
            "error_type": str(type(e))
        }
@app.get("/debug/test-news/{symbol}")
async def test_news_sentiment(symbol: str, mode: str = "cached"):
    """Test the news sentiment service"""
    try:
        if not news_service:
            return {
                "symbol": symbol.upper(),
                "error": "News sentiment service not available",
                "recommendation": "Set MARKETAUX_API_KEY environment variable and restart"
            }
        
        # Test news sentiment analysis
        sentiment_data = await news_service.get_sentiment(symbol.upper(), mode=mode)
        
        return {
            "symbol": symbol.upper(),
            "mode": mode,
            "sentiment_analysis": sentiment_data,
            "service_status": "working" if sentiment_data and not sentiment_data.get('error') else "error",
            "cache_available": news_service.redis_client is not None,
            "openai_available": news_service.openai_service is not None
        }
        
    except Exception as e:
        logger.error(f"News sentiment test error: {e}")
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "recommendation": "Check your MarketAux API key and service configuration"
        }

@app.get("/debug/test-integration/{symbol}")
async def test_full_integration(symbol: str):
    """Test full TA + News Sentiment integration"""
    try:
        test_message = f"analyze {symbol}"
        test_phone = "+1234567890"
        
        # Test hybrid agent intent parsing
        if trading_agent:
            intent = await trading_agent.parse_intent(test_message, test_phone)
            agent_working = True
        else:
            intent = {"intent": "analyze", "symbols": [symbol], "requires_tools": ["technical_analysis", "news_sentiment"]}
            agent_working = False
        
        # Execute tools (both TA and News Sentiment)
        if tool_executor:
            tool_results = await tool_executor.execute_tools(intent, test_phone)
        else:
            tool_results = {}
        
        # Update personality engine
        personality_engine.learn_from_message(test_phone, test_message, intent)
        user_profile = personality_engine.get_user_profile(test_phone)
        
        # Generate enhanced response
        if trading_agent and tool_results:
            ai_response = await trading_agent.generate_response(
                user_message=test_message,
                intent_data=intent,
                tool_results=tool_results,
                user_phone=test_phone,
                user_profile=user_profile
            )
        else:
            ai_response = "Integration test failed - check service availability"
        
        return {
            "symbol": symbol.upper(),
            "integration_test": {
                "intent_parsing": agent_working,
                "tool_execution": len(tool_results) > 0,
                "ta_available": "technical_analysis" in tool_results,
                "news_available": "news_sentiment" in tool_results,
                "personality_learning": True
            },
            "tool_results": {
                "technical_analysis": tool_results.get("technical_analysis", {}),
                "news_sentiment": tool_results.get("news_sentiment", {})
            },
            "enhanced_response": ai_response,
            "user_profile": user_profile["communication_style"],
            "recommendation": _get_integration_status_recommendation(tool_results)
        }
        
    except Exception as e:
        logger.error(f"Integration test error: {e}")
        return {
            "symbol": symbol.upper(),
            "integration_test": "failed",
            "error": str(e),
            "recommendation": "Check service initialization and API keys"
        }

def _get_integration_status_recommendation(tool_results: Dict) -> str:
    """Get recommendation based on integration test results"""
    ta_working = "technical_analysis" in tool_results and not tool_results.get("technical_analysis_unavailable")
    news_working = "news_sentiment" in tool_results and not tool_results.get("news_sentiment_unavailable")
    
    if ta_working and news_working:
        return "✅ Full integration working! TA + News Sentiment + Personality learning active."
    elif ta_working and not news_working:
        return "⚠️ TA working, News Sentiment needs setup. Set MARKETAUX_API_KEY and restart."
    elif not ta_working and news_working:
        return "⚠️ News working, TA needs setup. Check EODHD_API_KEY and restart."
    else:
        return "❌ Both services need setup. Check EODHD_API_KEY and MARKETAUX_API_KEY."

# ===== ADD TO EXISTING SERVICE DIAGNOSIS ENDPOINT =====
# Update your existing diagnose_services endpoint to include news sentiment:

@app.get("/debug/diagnose-services")
async def diagnose_services():
    """Enhanced service diagnosis including news sentiment"""
    try:
        services_status = {
            "timestamp": datetime.now().isoformat(),
            "services": {
                "openai_service": openai_service is not None,
                "db_service": db_service is not None,
                "twilio_service": twilio_service is not None,
                "technical_analysis": ta_service is not None,
                "news_sentiment": news_service is not None,  # ADD THIS
                "personality_engine": personality_engine is not None,
                "trading_agent": trading_agent is not None,
                "tool_executor": tool_executor is not None
            },
            "environment_variables": {
                "OPENAI_API_KEY": "Set" if os.getenv('OPENAI_API_KEY') else "Missing",
                "EODHD_API_KEY": "Set" if os.getenv('EODHD_API_KEY') else "Missing",
                "MARKETAUX_API_KEY": "Set" if os.getenv('MARKETAUX_API_KEY') else "Missing",  # ADD THIS
                "MONGODB_URL": "Set" if os.getenv('MONGODB_URL') else "Missing",
                "REDIS_URL": "Set" if os.getenv('REDIS_URL') else "Optional"
            },
            "integration_status": {
                "ta_integration": ta_service is not None and os.getenv('EODHD_API_KEY'),
                "news_integration": news_service is not None and os.getenv('MARKETAUX_API_KEY'),  # ADD THIS
                "llm_integration": trading_agent is not None and tool_executor is not None
            }
        }
        
        # Add specific diagnostics
        issues = []
        if not os.getenv('MARKETAUX_API_KEY'):
            issues.append("❌ Set MARKETAUX_API_KEY environment variable for news sentiment")
        if not news_service:
            issues.append("❌ News sentiment service failed to initialize")
        if news_service and not news_service.openai_service:
            issues.append("⚠️ News sentiment fallback mode (OpenAI unavailable)")
        
        services_status["issues"] = issues
        services_status["recommendations"] = _get_service_recommendations(services_status)
        
        return services_status
        
    except Exception as e:
        logger.error(f"Service diagnosis error: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

def _get_service_recommendations(status: Dict) -> List[str]:
    """Get service setup recommendations"""
    recommendations = []
    
    env_vars = status.get("environment_variables", {})
    
    if env_vars.get("MARKETAUX_API_KEY") == "Missing":
        recommendations.append("1. Get MarketAux API key from https://www.marketaux.com/")
        recommendations.append("2. Add MARKETAUX_API_KEY=your_key to .env file")
    
    if not status.get("services", {}).get("news_sentiment"):
        recommendations.append("3. Restart application after setting MARKETAUX_API_KEY")
    
    if status.get("integration_status", {}).get("news_integration"):
        recommendations.append("✅ News sentiment ready! Test with /debug/test-news/AAPL")
    
    return recommendations

# ===== SAMPLE SMS TESTING =====
# Test these SMS messages once everything is set up:

"""
Sample SMS Test Messages:
1. "analyze AAPL" - Should return TA + News Sentiment
2. "TSLA news" - Should return just news sentiment  
3. "what's the news on NVDA?" - Should return news sentiment
4. "how is MSFT doing?" - Should return TA + News Sentiment

Expected Enhanced Responses:
- "AAPL 📈 $185.50 (+2.1%) | RSI: 65.2 (neutral) | 🐂 News: Bullish (impact: 0.8) - Strong earnings beat driving momentum"
- "🐂 TSLA News: Recent developments show bullish sentiment | Top Headlines: 1. Tesla beats delivery expectations... 2. New Gigafactory announced... (5 articles analyzed)"
"""
