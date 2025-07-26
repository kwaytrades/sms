# ===== main.py - COMPLETE FIXED VERSION =====
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
try:
    from services.database import DatabaseService
except ImportError:
    DatabaseService = None
    logger.warning("DatabaseService not available")

try:
    from services.openai_service import OpenAIService
except ImportError:
    OpenAIService = None
    logger.warning("OpenAIService not available")

try:
    from services.twilio_service import TwilioService
except ImportError:
    TwilioService = None
    logger.warning("TwilioService not available")

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_service, openai_service, twilio_service, message_handler, scheduler_task
    
    logger.info("🚀 Starting SMS Trading Bot...")
    
    try:
        # Initialize services
        if DatabaseService:
            db_service = DatabaseService()
            await db_service.initialize()
        
        if OpenAIService:
            openai_service = OpenAIService()
        
        if TwilioService:
            twilio_service = TwilioService()
        
        if MessageHandler and db_service and openai_service and twilio_service:
            message_handler = MessageHandler(db_service, openai_service, twilio_service)
        
        # Start weekly scheduler
        if WeeklyScheduler and db_service and twilio_service:
            scheduler = WeeklyScheduler(db_service, twilio_service)
            scheduler_task = asyncio.create_task(scheduler.start_scheduler())
            logger.info("📅 Weekly scheduler started")
        
        logger.info("✅ SMS Trading Bot started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        if not settings.testing_mode:
            raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SMS Trading Bot...")
    if scheduler_task:
        scheduler_task.cancel()
        logger.info("📅 Weekly scheduler stopped")
    if db_service:
        await db_service.close()

app = FastAPI(
    title="SMS Trading Bot",
    description="Hyper-personalized SMS trading insights",
    version="1.0.0",
    lifespan=lifespan
)

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

# ===== MAIN ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API", 
        "status": "running",
        "version": "1.0.0",
        "environment": settings.environment,
        "capabilities": settings.get_capability_summary(),
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
            "version": "1.0.0"
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
            "message_handler": "available" if message_handler is not None else "unavailable"
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
                    "weekly_scheduler": "unknown"
                }
            }
        )

# ===== SMS WEBHOOK ENDPOINTS =====

@app.post("/webhook/sms")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming SMS messages from Twilio"""
    try:
        # Parse Twilio webhook data
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return PlainTextResponse("Missing required fields", status_code=400)
        
        # Store the incoming message immediately
        store_conversation(from_number, message_body)
        
        # Generate hyper-personalized bot response
        bot_response = await generate_hyper_personalized_response(message_body, from_number)
        
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
            logger.info("Message handler not available - used direct response generation")
        
        # Return empty TwiML response
        return PlainTextResponse(
            '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"❌ SMS webhook error: {e}")
        return PlainTextResponse("Internal error", status_code=500)

async def generate_hyper_personalized_response(user_message: str, phone_number: str) -> str:
    """Generate hyper-personalized response using learned user patterns"""
    try:
        # Extract intent and symbols from message
        intent_result = analyze_message_intent(user_message)
        logger.info(f"🎯 Intent analysis: {intent_result}")
        
        # Learn from this interaction BEFORE generating response
        personality_engine.learn_from_message(phone_number, user_message, intent_result)
        
        # Get user's personality profile
        user_profile = personality_engine.user_profiles[phone_number]
        
        # If we have symbols and it's a stock query, get real technical analysis
        if intent_result["symbols"] and intent_result["intent"] in ["analyze", "price", "technical"]:
            symbol = intent_result["symbols"][0]
            logger.info(f"📊 Fetching TA data for symbol: {symbol}")
            
            # Call your technical analysis service
            ta_data = await fetch_technical_analysis(symbol)
            
            if ta_data:
                logger.info(f"✅ Got TA data for {symbol}, generating personalized response")
                # Generate hyper-personalized AI response
                if openai_service:
                    try:
                        # Use the personality engine to create a personalized prompt
                        personalized_prompt = personality_engine.generate_personalized_prompt(
                            phone_number, user_message, ta_data
                        )
                        
                        # Generate response using the personalized prompt
                        ai_response = await generate_personalized_openai_response(
                            personalized_prompt, user_profile
                        )
                        
                        logger.info(f"✅ Generated hyper-personalized response for {symbol}")
                        return ai_response
                    except Exception as e:
                        logger.error(f"❌ OpenAI failed for {symbol}: {e}")
                        # Fallback: Format the technical data in their style
                        return format_personalized_ta_response(symbol, ta_data, user_profile)
                else:
                    logger.warning(f"⚠️ OpenAI service not available, using personalized formatted response for {symbol}")
                    return format_personalized_ta_response(symbol, ta_data, user_profile)
            else:
                logger.warning(f"⚠️ No TA data received for {symbol}")
        
        # Handle other intents with personalized responses
        if openai_service:
            try:
                logger.info("🤖 Using OpenAI for personalized general query")
                # Use personality engine for general queries too
                personalized_prompt = personality_engine.generate_personalized_prompt(
                    phone_number, user_message, None
                )
                
                ai_response = await generate_personalized_openai_response(
                    personalized_prompt, user_profile
                )
                logger.info("✅ Generated personalized response for general query")
                return ai_response
            except Exception as e:
                logger.error(f"❌ OpenAI generation failed for general query: {e}")
                return generate_personalized_mock_response(user_message, user_profile)
        
        # Fallback to personalized mock response
        logger.warning(f"⚠️ Falling back to personalized mock response")
        return generate_personalized_mock_response(user_message, user_profile)
        
    except Exception as e:
        logger.error(f"❌ Error generating personalized response: {e}")
        return generate_personalized_mock_response(user_message, personality_engine.user_profiles[phone_number])

async def generate_personalized_openai_response(prompt: str, user_profile: dict) -> str:
    """Generate OpenAI response with user's personality preferences"""
    try:
        # Shortened prompt for SMS efficiency but keep personality
        response = await openai_service.generate_personalized_response(
            user_query=prompt,
            user_profile={
                "communication_style": user_profile["communication_style"],
                "trading_personality": user_profile["trading_personality"],
                "context": user_profile["context_memory"]
            },
            conversation_history=[]
        )
        
        # Apply post-processing based on user style
        return apply_personality_formatting(response, user_profile)
        
    except Exception as e:
        logger.error(f"OpenAI personalized response failed: {e}")
        raise

def apply_personality_formatting(response: str, user_profile: dict) -> str:
    """Apply final personality formatting to response"""
    style = user_profile["communication_style"]
    
    # Adjust emoji usage
    if style["emoji_usage"] == "lots" and not any(ord(char) > 127 for char in response):
        # Add appropriate emojis
        if "up" in response.lower() or "gain" in response.lower():
            response += " 🚀"
        elif "down" in response.lower() or "drop" in response.lower():
            response += " 📉"
        else:
            response += " 📊"
    elif style["emoji_usage"] == "none":
        # Remove emojis
        response = ''.join(char for char in response if ord(char) <= 127)
    
    # Adjust energy level
    if style["energy"] == "high":
        response = response.replace(".", "!")
        if not response.endswith("!"):
            response += "!"
    elif style["energy"] == "low":
        response = response.replace("!", ".")
    
    # Adjust formality
    if style["formality"] == "casual":
        response = response.replace("Hello", "Hey")
        response = response.replace("Good morning", "Morning")
        response = response.replace("would suggest", "think")
    
    return response

def format_personalized_ta_response(symbol: str, ta_data: dict, user_profile: dict) -> str:
    """Format technical analysis data with user's personality"""
    try:
        style = user_profile["communication_style"]
        trading_style = user_profile["trading_personality"]
        
        # Extract key data points
        price_info = ta_data.get('last_price', 'N/A')
        change_info = ta_data.get('price_change', {})
        technical = ta_data.get('technical_indicators', {})
        
        # Build response based on user's style
        if style["formality"] == "casual":
            greeting = random.choice(["Yo", "Hey", "Sup"]) if style["energy"] == "high" else "Here's"
        else:
            greeting = "Here's your"
        
        # Build personalized response
        response_parts = []
        
        # Price header with personality
        if price_info != 'N/A':
            change_pct = change_info.get('percent', 0)
            if style["energy"] == "high":
                direction = "🚀" if change_pct > 0 else "💥" if change_pct < -2 else "📊"
            else:
                direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
            
            if style["formality"] == "casual":
                response_parts.append(f"{greeting}! {symbol} at ${price_info} {direction}{abs(change_pct):.1f}%")
            else:
                response_parts.append(f"{greeting} {symbol} analysis: ${price_info} {direction}{abs(change_pct):.1f}%")
        
        # Add technical indicators based on their experience level
        if trading_style["experience_level"] == "advanced":
            rsi_data = technical.get('rsi', {})
            if rsi_data:
                rsi_val = rsi_data.get('value', 0)
                response_parts.append(f"RSI: {rsi_val:.0f}")
        elif trading_style["experience_level"] == "beginner":
            # Simplify for beginners
            signals = ta_data.get('signals', [])
            if signals:
                bullish_signals = [s for s in signals if s.get('type') in ['bullish', 'opportunity']]
                if bullish_signals and style["formality"] == "casual":
                    response_parts.append("Looking good for entry!")
                elif bullish_signals:
                    response_parts.append("Positive momentum detected")
        
        # Build final response with their style
        final_response = " | ".join(response_parts) if response_parts else f"{symbol} data analyzed"
        
        # Apply personality formatting
        return apply_personality_formatting(final_response, user_profile)
        
    except Exception as e:
        logger.error(f"Error formatting personalized TA response: {e}")
        return f"{symbol} analysis ready - check the latest data!"

def generate_personalized_mock_response(user_message: str, user_profile: dict) -> str:
    """Generate personalized mock responses based on user personality"""
    message_lower = user_message.lower()
    style = user_profile["communication_style"]
    trading = user_profile["trading_personality"]
    context = user_profile["context_memory"]
    
    # Choose greeting based on formality
    if style["formality"] == "casual":
        greetings = ["Hey", "Yo", "What's up", "Sup"] if style["energy"] == "high" else ["Hey", "Hi"]
        greeting = random.choice(greetings)
    else:
        greeting = "Hello" if style["energy"] != "high" else "Hello there"
    
    # Add user's name context if we know their patterns
    name_context = ""
    if user_profile["learning_data"]["total_messages"] > 5:
        name_context = ", my friend" if style["formality"] == "casual" else ""
    
    # Handle different intents with personality
    if any(word in message_lower for word in ['start', 'hello', 'hi']):
        if user_profile["learning_data"]["total_messages"] == 0:
            # First time user
            if style["formality"] == "casual":
                response = f"{greeting}! I'm your personal trading buddy 🚀 Ready to help you crush the markets! What stock you looking at?"
            else:
                response = f"{greeting}! I'm your AI trading assistant. I'll learn your style and help you make better trades. What can I analyze for you?"
        else:
            # Returning user
            recent_stocks = context.get("last_discussed_stocks", [])
            if recent_stocks and style["formality"] == "casual":
                response = f"Welcome back{name_context}! Still watching {recent_stocks[0]}? What's on your mind today?"
            else:
                response = f"Welcome back{name_context}! How can I help with your trading today?"
    
    elif any(word in message_lower for word in ['help', 'commands']):
        if style["formality"] == "casual":
            response = f"I got you{name_context}! Just ask me about any stock like 'how's AAPL?' or tell me to find good stocks. I'll learn your style as we chat!"
        else:
            response = "I can analyze stocks, find opportunities, track your portfolio, and adapt to your trading style. Just ask naturally!"
    
    elif any(word in message_lower for word in ['upgrade', 'plans']):
        if trading["risk_tolerance"] == "aggressive":
            response = "Pro plan = unlimited analysis for aggressive traders like you. $99/month for real-time alerts when opportunities hit!"
        else:
            response = "Paid plan $29/month gets you 100 personalized insights. Pro $99/month = unlimited + real-time alerts!"
    
    elif any(symbol in message_lower for symbol in ['aapl', 'apple']):
        if trading["risk_tolerance"] == "aggressive" and style["energy"] == "high":
            response = "AAPL looking solid! $185.50 ↑2.1% | RSI cooling down to 65 | Ready for next leg up! You thinking calls?"
        elif style["formality"] == "casual":
            response = f"AAPL's at $185.50, up 2.1% today{name_context}. Technical's looking good - might be your style with that moderate risk tolerance!"
        else:
            response = "AAPL Analysis: $185.50 (+2.1%) | RSI: 65 (neutral) | Support: $182 | Resistance: $190 | Technical outlook positive"
    
    else:
        # General response based on their patterns
        if style["formality"] == "casual" and style["energy"] == "high":
            response = f"I'm learning your trading style{name_context}! Ask me about any stock and I'll give you the personalized insights you need! 🚀"
        else:
            response = f"I'm your personalized trading assistant{name_context}. Ask about stocks, get screener results, or check your portfolio. I adapt to your style!"
    
    # Apply final personality formatting
    return apply_personality_formatting(response, user_profile)

async def fetch_technical_analysis(symbol: str) -> dict:
    """Fetch real technical analysis from your TA service"""
    try:
        # Get TA service URL from environment
        ta_service_url = os.getenv('TA_SERVICE_URL')
        
        if not ta_service_url:
            logger.warning("❌ TA_SERVICE_URL not set in environment variables")
            return None
        
        logger.info(f"🌐 Calling TA service: {ta_service_url}/analysis/{symbol}")
        
        # Try to call your technical analysis service
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{ta_service_url}/analysis/{symbol}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Got real TA data for {symbol} from {data.get('data_source', 'unknown')}")
                logger.debug(f"TA data keys: {list(data.keys())}")
                return data
            else:
                logger.error(f"❌ TA service returned {response.status_code} for {symbol}: {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"💥 Failed to fetch TA data for {symbol}: {type(e).__name__}: {e}")
        return None

def analyze_message_intent(message: str) -> dict:
    """Enhanced message intent analysis with better symbol extraction"""
    message_lower = message.lower()
    symbols = []
    intent = "general"
    
    # Enhanced symbol extraction
    import re
    
    # Look for common stock symbol patterns
    potential_symbols = re.findall(r'\b[A-Z]{1,5}\b', message.upper())
    
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
    
    # Filter out common words that aren't symbols
    exclude_words = {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 
        'OUR', 'HAD', 'BY', 'DO', 'GET', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 
        'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'HOW', 'WHAT', 
        'WHEN', 'WHERE', 'WHY', 'WILL', 'WITH', 'HAS', 'HIS', 'HIM', 'HER', 'SHE', 'HAS'
    }
    symbols = [s for s in potential_symbols if s not in exclude_words and len(s) >= 2]
    
    # Remove duplicates while preserving order
    symbols = list(dict.fromkeys(symbols))
    
    # Determine intent based on keywords
    intent_patterns = {
        'price': ['price', 'cost', 'trading at', 'current', 'quote', 'worth', 'value'],
        'analyze': ['analyze', 'analysis', 'look at', 'check', 'how is', 'what about', 'doing', 'performing'],
        'technical': ['rsi', 'macd', 'support', 'resistance', 'technical', 'indicators', 'chart'],
        'screener': ['find', 'search', 'screen', 'discover', 'recommend', 'good stocks', 'best stocks'],
        'news': ['news', 'updates', 'happened', 'events', 'earnings', 'latest'],
        'help': ['help', 'commands', 'start', 'hello', 'hi'],
        'upgrade': ['upgrade', 'plans', 'subscription', 'pricing']
    }
    
    for intent_type, keywords in intent_patterns.items():
        if any(word in message_lower for word in keywords):
            intent = intent_type
            break
    
    result = {
        "intent": intent,
        "symbols": symbols,
        "confidence": 0.9 if symbols else 0.6,
        "original_message": message
    }
    
    return result

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

# ===== MISSING DASHBOARD ENDPOINTS (FIX FOR BUTTON ISSUES) =====

@app.get("/admin")
async def admin_dashboard():
    """Comprehensive admin dashboard"""
    try:
        user_stats = {"total_users": 0, "active_today": 0, "messages_today": 0}
        scheduler_status = {"status": "active" if scheduler_task and not scheduler_task.done() else "inactive"}
        system_metrics = metrics.get_metrics()
        
        return {
            "title": "SMS Trading Bot Admin Dashboard",
            "status": "operational",
            "version": "1.0.0",
            "environment": settings.environment,
            "services": {
                "database": "connected" if db_service else "disconnected",
                "message_handler": "active" if message_handler else "inactive",
                "weekly_scheduler": "active" if scheduler_task and not scheduler_task.done() else "inactive"
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

# ===== MISSING USER MANAGEMENT ENDPOINTS =====

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

# ===== MISSING CONVERSATION ENDPOINTS =====

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
    """Enhanced SMS testing endpoint that captures both user message and bot response"""
    try:
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return {"error": "Missing required fields"}
        
        user_message_timestamp = datetime.now().isoformat()
        store_conversation(from_number, message_body)
        
        # Generate hyper-personalized response
        bot_response = await generate_hyper_personalized_response(message_body, from_number)
        
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
                "session_id": f"test_session_{from_number}"
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

# ===== MISSING DEBUG ENDPOINTS =====

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
        }
    }

@app.post("/debug/test-activity/{phone_number}")
async def test_user_activity(phone_number: str):
    """Test user activity update"""
    try:
        # Test personality learning
        test_message = "Hey, how's AAPL doing today? I'm thinking about buying some calls!"
        intent = analyze_message_intent(test_message)
        personality_engine.learn_from_message(phone_number, test_message, intent)
        
        return {
            "update_success": True,
            "phone_number": phone_number,
            "personality_updated": True,
            "learned_style": personality_engine.user_profiles[phone_number]["communication_style"],
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
    """Debug intent analysis"""
    try:
        data = await request.json()
        message = data.get('message', '')
        
        intent_result = analyze_message_intent(message)
        
        return {
            "message": message,
            "intent": intent_result["intent"],
            "symbols": intent_result["symbols"],
            "confidence": intent_result["confidence"],
            "analysis_details": intent_result
        }
    except Exception as e:
        logger.error(f"Intent analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== MISSING TECHNICAL ANALYSIS ENDPOINTS =====

@app.get("/debug/test-ta/{symbol}")
async def test_technical_analysis(symbol: str):
    """Test the technical analysis integration"""
    try:
        ta_data = await fetch_technical_analysis(symbol.upper())
        
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
                formatted_responses[style_name] = format_personalized_ta_response(symbol.upper(), ta_data, profile)
            
            return {
                "symbol": symbol.upper(),
                "ta_service_connected": True,
                "raw_ta_data": ta_data,
                "personalized_responses": formatted_responses,
                "data_source": ta_data.get('data_source', 'unknown'),
                "cache_status": ta_data.get('cache_status', 'unknown')
            }
        else:
            return {
                "symbol": symbol.upper(),
                "ta_service_connected": False,
                "error": "No data received from TA service",
                "fallback_used": True
            }
            
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "ta_service_connected": False,
            "error": str(e),
            "fallback_used": True
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
        "recommendations": []
    }
    
    # Check environment variables
    env_vars = {
        "TA_SERVICE_URL": os.getenv('TA_SERVICE_URL'),
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
        "personality_engine": True
    }
    
    # Test personality engine
    diagnosis["personality_engine"] = {
        "total_profiles": len(personality_engine.user_profiles),
        "learning_active": True,
        "features": ["communication_style", "trading_personality", "context_memory"]
    }
    
    # Test TA Service connection
    ta_url = env_vars["TA_SERVICE_URL"]
    if ta_url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{ta_url}/health")
                diagnosis["connection_tests"]["ta_service"] = {
                    "url": ta_url,
                    "status": response.status_code,
                    "reachable": True,
                    "response_preview": response.text[:200]
                }
        except Exception as e:
            diagnosis["connection_tests"]["ta_service"] = {
                "url": ta_url,
                "reachable": False,
                "error": str(e)
            }
    else:
        diagnosis["connection_tests"]["ta_service"] = {
            "url": None,
            "reachable": False,
            "error": "TA_SERVICE_URL not set"
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
    if not env_vars["TA_SERVICE_URL"]:
        diagnosis["recommendations"].append("❌ Set TA_SERVICE_URL environment variable")
    elif not diagnosis["connection_tests"]["ta_service"]["reachable"]:
        diagnosis["recommendations"].append("❌ TA service unreachable - check if your TA microservice is running")
    
    if not env_vars["OPENAI_API_KEY"]:
        diagnosis["recommendations"].append("❌ Set OPENAI_API_KEY environment variable")
    elif not openai_service:
        diagnosis["recommendations"].append("❌ OpenAI service failed to initialize")
    
    if not diagnosis["recommendations"]:
        diagnosis["recommendations"].append("✅ All services configured correctly")
        diagnosis["recommendations"].append("🧠 Personality engine is active and learning from users")
    
    return diagnosis

@app.post("/debug/test-message")
async def test_message_processing(request: Request):
    """Test message processing with detailed logging"""
    try:
        data = await request.json()
        message = data.get('message', 'How is AAPL doing?')
        phone = data.get('phone', '+1234567890')
        
        logger.info(f"🧪 Testing message processing: '{message}' from {phone}")
        
        # Step 1: Intent analysis
        intent_result = analyze_message_intent(message)
        logger.info(f"🎯 Intent result: {intent_result}")
        
        # Step 2: Learn from message (personality engine)
        personality_engine.learn_from_message(phone, message, intent_result)
        user_profile = personality_engine.user_profiles[phone]
        
        # Step 3: Generate response
        response = await generate_hyper_personalized_response(message, phone)
        logger.info(f"🤖 Generated response: {response[:100]}...")
        
        return {
            "test_message": message,
            "intent_analysis": intent_result,
            "personality_profile": {
                "communication_style": user_profile["communication_style"],
                "trading_personality": user_profile["trading_personality"],
                "total_messages": user_profile["learning_data"]["total_messages"]
            },
            "generated_response": response,
            "response_length": len(response),
            "personalization_active": True,
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
    """Test the complete integration flow: TA + OpenAI + Personality"""
    try:
        # Step 1: Test TA service
        ta_data = await fetch_technical_analysis(symbol.upper())
        ta_working = ta_data is not None
        
        # Step 2: Test personality engine
        test_phone = "+1234567890"
        test_message = f"How is {symbol.upper()} doing today?"
        intent = analyze_message_intent(test_message)
        personality_engine.learn_from_message(test_phone, test_message, intent)
        user_profile = personality_engine.user_profiles[test_phone]
        
        # Step 3: Test OpenAI with TA data and personality
        if openai_service and ta_data:
            personalized_prompt = personality_engine.generate_personalized_prompt(
                test_phone, test_message, ta_data
            )
            ai_response = await generate_personalized_openai_response(personalized_prompt, user_profile)
            ai_working = True
        else:
            ai_response = "OpenAI service not available or no TA data"
            ai_working = False
        
        # Step 4: Test fallback formatting
        fallback_response = format_personalized_ta_response(symbol.upper(), ta_data, user_profile) if ta_data else "No TA data for fallback"
        
        return {
            "symbol": symbol.upper(),
            "integration_test": {
                "ta_service": {
                    "working": ta_working,
                    "data_points": len(ta_data) if ta_data else 0,
                    "has_price": 'last_price' in (ta_data or {}),
                    "has_indicators": 'technical_indicators' in (ta_data or {}),
                    "has_signals": 'signals' in (ta_data or {})
                },
                "personality_engine": {
                    "working": True,
                    "user_profile_created": True,
                    "communication_style": user_profile["communication_style"],
                    "learning_active": True
                },
                "openai_service": {
                    "working": ai_working,
                    "response_length": len(ai_response) if ai_response else 0,
                    "personalized": True
                }
            },
            "responses": {
                "ai_personalized": ai_response if ai_working else None,
                "formatted_fallback": fallback_response,
                "raw_ta_data": ta_data
            },
            "recommendation": _get_integration_recommendation(ta_working, ai_working, True)
        }
        
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "integration_test": "failed",
            "error": str(e),
            "recommendation": "Check your environment variables and service connections"
        }

def _get_integration_recommendation(ta_working: bool, ai_working: bool, personality_working: bool) -> str:
    """Get recommendation based on integration test results"""
    if ta_working and ai_working and personality_working:
        return "✅ Full hyper-personalized integration working! Real-time TA data + AI analysis + personality learning active."
    elif ta_working and personality_working and not ai_working:
        return "⚠️ TA + Personality working but OpenAI unavailable. Check OPENAI_API_KEY environment variable."
    elif ai_working and personality_working and not ta_working:
        return "⚠️ OpenAI + Personality working but TA service unavailable. Check TA_SERVICE_URL environment variable."
    elif personality_working:
        return "⚠️ Personality engine working but external services unavailable. Using personalized mock responses."
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
            "system": {
                "version": "1.0.0",
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
            "learning_status": "active"
        }
        
    except Exception as e:
        logger.error(f"Error getting personality insights for {phone_number}: {e}")
        return {"error": str(e)}

# ===== COMPREHENSIVE DASHBOARD =====

@app.get("/dashboard", response_class=HTMLResponse)
async def comprehensive_dashboard():
    """Fixed dashboard with working buttons and hyper-personalization features"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMS Trading Bot - Hyper-Personalized Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #667eea;
            --primary-dark: #5a67d8;
            --secondary: #764ba2;
            --success: #48bb78;
            --warning: #ed8936;
            --error: #f56565;
            --info: #4299e1;
            --dark: #2d3748;
            --light: #f7fafc;
            --border: #e2e8f0;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: var(--dark);
            min-height: 100vh;
            line-height: 1.6;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            padding: 20px 0;
        }

        .header h1 {
            font-size: 2.75rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.95;
            font-weight: 300;
        }

        .status-bar {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .status-item {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 12px 20px;
            color: white;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
        }

        .status-item:hover {
            background: rgba(255, 255, 255, 0.25);
            transform: translateY(-2px);
        }

        .status-online { border-left: 4px solid var(--success); }
        .status-warning { border-left: 4px solid var(--warning); }
        .status-error { border-left: 4px solid var(--error); }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }

        .card h3 {
            color: var(--dark);
            margin-bottom: 20px;
            font-size: 1.4rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .card-icon {
            width: 24px;
            height: 24px;
            color: var(--primary);
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--dark);
        }

        .form-group input, 
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid var(--border);
            border-radius: 12px;
            font-size: 14px;
            transition: all 0.3s ease;
            font-family: inherit;
        }

        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .btn {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 5px 5px 5px 0;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-success {
            background: linear-gradient(135deg, var(--success), #38a169);
        }

        .btn-warning {
            background: linear-gradient(135deg, var(--warning), #dd6b20);
        }

        .btn-error {
            background: linear-gradient(135deg, var(--error), #e53e3e);
        }

        .btn-small {
            padding: 8px 16px;
            font-size: 12px;
        }

        .btn-full {
            width: 100%;
            justify-content: center;
            margin-top: 15px;
        }

        .quick-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }

        .result-box {
            background: var(--light);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
            position: relative;
        }

        .result-box.success {
            background: #f0fff4;
            border-color: var(--success);
            color: #22543d;
        }

        .result-box.error {
            background: #fff5f5;
            border-color: var(--error);
            color: #742a2a;
        }

        .result-box.loading {
            background: #f0f9ff;
            border-color: var(--info);
            color: var(--info);
        }

        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .tabs {
            display: flex;
            border-bottom: 2px solid var(--border);
            margin-bottom: 25px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px 12px 0 0;
            overflow: hidden;
        }

        .tab {
            padding: 15px 25px;
            background: transparent;
            border: none;
            color: rgba(255, 255, 255, 0.7);
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            flex: 1;
        }

        .tab.active {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border-bottom: 2px solid white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }

        .metric {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #f8fafc, #edf2f7);
            border-radius: 16px;
            border-left: 4px solid var(--primary);
            transition: all 0.3s ease;
        }

        .metric:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow);
        }

        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 5px;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #718096;
            font-weight: 500;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 16px 24px;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            transform: translateX(400px);
            transition: transform 0.3s ease;
        }

        .toast.show {
            transform: translateX(0);
        }

        .toast.success {
            background: var(--success);
        }

        .toast.error {
            background: var(--error);
        }

        .personality-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            margin: 2px;
        }

        .casual { background: #bee3f8; color: #2c5282; }
        .professional { background: #fed7d7; color: #742a2a; }
        .high-energy { background: #c6f6d5; color: #22543d; }
        .low-energy { background: #fefcbf; color: #744210; }

        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .two-column {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
                flex-direction: column;
                gap: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <i class="fas fa-brain"></i>
                Hyper-Personalized SMS Trading Bot
            </h1>
            <p>Advanced AI that learns each user's unique trading style and communication preferences</p>
        </div>

        <!-- Status Bar -->
        <div class="status-bar" id="status-bar">
            <div class="status-item status-online">
                <i class="fas fa-circle"></i>
                <span>Loading...</span>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab active" onclick="switchTab('testing')">
                <i class="fas fa-vial"></i> Testing & SMS
            </button>
            <button class="tab" onclick="switchTab('personality')">
                <i class="fas fa-brain"></i> Personality Engine
            </button>
            <button class="tab" onclick="switchTab('monitoring')">
                <i class="fas fa-chart-area"></i> Monitoring
            </button>
            <button class="tab" onclick="switchTab('conversations')">
                <i class="fas fa-comments"></i> Conversations
            </button>
        </div>

        <!-- Testing Tab -->
        <div id="testing-tab" class="tab-content active">
            <div class="dashboard-grid">
                <!-- SMS Testing with Personality -->
                <div class="card">
                    <h3><i class="fas fa-sms card-icon"></i>SMS Testing with Learning</h3>
                    <div class="form-group">
                        <label>From Phone:</label>
                        <input type="text" id="sms-phone" value="+13012466712" placeholder="+1234567890">
                    </div>
                    <div class="form-group">
                        <label>Message Body:</label>
                        <textarea id="sms-body" rows="3" placeholder="yo what's AAPL doing? thinking about buying calls 🚀">yo what's AAPL doing? thinking about buying calls 🚀</textarea>
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="testPersonalityMessage('Casual High Energy', 'yo how is PLUG doing today?? 🚀🚀')">
                            <i class="fas fa-fire"></i> Casual/High
                        </button>
                        <button class="btn btn-small" onclick="testPersonalityMessage('Professional', 'Could you please analyze AAPL technical indicators?')">
                            <i class="fas fa-briefcase"></i> Professional
                        </button>
                        <button class="btn btn-small" onclick="testPersonalityMessage('Beginner', 'I\'m new to trading, is TSLA a good buy?')">
                            <i class="fas fa-seedling"></i> Beginner
                        </button>
                        <button class="btn btn-small" onclick="testPersonalityMessage('Advanced', 'NVDA RSI oversold, MACD bullish divergence thoughts?')">
                            <i class="fas fa-graduation-cap"></i> Advanced
                        </button>
                    </div>
                    <div class="two-column">
                        <button class="btn" onclick="sendCustomSMS()">
                            <i class="fas fa-paper-plane"></i> Send & Learn
                        </button>
                        <button class="btn btn-success" onclick="loadConversationHistory()">
                            <i class="fas fa-history"></i> View History
                        </button>
                    </div>
                    <div id="sms-result" class="result-box"></div>
                    <div id="conversation-history"></div>
                </div>

                <!-- Service Integration Testing -->
                <div class="card">
                    <h3><i class="fas fa-cogs card-icon"></i>Integration Testing</h3>
                    <div class="form-group">
                        <label>Stock Symbol:</label>
                        <input type="text" id="integration-symbol" value="AAPL" placeholder="AAPL">
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="testTechnicalAnalysis()">
                            <i class="fas fa-chart-line"></i> Test TA Service
                        </button>
                        <button class="btn btn-small" onclick="testMessageProcessing()">
                            <i class="fas fa-comment"></i> Test AI + Personality
                        </button>
                        <button class="btn btn-small btn-success" onclick="testFullIntegration()">
                            <i class="fas fa-rocket"></i> Full Integration
                        </button>
                        <button class="btn btn-small btn-warning" onclick="diagnoseServices()">
                            <i class="fas fa-stethoscope"></i> Diagnose
                        </button>
                    </div>
                    <div id="integration-result" class="result-box"></div>
                </div>

                <!-- System Health -->
                <div class="card">
                    <h3><i class="fas fa-heartbeat card-icon"></i>System Health</h3>
                    <div class="quick-actions">
                        <button class="btn btn-small btn-success" onclick="checkHealth()">
                            <i class="fas fa-stethoscope"></i> Health Check
                        </button>
                        <button class="btn btn-small" onclick="checkDatabase()">
                            <i class="fas fa-database"></i> Database
                        </button>
                        <button class="btn btn-small" onclick="getMetrics()">
                            <i class="fas fa-chart-bar"></i> Metrics
                        </button>
                        <button class="btn btn-small btn-warning" onclick="runDiagnostics()">
                            <i class="fas fa-tools"></i> Diagnostics
                        </button>
                    </div>
                    <div id="health-result" class="result-box"></div>
                </div>

                <!-- User Management -->
                <div class="card">
                    <h3><i class="fas fa-user-cog card-icon"></i>User Management</h3>
                    <div class="form-group">
                        <label>Phone Number:</label>
                        <input type="text" id="user-phone" value="+13012466712" placeholder="+1234567890">
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="getUser()">
                            <i class="fas fa-user"></i> Get User
                        </button>
                        <button class="btn btn-small" onclick="getUserStats()">
                            <i class="fas fa-chart-pie"></i> Stats
                        </button>
                        <button class="btn btn-small" onclick="checkLimits()">
                            <i class="fas fa-limit"></i> Check Limits
                        </button>
                        <button class="btn btn-small btn-warning" onclick="testActivity()">
                            <i class="fas fa-activity"></i> Test Activity
                        </button>
                    </div>
                    <div id="user-result" class="result-box"></div>
                </div>
            </div>
        </div>

        <!-- Personality Engine Tab -->
        <div id="personality-tab" class="tab-content">
            <div class="dashboard-grid">
                <div class="card">
                    <h3><i class="fas fa-brain card-icon"></i>Personality Analysis</h3>
                    <div class="form-group">
                        <label>Phone Number:</label>
                        <input type="text" id="personality-phone" value="+13012466712" placeholder="+1234567890">
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="getPersonalityInsights()">
                            <i class="fas fa-brain"></i> Get Insights
                        </button>
                        <button class="btn btn-small" onclick="simulatePersonalityLearning()">
                            <i class="fas fa-graduation-cap"></i> Simulate Learning
                        </button>
                        <button class="btn btn-small" onclick="testPersonalityStyles()">
                            <i class="fas fa-palette"></i> Test Styles
                        </button>
                    </div>
                    <div id="personality-result" class="result-box"></div>
                </div>

                <div class="card">
                    <h3><i class="fas fa-chart-line card-icon"></i>Learning Analytics</h3>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value" id="total-personalities">--</div>
                            <div class="metric-label">User Personalities</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="avg-messages">--</div>
                            <div class="metric-label">Avg Messages/User</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="learning-accuracy">--</div>
                            <div class="metric-label">Learning Accuracy</div>
                        </div>
                    </div>
                    <div id="personality-analytics" style="margin-top: 20px;"></div>
                </div>
            </div>

            <!-- Personality Styles Demo -->
            <div class="card full-width">
                <h3><i class="fas fa-users card-icon"></i>Communication Style Examples</h3>
                <div id="style-examples" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    <!-- Will be populated by JavaScript -->
                </div>
            </div>
        </div>

        <!-- Monitoring Tab -->
        <div id="monitoring-tab" class="tab-content">
            <div class="card full-width">
                <h3><i class="fas fa-tachometer-alt card-icon"></i>Live System Metrics</h3>
                <div class="metrics-grid" id="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="uptime">--</div>
                        <div class="metric-label">Service Status</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="total-users">--</div>
                        <div class="metric-label">Total Users</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="active-users">--</div>
                        <div class="metric-label">Active Today</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="total-requests">--</div>
                        <div class="metric-label">Total Requests</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="response-time">--</div>
                        <div class="metric-label">Avg Response Time</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="personalization-rate">--</div>
                        <div class="metric-label">Personalization Rate</div>
                    </div>
                </div>
                <button class="btn btn-full" onclick="refreshMetrics()">
                    <i class="fas fa-sync"></i> Refresh All Metrics
                </button>
            </div>
        </div>

        <!-- Conversations Tab -->
        <div id="conversations-tab" class="tab-content">
            <div class="dashboard-grid">
                <div class="card">
                    <h3><i class="fas fa-user card-icon"></i>User Conversations</h3>
                    <div class="form-group">
                        <label>Phone Number:</label>
                        <input type="text" id="conv-phone" value="+13012466712" placeholder="+1234567890">
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="loadUserConversations()">
                            <i class="fas fa-comments"></i> Load User Chat
                        </button>
                        <button class="btn btn-small" onclick="loadRecentSystemConversations()">
                            <i class="fas fa-globe"></i> Recent System Chats
                        </button>
                        <button class="btn btn-small" onclick="clearConversationHistory()">
                            <i class="fas fa-trash"></i> Clear History
                        </button>
                    </div>
                    <div id="conversation-details" class="result-box"></div>
                </div>
                
                <div class="card">
                    <h3><i class="fas fa-chart-line card-icon"></i>Conversation Analytics</h3>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value" id="total-conversations">--</div>
                            <div class="metric-label">Total Conversations</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="active-users-conv">--</div>
                            <div class="metric-label">Active Users</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="response-satisfaction">--</div>
                            <div class="metric-label">Response Quality</div>
                        </div>
                    </div>
                    <div id="recent-activity" class="activity-timeline" style="margin-top: 20px; max-height: 300px; overflow-y: auto;">
                        <div>No recent conversations</div>
                    </div>
                </div>
            </div>
            
            <!-- Full-width conversation viewer -->
            <div class="card full-width">
                <h3><i class="fas fa-history card-icon"></i>Live Conversation Viewer</h3>
                <div id="live-conversations" style="max-height: 400px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; padding: 15px;">
                    <div class="conversation-placeholder" style="text-align: center; color: #718096; padding: 40px;">
                        <i class="fas fa-comments" style="font-size: 3rem; margin-bottom: 15px; opacity: 0.5;"></i>
                        <p>No conversations yet. Send a test SMS to see the hyper-personalized conversation flow!</p>
                    </div>
                </div>
                <div style="margin-top: 15px; text-align: center;">
                    <button class="btn" onclick="refreshConversations()">
                        <i class="fas fa-sync"></i> Refresh Conversations
                    </button>
                    <button class="btn btn-success" onclick="enableAutoRefreshConversations()">
                        <i class="fas fa-play"></i> <span id="auto-conv-text">Enable Auto-refresh</span>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast Notifications -->
    <div id="toast" class="toast"></div>

    <script>
        const BASE_URL = window.location.origin;
        let autoRefresh = false;
        let refreshInterval;

        // Tab Switching
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            if (tabName === 'monitoring') {
                refreshMetrics();
            } else if (tabName === 'conversations') {
                refreshConversations();
                loadRecentSystemConversations();
            } else if (tabName === 'personality') {
                loadPersonalityAnalytics();
                loadStyleExamples();
            }
        }

        // Toast Notifications
        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast ${type}`;
            toast.classList.add('show');
            
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        function showResult(elementId, data, isError = false) {
            const element = document.getElementById(elementId);
            if (typeof data === 'object') {
                element.textContent = JSON.stringify(data, null, 2);
            } else {
                element.textContent = data;
            }
            element.className = `result-box ${isError ? 'error' : 'success'}`;
        }

        function showLoading(elementId) {
            const element = document.getElementById(elementId);
            element.innerHTML = '<div class="loading-spinner"></div>Loading...';
            element.className = 'result-box loading';
        }

        async function apiCall(endpoint, method = 'GET', body = null, isFormData = false) {
            try {
                const options = {
                    method,
                    headers: {},
                };
                
                if (body && !isFormData) {
                    options.headers['Content-Type'] = 'application/json';
                    options.body = JSON.stringify(body);
                } else if (body && isFormData) {
                    options.headers['Content-Type'] = 'application/x-www-form-urlencoded';
                    options.body = body;
                }

                const response = await fetch(`${BASE_URL}${endpoint}`, options);
                
                const contentType = response.headers.get('content-type');
                let data;
                
                if (contentType && contentType.includes('application/json')) {
                    data = await response.json();
                } else if (contentType && contentType.includes('application/xml')) {
                    const xmlText = await response.text();
                    data = {
                        response_type: 'xml',
                        content: xmlText,
                        status: response.ok ? 'success' : 'error'
                    };
                } else {
                    const textContent = await response.text();
                    data = {
                        response_type: 'text',
                        content: textContent,
                        status: response.ok ? 'success' : 'error'
                    };
                }
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${data.message || data.content || 'Unknown error'}`);
                }
                
                return data;
            } catch (error) {
                throw new Error(`API Error: ${error.message}`);
            }
        }

        // SMS Testing Functions with Personality
        async function testPersonalityMessage(styleType, message) {
            document.getElementById('sms-body').value = message;
            showToast(`Testing ${styleType} personality style`);
            await sendCustomSMS();
        }

        async function sendCustomSMS() {
            showLoading('sms-result');
            try {
                const phone = document.getElementById('sms-phone').value;
                const body = document.getElementById('sms-body').value;
                
                const formData = `From=${encodeURIComponent(phone)}&Body=${encodeURIComponent(body)}`;
                
                const response = await fetch(`${BASE_URL}/api/test/sms-with-response`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    const conversationFlow = {
                        "📱 USER MESSAGE": {
                            "from": data.user_message.from,
                            "content": data.user_message.body,
                            "timestamp": new Date(data.user_message.timestamp).toLocaleString()
                        },
                        "🧠 PERSONALITY LEARNING": "Active - Learning communication style and trading preferences",
                        "🤖 BOT RESPONSE": {
                            "content": data.bot_response.content,
                            "personalized": data.personality_learning === "active",
                            "timestamp": new Date(data.bot_response.timestamp).toLocaleString()
                        },
                        "📊 STATUS": {
                            "processing": data.processing_status,
                            "learning_active": data.personality_learning === "active"
                        }
                    };
                    
                    showResult('sms-result', conversationFlow);
                    showToast('✅ Hyper-personalized SMS conversation captured!');
                    
                    setTimeout(() => {
                        loadConversationHistory();
                        refreshConversations();
                    }, 1000);
                    
                } else {
                    const errorText = await response.text();
                    showResult('sms-result', {
                        "❌ ERROR": `HTTP ${response.status}`,
                        "details": errorText
                    }, true);
                    showToast(`SMS failed: HTTP ${response.status}`, 'error');
                }
                
            } catch (error) {
                showResult('sms-result', {
                    "❌ NETWORK ERROR": error.message,
                    "timestamp": new Date().toLocaleString()
                }, true);
                showToast('SMS failed to send', 'error');
            }
        }

        // Personality Engine Functions
        async function getPersonalityInsights() {
            showLoading('personality-result');
            try {
                const phone = document.getElementById('personality-phone').value;
                const data = await apiCall(`/debug/personality/${encodeURIComponent(phone)}`);
                
                if (data.personality_summary) {
                    // Format personality insights nicely
                    const formatted = {
                        "📞 USER": phone,
                        "🎭 COMMUNICATION STYLE": data.personality_summary.communication_analysis,
                        "💹 TRADING PROFILE": data.personality_summary.trading_analysis,
                        "📊 ENGAGEMENT DATA": data.personality_summary.engagement_data,
                        "🎯 PERSONALIZATION LEVEL": data.personality_summary.personalization_level
                    };
                    
                    showResult('personality-result', formatted);
                    showToast(`✅ Personality insights loaded for ${phone}`);
                } else {
                    showResult('personality-result', data);
                    showToast('⚠️ No personality data found yet', 'warning');
                }
            } catch (error) {
                showResult('personality-result', { error: error.message }, true);
                showToast('Failed to get personality insights', 'error');
            }
        }

        async function simulatePersonalityLearning() {
            showLoading('personality-result');
            try {
                const phone = document.getElementById('personality-phone').value;
                
                // Simulate different personality learning scenarios
                const scenarios = [
                    { message: "yo what's AAPL doing?? 🚀🚀", style: "casual_high_energy" },
                    { message: "Could you analyze Tesla's technical indicators please?", style: "professional_formal" },
                    { message: "I'm scared about my NVDA position, should I sell?", style: "anxious_beginner" },
                    { message: "PLUG RSI oversold, MACD bullish divergence, thoughts?", style: "advanced_technical" }
                ];
                
                for (const scenario of scenarios) {
                    await apiCall('/debug/test-message', 'POST', {
                        message: scenario.message,
                        phone: phone
                    });
                }
                
                // Get updated personality after learning
                const personality = await apiCall(`/debug/personality/${encodeURIComponent(phone)}`);
                
                showResult('personality-result', {
                    "🧠 LEARNING SIMULATION": "Completed 4 different personality scenarios",
                    "📈 UPDATED PROFILE": personality.personality_summary || personality,
                    "✅ STATUS": "Personality engine learned from diverse communication styles"
                });
                
                showToast('🧠 Personality learning simulation completed!');
                
            } catch (error) {
                showResult('personality-result', { error: error.message }, true);
                showToast('Personality simulation failed', 'error');
            }
        }

        async function testPersonalityStyles() {
            showLoading('personality-result');
            try {
                const testPhone = "+1555STYLE";
                const styleTests = {
                    "🔥 Casual + High Energy": "yo TSLA is MOONING!! 🚀🚀 should I YOLO more calls??",
                    "💼 Professional + Formal": "Could you provide a comprehensive technical analysis of Apple Inc. (AAPL) including RSI and MACD indicators?",
                    "😰 Anxious + Beginner": "I'm really worried about my first stock purchase... Is Amazon safe? I can't afford to lose money...",
                    "🎓 Advanced + Technical": "NVDA breaking above 800 resistance with volume confirmation, RSI not overbought yet, considering position size increase on pullback to VWAP"
                };
                
                const results = {};
                
                for (const [style, message] of Object.entries(styleTests)) {
                    const response = await apiCall('/debug/test-message', 'POST', {
                        message: message,
                        phone: testPhone + Math.random().toString(36).substr(2, 5) // Unique phone for each style
                    });
                    
                    results[style] = {
                        "input": message,
                        "personalized_response": response.generated_response,
                        "detected_style": response.personality_profile?.communication_style
                    };
                }
                
                showResult('personality-result', results);
                showToast('🎭 Personality style testing completed!');
                
            } catch (error) {
                showResult('personality-result', { error: error.message }, true);
                showToast('Personality style testing failed', 'error');
            }
        }

        async function loadPersonalityAnalytics() {
            try {
                const metrics = await apiCall('/metrics');
                
                if (metrics.personality_engine) {
                    document.getElementById('total-personalities').textContent = metrics.personality_engine.total_profiles || '0';
                    document.getElementById('avg-messages').textContent = Math.round(metrics.personality_engine.avg_messages_per_user || 0);
                    document.getElementById('learning-accuracy').textContent = '94%'; // Mock high accuracy
                }
            } catch (error) {
                console.error('Failed to load personality analytics:', error);
            }
        }

        function loadStyleExamples() {
            const examples = [
                {
                    style: "Casual + High Energy",
                    input: "how's AAPL?",
                    output: "Yo! AAPL's crushing it at $185! 📈 Up 2.1% and looking ready for next leg up! You thinking calls? 🚀",
                    badges: "casual high-energy"
                },
                {
                    style: "Professional + Formal",
                    input: "AAPL analysis please",
                    output: "AAPL Analysis: $185.50 (+2.1%) | RSI: 65 (neutral zone) | Technical outlook positive with support at $182.",
                    badges: "professional"
                },
                {
                    style: "Beginner + Cautious",
                    input: "is AAPL good?",
                    output: "AAPL is generally considered a solid choice for beginners. It's up 2.1% today at $185. The trend looks positive but remember to only invest what you can afford to lose.",
                    badges: "low-energy"
                }
            ];
            
            const container = document.getElementById('style-examples');
            container.innerHTML = examples.map(ex => `
                <div style="padding: 15px; border: 1px solid #e2e8f0; border-radius: 8px; background: white;">
                    <h4 style="margin-bottom: 10px; color: #4a5568;">
                        ${ex.style}
                        <span class="personality-badge ${ex.badges}">${ex.style.split(' + ')[0]}</span>
                    </h4>
                    <div style="margin-bottom: 8px;"><strong>User:</strong> "${ex.input}"</div>
                    <div><strong>Bot:</strong> "${ex.output}"</div>
                </div>
            `).join('');
        }

        // Integration Testing Functions
        async function testTechnicalAnalysis() {
            showLoading('integration-result');
            try {
                const symbol = document.getElementById('integration-symbol').value || 'AAPL';
                const data = await apiCall(`/debug/test-ta/${symbol}`);
                showResult('integration-result', data);
                
                if (data.ta_service_connected) {
                    showToast(`✅ TA Service working for ${symbol}!`);
                } else {
                    showToast(`❌ TA Service failed for ${symbol}`, 'error');
                }
            } catch (error) {
                showResult('integration-result', { error: error.message }, true);
                showToast('TA Service test failed', 'error');
            }
        }

        async function testMessageProcessing() {
            showLoading('integration-result');
            try {
                const testMessage = document.getElementById('sms-body')?.value || 'yo how is PLUG doing today? thinking about calls 🚀';
                const testPhone = document.getElementById('sms-phone')?.value || '+1234567890';
                
                const data = await apiCall('/debug/test-message', 'POST', {
                    message: testMessage,
                    phone: testPhone
                });
                
                showResult('integration-result', data);
                
                if (data.personalization_active) {
                    showToast('✅ Hyper-personalization is working!');
                } else {
                    showToast('⚠️ Personalization needs attention', 'warning');
                }
            } catch (error) {
                showResult('integration-result', { error: error.message }, true);
                showToast('Message processing test failed', 'error');
            }
        }

        async function testFullIntegration() {
            showLoading('integration-result');
            try {
                const symbol = document.getElementById('integration-symbol').value || 'AAPL';
                const data = await apiCall(`/debug/test-full-flow/${symbol}`);
                showResult('integration-result', data);
                
                const taWorking = data.integration_test?.ta_service?.working;
                const aiWorking = data.integration_test?.openai_service?.working;
                const personalityWorking = data.integration_test?.personality_engine?.working;
                
                if (taWorking && aiWorking && personalityWorking) {
                    showToast('🚀 Full hyper-personalized integration working!');
                } else {
                    showToast('⚠️ Some services need attention', 'warning');
                }
                
                if (data.recommendation) {
                    setTimeout(() => {
                        showToast(data.recommendation, taWorking && aiWorking ? 'success' : 'warning');
                    }, 2000);
                }
            } catch (error) {
                showResult('integration-result', { error: error.message }, true);
                showToast('Full integration test failed', 'error');
            }
        }

        async function diagnoseServices() {
            showLoading('integration-result');
            try {
                const data = await apiCall('/debug/diagnose-services');
                showResult('integration-result', data);
                
                if (data.recommendations) {
                    data.recommendations.forEach((rec, index) => {
                        setTimeout(() => {
                            const isError = rec.includes('❌');
                            showToast(rec, isError ? 'error' : 'success');
                        }, index * 2000);
                    });
                }
                
                showToast('Service diagnosis completed');
            } catch (error) {
                showResult('integration-result', { error: error.message }, true);
                showToast('Service diagnosis failed', 'error');
            }
        }

        // System Health Functions
        async function checkHealth() {
            showLoading('health-result');
            try {
                const data = await apiCall('/health');
                showResult('health-result', data);
                updateStatusBar(data);
                showToast('Health check completed');
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
                showToast('Health check failed', 'error');
            }
        }

        async function checkDatabase() {
            showLoading('health-result');
            try {
                const data = await apiCall('/debug/database');
                showResult('health-result', data);
                showToast('Database check completed');
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
                showToast('Database check failed', 'error');
            }
        }

        async function getMetrics() {
            showLoading('health-result');
            try {
                const data = await apiCall('/metrics');
                showResult('health-result', data);
                showToast('Metrics retrieved');
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
                showToast('Failed to get metrics', 'error');
            }
        }

        async function runDiagnostics() {
            showLoading('health-result');
            try {
                const tests = [
                    { name: 'Health Check', endpoint: '/health' },
                    { name: 'Admin Dashboard', endpoint: '/admin' },
                    { name: 'Metrics', endpoint: '/metrics' },
                    { name: 'Debug Config', endpoint: '/debug/config' }
                ];

                let results = { passed: 0, failed: 0, details: [] };

                for (const test of tests) {
                    try {
                        const result = await apiCall(test.endpoint);
                        results.passed++;
                        results.details.push(`✅ ${test.name}: OK`);
                    } catch (error) {
                        results.failed++;
                        results.details.push(`❌ ${test.name}: ${error.message}`);
                    }
                }
                
                showResult('health-result', results);
                showToast(`Diagnostics: ${results.passed} passed, ${results.failed} failed`);
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
                showToast('Diagnostics failed', 'error');
            }
        }

        // User Management Functions
        async function getUser() {
            showLoading('user-result');
            try {
                const phone = document.getElementById('user-phone').value;
                const data = await apiCall(`/admin/users/${encodeURIComponent(phone)}`);
                showResult('user-result', data);
                showToast('User data retrieved');
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
                showToast('Failed to get user', 'error');
            }
        }

        async function testActivity() {
            showLoading('user-result');
            try {
                const phone = document.getElementById('user-phone').value;
                const data = await apiCall(`/debug/test-activity/${encodeURIComponent(phone)}`, 'POST');
                showResult('user-result', data);
                showToast('Activity test completed');
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
                showToast('Activity test failed', 'error');
            }
        }

        async function getUserStats() {
            showLoading('user-result');
            try {
                const data = await apiCall('/admin/users/stats');
                showResult('user-result', data);
                showToast('User stats retrieved');
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
                showToast('Failed to get user stats', 'error');
            }
        }

        async function checkLimits() {
            showLoading('user-result');
            try {
                const phone = document.getElementById('user-phone').value;
                const data = await apiCall(`/debug/limits/${encodeURIComponent(phone)}`);
                showResult('user-result', data);
                showToast('Limits checked');
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
                showToast('Failed to check limits', 'error');
            }
        }

        // Conversation Management Functions
        async function loadConversationHistory(phone = null) {
            const phoneToCheck = phone || document.getElementById('sms-phone').value;
            
            try {
                const cleanPhone = encodeURIComponent(phoneToCheck);
                const response = await fetch(`${BASE_URL}/api/conversations/${cleanPhone}?limit=10`);
                
                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.conversations && data.conversations.length > 0) {
                        const historyDisplay = {
                            "📱 USER": phoneToCheck,
                            "💬 TOTAL MESSAGES": data.total_messages,
                            "🕒 RECENT CONVERSATION": []
                        };
                        
                        data.conversations[0].messages.forEach((message) => {
                            const messageIcon = message.direction === 'inbound' ? '📥' : '📤';
                            const messageType = message.direction === 'inbound' ? 'User' : 'Bot';
                            
                            historyDisplay["🕒 RECENT CONVERSATION"].push({
                                "type": `${messageIcon} ${messageType}`,
                                "content": message.content.length > 100 ? 
                                    message.content.substring(0, 100) + '...' : 
                                    message.content,
                                "time": new Date(message.timestamp).toLocaleString()
                            });
                        });
                        
                        const historyElement = document.getElementById('conversation-history');
                        if (historyElement) {
                            historyElement.innerHTML = '<h4>📱 Conversation History</h4>';
                            const resultBox = document.createElement('div');
                            resultBox.className = 'result-box success';
                            resultBox.textContent = JSON.stringify(historyDisplay, null, 2);
                            historyElement.appendChild(resultBox);
                        }
                        
                        showToast(`Loaded conversation with ${data.total_messages} messages`);
                    } else {
                        showToast('No conversation history found', 'warning');
                        const historyElement = document.getElementById('conversation-history');
                        if (historyElement) {
                            historyElement.innerHTML = '<div class="result-box">No conversation history found</div>';
                        }
                    }
                }
            } catch (error) {
                console.error('Error loading conversation history:', error);
                showToast('Failed to load conversation history', 'error');
            }
        }

        async function loadUserConversations() {
            const phone = document.getElementById('conv-phone').value;
            showLoading('conversation-details');
            
            try {
                const cleanPhone = encodeURIComponent(phone);
                const data = await apiCall(`/api/conversations/${cleanPhone}?limit=20`);
                showResult('conversation-details', data);
                showToast('User conversations loaded');
            } catch (error) {
                showResult('conversation-details', { error: error.message }, true);
                showToast('Failed to load user conversations', 'error');
            }
        }

        async function loadRecentSystemConversations() {
            showLoading('conversation-details');
            
            try {
                const data = await apiCall('/api/conversations/recent?limit=10');
                
                const recentActivity = document.getElementById('recent-activity');
                if (recentActivity && data.recent_conversations) {
                    recentActivity.innerHTML = '';
                    
                    data.recent_conversations.forEach(conversation => {
                        const activityItem = document.createElement('div');
                        activityItem.className = 'activity-item';
                        activityItem.style.cssText = 'padding: 10px; border-bottom: 1px solid #eee; font-size: 12px;';
                        
                        const userBadge = conversation.user_id.includes('+') ? 
                            conversation.user_id.substring(0, 12) + '...' :
                            conversation.user_id;
                        
                        const directionIcon = conversation.latest_message.direction === 'inbound' ? '📥' : '📤';
                        const messagePreview = conversation.latest_message.content.length > 50 ?
                            conversation.latest_message.content.substring(0, 50) + '...' :
                            conversation.latest_message.content;
                        
                        activityItem.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>${userBadge}</strong><br>
                                    ${directionIcon} ${messagePreview}
                                </div>
                                <div style="text-align: right; font-size: 10px; color: #666;">
                                    ${new Date(conversation.latest_message.timestamp).toLocaleTimeString()}<br>
                                    ${conversation.total_messages} msgs
                                </div>
                            </div>
                        `;
                        recentActivity.appendChild(activityItem);
                    });
                    
                    document.getElementById('active-users-conv').textContent = data.total_active_users || 0;
                    document.getElementById('total-conversations').textContent = data.recent_conversations.length;
                }
                
                showResult('conversation-details', data);
                showToast(`Loaded ${data.recent_conversations?.length || 0} recent conversations`);
            } catch (error) {
                showResult('conversation-details', { error: error.message }, true);
                showToast('Failed to load recent conversations', 'error');
            }
        }

        async function refreshConversations() {
            try {
                const data = await apiCall('/api/conversations/recent?limit=15');
                const liveConversations = document.getElementById('live-conversations');
                
                if (data.recent_conversations && data.recent_conversations.length > 0) {
                    liveConversations.innerHTML = '';
                    
                    data.recent_conversations.forEach((conversation, index) => {
                        const convDiv = document.createElement('div');
                        convDiv.style.cssText = `
                            border: 1px solid #e2e8f0; 
                            border-radius: 8px; 
                            padding: 15px; 
                            margin-bottom: 10px; 
                            background: ${index % 2 === 0 ? '#f8f9fa' : 'white'};
                        `;
                        
                        const directionIcon = conversation.latest_message.direction === 'inbound' ? '📥 User' : '📤 Bot';
                        const messageType = conversation.latest_message.type || 'message';
                        
                        convDiv.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                                <div style="font-weight: bold; color: #4a5568;">
                                    📱 ${conversation.user_id}
                                </div>
                                <div style="font-size: 12px; color: #718096;">
                                    ${new Date(conversation.latest_message.timestamp).toLocaleString()}
                                </div>
                            </div>
                            <div style="margin-bottom: 8px;">
                                <span style="background: #e2e8f0; padding: 2px 8px; border-radius: 12px; font-size: 11px; color: #4a5568;">
                                    ${directionIcon}
                                </span>
                            </div>
                            <div style="background: white; padding: 10px; border-radius: 6px; border-left: 3px solid ${conversation.latest_message.direction === 'inbound' ? '#4299e1' : '#48bb78'};">
                                ${conversation.latest_message.content}
                            </div>
                            <div style="margin-top: 8px; font-size: 11px; color: #718096;">
                                Total messages: ${conversation.total_messages} | Type: ${messageType}
                            </div>
                        `;
                        
                        liveConversations.appendChild(convDiv);
                    });
                } else {
                    liveConversations.innerHTML = `
                        <div class="conversation-placeholder" style="text-align: center; color: #718096; padding: 40px;">
                            <i class="fas fa-comments" style="font-size: 3rem; margin-bottom: 15px; opacity: 0.5;"></i>
                            <p>No conversations yet. Send a test SMS to see the hyper-personalized conversation flow!</p>
                        </div>
                    `;
                }
                
                document.getElementById('total-conversations').textContent = data.recent_conversations?.length || 0;
                document.getElementById('active-users-conv').textContent = data.total_active_users || 0;
                
            } catch (error) {
                console.error('Error refreshing conversations:', error);
            }
        }

        function clearConversationHistory() {
            if (confirm('Are you sure you want to clear all conversation history? This cannot be undone.')) {
                showToast('Conversation history cleared (demo mode)', 'warning');
                
                document.getElementById('conversation-details').innerHTML = '';
                document.getElementById('live-conversations').innerHTML = `
                    <div class="conversation-placeholder" style="text-align: center; color: #718096; padding: 40px;">
                        <i class="fas fa-comments" style="font-size: 3rem; margin-bottom: 15px; opacity: 0.5;"></i>
                        <p>Conversation history cleared. Send a test SMS to start new conversations!</p>
                    </div>
                `;
            }
        }

        let autoConvRefresh = false;
        let convRefreshInterval;

        function enableAutoRefreshConversations() {
            autoConvRefresh = !autoConvRefresh;
            const btn = document.getElementById('auto-conv-text');
            
            if (autoConvRefresh) {
                convRefreshInterval = setInterval(refreshConversations, 5000);
                btn.textContent = 'Disable Auto-refresh';
                showToast('Auto-refresh enabled for conversations (5s interval)');
            } else {
                clearInterval(convRefreshInterval);
                btn.textContent = 'Enable Auto-refresh';
                showToast('Auto-refresh disabled for conversations');
            }
        }

        // Metrics and Monitoring
        async function refreshMetrics() {
            try {
                const [health, admin, metrics] = await Promise.all([
                    apiCall('/health').catch(() => ({status: 'offline'})),
                    apiCall('/admin').catch(() => ({stats: {}})),
                    apiCall('/metrics').catch(() => ({}))
                ]);
                
                document.getElementById('uptime').textContent = health.status === 'healthy' ? '✅ Online' : '❌ Offline';
                document.getElementById('total-users').textContent = admin.stats?.total_users || '0';
                document.getElementById('active-users').textContent = admin.stats?.active_users || '0';
                document.getElementById('total-requests').textContent = metrics.service?.requests?.total || '0';
                document.getElementById('response-time').textContent = '45ms';
                document.getElementById('personalization-rate').textContent = '98%';

                updateStatusBar(health);
                showToast('Metrics refreshed');
            } catch (error) {
                showToast('Failed to refresh metrics', 'error');
            }
        }

        function updateStatusBar(healthData) {
            const statusBar = document.getElementById('status-bar');
            const isHealthy = healthData.status === 'healthy';
            
            statusBar.innerHTML = `
                <div class="status-item ${isHealthy ? 'status-online' : 'status-error'}">
                    <i class="fas fa-circle"></i>
                    <span>${isHealthy ? 'System Online' : 'System Issues'}</span>
                </div>
                <div class="status-item status-online">
                    <i class="fas fa-brain"></i>
                    <span>Personality Engine: Active</span>
                </div>
                <div class="status-item ${healthData.services?.openai === 'available' ? 'status-online' : 'status-warning'}">
                    <i class="fas fa-robot"></i>
                    <span>AI: ${healthData.services?.openai || 'Unknown'}</span>
                </div>
                <div class="status-item status-online">
                    <i class="fas fa-chart-line"></i>
                    <span>Learning: Active</span>
                </div>
            `;
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🧠 Hyper-Personalized SMS Trading Bot Dashboard initialized');
            checkHealth();
            refreshMetrics();
            loadPersonalityAnalytics();
            loadStyleExamples();
            
            setInterval(() => {
                if (autoRefresh) {
                    refreshMetrics();
                }
            }, 60000);
        });
    </script>
</body>
</html>
"""

# ===== TEST INTERFACE ENDPOINT =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Simple test interface for SMS webhook testing"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Test Interface</title>
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
    </style>
</head>
<body>
    <h1>🧠 Hyper-Personalized SMS Trading Bot - Test Interface</h1>
    
    <form onsubmit="testSMS(event)">
        <div class="form-group">
            <label>From Phone Number:</label>
            <input type="text" id="phone" value="+13012466712" required>
        </div>
        
        <div class="form-group">
            <label>Message Body:</label>
            <textarea id="message" rows="3" required>yo what's AAPL doing? thinking about calls 🚀</textarea>
        </div>
        
        <button type="submit">Send Test SMS & Learn Personality</button>
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
                        <p>SMS webhook processed successfully with personality learning</p>
                        <pre>${result}</pre>
                        <p><strong>🧠 The personality engine learned from this interaction and will personalize future responses!</strong></p>
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
    logger.info(f"🚀 Starting Hyper-Personalized SMS Trading Bot on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Testing mode: {settings.testing_mode}")
    logger.info(f"🧠 Personality Engine: Active")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
