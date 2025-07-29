# ===== main.py - CLAUDE-POWERED WITH BACKGROUND JOBS =====
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os
from datetime import datetime, timezone
import time
import asyncio
from collections import defaultdict
from typing import Dict, List, Any

# Import configuration
try:
    from config import settings
    # Ensure the settings object has all required attributes
    if not hasattr(settings, 'anthropic_api_key'):
        # Monkey patch the missing attribute if config.py doesn't have it
        settings.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not hasattr(settings, 'prefer_claude'):
        settings.prefer_claude = os.getenv('PREFER_CLAUDE', 'true').lower() == 'true'
    if not hasattr(settings, 'eodhd_api_key'):
        settings.eodhd_api_key = os.getenv('EODHD_API_KEY')
    if not hasattr(settings, 'ta_service_url'):
        settings.ta_service_url = os.getenv('TA_SERVICE_URL', 'http://localhost:8001')
    logger.info("✅ Configuration loaded from config.py")
except ImportError:
    # Fallback configuration
    class Settings:
        def __init__(self):
            self.environment = "development"
            self.log_level = "INFO"
            self.testing_mode = True
            self.mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/ai')
            self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
            self.eodhd_api_key = os.getenv('EODHD_API_KEY')
            self.ta_service_url = os.getenv('TA_SERVICE_URL', 'http://localhost:8001')
            self.prefer_claude = os.getenv('PREFER_CLAUDE', 'true').lower() == 'true'
    
    settings = Settings()
    logger.info("✅ Configuration loaded from fallback")

# Import services with error handling
try:
    from services.database import DatabaseService
    logger.info("✅ DatabaseService imported")
except Exception as e:
    DatabaseService = None
    logger.error(f"❌ DatabaseService failed: {e}")

try:
    from services.openai_service import OpenAIService
    logger.info("✅ OpenAIService imported")
except Exception as e:
    OpenAIService = None
    logger.error(f"❌ OpenAIService failed: {e}")

try:
    import anthropic
    logger.info("✅ Anthropic library imported")
except Exception as e:
    anthropic = None
    logger.error(f"❌ Anthropic library failed: {e}")

try:
    from services.twilio_service import TwilioService
    logger.info("✅ TwilioService imported")
except Exception as e:
    TwilioService = None
    logger.error(f"❌ TwilioService failed: {e}")

try:
    from services.technical_analysis import TAEngine
    logger.info("✅ TAEngine imported")
except Exception as e:
    TAEngine = None
    logger.error(f"❌ TAEngine failed: {e}")

# Import both OpenAI and Claude agents
try:
    from services.llm_agent import ComprehensiveMessageProcessor
    logger.info("✅ OpenAI Agent imported")
except Exception as e:
    ComprehensiveMessageProcessor = None
    logger.error(f"❌ OpenAI Agent failed: {e}")

try:
    from services.claude_data_driven_agent import ClaudeMessageProcessor, HybridProcessor
    logger.info("✅ Claude Agent imported")
except Exception as e:
    ClaudeMessageProcessor = None
    HybridProcessor = None
    logger.error(f"❌ Claude Agent failed: {e}")

try:
    from services.cache_service import CacheService
    logger.info("✅ CacheService imported")
except Exception as e:
    CacheService = None
    logger.error(f"❌ CacheService failed: {e}")

try:
    from services.news_sentiment import NewsSentimentService
    logger.info("✅ NewsSentimentService imported")
except Exception as e:
    NewsSentimentService = None
    logger.error(f"❌ NewsSentimentService failed: {e}")

try:
    from services.fundamental_analysis import FAEngine
    logger.info("✅ FAEngine imported")
except Exception as e:
    FAEngine = None
    logger.error(f"❌ FAEngine failed: {e}")

# Import background job services
try:
    from services.backgroundjob.data_pipeline import BackgroundDataPipeline
    logger.info("✅ BackgroundDataPipeline imported")
except Exception as e:
    BackgroundDataPipeline = None
    logger.error(f"❌ BackgroundDataPipeline failed: {e}")

try:
    from services.backgroundjob.screener_service import EODHDScreener
    logger.info("✅ EODHDScreener imported")
except Exception as e:
    EODHDScreener = None
    logger.error(f"❌ EODHDScreener failed: {e}")

try:
    from services.backgroundjob.options_service import OptionsAnalyzer
    logger.info("✅ OptionsAnalyzer imported")
except Exception as e:
    OptionsAnalyzer = None
    logger.error(f"❌ OptionsAnalyzer failed: {e}")

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level)

# ===== SIMPLIFIED PERSONALITY ENGINE =====

class UserPersonalityEngine:
    """Simplified personality engine for conversation learning"""
    
    def __init__(self):
        self.user_profiles = defaultdict(lambda: {
            "communication_style": {"formality": "professional"},
            "trading_personality": {"experience_level": "intermediate"},
            "learning_data": {"total_messages": 0}
        })
    
    def learn_from_message(self, phone_number: str, message: str, intent: dict):
        """Simple learning from user interactions"""
        profile = self.user_profiles[phone_number]
        profile["learning_data"]["total_messages"] += 1
        
        # Simple formality detection
        if any(word in message.lower() for word in ['yo', 'hey', 'gonna']):
            profile["communication_style"]["formality"] = "casual"
        elif any(word in message.lower() for word in ['please', 'analysis']):
            profile["communication_style"]["formality"] = "professional"
    
    def get_user_profile(self, phone_number: str) -> dict:
        """Get user profile"""
        return self.user_profiles.get(phone_number, {})

# Initialize personality engine
personality_engine = UserPersonalityEngine()

# ===== GLOBAL SERVICES =====
db_service = None
openai_service = None
anthropic_client = None
twilio_service = None
ta_service = None
news_service = None
fundamental_tool = None
cache_service = None
openai_agent = None
claude_agent = None
hybrid_agent = None
active_agent = None

# Background job services
background_pipeline = None
cached_screener = None
options_analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown with Claude support and background jobs"""
    global db_service, openai_service, anthropic_client, twilio_service, ta_service, news_service
    global fundamental_tool, cache_service, openai_agent, claude_agent, hybrid_agent, active_agent
    global background_pipeline, cached_screener, options_analyzer
    
    logger.info("🚀 Starting SMS Trading Bot with Claude-Powered Agent and Background Jobs...")
    
    try:
        # Initialize core services
        if DatabaseService:
            db_service = DatabaseService()
            await db_service.initialize()
            logger.info("✅ Database service initialized")
        
        if CacheService and db_service and hasattr(db_service, 'redis'):
            cache_service = CacheService(db_service.redis)
            logger.info("✅ Cache service initialized")
        
        if OpenAIService:
            openai_service = OpenAIService()
            logger.info("✅ OpenAI service initialized")
        
        # Initialize Anthropic client
        if anthropic and settings.anthropic_api_key:
            anthropic_client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key
            )
            logger.info("✅ Anthropic client initialized")
        else:
            logger.warning("⚠️ Anthropic client not available - missing API key or library")
        
        if TwilioService:
            twilio_service = TwilioService()
            logger.info("✅ Twilio service initialized")
        
        if TAEngine:
            ta_service = TAEngine()
            logger.info("✅ Technical Analysis service initialized")
        
        if NewsSentimentService:
            news_service = NewsSentimentService(
                redis_client=db_service.redis if db_service else None,
                openai_service=openai_service
            )
            logger.info("✅ News Sentiment service initialized")
        
        if FAEngine and settings.eodhd_api_key:
            fundamental_tool = FAEngine(
                eodhd_api_key=settings.eodhd_api_key,
                redis_client=db_service.redis if db_service else None
            )
            logger.info("✅ Fundamental Analysis tool initialized")
        
        TechnicalAnalysisService = TAEngine  # Backward compatibility
        FundamentalAnalysisTool = FAEngine   # Backward compatibility

        # Initialize background data pipeline
        if BackgroundDataPipeline and settings.eodhd_api_key and settings.mongodb_url:
            try:
                background_pipeline = BackgroundDataPipeline(
                    mongodb_url=settings.mongodb_url,
                    redis_url=settings.redis_url,
                    eodhd_api_key=settings.eodhd_api_key,
                    ta_service_url=settings.ta_service_url
                )
                await background_pipeline.initialize()
                background_pipeline.start_scheduler()
                logger.info("✅ Background data pipeline started")
                
                # Check if we need initial data load
                stock_count = await background_pipeline.db.stocks.count_documents({})
                if stock_count == 0:
                    logger.info("📊 Cache empty - triggering initial data load...")
                    asyncio.create_task(background_pipeline.daily_data_refresh_job())
                
            except Exception as e:
                logger.warning(f"⚠️ Background pipeline startup failed: {e}")
                background_pipeline = None
        else:
            logger.info("⚠️ Background pipeline disabled - missing EODHD_API_KEY or MongoDB URL")
        
        # Initialize cached screener if pipeline is available
        if background_pipeline and EODHDScreener:
            try:
                cached_screener = EODHDScreener(
                    eodhd_api_key=settings.eodhd_api_key,
                    redis_client=background_pipeline.redis_client
                )
                logger.info("✅ Cached screener service initialized")
            except Exception as e:
                logger.warning(f"⚠️ Cached screener initialization failed: {e}")
        
        # Initialize options analyzer
        if OptionsAnalyzer and settings.eodhd_api_key:
            try:
                options_analyzer = OptionsAnalyzer(
                    eodhd_api_key=settings.eodhd_api_key,
                    redis_client=db_service.redis if db_service else None
                )
                logger.info("✅ Options analyzer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Options analyzer initialization failed: {e}")
        
        # Initialize OpenAI agent
        if ComprehensiveMessageProcessor and openai_service:
            openai_agent = ComprehensiveMessageProcessor(
                openai_client=openai_service.client,
                ta_service=ta_service,
                personality_engine=personality_engine,
                cache_service=cache_service,
                news_service=news_service,
                fundamental_tool=fundamental_tool
            )
            logger.info("✅ OpenAI Agent initialized")
        
        # Initialize Claude agent
        if ClaudeMessageProcessor and anthropic_client:
            claude_agent = ClaudeMessageProcessor(
                anthropic_client=anthropic_client,
                ta_service=ta_service,
                personality_engine=personality_engine,
                cache_service=cache_service,
                news_service=news_service,
                fundamental_tool=fundamental_tool
            )
            logger.info("✅ Claude Agent initialized")
        
        # Initialize hybrid agent if both are available
        if HybridProcessor and claude_agent and openai_agent:
            hybrid_agent = HybridProcessor(claude_agent, openai_agent)
            logger.info("✅ Hybrid Agent initialized")
        
        # Determine active agent based on preference and availability
        prefer_claude = getattr(settings, 'prefer_claude', True)  # Default to preferring Claude
        
        if prefer_claude and claude_agent:
            active_agent = claude_agent
            agent_type = "Claude (Primary)"
        elif hybrid_agent:
            active_agent = hybrid_agent
            agent_type = "Hybrid (Claude + OpenAI)"
        elif claude_agent:
            active_agent = claude_agent
            agent_type = "Claude (Only Available)"
        elif openai_agent:
            active_agent = openai_agent
            agent_type = "OpenAI (Fallback)"
        else:
            active_agent = None
            agent_type = "None Available"
        
        # Status report
        logger.info(f"🤖 Active Agent: {agent_type}")
        logger.info(f"🔧 Available Agents: Claude={claude_agent is not None}, OpenAI={openai_agent is not None}, Hybrid={hybrid_agent is not None}")
        logger.info(f"📊 Background Jobs: Pipeline={background_pipeline is not None}, Screener={cached_screener is not None}, Options={options_analyzer is not None}")
        logger.info("✅ Startup completed")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.warning("⚠️ Continuing in degraded mode")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    try:
        if background_pipeline:
            await background_pipeline.close()
            logger.info("✅ Background pipeline shutdown")
        if db_service and hasattr(db_service, 'close'):
            await db_service.close()
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# ===== CREATE APP =====

app = FastAPI(
    title="SMS Trading Bot",
    description="Claude-Powered SMS Trading Assistant with Background Data Pipeline",
    version="4.1.0",
    lifespan=lifespan
)

# ===== CORE ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API",
        "status": "running",
        "version": "4.1.0",
        "agent_type": get_agent_type(),
        "architecture": "claude_powered_with_background_jobs",
        "services": {
            "database": db_service is not None,
            "cache": cache_service is not None,
            "technical_analysis": ta_service is not None,
            "news_sentiment": news_service is not None,
            "fundamental_analysis": fundamental_tool is not None,
            "openai_agent": openai_agent is not None,
            "claude_agent": claude_agent is not None,
            "hybrid_agent": hybrid_agent is not None,
            "active_agent": active_agent is not None,
            "background_pipeline": background_pipeline is not None,
            "cached_screener": cached_screener is not None,
            "options_analyzer": options_analyzer is not None
        }
    }

def get_agent_type():
    """Get current active agent type"""
    if active_agent == claude_agent:
        return "claude_primary"
    elif active_agent == hybrid_agent:
        return "hybrid_claude_openai"
    elif active_agent == openai_agent:
        return "openai_fallback"
    else:
        return "none_available"

@app.get("/health")
async def health_check():
    """Health check endpoint with Claude status and background jobs"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "4.1.0",
            "agent_type": get_agent_type(),
            "services": {
                "openai": "available" if openai_service else "unavailable",
                "anthropic": "available" if anthropic_client else "unavailable",
                "twilio": "available" if twilio_service else "unavailable",
                "technical_analysis": "available" if ta_service else "unavailable",
                "news_sentiment": "available" if news_service else "unavailable",
                "fundamental_analysis": "available" if fundamental_tool else "unavailable",
                "cache": "available" if cache_service else "unavailable",
                "openai_agent": "available" if openai_agent else "unavailable",
                "claude_agent": "available" if claude_agent else "unavailable",
                "hybrid_agent": "available" if hybrid_agent else "unavailable"
            },
            "background_jobs": {
                "data_pipeline": "active" if background_pipeline else "inactive",
                "cached_screener": "available" if cached_screener else "unavailable",
                "options_analyzer": "available" if options_analyzer else "unavailable"
            },
            "preferences": {
                "prefer_claude": settings.prefer_claude,
                "active_agent": get_agent_type()
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(status_code=503, content={"status": "error", "error": str(e)})

# ===== SMS WEBHOOK (CLAUDE-POWERED) =====

@app.post("/webhook/sms")
async def sms_webhook(request: Request):
    """Handle incoming SMS messages with Claude-powered processing"""
    try:
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return PlainTextResponse("Missing required fields", status_code=400)
        
        # Process with Claude-powered agent
        response_text = await process_sms_message(message_body, from_number)
        
        # Return Twilio XML response
        return PlainTextResponse(
            '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"❌ SMS webhook error: {e}")
        return PlainTextResponse("Internal error", status_code=500)

async def process_sms_message(message_body: str, phone_number: str) -> str:
    """SMS processing with Claude-powered agent"""
    
    start_time = time.time()
    
    try:
        logger.info(f"📱 Processing: '{message_body}' from {phone_number}")
        
        # Use active agent (Claude preferred)
        if active_agent:
            response_text = await active_agent.process_message(message_body, phone_number)
            processing_time = time.time() - start_time
            agent_used = get_agent_type()
            logger.info(f"✅ {agent_used} processed in {processing_time:.2f}s")
        else:
            # Simple fallback
            response_text = "Trading assistant temporarily unavailable. Please try again in a moment."
            logger.warning("⚠️ Using fallback - no agents available")
        
        # Send SMS
        if twilio_service and response_text:
            sms_sent = await twilio_service.send_message(phone_number, response_text)
            if not sms_sent:
                logger.error(f"❌ Failed to send SMS to {phone_number}")
        
        return response_text
        
    except Exception as e:
        logger.error(f"💥 SMS processing failed: {e}")
        error_response = "Sorry, I'm having technical issues. Please try again!"
        
        if twilio_service:
            await twilio_service.send_message(phone_number, error_response)
        
        return error_response

# ===== BACKGROUND JOB ADMIN ENDPOINTS =====

@app.get("/admin/backgroundjob/status")
async def get_background_job_status():
    """Get background job status and statistics"""
    if not background_pipeline:
        return {
            "status": "not_initialized", 
            "reason": "Missing EODHD_API_KEY, MongoDB URL, or initialization failed",
            "available_services": {
                "screener": cached_screener is not None,
                "options": options_analyzer is not None
            }
        }
    
    try:
        pipeline_status = await background_pipeline.get_job_status()
        
        # Add database stats
        stock_count = await background_pipeline.db.stocks.count_documents({})
        latest_update = await background_pipeline.db.stocks.find_one(
            sort=[("last_updated", -1)]
        )
        
        # Add service availability
        pipeline_status.update({
            "database_stats": {
                "total_stocks_in_db": stock_count,
                "latest_update": latest_update.get("last_updated") if latest_update else None,
                "collection_exists": stock_count > 0
            },
            "available_services": {
                "cached_screener": cached_screener is not None,
                "options_analyzer": options_analyzer is not None,
                "background_pipeline": True
            },
            "schedules": {
                "daily_refresh": "06:00 AM ET daily",
                "weekly_cleanup": "Sunday 02:00 AM ET"
            }
        })
        
        return pipeline_status
        
    except Exception as e:
        logger.error(f"❌ Background job status error: {e}")
        return {"error": str(e)}

@app.post("/admin/backgroundjob/run-daily")  
async def trigger_daily_job():
    """Manually trigger daily data refresh"""
    if not background_pipeline:
        return {"error": "Background pipeline not initialized"}
    
    try:
        # Run in background to avoid timeout
        asyncio.create_task(background_pipeline.force_run_daily_job())
        
        return {
            "success": True,
            "message": "Daily data refresh job triggered",
            "note": "Job running in background - check /admin/backgroundjob/status for progress",
            "estimated_duration": "5-15 minutes depending on API response times"
        }
        
    except Exception as e:
        logger.error(f"❌ Manual daily job trigger failed: {e}")
        return {"error": str(e)}

@app.post("/admin/backgroundjob/cleanup")
async def trigger_cleanup_job():
    """Manually trigger weekly cleanup"""  
    if not background_pipeline:
        return {"error": "Background pipeline not initialized"}
    
    try:
        # Run in background
        asyncio.create_task(background_pipeline.force_run_cleanup())
        
        return {
            "success": True,
            "message": "Weekly cleanup job triggered",
            "note": "Job running in background - check /admin/backgroundjob/status for progress",
            "warning": "This will clear all cached stock data"
        }
        
    except Exception as e:
        logger.error(f"❌ Manual cleanup trigger failed: {e}")
        return {"error": str(e)}

@app.get("/admin/backgroundjob/sample-data/{symbol}")
async def get_sample_cached_data(symbol: str):
    """Get sample cached data for a symbol (debugging)"""
    if not background_pipeline:
        return {"error": "Background pipeline not initialized"}
    
    try:
        symbol = symbol.upper()
        
        # Get from MongoDB
        mongo_data = await background_pipeline.db.stocks.find_one({"symbol": symbol})
        
        # Get from Redis
        redis_basic = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:basic")
        redis_technical = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:technical")
        redis_fundamental = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:fundamental")
        redis_tags = await background_pipeline.redis_client.smembers(f"stock:{symbol}:tags")
        
        return {
            "symbol": symbol,
            "mongodb_data": mongo_data,
            "redis_data": {
                "basic": dict(redis_basic) if redis_basic else {},
                "technical": dict(redis_technical) if redis_technical else {},
                "fundamental": dict(redis_fundamental) if redis_fundamental else {},
                "tags": list(redis_tags) if redis_tags else []
            },
            "data_available": {
                "in_mongodb": bool(mongo_data),
                "in_redis": bool(redis_basic)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Sample data fetch failed: {e}")
        return {"error": str(e)}

@app.get("/admin/backgroundjob/stocks-by-tag/{tag}")
async def get_stocks_by_screening_tag(tag: str):
    """Get stocks by screening tag for testing"""
    if not background_pipeline:
        return {"error": "Background pipeline not initialized"}
    
    try:
        # Query MongoDB for stocks with specific tag
        stocks = await background_pipeline.db.stocks.find(
            {"screening_tags": tag}
        ).limit(20).to_list(length=20)
        
        result = []
        for stock in stocks:
            result.append({
                "symbol": stock["symbol"],
                "price": stock.get("basic", {}).get("price", 0),
                "sector": stock.get("basic", {}).get("sector", "Unknown"),
                "market_cap": stock.get("basic", {}).get("market_cap", 0),
                "tags": stock.get("screening_tags", [])
            })
        
        return {
            "tag": tag,
            "matching_stocks": len(result),
            "stocks": result,
            "available_tags": ["large_cap", "mid_cap", "small_cap", "mega_cap", "overbought", "oversold", "neutral_rsi", "value", "growth", "sector_*"]
        }
        
    except Exception as e:
        logger.error(f"❌ Tag query failed: {e}")
        return {"error": str(e)}

# ===== OPTIONS AND SCREENER ENDPOINTS =====

@app.get("/admin/screener/test/{criteria}")
async def test_cached_screener(criteria: str):
    """Test cached screener with preset criteria"""
    if not cached_screener:
        return {"error": "Cached screener not available"}
    
    try:
        # Preset criteria mapping
        preset_criteria = {
            "momentum": {
                "change_1d_min": 2,
                "volume_min": 1000000,
                "market_cap_min": 1000000000
            },
            "value": {
                "pe_max": 20,
                "market_cap_min": 2000000000,
                "volume_min": 500000
            },
            "growth": {
                "change_1m_min": 10,
                "volume_min": 1000000,
                "market_cap_min": 1000000000
            }
        }
        
        if criteria in preset_criteria:
            # Use cached screener if background pipeline is available
            if background_pipeline:
                # Query MongoDB directly for faster results
                filters = preset_criteria[criteria]
                query = {}
                
                if "market_cap_min" in filters:
                    query["basic.market_cap"] = {"$gte": filters["market_cap_min"]}
                if "pe_max" in filters:
                    query["fundamental.pe"] = {"$lte": filters["pe_max"]}
                if "change_1d_min" in filters:
                    query["basic.change_1d"] = {"$gte": filters["change_1d_min"]}
                
                cursor = background_pipeline.db.stocks.find(query).limit(10)
                results = await cursor.to_list(length=10)
                
                return {
                    "criteria": criteria,
                    "filters_applied": filters,
                    "results_count": len(results),
                    "results": [
                        {
                            "symbol": r["symbol"],
                            "price": r.get("basic", {}).get("price", 0),
                            "change_1d": r.get("basic", {}).get("change_1d", 0),
                            "pe": r.get("fundamental", {}).get("pe", 0),
                            "market_cap": r.get("basic", {}).get("market_cap", 0)
                        } for r in results
                    ],
                    "data_source": "cached_mongodb"
                }
            else:
                # Fallback to API screener
                results = await cached_screener.screen_stocks(preset_criteria[criteria])
                return results
        else:
            return {
                "error": f"Unknown criteria '{criteria}'",
                "available_criteria": list(preset_criteria.keys())
            }
            
    except Exception as e:
        logger.error(f"❌ Screener test failed: {e}")
        return {"error": str(e)}

@app.get("/admin/options/test/{symbol}")
async def test_options_analyzer(symbol: str):
    """Test options analyzer for a symbol"""
    if not options_analyzer:
        return {"error": "Options analyzer not available - missing EODHD_API_KEY"}
    
    try:
        symbol = symbol.upper()
        results = await options_analyzer.analyze_options_chain(symbol)
        
        return {
            "symbol": symbol,
            "analysis": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Options analysis test failed: {e}")
        return {"error": str(e)}

# ===== ADMIN ENDPOINTS =====

@app.get("/admin")
async def admin_dashboard():
    """Admin dashboard with Claude status and background jobs"""
    try:
        return {
            "title": "SMS Trading Bot Admin - Claude-Powered with Background Jobs",
            "status": "operational",
            "version": "4.1.0",
            "architecture": "claude_powered_with_background_jobs",
            "agent_type": get_agent_type(),
            "services": {
                "database": "connected" if db_service else "disconnected",
                "cache": "active" if cache_service else "inactive",
                "technical_analysis": "active" if ta_service else "inactive",
                "news_sentiment": "active" if news_service else "inactive",
                "fundamental_analysis": "active" if fundamental_tool else "inactive",
                "openai_agent": "active" if openai_agent else "inactive",
                "claude_agent": "active" if claude_agent else "inactive",
                "hybrid_agent": "active" if hybrid_agent else "inactive"
            },
            "background_jobs": {
                "data_pipeline": "active" if background_pipeline else "inactive",
                "cached_screener": "available" if cached_screener else "unavailable",
                "options_analyzer": "available" if options_analyzer else "unavailable",
                "schedule": {
                    "daily_refresh": "06:00 AM ET",
                    "weekly_cleanup": "Sunday 02:00 AM ET"
                }
            },
            "ai_providers": {
                "openai": {
                    "available": openai_service is not None,
                    "agent_ready": openai_agent is not None
                },
                "anthropic": {
                    "available": anthropic_client is not None,
                    "agent_ready": claude_agent is not None
                },
                "hybrid": {
                    "available": hybrid_agent is not None,
                    "active": active_agent == hybrid_agent
                }
            },
            "users": {
                "total_profiles": len(personality_engine.user_profiles)
            },
            "features": {
                "claude_reasoning": claude_agent is not None,
                "hybrid_fallback": hybrid_agent is not None,
                "conversation_awareness": True,
                "smart_tool_calling": True,
                "professional_responses": True,
                "context_retention": True,
                "superior_analysis": claude_agent is not None,
                "cached_data_pipeline": background_pipeline is not None,
                "advanced_screening": cached_screener is not None,
                "options_analysis": options_analyzer is not None
            },
            "preferences": {
                "prefer_claude": settings.prefer_claude,
                "active_agent": get_agent_type()
            }
        }
    except Exception as e:
        logger.error(f"❌ Admin dashboard error: {e}")
        return {"error": str(e)}

# ===== DEBUG ENDPOINTS =====

@app.post("/debug/test-message")
async def test_message_processing(request: Request):
    """Test Claude-powered message processing"""
    try:
        data = await request.json()
        message = data.get('message', 'Would you advise buying PLTR right now?')
        phone = data.get('phone', '+1555TEST')
        force_agent = data.get('force_agent', None)  # 'claude', 'openai', or 'hybrid'
        
        # Select agent based on force_agent parameter
        if force_agent == 'claude' and claude_agent:
            test_agent = claude_agent
            agent_used = 'claude_forced'
        elif force_agent == 'openai' and openai_agent:
            test_agent = openai_agent
            agent_used = 'openai_forced'
        elif force_agent == 'hybrid' and hybrid_agent:
            test_agent = hybrid_agent
            agent_used = 'hybrid_forced'
        else:
            test_agent = active_agent
            agent_used = get_agent_type()
        
        if test_agent:
            response = await test_agent.process_message(message, phone)
        else:
            response = "No agents available for testing"
            agent_used = "none"
        
        return {
            "success": True,
            "input_message": message,
            "phone_number": phone,
            "bot_response": response,
            "agent_used": agent_used,
            "available_agents": {
                "claude": claude_agent is not None,
                "openai": openai_agent is not None,
                "hybrid": hybrid_agent is not None
            },
            "background_services": {
                "data_pipeline": background_pipeline is not None,
                "cached_screener": cached_screener is not None,
                "options_analyzer": options_analyzer is not None
            },
            "features_active": {
                "claude_reasoning": claude_agent is not None,
                "conversation_awareness": active_agent is not None,
                "context_retention": cache_service is not None,
                "smart_tool_calling": True,
                "superior_analysis": claude_agent is not None,
                "cached_screening": background_pipeline is not None,
                "options_analysis": options_analyzer is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug/diagnose")
async def diagnose_services():
    """Service diagnosis with Claude status and background jobs"""
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "environment_variables": {
            "OPENAI_API_KEY": "Set" if settings.openai_api_key else "Missing",
            "ANTHROPIC_API_KEY": "Set" if settings.anthropic_api_key else "Missing",
            "EODHD_API_KEY": "Set" if settings.eodhd_api_key else "Missing",
            "MONGODB_URL": "Set" if settings.mongodb_url else "Missing",
            "REDIS_URL": "Set" if settings.redis_url else "Missing",
            "TWILIO_ACCOUNT_SID": "Set" if settings.twilio_account_sid else "Missing",
            "TA_SERVICE_URL": "Set" if settings.ta_service_url else "Missing",
            "PREFER_CLAUDE": settings.prefer_claude
        },
        "service_status": {
            "database": db_service is not None,
            "cache": cache_service is not None,
            "openai": openai_service is not None,
            "anthropic": anthropic_client is not None,
            "twilio": twilio_service is not None,
            "technical_analysis": ta_service is not None,
            "news_sentiment": news_service is not None,
            "fundamental_analysis": fundamental_tool is not None,
            "openai_agent": openai_agent is not None,
            "claude_agent": claude_agent is not None,
            "hybrid_agent": hybrid_agent is not None,
            "background_pipeline": background_pipeline is not None,
            "cached_screener": cached_screener is not None,
            "options_analyzer": options_analyzer is not None
        },
        "architecture": "claude_powered_with_background_jobs",
        "active_agent": get_agent_type(),
        "recommendations": []
    }
    
    # Generate recommendations
    if not settings.anthropic_api_key:
        diagnosis["recommendations"].append("❌ Set ANTHROPIC_API_KEY for Claude-powered responses")
    if not settings.openai_api_key:
        diagnosis["recommendations"].append("⚠️ Set OPENAI_API_KEY for OpenAI fallback")
    if not settings.eodhd_api_key:
        diagnosis["recommendations"].append("❌ Set EODHD_API_KEY for market data and background jobs")
    if not settings.mongodb_url:
        diagnosis["recommendations"].append("❌ Set MONGODB_URL for data storage")
    if not active_agent:
        diagnosis["recommendations"].append("❌ No agents available - check API keys")
    if not background_pipeline:
        diagnosis["recommendations"].append("⚠️ Background data pipeline not running - check EODHD_API_KEY and MongoDB")
    
    if claude_agent and openai_agent and background_pipeline:
        diagnosis["recommendations"].append("✅ All systems operational - Claude + OpenAI + Background Jobs active!")
    elif claude_agent and background_pipeline:
        diagnosis["recommendations"].append("✅ Claude agent + Background jobs operational - excellent setup!")
    elif background_pipeline:
        diagnosis["recommendations"].append("✅ Background jobs operational - cached data available!")
    
    if not diagnosis["recommendations"]:
        diagnosis["recommendations"].append("✅ All systems operational!")
    
    return diagnosis

# ===== CLAUDE TEST INTERFACE =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Test interface with Claude vs OpenAI comparison and background job status"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Claude + Background Jobs Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button.claude { background: #8B5CF6; }
        button.background { background: #10B981; }
        button.compare { background: #F59E0B; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; background: #f8f9fa; }
        .comparison { display: flex; gap: 20px; }
        .agent-result { flex: 1; padding: 15px; border-radius: 4px; }
        .claude-result { background: #F3F4F6; border-left: 4px solid #8B5CF6; }
        .openai-result { background: #FFF7ED; border-left: 4px solid #F59E0B; }
        .quick-tests { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .badge { background: #8B5CF6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px; }
        .badge.openai { background: #F59E0B; }
        .badge.hybrid { background: #10B981; }
        .badge.background { background: #6366F1; }
        .status-panel { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>SMS Trading Bot - Claude + Background Jobs Test Interface</h1>
    <div>
        <span class="badge">Claude-Powered</span>
        <span class="badge openai">OpenAI Fallback</span>
        <span class="badge hybrid">Hybrid Available</span>
        <span class="badge background">Background Pipeline</span>
    </div>
    
    <div class="status-panel">
        <h3>System Status</h3>
        <div id="system-status">Loading system status...</div>
        <button onclick="refreshStatus()">Refresh Status</button>
    </div>
    
    <div class="quick-tests">
        <button onclick="quickTest('Would you advise buying PLTR right now?')">Test Advice Question</button>
        <button onclick="quickTest('Screen momentum stocks')">Test Cached Screening</button>
        <button onclick="quickTest('AAPL options analysis')">Test Options Analysis</button>
        <button onclick="quickTest('What are the fundamentals for NVDA?')">Test Data Question</button>
        <button onclick="quickTest('How is TSLA doing technically?')">Test Technical</button>
        <button onclick="quickTest('Tell me about AAPL news')">Test News</button>
    </div>
    
    <div class="quick-tests">
        <button class="background" onclick="testBackgroundJob('status')">Check Background Status</button>
        <button class="background" onclick="testBackgroundJob('daily')">Trigger Daily Job</button>
        <button class="background" onclick="testScreener('momentum')">Test Momentum Screener</button>
        <button class="background" onclick="testOptions('AAPL')">Test Options (AAPL)</button>
    </div>
    
    <form onsubmit="testSMS(event)">
        <div class="form-group">
            <label>Phone Number:</label>
            <input type="text" id="phone" value="+1555TEST" required>
        </div>
        
        <div class="form-group">
            <label>Message:</label>
            <textarea id="message" rows="3" required>Would you advise buying PLTR right now?</textarea>
        </div>
        
        <div class="form-group">
            <label>Test Agent:</label>
            <select id="agent">
                <option value="">Active Agent (Claude Preferred)</option>
                <option value="claude">Force Claude</option>
                <option value="openai">Force OpenAI</option>
                <option value="hybrid">Force Hybrid</option>
            </select>
        </div>
        
        <button type="submit">Test Active Agent</button>
        <button type="button" class="claude" onclick="testSpecificAgent('claude')">Test Claude</button>
        <button type="button" onclick="testSpecificAgent('openai')">Test OpenAI</button>
        <button type="button" class="compare" onclick="compareAgents()">Compare Both</button>
    </form>
    
    <div id="result"></div>
    
    <script>
        // Load system status on page load
        window.onload = function() {
            refreshStatus();
        };
        
        async function refreshStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                
                document.getElementById('system-status').innerHTML = `
                    <strong>Agent:</strong> ${data.agent_type} | 
                    <strong>Claude:</strong> ${data.services.claude_agent} | 
                    <strong>OpenAI:</strong> ${data.services.openai_agent} | 
                    <strong>Background Pipeline:</strong> ${data.background_jobs.data_pipeline} | 
                    <strong>Screener:</strong> ${data.background_jobs.cached_screener} | 
                    <strong>Options:</strong> ${data.background_jobs.options_analyzer}
                `;
            } catch (error) {
                document.getElementById('system-status').innerHTML = 'Error loading status';
            }
        }
        
        function quickTest(message) {
            document.getElementById('message').value = message;
            testSMS(new Event('submit'));
        }
        
        async function testBackgroundJob(jobType) {
            document.getElementById('result').innerHTML = `<div class="result">🔄 Testing background ${jobType}...</div>`;
            
            try {
                let url = '';
                if (jobType === 'status') {
                    url = '/admin/backgroundjob/status';
                } else if (jobType === 'daily') {
                    url = '/admin/backgroundjob/run-daily';
                }
                
                const method = jobType === 'status' ? 'GET' : 'POST';
                const response = await fetch(url, { method });
                const data = await response.json();
                
                document.getElementById('result').innerHTML = 
                    `<div class="result">
                        <h3>✅ Background Job ${jobType}</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function testScreener(criteria) {
            document.getElementById('result').innerHTML = `<div class="result">🔄 Testing ${criteria} screener...</div>`;
            
            try {
                const response = await fetch(`/admin/screener/test/${criteria}`);
                const data = await response.json();
                
                document.getElementById('result').innerHTML = 
                    `<div class="result">
                        <h3>✅ Screener Test: ${criteria}</h3>
                        <p><strong>Results:</strong> ${data.results_count || 'N/A'} stocks found</p>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function testOptions(symbol) {
            document.getElementById('result').innerHTML = `<div class="result">🔄 Testing options analysis for ${symbol}...</div>`;
            
            try {
                const response = await fetch(`/admin/options/test/${symbol}`);
                const data = await response.json();
                
                document.getElementById('result').innerHTML = 
                    `<div class="result">
                        <h3>✅ Options Analysis: ${symbol}</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function testSMS(event) {
            event.preventDefault();
            
            const phone = document.getElementById('phone').value;
            const message = document.getElementById('message').value;
            const agent = document.getElementById('agent').value;
            
            document.getElementById('result').innerHTML = '<div class="result">🔄 Processing...</div>';
            
            try {
                const response = await fetch('/debug/test-message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, phone, force_agent: agent || undefined })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('result').innerHTML = 
                        `<div class="result">
                            <h3>✅ Response from ${data.agent_used}</h3>
                            <p><strong>Input:</strong> ${data.input_message}</p>
                            <p><strong>Response:</strong> ${data.bot_response}</p>
                            <p><strong>Background Services:</strong> 
                                Pipeline: ${data.background_services?.data_pipeline ? '✅' : '❌'}, 
                                Screener: ${data.background_services?.cached_screener ? '✅' : '❌'}, 
                                Options: ${data.background_services?.options_analyzer ? '✅' : '❌'}
                            </p>
                        </div>`;
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function testSpecificAgent(agentType) {
            const phone = document.getElementById('phone').value;
            const message = document.getElementById('message').value;
            
            document.getElementById('result').innerHTML = `<div class="result">🔄 Testing ${agentType}...</div>`;
            
            try {
                const response = await fetch('/debug/test-message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, phone, force_agent: agentType })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('result').innerHTML = 
                        `<div class="result">
                            <h3>✅ ${agentType.toUpperCase()} Response</h3>
                            <p><strong>Input:</strong> ${data.input_message}</p>
                            <p><strong>Response:</strong> ${data.bot_response}</p>
                        </div>`;
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function compareAgents() {
            const phone = document.getElementById('phone').value;
            const message = document.getElementById('message').value;
            
            document.getElementById('result').innerHTML = '<div class="result">🔄 Comparing Claude vs OpenAI...</div>';
            
            try {
                const response = await fetch('/debug/compare-agents', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, phone })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    const claudeResult = data.comparison.claude;
                    const openaiResult = data.comparison.openai;
                    
                    document.getElementById('result').innerHTML = 
                        `<div class="result">
                            <h3>🔍 Agent Comparison</h3>
                            <p><strong>Input:</strong> ${data.input_message}</p>
                            <div class="comparison">
                                <div class="agent-result claude-result">
                                    <h4>🧠 Claude Response</h4>
                                    <p><strong>Status:</strong> ${claudeResult.status}</p>
                                    ${claudeResult.response ? `<p><strong>Response:</strong> ${claudeResult.response}</p>` : ''}
                                </div>
                                <div class="agent-result openai-result">
                                    <h4>🤖 OpenAI Response</h4>
                                    <p><strong>Status:</strong> ${openaiResult.status}</p>
                                    ${openaiResult.response ? `<p><strong>Response:</strong> ${openaiResult.response}</p>` : ''}
                                </div>
                            </div>
                            <p><strong>Recommendation:</strong> ${data.recommendation}</p>
                        </div>`;
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

# ===== RUN SERVER =====

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Claude-Powered SMS Trading Bot with Background Jobs on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Claude Preference: {settings.prefer_claude}")
    logger.info(f"Available APIs: Claude={anthropic is not None and settings.anthropic_api_key}, OpenAI={settings.openai_api_key is not None}, EODHD={settings.eodhd_api_key is not None}")
    logger.info(f"Background Jobs: Enabled={BackgroundDataPipeline is not None and settings.eodhd_api_key is not None}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development"
    )
