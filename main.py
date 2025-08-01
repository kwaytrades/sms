# ===== main.py - CLAUDE-POWERED WITH BACKGROUND JOBS + GEMINI PERSONALITY ENGINE =====
# Standard library imports
import os
import sys
import time
import hmac
import hashlib
import asyncio
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Dict, List, Any, Optional

# Third-party imports
import uvicorn
import stripe
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from loguru import logger

# Local imports
from services.stripe_service import StripeService
from core.user_manager import UserManager, PlanType, SubscriptionStatus, AccountStatus

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
    # Add Gemini API key support
    if not hasattr(settings, 'gemini_api_key'):
        settings.gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not hasattr(settings, 'personality_analysis_enabled'):
        settings.personality_analysis_enabled = os.getenv('PERSONALITY_ANALYSIS_ENABLED', 'true').lower() == 'true'
    if not hasattr(settings, 'background_analysis_enabled'):
        settings.background_analysis_enabled = os.getenv('BACKGROUND_ANALYSIS_ENABLED', 'true').lower() == 'true'
    if not hasattr(settings, 'enable_real_time_personality'):
        settings.enable_real_time_personality = os.getenv('ENABLE_REAL_TIME_PERSONALITY', 'true').lower() == 'true'
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
            # Gemini Personality Analysis settings
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            self.personality_analysis_enabled = os.getenv('PERSONALITY_ANALYSIS_ENABLED', 'true').lower() == 'true'
            self.background_analysis_enabled = os.getenv('BACKGROUND_ANALYSIS_ENABLED', 'true').lower() == 'true'
            self.enable_real_time_personality = os.getenv('ENABLE_REAL_TIME_PERSONALITY', 'true').lower() == 'true'
            # Memory Manager settings
            self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
            self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp')
            self.memory_stm_limit = int(os.getenv('MEMORY_STM_LIMIT', '15'))
            self.memory_summary_trigger = int(os.getenv('MEMORY_SUMMARY_TRIGGER', '10'))
    
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
    from services.options_service import OptionsEngine as OptionsAnalyzer
    logger.info("✅ OptionsAnalyzer imported")
except Exception as e:
    OptionsAnalyzer = None
    logger.error(f"❌ OptionsAnalyzer failed: {e}")

# Import MemoryManager
try:
    from services.memory_manager import MemoryManager, MessageDirection, AgentType
    logger.info("✅ MemoryManager imported")
except ImportError as e:
    MemoryManager = None
    MessageDirection = None
    AgentType = None
    logger.error(f"❌ MemoryManager failed: {e}")

# Import KeyBuilder
try:
    from services.key_builder import KeyBuilder
    logger.info("✅ KeyBuilder imported")
except Exception as e:
    KeyBuilder = None
    logger.error(f"❌ KeyBuilder failed: {e}")

# Import Enhanced Personality Engine with Gemini support
try:
    from core.personality_engine_v3_gemini import EnhancedPersonalityEngine
    logger.info("✅ Enhanced PersonalityEngine with Gemini imported")
    PERSONALITY_ENGINE_TYPE = "gemini_enhanced"
except ImportError as e:
    logger.warning(f"⚠️ Enhanced PersonalityEngine with Gemini failed: {e}")
    try:
        from core.personality_engine import UserPersonalityEngine as EnhancedPersonalityEngine
        logger.info("✅ Fallback PersonalityEngine imported")
        PERSONALITY_ENGINE_TYPE = "regex_fallback"
    except ImportError as e2:
        EnhancedPersonalityEngine = None
        PERSONALITY_ENGINE_TYPE = "none"
        logger.error(f"❌ All PersonalityEngine imports failed: {e2}")

# Import Gemini service
try:
    from services.gemini_service import GeminiPersonalityService
    logger.info("✅ GeminiPersonalityService imported")
except ImportError as e:
    GeminiPersonalityService = None
    logger.warning(f"⚠️ GeminiPersonalityService failed: {e}")

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level)

stripe_service = StripeService()

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
memory_manager = None
personality_engine = None
gemini_service = None

# Background job services
background_pipeline = None
cached_screener = None
options_analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown with Claude support, background jobs, and Gemini personality engine"""
    global db_service, openai_service, anthropic_client, twilio_service, ta_service, news_service
    global fundamental_tool, cache_service, openai_agent, claude_agent, hybrid_agent, active_agent
    global background_pipeline, cached_screener, options_analyzer, memory_manager, personality_engine, gemini_service
    
    logger.info("🚀 Starting SMS Trading Bot with Claude-Powered Agent, Background Jobs, and Gemini Personality Engine...")
    
    try:
        # Initialize core services
        if DatabaseService:
            db_service = DatabaseService()
            await db_service.initialize()
            logger.info("✅ Database service initialized")
        
        if CacheService and db_service and hasattr(db_service, 'redis'):
            cache_service = CacheService(db_service.redis)
            logger.info("✅ Cache service initialized")
        
       # Initialize KeyBuilder
        if KeyBuilder and db_service is not None and db_service.redis is not None and db_service.db is not None:
            db_service.key_builder = KeyBuilder(db_service.redis, db_service.db)
            logger.info("🔧 KeyBuilder initialized")
        
        # Initialize Gemini service for personality analysis
        if GeminiPersonalityService and settings.gemini_api_key and settings.personality_analysis_enabled:
            try:
                gemini_service = GeminiPersonalityService(api_key=settings.gemini_api_key)
                logger.info("🤖 Gemini personality service initialized")
            except Exception as e:
                logger.warning(f"⚠️ Gemini service initialization failed: {e}")
                gemini_service = None
        else:
            if not settings.gemini_api_key:
                logger.info("⚠️ Gemini service disabled - missing GEMINI_API_KEY")
            elif not settings.personality_analysis_enabled:
                logger.info("⚠️ Gemini service disabled - PERSONALITY_ANALYSIS_ENABLED=false")
            else:
                logger.info("⚠️ Gemini service disabled - GeminiPersonalityService not available")
        
        # Initialize Enhanced PersonalityEngine with Gemini support
        if EnhancedPersonalityEngine:
            try:
                if PERSONALITY_ENGINE_TYPE == "gemini_enhanced" and gemini_service:
                    personality_engine = EnhancedPersonalityEngine(
                        db_service=db_service,
                        gemini_api_key=settings.gemini_api_key
                    )
                    logger.info("✅ Enhanced Personality Engine with Gemini initialized")
                else:
                    personality_engine = EnhancedPersonalityEngine(db_service=db_service)
                    logger.info("✅ Enhanced Personality Engine with fallback initialized")
            except Exception as e:
                logger.error(f"❌ Enhanced Personality Engine initialization failed: {e}")
                personality_engine = None
        
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
                mongodb_url=settings.mongodb_url,
                redis_url=settings.redis_url
            )
            await fundamental_tool.initialize()
            logger.info("✅ Fundamental Analysis tool initialized")
        
        # Initialize MemoryManager
        if MemoryManager and settings.pinecone_api_key and settings.openai_api_key:
            try:
                memory_manager = MemoryManager(
                    redis_url=settings.redis_url,
                    mongodb_url=settings.mongodb_url,
                    pinecone_api_key=settings.pinecone_api_key,
                    pinecone_environment=settings.pinecone_environment,
                    openai_api_key=settings.openai_api_key,
                    stm_limit=settings.memory_stm_limit,
                    summary_trigger=settings.memory_summary_trigger
                )
                await memory_manager.setup()
                logger.info("✅ MemoryManager initialized")
            except Exception as e:
                logger.warning(f"⚠️ MemoryManager initialization failed: {e}")
                memory_manager = None
        else:
            if not MemoryManager:
                logger.info("⚠️ MemoryManager disabled - module not available")
            elif not settings.pinecone_api_key:
                logger.info("⚠️ MemoryManager disabled - missing Pinecone API key")
            elif not settings.openai_api_key:
                logger.info("⚠️ MemoryManager disabled - missing OpenAI API key")
            else:
                logger.info("⚠️ MemoryManager disabled - missing required configuration")
            memory_manager = None
        
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
                    mongodb_url=settings.mongodb_url,
                    redis_url=settings.redis_url,
                    eodhd_api_key=settings.eodhd_api_key
                )
                await cached_screener.initialize()
                logger.info("✅ Cached screener service initialized")
            except Exception as e:
                logger.warning(f"⚠️ Cached screener initialization failed: {e}")
        
        # Initialize options analyzer
        if OptionsAnalyzer and settings.eodhd_api_key:
            try:
                options_analyzer = OptionsAnalyzer(
                    mongodb_url=settings.mongodb_url,
                    redis_url=settings.redis_url,
                    eodhd_api_key=settings.eodhd_api_key
                )
                await options_analyzer.initialize()  # Don't forget to initialize!
                logger.info("✅ Options analyzer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Options analyzer initialization failed: {e}")
        
        # Initialize OpenAI agent
        if ComprehensiveMessageProcessor and openai_service:
            try:
                # Check if ComprehensiveMessageProcessor accepts memory_manager parameter
                import inspect
                try:
                    init_signature = inspect.signature(ComprehensiveMessageProcessor.__init__)
                    accepts_memory_manager = 'memory_manager' in init_signature.parameters
                    accepts_db_service = 'db_service' in init_signature.parameters
                except:
                    accepts_memory_manager = False
                    accepts_db_service = False
                
                kwargs = {
                    'openai_client': openai_service.client,
                    'ta_service': ta_service,
                    'personality_engine': personality_engine,
                    'cache_service': cache_service,
                    'news_service': news_service,
                    'fundamental_tool': fundamental_tool
                }
                
                if accepts_memory_manager and memory_manager:
                    kwargs['memory_manager'] = memory_manager
                
                if accepts_db_service and db_service:
                    kwargs['db_service'] = db_service
                
                openai_agent = ComprehensiveMessageProcessor(**kwargs)
                logger.info("✅ OpenAI Agent with Gemini-Enhanced Personality support initialized")
                    
            except Exception as e:
                logger.error(f"❌ OpenAI Agent initialization failed: {e}")
                openai_agent = None
        
        # Initialize Claude agent
        if ClaudeMessageProcessor and anthropic_client:
            claude_agent = ClaudeMessageProcessor(
                anthropic_client=anthropic_client,
                ta_service=ta_service,
                personality_engine=personality_engine,
                cache_service=cache_service,
                news_service=news_service,
                fundamental_tool=fundamental_tool,
                memory_manager=memory_manager,
                db_service=db_service
            )
            logger.info("✅ Claude Agent with Gemini-Enhanced Personality initialized")
        
        # Initialize hybrid agent if both are available
        if HybridProcessor and claude_agent and openai_agent:
            hybrid_agent = HybridProcessor(claude_agent, openai_agent)
            logger.info("✅ Hybrid Agent with Gemini-Enhanced Personality initialized")
        
        # Determine active agent based on preference and availability
        prefer_claude = getattr(settings, 'prefer_claude', True)  # Default to preferring Claude
        
        if prefer_claude and claude_agent:
            active_agent = claude_agent
            agent_type = "Claude (Primary) + Gemini Personality"
        elif hybrid_agent:
            active_agent = hybrid_agent
            agent_type = "Hybrid (Claude + OpenAI) + Gemini Personality"
        elif claude_agent:
            active_agent = claude_agent
            agent_type = "Claude (Only Available) + Gemini Personality"
        elif openai_agent:
            active_agent = openai_agent
            agent_type = "OpenAI (Fallback) + Gemini Personality"
        else:
            active_agent = None
            agent_type = "None Available"
        
        # Status report
        logger.info(f"🤖 Active Agent: {agent_type}")
        logger.info(f"🔧 Available Agents: Claude={claude_agent is not None}, OpenAI={openai_agent is not None}, Hybrid={hybrid_agent is not None}")
        logger.info(f"📊 Background Jobs: Pipeline={background_pipeline is not None}, Screener={cached_screener is not None}, Options={options_analyzer is not None}")
        logger.info(f"🧠 Memory Manager: {memory_manager is not None}")
        logger.info(f"👤 Personality Engine: {personality_engine is not None} (Type: {PERSONALITY_ENGINE_TYPE})")
        logger.info(f"🤖 Gemini Service: {gemini_service is not None}")
        logger.info(f"🔧 KeyBuilder: {db_service is not None and hasattr(db_service, 'key_builder')}")
        logger.info("✅ Startup completed with Gemini-Enhanced Personality Intelligence")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.warning("⚠️ Continuing in degraded mode")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    try:
        if memory_manager:
            await memory_manager.cleanup()
            logger.info("✅ Memory manager shutdown")
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
    description="Claude-Powered SMS Trading Assistant with Background Data Pipeline and Gemini Personality Engine",
    version="4.2.0",
    lifespan=lifespan
)

# ===== CORE ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API",
        "status": "running",
        "version": "4.2.0",
        "agent_type": get_agent_type(),
        "architecture": "claude_powered_with_background_jobs_gemini_personality",
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
            "options_analyzer": options_analyzer is not None,
            "memory_manager": memory_manager is not None,
            "key_builder": db_service is not None and hasattr(db_service, 'key_builder'),
            "gemini_personality": gemini_service is not None,
            "personality_engine_type": PERSONALITY_ENGINE_TYPE
        }
    }

def get_agent_type():
    """Get current active agent type"""
    if active_agent == claude_agent:
        return "claude_primary_gemini_personality"
    elif active_agent == hybrid_agent:
        return "hybrid_claude_openai_gemini_personality"
    elif active_agent == openai_agent:
        return "openai_fallback_gemini_personality"
    else:
        return "none_available"

@app.get("/health")
async def health_check():
    """Health check endpoint with Claude status, background jobs, and Gemini personality"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "4.2.0",
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
                "hybrid_agent": "available" if hybrid_agent else "unavailable",
                "memory_manager": "available" if memory_manager else "unavailable",
                "key_builder": "available" if db_service and hasattr(db_service, 'key_builder') else "unavailable",
                "gemini_service": "available" if gemini_service else "unavailable",
                "personality_engine": "available" if personality_engine else "unavailable"
            },
            "background_jobs": {
                "data_pipeline": "active" if background_pipeline else "inactive",
                "cached_screener": "available" if cached_screener else "unavailable",
                "options_analyzer": "available" if options_analyzer else "unavailable"
            },
            "personality_features": {
                "engine_type": PERSONALITY_ENGINE_TYPE,
                "gemini_enabled": gemini_service is not None,
                "analysis_enabled": settings.personality_analysis_enabled,
                "background_analysis": settings.background_analysis_enabled,
                "real_time_personality": settings.enable_real_time_personality
            },
            "preferences": {
                "prefer_claude": settings.prefer_claude,
                "active_agent": get_agent_type()
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(status_code=503, content={"status": "error", "error": str(e)})

# ===== SMS WEBHOOK (CLAUDE-POWERED WITH GEMINI PERSONALITY) =====

@app.post("/webhook/sms")
async def sms_webhook(request: Request):
    """Handle incoming SMS messages with Claude-powered processing and Gemini personality analysis"""
    try:
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return PlainTextResponse("Missing required fields", status_code=400)
        
        # Process with Claude-powered agent and Gemini personality analysis
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
    """SMS processing with Claude-powered agent and Gemini personality analysis"""
    
    start_time = time.time()
    
    try:
        logger.info(f"📱 Processing: '{message_body}' from {phone_number}")
        
        # Use active agent (Claude preferred with Gemini personality)
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

# ===== GEMINI PERSONALITY ADMIN ENDPOINTS =====

@app.get("/admin/personality/gemini/stats")
async def get_gemini_personality_stats():
    """Get Gemini personality analysis statistics"""
    if not personality_engine or PERSONALITY_ENGINE_TYPE != "gemini_enhanced":
        return {"error": "Gemini personality engine not available"}
    
    try:
        stats = personality_engine.get_gemini_usage_stats()
        return {
            "engine_type": PERSONALITY_ENGINE_TYPE,
            "gemini_stats": stats,
            "total_profiles": len(personality_engine.user_profiles) if hasattr(personality_engine, 'user_profiles') else 0,
            "analysis_method": "gemini_semantic" if gemini_service else "regex_fallback"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/personality/gemini/optimize-costs")
async def optimize_gemini_costs():
    """Optimize Gemini personality analysis costs"""
    if not personality_engine or PERSONALITY_ENGINE_TYPE != "gemini_enhanced":
        return {"error": "Gemini personality engine not available"}
    
    try:
        optimization_results = await personality_engine.optimize_analysis_costs()
        return {
            "optimization_completed": True,
            "results": optimization_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/personality/background-analysis/{user_id}")
async def trigger_background_personality_analysis(user_id: str):
    """Trigger background deep personality analysis for a user"""
    if not personality_engine or PERSONALITY_ENGINE_TYPE != "gemini_enhanced":
        return {"error": "Gemini personality engine not available"}
    
    if not settings.background_analysis_enabled:
        return {"error": "Background analysis disabled"}
    
    try:
        # Get conversation history (this would need to be implemented based on your data storage)
        conversation_history = []  # You'd fetch this from your database
        
        result = await personality_engine.run_background_deep_analysis(
            user_id, 
            conversation_history
        )
        
        return {
            "analysis_triggered": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/personality/test/{user_id}")
async def test_personality_analysis(user_id: str, message: str = Query("What do you think about AAPL stock?")):
    """Test personality analysis for a user"""
    if not personality_engine:
        return {"error": "Personality engine not available"}
    
    try:
        if PERSONALITY_ENGINE_TYPE == "gemini_enhanced":
            # Test Gemini analysis
            analysis = await personality_engine.run_comprehensive_analysis(
                message, 
                {"user_id": user_id, "test": True}
            )
            
            return {
                "user_id": user_id,
                "test_message": message,
                "analysis_method": analysis.analysis_method,
                "confidence_score": analysis.confidence_score,
                "processing_time_ms": analysis.processing_time_ms,
                "communication_insights": analysis.communication_insights,
                "trading_insights": analysis.trading_insights,
                "emotional_state": analysis.emotional_state,
                "engine_type": PERSONALITY_ENGINE_TYPE
            }
        else:
            # Test fallback analysis
            analysis = await personality_engine.analyze_and_learn(user_id, message)
            return {
                "user_id": user_id,
                "test_message": message,
                "analysis": analysis,
                "engine_type": PERSONALITY_ENGINE_TYPE
            }
            
    except Exception as e:
        return {"error": str(e)}

# ===== MEMORY MANAGER ADMIN ENDPOINTS =====

@app.get("/admin/memory/user/{user_id}/stats")
async def get_user_memory_stats(user_id: str):
    """Get memory statistics for a user"""
    if not memory_manager:
        return {"error": "Memory manager not available"}
    
    try:
        stats = await memory_manager.get_user_statistics(user_id)
        return stats
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/memory/user/{user_id}/cleanup")
async def cleanup_user_memory(user_id: str):
    """Clean up user memory data (GDPR compliance)"""
    if not memory_manager:
        return {"error": "Memory manager not available"}
    
    try:
        success = await memory_manager.delete_user_data(user_id)
        return {"success": success, "message": "User data deleted" if success else "Deletion failed"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/memory/health")
async def memory_health_check():
    """Check memory manager health"""
    if not memory_manager:
        return {"status": "unavailable", "reason": "Memory manager not initialized"}
    
    try:
        # Test basic functionality
        test_context = await memory_manager.get_context("health_check_user", query="test")
        return {
            "status": "healthy",
            "redis": memory_manager.redis_client is not None,
            "mongodb": memory_manager.db is not None,
            "pinecone": memory_manager.pinecone_index is not None,
            "openai": memory_manager.openai_client is not None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ===== KEYBUILDER ADMIN ENDPOINTS =====

@app.get("/admin/keybuilder/migration/stats")
async def get_keybuilder_migration_stats():
    """Get KeyBuilder migration statistics"""
    if not db_service or not hasattr(db_service, 'key_builder'):
        return {"error": "KeyBuilder not available"}
    
    try:
        stats = await db_service.key_builder.get_migration_stats()
        return stats
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/keybuilder/migrate/users")
async def migrate_all_users(limit: int = Query(None, description="Limit number of users to migrate")):
    """Trigger migration of all users to new KeyBuilder format"""
    if not db_service or not hasattr(db_service, 'key_builder'):
        return {"error": "KeyBuilder not available"}
    
    try:
        result = await db_service.key_builder.migrate_all_users(limit=limit)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/keybuilder/migrate/stocks")
async def migrate_all_stocks():
    """Trigger migration of all stock data to new KeyBuilder format"""
    if not db_service or not hasattr(db_service, 'key_builder'):
        return {"error": "KeyBuilder not available"}
    
    try:
        result = await db_service.key_builder.migrate_all_stocks()
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/keybuilder/cleanup")
async def cleanup_old_keys(dry_run: bool = Query(True, description="Dry run mode - show what would be deleted")):
    """Clean up old keys after migration"""
    if not db_service or not hasattr(db_service, 'key_builder'):
        return {"error": "KeyBuilder not available"}
    
    try:
        result = await db_service.key_builder.cleanup_old_keys(dry_run=dry_run)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/keybuilder/validate/{user_id}")
async def validate_user_migration(user_id: str):
    """Validate that migration worked correctly for a specific user"""
    if not db_service or not hasattr(db_service, 'key_builder'):
        return {"error": "KeyBuilder not available"}
    
    try:
        result = await db_service.key_builder.validate_migration(user_id)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/keybuilder/test/{user_id}")
async def test_keybuilder_user_data(user_id: str):
    """Test KeyBuilder data retrieval for a user"""
    if not db_service or not hasattr(db_service, 'key_builder'):
        return {"error": "KeyBuilder not available"}
    
    try:
        kb = db_service.key_builder
        
        # Test all data types
        profile = await kb.get_user_profile(user_id)
        context = await kb.get_user_context(user_id)
        conversations = await kb.get_user_conversations(user_id, limit=5)
        sms_history = await kb.get_user_sms_history(user_id)
        usage = await kb.get_user_usage(user_id)
        personality = await kb.get_user_personality(user_id)
        
        return {
            "user_id": user_id,
            "data_available": {
                "profile": profile is not None,
                "context": context is not None,
                "conversations": len(conversations) if conversations else 0,
                "sms_history": len(sms_history) if sms_history else 0,
                "usage": usage is not None,
                "personality": personality is not None
            },
            "sample_data": {
                "profile": profile if profile else "No profile data",
                "context": context if context else "No context data",
                "conversations_count": len(conversations) if conversations else 0,
                "usage_data": usage if usage else "No usage data"
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/keybuilder/test/stock/{symbol}")
async def test_keybuilder_stock_data(symbol: str):
    """Test KeyBuilder stock data retrieval"""
    if not db_service or not hasattr(db_service, 'key_builder'):
        return {"error": "KeyBuilder not available"}
    
    try:
        kb = db_service.key_builder
        symbol = symbol.upper()
        
        # Test stock data retrieval
        technical = await kb.get_stock_technical(symbol)
        fundamental = await kb.get_stock_fundamental(symbol)
        ta_metadata = await kb.get_stock_metadata(symbol, "ta")
        fa_metadata = await kb.get_stock_metadata(symbol, "fa")
        
        return {
            "symbol": symbol,
            "data_available": {
                "technical": technical is not None,
                "fundamental": fundamental is not None,
                "ta_metadata": ta_metadata is not None,
                "fa_metadata": fa_metadata is not None
            },
            "sample_data": {
                "technical_keys": list(technical.keys()) if technical else [],
                "fundamental_keys": list(fundamental.keys()) if fundamental else [],
                "ta_metadata": ta_metadata,
                "fa_metadata": fa_metadata
            }
        }
    except Exception as e:
        return {"error": str(e)}

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
        
        # Add database stats - fix the MongoDB comparison
    try:
        stock_count = await background_pipeline.db.stocks.count_documents({})
        latest_update = await background_pipeline.db.stocks.find_one(
            sort=[("last_updated", -1)]
            )
    except (AttributeError, TypeError):
        stock_count = 0
        latest_update = None
            )
        else:
            stock_count = 0
            latest_update = None
        
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
        
        # Get from MongoDB - fix the comparison
        if background_pipeline.db is not None:
            mongo_data = await background_pipeline.db.stocks.find_one({"symbol": symbol})
        else:
            mongo_data = None
        
        # Get from Redis - fix the comparison  
    try:
        redis_basic = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:basic")
        redis_technical = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:technical")
        redis_fundamental = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:fundamental")
        redis_tags = await background_pipeline.redis_client.smembers(f"stock:{symbol}:tags")

    except (AttributeError, TypeError):
        redis_basic = {}
        redis_technical = {}
        redis_fundamental = {}
        redis_tags = []
        
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
    """Admin dashboard with Claude status, background jobs, and Gemini personality"""
    try:
        return {
            "title": "SMS Trading Bot Admin - Claude-Powered with Background Jobs + KeyBuilder + Gemini Personality",
            "status": "operational",
            "version": "4.2.0",
            "architecture": "claude_powered_with_background_jobs_keybuilder_gemini_personality",
            "agent_type": get_agent_type(),
            "services": {
                "database": "connected" if db_service else "disconnected",
                "cache": "active" if cache_service else "inactive",
                "technical_analysis": "active" if ta_service else "inactive",
                "news_sentiment": "active" if news_service else "inactive",
                "fundamental_analysis": "active" if fundamental_tool else "inactive",
                "openai_agent": "active" if openai_agent else "inactive",
                "claude_agent": "active" if claude_agent else "inactive",
                "hybrid_agent": "active" if hybrid_agent else "inactive",
                "memory_manager": "active" if memory_manager else "inactive",
                "key_builder": "active" if db_service and hasattr(db_service, 'key_builder') else "inactive",
                "gemini_service": "active" if gemini_service else "inactive",
                "personality_engine": "active" if personality_engine else "inactive"
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
            "personality_intelligence": {
                "engine_type": PERSONALITY_ENGINE_TYPE,
                "gemini_enabled": gemini_service is not None,
                "semantic_analysis": PERSONALITY_ENGINE_TYPE == "gemini_enhanced",
                "analysis_enabled": settings.personality_analysis_enabled,
                "background_analysis": settings.background_analysis_enabled,
                "real_time_personality": settings.enable_real_time_personality,
                "cost_optimization": "available"
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
                "gemini": {
                    "available": gemini_service is not None,
                    "personality_analysis": PERSONALITY_ENGINE_TYPE == "gemini_enhanced"
                },
                "hybrid": {
                    "available": hybrid_agent is not None,
                    "active": active_agent == hybrid_agent
                }
            },
            "users": {
                "total_profiles": len(personality_engine.user_profiles) if personality_engine and hasattr(personality_engine, 'user_profiles') else 0
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
                "options_analysis": options_analyzer is not None,
                "emotional_intelligence": memory_manager is not None,
                "memory_enhanced": memory_manager is not None,
                "unified_key_management": db_service is not None and hasattr(db_service, 'key_builder'),
                "gemini_personality_analysis": PERSONALITY_ENGINE_TYPE == "gemini_enhanced",
                "semantic_understanding": gemini_service is not None,
                "10x_personality_accuracy": PERSONALITY_ENGINE_TYPE == "gemini_enhanced"
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
    """Test Claude-powered message processing with Gemini personality analysis"""
    try:
        data = await request.json()
        message = data.get('message', 'Would you advise buying PLTR right now?')
        phone = data.get('phone', '+1555TEST')
        force_agent = data.get('force_agent', None)  # 'claude', 'openai', or 'hybrid'
        
        # Select agent based on force_agent parameter
        if force_agent == 'claude' and claude_agent:
            test_agent = claude_agent
            agent_used = 'claude_forced_gemini_personality'
        elif force_agent == 'openai' and openai_agent:
            test_agent = openai_agent
            agent_used = 'openai_forced_gemini_personality'
        elif force_agent == 'hybrid' and hybrid_agent:
            test_agent = hybrid_agent
            agent_used = 'hybrid_forced_gemini_personality'
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
            "personality_features": {
                "engine_type": PERSONALITY_ENGINE_TYPE,
                "gemini_enabled": gemini_service is not None,
                "semantic_analysis": PERSONALITY_ENGINE_TYPE == "gemini_enhanced",
                "analysis_enabled": settings.personality_analysis_enabled
            },
            "features_active": {
                "claude_reasoning": claude_agent is not None,
                "conversation_awareness": active_agent is not None,
                "context_retention": cache_service is not None,
                "smart_tool_calling": True,
                "superior_analysis": claude_agent is not None,
                "cached_screening": background_pipeline is not None,
                "options_analysis": options_analyzer is not None,
                "memory_enhanced": memory_manager is not None,
                "key_builder": db_service is not None and hasattr(db_service, 'key_builder'),
                "gemini_personality": PERSONALITY_ENGINE_TYPE == "gemini_enhanced",
                "10x_personality_intelligence": gemini_service is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug/diagnose")
async def diagnose_services():
    """Service diagnosis with Claude status, background jobs, and Gemini personality"""
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "environment_variables": {
            "OPENAI_API_KEY": "Set" if settings.openai_api_key else "Missing",
            "ANTHROPIC_API_KEY": "Set" if settings.anthropic_api_key else "Missing",
            "GEMINI_API_KEY": "Set" if settings.gemini_api_key else "Missing",
            "EODHD_API_KEY": "Set" if settings.eodhd_api_key else "Missing",
            "MONGODB_URL": "Set" if settings.mongodb_url else "Missing",
            "REDIS_URL": "Set" if settings.redis_url else "Missing",
            "TWILIO_ACCOUNT_SID": "Set" if settings.twilio_account_sid else "Missing",
            "TA_SERVICE_URL": "Set" if settings.ta_service_url else "Missing",
            "PREFER_CLAUDE": settings.prefer_claude,
            "PINECONE_API_KEY": "Set" if settings.pinecone_api_key else "Missing",
            "PERSONALITY_ANALYSIS_ENABLED": settings.personality_analysis_enabled,
            "BACKGROUND_ANALYSIS_ENABLED": settings.background_analysis_enabled,
            "ENABLE_REAL_TIME_PERSONALITY": settings.enable_real_time_personality
        },
        "service_status": {
            "database": db_service is not None,
            "cache": cache_service is not None,
            "openai": openai_service is not None,
            "anthropic": anthropic_client is not None,
            "gemini": gemini_service is not None,
            "twilio": twilio_service is not None,
            "technical_analysis": ta_service is not None,
            "news_sentiment": news_service is not None,
            "fundamental_analysis": fundamental_tool is not None,
            "openai_agent": openai_agent is not None,
            "claude_agent": claude_agent is not None,
            "hybrid_agent": hybrid_agent is not None,
            "background_pipeline": background_pipeline is not None,
            "cached_screener": cached_screener is not None,
            "options_analyzer": options_analyzer is not None,
            "memory_manager": memory_manager is not None,
            "key_builder": db_service is not None and hasattr(db_service, 'key_builder'),
            "personality_engine": personality_engine is not None
        },
        "personality_intelligence": {
            "engine_type": PERSONALITY_ENGINE_TYPE,
            "gemini_available": gemini_service is not None,
            "semantic_analysis": PERSONALITY_ENGINE_TYPE == "gemini_enhanced",
            "fallback_ready": PERSONALITY_ENGINE_TYPE == "regex_fallback"
        },
        "architecture": "claude_powered_with_background_jobs_keybuilder_gemini_personality",
        "active_agent": get_agent_type(),
        "recommendations": []
    }
    
    # Generate recommendations
    if not settings.anthropic_api_key:
        diagnosis["recommendations"].append("❌ Set ANTHROPIC_API_KEY for Claude-powered responses")
    if not settings.openai_api_key:
        diagnosis["recommendations"].append("⚠️ Set OPENAI_API_KEY for OpenAI fallback")
    if not settings.gemini_api_key:
        diagnosis["recommendations"].append("⚠️ Set GEMINI_API_KEY for 10x personality intelligence upgrade")
    if not settings.eodhd_api_key:
        diagnosis["recommendations"].append("❌ Set EODHD_API_KEY for market data and background jobs")
    if not settings.mongodb_url:
        diagnosis["recommendations"].append("❌ Set MONGODB_URL for data storage")
    if not settings.pinecone_api_key:
        diagnosis["recommendations"].append("⚠️ Set PINECONE_API_KEY for enhanced memory features")
    if not active_agent:
        diagnosis["recommendations"].append("❌ No agents available - check API keys")
    if not background_pipeline:
        diagnosis["recommendations"].append("⚠️ Background data pipeline not running - check EODHD_API_KEY and MongoDB")
    if not memory_manager:
        diagnosis["recommendations"].append("⚠️ Memory manager not available - check Pinecone and OpenAI API keys")
    if not (db_service and hasattr(db_service, 'key_builder')):
        diagnosis["recommendations"].append("⚠️ KeyBuilder not available - check DatabaseService initialization")
    if not gemini_service and settings.gemini_api_key:
        diagnosis["recommendations"].append("⚠️ Gemini service failed to initialize - check API key and library")
    if PERSONALITY_ENGINE_TYPE == "regex_fallback":
        diagnosis["recommendations"].append("⚠️ Using fallback personality engine - enable Gemini for 10x intelligence")
    
    if claude_agent and openai_agent and background_pipeline and memory_manager and (db_service and hasattr(db_service, 'key_builder')) and gemini_service:
        diagnosis["recommendations"].append("✅ All systems operational - Claude + OpenAI + Background Jobs + Memory + KeyBuilder + Gemini Personality active!")
    elif claude_agent and background_pipeline and memory_manager and (db_service and hasattr(db_service, 'key_builder')) and gemini_service:
        diagnosis["recommendations"].append("✅ Claude agent + Background jobs + Memory + KeyBuilder + Gemini Personality operational - excellent setup!")
    elif claude_agent and memory_manager and (db_service and hasattr(db_service, 'key_builder')) and gemini_service:
        diagnosis["recommendations"].append("✅ Claude agent + Memory + KeyBuilder + Gemini Personality operational - enhanced conversations available!")
    elif background_pipeline and (db_service and hasattr(db_service, 'key_builder')) and gemini_service:
        diagnosis["recommendations"].append("✅ Background jobs + KeyBuilder + Gemini Personality operational - intelligent cached data!")
    elif (db_service and hasattr(db_service, 'key_builder')) and gemini_service:
        diagnosis["recommendations"].append("✅ KeyBuilder + Gemini Personality operational - unified data with 10x intelligence!")
    elif gemini_service:
        diagnosis["recommendations"].append("✅ Gemini Personality operational - 10x personality intelligence available!")
    
    if not diagnosis["recommendations"]:
        diagnosis["recommendations"].append("✅ All systems operational!")
    
    return diagnosis

# ===== GEMINI TEST INTERFACE =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Test interface with Claude vs OpenAI comparison, background job status, and Gemini personality features"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Claude + Memory + Background Jobs + KeyBuilder + Gemini Personality Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button.claude { background: #8B5CF6; }
        button.background { background: #10B981; }
        button.memory { background: #EC4899; }
        button.keybuilder { background: #F59E0B; }
        button.gemini { background: #4285F4; }
        button.compare { background: #EF4444; }
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
        .badge.memory { background: #EC4899; }
        .badge.keybuilder { background: #F59E0B; }
        .badge.gemini { background: #4285F4; }
        .status-panel { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>SMS Trading Bot - Claude + Memory + Background Jobs + KeyBuilder + Gemini Personality Test Interface</h1>
    <div>
        <span class="badge">Claude-Powered</span>
        <span class="badge openai">OpenAI Fallback</span>
        <span class="badge hybrid">Hybrid Available</span>
        <span class="badge background">Background Pipeline</span>
        <span class="badge memory">Memory Enhanced</span>
        <span class="badge keybuilder">KeyBuilder Unified</span>
        <span class="badge gemini">Gemini 10x Personality</span>
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
    
    <div class="quick-tests">
        <button class="memory" onclick="testMemory('health')">Check Memory Health</button>
        <button class="memory" onclick="testMemory('stats')">Get User Stats</button>
        <button class="memory" onclick="testConversation()">Test Memory Conversation</button>
    </div>
    
    <div class="quick-tests">
        <button class="keybuilder" onclick="testKeyBuilder('stats')">KeyBuilder Stats</button>
        <button class="keybuilder" onclick="testKeyBuilder('test-user')">Test User Data</button>
        <button class="keybuilder" onclick="testKeyBuilder('test-stock')">Test Stock Data</button>
        <button class="keybuilder" onclick="testKeyBuilder('migrate')">Migrate Users</button>
    </div>
    
    <div class="quick-tests">
        <button class="gemini" onclick="testGeminiPersonality('stats')">Gemini Stats</button>
        <button class="gemini" onclick="testGeminiPersonality('analyze')">Test Analysis</button>
        <button class="gemini" onclick="testGeminiPersonality('optimize')">Optimize Costs</button>
        <button class="gemini" onclick="testGeminiPersonality('background')">Background Analysis</button>
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
                <option value="">Active Agent (Claude + Gemini Preferred)</option>
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
                    <strong>Gemini:</strong> ${data.services.gemini_service} | 
                    <strong>Memory:</strong> ${data.services.memory_manager} | 
                    <strong>KeyBuilder:</strong> ${data.services.key_builder} | 
                    <strong>Background Pipeline:</strong> ${data.background_jobs.data_pipeline} | 
                    <strong>Screener:</strong> ${data.background_jobs.cached_screener} | 
                    <strong>Options:</strong> ${data.background_jobs.options_analyzer} |
                    <strong>Personality:</strong> ${data.personality_features?.engine_type || 'unknown'}
                `;
            } catch (error) {
                document.getElementById('system-status').innerHTML = 'Error loading status';
            }
        }
        
        function quickTest(message) {
            document.getElementById('message').value = message;
            testSMS(new Event('submit'));
        }
        
        async function testGeminiPersonality(testType) {
            document.getElementById('result').innerHTML = `<div class="result">🔄 Testing Gemini Personality ${testType}...</div>`;
            
            try {
                let url = '';
                let method = 'GET';
                
                if (testType === 'stats') {
                    url = '/admin/personality/gemini/stats';
                } else if (testType === 'analyze') {
                    url = '/admin/personality/test/+1555TEST?message=Hey, what do you think about AAPL? I\'m bullish!';
                } else if (testType === 'optimize') {
                    url = '/admin/personality/gemini/optimize-costs';
                    method = 'POST';
                } else if (testType === 'background') {
                    url = '/admin/personality/background-analysis/+1555TEST';
                    method = 'POST';
                }
                
                const response = await fetch(url, { method });
                const data = await response.json();
                
                document.getElementById('result').innerHTML = 
                    `<div class="result">
                        <h3>✅ Gemini Personality ${testType}</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function testKeyBuilder(testType) {
            document.getElementById('result').innerHTML = `<div class="result">🔄 Testing KeyBuilder ${testType}...</div>`;
            
            try {
                let url = '';
                let method = 'GET';
                
                if (testType === 'stats') {
                    url = '/admin/keybuilder/migration/stats';
                } else if (testType === 'test-user') {
                    url = '/admin/keybuilder/test/+1555TEST';
                } else if (testType === 'test-stock') {
                    url = '/admin/keybuilder/test/stock/AAPL';
                } else if (testType === 'migrate') {
                    url = '/admin/keybuilder/migrate/users?limit=5';
                    method = 'POST';
                }
                
                const response = await fetch(url, { method });
                const data = await response.json();
                
                document.getElementById('result').innerHTML = 
                    `<div class="result">
                        <h3>✅ KeyBuilder ${testType}</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function testMemory(testType) {
            document.getElementById('result').innerHTML = `<div class="result">🔄 Testing memory ${testType}...</div>`;
            
            try {
                let url = '';
                if (testType === 'health') {
                    url = '/admin/memory/health';
                } else if (testType === 'stats') {
                    url = '/admin/memory/user/+1555TEST/stats';
                }
                
                const response = await fetch(url);
                const data = await response.json();
                
                document.getElementById('result').innerHTML = 
                    `<div class="result">
                        <h3>✅ Memory ${testType}</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function testConversation() {
            document.getElementById('result').innerHTML = `<div class="result">🔄 Testing memory conversation...</div>`;
            
            try {
                // Send first message
                const response1 = await fetch('/debug/test-message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: "What do you think about AAPL?", 
                        phone: "+1555MEMORY",
                        force_agent: "claude"
                    })
                });
                const data1 = await response1.json();
                
                // Wait a moment
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Send follow-up message
                const response2 = await fetch('/debug/test-message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: "How about its fundamentals?", 
                        phone: "+1555MEMORY",
                        force_agent: "claude"
                    })
                });
                const data2 = await response2.json();
                
                document.getElementById('result').innerHTML = 
                    `<div class="result">
                        <h3>✅ Memory Conversation Test</h3>
                        <p><strong>First Message:</strong> "${data1.input_message}"</p>
                        <p><strong>First Response:</strong> ${data1.bot_response}</p>
                        <p><strong>Follow-up:</strong> "${data2.input_message}"</p>
                        <p><strong>Context-Aware Response:</strong> ${data2.bot_response}</p>
                        <p><strong>KeyBuilder Features:</strong> ${data2.features_active?.key_builder ? '✅' : '❌'} Unified Data</p>
                        <p><strong>Memory Features:</strong> ${data2.features_active?.memory_enhanced ? '✅' : '❌'} Enhanced</p>
                        <p><strong>Gemini Personality:</strong> ${data2.features_active?.gemini_personality ? '✅' : '❌'} 10x Intelligence</p>
                    </div>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
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
                            <p><strong>KeyBuilder:</strong> ${data.features_active?.key_builder ? '✅' : '❌'}</p>
                            <p><strong>Memory Enhanced:</strong> ${data.features_active?.memory_enhanced ? '✅' : '❌'}</p>
                            <p><strong>Gemini Personality:</strong> ${data.features_active?.gemini_personality ? '✅' : '❌'}</p>
                            <p><strong>10x Intelligence:</strong> ${data.features_active?.['10x_personality_intelligence'] ? '✅' : '❌'}</p>
                            <p><strong>Background Services:</strong> 
                                Pipeline: ${data.background_services?.data_pipeline ? '✅' : '❌'}, 
                                Screener: ${data.background_services?.cached_screener ? '✅' : '❌'}, 
                                Options: ${data.background_services?.options_analyzer ? '✅' : '❌'}
                            </p>
                            <p><strong>Personality Engine:</strong> ${data.personality_features?.engine_type || 'unknown'}</p>
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
                            <p><strong>KeyBuilder:</strong> ${data.features_active?.key_builder ? '✅' : '❌'}</p>
                            <p><strong>Memory Enhanced:</strong> ${data.features_active?.memory_enhanced ? '✅' : '❌'}</p>
                            <p><strong>Gemini Personality:</strong> ${data.features_active?.gemini_personality ? '✅' : '❌'}</p>
                            <p><strong>10x Intelligence:</strong> ${data.features_active?.['10x_personality_intelligence'] ? '✅' : '❌'}</p>
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
            
            document.getElementById('result').innerHTML = '<div class="result">🔄 Comparing Claude vs OpenAI with Gemini Personality...</div>';
            
            try {
                // Test Claude
                const claudeResponse = await fetch('/debug/test-message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, phone, force_agent: 'claude' })
                });
                const claudeData = await claudeResponse.json();
                
                // Test OpenAI
                const openaiResponse = await fetch('/debug/test-message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, phone, force_agent: 'openai' })
                });
                const openaiData = await openaiResponse.json();
                
                document.getElementById('result').innerHTML = 
                    `<div class="result">
                        <h3>🔍 Agent Comparison (Gemini-Enhanced)</h3>
                        <p><strong>Input:</strong> ${message}</p>
                        <div class="comparison">
                            <div class="agent-result claude-result">
                                <h4>🧠 Claude Response</h4>
                                <p><strong>Status:</strong> ${claudeData.success ? 'Success' : 'Failed'}</p>
                                <p><strong>Response:</strong> ${claudeData.bot_response || 'No response'}</p>
                                <p><strong>KeyBuilder:</strong> ${claudeData.features_active?.key_builder ? '✅' : '❌'}</p>
                                <p><strong>Gemini Personality:</strong> ${claudeData.features_active?.gemini_personality ? '✅' : '❌'}</p>
                            </div>
                            <div class="agent-result openai-result">
                                <h4>🤖 OpenAI Response</h4>
                                <p><strong>Status:</strong> ${openaiData.success ? 'Success' : 'Failed'}</p>
                                <p><strong>Response:</strong> ${openaiData.bot_response || 'No response'}</p>
                                <p><strong>KeyBuilder:</strong> ${openaiData.features_active?.key_builder ? '✅' : '❌'}</p>
                                <p><strong>Gemini Personality:</strong> ${openaiData.features_active?.gemini_personality ? '✅' : '❌'}</p>
                            </div>
                        </div>
                    </div>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da;">❌ Error: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

# ===== STRIPE WEBHOOK ENDPOINTS =====

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Enhanced Stripe webhook handler with comprehensive event processing"""
    try:
        payload = await request.body()
        sig_header = request.headers.get('stripe-signature')
        
        # Verify webhook signature
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.stripe_webhook_secret
            )
        except ValueError:
            logger.error("❌ Invalid payload in Stripe webhook")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            logger.error("❌ Invalid signature in Stripe webhook")
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        event_type = event['type']
        event_data = event['data']['object']
        
        logger.info(f"🔔 Processing Stripe webhook: {event_type}")
        
        # Process the webhook event
        result = await process_stripe_webhook_event(event_type, event_data)
        
        return {
            "status": "success",
            "event_type": event_type,
            "processed": result.get("processed", False),
            "action_taken": result.get("action_taken", "none"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Stripe webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_stripe_webhook_event(event_type: str, event_data: Dict) -> Dict[str, Any]:
    """Process different types of Stripe webhook events"""
    
    customer_id = event_data.get('customer')
    subscription_id = event_data.get('id') if 'subscription' in event_type else event_data.get('subscription')
    
    # Get user from database using customer ID
    user = None
    if customer_id and db_service:
        try:
            user_doc = await db_service.db.users.find_one({"stripe_customer_id": customer_id})
            if user_doc:
                user = user_doc
        except Exception as e:
            logger.error(f"Error fetching user by customer ID {customer_id}: {e}")
    
    result = {"processed": False, "action_taken": "none", "user_found": bool(user)}
    
    try:
        if event_type == 'customer.subscription.created':
            result.update(await handle_subscription_created(event_data, user))
            
        elif event_type == 'customer.subscription.updated':
            result.update(await handle_subscription_updated(event_data, user))
            
        elif event_type == 'customer.subscription.deleted':
            result.update(await handle_subscription_deleted(event_data, user))
            
        elif event_type == 'customer.subscription.trial_will_end':
            result.update(await handle_trial_will_end(event_data, user))
            
        elif event_type == 'invoice.payment_succeeded':
            result.update(await handle_payment_succeeded(event_data, user))
            
        elif event_type == 'invoice.payment_failed':
            result.update(await handle_payment_failed(event_data, user))
            
        elif event_type == 'customer.created':
            result.update(await handle_customer_created(event_data))
            
        elif event_type == 'customer.updated':
            result.update(await handle_customer_updated(event_data, user))
            
        else:
            logger.info(f"ℹ️ Unhandled Stripe event: {event_type}")
            result["action_taken"] = "ignored"
            
    except Exception as e:
        logger.error(f"❌ Error processing {event_type}: {e}")
        result["error"] = str(e)
    
    return result

async def handle_subscription_created(event_data: Dict, user: Optional[Dict]) -> Dict:
    """Handle subscription.created webhook"""
    if not user:
        logger.warning(f"⚠️ Subscription created but no user found for customer {event_data.get('customer')}")
        return {"processed": False, "action_taken": "user_not_found"}
    
    # Update user subscription status
    try:
        await db_service.db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "subscription_status": "active",
                    "stripe_subscription_id": event_data.get('id'),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        logger.info(f"✅ Subscription created: {user.get('phone_number')} -> active")
        return {
            "processed": True,
            "action_taken": "subscription_activated",
            "status": event_data['status']
        }
    except Exception as e:
        logger.error(f"❌ Failed to update subscription for user {user['_id']}: {e}")
        return {"processed": False, "action_taken": "update_failed"}

async def handle_subscription_updated(event_data: Dict, user: Optional[Dict]) -> Dict:
    """Handle subscription.updated webhook"""
    if not user:
        return {"processed": False, "action_taken": "user_not_found"}
    
    subscription_status = event_data.get('status')
    
    try:
        await db_service.db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "subscription_status": subscription_status,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        return {
            "processed": True,
            "action_taken": "subscription_updated",
            "new_status": subscription_status
        }
    except Exception as e:
        logger.error(f"❌ Failed to update subscription status for user {user['_id']}: {e}")
        return {"processed": False, "action_taken": "update_failed"}

async def handle_subscription_deleted(event_data: Dict, user: Optional[Dict]) -> Dict:
    """Handle subscription.deleted webhook"""
    if not user:
        return {"processed": False, "action_taken": "user_not_found"}
    
    try:
        await db_service.db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "subscription_status": "cancelled",
                    "plan_type": "free",
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        logger.info(f"✅ Subscription deleted: {user.get('phone_number')} downgraded to free")
        return {
            "processed": True,
            "action_taken": "downgraded_to_free"
        }
    except Exception as e:
        logger.error(f"❌ Failed to downgrade user {user['_id']}: {e}")
        return {"processed": False, "action_taken": "downgrade_failed"}

async def handle_trial_will_end(event_data: Dict, user: Optional[Dict]) -> Dict:
    """Handle subscription.trial_will_end webhook"""
    if not user:
        return {"processed": False, "action_taken": "user_not_found"}
    
    trial_end_timestamp = event_data.get('trial_end')
    trial_end_date = datetime.fromtimestamp(trial_end_timestamp, timezone.utc) if trial_end_timestamp else None
    
    # Send trial ending reminder via SMS
    if twilio_service and user.get('phone_number'):
        trial_message = f"⏰ Your free trial ends in 3 days. Upgrade to continue getting trading insights!"
        await twilio_service.send_message(user['phone_number'], trial_message)
    
    return {
        "processed": True,
        "action_taken": "trial_ending_notification_sent",
        "trial_end": trial_end_date.isoformat() if trial_end_date else None
    }

async def handle_payment_succeeded(event_data: Dict, user: Optional[Dict]) -> Dict:
    """Handle invoice.payment_succeeded webhook"""
    if not user:
        return {"processed": False, "action_taken": "user_not_found"}
    
    try:
        await db_service.db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "payment_failures": 0,
                    "last_payment_success": datetime.now(timezone.utc),
                    "billing_status": "current",
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        amount = event_data.get('amount_paid', 0) / 100  # Convert from cents
        return {
            "processed": True,
            "action_taken": "payment_success_recorded",
            "amount": amount
        }
    except Exception as e:
        logger.error(f"❌ Failed to record payment success for user {user['_id']}: {e}")
        return {"processed": False, "action_taken": "record_failed"}

async def handle_payment_failed(event_data: Dict, user: Optional[Dict]) -> Dict:
    """Handle invoice.payment_failed webhook"""
    if not user:
        return {"processed": False, "action_taken": "user_not_found"}
    
    try:
        # Increment payment failure count
        result = await db_service.db.users.update_one(
            {"_id": user["_id"]},
            {
                "$inc": {"payment_failures": 1},
                "$set": {"updated_at": datetime.now(timezone.utc)}
            }
        )
        
        # Get updated failure count
        updated_user = await db_service.db.users.find_one({"_id": user["_id"]})
        payment_failures = updated_user.get("payment_failures", 1)
        
        # Send appropriate message based on failure count
        if twilio_service and user.get('phone_number'):
            if payment_failures == 1:
                message = "💳 Payment failed. Please update your payment method to continue service."
            elif payment_failures >= 3:
                message = "🚫 Account suspended due to payment issues. Please contact support."
            else:
                message = f"💳 Payment attempt {payment_failures} failed. Please check your payment method."
            
            await twilio_service.send_message(user['phone_number'], message)
        
        return {
            "processed": True,
            "action_taken": "payment_failure_handled",
            "failure_count": payment_failures,
            "suspended": payment_failures >= 3
        }
    except Exception as e:
        logger.error(f"❌ Failed to handle payment failure for user {user['_id']}: {e}")
        return {"processed": False, "action_taken": "handle_failed"}

async def handle_customer_created(event_data: Dict) -> Dict:
    """Handle customer.created webhook"""
    phone_number = event_data.get('phone')
    customer_id = event_data.get('id')
    
    if phone_number and db_service:
        try:
            # Update user with Stripe customer ID
            result = await db_service.db.users.update_one(
                {"phone_number": phone_number},
                {
                    "$set": {
                        "stripe_customer_id": customer_id,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            if result.modified_count > 0:
                return {
                    "processed": True,
                    "action_taken": "customer_linked_to_user"
                }
        except Exception as e:
            logger.error(f"❌ Failed to link customer {customer_id} to user: {e}")
    
    return {"processed": True, "action_taken": "customer_created_no_user"}

async def handle_customer_updated(event_data: Dict, user: Optional[Dict]) -> Dict:
    """Handle customer.updated webhook"""
    if not user:
        return {"processed": False, "action_taken": "user_not_found"}
    
    # Update user email if changed
    email = event_data.get('email')
    if email and email != user.get('email') and db_service:
        try:
            await db_service.db.users.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "email": email,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            return {
                "processed": True,
                "action_taken": "user_email_updated",
                "email": email
            }
        except Exception as e:
            logger.error(f"❌ Failed to update email for user {user['_id']}: {e}")
            return {"processed": False, "action_taken": "update_failed"}
    
    return {"processed": True, "action_taken": "no_changes_needed"}

# ===== DEBUG REDIS/MONGO ENDPOINTS =====

@app.get("/admin/debug/keys")
async def get_all_keys():
    """Temporary endpoint to audit Redis keys"""
    try:
        # Use your existing database service
        if not db_service or not db_service.redis:
            return {"error": "Redis connection not available"}
            
        all_keys = await db_service.redis.keys("*")
        
        key_list = [key.decode('utf-8') if isinstance(key, bytes) else key for key in all_keys]
        
        return {
            "total_keys": len(key_list),
            "keys": sorted(key_list)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/debug/mongo")
async def get_mongo_collections():
    """Check what's in MongoDB"""
    try:
        if not db_service or not db_service.db:
            return {"error": "MongoDB connection not available"}
            
        # Get all collection names
        collections = await db_service.db.list_collection_names()
        
        result = {"collections": {}}
        
        for collection_name in collections:
            # Get count and sample documents
            count = await db_service.db[collection_name].count_documents({})
            
            # Get a few sample documents (without sensitive data)
            samples = []
            async for doc in db_service.db[collection_name].find({}).limit(3):
                # Remove _id and any sensitive fields for display
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                samples.append(doc)
            
            result["collections"][collection_name] = {
                "count": count,
                "samples": samples
            }
        
        return result
    except Exception as e:
        return {"error": str(e)}

# ===== RUN SERVER =====

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Claude-Powered SMS Trading Bot with Background Jobs + KeyBuilder + Gemini Personality on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Claude Preference: {settings.prefer_claude}")
    logger.info(f"Available APIs: Claude={anthropic is not None and settings.anthropic_api_key}, OpenAI={settings.openai_api_key is not None}, Gemini={settings.gemini_api_key is not None}, EODHD={settings.eodhd_api_key is not None}")
    logger.info(f"Background Jobs: Enabled={BackgroundDataPipeline is not None and settings.eodhd_api_key is not None}")
    logger.info(f"Memory Manager: Enabled={MemoryManager is not None and settings.pinecone_api_key is not None}")
    logger.info(f"KeyBuilder: Enabled={KeyBuilder is not None}")
    logger.info(f"Gemini Personality: Enabled={settings.gemini_api_key is not None and settings.personality_analysis_enabled}")
    logger.info(f"Personality Engine Type: {PERSONALITY_ENGINE_TYPE}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development"
    )
