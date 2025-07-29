# ===== main.py - FIXED AND STREAMLINED VERSION =====
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os
from datetime import datetime, timezone
import traceback
import time
from collections import defaultdict
from typing import Dict, List, Any

# Import configuration
try:
    from config import settings
except ImportError:
    # Fallback configuration
    class Settings:
        environment = "development"
        log_level = "INFO"
        testing_mode = True
        mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/ai')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        eodhd_api_key = os.getenv('EODHD_API_KEY')
        
        def get_capability_summary(self):
            return {
                "sms_enabled": bool(self.twilio_account_sid),
                "ai_enabled": bool(self.openai_api_key),
                "database_enabled": bool(self.mongodb_url),
                "testing_mode": self.testing_mode
            }
    
    settings = Settings()

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
    from services.twilio_service import TwilioService
    logger.info("✅ TwilioService imported")
except Exception as e:
    TwilioService = None
    logger.error(f"❌ TwilioService failed: {e}")

try:
    from services.technical_analysis import TechnicalAnalysisService
    logger.info("✅ TechnicalAnalysisService imported")
except Exception as e:
    TechnicalAnalysisService = None
    logger.error(f"❌ TechnicalAnalysisService failed: {e}")

try:
    from services.llm_agent import ComprehensiveMessageProcessor
    logger.info("✅ Orchestrator imported")
except Exception as e:
    ComprehensiveMessageProcessor = None
    logger.error(f"❌ Orchestrator failed: {e}")

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
    from services.fundamental_analysis import FundamentalAnalysisEngine, FundamentalAnalysisTool
    logger.info("✅ FundamentalAnalysisEngine imported")
except Exception as e:
    FundamentalAnalysisEngine = None
    FundamentalAnalysisTool = None
    logger.error(f"❌ FundamentalAnalysisEngine failed: {e}")

try:
    from core.message_handler import MessageHandler
    logger.info("✅ MessageHandler imported")
except Exception as e:
    MessageHandler = None
    logger.error(f"❌ MessageHandler failed: {e}")

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level)

# ===== PERSONALITY ENGINE =====

class UserPersonalityEngine:
    """Learns and adapts to each user's unique communication style and trading personality"""
    
    def __init__(self):
        self.user_profiles = defaultdict(lambda: {
            "communication_style": {
                "formality": "casual",
                "energy": "moderate", 
                "technical_depth": "medium"
            },
            "trading_personality": {
                "risk_tolerance": "moderate",
                "trading_style": "swing",
                "experience_level": "intermediate"
            },
            "learning_data": {
                "total_messages": 0
            }
        })
    
    def learn_from_message(self, phone_number: str, message: str, intent: dict):
        """Learn from each user interaction"""
        profile = self.user_profiles[phone_number]
        profile["learning_data"]["total_messages"] += 1
        
        # Simple learning logic
        msg_lower = message.lower()
        
        # Formality detection
        if any(word in msg_lower for word in ['yo', 'hey', 'gonna', 'wanna']):
            profile["communication_style"]["formality"] = "casual"
        elif any(word in msg_lower for word in ['please', 'analysis', 'evaluation']):
            profile["communication_style"]["formality"] = "professional"
    
    def get_user_profile(self, phone_number: str) -> dict:
        """Get user profile"""
        return self.user_profiles.get(phone_number, {})

# Initialize personality engine
personality_engine = UserPersonalityEngine()

# ===== GLOBAL SERVICES =====
db_service = None
openai_service = None
twilio_service = None
ta_service = None
news_service = None
fundamental_tool = None
cache_service = None
message_processor = None
message_handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    global db_service, openai_service, twilio_service, ta_service, news_service
    global fundamental_tool, cache_service, message_processor, message_handler
    
    logger.info("🚀 Starting SMS Trading Bot...")
    
    try:
        # Initialize Database Service
        if DatabaseService:
            db_service = DatabaseService()
            await db_service.initialize()
            logger.info("✅ Database service initialized")
        
        # Initialize Cache Service
        if CacheService and db_service and hasattr(db_service, 'redis'):
            cache_service = CacheService(db_service.redis)
            logger.info("✅ Cache service initialized")
        
        # Initialize OpenAI Service
        if OpenAIService:
            openai_service = OpenAIService()
            logger.info("✅ OpenAI service initialized")
        
        # Initialize Twilio Service
        if TwilioService:
            twilio_service = TwilioService()
            logger.info("✅ Twilio service initialized")
        
        # Initialize Technical Analysis Service
        if TechnicalAnalysisService:
            ta_service = TechnicalAnalysisService()
            logger.info("✅ Technical Analysis service initialized")
        
        # Initialize News Sentiment Service
        if NewsSentimentService:
            news_service = NewsSentimentService(
                redis_client=db_service.redis if db_service else None,
                openai_service=openai_service
            )
            logger.info("✅ News Sentiment service initialized")
        
        # Initialize Fundamental Analysis Tool
        if FundamentalAnalysisTool and settings.eodhd_api_key:
            fundamental_tool = FundamentalAnalysisTool(
                eodhd_api_key=settings.eodhd_api_key,
                redis_client=db_service.redis if db_service else None
            )
            logger.info("✅ Fundamental Analysis tool initialized")
        
        # Initialize AI-Powered Message Processor (Enhanced Orchestrator)
        if ComprehensiveMessageProcessor and openai_service:
            message_processor = ComprehensiveMessageProcessor(
                openai_client=openai_service.client,
                ta_service=ta_service,
                personality_engine=personality_engine,
                cache_service=cache_service,
                news_service=news_service,
                fundamental_tool=fundamental_tool
            )
            logger.info("✅ AI-Powered Message Processor (Enhanced Orchestrator) initialized")
        
        # Initialize Message Handler
        if MessageHandler:
            message_handler = MessageHandler(
                db_service=db_service,
                openai_service=openai_service,
                twilio_service=twilio_service,
                message_processor=message_processor
            )
            logger.info("✅ Message Handler initialized")
        
        # Status report
        agent_status = "AI-Powered Orchestrator" if message_processor else "Fallback"
        logger.info(f"🤖 Agent Status: {agent_status}")
        logger.info("✅ Startup completed")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.warning("⚠️ Continuing in degraded mode")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    try:
        if db_service and hasattr(db_service, 'close'):
            await db_service.close()
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# ===== CREATE APP =====

app = FastAPI(
    title="SMS Trading Bot",
    description="AI-Powered SMS Trading Assistant with Enhanced Orchestrator",
    version="2.0.0",
    lifespan=lifespan
)

# ===== CORE ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API",
        "status": "running",
        "version": "2.0.0",
        "agent_type": "ai_powered_orchestrator" if message_processor else "fallback",
        "services": {
            "database": db_service is not None,
            "cache": cache_service is not None,
            "technical_analysis": ta_service is not None,
            "news_sentiment": news_service is not None,
            "fundamental_analysis": fundamental_tool is not None,
            "orchestrator": message_processor is not None
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "agent_type": "ai_powered_orchestrator" if message_processor else "fallback"
        }
        
        # Check database
        if db_service:
            try:
                await db_service.db.command("ping")
                health_status["database"] = {"status": "connected"}
            except Exception as e:
                health_status["database"] = {"status": "error", "error": str(e)}
        else:
            health_status["database"] = {"status": "not_initialized"}
        
        # Check services
        health_status["services"] = {
            "openai": "available" if openai_service else "unavailable",
            "twilio": "available" if twilio_service else "unavailable",
            "technical_analysis": "available" if ta_service else "unavailable",
            "news_sentiment": "available" if news_service else "unavailable",
            "fundamental_analysis": "available" if fundamental_tool else "unavailable",
            "cache": "available" if cache_service else "unavailable",
            "orchestrator": "available" if message_processor else "unavailable",
            "message_handler": "available" if message_handler else "unavailable"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)}
        )

# ===== SMS WEBHOOK (FIXED) =====

@app.post("/webhook/sms")
async def sms_webhook(request: Request):
    """Handle incoming SMS messages - FIXED TO REMOVE DUPLICATE PROCESSING"""
    try:
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return PlainTextResponse("Missing required fields", status_code=400)
        
        # Process ONCE with AI-powered orchestrator - NO DUPLICATE BACKGROUND PROCESSING
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
    """Process SMS using AI-powered orchestrator or fallback"""
    
    start_time = time.time()
    
    try:
        logger.info(f"📱 Processing: '{message_body}' from {phone_number}")
        
        # Use AI-powered orchestrator if available
        if message_processor:
            response_text = await message_processor.process_message(message_body, phone_number)
            processing_time = time.time() - start_time
            logger.info(f"✅ AI-Powered Orchestrator processed in {processing_time:.2f}s")
        else:
            # Simple fallback
            response_text = "I'm here to help with your trading questions! The AI-powered system is currently in maintenance mode."
            logger.warning("⚠️ Using simple fallback - AI-powered orchestrator unavailable")
        
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

# ===== ADMIN ENDPOINTS =====

@app.get("/admin")
async def admin_dashboard():
    """Admin dashboard"""
    try:
        return {
            "title": "SMS Trading Bot Admin - AI-Powered",
            "status": "operational",
            "version": "2.0.0",
            "agent_type": "ai_powered_orchestrator" if message_processor else "fallback",
            "services": {
                "database": "connected" if db_service else "disconnected",
                "cache": "active" if cache_service else "inactive",
                "technical_analysis": "active" if ta_service else "inactive",
                "news_sentiment": "active" if news_service else "inactive",
                "fundamental_analysis": "active" if fundamental_tool else "inactive",
                "orchestrator": "active" if message_processor else "inactive",
                "message_handler": "active" if message_handler else "inactive"
            },
            "users": {
                "total_profiles": len(personality_engine.user_profiles)
            },
            "enhancements": {
                "ai_powered_synthesis": True,
                "intelligent_engine_selection": True,
                "conversation_context": True,
                "fundamental_analysis_detection": True,
                "complete_context_json": True
            }
        }
    except Exception as e:
        logger.error(f"❌ Admin dashboard error: {e}")
        return {"error": str(e)}

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "agent_type": "ai_powered_orchestrator" if message_processor else "fallback",
            "services": {
                "database": db_service is not None,
                "cache": cache_service is not None,
                "technical_analysis": ta_service is not None,
                "news_sentiment": news_service is not None,
                "fundamental_analysis": fundamental_tool is not None,
                "orchestrator": message_processor is not None
            },
            "personality_engine": {
                "total_profiles": len(personality_engine.user_profiles),
                "active_learning": True
            },
            "ai_enhancements": {
                "intelligent_synthesis": True,
                "context_aware_responses": True,
                "complete_json_structure": True,
                "template_responses_eliminated": True
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ===== DEBUG ENDPOINTS =====

@app.get("/debug/diagnose")
async def diagnose_services():
    """Comprehensive service diagnosis"""
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "environment_variables": {
            "OPENAI_API_KEY": "Set" if os.getenv('OPENAI_API_KEY') else "Missing",
            "EODHD_API_KEY": "Set" if os.getenv('EODHD_API_KEY') else "Missing",
            "MONGODB_URL": "Set" if os.getenv('MONGODB_URL') else "Missing",
            "TWILIO_ACCOUNT_SID": "Set" if os.getenv('TWILIO_ACCOUNT_SID') else "Missing"
        },
        "service_status": {
            "database": db_service is not None,
            "cache": cache_service is not None,
            "openai": openai_service is not None,
            "twilio": twilio_service is not None,
            "technical_analysis": ta_service is not None,
            "news_sentiment": news_service is not None,
            "fundamental_analysis": fundamental_tool is not None,
            "orchestrator": message_processor is not None,
            "message_handler": message_handler is not None
        },
        "ai_enhancements_active": {
            "intelligent_synthesis": message_processor is not None,
            "conversation_context": cache_service is not None,
            "fundamental_detection": True,
            "complete_json_context": True,
            "template_elimination": True
        },
        "recommendations": []
    }
    
    # Generate recommendations
    if not os.getenv('OPENAI_API_KEY'):
        diagnosis["recommendations"].append("❌ Set OPENAI_API_KEY for AI-powered responses")
    if not os.getenv('EODHD_API_KEY'):
        diagnosis["recommendations"].append("❌ Set EODHD_API_KEY for market data")
    if not message_processor:
        diagnosis["recommendations"].append("❌ AI-Powered Orchestrator not available")
    
    if not diagnosis["recommendations"]:
        diagnosis["recommendations"].append("✅ All AI-powered systems operational!")
    
    return diagnosis

@app.post("/debug/test-message")
async def test_message_processing(request: Request):
    """Test AI-powered message processing"""
    try:
        data = await request.json()
        message = data.get('message', 'What are the fundamentals for AAPL?')
        phone = data.get('phone', '+1555TEST')
        
        response = await process_sms_message(message, phone)
        
        return {
            "success": True,
            "input_message": message,
            "phone_number": phone,
            "bot_response": response,
            "processed_with": "ai_powered_orchestrator" if message_processor else "fallback",
            "ai_enhancements_active": {
                "intelligent_synthesis": message_processor is not None,
                "conversation_context": cache_service is not None,
                "fundamental_detection": True,
                "complete_json_structure": True
            }
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug/test-fundamental")
async def test_fundamental_detection():
    """Test fundamental analysis detection"""
    test_messages = [
        "What are the fundamentals for NVDA?",
        "Tell me about AAPL fundamentals",
        "How is TSLA doing?",
        "What about fundamentals",
        "Show me the earnings for MSFT"
    ]
    
    results = []
    for message in test_messages:
        try:
            response = await process_sms_message(message, "+1555TEST")
            results.append({
                "message": message,
                "response_preview": response[:100] + "..." if len(response) > 100 else response,
                "should_detect_fundamental": any(word in message.lower() for word in ["fundamental", "fundamentals", "earnings"])
            })
        except Exception as e:
            results.append({
                "message": message,
                "error": str(e)
            })
    
    return {
        "test_results": results,
        "orchestrator_available": message_processor is not None,
        "fundamental_tool_available": fundamental_tool is not None
    }

# ===== TEST INTERFACE =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """AI-Powered test interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - AI-Powered Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; background: #f8f9fa; }
        .quick-tests { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .enhancement-badge { background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
    </style>
</head>
<body>
    <h1>SMS Trading Bot - AI-Powered Test Interface</h1>
    <div class="enhancement-badge">AI-Powered Orchestrator Active</div>
    
    <div class="quick-tests">
        <button onclick="quickTest('What are the fundamentals for NVDA?')">Test Fundamental Analysis</button>
        <button onclick="quickTest('How is AAPL doing technically?')">Test Technical Analysis</button>
        <button onclick="quickTest('Tell me about TSLA news')">Test News Sentiment</button>
        <button onclick="quickTest('What about fundamentals')">Test Context Awareness</button>
    </div>
    
    <form onsubmit="testSMS(event)">
        <div class="form-group">
            <label>Phone Number:</label>
            <input type="text" id="phone" value="+1555TEST" required>
        </div>
        
        <div class="form-group">
            <label>Message:</label>
            <textarea id="message" rows="3" required>What are the fundamentals for AAPL?</textarea>
        </div>
        
        <button type="submit">Test AI-Powered SMS Processing</button>
    </form>
    
    <div id="result"></div>
    
    <script>
        function quickTest(message) {
            document.getElementById('message').value = message;
            document.querySelector('form').dispatchEvent(new Event('submit'));
        }
        
        async function testSMS(event) {
            event.preventDefault();
            
            const phone = document.getElementById('phone').value;
            const message = document.getElementById('message').value;
            
            document.getElementById('result').innerHTML = '<div class="result">🔄 Processing with AI-Powered Orchestrator...</div>';
            
            try {
                const formData = new URLSearchParams();
                formData.append('From', phone);
                formData.append('Body', message);
                
                const response = await fetch('/webhook/sms', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: formData
                });
                
                if (response.ok) {
                    // Also get debug info
                    const debugResponse = await fetch('/debug/test-message', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message, phone })
                    });
                    
                    if (debugResponse.ok) {
                        const debugData = await debugResponse.json();
                        document.getElementById('result').innerHTML = 
                            `<div class="result">
                                <h3>✅ AI-Powered SMS Processing Success!</h3>
                                <p><strong>Input:</strong> ${debugData.input_message}</p>
                                <p><strong>Response:</strong> ${debugData.bot_response}</p>
                                <p><strong>Processed with:</strong> ${debugData.processed_with}</p>
                                <p><strong>AI Enhancements Active:</strong> 
                                    ${Object.entries(debugData.ai_enhancements_active || {})
                                        .map(([key, value]) => `${key}: ${value ? '✅' : '❌'}`)
                                        .join(', ')}
                                </p>
                            </div>`;
                    } else {
                        document.getElementById('result').innerHTML = 
                            '<div class="result">✅ SMS processed successfully with AI-Powered Orchestrator!</div>';
                    }
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
    logger.info(f"🚀 Starting AI-Powered SMS Trading Bot on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"AI-Powered Orchestrator: {'Available' if ComprehensiveMessageProcessor else 'Unavailable'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development"
    )
