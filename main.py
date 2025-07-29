# ===== main.py - SIMPLIFIED FOR CONVERSATION-AWARE AGENT =====
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os
from datetime import datetime, timezone
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
    logger.info("✅ Conversation-Aware Agent imported")
except Exception as e:
    ComprehensiveMessageProcessor = None
    logger.error(f"❌ Conversation-Aware Agent failed: {e}")

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
    from services.fundamental_analysis import FundamentalAnalysisTool
    logger.info("✅ FundamentalAnalysisTool imported")
except Exception as e:
    FundamentalAnalysisTool = None
    logger.error(f"❌ FundamentalAnalysisTool failed: {e}")

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
twilio_service = None
ta_service = None
news_service = None
fundamental_tool = None
cache_service = None
conversation_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified application startup and shutdown"""
    global db_service, openai_service, twilio_service, ta_service, news_service
    global fundamental_tool, cache_service, conversation_agent
    
    logger.info("🚀 Starting SMS Trading Bot with Conversation-Aware Agent...")
    
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
        
        if TwilioService:
            twilio_service = TwilioService()
            logger.info("✅ Twilio service initialized")
        
        if TechnicalAnalysisService:
            ta_service = TechnicalAnalysisService()
            logger.info("✅ Technical Analysis service initialized")
        
        if NewsSentimentService:
            news_service = NewsSentimentService(
                redis_client=db_service.redis if db_service else None,
                openai_service=openai_service
            )
            logger.info("✅ News Sentiment service initialized")
        
        if FundamentalAnalysisTool and settings.eodhd_api_key:
            fundamental_tool = FundamentalAnalysisTool(
                eodhd_api_key=settings.eodhd_api_key,
                redis_client=db_service.redis if db_service else None
            )
            logger.info("✅ Fundamental Analysis tool initialized")
        
        # Initialize the simplified conversation-aware message processor
        if ComprehensiveMessageProcessor and openai_service:
            conversation_agent = ComprehensiveMessageProcessor(
                openai_client=openai_service.client,
                ta_service=ta_service,
                personality_engine=personality_engine,
                cache_service=cache_service,
                news_service=news_service,
                fundamental_tool=fundamental_tool
            )
            logger.info("✅ Conversation-Aware Agent initialized")
        
        # Status report
        agent_status = "Conversation-Aware Agent" if conversation_agent else "Fallback"
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
    description="Conversation-Aware SMS Trading Assistant",
    version="3.0.0",
    lifespan=lifespan
)

# ===== CORE ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API",
        "status": "running",
        "version": "3.0.0",
        "agent_type": "conversation_aware" if conversation_agent else "fallback",
        "architecture": "simplified",
        "services": {
            "database": db_service is not None,
            "cache": cache_service is not None,
            "technical_analysis": ta_service is not None,
            "news_sentiment": news_service is not None,
            "fundamental_analysis": fundamental_tool is not None,
            "conversation_agent": conversation_agent is not None
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "3.0.0",
            "agent_type": "conversation_aware" if conversation_agent else "fallback",
            "services": {
                "openai": "available" if openai_service else "unavailable",
                "twilio": "available" if twilio_service else "unavailable",
                "technical_analysis": "available" if ta_service else "unavailable",
                "news_sentiment": "available" if news_service else "unavailable",
                "fundamental_analysis": "available" if fundamental_tool else "unavailable",
                "cache": "available" if cache_service else "unavailable",
                "conversation_agent": "available" if conversation_agent else "unavailable"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(status_code=503, content={"status": "error", "error": str(e)})

# ===== SMS WEBHOOK (SIMPLIFIED) =====

@app.post("/webhook/sms")
async def sms_webhook(request: Request):
    """Handle incoming SMS messages with conversation-aware processing"""
    try:
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        
        if not from_number or not message_body:
            return PlainTextResponse("Missing required fields", status_code=400)
        
        # Process with conversation-aware agent
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
    """Simplified SMS processing with conversation-aware agent"""
    
    start_time = time.time()
    
    try:
        logger.info(f"📱 Processing: '{message_body}' from {phone_number}")
        
        # Use conversation-aware agent if available
        if conversation_agent:
            response_text = await conversation_agent.process_message(message_body, phone_number)
            processing_time = time.time() - start_time
            logger.info(f"✅ Conversation-Aware Agent processed in {processing_time:.2f}s")
        else:
            # Simple fallback
            response_text = "Trading assistant temporarily unavailable. Please try again in a moment."
            logger.warning("⚠️ Using fallback - conversation-aware agent unavailable")
        
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
    """Simplified admin dashboard"""
    try:
        return {
            "title": "SMS Trading Bot Admin - Conversation-Aware",
            "status": "operational",
            "version": "3.0.0",
            "architecture": "simplified",
            "agent_type": "conversation_aware" if conversation_agent else "fallback",
            "services": {
                "database": "connected" if db_service else "disconnected",
                "cache": "active" if cache_service else "inactive",
                "technical_analysis": "active" if ta_service else "inactive",
                "news_sentiment": "active" if news_service else "inactive",
                "fundamental_analysis": "active" if fundamental_tool else "inactive",
                "conversation_agent": "active" if conversation_agent else "inactive"
            },
            "users": {
                "total_profiles": len(personality_engine.user_profiles)
            },
            "features": {
                "conversation_awareness": True,
                "smart_tool_calling": True,
                "professional_responses": True,
                "context_retention": True,
                "simplified_architecture": True
            }
        }
    except Exception as e:
        logger.error(f"❌ Admin dashboard error: {e}")
        return {"error": str(e)}

# ===== DEBUG ENDPOINTS =====

@app.post("/debug/test-message")
async def test_message_processing(request: Request):
    """Test conversation-aware message processing"""
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
            "processed_with": "conversation_aware_agent" if conversation_agent else "fallback",
            "features_active": {
                "conversation_awareness": conversation_agent is not None,
                "context_retention": cache_service is not None,
                "smart_tool_calling": True,
                "professional_tone": True
            }
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug/diagnose")
async def diagnose_services():
    """Simplified service diagnosis"""
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
            "conversation_agent": conversation_agent is not None
        },
        "architecture": "simplified_conversation_aware",
        "recommendations": []
    }
    
    # Generate recommendations
    if not os.getenv('OPENAI_API_KEY'):
        diagnosis["recommendations"].append("❌ Set OPENAI_API_KEY for conversation-aware responses")
    if not os.getenv('EODHD_API_KEY'):
        diagnosis["recommendations"].append("❌ Set EODHD_API_KEY for market data")
    if not conversation_agent:
        diagnosis["recommendations"].append("❌ Conversation-Aware Agent not available")
    
    if not diagnosis["recommendations"]:
        diagnosis["recommendations"].append("✅ All conversation-aware systems operational!")
    
    return diagnosis

# ===== SIMPLIFIED TEST INTERFACE =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Simplified test interface for conversation-aware agent"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Conversation-Aware Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; background: #f8f9fa; }
        .quick-tests { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .badge { background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
    </style>
</head>
<body>
    <h1>SMS Trading Bot - Conversation-Aware Test Interface</h1>
    <div class="badge">Conversation-Aware Agent</div>
    
    <div class="quick-tests">
        <button onclick="quickTest('What are the fundamentals for NVDA?')">Test Fundamentals</button>
        <button onclick="quickTest('How is AAPL doing technically?')">Test Technical</button>
        <button onclick="quickTest('Tell me about TSLA news')">Test News</button>
        <button onclick="quickTest('What about the fundamentals?')">Test Context</button>
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
        
        <button type="submit">Test Conversation-Aware Agent</button>
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
            
            document.getElementById('result').innerHTML = '<div class="result">🔄 Processing with Conversation-Aware Agent...</div>';
            
            try {
                const debugResponse = await fetch('/debug/test-message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, phone })
                });
                
                if (debugResponse.ok) {
                    const debugData = await debugResponse.json();
                    document.getElementById('result').innerHTML = 
                        `<div class="result">
                            <h3>✅ Conversation-Aware Processing Success!</h3>
                            <p><strong>Input:</strong> ${debugData.input_message}</p>
                            <p><strong>Response:</strong> ${debugData.bot_response}</p>
                            <p><strong>Architecture:</strong> ${debugData.processed_with}</p>
                            <p><strong>Features Active:</strong> 
                                ${Object.entries(debugData.features_active || {})
                                    .map(([key, value]) => `${key}: ${value ? '✅' : '❌'}`)
                                    .join(', ')}
                            </p>
                        </div>`;
                } else {
                    throw new Error(`HTTP ${debugResponse.status}`);
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
    logger.info(f"🚀 Starting Conversation-Aware SMS Trading Bot on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Conversation Agent: {'Available' if ComprehensiveMessageProcessor else 'Unavailable'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development"
    )
