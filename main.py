# ===== main.py - CLAUDE-POWERED CONVERSATION-AWARE AGENT =====
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
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        eodhd_api_key = os.getenv('EODHD_API_KEY')
        prefer_claude = os.getenv('PREFER_CLAUDE', 'true').lower() == 'true'
    
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
    from services.technical_analysis import TechnicalAnalysisService
    logger.info("✅ TechnicalAnalysisService imported")
except Exception as e:
    TechnicalAnalysisService = None
    logger.error(f"❌ TechnicalAnalysisService failed: {e}")

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown with Claude support"""
    global db_service, openai_service, anthropic_client, twilio_service, ta_service, news_service
    global fundamental_tool, cache_service, openai_agent, claude_agent, hybrid_agent, active_agent
    
    logger.info("🚀 Starting SMS Trading Bot with Claude-Powered Agent...")
    
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
        if settings.prefer_claude and claude_agent:
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
    description="Claude-Powered SMS Trading Assistant",
    version="4.0.0",
    lifespan=lifespan
)

# ===== CORE ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API",
        "status": "running",
        "version": "4.0.0",
        "agent_type": get_agent_type(),
        "architecture": "claude_powered",
        "services": {
            "database": db_service is not None,
            "cache": cache_service is not None,
            "technical_analysis": ta_service is not None,
            "news_sentiment": news_service is not None,
            "fundamental_analysis": fundamental_tool is not None,
            "openai_agent": openai_agent is not None,
            "claude_agent": claude_agent is not None,
            "hybrid_agent": hybrid_agent is not None,
            "active_agent": active_agent is not None
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
    """Health check endpoint with Claude status"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "4.0.0",
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

# ===== ADMIN ENDPOINTS =====

@app.get("/admin")
async def admin_dashboard():
    """Admin dashboard with Claude status"""
    try:
        return {
            "title": "SMS Trading Bot Admin - Claude-Powered",
            "status": "operational",
            "version": "4.0.0",
            "architecture": "claude_powered",
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
                "superior_analysis": claude_agent is not None
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
            "features_active": {
                "claude_reasoning": claude_agent is not None,
                "conversation_awareness": active_agent is not None,
                "context_retention": cache_service is not None,
                "smart_tool_calling": True,
                "superior_analysis": claude_agent is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/debug/compare-agents")
async def compare_agents(request: Request):
    """Compare responses from Claude vs OpenAI"""
    try:
        data = await request.json()
        message = data.get('message', 'Would you advise buying PLTR right now?')
        phone = data.get('phone', '+1555COMPARE')
        
        results = {}
        
        # Test Claude
        if claude_agent:
            try:
                claude_response = await claude_agent.process_message(message, phone)
                results['claude'] = {
                    "response": claude_response,
                    "status": "success"
                }
            except Exception as e:
                results['claude'] = {
                    "response": None,
                    "status": f"error: {str(e)}"
                }
        else:
            results['claude'] = {
                "response": None,
                "status": "not_available"
            }
        
        # Test OpenAI
        if openai_agent:
            try:
                openai_response = await openai_agent.process_message(message, phone)
                results['openai'] = {
                    "response": openai_response,
                    "status": "success"
                }
            except Exception as e:
                results['openai'] = {
                    "response": None,
                    "status": f"error: {str(e)}"
                }
        else:
            results['openai'] = {
                "response": None,
                "status": "not_available"
            }
        
        return {
            "success": True,
            "input_message": message,
            "comparison": results,
            "recommendation": "Claude is expected to provide more natural conversational advice" if claude_agent else "Only OpenAI available"
        }
        
    except Exception as e:
        logger.error(f"Agent comparison failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug/diagnose")
async def diagnose_services():
    """Service diagnosis with Claude status"""
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "environment_variables": {
            "OPENAI_API_KEY": "Set" if os.getenv('OPENAI_API_KEY') else "Missing",
            "ANTHROPIC_API_KEY": "Set" if os.getenv('ANTHROPIC_API_KEY') else "Missing",
            "EODHD_API_KEY": "Set" if os.getenv('EODHD_API_KEY') else "Missing",
            "MONGODB_URL": "Set" if os.getenv('MONGODB_URL') else "Missing",
            "TWILIO_ACCOUNT_SID": "Set" if os.getenv('TWILIO_ACCOUNT_SID') else "Missing",
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
            "hybrid_agent": hybrid_agent is not None
        },
        "architecture": "claude_powered_with_fallback",
        "active_agent": get_agent_type(),
        "recommendations": []
    }
    
    # Generate recommendations
    if not os.getenv('ANTHROPIC_API_KEY'):
        diagnosis["recommendations"].append("❌ Set ANTHROPIC_API_KEY for Claude-powered responses")
    if not os.getenv('OPENAI_API_KEY'):
        diagnosis["recommendations"].append("⚠️ Set OPENAI_API_KEY for OpenAI fallback")
    if not os.getenv('EODHD_API_KEY'):
        diagnosis["recommendations"].append("❌ Set EODHD_API_KEY for market data")
    if not active_agent:
        diagnosis["recommendations"].append("❌ No agents available - check API keys")
    
    if claude_agent and openai_agent:
        diagnosis["recommendations"].append("✅ Both Claude and OpenAI agents operational - excellent setup!")
    elif claude_agent:
        diagnosis["recommendations"].append("✅ Claude agent operational - superior reasoning active!")
    elif openai_agent:
        diagnosis["recommendations"].append("⚠️ Only OpenAI agent available - consider adding Claude for better responses")
    
    if not diagnosis["recommendations"]:
        diagnosis["recommendations"].append("✅ All systems operational!")
    
    return diagnosis

# ===== CLAUDE TEST INTERFACE =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Test interface with Claude vs OpenAI comparison"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Claude vs OpenAI Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button.claude { background: #8B5CF6; }
        button.compare { background: #10B981; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; background: #f8f9fa; }
        .comparison { display: flex; gap: 20px; }
        .agent-result { flex: 1; padding: 15px; border-radius: 4px; }
        .claude-result { background: #F3F4F6; border-left: 4px solid #8B5CF6; }
        .openai-result { background: #FFF7ED; border-left: 4px solid #F59E0B; }
        .quick-tests { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .badge { background: #8B5CF6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px; }
        .badge.openai { background: #F59E0B; }
        .badge.hybrid { background: #10B981; }
    </style>
</head>
<body>
    <h1>SMS Trading Bot - Claude vs OpenAI Test Interface</h1>
    <div>
        <span class="badge">Claude-Powered</span>
        <span class="badge openai">OpenAI Fallback</span>
        <span class="badge hybrid">Hybrid Available</span>
    </div>
    
    <div class="quick-tests">
        <button onclick="quickTest('Would you advise buying PLTR right now?')">Test Advice Question</button>
        <button onclick="quickTest('What are the fundamentals for NVDA?')">Test Data Question</button>
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
        function quickTest(message) {
            document.getElementById('message').value = message;
            document.querySelector('form').dispatchEvent(new Event('submit'));
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
                            <p><strong>Available Agents:</strong> 
                                Claude: ${data.available_agents.claude ? '✅' : '❌'}, 
                                OpenAI: ${data.available_agents.openai ? '✅' : '❌'}, 
                                Hybrid: ${data.available_agents.hybrid ? '✅' : '❌'}
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
    logger.info(f"🚀 Starting Claude-Powered SMS Trading Bot on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Claude Preference: {settings.prefer_claude}")
    logger.info(f"Available: Claude={anthropic is not None and settings.anthropic_api_key}, OpenAI={settings.openai_api_key is not None}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development"
    )
