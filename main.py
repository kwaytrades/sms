# ===== main.py - FIXED FASTAPI VERSION =====
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os
from datetime import datetime, timedelta
import traceback
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any

# Import configuration
from config import settings

# Import services
from services.database import DatabaseService
from services.openai_service import OpenAIService
from services.twilio_service import TwilioService
from services.weekly_scheduler import WeeklyScheduler
from core.message_handler import MessageHandler

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level)

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
        "weekly_limit": settings.free_weekly_limit,
        "price": 0,
        "features": ["Basic market updates", "Stock analysis on demand"]
    },
    "paid": {
        "monthly_limit": settings.paid_monthly_limit,
        "price": 29,
        "features": ["Personalized insights", "Portfolio tracking", "Market analytics"]
    },
    "pro": {
        "unlimited": True,
        "daily_cooloff": settings.pro_daily_cooloff,
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

# Global services
db_service = None
openai_service = None
twilio_service = None
message_handler = None
scheduler_task = None
metrics = MetricsCollector()

# ===== FASTAPI MIDDLEWARE FOR METRICS =====

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_service, openai_service, twilio_service, message_handler, scheduler_task
    
    logger.info("🚀 Starting SMS Trading Bot...")
    
    try:
        # Initialize services
        db_service = DatabaseService()
        await db_service.initialize()
        
        openai_service = OpenAIService()
        twilio_service = TwilioService()
        message_handler = MessageHandler(db_service, openai_service, twilio_service)
        
        # Start weekly scheduler
        scheduler = WeeklyScheduler(db_service, twilio_service)
        scheduler_task = asyncio.create_task(scheduler.start_scheduler())
        logger.info("📅 Weekly scheduler started")
        
        logger.info("✅ SMS Trading Bot started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise in testing mode
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

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

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
            "test_interface": "/test",
            "metrics": "/metrics",
            "user_management": "/admin/users/*",
            "scheduler": "/admin/scheduler/*"
        }
    }

@app.get("/health")
async def health_check():
    try:
        # Get runtime requirements validation
        validation = settings.validate_runtime_requirements()
        
        return {
            "status": "healthy" if validation["ready_for_production"] or settings.testing_mode else "degraded",
            "environment": settings.environment,
            "testing_mode": settings.testing_mode,
            "services": {
                "database": "connected" if db_service and db_service.db else "disconnected",
                "redis": "connected" if db_service and db_service.redis else "disconnected",
                "message_handler": "active" if message_handler else "inactive",
                "weekly_scheduler": "active" if scheduler_task and not scheduler_task.done() else "inactive"
            },
            "capabilities": settings.get_capability_summary(),
            "validation": validation,
            "uptime": metrics.get_metrics()["uptime"]
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
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
        
        # Process message in background
        if message_handler:
            background_tasks.add_task(
                message_handler.process_incoming_message,
                from_number,
                message_body
            )
        else:
            logger.warning("Message handler not available - testing mode")
        
        # Return empty TwiML response
        return PlainTextResponse(
            '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"❌ SMS webhook error: {e}")
        return PlainTextResponse("Internal error", status_code=500)

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

# ===== ADMIN ENDPOINTS =====

@app.get("/admin")
async def admin_dashboard():
    """Comprehensive admin dashboard"""
    try:
        user_stats = {}
        if db_service:
            try:
                user_stats = await db_service.get_user_stats()
            except Exception as e:
                logger.warning(f"Could not get user stats: {e}")
        
        # Get scheduler status
        scheduler_status = {"status": "unknown"}
        if db_service and twilio_service:
            try:
                temp_scheduler = WeeklyScheduler(db_service, twilio_service)
                scheduler_status = {"status": "active" if scheduler_task and not scheduler_task.done() else "inactive"}
            except Exception as e:
                scheduler_status = {"status": "error", "error": str(e)}
        
        # Get metrics
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
            "user_stats": user_stats,
            "scheduler": scheduler_status,
            "metrics": system_metrics,
            "configuration": settings.get_capability_summary(),
            "plan_limits": PLAN_LIMITS,
            "popular_tickers": {
                "count": len(POPULAR_TICKERS),
                "sample": POPULAR_TICKERS[:10]
            }
        }
    except Exception as e:
        logger.error(f"❌ Admin dashboard error: {e}")
        return {"error": str(e)}

# ===== USER MANAGEMENT ENDPOINTS =====

@app.get("/admin/users/{phone_number}")
async def get_user_profile(phone_number: str):
    """Get user profile for admin"""
    if not db_service:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    
    try:
        user = await db_service.get_user_by_phone(phone_number)
        if user:
            return user.to_dict()
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error getting user {phone_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/users/stats")
async def get_user_stats():
    """Get user statistics"""
    if not db_service:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    
    try:
        return await db_service.get_user_stats()
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/users/{phone_number}/subscription")
async def update_user_subscription(phone_number: str, request: Request):
    """Update user subscription"""
    if not db_service:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    
    try:
        plan_data = await request.json()
        success = await db_service.update_subscription(
            phone_number,
            plan_data.get('plan_type'),
            plan_data.get('stripe_customer_id'),
            plan_data.get('stripe_subscription_id')
        )
        
        return {"success": success}
    except Exception as e:
        logger.error(f"Error updating subscription for {phone_number}: {e}")
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
            "system": {
                "version": "1.0.0",
                "environment": settings.environment,
                "testing_mode": settings.testing_mode
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ===== DEBUG ENDPOINTS =====

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
        "popular_tickers_count": len(POPULAR_TICKERS)
    }

@app.post("/debug/analyze-intent")
async def debug_analyze_intent(request: Request):
    """Debug intent analysis"""
    try:
        data = await request.json()
        message = data.get('message', '')
        
        # Simple intent classification
        intent = "general"
        symbols = []
        
        if any(word in message.lower() for word in ['price', 'cost', 'trading']):
            intent = 'price'
        elif any(word in message.lower() for word in ['analyze', 'analysis', 'check']):
            intent = 'analyze'
        elif any(word in message.lower() for word in ['find', 'search', 'recommend']):
            intent = 'screener'
        
        # Extract potential symbols
        import re
        potential_symbols = re.findall(r'\b[A-Z]{1,5}\b', message.upper())
        symbols = [s for s in potential_symbols if s in POPULAR_TICKERS]
        
        return {
            "message": message,
            "intent": intent,
            "symbols": symbols,
            "confidence": 0.8 if symbols else 0.5
        }
    except Exception as e:
        logger.error(f"Intent analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== TEST INTERFACE =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Simplified test interface"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Test Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #0056b3; }
        input, textarea { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
        .result { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px; margin: 10px 0; font-family: monospace; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📱 SMS Trading Bot Test Dashboard</h1>
        
        <div class="card">
            <h3>📊 System Health</h3>
            <button class="btn" onclick="checkHealth()">Health Check</button>
            <button class="btn" onclick="getMetrics()">Get Metrics</button>
            <button class="btn" onclick="debugConfig()">Debug Config</button>
            <div id="health-result" class="result"></div>
        </div>
        
        <div class="card">
            <h3>💬 SMS Testing</h3>
            <input type="text" id="sms-phone" value="+13012466712" placeholder="Phone Number">
            <textarea id="sms-body" rows="3" placeholder="Message">How is AAPL doing?</textarea>
            <button class="btn" onclick="sendSMS()">Send SMS</button>
            <div id="sms-result" class="result"></div>
        </div>
        
        <div class="card">
            <h3>🧠 Intent Analysis</h3>
            <textarea id="intent-message" rows="2" placeholder="Message to analyze">What's the RSI for TSLA?</textarea>
            <button class="btn" onclick="analyzeIntent()">Analyze Intent</button>
            <div id="intent-result" class="result"></div>
        </div>
    </div>

    <script>
        const BASE_URL = window.location.origin;
        
        function showResult(elementId, data) {
            document.getElementById(elementId).textContent = JSON.stringify(data, null, 2);
        }
        
        async function apiCall(endpoint, method = 'GET', body = null) {
            const options = { method, headers: {} };
            if (body) {
                if (typeof body === 'string') {
                    options.headers['Content-Type'] = 'application/x-www-form-urlencoded';
                    options.body = body;
                } else {
                    options.headers['Content-Type'] = 'application/json';
                    options.body = JSON.stringify(body);
                }
            }
            
            const response = await fetch(BASE_URL + endpoint, options);
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('application/json')) {
                return response.json();
            } else {
                return response.text();
            }
        }
        
        async function checkHealth() {
            try {
                const data = await apiCall('/health');
                showResult('health-result', data);
            } catch (error) {
                showResult('health-result', { error: error.message });
            }
        }
        
        async function getMetrics() {
            try {
                const data = await apiCall('/metrics');
                showResult('health-result', data);
            } catch (error) {
                showResult('health-result', { error: error.message });
            }
        }
        
        async function debugConfig() {
            try {
                const data = await apiCall('/debug/config');
                showResult('health-result', data);
            } catch (error) {
                showResult('health-result', { error: error.message });
            }
        }
        
        async function sendSMS() {
            try {
                const phone = document.getElementById('sms-phone').value;
                const body = document.getElementById('sms-body').value;
                const formData = `From=${encodeURIComponent(phone)}&Body=${encodeURIComponent(body)}`;
                const data = await apiCall('/webhook/sms', 'POST', formData);
                showResult('sms-result', data);
            } catch (error) {
                showResult('sms-result', { error: error.message });
            }
        }
        
        async function analyzeIntent() {
            try {
                const message = document.getElementById('intent-message').value;
                const data = await apiCall('/debug/analyze-intent', 'POST', { message });
                showResult('intent-result', data);
            } catch (error) {
                showResult('intent-result', { error: error.message });
            }
        }
    </script>
</body>
</html>
    '''

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting SMS Trading Bot on port {port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Testing mode: {settings.testing_mode}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
