# ===== main.py - COMPLETE VERSION WITH FIXED FASTAPI MIDDLEWARE =====
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
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

# Import configuration - FIXED IMPORTS (removed non-existent items)
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
        self.recent_requests = deque(maxlen=100)  # Last 100 requests
        
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
                # Keep only last 50 response times per endpoint
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
                    avg_response_times[endpoint] = round(sum(times) / len(times) * 1000, 2)  # Convert to ms
            
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
                    "formatted": str(uptime).split('.')[0]  # Remove microseconds
                },
                "requests": {
                    "total": self.total_requests,
                    "per_minute": requests_per_minute,
                    "by_endpoint": dict(self.requests_by_endpoint),
                    "recent": list(self.recent_requests)[-10:]  # Last 10 requests
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
                    "recent": list(self.recent_errors)[-5:]  # Last 5 errors
                }
            }

# ===== PLAN LIMITS CONFIGURATION =====
# Since we removed this from config, define it here
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

# Popular tickers list (since removed from config)
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

# ===== FASTAPI MIDDLEWARE FOR METRICS (CORRECTED) =====

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
        # TODO: Implement Stripe webhook signature verification
        # TODO: Handle subscription events
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

# ===== SCHEDULER ENDPOINTS =====

@app.get("/admin/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    if not scheduler_task:
        return {"status": "inactive", "message": "Scheduler not running"}
    
    try:
        if db_service and twilio_service:
            temp_scheduler = WeeklyScheduler(db_service, twilio_service)
            return {"status": "active" if scheduler_task and not scheduler_task.done() else "inactive"}
        else:
            return {"status": "unavailable", "message": "Required services not available"}
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/admin/scheduler/manual-reset")
async def manual_reset_trigger():
    """Manually trigger reset notifications (for testing)"""
    if not db_service or not twilio_service:
        raise HTTPException(status_code=503, detail="Required services unavailable")
    
    try:
        temp_scheduler = WeeklyScheduler(db_service, twilio_service)
        logger.info("📅 Manual reset trigger executed")
        return {"status": "success", "message": "Manual reset trigger executed"}
    except Exception as e:
        logger.error(f"Manual reset trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/scheduler/manual-reminder")
async def manual_reminder_trigger():
    """Manually trigger 24-hour reminders (for testing)"""
    if not db_service or not twilio_service:
        raise HTTPException(status_code=503, detail="Required services unavailable")
    
    try:
        temp_scheduler = WeeklyScheduler(db_service, twilio_service)
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
            "system": {
                "version": "1.0.0",
                "environment": settings.environment,
                "testing_mode": settings.testing_mode
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ===== DEBUG ENDPOINTS =====

@app.get("/debug/database")
async def debug_database():
    """Debug database connections and collections"""
    if not db_service:
        return {"error": "Database service not available"}
    
    try:
        # Test connection
        user_count = await db_service.db.users.count_documents({})
        collections = await db_service.db.list_collection_names()
        
        # Test user retrieval
        sample_user = await db_service.db.users.find_one()
        
        return {
            "database_name": db_service.db.name,
            "collections": collections,
            "users_count": user_count,
            "sample_user_phone": sample_user.get("phone_number") if sample_user else None,
            "sample_user_fields": list(sample_user.keys()) if sample_user else [],
            "connection_status": "connected"
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
        "popular_tickers_count": len(POPULAR_TICKERS)
    }

@app.post("/debug/test-activity/{phone_number}")
async def test_user_activity(phone_number: str):
    """Test user activity update"""
    if not db_service:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    
    try:
        # Ensure user exists
        user = await db_service.get_or_create_user(phone_number)
        
        # Test activity update
        success = await db_service.update_user_activity(phone_number, "received")
        
        # Get updated user
        updated_user = await db_service.get_user_by_phone(phone_number)
        
        return {
            "update_success": success,
            "user_data": updated_user.to_dict() if updated_user else None
        }
    except Exception as e:
        logger.error(f"Test activity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/limits/{phone_number}")
async def debug_limits(phone_number: str):
    """Debug user limits"""
    if not db_service:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    
    try:
        limit_check = await db_service.check_message_limits(phone_number)
        return limit_check
    except Exception as e:
        logger.error(f"Debug limits error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/analyze-intent")
async def debug_analyze_intent(request: Request):
    """Debug intent analysis"""
    try:
        data = await request.json()
        message = data.get('message', '')
        
        # Simple intent classification (since we don't have the full analyzer)
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

# ===== COMPREHENSIVE TEST INTERFACE =====

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Comprehensive test interface"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMS Trading Bot - Comprehensive Test Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }

        .card h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .emoji {
            font-size: 1.5rem;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }

        .form-group input, 
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }

        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 5px 5px 5px 0;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn-small {
            padding: 8px 16px;
            font-size: 12px;
        }

        .btn-full {
            width: 100%;
            margin-top: 10px;
        }

        .result-box {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        .result-box.success {
            background: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }

        .result-box.error {
            background: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }

        .quick-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 15px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .metric {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #333;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 5px;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-online { background: #28a745; }
        .status-offline { background: #dc3545; }
        .status-warning { background: #ffc107; }

        .loading {
            opacity: 0.6;
            pointer-events: none;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .two-column {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }

        .info-box {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }

        .info-box h4 {
            color: #0066cc;
            margin-bottom: 10px;
        }

        .info-box ul {
            margin-left: 20px;
        }

        .info-box li {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📱 SMS Trading Bot Test Dashboard</h1>
            <p>Comprehensive testing interface for all microservice endpoints</p>
        </div>

        <div class="dashboard-grid">
            <!-- SMS Message Testing -->
            <div class="card">
                <h3><span class="emoji">💬</span>SMS Message Testing</h3>
                <div class="form-group">
                    <label>From Phone:</label>
                    <input type="text" id="sms-phone" value="+13012466712">
                </div>
                <div class="form-group">
                    <label>Message Body:</label>
                    <textarea id="sms-body" rows="3" placeholder="How is AAPL doing?">How is AAPL doing?</textarea>
                </div>
                <div class="quick-actions">
                    <button class="btn btn-small" onclick="testSMS('START')">START</button>
                    <button class="btn btn-small" onclick="testSMS('How is AAPL?')">Stock Query</button>
                    <button class="btn btn-small" onclick="testSMS('Find me good stocks')">Screener</button>
                    <button class="btn btn-small" onclick="testSMS('UPGRADE')">Upgrade</button>
                </div>
                <button class="btn btn-full" onclick="sendCustomSMS()">Send Custom Message</button>
                <div id="sms-result" class="result-box"></div>
            </div>

            <!-- System Health -->
            <div class="card">
                <h3><span class="emoji">🏥</span>System Health</h3>
                <div class="quick-actions">
                    <button class="btn btn-small" onclick="checkHealth()">Health Check</button>
                    <button class="btn btn-small" onclick="checkDatabase()">Database Info</button>
                    <button class="btn btn-small" onclick="getMetrics()">Get Metrics</button>
                    <button class="btn btn-small" onclick="debugConfig()">Debug Config</button>
                </div>
                <div id="health-result" class="result-box"></div>
            </div>

            <!-- User Management -->
            <div class="card">
                <h3><span class="emoji">👤</span>User Management</h3>
                <div class="form-group">
                    <label>Phone Number:</label>
                    <input type="text" id="user-phone" value="+13012466712">
                </div>
                <div class="quick-actions">
                    <button class="btn btn-small" onclick="getUser()">Get User</button>
                    <button class="btn btn-small" onclick="testActivity()">Test Activity</button>
                    <button class="btn btn-small" onclick="getUserStats()">User Stats</button>
                    <button class="btn btn-small" onclick="checkLimits()">Check Limits</button>
                </div>
                <div id="user-result" class="result-box"></div>
            </div>

            <!-- Intent Analysis -->
            <div class="card">
                <h3><span class="emoji">🧠</span>Intent Analysis</h3>
                <div class="form-group">
                    <label>Message to Analyze:</label>
                    <textarea id="intent-message" rows="2" placeholder="What's the RSI for TSLA and NVDA?">What's the RSI for TSLA and NVDA?</textarea>
                </div>
                <button class="btn btn-full" onclick="analyzeIntent()">Analyze Intent</button>
                <div id="intent-result" class="result-box"></div>
            </div>

            <!-- Subscription Testing -->
            <div class="card">
                <h3><span class="emoji">💳</span>Subscription Testing</h3>
                <div class="two-column">
                    <div>
                        <div class="form-group">
                            <label>Phone:</label>
                            <input type="text" id="sub-phone" value="+13012466712">
                        </div>
                        <div class="form-group">
                            <label>Plan Type:</label>
                            <select id="sub-plan">
                                <option value="free">Free</option>
                                <option value="paid">Paid ($29/month)</option>
                                <option value="pro">Pro ($99/month)</option>
                            </select>
                        </div>
                    </div>
                    <div>
                        <div class="form-group">
                            <label>Stripe Customer ID:</label>
                            <input type="text" id="stripe-customer" placeholder="cus_test123">
                        </div>
                        <div class="form-group">
                            <label>Stripe Subscription ID:</label>
                            <input type="text" id="stripe-subscription" placeholder="sub_test123">
                        </div>
                    </div>
                </div>
                <button class="btn btn-full" onclick="updateSubscription()">Update Subscription</button>
                <div id="subscription-result" class="result-box"></div>
            </div>

            <!-- Scheduler Management -->
            <div class="card">
                <h3><span class="emoji">📅</span>Scheduler Management</h3>
                <div class="quick-actions">
                    <button class="btn btn-small" onclick="schedulerStatus()">Status</button>
                    <button class="btn btn-small" onclick="manualReset()">Manual Reset</button>
                    <button class="btn btn-small" onclick="manualReminder()">Manual Reminder</button>
                </div>
                <div id="scheduler-result" class="result-box"></div>
            </div>

            <!-- Live Metrics -->
            <div class="card full-width">
                <h3><span class="emoji">📊</span>Live System Metrics</h3>
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
                        <div class="metric-value" id="cache-hits">--</div>
                        <div class="metric-label">Cache Performance</div>
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
                </div>
                <button class="btn btn-full" onclick="refreshMetrics()">Refresh All Metrics</button>
            </div>

            <!-- Plan Information -->
            <div class="card full-width">
                <h3><span class="emoji">📋</span>Plan Information & Limits</h3>
                <div class="info-box">
                    <h4>Current Plan Structure:</h4>
                    <ul>
                        <li><strong>Free Plan:</strong> 10 messages/week</li>
                        <li><strong>Paid Plan ($29/month):</strong> 100 messages/month</li>
                        <li><strong>Pro Plan ($99/month):</strong> Unlimited (with 50 msg/day cooloff)</li>
                    </ul>
                </div>
                <div class="info-box">
                    <h4>Smart Warning System:</h4>
                    <ul>
                        <li>75% usage: Warning message sent</li>
                        <li>90% usage: Urgent warning with upgrade prompt</li>
                        <li>100% usage: Limit exceeded, upgrade required</li>
                        <li>Pro users: Daily cooloff after 50 messages</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <script>
        const BASE_URL = window.location.origin;

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
            element.innerHTML = '<span class="spinner"></span>Loading...';
            element.className = 'result-box';
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
                } else {
                    data = await response.text();
                }
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${typeof data === 'object' ? JSON.stringify(data) : data}`);
                }
                
                return data;
            } catch (error) {
                throw new Error(`API Error: ${error.message}`);
            }
        }

        // SMS Testing Functions
        async function testSMS(message) {
            document.getElementById('sms-body').value = message;
            await sendCustomSMS();
        }

        async function sendCustomSMS() {
            showLoading('sms-result');
            try {
                const phone = document.getElementById('sms-phone').value;
                const body = document.getElementById('sms-body').value;
                
                const formData = `From=${encodeURIComponent(phone)}&Body=${encodeURIComponent(body)}`;
                const data = await apiCall('/webhook/sms', 'POST', formData, true);
                showResult('sms-result', data);
            } catch (error) {
                showResult('sms-result', { error: error.message }, true);
            }
        }

        // System Health Functions
        async function checkHealth() {
            showLoading('health-result');
            try {
                const data = await apiCall('/health');
                showResult('health-result', data);
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
            }
        }

        async function checkDatabase() {
            showLoading('health-result');
            try {
                const data = await apiCall('/debug/database');
                showResult('health-result', data);
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
            }
        }

        async function getMetrics() {
            showLoading('health-result');
            try {
                const data = await apiCall('/metrics');
                showResult('health-result', data);
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
            }
        }

        async function debugConfig() {
            showLoading('health-result');
            try {
                const data = await apiCall('/debug/config');
                showResult('health-result', data);
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
            }
        }

        // User Management Functions
        async function getUser() {
            showLoading('user-result');
            try {
                const phone = document.getElementById('user-phone').value;
                const data = await apiCall(`/admin/users/${encodeURIComponent(phone)}`);
                showResult('user-result', data);
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
            }
        }

        async function testActivity() {
            showLoading('user-result');
            try {
                const phone = document.getElementById('user-phone').value;
                const data = await apiCall(`/debug/test-activity/${encodeURIComponent(phone)}`, 'POST');
                showResult('user-result', data);
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
            }
        }

        async function getUserStats() {
            showLoading('user-result');
            try {
                const data = await apiCall('/admin/users/stats');
                showResult('user-result', data);
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
            }
        }

        async function checkLimits() {
            showLoading('user-result');
            try {
                const phone = document.getElementById('user-phone').value;
                const data = await apiCall(`/debug/limits/${encodeURIComponent(phone)}`);
                showResult('user-result', data);
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
            }
        }

        // Intent Analysis
        async function analyzeIntent() {
            showLoading('intent-result');
            try {
                const message = document.getElementById('intent-message').value;
                const data = await apiCall('/debug/analyze-intent', 'POST', { message });
                showResult('intent-result', data);
            } catch (error) {
                showResult('intent-result', { error: error.message }, true);
            }
        }

        // Subscription Management
        async function updateSubscription() {
            showLoading('subscription-result');
            try {
                const phone = document.getElementById('sub-phone').value;
                const planData = {
                    plan_type: document.getElementById('sub-plan').value,
                    stripe_customer_id: document.getElementById('stripe-customer').value,
                    stripe_subscription_id: document.getElementById('stripe-subscription').value
                };
                
                const data = await apiCall(`/admin/users/${encodeURIComponent(phone)}/subscription`, 'POST', planData);
                showResult('subscription-result', data);
            } catch (error) {
                showResult('subscription-result', { error: error.message }, true);
            }
        }

        // Scheduler Functions
        async function schedulerStatus() {
            showLoading('scheduler-result');
            try {
                const data = await apiCall('/admin/scheduler/status');
                showResult('scheduler-result', data);
            } catch (error) {
                showResult('scheduler-result', { error: error.message }, true);
            }
        }

        async function manualReset() {
            showLoading('scheduler-result');
            try {
                const data = await apiCall('/admin/scheduler/manual-reset', 'POST');
                showResult('scheduler-result', data);
            } catch (error) {
                showResult('scheduler-result', { error: error.message }, true);
            }
        }

        async function manualReminder() {
            showLoading('scheduler-result');
            try {
                const data = await apiCall('/admin/scheduler/manual-reminder', 'POST');
                showResult('scheduler-result', data);
            } catch (error) {
                showResult('scheduler-result', { error: error.message }, true);
            }
        }

        // Metrics and Monitoring
        async function refreshMetrics() {
            try {
                const data = await apiCall('/metrics');
                
                if (data.service) {
                    const service = data.service;
                    
                    document.getElementById('uptime').textContent = service.uptime?.formatted || '--';
                    document.getElementById('total-requests').textContent = service.requests?.total || '0';
                    document.getElementById('cache-hits').textContent = service.performance?.cache_hit_rate + '%' || '0%';
                    document.getElementById('active-users').textContent = service.tickers?.unique_count || '0';
                    document.getElementById('total-requests').textContent = service.requests?.total || '0';
                    document.getElementById('response-time').textContent = 
                        Object.values(service.performance?.avg_response_times_ms || {})[0] + 'ms' || '--';
                }

            } catch (error) {
                console.error('Error refreshing metrics:', error);
            }
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            console.log('SMS Trading Bot Test Dashboard initialized');
            refreshMetrics();
        });
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
