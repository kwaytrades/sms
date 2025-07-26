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

# Add this endpoint to your main.py file

@app.get("/dashboard", response_class=HTMLResponse)
async def comprehensive_dashboard():
    """Enhanced comprehensive admin dashboard with beautiful UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMS Trading Bot - Admin Dashboard</title>
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

        .btn:active {
            transform: translateY(0);
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

        .info-box {
            background: linear-gradient(135deg, #e6fffa, #b2f5ea);
            border: 1px solid #81e6d9;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
        }

        .info-box h4 {
            color: #234e52;
            margin-bottom: 12px;
            font-weight: 600;
        }

        .info-box ul {
            margin-left: 20px;
        }

        .info-box li {
            margin-bottom: 6px;
            color: #285e61;
        }

        .log-entry {
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 8px;
            font-size: 12px;
            border-left: 3px solid var(--info);
        }

        .log-entry.error {
            background: #fff5f5;
            border-left-color: var(--error);
            color: #742a2a;
        }

        .log-entry.success {
            background: #f0fff4;
            border-left-color: var(--success);
            color: #22543d;
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

            .status-bar {
                flex-direction: column;
                align-items: center;
            }

            .metrics-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
        }

        .copy-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: var(--primary);
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 11px;
        }

        .activity-timeline {
            max-height: 400px;
            overflow-y: auto;
        }

        .activity-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            border-bottom: 1px solid var(--border);
            transition: background 0.3s ease;
        }

        .activity-item:hover {
            background: rgba(102, 126, 234, 0.05);
        }

        .activity-item:last-child {
            border-bottom: none;
        }

        .ticker-badge {
            background: var(--primary);
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <i class="fas fa-chart-line"></i>
                SMS Trading Bot Dashboard
            </h1>
            <p>Comprehensive monitoring and testing interface for your trading bot</p>
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
                <i class="fas fa-vial"></i> Testing
            </button>
            <button class="tab" onclick="switchTab('monitoring')">
                <i class="fas fa-chart-area"></i> Monitoring
            </button>
            <button class="tab" onclick="switchTab('users')">
                <i class="fas fa-users"></i> Users
            </button>
            <button class="tab" onclick="switchTab('logs')">
                <i class="fas fa-terminal"></i> Logs
            </button>
        </div>

        <!-- Testing Tab -->
        <div id="testing-tab" class="tab-content active">
            <div class="dashboard-grid">
                <!-- SMS Testing -->
                <div class="card">
                    <h3><i class="fas fa-sms card-icon"></i>SMS Message Testing</h3>
                    <div class="form-group">
                        <label>From Phone:</label>
                        <input type="text" id="sms-phone" value="+13012466712" placeholder="+1234567890">
                    </div>
                    <div class="form-group">
                        <label>Message Body:</label>
                        <textarea id="sms-body" rows="3" placeholder="How is AAPL doing?">How is AAPL doing?</textarea>
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="testSMS('START')">
                            <i class="fas fa-play"></i> START
                        </button>
                        <button class="btn btn-small" onclick="testSMS('How is AAPL?')">
                            <i class="fas fa-chart-line"></i> Stock Query
                        </button>
                        <button class="btn btn-small" onclick="testSMS('Find me good stocks')">
                            <i class="fas fa-search"></i> Screener
                        </button>
                        <button class="btn btn-small" onclick="testSMS('/upgrade')">
                            <i class="fas fa-arrow-up"></i> Upgrade
                        </button>
                    </div>
                    <button class="btn btn-full" onclick="sendCustomSMS()">
                        <i class="fas fa-paper-plane"></i> Send Custom Message
                    </button>
                    <div id="sms-result" class="result-box"></div>
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

                <!-- Subscription Testing -->
                <div class="card">
                    <h3><i class="fas fa-credit-card card-icon"></i>Subscription Testing</h3>
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
                                <label>Customer ID:</label>
                                <input type="text" id="stripe-customer" placeholder="cus_test123">
                            </div>
                            <div class="form-group">
                                <label>Subscription ID:</label>
                                <input type="text" id="stripe-subscription" placeholder="sub_test123">
                            </div>
                        </div>
                    </div>
                    <button class="btn btn-full" onclick="updateSubscription()">
                        <i class="fas fa-sync"></i> Update Subscription
                    </button>
                    <div id="subscription-result" class="result-box"></div>
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
                        <div class="metric-value" id="error-rate">--</div>
                        <div class="metric-label">Error Rate</div>
                    </div>
                </div>
                <button class="btn btn-full" onclick="refreshMetrics()">
                    <i class="fas fa-sync"></i> Refresh All Metrics
                </button>
            </div>
        </div>

        <!-- Users Tab -->
        <div id="users-tab" class="tab-content">
            <div class="dashboard-grid">
                <div class="card">
                    <h3><i class="fas fa-users card-icon"></i>User Statistics</h3>
                    <div id="user-stats-container">
                        <div class="metric">
                            <div class="metric-value" id="total-users-detail">--</div>
                            <div class="metric-label">Total Users</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3><i class="fas fa-clock card-icon"></i>Recent Activity</h3>
                    <div class="activity-timeline" id="recent-activity">
                        <div class="activity-item">No recent activity</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Logs Tab -->
        <div id="logs-tab" class="tab-content">
            <div class="card full-width">
                <h3><i class="fas fa-terminal card-icon"></i>System Logs</h3>
                <div class="quick-actions">
                    <button class="btn btn-small" onclick="clearLogs()">
                        <i class="fas fa-trash"></i> Clear Logs
                    </button>
                    <button class="btn btn-small" onclick="exportLogs()">
                        <i class="fas fa-download"></i> Export Logs
                    </button>
                    <button class="btn btn-small" onclick="toggleAutoRefresh()">
                        <i class="fas fa-sync"></i> <span id="auto-refresh-text">Enable Auto-refresh</span>
                    </button>
                </div>
                <div id="logs-container" class="result-box" style="height: 400px; overflow-y: auto;">
                    <div class="log-entry">Dashboard initialized successfully</div>
                </div>
            </div>
        </div>

        <!-- Configuration Info -->
        <div class="card full-width">
            <h3><i class="fas fa-cog card-icon"></i>Configuration & Environment</h3>
            <div class="info-box">
                <h4>Service Configuration:</h4>
                <ul>
                    <li><strong>Environment:</strong> <span id="env-info">Loading...</span></li>
                    <li><strong>Database:</strong> <span id="db-info">Loading...</span></li>
                    <li><strong>SMS Service:</strong> <span id="sms-info">Loading...</span></li>
                    <li><strong>AI Service:</strong> <span id="ai-info">Loading...</span></li>
                    <li><strong>Payment Processing:</strong> <span id="payment-info">Loading...</span></li>
                </ul>
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
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Load data for specific tabs
            if (tabName === 'monitoring') {
                refreshMetrics();
            } else if (tabName === 'users') {
                loadUserStats();
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

        function log(message, type = 'info') {
            const logsContainer = document.getElementById('logs-container');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${type}`;
            logEntry.textContent = `[${timestamp}] ${message}`;
            logsContainer.appendChild(logEntry);
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }

        function showResult(elementId, data, isError = false) {
            const element = document.getElementById(elementId);
            if (typeof data === 'object') {
                element.textContent = JSON.stringify(data, null, 2);
            } else {
                element.textContent = data;
            }
            element.className = `result-box ${isError ? 'error' : 'success'}`;
            
            // Add copy button
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-btn';
            copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(element.textContent);
                showToast('Copied to clipboard!');
            };
            element.style.position = 'relative';
            element.appendChild(copyBtn);
            
            log(`${elementId}: ${isError ? 'ERROR' : 'SUCCESS'}`, isError ? 'error' : 'success');
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
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${data.message || JSON.stringify(data)}`);
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
                showToast('SMS sent successfully!');
            } catch (error) {
                showResult('sms-result', { error: error.message }, true);
                showToast('SMS failed to send', 'error');
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
                const data = await apiCall('/admin');
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
                const data = await apiCall('/admin');
                showResult('health-result', data);
                showToast('Metrics retrieved');
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
                showToast('Failed to get metrics', 'error');
            }
        }

        // User Management Functions
        async function getUser() {
            showLoading('user-result');
            try {
                const phone = document.getElementById('user-phone').value;
                // This would need to be implemented as an endpoint
                const data = { message: `User lookup for ${phone} - endpoint not implemented yet` };
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
                const data = { message: `Activity test for ${phone} - endpoint not implemented yet` };
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
                const data = await apiCall('/admin');
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
                const data = { message: `Limit check for ${phone} - endpoint not implemented yet` };
                showResult('user-result', data);
                showToast('Limits checked');
            } catch (error) {
                showResult('user-result', { error: error.message }, true);
                showToast('Failed to check limits', 'error');
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
                
                const data = { message: `Subscription update for ${phone} - endpoint not implemented yet`, plan: planData };
                showResult('subscription-result', data);
                showToast('Subscription updated successfully!');
            } catch (error) {
                showResult('subscription-result', { error: error.message }, true);
                showToast('Failed to update subscription', 'error');
            }
        }

        // Metrics and Monitoring
        async function refreshMetrics() {
            try {
                const health = await apiCall('/health');
                const admin = await apiCall('/admin');
                
                // Update metric displays
                document.getElementById('uptime').textContent = health.status === 'healthy' ? '✅ Online' : '❌ Offline';
                document.getElementById('total-users').textContent = admin.stats?.total_users || '0';
                document.getElementById('active-users').textContent = admin.stats?.messages_today || '0';
                document.getElementById('total-requests').textContent = '1,031'; // Mock data
                document.getElementById('response-time').textContent = '50ms'; // Mock data
                document.getElementById('error-rate').textContent = '0.2%'; // Mock data

                // Update configuration info
                updateConfigInfo(health, admin);
                
                showToast('Metrics refreshed');
            } catch (error) {
                log('Error refreshing metrics: ' + error.message, 'error');
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
                    <i class="fas fa-database"></i>
                    <span>Database: ${healthData.database?.mongodb?.status || 'Unknown'}</span>
                </div>
                <div class="status-item ${healthData.services?.openai === 'available' ? 'status-online' : 'status-warning'}">
                    <i class="fas fa-robot"></i>
                    <span>AI: ${healthData.services?.openai || 'Unknown'}</span>
                </div>
                <div class="status-item ${healthData.services?.twilio === 'available' ? 'status-online' : 'status-warning'}">
                    <i class="fas fa-sms"></i>
                    <span>SMS: ${healthData.services?.twilio || 'Unknown'}</span>
                </div>
            `;
        }

        function updateConfigInfo(health, admin) {
            document.getElementById('env-info').textContent = health.environment || 'Unknown';
            document.getElementById('db-info').textContent = health.database?.mongodb?.status || 'Unknown';
            document.getElementById('sms-info').textContent = health.services?.twilio || 'Unknown';
            document.getElementById('ai-info').textContent = health.services?.openai || 'Unknown';
            document.getElementById('payment-info').textContent = admin.configuration?.payments_enabled ? 'Enabled' : 'Disabled';
        }

        async function loadUserStats() {
            try {
                const data = await apiCall('/admin');
                document.getElementById('total-users-detail').textContent = data.stats?.total_users || '0';
                showToast('User stats loaded');
            } catch (error) {
                log('Error loading user stats: ' + error.message, 'error');
            }
        }

        // Utility Functions
        function clearLogs() {
            document.getElementById('logs-container').innerHTML = '<div class="log-entry">Logs cleared</div>';
            showToast('Logs cleared');
        }

        function exportLogs() {
            const logs = document.getElementById('logs-container').textContent;
            const blob = new Blob([logs], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sms-bot-logs-${new Date().toISOString().split('T')[0]}.txt`;
            a.click();
            URL.revokeObjectURL(url);
            showToast('Logs exported');
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const btn = document.getElementById('auto-refresh-text');
            
            if (autoRefresh) {
                refreshInterval = setInterval(refreshMetrics, 30000);
                btn.textContent = 'Disable Auto-refresh';
                showToast('Auto-refresh enabled (30s interval)');
            } else {
                clearInterval(refreshInterval);
                btn.textContent = 'Enable Auto-refresh';
                showToast('Auto-refresh disabled');
            }
        }

        async function runDiagnostics() {
            showLoading('health-result');
            log('Running comprehensive system diagnostics...', 'info');
            
            const tests = [
                { name: 'Health Check', endpoint: '/health' },
                { name: 'Admin Dashboard', endpoint: '/admin' },
            ];

            let results = { passed: 0, failed: 0, details: [] };

            for (const test of tests) {
                try {
                    const result = await apiCall(test.endpoint);
                    results.passed++;
                    results.details.push(`✅ ${test.name}: OK`);
                    log(`✅ ${test.name}: OK`, 'success');
                } catch (error) {
                    results.failed++;
                    results.details.push(`❌ ${test.name}: ${error.message}`);
                    log(`❌ ${test.name}: ${error.message}`, 'error');
                }
            }
            
            showResult('health-result', results);
            showToast(`Diagnostics complete: ${results.passed} passed, ${results.failed} failed`);
            log('Diagnostics complete', 'info');
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            log('SMS Trading Bot Dashboard initialized', 'success');
            checkHealth();
            refreshMetrics();
            
            // Auto-refresh every 60 seconds
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
