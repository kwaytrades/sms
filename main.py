# main.py - Complete Unified SMS Trading Bot with Full Feature Set
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger

# Import all services and core modules
from config import settings, PLAN_LIMITS, POPULAR_TICKERS, is_popular_ticker
from services.database import DatabaseService
from services.openai_service import OpenAIService
from services.twilio_service import TwilioService
from services.stripe_service import StripeService
from services.technical_analysis import TechnicalAnalysisService
from core.message_handler import MessageHandler
from core.user_manager import UserManager
from utils.validators import validate_phone_number, sanitize_input

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Global services - will be initialized in lifespan
db_service = None
openai_service = None
twilio_service = None
stripe_service = None
ta_service = None
message_handler = None
user_manager = None

# Metrics collector for monitoring
class MetricsCollector:
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.total_requests = 0
        self.sms_messages_processed = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.user_signups = 0
        self.subscription_changes = 0
        
    def record_request(self):
        self.total_requests += 1
    
    def record_sms_processed(self):
        self.sms_messages_processed += 1
    
    def record_analysis(self, success: bool):
        if success:
            self.successful_analyses += 1
        else:
            self.failed_analyses += 1
    
    def record_cache(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_signup(self):
        self.user_signups += 1
    
    def record_subscription_change(self):
        self.subscription_changes += 1
    
    def get_stats(self) -> Dict[str, Any]:
        uptime = datetime.utcnow() - self.start_time
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0
        
        return {
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_formatted": str(uptime).split('.')[0],
            "total_requests": self.total_requests,
            "sms_messages_processed": self.sms_messages_processed,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "user_signups": self.user_signups,
            "subscription_changes": self.subscription_changes
        }

# Global metrics instance
metrics = MetricsCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_service, openai_service, twilio_service, stripe_service, ta_service, message_handler, user_manager
    
    logger.info("🚀 Starting Unified SMS Trading Bot with Full Feature Set...")
    
    try:
        # Initialize database service first
        db_service = DatabaseService()
        await db_service.initialize()
        logger.info("✅ Database service initialized")
        
        # Initialize other services
        openai_service = OpenAIService()
        logger.info("✅ OpenAI service initialized")
        
        twilio_service = TwilioService()
        logger.info("✅ Twilio service initialized")
        
        stripe_service = StripeService()
        logger.info("✅ Stripe service initialized")
        
        # Initialize technical analysis service with database cache
        ta_service = TechnicalAnalysisService()
        await ta_service.initialize(db_service)
        logger.info("✅ Technical Analysis service initialized")
        
        # Initialize user manager
        user_manager = UserManager(db_service)
        logger.info("✅ User Manager initialized")
        
        # Initialize message handler with all services
        message_handler = MessageHandler(
            db_service, 
            openai_service, 
            twilio_service, 
            ta_service,
            user_manager
        )
        logger.info("✅ Message Handler initialized")
        
        # Perform startup tasks
        await _startup_tasks()
        
        logger.info("🎉 All services initialized successfully - Ready to serve!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down services...")
    
    try:
        if ta_service:
            await ta_service.close()
        if db_service:
            await db_service.close()
        logger.info("✅ All services shut down gracefully")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

async def _startup_tasks():
    """Perform startup tasks like database cleanup"""
    try:
        # Clean up old data (older than 30 days)
        if db_service:
            cleanup_result = await db_service.cleanup_old_data(days=30)
            logger.info(f"🧹 Startup cleanup: {cleanup_result}")
        
        # Warm up popular ticker cache if market is open
        if ta_service and ta_service.market_scheduler.is_market_hours():
            logger.info("🔥 Market is open - warming up popular ticker cache...")
            # This would pre-fetch popular tickers in the background
            
    except Exception as e:
        logger.warning(f"⚠️ Startup tasks failed (non-critical): {e}")

# Create FastAPI app
app = FastAPI(
    title="Unified SMS Trading Bot",
    description="Complete SMS-based AI trading assistant with technical analysis, user management, and subscription billing",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None
)

# Add CORS middleware for development
if settings.environment == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Middleware to track requests
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    metrics.record_request()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# ============================================================================
# CORE ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root with comprehensive service information"""
    return {
        "service": "Unified SMS Trading Bot",
        "version": "3.0.0",
        "status": "operational",
        "description": "Complete SMS-based AI trading assistant",
        "features": {
            "sms_interface": "Natural language SMS conversations via Twilio",
            "technical_analysis": "Built-in RSI, MACD, Support/Resistance, Gap Analysis",
            "ai_responses": "Personalized responses using OpenAI GPT",
            "user_management": "Behavioral learning and personalization",
            "subscription_billing": "Stripe-powered subscription management",
            "smart_caching": "Redis-based caching with market-aware TTL",
            "real_time_data": "EODHD market data integration",
            "security": "Input validation, webhook verification, secure secrets"
        },
        "plans": {
            "free": {"price": "$0", "limit": "4 messages/week", "features": ["Basic analysis"]},
            "paid": {"price": "$29/month", "limit": "40 messages/week", "features": ["Advanced analysis", "Personalization"]},
            "pro": {"price": "$99/month", "limit": "120 messages/week", "features": ["Unlimited daily", "Real-time alerts", "Priority support"]}
        },
        "endpoints": {
            "sms_webhook": "/webhook/sms",
            "stripe_webhook": "/webhook/stripe",
            "technical_analysis": "/analysis/{symbol}",
            "trading_signals": "/signals/{symbol}",
            "health_check": "/health",
            "admin_dashboard": "/admin",
            "metrics": "/metrics",
            "cache_management": "/cache/*",
            "user_management": "/admin/users/*"
        },
        "supported_commands": [
            "Natural language: 'How is AAPL doing?'",
            "Price queries: 'TSLA price'",
            "Commands: START, HELP, UPGRADE, STATUS, WATCHLIST"
        ],
        "data_sources": {
            "market_data": "EODHD Professional API",
            "ai_responses": "OpenAI GPT-4",
            "user_data": "MongoDB Atlas",
            "caching": "Redis"
        },
        "uptime": metrics.get_stats()["uptime_formatted"],
        "environment": settings.environment
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with detailed service status"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "3.0.0",
            "environment": settings.environment,
            "services": {},
            "metrics": metrics.get_stats(),
            "configuration": {
                "cache_enabled": True,
                "market_hours_aware": True,
                "behavioral_learning": True,
                "subscription_management": True
            }
        }
        
        # Check each service
        service_checks = []
        
        # Database check
        if db_service and db_service.db:
            try:
                await db_service.db.command('ping')
                health_status["services"]["mongodb"] = {"status": "connected", "database": "sms_trading_bot"}
                service_checks.append(True)
            except Exception as e:
                health_status["services"]["mongodb"] = {"status": "error", "error": str(e)}
                service_checks.append(False)
        else:
            health_status["services"]["mongodb"] = {"status": "not_initialized"}
            service_checks.append(False)
        
        # Redis check
        if db_service and db_service.redis:
            try:
                await db_service.redis.ping()
                health_status["services"]["redis"] = {"status": "connected"}
                service_checks.append(True)
            except Exception as e:
                health_status["services"]["redis"] = {"status": "error", "error": str(e)}
                service_checks.append(False)
        else:
            health_status["services"]["redis"] = {"status": "not_initialized"}
            service_checks.append(False)
        
        # OpenAI check
        health_status["services"]["openai"] = {
            "status": "configured" if openai_service and openai_service.client else "not_configured"
        }
        service_checks.append(bool(openai_service and openai_service.client))
        
        # Twilio check
        health_status["services"]["twilio"] = {
            "status": "configured" if twilio_service and twilio_service.client else "not_configured"
        }
        service_checks.append(bool(twilio_service))
        
        # Stripe check
        health_status["services"]["stripe"] = {
            "status": "configured" if stripe_service else "not_configured"
        }
        service_checks.append(bool(stripe_service))
        
        # Technical Analysis check
        health_status["services"]["technical_analysis"] = {
            "status": "active" if ta_service else "inactive",
            "market_hours": ta_service.market_scheduler.is_market_hours() if ta_service else False
        }
        service_checks.append(bool(ta_service))
        
        # Message Handler check
        health_status["services"]["message_handler"] = {
            "status": "active" if message_handler else "inactive"
        }
        service_checks.append(bool(message_handler))
        
        # User Manager check
        health_status["services"]["user_manager"] = {
            "status": "active" if user_manager else "inactive"
        }
        service_checks.append(bool(user_manager))
        
        # Overall health determination
        critical_services_healthy = all(service_checks[:3])  # MongoDB, Redis, OpenAI are critical
        all_services_healthy = all(service_checks)
        
        if all_services_healthy:
            health_status["status"] = "healthy"
            status_code = 200
        elif critical_services_healthy:
            health_status["status"] = "degraded"
            status_code = 200
        else:
            health_status["status"] = "unhealthy"
            status_code = 503
        
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# ============================================================================
# SMS WEBHOOK ENDPOINT
# ============================================================================

@app.post("/webhook/sms")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming SMS messages from Twilio
    This is the main entry point for user interactions
    """
    try:
        # Parse Twilio webhook data
        form_data = await request.form()
        from_number = form_data.get('From')
        message_body = form_data.get('Body', '').strip()
        twilio_message_sid = form_data.get('MessageSid')
        
        # Log incoming message
        logger.info(f"📱 SMS received from {from_number}: {message_body[:50]}{'...' if len(message_body) > 50 else ''}")
        
        # Validate required fields
        if not from_number or not message_body:
            logger.error("❌ Missing required fields in SMS webhook")
            return PlainTextResponse("Missing required fields", status_code=400)
        
        # Validate and sanitize phone number
        try:
            validated_phone = validate_phone_number(from_number)
        except ValueError as e:
            logger.error(f"❌ Invalid phone number {from_number}: {e}")
            return PlainTextResponse("Invalid phone number", status_code=400)
        
        # Sanitize message body
        sanitized_message = sanitize_input(message_body)
        
        # Check if system is healthy enough to process
        if not message_handler:
            logger.error("❌ Message handler not available")
            # Send error SMS if Twilio is available
            if twilio_service:
                error_msg = "System temporarily unavailable. Please try again in a few minutes."
                await twilio_service.send_sms(validated_phone, error_msg)
            return PlainTextResponse("Service unavailable", status_code=503)
        
        # Record metrics
        metrics.record_sms_processed()
        
        # Process message in background to return quickly to Twilio
        background_tasks.add_task(
            _process_sms_message,
            validated_phone,
            sanitized_message,
            twilio_message_sid
        )
        
        # Return empty TwiML response immediately
        # (We send SMS responses directly via Twilio API, not TwiML)
        return PlainTextResponse(
            '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"❌ SMS webhook error: {e}")
        return PlainTextResponse("Internal error", status_code=500)

async def _process_sms_message(phone_number: str, message_body: str, message_sid: str):
    """Background task to process SMS message"""
    try:
        logger.info(f"🔄 Processing SMS {message_sid} from {phone_number}")
        
        # Process message through message handler
        success = await message_handler.process_incoming_message(phone_number, message_body)
        
        if success:
            logger.info(f"✅ Successfully processed SMS {message_sid}")
        else:
            logger.error(f"❌ Failed to process SMS {message_sid}")
            
    except Exception as e:
        logger.error(f"❌ Background SMS processing error for {message_sid}: {e}")

# ============================================================================
# STRIPE WEBHOOK ENDPOINT
# ============================================================================

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events for subscription management"""
    try:
        payload = await request.body()
        sig_header = request.headers.get('stripe-signature')
        
        logger.info("💳 Stripe webhook received")
        
        if not stripe_service:
            logger.error("❌ Stripe service not available")
            return JSONResponse(status_code=503, content={"error": "Stripe service unavailable"})
        
        # Verify webhook signature and get event
        event = await stripe_service.handle_webhook(payload, sig_header)
        
        if not event:
            logger.error("❌ Failed to verify Stripe webhook")
            return JSONResponse(status_code=400, content={"error": "Invalid webhook"})
        
        # Handle different event types
        event_type = event['type']
        logger.info(f"💳 Processing Stripe event: {event_type}")
        
        if event_type in ['customer.subscription.created', 'customer.subscription.updated']:
            subscription = event['data']['object']
            customer_id = subscription['customer']
            
            # Update user subscription in database
            if user_manager:
                success = await user_manager.update_subscription_from_stripe(customer_id, subscription)
                if success:
                    metrics.record_subscription_change()
                    logger.info(f"✅ Updated subscription for customer {customer_id}")
                else:
                    logger.error(f"❌ Failed to update subscription for customer {customer_id}")
        
        elif event_type == 'customer.subscription.deleted':
            subscription = event['data']['object']
            customer_id = subscription['customer']
            
            # Handle subscription cancellation
            if user_manager:
                # Find user and update to free plan
                user = await user_manager.db.db.users.find_one({"stripe_customer_id": customer_id})
                if user:
                    await user_manager.update_subscription(user["phone_number"], "free")
                    metrics.record_subscription_change()
                    logger.info(f"✅ Cancelled subscription for customer {customer_id}")
        
        elif event_type == 'invoice.payment_failed':
            invoice = event['data']['object']
            customer_id = invoice['customer']
            logger.warning(f"⚠️ Payment failed for customer {customer_id}")
            
            # Optionally notify user via SMS about payment failure
            if user_manager and twilio_service:
                user = await user_manager.db.db.users.find_one({"stripe_customer_id": customer_id})
                if user:
                    message = "⚠️ Payment failed for your SMS Trading Bot subscription. Please update your payment method to continue receiving premium features."
                    await twilio_service.send_sms(user["phone_number"], message)
        
        return {"status": "success", "event_type": event_type}
        
    except Exception as e:
        logger.error(f"❌ Stripe webhook error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

# ============================================================================
# TECHNICAL ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/analysis/{symbol}")
async def get_technical_analysis(
    symbol: str,
    interval: str = Query("1d", description="Time interval (1d, 4h, 1h)"),
    period: str = Query("1mo", description="Time period (1mo, 3mo, 6mo, 1y)")
):
    """
    Get comprehensive technical analysis for a symbol
    Includes price data, indicators, support/resistance, signals
    """
    try:
        if not ta_service:
            raise HTTPException(status_code=503, detail="Technical analysis service not available")
        
        logger.info(f"📊 Technical analysis requested for {symbol}")
        
        # Get analysis
        result = await ta_service.analyze_symbol(symbol, interval, period)
        
        # Record metrics
        if "error" in result:
            metrics.record_analysis(False)
            logger.error(f"❌ Technical analysis failed for {symbol}: {result['error']}")
        else:
            metrics.record_analysis(True)
            # Record cache hit/miss
            cache_status = result.get("cache_status", "unknown")
            metrics.record_cache(cache_status == "hit")
            logger.info(f"✅ Technical analysis completed for {symbol} (cache: {cache_status})")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Technical analysis endpoint error: {e}")
        metrics.record_analysis(False)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/{symbol}")
async def get_trading_signals(symbol: str):
    """Get just the trading signals for a symbol (lighter endpoint)"""
    try:
        if not ta_service:
            raise HTTPException(status_code=503, detail="Technical analysis service not available")
        
        signals = await ta_service.get_trading_signals(symbol)
        
        return {
            "symbol": symbol.upper(),
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(signals)
        }
        
    except Exception as e:
        logger.error(f"❌ Trading signals error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/price/{symbol}")
async def get_current_price(symbol: str):
    """Get just the current price for a symbol (fastest endpoint)"""
    try:
        if not ta_service:
            raise HTTPException(status_code=503, detail="Technical analysis service not available")
        
        # Get basic analysis (should hit cache for popular symbols)
        analysis = await ta_service.analyze_symbol(symbol, "1d", "1mo")
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        return {
            "symbol": symbol.upper(),
            "price": analysis["current_price"],
            "change": analysis["price_change"],
            "volume": analysis["volume"],
            "timestamp": analysis["timestamp"],
            "cache_status": analysis.get("cache_status", "unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Price endpoint error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/cache/stats")
async def get_cache_stats():
    """Get comprehensive cache statistics"""
    try:
        if not ta_service:
            return {"error": "Technical analysis service not available"}
        
        stats = await ta_service.get_cache_stats()
        
        # Add service-level cache metrics
        stats.update({
            "service_metrics": {
                "cache_hits": metrics.cache_hits,
                "cache_misses": metrics.cache_misses,
                "hit_rate": round((metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) * 100), 2) if (metrics.cache_hits + metrics.cache_misses) > 0 else 0
            },
            "popular_tickers": {
                "count": len(POPULAR_TICKERS),
                "examples": POPULAR_TICKERS[:10]
            }
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Cache stats error: {e}")
        return {"error": str(e)}

@app.post("/cache/clear")
async def clear_all_cache():
    """Clear all technical analysis cache"""
    try:
        if not ta_service:
            return {"error": "Technical analysis service not available"}
        
        success = await ta_service.clear_cache()
        
        if success:
            logger.info("🗑️ All cache cleared by admin")
            return {"message": "All cache cleared successfully", "timestamp": datetime.utcnow().isoformat()}
        else:
            return {"error": "Failed to clear cache"}
        
    except Exception as e:
        logger.error(f"❌ Clear cache error: {e}")
        return {"error": str(e)}

@app.delete("/cache/{symbol}")
async def invalidate_symbol_cache(symbol: str):
    """Invalidate cache for specific symbol"""
    try:
        if not ta_service:
            return {"error": "Technical analysis service not available"}
        
        success = await ta_service.invalidate_cache(symbol)
        
        message = f"Cache {'invalidated' if success else 'not found'} for {symbol.upper()}"
        logger.info(f"🗑️ {message}")
        
        return {
            "message": message,
            "symbol": symbol.upper(),
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Invalidate cache error for {symbol}: {e}")
        return {"error": str(e)}

@app.post("/cache/prewarm")
async def prewarm_popular_cache():
    """Pre-warm cache for popular tickers (admin function)"""
    try:
        if not ta_service:
            return {"error": "Technical analysis service not available"}
        
        # This would trigger pre-warming of popular tickers
        logger.info("🔥 Cache pre-warming triggered by admin")
        
        # For now, just return success
        return {
            "message": "Cache pre-warming initiated",
            "popular_tickers": len(POPULAR_TICKERS),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Cache pre-warm error: {e}")
        return {"error": str(e)}

# ============================================================================
# ADMIN AND USER MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/admin")
async def admin_dashboard():
    """Comprehensive admin dashboard with system statistics"""
    try:
        dashboard_data = {
            "title": "SMS Trading Bot Admin Dashboard",
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "3.0.0",
            "environment": settings.environment
        }
        
        # Get system metrics
        dashboard_data["metrics"] = metrics.get_stats()
        
        # Get user statistics
        if user_manager:
            dashboard_data["users"] = await user_manager.get_user_stats()
        
        # Get database statistics
        if db_service:
            dashboard_data["database"] = await db_service.get_stats()
        
        # Get technical analysis statistics
        if ta_service:
            dashboard_data["technical_analysis"] = await ta_service.get_stats()
        
        # Add service health
        dashboard_data["services"] = {
            "database": "connected" if db_service and db_service.db else "disconnected",
            "redis": "connected" if db_service and db_service.redis else "disconnected",
            "openai": "configured" if openai_service and openai_service.client else "not_configured",
            "twilio": "configured" if twilio_service and twilio_service.client else "not_configured",
            "stripe": "configured" if stripe_service else "not_configured",
            "technical_analysis": "active" if ta_service else "inactive",
            "message_handler": "active" if message_handler else "inactive"
        }
        
        # Add configuration info
        dashboard_data["configuration"] = {
            "plans": PLAN_LIMITS,
            "popular_tickers_count": len(POPULAR_TICKERS),
            "cache_ttl": {
                "popular": settings.cache_popular_ttl,
                "ondemand": settings.cache_ondemand_ttl,
                "afterhours": settings.cache_afterhours_ttl
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"❌ Admin dashboard error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

@app.get("/admin/users/{phone_number}")
async def get_user_profile(phone_number: str):
    """Get detailed user profile for admin"""
    try:
        if not user_manager:
            raise HTTPException(status_code=503, detail="User manager not available")
        
        # Validate phone number format
        try:
            validated_phone = validate_phone_number(phone_number)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid phone number: {e}")
        
        user = await user_manager.get_user_by_phone(validated_phone)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get additional user data
        user_data = {
            "profile": user,
            "usage": {
                "weekly": await db_service.get_weekly_usage(validated_phone),
                "daily": await db_service.get_daily_usage(validated_phone)
            },
            "conversation_history": await db_service.get_conversation_history(validated_phone, days=7)
        }
        
        return user_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get user profile error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/users/{phone_number}/subscription")
async def update_user_subscription(phone_number: str, request: Request):
    """Update user subscription (admin function)"""
    try:
        if not user_manager:
            raise HTTPException(status_code=503, detail="User manager not available")
        
        # Validate phone number
        try:
            validated_phone = validate_phone_number(phone_number)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid phone number: {e}")
        
        # Parse request data
        data = await request.json()
        plan_type = data.get('plan_type')
        
        if plan_type not in PLAN_LIMITS:
            raise HTTPException(status_code=400, detail=f"Invalid plan type. Must be one of: {list(PLAN_LIMITS.keys())}")
        
        # Update subscription
        success = await user_manager.update_subscription(
            validated_phone,
            plan_type,
            data.get('stripe_customer_id'),
            data.get('stripe_subscription_id')
        )
        
        if success:
            metrics.record_subscription_change()
            logger.info(f"✅ Admin updated subscription for {validated_phone} to {plan_type}")
            return {"success": True, "message": f"Subscription updated to {plan_type}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update subscription")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Update subscription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/users/stats")
async def get_user_statistics():
    """Get comprehensive user statistics"""
    try:
        if not user_manager:
            raise HTTPException(status_code=503, detail="User manager not available")
        
        stats = await user_manager.get_user_stats()
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# METRICS AND MONITORING ENDPOINTS
# ============================================================================

@app.get("/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics for monitoring"""
    try:
        system_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": {
                "name": "SMS Trading Bot",
                "version": "3.0.0",
                "environment": settings.environment
            },
            "performance": metrics.get_stats(),
            "health": {}
        }
        
        # Add health status for each service
        system_metrics["health"] = {
            "database": "healthy" if db_service and db_service.db else "unhealthy",
            "redis": "healthy" if db_service and db_service.redis else "unhealthy",
            "openai": "configured" if openai_service and openai_service.client else "not_configured",
            "twilio": "configured" if twilio_service else "not_configured",
            "stripe": "configured" if stripe_service else "not_configured",
            "technical_analysis": "healthy" if ta_service else "unhealthy"
        }
        
        # Add cache statistics if available
        if ta_service:
            cache_stats = await ta_service.get_cache_stats()
            system_metrics["cache"] = cache_stats
        
        # Add user statistics if available
        if user_manager:
            user_stats = await user_manager.get_user_stats()
            system_metrics["users"] = user_stats
        
        return system_metrics
        
    except Exception as e:
        logger.error(f"❌ Metrics error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

# ============================================================================
# DEBUG AND TESTING ENDPOINTS (Development only)
# ============================================================================

if settings.environment == "development":
    
    @app.post("/debug/test-sms")
    async def test_sms_processing(request: Request):
        """Test SMS processing without Twilio (development only)"""
        try:
            data = await request.json()
            phone_number = data.get('phone_number', '+1234567890')
            message = data.get('message', 'How is AAPL?')
            
            # Validate phone
            validated_phone = validate_phone_number(phone_number)
            
            # Process message
            if message_handler:
                success = await message_handler.process_incoming_message(validated_phone, message)
                return {"success": success, "phone": validated_phone, "message": message}
            else:
                return {"error": "Message handler not available"}
                
        except Exception as e:
            return {"error": str(e)}
    
    @app.get("/debug/create-test-user/{phone_number}")
    async def create_test_user(phone_number: str):
        """Create a test user (development only)"""
        try:
            validated_phone = validate_phone_number(phone_number)
            
            if user_manager:
                user = await user_manager.get_or_create_user(validated_phone)
                metrics.record_signup()
                return {"success": True, "user": user}
            else:
                return {"error": "User manager not available"}
                
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url.path),
            "message": "Check /docs for available endpoints",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Custom 500 handler"""
    logger.error(f"Internal server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Starting Unified SMS Trading Bot v3.0.0")
    logger.info(f"🌍 Environment: {settings.environment}")
    logger.info(f"🔧 Port: {port}")
    logger.info(f"📊 Features: Technical Analysis, User Management, Subscriptions, AI Responses")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
