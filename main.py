# ===== main.py =====
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
import uvicorn
from contextlib import asynccontextmanager
from loguru import logger
import sys

from config import settings
from services.database import DatabaseService
from services.openai_service import OpenAIService
from services.twilio_service import TwilioService
from core.message_handler import MessageHandler
#from core.scheduler import DailyScheduler
from webhooks.twilio_webhook import handle_incoming_sms
from webhooks.stripe_webhook import handle_stripe_webhook
from admin.dashboard import admin_router

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level)

# Global services
db_service = None
openai_service = None
twilio_service = None
message_handler = None
scheduler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_service, openai_service, twilio_service, message_handler, scheduler
    
    logger.info("🚀 Starting SMS Trading Bot...")
    
    # Initialize services
    db_service = DatabaseService()
    await db_service.initialize()
    
    openai_service = OpenAIService()
    twilio_service = TwilioService()
    message_handler = MessageHandler(db_service, openai_service, twilio_service)
    
    # Start scheduler
    #scheduler = DailyScheduler(db_service, message_handler)
    #await scheduler.start()
    
    logger.info("✅ SMS Trading Bot started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SMS Trading Bot...")
    if scheduler:
        await scheduler.stop()
    if db_service:
        await db_service.close()

app = FastAPI(
    title="SMS Trading Bot",
    description="Hyper-personalized SMS trading insights",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(admin_router, prefix="/admin", tags=["admin"])

@app.get("/")
async def root():
    return {"message": "SMS Trading Bot API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "services": {
            "database": "connected" if db_service and db_service.db else "disconnected",
            "redis": "connected" if db_service and db_service.redis else "disconnected"
        }
    }

@app.post("/webhook/sms")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming SMS messages from Twilio"""
    return await handle_incoming_sms(request, background_tasks, message_handler)

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    return await handle_stripe_webhook(request, db_service)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )
