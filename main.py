# main.py - Minimal working version
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

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level)

# Global services
db_service = None
openai_service = None
twilio_service = None
message_handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_service, openai_service, twilio_service, message_handler
    
    logger.info("🚀 Starting SMS Trading Bot...")
    
    try:
        # Initialize services
        db_service = DatabaseService()
        await db_service.initialize()
        
        openai_service = OpenAIService()
        twilio_service = TwilioService()
        message_handler = MessageHandler(db_service, openai_service, twilio_service)
        
        logger.info("✅ SMS Trading Bot started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SMS Trading Bot...")
    if db_service:
        await db_service.close()

app = FastAPI(
    title="SMS Trading Bot",
    description="Hyper-personalized SMS trading insights",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "SMS Trading Bot API", 
        "status": "running",
        "version": "1.0.0",
        "environment": settings.environment
    }

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "environment": settings.environment,
            "services": {
                "database": "connected" if db_service and db_service.db else "disconnected",
                "redis": "connected" if db_service and db_service.redis else "disconnected",
                "message_handler": "active" if message_handler else "inactive"
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "services": {
                "database": "unknown",
                "redis": "unknown"
            }
        }

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
    """Handle Stripe webhook events (placeholder)"""
    try:
        payload = await request.body()
        logger.info("Stripe webhook received (not implemented yet)")
        return {"status": "received", "implemented": False}
        
    except Exception as e:
        logger.error(f"❌ Stripe webhook error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/admin")
async def admin_dashboard():
    """Simple admin dashboard"""
    return {
        "title": "SMS Trading Bot Admin",
        "status": "running",
        "services": {
            "database": "connected" if db_service else "disconnected",
            "message_handler": "active" if message_handler else "inactive"
        },
        "stats": {
            "total_users": "N/A - implement with real DB queries",
            "messages_today": "N/A - implement with analytics"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )
@app.get("/test")
async def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>SMS Bot Test</title></head>
    <body>
        <h2>Test SMS Webhook</h2>
        <form action="/webhook/sms" method="POST">
            <label>From Phone:</label><br>
            <input type="text" name="From" value="+1234567890"><br><br>
            
            <label>Message:</label><br>
            <input type="text" name="Body" value="START"><br><br>
            
            <input type="submit" value="Send Test SMS">
        </form>
        
        <hr>
        <h3>Quick Tests:</h3>
        <button onclick="testCommand('START')">Test START</button>
        <button onclick="testCommand('/help')">Test /help</button>
        <button onclick="testCommand('/upgrade')">Test /upgrade</button>
        
        <div id="results"></div>
        
        <script>
        function testCommand(message) {
            fetch('/webhook/sms', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: `From=%2B1234567890&Body=${encodeURIComponent(message)}`
            })
            .then(response => response.text())
            .then(data => {
                document.getElementById('results').innerHTML = 
                    `<h4>Response for "${message}":</h4><pre>${data}</pre>`;
            });
        }
        </script>
    </body>
    </html>
    """
