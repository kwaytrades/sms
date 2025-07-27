"""
Dashboard Routes Module - Separated from main.py
Contains all dashboard-related FastAPI routes and endpoints
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Create dashboard router
dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Static files router for dashboard assets
static_router = APIRouter()


    # Mount static files
    #app.mount("/static", StaticFiles(directory="static"), name="static")

@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Main dashboard page - serves the HTML template"""
    try:
        # Read the dashboard HTML template
        dashboard_path = Path("templates/dashboard.html")
        
        if dashboard_path.exists():
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            # Fallback: return a simple message if template doesn't exist
            return HTMLResponse(content="""
            <html>
                <head><title>Dashboard</title></head>
                <body>
                    <h1>Dashboard Template Missing</h1>
                    <p>Please ensure templates/dashboard.html exists</p>
                    <p>Current working directory: {}</p>
                </body>
            </html>
            """.format(os.getcwd()))
            
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return HTMLResponse(content=f"""
        <html>
            <head><title>Dashboard Error</title></head>
            <body>
                <h1>Dashboard Error</h1>
                <p>Error: {str(e)}</p>
            </body>
        </html>
        """, status_code=500)

@static_router.get("/static/dashboard.js")
async def dashboard_js():
    """Serve dashboard JavaScript file"""
    try:
        js_path = Path("static/dashboard.js")
        if js_path.exists():
            return FileResponse(js_path, media_type="application/javascript")
        else:
            # Return empty JS if file doesn't exist to prevent 404s
            return HTMLResponse(content="// Dashboard JS file not found", media_type="application/javascript")
    except Exception as e:
        logger.error(f"Error serving dashboard.js: {e}")
        return HTMLResponse(content=f"// Error loading dashboard.js: {str(e)}", media_type="application/javascript")

@dashboard_router.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Simple test interface for SMS webhook testing"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>SMS Trading Bot - Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    </style>
</head>
<body>
    <h1>🧠 Hyper-Personalized SMS Trading Bot - Test Interface</h1>
    
    <form onsubmit="testSMS(event)">
        <div class="form-group">
            <label>From Phone Number:</label>
            <input type="text" id="phone" value="+13012466712" required>
        </div>
        
        <div class="form-group">
            <label>Message Body:</label>
            <textarea id="message" rows="3" required>yo what's AAPL doing? thinking about buying calls 🚀</textarea>
        </div>
        
        <button type="submit">🚀 Send Test SMS</button>
    </form>
    
    <div id="result"></div>
    
    <div style="margin-top: 30px;">
        <h3>📋 Quick Test Messages</h3>
        <button onclick="setMessage('yo what\\'s AAPL doing?? 🚀🚀')">Casual/High Energy</button>
        <button onclick="setMessage('Could you please analyze TSLA technical indicators?')">Professional</button>
        <button onclick="setMessage('I\\'m new to trading, is NVDA a good buy?')">Beginner</button>
        <button onclick="setMessage('PLUG RSI oversold, MACD bullish divergence thoughts?')">Advanced</button>
    </div>
    
    <script>
        function setMessage(msg) {
            document.getElementById('message').value = msg;
        }
        
        async function testSMS(event) {
            event.preventDefault();
            
            const phone = document.getElementById('phone').value;
            const message = document.getElementById('message').value;
            const resultDiv = document.getElementById('result');
            
            resultDiv.innerHTML = '<div class="result">🔄 Sending test SMS...</div>';
            
            try {
                const formData = new URLSearchParams();
                formData.append('From', phone);
                formData.append('Body', message);
                
                const response = await fetch('/api/test/sms-with-response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h3>✅ Test Successful!</h3>
                            <p><strong>User Message:</strong> ${data.user_message?.body || message}</p>
                            <p><strong>Bot Response:</strong> ${data.bot_response?.content || 'No response generated'}</p>
                            <p><strong>Personality Learning:</strong> ${data.personality_learning || 'inactive'}</p>
                            <p><strong>Agent Type:</strong> ${data.bot_response?.agent_type || 'unknown'}</p>
                        </div>
                    `;
                } else {
                    const errorText = await response.text();
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>❌ Test Failed</h3>
                            <p><strong>Status:</strong> ${response.status}</p>
                            <p><strong>Error:</strong> ${errorText}</p>
                        </div>
                    `;
                }
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="result error">
                        <h3>❌ Network Error</h3>
                        <p><strong>Error:</strong> ${error.message}</p>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>
    """)

# Dashboard API routes that were mixed in main.py
@dashboard_router.get("/health")
async def dashboard_health():
    """Dashboard-specific health check"""
    return {
        "status": "healthy",
        "component": "dashboard",
        "routes": [
            "/dashboard/",
            "/dashboard/test", 
            "/dashboard/health",
            "/static/dashboard.js"
        ]
    }
