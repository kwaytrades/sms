# main.py - Updated with Weekly Scheduler
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, HTMLResponse
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from loguru import logger
import sys

from config import settings
from services.database import DatabaseService
from services.openai_service import OpenAIService
from services.twilio_service import TwilioService
from services.weekly_scheduler import WeeklyScheduler
from core.message_handler import MessageHandler

# Configure logging
logger.remove()
logger.add(sys.stdout, level=settings.log_level)

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
                "message_handler": "active" if message_handler else "inactive",
                "weekly_scheduler": "active" if scheduler_task and not scheduler_task.done() else "inactive"
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "services": {
                "database": "unknown",
                "redis": "unknown",
                "message_handler": "unknown",
                "weekly_scheduler": "unknown"
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
            "message_handler": "active" if message_handler else "inactive",
            "weekly_scheduler": "active" if scheduler_task and not scheduler_task.done() else "inactive"
        },
        "stats": {
            "total_users": "N/A - implement with real DB queries",
            "messages_today": "N/A - implement with analytics"
        },
        "weekly_limits": {
            "free": "4 messages/week",
            "standard": "40 messages/week",
            "vip": "120 messages/week",
            "reset_time": "Monday 9:30 AM EST"
        }
    }

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SMS Trading Bot - Comprehensive Test Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
                        <button class="btn btn-small" onclick="getCacheStats()">Cache Stats</button>
                        <button class="btn btn-small" onclick="getMetrics()">Get Metrics</button>
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

                <!-- Technical Analysis -->
                <div class="card">
                    <h3><span class="emoji">📈</span>Technical Analysis</h3>
                    <div class="form-group">
                        <label>Stock Symbol:</label>
                        <input type="text" id="ta-symbol" value="AAPL" placeholder="AAPL">
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="getAnalysis()">Full Analysis</button>
                        <button class="btn btn-small" onclick="getPrice()">Price Only</button>
                        <button class="btn btn-small" onclick="getTechnical()">Technical Only</button>
                        <button class="btn btn-small" onclick="getNews()">News</button>
                    </div>
                    <div id="ta-result" class="result-box"></div>
                </div>

                <!-- Cache Management -->
                <div class="card">
                    <h3><span class="emoji">⚡</span>Cache Management</h3>
                    <div class="form-group">
                        <label>Symbol for Cache Operations:</label>
                        <input type="text" id="cache-symbol" value="AAPL">
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="prewarmCache()">Prewarm Popular</button>
                        <button class="btn btn-small" onclick="clearCache()">Clear Symbol</button>
                        <button class="btn btn-small" onclick="debugCache()">Debug Cache</button>
                        <button class="btn btn-small" onclick="getCacheStats()">Cache Stats</button>
                    </div>
                    <div id="cache-result" class="result-box"></div>
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

                <!-- Stock Screener -->
                <div class="card">
                    <h3><span class="emoji">🔍</span>Stock Screener</h3>
                    <div class="form-group">
                        <label>Screening Criteria (JSON):</label>
                        <textarea id="screener-criteria" rows="3" placeholder='{"market_cap": ">1B", "pe_ratio": "<25"}'>{"market_cap": ">1B", "pe_ratio": "<25"}</textarea>
                    </div>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="setScreenerPreset('growth')">Growth Stocks</button>
                        <button class="btn btn-small" onclick="setScreenerPreset('value')">Value Stocks</button>
                        <button class="btn btn-small" onclick="setScreenerPreset('dividend')">Dividend Stocks</button>
                    </div>
                    <button class="btn btn-full" onclick="runScreener()">Run Screener</button>
                    <div id="screener-result" class="result-box"></div>
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

                <!-- Debug Console -->
                <div class="card full-width">
                    <h3><span class="emoji">🔧</span>Debug Console</h3>
                    <div class="quick-actions">
                        <button class="btn btn-small" onclick="clearLogs()">Clear Logs</button>
                        <button class="btn btn-small" onclick="exportLogs()">Export Logs</button>
                        <button class="btn btn-small" onclick="toggleAutoRefresh()">Toggle Auto-refresh</button>
                        <button class="btn btn-small" onclick="runDiagnostics()">Run Diagnostics</button>
                    </div>
                    <div id="logs-container" class="result-box" style="height: 250px; overflow-y: auto;"></div>
                </div>
            </div>
        </div>

        <script>
            const BASE_URL = window.location.origin; // Uses current domain
            let autoRefresh = false;
            let refreshInterval;

            function log(message, type = 'info') {
                const logsContainer = document.getElementById('logs-container');
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = `[${timestamp}] ${type.toUpperCase()}: ${message}\\n`;
                logsContainer.textContent += logEntry;
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
                log(`Result for ${elementId}: ${isError ? 'ERROR' : 'SUCCESS'}`, isError ? 'error' : 'info');
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
                    
                    // Handle different response types
                    const contentType = response.headers.get('content-type');
                    let data;
                    
                    if (contentType && contentType.includes('application/json')) {
                        data = await response.json();
                    } else if (contentType && contentType.includes('application/xml')) {
                        data = await response.text();
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

            // Technical Analysis Functions
            async function getAnalysis() {
                showLoading('ta-result');
                try {
                    const symbol = document.getElementById('ta-symbol').value;
                    const data = await apiCall(`/analysis/${symbol}`);
                    showResult('ta-result', data);
                } catch (error) {
                    showResult('ta-result', { error: error.message }, true);
                }
            }

            async function getPrice() {
                showLoading('ta-result');
                try {
                    const symbol = document.getElementById('ta-symbol').value;
                    const data = await apiCall(`/price/${symbol}`);
                    showResult('ta-result', data);
                } catch (error) {
                    showResult('ta-result', { error: error.message }, true);
                }
            }

            async function getTechnical() {
                showLoading('ta-result');
                try {
                    const symbol = document.getElementById('ta-symbol').value;
                    const data = await apiCall(`/technical/${symbol}`);
                    showResult('ta-result', data);
                } catch (error) {
                    showResult('ta-result', { error: error.message }, true);
                }
            }

            async function getNews() {
                showLoading('ta-result');
                try {
                    const symbol = document.getElementById('ta-symbol').value;
                    const data = await apiCall(`/news/${symbol}`);
                    showResult('ta-result', data);
                } catch (error) {
                    showResult('ta-result', { error: error.message }, true);
                }
            }

            // Cache Management Functions
            async function prewarmCache() {
                showLoading('cache-result');
                try {
                    const data = await apiCall('/cache/prewarm', 'POST');
                    showResult('cache-result', data);
                } catch (error) {
                    showResult('cache-result', { error: error.message }, true);
                }
            }

            async function clearCache() {
                showLoading('cache-result');
                try {
                    const symbol = document.getElementById('cache-symbol').value;
                    const data = await apiCall(`/cache/clear/${symbol}`, 'POST');
                    showResult('cache-result', data);
                } catch (error) {
                    showResult('cache-result', { error: error.message }, true);
                }
            }

            async function debugCache() {
                showLoading('cache-result');
                try {
                    const symbol = document.getElementById('cache-symbol').value;
                    const data = await apiCall(`/debug/cache/${symbol}`);
                    showResult('cache-result', data);
                } catch (error) {
                    showResult('cache-result', { error: error.message }, true);
                }
            }

            async function getCacheStats() {
                showLoading('cache-result');
                try {
                    const data = await apiCall('/cache/stats');
                    showResult('cache-result', data);
                } catch (error) {
                    showResult('cache-result', { error: error.message }, true);
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

            // Stock Screener
            function setScreenerPreset(type) {
                const criteria = {
                    growth: '{"revenue_growth": ">20", "earnings_growth": ">15", "pe_ratio": "<30"}',
                    value: '{"pe_ratio": "<15", "pb_ratio": "<2", "debt_equity": "<0.5"}',
                    dividend: '{"dividend_yield": ">3", "payout_ratio": "<60", "market_cap": ">1B"}'
                };
                document.getElementById('screener-criteria').value = criteria[type];
            }

            async function runScreener() {
                showLoading('screener-result');
                try {
                    const criteriaText = document.getElementById('screener-criteria').value;
                    const criteria = JSON.parse(criteriaText);
                    const data = await apiCall('/screener', 'POST', criteria);
                    showResult('screener-result', data);
                } catch (error) {
                    showResult('screener-result', { error: error.message }, true);
                }
            }

            // Metrics and Monitoring
            async function refreshMetrics() {
                try {
                    // Get various metrics
                    const [health, userStats, cacheStats] = await Promise.all([
                        apiCall('/health').catch(() => ({status: 'offline'})),
                        apiCall('/admin/users/stats').catch(() => ({total_users: 0})),
                        apiCall('/cache/stats').catch(() => ({total_keys: 0}))
                    ]);

                    // Update metric displays
                    document.getElementById('uptime').textContent = health.status === 'ok' ? '✅ Online' : '❌ Offline';
                    document.getElementById('total-users').textContent = userStats.total_users || '0';
                    document.getElementById('active-users').textContent = userStats.active_today || '0';
                    document.getElementById('cache-hits').textContent = cacheStats.total_keys || '0';
                    document.getElementById('total-requests').textContent = '1,031'; // You can make this dynamic
                    document.getElementById('response-time').textContent = '50ms'; // You can make this dynamic

                } catch (error) {
                    log('Error refreshing metrics: ' + error.message, 'error');
                }
            }

            // Utility Functions
            function clearLogs() {
                document.getElementById('logs-container').textContent = '';
                log('Logs cleared', 'info');
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
                log('Logs exported', 'info');
            }

            function toggleAutoRefresh() {
                autoRefresh = !autoRefresh;
                if (autoRefresh) {
                    refreshInterval = setInterval(refreshMetrics, 30000);
                    log('Auto-refresh enabled (30s interval)', 'info');
                } else {
                    clearInterval(refreshInterval);
                    log('Auto-refresh disabled', 'info');
                }
            }

            async function runDiagnostics() {
                log('Running system diagnostics...', 'info');
                
                const tests = [
                    { name: 'Health Check', endpoint: '/health' },
                    { name: 'Database Connection', endpoint: '/debug/database' },
                    { name: 'Cache Status', endpoint: '/cache/stats' },
                    { name: 'User Stats', endpoint: '/admin/users/stats' }
                ];

                for (const test of tests) {
                    try {
                        const result = await apiCall(test.endpoint);
                        log(`✅ ${test.name}: OK`, 'info');
                    } catch (error) {
                        log(`❌ ${test.name}: ${error.message}`, 'error');
                    }
                }
                
                log('Diagnostics complete', 'info');
            }

            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                log('SMS Trading Bot Test Dashboard initialized', 'info');
                refreshMetrics();
            });
        </script>
    </body>
    </html>
    """
