from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Create dashboard router
dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])

def setup_static_files(app):
    """Setup static file serving for dashboard assets"""
    # Create static directories if they don't exist
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ Static files mounted successfully")

@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Main dashboard page - serves the HTML template"""
    try:
        dashboard_path = Path("templates/dashboard.html")
        
        if dashboard_path.exists():
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            return HTMLResponse(content="<h1>Dashboard template missing</h1>")
            
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return HTMLResponse(content=f"<h1>Dashboard Error: {str(e)}</h1>", status_code=500)
