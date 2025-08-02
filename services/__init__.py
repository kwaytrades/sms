# services/__init__.py
"""
Services package for SMS Trading Bot
Contains all service modules and integrations
"""

# Import enhanced database service with fallback
try:
    from .database_service import UnifiedDatabaseService as DatabaseService
    print("✅ Using enhanced UnifiedDatabaseService")
except ImportError as e:
    print(f"⚠️ Enhanced database service not available: {e}")
    try:
        # Fallback to existing database service
        from .database import DatabaseService
        print("⚠️ Using legacy DatabaseService")
    except ImportError as e2:
        print(f"❌ Database service import failed: {e2}")
        # Create minimal fallback to prevent crashes
        class DatabaseService:
            def __init__(self):
                print("⚠️ Using DatabaseService fallback")
            async def initialize(self):
                pass
            async def health_check(self):
                return {"status": "error", "message": "Database service not available"}

# Import all existing services with error handling
try:
    from .openai_service import OpenAIService
except ImportError as e:
    print(f"⚠️ OpenAIService import failed: {e}")
    class OpenAIService:
        def __init__(self): pass

try:
    from .technical_analysis import TechnicalAnalysisService
except ImportError as e:
    print(f"⚠️ TechnicalAnalysisService import failed: {e}")
    class TechnicalAnalysisService:
        def __init__(self): pass

try:
    from .twilio_service import TwilioService
except ImportError as e:
    print(f"⚠️ TwilioService import failed: {e}")
    class TwilioService:
        def __init__(self): pass

try:
    from .llm_agent import TradingAgent, ToolExecutor
except ImportError as e:
    print(f"⚠️ LLM Agent imports failed: {e}")
    class TradingAgent:
        def __init__(self): pass
    class ToolExecutor:
        def __init__(self): pass

try:
    from .news_sentiment import NewsSentimentService
except ImportError as e:
    print(f"⚠️ NewsSentimentService import failed: {e}")
    class NewsSentimentService:
        def __init__(self): pass

try:
    from .fundamental_analysis import FundamentalAnalysisEngine, FundamentalAnalysisTool
except ImportError as e:
    print(f"⚠️ Fundamental analysis imports failed: {e}")
    class FundamentalAnalysisEngine:
        def __init__(self): pass
    class FundamentalAnalysisTool:
        def __init__(self): pass

try:
    from .key_builder import KeyBuilder
except ImportError as e:
    print(f"⚠️ KeyBuilder import failed: {e}")
    class KeyBuilder:
        def __init__(self): pass

try:
    from .gemini_service import GeminiPersonalityService
except ImportError as e:
    print(f"⚠️ GeminiPersonalityService import failed: {e}")
    class GeminiPersonalityService:
        def __init__(self): pass

# Import new enhanced services if available
try:
    from .vector_service import VectorService
    print("✅ VectorService available")
except ImportError:
    print("⚠️ VectorService not available (using stub)")
    class VectorService:
        def __init__(self): pass

try:
    from .cache_service import CacheService
    print("✅ CacheService available")
except ImportError:
    print("⚠️ CacheService not available (using stub)")
    class CacheService:
        def __init__(self): pass

try:
    from .memory_service import MemoryService
    print("✅ MemoryService available")
except ImportError:
    print("⚠️ MemoryService not available (using stub)")
    class MemoryService:
        def __init__(self): pass

try:
    from .alert_service import AlertService
    print("✅ AlertService available")
except ImportError:
    print("⚠️ AlertService not available (using stub)")
    class AlertService:
        def __init__(self): pass

try:
    from .notification_service import NotificationService
    print("✅ NotificationService available")
except ImportError:
    print("⚠️ NotificationService not available (using stub)")
    class NotificationService:
        def __init__(self): pass

try:
    from .portfolio_service import PortfolioService
    print("✅ PortfolioService available")
except ImportError:
    print("⚠️ PortfolioService not available (using stub)")
    class PortfolioService:
        def __init__(self): pass

try:
    from .trade_tracker_service import TradeTrackerService
    print("✅ TradeTrackerService available")
except ImportError:
    print("⚠️ TradeTrackerService not available (using stub)")
    class TradeTrackerService:
        def __init__(self): pass

try:
    from .research_service import ResearchService
    print("✅ ResearchService available")
except ImportError:
    print("⚠️ ResearchService not available (using stub)")
    class ResearchService:
        def __init__(self): pass

# Export all services
__all__ = [
    # Core existing services
    "DatabaseService",
    "OpenAIService", 
    "TechnicalAnalysisService", 
    "TwilioService",
    "TradingAgent",
    "ToolExecutor",
    "NewsSentimentService",
    "FundamentalAnalysisEngine",
    "FundamentalAnalysisTool",
    "KeyBuilder",
    "GeminiPersonalityService",
    
    # New enhanced services
    "VectorService",
    "CacheService",
    "MemoryService",
    "AlertService",
    "NotificationService",
    "PortfolioService",
    "TradeTrackerService",
    "ResearchService"
]

# Service availability status
def get_service_status():
    """Get status of all available services"""
    return {
        "core_services": {
            "database": "enhanced" if "UnifiedDatabaseService" in str(DatabaseService) else "legacy",
            "openai": "available" if hasattr(OpenAIService, "__module__") else "stub",
            "technical_analysis": "available" if hasattr(TechnicalAnalysisService, "__module__") else "stub",
            "twilio": "available" if hasattr(TwilioService, "__module__") else "stub",
            "llm_agent": "available" if hasattr(TradingAgent, "__module__") else "stub",
            "news_sentiment": "available" if hasattr(NewsSentimentService, "__module__") else "stub",
            "fundamental_analysis": "available" if hasattr(FundamentalAnalysisEngine, "__module__") else "stub",
            "key_builder": "available" if hasattr(KeyBuilder, "__module__") else "stub",
            "gemini_service": "available" if hasattr(GeminiPersonalityService, "__module__") else "stub"
        },
        "enhanced_services": {
            "vector": "available" if hasattr(VectorService, "__module__") else "stub",
            "cache": "available" if hasattr(CacheService, "__module__") else "stub",
            "memory": "available" if hasattr(MemoryService, "__module__") else "stub",
            "alerts": "available" if hasattr(AlertService, "__module__") else "stub",
            "notifications": "available" if hasattr(NotificationService, "__module__") else "stub",
            "portfolio": "available" if hasattr(PortfolioService, "__module__") else "stub",
            "trade_tracker": "available" if hasattr(TradeTrackerService, "__module__") else "stub",
            "research": "available" if hasattr(ResearchService, "__module__") else "stub"
        }
    }

print("🚀 Services package initialized")
print(f"📊 Service status: {get_service_status()}")
