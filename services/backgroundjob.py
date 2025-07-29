from services.background_data_pipeline import BackgroundDataPipeline
import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

# Global pipeline instance
background_pipeline = None

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager - start/stop background jobs"""
    global background_pipeline
    
    try:
        # Initialize background pipeline on startup
        logger.info("🚀 Initializing background data pipeline...")
        
        background_pipeline = BackgroundDataPipeline(
            mongodb_url=settings.mongodb_url,
            redis_url=settings.redis_url,
            eodhd_api_key=settings.eodhd_api_key,
            ta_service_url=settings.ta_service_url or "http://localhost:8001"
        )
        
        await background_pipeline.initialize()
        
        # Start the scheduler
        background_pipeline.start_scheduler()
        
        logger.info("✅ Background pipeline started successfully")
        
        # Optional: Run initial data load if cache is empty
        stock_count = await background_pipeline.db.stocks.count_documents({})
        if stock_count == 0:
            logger.info("📊 Cache empty - running initial data load...")
            await background_pipeline.daily_data_refresh_job()
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Background pipeline startup failed: {e}")
        yield
    
    finally:
        # Cleanup on shutdown
        if background_pipeline:
            await background_pipeline.close()
            logger.info("✅ Background pipeline shutdown complete")

# Update your FastAPI app initialization
# Replace your existing lifespan or add if you don't have one
app = FastAPI(lifespan=lifespan)

# ===== ADMIN ENDPOINTS FOR PIPELINE MONITORING =====

@app.get("/admin/pipeline/status")
async def get_pipeline_status():
    """Get background pipeline status"""
    if not background_pipeline:
        return {"error": "Pipeline not initialized"}
    
    try:
        status = await background_pipeline.get_job_status()
        
        # Add database stats
        stock_count = await background_pipeline.db.stocks.count_documents({})
        latest_update = await background_pipeline.db.stocks.find_one(
            sort=[("last_updated", -1)]
        )
        
        status.update({
            "database_stats": {
                "total_stocks_in_db": stock_count,
                "latest_update": latest_update.get("last_updated") if latest_update else None
            }
        })
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Pipeline status error: {e}")
        return {"error": str(e)}

@app.post("/admin/pipeline/run-daily-job")
async def trigger_daily_job():
    """Manually trigger daily data refresh (admin only)"""
    if not background_pipeline:
        return {"error": "Pipeline not initialized"}
    
    try:
        # Run in background to avoid timeout
        asyncio.create_task(background_pipeline.force_run_daily_job())
        
        return {
            "success": True,
            "message": "Daily data refresh job triggered",
            "note": "Job running in background - check /admin/pipeline/status for progress"
        }
        
    except Exception as e:
        logger.error(f"❌ Manual daily job trigger failed: {e}")
        return {"error": str(e)}

@app.post("/admin/pipeline/run-cleanup")
async def trigger_cleanup_job():
    """Manually trigger weekly cleanup (admin only)"""
    if not background_pipeline:
        return {"error": "Pipeline not initialized"}
    
    try:
        # Run in background
        asyncio.create_task(background_pipeline.force_run_cleanup())
        
        return {
            "success": True,
            "message": "Weekly cleanup job triggered",
            "note": "Job running in background - check /admin/pipeline/status for progress"
        }
        
    except Exception as e:
        logger.error(f"❌ Manual cleanup trigger failed: {e}")
        return {"error": str(e)}

@app.get("/admin/pipeline/sample-data/{symbol}")
async def get_sample_cached_data(symbol: str):
    """Get sample cached data for a symbol (debugging)"""
    if not background_pipeline:
        return {"error": "Pipeline not initialized"}
    
    try:
        symbol = symbol.upper()
        
        # Get from MongoDB
        mongo_data = await background_pipeline.db.stocks.find_one({"symbol": symbol})
        
        # Get from Redis
        redis_basic = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:basic")
        redis_technical = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:technical")
        redis_fundamental = await background_pipeline.redis_client.hgetall(f"stock:{symbol}:fundamental")
        redis_tags = await background_pipeline.redis_client.smembers(f"stock:{symbol}:tags")
        
        return {
            "symbol": symbol,
            "mongodb_data": mongo_data,
            "redis_data": {
                "basic": redis_basic,
                "technical": redis_technical, 
                "fundamental": redis_fundamental,
                "tags": list(redis_tags) if redis_tags else []
            },
            "data_available": {
                "in_mongodb": bool(mongo_data),
                "in_redis": bool(redis_basic)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Sample data fetch failed: {e}")
        return {"error": str(e)}

@app.get("/admin/pipeline/stocks-by-tag/{tag}")
async def get_stocks_by_screening_tag(tag: str):
    """Get stocks by screening tag for testing"""
    if not background_pipeline:
        return {"error": "Pipeline not initialized"}
    
    try:
        # Query MongoDB for stocks with specific tag
        stocks = await background_pipeline.db.stocks.find(
            {"screening_tags": tag}
        ).limit(20).to_list(length=20)
        
        result = []
        for stock in stocks:
            result.append({
                "symbol": stock["symbol"],
                "price": stock.get("basic", {}).get("price", 0),
                "sector": stock.get("basic", {}).get("sector", "Unknown"),
                "market_cap": stock.get("basic", {}).get("market_cap", 0),
                "tags": stock.get("screening_tags", [])
            })
        
        return {
            "tag": tag,
            "matching_stocks": len(result),
            "stocks": result
        }
        
    except Exception as e:
        logger.error(f"❌ Tag query failed: {e}")
        return {"error": str(e)}

# ===== UPDATE YOUR EXISTING SCREENER SERVICE =====

# Now your screener service can query the cached data instead of API calls
class CachedScreenerService:
    """Updated screener that uses cached data"""
    
    def __init__(self, mongodb_client, redis_client):
        self.db = mongodb_client.trading_bot
        self.redis = redis_client
    
    async def screen_stocks_from_cache(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Screen stocks using cached MongoDB data"""
        try:
            # Build MongoDB query from criteria
            query = self._build_mongodb_query(criteria)
            
            # Execute query against cached data
            cursor = self.db.stocks.find(query).limit(100)
            results = await cursor.to_list(length=100)
            
            return {
                "success": True,
                "total_results": len(results),
                "results": results,
                "screened_at": datetime.now().isoformat(),
                "data_source": "cached_mongodb",
                "criteria_applied": criteria
            }
            
        except Exception as e:
            logger.error(f"❌ Cached screening failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_mongodb_query(self, criteria: Dict[str, Any]) -> Dict:
        """Convert screening criteria to MongoDB query"""
        query = {}
        
        # Market cap filters
        if "market_cap_min" in criteria:
            query["basic.market_cap"] = {"$gte": criteria["market_cap_min"]}
        if "market_cap_max" in criteria:
            query.setdefault("basic.market_cap", {})["$lte"] = criteria["market_cap_max"]
        
        # Price filters
        if "price_min" in criteria:
            query["basic.price"] = {"$gte": criteria["price_min"]}
        if "price_max" in criteria:
            query.setdefault("basic.price", {})["$lte"] = criteria["price_max"]
        
        # Volume filters
        if "volume_min" in criteria:
            query["basic.volume"] = {"$gte": criteria["volume_min"]}
        
        # Technical filters
        if "rsi_min" in criteria:
            query["technical.rsi"] = {"$gte": criteria["rsi_min"]}
        if "rsi_max" in criteria:
            query.setdefault("technical.rsi", {})["$lte"] = criteria["rsi_max"]
        
        # Fundamental filters
        if "pe_min" in criteria:
            query["fundamental.pe"] = {"$gte": criteria["pe_min"]}
        if "pe_max" in criteria:
            query.setdefault("fundamental.pe", {})["$lte"] = criteria["pe_max"]
        
        # Tag-based filters (much faster than individual field queries)
        tags = []
        if "large_cap" in str(criteria.values()):
            tags.append("large_cap")
        if "momentum" in str(criteria.values()):
            tags.append("momentum")
        
        if tags:
            query["screening_tags"] = {"$in": tags}
        
        return query

# Initialize the cached screener when pipeline is ready
cached_screener = None

async def initialize_cached_screener():
    """Initialize cached screener after pipeline is ready"""
    global cached_screener
    if background_pipeline and background_pipeline.mongo_client:
        cached_screener = CachedScreenerService(
            background_pipeline.mongo_client,
            background_pipeline.redis_client
        )
        logger.info("✅ Cached screener initialized")

# ===== ENVIRONMENT VARIABLES NEEDED =====

# Add these to your .env file:
"""
# Background Pipeline Configuration
EODHD_API_KEY=your_eodhd_api_key_here
TA_SERVICE_URL=http://localhost:8001
MONGODB_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379

# Job Schedule (Optional - defaults shown)
DAILY_JOB_TIME=06:00
WEEKLY_CLEANUP_TIME=02:00
WEEKLY_CLEANUP_DAY=sunday
"""
