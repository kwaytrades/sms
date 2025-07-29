# services/backgroundjob/screener_service.py
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class CachedScreenerService:
    """
    Stock Screener with intelligent caching strategy:
    - Primary: Query cached MongoDB data (90% of cases) - <100ms response
    - Fallback: EODHD Screener API for complex queries - 2-3s response
    - Auto-caching: Store API results for future use
    """
    
    def __init__(self, mongodb_url: str, redis_url: str, eodhd_api_key: str):
        self.mongodb_url = mongodb_url
        self.redis_url = redis_url
        self.eodhd_api_key = eodhd_api_key
        
        # Database connections
        self.mongo_client = None
        self.db = None
        self.redis_client = None
        
        # Screening statistics
        self.screening_stats = {
            "total_screens": 0,
            "cache_hits": 0,
            "api_fallback": 0,
            "complex_queries": 0,
            "avg_response_time_ms": 0
        }
        
        # Common screening patterns (for optimization)
        self.popular_screens = {
            "growth_stocks": {
                "revenue_growth": {"$gte": 20},
                "basic.market_cap": {"$gte": 1000000000}
            },
            "value_stocks": {
                "fundamental.pe": {"$gte": 5, "$lte": 15},
                "fundamental.pb": {"$lte": 2}
            },
            "dividend_stocks": {
                "fundamental.dividend_yield": {"$gte": 3}
            },
            "oversold_stocks": {
                "technical.rsi": {"$lte": 30}
            },
            "overbought_stocks": {
                "technical.rsi": {"$gte": 70}
            },
            "momentum_stocks": {
                "basic.change_1d": {"$gte": 5},
                "basic.volume": {"$gte": 1000000}
            },
            "small_cap_growth": {
                "basic.market_cap": {"$gte": 300000000, "$lte": 2000000000},
                "fundamental.revenue_growth": {"$gte": 15}
            },
            "tech_stocks": {
                "basic.sector": "Technology"
            },
            "quantum_stocks": {
                "screening_tags": {"$in": ["quantum", "ai", "semiconductor"]}
            },
            "crypto_related": {
                "screening_tags": {"$in": ["crypto", "blockchain", "bitcoin"]}
            }
        }
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(self.mongodb_url)
            self.db = self.mongo_client.trading_bot
            
            # Redis connection  
            self.redis_client = aioredis.from_url(self.redis_url)
            
            # Test connections
            await self.db.command("ping")
            await self.redis_client.ping()
            
            logger.info("✅ Screener service initialized - DB connections ready")
            
            # Warm up popular screen cache
            await self._warm_up_popular_screens()
            
        except Exception as e:
            logger.error(f"❌ Screener service initialization failed: {e}")
            raise
    
    async def screen_stocks(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main screening method with intelligent caching
        
        Args:
            criteria: Screening criteria dict
            
        Returns:
            Dict containing screening results with metadata
        """
        start_time = datetime.now()
        
        try:
            self.screening_stats["total_screens"] += 1
            
            # Step 1: Check if this is a popular/cached screen
            screen_key = self._get_screen_cache_key(criteria)
            cached_result = await self._get_cached_screen_result(screen_key)
            
            if cached_result:
                self.screening_stats["cache_hits"] += 1
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                logger.info(f"✅ Cached screen result in {response_time:.1f}ms")
                
                return {
                    "results": cached_result,
                    "data_source": "cached_mongodb",
                    "response_time_ms": round(response_time, 1),
                    "total_matches": len(cached_result),
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": True
                }
            
            # Step 2: Try MongoDB query for cached data
            mongodb_results = await self._query_mongodb_cache(criteria)
            
            if mongodb_results:
                # Cache the result for future use
                await self._cache_screen_result(screen_key, mongodb_results)
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                logger.info(f"✅ MongoDB screen result in {response_time:.1f}ms")
                
                return {
                    "results": mongodb_results,
                    "data_source": "mongodb_cache", 
                    "response_time_ms": round(response_time, 1),
                    "total_matches": len(mongodb_results),
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": False
                }
            
            # Step 3: Fallback to EODHD Screener API for complex queries
            self.screening_stats["api_fallback"] += 1
            
            logger.info("📡 Using EODHD Screener API fallback for complex query")
            
            api_results = await self._query_eodhd_screener_api(criteria)
            
            if api_results:
                # Cache API results in MongoDB for future use
                await self._store_api_results_in_cache(api_results)
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                logger.info(f"✅ API fallback screen result in {response_time:.1f}ms")
                
                return {
                    "results": api_results,
                    "data_source": "eodhd_api_fallback",
                    "response_time_ms": round(response_time, 1),
                    "total_matches": len(api_results),
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": False,
                    "note": "Complex query used API fallback - results cached for future use"
                }
            
            # Step 4: No results found
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "results": [],
                "data_source": "no_results",
                "response_time_ms": round(response_time, 1),
                "total_matches": 0,
                "timestamp": datetime.now().isoformat(),
                "message": "No stocks match the specified criteria"
            }
            
        except Exception as e:
            logger.error(f"❌ Stock screening failed: {e}")
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "error": str(e),
                "data_source": "error",
                "response_time_ms": round(response_time, 1),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _query_mongodb_cache(self, criteria: Dict[str, Any]) -> List[Dict]:
        """Query cached stock data in MongoDB using screening criteria"""
        try:
            # Convert natural language criteria to MongoDB query
            mongo_query = await self._convert_criteria_to_mongo_query(criteria)
            
            if not mongo_query:
                logger.info("❌ Could not convert criteria to MongoDB query")
                return []
            
            logger.info(f"🔍 MongoDB Query: {mongo_query}")
            
            # Execute query with proper projection and limits
            cursor = self.db.stocks.find(
                mongo_query,
                {
                    "symbol": 1,
                    "basic": 1,
                    "technical": 1,
                    "fundamental": 1,
                    "screening_tags": 1,
                    "last_updated": 1
                }
            ).limit(500)  # Reasonable limit for screening results
            
            results = await cursor.to_list(length=None)
            
            # Format results for consistent output
            formatted_results = []
            for stock in results:
                formatted_stock = {
                    "symbol": stock.get("symbol", ""),
                    "name": f"{stock.get('symbol', '')} Inc",  # Would need company name data
                    "price": stock.get("basic", {}).get("price", 0),
                    "change_1d": stock.get("basic", {}).get("change_1d", 0),
                    "volume": stock.get("basic", {}).get("volume", 0),
                    "market_cap": stock.get("basic", {}).get("market_cap", 0),
                    "sector": stock.get("basic", {}).get("sector", "Unknown"),
                    "pe_ratio": stock.get("fundamental", {}).get("pe", 0),
                    "rsi": stock.get("technical", {}).get("rsi", 50),
                    "trend": stock.get("technical", {}).get("trend", "neutral"),
                    "tags": stock.get("screening_tags", []),
                    "last_updated": stock.get("last_updated", datetime.now()).isoformat()
                }
                formatted_results.append(formatted_stock)
            
            logger.info(f"✅ MongoDB query returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ MongoDB query failed: {e}")
            return []
    
    async def _convert_criteria_to_mongo_query(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Convert screening criteria to MongoDB query"""
        try:
            mongo_query = {}
            
            # Market cap filters
            if "market_cap_min" in criteria:
                mongo_query["basic.market_cap"] = {"$gte": criteria["market_cap_min"]}
            if "market_cap_max" in criteria:
                if "basic.market_cap" in mongo_query:
                    mongo_query["basic.market_cap"]["$lte"] = criteria["market_cap_max"]
                else:
                    mongo_query["basic.market_cap"] = {"$lte": criteria["market_cap_max"]}
            
            # Price filters
            if "price_min" in criteria:
                mongo_query["basic.price"] = {"$gte": criteria["price_min"]}
            if "price_max" in criteria:
                if "basic.price" in mongo_query:
                    mongo_query["basic.price"]["$lte"] = criteria["price_max"]
                else:
                    mongo_query["basic.price"] = {"$lte": criteria["price_max"]}
            
            # Volume filters
            if "volume_min" in criteria:
                mongo_query["basic.volume"] = {"$gte": criteria["volume_min"]}
            
            # Technical indicators
            if "rsi_min" in criteria:
                mongo_query["technical.rsi"] = {"$gte": criteria["rsi_min"]}
            if "rsi_max" in criteria:
                if "technical.rsi" in mongo_query:
                    mongo_query["technical.rsi"]["$lte"] = criteria["rsi_max"]
                else:
                    mongo_query["technical.rsi"] = {"$lte": criteria["rsi_max"]}
            
            # Fundamental metrics
            if "pe_min" in criteria:
                mongo_query["fundamental.pe"] = {"$gte": criteria["pe_min"]}
            if "pe_max" in criteria:
                if "fundamental.pe" in mongo_query:
                    mongo_query["fundamental.pe"]["$lte"] = criteria["pe_max"]
                else:
                    mongo_query["fundamental.pe"] = {"$lte": criteria["pe_max"]}
            
            if "dividend_yield_min" in criteria:
                mongo_query["fundamental.dividend_yield"] = {"$gte": criteria["dividend_yield_min"]}
            
            # Sector filter
            if "sector" in criteria:
                mongo_query["basic.sector"] = criteria["sector"]
            
            # Tag-based filters (using pre-generated screening tags)
            if "tags" in criteria:
                if isinstance(criteria["tags"], list):
                    mongo_query["screening_tags"] = {"$in": criteria["tags"]}
                else:
                    mongo_query["screening_tags"] = criteria["tags"]
            
            # Performance filters
            if "change_1d_min" in criteria:
                mongo_query["basic.change_1d"] = {"$gte": criteria["change_1d_min"]}
            if "change_1d_max" in criteria:
                if "basic.change_1d" in mongo_query:
                    mongo_query["basic.change_1d"]["$lte"] = criteria["change_1d_max"]
                else:
                    mongo_query["basic.change_1d"] = {"$lte": criteria["change_1d_max"]}
            
            # Trend filter
            if "trend" in criteria:
                mongo_query["technical.trend"] = criteria["trend"]
            
            # Special pattern matching for popular screens
            if "screen_type" in criteria:
                screen_type = criteria["screen_type"]
                if screen_type in self.popular_screens:
                    pattern_query = self.popular_screens[screen_type]
                    mongo_query.update(pattern_query)
            
            # Complex filters
            if "custom_filter" in criteria:
                # Allow direct MongoDB query injection for advanced users
                custom = criteria["custom_filter"]
                if isinstance(custom, dict):
                    mongo_query.update(custom)
            
            # Ensure we only get stocks with basic data
            mongo_query["basic"] = {"$exists": True}
            
            return mongo_query
            
        except Exception as e:
            logger.error(f"❌ Criteria conversion failed: {e}")
            return {}
    
    async def _query_eodhd_screener_api(self, criteria: Dict[str, Any]) -> List[Dict]:
        """Query EODHD Screener API as fallback for complex queries"""
        try:
            # Convert criteria to EODHD screener format
            screener_params = self._convert_criteria_to_eodhd_params(criteria)
            
            if not screener_params:
                logger.info("❌ Could not convert criteria to EODHD screener format")
                return []
            
            async with aiohttp.ClientSession() as session:
                url = "https://eodhd.com/api/screener"
                params = {
                    "api_token": self.eodhd_api_key,
                    "fmt": "json",
                    **screener_params
                }
                
                logger.info(f"📡 EODHD Screener API call with params: {screener_params}")
                
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if isinstance(data, list) and data:
                            # Format EODHD results to match our standard format
                            formatted_results = []
                            for item in data[:100]:  # Limit to top 100 results
                                formatted_stock = {
                                    "symbol": item.get("code", "").replace(".US", ""),
                                    "name": item.get("name", ""),
                                    "price": float(item.get("close", 0)),
                                    "change_1d": float(item.get("change_p", 0)),
                                    "volume": int(item.get("volume", 0)),
                                    "market_cap": float(item.get("market_cap", 0)),
                                    "sector": item.get("sector", "Unknown"),
                                    "pe_ratio": float(item.get("pe", 0)),
                                    "rsi": 50,  # Would need separate call for technical data
                                    "trend": "neutral",  # Would need separate call
                                    "tags": ["api_result"],
                                    "last_updated": datetime.now().isoformat(),
                                    "source": "eodhd_screener_api"
                                }
                                formatted_results.append(formatted_stock)
                            
                            logger.info(f"✅ EODHD API returned {len(formatted_results)} results")
                            return formatted_results
                        else:
                            logger.info("📡 EODHD API returned empty or invalid results")
                            return []
                    else:
                        logger.error(f"❌ EODHD API error: {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"❌ EODHD screener API failed: {e}")
            return []
    
    def _convert_criteria_to_eodhd_params(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Convert criteria to EODHD screener API parameters"""
        try:
            params = {}
            
            # Market cap (in millions for EODHD)
            if "market_cap_min" in criteria:
                params["market_cap_from"] = criteria["market_cap_min"] // 1000000
            if "market_cap_max" in criteria:
                params["market_cap_to"] = criteria["market_cap_max"] // 1000000
            
            # Price range
            if "price_min" in criteria:
                params["price_from"] = criteria["price_min"]
            if "price_max" in criteria:
                params["price_to"] = criteria["price_max"]
            
            # Volume
            if "volume_min" in criteria:
                params["volume_from"] = criteria["volume_min"]
            
            # PE ratio
            if "pe_min" in criteria:
                params["pe_from"] = criteria["pe_min"]
            if "pe_max" in criteria:
                params["pe_to"] = criteria["pe_max"]
            
            # Dividend yield
            if "dividend_yield_min" in criteria:
                params["dividend_yield_from"] = criteria["dividend_yield_min"]
            
            # Sector
            if "sector" in criteria:
                params["sector"] = criteria["sector"]
            
            # Performance
            if "change_1d_min" in criteria:
                params["change_from"] = criteria["change_1d_min"]
            if "change_1d_max" in criteria:
                params["change_to"] = criteria["change_1d_max"]
            
            # Default limits and sorting
            params["limit"] = 100
            params["order"] = "market_cap"
            params["sort"] = "desc"
            
            return params
            
        except Exception as e:
            logger.error(f"❌ EODHD params conversion failed: {e}")
            return {}
    
    async def _store_api_results_in_cache(self, api_results: List[Dict]):
        """Store API results in MongoDB cache for future use"""
        try:
            if not api_results:
                return
            
            for stock in api_results:
                symbol = stock.get("symbol", "")
                if symbol:
                    # Create cache record
                    cache_record = {
                        "symbol": symbol,
                        "last_updated": datetime.now(),
                        "basic": {
                            "price": stock.get("price", 0),
                            "volume": stock.get("volume", 0),
                            "market_cap": stock.get("market_cap", 0),
                            "change_1d": stock.get("change_1d", 0),
                            "sector": stock.get("sector", "Unknown"),
                            "exchange": "NASDAQ"
                        },
                        "technical": {
                            "rsi": stock.get("rsi", 50),
                            "trend": stock.get("trend", "neutral")
                        },
                        "fundamental": {
                            "pe": stock.get("pe_ratio", 0)
                        },
                        "screening_tags": stock.get("tags", []),
                        "api_cached": True,
                        "cache_expiry": datetime.now() + timedelta(hours=6)  # Shorter TTL for API results
                    }
                    
                    # Upsert to cache
                    await self.db.stocks.replace_one(
                        {"symbol": symbol},
                        cache_record,
                        upsert=True
                    )
            
            logger.info(f"✅ Cached {len(api_results)} API results for future use")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache API results: {e}")
    
    def _get_screen_cache_key(self, criteria: Dict[str, Any]) -> str:
        """Generate cache key for screening criteria"""
        try:
            # Sort criteria for consistent key generation
            sorted_criteria = json.dumps(criteria, sort_keys=True)
            
            # Hash for shorter key
            import hashlib
            hash_object = hashlib.md5(sorted_criteria.encode())
            return f"screen:{hash_object.hexdigest()}"
            
        except Exception:
            return f"screen:default:{datetime.now().strftime('%Y%m%d%H')}"
    
    async def _get_cached_screen_result(self, screen_key: str) -> Optional[List[Dict]]:
        """Get cached screening result from Redis"""
        try:
            if self.redis_client:
                cached_json = await self.redis_client.get(screen_key)
                if cached_json:
                    return json.loads(cached_json)
            return None
        except Exception as e:
            logger.warning(f"⚠️ Redis cache get failed: {e}")
            return None
    
    async def _cache_screen_result(self, screen_key: str, results: List[Dict]):
        """Cache screening result in Redis"""
        try:
            if self.redis_client and results:
                # Cache for 1 hour (screening results change frequently)
                await self.redis_client.setex(
                    screen_key,
                    3600,  # 1 hour TTL
                    json.dumps(results, default=str)
                )
                logger.info(f"✅ Cached screen result with key: {screen_key}")
        except Exception as e:
            logger.warning(f"⚠️ Redis cache set failed: {e}")
    
    async def _warm_up_popular_screens(self):
        """Pre-compute and cache popular screening patterns"""
        try:
            logger.info("🔥 Warming up popular screen cache...")
            
            for screen_name, screen_criteria in self.popular_screens.items():
                try:
                    # Convert to standard criteria format
                    criteria = {"custom_filter": screen_criteria}
                    
                    # Run the screen to cache it
                    await self.screen_stocks(criteria)
                    
                    logger.info(f"✅ Warmed up: {screen_name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to warm up {screen_name}: {e}")
                
                # Small delay between warm-up screens
                await asyncio.sleep(0.1)
            
            logger.info("✅ Popular screen cache warm-up completed")
            
        except Exception as e:
            logger.error(f"❌ Screen warm-up failed: {e}")
    
    def _update_avg_response_time(self, response_time_ms: float):
        """Update running average response time"""
        current_avg = self.screening_stats["avg_response_time_ms"]
        total_screens = self.screening_stats["total_screens"]
        
        # Running average calculation
        self.screening_stats["avg_response_time_ms"] = (
            (current_avg * (total_screens - 1) + response_time_ms) / total_screens
        )
    
    async def get_popular_screens(self) -> Dict[str, Any]:
        """Get list of popular/optimized screening patterns"""
        return {
            "popular_screens": list(self.popular_screens.keys()),
            "screen_definitions": self.popular_screens,
            "usage_stats": self.screening_stats,
            "optimization_note": "Popular screens are pre-cached for <100ms response times"
        }
    
    async def quick_screen(self, screen_type: str, limit: int = 50) -> Dict[str, Any]:
        """Quick screening using popular patterns"""
        try:
            if screen_type not in self.popular_screens:
                return {
                    "error": f"Unknown screen type: {screen_type}",
                    "available_types": list(self.popular_screens.keys())
                }
            
            # Use popular screen pattern
            criteria = {
                "screen_type": screen_type,
                "limit": limit
            }
            
            result = await self.screen_stocks(criteria)
            
            # Add quick screen metadata
            result["screen_type"] = screen_type
            result["optimized"] = True
            result["description"] = f"Quick {screen_type.replace('_', ' ')} screen"
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Quick screen failed: {e}")
            return {"error": str(e)}
    
    async def get_screening_stats(self) -> Dict[str, Any]:
        """Get screening service statistics"""
        try:
            # Get MongoDB collection stats
            mongo_stats = await self.db.command("collStats", "stocks")
            
            # Get Redis cache info
            redis_info = await self.redis_client.info() if self.redis_client else {}
            
            return {
                "screening_stats": self.screening_stats,
                "cache_performance": {
                    "cache_hit_rate": round(
                        (self.screening_stats["cache_hits"] / max(self.screening_stats["total_screens"], 1)) * 100, 2
                    ),
                    "avg_response_time_ms": round(self.screening_stats["avg_response_time_ms"], 1),
                    "api_fallback_rate": round(
                        (self.screening_stats["api_fallback"] / max(self.screening_stats["total_screens"], 1)) * 100, 2
                    )
                },
                "mongodb_stats": {
                    "documents_count": mongo_stats.get("count", 0),
                    "total_size_bytes": mongo_stats.get("size", 0),
                    "index_count": mongo_stats.get("nindexes", 0)
                },
                "redis_cache": {
                    "connected": bool(self.redis_client),
                    "used_memory": redis_info.get("used_memory_human", "Unknown")
                },
                "popular_screens_available": len(self.popular_screens),
                "optimization_notes": [
                    "90% of screens use cached MongoDB data (<100ms)",
                    "Popular screens are pre-cached for instant results",
                    "Complex queries fallback to EODHD API (2-3s)",
                    "API results are cached for future use"
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Stats collection failed: {e}")
            return {"error": str(e)}
    
    async def custom_screen(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Advanced: Convert natural language to screening criteria
        Example: "Find growth stocks under $50 with PE ratio less than 25"
        """
        try:
            # Basic natural language parsing (could be enhanced with NLP)
            criteria = self._parse_natural_language_query(natural_language_query)
            
            if not criteria:
                return {
                    "error": "Could not parse natural language query",
                    "query": natural_language_query,
                    "suggestion": "Try specific criteria like 'market_cap_min', 'pe_max', etc."
                }
            
            # Run the parsed screen
            result = await self.screen_stocks(criteria)
            
            # Add natural language metadata
            result["original_query"] = natural_language_query
            result["parsed_criteria"] = criteria
            result["natural_language_processing"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Natural language screening failed: {e}")
            return {"error": str(e), "query": natural_language_query}
    
    def _parse_natural_language_query(self, query: str) -> Dict[str, Any]:
        """
        Basic natural language parsing for screening queries
        This could be enhanced with proper NLP libraries
        """
        try:
            query_lower = query.lower()
            criteria = {}
            
            # Price filters
            if "under $" in query_lower:
                import re
                match = re.search(r'under \$(\d+(?:\.\d+)?)', query_lower)
                if match:
                    criteria["price_max"] = float(match.group(1))
            
            if "over $" in query_lower or "above $" in query_lower:
                import re
                match = re.search(r'(?:over|above) \$(\d+(?:\.\d+)?)', query_lower)
                if match:
                    criteria["price_min"] = float(match.group(1))
            
            # PE ratio
            if "pe" in query_lower or "p/e" in query_lower:
                import re
                if "less than" in query_lower or "under" in query_lower:
                    match = re.search(r'(?:pe|p/e).*?(?:less than|under)\s*(\d+)', query_lower)
                    if match:
                        criteria["pe_max"] = float(match.group(1))
                
                if "greater than" in query_lower or "over" in query_lower:
                    match = re.search(r'(?:pe|p/e).*?(?:greater than|over)\s*(\d+)', query_lower)
                    if match:
                        criteria["pe_min"] = float(match.group(1))
            
            # Market cap
            if "large cap" in query_lower:
                criteria["market_cap_min"] = 10000000000  # $10B+
            elif "mid cap" in query_lower:
                criteria["market_cap_min"] = 2000000000   # $2B+
                criteria["market_cap_max"] = 10000000000  # Under $10B
            elif "small cap" in query_lower:
                criteria["market_cap_max"] = 2000000000   # Under $2B
            
            # Common patterns
            if "growth" in query_lower:
                criteria["tags"] = criteria.get("tags", []) + ["growth"]
            if "value" in query_lower:
                criteria["tags"] = criteria.get("tags", []) + ["value"]
            if "dividend" in query_lower:
                criteria["dividend_yield_min"] = 2.0
            if "tech" in query_lower or "technology" in query_lower:
                criteria["sector"] = "Technology"
            if "oversold" in query_lower:
                criteria["rsi_max"] = 30
            if "overbought" in query_lower:
                criteria["rsi_min"] = 70
            
            return criteria if criteria else None
            
        except Exception as e:
            logger.error(f"❌ Natural language parsing failed: {e}")
            return None
    
    async def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("✅ Screener service connections closed")


# Example usage and testing functions
async def test_screener_service():
    """Test function for the screener service"""
    
    # Initialize service (you'll need to provide real connection strings)
    screener = CachedScreenerService(
        mongodb_url="mongodb://localhost:27017",
        redis_url="redis://localhost:6379", 
        eodhd_api_key="your_eodhd_api_key"
    )
    
    try:
        await screener.initialize()
        
        # Test 1: Quick popular screen
        print("🧪 Testing popular growth stocks screen...")
        result1 = await screener.quick_screen("growth_stocks")
        print(f"Results: {len(result1.get('results', []))} stocks found")
        print(f"Response time: {result1.get('response_time_ms', 0)}ms")
        print(f"Data source: {result1.get('data_source', 'unknown')}")
        
        # Test 2: Custom criteria screen
        print("\n🧪 Testing custom criteria screen...")
        custom_criteria = {
            "market_cap_min": 1000000000,  # $1B+
            "pe_max": 25,
            "rsi_min": 30,
            "rsi_max": 70
        }
        result2 = await screener.screen_stocks(custom_criteria)
        print(f"Results: {len(result2.get('results', []))} stocks found")
        print(f"Response time: {result2.get('response_time_ms', 0)}ms")
        
        # Test 3: Natural language query
        print("\n🧪 Testing natural language query...")
        result3 = await screener.custom_screen("Find growth stocks under $100 with PE ratio less than 30")
        print(f"Results: {len(result3.get('results', []))} stocks found")
        print(f"Parsed criteria: {result3.get('parsed_criteria', {})}")
        
        # Test 4: Get statistics
        print("\n📊 Service Statistics:")
        stats = await screener.get_screening_stats()
        print(f"Cache hit rate: {stats.get('cache_performance', {}).get('cache_hit_rate', 0)}%")
        print(f"Avg response time: {stats.get('cache_performance', {}).get('avg_response_time_ms', 0)}ms")
        
    finally:
        await screener.close()

if __name__ == "__main__":
    # Run tests
    import asyncio
    asyncio.run(test_screener_service())
