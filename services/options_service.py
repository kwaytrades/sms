# services/options_service.py
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class OptionsEngine:
    """
    Options Data Service - Fetches raw options data for LLM analysis
    Just gets the data, lets LLM handle strategy analysis and explanations
    """
    
    def __init__(self, mongodb_url: str, redis_url: str, eodhd_api_key: str):
        self.mongodb_url = mongodb_url
        self.redis_url = redis_url
        self.eodhd_api_key = eodhd_api_key
        
        # Database connections
        self.mongo_client = None
        self.db = None
        self.redis_client = None
        
        # Simple fetch statistics
        self.fetch_stats = {
            "total_fetches": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "avg_response_time_ms": 0
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
            
            logger.info("✅ Options data service initialized - DB connections ready")
            
        except Exception as e:
            logger.error(f"❌ Options data service initialization failed: {e}")
            raise

    async def get_options_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get raw options data for LLM to analyze
        Returns clean, structured data that LLM can easily understand
        """
        start_time = datetime.now()
        
        try:
            self.fetch_stats["total_fetches"] += 1
            symbol = symbol.upper()
            
            # Check cache first
            cache_key = f"options_raw:{symbol}"
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                self.fetch_stats["cache_hits"] += 1
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                return {
                    "symbol": symbol,
                    "success": True,
                    "options_data": cached_result,
                    "data_source": "cached",
                    "response_time_ms": round(response_time, 1),
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": True
                }
            
            # Fetch fresh data from API
            self.fetch_stats["api_calls"] += 1
            raw_options = await self._fetch_raw_options_data(symbol)
            
            if raw_options:
                # Structure data for LLM consumption
                structured_data = self._structure_for_llm(raw_options, symbol)
                
                # Cache for 30 minutes (options change frequently)
                await self._cache_result(cache_key, structured_data, ttl=1800)
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                return {
                    "symbol": symbol,
                    "success": True,
                    "options_data": structured_data,
                    "data_source": "live_api",
                    "response_time_ms": round(response_time, 1),
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": False
                }
            else:
                return {
                    "symbol": symbol,
                    "success": False,
                    "error": f"No options data available for {symbol}",
                    "data_source": "api_failed",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Options data fetch failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "success": False,
                "error": str(e),
                "data_source": "error",
                "timestamp": datetime.now().isoformat()
            }

    async def _fetch_raw_options_data(self, symbol: str) -> Optional[Dict]:
        """Fetch raw options data from EODHD API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://eodhd.com/api/options/{symbol}.US"
                params = {
                    "api_token": self.eodhd_api_key,
                    "fmt": "json"
                }
                
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data if data else None
                    else:
                        logger.warning(f"⚠️ EODHD options API error: {response.status} for {symbol}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Raw options fetch failed for {symbol}: {e}")
            return None

    def _structure_for_llm(self, raw_data: Dict, symbol: str) -> Dict:
        """
        Structure raw options data in a clean format for LLM analysis
        LLM can easily understand this format and provide strategic advice
        """
        try:
            structured = {
                "symbol": symbol,
                "underlying_info": {
                    "current_price": 0,
                    "last_updated": datetime.now().isoformat()
                },
                "expiration_dates": [],
                "calls_summary": [],
                "puts_summary": [],
                "market_context": {
                    "total_call_volume": 0,
                    "total_put_volume": 0,
                    "avg_implied_volatility": 0,
                    "most_active_strikes": []
                }
            }
            
            if not raw_data or "data" not in raw_data:
                return structured
            
            total_call_vol = 0
            total_put_vol = 0
            all_ivs = []
            strike_volumes = {}
            
            # Process each expiration date
            for expiry_date, options in raw_data["data"].items():
                structured["expiration_dates"].append(expiry_date)
                
                # Process calls
                if "calls" in options:
                    for strike_str, call_data in options["calls"].items():
                        try:
                            strike = float(strike_str)
                            volume = call_data.get("volume", 0)
                            iv = call_data.get("impliedVolatility", 0)
                            
                            call_summary = {
                                "strike": strike,
                                "expiry": expiry_date,
                                "last_price": call_data.get("lastPrice", 0),
                                "bid": call_data.get("bid", 0),
                                "ask": call_data.get("ask", 0),
                                "volume": volume,
                                "open_interest": call_data.get("openInterest", 0),
                                "implied_volatility": iv,
                                "in_the_money": False  # Would need current price to determine
                            }
                            
                            structured["calls_summary"].append(call_summary)
                            total_call_vol += volume
                            
                            if iv > 0:
                                all_ivs.append(iv)
                            
                            if strike in strike_volumes:
                                strike_volumes[strike] += volume
                            else:
                                strike_volumes[strike] = volume
                                
                        except (ValueError, TypeError):
                            continue
                
                # Process puts
                if "puts" in options:
                    for strike_str, put_data in options["puts"].items():
                        try:
                            strike = float(strike_str)
                            volume = put_data.get("volume", 0)
                            iv = put_data.get("impliedVolatility", 0)
                            
                            put_summary = {
                                "strike": strike,
                                "expiry": expiry_date,
                                "last_price": put_data.get("lastPrice", 0),
                                "bid": put_data.get("bid", 0),
                                "ask": put_data.get("ask", 0),
                                "volume": volume,
                                "open_interest": put_data.get("openInterest", 0),
                                "implied_volatility": iv,
                                "in_the_money": False  # Would need current price to determine
                            }
                            
                            structured["puts_summary"].append(put_summary)
                            total_put_vol += volume
                            
                            if iv > 0:
                                all_ivs.append(iv)
                            
                            if strike in strike_volumes:
                                strike_volumes[strike] += volume
                            else:
                                strike_volumes[strike] = volume
                                
                        except (ValueError, TypeError):
                            continue
            
            # Calculate market context
            structured["market_context"]["total_call_volume"] = total_call_vol
            structured["market_context"]["total_put_volume"] = total_put_vol
            
            if all_ivs:
                structured["market_context"]["avg_implied_volatility"] = round(sum(all_ivs) / len(all_ivs), 4)
            
            # Most active strikes
            if strike_volumes:
                sorted_strikes = sorted(strike_volumes.items(), key=lambda x: x[1], reverse=True)
                structured["market_context"]["most_active_strikes"] = [
                    {"strike": strike, "total_volume": volume} 
                    for strike, volume in sorted_strikes[:5]
                ]
            
            # Sort options by expiry and strike for easier LLM processing
            structured["calls_summary"].sort(key=lambda x: (x["expiry"], x["strike"]))
            structured["puts_summary"].sort(key=lambda x: (x["expiry"], x["strike"]))
            
            return structured
            
        except Exception as e:
            logger.error(f"❌ Data structuring failed: {e}")
            return {"symbol": symbol, "error": "Data processing failed"}

    async def get_simple_options_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get a simple options summary perfect for SMS/LLM analysis
        Returns just the key data points LLM needs for quick strategy advice
        """
        try:
            full_data = await self.get_options_data(symbol)
            
            if not full_data.get("success"):
                return full_data
            
            options_data = full_data["options_data"]
            
            # Extract just the essential info for LLM
            summary = {
                "symbol": symbol,
                "success": True,
                "summary": {
                    "expiration_dates_available": len(options_data.get("expiration_dates", [])),
                    "nearest_expiry": options_data.get("expiration_dates", [None])[0],
                    "total_calls": len(options_data.get("calls_summary", [])),
                    "total_puts": len(options_data.get("puts_summary", [])),
                    "avg_implied_volatility": options_data.get("market_context", {}).get("avg_implied_volatility", 0),
                    "call_put_volume_ratio": self._calculate_call_put_ratio(options_data),
                    "most_active_strikes": options_data.get("market_context", {}).get("most_active_strikes", [])[:3],
                    "high_volume_calls": self._get_high_volume_options(options_data.get("calls_summary", []), limit=3),
                    "high_volume_puts": self._get_high_volume_options(options_data.get("puts_summary", []), limit=3)
                },
                "data_source": full_data["data_source"],
                "timestamp": full_data["timestamp"]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Simple options summary failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "success": False,
                "error": str(e)
            }

    def _calculate_call_put_ratio(self, options_data: Dict) -> float:
        """Calculate call/put volume ratio"""
        try:
            call_vol = options_data.get("market_context", {}).get("total_call_volume", 0)
            put_vol = options_data.get("market_context", {}).get("total_put_volume", 0)
            
            if put_vol > 0:
                return round(call_vol / put_vol, 2)
            else:
                return 0.0
        except:
            return 0.0

    def _get_high_volume_options(self, options_list: List[Dict], limit: int = 3) -> List[Dict]:
        """Get highest volume options"""
        try:
            sorted_options = sorted(options_list, key=lambda x: x.get("volume", 0), reverse=True)
            return [{
                "strike": opt["strike"],
                "expiry": opt["expiry"],
                "volume": opt["volume"],
                "last_price": opt["last_price"]
            } for opt in sorted_options[:limit] if opt.get("volume", 0) > 0]
        except:
            return []

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result from Redis"""
        try:
            if self.redis_client:
                cached_json = await self.redis_client.get(cache_key)
                if cached_json:
                    return json.loads(cached_json)
            return None
        except Exception as e:
            logger.warning(f"⚠️ Redis cache get failed: {e}")
            return None

    async def _cache_result(self, cache_key: str, data: Dict, ttl: int = 1800):
        """Cache result in Redis"""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(data, default=str)
                )
        except Exception as e:
            logger.warning(f"⚠️ Redis cache set failed: {e}")

    def _update_avg_response_time(self, response_time_ms: float):
        """Update running average response time"""
        current_avg = self.fetch_stats["avg_response_time_ms"]
        total_fetches = self.fetch_stats["total_fetches"]
        
        self.fetch_stats["avg_response_time_ms"] = (
            (current_avg * (total_fetches - 1) + response_time_ms) / total_fetches
        )

    async def get_fetch_stats(self) -> Dict[str, Any]:
        """Get options data fetch statistics"""
        return {
            "fetch_stats": self.fetch_stats,
            "cache_performance": {
                "cache_hit_rate": round(
                    (self.fetch_stats["cache_hits"] / max(self.fetch_stats["total_fetches"], 1)) * 100, 2
                ),
                "avg_response_time_ms": round(self.fetch_stats["avg_response_time_ms"], 1)
            },
            "service_type": "options_data_fetcher",
            "analysis_approach": "LLM-powered strategic analysis",
            "features": [
                "Raw options data fetching",
                "Intelligent caching",
                "LLM-friendly data formatting",
                "Simple summary generation"
            ]
        }

    async def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("✅ Options data service connections closed")
