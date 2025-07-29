# services/technical_analysis.py - MODIFIED for cache-first hybrid approach

import pandas as pd
import numpy as np
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from loguru import logger
import os
import json
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis

class HybridTechnicalAnalysisService:
    """
    Hybrid TA Service - Cache First with Live API Fallback:
    1. Check background cache (MongoDB/Redis) - <100ms
    2. If cache miss, use live EODHD API - 3-5s
    3. Cache the live result for future requests
    4. Supports both cached (800+ stocks) and any stock via API
    """
    
    def __init__(self, mongodb_url: str = None, redis_url: str = None):
        # Original API-based functionality
        self.eodhd_api_key = os.getenv('EODHD_API_KEY')
        self.memory_cache = {}
        self.memory_cache_ttl = 300  # 5 minutes
        
        # New background cache connections
        self.mongodb_url = mongodb_url or os.getenv('MONGODB_URL')
        self.redis_url = redis_url or os.getenv('REDIS_URL')
        self.mongo_client = None
        self.db = None
        self.redis_client = None
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "background_cache_hits": 0,
            "memory_cache_hits": 0,
            "live_api_calls": 0,
            "avg_response_time_ms": 0
        }
        
        logger.info(f"✅ Hybrid TA Service initialized")
        logger.info(f"   EODHD API: {'Set' if self.eodhd_api_key else 'Not Set'}")
        logger.info(f"   Background Cache: {'Enabled' if self.mongodb_url else 'Disabled'}")
    
    async def initialize(self):
        """Initialize background cache connections"""
        try:
            if self.mongodb_url and self.redis_url:
                # MongoDB connection
                self.mongo_client = AsyncIOMotorClient(self.mongodb_url)
                self.db = self.mongo_client.trading_bot
                
                # Redis connection
                self.redis_client = aioredis.from_url(self.redis_url)
                
                # Test connections
                await self.db.command("ping")
                await self.redis_client.ping()
                
                logger.info("✅ Background cache connections established")
            else:
                logger.info("📡 Running in API-only mode (no background cache)")
                
        except Exception as e:
            logger.warning(f"⚠️ Background cache unavailable, using API-only mode: {e}")
            self.mongo_client = None
            self.db = None
            self.redis_client = None
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Main analysis function - Cache first, API fallback
        
        Order of operations:
        1. Check background cache (Redis/MongoDB) - ~50ms
        2. Check memory cache - ~1ms  
        3. Make live API call - ~3-5s
        4. Cache result in both systems
        """
        start_time = datetime.now()
        symbol = symbol.upper()
        
        try:
            self.stats["total_requests"] += 1
            
            # Step 1: Check background cache first (fastest for popular stocks)
            if self.redis_client or self.db:
                background_result = await self._get_from_background_cache(symbol)
                if background_result:
                    self.stats["background_cache_hits"] += 1
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_avg_response_time(response_time)
                    
                    logger.info(f"⚡ Background cache HIT for {symbol} in {response_time:.1f}ms")
                    
                    # Add metadata
                    background_result.update({
                        "data_source": "background_cache",
                        "response_time_ms": round(response_time, 1),
                        "cache_level": "background",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    return background_result
            
            # Step 2: Check memory cache (existing logic)
            memory_result = self._get_from_memory_cache(symbol)
            if memory_result:
                self.stats["memory_cache_hits"] += 1
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                logger.info(f"💾 Memory cache HIT for {symbol} in {response_time:.1f}ms")
                
                # Add metadata
                memory_result.update({
                    "data_source": "memory_cache",
                    "response_time_ms": round(response_time, 1),
                    "cache_level": "memory",
                    "timestamp": datetime.now().isoformat()
                })
                
                return memory_result
            
            # Step 3: Cache miss - use live API (your existing logic)
            self.stats["live_api_calls"] += 1
            
            logger.info(f"📡 Cache MISS for {symbol} - fetching from live API")
            
            live_result = await self._analyze_with_live_api(symbol)
            
            if live_result and not live_result.get('error'):
                # Step 4: Cache the result in both systems
                await self._cache_live_result(symbol, live_result)
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                # Add metadata
                live_result.update({
                    "data_source": "live_api",
                    "response_time_ms": round(response_time, 1),
                    "cache_level": "none",
                    "timestamp": datetime.now().isoformat(),
                    "note": "Fresh calculation cached for future requests"
                })
                
                logger.info(f"✅ Live API success for {symbol} in {response_time:.1f}ms")
            else:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"❌ Live API failed for {symbol}")
            
            return live_result
            
        except Exception as e:
            logger.error(f"❌ TA analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'data_source': 'error',
                'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
    
    async def _get_from_background_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get TA data from background cache (Redis/MongoDB)"""
        try:
            # Try Redis first (fastest)
            if self.redis_client:
                redis_data = await self._get_redis_ta_data(symbol)
                if redis_data:
                    return self._format_background_cache_result(symbol, redis_data, "redis")
            
            # Try MongoDB (daily pipeline data)
            if self.db:
                mongodb_data = await self._get_mongodb_ta_data(symbol)
                if mongodb_data:
                    # Promote to Redis for next time
                    if self.redis_client:
                        await self._promote_to_redis_cache(symbol, mongodb_data)
                    
                    return self._format_background_cache_result(symbol, mongodb_data, "mongodb")
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Background cache error for {symbol}: {e}")
            return None
    
    async def _get_redis_ta_data(self, symbol: str) -> Optional[Dict]:
        """Get TA data from Redis cache"""
        try:
            # Get technical data from Redis hash
            technical_data = await self.redis_client.hgetall(f"stock:{symbol}:technical")
            basic_data = await self.redis_client.hgetall(f"stock:{symbol}:basic")
            
            if technical_data and basic_data:
                # Convert Redis data back to proper format
                converted_tech = {}
                for key, value in technical_data.items():
                    key = key.decode() if isinstance(key, bytes) else key
                    value = value.decode() if isinstance(value, bytes) else value
                    
                    if key in ['rsi', 'sma_20', 'sma_50', 'sma_200', 'volatility']:
                        try:
                            converted_tech[key] = float(value)
                        except (ValueError, TypeError):
                            converted_tech[key] = value
                    elif key in ['support_levels', 'resistance_levels']:
                        try:
                            converted_tech[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            converted_tech[key] = []
                    else:
                        converted_tech[key] = value
                
                converted_basic = {}
                for key, value in basic_data.items():
                    key = key.decode() if isinstance(key, bytes) else key
                    value = value.decode() if isinstance(value, bytes) else value
                    
                    if key in ['price', 'volume', 'change_1d', 'high', 'low', 'open']:
                        try:
                            converted_basic[key] = float(value)
                        except (ValueError, TypeError):
                            converted_basic[key] = value
                    else:
                        converted_basic[key] = value
                
                return {"technical": converted_tech, "basic": converted_basic}
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Redis get failed for {symbol}: {e}")
            return None
    
    async def _get_mongodb_ta_data(self, symbol: str) -> Optional[Dict]:
        """Get TA data from MongoDB (daily pipeline)"""
        try:
            stock_data = await self.db.stocks.find_one(
                {"symbol": symbol},
                {"technical": 1, "basic": 1, "last_updated": 1}
            )
            
            if stock_data and stock_data.get("technical") and stock_data.get("basic"):
                # Check data freshness (within 2 days)
                last_updated = stock_data.get("last_updated", datetime.now())
                if isinstance(last_updated, str):
                    try:
                        last_updated = datetime.fromisoformat(last_updated)
                    except ValueError:
                        last_updated = datetime.now()
                
                age = datetime.now() - last_updated
                if age < timedelta(days=2):
                    return {
                        "technical": stock_data["technical"],
                        "basic": stock_data["basic"]
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB get failed for {symbol}: {e}")
            return None
    
    def _format_background_cache_result(self, symbol: str, cache_data: Dict, cache_source: str) -> Dict[str, Any]:
        """Format background cache data to match API response format"""
        try:
            technical = cache_data.get("technical", {})
            basic = cache_data.get("basic", {})
            
            # Calculate price change if we have current and previous
            current_price = basic.get("price", 0)
            change = basic.get("change_1d", 0)
            change_percent = (change / (current_price - change) * 100) if (current_price - change) > 0 else 0
            
            return {
                "symbol": symbol,
                "price": {
                    "current": round(float(current_price), 2) if current_price else 0,
                    "open": round(float(basic.get("open", 0)), 2),
                    "high": round(float(basic.get("high", 0)), 2),
                    "low": round(float(basic.get("low", 0)), 2),
                    "volume": int(basic.get("volume", 0)),
                    "change": round(float(change), 2),
                    "change_percent": round(float(change_percent), 2)
                },
                "technical_indicators": {
                    "rsi": round(float(technical.get("rsi", 50)), 2),
                    "macd": {
                        "signal": technical.get("macd_signal", "neutral")
                    },
                    "moving_averages": {
                        "ma_20": round(float(technical.get("sma_20", 0)), 2) if technical.get("sma_20") else None,
                        "ma_50": round(float(technical.get("sma_50", 0)), 2) if technical.get("sma_50") else None,
                        "ma_200": round(float(technical.get("sma_200", 0)), 2) if technical.get("sma_200") else None
                    },
                    "trend": technical.get("trend", "neutral"),
                    "support_levels": technical.get("support_levels", []),
                    "resistance_levels": technical.get("resistance_levels", []),
                    "volatility": round(float(technical.get("volatility", 0)), 2) if technical.get("volatility") else None
                },
                "cache_source": cache_source,
                "cached_data": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error formatting cache data for {symbol}: {e}")
            return None
    
    def _get_from_memory_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get from existing memory cache (your original logic)"""
        try:
            cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            if cache_key in self.memory_cache:
                cached_data = self.memory_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.memory_cache_ttl:
                    return cached_data['data']
            return None
        except Exception as e:
            logger.warning(f"⚠️ Memory cache error: {e}")
            return None
    
    async def _analyze_with_live_api(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Use your existing live API analysis logic"""
        try:
            # Use your existing market data fetching
            market_data = self._fetch_market_data_direct(symbol)
            
            if market_data is None or market_data.empty:
                return {
                    'symbol': symbol,
                    'error': 'No market data available',
                    'message': f'Unable to fetch data for {symbol}. Symbol may not be available or markets may be closed.'
                }
            
            # Use your existing technical calculation
            analysis = self._calculate_technical_indicators(market_data, symbol)
            
            # Store in memory cache (your existing logic)
            cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            self.memory_cache[cache_key] = {
                'data': analysis,
                'timestamp': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Live API analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'message': f'Technical error analyzing {symbol}. Please try again.'
            }
    
    async def _cache_live_result(self, symbol: str, result: Dict[str, Any]):
        """Cache live API result in background cache for future requests"""
        try:
            if not result or result.get('error'):
                return
            
            # Extract data for caching
            price_data = result.get("price", {})
            tech_data = result.get("technical_indicators", {})
            
            # Prepare cache data structure
            cache_data = {
                "symbol": symbol,
                "last_updated": datetime.now(),
                "basic": {
                    "price": price_data.get("current", 0),
                    "open": price_data.get("open", 0),
                    "high": price_data.get("high", 0),
                    "low": price_data.get("low", 0),
                    "volume": price_data.get("volume", 0),
                    "change_1d": price_data.get("change", 0),
                    "sector": "Unknown",
                    "exchange": "US"
                },
                "technical": {
                    "rsi": tech_data.get("rsi", 50),
                    "macd_signal": tech_data.get("macd", {}).get("signal", "neutral"),
                    "sma_20": tech_data.get("moving_averages", {}).get("ma_20", 0),
                    "sma_50": tech_data.get("moving_averages", {}).get("ma_50", 0),
                    "trend": "neutral",
                    "volatility": 0,
                    "support_levels": [],
                    "resistance_levels": []
                },
                "api_cached": True,  # Mark as API-cached data
                "cache_source": "live_api"
            }
            
            # Store in MongoDB
            if self.db:
                await self.db.stocks.replace_one(
                    {"symbol": symbol},
                    cache_data,
                    upsert=True
                )
            
            # Store in Redis
            if self.redis_client:
                await self._store_in_redis_cache(symbol, cache_data)
            
            logger.info(f"✅ Cached live result for {symbol}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to cache live result for {symbol}: {e}")
    
    async def _promote_to_redis_cache(self, symbol: str, mongodb_data: Dict):
        """Promote MongoDB data to Redis for faster future access"""
        try:
            if not self.redis_client:
                return
            
            await self._store_in_redis_cache(symbol, {
                "basic": mongodb_data.get("basic", {}),
                "technical": mongodb_data.get("technical", {})
            })
            
            logger.info(f"✅ Promoted {symbol} to Redis cache")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis promotion failed for {symbol}: {e}")
    
    async def _store_in_redis_cache(self, symbol: str, cache_data: Dict):
        """Store data in Redis cache"""
        try:
            pipe = self.redis_client.pipeline()
            
            # Store basic data
            if "basic" in cache_data:
                basic_data = {k: str(v) for k, v in cache_data["basic"].items()}
                pipe.hset(f"stock:{symbol}:basic", mapping=basic_data)
                pipe.expire(f"stock:{symbol}:basic", 7200)  # 2 hours for API-cached data
            
            # Store technical data  
            if "technical" in cache_data:
                tech_data = cache_data["technical"].copy()
                for key, value in tech_data.items():
                    if isinstance(value, (list, dict)):
                        tech_data[key] = json.dumps(value)
                    else:
                        tech_data[key] = str(value)
                
                pipe.hset(f"stock:{symbol}:technical", mapping=tech_data)
                pipe.expire(f"stock:{symbol}:technical", 7200)  # 2 hours for API-cached data
            
            # Store timestamp
            pipe.set(f"stock:{symbol}:last_updated", datetime.now().isoformat())
            pipe.expire(f"stock:{symbol}:last_updated", 7200)
            
            await pipe.execute()
            
        except Exception as e:
            logger.warning(f"⚠️ Redis storage failed for {symbol}: {e}")
    
    def _update_avg_response_time(self, response_time_ms: float):
        """Update running average response time"""
        current_avg = self.stats["avg_response_time_ms"]
        total_requests = self.stats["total_requests"]
        
        self.stats["avg_response_time_ms"] = (
            (current_avg * (total_requests - 1) + response_time_ms) / total_requests
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        total_requests = self.stats["total_requests"]
        
        return {
            "service_mode": "hybrid_cache_first",
            "performance_stats": self.stats,
            "cache_efficiency": {
                "background_cache_hit_rate": round((self.stats["background_cache_hits"] / max(total_requests, 1)) * 100, 2),
                "memory_cache_hit_rate": round((self.stats["memory_cache_hits"] / max(total_requests, 1)) * 100, 2),
                "overall_cache_hit_rate": round(((self.stats["background_cache_hits"] + self.stats["memory_cache_hits"]) / max(total_requests, 1)) * 100, 2),
                "live_api_usage_rate": round((self.stats["live_api_calls"] / max(total_requests, 1)) * 100, 2)
            },
            "response_times": {
                "background_cache": "~50ms",
                "memory_cache": "~1ms", 
                "live_api": "~3-5s",
                "avg_response_time_ms": round(self.stats["avg_response_time_ms"], 1)
            },
            "capabilities": {
                "cached_stocks": "800+ popular stocks (instant)",
                "live_api_stocks": "Any stock symbol (3-5s)",
                "auto_caching": "Live results cached for future requests",
                "background_cache": "Enabled" if self.db else "Disabled"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    # === YOUR EXISTING METHODS (unchanged) ===
    
    def _fetch_market_data_direct(self, symbol: str) -> Optional[pd.DataFrame]:
        """Your existing market data fetching logic - UNCHANGED"""
        if not self.eodhd_api_key:
            logger.error("EODHD API key not available")
            return None
        
        try:
            # Calculate date range (last 30 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Add .US suffix if not present
            if '.' not in symbol:
                symbol_with_exchange = f"{symbol}.US"
            else:
                symbol_with_exchange = symbol
            
            logger.info(f"🔍 Fetching EODHD data for {symbol_with_exchange}")
            
            # Direct API call
            url = f"https://eodhd.com/api/eod/{symbol_with_exchange}"
            params = {
                'api_token': self.eodhd_api_key,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"EODHD API error {response.status_code} for {symbol}: {response.text}")
                return None
            
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response for {symbol}: {e}")
                return None
            
            if not data or not isinstance(data, list):
                logger.warning(f"No valid data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Check required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                # Try with adjusted_close
                if 'adjusted_close' in df.columns:
                    df['close'] = df['adjusted_close']
                else:
                    logger.error(f"Missing required columns for {symbol}. Available: {df.columns.tolist()}")
                    return None
            
            # Process DataFrame
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove NaN rows
            df = df.dropna()
            
            if df.empty:
                logger.error(f"No valid data after cleaning for {symbol}")
                return None
            
            logger.info(f"✅ Fetched {len(df)} days of data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error fetching data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Your existing technical calculation logic - UNCHANGED"""
        try:
            if df.empty or len(df) < 14:
                return {
                    'symbol': symbol,
                    'error': 'Insufficient data for technical analysis'
                }
            
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            # Calculate indicators
            rsi = self._calculate_rsi(df['close'])
            macd_data = self._calculate_macd(df['close'])
            
            # Moving averages
            ma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            # Price change
            price_change = latest['close'] - previous['close']
            price_change_pct = (price_change / previous['close']) * 100
            
            return {
                'symbol': symbol,
                'price': {
                    'current': round(float(latest['close']), 2),
                    'open': round(float(latest['open']), 2),
                    'high': round(float(latest['high']), 2),
                    'low': round(float(latest['low']), 2),
                    'volume': int(latest['volume']) if not pd.isna(latest['volume']) else 0,
                    'change': round(float(price_change), 2),
                    'change_percent': round(float(price_change_pct), 2)
                },
                'technical_indicators': {
                    'rsi': round(float(rsi), 2) if not pd.isna(rsi) else None,
                    'macd': {
                        'macd': round(float(macd_data['macd']), 2) if not pd.isna(macd_data['macd']) else None,
                        'signal': round(float(macd_data['signal']), 2) if not pd.isna(macd_data['signal']) else None,
                        'histogram': round(float(macd_data['histogram']), 2) if not pd.isna(macd_data['histogram']) else None
                    },
                    'moving_averages': {
                        'ma_20': round(float(ma_20), 2) if ma_20 and not pd.isna(ma_20) else None,
                        'ma_50': round(float(ma_50), 2) if ma_50 and not pd.isna(ma_50) else None
                    }
                },
                'timestamp': datetime.now().isoformat(),
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"❌ Technical calculation failed: {e}")
            return {
                'symbol': symbol,
                'error': f'Technical calculation failed: {str(e)}'
            }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Your existing RSI calculation - UNCHANGED"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return np.nan
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Your existing MACD calculation - UNCHANGED"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal_line.iloc[-1], 
                'histogram': histogram.iloc[-1]
            }
        except:
            return {
                'macd': np.nan,
                'signal': np.nan,
                'histogram': np.nan
            }
    
    async def close(self):
        """Cleanup method"""
        self.memory_cache.clear()
        
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("✅ Hybrid Technical Analysis service closed")


# Example usage in your main app
async def setup_hybrid_ta_service():
    """Setup the hybrid TA service"""
    ta_service = HybridTechnicalAnalysisService(
        mongodb_url=os.getenv('MONGODB_URL'),
        redis_url=os.getenv('REDIS_URL')
    )
    
    await ta_service.initialize()
    return ta_service

# Usage example
async def test_hybrid_service():
    ta_service = await setup_hybrid_ta_service()
    
    # This will check cache first, then API
    result = await ta_service.analyze_symbol("AAPL")
    print(f"Data source: {result.get('data_source')}")
    print(f"Response time: {result.get('response_time_ms')}ms")
    
    # Get performance stats
    stats = await ta_service.get_performance_stats()
    print(f"Cache hit rate: {stats['cache_efficiency']['overall_cache_hit_rate']}%")
