# services/cache_service.py - Production Cache Service
"""
Production Cache Service with Market-Aware TTL and Intelligent Invalidation
Handles Redis caching with sophisticated strategies for financial data
"""

import asyncio
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timezone, time as dt_time
from enum import Enum
from dataclasses import dataclass

import redis.asyncio as aioredis
from loguru import logger

from config import settings


class TTLStrategy(Enum):
    """TTL strategies for different data types"""
    MARKET_HOURS = "market_hours"      # Short TTL during market hours, longer after
    STOCK_DATA = "stock_data"          # Popular stocks get longer TTL
    USER_DATA = "user_data"           # User-specific data caching
    STATIC_DATA = "static_data"       # Rarely changing data
    REAL_TIME = "real_time"           # Very short TTL for real-time data
    NEWS_DATA = "news_data"           # News-specific caching


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


class CacheService:
    """
    Production Cache Service with Market Intelligence
    
    Features:
    - Market-aware TTL strategies
    - Popular ticker enhanced caching
    - Batch operations for efficiency
    - Cache warming and preloading
    - Performance metrics and monitoring
    - Intelligent invalidation patterns
    - Fallback mechanisms
    - Compression for large values
    """
    
    def __init__(self, base_service):
        self.base_service = base_service
        self.redis: Optional[aioredis.Redis] = None
        self.metrics = CacheMetrics()
        
        # Market hours (Eastern Time)
        self.market_open = dt_time(9, 30)    # 9:30 AM
        self.market_close = dt_time(16, 0)   # 4:00 PM
        
        # Popular tickers for enhanced caching
        self.popular_tickers = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
            'BRK.B', 'UNH', 'JNJ', 'V', 'JPM', 'PG', 'XOM', 'HD'
        }
        
        # TTL configurations (in seconds)
        self.ttl_configs = {
            TTLStrategy.MARKET_HOURS: {
                "market_open": 300,      # 5 minutes during market hours
                "market_closed": 1800,   # 30 minutes after hours
                "weekend": 3600          # 1 hour on weekends
            },
            TTLStrategy.STOCK_DATA: {
                "popular_market_open": 600,     # 10 minutes for popular stocks
                "popular_market_closed": 1800,  # 30 minutes for popular stocks
                "regular_market_open": 300,     # 5 minutes for regular stocks
                "regular_market_closed": 900,   # 15 minutes for regular stocks
            },
            TTLStrategy.USER_DATA: {
                "profile": 1800,         # 30 minutes for user profiles
                "preferences": 3600,     # 1 hour for preferences
                "usage_stats": 600       # 10 minutes for usage stats
            },
            TTLStrategy.STATIC_DATA: {
                "company_info": 86400,   # 24 hours for company information
                "market_holidays": 86400, # 24 hours for market holidays
                "ticker_lists": 3600     # 1 hour for ticker lists
            },
            TTLStrategy.REAL_TIME: {
                "quotes": 60,            # 1 minute for real-time quotes
                "alerts": 30,            # 30 seconds for alert data
                "news_breaking": 120     # 2 minutes for breaking news
            },
            TTLStrategy.NEWS_DATA: {
                "headlines": 600,        # 10 minutes for news headlines
                "analysis": 1800,        # 30 minutes for news analysis
                "sentiment": 900         # 15 minutes for sentiment data
            }
        }
        
        logger.info("📦 CacheService initialized")

    async def initialize(self):
        """Initialize Redis connection and test functionality"""
        try:
            # Use Redis connection from base service
            self.redis = self.base_service.redis
            
            if not self.redis:
                raise Exception("Redis connection not available from base service")
            
            # Test Redis connection
            await self.redis.ping()
            
            # Initialize metrics tracking
            await self._initialize_metrics()
            
            logger.info("✅ CacheService Redis connection established")
            
        except Exception as e:
            logger.exception(f"❌ CacheService initialization failed: {e}")
            raise

    async def _initialize_metrics(self):
        """Initialize cache metrics tracking"""
        try:
            # Get existing metrics from Redis
            metrics_data = await self.redis.get("cache:metrics")
            if metrics_data:
                metrics_dict = json.loads(metrics_data)
                self.metrics = CacheMetrics(**metrics_dict)
            
            logger.info(f"Cache metrics initialized: {self.metrics.hit_rate:.2%} hit rate")
            
        except Exception as e:
            logger.warning(f"Could not load existing metrics: {e}")
            self.metrics = CacheMetrics()

    async def health_check(self) -> Dict[str, Any]:
        """Health check for cache service"""
        health = {"status": "healthy", "metrics": {}}
        
        try:
            # Test Redis connection
            await self.redis.ping()
            
            # Get Redis info
            redis_info = await self.redis.info()
            
            health["metrics"] = {
                "hit_rate": f"{self.metrics.hit_rate:.2%}",
                "total_operations": self.metrics.hits + self.metrics.misses + self.metrics.sets,
                "redis_memory_used": redis_info.get("used_memory_human", "unknown"),
                "redis_connected_clients": redis_info.get("connected_clients", 0),
                "redis_uptime": redis_info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health

    def _is_market_open(self, dt: datetime = None) -> bool:
        """Check if market is currently open (simplified - doesn't account for holidays)"""
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        # Convert to Eastern Time (approximate)
        et_time = dt.replace(tzinfo=timezone.utc).astimezone()
        
        # Check if weekday
        if et_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within market hours
        current_time = et_time.time()
        return self.market_open <= current_time <= self.market_close

    def _get_market_context(self) -> str:
        """Get current market context for TTL calculation"""
        now = datetime.now(timezone.utc)
        
        if now.weekday() >= 5:  # Weekend
            return "weekend"
        elif self._is_market_open(now):
            return "market_open"
        else:
            return "market_closed"

    def _calculate_ttl(self, strategy: TTLStrategy, context: Dict = None) -> int:
        """Calculate TTL based on strategy and context"""
        try:
            if strategy not in self.ttl_configs:
                return 300  # Default 5 minutes
            
            config = self.ttl_configs[strategy]
            market_context = self._get_market_context()
            
            if strategy == TTLStrategy.MARKET_HOURS:
                return config.get(market_context, 300)
            
            elif strategy == TTLStrategy.STOCK_DATA:
                symbol = context.get("symbol", "").upper() if context else ""
                is_popular = symbol in self.popular_tickers
                
                if is_popular:
                    key = f"popular_{market_context}"
                else:
                    key = f"regular_{market_context}"
                
                return config.get(key, 300)
            
            elif strategy == TTLStrategy.USER_DATA:
                data_type = context.get("type", "profile") if context else "profile"
                return config.get(data_type, 1800)
            
            elif strategy == TTLStrategy.STATIC_DATA:
                data_type = context.get("type", "company_info") if context else "company_info"
                return config.get(data_type, 86400)
            
            elif strategy == TTLStrategy.REAL_TIME:
                data_type = context.get("type", "quotes") if context else "quotes"
                return config.get(data_type, 60)
            
            elif strategy == TTLStrategy.NEWS_DATA:
                data_type = context.get("type", "headlines") if context else "headlines"
                return config.get(data_type, 600)
            
            else:
                return 300  # Default fallback
                
        except Exception as e:
            logger.warning(f"Error calculating TTL: {e}")
            return 300  # Safe default

    async def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage with compression for large objects"""
        try:
            if isinstance(value, (str, int, float, bool)):
                # Simple types as JSON
                return json.dumps(value).encode('utf-8')
            else:
                # Complex types with pickle (more efficient for large objects)
                serialized = pickle.dumps(value)
                
                # Compress if large (>1KB)
                if len(serialized) > 1024:
                    import gzip
                    compressed = gzip.compress(serialized)
                    return b"gzip:" + compressed
                else:
                    return b"pickle:" + serialized
                    
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            raise

    async def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from Redis storage"""
        try:
            if data.startswith(b"gzip:"):
                import gzip
                compressed_data = data[5:]  # Remove "gzip:" prefix
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b"pickle:"):
                pickle_data = data[7:]  # Remove "pickle:" prefix
                return pickle.loads(pickle_data)
            else:
                # Assume JSON for backward compatibility
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error deserializing value: {e}")
            raise

    async def set_cache(self, key: str, value: Any, ttl: int = None,
                       strategy: TTLStrategy = None, context: Dict = None) -> bool:
        """Set cache value with optional TTL strategy"""
        try:
            # Calculate TTL if strategy provided
            if strategy and ttl is None:
                ttl = self._calculate_ttl(strategy, context)
            
            if ttl is None:
                ttl = 300  # Default 5 minutes
            
            # Serialize value
            serialized_value = await self._serialize_value(value)
            
            # Set in Redis
            await self.redis.setex(key, ttl, serialized_value)
            
            # Update metrics
            self.metrics.sets += 1
            await self._update_metrics()
            
            logger.debug(f"✅ Cache set: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error setting cache {key}: {e}")
            return False

    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            data = await self.redis.get(key)
            
            if data is None:
                self.metrics.misses += 1
                await self._update_metrics()
                return None
            
            # Deserialize value
            value = await self._deserialize_value(data)
            
            # Update metrics
            self.metrics.hits += 1
            await self._update_metrics()
            
            logger.debug(f"✅ Cache hit: {key}")
            return value
            
        except Exception as e:
            logger.exception(f"❌ Error getting cache {key}: {e}")
            self.metrics.misses += 1
            await self._update_metrics()
            return None

    async def delete_cache(self, key: str) -> bool:
        """Delete cache key"""
        try:
            result = await self.redis.delete(key)
            
            if result > 0:
                self.metrics.deletes += 1
                await self._update_metrics()
                logger.debug(f"✅ Cache deleted: {key}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.exception(f"❌ Error deleting cache {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if cache key exists"""
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.exception(f"❌ Error checking cache existence {key}: {e}")
            return False

    async def get_ttl(self, key: str) -> int:
        """Get TTL for cache key"""
        try:
            return await self.redis.ttl(key)
        except Exception as e:
            logger.exception(f"❌ Error getting TTL for {key}: {e}")
            return -1

    # Market-aware methods
    async def set_market_aware(self, key: str, value: Any, 
                              ttl_strategy: str = "market_hours",
                              context: Dict = None) -> bool:
        """Set cache with market-aware TTL"""
        try:
            strategy = TTLStrategy(ttl_strategy)
            return await self.set_cache(key, value, strategy=strategy, context=context)
        except ValueError:
            logger.warning(f"Unknown TTL strategy: {ttl_strategy}")
            return await self.set_cache(key, value, ttl=300)  # Default fallback

    async def get_with_fallback(self, key: str, fallback_func: Callable,
                               cache_ttl: int = 300, 
                               fallback_args: tuple = None,
                               fallback_kwargs: dict = None) -> Any:
        """Get cached value with fallback function"""
        try:
            # Try cache first
            cached_value = await self.get_cache(key)
            if cached_value is not None:
                return cached_value
            
            # Execute fallback function
            if fallback_args is None:
                fallback_args = ()
            if fallback_kwargs is None:
                fallback_kwargs = {}
            
            if asyncio.iscoroutinefunction(fallback_func):
                fresh_value = await fallback_func(*fallback_args, **fallback_kwargs)
            else:
                fresh_value = fallback_func(*fallback_args, **fallback_kwargs)
            
            # Cache the fresh value
            if fresh_value is not None:
                await self.set_cache(key, fresh_value, ttl=cache_ttl)
            
            return fresh_value
            
        except Exception as e:
            logger.exception(f"❌ Error in get_with_fallback for {key}: {e}")
            return None

    # Batch operations
    async def set_many(self, key_value_pairs: Dict[str, Any], ttl: int = 300) -> Dict[str, bool]:
        """Set multiple cache values"""
        results = {}
        
        try:
            # Use pipeline for efficiency
            pipeline = self.redis.pipeline()
            
            serialized_pairs = {}
            for key, value in key_value_pairs.items():
                try:
                    serialized_value = await self._serialize_value(value)
                    serialized_pairs[key] = serialized_value
                    pipeline.setex(key, ttl, serialized_value)
                except Exception as e:
                    logger.error(f"Error serializing {key}: {e}")
                    results[key] = False
            
            # Execute pipeline
            pipeline_results = await pipeline.execute()
            
            # Process results
            for i, (key, _) in enumerate(serialized_pairs.items()):
                results[key] = pipeline_results[i] is True
                if results[key]:
                    self.metrics.sets += 1
            
            await self._update_metrics()
            logger.debug(f"✅ Batch set: {len(results)} keys")
            
        except Exception as e:
            logger.exception(f"❌ Error in batch set: {e}")
            for key in key_value_pairs.keys():
                results[key] = False
        
        return results

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple cache values"""
        results = {}
        
        try:
            # Use pipeline for efficiency
            pipeline = self.redis.pipeline()
            for key in keys:
                pipeline.get(key)
            
            pipeline_results = await pipeline.execute()
            
            # Process results
            for i, key in enumerate(keys):
                data = pipeline_results[i]
                if data is not None:
                    try:
                        results[key] = await self._deserialize_value(data)
                        self.metrics.hits += 1
                    except Exception as e:
                        logger.error(f"Error deserializing {key}: {e}")
                        results[key] = None
                        self.metrics.misses += 1
                else:
                    results[key] = None
                    self.metrics.misses += 1
            
            await self._update_metrics()
            logger.debug(f"✅ Batch get: {len(keys)} keys, {sum(1 for v in results.values() if v is not None)} hits")
            
        except Exception as e:
            logger.exception(f"❌ Error in batch get: {e}")
            for key in keys:
                results[key] = None
                self.metrics.misses += 1
        
        return results

    # Invalidation methods
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        try:
            keys = []
            cursor = 0
            
            # Scan for matching keys
            while True:
                cursor, partial_keys = await self.redis.scan(cursor, match=pattern, count=100)
                keys.extend(partial_keys)
                if cursor == 0:
                    break
            
            if keys:
                deleted_count = await self.redis.delete(*keys)
                self.metrics.deletes += deleted_count
                await self._update_metrics()
                logger.info(f"✅ Invalidated {deleted_count} keys matching pattern: {pattern}")
                return deleted_count
            else:
                return 0
                
        except Exception as e:
            logger.exception(f"❌ Error invalidating pattern {pattern}: {e}")
            return 0

    async def invalidate_symbol(self, symbol: str) -> bool:
        """Invalidate all cache entries for a symbol"""
        symbol = symbol.upper()
        patterns = [
            f"stock:{symbol}:*",
            f"analysis:{symbol}:*",
            f"news:{symbol}:*",
            f"quote:{symbol}",
            f"chart:{symbol}:*"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            deleted = await self.invalidate_pattern(pattern)
            total_deleted += deleted
        
        logger.info(f"✅ Invalidated {total_deleted} cache entries for {symbol}")
        return total_deleted > 0

    async def burst_invalidate(self, symbols: List[str]) -> Dict[str, bool]:
        """Invalidate cache for multiple symbols (volatility spike scenario)"""
        results = {}
        
        for symbol in symbols:
            try:
                success = await self.invalidate_symbol(symbol)
                results[symbol] = success
            except Exception as e:
                logger.error(f"Error invalidating {symbol}: {e}")
                results[symbol] = False
        
        logger.info(f"Burst invalidation results: {results}")
        return results

    # Cache warming
    async def warm_popular_tickers(self, data_fetcher: Callable) -> Dict[str, bool]:
        """Pre-warm cache for popular tickers"""
        results = {}
        
        for symbol in self.popular_tickers:
            try:
                # Fetch fresh data
                data = await data_fetcher(symbol) if asyncio.iscoroutinefunction(data_fetcher) else data_fetcher(symbol)
                
                if data:
                    # Cache with extended TTL for popular stocks
                    cache_key = f"stock:{symbol}:warmed"
                    success = await self.set_market_aware(
                        cache_key, data, "stock_data", {"symbol": symbol}
                    )
                    results[symbol] = success
                else:
                    results[symbol] = False
                    
            except Exception as e:
                logger.error(f"Error warming cache for {symbol}: {e}")
                results[symbol] = False
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Cache warming completed: {successful}/{len(self.popular_tickers)} successful")
        
        return results

    async def _update_metrics(self):
        """Update cache metrics in Redis"""
        try:
            metrics_data = {
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "sets": self.metrics.sets,
                "deletes": self.metrics.deletes,
                "evictions": self.metrics.evictions,
                "last_updated": time.time()
            }
            
            await self.redis.setex("cache:metrics", 86400, json.dumps(metrics_data))
            
        except Exception as e:
            logger.warning(f"Could not update cache metrics: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current cache metrics"""
        return {
            "hit_rate": f"{self.metrics.hit_rate:.2%}",
            "total_hits": self.metrics.hits,
            "total_misses": self.metrics.misses,
            "total_sets": self.metrics.sets,
            "total_deletes": self.metrics.deletes,
            "total_operations": self.metrics.hits + self.metrics.misses + self.metrics.sets
        }

    async def flush_all(self) -> bool:
        """Flush all cache data (use with caution)"""
        try:
            await self.redis.flushdb()
            self.metrics = CacheMetrics()
            logger.warning("🗑️ All cache data flushed")
            return True
        except Exception as e:
            logger.exception(f"❌ Error flushing cache: {e}")
            return False
