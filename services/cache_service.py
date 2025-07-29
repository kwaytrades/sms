# services/cache_service.py - NEW FILE NEEDED

import json
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
import redis.asyncio as redis

class CacheService:
    """
    Cache service for fast context retrieval in orchestrator
    Provides methods for storing and retrieving conversation context
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_list(self, key: str, limit: int = 5) -> List[Any]:
        """Get list items from cache"""
        try:
            # Get list items (most recent first)
            items = await self.redis.lrange(key, 0, limit - 1)
            
            # Deserialize JSON items
            result = []
            for item in items:
                try:
                    if isinstance(item, bytes):
                        item = item.decode('utf-8')
                    result.append(json.loads(item))
                except json.JSONDecodeError:
                    # Handle non-JSON items (like simple strings)
                    result.append(item)
            
            return result
            
        except Exception as e:
            logger.warning(f"Cache get_list failed for {key}: {e}")
            return []
    
    async def add_to_list(self, key: str, item: Any, max_length: int = 10) -> None:
        """Add item to list cache with automatic trimming"""
        try:
            # Serialize item to JSON
            if isinstance(item, (dict, list)):
                serialized_item = json.dumps(item)
            else:
                serialized_item = str(item)
            
            # Add to front of list
            await self.redis.lpush(key, serialized_item)
            
            # Trim list to max length
            await self.redis.ltrim(key, 0, max_length - 1)
            
            # Set TTL if this is a new key
            ttl = await self.redis.ttl(key)
            if ttl == -1:  # No TTL set
                await self.redis.expire(key, 86400)  # 24 hours default
                
        except Exception as e:
            logger.warning(f"Cache add_to_list failed for {key}: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get single value from cache"""
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set single value in cache with TTL"""
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            # Set with TTL
            await self.redis.setex(key, ttl, serialized_value)
            
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete key from cache"""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
    
    async def get_keys_pattern(self, pattern: str) -> List[str]:
        """Get all keys matching pattern"""
        try:
            keys = await self.redis.keys(pattern)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.warning(f"Cache get_keys_pattern failed for {pattern}: {e}")
            return []
