# services/cache_service.py - FIXED VERSION

import json
import asyncio
from typing import Optional, Dict, Any, List, Union
from loguru import logger
from datetime import datetime

class CacheService:
    """Cache service that properly handles lists and serialization"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def get(self, key: str) -> Optional[Any]:
        """Get single value from cache"""
        try:
            if not self.redis:
                return None
                
            cached_data = await self.redis.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set single value in cache"""
        try:
            if not self.redis:
                return False
                
            serialized_value = json.dumps(value, default=self._json_serializer)
            await self.redis.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            return False
    
    async def get_list(self, key: str, limit: int = 10) -> List[Any]:
        """Get list from cache (stored as JSON array)"""
        try:
            if not self.redis:
                return []
                
            cached_data = await self.redis.get(key)
            if cached_data:
                data = json.loads(cached_data)
                if isinstance(data, list):
                    return data[:limit]  # Return limited number of items
                else:
                    # If it's not a list, return it as single item
                    return [data]
            return []
            
        except Exception as e:
            logger.error(f"Cache get_list error for {key}: {e}")
            return []
    
    async def add_to_list(self, key: str, items: Union[Any, List[Any]], max_length: int = 10) -> bool:
        """Add items to list in cache"""
        try:
            if not self.redis:
                return False
            
            # Get current list
            current_list = await self.get_list(key, limit=max_length)
            
            # Add new items
            if isinstance(items, list):
                # If items is a list, extend current list
                current_list.extend(items)
            else:
                # If items is a single item, append it
                current_list.append(items)
            
            # Keep only the most recent items
            if len(current_list) > max_length:
                current_list = current_list[-max_length:]
            
            # Store back as JSON
            await self.set(key, current_list, ttl=86400)  # 24 hour TTL
            return True
            
        except Exception as e:
            logger.error(f"Cache add_to_list error for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if not self.redis:
                return False
                
            await self.redis.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
            return False
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
