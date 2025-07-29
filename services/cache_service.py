# services/cache_service.py - FIXED VERSION FOR REDIS TYPE CONFLICTS

import json
import asyncio
from typing import Optional, Dict, Any, List, Union
from loguru import logger
from datetime import datetime

class CacheService:
    """Cache service that properly handles lists and Redis type conflicts"""
    
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
        """Get list from cache with Redis type conflict handling"""
        try:
            if not self.redis:
                return []
            
            # First, try to get as string (JSON format)
            try:
                cached_data = await self.redis.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    if isinstance(data, list):
                        return data[:limit]
                    else:
                        # If it's not a list, return it as single item
                        return [data]
                return []
                
            except Exception as json_error:
                # If JSON parsing fails, might be a Redis list type
                logger.warning(f"JSON parse failed for {key}, trying Redis list operations: {json_error}")
                
                try:
                    # Check if it's a Redis list type
                    key_type = await self.redis.type(key)
                    if key_type == b'list':
                        # Get items from Redis list
                        list_items = await self.redis.lrange(key, 0, limit - 1)
                        return [item.decode('utf-8') if isinstance(item, bytes) else item for item in list_items]
                    else:
                        # Clear the incompatible key and return empty list
                        logger.warning(f"Clearing incompatible cache key {key} of type {key_type}")
                        await self.redis.delete(key)
                        return []
                        
                except Exception as redis_error:
                    logger.error(f"Redis list operation failed for {key}: {redis_error}")
                    # Clear the problematic key
                    try:
                        await self.redis.delete(key)
                    except:
                        pass
                    return []
            
        except Exception as e:
            logger.error(f"Cache get_list error for {key}: {e}")
            return []
    
    async def add_to_list(self, key: str, items: Union[Any, List[Any]], max_length: int = 10) -> bool:
        """Add items to list in cache with conflict prevention"""
        try:
            if not self.redis:
                return False
            
            # Check if key exists and clear if it's the wrong type
            try:
                key_type = await self.redis.type(key)
                if key_type not in [b'none', b'string']:
                    logger.warning(f"Clearing incompatible cache key {key} of type {key_type}")
                    await self.redis.delete(key)
            except Exception as type_check_error:
                logger.warning(f"Type check failed for {key}: {type_check_error}")
                # Try to delete the key to clear any conflicts
                try:
                    await self.redis.delete(key)
                except:
                    pass
            
            # Get current list (this will handle type conflicts)
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
            
            # Store back as JSON string (consistent format)
            await self.set(key, current_list, ttl=86400)  # 24 hour TTL
            return True
            
        except Exception as e:
            logger.error(f"Cache add_to_list error for {key}: {e}")
            # Try to clear the problematic key
            try:
                await self.redis.delete(key)
            except:
                pass
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
    
    async def clear_user_cache(self, phone_number: str) -> bool:
        """Clear all cache keys for a specific user"""
        try:
            if not self.redis:
                return False
            
            patterns = [
                f"recent_messages:{phone_number}",
                f"recent_symbols:{phone_number}",
                f"user_session:{phone_number}"
            ]
            
            for pattern in patterns:
                try:
                    await self.redis.delete(pattern)
                    logger.info(f"Cleared cache key: {pattern}")
                except Exception as e:
                    logger.warning(f"Failed to clear cache key {pattern}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing user cache for {phone_number}: {e}")
            return False
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
