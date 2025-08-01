# services/key_builder.py
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from loguru import logger
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorDatabase


class KeyBuilder:
    """
    Centralized key management system that handles migration from old scattered
    data structure to new clean naming convention while maintaining backward compatibility.
    """
    
    def __init__(self, redis_client: aioredis.Redis, mongo_db: AsyncIOMotorDatabase):
        self.redis = redis_client
        self.db = mongo_db
        self.migration_log = []
        
    # ==========================================
    # STOCK DATA METHODS
    # ==========================================
    
    async def get_stock_technical(self, symbol: str) -> Optional[Dict]:
        """Get technical analysis data for symbol"""
        symbol = symbol.upper()
        
        # Try new format first
        new_key = f"{symbol}:ta"
        data = await self.redis.get(new_key)
        if data:
            logger.debug(f"✅ Found {new_key} in new format")
            return json.loads(data)
        
        # Migrate from old format
        old_key = f"stock:{symbol}:technical"
        data = await self.redis.get(old_key)
        if data:
            logger.info(f"🔄 Migrating {old_key} → {new_key}")
            await self.redis.set(new_key, data, ex=3600)
            self._log_migration(old_key, new_key, "technical_analysis")
            return json.loads(data)
        
        logger.debug(f"❌ No technical data found for {symbol}")
        return None
    
    async def get_stock_fundamental(self, symbol: str) -> Optional[Dict]:
        """Get fundamental analysis data for symbol"""
        symbol = symbol.upper()
        
        # Try new format first
        new_key = f"{symbol}:fa"
        data = await self.redis.get(new_key)
        if data:
            logger.debug(f"✅ Found {new_key} in new format")
            return json.loads(data)
        
        # Migrate from old format
        old_key = f"stock:{symbol}:fundamental"
        data = await self.redis.get(old_key)
        if data:
            logger.info(f"🔄 Migrating {old_key} → {new_key}")
            await self.redis.set(new_key, data, ex=3600)
            self._log_migration(old_key, new_key, "fundamental_analysis")
            return json.loads(data)
        
        logger.debug(f"❌ No fundamental data found for {symbol}")
        return None
    
    async def get_stock_metadata(self, symbol: str, data_type: str = "ta") -> Optional[Dict]:
        """Get metadata for stock data (timestamps, tags, etc.)"""
        symbol = symbol.upper()
        
        # Try new format first
        new_key = f"{symbol}:{data_type}:metadata"
        data = await self.redis.get(new_key)
        if data:
            return json.loads(data)
        
        # Migrate from old scattered keys
        metadata = {}
        
        # Get last_updated
        old_updated_key = f"stock:{symbol}:last_updated"
        last_updated = await self.redis.get(old_updated_key)
        if last_updated:
            metadata["last_updated"] = last_updated.decode() if isinstance(last_updated, bytes) else last_updated
        
        # Get tags
        old_tags_key = f"stock:{symbol}:tags"
        tags_data = await self.redis.get(old_tags_key)
        if tags_data:
            try:
                metadata["tags"] = json.loads(tags_data)
            except:
                metadata["tags"] = tags_data.decode() if isinstance(tags_data, bytes) else tags_data
        
        if metadata:
            metadata["migrated_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"🔄 Migrating metadata for {symbol}:{data_type}")
            await self.redis.set(new_key, json.dumps(metadata), ex=86400)
            
            # Clean up old keys
            await self.redis.delete(old_updated_key)
            await self.redis.delete(old_tags_key)
            
            self._log_migration(f"{old_updated_key}, {old_tags_key}", new_key, "metadata")
            return metadata
        
        return None
    
    async def set_stock_data(self, symbol: str, data_type: str, data: Dict, ttl: int = 3600) -> bool:
        """Set stock data using new naming convention"""
        symbol = symbol.upper()
        key = f"{symbol}:{data_type}"
        
        try:
            await self.redis.set(key, json.dumps(data), ex=ttl)
            logger.debug(f"✅ Set {key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set {key}: {e}")
            return False
    
    # ==========================================
    # USER DATA METHODS
    # ==========================================
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get consolidated user profile"""
        # Try new format first
        new_key = f"user:{user_id}:profile"
        data = await self.redis.get(new_key)
        if data:
            logger.debug(f"✅ Found {new_key} in new format")
            return json.loads(data)
        
        # Migrate from MongoDB
        try:
            user_doc = await self.db.users.find_one({"_id": user_id})
            if not user_doc:
                logger.debug(f"❌ No user found for {user_id}")
                return None
            
            # Build consolidated profile
            profile = {
                "user_id": user_id,
                "phone_number": user_doc.get("phone_number"),
                "email": user_doc.get("email"),
                "first_name": user_doc.get("first_name"),
                "timezone": user_doc.get("timezone", "US/Eastern"),
                "plan_type": user_doc.get("plan_type", "free"),
                "subscription_status": user_doc.get("subscription_status", "trialing"),
                "risk_tolerance": user_doc.get("risk_tolerance", "medium"),
                "trading_experience": user_doc.get("trading_experience", "intermediate"),
                "trading_style": user_doc.get("trading_style", "swing"),
                "preferred_sectors": user_doc.get("preferred_sectors", []),
                "watchlist": user_doc.get("watchlist", []),
                "daily_insights_enabled": user_doc.get("daily_insights_enabled", True),
                "premarket_enabled": user_doc.get("premarket_enabled", True),
                "market_close_enabled": user_doc.get("market_close_enabled", True),
                "created_at": user_doc.get("created_at"),
                "last_active_at": user_doc.get("last_active_at"),
                "updated_at": user_doc.get("updated_at"),
                "migrated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Cache in new format
            await self.redis.set(new_key, json.dumps(profile), ex=3600)
            logger.info(f"🔄 Migrated user profile for {user_id}")
            self._log_migration(f"users.{user_id}", new_key, "user_profile")
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate user profile for {user_id}: {e}")
            return None
    
    async def get_user_context(self, user_id: str) -> Optional[Dict]:
        """Get current conversation context"""
        # Try new format first
        new_key = f"user:{user_id}:context"
        data = await self.redis.get(new_key)
        if data:
            return json.loads(data)
        
        # Migrate from conversation_context collection
        try:
            user_doc = await self.db.users.find_one({"_id": user_id})
            if not user_doc:
                return None
            
            context_doc = await self.db.conversation_context.find_one({
                "phone_number": user_doc["phone_number"]
            })
            
            if not context_doc:
                # Create empty context
                context = {
                    "recent_topics": [],
                    "last_discussed_symbols": [],
                    "relationship_stage": "new",
                    "conversation_frequency": "occasional",
                    "preferred_analysis_depth": "standard",
                    "total_conversations": 0,
                    "pending_decisions": [],
                    "successful_response_patterns": [],
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            else:
                # Build consolidated context
                context = {
                    "recent_topics": context_doc.get("recent_topics", []),
                    "last_discussed_symbols": context_doc.get("last_discussed_symbols", []),
                    "relationship_stage": context_doc.get("relationship_stage", "new"),
                    "conversation_frequency": context_doc.get("conversation_frequency", "occasional"),
                    "preferred_analysis_depth": context_doc.get("preferred_analysis_depth", "standard"),
                    "total_conversations": context_doc.get("total_conversations", 0),
                    "pending_decisions": context_doc.get("pending_decisions", []),
                    "successful_response_patterns": context_doc.get("successful_response_patterns", []),
                    "created_at": context_doc.get("created_at"),
                    "updated_at": context_doc.get("updated_at"),
                    "migrated_at": datetime.now(timezone.utc).isoformat()
                }
            
            # Cache in new format
            await self.redis.set(new_key, json.dumps(context), ex=1800)
            logger.info(f"🔄 Migrated user context for {user_id}")
            self._log_migration(f"conversation_context.{user_doc['phone_number']}", new_key, "user_context")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate user context for {user_id}: {e}")
            return None
    
    async def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user conversation history"""
        # Try new format first
        new_key = f"user:{user_id}:conversations"
        data = await self.redis.get(new_key)
        if data:
            return json.loads(data)
        
        # Migrate from enhanced_conversations
        try:
            user_doc = await self.db.users.find_one({"_id": user_id})
            if not user_doc:
                return []
            
            conversations = await self.db.enhanced_conversations.find({
                "phone_number": user_doc["phone_number"]
            }).sort("timestamp", -1).limit(limit).to_list(limit)
            
            # Convert to new format
            formatted_conversations = []
            for conv in conversations:
                formatted_conversations.append({
                    "timestamp": conv.get("timestamp"),
                    "date": conv.get("date"),
                    "user_message": conv.get("user_message", ""),
                    "bot_response": conv.get("bot_response", ""),
                    "symbols": conv.get("symbols_mentioned", []),
                    "intent": conv.get("intent_data", {}).get("intent", "unknown"),
                    "emotional_state": conv.get("intent_data", {}).get("emotional_state"),
                    "confidence_level": conv.get("intent_data", {}).get("confidence_level"),
                    "context_used": conv.get("context_used", ""),
                    "tools_used": conv.get("intent_data", {}).get("requires_tools", [])
                })
            
            # Cache in new format
            if formatted_conversations:
                await self.redis.set(new_key, json.dumps(formatted_conversations), ex=1800)
                logger.info(f"🔄 Migrated {len(formatted_conversations)} conversations for {user_id}")
                self._log_migration(f"enhanced_conversations.{user_doc['phone_number']}", new_key, "conversations")
            
            return formatted_conversations
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate conversations for {user_id}: {e}")
            return []
    
    async def get_user_sms_history(self, user_id: str) -> List[Dict]:
        """Get SMS conversation history"""
        # Try new format first
        new_key = f"user:{user_id}:sms:history"
        data = await self.redis.get(new_key)
        if data:
            return json.loads(data)
        
        # Migrate from conversation_thread Redis keys
        try:
            user_doc = await self.db.users.find_one({"_id": user_id})
            if not user_doc:
                return []
            
            old_key = f"conversation_thread:{user_doc['phone_number']}"
            old_data = await self.redis.get(old_key)
            if old_data:
                # Store in new format
                await self.redis.set(new_key, old_data, ex=86400)
                logger.info(f"🔄 Migrated SMS history: {old_key} → {new_key}")
                self._log_migration(old_key, new_key, "sms_history")
                return json.loads(old_data)
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate SMS history for {user_id}: {e}")
            return []
    
    async def get_user_usage(self, user_id: str, period: str = "current") -> Optional[Dict]:
        """Get user usage data"""
        # Try new format first
        new_key = f"user:{user_id}:usage:{period}"
        data = await self.redis.get(new_key)
        if data:
            return json.loads(data)
        
        # Migrate from old usage keys
        old_keys = [
            f"usage:{user_id}:week",
            f"usage:{user_id}:weekly"
        ]
        
        for old_key in old_keys:
            count = await self.redis.get(old_key)
            if count:
                usage_data = {
                    "count": int(count),
                    "period": period,
                    "migrated_from": old_key,
                    "migrated_at": datetime.now(timezone.utc).isoformat()
                }
                
                # Store in new format
                await self.redis.set(new_key, json.dumps(usage_data), ex=86400)
                logger.info(f"🔄 Migrated usage: {old_key} → {new_key}")
                self._log_migration(old_key, new_key, "usage_data")
                
                # Clean up old key
                await self.redis.delete(old_key)
                
                return usage_data
        
        return None
    
    async def get_user_personality(self, user_id: str) -> Optional[Dict]:
        """Get user personality data"""
        # Try new format first
        new_key = f"user:{user_id}:personality"
        data = await self.redis.get(new_key)
        if data:
            return json.loads(data)
        
        # Migrate from MongoDB user document
        try:
            user_doc = await self.db.users.find_one({"_id": user_id})
            if not user_doc:
                return None
            
            # Consolidate personality data from scattered fields
            personality = {
                "communication_style": user_doc.get("communication_style", {}),
                "response_patterns": user_doc.get("response_patterns", {}),
                "trading_behavior": user_doc.get("trading_behavior", {}),
                "speech_patterns": user_doc.get("speech_patterns", {}),
                "learning_data": {
                    "total_messages": user_doc.get("total_messages_sent", 0) + user_doc.get("total_messages_received", 0),
                    "successful_patterns": [],
                    "improvement_areas": []
                },
                "migrated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Cache in new format
            await self.redis.set(new_key, json.dumps(personality), ex=3600)
            logger.info(f"🔄 Migrated personality data for {user_id}")
            self._log_migration(f"users.{user_id}.personality_fields", new_key, "personality")
            
            return personality
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate personality for {user_id}: {e}")
            return None
    
    # ==========================================
    # SETTER METHODS
    # ==========================================
    
    async def set_user_profile(self, user_id: str, profile: Dict, ttl: int = 3600) -> bool:
        """Set user profile data"""
        key = f"user:{user_id}:profile"
        try:
            profile["updated_at"] = datetime.now(timezone.utc).isoformat()
            await self.redis.set(key, json.dumps(profile), ex=ttl)
            logger.debug(f"✅ Set {key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set {key}: {e}")
            return False
    
    async def set_user_context(self, user_id: str, context: Dict, ttl: int = 1800) -> bool:
        """Set user context data"""
        key = f"user:{user_id}:context"
        try:
            context["updated_at"] = datetime.now(timezone.utc).isoformat()
            await self.redis.set(key, json.dumps(context), ex=ttl)
            logger.debug(f"✅ Set {key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set {key}: {e}")
            return False
    
    async def set_user_conversations(self, user_id: str, conversations: List[Dict], ttl: int = 1800) -> bool:
        """Set user conversations"""
        key = f"user:{user_id}:conversations"
        try:
            await self.redis.set(key, json.dumps(conversations), ex=ttl)
            logger.debug(f"✅ Set {key} with {len(conversations)} conversations")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set {key}: {e}")
            return False
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def _log_migration(self, old_key: str, new_key: str, data_type: str):
        """Log migration for tracking purposes"""
        migration_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_key": old_key,
            "new_key": new_key,
            "data_type": data_type
        }
        self.migration_log.append(migration_entry)
        
        # Keep only last 1000 migrations in memory
        if len(self.migration_log) > 1000:
            self.migration_log = self.migration_log[-1000:]
    
    async def get_migration_stats(self) -> Dict:
        """Get migration statistics"""
        stats = {
            "total_migrations": len(self.migration_log),
            "by_type": {},
            "recent_migrations": self.migration_log[-10:] if self.migration_log else []
        }
        
        for entry in self.migration_log:
            data_type = entry["data_type"]
            stats["by_type"][data_type] = stats["by_type"].get(data_type, 0) + 1
        
        return stats
    
    async def cleanup_old_keys(self, dry_run: bool = True) -> Dict:
        """Clean up old keys after migration"""
        cleanup_stats = {
            "total_deleted": 0,
            "keys_deleted": [],
            "errors": []
        }
        
        # Define old key patterns to clean up
        old_patterns = [
            "stock:*:last_updated",
            "stock:*:tags", 
            "usage:*:weekly",
            "usage:None:*",
            "conversation_thread:*"
        ]
        
        for pattern in old_patterns:
            try:
                keys = await self.redis.keys(pattern)
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    
                    if not dry_run:
                        await self.redis.delete(key_str)
                        cleanup_stats["keys_deleted"].append(key_str)
                        cleanup_stats["total_deleted"] += 1
                    else:
                        cleanup_stats["keys_deleted"].append(f"[DRY RUN] {key_str}")
                        
            except Exception as e:
                cleanup_stats["errors"].append(f"Error processing {pattern}: {e}")
        
        return cleanup_stats
    
    async def validate_migration(self, user_id: str) -> Dict:
        """Validate that migration worked correctly for a user"""
        validation = {
            "user_id": user_id,
            "profile": False,
            "context": False,
            "conversations": False,
            "sms_history": False,
            "usage": False,
            "personality": False,
            "errors": []
        }
        
        try:
            # Check each data type
            profile = await self.get_user_profile(user_id)
            validation["profile"] = profile is not None
            
            context = await self.get_user_context(user_id)
            validation["context"] = context is not None
            
            conversations = await self.get_user_conversations(user_id)
            validation["conversations"] = len(conversations) > 0
            
            sms_history = await self.get_user_sms_history(user_id)
            validation["sms_history"] = len(sms_history) > 0
            
            usage = await self.get_user_usage(user_id)
            validation["usage"] = usage is not None
            
            personality = await self.get_user_personality(user_id)
            validation["personality"] = personality is not None
            
        except Exception as e:
            validation["errors"].append(str(e))
        
        return validation
    
    # ==========================================
    # BULK MIGRATION METHODS
    # ==========================================
    
    async def migrate_all_users(self, limit: int = None) -> Dict:
        """Migrate all users to new format"""
        migration_stats = {
            "total_users": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "errors": []
        }
        
        try:
            # Get all users
            query = {}
            if limit:
                cursor = self.db.users.find(query).limit(limit)
            else:
                cursor = self.db.users.find(query)
            
            async for user_doc in cursor:
                migration_stats["total_users"] += 1
                user_id = user_doc["_id"]
                
                try:
                    # Migrate each data type
                    await self.get_user_profile(user_id)
                    await self.get_user_context(user_id)
                    await self.get_user_conversations(user_id)
                    await self.get_user_sms_history(user_id)
                    await self.get_user_usage(user_id)
                    await self.get_user_personality(user_id)
                    
                    migration_stats["successful_migrations"] += 1
                    logger.info(f"✅ Migrated user {user_id}")
                    
                except Exception as e:
                    migration_stats["failed_migrations"] += 1
                    migration_stats["errors"].append(f"User {user_id}: {str(e)}")
                    logger.error(f"❌ Failed to migrate user {user_id}: {e}")
        
        except Exception as e:
            migration_stats["errors"].append(f"Bulk migration error: {str(e)}")
        
        return migration_stats
    
    async def migrate_all_stocks(self) -> Dict:
        """Migrate all stock data to new format"""
        migration_stats = {
            "total_symbols": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "errors": []
        }
        
        try:
            # Get all stock keys
            stock_keys = await self.redis.keys("stock:*:technical")
            
            for key in stock_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                symbol = key_str.split(":")[1]  # Extract symbol from stock:SYMBOL:technical
                
                migration_stats["total_symbols"] += 1
                
                try:
                    # Migrate both technical and fundamental data
                    await self.get_stock_technical(symbol)
                    await self.get_stock_fundamental(symbol)
                    await self.get_stock_metadata(symbol, "ta")
                    await self.get_stock_metadata(symbol, "fa")
                    
                    migration_stats["successful_migrations"] += 1
                    logger.info(f"✅ Migrated stock {symbol}")
                    
                except Exception as e:
                    migration_stats["failed_migrations"] += 1
                    migration_stats["errors"].append(f"Symbol {symbol}: {str(e)}")
                    logger.error(f"❌ Failed to migrate stock {symbol}: {e}")
        
        except Exception as e:
            migration_stats["errors"].append(f"Stock migration error: {str(e)}")
        
        return migration_stats


# Example usage and integration
async def initialize_key_builder(redis_client, mongo_db):
    """Initialize KeyBuilder with your existing connections"""
    key_builder = KeyBuilder(redis_client, mongo_db)
    
    # Log successful initialization
    logger.info("🔧 KeyBuilder initialized successfully")
    
    return key_builder


# Integration with your existing database service
def integrate_with_database_service(db_service):
    """Integrate KeyBuilder with your existing DatabaseService"""
    
    # Add KeyBuilder to your DatabaseService
    db_service.key_builder = None
    
    async def init_key_builder():
        if db_service.redis and db_service.db:
            db_service.key_builder = KeyBuilder(db_service.redis, db_service.db)
            logger.info("🔧 KeyBuilder integrated with DatabaseService")
    
    # Add method to DatabaseService to initialize KeyBuilder
    db_service.init_key_builder = init_key_builder
    
    return db_service
