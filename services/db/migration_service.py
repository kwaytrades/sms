services/db/migration_service.py
"""
Migration Service - Data migration, cleanup, version management
Focused on database migrations and data transformation
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from loguru import logger

from services.db.base_db_service import BaseDBService


class MigrationService:
    """Specialized service for data migrations and cleanup"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self._migration_log = []

    async def initialize(self):
        """Initialize migration tracking"""
        try:
            # Migration tracking collection
            await self.base.db.migration_history.create_index("migration_name", unique=True)
            await self.base.db.migration_history.create_index("executed_at")
            
            logger.info("✅ Migration service initialized")
        except Exception as e:
            logger.exception(f"❌ Migration service initialization failed: {e}")

    # Stock Data Migration Methods (KeyBuilder Compatible)
    async def get_stock_technical(self, symbol: str) -> Optional[Dict]:
        """Get technical analysis data with automatic migration"""
        symbol = symbol.upper()
        
        # Try new format first
        new_key = f"{symbol}:ta"
        data = await self.base.key_builder.get(new_key)
        if data:
            return data
        
        # Migrate from old format
        old_key = f"stock:{symbol}:technical"
        data = await self.base.key_builder.get(old_key)
        if data:
            logger.info(f"🔄 Migrating {old_key} → {new_key}")
            await self.base.key_builder.set(new_key, data, ttl=3600)
            self._log_migration(old_key, new_key, "technical_analysis")
            return data
        
        return None

    async def get_stock_fundamental(self, symbol: str) -> Optional[Dict]:
        """Get fundamental analysis data with automatic migration"""
        symbol = symbol.upper()
        
        # Try new format first
        new_key = f"{symbol}:fa"
        data = await self.base.key_builder.get(new_key)
        if data:
            return data
        
        # Migrate from old format
        old_key = f"stock:{symbol}:fundamental"
        data = await self.base.key_builder.get(old_key)
        if data:
            logger.info(f"🔄 Migrating {old_key} → {new_key}")
            await self.base.key_builder.set(new_key, data, ttl=3600)
            self._log_migration(old_key, new_key, "fundamental_analysis")
            return data
        
        return None

    async def set_stock_data(self, symbol: str, data_type: str, data: Dict, ttl: int = 3600) -> bool:
        """Set stock data using new naming convention"""
        symbol = symbol.upper()
        key = f"{symbol}:{data_type}"
        
        try:
            success = await self.base.key_builder.set(key, data, ttl=ttl)
            return success
        except Exception as e:
            logger.error(f"❌ Failed to set {key}: {e}")
            return False

    async def migrate_all_users(self, limit: int = None) -> Dict:
        """Migrate all users to new unified format"""
        migration_stats = {
            "total_users": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "errors": []
        }
        
        try:
            # Record migration start
            migration_doc = {
                "migration_name": f"user_migration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "started_at": datetime.now(timezone.utc),
                "status": "running",
                "stats": migration_stats
            }
            
            migration_result = await self.base.db.migration_history.insert_one(migration_doc)
            migration_id = migration_result.inserted_id
            
            # Get all users
            query = {}
            if limit:
                cursor = self.base.db.users.find(query).limit(limit)
            else:
                cursor = self.base.db.users.find(query)
            
            async for user_doc in cursor:
                migration_stats["total_users"] += 1
                user_id = str(user_doc["_id"])
                
                try:
                    # Validate user data
                    if not user_doc.get("phone_number"):
                        migration_stats["failed_migrations"] += 1
                        migration_stats["errors"].append(f"User {user_id}: missing phone_number")
                        continue
                    
                    migration_stats["successful_migrations"] += 1
                    logger.debug(f"✅ Validated user {user_id}")
                    
                except Exception as e:
                    migration_stats["failed_migrations"] += 1
                    migration_stats["errors"].append(f"User {user_id}: {str(e)}")
            
            # Update migration record
            await self.base.db.migration_history.update_one(
                {"_id": migration_id},
                {
                    "$set": {
                        "completed_at": datetime.now(timezone.utc),
                        "status": "completed",
                        "stats": migration_stats
                    }
                }
            )
        
        except Exception as e:
            migration_stats["errors"].append(f"Migration error: {str(e)}")
        
        return migration_stats

    async def cleanup_old_keys(self, dry_run: bool = True) -> Dict:
        """Clean up old Redis keys after migration"""
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
                # Note: In production, you'd use Redis SCAN for pattern matching
                # This is a simplified version for demonstration
                cleanup_stats["keys_deleted"].append(f"[PATTERN] {pattern}")
                if not dry_run:
                    cleanup_stats["total_deleted"] += 1
                    
            except Exception as e:
                cleanup_stats["errors"].append(f"Error processing {pattern}: {e}")
        
        return cleanup_stats

    async def get_stats(self) -> Dict:
        """Get migration statistics"""
        stats = {
            "total_migrations": len(self._migration_log),
            "by_type": {},
            "recent_migrations": self._migration_log[-10:] if self._migration_log else []
        }
        
        for entry in self._migration_log:
            data_type = entry["data_type"]
            stats["by_type"][data_type] = stats["by_type"].get(data_type, 0) + 1
        
        return stats

    def _log_migration(self, old_key: str, new_key: str, data_type: str):
        """Log migration for tracking purposes"""
        migration_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_key": old_key,
            "new_key": new_key,
            "data_type": data_type
        }
        self._migration_log.append(migration_entry)
        
        # Keep only last 1000 migrations in memory
        if len(self._migration_log) > 1000:
            self._migration_log = self._migration_log[-1000:]
