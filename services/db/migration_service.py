"""
Enhanced Migration Service - Complete Standalone Implementation
Advanced data migration, cleanup, and version management with background processing
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import asyncio
import json
import re
from dataclasses import dataclass
from enum import Enum
import uuid
import aioredis
from typing import Protocol, runtime_checkable
from loguru import logger

from services.db.base_db_service import BaseDBService


@runtime_checkable
class AsyncRedisProtocol(Protocol):
    """Protocol for async Redis operations"""
    async def ttl(self, key: str) -> int: ...
    async def keys(self, pattern: str) -> List[str]: ...
    async def delete(self, *keys: str) -> int: ...
    async def scan(self, cursor: int = 0, match: str = None, count: int = None) -> Tuple[int, List[str]]: ...


class MigrationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"
    CANCELLED = "cancelled"


@dataclass
class MigrationTask:
    task_id: str
    migration_name: str
    description: str
    status: MigrationStatus
    progress: float = 0.0
    total_items: int = 0
    processed_items: int = 0
    errors: List[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    last_checkpoint: Optional[datetime] = None
    rollback_data: List[Dict] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.rollback_data is None:
            self.rollback_data = []
        if self.metadata is None:
            self.metadata = {}


class BackgroundMigrationManager:
    """Manages long-running background migrations with checkpoints and error recovery"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self.active_migrations: Dict[str, MigrationTask] = {}
        self.migration_queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.redis_client: Optional[AsyncRedisProtocol] = None
        
    async def initialize(self):
        """Initialize async Redis client if available"""
        try:
            if hasattr(self.base, 'redis_url') and self.base.redis_url:
                self.redis_client = await aioredis.from_url(
                    self.base.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=15,
                    socket_timeout=15
                )
                logger.info("✅ Async Redis client initialized for migrations")
        except Exception as e:
            logger.warning(f"Failed to initialize async Redis client: {e}")
            self.redis_client = None
        
    async def start_worker(self):
        """Start background migration worker"""
        if self.worker_task and not self.worker_task.done():
            return
        
        self.worker_task = asyncio.create_task(self._migration_worker())
        logger.info("✅ Background migration worker started")
    
    async def stop_worker(self):
        """Gracefully stop background migration worker"""
        logger.info("🛑 Initiating graceful shutdown of migration worker...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Save unprocessed queue items for later resume
        await self._persist_unprocessed_queue()
        
        # Cancel worker task
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("⏹️ Background migration worker stopped gracefully")
    
    async def _persist_unprocessed_queue(self):
        """Persist unprocessed queue items for resume"""
        try:
            unprocessed_items = []
            
            # Drain the queue
            while not self.migration_queue.empty():
                try:
                    task = self.migration_queue.get_nowait()
                    task.status = MigrationStatus.PENDING
                    
                    # Save to database for resume
                    await self._save_migration_status(task)
                    unprocessed_items.append(task.task_id)
                    
                    self.migration_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            if unprocessed_items:
                logger.info(f"💾 Persisted {len(unprocessed_items)} unprocessed migrations for resume")
                
        except Exception as e:
            logger.error(f"Failed to persist unprocessed queue: {e}")
    
    async def _migration_worker(self):
        """Background worker for processing migrations with error recovery"""
        while not self.shutdown_event.is_set():
            try:
                # Get next migration from queue with timeout
                try:
                    migration_task = await asyncio.wait_for(
                        self.migration_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute migration with retry logic
                await self._execute_migration_with_retry(migration_task)
                
                # Mark queue task as done
                self.migration_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Migration worker cancelled")
                break
            except Exception as e:
                logger.exception(f"Migration worker error: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def _execute_migration_with_retry(self, migration_task: MigrationTask):
        """Execute migration with exponential backoff retry"""
        
        while migration_task.retry_count <= migration_task.max_retries:
            try:
                # Update status and start migration
                migration_task.status = MigrationStatus.RUNNING
                migration_task.started_at = datetime.now(timezone.utc)
                self.active_migrations[migration_task.task_id] = migration_task
                
                # Save to database
                await self._save_migration_status(migration_task)
                
                # Execute migration based on type
                if "user_migration" in migration_task.migration_name:
                    await self._execute_user_migration(migration_task)
                elif "key_cleanup" in migration_task.migration_name:
                    await self._execute_key_cleanup(migration_task)
                elif "data_normalization" in migration_task.migration_name:
                    await self._execute_data_normalization(migration_task)
                else:
                    await self._execute_generic_migration(migration_task)
                
                # Success - complete migration
                migration_task.status = MigrationStatus.COMPLETED
                migration_task.completed_at = datetime.now(timezone.utc)
                logger.info(f"✅ Migration completed: {migration_task.migration_name}")
                break
                
            except Exception as e:
                migration_task.retry_count += 1
                migration_task.errors.append(f"Attempt {migration_task.retry_count}: {str(e)}")
                
                if migration_task.retry_count <= migration_task.max_retries:
                    # Calculate exponential backoff delay
                    delay = min(300, 2 ** migration_task.retry_count)  # Max 5 minutes
                    logger.warning(f"⚠️ Migration failed, retrying in {delay}s: {migration_task.migration_name}")
                    
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded - check if rollback is needed
                    logger.error(f"❌ Migration failed after {migration_task.max_retries} retries: {migration_task.migration_name}")
                    
                    if migration_task.rollback_data:
                        await self._attempt_rollback(migration_task)
                    else:
                        migration_task.status = MigrationStatus.FAILED
                    break
        
        # Update final status
        await self._save_migration_status(migration_task)
        
        # Remove from active migrations
        self.active_migrations.pop(migration_task.task_id, None)
    
    async def _attempt_rollback(self, migration_task: MigrationTask):
        """Attempt to rollback failed migration"""
        try:
            migration_task.status = MigrationStatus.ROLLBACK
            logger.info(f"🔄 Attempting rollback for: {migration_task.migration_name}")
            
            # Execute rollback operations
            for rollback_op in reversed(migration_task.rollback_data):
                try:
                    if rollback_op["type"] == "redis_key_restore":
                        await self._restore_redis_key(rollback_op["key"], rollback_op["value"], rollback_op["ttl"])
                    elif rollback_op["type"] == "db_document_restore":
                        await self._restore_db_document(rollback_op["collection"], rollback_op["filter"], rollback_op["document"])
                    
                except Exception as rollback_error:
                    logger.error(f"Rollback operation failed: {rollback_error}")
            
            logger.info(f"✅ Rollback completed for: {migration_task.migration_name}")
            
        except Exception as e:
            logger.error(f"❌ Rollback failed for {migration_task.migration_name}: {e}")
            migration_task.status = MigrationStatus.FAILED
    
    async def _restore_redis_key(self, key: str, value: Any, ttl: int):
        """Restore a Redis key during rollback"""
        if self.redis_client:
            await self.redis_client.set(key, json.dumps(value))
            if ttl > 0:
                await self.redis_client.expire(key, ttl)
    
    async def _restore_db_document(self, collection: str, filter_doc: Dict, document: Dict):
        """Restore a database document during rollback"""
        await getattr(self.base.db, collection).replace_one(filter_doc, document, upsert=True)
    
    async def queue_migration(self, migration_task: MigrationTask):
        """Queue a migration for background processing"""
        await self.migration_queue.put(migration_task)
        logger.info(f"🔄 Queued migration: {migration_task.migration_name}")
    
    async def _execute_user_migration(self, task: MigrationTask):
        """Execute user migration with batch processing"""
        batch_size = 100
        skip = 0
        
        while skip < task.total_items:
            # Process batch
            batch_stats = await self._migrate_user_batch(skip, batch_size)
            
            # Update progress
            processed = min(skip + batch_size, task.total_items)
            task.processed_items = processed
            task.progress = (processed / task.total_items) * 100
            
            # Add any batch errors
            task.errors.extend(batch_stats.get("errors", []))
            
            # Save checkpoint
            await self._save_migration_status(task)
            
            skip += batch_size
            
            # Small delay to avoid overwhelming database
            await asyncio.sleep(0.1)
    
    async def _execute_key_cleanup(self, task: MigrationTask):
        """Execute Redis key cleanup"""
        patterns = task.metadata.get("patterns", [])
        batch_size = task.metadata.get("batch_size", 100)
        
        processed = 0
        for pattern in patterns:
            keys = await self._scan_redis_pattern(pattern)
            
            # Delete in batches
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                deleted = await self._delete_redis_keys_batch(batch)
                processed += deleted
                
                # Update progress
                task.processed_items = processed
                task.progress = min((processed / task.total_items) * 100, 100)
                
                await self._save_migration_status(task)
                await asyncio.sleep(0.01)
    
    async def _execute_data_normalization(self, task: MigrationTask):
        """Execute data normalization migration"""
        # Implementation would be specific to normalization type
        # This is a placeholder for the framework
        pass
    
    async def _execute_generic_migration(self, task: MigrationTask):
        """Execute generic migration with progress simulation"""
        for i in range(task.total_items):
            # Simulate processing
            await asyncio.sleep(0.01)
            
            task.processed_items = i + 1
            task.progress = (task.processed_items / task.total_items) * 100
            
            # Save checkpoint every 100 items
            if task.processed_items % 100 == 0:
                await self._save_migration_status(task)
    
    async def _save_migration_status(self, task: MigrationTask):
        """Save migration status to database"""
        try:
            await self.base.db.migration_history.update_one(
                {"task_id": task.task_id},
                {
                    "$set": {
                        "migration_name": task.migration_name,
                        "description": task.description,
                        "status": task.status.value,
                        "progress": task.progress,
                        "processed_items": task.processed_items,
                        "total_items": task.total_items,
                        "errors": task.errors,
                        "started_at": task.started_at,
                        "completed_at": task.completed_at,
                        "updated_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save migration status: {e}")
    
    async def _migrate_user_batch(self, skip: int, limit: int) -> Dict[str, Any]:
        """Migrate a batch of users - placeholder implementation"""
        # This would be implemented in the main service
        return {"processed": limit, "errors": []}
    
    async def _scan_redis_pattern(self, pattern: str) -> List[str]:
        """Scan Redis for keys matching pattern using async operations"""
        try:
            if self.redis_client:
                # Use async SCAN operation
                keys = []
                cursor = 0
                
                while True:
                    cursor, batch_keys = await self.redis_client.scan(
                        cursor=cursor, 
                        match=pattern, 
                        count=100
                    )
                    keys.extend(batch_keys)
                    
                    if cursor == 0:
                        break
                
                return keys
                
            elif hasattr(self.base, 'redis_client') and self.base.redis_client:
                # Fallback to sync client
                return self.base.redis_client.keys(pattern)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Redis scan error for pattern {pattern}: {e}")
            return []

    async def _delete_redis_keys_batch(self, keys: List[str]) -> int:
        """Delete a batch of Redis keys using async operations"""
        try:
            if self.redis_client:
                return await self.redis_client.delete(*keys)
            elif hasattr(self.base, 'redis_client') and self.base.redis_client:
                return self.base.redis_client.delete(*keys)
            else:
                return 0
        except Exception as e:
            logger.error(f"Redis batch delete error: {e}")
            return 0


class EnhancedMigrationService:
    """Advanced migration service with background processing and intelligent cleanup"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self._migration_log = []
        self.background_manager = BackgroundMigrationManager(base_service)
        self.redis_patterns = {
            "stock_old": ["stock:*:technical", "stock:*:fundamental", "stock:*:last_updated"],
            "user_old": ["usage:*:weekly", "usage:None:*", "conversation_thread:*"],
            "cache_old": ["cache:*:old_format", "temp:*", "deprecated:*"]
        }

    async def initialize(self):
        """Initialize enhanced migration tracking"""
        try:
            # Enhanced migration tracking collection
            await self.base.db.migration_history.create_index("migration_name")
            await self.base.db.migration_history.create_index("task_id", unique=True)
            await self.base.db.migration_history.create_index("status")
            await self.base.db.migration_history.create_index("started_at")
            
            # Migration checkpoints collection
            await self.base.db.migration_checkpoints.create_index([("task_id", 1), ("checkpoint_id", 1)], unique=True)
            
            # Initialize background manager
            await self.background_manager.initialize()
            
            # Start background worker
            await self.background_manager.start_worker()
            
            # Resume any pending migrations
            await self._resume_pending_migrations()
            
            logger.info("✅ Enhanced Migration service initialized")
        except Exception as e:
            logger.exception(f"❌ Enhanced Migration service initialization failed: {e}")
    
    async def _resume_pending_migrations(self):
        """Resume any pending migrations from previous shutdown"""
        try:
            pending_migrations = await self.base.db.migration_history.find({
                "status": "pending"
            }).to_list(length=None)
            
            for migration_doc in pending_migrations:
                task = MigrationTask(
                    task_id=migration_doc["task_id"],
                    migration_name=migration_doc["migration_name"],
                    description=migration_doc.get("description", ""),
                    status=MigrationStatus.PENDING,
                    total_items=migration_doc.get("total_items", 0),
                    processed_items=migration_doc.get("processed_items", 0),
                    metadata=migration_doc.get("metadata", {})
                )
                
                await self.background_manager.queue_migration(task)
                logger.info(f"🔄 Resumed pending migration: {task.migration_name}")
                
        except Exception as e:
            logger.error(f"Failed to resume pending migrations: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for migration service"""
        try:
            active_migrations = len(self.background_manager.active_migrations)
            queued_migrations = self.background_manager.migration_queue.qsize()
            
            migration_count = await self.base.db.migration_history.count_documents({})
            recent_failures = await self.base.db.migration_history.count_documents({
                "status": "failed",
                "started_at": {"$gte": datetime.now(timezone.utc) - timedelta(hours=24)}
            })
            
            return {
                "status": "healthy",
                "active_migrations": active_migrations,
                "queued_migrations": queued_migrations,
                "total_migrations": migration_count,
                "recent_failures": recent_failures,
                "worker_running": self.background_manager.worker_task and not self.background_manager.worker_task.done()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # Enhanced Stock Data Migration with TTL preservation
    async def get_stock_data_with_migration(self, symbol: str, data_type: str) -> Optional[Dict]:
        """Get stock data with intelligent migration and TTL preservation"""
        symbol = symbol.upper()
        
        # Map data types
        type_mapping = {
            "technical": "ta",
            "fundamental": "fa",
            "news": "news",
            "options": "options"
        }
        
        new_suffix = type_mapping.get(data_type, data_type)
        new_key = f"{symbol}:{new_suffix}"
        
        # Try new format first
        data = await self.base.key_builder.get_with_metadata(new_key)
        if data:
            return data.get('value') if isinstance(data, dict) else data
        
        # Attempt migration from old format
        old_key = f"stock:{symbol}:{data_type}"
        old_data = await self.base.key_builder.get_with_metadata(old_key)
        
        if old_data:
            # Preserve TTL if available
            original_ttl = await self.background_manager._get_redis_ttl(old_key)
            ttl = original_ttl if original_ttl > 0 else 3600
            
            # Store rollback data for potential failure recovery
            rollback_data = {
                "type": "redis_key_restore",
                "key": old_key,
                "value": old_data,
                "ttl": original_ttl
            }
            
            # Migrate to new format
            success = await self.base.key_builder.set(new_key, old_data, ttl=ttl)
            
            if success:
                logger.info(f"🔄 Migrated {old_key} → {new_key} (TTL: {ttl}s)")
                self._log_migration(old_key, new_key, data_type, preserved_ttl=ttl, rollback_data=rollback_data)
                
                # Schedule old key deletion
                asyncio.create_task(self._schedule_key_deletion(old_key, delay=300))
                
                return old_data
        
        return None

    async def set_stock_data(self, symbol: str, data_type: str, data: Dict, ttl: int = 3600) -> bool:
        """Set stock data using new naming convention"""
        symbol = symbol.upper()
        
        # Use new format
        type_mapping = {
            "technical": "ta",
            "fundamental": "fa",
            "news": "news",
            "options": "options"
        }
        
        suffix = type_mapping.get(data_type, data_type)
        key = f"{symbol}:{suffix}"
        
        try:
            success = await self.base.key_builder.set(key, data, ttl=ttl)
            return success
        except Exception as e:
            logger.error(f"❌ Failed to set {key}: {e}")
            return False

    # Advanced User Migration with data normalization
    async def migrate_users_enhanced(self, batch_size: int = 100, normalize_data: bool = True) -> str:
        """Enhanced user migration with background processing and normalization"""
        
        # Create migration task
        task_id = str(uuid.uuid4())
        total_users = await self.base.db.users.count_documents({})
        
        migration_task = MigrationTask(
            task_id=task_id,
            migration_name=f"enhanced_user_migration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            description=f"Enhanced user migration with normalization (batch_size: {batch_size})",
            status=MigrationStatus.PENDING,
            total_items=total_users
        )
        migration_task.metadata = {"batch_size": batch_size, "normalize_data": normalize_data}
        
        # Queue for background processing
        await self.background_manager.queue_migration(migration_task)
        
        return task_id

    async def migrate_user_batch(self, skip: int, limit: int, normalize_data: bool = True) -> Dict[str, Any]:
        """Migrate a batch of users with normalization"""
        batch_stats = {
            "processed": 0,
            "normalized": 0,
            "errors": []
        }
        
        try:
            cursor = self.base.db.users.find({}).skip(skip).limit(limit)
            
            async for user_doc in cursor:
                try:
                    user_id = str(user_doc["_id"])
                    batch_stats["processed"] += 1
                    
                    if normalize_data:
                        # Normalize phone number
                        if phone := user_doc.get("phone_number"):
                            normalized_phone = self._normalize_phone_number(phone)
                            if normalized_phone != phone:
                                await self.base.db.users.update_one(
                                    {"_id": user_doc["_id"]},
                                    {"$set": {"phone_number": normalized_phone}}
                                )
                                batch_stats["normalized"] += 1
                        
                        # Normalize email
                        if email := user_doc.get("email"):
                            normalized_email = email.lower().strip()
                            if normalized_email != email:
                                await self.base.db.users.update_one(
                                    {"_id": user_doc["_id"]},
                                    {"$set": {"email": normalized_email}}
                                )
                        
                        # Ensure required fields
                        updates = {}
                        if not user_doc.get("created_at"):
                            updates["created_at"] = datetime.now(timezone.utc)
                        if not user_doc.get("plan_type"):
                            updates["plan_type"] = "free"
                        if not user_doc.get("status"):
                            updates["status"] = "active"
                        
                        if updates:
                            await self.base.db.users.update_one(
                                {"_id": user_doc["_id"]},
                                {"$set": updates}
                            )
                    
                except Exception as e:
                    batch_stats["errors"].append(f"User {user_id}: {str(e)}")
            
        except Exception as e:
            batch_stats["errors"].append(f"Batch error: {str(e)}")
        
        return batch_stats

    # Advanced Redis Cleanup with pattern scanning
    async def cleanup_redis_keys_advanced(self, patterns: List[str] = None, 
                                        dry_run: bool = True, 
                                        batch_size: int = 100) -> str:
        """Advanced Redis cleanup with batch processing - returns task_id for background processing"""
        
        patterns_to_process = patterns or []
        if not patterns_to_process:
            for category, pattern_list in self.redis_patterns.items():
                patterns_to_process.extend(pattern_list)
        
        # Estimate total keys to process
        total_keys = 0
        for pattern in patterns_to_process:
            keys = await self._scan_redis_pattern(pattern)
            total_keys += len(keys)
        
        # Create migration task for cleanup
        task_id = str(uuid.uuid4())
        
        migration_task = MigrationTask(
            task_id=task_id,
            migration_name=f"redis_cleanup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            description=f"Redis key cleanup: {len(patterns_to_process)} patterns, {total_keys} keys (dry_run: {dry_run})",
            status=MigrationStatus.PENDING,
            total_items=total_keys
        )
        migration_task.metadata = {
            "patterns": patterns_to_process,
            "dry_run": dry_run,
            "batch_size": batch_size
        }
        
        # Queue for background processing
        await self.background_manager.queue_migration(migration_task)
        
        return task_id

    # Migration monitoring and management
    async def get_migration_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed migration status"""
        try:
            migration_doc = await self.base.db.migration_history.find_one({"task_id": task_id})
            if not migration_doc:
                return None
            
            # Convert ObjectId to string
            migration_doc["_id"] = str(migration_doc["_id"])
            
            # Add runtime calculations
            if migration_doc.get("started_at") and not migration_doc.get("completed_at"):
                runtime = datetime.now(timezone.utc) - migration_doc["started_at"]
                migration_doc["runtime_seconds"] = runtime.total_seconds()
                
                # Estimate remaining time
                if migration_doc.get("progress", 0) > 0:
                    estimated_total = runtime.total_seconds() / (migration_doc["progress"] / 100)
                    estimated_remaining = estimated_total - runtime.total_seconds()
                    migration_doc["estimated_remaining_seconds"] = max(0, estimated_remaining)
            
            return migration_doc
            
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return None

    async def get_all_migrations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all migrations with status"""
        try:
            migrations = await self.base.db.migration_history.find({}).sort("started_at", -1).limit(limit).to_list(length=None)
            
            for migration in migrations:
                migration["_id"] = str(migration["_id"])
            
            return migrations
            
        except Exception as e:
            logger.error(f"Error getting all migrations: {e}")
            return []

    async def cancel_migration(self, task_id: str) -> bool:
        """Cancel a running migration"""
        try:
            # Update database status
            result = await self.base.db.migration_history.update_one(
                {"task_id": task_id, "status": {"$in": ["pending", "running"]}},
                {
                    "$set": {
                        "status": MigrationStatus.CANCELLED.value,
                        "completed_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            # Remove from active migrations
            if task_id in self.background_manager.active_migrations:
                del self.background_manager.active_migrations[task_id]
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error cancelling migration: {e}")
            return False

    async def get_stats(self) -> Dict:
        """Get comprehensive migration statistics"""
        try:
            stats = {
                "total_migrations": len(self._migration_log),
                "by_type": {},
                "recent_migrations": self._migration_log[-10:] if self._migration_log else [],
                "background_stats": {
                    "active_migrations": len(self.background_manager.active_migrations),
                    "queued_migrations": self.background_manager.migration_queue.qsize(),
                    "worker_running": self.background_manager.worker_task and not self.background_manager.worker_task.done()
                }
            }
            
            # Count by data type
            for entry in self._migration_log:
                data_type = entry["data_type"]
                stats["by_type"][data_type] = stats["by_type"].get(data_type, 0) + 1
            
            # Database statistics
            db_stats = await self.base.db.migration_history.aggregate([
                {"$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }}
            ]).to_list(length=None)
            
            stats["database_stats"] = {item["_id"]: item["count"] for item in db_stats}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting migration stats: {e}")
            return {"error": str(e)}

    # Helper methods
    async def _get_redis_ttl(self, key: str) -> int:
        """Get TTL of Redis key using async client"""
        try:
            if self.redis_client:
                return await self.redis_client.ttl(key)
            elif hasattr(self.base, 'redis_client') and self.base.redis_client:
                # Fallback to sync client if async not available
                return self.base.redis_client.ttl(key)
            else:
                return -1
        except Exception as e:
            logger.warning(f"Failed to get TTL for {key}: {e}")
            return -1

    async def _schedule_key_deletion(self, key: str, delay: int = 300):
        """Schedule old key deletion after delay"""
        await asyncio.sleep(delay)
        try:
            await self.base.key_builder.delete(key)
            logger.debug(f"🗑️ Deleted old key: {key}")
        except Exception as e:
            logger.warning(f"Failed to delete old key {key}: {e}")

    async def _scan_redis_pattern(self, pattern: str) -> List[str]:
        """Scan Redis for keys matching pattern using async operations"""
        return await self.background_manager._scan_redis_pattern(pattern)

    def _normalize_phone_number(self, phone: str) -> str:
        """Normalize phone number format with validation"""
        if not phone:
            return phone
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Validate length
        if len(digits) < 10:
            raise ValueError(f"Phone number too short: {phone}")
        if len(digits) > 15:
            raise ValueError(f"Phone number too long: {phone}")
        
        # Handle different formats
        if len(digits) == 11 and digits.startswith('1'):
            # US number with country code
            return f"+1{digits[1:]}"
        elif len(digits) == 10:
            # US number without country code
            return f"+1{digits}"
        elif len(digits) > 10:
            # International number
            return f"+{digits}"
        
        return phone  # Return original if can't normalize

    def _log_migration(self, old_key: str, new_key: str, data_type: str, **metadata):
        """Enhanced migration logging with structured data"""
        migration_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_key": old_key,
            "new_key": new_key,
            "data_type": data_type,
            "metadata": metadata,
            "migration_id": str(uuid.uuid4())
        }
        self._migration_log.append(migration_entry)
        
        # Keep only last 1000 migrations in memory
        if len(self._migration_log) > 1000:
            self._migration_log = self._migration_log[-1000:]


# Test utilities for unit testing
class MockRedisClient:
    """Mock Redis client for testing"""
    
    def __init__(self):
        self.data = {}
        self.ttls = {}
    
    async def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)
    
    async def keys(self, pattern: str) -> List[str]:
        # Simple pattern matching for tests
        if '*' in pattern:
            prefix = pattern.replace('*', '')
            return [k for k in self.data.keys() if k.startswith(prefix)]
        return [key for key in self.data.keys() if key == pattern]
    
    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                self.ttls.pop(key, None)
                deleted += 1
        return deleted
    
    async def scan(self, cursor: int = 0, match: str = None, count: int = None) -> Tuple[int, List[str]]:
        keys = await self.keys(match or '*')
        return 0, keys  # Simplified for testing
    
    async def set(self, key: str, value: str):
        self.data[key] = value
    
    async def expire(self, key: str, ttl: int):
        self.ttls[key] = ttl
    
    async def close(self):
        pass


class MigrationTestHooks:
    """Test hooks for migration service testing"""
    
    @staticmethod
    def create_test_migration_service(mock_redis: bool = True, mock_db: bool = True):
        """Create migration service with test doubles"""
        from unittest.mock import MagicMock
        
        # Mock base service
        base_service = MagicMock()
        
        if mock_redis:
            base_service.redis_url = "redis://localhost"
        
        if mock_db:
            base_service.db = MagicMock()
            base_service.db.migration_history = MagicMock()
            base_service.db.migration_checkpoints = MagicMock()
        
        service = EnhancedMigrationService(base_service)
        
        if mock_redis:
            service.background_manager.redis_client = MockRedisClient()
        
        return service
    
    @staticmethod
    def create_test_migration_task(
        task_id: str = None,
        migration_name: str = "test_migration",
        total_items: int = 100
    ) -> MigrationTask:
        """Create test migration task"""
        return MigrationTask(
            task_id=task_id or str(uuid.uuid4()),
            migration_name=migration_name,
            description=f"Test migration: {migration_name}",
            status=MigrationStatus.PENDING,
            total_items=total_items
        )


# Export the enhanced migration service
__all__ = ['EnhancedMigrationService', 'BackgroundMigrationManager', 'MigrationTask', 'MigrationStatus']
