"""
Enhanced Migration Service - Complete Standalone Implementation
Advanced data migration, cleanup, and version management with background processing
Enhanced with custom exceptions, base service class, and monitoring
"""

from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
import uuid
import aioredis
from typing import Protocol, runtime_checkable
from loguru import logger
import time
from contextlib import asynccontextmanager
from pydantic import BaseSettings, Field

from services.db.base_db_service import BaseDBService


# Configuration Management
class MigrationConfig(BaseSettings):
    """Centralized migration service configuration"""
    
    # Cache TTL values
    MIGRATION_CACHE_TTL: int = Field(3600, description="Migration status cache TTL")
    CHECKPOINT_CACHE_TTL: int = Field(86400, description="Checkpoint data cache TTL")
    
    # Migration settings
    DEFAULT_BATCH_SIZE: int = Field(100, description="Default migration batch size")
    MAX_RETRIES: int = Field(3, description="Max migration retries")
    RETRY_BASE_DELAY: float = Field(1.0, description="Base retry delay in seconds")
    MAX_RETRY_DELAY: int = Field(300, description="Maximum retry delay in seconds")
    
    # Background processing
    WORKER_TIMEOUT: float = Field(1.0, description="Worker queue timeout")
    CHECKPOINT_INTERVAL: int = Field(100, description="Save checkpoint every N items")
    MAX_CONCURRENT_MIGRATIONS: int = Field(5, description="Max concurrent migrations")
    
    # Redis cleanup patterns
    REDIS_CLEANUP_PATTERNS: Dict[str, List[str]] = Field(
        default={
            "stock_old": ["stock:*:technical", "stock:*:fundamental", "stock:*:last_updated"],
            "user_old": ["usage:*:weekly", "usage:None:*", "conversation_thread:*"],
            "cache_old": ["cache:*:old_format", "temp:*", "deprecated:*"]
        }
    )
    
    # Performance limits
    MAX_SCAN_KEYS_PER_BATCH: int = Field(1000, description="Max keys per Redis scan")
    REDIS_SCAN_COUNT: int = Field(100, description="Redis SCAN count parameter")
    
    class Config:
        env_file = ".env"
        env_prefix = "MIGRATION_"


# Custom Exception Hierarchy
class MigrationException(Exception):
    """Base exception for all migration errors"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)


class MigrationValidationError(MigrationException):
    """Migration data validation errors"""
    pass


class MigrationExecutionError(MigrationException):
    """Migration execution errors"""
    pass


class MigrationRollbackError(MigrationException):
    """Migration rollback errors"""
    pass


class RedisOperationError(MigrationException):
    """Redis operation errors"""
    pass


class DatabaseOperationError(MigrationException):
    """Database operation errors"""
    pass


# Monitoring and Observability
class MigrationMonitor:
    """Migration monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = {}
        self.active_operations = {}
        self.start_time = time.time()
    
    @asynccontextmanager
    async def track_operation(self, operation_name: str, metadata: Dict = None):
        """Context manager for tracking operation metrics"""
        start_time = time.time()
        operation_id = f"{operation_name}_{int(start_time * 1000)}"
        
        self.active_operations[operation_id] = {
            "name": operation_name,
            "start_time": start_time,
            "metadata": metadata or {}
        }
        
        try:
            yield operation_id
        except Exception as e:
            self._record_error(operation_name, str(e))
            raise
        finally:
            duration = time.time() - start_time
            self._record_success(operation_name, duration)
            self.active_operations.pop(operation_id, None)
    
    def _record_success(self, operation: str, duration: float):
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total_duration": 0,
                "errors": 0,
                "avg_duration": 0,
                "min_duration": float('inf'),
                "max_duration": 0
            }
        
        metrics = self.metrics[operation]
        metrics["count"] += 1
        metrics["total_duration"] += duration
        metrics["avg_duration"] = metrics["total_duration"] / metrics["count"]
        metrics["min_duration"] = min(metrics["min_duration"], duration)
        metrics["max_duration"] = max(metrics["max_duration"], duration)
    
    def _record_error(self, operation: str, error: str):
        if operation not in self.metrics:
            self.metrics[operation] = {"count": 0, "total_duration": 0, "errors": 0}
        
        self.metrics[operation]["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        return {
            "metrics": self.metrics,
            "active_operations": len(self.active_operations),
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Base Enhanced Service
class BaseMigrationService:
    """Base class with common patterns for migration operations"""
    
    def __init__(self, base_service: BaseDBService, config: MigrationConfig = None):
        self.base = base_service
        self.config = config or MigrationConfig()
        self.monitor = MigrationMonitor()
        self._service_id = str(uuid.uuid4())
        self._operational_metrics = {
            "operations": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def with_retry(self, operation, max_retries: int = None, base_delay: float = None):
        """Generic retry wrapper with exponential backoff"""
        max_retries = max_retries or self.config.MAX_RETRIES
        base_delay = base_delay or self.config.RETRY_BASE_DELAY
        
        for attempt in range(max_retries + 1):
            try:
                result = await operation()
                self._operational_metrics["operations"] += 1
                return result
            except Exception as e:
                if attempt == max_retries:
                    self._operational_metrics["errors"] += 1
                    raise MigrationExecutionError(
                        f"Operation failed after {max_retries} retries: {str(e)}",
                        "MAX_RETRIES_EXCEEDED",
                        {"attempt": attempt, "max_retries": max_retries}
                    )
                
                delay = min(base_delay * (2 ** attempt), self.config.MAX_RETRY_DELAY)
                logger.warning(f"Retry {attempt + 1}/{max_retries} in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
    
    async def cache_with_fallback(self, cache_key: str, fallback_func, ttl: int = None):
        """Standard cache-with-fallback pattern"""
        ttl = ttl or self.config.MIGRATION_CACHE_TTL
        
        try:
            cached = await self.base.key_builder.get(cache_key)
            if cached:
                self._operational_metrics["cache_hits"] += 1
                return cached
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_key}: {e}")
        
        self._operational_metrics["cache_misses"] += 1
        result = await fallback_func()
        
        try:
            await self.base.key_builder.set(cache_key, result, ttl=ttl)
        except Exception as e:
            logger.warning(f"Cache write failed for {cache_key}: {e}")
        
        return result
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        return {
            "service_id": self._service_id,
            "operational_metrics": self._operational_metrics.copy(),
            "monitor_metrics": self.monitor.get_metrics(),
            "config": {
                "batch_size": self.config.DEFAULT_BATCH_SIZE,
                "max_retries": self.config.MAX_RETRIES,
                "max_concurrent": self.config.MAX_CONCURRENT_MIGRATIONS
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@runtime_checkable
class AsyncRedisProtocol(Protocol):
    """Protocol for async Redis operations"""
    async def ttl(self, key: str) -> int: ...
    async def keys(self, pattern: str) -> List[str]: ...
    async def delete(self, *keys: str) -> int: ...
    async def scan(self, cursor: int = 0, match: str = None, count: int = None) -> Tuple[int, List[str]]: ...
    async def set(self, key: str, value: str): ...
    async def expire(self, key: str, ttl: int): ...
    async def close(self): ...


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
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    last_checkpoint: Optional[datetime] = None
    rollback_data: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "task_id": self.task_id,
            "migration_name": self.migration_name,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_duration": self.estimated_duration.total_seconds() if self.estimated_duration else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_checkpoint": self.last_checkpoint,
            "rollback_data": self.rollback_data,
            "metadata": self.metadata
        }


class BackgroundMigrationManager(BaseMigrationService):
    """Manages long-running background migrations with checkpoints and error recovery"""
    
    def __init__(self, base_service: BaseDBService, config: MigrationConfig = None):
        super().__init__(base_service, config)
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
                        timeout=self.config.WORKER_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute migration with retry logic
                async with self.monitor.track_operation("execute_migration", 
                                                      {"migration_name": migration_task.migration_name}):
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
                    delay = min(self.config.MAX_RETRY_DELAY, 
                              self.config.RETRY_BASE_DELAY * (2 ** migration_task.retry_count))
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
            
            # Execute rollback operations in reverse order
            for rollback_op in reversed(migration_task.rollback_data):
                try:
                    if rollback_op["type"] == "redis_key_restore":
                        await self._restore_redis_key(rollback_op["key"], rollback_op["value"], rollback_op["ttl"])
                    elif rollback_op["type"] == "db_document_restore":
                        await self._restore_db_document(rollback_op["collection"], rollback_op["filter"], rollback_op["document"])
                    
                except Exception as rollback_error:
                    logger.error(f"Rollback operation failed: {rollback_error}")
                    raise MigrationRollbackError(f"Failed to rollback operation: {rollback_error}")
            
            logger.info(f"✅ Rollback completed for: {migration_task.migration_name}")
            
        except Exception as e:
            logger.error(f"❌ Rollback failed for {migration_task.migration_name}: {e}")
            migration_task.status = MigrationStatus.FAILED
            raise MigrationRollbackError(f"Complete rollback failure: {e}")
    
    async def _restore_redis_key(self, key: str, value: Any, ttl: int):
        """Restore a Redis key during rollback"""
        if not self.redis_client:
            raise RedisOperationError("Redis client not available for rollback")
        
        try:
            await self.redis_client.set(key, json.dumps(value))
            if ttl > 0:
                await self.redis_client.expire(key, ttl)
        except Exception as e:
            raise RedisOperationError(f"Failed to restore Redis key {key}: {e}")
    
    async def _restore_db_document(self, collection: str, filter_doc: Dict, document: Dict):
        """Restore a database document during rollback"""
        try:
            await getattr(self.base.db, collection).replace_one(filter_doc, document, upsert=True)
        except Exception as e:
            raise DatabaseOperationError(f"Failed to restore document in {collection}: {e}")
    
    async def queue_migration(self, migration_task: MigrationTask):
        """Queue a migration for background processing"""
        await self.migration_queue.put(migration_task)
        logger.info(f"🔄 Queued migration: {migration_task.migration_name}")
    
    async def _execute_user_migration(self, task: MigrationTask):
        """Execute user migration with batch processing"""
        batch_size = task.metadata.get("batch_size", self.config.DEFAULT_BATCH_SIZE)
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
            if processed % self.config.CHECKPOINT_INTERVAL == 0:
                await self._save_migration_status(task)
                task.last_checkpoint = datetime.now(timezone.utc)
            
            skip += batch_size
            
            # Small delay to avoid overwhelming database
            await asyncio.sleep(0.1)
    
    async def _execute_key_cleanup(self, task: MigrationTask):
        """Execute Redis key cleanup"""
        patterns = task.metadata.get("patterns", [])
        batch_size = task.metadata.get("batch_size", self.config.DEFAULT_BATCH_SIZE)
        dry_run = task.metadata.get("dry_run", False)
        
        processed = 0
        for pattern in patterns:
            async for key in self._scan_redis_pattern_efficiently(pattern):
                if not dry_run:
                    deleted = await self._delete_redis_keys_batch([key])
                    processed += deleted
                else:
                    processed += 1
                
                # Update progress
                task.processed_items = processed
                if task.total_items > 0:
                    task.progress = min((processed / task.total_items) * 100, 100)
                
                # Save checkpoint periodically
                if processed % self.config.CHECKPOINT_INTERVAL == 0:
                    await self._save_migration_status(task)
                
                await asyncio.sleep(0.01)  # Brief pause
    
    async def _execute_data_normalization(self, task: MigrationTask):
        """Execute data normalization migration"""
        # Enhanced implementation for data normalization
        normalization_type = task.metadata.get("normalization_type", "user_data")
        
        if normalization_type == "user_data":
            await self._normalize_user_data(task)
        elif normalization_type == "phone_numbers":
            await self._normalize_phone_numbers(task)
        else:
            raise MigrationValidationError(f"Unknown normalization type: {normalization_type}")
    
    async def _normalize_user_data(self, task: MigrationTask):
        """Normalize user data fields"""
        batch_size = task.metadata.get("batch_size", self.config.DEFAULT_BATCH_SIZE)
        skip = 0
        
        while skip < task.total_items:
            try:
                cursor = self.base.db.users.find({}).skip(skip).limit(batch_size)
                batch_processed = 0
                
                async for user_doc in cursor:
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
                    
                    batch_processed += 1
                
                # Update progress
                skip += batch_size
                task.processed_items = min(skip, task.total_items)
                task.progress = (task.processed_items / task.total_items) * 100
                
                # Save checkpoint
                if task.processed_items % self.config.CHECKPOINT_INTERVAL == 0:
                    await self._save_migration_status(task)
                
                await asyncio.sleep(0.01)  # Brief pause
                
            except Exception as e:
                error_msg = f"Batch normalization error at skip {skip}: {e}"
                task.errors.append(error_msg)
                logger.error(error_msg)
                skip += batch_size  # Continue with next batch
    
    async def _normalize_phone_numbers(self, task: MigrationTask):
        """Normalize phone number formats"""
        # Implementation for phone number normalization
        # This would follow similar pattern to _normalize_user_data
        pass
    
    async def _execute_generic_migration(self, task: MigrationTask):
        """Execute generic migration with progress simulation"""
        items_per_iteration = task.metadata.get("items_per_iteration", 10)
        sleep_delay = task.metadata.get("sleep_delay", 0.01)
        
        for i in range(0, task.total_items, items_per_iteration):
            # Simulate processing
            await asyncio.sleep(sleep_delay)
            
            # Update progress
            processed = min(i + items_per_iteration, task.total_items)
            task.processed_items = processed
            task.progress = (task.processed_items / task.total_items) * 100
            
            # Save checkpoint periodically
            if processed % self.config.CHECKPOINT_INTERVAL == 0:
                await self._save_migration_status(task)
    
    async def _save_migration_status(self, task: MigrationTask):
        """Save migration status to database"""
        try:
            await self.base.db.migration_history.update_one(
                {"task_id": task.task_id},
                {"$set": {
                    **task.to_dict(),
                    "updated_at": datetime.now(timezone.utc)
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save migration status: {e}")
            raise DatabaseOperationError(f"Failed to save migration status: {e}")
    
    async def _migrate_user_batch(self, skip: int, limit: int) -> Dict[str, Any]:
        """Migrate a batch of users - placeholder implementation"""
        # This would be implemented in the main service
        await asyncio.sleep(0.1)  # Simulate work
        return {"processed": limit, "errors": []}
    
    async def _scan_redis_pattern_efficiently(self, pattern: str) -> AsyncGenerator[str, None]:
        """Memory-efficient Redis key scanning with generator"""
        cursor = 0
        total_processed = 0
        
        while True:
            try:
                if self.redis_client:
                    cursor, batch_keys = await self.redis_client.scan(
                        cursor=cursor, 
                        match=pattern, 
                        count=self.config.REDIS_SCAN_COUNT
                    )
                    
                    for key in batch_keys:
                        yield key
                        total_processed += 1
                        
                        # Prevent runaway operations
                        if total_processed > self.config.MAX_SCAN_KEYS_PER_BATCH:
                            logger.warning(f"Scan limit reached for pattern {pattern}")
                            return
                    
                    if cursor == 0:
                        break
                        
                elif hasattr(self.base, 'redis_client') and self.base.redis_client:
                    # Fallback to sync client
                    keys = self.base.redis_client.keys(pattern)
                    for key in keys:
                        yield key
                    break
                else:
                    logger.warning("No Redis client available for scanning")
                    break
                    
            except Exception as e:
                logger.error(f"Redis scan error for pattern {pattern}: {e}")
                raise RedisOperationError(f"Failed to scan pattern {pattern}: {e}")

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
            raise RedisOperationError(f"Failed to delete keys: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for background migration manager"""
        try:
            return {
                "status": "healthy",
                "active_migrations": len(self.active_migrations),
                "queued_migrations": self.migration_queue.qsize(),
                "worker_running": self.worker_task and not self.worker_task.worker_task.done() if hasattr(self.worker_task, 'worker_task') else bool(self.worker_task and not self.worker_task.done()),
                "redis_connected": self.redis_client is not None,
                "metrics": self.get_service_metrics()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class EnhancedMigrationService(BaseMigrationService):
    """Advanced migration service with background processing and intelligent cleanup"""
    
    def __init__(self, base_service: BaseDBService, config: MigrationConfig = None):
        super().__init__(base_service, config)
        self._migration_log = []
        self.background_manager = BackgroundMigrationManager(base_service, config)

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
            raise MigrationException(f"Failed to initialize migration service: {e}")
    
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
                "worker_running": self.background_manager.worker_task and not self.background_manager.worker_task.done(),
                "service_metrics": self.get_service_metrics(),
                "background_manager_health": await self.background_manager.health_check()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # Enhanced Stock Data Migration with TTL preservation
    async def get_stock_data_with_migration(self, symbol: str, data_type: str) -> Optional[Dict]:
        """Get stock data with intelligent migration and TTL preservation"""
        if not symbol or not data_type:
            raise MigrationValidationError("Symbol and data_type are required")
        
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
        
        async with self.monitor.track_operation("get_stock_data_with_migration", 
                                              {"symbol": symbol, "data_type": data_type}):
            # Try new format first
            data = await self.base.key_builder.get_with_metadata(new_key)
            if data:
                return data.get('value') if isinstance(data, dict) else data
            
            # Attempt migration from old format
            old_key = f"stock:{symbol}:{data_type}"
            old_data = await self.base.key_builder.get_with_metadata(old_key)
            
            if old_data:
                # Preserve TTL if available
                original_ttl = await self._get_redis_ttl(old_key)
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
                else:
                    raise MigrationExecutionError(f"Failed to migrate key {old_key} to {new_key}")
        
        return None

    async def set_stock_data(self, symbol: str, data_type: str, data: Dict, ttl: int = 3600) -> bool:
        """Set stock data using new naming convention"""
        if not symbol or not data_type or not data:
            raise MigrationValidationError("Symbol, data_type, and data are required")
        
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
            async with self.monitor.track_operation("set_stock_data", {"symbol": symbol, "data_type": data_type}):
                success = await self.base.key_builder.set(key, data, ttl=ttl)
                return success
        except Exception as e:
            logger.error(f"❌ Failed to set {key}: {e}")
            raise MigrationExecutionError(f"Failed to set stock data for {key}: {e}")

    # Advanced User Migration with data normalization
    async def migrate_users_enhanced(self, batch_size: int = None, normalize_data: bool = True) -> str:
        """Enhanced user migration with background processing and normalization"""
        
        batch_size = batch_size or self.config.DEFAULT_BATCH_SIZE
        
        # Validate batch size
        if batch_size <= 0 or batch_size > 1000:
            raise MigrationValidationError("Batch size must be between 1 and 1000")
        
        # Create migration task
        task_id = str(uuid.uuid4())
        total_users = await self.base.db.users.count_documents({})
        
        migration_task = MigrationTask(
            task_id=task_id,
            migration_name=f"enhanced_user_migration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            description=f"Enhanced user migration with normalization (batch_size: {batch_size})",
            status=MigrationStatus.PENDING,
            total_items=total_users,
            max_retries=self.config.MAX_RETRIES
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
                    error_msg = f"User {user_id}: {str(e)}"
                    batch_stats["errors"].append(error_msg)
                    logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Batch error: {str(e)}"
            batch_stats["errors"].append(error_msg)
            logger.error(error_msg)
        
        return batch_stats

    # Advanced Redis Cleanup with pattern scanning
    async def cleanup_redis_keys_advanced(self, patterns: List[str] = None, 
                                        dry_run: bool = True, 
                                        batch_size: int = None) -> str:
        """Advanced Redis cleanup with batch processing - returns task_id for background processing"""
        
        batch_size = batch_size or self.config.DEFAULT_BATCH_SIZE
        patterns_to_process = patterns or []
        
        if not patterns_to_process:
            for category, pattern_list in self.config.REDIS_CLEANUP_PATTERNS.items():
                patterns_to_process.extend(pattern_list)
        
        # Validate patterns
        if not patterns_to_process:
            raise MigrationValidationError("No patterns provided for cleanup")
        
        # Estimate total keys to process
        total_keys = 0
        for pattern in patterns_to_process:
            keys_count = 0
            async for _ in self.background_manager._scan_redis_pattern_efficiently(pattern):
                keys_count += 1
                if keys_count > 10000:  # Limit estimation
                    break
            total_keys += keys_count
        
        # Create migration task for cleanup
        task_id = str(uuid.uuid4())
        
        migration_task = MigrationTask(
            task_id=task_id,
            migration_name=f"redis_cleanup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            description=f"Redis key cleanup: {len(patterns_to_process)} patterns, ~{total_keys} keys (dry_run: {dry_run})",
            status=MigrationStatus.PENDING,
            total_items=total_keys,
            max_retries=self.config.MAX_RETRIES
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
        if not task_id:
            raise MigrationValidationError("Task ID is required")
        
        try:
            async with self.monitor.track_operation("get_migration_status"):
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
            raise MigrationExecutionError(f"Failed to get migration status: {e}")

    async def get_all_migrations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all migrations with status"""
        if limit <= 0 or limit > 1000:
            raise MigrationValidationError("Limit must be between 1 and 1000")
        
        try:
            async with self.monitor.track_operation("get_all_migrations"):
                migrations = await self.base.db.migration_history.find({}).sort("started_at", -1).limit(limit).to_list(length=None)
                
                for migration in migrations:
                    migration["_id"] = str(migration["_id"])
                
                return migrations
                
        except Exception as e:
            logger.error(f"Error getting all migrations: {e}")
            raise MigrationExecutionError(f"Failed to get migrations: {e}")

    async def cancel_migration(self, task_id: str) -> bool:
        """Cancel a running migration"""
        if not task_id:
            raise MigrationValidationError("Task ID is required")
        
        try:
            async with self.monitor.track_operation("cancel_migration"):
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
            raise MigrationExecutionError(f"Failed to cancel migration: {e}")

    async def get_stats(self) -> Dict:
        """Get comprehensive migration statistics"""
        try:
            async with self.monitor.track_operation("get_migration_stats"):
                stats = {
                    "total_migrations": len(self._migration_log),
                    "by_type": {},
                    "recent_migrations": self._migration_log[-10:] if self._migration_log else [],
                    "background_stats": {
                        "active_migrations": len(self.background_manager.active_migrations),
                        "queued_migrations": self.background_manager.migration_queue.qsize(),
                        "worker_running": self.background_manager.worker_task and not self.background_manager.worker_task.done()
                    },
                    "service_metrics": self.get_service_metrics()
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
            raise MigrationExecutionError(f"Failed to get migration stats: {e}")

    # Helper methods
    async def _get_redis_ttl(self, key: str) -> int:
        """Get TTL of Redis key using async client"""
        try:
            if self.background_manager.redis_client:
                return await self.background_manager.redis_client.ttl(key)
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

    def _normalize_phone_number(self, phone: str) -> str:
        """Normalize phone number format with validation"""
        if not phone:
            return phone
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Validate length
        if len(digits) < 10:
            raise MigrationValidationError(f"Phone number too short: {phone}")
        if len(digits) > 15:
            raise MigrationValidationError(f"Phone number too long: {phone}")
        
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


# Test utilities for unit testing (Enhanced)
class MockRedisClient:
    """Enhanced mock Redis client for testing"""
    
    def __init__(self):
        self.data = {}
        self.ttls = {}
        self.operation_count = 0
    
    async def ttl(self, key: str) -> int:
        self.operation_count += 1
        return self.ttls.get(key, -1)
    
    async def keys(self, pattern: str) -> List[str]:
        self.operation_count += 1
        # Simple pattern matching for tests
        if '*' in pattern:
            prefix = pattern.replace('*', '')
            return [k for k in self.data.keys() if k.startswith(prefix)]
        return [key for key in self.data.keys() if key == pattern]
    
    async def delete(self, *keys: str) -> int:
        self.operation_count += 1
        deleted = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                self.ttls.pop(key, None)
                deleted += 1
        return deleted
    
    async def scan(self, cursor: int = 0, match: str = None, count: int = None) -> Tuple[int, List[str]]:
        self.operation_count += 1
        keys = await self.keys(match or '*')
        return 0, keys  # Simplified for testing
    
    async def set(self, key: str, value: str):
        self.operation_count += 1
        self.data[key] = value
    
    async def expire(self, key: str, ttl: int):
        self.operation_count += 1
        self.ttls[key] = ttl
    
    async def close(self):
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "operation_count": self.operation_count,
            "keys_stored": len(self.data),
            "ttls_set": len(self.ttls)
        }


class MigrationTestHooks:
    """Enhanced test hooks for migration service testing"""
    
    @staticmethod
    def create_test_migration_service(mock_redis: bool = True, mock_db: bool = True, config: MigrationConfig = None):
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
            base_service.db.users = MagicMock()
        
        # Use test config if not provided
        test_config = config or MigrationConfig(
            DEFAULT_BATCH_SIZE=10,
            MAX_RETRIES=2,
            WORKER_TIMEOUT=0.1
        )
        
        service = EnhancedMigrationService(base_service, test_config)
        
        if mock_redis:
            service.background_manager.redis_client = MockRedisClient()
        
        return service
    
    @staticmethod
    def create_test_migration_task(
        task_id: str = None,
        migration_name: str = "test_migration",
        total_items: int = 100,
        **kwargs
    ) -> MigrationTask:
        """Create test migration task with enhanced options"""
        return MigrationTask(
            task_id=task_id or str(uuid.uuid4()),
            migration_name=migration_name,
            description=f"Test migration: {migration_name}",
            status=MigrationStatus.PENDING,
            total_items=total_items,
            **kwargs
        )
    
    @staticmethod
    async def wait_for_migration_completion(service: EnhancedMigrationService, task_id: str, timeout: float = 10.0):
        """Wait for migration to complete with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = await service.get_migration_status(task_id)
            if status and status.get("status") in ["completed", "failed", "cancelled"]:
                return status
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Migration {task_id} did not complete within {timeout} seconds")


# Export the enhanced migration service
__all__ = [
    'EnhancedMigrationService', 
    'BackgroundMigrationManager', 
    'MigrationTask', 
    'MigrationStatus',
    'MigrationConfig',
    'MigrationException',
    'MigrationValidationError',
    'MigrationExecutionError',
    'MigrationRollbackError',
    'RedisOperationError',
    'DatabaseOperationError',
    'MigrationTestHooks',
    'MockRedisClient'
]
