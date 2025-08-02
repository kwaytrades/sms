# services/database/mixins.py
"""
Database Service Mixins - DRY Patterns
Reusable mixins for common patterns: error handling, caching, monitoring, etc.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union, Type
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from functools import wraps
from loguru import logger

from .config import DatabaseServiceConfig
from .monitoring import DatabaseServiceMonitor
from .exceptions import (
    DatabaseServiceException,
    ServiceUnavailableError,
    ServiceTimeoutError,
    CacheOperationError,
    RetryExhaustedException
)

T = TypeVar('T')


# ==========================================
# ERROR HANDLING MIXIN
# ==========================================

class ErrorHandlingMixin:
    """Mixin providing standardized error handling patterns"""
    
    def __init__(self, config: DatabaseServiceConfig, monitor: DatabaseServiceMonitor):
        self.config = config
        self.monitor = monitor
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    def with_retry(
        self, 
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        exceptions: tuple = (Exception,),
        circuit_breaker_key: Optional[str] = None
    ):
        """
        Decorator for retry logic with exponential backoff and circuit breaker
        
        Args:
            max_retries: Maximum retry attempts (defaults to config)
            base_delay: Base delay for exponential backoff (defaults to config)
            exceptions: Tuple of exceptions to retry on
            circuit_breaker_key: Key for circuit breaker (enables circuit breaker)
        """
        max_retries = max_retries or self.config.MAX_RETRIES
        base_delay = base_delay or self.config.RETRY_BASE_DELAY
        
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                # Check circuit breaker
                if circuit_breaker_key and self._is_circuit_open(circuit_breaker_key):
                    raise ServiceUnavailableError(f"Circuit breaker open for {circuit_breaker_key}")
                
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Reset circuit breaker on success
                        if circuit_breaker_key:
                            self._reset_circuit_breaker(circuit_breaker_key)
                        
                        return result
                        
                    except exceptions as e:
                        last_exception = e
                        
                        # Record failure for circuit breaker
                        if circuit_breaker_key:
                            self._record_failure(circuit_breaker_key)
                        
                        if attempt == max_retries:
                            logger.error(f"❌ Operation failed after {max_retries} retries: {e}")
                            self.monitor.increment_counter("retry_exhausted")
                            raise RetryExhaustedException(f"Max retries exceeded: {e}")
                        
                        delay = self.config.get_retry_delay(attempt)
                        logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} in {delay}s: {e}")
                        self.monitor.increment_counter("retries_attempted")
                        
                        await asyncio.sleep(delay)
                
                # This should never be reached, but just in case
                raise last_exception or Exception("Unknown retry error")
            
            return wrapper
        return decorator
    
    def with_timeout(self, timeout_seconds: Optional[float] = None):
        """Decorator for operation timeouts"""
        timeout_seconds = timeout_seconds or self.config.HEALTH_CHECK_TIMEOUT
        
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    self.monitor.increment_counter("timeouts")
                    raise ServiceTimeoutError(f"Operation timed out after {timeout_seconds}s")
            
            return wrapper
        return decorator
    
    def with_exception_mapping(self, exception_map: Dict[Type[Exception], Type[Exception]]):
        """Decorator for mapping exceptions to service-specific exceptions"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if we should map this exception
                    for source_exc, target_exc in exception_map.items():
                        if isinstance(e, source_exc):
                            raise target_exc(f"Mapped from {type(e).__name__}: {e}")
                    
                    # Re-raise if no mapping found
                    raise
            
            return wrapper
        return decorator
    
    async def handle_service_error(
        self, 
        operation_name: str, 
        error: Exception,
        fallback_value: Any = None,
        re_raise: bool = True
    ) -> Any:
        """
        Centralized service error handling
        
        Args:
            operation_name: Name of the operation that failed
            error: The exception that occurred
            fallback_value: Value to return instead of re-raising
            re_raise: Whether to re-raise the exception
        """
        error_details = {
            "operation": operation_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.error(f"❌ Service error in {operation_name}: {error}", extra=error_details)
        self.monitor.record_error(operation_name, error_details)
        
        if re_raise:
            if isinstance(error, DatabaseServiceException):
                raise error
            else:
                raise DatabaseServiceException(f"Error in {operation_name}: {error}")
        
        return fallback_value
    
    def _is_circuit_open(self, key: str) -> bool:
        """Check if circuit breaker is open"""
        if not self.config.CIRCUIT_BREAKER_ENABLED:
            return False
        
        circuit = self._circuit_breakers.get(key)
        if not circuit:
            return False
        
        if circuit["state"] == "open":
            # Check if we should try to recover
            if time.time() - circuit["opened_at"] > self.config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT:
                circuit["state"] = "half_open"
                logger.info(f"🔄 Circuit breaker {key} entering half-open state")
            return circuit["state"] == "open"
        
        return False
    
    def _record_failure(self, key: str) -> None:
        """Record failure for circuit breaker"""
        if not self.config.CIRCUIT_BREAKER_ENABLED:
            return
        
        if key not in self._circuit_breakers:
            self._circuit_breakers[key] = {
                "state": "closed",
                "failures": 0,
                "opened_at": None
            }
        
        circuit = self._circuit_breakers[key]
        circuit["failures"] += 1
        
        if circuit["failures"] >= self.config.CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            circuit["state"] = "open"
            circuit["opened_at"] = time.time()
            logger.warning(f"⚠️ Circuit breaker {key} opened after {circuit['failures']} failures")
    
    def _reset_circuit_breaker(self, key: str) -> None:
        """Reset circuit breaker on successful operation"""
        if key in self._circuit_breakers:
            circuit = self._circuit_breakers[key]
            if circuit["state"] in ["half_open", "open"]:
                logger.info(f"✅ Circuit breaker {key} reset")
            
            circuit["state"] = "closed"
            circuit["failures"] = 0
            circuit["opened_at"] = None


# ==========================================
# CACHING MIXIN
# ==========================================

class CachingMixin:
    """Mixin providing standardized caching patterns"""
    
    def __init__(self, config: DatabaseServiceConfig, monitor: DatabaseServiceMonitor):
        self.config = config
        self.monitor = monitor
    
    async def cache_with_fallback(
        self,
        cache_key: str,
        fallback_func: Callable[[], T],
        ttl: Optional[int] = None,
        cache_type: str = "default"
    ) -> T:
        """
        Standard cache-with-fallback pattern
        
        Args:
            cache_key: Redis cache key
            fallback_func: Function to call if cache miss
            ttl: Time to live for cache entry
            cache_type: Type of cache for TTL lookup
        """
        if not self.config.CACHE_ENABLED:
            return await fallback_func()
        
        ttl = ttl or self.config.get_cache_ttl(cache_type)
        
        try:
            # Try cache first
            cached_value = await self._get_from_cache(cache_key)
            if cached_value is not None:
                self.monitor.increment_counter("cache_hits")
                return cached_value
        except Exception as e:
            if not self.config.CACHE_FALLBACK_ENABLED:
                raise CacheOperationError(f"Cache read failed: {e}")
            
            logger.warning(f"Cache read failed for {cache_key}: {e}")
        
        # Cache miss - call fallback
        self.monitor.increment_counter("cache_misses")
        result = await fallback_func()
        
        # Try to cache the result
        try:
            await self._set_to_cache(cache_key, result, ttl)
        except Exception as e:
            if not self.config.CACHE_FALLBACK_ENABLED:
                raise CacheOperationError(f"Cache write failed: {e}")
            
            logger.warning(f"Cache write failed for {cache_key}: {e}")
        
        return result
    
    async def invalidate_cache_pattern(self, pattern: str, max_keys: int = 1000) -> int:
        """
        Invalidate cache keys matching pattern
        
        Args:
            pattern: Redis key pattern (e.g., "user:*")
            max_keys: Maximum keys to delete in one operation
        
        Returns:
            Number of keys deleted
        """
        try:
            if not hasattr(self, 'key_builder') or not self.key_builder:
                logger.warning("Key builder not available for cache invalidation")
                return 0
            
            keys = await self.key_builder.get_pattern_keys(pattern, max_keys)
            deleted_count = 0
            
            # Delete in batches to avoid blocking Redis
            batch_size = 100
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                for key in batch:
                    if await self.key_builder.delete(key):
                        deleted_count += 1
                
                # Small delay between batches
                if i + batch_size < len(keys):
                    await asyncio.sleep(0.01)
            
            logger.info(f"🗑️ Invalidated {deleted_count} cache keys matching pattern: {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cache invalidation failed for pattern {pattern}: {e}")
            raise CacheOperationError(f"Cache invalidation failed: {e}")
    
    async def warm_cache(self, cache_operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Warm cache with multiple operations
        
        Args:
            cache_operations: List of dicts with 'key', 'value', 'ttl'
        
        Returns:
            Dict with success/failure counts
        """
        results = {"success": 0, "failed": 0, "errors": []}
        
        for operation in cache_operations:
            try:
                key = operation["key"]
                value = operation["value"]
                ttl = operation.get("ttl", 3600)
                
                await self._set_to_cache(key, value, ttl)
                results["success"] += 1
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Key {operation.get('key', 'unknown')}: {e}")
        
        logger.info(f"🔥 Cache warming complete: {results['success']} success, {results['failed']} failed")
        return results
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache with proper error handling"""
        if not hasattr(self, 'key_builder') or not self.key_builder:
            return None
        
        return await self.key_builder.get(key)
    
    async def _set_to_cache(self, key: str, value: Any, ttl: int) -> bool:
        """Set value to cache with proper error handling"""
        if not hasattr(self, 'key_builder') or not self.key_builder:
            return False
        
        return await self.key_builder.set(key, value, ttl)


# ==========================================
# INFERENCE MIXIN
# ==========================================

class InferenceMixin:
    """Mixin providing auto-inference capabilities"""
    
    def __init__(self, config: DatabaseServiceConfig, monitor: DatabaseServiceMonitor):
        self.config = config
        self.monitor = monitor
        self._last_inference_check: Dict[str, datetime] = {}
    
    async def should_run_inference(self, user_id: str, last_inference_at: Optional[datetime] = None) -> bool:
        """
        Determine if auto-inference should run for user
        
        Args:
            user_id: User identifier
            last_inference_at: When inference was last run
        
        Returns:
            True if inference should run
        """
        if not self.config.INFERENCE_ENABLED:
            return False
        
        # Check if we've already checked recently (avoid repeated checks)
        last_check = self._last_inference_check.get(user_id)
        if last_check and (datetime.now(timezone.utc) - last_check).seconds < 300:  # 5 minutes
            return False
        
        self._last_inference_check[user_id] = datetime.now(timezone.utc)
        
        # Check if enough time has passed since last inference
        if last_inference_at:
            time_since_inference = datetime.now(timezone.utc) - last_inference_at
            interval = timedelta(hours=self.config.INFERENCE_INTERVAL_HOURS)
            
            if time_since_inference < interval:
                return False
        
        return True
    
    async def run_inference_with_tracking(
        self,
        user_id: str,
        inference_func: Callable[[], T],
        inference_type: str = "profile"
    ) -> T:
        """
        Run inference with proper tracking and error handling
        
        Args:
            user_id: User identifier
            inference_func: Function that performs inference
            inference_type: Type of inference being performed
        
        Returns:
            Result of inference function
        """
        start_time = time.time()
        
        try:
            logger.info(f"🧠 Running {inference_type} inference for user {user_id}")
            
            result = await inference_func()
            
            duration = time.time() - start_time
            self.monitor.record_inference_success(inference_type, duration)
            self.monitor.increment_counter("inference_runs")
            
            logger.info(f"✅ {inference_type} inference completed for user {user_id} in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_inference_failure(inference_type, str(e), duration)
            
            logger.warning(f"⚠️ {inference_type} inference failed for user {user_id}: {e}")
            raise
    
    def should_cache_inference_result(self, inference_type: str, result: Any) -> bool:
        """
        Determine if inference result should be cached
        
        Args:
            inference_type: Type of inference
            result: Inference result
        
        Returns:
            True if result should be cached
        """
        if not self.config.CACHE_ENABLED:
            return False
        
        # Don't cache empty or error results
        if not result or (isinstance(result, dict) and result.get("error")):
            return False
        
        return True


# ==========================================
# HEALTH CHECK MIXIN
# ==========================================

class HealthCheckMixin:
    """Mixin providing standardized health check patterns"""
    
    def __init__(self, config: DatabaseServiceConfig, monitor: DatabaseServiceMonitor):
        self.config = config
        self.monitor = monitor
    
    async def run_health_check_with_timeout(
        self,
        service_name: str,
        health_check_func: Callable[[], Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run health check with timeout and standardized response format
        
        Args:
            service_name: Name of the service being checked
            health_check_func: Function that performs the health check
            timeout: Timeout for health check
        
        Returns:
            Standardized health check response
        """
        timeout = timeout or self.config.HEALTH_CHECK_TIMEOUT
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(health_check_func(), timeout=timeout)
            
            duration = time.time() - start_time
            
            # Standardize response format
            standardized_result = {
                "service": service_name,
                "status": result.get("status", "unknown"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response_time_ms": round(duration * 1000, 2),
                "details": result
            }
            
            self.monitor.record_health_check(service_name, standardized_result)
            
            return standardized_result
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            
            result = {
                "service": service_name,
                "status": "timeout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response_time_ms": round(duration * 1000, 2),
                "error": f"Health check timed out after {timeout}s"
            }
            
            self.monitor.record_health_check(service_name, result)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            result = {
                "service": service_name,
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response_time_ms": round(duration * 1000, 2),
                "error": str(e)
            }
            
            self.monitor.record_health_check(service_name, result)
            
            return result
    
    def determine_overall_health(self, service_healths: List[Dict[str, Any]]) -> str:
        """
        Determine overall health status from individual service healths
        
        Args:
            service_healths: List of individual service health results
        
        Returns:
            Overall health status: "healthy", "degraded", "unhealthy"
        """
        if not service_healths:
            return "unknown"
        
        healthy_count = sum(1 for health in service_healths if health.get("status") == "healthy")
        total_count = len(service_healths)
        
        health_ratio = healthy_count / total_count
        
        if health_ratio == 1.0:
            return "healthy"
        elif health_ratio >= self.config.DEGRADED_THRESHOLD:
            return "degraded"
        else:
            return "unhealthy"


# ==========================================
# MONITORING MIXIN
# ==========================================

class MonitoringMixin:
    """Mixin providing standardized monitoring and metrics patterns"""
    
    def __init__(self, config: DatabaseServiceConfig, monitor: DatabaseServiceMonitor):
        self.config = config
        self.monitor = monitor
    
    @asynccontextmanager
    async def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking operation performance
        
        Args:
            operation_name: Name of the operation
            metadata: Additional metadata to track
        """
        start_time = time.time()
        operation_id = f"{operation_name}_{int(start_time * 1000)}"
        
        self.monitor.start_operation(operation_id, operation_name, metadata or {})
        
        try:
            yield operation_id
            
            duration = time.time() - start_time
            self.monitor.complete_operation(operation_id, duration)
            
            # Log slow operations
            if duration > self.config.SLOW_OPERATION_THRESHOLD:
                logger.warning(f"🐌 Slow operation {operation_name}: {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.fail_operation(operation_id, str(e), duration)
            raise
        finally:
            self.monitor.end_operation(operation_id)
    
    def log_business_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log business events for analytics
        
        Args:
            event_type: Type of business event
            user_id: User associated with event
            details: Additional event details
        """
        event_data = {
            "event_type": event_type,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }
        
        self.monitor.record_business_event(event_data)
        
        if self.config.ENABLE_REQUEST_LOGGING:
            logger.info(f"📊 Business event: {event_type}", extra=event_data)


# ==========================================
# SERVICE VALIDATION MIXIN
# ==========================================

class ServiceValidationMixin:
    """Mixin providing service validation and dependency checking"""
    
    def __init__(self, config: DatabaseServiceConfig, monitor: DatabaseServiceMonitor):
        self.config = config
        self.monitor = monitor
    
    def require_service(self, service_name: str):
        """
        Decorator to ensure a service is available before method execution
        
        Args:
            service_name: Name of required service
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(self_inner, *args, **kwargs) -> T:
                service = getattr(self_inner, service_name, None)
                if not service:
                    raise ServiceUnavailableError(f"Required service '{service_name}' is not available")
                
                return await func(self_inner, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def fallback_on_service_unavailable(
        self,
        service_name: str,
        fallback_value: Any = None,
        log_fallback: bool = True
    ):
        """
        Decorator to provide fallback when service is unavailable
        
        Args:
            service_name: Name of service to check
            fallback_value: Value to return if service unavailable
            log_fallback: Whether to log fallback usage
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(self_inner, *args, **kwargs) -> Union[T, Any]:
                service = getattr(self_inner, service_name, None)
                if not service:
                    if log_fallback:
                        logger.warning(f"⚠️ Service '{service_name}' unavailable, using fallback")
                        self.monitor.increment_counter("fallback_used")
                    
                    return fallback_value
                
                return await func(self_inner, *args, **kwargs)
            
            return wrapper
        return decorator
    
    async def validate_dependencies(self, required_services: List[str]) -> Dict[str, Any]:
        """
        Validate that all required dependencies are available
        
        Args:
            required_services: List of required service names
        
        Returns:
            Validation result with details
        """
        validation_result = {
            "valid": True,
            "missing_services": [],
            "available_services": [],
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        for service_name in required_services:
            service = getattr(self, service_name, None)
            if service:
                validation_result["available_services"].append(service_name)
            else:
                validation_result["valid"] = False
                validation_result["missing_services"].append(service_name)
        
        return validation_result


# Export all mixins
__all__ = [
    'ErrorHandlingMixin',
    'CachingMixin', 
    'InferenceMixin',
    'HealthCheckMixin',
    'MonitoringMixin',
    'ServiceValidationMixin'
]
