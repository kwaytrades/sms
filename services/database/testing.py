# services/database/testing.py
"""
Database Service Testing Framework
Structured interface for testing with dependency injection and mocking
"""

from typing import Dict, List, Optional, Any, Type, Union, Callable
from unittest.mock import MagicMock, AsyncMock, patch
from contextlib import asynccontextmanager
import asyncio
import uuid
from datetime import datetime, timezone

from .config import DatabaseServiceConfig, DevelopmentConfig
from .core import CoreDatabaseService
from .exceptions import DatabaseServiceException


class MockServiceBuilder:
    """Builder for creating mock services with realistic behavior"""
    
    def __init__(self):
        self._mocks: Dict[str, Any] = {}
        self._behaviors: Dict[str, Dict[str, Any]] = {}
    
    def add_service_mock(
        self,
        service_name: str,
        mock_type: str = "async",
        methods: Optional[List[str]] = None,
        return_values: Optional[Dict[str, Any]] = None,
        side_effects: Optional[Dict[str, Any]] = None
    ) -> 'MockServiceBuilder':
        """
        Add a mock service with specified behavior
        
        Args:
            service_name: Name of the service to mock
            mock_type: Type of mock ("async", "sync", "class")
            methods: List of methods to mock
            return_values: Default return values for methods
            side_effects: Side effects for methods (exceptions, etc.)
        """
        if mock_type == "async":
            mock = AsyncMock()
        elif mock_type == "class":
            mock = MagicMock()
        else:
            mock = MagicMock()
        
        # Set up method behaviors
        if methods:
            for method_name in methods:
                method_mock = AsyncMock() if mock_type == "async" else MagicMock()
                
                # Set return value if specified
                if return_values and method_name in return_values:
                    if mock_type == "async":
                        method_mock.return_value = return_values[method_name]
                    else:
                        method_mock.return_value = return_values[method_name]
                
                # Set side effect if specified
                if side_effects and method_name in side_effects:
                    method_mock.side_effect = side_effects[method_name]
                
                setattr(mock, method_name, method_mock)
        
        self._mocks[service_name] = mock
        self._behaviors[service_name] = {
            "type": mock_type,
            "methods": methods or [],
            "return_values": return_values or {},
            "side_effects": side_effects or {}
        }
        
        return self
    
    def add_users_service_mock(
        self,
        return_user: Optional[Dict[str, Any]] = None,
        health_status: str = "healthy"
    ) -> 'MockServiceBuilder':
        """Add realistic users service mock"""
        default_user = {
            "_id": "test_user_id",
            "phone_number": "+1234567890",
            "plan_type": "free",
            "status": "active",
            "created_at": datetime.now(timezone.utc)
        }
        
        return self.add_service_mock(
            "users",
            mock_type="async",
            methods=[
                "get_by_phone", "get_by_id", "save", "update_activity",
                "get_usage_count", "increment_usage", "check_usage_limits",
                "get_personality_profile", "save_personality_profile",
                "health_check", "get_service_metrics"
            ],
            return_values={
                "get_by_phone": return_user or default_user,
                "get_by_id": return_user or default_user,
                "save": "test_user_id",
                "update_activity": True,
                "get_usage_count": 5,
                "increment_usage": None,
                "check_usage_limits": {
                    "limits_exceeded": {"weekly_limit": False},
                    "remaining": {"weekly_limit": 5}
                },
                "get_personality_profile": {"risk_tolerance": "moderate"},
                "save_personality_profile": True,
                "health_check": {"status": health_status},
                "get_service_metrics": {"operations": 100, "errors": 0}
            }
        )
    
    def add_trading_service_mock(
        self,
        health_status: str = "healthy"
    ) -> 'MockServiceBuilder':
        """Add realistic trading service mock"""
        return self.add_service_mock(
            "trading",
            mock_type="async",
            methods=[
                "save_goal", "get_goals", "update_goal_progress",
                "save_alert", "get_active_alerts", "record_alert_trigger",
                "save_trade_marker", "get_trade_performance",
                "health_check", "get_service_metrics"
            ],
            return_values={
                "save_goal": "test_goal_id",
                "get_goals": [{"_id": "test_goal_id", "title": "Test Goal"}],
                "update_goal_progress": True,
                "save_alert": "test_alert_id",
                "get_active_alerts": [{"_id": "test_alert_id", "symbol": "AAPL"}],
                "record_alert_trigger": True,
                "save_trade_marker": "test_marker_id",
                "get_trade_performance": [{"_id": "test_marker_id"}],
                "health_check": {"status": health_status},
                "get_service_metrics": {"operations": 50, "errors": 0}
            }
        )
    
    def add_migration_service_mock(
        self,
        health_status: str = "healthy"
    ) -> 'MockServiceBuilder':
        """Add realistic migration service mock"""
        return self.add_service_mock(
            "migrations",
            mock_type="async",
            methods=[
                "get_stock_data_with_migration", "set_stock_data",
                "migrate_users_enhanced", "cleanup_redis_keys_advanced",
                "get_migration_status", "get_stats",
                "health_check", "get_service_metrics"
            ],
            return_values={
                "get_stock_data_with_migration": {"price": 150.0, "symbol": "AAPL"},
                "set_stock_data": True,
                "migrate_users_enhanced": "test_migration_id",
                "cleanup_redis_keys_advanced": "test_cleanup_id",
                "get_migration_status": {"status": "completed", "progress": 100},
                "get_stats": {"total_migrations": 10, "successful": 9},
                "health_check": {"status": health_status},
                "get_service_metrics": {"operations": 25, "errors": 1}
            }
        )
    
    def add_base_service_mock(self) -> 'MockServiceBuilder':
        """Add base database service mock"""
        return self.add_service_mock(
            "base_service",
            mock_type="class",
            methods=["health_check", "check_rate_limit", "reset_rate_limit", "close"],
            return_values={
                "health_check": {"status": "healthy", "components": {"mongodb": {"status": "healthy"}}},
                "check_rate_limit": (True, 1),
                "reset_rate_limit": True,
                "close": None
            }
        )
    
    def build(self) -> Dict[str, Any]:
        """Build and return all mocks"""
        return self._mocks.copy()


class DatabaseServiceTestConfig(DevelopmentConfig):
    """Test-specific configuration"""
    
    # Disable external dependencies
    INFERENCE_ENABLED: bool = False
    CACHE_ENABLED: bool = False
    METRICS_ENABLED: bool = True
    
    # Faster timeouts for testing
    HEALTH_CHECK_TIMEOUT: float = 1.0
    SERVICE_INITIALIZATION_TIMEOUT: int = 5
    
    # Reduced limits for testing
    MAX_RETRIES: int = 2
    RETRY_BASE_DELAY: float = 0.1
    MAX_RETRY_DELAY: int = 1
    
    # Test-specific settings
    CIRCUIT_BREAKER_ENABLED: bool = False
    ENABLE_REQUEST_LOGGING: bool = False
    DEBUG_MODE: bool = True


class TestDatabaseService(CoreDatabaseService):
    """
    Test version of database service with dependency injection support
    """
    
    def __init__(self, config: Optional[DatabaseServiceConfig] = None):
        super().__init__(config or DatabaseServiceTestConfig())
        self._test_mode = True
        self._injected_services: Dict[str, Any] = {}
    
    def inject_services(self, service_mocks: Dict[str, Any]) -> None:
        """
        Inject mock services for testing
        
        Args:
            service_mocks: Dictionary of service_name -> mock_object
        """
        self._injected_services.update(service_mocks)
        
        # Inject into service manager
        self.service_manager.inject_test_services(service_mocks)
        
        # Also set as direct attributes for property access
        for service_name, mock_service in service_mocks.items():
            setattr(self, service_name, mock_service)
    
    async def initialize_for_testing(self, mock_base_service: bool = True) -> None:
        """
        Initialize service for testing with optional base service mocking
        
        Args:
            mock_base_service: Whether to mock the base database service
        """
        if mock_base_service:
            # Create mock base service
            base_mock = AsyncMock()
            base_mock.initialize = AsyncMock()
            base_mock.health_check = AsyncMock(return_value={"status": "healthy"})
            base_mock.mongo_client = MagicMock()
            base_mock.db = MagicMock()
            base_mock.redis = AsyncMock()
            base_mock.key_builder = AsyncMock()
            
            self.base_service = base_mock
            
            # Skip actual initialization
            await self.service_manager.initialize_test_services(base_mock)
        else:
            # Use real initialization
            await self.initialize()
    
    def get_mock_service(self, service_name: str) -> Optional[Any]:
        """Get injected mock service by name"""
        return self._injected_services.get(service_name)
    
    def assert_service_called(self, service_name: str, method_name: str, *args, **kwargs) -> None:
        """Assert that a service method was called with specific arguments"""
        service = self.get_mock_service(service_name)
        if not service:
            raise AssertionError(f"Service '{service_name}' not found in injected services")
        
        method = getattr(service, method_name, None)
        if not method:
            raise AssertionError(f"Method '{method_name}' not found on service '{service_name}'")
        
        if args or kwargs:
            method.assert_called_with(*args, **kwargs)
        else:
            method.assert_called()
    
    def get_service_call_count(self, service_name: str, method_name: str) -> int:
        """Get the number of times a service method was called"""
        service = self.get_mock_service(service_name)
        if not service:
            return 0
        
        method = getattr(service, method_name, None)
        if not method:
            return 0
        
        return method.call_count


class DatabaseServiceTestRunner:
    """
    Test runner for database service with common test scenarios
    """
    
    def __init__(self, service: TestDatabaseService):
        self.service = service
        self.test_results: List[Dict[str, Any]] = []
    
    async def run_basic_functionality_tests(self) -> Dict[str, Any]:
        """Run basic functionality tests"""
        test_suite = {
            "name": "Basic Functionality",
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
        
        # Test user operations
        user_tests = await self._test_user_operations()
        test_suite["tests"].extend(user_tests)
        
        # Test trading operations
        trading_tests = await self._test_trading_operations()
        test_suite["tests"].extend(trading_tests)
        
        # Test health checks
        health_tests = await self._test_health_checks()
        test_suite["tests"].extend(health_tests)
        
        # Calculate summary
        test_suite["summary"]["total"] = len(test_suite["tests"])
        test_suite["summary"]["passed"] = sum(1 for t in test_suite["tests"] if t["passed"])
        test_suite["summary"]["failed"] = test_suite["summary"]["total"] - test_suite["summary"]["passed"]
        
        return test_suite
    
    async def run_error_handling_tests(self) -> Dict[str, Any]:
        """Run error handling and resilience tests"""
        test_suite = {
            "name": "Error Handling",
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
        
        # Test service unavailable scenarios
        unavailable_tests = await self._test_service_unavailable_scenarios()
        test_suite["tests"].extend(unavailable_tests)
        
        # Test timeout handling
        timeout_tests = await self._test_timeout_handling()
        test_suite["tests"].extend(timeout_tests)
        
        # Test retry mechanisms
        retry_tests = await self._test_retry_mechanisms()
        test_suite["tests"].extend(retry_tests)
        
        # Calculate summary
        test_suite["summary"]["total"] = len(test_suite["tests"])
        test_suite["summary"]["passed"] = sum(1 for t in test_suite["tests"] if t["passed"])
        test_suite["summary"]["failed"] = test_suite["summary"]["total"] - test_suite["summary"]["passed"]
        
        return test_suite
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and load tests"""
        test_suite = {
            "name": "Performance",
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
        
        # Test concurrent operations
        concurrent_tests = await self._test_concurrent_operations()
        test_suite["tests"].extend(concurrent_tests)
        
        # Test memory usage
        memory_tests = await self._test_memory_usage()
        test_suite["tests"].extend(memory_tests)
        
        # Calculate summary
        test_suite["summary"]["total"] = len(test_suite["tests"])
        test_suite["summary"]["passed"] = sum(1 for t in test_suite["tests"] if t["passed"])
        test_suite["summary"]["failed"] = test_suite["summary"]["total"] - test_suite["summary"]["passed"]
        
        return test_suite
    
    async def _test_user_operations(self) -> List[Dict[str, Any]]:
        """Test user-related operations"""
        tests = []
        
        # Test get user by phone
        try:
            user = await self.service.get_user_by_phone("+1234567890")
            tests.append({
                "name": "get_user_by_phone",
                "passed": user is not None,
                "details": f"Returned user: {user is not None}"
            })
        except Exception as e:
            tests.append({
                "name": "get_user_by_phone",
                "passed": False,
                "error": str(e)
            })
        
        # Test get user by ID
        try:
            user = await self.service.get_user_by_id("test_user_id")
            tests.append({
                "name": "get_user_by_id",
                "passed": user is not None,
                "details": f"Returned user: {user is not None}"
            })
        except Exception as e:
            tests.append({
                "name": "get_user_by_id",
                "passed": False,
                "error": str(e)
            })
        
        # Test usage tracking
        try:
            usage_count = await self.service.get_usage_count("test_user_id", "week")
            tests.append({
                "name": "get_usage_count",
                "passed": isinstance(usage_count, int),
                "details": f"Usage count: {usage_count}"
            })
        except Exception as e:
            tests.append({
                "name": "get_usage_count",
                "passed": False,
                "error": str(e)
            })
        
        return tests
    
    async def _test_trading_operations(self) -> List[Dict[str, Any]]:
        """Test trading-related operations"""
        tests = []
        
        # Test save goal
        try:
            goal_data = {
                "title": "Test Goal",
                "target_amount": 10000,
                "goal_type": "savings"
            }
            goal_id = await self.service.save_financial_goal("test_user_id", goal_data)
            tests.append({
                "name": "save_financial_goal",
                "passed": goal_id is not None,
                "details": f"Goal ID: {goal_id}"
            })
        except Exception as e:
            tests.append({
                "name": "save_financial_goal",
                "passed": False,
                "error": str(e)
            })
        
        # Test get goals
        try:
            goals = await self.service.get_user_goals("test_user_id")
            tests.append({
                "name": "get_user_goals",
                "passed": isinstance(goals, list),
                "details": f"Goals count: {len(goals)}"
            })
        except Exception as e:
            tests.append({
                "name": "get_user_goals",
                "passed": False,
                "error": str(e)
            })
        
        # Test save alert
        try:
            alert_data = {
                "alert_type": "price_above",
                "symbol": "AAPL",
                "target_value": 150.0,
                "condition": "price > 150"
            }
            alert_id = await self.service.save_user_alert("test_user_id", alert_data)
            tests.append({
                "name": "save_user_alert",
                "passed": alert_id is not None,
                "details": f"Alert ID: {alert_id}"
            })
        except Exception as e:
            tests.append({
                "name": "save_user_alert",
                "passed": False,
                "error": str(e)
            })
        
        return tests
    
    async def _test_health_checks(self) -> List[Dict[str, Any]]:
        """Test health check functionality"""
        tests = []
        
        # Test service health check
        try:
            health = await self.service.health_check()
            tests.append({
                "name": "service_health_check",
                "passed": health.get("overall_status") in ["healthy", "degraded"],
                "details": f"Status: {health.get('overall_status')}"
            })
        except Exception as e:
            tests.append({
                "name": "service_health_check",
                "passed": False,
                "error": str(e)
            })
        
        # Test service metrics
        try:
            metrics = await self.service.get_service_metrics()
            tests.append({
                "name": "get_service_metrics",
                "passed": isinstance(metrics, dict),
                "details": f"Metrics keys: {list(metrics.keys())}"
            })
        except Exception as e:
            tests.append({
                "name": "get_service_metrics",
                "passed": False,
                "error": str(e)
            })
        
        return tests
    
    async def _test_service_unavailable_scenarios(self) -> List[Dict[str, Any]]:
        """Test behavior when services are unavailable"""
        tests = []
        
        # Temporarily remove a service to test fallback
        original_users_service = self.service.users
        self.service.users = None
        
        try:
            # This should use fallback or raise appropriate exception
            user = await self.service.get_user_by_phone("+1234567890")
            tests.append({
                "name": "service_unavailable_fallback",
                "passed": True,  # If it doesn't crash, it's handling the unavailable service
                "details": "Service unavailable handled gracefully"
            })
        except Exception as e:
            # Check if it's the expected exception type
            expected_exception = "ServiceUnavailableError" in str(type(e))
            tests.append({
                "name": "service_unavailable_exception",
                "passed": expected_exception,
                "details": f"Exception type: {type(e).__name__}"
            })
        finally:
            # Restore service
            self.service.users = original_users_service
        
        return tests
    
    async def _test_timeout_handling(self) -> List[Dict[str, Any]]:
        """Test timeout handling"""
        tests = []
        
        # Mock a slow operation
        if self.service.users:
            original_method = self.service.users.health_check
            
            async def slow_health_check():
                await asyncio.sleep(2.0)  # Longer than timeout
                return {"status": "healthy"}
            
            self.service.users.health_check = slow_health_check
            
            try:
                # This should timeout
                health = await self.service.run_health_check_with_timeout(
                    "users",
                    self.service.users.health_check,
                    timeout=0.5
                )
                
                tests.append({
                    "name": "timeout_handling",
                    "passed": health.get("status") == "timeout",
                    "details": f"Health status: {health.get('status')}"
                })
            except Exception as e:
                tests.append({
                    "name": "timeout_handling",
                    "passed": False,
                    "error": str(e)
                })
            finally:
                # Restore original method
                self.service.users.health_check = original_method
        
        return tests
    
    async def _test_retry_mechanisms(self) -> List[Dict[str, Any]]:
        """Test retry mechanisms"""
        tests = []
        
        # Test retry decorator
        call_count = 0
        
        @self.service.with_retry(max_retries=2, base_delay=0.1)
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        try:
            result = await failing_operation()
            tests.append({
                "name": "retry_mechanism_success",
                "passed": result == "success" and call_count == 3,
                "details": f"Call count: {call_count}, Result: {result}"
            })
        except Exception as e:
            tests.append({
                "name": "retry_mechanism_success",
                "passed": False,
                "error": str(e)
            })
        
        return tests
    
    async def _test_concurrent_operations(self) -> List[Dict[str, Any]]:
        """Test concurrent operations"""
        tests = []
        
        # Test concurrent user lookups
        try:
            tasks = []
            for i in range(10):
                task = self.service.get_user_by_phone(f"+123456789{i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            tests.append({
                "name": "concurrent_user_lookups",
                "passed": len(successful_results) >= 8,  # Allow some failures
                "details": f"Successful: {len(successful_results)}/10"
            })
        except Exception as e:
            tests.append({
                "name": "concurrent_user_lookups",
                "passed": False,
                "error": str(e)
            })
        
        return tests
    
    async def _test_memory_usage(self) -> List[Dict[str, Any]]:
        """Test memory usage patterns"""
        tests = []
        
        # Test that metrics don't grow unbounded
        initial_metrics = self.service.monitor.get_metrics()
        
        # Perform many operations
        for i in range(100):
            await self.service.get_user_by_phone(f"+123456789{i % 10}")
        
        final_metrics = self.service.monitor.get_metrics()
        
        # Check that active operations don't accumulate
        tests.append({
            "name": "memory_usage_operations",
            "passed": final_metrics["active_operations"] == 0,
            "details": f"Active operations: {final_metrics['active_operations']}"
        })
        
        return tests


class DatabaseServiceTestSuite:
    """
    Complete test suite for database service
    """
    
    @classmethod
    async def run_full_test_suite(
        cls,
        include_performance: bool = False,
        custom_config: Optional[DatabaseServiceConfig] = None
    ) -> Dict[str, Any]:
        """
        Run complete test suite
        
        Args:
            include_performance: Whether to include performance tests
            custom_config: Custom configuration for testing
        
        Returns:
            Complete test results
        """
        test_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_suites": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            }
        }
        
        # Create test service
        service = TestDatabaseService(custom_config)
        
        # Set up mocks
        mock_builder = MockServiceBuilder()
        mocks = (mock_builder
                .add_users_service_mock()
                .add_trading_service_mock()
                .add_migration_service_mock()
                .add_base_service_mock()
                .build())
        
        service.inject_services(mocks)
        await service.initialize_for_testing(mock_base_service=True)
        
        # Create test runner
        runner = DatabaseServiceTestRunner(service)
        
        try:
            # Run test suites
            basic_tests = await runner.run_basic_functionality_tests()
            test_results["test_suites"].append(basic_tests)
            
            error_tests = await runner.run_error_handling_tests()
            test_results["test_suites"].append(error_tests)
            
            if include_performance:
                performance_tests = await runner.run_performance_tests()
                test_results["test_suites"].append(performance_tests)
            
            # Calculate overall summary
            for suite in test_results["test_suites"]:
                test_results["summary"]["total_tests"] += suite["summary"]["total"]
                test_results["summary"]["passed_tests"] += suite["summary"]["passed"]
                test_results["summary"]["failed_tests"] += suite["summary"]["failed"]
            
            if test_results["summary"]["total_tests"] > 0:
                test_results["summary"]["success_rate"] = (
                    test_results["summary"]["passed_tests"] / 
                    test_results["summary"]["total_tests"]
                ) * 100
            
        finally:
            # Cleanup
            await service.close()
        
        return test_results
    
    @classmethod
    async def run_integration_tests(
        cls,
        real_database: bool = False
    ) -> Dict[str, Any]:
        """
        Run integration tests with optional real database connections
        
        Args:
            real_database: Whether to use real database connections
        
        Returns:
            Integration test results
        """
        test_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_type": "integration",
            "real_database": real_database,
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
        
        # Create service
        if real_database:
            service = CoreDatabaseService()
            try:
                await service.initialize()
            except Exception as e:
                test_results["tests"].append({
                    "name": "database_connection",
                    "passed": False,
                    "error": f"Failed to connect to real database: {e}"
                })
                return test_results
        else:
            service = TestDatabaseService()
            mock_builder = MockServiceBuilder()
            mocks = (mock_builder
                    .add_users_service_mock()
                    .add_trading_service_mock()
                    .add_migration_service_mock()
                    .add_base_service_mock()
                    .build())
            
            service.inject_services(mocks)
            await service.initialize_for_testing(mock_base_service=True)
        
        try:
            # Test full user workflow
            workflow_test = await cls._test_user_workflow(service)
            test_results["tests"].append(workflow_test)
            
            # Test trading workflow
            trading_test = await cls._test_trading_workflow(service)
            test_results["tests"].append(trading_test)
            
            # Test service coordination
            coordination_test = await cls._test_service_coordination(service)
            test_results["tests"].append(coordination_test)
            
            # Calculate summary
            test_results["summary"]["total"] = len(test_results["tests"])
            test_results["summary"]["passed"] = sum(1 for t in test_results["tests"] if t["passed"])
            test_results["summary"]["failed"] = test_results["summary"]["total"] - test_results["summary"]["passed"]
            
        finally:
            await service.close()
        
        return test_results
    
    @classmethod
    async def _test_user_workflow(cls, service) -> Dict[str, Any]:
        """Test complete user workflow"""
        try:
            # Create user workflow
            phone = "+1234567890"
            
            # Get user (might not exist)
            user = await service.get_user_by_phone(phone)
            
            # Update activity
            await service.update_user_activity(phone)
            
            # Check usage
            if user:
                usage = await service.get_usage_count(user.get("_id", "test"), "week")
            
            return {
                "name": "user_workflow",
                "passed": True,
                "details": "User workflow completed successfully"
            }
        except Exception as e:
            return {
                "name": "user_workflow",
                "passed": False,
                "error": str(e)
            }
    
    @classmethod
    async def _test_trading_workflow(cls, service) -> Dict[str, Any]:
        """Test complete trading workflow"""
        try:
            user_id = "test_user_id"
            
            # Create goal
            goal_data = {
                "title": "Integration Test Goal",
                "target_amount": 50000,
                "goal_type": "investment"
            }
            goal_id = await service.save_financial_goal(user_id, goal_data)
            
            # Get goals
            goals = await service.get_user_goals(user_id)
            
            # Create alert
            alert_data = {
                "alert_type": "price_above",
                "symbol": "TSLA",
                "target_value": 200.0,
                "condition": "price > 200"
            }
            alert_id = await service.save_user_alert(user_id, alert_data)
            
            # Get alerts
            alerts = await service.get_active_alerts(user_id)
            
            return {
                "name": "trading_workflow",
                "passed": True,
                "details": f"Created goal {goal_id} and alert {alert_id}"
            }
        except Exception as e:
            return {
                "name": "trading_workflow",
                "passed": False,
                "error": str(e)
            }
    
    @classmethod
    async def _test_service_coordination(cls, service) -> Dict[str, Any]:
        """Test coordination between services"""
        try:
            # Test health checks across services
            health = await service.health_check()
            
            # Test metrics collection
            metrics = await service.get_service_metrics()
            
            # Test service validation
            validation = await service.validate_service_availability()
            
            return {
                "name": "service_coordination",
                "passed": (
                    health.get("overall_status") in ["healthy", "degraded"] and
                    isinstance(metrics, dict) and
                    isinstance(validation, dict)
                ),
                "details": f"Health: {health.get('overall_status')}, Services validated: {validation.get('all_required_available')}"
            }
        except Exception as e:
            return {
                "name": "service_coordination",
                "passed": False,
                "error": str(e)
            }


# Utility functions for testing
def create_test_service_with_mocks(
    mock_users: bool = True,
    mock_trading: bool = True,
    mock_migrations: bool = True,
    custom_config: Optional[DatabaseServiceConfig] = None
) -> TestDatabaseService:
    """
    Convenience function to create test service with common mocks
    
    Args:
        mock_users: Whether to mock users service
        mock_trading: Whether to mock trading service
        mock_migrations: Whether to mock migrations service
        custom_config: Custom configuration
    
    Returns:
        Configured test service
    """
    service = TestDatabaseService(custom_config)
    
    mock_builder = MockServiceBuilder()
    
    if mock_users:
        mock_builder.add_users_service_mock()
    
    if mock_trading:
        mock_builder.add_trading_service_mock()
    
    if mock_migrations:
        mock_builder.add_migration_service_mock()
    
    mock_builder.add_base_service_mock()
    
    mocks = mock_builder.build()
    service.inject_services(mocks)
    
    return service


@asynccontextmanager
async def test_database_service(
    mock_services: Optional[Dict[str, Any]] = None,
    custom_config: Optional[DatabaseServiceConfig] = None
):
    """
    Async context manager for test database service
    
    Args:
        mock_services: Custom mock services to inject
        custom_config: Custom configuration
    
    Yields:
        Configured and initialized test service
    """
    service = TestDatabaseService(custom_config)
    
    if mock_services:
        service.inject_services(mock_services)
    else:
        # Use default mocks
        mock_builder = MockServiceBuilder()
        mocks = (mock_builder
                .add_users_service_mock()
                .add_trading_service_mock()
                .add_migration_service_mock()
                .add_base_service_mock()
                .build())
        service.inject_services(mocks)
    
    try:
        await service.initialize_for_testing(mock_base_service=True)
        yield service
    finally:
        await service.close()


# Export testing utilities
__all__ = [
    'MockServiceBuilder',
    'DatabaseServiceTestConfig',
    'TestDatabaseService',
    'DatabaseServiceTestRunner',
    'DatabaseServiceTestSuite',
    'create_test_service_with_mocks',
    'test_database_service'
]
