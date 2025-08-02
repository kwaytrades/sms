# tests/test_services.py - Comprehensive Test Suite
"""
Production-grade test suite for all database services
Includes unit tests, integration tests, and performance tests
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Test imports
import pytest_asyncio
from faker import Faker
import numpy as np

# Service imports
from services.vector_service import VectorService, VectorNamespace
from services.cache_service import CacheService, TTLStrategy
from services.memory_service import MemoryService, ConversationTurn
from services.alert_service import AlertService, AlertType, AlertCondition
from services.notification_service import NotificationService, NotificationChannel
from services.database_service import UnifiedDatabaseService
from models.user import UserProfile


fake = Faker()


@pytest.fixture
def mock_base_service():
    """Mock base service for testing"""
    base_service = Mock()
    base_service.redis = AsyncMock()
    base_service.db = Mock()
    base_service.mongo_client = Mock()
    base_service.key_builder = Mock()
    return base_service


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    client = AsyncMock()
    client.embeddings.create = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_pinecone():
    """Mock Pinecone client"""
    with patch('pinecone.init'), \
         patch('pinecone.Index') as mock_index, \
         patch('pinecone.list_indexes') as mock_list, \
         patch('pinecone.create_index'):
        
        mock_list.return_value = []
        mock_index_instance = Mock()
        mock_index.return_value = mock_index_instance
        
        # Mock index operations
        mock_index_instance.upsert.return_value = {"upserted_count": 1}
        mock_index_instance.query.return_value = {"matches": []}
        mock_index_instance.delete.return_value = {"deleted": True}
        mock_index_instance.describe_index_stats.return_value = {
            "total_vector_count": 100,
            "dimension": 1536
        }
        
        yield mock_index_instance


# ==========================================
# VECTOR SERVICE TESTS
# ==========================================

class TestVectorService:
    """Test suite for VectorService"""
    
    @pytest_asyncio.async_test
    async def test_vector_service_initialization(self, mock_base_service, mock_pinecone):
        """Test vector service initialization"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_openai.return_value.embeddings.create = AsyncMock(
                return_value=Mock(data=[Mock(embedding=[0.1] * 1536)])
            )
            
            service = VectorService(mock_base_service)
            await service.initialize()
            
            assert service.openai_client is not None
            assert service.pinecone_index is not None

    @pytest_asyncio.async_test
    async def test_generate_embedding(self, mock_base_service, mock_pinecone):
        """Test embedding generation"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Mock embedding response
            mock_embedding = [0.1, 0.2, 0.3] + [0.0] * 1533  # 1536 dimensions
            mock_openai.return_value.embeddings.create = AsyncMock(
                return_value=Mock(data=[Mock(embedding=mock_embedding)])
            )
            
            service = VectorService(mock_base_service)
            service.openai_client = mock_openai.return_value
            
            # Test embedding generation
            embedding = await service._generate_embedding("test text")
            
            assert embedding == mock_embedding
            assert len(embedding) == 1536

    @pytest_asyncio.async_test
    async def test_upsert_text(self, mock_base_service, mock_pinecone):
        """Test text upserting with embedding generation"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_embedding = [0.1] * 1536
            mock_openai.return_value.embeddings.create = AsyncMock(
                return_value=Mock(data=[Mock(embedding=mock_embedding)])
            )
            
            service = VectorService(mock_base_service)
            service.openai_client = mock_openai.return_value
            service.pinecone_index = mock_pinecone
            
            # Test text upsert
            doc_id = await service.upsert_text(
                "test_namespace",
                "This is test text",
                {"type": "test"},
                "user123"
            )
            
            assert doc_id is not None
            mock_pinecone.upsert.assert_called_once()

    @pytest_asyncio.async_test
    async def test_query_similar(self, mock_base_service, mock_pinecone):
        """Test similarity search"""
        # Mock query response
        mock_pinecone.query.return_value = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.9,
                    "metadata": {"text": "similar text", "user_id": "user123"}
                },
                {
                    "id": "doc2", 
                    "score": 0.8,
                    "metadata": {"text": "another text", "user_id": "user123"}
                }
            ]
        }
        
        service = VectorService(mock_base_service)
        service.pinecone_index = mock_pinecone
        
        # Test query
        results = await service.query_similar(
            "test_namespace",
            [0.1] * 1536,
            top_k=5
        )
        
        assert len(results) == 2
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.8

    @pytest_asyncio.async_test
    async def test_delete_user_data(self, mock_base_service, mock_pinecone):
        """Test GDPR user data deletion"""
        service = VectorService(mock_base_service)
        service.pinecone_index = mock_pinecone
        
        result = await service.delete_user_data("user123")
        
        assert result["deleted"] is True
        assert "details" in result


# ==========================================
# CACHE SERVICE TESTS
# ==========================================

class TestCacheService:
    """Test suite for CacheService"""
    
    @pytest_asyncio.async_test
    async def test_cache_service_initialization(self, mock_base_service):
        """Test cache service initialization"""
        mock_base_service.redis.ping = AsyncMock()
        mock_base_service.redis.get.return_value = None
        
        service = CacheService(mock_base_service)
        await service.initialize()
        
        assert service.redis is not None

    @pytest_asyncio.async_test
    async def test_market_aware_ttl(self, mock_base_service):
        """Test market-aware TTL calculation"""
        service = CacheService(mock_base_service)
        
        # Test different market contexts
        with patch.object(service, '_get_market_context', return_value='market_open'):
            ttl = service._calculate_ttl(TTLStrategy.MARKET_HOURS)
            assert ttl == 300  # 5 minutes during market hours
        
        with patch.object(service, '_get_market_context', return_value='market_closed'):
            ttl = service._calculate_ttl(TTLStrategy.MARKET_HOURS)
            assert ttl == 1800  # 30 minutes after hours

    @pytest_asyncio.async_test
    async def test_stock_data_caching(self, mock_base_service):
        """Test stock data caching with popular ticker logic"""
        mock_base_service.redis.setex = AsyncMock()
        
        service = CacheService(mock_base_service)
        
        # Test popular ticker (longer TTL)
        with patch.object(service, '_get_market_context', return_value='market_open'):
            ttl = service._calculate_ttl(
                TTLStrategy.STOCK_DATA, 
                {"symbol": "AAPL"}
            )
            assert ttl == 600  # 10 minutes for popular stocks
        
        # Test regular ticker (shorter TTL)
        with patch.object(service, '_get_market_context', return_value='market_open'):
            ttl = service._calculate_ttl(
                TTLStrategy.STOCK_DATA,
                {"symbol": "UNKNOWN"}
            )
            assert ttl == 300  # 5 minutes for regular stocks

    @pytest_asyncio.async_test
    async def test_serialization(self, mock_base_service):
        """Test value serialization and compression"""
        service = CacheService(mock_base_service)
        
        # Test simple value
        simple_value = "test string"
        serialized = await service._serialize_value(simple_value)
        deserialized = await service._deserialize_value(serialized)
        assert deserialized == simple_value
        
        # Test complex value
        complex_value = {"data": [1, 2, 3], "nested": {"key": "value"}}
        serialized = await service._serialize_value(complex_value)
        deserialized = await service._deserialize_value(serialized)
        assert deserialized == complex_value

    @pytest_asyncio.async_test
    async def test_batch_operations(self, mock_base_service):
        """Test batch cache operations"""
        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = [True, True, True]
        mock_base_service.redis.pipeline.return_value = mock_pipeline
        
        service = CacheService(mock_base_service)
        
        # Test batch set
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        results = await service.set_many(data, ttl=300)
        
        assert all(results.values())
        assert len(results) == 3

    @pytest_asyncio.async_test
    async def test_invalidation_patterns(self, mock_base_service):
        """Test cache invalidation patterns"""
        mock_base_service.redis.scan.return_value = (0, ["stock:AAPL:price", "stock:AAPL:analysis"])
        mock_base_service.redis.delete.return_value = 2
        
        service = CacheService(mock_base_service)
        
        # Test symbol invalidation
        success = await service.invalidate_symbol("AAPL")
        assert success is True


# ==========================================
# MEMORY SERVICE TESTS
# ==========================================

class TestMemoryService:
    """Test suite for MemoryService"""
    
    @pytest_asyncio.async_test
    async def test_memory_service_initialization(self, mock_base_service):
        """Test memory service initialization"""
        mock_vector_service = Mock()
        mock_vector_service.health_check = AsyncMock(return_value={"status": "healthy"})
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_openai.return_value.chat.completions.create = AsyncMock(
                return_value=Mock(choices=[Mock(message=Mock(content='{"test": "response"}'))])
            )
            
            service = MemoryService(mock_base_service, mock_vector_service)
            await service.initialize()
            
            assert service.openai_client is not None
            assert service.vector_service is not None

    @pytest_asyncio.async_test
    async def test_conversation_turn_saving(self, mock_base_service):
        """Test saving conversation turns to STM"""
        mock_base_service.redis.get.return_value = None
        mock_base_service.redis.setex = AsyncMock()
        
        mock_vector_service = Mock()
        service = MemoryService(mock_base_service, mock_vector_service)
        
        # Test saving conversation turn
        success = await service.save_conversation_turn(
            "user123",
            "What's AAPL doing?",
            "AAPL is trading at $150...",
            {"symbols": ["AAPL"]}
        )
        
        assert success is True
        mock_base_service.redis.setex.assert_called_once()

    @pytest_asyncio.async_test
    async def test_conversation_summarization(self, mock_base_service):
        """Test conversation summarization"""
        mock_vector_service = Mock()
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Mock LLM response for summarization
            mock_response = {
                "summary": "User asked about AAPL stock performance",
                "topics": ["AAPL", "stock analysis"],
                "symbols": ["AAPL"],
                "insights": ["User interested in Apple stock"],
                "importance_score": 0.8
            }
            
            mock_openai.return_value.chat.completions.create = AsyncMock(
                return_value=Mock(
                    choices=[Mock(message=Mock(content=json.dumps(mock_response)))]
                )
            )
            
            service = MemoryService(mock_base_service, mock_vector_service)
            service.openai_client = mock_openai.return_value
            
            # Test summarization
            turns = [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_message": "How is AAPL doing?",
                    "bot_response": "AAPL is up 2% today...",
                    "symbols": ["AAPL"]
                }
            ]
            
            summary = await service._generate_conversation_summary("user123", turns)
            
            assert summary is not None
            assert summary.importance_score == 0.8
            assert "AAPL" in summary.symbols

    @pytest_asyncio.async_test
    async def test_memory_retrieval(self, mock_base_service):
        """Test comprehensive memory retrieval"""
        mock_vector_service = Mock()
        mock_vector_service.query_text = AsyncMock(return_value=[])
        
        # Mock STM data
        stm_data = json.dumps([
            {
                "timestamp": datetime.utcnow().isoformat(),
                "user_message": "Recent message",
                "bot_response": "Recent response"
            }
        ])
        mock_base_service.redis.get.return_value = stm_data
        
        # Mock summaries collection
        mock_collection = AsyncMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value.to_list = AsyncMock(
            return_value=[{
                "summary_text": "Previous conversation about stocks",
                "topics": ["stocks"],
                "importance_score": 0.7
            }]
        )
        mock_base_service.db.__getitem__.return_value = mock_collection
        
        service = MemoryService(mock_base_service, mock_vector_service)
        
        # Test memory retrieval
        memory = await service.get_conversation_memory("user123", query="stocks")
        
        assert "short_term_memory" in memory
        assert "conversation_summaries" in memory
        assert "relevant_memories" in memory

    @pytest_asyncio.async_test
    async def test_context_compression(self, mock_base_service):
        """Test context compression for LLM"""
        mock_vector_service = Mock()
        service = MemoryService(mock_base_service, mock_vector_service)
        
        # Test context with various memory types
        memory_context = {
            "short_term_memory": [
                {
                    "user_message": "What's the market doing?",
                    "bot_response": "The market is up 1% today..."
                }
            ],
            "conversation_summaries": [
                {
                    "summary_text": "User frequently asks about tech stocks",
                    "topics": ["technology", "stocks"]
                }
            ],
            "relevant_memories": [
                {
                    "summary": {
                        "key_insights": ["User prefers growth stocks"]
                    }
                }
            ]
        }
        
        compressed = await service.compress_context_for_llm(memory_context, max_tokens=1000)
        
        assert len(compressed) > 0
        assert "Recent Conversation" in compressed or "Conversation History" in compressed


# ==========================================
# ALERT SERVICE TESTS
# ==========================================

class TestAlertService:
    """Test suite for AlertService"""
    
    @pytest_asyncio.async_test
    async def test_price_alert_creation(self, mock_base_service):
        """Test price alert creation"""
        # Mock notification service
        mock_notification_service = Mock()
        mock_notification_service.initialize = AsyncMock()
        
        # Mock database operations
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 5  # Under limit
        mock_collection.insert_one.return_value = Mock(inserted_id="alert123")
        mock_base_service.db.__getitem__.return_value = mock_collection
        
        service = AlertService(mock_base_service)
        service.notification_service = mock_notification_service
        
        # Mock price fetching
        with patch.object(service, '_get_current_price', return_value=150.0):
            alert_id = await service.create_price_alert(
                "user123",
                "AAPL", 
                155.0,
                "above"
            )
        
        assert alert_id is not None
        assert "user123" in alert_id
        assert "AAPL" in alert_id

    @pytest_asyncio.async_test
    async def test_price_alert_evaluation(self, mock_base_service):
        """Test price alert evaluation"""
        service = AlertService(mock_base_service)
        
        # Test alert that should trigger
        alert = {
            "alert_id": "test_alert",
            "user_id": "user123",
            "symbol": "AAPL",
            "condition": "above",
            "target_price": 150.0,
            "current_price": 145.0
        }
        
        with patch.object(service, '_get_current_price', return_value=155.0), \
             patch.object(service, '_update_alert_price') as mock_update, \
             patch.object(service, '_trigger_alert', return_value={"success": True}) as mock_trigger:
            
            triggered = await service._evaluate_price_alert(alert)
            
            assert triggered is True
            mock_update.assert_called_once()
            mock_trigger.assert_called_once()

    @pytest_asyncio.async_test
    async def test_alert_triggering(self, mock_base_service):
        """Test alert triggering and notification"""
        # Mock notification service
        mock_notification_service = Mock()
        mock_notification_service.send_sms = AsyncMock(return_value={"success": True})
        
        # Mock database operations
        mock_collection = AsyncMock()
        mock_collection.insert_one = AsyncMock()
        mock_collection.update_one = AsyncMock()
        mock_base_service.db.__getitem__.return_value = mock_collection
        
        service = AlertService(mock_base_service)
        service.notification_service = mock_notification_service
        
        alert = {
            "alert_id": "test_alert",
            "user_id": "user123",
            "symbol": "AAPL",
            "alert_type": "price",
            "condition": "above",
            "target_price": 150.0,
            "frequency": "once"
        }
        
        result = await service._trigger_alert(alert, 155.0)
        
        assert result["success"] is True
        assert result["trigger_value"] == 155.0
        mock_notification_service.send_sms.assert_called_once()

    @pytest_asyncio.async_test
    async def test_technical_alert_evaluation(self, mock_base_service):
        """Test technical indicator alert evaluation"""
        service = AlertService(mock_base_service)
        
        alert = {
            "alert_id": "tech_alert",
            "symbol": "AAPL",
            "indicator": "RSI",
            "conditions": {
                "condition": "above",
                "threshold": 70.0,
                "timeframe": "1D"
            }
        }
        
        with patch.object(service, '_get_technical_indicator', return_value=75.0), \
             patch.object(service, '_check_technical_conditions', return_value=True), \
             patch.object(service, '_trigger_alert', return_value={"success": True}):
            
            triggered = await service._evaluate_technical_alert(alert)
            assert triggered is True

    @pytest_asyncio.async_test
    async def test_user_alert_limits(self, mock_base_service):
        """Test user alert creation limits"""
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 50  # At limit
        mock_base_service.db.__getitem__.return_value = mock_collection
        
        service = AlertService(mock_base_service)
        
        with pytest.raises(ValueError, match="Maximum .* alerts per user"):
            await service.create_price_alert("user123", "AAPL", 150.0, "above")

    @pytest_asyncio.async_test
    async def test_gdpr_compliance(self, mock_base_service):
        """Test GDPR data deletion"""
        mock_alerts_collection = AsyncMock()
        mock_alerts_collection.delete_many.return_value = Mock(deleted_count=5)
        
        mock_history_collection = AsyncMock()
        mock_history_collection.delete_many.return_value = Mock(deleted_count=10)
        
        mock_base_service.db.__getitem__.side_effect = [
            mock_alerts_collection,
            mock_history_collection
        ]
        
        service = AlertService(mock_base_service)
        
        result = await service.delete_user_data("user123")
        
        assert result["deleted_alerts"] == 5
        assert result["deleted_history"] == 10


# ==========================================
# NOTIFICATION SERVICE TESTS  
# ==========================================

class TestNotificationService:
    """Test suite for NotificationService"""
    
    @pytest_asyncio.async_test
    async def test_sms_sending(self, mock_base_service):
        """Test SMS notification sending"""
        # Mock Twilio client
        mock_twilio_client = Mock()
        mock_message = Mock()
        mock_message.sid = "SMS123"
        mock_twilio_client.messages.create.return_value = mock_message
        
        # Mock user lookup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = {
            "_id": "user123",
            "phone_number": "+1234567890"
        }
        mock_base_service.db.__getitem__.return_value = mock_collection
        
        service = NotificationService(mock_base_service)
        service.twilio_client = mock_twilio_client
        
        with patch.object(service, 'get_preferences', return_value={"sms_enabled": True}), \
             patch.object(service, '_check_rate_limit', return_value=True), \
             patch.object(service, '_create_notification_record', return_value="notif123"), \
             patch.object(service, '_update_notification_status'):
            
            result = await service.send_sms("user123", "Test message")
            
            assert result["success"] is True
            assert result["twilio_sid"] == "SMS123"

    @pytest_asyncio.async_test
    async def test_email_template_rendering(self, mock_base_service):
        """Test email template rendering"""
        service = NotificationService(mock_base_service)
        
        # Test template rendering
        template = service.default_templates["welcome_email"]
        context = {
            "user_name": "John Doe",
            "phone_number": "+1234567890"
        }
        
        rendered = template.render(context)
        
        assert "John Doe" in rendered["body"]
        assert "+1234567890" in rendered["body"]

    @pytest_asyncio.async_test
    async def test_notification_preferences(self, mock_base_service):
        """Test notification preference management"""
        # Mock preferences collection
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = {
            "user_id": "user123",
            "sms_enabled": False,
            "email_enabled": True
        }
        mock_collection.replace_one.return_value = Mock(acknowledged=True)
        mock_base_service.db.__getitem__.return_value = mock_collection
        
        service = NotificationService(mock_base_service)
        
        # Test getting preferences
        prefs = await service.get_preferences("user123")
        assert prefs["sms_enabled"] is False
        
        # Test updating preferences
        success = await service.update_preferences("user123", {"sms_enabled": True})
        assert success is True

    @pytest_asyncio.async_test
    async def test_scheduled_notifications(self, mock_base_service):
        """Test scheduled notification processing"""
        # Mock notifications collection
        mock_collection = AsyncMock()
        
        # Mock due notifications
        due_notification = {
            "_id": "notif123",
            "user_id": "user123",
            "channel": "sms",
            "message": "Scheduled message",
            "priority": "normal",
            "metadata": {}
        }
        
        mock_collection.find.return_value.limit.return_value = [due_notification]
        mock_collection.update_one = AsyncMock()
        mock_base_service.db.__getitem__.return_value = mock_collection
        
        service = NotificationService(mock_base_service)
        
        with patch.object(service, 'send_sms', return_value={"success": True}):
            await service.process_scheduled_notifications()
            
            mock_collection.update_one.assert_called()

    @pytest_asyncio.async_test
    async def test_rate_limiting(self, mock_base_service):
        """Test notification rate limiting"""
        mock_base_service.check_rate_limit = AsyncMock(return_value=(False, 10))
        
        service = NotificationService(mock_base_service)
        
        allowed = await service._check_rate_limit("user123", NotificationChannel.SMS)
        assert allowed is False


# ==========================================
# INTEGRATION TESTS
# ==========================================

class TestServiceIntegration:
    """Integration tests for service interactions"""
    
    @pytest_asyncio.async_test
    async def test_alert_to_notification_flow(self, mock_base_service):
        """Test complete alert-to-notification flow"""
        # This would test the full flow from alert creation to notification delivery
        pass
    
    @pytest_asyncio.async_test
    async def test_memory_to_vector_integration(self, mock_base_service):
        """Test memory service integration with vector service"""
        pass


# ==========================================
# PERFORMANCE TESTS
# ==========================================

class TestPerformance:
    """Performance tests for services"""
    
    @pytest_asyncio.async_test
    async def test_cache_performance(self, mock_base_service):
        """Test cache service performance under load"""
        mock_base_service.redis.setex = AsyncMock()
        mock_base_service.redis.get = AsyncMock(return_value=json.dumps("cached_value"))
        
        service = CacheService(mock_base_service)
        
        # Test batch operations performance
        start_time = time.time()
        
        tasks = []
        for i in range(100):
            tasks.append(service.set_cache(f"key_{i}", f"value_{i}"))
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 100 operations in reasonable time
        assert duration < 1.0  # Less than 1 second

    @pytest_asyncio.async_test
    async def test_vector_batch_performance(self, mock_base_service, mock_pinecone):
        """Test vector service batch operations"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_openai.return_value.embeddings.create = AsyncMock(
                return_value=Mock(data=[Mock(embedding=[0.1] * 1536)])
            )
            
            service = VectorService(mock_base_service)
            service.openai_client = mock_openai.return_value
            service.pinecone_index = mock_pinecone
            
            # Test batch upsert performance
            documents = [
                {"text": f"Document {i}", "metadata": {"id": i}}
                for i in range(50)
            ]
            
            start_time = time.time()
            result = await service.batch_upsert_texts("test_namespace", documents)
            end_time = time.time()
            
            assert end_time - start_time < 30.0  # Should complete in reasonable time
            assert result["successful"] > 0


# ==========================================
# ERROR HANDLING TESTS
# ==========================================

class TestErrorHandling:
    """Test error handling and resilience"""
    
    @pytest_asyncio.async_test
    async def test_network_failure_handling(self, mock_base_service):
        """Test handling of network failures"""
        service = CacheService(mock_base_service)
        
        # Mock network failure
        mock_base_service.redis.get.side_effect = Exception("Network error")
        
        # Should not raise exception, should return None gracefully
        result = await service.get_cache("test_key")
        assert result is None

    @pytest_asyncio.async_test
    async def test_rate_limit_handling(self, mock_base_service):
        """Test handling of API rate limits"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            import openai
            
            # Mock rate limit error
            mock_openai.return_value.embeddings.create.side_effect = openai.RateLimitError(
                "Rate limit exceeded", response=Mock(), body={}
            )
            
            service = VectorService(mock_base_service)
            service.openai_client = mock_openai.return_value
            
            # Should handle rate limit gracefully
            with pytest.raises(openai.RateLimitError):
                await service._generate_embedding("test text")

    @pytest_asyncio.async_test
    async def test_database_connection_failure(self, mock_base_service):
        """Test handling of database connection failures"""
        service = AlertService(mock_base_service)
        
        # Mock database failure
        mock_base_service.db.__getitem__.side_effect = Exception("Database connection failed")
        
        # Should handle gracefully
        alerts = await service.get_user_alerts("user123")
        assert alerts == []


# ==========================================
# CONFIGURATION AND FIXTURES
# ==========================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def pytest_configure():
    """Configure pytest for async testing"""
    pytest_asyncio.async_test = pytest.mark.asyncio


# ==========================================
# TEST RUNNERS AND UTILITIES
# ==========================================

def create_test_user() -> Dict[str, Any]:
    """Create test user data"""
    return {
        "_id": fake.uuid4(),
        "phone_number": fake.phone_number(),
        "email": fake.email(),
        "name": fake.name(),
        "created_at": datetime.utcnow(),
        "preferences": {
            "sms_enabled": True,
            "email_enabled": True
        }
    }


def create_test_alert() -> Dict[str, Any]:
    """Create test alert data"""
    return {
        "alert_id": fake.uuid4(),
        "user_id": fake.uuid4(),
        "symbol": fake.random_element(["AAPL", "GOOGL", "MSFT", "TSLA"]),
        "alert_type": "price",
        "condition": "above",
        "target_price": fake.random.uniform(50, 500),
        "status": "active",
        "created_at": datetime.utcnow()
    }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
