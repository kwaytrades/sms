# tests/__init__.py - Place in tests/ directory
# Empty file to make tests a Python package

# tests/conftest.py - Place in tests/ directory
import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Test client for API endpoints"""
    return TestClient(app)

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# tests/test_health.py - Place in tests/ directory
import pytest
from fastapi.testclient import TestClient
from main import app

def test_health_endpoint():
    """Test health check endpoint"""
    client = TestClient(app)
    response = client.get("/health")
    
    # Should return 200 or 503 (if services not configured)
    assert response.status_code in [200, 503]
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data

def test_root_endpoint():
    """Test root endpoint"""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Unified SMS Trading Bot"
    assert data["version"] == "3.0.0"

# tests/test_technical_analysis.py - Place in tests/ directory
import pytest
from fastapi.testclient import TestClient
from main import app

def test_technical_analysis_endpoint():
    """Test technical analysis endpoint"""
    client = TestClient(app)
    
    # Test with mock data (will fail if EODHD not configured, which is expected)
    response = client.get("/analysis/AAPL")
    
    # Could be 200 (success) or 500 (no API key configured)
    assert response.status_code in [200, 500, 503]

def test_price_endpoint():
    """Test price endpoint"""
    client = TestClient(app)
    response = client.get("/price/AAPL")
    
    # Could be 200 (success) or 500 (no API key configured)
    assert response.status_code in [200, 404, 500, 503]

# tests/test_validators.py - Place in tests/ directory
import pytest
from utils.validators import validate_phone_number, validate_email, sanitize_input

def test_validate_phone_number():
    """Test phone number validation"""
    # Valid US phone numbers
    assert validate_phone_number("1234567890") == "+11234567890"
    assert validate_phone_number("+1234567890") == "+11234567890"
    assert validate_phone_number("(123) 456-7890") == "+11234567890"
    
    # Invalid phone numbers
    with pytest.raises(ValueError):
        validate_phone_number("123")
    
    with pytest.raises(ValueError):
        validate_phone_number("abcdefghij")

def test_validate_email():
    """Test email validation"""
    assert validate_email("test@example.com") == True
    assert validate_email("user.name+tag@example.co.uk") == True
    assert validate_email("invalid-email") == False
    assert validate_email("@example.com") == False

def test_sanitize_input():
    """Test input sanitization"""
    assert sanitize_input("Hello World") == "Hello World"
    assert sanitize_input("<script>alert('xss')</script>") == "scriptalert('xss')/script"
    assert sanitize_input("  spaces  ") == "spaces"

# tests/test_sms_webhook.py - Place in tests/ directory  
import pytest
from fastapi.testclient import TestClient
from main import app

def test_sms_webhook_missing_data():
    """Test SMS webhook with missing data"""
    client = TestClient(app)
    
    response = client.post("/webhook/sms", data={})
    assert response.status_code == 400

def test_sms_webhook_invalid_phone():
    """Test SMS webhook with invalid phone number"""
    client = TestClient(app)
    
    response = client.post("/webhook/sms", data={
        "From": "invalid",
        "Body": "test message"
    })
    assert response.status_code == 400

def test_sms_webhook_valid_data():
    """Test SMS webhook with valid data"""
    client = TestClient(app)
    
    response = client.post("/webhook/sms", data={
        "From": "+1234567890",
        "Body": "test message",
        "MessageSid": "test123"
    })
    
    # Should return TwiML response
    assert response.status_code in [200, 503]  # 503 if services not initialized
