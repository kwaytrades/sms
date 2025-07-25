# config.py - Updated version
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database (Required)
    mongodb_url: str
    redis_url: str = "redis://localhost:6379"  # Default for testing
    
    # External APIs (Optional for testing)
    openai_api_key: Optional[str] = None
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_paid_price_id: Optional[str] = "price_mock_paid"  # Mock default
    stripe_pro_price_id: Optional[str] = "price_mock_pro"   # Mock default
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    plaid_client_id: Optional[str] = None
    plaid_secret: Optional[str] = None
    plaid_env: str = "sandbox"
    eodhd_api_key: Optional[str] = None
    ta_service_url: Optional[str] = "https://mock-ta-service.com"
    
    # App settings
    environment: str = "development"
    log_level: str = "INFO"
    testing_mode: bool = True  # Enable testing mode by default
    
    class Config:
        env_file = ".env"

settings = Settings()
