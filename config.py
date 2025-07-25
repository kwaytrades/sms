# ===== config.py =====
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    mongodb_url: str
    redis_url: str
    
    # External APIs
    openai_api_key: str
    stripe_secret_key: str
    stripe_webhook_secret: str
    stripe_paid_price_id: str
    stripe_pro_price_id: str
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    plaid_client_id: str
    plaid_secret: str
    plaid_env: str = "sandbox"
    eodhd_api_key: str
    ta_service_url: str
    
    # App settings
    environment: str = "development"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
