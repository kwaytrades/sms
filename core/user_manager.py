# core/user_manager.py - Unified User Management
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from loguru import logger

from config import settings, PLAN_LIMITS
from services.database import DatabaseService

class UserManager:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service
    
    async def get_user_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Get user by phone number"""
        try:
            user = await self.db.db.users.find_one({"phone_number": phone_number})
            if user:
                user["_id"] = str(user["_id"])
                logger.info(f"✅ Found user: {phone_number}")
                return user
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding user {phone_number}: {e}")
            return None
    
    async def get_or_create_user(self, phone_number: str) -> Dict[str, Any]:
        """Get existing user or create new one"""
        user = await self.get_user_by_phone(phone_number)
        
        if user:
            # Update existing user schema if needed
            await self._update_user_schema(user)
            return await self.get_user_by_phone(phone_number)
        else:
            # Create new user
            return await self._create_new_user(phone_number)
    
    async def _create_new_user(self, phone_number: str) -> Dict[str, Any]:
        """Create a new user with complete schema"""
        now = datetime.now(timezone.utc)
        
        new_user = {
            "phone_number": phone_number,
            "email": None,
            "first_name": None,
            "timezone": "US/Eastern",
            
            # Subscription
            "plan_type": "free",
            "subscription_status": "active",
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            "trial_ends_at": None,
            
            # Trading profile
            "risk_tolerance": "medium",
            "trading_experience": "intermediate",
            "preferred_sectors": [],
            "watchlist": [],
            "trading_style": "swing",
            
            # Communication preferences
            "communication_style": {
                "formality": "casual",
                "message_length": "medium",
                "emoji_usage": True,
                "technical_depth": "medium"
            },
            
            # Behavioral patterns (learned over time)
            "response_patterns": {
                "preferred_response_time": "immediate",
                "engagement_triggers": [],
                "successful_trade_patterns": [],
                "loss_triggers": []
            },
            
            # Trading behavior tracking
            "trading_behavior": {
                "successful_sectors": [],
                "preferred_position_sizes": [],
                "typical_hold_time": None,
                "win_rate": None,
                "average_gain": None,
                "average_loss": None
            },
            
            # Speech and interaction patterns
            "speech_patterns": {
                "vocabulary_level": "intermediate",
                "question_types": [],
                "common_tickers": [],
                "interaction_frequency": "moderate"
            },
            
            # Usage tracking
            "total_messages_sent": 0,
            "total_messages
