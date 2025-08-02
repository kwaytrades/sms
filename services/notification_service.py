# services/notification_service.py - Production Notification Service
"""
Production Notification Service for Multi-Channel Delivery
Handles SMS (Twilio), Email, and Push notifications with intelligent delivery
"""

import asyncio
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass

import aiohttp
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioException
from loguru import logger
from jinja2 import Environment, BaseLoader

from config import settings


class NotificationChannel(Enum):
    """Notification delivery channels"""
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(Enum):
    """Notification delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class NotificationTemplate:
    """Notification template structure"""
    name: str
    subject_template: str
    body_template: str
    channel: NotificationChannel
    priority: NotificationPriority
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Render template with context data"""
        env = Environment(loader=BaseLoader())
        
        subject = env.from_string(self.subject_template).render(context)
        body = env.from_string(self.body_template).render(context)
        
        return {"subject": subject, "body": body}


@dataclass
class NotificationDelivery:
    """Notification delivery record"""
    notification_id: str
    user_id: str
    channel: NotificationChannel
    status: NotificationStatus
    sent_at: Optional[datetime]
    delivered_at: Optional[datetime]
    failed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]


class NotificationService:
    """
    Production Notification Service
    
    Features:
    - Multi-channel delivery (SMS, Email, Push)
    - Intelligent delivery timing
    - Template management and rendering
    - Delivery tracking and analytics
    - Retry logic with exponential backoff
    - User preference management
    - Rate limiting and cost optimization
    - GDPR compliance and opt-out handling
    """
    
    def __init__(self, base_service):
        self.base_service = base_service
        
        # Service clients
        self.twilio_client = None
        self.email_client = None
        
        # Configuration
        self.max_retries = 3
        self.retry_delays = [60, 300, 900]  # 1min, 5min, 15min
        self.rate_limits = {
            NotificationChannel.SMS: 10,      # 10 SMS per minute
            NotificationChannel.EMAIL: 20,    # 20 emails per minute
            NotificationChannel.PUSH: 100     # 100 push per minute
        }
        
        # Collections
        self.notifications_collection = "notifications"
        self.preferences_collection = "notification_preferences"
        self.templates_collection = "notification_templates"
        
        # Default templates
        self.default_templates = self._init_default_templates()
        
        logger.info("📢 NotificationService initialized")

    async def initialize(self):
        """Initialize notification service"""
        try:
            # Initialize Twilio client
            if settings.twilio_account_sid and settings.twilio_auth_token:
                self.twilio_client = TwilioClient(
                    settings.twilio_account_sid,
                    settings.twilio_auth_token
                )
                logger.info("✅ Twilio client initialized")
            else:
                logger.warning("⚠️ Twilio credentials not configured")
            
            # Initialize email client (SMTP)
            self.email_client = self._init_email_client()
            
            # Ensure collections and indexes
            await self._ensure_collections()
            
            # Load default templates
            await self._load_default_templates()
            
            logger.info("✅ NotificationService initialized successfully")
            
        except Exception as e:
            logger.exception(f"❌ NotificationService initialization failed: {e}")
            raise

    def _init_email_client(self):
        """Initialize email client"""
        try:
            if hasattr(settings, 'smtp_server') and settings.smtp_server:
                return {
                    "server": settings.smtp_server,
                    "port": getattr(settings, 'smtp_port', 587),
                    "username": getattr(settings, 'smtp_username', ''),
                    "password": getattr(settings, 'smtp_password', ''),
                    "use_tls": getattr(settings, 'smtp_use_tls', True)
                }
            else:
                logger.warning("⚠️ Email SMTP settings not configured")
                return None
        except Exception as e:
            logger.warning(f"Email client initialization failed: {e}")
            return None

    async def _ensure_collections(self):
        """Ensure MongoDB collections with proper indexes"""
        try:
            db = self.base_service.db
            
            # Notifications collection
            notifications = db[self.notifications_collection]
            await notifications.create_index([("user_id", 1), ("created_at", -1)])
            await notifications.create_index([("status", 1), ("scheduled_at", 1)])
            await notifications.create_index([("channel", 1), ("created_at", -1)])
            
            # Preferences collection
            preferences = db[self.preferences_collection]
            await preferences.create_index([("user_id", 1)], unique=True)
            
            # Templates collection
            templates = db[self.templates_collection]
            await templates.create_index([("name", 1)], unique=True)
            
            logger.info("✅ Notification collections ensured")
            
        except Exception as e:
            logger.exception(f"❌ Error ensuring collections: {e}")
            raise

    def _init_default_templates(self) -> Dict[str, NotificationTemplate]:
        """Initialize default notification templates"""
        return {
            "alert_price": NotificationTemplate(
                name="alert_price",
                subject_template="🔔 Price Alert: {{symbol}} {{condition}}",
                body_template="{{symbol}} has {{condition}} your target price of ${{target_price}}. Current price: ${{current_price}}",
                channel=NotificationChannel.SMS,
                priority=NotificationPriority.HIGH
            ),
            "alert_technical": NotificationTemplate(
                name="alert_technical",
                subject_template="📊 Technical Alert: {{symbol}}",
                body_template="{{symbol}}: {{indicator}} signal detected. {{details}}",
                channel=NotificationChannel.SMS,
                priority=NotificationPriority.NORMAL
            ),
            "portfolio_update": NotificationTemplate(
                name="portfolio_update",
                subject_template="📈 Portfolio Update",
                body_template="Your portfolio value changed by {{change_percent}}% today. Current value: ${{current_value}}",
                channel=NotificationChannel.SMS,
                priority=NotificationPriority.NORMAL
            ),
            "goal_milestone": NotificationTemplate(
                name="goal_milestone",
                subject_template="🎯 Goal Milestone Reached!",
                body_template="Congratulations! You've reached {{milestone_percent}}% of your {{goal_name}} goal. Keep it up!",
                channel=NotificationChannel.SMS,
                priority=NotificationPriority.NORMAL
            ),
            "welcome_email": NotificationTemplate(
                name="welcome_email",
                subject_template="Welcome to SMS Trading Bot!",
                body_template="""
                <h2>Welcome to SMS Trading Bot!</h2>
                <p>Hi {{user_name}},</p>
                <p>Thank you for joining SMS Trading Bot. You now have access to AI-powered trading insights via SMS.</p>
                <p>To get started, simply text us at {{phone_number}} with questions like:</p>
                <ul>
                    <li>"Analyze AAPL"</li>
                    <li>"What's the market doing?"</li>
                    <li>"Set price alert for TSLA at $200"</li>
                </ul>
                <p>Best regards,<br>SMS Trading Bot Team</p>
                """,
                channel=NotificationChannel.EMAIL,
                priority=NotificationPriority.LOW
            )
        }

    async def _load_default_templates(self):
        """Load default templates into database"""
        try:
            collection = self.base_service.db[self.templates_collection]
            
            for template_name, template in self.default_templates.items():
                # Check if template exists
                existing = await collection.find_one({"name": template_name})
                
                if not existing:
                    template_doc = {
                        "name": template.name,
                        "subject_template": template.subject_template,
                        "body_template": template.body_template,
                        "channel": template.channel.value,
                        "priority": template.priority.value,
                        "created_at": datetime.utcnow(),
                        "is_default": True
                    }
                    
                    await collection.insert_one(template_doc)
                    logger.debug(f"✅ Default template loaded: {template_name}")
            
        except Exception as e:
            logger.exception(f"❌ Error loading default templates: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for notification service"""
        health = {"status": "healthy", "components": {}}
        
        try:
            # Test Twilio
            if self.twilio_client:
                try:
                    # Simple account info check
                    account = self.twilio_client.api.accounts(
                        self.twilio_client.username
                    ).fetch()
                    health["components"]["twilio"] = {
                        "status": "healthy",
                        "account_status": account.status
                    }
                except Exception as e:
                    health["components"]["twilio"] = {"status": "unhealthy", "error": str(e)}
                    health["status"] = "degraded"
            else:
                health["components"]["twilio"] = {"status": "not_configured"}
            
            # Test email
            if self.email_client:
                health["components"]["email"] = {"status": "configured"}
            else:
                health["components"]["email"] = {"status": "not_configured"}
            
            # Get notification statistics
            stats = await self.get_notification_stats()
            health["statistics"] = stats
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health

    # ==========================================
    # SMS NOTIFICATIONS
    # ==========================================

    async def send_sms(self, user_id: str, message: str, 
                      priority: str = "normal", metadata: Dict = None) -> Dict[str, Any]:
        """Send SMS notification"""
        try:
            if not self.twilio_client:
                return {"success": False, "error": "Twilio not configured"}
            
            # Get user phone number
            user = await self._get_user_info(user_id)
            if not user or not user.get("phone_number"):
                return {"success": False, "error": "User phone number not found"}
            
            phone_number = user["phone_number"]
            
            # Check user preferences
            preferences = await self.get_preferences(user_id)
            if not preferences.get("sms_enabled", True):
                return {"success": False, "error": "SMS notifications disabled for user"}
            
            # Check rate limiting
            if not await self._check_rate_limit(user_id, NotificationChannel.SMS):
                return {"success": False, "error": "Rate limit exceeded"}
            
            # Create notification record
            notification_id = await self._create_notification_record(
                user_id, NotificationChannel.SMS, message, priority, metadata
            )
            
            # Send SMS via Twilio
            try:
                twilio_message = self.twilio_client.messages.create(
                    body=message,
                    from_=settings.twilio_phone_number,
                    to=phone_number
                )
                
                # Update notification status
                await self._update_notification_status(
                    notification_id, 
                    NotificationStatus.SENT,
                    {"twilio_sid": twilio_message.sid}
                )
                
                logger.info(f"✅ SMS sent to {user_id}: {twilio_message.sid}")
                
                return {
                    "success": True,
                    "notification_id": notification_id,
                    "twilio_sid": twilio_message.sid,
                    "status": "sent"
                }
                
            except TwilioException as e:
                await self._update_notification_status(
                    notification_id,
                    NotificationStatus.FAILED,
                    {"error": str(e)}
                )
                
                logger.error(f"❌ Twilio SMS failed for {user_id}: {e}")
                return {"success": False, "error": f"SMS delivery failed: {e}"}
            
        except Exception as e:
            logger.exception(f"❌ Error sending SMS to {user_id}: {e}")
            return {"success": False, "error": str(e)}

    # ==========================================
    # EMAIL NOTIFICATIONS
    # ==========================================

    async def send_email(self, user_id: str, subject: str, content: str,
                        template: str = None, context: Dict = None) -> Dict[str, Any]:
        """Send email notification"""
        try:
            if not self.email_client:
                return {"success": False, "error": "Email not configured"}
            
            # Get user email
            user = await self._get_user_info(user_id)
            if not user or not user.get("email"):
                return {"success": False, "error": "User email not found"}
            
            email_address = user["email"]
            
            # Check user preferences
            preferences = await self.get_preferences(user_id)
            if not preferences.get("email_enabled", True):
                return {"success": False, "error": "Email notifications disabled for user"}
            
            # Check rate limiting
            if not await self._check_rate_limit(user_id, NotificationChannel.EMAIL):
                return {"success": False, "error": "Rate limit exceeded"}
            
            # Render template if provided
            if template and context:
                template_obj = await self._get_template(template)
                if template_obj:
                    rendered = template_obj.render(context)
                    subject = rendered["subject"]
                    content = rendered["body"]
            
            # Create notification record
            notification_id = await self._create_notification_record(
                user_id, NotificationChannel.EMAIL, content, "normal",
                {"subject": subject, "template": template}
            )
            
            # Send email
            try:
                await self._send_smtp_email(email_address, subject, content)
                
                # Update notification status
                await self._update_notification_status(
                    notification_id,
                    NotificationStatus.SENT
                )
                
                logger.info(f"✅ Email sent to {user_id}")
                
                return {
                    "success": True,
                    "notification_id": notification_id,
                    "status": "sent"
                }
                
            except Exception as e:
                await self._update_notification_status(
                    notification_id,
                    NotificationStatus.FAILED,
                    {"error": str(e)}
                )
                
                logger.error(f"❌ Email failed for {user_id}: {e}")
                return {"success": False, "error": f"Email delivery failed: {e}"}
            
        except Exception as e:
            logger.exception(f"❌ Error sending email to {user_id}: {e}")
            return {"success": False, "error": str(e)}

    async def _send_smtp_email(self, to_email: str, subject: str, content: str):
        """Send email via SMTP"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_client["username"]
            msg['To'] = to_email
            
            # Add HTML content
            html_part = MIMEText(content, 'html')
            msg.attach(html_part)
            
            # Send via SMTP
            server = smtplib.SMTP(
                self.email_client["server"], 
                self.email_client["port"]
            )
            
            if self.email_client["use_tls"]:
                server.starttls()
            
            if self.email_client["username"] and self.email_client["password"]:
                server.login(
                    self.email_client["username"],
                    self.email_client["password"]
                )
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.exception(f"SMTP email error: {e}")
            raise

    # ==========================================
    # SCHEDULED NOTIFICATIONS
    # ==========================================

    async def schedule(self, user_id: str, notification_data: Dict,
                      send_at: datetime) -> str:
        """Schedule future notification"""
        try:
            # Create scheduled notification record
            notification_doc = {
                "user_id": user_id,
                "channel": notification_data.get("channel", "sms"),
                "message": notification_data.get("message", ""),
                "subject": notification_data.get("subject", ""),
                "template": notification_data.get("template"),
                "context": notification_data.get("context", {}),
                "priority": notification_data.get("priority", "normal"),
                "status": NotificationStatus.PENDING.value,
                "scheduled_at": send_at,
                "created_at": datetime.utcnow(),
                "metadata": notification_data.get("metadata", {})
            }
            
            collection = self.base_service.db[self.notifications_collection]
            result = await collection.insert_one(notification_doc)
            
            notification_id = str(result.inserted_id)
            
            logger.info(f"✅ Notification scheduled: {notification_id} for {send_at}")
            return notification_id
            
        except Exception as e:
            logger.exception(f"❌ Error scheduling notification: {e}")
            raise

    async def process_scheduled_notifications(self):
        """Process due scheduled notifications"""
        try:
            now = datetime.utcnow()
            
            collection = self.base_service.db[self.notifications_collection]
            
            # Find due notifications
            due_notifications = collection.find({
                "status": NotificationStatus.PENDING.value,
                "scheduled_at": {"$lte": now}
            }).limit(50)  # Process in batches
            
            async for notification in due_notifications:
                try:
                    user_id = notification["user_id"]
                    channel = notification["channel"]
                    
                    if channel == NotificationChannel.SMS.value:
                        result = await self.send_sms(
                            user_id,
                            notification["message"],
                            notification["priority"],
                            notification.get("metadata", {})
                        )
                    elif channel == NotificationChannel.EMAIL.value:
                        result = await self.send_email(
                            user_id,
                            notification.get("subject", ""),
                            notification["message"],
                            notification.get("template"),
                            notification.get("context", {})
                        )
                    else:
                        logger.warning(f"Unsupported notification channel: {channel}")
                        continue
                    
                    # Update status based on result
                    if result.get("success"):
                        await collection.update_one(
                            {"_id": notification["_id"]},
                            {
                                "$set": {
                                    "status": NotificationStatus.SENT.value,
                                    "sent_at": datetime.utcnow(),
                                    "delivery_result": result
                                }
                            }
                        )
                    else:
                        await collection.update_one(
                            {"_id": notification["_id"]},
                            {
                                "$set": {
                                    "status": NotificationStatus.FAILED.value,
                                    "failed_at": datetime.utcnow(),
                                    "error_message": result.get("error", "Unknown error")
                                }
                            }
                        )
                    
                except Exception as e:
                    logger.exception(f"Error processing notification {notification['_id']}: {e}")
            
        except Exception as e:
            logger.exception(f"❌ Error processing scheduled notifications: {e}")

    # ==========================================
    # USER PREFERENCES
    # ==========================================

    async def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user notification preferences"""
        try:
            collection = self.base_service.db[self.preferences_collection]
            preferences = await collection.find_one({"user_id": user_id})
            
            if preferences:
                return preferences
            else:
                # Return default preferences
                return {
                    "user_id": user_id,
                    "sms_enabled": True,
                    "email_enabled": True,
                    "push_enabled": True,
                    "quiet_hours": {
                        "enabled": False,
                        "start": "22:00",
                        "end": "08:00",
                        "timezone": "UTC"
                    },
                    "channels": {
                        "alerts": ["sms"],
                        "portfolio_updates": ["sms"],
                        "goal_milestones": ["sms", "email"],
                        "marketing": ["email"]
                    }
                }
                
        except Exception as e:
            logger.exception(f"❌ Error getting preferences for {user_id}: {e}")
            return {"user_id": user_id, "sms_enabled": True, "email_enabled": True}

    async def update_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user notification preferences"""
        try:
            collection = self.base_service.db[self.preferences_collection]
            
            update_doc = {
                "user_id": user_id,
                "updated_at": datetime.utcnow(),
                **preferences
            }
            
            result = await collection.replace_one(
                {"user_id": user_id},
                update_doc,
                upsert=True
            )
            
            logger.info(f"✅ Preferences updated for {user_id}")
            return result.acknowledged
            
        except Exception as e:
            logger.exception(f"❌ Error updating preferences for {user_id}: {e}")
            return False

    # ==========================================
    # ANALYTICS & MONITORING
    # ==========================================

    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification service statistics"""
        try:
            collection = self.base_service.db[self.notifications_collection]
            
            # Get stats for last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            
            pipeline = [
                {"$match": {"created_at": {"$gte": since}}},
                {
                    "$group": {
                        "_id": {
                            "channel": "$channel",
                            "status": "$status"
                        },
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            results = await collection.aggregate(pipeline).to_list(length=None)
            
            stats = {
                "period": "24_hours",
                "channels": {},
                "total_sent": 0,
                "total_failed": 0
            }
            
            for result in results:
                channel = result["_id"]["channel"]
                status = result["_id"]["status"]
                count = result["count"]
                
                if channel not in stats["channels"]:
                    stats["channels"][channel] = {"sent": 0, "failed": 0}
                
                if status == NotificationStatus.SENT.value:
                    stats["channels"][channel]["sent"] += count
                    stats["total_sent"] += count
                elif status == NotificationStatus.FAILED.value:
                    stats["channels"][channel]["failed"] += count
                    stats["total_failed"] += count
            
            return stats
            
        except Exception as e:
            logger.exception(f"❌ Error getting notification stats: {e}")
            return {"error": str(e)}

    async def get_user_notification_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's notification history"""
        try:
            collection = self.base_service.db[self.notifications_collection]
            
            cursor = collection.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(limit)
            
            notifications = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for notification in notifications:
                notification["_id"] = str(notification["_id"])
            
            return notifications
            
        except Exception as e:
            logger.exception(f"❌ Error getting notification history for {user_id}: {e}")
            return []

    # ==========================================
    # HELPER METHODS
    # ==========================================

    async def _get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information for notifications"""
        try:
            # This should integrate with your user service
            collection = self.base_service.db["users"]
            user = await collection.find_one({"_id": user_id})
            return user
        except Exception as e:
            logger.exception(f"Error getting user info for {user_id}: {e}")
            return None

    async def _check_rate_limit(self, user_id: str, channel: NotificationChannel) -> bool:
        """Check if user is within rate limits"""
        try:
            limit = self.rate_limits.get(channel, 10)
            window = 60  # 1 minute window
            
            key = f"notification_rate_limit:{user_id}:{channel.value}"
            
            # Use base service rate limiting
            allowed, current = await self.base_service.check_rate_limit(key, limit, window)
            
            return allowed
            
        except Exception as e:
            logger.exception(f"Error checking rate limit: {e}")
            return True  # Allow on error

    async def _create_notification_record(self, user_id: str, channel: NotificationChannel,
                                        message: str, priority: str, metadata: Dict = None) -> str:
        """Create notification record in database"""
        try:
            notification_doc = {
                "user_id": user_id,
                "channel": channel.value,
                "message": message,
                "priority": priority,
                "status": NotificationStatus.PENDING.value,
                "created_at": datetime.utcnow(),
                "metadata": metadata or {}
            }
            
            collection = self.base_service.db[self.notifications_collection]
            result = await collection.insert_one(notification_doc)
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.exception(f"Error creating notification record: {e}")
            raise

    async def _update_notification_status(self, notification_id: str,
                                        status: NotificationStatus, metadata: Dict = None):
        """Update notification status"""
        try:
            from bson import ObjectId
            
            update_doc = {
                "status": status.value,
                "updated_at": datetime.utcnow()
            }
            
            if status == NotificationStatus.SENT:
                update_doc["sent_at"] = datetime.utcnow()
            elif status == NotificationStatus.FAILED:
                update_doc["failed_at"] = datetime.utcnow()
            
            if metadata:
                update_doc.update(metadata)
            
            collection = self.base_service.db[self.notifications_collection]
            await collection.update_one(
                {"_id": ObjectId(notification_id)},
                {"$set": update_doc}
            )
            
        except Exception as e:
            logger.exception(f"Error updating notification status: {e}")

    async def _get_template(self, template_name: str) -> Optional[NotificationTemplate]:
        """Get notification template"""
        try:
            # Check default templates first
            if template_name in self.default_templates:
                return self.default_templates[template_name]
            
            # Check database templates
            collection = self.base_service.db[self.templates_collection]
            template_doc = await collection.find_one({"name": template_name})
            
            if template_doc:
                return NotificationTemplate(
                    name=template_doc["name"],
                    subject_template=template_doc["subject_template"],
                    body_template=template_doc["body_template"],
                    channel=NotificationChannel(template_doc["channel"]),
                    priority=NotificationPriority(template_doc["priority"])
                )
            
            return None
            
        except Exception as e:
            logger.exception(f"Error getting template {template_name}: {e}")
            return None
