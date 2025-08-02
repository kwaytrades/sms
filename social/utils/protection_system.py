# social/utils/protection_system.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import re
import hashlib
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class SpamLevel(Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious" 
    LIKELY_SPAM = "likely_spam"
    DEFINITE_SPAM = "definite_spam"

class ActionType(Enum):
    ALLOW = "allow"
    RATE_LIMIT = "rate_limit"
    REQUIRE_REVIEW = "require_review"
    BLOCK = "block"
    SHADOW_BAN = "shadow_ban"

@dataclass
class SpamAnalysis:
    """Results of spam detection analysis"""
    spam_level: SpamLevel
    confidence_score: float  # 0.0 - 1.0
    triggers: List[str]  # What triggered the spam detection
    action_recommended: ActionType
    explanation: str

@dataclass
class RateLimit:
    """Rate limiting configuration"""
    max_requests: int
    time_window_minutes: int
    description: str

class SocialProtectionSystem:
    """Comprehensive protection against spam, abuse, and rate limiting"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.request_history = defaultdict(deque)  # Fallback if no Redis
        
        # Rate limiting rules
        self.rate_limits = {
            "comments_per_hour": RateLimit(10, 60, "Comments per hour per user"),
            "dms_per_hour": RateLimit(5, 60, "DMs per hour per user"),
            "mentions_per_hour": RateLimit(15, 60, "Mentions per hour"),
            "responses_per_minute": RateLimit(3, 1, "Our responses per minute globally"),
            "competitive_engagement_per_day": RateLimit(20, 1440, "Competitive engagements per day")
        }
        
        # Spam detection patterns
        self.spam_patterns = {
            "excessive_caps": re.compile(r'[A-Z]{5,}'),
            "excessive_emojis": re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]{5,}'),
            "repeated_chars": re.compile(r'(.)\1{4,}'),
            "urls": re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            "phone_numbers": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "email_addresses": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "crypto_addresses": re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|0x[a-fA-F0-9]{40}\b'),
            "excessive_punctuation": re.compile(r'[!?.]{3,}')
        }
        
        # Spam keyword lists
        self.spam_keywords = {
            "financial_scams": [
                "guaranteed profit", "risk free", "get rich quick", "make money fast",
                "investment opportunity", "double your money", "insider trading",
                "pump and dump", "easy money", "financial freedom"
            ],
            "promotional": [
                "check out my", "follow me", "subscribe to", "join my channel",
                "click here", "dm me", "message me", "contact me for",
                "buy now", "limited time", "act fast", "exclusive offer"
            ],
            "crypto_scams": [
                "airdrop", "free crypto", "bitcoin giveaway", "send me crypto",
                "wallet address", "private key", "seed phrase", "recovery phrase"
            ],
            "generic_spam": [
                "congratulations you won", "claim your prize", "verify your account",
                "suspended account", "click to verify", "account will be closed"
            ]
        }
        
        # User behavior tracking
        self.user_behavior_cache = {}  # User ID -> behavior metrics
        
    def check_rate_limit(self, user_id: str, action_type: str, platform: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Check if user has exceeded rate limits"""
        key = f"{platform}_{user_id}_{action_type}" if platform else f"{user_id}_{action_type}"
        
        if action_type not in self.rate_limits:
            return True, {"allowed": True, "reason": "no_rate_limit_defined"}
        
        rate_limit = self.rate_limits[action_type]
        current_time = datetime.utcnow()
        
        # Use Redis if available, otherwise fallback to memory
        if self.redis:
            return self._check_rate_limit_redis(key, rate_limit, current_time)
        else:
            return self._check_rate_limit_memory(key, rate_limit, current_time)
    
    def _check_rate_limit_redis(self, key: str, rate_limit: RateLimit, current_time: datetime) -> Tuple[bool, Dict[str, Any]]:
        """Redis-based rate limiting"""
        window_start = current_time - timedelta(minutes=rate_limit.time_window_minutes)
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start.timestamp())
        
        # Count current requests
        current_count = self.redis.zcard(key)
        
        if current_count >= rate_limit.max_requests:
            ttl = self.redis.ttl(key)
            return False, {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "current_count": current_count,
                "max_requests": rate_limit.max_requests,
                "reset_in_seconds": ttl
            }
        
        # Add current request
        self.redis.zadd(key, {str(current_time.timestamp()): current_time.timestamp()})
        self.redis.expire(key, rate_limit.time_window_minutes * 60)
        
        return True, {
            "allowed": True,
            "current_count": current_count + 1,
            "max_requests": rate_limit.max_requests
        }
    
    def _check_rate_limit_memory(self, key: str, rate_limit: RateLimit, current_time: datetime) -> Tuple[bool, Dict[str, Any]]:
        """Memory-based rate limiting fallback"""
        window_start = current_time - timedelta(minutes=rate_limit.time_window_minutes)
        
        # Clean old entries
        while self.request_history[key] and self.request_history[key][0] < window_start:
            self.request_history[key].popleft()
        
        current_count = len(self.request_history[key])
        
        if current_count >= rate_limit.max_requests:
            return False, {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "current_count": current_count,
                "max_requests": rate_limit.max_requests
            }
        
        # Add current request
        self.request_history[key].append(current_time)
        
        return True, {
            "allowed": True,
            "current_count": current_count + 1,
            "max_requests": rate_limit.max_requests
        }
    
    def analyze_spam_content(self, content: str, user_context: Dict[str, Any] = None) -> SpamAnalysis:
        """Comprehensive spam analysis of message content"""
        triggers = []
        spam_score = 0.0
        
        # Content-based analysis
        content_lower = content.lower()
        
        # Pattern matching
        for pattern_name, pattern in self.spam_patterns.items():
            if pattern.search(content):
                triggers.append(f"pattern_{pattern_name}")
                spam_score += self._get_pattern_weight(pattern_name)
        
        # Keyword matching
        for category, keywords in self.spam_keywords.items():
            matched_keywords = [kw for kw in keywords if kw in content_lower]
            if matched_keywords:
                triggers.append(f"keywords_{category}")
                spam_score += len(matched_keywords) * 0.1
        
        # Content characteristics
        if len(content) < 5:
            triggers.append("too_short")
            spam_score += 0.2
        
        if len(content) > 1000:
            triggers.append("too_long") 
            spam_score += 0.1
        
        # Repetitive content check
        if self._is_repetitive_content(content):
            triggers.append("repetitive_content")
            spam_score += 0.3
        
        # User behavior analysis
        if user_context:
            behavior_score = self._analyze_user_behavior(user_context)
            spam_score += behavior_score
            if behavior_score > 0.3:
                triggers.append("suspicious_user_behavior")
        
        # Determine spam level and action
        spam_level, action = self._calculate_spam_level_and_action(spam_score)
        
        explanation = self._generate_explanation(spam_score, triggers)
        
        return SpamAnalysis(
            spam_level=spam_level,
            confidence_score=min(spam_score, 1.0),
            triggers=triggers,
            action_recommended=action,
            explanation=explanation
        )
    
    def _get_pattern_weight(self, pattern_name: str) -> float:
        """Get spam weight for different patterns"""
        weights = {
            "excessive_caps": 0.2,
            "excessive_emojis": 0.15,
            "repeated_chars": 0.1,
            "urls": 0.4,
            "phone_numbers": 0.5,
            "email_addresses": 0.3,
            "crypto_addresses": 0.6,
            "excessive_punctuation": 0.1
        }
        return weights.get(pattern_name, 0.1)
    
    def _is_repetitive_content(self, content: str) -> bool:
        """Check if content is repetitive"""
        words = content.lower().split()
        if len(words) < 3:
            return False
        
        # Check for repeated words
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, it's repetitive
        max_frequency = max(word_count.values())
        return max_frequency / len(words) > 0.3
    
    def _analyze_user_behavior(self, user_context: Dict[str, Any]) -> float:
        """Analyze user behavior patterns for spam indicators"""
        behavior_score = 0.0
        
        # New account (less than 24 hours old)
        if user_context.get("account_age_hours", 24) < 24:
            behavior_score += 0.2
        
        # Low follower count
        follower_count = user_context.get("follower_count", 0)
        if follower_count < 10:
            behavior_score += 0.2
        elif follower_count < 100:
            behavior_score += 0.1
        
        # High following to follower ratio (sign of spam account)
        following_count = user_context.get("following_count", 0)
        if follower_count > 0 and following_count / follower_count > 10:
            behavior_score += 0.3
        
        # No profile picture
        if not user_context.get("has_profile_picture", True):
            behavior_score += 0.1
        
        # Generic username patterns
        username = user_context.get("username", "")
        if re.match(r'^[a-zA-Z]+\d{4,}$', username):  # Like "user12345"
            behavior_score += 0.15
        
        # Recent burst of activity
        recent_message_count = user_context.get("messages_last_hour", 0)
        if recent_message_count > 10:
            behavior_score += 0.4
        
        return behavior_score
    
    def _calculate_spam_level_and_action(self, spam_score: float) -> Tuple[SpamLevel, ActionType]:
        """Calculate spam level and recommended action"""
        if spam_score >= 0.8:
            return SpamLevel.DEFINITE_SPAM, ActionType.BLOCK
        elif spam_score >= 0.6:
            return SpamLevel.LIKELY_SPAM, ActionType.SHADOW_BAN
        elif spam_score >= 0.4:
            return SpamLevel.SUSPICIOUS, ActionType.REQUIRE_REVIEW
        elif spam_score >= 0.2:
            return SpamLevel.SUSPICIOUS, ActionType.RATE_LIMIT
        else:
            return SpamLevel.CLEAN, ActionType.ALLOW
    
    def _generate_explanation(self, spam_score: float, triggers: List[str]) -> str:
        """Generate human-readable explanation"""
        if spam_score < 0.2:
            return "Content appears clean"
        
        explanation = f"Spam score: {spam_score:.2f}. "
        
        if triggers:
            explanation += f"Triggered by: {', '.join(triggers[:3])}"
            if len(triggers) > 3:
                explanation += f" and {len(triggers) - 3} more"
        
        return explanation
    
    def should_respond_to_message(self, user_id: str, content: str, platform: str, 
                                user_context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive check if we should respond to a message"""
        result = {
            "should_respond": True,
            "reason": "clean",
            "rate_limit_status": {},
            "spam_analysis": None,
            "action_taken": ActionType.ALLOW
        }
        
        # Check rate limits first
        rate_allowed, rate_info = self.check_rate_limit(user_id, "comments_per_hour", platform)
        result["rate_limit_status"] = rate_info
        
        if not rate_allowed:
            result.update({
                "should_respond": False,
                "reason": "rate_limit_exceeded",
                "action_taken": ActionType.RATE_LIMIT
            })
            return False, result
        
        # Analyze content for spam
        spam_analysis = self.analyze_spam_content(content, user_context)
        result["spam_analysis"] = spam_analysis
        
        # Determine final action
        if spam_analysis.action_recommended in [ActionType.BLOCK, ActionType.SHADOW_BAN]:
            result.update({
                "should_respond": False,
                "reason": f"spam_detected_{spam_analysis.spam_level.value}",
                "action_taken": spam_analysis.action_recommended
            })
            return False, result
        elif spam_analysis.action_recommended == ActionType.REQUIRE_REVIEW:
            result.update({
                "should_respond": False,
                "reason": "requires_manual_review",
                "action_taken": ActionType.REQUIRE_REVIEW
            })
            return False, result
        
        # Check our own response rate limits
        our_rate_allowed, our_rate_info = self.check_rate_limit("system", "responses_per_minute")
        if not our_rate_allowed:
            result.update({
                "should_respond": False,
                "reason": "system_rate_limit",
                "action_taken": ActionType.RATE_LIMIT
            })
            return False, result
        
        return True, result
    
    def log_interaction(self, user_id: str, platform: str, content: str, 
                       response_decision: Dict[str, Any]):
        """Log interaction for monitoring and improvement"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "platform": platform,
            "content_length": len(content),
            "decision": response_decision["reason"],
            "spam_score": response_decision.get("spam_analysis", {}).get("confidence_score", 0),
            "action_taken": response_decision.get("action_taken", "unknown").value if hasattr(response_decision.get("action_taken"), 'value') else str(response_decision.get("action_taken"))
        }
        
        # Log to your monitoring system
        logger.info(f"Social interaction logged: {log_data}")
    
    def get_protection_statistics(self) -> Dict[str, Any]:
        """Get protection system statistics"""
        # This would integrate with your monitoring/analytics system
        return {
            "rate_limits_configured": len(self.rate_limits),
            "spam_patterns_active": len(self.spam_patterns),
            "spam_keywords_total": sum(len(keywords) for keywords in self.spam_keywords.values()),
            "protection_level": "comprehensive"
        }
    
    def update_spam_patterns(self, new_patterns: Dict[str, Any]):
        """Update spam detection patterns (for continuous improvement)"""
        for pattern_name, pattern_data in new_patterns.items():
            if "regex" in pattern_data:
                self.spam_patterns[pattern_name] = re.compile(pattern_data["regex"])
            if "keywords" in pattern_data and "category" in pattern_data:
                category = pattern_data["category"]
                if category not in self.spam_keywords:
                    self.spam_keywords[category] = []
                self.spam_keywords[category].extend(pattern_data["keywords"])
        
        logger.info(f"Updated spam patterns: {list(new_patterns.keys())}")

# Utility functions for integration
def create_protection_system(redis_client=None) -> SocialProtectionSystem:
    """Factory function to create protection system"""
    return SocialProtectionSystem(redis_client)

def is_message_safe_to_process(protection_system: SocialProtectionSystem, 
                             user_id: str, content: str, platform: str,
                             user_context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
    """High-level function to check if message is safe to process"""
    should_respond, analysis = protection_system.should_respond_to_message(
        user_id, content, platform, user_context
    )
    
    # Log the interaction
    protection_system.log_interaction(user_id, platform, content, analysis)
    
    return should_respond, analysis
