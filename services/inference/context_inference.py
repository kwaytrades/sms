services/inference/context_inference.py
"""
Context Inference Service - Enhance conversation context intelligently
Smart context building and conversation intelligence
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from loguru import logger

from services.db.base_db_service import BaseDBService


class ContextInferenceService:
    """
    Context inference service for intelligent conversation enhancement
    
    Features:
    - Topic evolution tracking
    - Intent prediction from conversation patterns
    - Relationship stage progression
    - Conversation frequency analysis
    - Context quality scoring
    - Intelligent context compression
    """
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self._topic_memory = {}
        self._intent_patterns = self._build_intent_patterns()

    async def initialize(self):
        """Initialize context inference service"""
        try:
            # Create context analytics collection
            await self.base.db.context_analytics.create_index("phone_number")
            await self.base.db.context_analytics.create_index("created_at")
            
            logger.info("✅ Context inference service initialized")
        except Exception as e:
            logger.exception(f"❌ Context inference service initialization failed: {e}")

    async def enhance_context(self, context: Dict, phone_number: str) -> Dict:
        """
        Main context enhancement method
        """
        try:
            if not context or not phone_number:
                return context
            
            # Analyze conversation patterns
            pattern_analysis = await self._analyze_conversation_patterns(phone_number)
            
            # Enhance relationship stage
            enhanced_stage = await self._enhance_relationship_stage(context, pattern_analysis)
            
            # Add conversation intelligence
            conversation_intel = await self._build_conversation_intelligence(phone_number, pattern_analysis)
            
            # Predict next topics
            topic_predictions = await self._predict_next_topics(phone_number, context)
            
            # Enhanced context
            enhanced_context = context.copy()
            enhanced_context.update({
                "relationship_stage": enhanced_stage,
                "conversation_intelligence": conversation_intel,
                "topic_predictions": topic_predictions,
                "pattern_analysis": pattern_analysis,
                "context_quality_score": self._calculate_context_quality(enhanced_context),
                "enhancement_timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Update context analytics
            await self._update_context_analytics(phone_number, enhanced_context)
            
            return enhanced_context
            
        except Exception as e:
            logger.exception(f"❌ Error enhancing context: {e}")
            return context

    async def update_from_message(self, phone_number: str, user_message: str, 
                                bot_response: str, intent_data: Dict, symbols: List[str]):
        """
        Update context intelligence from new message
        """
        try:
            # Extract insights from the new message
            message_insights = await self._extract_message_insights(
                user_message, bot_response, intent_data, symbols
            )
            
            # Update topic evolution
            await self._update_topic_evolution(phone_number, message_insights)
            
            # Update conversation patterns
            await self._update_conversation_patterns(phone_number, message_insights)
            
            # Check for context triggers
            await self._check_context_triggers(phone_number, message_insights)
            
        except Exception as e:
            logger.exception(f"❌ Error updating context from message: {e}")

    def _build_intent_patterns(self) -> Dict[str, Dict]:
        """Build intent prediction patterns"""
        return {
            "information_seeking": {
                "keywords": ["what", "how", "when", "where", "why", "explain", "tell me"],
                "patterns": ["what is", "how do", "can you explain"],
                "confidence_boost": 0.3
            },
            "analysis_request": {
                "keywords": ["analyze", "analysis", "look at", "thoughts on", "opinion"],
                "patterns": ["what do you think", "analysis of", "your thoughts"],
                "confidence_boost": 0.4
            },
            "decision_support": {
                "keywords": ["should i", "recommend", "suggest", "advice", "help decide"],
                "patterns": ["should i buy", "what would you", "help me decide"],
                "confidence_boost": 0.5
            },
            "portfolio_management": {
                "keywords": ["portfolio", "holdings", "positions", "allocate", "rebalance"],
                "patterns": ["my portfolio", "portfolio allocation", "position size"],
                "confidence_boost": 0.4
            },
            "market_discussion": {
                "keywords": ["market", "economy", "fed", "earnings", "news", "trend"],
                "patterns": ["market outlook", "economic conditions", "market trend"],
                "confidence_boost": 0.3
            }
        }

    async def _analyze_conversation_patterns(self, phone_number: str) -> Dict[str, Any]:
        """Analyze conversation patterns for intelligence"""
        try:
            # Get recent conversation history
            conversations = await self.base.db.enhanced_conversations.find(
                {"phone_number": phone_number}
            ).sort("timestamp", -1).limit(50).to_list(length=50)
            
            if not conversations:
                return {}
            
            # Analyze patterns
            analysis = {
                "total_conversations": len(conversations),
                "time_span_days": 0,
                "avg_messages_per_day": 0,
                "common_intents": [],
                "topic_diversity": 0,
                "engagement_level": "medium",
                "conversation_frequency": "occasional",
                "peak_activity_hours": [],
                "response_patterns": {}
            }
            
            # Time analysis
            if len(conversations) > 1:
                latest = conversations[0]["timestamp"]
                earliest = conversations[-1]["timestamp"]
                if isinstance(latest, str):
                    latest = datetime.fromisoformat(latest)
                if isinstance(earliest, str):
                    earliest = datetime.fromisoformat(earliest)
                
                time_span = (latest - earliest).days
                analysis["time_span_days"] = max(time_span, 1)
                analysis["avg_messages_per_day"] = len(conversations) / analysis["time_span_days"]
            
            # Intent analysis
            intent_counts = defaultdict(int)
            for conv in conversations:
                intent = conv.get("intent_data", {}).get("intent", "general")
                intent_counts[intent] += 1
            
            # Sort by frequency
            sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
            analysis["common_intents"] = [{"intent": intent, "count": count} for intent, count in sorted_intents[:5]]
            
            # Topic diversity
            all_symbols = set()
            for conv in conversations:
                symbols = conv.get("symbols_mentioned", [])
                all_symbols.update(symbols)
            analysis["topic_diversity"] = len(all_symbols)
            
            # Engagement level
            avg_user_message_length = sum(
                len(conv.get("user_message", "").split()) 
                for conv in conversations
            ) / len(conversations)
            
            if avg_user_message_length > 15:
                analysis["engagement_level"] = "high"
            elif avg_user_message_length < 5:
                analysis["engagement_level"] = "low"
            else:
                analysis["engagement_level"] = "medium"
            
            # Conversation frequency
            if analysis["avg_messages_per_day"] > 3:
                analysis["conversation_frequency"] = "frequent"
            elif analysis["avg_messages_per_day"] > 1:
                analysis["conversation_frequency"] = "regular"
            else:
                analysis["conversation_frequency"] = "occasional"
            
            return analysis
            
        except Exception as e:
            logger.exception(f"❌ Error analyzing conversation patterns: {e}")
            return {}

    async def _enhance_relationship_stage(self, context: Dict, pattern_analysis: Dict) -> str:
        """Enhance relationship stage based on patterns"""
        try:
            current_stage = context.get("conversation_context", {}).get("relationship_stage", "new")
            total_conversations = pattern_analysis.get("total_conversations", 0)
            engagement_level = pattern_analysis.get("engagement_level", "medium")
            conversation_frequency = pattern_analysis.get("conversation_frequency", "occasional")
            
            # Stage progression logic
            if total_conversations < 3:
                stage = "new"
            elif total_conversations < 10:
                if engagement_level == "high" and conversation_frequency in ["frequent", "regular"]:
                    stage = "engaged"
                else:
                    stage = "exploring"
            elif total_conversations < 25:
                if engagement_level == "high":
                    stage = "trusted"
                else:
                    stage = "regular"
            else:
                if engagement_level == "high" and conversation_frequency == "frequent":
                    stage = "highly_engaged"
                else:
                    stage = "established"
            
            return stage
            
        except Exception as e:
            logger.exception(f"❌ Error enhancing relationship stage: {e}")
            return "new"

    async def _build_conversation_intelligence(self, phone_number: str, pattern_analysis: Dict) -> Dict[str, Any]:
        """Build conversation intelligence insights"""
        try:
            intelligence = {
                "user_persona": await self._infer_user_persona(pattern_analysis),
                "conversation_style": await self._analyze_conversation_style(phone_number),
                "topic_preferences": await self._analyze_topic_preferences(phone_number),
                "response_optimization": await self._suggest_response_optimization(pattern_analysis),
                "engagement_triggers": await self._identify_engagement_triggers(phone_number)
            }
            
            return intelligence
            
        except Exception as e:
            logger.exception(f"❌ Error building conversation intelligence: {e}")
            return {}

    async def _predict_next_topics(self, phone_number: str, context: Dict) -> List[Dict]:
        """Predict likely next topics based on conversation flow"""
        try:
            predictions = []
            
            # Get recent topics
            recent_symbols = context.get("conversation_context", {}).get("recent_symbols", [])
            recent_topics = context.get("conversation_context", {}).get("recent_topics", [])
            
            # Predict based on recent symbols
            for symbol in recent_symbols[-3:]:  # Last 3 symbols
                predictions.append({
                    "type": "symbol_followup",
                    "topic": f"{symbol} update",
                    "confidence": 0.7,
                    "reason": f"Recent discussion about {symbol}"
                })
            
            # Predict based on conversation patterns
            if any(topic.get("topic") == "analysis" for topic in recent_topics):
                predictions.append({
                    "type": "analysis_request",
                    "topic": "market analysis",
                    "confidence": 0.6,
                    "reason": "Pattern of requesting analysis"
                })
            
            # Predict based on time patterns
            now = datetime.now(timezone.utc)
            if now.hour < 10:  # Morning
                predictions.append({
                    "type": "market_open",
                    "topic": "pre-market discussion",
                    "confidence": 0.5,
                    "reason": "Pre-market hours"
                })
            elif now.hour > 16:  # After market close
                predictions.append({
                    "type": "market_close",
                    "topic": "post-market analysis",
                    "confidence": 0.5,
                    "reason": "Post-market hours"
                })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return predictions[:5]  # Top 5 predictions
            
        except Exception as e:
            logger.exception(f"❌ Error predicting next topics: {e}")
            return []

    def _calculate_context_quality(self, context: Dict) -> float:
        """Calculate context quality score"""
        try:
            score = 0.0
            
            # Profile completeness (0.3 weight)
            profile = context.get("user_profile", {})
            profile_fields = ["phone_number", "risk_tolerance", "trading_style", "plan_type"]
            profile_completeness = sum(1 for field in profile_fields if profile.get(field)) / len(profile_fields)
            score += profile_completeness * 0.3
            
            # Conversation history (0.3 weight)
            recent_messages = context.get("recent_messages", [])
            if len(recent_messages) >= 5:
                score += 0.3
            elif len(recent_messages) >= 2:
                score += 0.15
            
            # Context richness (0.2 weight)
            conv_context = context.get("conversation_context", {})
            if conv_context.get("recent_symbols"):
                score += 0.1
            if conv_context.get("recent_topics"):
                score += 0.1
            
            # Session data (0.2 weight)
            today_session = context.get("today_session", {})
            if today_session.get("message_count", 0) > 0:
                score += 0.1
            if today_session.get("symbols_mentioned"):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.exception(f"❌ Error calculating context quality: {e}")
            return 0.0

    async def _extract_message_insights(self, user_message: str, bot_response: str, 
                                      intent_data: Dict, symbols: List[str]) -> Dict[str, Any]:
        """Extract insights from a single message exchange"""
        try:
            insights = {
                "user_message_length": len(user_message.split()),
                "bot_response_length": len(bot_response.split()),
                "intent": intent_data.get("intent", "general"),
                "symbols_mentioned": symbols,
                "question_count": user_message.count("?"),
                "urgency_indicators": self._detect_urgency(user_message),
                "sentiment": self._analyze_simple_sentiment(user_message),
                "complexity_level": self._assess_complexity(user_message),
                "timestamp": datetime.now(timezone.utc)
            }
            
            return insights
            
        except Exception as e:
            logger.exception(f"❌ Error extracting message insights: {e}")
            return {}

    def _detect_urgency(self, message: str) -> List[str]:
        """Detect urgency indicators in message"""
        urgency_keywords = ["urgent", "asap", "quickly", "fast", "hurry", "immediate", "now"]
        found_indicators = [keyword for keyword in urgency_keywords if keyword.lower() in message.lower()]
        return found_indicators

    def _analyze_simple_sentiment(self, message: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ["good", "great", "excellent", "awesome", "love", "happy", "satisfied"]
        negative_words = ["bad", "terrible", "awful", "hate", "frustrated", "angry", "disappointed"]
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _assess_complexity(self, message: str) -> str:
        """Assess message complexity level"""
        advanced_terms = ["volatility", "correlation", "beta", "options", "derivatives", "technical analysis"]
        
        if any(term in message.lower() for term in advanced_terms):
            return "advanced"
        elif len(message.split()) > 20:
            return "detailed"
        elif len(message.split()) < 5:
            return "simple"
        else:
            return "moderate"

    async def _update_topic_evolution(self, phone_number: str, insights: Dict):
        """Update topic evolution tracking"""
        try:
            if phone_number not in self._topic_memory:
                self._topic_memory[phone_number] = {
                    "topic_sequence": [],
                    "symbol_frequency": defaultdict(int),
                    "intent_patterns": defaultdict(list)
                }
            
            memory = self._topic_memory[phone_number]
            
            # Update topic sequence
            if insights.get("symbols_mentioned"):
                memory["topic_sequence"].extend(insights["symbols_mentioned"])
                memory["topic_sequence"] = memory["topic_sequence"][-20:]  # Keep last 20
                
                # Update symbol frequency
                for symbol in insights["symbols_mentioned"]:
                    memory["symbol_frequency"][symbol] += 1
            
            # Update intent patterns
            intent = insights.get("intent", "general")
            memory["intent_patterns"][intent].append(insights.get("timestamp"))
            
        except Exception as e:
            logger.exception(f"❌ Error updating topic evolution: {e}")

    async def _update_conversation_patterns(self, phone_number: str, insights: Dict):
        """Update conversation pattern analysis"""
        try:
            # Update in database for persistence
            pattern_doc = {
                "phone_number": phone_number,
                "message_insights": insights,
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.context_analytics.insert_one(pattern_doc)
            
        except Exception as e:
            logger.exception(f"❌ Error updating conversation patterns: {e}")

    async def _check_context_triggers(self, phone_number: str, insights: Dict):
        """Check for context-based triggers"""
        try:
            # Check for engagement opportunity
            if insights.get("urgency_indicators"):
                logger.info(f"🚨 Urgency detected for {phone_number}: {insights['urgency_indicators']}")
            
            # Check for complexity mismatch
            if insights.get("complexity_level") == "advanced":
                logger.info(f"📈 Advanced user detected: {phone_number}")
            
            # Check for negative sentiment
            if insights.get("sentiment") == "negative":
                logger.info(f"😟 Negative sentiment detected: {phone_number}")
            
        except Exception as e:
            logger.exception(f"❌ Error checking context triggers: {e}")

    async def _update_context_analytics(self, phone_number: str, enhanced_context: Dict):
        """Update context analytics for improvement tracking"""
        try:
            analytics_doc = {
                "phone_number": phone_number,
                "context_quality_score": enhanced_context.get("context_quality_score", 0),
                "relationship_stage": enhanced_context.get("relationship_stage", "new"),
                "enhancement_applied": True,
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.context_analytics.insert_one(analytics_doc)
            
        except Exception as e:
            logger.exception(f"❌ Error updating context analytics: {e}")

    # Additional helper methods for conversation intelligence
    async def _infer_user_persona(self, pattern_analysis: Dict) -> Dict[str, Any]:
        """Infer user persona from patterns"""
        engagement = pattern_analysis.get("engagement_level", "medium")
        frequency = pattern_analysis.get("conversation_frequency", "occasional")
        
        if engagement == "high" and frequency == "frequent":
            persona = "power_user"
        elif engagement == "high":
            persona = "engaged_learner"
        elif frequency == "frequent":
            persona = "regular_user"
        else:
            persona = "casual_user"
        
        return {"persona": persona, "confidence": 0.7}

    async def _analyze_conversation_style(self, phone_number: str) -> Dict[str, Any]:
        """Analyze user's conversation style"""
        # Get recent messages for style analysis
        conversations = await self.base.db.enhanced_conversations.find(
            {"phone_number": phone_number}
        ).sort("timestamp", -1).limit(10).to_list(length=10)
        
        if not conversations:
            return {}
        
        user_messages = [conv.get("user_message", "") for conv in conversations]
        
        # Analyze style characteristics
        avg_length = sum(len(msg.split()) for msg in user_messages) / len(user_messages)
        question_ratio = sum(msg.count("?") for msg in user_messages) / len(user_messages)
        
        style = {
            "communication_length": "detailed" if avg_length > 15 else "concise" if avg_length < 5 else "moderate",
            "question_frequency": "high" if question_ratio > 1 else "low" if question_ratio < 0.5 else "moderate",
            "avg_message_length": avg_length
        }
        
        return style

    async def _analyze_topic_preferences(self, phone_number: str) -> Dict[str, Any]:
        """Analyze user's topic preferences"""
        # Analyze symbol mentions and intent patterns
        conversations = await self.base.db.enhanced_conversations.find(
            {"phone_number": phone_number}
        ).sort("timestamp", -1).limit(30).to_list(length=30)
        
        symbol_counts = defaultdict(int)
        intent_counts = defaultdict(int)
        
        for conv in conversations:
            for symbol in conv.get("symbols_mentioned", []):
                symbol_counts[symbol] += 1
            
            intent = conv.get("intent_data", {}).get("intent", "general")
            intent_counts[intent] += 1
        
        # Top preferences
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "preferred_symbols": [{"symbol": s, "frequency": f} for s, f in top_symbols],
            "preferred_intents": [{"intent": i, "frequency": f} for i, f in top_intents]
        }

    async def _suggest_response_optimization(self, pattern_analysis: Dict) -> Dict[str, Any]:
        """Suggest response optimizations based on patterns"""
        suggestions = []
        
        engagement = pattern_analysis.get("engagement_level", "medium")
        if engagement == "low":
            suggestions.append({
                "type": "engagement",
                "suggestion": "Use more interactive elements and questions",
                "priority": "high"
            })
        
        frequency = pattern_analysis.get("conversation_frequency", "occasional")
        if frequency == "occasional":
            suggestions.append({
                "type": "retention",
                "suggestion": "Send proactive insights to increase engagement",
                "priority": "medium"
            })
        
        return {"suggestions": suggestions}

    async def _identify_engagement_triggers(self, phone_number: str) -> List[Dict]:
        """Identify what triggers user engagement"""
        # Analyze patterns of high-engagement conversations
        conversations = await self.base.db.enhanced_conversations.find(
            {"phone_number": phone_number}
        ).sort("timestamp", -1).limit(20).to_list(length=20)
        
        triggers = []
        
        # Look for patterns in engaged conversations (longer user messages)
        for conv in conversations:
            user_msg_length = len(conv.get("user_message", "").split())
            if user_msg_length > 10:  # Engaged response
                intent = conv.get("intent_data", {}).get("intent", "general")
                symbols = conv.get("symbols_mentioned", [])
                
                if intent != "general":
                    triggers.append({
                        "type": "intent",
                        "trigger": intent,
                        "confidence": 0.6
                    })
                
                for symbol in symbols:
                    triggers.append({
                        "type": "symbol",
                        "trigger": symbol,
                        "confidence": 0.5
                    })
        
        # Count frequency and return top triggers
        trigger_counts = defaultdict(int)
        for trigger in triggers:
            key = f"{trigger['type']}:{trigger['trigger']}"
            trigger_counts[key] += 1
        
        top_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [
            {
                "type": trigger.split(":")[0],
                "trigger": trigger.split(":")[1],
                "frequency": count,
                "confidence": 0.7
            }
            for trigger, count in top_triggers
        ]
