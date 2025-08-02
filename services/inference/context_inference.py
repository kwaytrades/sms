# services/inference/context_inference.py - Enhanced with Profile Integration
"""
Context Inference Service v2.0 - Enhanced with ProfileInferenceService integration
Real-time context scoring with profile inference feedback loop
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from loguru import logger

from services.db.base_db_service import BaseDBService


class ContextInferenceService:
    """
    Enhanced Context inference service with profile integration
    
    NEW FEATURES v2.0:
    - Real-time feedback from ProfileInferenceService
    - Dynamic context quality scoring based on profile completeness
    - Profile-aware conversation intelligence
    - Adaptive context weighting based on inference results
    - Cross-service learning and improvement
    """
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self._topic_memory = {}
        self._intent_patterns = self._build_intent_patterns()
        self._profile_feedback_cache = {}  # Cache for profile inference feedback

    async def initialize(self):
        """Initialize enhanced context inference service"""
        try:
            # Existing indexes
            await self.base.db.context_analytics.create_index("phone_number")
            await self.base.db.context_analytics.create_index("created_at")
            
            # NEW: Profile integration indexes
            await self.base.db.profile_context_feedback.create_index("phone_number")
            await self.base.db.profile_context_feedback.create_index("created_at")
            await self.base.db.profile_context_feedback.create_index("feedback_type")
            
            # NEW: Context quality tracking
            await self.base.db.context_quality_history.create_index("phone_number")
            await self.base.db.context_quality_history.create_index("created_at")
            
            logger.info("✅ Enhanced Context inference service v2.0 initialized")
        except Exception as e:
            logger.exception(f"❌ Enhanced context inference service initialization failed: {e}")

    async def enhance_context(self, context: Dict, phone_number: str) -> Dict:
        """
        Enhanced context enhancement with profile integration feedback
        """
        try:
            if not context or not phone_number:
                return context
            
            # Get profile feedback for this user
            profile_feedback = await self._get_profile_feedback(phone_number)
            
            # Analyze conversation patterns with profile context
            pattern_analysis = await self._analyze_conversation_patterns(phone_number, profile_feedback)
            
            # Enhanced relationship stage with profile intelligence
            enhanced_stage = await self._enhance_relationship_stage(context, pattern_analysis, profile_feedback)
            
            # Build conversation intelligence with profile awareness
            conversation_intel = await self._build_conversation_intelligence(phone_number, pattern_analysis, profile_feedback)
            
            # Predict next topics with profile insights
            topic_predictions = await self._predict_next_topics(phone_number, context, profile_feedback)
            
            # Calculate enhanced context quality score
            context_quality = await self._calculate_enhanced_context_quality(context, profile_feedback)
            
            # Enhanced context with profile integration
            enhanced_context = context.copy()
            enhanced_context.update({
                "relationship_stage": enhanced_stage,
                "conversation_intelligence": conversation_intel,
                "topic_predictions": topic_predictions,
                "pattern_analysis": pattern_analysis,
                "context_quality_score": context_quality,
                "profile_integration": {
                    "has_profile_feedback": bool(profile_feedback),
                    "profile_completeness": profile_feedback.get("profile_completeness", 0.0) if profile_feedback else 0.0,
                    "last_inference_quality": profile_feedback.get("inference_quality", 0.0) if profile_feedback else 0.0,
                    "profile_confidence": profile_feedback.get("average_confidence", 0.0) if profile_feedback else 0.0
                },
                "enhancement_timestamp": datetime.now(timezone.utc).isoformat(),
                "algorithm_version": "v2.0"
            })
            
            # Store context quality history
            await self._store_context_quality_history(phone_number, context_quality, enhanced_context)
            
            # Update context analytics with profile integration
            await self._update_enhanced_context_analytics(phone_number, enhanced_context, profile_feedback)
            
            return enhanced_context
            
        except Exception as e:
            logger.exception(f"❌ Error in enhanced context processing: {e}")
            return context

    async def update_from_profile_inference(self, profile_update: Dict):
        """
        NEW: Receive and process feedback from ProfileInferenceService
        """
        try:
            phone_number = profile_update.get("phone_number")
            if not phone_number:
                return
            
            # Store profile feedback
            feedback_doc = {
                "phone_number": phone_number,
                "inference_applied": profile_update.get("inference_applied", False),
                "inference_quality": profile_update.get("inference_quality", 0.0),
                "profile_improvements": profile_update.get("profile_improvements", {}),
                "inferred_patterns": profile_update.get("inferred_patterns", {}),
                "feedback_timestamp": profile_update.get("timestamp"),
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.profile_context_feedback.insert_one(feedback_doc)
            
            # Update cached feedback
            self._profile_feedback_cache[phone_number] = {
                "last_updated": datetime.now(timezone.utc),
                "data": feedback_doc
            }
            
            # Immediately update context intelligence with new profile insights
            await self._update_context_with_profile_insights(phone_number, profile_update)
            
            logger.debug(f"📥 Processed profile inference feedback for {phone_number}")
            
        except Exception as e:
            logger.exception(f"❌ Error processing profile inference feedback: {e}")

    async def _get_profile_feedback(self, phone_number: str) -> Optional[Dict]:
        """Get latest profile feedback for enhanced context scoring"""
        try:
            # Check cache first
            if phone_number in self._profile_feedback_cache:
                cached = self._profile_feedback_cache[phone_number]
                if (datetime.now(timezone.utc) - cached["last_updated"]).total_seconds() < 1800:  # 30 minutes
                    return cached["data"]
            
            # Get from database
            feedback = await self.base.db.profile_context_feedback.find_one(
                {"phone_number": phone_number},
                sort=[("created_at", -1)]
            )
            
            if feedback:
                # Cache the result
                self._profile_feedback_cache[phone_number] = {
                    "last_updated": datetime.now(timezone.utc),
                    "data": feedback
                }
                
                return feedback
            
            return None
            
        except Exception as e:
            logger.exception(f"❌ Error getting profile feedback: {e}")
            return None

    async def _analyze_conversation_patterns(self, phone_number: str, profile_feedback: Dict = None) -> Dict[str, Any]:
        """Enhanced pattern analysis with profile context"""
        try:
            # Get basic pattern analysis
            basic_analysis = await self._get_basic_conversation_patterns(phone_number)
            
            # Enhance with profile feedback
            if profile_feedback:
                enhanced_analysis = basic_analysis.copy()
                
                # Profile-aware engagement scoring
                profile_improvements = profile_feedback.get("profile_improvements", {})
                if profile_improvements:
                    # Higher engagement if profile is being actively improved
                    engagement_boost = len(profile_improvements) * 0.1
                    current_engagement = enhanced_analysis.get("engagement_level", "medium")
                    
                    if engagement_boost > 0.2 and current_engagement == "medium":
                        enhanced_analysis["engagement_level"] = "high"
                        enhanced_analysis["profile_driven_engagement"] = True
                
                # Profile-aware conversation frequency
                inference_quality = profile_feedback.get("inference_quality", 0.0)
                if inference_quality > 0.7:
                    # High-quality inferences suggest deep engagement
                    enhanced_analysis["conversation_depth"] = "deep"
                else:
                    enhanced_analysis["conversation_depth"] = "surface"
                
                # Add profile insights
                enhanced_analysis["profile_insights"] = {
                    "recent_improvements": list(profile_improvements.keys()),
                    "inference_quality": inference_quality,
                    "profile_driven_patterns": profile_feedback.get("inferred_patterns", {})
                }
                
                return enhanced_analysis
            
            return basic_analysis
            
        except Exception as e:
            logger.exception(f"❌ Error in enhanced pattern analysis: {e}")
            return {}

    async def _calculate_enhanced_context_quality(self, context: Dict, profile_feedback: Dict = None) -> float:
        """Enhanced context quality calculation with profile integration"""
        try:
            # Base quality calculation
            base_quality = self._calculate_base_context_quality(context)
            
            # Profile enhancement factor
            profile_factor = 0.0
            
            if profile_feedback:
                # Profile completeness factor (0.2 weight)
                profile_improvements = profile_feedback.get("profile_improvements", {})
                completeness_factor = min(len(profile_improvements) / 5.0, 1.0) * 0.2
                
                # Inference quality factor (0.3 weight)
                inference_quality = profile_feedback.get("inference_quality", 0.0)
                quality_factor = inference_quality * 0.3
                
                # Pattern recognition factor (0.2 weight)
                inferred_patterns = profile_feedback.get("inferred_patterns", {})
                pattern_factor = min(len(inferred_patterns) / 3.0, 1.0) * 0.2
                
                # Recency factor (0.3 weight)
                feedback_timestamp = profile_feedback.get("feedback_timestamp")
                if feedback_timestamp:
                    try:
                        feedback_time = datetime.fromisoformat(feedback_timestamp)
                        hours_since = (datetime.now(timezone.utc) - feedback_time).total_seconds() / 3600
                        recency_factor = max(0, 1.0 - (hours_since / 24)) * 0.3  # Decay over 24 hours
                    except:
                        recency_factor = 0.1
                else:
                    recency_factor = 0.1
                
                profile_factor = completeness_factor + quality_factor + pattern_factor + recency_factor
            
            # Combine base quality with profile factor
            enhanced_quality = base_quality * 0.7 + profile_factor * 0.3
            
            return min(enhanced_quality, 1.0)
            
        except Exception as e:
            logger.exception(f"❌ Error calculating enhanced context quality: {e}")
            return 0.5

    def _calculate_base_context_quality(self, context: Dict) -> float:
        """Calculate base context quality (original method)"""
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
            logger.exception(f"❌ Error calculating base context quality: {e}")
            return 0.0

    async def _enhance_relationship_stage(self, context: Dict, pattern_analysis: Dict, profile_feedback: Dict = None) -> str:
        """Enhanced relationship stage with profile intelligence"""
        try:
            # Get base stage
            base_stage = await self._get_base_relationship_stage(context, pattern_analysis)
            
            if not profile_feedback:
                return base_stage
            
            # Profile-driven stage enhancement
            profile_improvements = profile_feedback.get("profile_improvements", {})
            inference_quality = profile_feedback.get("inference_quality", 0.0)
            
            # Strong profile inference suggests deeper engagement
            if len(profile_improvements) >= 3 and inference_quality > 0.7:
                stage_progression = {
                    "new": "exploring",
                    "exploring": "engaged",
                    "engaged": "trusted",
                    "regular": "trusted",
                    "trusted": "established",
                    "established": "highly_engaged"
                }
                return stage_progression.get(base_stage, base_stage)
            
            # Medium profile activity suggests moderate progression
            elif len(profile_improvements) >= 1 and inference_quality > 0.5:
                if base_stage == "new":
                    return "exploring"
                elif base_stage == "exploring":
                    return "engaged"
            
            return base_stage
            
        except Exception as e:
            logger.exception(f"❌ Error enhancing relationship stage: {e}")
            return "new"

    async def _build_conversation_intelligence(self, phone_number: str, pattern_analysis: Dict, profile_feedback: Dict = None) -> Dict[str, Any]:
        """Enhanced conversation intelligence with profile awareness"""
        try:
            # Get base intelligence
            base_intelligence = await self._get_base_conversation_intelligence(phone_number, pattern_analysis)
            
            if not profile_feedback:
                return base_intelligence
            
            # Add profile-driven insights
            enhanced_intelligence = base_intelligence.copy()
            
            # Profile-driven user persona refinement
            inferred_patterns = profile_feedback.get("inferred_patterns", {})
            
            if "communication" in inferred_patterns:
                comm_style = inferred_patterns["communication"]
                enhanced_intelligence["profile_driven_persona"] = {
                    "communication_style": comm_style,
                    "inference_source": "profile_analysis",
                    "confidence": profile_feedback.get("inference_quality", 0.0)
                }
            
            # Financial goal insights
            if "financial_goals" in inferred_patterns:
                goals = inferred_patterns["financial_goals"]
                enhanced_intelligence["financial_intelligence"] = {
                    "inferred_goals": goals,
                    "goal_complexity": len(goals),
                    "primary_goal_type": goals[0].get("type") if goals else None
                }
            
            # Profile completeness insights
            profile_improvements = profile_feedback.get("profile_improvements", {})
            enhanced_intelligence["profile_insights"] = {
                "active_improvement_areas": list(profile_improvements.keys()),
                "profile_maturity": "developing" if len(profile_improvements) > 2 else "established",
                "inference_accuracy": profile_feedback.get("inference_quality", 0.0)
            }
            
            return enhanced_intelligence
            
        except Exception as e:
            logger.exception(f"❌ Error building enhanced conversation intelligence: {e}")
            return {}

    async def _predict_next_topics(self, phone_number: str, context: Dict, profile_feedback: Dict = None) -> List[Dict]:
        """Enhanced topic prediction with profile insights"""
        try:
            # Get base predictions
            base_predictions = await self._get_base_topic_predictions(phone_number, context)
            
            if not profile_feedback:
                return base_predictions
            
            enhanced_predictions = base_predictions.copy()
            
            # Add profile-driven predictions
            inferred_patterns = profile_feedback.get("inferred_patterns", {})
            
            # Goal-based predictions
            if "financial_goals" in inferred_patterns:
                goals = inferred_patterns["financial_goals"]
                for goal in goals[:2]:  # Top 2 goals
                    goal_type = goal.get("type", "unknown")
                    enhanced_predictions.append({
                        "type": "goal_related",
                        "topic": f"{goal_type} strategy discussion",
                        "confidence": goal.get("confidence", 0.5) * 0.8,  # Slightly lower confidence
                        "reason": f"Inferred {goal_type} goal from profile analysis",
                        "source": "profile_inference"
                    })
            
            # Communication style predictions
            if "communication" in inferred_patterns:
                comm_style = inferred_patterns["communication"]
                
                if comm_style.get("detail_preference") == "detailed":
                    enhanced_predictions.append({
                        "type": "detailed_analysis",
                        "topic": "in-depth market analysis",
                        "confidence": 0.7,
                        "reason": "User prefers detailed information",
                        "source": "communication_style_inference"
                    })
                elif comm_style.get("detail_preference") == "brief":
                    enhanced_predictions.append({
                        "type": "quick_update",
                        "topic": "market summary",
                        "confidence": 0.7,
                        "reason": "User prefers brief updates",
                        "source": "communication_style_inference"
                    })
            
            # Profile improvement suggestions
            profile_improvements = profile_feedback.get("profile_improvements", {})
            missing_areas = ["demographics", "financial", "goals"] 
            for area in missing_areas:
                if area not in profile_improvements:
                    enhanced_predictions.append({
                        "type": "profile_completion",
                        "topic": f"{area} information gathering",
                        "confidence": 0.4,
                        "reason": f"Could improve {area} profile information",
                        "source": "profile_gap_analysis"
                    })
            
            # Sort by confidence and limit
            enhanced_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return enhanced_predictions[:8]  # Top 8 predictions
            
        except Exception as e:
            logger.exception(f"❌ Error in enhanced topic prediction: {e}")
            return []

    async def _update_context_with_profile_insights(self, phone_number: str, profile_update: Dict):
        """Update context intelligence with new profile insights"""
        try:
            # Update topic memory with profile insights
            if phone_number not in self._topic_memory:
                self._topic_memory[phone_number] = {
                    "topic_sequence": [],
                    "symbol_frequency": defaultdict(int),
                    "intent_patterns": defaultdict(list),
                    "profile_insights": []
                }
            
            memory = self._topic_memory[phone_number]
            
            # Add profile insights to memory
            profile_insight = {
                "timestamp": datetime.now(timezone.utc),
                "inference_quality": profile_update.get("inference_quality", 0.0),
                "improvements": list(profile_update.get("profile_improvements", {}).keys()),
                "patterns": profile_update.get("inferred_patterns", {})
            }
            
            memory["profile_insights"].append(profile_insight)
            memory["profile_insights"] = memory["profile_insights"][-5:]  # Keep last 5
            
            # Update context analytics immediately
            await self._store_profile_context_update(phone_number, profile_update)
            
        except Exception as e:
            logger.exception(f"❌ Error updating context with profile insights: {e}")

    async def _store_context_quality_history(self, phone_number: str, quality_score: float, context: Dict):
        """Store context quality history for trend analysis"""
        try:
            quality_doc = {
                "phone_number": phone_number,
                "quality_score": quality_score,
                "base_quality": self._calculate_base_context_quality(context),
                "profile_enhancement": quality_score - self._calculate_base_context_quality(context),
                "context_metadata": {
                    "message_count": len(context.get("recent_messages", [])),
                    "relationship_stage": context.get("relationship_stage", "new"),
                    "has_profile_integration": bool(context.get("profile_integration")),
                    "algorithm_version": "v2.0"
                },
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.context_quality_history.insert_one(quality_doc)
            
        except Exception as e:
            logger.exception(f"❌ Error storing context quality history: {e}")

    async def _store_profile_context_update(self, phone_number: str, profile_update: Dict):
        """Store immediate profile context update"""
        try:
            update_doc = {
                "phone_number": phone_number,
                "update_type": "profile_feedback",
                "inference_quality": profile_update.get("inference_quality", 0.0),
                "improvements_count": len(profile_update.get("profile_improvements", {})),
                "patterns_count": len(profile_update.get("inferred_patterns", {})),
                "immediate_impact": True,
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.context_analytics.insert_one(update_doc)
            
        except Exception as e:
            logger.exception(f"❌ Error storing profile context update: {e}")

    async def get_context_quality_trend(self, phone_number: str, days: int = 7) -> Dict[str, Any]:
        """Get context quality trend analysis"""
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            history = await self.base.db.context_quality_history.find({
                "phone_number": phone_number,
                "created_at": {"$gte": start_date}
            }).sort("created_at", 1).to_list(length=None)
            
            if not history:
                return {"error": "No quality history found"}
            
            # Calculate trend metrics
            quality_scores = [h["quality_score"] for h in history]
            base_scores = [h["base_quality"] for h in history]
            profile_enhancements = [h["profile_enhancement"] for h in history]
            
            trend_analysis = {
                "phone_number": phone_number,
                "period_days": days,
                "data_points": len(history),
                "current_quality": quality_scores[-1] if quality_scores else 0,
                "average_quality": sum(quality_scores) / len(quality_scores),
                "quality_trend": "improving" if quality_scores[-1] > quality_scores[0] else "declining",
                "profile_contribution": {
                    "average_enhancement": sum(profile_enhancements) / len(profile_enhancements),
                    "max_enhancement": max(profile_enhancements),
                    "enhancement_trend": "improving" if profile_enhancements[-1] > profile_enhancements[0] else "stable"
                },
                "quality_stability": self._calculate_stability(quality_scores),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.exception(f"❌ Error getting context quality trend: {e}")
            return {"error": str(e)}

    def _calculate_stability(self, scores: List[float]) -> str:
        """Calculate quality stability from score sequence"""
        try:
            if len(scores) < 3:
                return "insufficient_data"
            
            # Calculate variance
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            
            if variance < 0.01:
                return "very_stable"
            elif variance < 0.05:
                return "stable"
            elif variance < 0.1:
                return "moderate"
            else:
                return "volatile"
                
        except Exception as e:
            logger.exception(f"❌ Error calculating stability: {e}")
            return "unknown"

    # Helper methods for base functionality
    async def _get_basic_conversation_patterns(self, phone_number: str) -> Dict[str, Any]:
        """Get basic conversation patterns (original method)"""
        try:
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

    async def _get_base_relationship_stage(self, context: Dict, pattern_analysis: Dict) -> str:
        """Get base relationship stage (original method)"""
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
            logger.exception(f"❌ Error getting base relationship stage: {e}")
            return "new"

    async def _get_base_conversation_intelligence(self, phone_number: str, pattern_analysis: Dict) -> Dict[str, Any]:
        """Get base conversation intelligence (original method)"""
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
            logger.exception(f"❌ Error building base conversation intelligence: {e}")
            return {}

    async def _get_base_topic_predictions(self, phone_number: str, context: Dict) -> List[Dict]:
        """Get base topic predictions (original method)"""
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
                    "reason": f"Recent discussion about {symbol}",
                    "source": "symbol_history"
                })
            
            # Predict based on conversation patterns
            if any(topic.get("topic") == "analysis" for topic in recent_topics):
                predictions.append({
                    "type": "analysis_request",
                    "topic": "market analysis",
                    "confidence": 0.6,
                    "reason": "Pattern of requesting analysis",
                    "source": "conversation_patterns"
                })
            
            # Predict based on time patterns
            now = datetime.now(timezone.utc)
            if now.hour < 10:  # Morning
                predictions.append({
                    "type": "market_open",
                    "topic": "pre-market discussion",
                    "confidence": 0.5,
                    "reason": "Pre-market hours",
                    "source": "time_based"
                })
            elif now.hour > 16:  # After market close
                predictions.append({
                    "type": "market_close",
                    "topic": "post-market analysis",
                    "confidence": 0.5,
                    "reason": "Post-market hours",
                    "source": "time_based"
                })
            
            return predictions
            
        except Exception as e:
            logger.exception(f"❌ Error getting base topic predictions: {e}")
            return []

    async def _update_enhanced_context_analytics(self, phone_number: str, enhanced_context: Dict, profile_feedback: Dict):
        """Update context analytics with enhanced information"""
        try:
            analytics_doc = {
                "phone_number": phone_number,
                "context_quality_score": enhanced_context.get("context_quality_score", 0),
                "relationship_stage": enhanced_context.get("relationship_stage", "new"),
                "enhancement_applied": True,
                "profile_integration": enhanced_context.get("profile_integration", {}),
                "algorithm_version": "v2.0",
                "has_profile_feedback": bool(profile_feedback),
                "created_at": datetime.now(timezone.utc)
            }
            
            # Add profile-specific metrics if available
            if profile_feedback:
                analytics_doc["profile_metrics"] = {
                    "inference_quality": profile_feedback.get("inference_quality", 0.0),
                    "improvements_count": len(profile_feedback.get("profile_improvements", {})),
                    "patterns_detected": len(profile_feedback.get("inferred_patterns", {}))
                }
            
            await self.base.db.context_analytics.insert_one(analytics_doc)
            
        except Exception as e:
            logger.exception(f"❌ Error updating enhanced context analytics: {e}")

    # Keep all remaining original helper methods...
    def _build_intent_patterns(self) -> Dict[str, Dict]:
        """Build intent prediction patterns (original method)"""
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

    # Additional helper methods (implementing missing ones from original)
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

    # Additional existing methods that remain unchanged...
    async def update_from_message(self, phone_number: str, user_message: str, 
                                bot_response: str, intent_data: Dict, symbols: List[str]):
        """Update context intelligence from new message (enhanced)"""
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
            
            # NEW: Check if this message should trigger profile inference
            await self._check_profile_inference_trigger(phone_number, message_insights)
            
        except Exception as e:
            logger.exception(f"❌ Error updating context from message: {e}")

    async def _check_profile_inference_trigger(self, phone_number: str, message_insights: Dict):
        """Check if message should trigger profile inference"""
        try:
            # Trigger profile inference if we detect profile-relevant information
            profile_triggers = [
                "occupation", "age", "income", "goals", "experience", "risk", "trading"
            ]
            
            user_message = message_insights.get("original_user_message", "").lower()
            
            if any(trigger in user_message for trigger in profile_triggers):
                # Log potential profile inference opportunity
                await self.base.db.profile_inference_triggers.insert_one({
                    "phone_number": phone_number,
                    "trigger_type": "message_content",
                    "message_insights": message_insights,
                    "detected_triggers": [t for t in profile_triggers if t in user_message],
                    "created_at": datetime.now(timezone.utc)
                })
                
                logger.debug(f"🎯 Profile inference trigger detected for {phone_number}")
        
        except Exception as e:
            logger.exception(f"❌ Error checking profile inference trigger: {e}")

    async def _extract_message_insights(self, user_message: str, bot_response: str, 
                                      intent_data: Dict, symbols: List[str]) -> Dict[str, Any]:
        """Extract insights from a single message exchange (enhanced)"""
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
                "original_user_message": user_message,  # Store for profile inference
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
        """Update topic evolution tracking (enhanced)"""
        try:
            if phone_number not in self._topic_memory:
                self._topic_memory[phone_number] = {
                    "topic_sequence": [],
                    "symbol_frequency": defaultdict(int),
                    "intent_patterns": defaultdict(list),
                    "profile_insights": []
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
        """Update conversation pattern analysis (enhanced)"""
        try:
            # Update in database for persistence
            pattern_doc = {
                "phone_number": phone_number,
                "message_insights": insights,
                "algorithm_version": "v2.0",
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.context_analytics.insert_one(pattern_doc)
            
        except Exception as e:
            logger.exception(f"❌ Error updating conversation patterns: {e}")

    async def _check_context_triggers(self, phone_number: str, insights: Dict):
        """Check for context-based triggers (enhanced)"""
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
                
                # Store negative sentiment for profile inference consideration
                await self.base.db.sentiment_alerts.insert_one({
                    "phone_number": phone_number,
                    "sentiment": "negative",
                    "message_length": insights.get("user_message_length", 0),
                    "urgency_indicators": insights.get("urgency_indicators", []),
                    "created_at": datetime.now(timezone.utc)
                })
            
        except Exception as e:
            logger.exception(f"❌ Error checking context triggers: {e}")
