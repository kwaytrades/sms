# services/inference/profile_inference.py - Enhanced with Versioning & Context Integration
"""
Profile Inference Service v2.0 - Enhanced with versioned snapshots and context integration
Intelligent data collection with complete audit trail and rollback capabilities
"""

import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
from loguru import logger

from models.user import UserProfile
from services.db.base_db_service import BaseDBService


class ProfileInferenceService:
    """
    Enhanced Auto-inference service with versioning and context integration
    
    NEW FEATURES v2.0:
    - Versioned profile snapshots with rollback capability
    - Integration with ContextInferenceService for real-time feedback
    - Confidence decay and improvement tracking
    - A/B testing for inference algorithms
    - Performance metrics and accuracy measurement
    - Rollback triggers for poor inference results
    """
    
    def __init__(self, base_service: BaseDBService, context_inference_service=None):
        self.base = base_service
        self.context_inference = context_inference_service  # Injected for tight integration
        self._inference_patterns = self._build_inference_patterns()
        self._confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        self._version_retention_days = 90  # Keep profile versions for 90 days
        self._rollback_triggers = {
            "low_confidence_streak": 3,  # 3 consecutive low-confidence inferences
            "negative_feedback_threshold": 0.3,  # Context quality drops below 30%
            "user_correction_weight": 0.8  # High weight for explicit user corrections
        }

    async def initialize(self):
        """Initialize enhanced inference service with versioning"""
        try:
            # Create inference tracking collection
            await self.base.db.inference_history.create_index("user_id")
            await self.base.db.inference_history.create_index("inference_type")
            await self.base.db.inference_history.create_index("created_at")
            
            # NEW: Profile version snapshots collection
            await self.base.db.profile_versions.create_index("user_id")
            await self.base.db.profile_versions.create_index("version_number")
            await self.base.db.profile_versions.create_index("created_at")
            await self.base.db.profile_versions.create_index([("user_id", 1), ("created_at", -1)])
            
            # NEW: Inference performance tracking
            await self.base.db.inference_performance.create_index("user_id")
            await self.base.db.inference_performance.create_index("algorithm_version")
            await self.base.db.inference_performance.create_index("created_at")
            
            # NEW: Inference feedback collection
            await self.base.db.inference_feedback.create_index("user_id")
            await self.base.db.inference_feedback.create_index("inference_id")
            await self.base.db.inference_feedback.create_index("feedback_type")
            
            logger.info("✅ Enhanced Profile inference service v2.0 initialized")
        except Exception as e:
            logger.exception(f"❌ Enhanced profile inference service initialization failed: {e}")

    async def enhance_user_profile(self, user: UserProfile) -> UserProfile:
        """
        Enhanced profile enhancement with versioning and context feedback
        """
        try:
            if not user or not user.phone_number:
                return user
            
            # Check if inference should run
            if not await self._should_run_inference(user):
                return user
            
            # Create profile snapshot BEFORE inference
            pre_inference_snapshot = await self._create_profile_snapshot(user, "pre_inference")
            
            # Get conversation history for analysis
            conversations = await self._get_conversation_history(user.phone_number)
            if not conversations:
                return user
            
            # Calculate current profile completeness and context quality
            original_completeness = await self.calculate_completeness(user)
            original_context_quality = await self._get_current_context_quality(user.phone_number)
            
            # Run enhanced inference with versioning
            enhanced_user, inference_results = await self._run_versioned_inference(
                user, conversations, pre_inference_snapshot
            )
            
            # Calculate post-inference metrics
            new_completeness = await self.calculate_completeness(enhanced_user)
            
            # Create post-inference snapshot
            post_inference_snapshot = await self._create_profile_snapshot(
                enhanced_user, "post_inference", inference_results
            )
            
            # Evaluate inference quality with context feedback
            quality_assessment = await self._assess_inference_quality(
                user, enhanced_user, inference_results, original_context_quality
            )
            
            # Decide whether to apply or rollback based on quality
            if quality_assessment["should_apply"]:
                # Apply inference and update context
                enhanced_user.last_inference_at = datetime.utcnow()
                
                # Feed results back to ContextInferenceService for immediate improvement
                if self.context_inference:
                    await self._update_context_with_inference_results(
                        user.phone_number, inference_results, quality_assessment
                    )
                
                # Log successful inference
                await self._log_enhanced_inference_results(
                    user.user_id or str(user._id), 
                    inference_results, 
                    quality_assessment,
                    original_completeness, 
                    new_completeness,
                    pre_inference_snapshot["version_id"],
                    post_inference_snapshot["version_id"]
                )
                
                logger.info(f"✅ Applied enhanced profile inference: {original_completeness:.2f} → {new_completeness:.2f} completeness (quality: {quality_assessment['confidence_score']:.2f})")
                
                return enhanced_user
            else:
                # Rollback triggered - return original user
                await self._handle_inference_rollback(
                    user, inference_results, quality_assessment, pre_inference_snapshot
                )
                
                logger.warning(f"🔄 Inference rollback triggered for user {user.phone_number}: {quality_assessment['rollback_reason']}")
                
                return user
            
        except Exception as e:
            logger.exception(f"❌ Error in enhanced user profile inference: {e}")
            return user

    async def _create_profile_snapshot(self, user: UserProfile, snapshot_type: str, 
                                     inference_results: Dict = None) -> Dict[str, Any]:
        """Create a versioned snapshot of the user profile"""
        try:
            user_id = user.user_id or str(user._id)
            
            # Get next version number
            latest_version = await self.base.db.profile_versions.find_one(
                {"user_id": user_id},
                sort=[("version_number", -1)]
            )
            
            version_number = (latest_version["version_number"] + 1) if latest_version else 1
            
            # Create snapshot
            snapshot = {
                "user_id": user_id,
                "version_number": version_number,
                "snapshot_type": snapshot_type,
                "profile_data": self._serialize_user_profile(user),
                "inference_results": inference_results,
                "created_at": datetime.now(timezone.utc),
                "version_id": self._generate_version_id(user_id, version_number),
                "metadata": {
                    "completeness_score": await self.calculate_completeness(user),
                    "algorithm_version": "v2.0",
                    "snapshot_hash": self._calculate_profile_hash(user)
                }
            }
            
            # Store snapshot
            await self.base.db.profile_versions.insert_one(snapshot)
            
            # Cleanup old versions (keep last 10 versions or 90 days)
            await self._cleanup_old_versions(user_id)
            
            logger.debug(f"📸 Created profile snapshot v{version_number} for {user_id}")
            
            return snapshot
            
        except Exception as e:
            logger.exception(f"❌ Error creating profile snapshot: {e}")
            return {}

    async def _run_versioned_inference(self, user: UserProfile, conversations: List[Dict], 
                                     pre_snapshot: Dict) -> Tuple[UserProfile, Dict]:
        """Run inference with enhanced tracking and versioning"""
        try:
            enhanced_user = user
            inference_results = {
                "algorithm_version": "v2.0",
                "pre_snapshot_id": pre_snapshot.get("version_id"),
                "inference_timestamp": datetime.now(timezone.utc).isoformat(),
                "results": {}
            }
            
            # Demographic inference with confidence tracking
            demo_results = await self._infer_demographics(conversations, enhanced_user)
            if demo_results:
                enhanced_user = self._apply_demographic_updates(enhanced_user, demo_results)
                inference_results["results"]["demographics"] = demo_results
            
            # Financial profile inference
            financial_results = await self._infer_financial_profile(conversations, enhanced_user)
            if financial_results:
                enhanced_user = self._apply_financial_updates(enhanced_user, financial_results)
                inference_results["results"]["financial"] = financial_results
            
            # Goal extraction with timeline prediction
            goal_results = await self._extract_goals(conversations, enhanced_user)
            if goal_results:
                inference_results["results"]["goals"] = goal_results
                await self._save_inferred_goals(user.user_id or str(user._id), goal_results)
            
            # Life event detection with impact analysis
            event_results = await self._detect_life_events(conversations, enhanced_user)
            if event_results:
                enhanced_user = self._apply_event_updates(enhanced_user, event_results)
                inference_results["results"]["life_events"] = event_results
            
            # Communication style inference
            style_results = await self._infer_communication_style(conversations, enhanced_user)
            if style_results:
                enhanced_user = self._apply_style_updates(enhanced_user, style_results)
                inference_results["results"]["communication_style"] = style_results
            
            # NEW: Investment personality inference
            personality_results = await self._infer_investment_personality(conversations, enhanced_user)
            if personality_results:
                enhanced_user = self._apply_personality_updates(enhanced_user, personality_results)
                inference_results["results"]["investment_personality"] = personality_results
            
            return enhanced_user, inference_results
            
        except Exception as e:
            logger.exception(f"❌ Error in versioned inference: {e}")
            return user, {}

    async def _assess_inference_quality(self, original_user: UserProfile, enhanced_user: UserProfile,
                                      inference_results: Dict, original_context_quality: float) -> Dict[str, Any]:
        """Assess quality of inference results with multiple metrics"""
        try:
            assessment = {
                "should_apply": True,
                "confidence_score": 0.0,
                "quality_metrics": {},
                "rollback_reason": None,
                "feedback_signals": {}
            }
            
            # 1. Confidence score analysis
            confidence_scores = []
            for category, results in inference_results.get("results", {}).items():
                if isinstance(results, dict) and "confidence" in results:
                    confidence_scores.append(results["confidence"])
                elif isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and "confidence" in result:
                            confidence_scores.append(result["confidence"])
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            assessment["confidence_score"] = avg_confidence
            assessment["quality_metrics"]["average_confidence"] = avg_confidence
            
            # 2. Profile completeness improvement
            original_completeness = await self.calculate_completeness(original_user)
            new_completeness = await self.calculate_completeness(enhanced_user)
            completeness_improvement = new_completeness - original_completeness
            
            assessment["quality_metrics"]["completeness_improvement"] = completeness_improvement
            
            # 3. Check for confidence streak
            recent_inferences = await self._get_recent_inference_history(original_user.user_id or str(original_user._id))
            low_confidence_streak = self._count_low_confidence_streak(recent_inferences)
            
            assessment["quality_metrics"]["low_confidence_streak"] = low_confidence_streak
            
            # 4. Context quality feedback (if ContextInferenceService available)
            new_context_quality = await self._get_current_context_quality(original_user.phone_number)
            context_quality_change = new_context_quality - original_context_quality
            
            assessment["quality_metrics"]["context_quality_change"] = context_quality_change
            assessment["feedback_signals"]["context_quality"] = new_context_quality
            
            # 5. Contradiction detection
            contradictions = await self._detect_profile_contradictions(enhanced_user, inference_results)
            assessment["quality_metrics"]["contradictions_found"] = len(contradictions)
            assessment["feedback_signals"]["contradictions"] = contradictions
            
            # 6. Apply rollback triggers
            if low_confidence_streak >= self._rollback_triggers["low_confidence_streak"]:
                assessment["should_apply"] = False
                assessment["rollback_reason"] = f"Low confidence streak: {low_confidence_streak} consecutive inferences"
            
            elif new_context_quality < self._rollback_triggers["negative_feedback_threshold"]:
                assessment["should_apply"] = False
                assessment["rollback_reason"] = f"Context quality too low: {new_context_quality:.2f}"
            
            elif len(contradictions) > 2:
                assessment["should_apply"] = False
                assessment["rollback_reason"] = f"Too many contradictions detected: {len(contradictions)}"
            
            elif avg_confidence < self._confidence_thresholds["low"]:
                assessment["should_apply"] = False
                assessment["rollback_reason"] = f"Average confidence too low: {avg_confidence:.2f}"
            
            # 7. Positive signals that boost confidence
            if completeness_improvement > 0.1:  # 10% improvement
                assessment["confidence_score"] += 0.1
            
            if context_quality_change > 0.05:  # 5% context improvement
                assessment["confidence_score"] += 0.1
            
            # Final confidence score
            assessment["confidence_score"] = min(assessment["confidence_score"], 1.0)
            
            return assessment
            
        except Exception as e:
            logger.exception(f"❌ Error assessing inference quality: {e}")
            return {"should_apply": False, "rollback_reason": "Assessment error", "confidence_score": 0.0}

    async def _update_context_with_inference_results(self, phone_number: str, 
                                                   inference_results: Dict, quality_assessment: Dict):
        """Feed inference results back to ContextInferenceService for immediate improvement"""
        try:
            if not self.context_inference:
                return
            
            # Create context update payload
            context_update = {
                "phone_number": phone_number,
                "inference_applied": True,
                "inference_quality": quality_assessment["confidence_score"],
                "profile_improvements": {},
                "inferred_patterns": {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Extract profile improvements
            for category, results in inference_results.get("results", {}).items():
                if category in ["demographics", "financial", "communication_style"]:
                    context_update["profile_improvements"][category] = {
                        "fields_updated": list(results.keys()) if isinstance(results, dict) else [],
                        "confidence": results.get("confidence", 0.0) if isinstance(results, dict) else 0.0
                    }
            
            # Extract behavioral patterns
            if "communication_style" in inference_results.get("results", {}):
                style_data = inference_results["results"]["communication_style"]
                if isinstance(style_data, dict) and "style" in style_data:
                    context_update["inferred_patterns"]["communication"] = style_data["style"]
            
            # Extract goal patterns
            if "goals" in inference_results.get("results", {}):
                goals = inference_results["results"]["goals"]
                if isinstance(goals, list):
                    context_update["inferred_patterns"]["financial_goals"] = [
                        {"type": goal.get("goal_type"), "confidence": goal.get("confidence")}
                        for goal in goals
                    ]
            
            # Send update to ContextInferenceService
            await self.context_inference.update_from_profile_inference(context_update)
            
            logger.debug(f"📡 Sent inference feedback to ContextInferenceService for {phone_number}")
            
        except Exception as e:
            logger.exception(f"❌ Error updating context with inference results: {e}")

    async def _handle_inference_rollback(self, original_user: UserProfile, inference_results: Dict,
                                       quality_assessment: Dict, pre_snapshot: Dict):
        """Handle inference rollback and learn from failure"""
        try:
            user_id = original_user.user_id or str(original_user._id)
            
            # Log rollback event
            rollback_doc = {
                "user_id": user_id,
                "rollback_reason": quality_assessment["rollback_reason"],
                "inference_results": inference_results,
                "quality_assessment": quality_assessment,
                "pre_snapshot_id": pre_snapshot.get("version_id"),
                "created_at": datetime.now(timezone.utc),
                "algorithm_version": "v2.0"
            }
            
            await self.base.db.inference_rollbacks.insert_one(rollback_doc)
            
            # Update inference performance tracking
            await self._update_performance_metrics(user_id, "rollback", quality_assessment)
            
            # Learn from the rollback for future improvements
            await self._learn_from_rollback(user_id, inference_results, quality_assessment)
            
            logger.info(f"🔄 Processed inference rollback for {user_id}: {quality_assessment['rollback_reason']}")
            
        except Exception as e:
            logger.exception(f"❌ Error handling inference rollback: {e}")

    async def rollback_to_version(self, user_id: str, version_id: str) -> Optional[UserProfile]:
        """Manually rollback user profile to a specific version"""
        try:
            # Find the target version
            target_version = await self.base.db.profile_versions.find_one({
                "user_id": user_id,
                "version_id": version_id
            })
            
            if not target_version:
                logger.error(f"❌ Version {version_id} not found for user {user_id}")
                return None
            
            # Deserialize the profile data
            profile_data = target_version["profile_data"]
            restored_user = self._deserialize_user_profile(profile_data)
            
            # Create rollback snapshot
            rollback_snapshot = await self._create_profile_snapshot(
                restored_user, "manual_rollback", {"target_version_id": version_id}
            )
            
            # Log manual rollback
            rollback_log = {
                "user_id": user_id,
                "rollback_type": "manual",
                "target_version_id": version_id,
                "new_version_id": rollback_snapshot["version_id"],
                "created_at": datetime.now(timezone.utc),
                "reason": "manual_rollback"
            }
            
            await self.base.db.inference_rollbacks.insert_one(rollback_log)
            
            logger.info(f"🔄 Manual rollback completed for {user_id} to version {version_id}")
            
            return restored_user
            
        except Exception as e:
            logger.exception(f"❌ Error rolling back to version: {e}")
            return None

    async def get_profile_version_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get profile version history for analysis"""
        try:
            versions = await self.base.db.profile_versions.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(limit).to_list(length=limit)
            
            # Process versions for display
            for version in versions:
                version["_id"] = str(version["_id"])
                if isinstance(version.get("created_at"), datetime):
                    version["created_at"] = version["created_at"].isoformat()
            
            return versions
            
        except Exception as e:
            logger.exception(f"❌ Error getting profile version history: {e}")
            return []

    async def get_inference_performance_metrics(self, user_id: str = None, days: int = 30) -> Dict[str, Any]:
        """Get inference performance metrics"""
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            query = {"created_at": {"$gte": start_date}}
            if user_id:
                query["user_id"] = user_id
            
            # Get performance data
            performance_docs = await self.base.db.inference_performance.find(query).to_list(length=None)
            rollback_docs = await self.base.db.inference_rollbacks.find(query).to_list(length=None)
            
            # Calculate metrics
            total_inferences = len(performance_docs)
            total_rollbacks = len(rollback_docs)
            success_rate = ((total_inferences - total_rollbacks) / total_inferences) if total_inferences > 0 else 0
            
            # Confidence distribution
            confidence_scores = [doc.get("confidence_score", 0) for doc in performance_docs]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Rollback reasons
            rollback_reasons = {}
            for rollback in rollback_docs:
                reason = rollback.get("rollback_reason", "unknown")
                rollback_reasons[reason] = rollback_reasons.get(reason, 0) + 1
            
            metrics = {
                "period_days": days,
                "total_inferences": total_inferences,
                "successful_inferences": total_inferences - total_rollbacks,
                "rollbacks": total_rollbacks,
                "success_rate": success_rate,
                "average_confidence": avg_confidence,
                "rollback_reasons": rollback_reasons,
                "algorithm_version": "v2.0",
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            if user_id:
                metrics["user_id"] = user_id
            
            return metrics
            
        except Exception as e:
            logger.exception(f"❌ Error getting inference performance metrics: {e}")
            return {"error": str(e)}

    # NEW HELPER METHODS

    async def _get_current_context_quality(self, phone_number: str) -> float:
        """Get current context quality score from ContextInferenceService"""
        try:
            if not self.context_inference:
                return 0.5  # Default neutral score
            
            # Get context and calculate quality
            context = await self.context_inference.base.db.conversation_context.find_one({
                "phone_number": phone_number
            })
            
            if not context:
                return 0.3  # Low score for missing context
            
            # Simple quality scoring based on context richness
            score = 0.0
            
            # Recent symbols (0.2 weight)
            recent_symbols = context.get("last_discussed_symbols", [])
            score += min(len(recent_symbols) / 5, 1.0) * 0.2
            
            # Recent topics (0.2 weight)
            recent_topics = context.get("recent_topics", [])
            score += min(len(recent_topics) / 3, 1.0) * 0.2
            
            # Total conversations (0.3 weight)
            total_convos = context.get("total_conversations", 0)
            score += min(total_convos / 20, 1.0) * 0.3
            
            # Relationship stage (0.3 weight)
            stage_scores = {
                "new": 0.2,
                "exploring": 0.4,
                "engaged": 0.6,
                "regular": 0.7,
                "trusted": 0.8,
                "established": 0.9,
                "highly_engaged": 1.0
            }
            stage = context.get("relationship_stage", "new")
            score += stage_scores.get(stage, 0.2) * 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.exception(f"❌ Error getting context quality: {e}")
            return 0.5

    async def _get_recent_inference_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get recent inference history for streak analysis"""
        try:
            history = await self.base.db.inference_history.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(limit).to_list(length=limit)
            
            return history
            
        except Exception as e:
            logger.exception(f"❌ Error getting recent inference history: {e}")
            return []

    def _count_low_confidence_streak(self, recent_inferences: List[Dict]) -> int:
        """Count consecutive low-confidence inferences"""
        try:
            streak = 0
            for inference in recent_inferences:
                avg_confidence = inference.get("average_confidence", 1.0)
                if avg_confidence < self._confidence_thresholds["medium"]:
                    streak += 1
                else:
                    break
            
            return streak
            
        except Exception as e:
            logger.exception(f"❌ Error counting confidence streak: {e}")
            return 0

    async def _detect_profile_contradictions(self, user: UserProfile, inference_results: Dict) -> List[Dict]:
        """Detect contradictions in inferred profile data"""
        try:
            contradictions = []
            
            # Age vs. career stage contradictions
            age_range = getattr(user, 'age_range', None)
            occupation = getattr(user, 'occupation', None)
            
            if age_range == "22-30" and occupation in ["senior", "executive"]:
                contradictions.append({
                    "type": "age_career_mismatch",
                    "details": f"Young age range ({age_range}) inconsistent with senior role ({occupation})",
                    "confidence": 0.8
                })
            
            # Income vs. portfolio size contradictions
            income_range = getattr(user, 'income_range', None)
            portfolio_size = getattr(user, 'portfolio_size', None)
            
            if income_range == "Under $50k" and portfolio_size == "very_large":
                contradictions.append({
                    "type": "income_portfolio_mismatch",
                    "details": f"Low income ({income_range}) inconsistent with large portfolio ({portfolio_size})",
                    "confidence": 0.7
                })
            
            # Risk tolerance vs. trading style contradictions
            risk_tolerance = getattr(user, 'risk_tolerance', None)
            trading_style = getattr(user, 'trading_style', None)
            
            if risk_tolerance == "conservative" and trading_style == "day_trading":
                contradictions.append({
                    "type": "risk_trading_mismatch",
                    "details": f"Conservative risk profile inconsistent with day trading style",
                    "confidence": 0.9
                })
            
            return contradictions
            
        except Exception as e:
            logger.exception(f"❌ Error detecting contradictions: {e}")
            return []

    async def _infer_investment_personality(self, conversations: List[Dict], user: UserProfile) -> Optional[Dict]:
        """NEW: Infer investment personality traits"""
        try:
            all_text = " ".join([
                conv.get("user_message", "") + " " + conv.get("bot_response", "")
                for conv in conversations
            ]).lower()
            
            personality_traits = {}
            
            # FOMO tendency
            fomo_indicators = ["missing out", "everyone else", "hot stock", "trending", "fear of missing"]
            fomo_score = sum(1 for indicator in fomo_indicators if indicator in all_text)
            if fomo_score > 0:
                personality_traits["fomo_tendency"] = min(fomo_score / 2.0, 1.0)
            
            # Research orientation
            research_indicators = ["research", "analysis", "study", "due diligence", "investigate"]
            research_score = sum(1 for indicator in research_indicators if indicator in all_text)
            if research_score > 0:
                personality_traits["research_orientation"] = min(research_score / 3.0, 1.0)
            
            # Emotional trading tendency
            emotion_indicators = ["panic", "excited", "nervous", "confident", "worried", "euphoric"]
            emotion_score = sum(1 for indicator in emotion_indicators if indicator in all_text)
            if emotion_score > 0:
                personality_traits["emotional_trading"] = min(emotion_score / 3.0, 1.0)
            
            if personality_traits:
                return {
                    "traits": personality_traits,
                    "confidence": 0.6,
                    "inferred_at": datetime.now(timezone.utc).isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.exception(f"❌ Error inferring investment personality: {e}")
            return None

    def _apply_personality_updates(self, user: UserProfile, results: Dict) -> UserProfile:
        """Apply investment personality results to user profile"""
        try:
            if results and results.get("confidence", 0) >= self._confidence_thresholds["low"]:
                user.investment_personality = results["traits"]
                logger.debug(f"✅ Updated investment personality: {results['traits']}")
            
            return user
            
        except Exception as e:
            logger.exception(f"❌ Error applying personality updates: {e}")
            return user

    async def _update_performance_metrics(self, user_id: str, result_type: str, quality_data: Dict):
        """Update inference performance tracking"""
        try:
            metric_doc = {
                "user_id": user_id,
                "result_type": result_type,
                "confidence_score": quality_data.get("confidence_score", 0.0),
                "quality_metrics": quality_data.get("quality_metrics", {}),
                "algorithm_version": "v2.0",
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.inference_performance.insert_one(metric_doc)
            
        except Exception as e:
            logger.exception(f"❌ Error updating performance metrics: {e}")

    async def _learn_from_rollback(self, user_id: str, inference_results: Dict, quality_assessment: Dict):
        """Learn from rollback to improve future inferences"""
        try:
            # Extract patterns that led to rollback
            failed_patterns = []
            
            for category, results in inference_results.get("results", {}).items():
                if isinstance(results, dict) and results.get("confidence", 0) < self._confidence_thresholds["medium"]:
                    failed_patterns.append({
                        "category": category,
                        "confidence": results.get("confidence"),
                        "pattern_type": results.get("pattern_type"),
                        "matches": results.get("matches", [])
                    })
            
            # Store learning data for algorithm improvement
            learning_doc = {
                "user_id": user_id,
                "rollback_reason": quality_assessment.get("rollback_reason"),
                "failed_patterns": failed_patterns,
                "algorithm_version": "v2.0",
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.inference_learning.insert_one(learning_doc)
            
            logger.debug(f"📚 Stored learning data from rollback for {user_id}")
            
        except Exception as e:
            logger.exception(f"❌ Error learning from rollback: {e}")

    def _serialize_user_profile(self, user: UserProfile) -> Dict[str, Any]:
        """Serialize user profile for versioning"""
        try:
            return {
                "phone_number": user.phone_number,
                "email": getattr(user, 'email', None),
                "first_name": getattr(user, 'first_name', None),
                "plan_type": getattr(user, 'plan_type', None),
                "age_range": getattr(user, 'age_range', None),
                "location": getattr(user, 'location', None),
                "occupation": getattr(user, 'occupation', None),
                "income_range": getattr(user, 'income_range', None),
                "risk_tolerance": getattr(user, 'risk_tolerance', None),
                "trading_style": getattr(user, 'trading_style', None),
                "investment_experience": getattr(user, 'investment_experience', None),
                "portfolio_size": getattr(user, 'portfolio_size', None),
                "communication_style": getattr(user, 'communication_style', None),
                "investment_personality": getattr(user, 'investment_personality', None),
                "life_events": getattr(user, 'life_events', None),
                "last_inference_at": getattr(user, 'last_inference_at', None),
                "_id": str(user._id) if user._id else None,
                "user_id": user.user_id
            }
        except Exception as e:
            logger.exception(f"❌ Error serializing user profile: {e}")
            return {}

    def _deserialize_user_profile(self, profile_data: Dict[str, Any]) -> UserProfile:
        """Deserialize user profile from version data"""
        try:
            # Create UserProfile instance from serialized data
            user = UserProfile(**profile_data)
            return user
        except Exception as e:
            logger.exception(f"❌ Error deserializing user profile: {e}")
            return None

    def _calculate_profile_hash(self, user: UserProfile) -> str:
        """Calculate hash of profile for change detection"""
        try:
            profile_str = str(sorted(self._serialize_user_profile(user).items()))
            return hashlib.md5(profile_str.encode()).hexdigest()
        except Exception as e:
            logger.exception(f"❌ Error calculating profile hash: {e}")
            return ""

    def _generate_version_id(self, user_id: str, version_number: int) -> str:
        """Generate unique version ID"""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return f"{user_id}_v{version_number}_{timestamp}"
        except Exception as e:
            logger.exception(f"❌ Error generating version ID: {e}")
            return f"{user_id}_v{version_number}_unknown"

    async def _cleanup_old_versions(self, user_id: str):
        """Clean up old profile versions"""
        try:
            # Keep last 10 versions
            versions = await self.base.db.profile_versions.find(
                {"user_id": user_id}
            ).sort("created_at", -1).to_list(length=None)
            
            if len(versions) > 10:
                old_versions = versions[10:]
                old_version_ids = [v["_id"] for v in old_versions]
                await self.base.db.profile_versions.delete_many({"_id": {"$in": old_version_ids}})
            
            # Also clean by date (older than retention period)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self._version_retention_days)
            await self.base.db.profile_versions.delete_many({
                "user_id": user_id,
                "created_at": {"$lt": cutoff_date}
            })
            
        except Exception as e:
            logger.exception(f"❌ Error cleaning up old versions: {e}")

    async def _log_enhanced_inference_results(self, user_id: str, inference_results: Dict, 
                                            quality_assessment: Dict, original_completeness: float, 
                                            new_completeness: float, pre_version_id: str, post_version_id: str):
        """Enhanced logging with versioning information"""
        try:
            log_doc = {
                "user_id": user_id,
                "inference_results": inference_results,
                "quality_assessment": quality_assessment,
                "original_completeness": original_completeness,
                "new_completeness": new_completeness,
                "improvement": new_completeness - original_completeness,
                "pre_version_id": pre_version_id,
                "post_version_id": post_version_id,
                "algorithm_version": "v2.0",
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.inference_history.insert_one(log_doc)
            
            # Also update performance metrics
            await self._update_performance_metrics(user_id, "success", quality_assessment)
            
        except Exception as e:
            logger.exception(f"❌ Error logging enhanced inference results: {e}")

    # Keep all existing methods from the original implementation...
    # (calculate_completeness, _should_run_inference, _get_conversation_history, etc.)
    # These remain unchanged but now work within the enhanced versioning system
