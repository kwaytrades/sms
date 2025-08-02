# services/inference/profile_inference.py
"""
Profile Inference Service - Auto-fill user demographics and financial profiles
Intelligent data collection without disrupting user experience
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
from loguru import logger

from models.user import UserProfile
from services.db.base_db_service import BaseDBService


class ProfileInferenceService:
    """
    Auto-inference service for user profile enhancement
    
    Features:
    - Demographic inference from conversation patterns
    - Financial profile building from trading behavior  
    - Goal extraction from natural language mentions
    - Life event detection and impact analysis
    - Confidence scoring for all inferred data
    - Progressive enhancement without user friction
    """
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self._inference_patterns = self._build_inference_patterns()
        self._confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }

    async def initialize(self):
        """Initialize inference service"""
        try:
            # Create inference tracking collection
            await self.base.db.inference_history.create_index("user_id")
            await self.base.db.inference_history.create_index("inference_type")
            await self.base.db.inference_history.create_index("created_at")
            
            logger.info("✅ Profile inference service initialized")
        except Exception as e:
            logger.exception(f"❌ Profile inference service initialization failed: {e}")

    async def enhance_user_profile(self, user: UserProfile) -> UserProfile:
        """
        Main enhancement method - analyzes conversation history and enhances profile
        """
        try:
            if not user or not user.phone_number:
                return user
            
            # Check if inference should run
            if not await self._should_run_inference(user):
                return user
            
            # Get conversation history for analysis
            conversations = await self._get_conversation_history(user.phone_number)
            if not conversations:
                return user
            
            # Calculate current profile completeness
            original_completeness = await self.calculate_completeness(user)
            
            # Run inference on different aspects
            enhanced_user = user
            inference_results = {}
            
            # Demographic inference
            demo_results = await self._infer_demographics(conversations, enhanced_user)
            if demo_results:
                enhanced_user = self._apply_demographic_updates(enhanced_user, demo_results)
                inference_results["demographics"] = demo_results
            
            # Financial profile inference
            financial_results = await self._infer_financial_profile(conversations, enhanced_user)
            if financial_results:
                enhanced_user = self._apply_financial_updates(enhanced_user, financial_results)
                inference_results["financial"] = financial_results
            
            # Goal extraction
            goal_results = await self._extract_goals(conversations, enhanced_user)
            if goal_results:
                inference_results["goals"] = goal_results
                # Goals are saved separately, not to user profile
                await self._save_inferred_goals(user.user_id or str(user._id), goal_results)
            
            # Life event detection
            event_results = await self._detect_life_events(conversations, enhanced_user)
            if event_results:
                enhanced_user = self._apply_event_updates(enhanced_user, event_results)
                inference_results["life_events"] = event_results
            
            # Communication style inference
            style_results = await self._infer_communication_style(conversations, enhanced_user)
            if style_results:
                enhanced_user = self._apply_style_updates(enhanced_user, style_results)
                inference_results["communication_style"] = style_results
            
            # Update inference metadata
            enhanced_user.last_inference_at = datetime.utcnow()
            new_completeness = await self.calculate_completeness(enhanced_user)
            
            # Log inference results
            if inference_results:
                await self._log_inference_results(
                    user.user_id or str(user._id), 
                    inference_results, 
                    original_completeness, 
                    new_completeness
                )
                
                logger.info(f"✅ Enhanced user profile: {original_completeness:.2f} → {new_completeness:.2f} completeness")
            
            return enhanced_user
            
        except Exception as e:
            logger.exception(f"❌ Error enhancing user profile: {e}")
            return user

    async def calculate_completeness(self, user: UserProfile) -> float:
        """Calculate profile completeness score (0.0 to 1.0)"""
        try:
            if not user:
                return 0.0
            
            total_fields = 0
            completed_fields = 0
            
            # Core fields (weight: 2)
            core_fields = ['phone_number', 'email', 'first_name', 'plan_type']
            for field in core_fields:
                total_fields += 2
                if hasattr(user, field) and getattr(user, field):
                    completed_fields += 2
            
            # Demographic fields (weight: 1)
            demo_fields = ['age_range', 'location', 'occupation', 'income_range']
            for field in demo_fields:
                total_fields += 1
                if hasattr(user, field) and getattr(user, field):
                    completed_fields += 1
            
            # Financial fields (weight: 1.5)
            financial_fields = ['risk_tolerance', 'trading_style', 'investment_experience', 'portfolio_size']
            for field in financial_fields:
                total_fields += 1.5
                if hasattr(user, field) and getattr(user, field):
                    completed_fields += 1.5
            
            # Preference fields (weight: 0.5)
            pref_fields = ['timezone', 'preferred_contact_method', 'notification_preferences']
            for field in pref_fields:
                total_fields += 0.5
                if hasattr(user, field) and getattr(user, field):
                    completed_fields += 0.5
            
            return min(completed_fields / total_fields, 1.0) if total_fields > 0 else 0.0
            
        except Exception as e:
            logger.exception(f"❌ Error calculating completeness: {e}")
            return 0.0

    async def _should_run_inference(self, user: UserProfile) -> bool:
        """Determine if inference should run"""
        try:
            # Check time since last inference
            if hasattr(user, 'last_inference_at') and user.last_inference_at:
                time_diff = datetime.utcnow() - user.last_inference_at
                if time_diff < timedelta(hours=24):
                    return False
            
            # Check profile completeness
            completeness = await self.calculate_completeness(user)
            if completeness > 0.9:  # Already very complete
                return False
            
            # Check if user has enough conversation history
            conversations = await self._get_conversation_history(user.phone_number, limit=5)
            if len(conversations) < 3:
                return False
            
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error checking inference requirements: {e}")
            return False

    async def _get_conversation_history(self, phone_number: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for analysis"""
        try:
            # Get recent conversations
            conversations = await self.base.db.enhanced_conversations.find(
                {"phone_number": phone_number}
            ).sort("timestamp", -1).limit(limit).to_list(length=limit)
            
            return conversations
            
        except Exception as e:
            logger.exception(f"❌ Error getting conversation history: {e}")
            return []

    def _build_inference_patterns(self) -> Dict[str, Dict]:
        """Build pattern matching rules for inference"""
        return {
            "age_patterns": {
                "young_adult": {
                    "keywords": ["college", "university", "student", "graduation", "first job", "entry level"],
                    "age_range": "22-30",
                    "confidence": 0.7
                },
                "mid_career": {
                    "keywords": ["promotion", "manager", "director", "mortgage", "kids", "family"],
                    "age_range": "30-45",
                    "confidence": 0.6
                },
                "experienced": {
                    "keywords": ["retirement", "401k", "senior", "executive", "grandkids", "medicare"],
                    "age_range": "45-65",
                    "confidence": 0.7
                },
                "retired": {
                    "keywords": ["retired", "pension", "social security", "medicare", "fixed income"],
                    "age_range": "65+",
                    "confidence": 0.8
                }
            },
            "income_patterns": {
                "low": {
                    "keywords": ["tight budget", "can't afford", "cheap", "budget", "paycheck to paycheck"],
                    "phrases": ["money is tight", "on a budget", "need to save"],
                    "income_range": "Under $50k",
                    "confidence": 0.6
                },
                "medium": {
                    "keywords": ["comfortable", "moderate", "steady income", "middle class"],
                    "phrases": ["doing okay", "comfortable financially"],
                    "income_range": "$50k-$100k",
                    "confidence": 0.5
                },
                "high": {
                    "keywords": ["high income", "well off", "luxury", "expensive", "portfolio"],
                    "phrases": ["money isn't an issue", "can afford"],
                    "income_range": "$100k+",
                    "confidence": 0.7
                }
            },
            "occupation_patterns": {
                "tech": {
                    "keywords": ["software", "developer", "engineer", "tech", "startup", "coding", "programming"],
                    "confidence": 0.8
                },
                "finance": {
                    "keywords": ["banking", "finance", "accounting", "analyst", "advisor", "wall street"],
                    "confidence": 0.8
                },
                "healthcare": {
                    "keywords": ["doctor", "nurse", "medical", "hospital", "healthcare", "physician"],
                    "confidence": 0.8
                },
                "education": {
                    "keywords": ["teacher", "professor", "education", "school", "university", "academic"],
                    "confidence": 0.7
                },
                "business": {
                    "keywords": ["manager", "director", "executive", "business", "sales", "marketing"],
                    "confidence": 0.6
                }
            },
            "risk_tolerance_patterns": {
                "conservative": {
                    "keywords": ["safe", "secure", "low risk", "conservative", "stable", "guaranteed"],
                    "phrases": ["don't want to lose money", "play it safe", "risk averse"],
                    "confidence": 0.8
                },
                "moderate": {
                    "keywords": ["balanced", "moderate", "some risk", "diversified"],
                    "phrases": ["balanced approach", "moderate risk"],
                    "confidence": 0.6
                },
                "aggressive": {
                    "keywords": ["aggressive", "high risk", "growth", "speculation", "volatile"],
                    "phrases": ["willing to take risks", "high growth potential"],
                    "confidence": 0.8
                }
            },
            "trading_style_patterns": {
                "buy_and_hold": {
                    "keywords": ["long term", "hold", "years", "decades", "retirement"],
                    "phrases": ["buy and hold", "long term investor"],
                    "confidence": 0.7
                },
                "swing_trading": {
                    "keywords": ["swing", "weeks", "months", "medium term", "trend"],
                    "phrases": ["swing trading", "hold for weeks"],
                    "confidence": 0.8
                },
                "day_trading": {
                    "keywords": ["day trading", "daily", "intraday", "scalping", "quick"],
                    "phrases": ["trade daily", "in and out same day"],
                    "confidence": 0.9
                }
            },
            "goal_patterns": {
                "retirement": {
                    "keywords": ["retirement", "retire", "401k", "ira", "pension", "nest egg"],
                    "phrases": ["save for retirement", "retirement fund"],
                    "goal_type": "retirement",
                    "confidence": 0.8
                },
                "emergency_fund": {
                    "keywords": ["emergency", "safety net", "backup", "rainy day"],
                    "phrases": ["emergency fund", "safety net", "rainy day fund"],
                    "goal_type": "emergency",
                    "confidence": 0.8
                },
                "home_purchase": {
                    "keywords": ["house", "home", "mortgage", "down payment", "property"],
                    "phrases": ["buy a house", "down payment", "first home"],
                    "goal_type": "home_purchase",
                    "confidence": 0.7
                },
                "education": {
                    "keywords": ["college", "education", "tuition", "school", "degree"],
                    "phrases": ["pay for college", "education fund"],
                    "goal_type": "education",
                    "confidence": 0.7
                }
            }
        }

    async def _infer_demographics(self, conversations: List[Dict], user: UserProfile) -> Dict[str, Any]:
        """Infer demographic information from conversations"""
        try:
            results = {}
            all_text = " ".join([
                conv.get("user_message", "") + " " + conv.get("bot_response", "")
                for conv in conversations
            ]).lower()
            
            # Age inference
            if not getattr(user, 'age_range', None):
                age_result = self._match_patterns(all_text, self._inference_patterns["age_patterns"])
                if age_result:
                    results["age_range"] = age_result
            
            # Income inference
            if not getattr(user, 'income_range', None):
                income_result = self._match_patterns(all_text, self._inference_patterns["income_patterns"])
                if income_result:
                    results["income_range"] = income_result
            
            # Occupation inference
            if not getattr(user, 'occupation', None):
                occupation_result = self._match_patterns(all_text, self._inference_patterns["occupation_patterns"])
                if occupation_result:
                    results["occupation"] = occupation_result
            
            return results if results else None
            
        except Exception as e:
            logger.exception(f"❌ Error inferring demographics: {e}")
            return None

    async def _infer_financial_profile(self, conversations: List[Dict], user: UserProfile) -> Dict[str, Any]:
        """Infer financial profile from trading behavior and conversations"""
        try:
            results = {}
            all_text = " ".join([
                conv.get("user_message", "") + " " + conv.get("bot_response", "")
                for conv in conversations
            ]).lower()
            
            # Risk tolerance inference
            if not getattr(user, 'risk_tolerance', None):
                risk_result = self._match_patterns(all_text, self._inference_patterns["risk_tolerance_patterns"])
                if risk_result:
                    results["risk_tolerance"] = risk_result
            
            # Trading style inference
            if not getattr(user, 'trading_style', None):
                style_result = self._match_patterns(all_text, self._inference_patterns["trading_style_patterns"])
                if style_result:
                    results["trading_style"] = style_result
            
            # Investment experience inference
            if not getattr(user, 'investment_experience', None):
                experience_level = self._infer_experience_level(conversations)
                if experience_level:
                    results["investment_experience"] = experience_level
            
            # Portfolio size estimation
            if not getattr(user, 'portfolio_size', None):
                portfolio_estimate = self._estimate_portfolio_size(all_text)
                if portfolio_estimate:
                    results["portfolio_size"] = portfolio_estimate
            
            return results if results else None
            
        except Exception as e:
            logger.exception(f"❌ Error inferring financial profile: {e}")
            return None

    async def _extract_goals(self, conversations: List[Dict], user: UserProfile) -> List[Dict]:
        """Extract financial goals from natural language mentions"""
        try:
            goals = []
            all_text = " ".join([
                conv.get("user_message", "")
                for conv in conversations
            ]).lower()
            
            # Match goal patterns
            for goal_type, pattern in self._inference_patterns["goal_patterns"].items():
                matches = self._find_goal_mentions(all_text, pattern)
                for match in matches:
                    goal = {
                        "title": match.get("title", f"{goal_type.replace('_', ' ').title()} Goal"),
                        "goal_type": pattern["goal_type"],
                        "confidence": match["confidence"],
                        "context": match.get("context", ""),
                        "inferred_at": datetime.now(timezone.utc).isoformat(),
                        "source": "conversation_analysis"
                    }
                    
                    # Try to extract target amount or timeline
                    amount_match = self._extract_amount_from_context(match.get("context", ""))
                    if amount_match:
                        goal["target_amount"] = amount_match
                    
                    timeline_match = self._extract_timeline_from_context(match.get("context", ""))
                    if timeline_match:
                        goal["target_date"] = timeline_match
                    
                    goals.append(goal)
            
            return goals if goals else None
            
        except Exception as e:
            logger.exception(f"❌ Error extracting goals: {e}")
            return None

    async def _detect_life_events(self, conversations: List[Dict], user: UserProfile) -> Dict[str, Any]:
        """Detect life events that might affect investment strategy"""
        try:
            results = {}
            all_text = " ".join([
                conv.get("user_message", "")
                for conv in conversations
            ]).lower()
            
            life_events = {
                "job_change": {
                    "keywords": ["new job", "started work", "promotion", "raise", "layoff", "unemployed"],
                    "impact": "income_change"
                },
                "marriage": {
                    "keywords": ["married", "wedding", "spouse", "husband", "wife"],
                    "impact": "financial_responsibility_change"
                },
                "children": {
                    "keywords": ["baby", "pregnant", "child", "kids", "daughter", "son"],
                    "impact": "expense_increase"
                },
                "home_purchase": {
                    "keywords": ["bought house", "new home", "mortgage", "moved"],
                    "impact": "major_expense"
                },
                "retirement": {
                    "keywords": ["retired", "retirement", "stop working"],
                    "impact": "income_reduction"
                }
            }
            
            detected_events = []
            for event_type, pattern in life_events.items():
                if any(keyword in all_text for keyword in pattern["keywords"]):
                    detected_events.append({
                        "event_type": event_type,
                        "impact": pattern["impact"],
                        "confidence": 0.6,
                        "detected_at": datetime.now(timezone.utc).isoformat()
                    })
            
            if detected_events:
                results["life_events"] = detected_events
            
            return results if results else None
            
        except Exception as e:
            logger.exception(f"❌ Error detecting life events: {e}")
            return None

    async def _infer_communication_style(self, conversations: List[Dict], user: UserProfile) -> Dict[str, Any]:
        """Infer communication preferences and style"""
        try:
            results = {}
            user_messages = [conv.get("user_message", "") for conv in conversations if conv.get("user_message")]
            
            if not user_messages:
                return None
            
            # Analyze message characteristics
            avg_length = sum(len(msg.split()) for msg in user_messages) / len(user_messages)
            total_text = " ".join(user_messages).lower()
            
            # Communication style analysis
            style = {}
            
            # Formality level
            formal_indicators = ["please", "thank you", "could you", "would you", "appreciate"]
            casual_indicators = ["hey", "hi", "thanks", "thx", "cool", "awesome"]
            
            formal_score = sum(1 for indicator in formal_indicators if indicator in total_text)
            casual_score = sum(1 for indicator in casual_indicators if indicator in total_text)
            
            if formal_score > casual_score:
                style["formality"] = "formal"
            elif casual_score > formal_score:
                style["formality"] = "casual"
            else:
                style["formality"] = "neutral"
            
            # Detail preference
            if avg_length > 15:
                style["detail_preference"] = "detailed"
            elif avg_length < 5:
                style["detail_preference"] = "brief"
            else:
                style["detail_preference"] = "moderate"
            
            # Response urgency
            urgent_indicators = ["urgent", "asap", "quickly", "fast", "hurry", "immediate"]
            if any(indicator in total_text for indicator in urgent_indicators):
                style["urgency_preference"] = "high"
            else:
                style["urgency_preference"] = "normal"
            
            # Question style
            question_count = sum(msg.count('?') for msg in user_messages)
            if question_count / len(user_messages) > 1.5:
                style["question_style"] = "inquisitive"
            else:
                style["question_style"] = "direct"
            
            results["communication_style"] = {
                "style": style,
                "confidence": 0.6,
                "message_count": len(user_messages),
                "avg_message_length": avg_length
            }
            
            return results if results else None
            
        except Exception as e:
            logger.exception(f"❌ Error inferring communication style: {e}")
            return None

    def _match_patterns(self, text: str, patterns: Dict) -> Optional[Dict]:
        """Match text against inference patterns"""
        try:
            best_match = None
            best_score = 0
            
            for category, pattern in patterns.items():
                score = 0
                matches = []
                
                # Check keywords
                keywords = pattern.get("keywords", [])
                for keyword in keywords:
                    if keyword in text:
                        score += 1
                        matches.append(keyword)
                
                # Check phrases
                phrases = pattern.get("phrases", [])
                for phrase in phrases:
                    if phrase in text:
                        score += 2  # Phrases are weighted higher
                        matches.append(phrase)
                
                # Calculate confidence based on matches and pattern confidence
                if score > 0:
                    base_confidence = pattern.get("confidence", 0.5)
                    match_confidence = min(score * 0.1, 0.4)  # Up to 0.4 boost from matches
                    total_confidence = min(base_confidence + match_confidence, 1.0)
                    
                    if total_confidence > best_score and total_confidence >= self._confidence_thresholds["low"]:
                        best_score = total_confidence
                        best_match = {
                            "category": category,
                            "value": pattern.get(category.split("_")[0] + "_range", pattern.get("value", category)),
                            "confidence": total_confidence,
                            "matches": matches,
                            "pattern_type": category
                        }
            
            return best_match
            
        except Exception as e:
            logger.exception(f"❌ Error matching patterns: {e}")
            return None

    def _infer_experience_level(self, conversations: List[Dict]) -> Optional[Dict]:
        """Infer investment experience level from conversation complexity"""
        try:
            all_text = " ".join([
                conv.get("user_message", "") + " " + conv.get("bot_response", "")
                for conv in conversations
            ]).lower()
            
            # Advanced terms indicate higher experience
            advanced_terms = [
                "options", "derivatives", "volatility", "beta", "alpha", "sharpe ratio",
                "diversification", "correlation", "portfolio theory", "hedge",
                "futures", "commodities", "forex", "technical analysis", "fundamental analysis"
            ]
            
            # Beginner terms indicate lower experience
            beginner_terms = [
                "what is", "how do i", "new to", "beginner", "start investing",
                "first time", "don't understand", "explain", "simple terms"
            ]
            
            advanced_count = sum(1 for term in advanced_terms if term in all_text)
            beginner_count = sum(1 for term in beginner_terms if term in all_text)
            
            if advanced_count > beginner_count and advanced_count >= 3:
                experience = "experienced"
                confidence = min(0.6 + (advanced_count * 0.1), 0.9)
            elif beginner_count > advanced_count and beginner_count >= 2:
                experience = "beginner"
                confidence = min(0.6 + (beginner_count * 0.1), 0.9)
            else:
                experience = "intermediate"
                confidence = 0.5
            
            return {
                "value": experience,
                "confidence": confidence,
                "advanced_terms_found": advanced_count,
                "beginner_terms_found": beginner_count
            }
            
        except Exception as e:
            logger.exception(f"❌ Error inferring experience level: {e}")
            return None

    def _estimate_portfolio_size(self, text: str) -> Optional[Dict]:
        """Estimate portfolio size from conversation context"""
        try:
            # Look for portfolio value mentions
            amount_patterns = [
                r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $1,000.00 format
                r'(\d+)k',  # 100k format
                r'(\d+) thousand',  # 100 thousand format
                r'(\d+) million',  # 1 million format
            ]
            
            amounts = []
            for pattern in amount_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        if 'k' in pattern:
                            amounts.append(float(match) * 1000)
                        elif 'thousand' in pattern:
                            amounts.append(float(match) * 1000)
                        elif 'million' in pattern:
                            amounts.append(float(match) * 1000000)
                        else:
                            amount_str = match.replace(',', '')
                            amounts.append(float(amount_str))
                    except ValueError:
                        continue
            
            if amounts:
                # Use the largest reasonable amount (likely portfolio value)
                reasonable_amounts = [a for a in amounts if 1000 <= a <= 10000000]  # $1k to $10M
                if reasonable_amounts:
                    estimated_value = max(reasonable_amounts)
                    
                    # Categorize portfolio size
                    if estimated_value < 10000:
                        size_category = "small"
                    elif estimated_value < 100000:
                        size_category = "medium"
                    elif estimated_value < 1000000:
                        size_category = "large"
                    else:
                        size_category = "very_large"
                    
                    return {
                        "category": size_category,
                        "estimated_value": estimated_value,
                        "confidence": 0.7
                    }
            
            return None
            
        except Exception as e:
            logger.exception(f"❌ Error estimating portfolio size: {e}")
            return None

    def _find_goal_mentions(self, text: str, pattern: Dict) -> List[Dict]:
        """Find goal mentions in text using patterns"""
        try:
            matches = []
            keywords = pattern.get("keywords", [])
            phrases = pattern.get("phrases", [])
            
            # Look for goal mentions with context
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip().lower()
                
                # Check if this sentence mentions the goal
                goal_mentioned = False
                for keyword in keywords:
                    if keyword in sentence:
                        goal_mentioned = True
                        break
                
                if not goal_mentioned:
                    for phrase in phrases:
                        if phrase in sentence:
                            goal_mentioned = True
                            break
                
                if goal_mentioned:
                    matches.append({
                        "context": sentence,
                        "confidence": pattern.get("confidence", 0.6)
                    })
            
            return matches
            
        except Exception as e:
            logger.exception(f"❌ Error finding goal mentions: {e}")
            return []

    def _extract_amount_from_context(self, context: str) -> Optional[float]:
        """Extract monetary amount from context"""
        try:
            # Amount patterns
            patterns = [
                r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+)k',
                r'(\d+) thousand',
                r'(\d+) million'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, context.lower())
                if match:
                    amount_str = match.group(1)
                    if 'k' in pattern:
                        return float(amount_str) * 1000
                    elif 'thousand' in pattern:
                        return float(amount_str) * 1000
                    elif 'million' in pattern:
                        return float(amount_str) * 1000000
                    else:
                        return float(amount_str.replace(',', ''))
            
            return None
            
        except Exception as e:
            logger.exception(f"❌ Error extracting amount: {e}")
            return None

    def _extract_timeline_from_context(self, context: str) -> Optional[str]:
        """Extract timeline from context"""
        try:
            # Timeline patterns
            timeline_patterns = {
                r'(\d+) years?': lambda x: f"{datetime.now().year + int(x)}-12-31",
                r'(\d+) months?': lambda x: (datetime.now() + timedelta(days=int(x)*30)).strftime("%Y-%m-%d"),
                r'by (\d{4})': lambda x: f"{x}-12-31",
                r'in (\d{4})': lambda x: f"{x}-12-31",
                r'retirement': lambda x: f"{datetime.now().year + 30}-12-31"  # Assume 30 years to retirement
            }
            
            for pattern, date_func in timeline_patterns.items():
                match = re.search(pattern, context.lower())
                if match:
                    if pattern == r'retirement':
                        return date_func(None)
                    else:
                        return date_func(match.group(1))
            
            return None
            
        except Exception as e:
            logger.exception(f"❌ Error extracting timeline: {e}")
            return None

    def _apply_demographic_updates(self, user: UserProfile, results: Dict) -> UserProfile:
        """Apply demographic inference results to user profile"""
        try:
            for field, result in results.items():
                if result["confidence"] >= self._confidence_thresholds["low"]:
                    setattr(user, field, result["value"])
                    logger.debug(f"✅ Updated {field}: {result['value']} (confidence: {result['confidence']:.2f})")
            
            return user
            
        except Exception as e:
            logger.exception(f"❌ Error applying demographic updates: {e}")
            return user

    def _apply_financial_updates(self, user: UserProfile, results: Dict) -> UserProfile:
        """Apply financial inference results to user profile"""
        try:
            for field, result in results.items():
                if result["confidence"] >= self._confidence_thresholds["low"]:
                    setattr(user, field, result["value"])
                    logger.debug(f"✅ Updated {field}: {result['value']} (confidence: {result['confidence']:.2f})")
            
            return user
            
        except Exception as e:
            logger.exception(f"❌ Error applying financial updates: {e}")
            return user

    def _apply_event_updates(self, user: UserProfile, results: Dict) -> UserProfile:
        """Apply life event results to user profile"""
        try:
            life_events = results.get("life_events", [])
            if life_events:
                # Store life events in user metadata or separate field
                if not hasattr(user, 'life_events'):
                    user.life_events = []
                user.life_events.extend(life_events)
                logger.debug(f"✅ Added {len(life_events)} life events to profile")
            
            return user
            
        except Exception as e:
            logger.exception(f"❌ Error applying event updates: {e}")
            return user

    def _apply_style_updates(self, user: UserProfile, results: Dict) -> UserProfile:
        """Apply communication style results to user profile"""
        try:
            style_data = results.get("communication_style", {})
            if style_data and style_data.get("confidence", 0) >= self._confidence_thresholds["low"]:
                user.communication_style = style_data["style"]
                logger.debug(f"✅ Updated communication style: {style_data['style']}")
            
            return user
            
        except Exception as e:
            logger.exception(f"❌ Error applying style updates: {e}")
            return user

    async def _save_inferred_goals(self, user_id: str, goals: List[Dict]) -> bool:
        """Save inferred goals to the database"""
        try:
            for goal in goals:
                if goal.get("confidence", 0) >= self._confidence_thresholds["medium"]:
                    # Check if similar goal already exists
                    existing_goal = await self.base.db.financial_goals.find_one({
                        "user_id": user_id,
                        "goal_type": goal["goal_type"]
                    })
                    
                    if not existing_goal:
                        goal_doc = {
                            "user_id": user_id,
                            "title": goal["title"],
                            "goal_type": goal["goal_type"],
                            "status": "inferred",
                            "target_amount": goal.get("target_amount"),
                            "target_date": goal.get("target_date"),
                            "confidence": goal["confidence"],
                            "source": goal["source"],
                            "created_at": datetime.now(timezone.utc),
                            "updated_at": datetime.now(timezone.utc)
                        }
                        
                        await self.base.db.financial_goals.insert_one(goal_doc)
                        logger.info(f"✅ Saved inferred goal: {goal['title']}")
            
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error saving inferred goals: {e}")
            return False

    async def _log_inference_results(self, user_id: str, results: Dict, 
                                   original_completeness: float, new_completeness: float):
        """Log inference results for tracking and improvement"""
        try:
            log_doc = {
                "user_id": user_id,
                "inference_results": results,
                "original_completeness": original_completeness,
                "new_completeness": new_completeness,
                "improvement": new_completeness - original_completeness,
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.inference_history.insert_one(log_doc)
            
        except Exception as e:
            logger.exception(f"❌ Error logging inference results: {e}")
