# social/services/social_trading_agent.py - SEMANTIC LLM-POWERED VERSION
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class SocialTradingAgent:
    """
    FULLY SEMANTIC social media trading agent
    Gemini for intent parsing → ToolExecutor → Claude for response generation
    ALL extraction and analysis now LLM-powered instead of keyword-based
    """
    
    def __init__(self, gemini_client, claude_client, social_memory_service):
        self.gemini_client = gemini_client
        self.claude_client = claude_client
        self.memory_service = social_memory_service
        
        # Intent classification system - same pattern as SMS bot
        self.social_intents = {
            "user_interaction": ["comment", "dm", "reply", "question"],
            "competitor_engagement": ["competitor_comment", "competitor_content"],
            "hashtag_engagement": ["hashtag_content", "trending_topic"],
            "content_monitoring": ["brand_mention", "stock_discussion"],
            "conversion_opportunity": ["trial_request", "pricing_question", "how_it_works"],
            "relationship_building": ["returning_user", "high_value_user", "influencer"],
            "competitive_analysis": ["competitor_weakness", "audience_education"],
            "help": ["support_request", "technical_question"],
            "general": ["generic_comment", "off_topic"]
        }
        
        # Response strategies - mirroring SMS personality system
        self.response_strategies = {
            "educational_onboarding": "New user education and trial conversion",
            "relationship_building": "Existing user appreciation and engagement",
            "influencer_partnership": "High-value user collaboration approach",
            "competitive_advantage": "Highlight unique value vs competitors",
            "skeptic_conversion": "Address doubts with proof and transparency",
            "community_building": "Foster engagement and belonging",
            "value_demonstration": "Show bot capabilities and results",
            "casual_engagement": "Light, friendly interaction"
        }
    
    async def parse_social_intent(self, content: str, username: str, platform: str,
                                interaction_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        COMPLETE SEMANTIC INTEGRATION - LLM results feed directly into user profile
        """
        try:
            # Get user for profile integration
            from ..models.social_user import PlatformType
            user = await self.memory_service.get_or_create_user(username, PlatformType(platform))
            
            # Get FULL user context with actual message history
            user_context = await self.memory_service.get_conversation_context(f"{platform}_{username}")
            
            # LLM SEMANTIC EXTRACTION with user history context
            symbols, topics = await self._extract_symbols_and_topics_with_history(content, user.interaction_history)
            
            # LLM ENGAGEMENT STYLE CLASSIFICATION (replaces keyword analysis)
            engagement_style = await self._classify_engagement_style_llm(content, user.interaction_history)
            
            # Build enhanced Gemini prompt with ACTUAL user messages
            prompt = self._build_enhanced_intent_prompt(
                content, username, platform, interaction_type, user_context, context, symbols, topics
            )
            
            # Call Gemini for intent classification
            response = await self._call_gemini_intent(prompt)
            intent_data = self._parse_gemini_response(response)
            
            # LLM VALIDATION LOOP
            validated_intent = await self._validate_intent_classification(content, intent_data)
            
            # Build final intent object with ALL semantic extractions
            final_intent = {
                "intent": validated_intent.get("primary_intent", "general"),
                "sub_intent": validated_intent.get("sub_intent", ""),
                "symbols": symbols,  # LLM-extracted
                "topics": topics,    # LLM-extracted
                "sentiment": validated_intent.get("sentiment", "neutral"),
                "urgency": validated_intent.get("urgency", "normal"),
                "user_type": validated_intent.get("user_type", "regular"),
                "requires_tools": validated_intent.get("requires_tools", []),
                "response_strategy": validated_intent.get("response_strategy", "casual_engagement"),
                "priority": validated_intent.get("priority", "medium"),
                "competitive_context": validated_intent.get("competitive_context", False),
                "conversion_potential": validated_intent.get("conversion_potential", "low"),
                "confidence": validated_intent.get("confidence", 0.8),
                "engagement_style_detected": engagement_style,  # LLM-classified
                "user_journey_stage": validated_intent.get("user_journey_stage", "discovery"),
                "timestamp": datetime.utcnow().isoformat(),
                "content": content  # Store for analysis
            }
            
            # DIRECT INTEGRATION: Feed LLM results into user profile
            await self._integrate_llm_results_into_user_profile(user, final_intent, symbols, topics, engagement_style)
            
            return final_intent
            
        except Exception as e:
            logger.error(f"Failed to parse social intent: {e}")
            return self._get_fallback_intent(content, interaction_type)

    async def _extract_symbols_and_topics_with_history(self, content: str, 
                                                     interaction_history: List) -> Tuple[List[str], List[str]]:
        """
        LLM extraction with user history context for better accuracy
        """
        try:
            # Get recent user messages for context
            recent_messages = [i.content for i in interaction_history[-5:]] if interaction_history else []
            history_context = " | ".join(recent_messages) if recent_messages else "No previous context"
            
            prompt = f"""
You are an expert at extracting trading information from social media conversations.

CURRENT MESSAGE: "{content}"

RECENT USER CONTEXT: {history_context}

Extract stock symbols and trading topics from the CURRENT MESSAGE, using the context to resolve ambiguities.

STOCK SYMBOLS (return actual tickers):
- Direct: AAPL, TSLA, BTC, NVDA, etc.
- Companies: "Apple" → AAPL, "Tesla" → TSLA, "Microsoft" → MSFT
- Crypto: "Bitcoin" → BTC, "Ethereum" → ETH
- ETFs: "S&P 500" → SPY, "Nasdaq" → QQQ
- Context-dependent: Use history to resolve "CAT" (Caterpillar vs cat animal)

TRADING TOPICS:
- technical_analysis: Charts, RSI, MACD, support/resistance, breakouts
- options: Calls, puts, strikes, IV, Greeks
- earnings: Reports, guidance, beats/misses
- crypto: Bitcoin, DeFi, altcoins, blockchain
- day_trading: Scalping, intraday, quick trades
- swing_trading: Multi-day holds, momentum
- news_trading: Catalysts, events, announcements
- ai_trading: Bots, algorithms, automation
- market_psychology: Sentiment, FOMO, fear
- risk_management: Stops, position sizing
- fundamental_analysis: P/E, revenue, growth
- sector_rotation: Industry moves, themes

Return JSON with actual symbols/topics found (empty arrays if none):
{{
    "symbols": ["AAPL", "TSLA"],
    "topics": ["technical_analysis", "earnings"],
    "confidence": 0.9,
    "reasoning": "User mentioned Apple earnings and RSI levels"
}}
"""
            
            response = await self.gemini_client.generate_content_async(prompt)
            extraction_data = self._parse_extraction_response(response.text)
            
            symbols = extraction_data.get("symbols", [])
            topics = extraction_data.get("topics", [])
            
            logger.info(f"LLM extraction with history - Symbols: {symbols}, Topics: {topics}")
            return symbols, topics
            
        except Exception as e:
            logger.error(f"Semantic extraction with history failed: {e}")
            return self._extract_symbols_fallback(content), self._extract_trading_topics_fallback(content)

    async def _classify_engagement_style_llm(self, content: str, interaction_history: List) -> str:
        """
        LLM-based engagement style classification (replaces keyword analysis)
        """
        try:
            # Get last 10 interactions for pattern analysis
            recent_interactions = interaction_history[-10:] if interaction_history else []
            
            if len(recent_interactions) < 3:
                # Not enough history, analyze current message only
                history_text = "Insufficient interaction history - analyze current message only"
            else:
                history_text = "\n".join([
                    f"- \"{interaction.content}\"" 
                    for interaction in recent_interactions
                ])
            
            prompt = f"""
Classify this user's engagement style based on their interaction pattern.

CURRENT MESSAGE: "{content}"

INTERACTION HISTORY:
{history_text}

ENGAGEMENT STYLES:
- question_asker: Frequently asks questions, seeks information and clarification
- advice_giver: Offers opinions, shares analysis, helps others, provides insights
- skeptic: Questions claims, doubts results, critical thinking, demands proof
- supporter: Positive about content, shares wins, encourages others, builds community
- lurker: Minimal engagement, short responses, observes more than participates
- technical_analyst: Uses trading terminology, discusses charts/indicators, technical focus
- influencer: High engagement, opinion leader, others respond to their comments

ANALYSIS CRITERIA:
- Question frequency and type
- Tone and language patterns
- Technical knowledge demonstrated
- Supportiveness vs skepticism
- Interaction depth and length
- Leadership vs following behavior

Return JSON:
{{
    "engagement_style": "question_asker",
    "confidence": 0.8,
    "reasoning": "User asks multiple clarifying questions and seeks explanations",
    "secondary_style": "technical_analyst",
    "style_evolution": "stable" or "evolving_from_X_to_Y"
}}
"""
            
            response = await self.gemini_client.generate_content_async(prompt)
            style_data = self._parse_gemini_response(response.text)
            
            engagement_style = style_data.get("engagement_style", "lurker")
            confidence = style_data.get("confidence", 0.5)
            
            logger.info(f"LLM classified engagement style: {engagement_style} (confidence: {confidence})")
            return engagement_style
            
        except Exception as e:
            logger.error(f"LLM engagement style classification failed: {e}")
            return "lurker"  # Safe fallback

    async def _integrate_llm_results_into_user_profile(self, user, intent_data: Dict, 
                                                     symbols: List[str], topics: List[str], 
                                                     engagement_style: str):
        """
        CONFIDENCE-FILTERED LLM integration with frequency tracking
        """
        try:
            confidence = intent_data.get("confidence", 0.0)
            
            # CONFIDENCE-BASED FILTERING
            if confidence < user.llm_confidence_threshold:
                logger.info(f"Skipping LLM update for @{user.primary_username}: confidence {confidence} below threshold {user.llm_confidence_threshold}")
                return
            
            # 1. STORE INTENT CLASSIFICATION (only high confidence)
            classification_record = {
                "timestamp": intent_data.get("timestamp"),
                "content": intent_data.get("content", ""),
                "intent": intent_data.get("intent"),
                "sentiment": intent_data.get("sentiment"),
                "engagement_style": engagement_style,
                "journey_stage": intent_data.get("user_journey_stage"),
                "conversion_potential": intent_data.get("conversion_potential"),
                "confidence": confidence,
                "symbols_mentioned": symbols,
                "topics_discussed": topics,
                "filtered": False  # Passed confidence filter
            }
            
            if not hasattr(user, 'intent_classifications'):
                user.intent_classifications = []
            user.intent_classifications.append(classification_record)
            
            # Keep only last 20 classifications
            if len(user.intent_classifications) > 20:
                user.intent_classifications = user.intent_classifications[-20:]
            
            # 2. UPDATE ENGAGEMENT STYLE (confidence-based)
            await self._update_engagement_style_from_llm_with_confidence(user, engagement_style, confidence)
            
            # 3. UPDATE JOURNEY PROGRESSION (confidence-based)
            await self._update_journey_progression_from_llm_with_confidence(user, intent_data.get("user_journey_stage"), confidence)
            
            # 4. FREQUENCY-WEIGHTED TRADING INTERESTS UPDATE
            user.update_trading_interests_with_frequency(symbols, topics, confidence)
            
            # 5. UPDATE COMMUNICATION STYLE
            await self._update_communication_style_from_llm(user, intent_data.get("content", ""), intent_data)
            
            # Save all updates
            await self.memory_service._save_user_updates(user)
            
            logger.info(f"LLM integration complete for @{user.primary_username} (confidence: {confidence})")
            
        except Exception as e:
            logger.error(f"Failed to integrate LLM results: {e}")

    async def _update_engagement_style_from_llm_with_confidence(self, user, llm_engagement_style: str, confidence: float):
        """Update engagement style only if confidence is high enough"""
        try:
            # Require higher confidence for style changes
            if confidence < 0.7:  # Higher threshold for style changes
                logger.debug(f"Skipping engagement style update: confidence {confidence} too low")
                return
            
            from ..models.social_user import EngagementStyle
            
            style_mapping = {
                "question_asker": EngagementStyle.QUESTION_ASKER,
                "advice_giver": EngagementStyle.ADVICE_GIVER,
                "skeptic": EngagementStyle.SKEPTIC,
                "supporter": EngagementStyle.SUPPORTER,
                "lurker": EngagementStyle.LURKER,
                "technical_analyst": EngagementStyle.TECHNICAL_ANALYST
            }
            
            new_style = style_mapping.get(llm_engagement_style, EngagementStyle.LURKER)
            old_style = user.engagement_style
            
            # Track evolution if style changed
            if new_style != old_style:
                if not hasattr(user, 'engagement_style_evolution'):
                    user.engagement_style_evolution = []
                
                evolution_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "from_style": old_style.value,
                    "to_style": new_style.value,
                    "confidence": confidence,
                    "llm_triggered": True
                }
                
                user.engagement_style_evolution.append(evolution_record)
                user.engagement_style = new_style
                
                logger.info(f"User @{user.primary_username} evolved: {old_style.value} → {new_style.value} (confidence: {confidence})")
            
        except Exception as e:
            logger.error(f"Failed to update engagement style from LLM: {e}")

    async def _update_journey_progression_from_llm_with_confidence(self, user, llm_journey_stage: str, confidence: float):
        """Update journey progression only if confidence is high enough"""
        try:
            # Require high confidence for journey stage changes
            if confidence < 0.75:  # Even higher threshold for journey changes
                logger.debug(f"Skipping journey progression update: confidence {confidence} too low")
                return
            
            from ..models.social_user import ConversionStage
            
            stage_mapping = {
                "discovery": ConversionStage.DISCOVERY,
                "engagement": ConversionStage.ENGAGEMENT,
                "recognition": ConversionStage.RECOGNITION,
                "collaboration": ConversionStage.COLLABORATION,
                "advocacy": ConversionStage.ADVOCACY,
                "converted": ConversionStage.CONVERTED
            }
            
            new_stage = stage_mapping.get(llm_journey_stage, ConversionStage.DISCOVERY)
            old_stage = user.conversion_stage
            
            # Track progression if stage changed
            if new_stage != old_stage:
                if not hasattr(user, 'journey_progression'):
                    user.journey_progression = []
                
                progression_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "from_stage": old_stage.value,
                    "to_stage": new_stage.value,
                    "progression_direction": self._get_progression_direction(old_stage.value, new_stage.value),
                    "confidence": confidence,
                    "llm_triggered": True
                }
                
                user.journey_progression.append(progression_record)
                user.advance_conversion_stage(new_stage)
                
                logger.info(f"User @{user.primary_username} journey: {old_stage.value} → {new_stage.value} (confidence: {confidence})")
            
        except Exception as e:
            logger.error(f"Failed to update journey progression from LLM: {e}")

    async def _update_communication_style_from_llm(self, user, content: str, intent_data: Dict):
        """Update communication style using LLM analysis instead of keywords"""
        try:
            # Use LLM sentiment and tone analysis
            sentiment = intent_data.get("sentiment", "neutral")
            urgency = intent_data.get("urgency", "normal")
            
            # Map to communication style
            if sentiment == "excited" or urgency == "high":
                user.communication_style.energy_level = "high"
            elif sentiment == "negative" or urgency == "low":
                user.communication_style.energy_level = "low"
            else:
                user.communication_style.energy_level = "moderate"
            
            # Analyze formality from content length and structure
            if len(content) > 100 and any(word in content.lower() for word in ["please", "thank you", "analysis"]):
                user.communication_style.formality = "formal"
            elif len(content) < 50 or any(word in content.lower() for word in ["yo", "hey", "lol"]):
                user.communication_style.formality = "casual"
            else:
                user.communication_style.formality = "neutral"
            
            # Emoji usage analysis
            emoji_count = sum(1 for char in content if ord(char) > 0x1F600)
            if emoji_count > 3:
                user.communication_style.emoji_usage = "heavy"
            elif emoji_count > 1:
                user.communication_style.emoji_usage = "moderate"
            else:
                user.communication_style.emoji_usage = "light"
            
        except Exception as e:
            logger.error(f"Failed to update communication style from LLM: {e}")
    
    def _build_enhanced_intent_prompt(self, content: str, username: str, platform: str,
                                    interaction_type: str, user_context: Dict, context: Dict = None,
                                    symbols: List[str] = None, topics: List[str] = None) -> str:
        """
        ENHANCED PROMPT with actual user message history for better context
        """
        
        # Get ACTUAL recent user messages instead of just summary
        user_message_history = self._get_actual_user_messages(user_context)
        user_summary = self._summarize_user_context(user_context)
        
        # Platform context
        platform_context = context or {}
        competitive_context = platform_context.get("competitor_account", "")
        our_content_topic = platform_context.get("our_content_topic", [])
        
        # Semantic extractions
        symbols_str = ', '.join(symbols) if symbols else 'None'
        topics_str = ', '.join(topics) if topics else 'None'
        
        prompt = f"""
You are a social media engagement intent classifier for a trading bot company. Analyze this interaction and classify the intent with FULL SEMANTIC UNDERSTANDING.

=== CURRENT INTERACTION ===
Platform: {platform}
Username: @{username}
Interaction Type: {interaction_type}
Content: "{content}"

=== SEMANTIC EXTRACTIONS ===
Symbols Mentioned: {symbols_str}
Trading Topics: {topics_str}

=== USER CONVERSATION HISTORY ===
{user_message_history}

=== USER PROFILE SUMMARY ===
{user_summary}

=== PLATFORM CONTEXT ===
Our Content Topics: {', '.join(our_content_topic) if our_content_topic else 'None'}
Competitor Context: {competitive_context if competitive_context else 'None'}

=== INTENT CATEGORIES ===
1. user_interaction: User engaging with our content (comments, questions, discussions)
2. competitor_engagement: Opportunity to engage on competitor's content
3. hashtag_engagement: Trending topic or hashtag opportunity
4. conversion_opportunity: User showing interest in trying our service
5. relationship_building: Existing user relationship maintenance
6. competitive_analysis: Highlighting our advantages over competitors
7. help: User needs support or has questions
8. general: Generic social interaction

=== RESPONSE STRATEGIES ===
- educational_onboarding: New user education and trial conversion
- relationship_building: Existing user appreciation and engagement  
- influencer_partnership: High-value user collaboration approach
- competitive_advantage: Highlight unique value vs competitors
- skeptic_conversion: Address doubts with proof and transparency
- community_building: Foster engagement and belonging
- value_demonstration: Show bot capabilities and results
- casual_engagement: Light, friendly interaction

=== ENGAGEMENT STYLE DETECTION ===
Based on current message AND conversation history, classify user's engagement style:
- question_asker: Frequently asks questions, seeks information
- advice_giver: Offers opinions, shares analysis, helps others
- skeptic: Questions claims, doubts results, critical thinking
- supporter: Positive about our content, shares wins, encourages others
- lurker: Minimal engagement, short responses
- technical_analyst: Uses trading terminology, discusses charts/indicators
- influencer: High engagement, large following, opinion leader

=== USER JOURNEY STAGE ===
Based on interaction history and current message:
- discovery: First few interactions, learning about us
- engagement: Regular interaction, building relationship
- recognition: Acknowledged value, trusted community member
- collaboration: Partnership discussions, content sharing
- advocacy: Actively promotes us, refers others
- converted: Already signed up for SMS service

=== ANALYSIS REQUIRED ===
Provide SEMANTIC ANALYSIS considering:
1. Actual conversation patterns from history
2. Evolution of user's engagement over time
3. Subtle sentiment and tone changes
4. Context of competitive landscape
5. Conversion signals and readiness

Return response as JSON:
{{
    "primary_intent": "",
    "sub_intent": "",
    "sentiment": "",
    "urgency": "",
    "user_type": "",
    "requires_tools": [],
    "response_strategy": "",
    "priority": "",
    "competitive_context": false,
    "conversion_potential": "",
    "confidence": 0.0,
    "engagement_style_detected": "",
    "user_journey_stage": "",
    "reasoning": "Explain your classification based on content and history"
}}
"""
        return prompt
    
    def _get_actual_user_messages(self, user_context: Dict) -> str:
        """Get actual recent user messages for semantic context"""
        try:
            user_profile = user_context.get("user_profile", {})
            interaction_history = user_profile.get("interaction_history", [])
            
            if not interaction_history:
                return "No previous interaction history available."
            
            # Get last 5 interactions for context
            recent_interactions = interaction_history[-5:]
            
            history_text = "Recent conversation history:\n"
            for i, interaction in enumerate(recent_interactions, 1):
                content = interaction.get("content", "")
                our_response = interaction.get("our_response", "")
                timestamp = interaction.get("timestamp", "")
                
                history_text += f"\n{i}. User: \"{content}\""
                if our_response:
                    history_text += f"\n   Our response: \"{our_response}\""
                if timestamp:
                    history_text += f"\n   Time: {timestamp}"
                history_text += "\n"
            
            return history_text
            
        except Exception as e:
            logger.error(f"Failed to get user messages: {e}")
            return "Error retrieving conversation history."
    
    async def _validate_intent_classification(self, content: str, intent_data: Dict) -> Dict[str, Any]:
        """
        LLM VALIDATION LOOP - Self-correction mechanism
        Catches hallucinations and wrong classifications
        """
        try:
            validation_prompt = f"""
You are a quality control system for intent classification. Review this classification and determine if it's accurate.

Original Content: "{content}"

Classification Made:
- Intent: {intent_data.get('primary_intent')}
- Sentiment: {intent_data.get('sentiment')}
- User Type: {intent_data.get('user_type')}
- Response Strategy: {intent_data.get('response_strategy')}
- Conversion Potential: {intent_data.get('conversion_potential')}
- Reasoning: {intent_data.get('reasoning', 'No reasoning provided')}

VALIDATION QUESTIONS:
1. Does the primary intent match the content?
2. Is the sentiment assessment accurate?
3. Is the user type classification reasonable?
4. Does the response strategy make sense?
5. Are there any obvious misclassifications?

If everything looks good, return the SAME classification.
If there are errors, provide CORRECTED classification.

Return as JSON with same structure:
{{
    "primary_intent": "",
    "sub_intent": "",
    "sentiment": "",
    "urgency": "",
    "user_type": "",
    "requires_tools": [],
    "response_strategy": "",
    "priority": "",
    "competitive_context": false,
    "conversion_potential": "",
    "confidence": 0.0,
    "engagement_style_detected": "",
    "user_journey_stage": "",
    "validation_notes": "Confirmed accurate / Corrected because..."
}}
"""
            
            response = await self.gemini_client.generate_content_async(validation_prompt)
            validated_data = self._parse_gemini_response(response.text)
            
            # Log validation results
            if validated_data.get("validation_notes", "").startswith("Corrected"):
                logger.warning(f"Intent classification corrected: {validated_data.get('validation_notes')}")
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Intent validation failed: {e}")
            return intent_data  # Return original if validation fails
    
    def _get_progression_direction(self, from_stage: str, to_stage: str) -> str:
        """Determine if user is progressing forward or backward in journey"""
        stage_order = ["discovery", "engagement", "recognition", "collaboration", "advocacy", "converted"]
        
        try:
            from_index = stage_order.index(from_stage)
            to_index = stage_order.index(to_stage)
            
            if to_index > from_index:
                return "forward"
            elif to_index < from_index:
                return "backward"
            else:
                return "stable"
        except ValueError:
            return "unknown"
    
    def _parse_extraction_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM extraction response"""
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            parsed = json.loads(clean_response)
            return {
                "symbols": parsed.get("symbols", []),
                "topics": parsed.get("topics", [])
            }
            
        except json.JSONDecodeError:
            logger.error("Failed to parse extraction response, using regex fallback")
            return self._regex_extraction_fallback(response)
    
    def _regex_extraction_fallback(self, response: str) -> Dict[str, List[str]]:
        """Regex fallback for extraction parsing"""
        symbols = re.findall(r'"([A-Z]{1,5})"', response)
        topics = re.findall(r'"([a-z_]+)"', response)
        
        return {
            "symbols": symbols[:5],  # Limit to 5
            "topics": topics[:5]     # Limit to 5
        }
    
    def _extract_symbols_fallback(self, content: str) -> List[str]:
        """Fallback symbol extraction using enhanced keyword matching"""
        content_upper = content.upper()
        
        # Enhanced symbol list
        common_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD",
            "NFLX", "CRM", "ORCL", "ADBE", "PYPL", "INTC", "CSCO", "IBM",
            "SPY", "QQQ", "IWM", "VTI", "VOO", "BTC", "ETH"
        ]
        
        # Company name mapping
        company_names = {
            "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL",
            "AMAZON": "AMZN", "TESLA": "TSLA", "FACEBOOK": "META", "META": "META",
            "NVIDIA": "NVDA", "NETFLIX": "NFLX", "BITCOIN": "BTC", "ETHEREUM": "ETH"
        }
        
        found_symbols = []
        
        # Direct symbol matches
        for symbol in common_symbols:
            if symbol in content_upper and self._is_valid_symbol_context(content, symbol):
                found_symbols.append(symbol)
        
        # Company name matches
        for company, symbol in company_names.items():
            if company in content_upper:
                found_symbols.append(symbol)
        
        return list(set(found_symbols))
    
    def _extract_trading_topics_fallback(self, content: str) -> List[str]:
        """Fallback topic extraction using keyword matching"""
        content_lower = content.lower()
        
        topic_keywords = {
            "technical_analysis": ["chart", "rsi", "macd", "support", "resistance", "breakout"],
            "options": ["call", "put", "option", "strike", "expiry", "premium"],
            "earnings": ["earnings", "eps", "guidance", "revenue", "profit"],
            "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain"],
            "day_trading": ["day trade", "scalp", "intraday", "quick"],
            "swing_trading": ["swing", "hold", "position", "weeks"],
            "news_trading": ["news", "catalyst", "announcement", "event"],
            "ai_trading": ["ai", "algorithm", "bot", "automated"]
        }
        
        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    # Rest of the methods remain the same as original...
    async def generate_response(self, intent_data: Dict, tool_results: Dict, 
                              user_context: Dict, platform_constraints: Dict) -> str:
        """Generate personalized response using Claude - SAME AS ORIGINAL"""
        try:
            response_context = self._build_response_context(
                intent_data, tool_results, user_context, platform_constraints
            )
            
            prompt = self._build_response_prompt(response_context)
            response = await self._call_claude_response(prompt)
            cleaned_response = self._clean_response(response)
            validated_response = self._validate_response(cleaned_response, platform_constraints)
            
            return validated_response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return self._get_fallback_response(intent_data, platform_constraints)
    
    async def _call_gemini_intent(self, prompt: str) -> str:
        """Call Gemini for intent classification - SAME AS ORIGINAL"""
        try:
            response = await self.gemini_client.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini intent classification failed: {e}")
            raise
    
    def _parse_gemini_response(self, response: str) -> Dict[str, Any]:
        """Parse Gemini JSON response - SAME AS ORIGINAL"""
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            parsed = json.loads(clean_response)
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return self._fallback_parse_gemini(response)
    
    def _build_response_context(self, intent_data: Dict, tool_results: Dict,
                              user_context: Dict, platform_constraints: Dict) -> Dict[str, Any]:
        """Build context for Claude response generation - SAME AS ORIGINAL"""
        return {
            "intent": intent_data,
            "tool_results": tool_results,
            "user_context": user_context,
            "platform": platform_constraints.get("platform", "unknown"),
            "max_length": platform_constraints.get("max_length", 280),
            "supports_emoji": platform_constraints.get("supports_emoji", True),
            "supports_hashtags": platform_constraints.get("supports_hashtags", True),
            "response_strategy": intent_data.get("response_strategy", "casual_engagement"),
            "user_relationship_score": user_context.get("relationship_score", 0),
            "user_influence_level": user_context.get("influence_metrics", {}).get("level", "unknown"),
            "conversation_stage": user_context.get("conversion_stage", "discovery")
        }
    
    def _build_response_prompt(self, context: Dict[str, Any]) -> str:
        """
        ENHANCED response prompt with frequency-weighted personalization
        """
        intent = context["intent"]
        user_context = context["user_context"]
        tool_results = context["tool_results"]
        platform = context["platform"]
        
        # User personalization with FREQUENCY DATA
        user_profile = user_context.get("user_profile", {})
        username = user_profile.get("primary_username", "")
        
        # GET FREQUENCY-WEIGHTED PERSONALIZATION
        personalization_context = {}
        if hasattr(user_profile, 'get_personalization_context'):
            personalization_context = user_profile.get_personalization_context()
        
        top_topics = personalization_context.get("top_topics", [])
        top_symbols = personalization_context.get("top_symbols", [])
        learning_velocity = personalization_context.get("learning_velocity", {})
        
        # SYMBOL RELATIONSHIP suggestions
        mentioned_symbols = intent.get("symbols", [])
        related_symbols = []
        if mentioned_symbols and hasattr(user_profile, 'get_related_symbols'):
            for symbol in mentioned_symbols[:1]:  # Check first mentioned symbol
                related = user_profile.get_related_symbols(symbol, 2)
                related_symbols.extend([r[0] for r in related])
        
        # Platform-specific voice rules
        platform_voice_rules = {
            "tiktok": "Casual, energetic, use emojis, trending language, under 150 chars",
            "twitter": "Concise, witty, hashtag-friendly, professional but accessible",
            "instagram": "Visual storytelling tone, emoji-rich, engaging, community-focused",
            "youtube": "Conversational, educational, longer-form OK, professional tone",
            "linkedin": "Professional, business-focused, industry expertise, formal tone"
        }
        
        voice_rule = platform_voice_rules.get(platform, "Professional but friendly tone")
        
        prompt = f"""
You are an expert social media manager for a trading bot company. Generate a personalized response using FREQUENCY-WEIGHTED user insights.

=== INTERACTION CONTEXT ===
Platform: {platform}
User: @{username}
Intent: {intent.get('primary_intent')} - {intent.get('sub_intent')}
Response Strategy: {intent.get('response_strategy')}
Current Symbols Mentioned: {', '.join(mentioned_symbols) if mentioned_symbols else 'None'}

=== FREQUENCY-WEIGHTED USER INSIGHTS ===
Top Discussed Topics: {', '.join([f"{topic}({score:.1f})" for topic, score in top_topics[:3]]) if top_topics else 'None'}
Top Discussed Symbols: {', '.join([f"{symbol}({score:.1f})" for symbol, score in top_symbols[:3]]) if top_symbols else 'None'}
Related Symbols: {', '.join(related_symbols) if related_symbols else 'None'}
Learning Velocity: {learning_velocity.get('learning_rate', 'unknown')} ({learning_velocity.get('updates_per_day', 0):.1f} updates/day)

=== USER PROFILE ===
Relationship Score: {user_context.get('relationship_score', 0)}/100
Journey Stage: {intent.get('user_journey_stage', 'discovery')}
Engagement Style: {intent.get('engagement_style_detected', 'unknown')}
Communication Style: {user_context.get('communication_style', {}).get('formality', 'neutral')}

=== TOOL RESULTS ===
{json.dumps(tool_results, indent=2) if tool_results else 'No tool results available'}

=== PERSONALIZATION STRATEGY ===
**Use Frequency Data for Personalization:**
- Reference their most discussed topics when relevant
- Mention their frequently traded symbols for context
- Suggest related symbols from their relationship graph
- Adapt depth based on their learning velocity

**Platform Optimization:**
- {voice_rule}
- Max Length: {context.get('max_length', 280)} characters
- Supports Emoji: {context.get('supports_emoji', True)}

**Response Guidelines:**
- If high learning velocity: Use more technical depth
- If top topics include "technical_analysis": Reference charts/indicators
- If related symbols available: Mention symbol relationships naturally
- If low frequency data: Keep response general but engaging

Generate a response that leverages the frequency-weighted insights for maximum personalization. Return only the response text.
"""
        return prompt
    
    async def _call_claude_response(self, prompt: str) -> str:
        """Call Claude for response generation - SAME AS ORIGINAL"""
        try:
            response = await self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude response generation failed: {e}")
            raise
    
    def _clean_response(self, response: str) -> str:
        """Clean response - SAME AS ORIGINAL"""
        if not response:
            return ""
        
        artifacts = [
            "Here's a response:", "Here's the response:", "Response:",
            "Here's what I'd say:", "I'd respond with:", "My response:",
            "The response would be:", "A good response would be:",
            "Here's a personalized response:", "I would say:",
        ]
        
        cleaned = response.strip()
        for artifact in artifacts:
            if cleaned.lower().startswith(artifact.lower()):
                cleaned = cleaned[len(artifact):].strip()
                break
        
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        
        return cleaned
    
    def _validate_response(self, response: str, platform_constraints: Dict) -> str:
        """Validate response meets platform constraints - SAME AS ORIGINAL"""
        max_length = platform_constraints.get("max_length", 280)
        
        if len(response) > max_length:
            truncated = response[:max_length-3]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                response = truncated[:last_space] + "..."
            else:
                response = truncated + "..."
        
        return response
    
    def _summarize_user_context(self, user_context: Dict) -> str:
        """Summarize user context for prompts - SAME AS ORIGINAL"""
        if not user_context.get("user_profile"):
            return "New user - no previous interactions"
        
        profile = user_context["user_profile"]
        return f"""
Previous Interactions: {profile.get('total_interactions', 0)}
Relationship Score: {user_context.get('relationship_score', 0)}/100
Influence Level: {user_context.get('influence_metrics', {}).get('level', 'unknown')}
Trading Interests: {', '.join(user_context.get('trading_interests', {}).get('symbols', [])) if user_context.get('trading_interests', {}).get('symbols') else 'None'}
Communication Style: {user_context.get('communication_style', {}).get('formality', 'unknown')}
Conversion Stage: {user_context.get('conversion_stage', 'discovery')}
"""
    
    def _is_valid_symbol_context(self, content: str, symbol: str) -> bool:
        """Check if symbol appears in valid context - SAME AS ORIGINAL"""
        content_lower = content.lower()
        
        false_positives = {
            "CAT": ["cat", "cats", "kitty"],
            "CRM": ["crm"],
            "IT": [" it ", "it's", "it is"],
            "AI": ["ai "],
            "GO": [" go ", "going", "goes"]
        }
        
        if symbol in false_positives:
            for fp in false_positives[symbol]:
                if fp in content_lower:
                    return False
        
        return True
    
    def _get_fallback_intent(self, content: str, interaction_type: str) -> Dict[str, Any]:
        """Fallback intent classification - ENHANCED WITH SEMANTIC DEFAULTS"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["?", "how", "what", "why", "when"]):
            intent = "user_interaction"
            strategy = "educational_onboarding"
            engagement_style = "question_asker"
        elif any(word in content_lower for word in ["thanks", "great", "awesome", "helpful"]):
            intent = "relationship_building"
            strategy = "community_building"
            engagement_style = "supporter"
        elif any(word in content_lower for word in ["scam", "fake", "doubt", "suspicious"]):
            intent = "user_interaction"
            strategy = "skeptic_conversion"
            engagement_style = "skeptic"
        else:
            intent = "general"
            strategy = "casual_engagement"
            engagement_style = "lurker"
        
        return {
            "intent": intent,
            "sub_intent": "fallback_classification",
            "symbols": self._extract_symbols_fallback(content),
            "topics": self._extract_trading_topics_fallback(content),
            "sentiment": "neutral",
            "urgency": "normal",
            "user_type": "regular",
            "requires_tools": [],
            "response_strategy": strategy,
            "priority": "medium",
            "competitive_context": False,
            "conversion_potential": "low",
            "confidence": 0.5,
            "engagement_style_detected": engagement_style,
            "user_journey_stage": "discovery"
        }
    
    def _get_fallback_response(self, intent_data: Dict, platform_constraints: Dict) -> str:
        """Fallback response - SAME AS ORIGINAL"""
        if intent_data.get("intent") == "user_interaction":
            return "Thanks for engaging! Our AI provides unique trading insights via SMS. Free trial available! 📈"
        elif intent_data.get("intent") == "competitor_engagement":
            return "Interesting perspective! Our AI takes a different approach to market analysis. Always valuable to compare methodologies! 📊"
        else:
            return "Great insight! 👍"
    
    def _fallback_parse_gemini(self, response: str) -> Dict[str, Any]:
        """Fallback parsing - SAME AS ORIGINAL"""
        intent_match = re.search(r'"primary_intent":\s*"([^"]+)"', response)
        sentiment_match = re.search(r'"sentiment":\s*"([^"]+)"', response)
        strategy_match = re.search(r'"response_strategy":\s*"([^"]+)"', response)
        
        return {
            "primary_intent": intent_match.group(1) if intent_match else "general",
            "sub_intent": "fallback_parse",
            "sentiment": sentiment_match.group(1) if sentiment_match else "neutral",
            "urgency": "normal",
            "user_type": "regular",
            "requires_tools": [],
            "response_strategy": strategy_match.group(1) if strategy_match else "casual_engagement",
            "priority": "medium",
            "competitive_context": False,
            "conversion_potential": "low",
            "confidence": 0.3,
            "engagement_style_detected": "unknown",
            "user_journey_stage": "discovery"
        }
