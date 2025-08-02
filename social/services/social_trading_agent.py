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
        SEMANTIC INTENT PARSING - Fully LLM-powered with validation loop
        """
        try:
            # Get FULL user context with actual message history
            user_context = await self.memory_service.get_conversation_context(f"{platform}_{username}")
            
            # SEMANTIC EXTRACTION: Get symbols and topics via LLM FIRST
            symbols, topics = await self._extract_symbols_and_topics_semantic(content)
            
            # Build enhanced Gemini prompt with ACTUAL user messages
            prompt = self._build_enhanced_intent_prompt(
                content, username, platform, interaction_type, user_context, context, symbols, topics
            )
            
            # Call Gemini for intent classification
            response = await self._call_gemini_intent(prompt)
            
            # Parse and validate response
            intent_data = self._parse_gemini_response(response)
            
            # LLM VALIDATION LOOP - Self-correction mechanism
            validated_intent = await self._validate_intent_classification(content, intent_data)
            
            # Build final intent object with semantic extractions
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
                "engagement_style_detected": validated_intent.get("engagement_style_detected", "unknown"),
                "user_journey_stage": validated_intent.get("user_journey_stage", "discovery")
            }
            
            # STORE CLASSIFICATION IN USER PROFILE for longitudinal analysis
            await self._store_intent_classification_in_user_profile(
                username, platform, final_intent, content
            )
            
            return final_intent
            
        except Exception as e:
            logger.error(f"Failed to parse social intent: {e}")
            return self._get_fallback_intent(content, interaction_type)
    
    async def _extract_symbols_and_topics_semantic(self, content: str) -> Tuple[List[str], List[str]]:
        """
        SEMANTIC SYMBOL & TOPIC EXTRACTION via LLM
        Replaces keyword-based extraction with true semantic understanding
        """
        try:
            prompt = f"""
You are an expert at extracting trading-related information from social media content.

Analyze this content and extract:
1. Stock symbols (tickers) mentioned - include company names converted to tickers
2. Trading topics discussed

Content: "{content}"

=== EXTRACTION RULES ===
**Stock Symbols:**
- Direct tickers: AAPL, TSLA, BTC, etc.
- Company names: "Apple" → AAPL, "Tesla" → TSLA, "Microsoft" → MSFT
- Crypto: "Bitcoin" → BTC, "Ethereum" → ETH
- ETFs: "S&P 500" → SPY, "Nasdaq" → QQQ
- Only extract if actually discussed in trading/financial context

**Trading Topics:**
- technical_analysis: Charts, indicators, patterns, levels
- options: Calls, puts, strikes, premiums, volatility
- earnings: Earnings reports, guidance, surprises
- crypto: Cryptocurrency discussion
- day_trading: Short-term trading, scalping, intraday
- swing_trading: Multi-day holds, momentum plays
- news_trading: Event-driven, catalysts, announcements
- ai_trading: AI, algorithms, bots, automation
- market_psychology: Sentiment, fear, greed, FOMO
- risk_management: Stop losses, position sizing, diversification
- fundamental_analysis: P/E ratios, revenue, growth
- sector_rotation: Industry trends, sector moves
- macro_economics: Fed policy, rates, inflation

Return ONLY valid symbols and topics that are actually discussed:

{{
    "symbols": ["AAPL", "TSLA"],
    "topics": ["technical_analysis", "earnings"]
}}
"""
            
            response = await self.gemini_client.generate_content_async(prompt)
            
            # Parse LLM response
            extraction_data = self._parse_extraction_response(response.text)
            
            symbols = extraction_data.get("symbols", [])
            topics = extraction_data.get("topics", [])
            
            logger.info(f"LLM extracted - Symbols: {symbols}, Topics: {topics}")
            return symbols, topics
            
        except Exception as e:
            logger.error(f"Semantic extraction failed: {e}")
            # Fallback to basic extraction
            return self._extract_symbols_fallback(content), self._extract_trading_topics_fallback(content)
    
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
    
    async def _store_intent_classification_in_user_profile(self, username: str, platform: str,
                                                         intent_data: Dict, content: str):
        """
        STORE CLASSIFICATION IN USER PROFILE for longitudinal analysis
        Tracks user journey progression over time
        """
        try:
            user = await self.memory_service.get_or_create_user(username, platform)
            
            # Store current classification
            classification_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "content": content,
                "intent": intent_data.get("intent"),
                "sentiment": intent_data.get("sentiment"),
                "engagement_style": intent_data.get("engagement_style_detected"),
                "journey_stage": intent_data.get("user_journey_stage"),
                "conversion_potential": intent_data.get("conversion_potential"),
                "confidence": intent_data.get("confidence")
            }
            
            # Add to user's classification history
            if not hasattr(user, 'intent_classifications'):
                user.intent_classifications = []
            
            user.intent_classifications.append(classification_record)
            
            # Keep only last 20 classifications
            if len(user.intent_classifications) > 20:
                user.intent_classifications = user.intent_classifications[-20:]
            
            # Update user's primary engagement style based on patterns
            await self._update_user_engagement_style_from_classifications(user)
            
            # Update user's journey stage progression
            await self._update_user_journey_progression(user, intent_data.get("user_journey_stage"))
            
            # Save updates
            await self.memory_service._save_user_updates(user)
            
            logger.info(f"Stored classification for @{username}: {intent_data.get('intent')} - {intent_data.get('engagement_style_detected')}")
            
        except Exception as e:
            logger.error(f"Failed to store intent classification: {e}")
    
    async def _update_user_engagement_style_from_classifications(self, user):
        """Update user's primary engagement style based on classification patterns"""
        try:
            if not hasattr(user, 'intent_classifications') or not user.intent_classifications:
                return
            
            # Count engagement style occurrences
            style_counts = {}
            for classification in user.intent_classifications[-10:]:  # Last 10 interactions
                style = classification.get("engagement_style", "unknown")
                style_counts[style] = style_counts.get(style, 0) + 1
            
            # Determine dominant style
            if style_counts:
                dominant_style = max(style_counts, key=style_counts.get)
                
                # Update user's engagement style
                from ..models.social_user import EngagementStyle
                try:
                    user.engagement_style = EngagementStyle(dominant_style.upper())
                except ValueError:
                    user.engagement_style = EngagementStyle.LURKER
                
                logger.info(f"Updated engagement style for {user.primary_username}: {dominant_style}")
            
        except Exception as e:
            logger.error(f"Failed to update engagement style: {e}")
    
    async def _update_user_journey_progression(self, user, current_stage: str):
        """Track user journey progression over time"""
        try:
            if not hasattr(user, 'journey_progression'):
                user.journey_progression = []
            
            # Check if stage has changed
            last_stage = user.journey_progression[-1].get("stage") if user.journey_progression else "discovery"
            
            if current_stage != last_stage:
                progression_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "from_stage": last_stage,
                    "to_stage": current_stage,
                    "progression_direction": self._get_progression_direction(last_stage, current_stage)
                }
                
                user.journey_progression.append(progression_record)
                
                # Update user's conversion stage
                from ..models.social_user import ConversionStage
                stage_mapping = {
                    "discovery": ConversionStage.DISCOVERY,
                    "engagement": ConversionStage.ENGAGEMENT,
                    "recognition": ConversionStage.RECOGNITION,
                    "collaboration": ConversionStage.COLLABORATION,
                    "advocacy": ConversionStage.ADVOCACY,
                    "converted": ConversionStage.CONVERTED
                }
                
                if current_stage in stage_mapping:
                    user.advance_conversion_stage(stage_mapping[current_stage])
                
                logger.info(f"User journey progression: {user.primary_username} moved from {last_stage} to {current_stage}")
            
        except Exception as e:
            logger.error(f"Failed to update journey progression: {e}")
    
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
        """Build Claude prompt for response generation - ENHANCED WITH PLATFORM VOICE"""
        intent = context["intent"]
        user_context = context["user_context"]
        tool_results = context["tool_results"]
        platform = context["platform"]
        
        # User personalization
        user_profile = user_context.get("user_profile", {})
        username = user_profile.get("primary_username", "")
        communication_style = user_context.get("communication_style", {})
        trading_interests = user_context.get("trading_interests", {})
        
        # PLATFORM-SPECIFIC VOICE RULES
        platform_voice_rules = {
            "tiktok": "Casual, energetic, use emojis, trending language, under 150 chars",
            "twitter": "Concise, witty, hashtag-friendly, professional but accessible",
            "instagram": "Visual storytelling tone, emoji-rich, engaging, community-focused",
            "youtube": "Conversational, educational, longer-form OK, professional tone",
            "linkedin": "Professional, business-focused, industry expertise, formal tone"
        }
        
        voice_rule = platform_voice_rules.get(platform, "Professional but friendly tone")
        
        prompt = f"""
You are an expert social media manager for a trading bot company. Generate a personalized response for this social media interaction.

=== INTERACTION CONTEXT ===
Platform: {platform}
User: @{username}
Intent: {intent.get('primary_intent')} - {intent.get('sub_intent')}
Response Strategy: {intent.get('response_strategy')}
Priority: {intent.get('priority')}
Sentiment: {intent.get('sentiment')}
User Journey Stage: {intent.get('user_journey_stage', 'discovery')}
Engagement Style: {intent.get('engagement_style_detected', 'unknown')}

=== USER PROFILE ===
Relationship Score: {context.get('user_relationship_score', 0)}/100
Influence Level: {context.get('user_influence_level', 'unknown')}
Conversion Stage: {context.get('conversation_stage', 'discovery')}
Communication Style: {communication_style.get('formality', 'neutral')}, {communication_style.get('energy', 'moderate')}
Trading Interests: {', '.join(trading_interests.get('symbols', [])) if trading_interests.get('symbols') else 'None'}
Trading Focus: {', '.join(trading_interests.get('focus', [])) if trading_interests.get('focus') else 'None'}

=== TOOL RESULTS ===
{json.dumps(tool_results, indent=2) if tool_results else 'No tool results available'}

=== PLATFORM CONSTRAINTS ===
Max Length: {context.get('max_length', 280)} characters
Supports Emoji: {context.get('supports_emoji', True)}
Supports Hashtags: {context.get('supports_hashtags', True)}
Platform Voice Rule: {voice_rule}

=== RESPONSE GUIDELINES ===

**Strategy: {intent.get('response_strategy')}**
- Match user's communication style and energy level
- Reference their trading interests when relevant
- Follow platform-specific voice rules
- Stay within character limits
- Include call-to-action when appropriate
- AVOID repeating disclaimers if user has seen them before

**Personalization Rules:**
- Use @username if relationship score > 30
- Match formality level: {communication_style.get('formality', 'neutral')}
- Match energy level: {communication_style.get('energy', 'moderate')}
- Reference journey stage progression when relevant

**Social Proof Injection:**
- If competitive context: Reference our unique advantages directly
- If skeptical user: Provide specific proof points
- If conversion opportunity: Mention user success stories

**Platform-Specific Adaptations:**
- {voice_rule}
- Use appropriate symbols/hashtags: {', '.join(intent.get('symbols', []))}
- Include relevant topics: {', '.join(intent.get('topics', []))}

Generate a response that follows these guidelines and matches the user's style perfectly. Return only the response text, no meta-commentary.
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
