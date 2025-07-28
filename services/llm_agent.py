# services/llm_agent.py - COMPLETE ENHANCED HUMAN-LIKE VERSION WITH CONTEXT-RICH CONVERSATION

import json
import asyncio
import re
from typing import Dict, List, Optional, Any
from loguru import logger
import openai
from datetime import datetime
from openai import AsyncOpenAI

class TradingAgent:
    """Human-like trading companion that adapts to user personality with rich conversation context"""
    
    def __init__(self, openai_client, personality_engine, database_service=None):
        self.openai_client = openai_client
        self.personality_engine = personality_engine
        self.database_service = database_service  # NEW: Database service for context retrieval
        
        # Human conversation patterns for natural responses
        self.conversation_starters = {
            'casual': ["yooo", "ayy", "what's good", "sup", "hey buddy", "yo"],
            'friendly': ["hey there", "what's up", "howdy", "hey friend"],
            'professional': ["hello", "good to see you", "greetings"]
        }
        
        # Natural transition phrases that humans actually use
        self.natural_transitions = [
            "btw", "also", "oh and", "plus", "real talk", "honestly", 
            "tbh", "ngl", "lowkey", "for real", "straight up"
        ]
        
        # Personality-based excitement indicators
        self.excitement_phrases = {
            'high': ["YOOO", "holy shit", "damn!", "no way!", "LFG!", "🔥🔥"],
            'medium': ["nice!", "solid move", "looking good", "not bad"],
            'low': ["interesting", "noted", "I see", "makes sense"]
        }
    
    async def parse_intent(self, message: str, user_phone: str = None) -> Dict[str, Any]:
        """Parse intent with rich conversation context awareness"""
        
        # NEW: Get conversation context if database service available
        conversation_context = {}
        if self.database_service and user_phone:
            try:
                conversation_context = await self.database_service.get_conversation_context(user_phone)
            except Exception as e:
                logger.warning(f"Failed to get conversation context: {e}")
                conversation_context = {}
        
        # Get user context for better parsing (existing functionality)
        user_context = ""
        if user_phone and self.personality_engine:
            profile = self.personality_engine.get_user_profile(user_phone)
            experience = profile.get('trading_personality', {}).get('experience_level', 'intermediate')
            style = profile.get('communication_style', {}).get('formality', 'casual')
            energy = profile.get('communication_style', {}).get('energy', 'moderate')
            user_context = f"User is {experience} trader with {style}/{energy} style."
        
        # NEW: Build enhanced context with conversation history
        enhanced_context = self._build_enhanced_context(user_context, conversation_context)
        
        prompt = f"""You're parsing a message from a real person to their trading buddy. Understand the HUMAN context and emotion behind their words.

{enhanced_context}

Extract intent but think like a human - what is this person REALLY asking/feeling?

Message: "{message}"

CONVERSATION CONTEXT AWARENESS:
{self._format_conversation_context(conversation_context)}

HUMAN CONVERSATION RULES:
- "yo what's google doing?" = they're casually checking GOOGL performance, probably considering a trade
- "tesla looking spicy 🌶️" = they think TSLA has momentum, want confirmation
- "should I dump my AAPL?" = they're worried, need emotional support + analysis
- "NVDA to the moon! 🚀🚀" = they're excited, want to celebrate/get validation
- "ugh my portfolio is bleeding" = they're frustrated, need reassurance + analysis
- "thoughts on META earnings?" = they want your opinion on upcoming catalyst
- "is Silver etf a good pick?" = they want analysis of SLV, treat as analyze intent

SYMBOL EXTRACTION - NATURAL LANGUAGE:
- "google" → "GOOGL" (obvious)
- "tesla/elon's company" → "TSLA" 
- "the fruit company" → "AAPL"
- "that AI chip company" → "NVDA"
- "zuck's thing" → "META"
- "silver etf" → "SLV"
- "my FAANG positions" → extract multiple if mentioned

INTENT CLASSIFICATION:
- If they mention a specific stock/ETF and want to know about it → "analyze" (NOT "general")
- If they ask "is X a good pick/investment" → "analyze" 
- If they want fundamentals specifically → "fundamental_analysis"
- If they want portfolio info → "portfolio"
- General chat without stock mentions → "general"

EMOTIONAL CONTEXT DETECTION:
- Detect: excited, worried, frustrated, curious, FOMO, celebrating, seeking validation
- This affects what tools they REALLY need vs what they asked for

Return JSON:
{{
    "intent": "analyze|celebrate|worry_check|validation_seeking|fomo_check|earnings_play|general|fundamental_analysis",
    "symbols": ["SLV"],
    "emotional_state": "excited|worried|frustrated|curious|celebrating|neutral",
    "confidence_level": "high|medium|low", 
    "human_subtext": "they want validation for their investment thesis",
    "requires_tools": ["technical_analysis", "news_sentiment"],
    "response_tone": "celebratory|reassuring|analytical|validating",
    "urgency": "low|medium|high",
    "conversation_continuity": "new_topic|continuing_discussion|follow_up_question|reference_to_previous"
}}

Examples:
- "TSLA fucking mooning! 🚀" → {{"intent": "celebrate", "emotional_state": "excited", "response_tone": "celebratory"}}
- "shit should I sell my NVDA?" → {{"intent": "worry_check", "emotional_state": "worried", "response_tone": "reassuring"}}
- "thoughts on AAPL earnings?" → {{"intent": "earnings_play", "emotional_state": "curious", "response_tone": "analytical"}}
- "is Silver etf a good pick?" → {{"intent": "analyze", "symbols": ["SLV"], "emotional_state": "curious", "requires_tools": ["technical_analysis", "news_sentiment"]}}
"""

        try:
            if hasattr(self.openai_client, 'client'):
                response = await self.openai_client.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Slightly higher for more natural understanding
                    max_tokens=300,  # Increased for conversation context
                    response_format={"type": "json_object"}
                )
            else:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )
            
            intent_data = json.loads(response.choices[0].message.content)
            intent_data = self._validate_intent(intent_data, message)
            
            # NEW: Add conversation context to intent data
            intent_data["conversation_context"] = conversation_context.get("context_summary", "")
            
            logger.info(f"Human intent parsed: {intent_data.get('intent')} | Emotion: {intent_data.get('emotional_state')} | Symbols: {intent_data.get('symbols', [])} | Context: {intent_data.get('conversation_continuity', 'unknown')}")
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            return self._fallback_intent_parsing(message)
    
    def _build_enhanced_context(self, user_context: str, conversation_context: Dict) -> str:
        """Build enhanced context including conversation history"""
        enhanced_parts = [user_context] if user_context else []
        
        if conversation_context:
            conv_ctx = conversation_context.get("conversation_context", {})
            today_session = conversation_context.get("today_session", {})
            
            # Add relationship context
            relationship = conv_ctx.get("relationship_stage", "new")
            total_convos = conv_ctx.get("total_conversations", 0)
            enhanced_parts.append(f"Relationship: {relationship} ({total_convos} conversations)")
            
            # Add recent discussion context
            recent_symbols = conv_ctx.get("recent_symbols", [])
            if recent_symbols:
                enhanced_parts.append(f"Recently discussed: {', '.join(recent_symbols[-3:])}")
            
            # Add today's session context
            if today_session.get("message_count", 0) > 0:
                topics_today = today_session.get("topics_discussed", [])
                mood_today = today_session.get("session_mood", "neutral")
                enhanced_parts.append(f"Today: {len(topics_today)} topics, {mood_today} mood")
            else:
                enhanced_parts.append("First message today")
        
        return "\n".join(enhanced_parts)
    
    def _format_conversation_context(self, conversation_context: Dict) -> str:
        """Format conversation context for prompt"""
        if not conversation_context:
            return "- New conversation, no previous context"
        
        context_lines = []
        
        # Recent topics and symbols
        conv_ctx = conversation_context.get("conversation_context", {})
        recent_symbols = conv_ctx.get("recent_symbols", [])
        recent_topics = conv_ctx.get("recent_topics", [])
        
        if recent_symbols:
            context_lines.append(f"- Recently discussed symbols: {', '.join(recent_symbols[-5:])}")
        
        if recent_topics:
            topic_strs = []
            for topic in recent_topics[-3:]:
                if isinstance(topic, dict):
                    topic_strs.append(f"{topic.get('topic', 'unknown')} ({', '.join(topic.get('symbols', []))})")
                else:
                    topic_strs.append(str(topic))
            context_lines.append(f"- Recent topics: {', '.join(topic_strs)}")
        
        # Pending decisions
        pending = conv_ctx.get("pending_decisions", [])
        if pending:
            decisions = [d.get("decision", str(d)) if isinstance(d, dict) else str(d) for d in pending[-2:]]
            context_lines.append(f"- Pending decisions: {', '.join(decisions)}")
        
        # Session context
        today_session = conversation_context.get("today_session", {})
        if today_session.get("message_count", 0) > 0:
            context_lines.append(f"- Today's session: {today_session.get('message_count')} messages, {today_session.get('session_mood', 'neutral')} mood")
        else:
            context_lines.append("- First message of the day")
        
        return "\n".join(context_lines) if context_lines else "- No significant conversation context"
    
    def _validate_intent(self, intent_data: Dict, original_message: str) -> Dict:
        """Validate and clean the parsed intent"""
        
        # Ensure required fields exist
        if "intent" not in intent_data:
            intent_data["intent"] = "general"
        
        if "symbols" not in intent_data:
            intent_data["symbols"] = []
        
        if "confidence" not in intent_data:
            intent_data["confidence"] = 0.5
        
        if "requires_tools" not in intent_data:
            intent_data["requires_tools"] = []
        
        # Clean symbols (remove duplicates, validate format)
        if intent_data["symbols"]:
            cleaned_symbols = []
            for symbol in intent_data["symbols"]:
                if isinstance(symbol, str) and 1 <= len(symbol) <= 5 and symbol.isalpha():
                    cleaned_symbols.append(symbol.upper())
            intent_data["symbols"] = list(dict.fromkeys(cleaned_symbols))  # Remove duplicates
        
        # CRITICAL FIX: Auto-determine required tools based on intent AND symbols
        if intent_data["intent"] == "analyze" and intent_data["symbols"]:
            if "technical_analysis" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("technical_analysis")
            if "news_sentiment" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("news_sentiment")
        
        if intent_data["intent"] == "fundamental_analysis" and intent_data["symbols"]:
            if "fundamental_analysis" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("fundamental_analysis")
        
        if intent_data["intent"] == "comprehensive_analysis" and intent_data["symbols"]:
            required_tools = ["technical_analysis", "news_sentiment", "fundamental_analysis"]
            for tool in required_tools:
                if tool not in intent_data["requires_tools"]:
                    intent_data["requires_tools"].append(tool)
        
        if intent_data["intent"] == "news" and intent_data["symbols"]:
            if "news_sentiment" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("news_sentiment")
        
        if intent_data["intent"] == "portfolio":
            if "portfolio_check" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("portfolio_check")
        
        # CRITICAL FIX: If we have symbols but no tools, auto-add analysis tools
        if intent_data["symbols"] and not intent_data["requires_tools"]:
            intent_data["requires_tools"] = ["technical_analysis", "news_sentiment"]
            if intent_data["intent"] == "general":
                intent_data["intent"] = "analyze"  # Fix intent if we have symbols
        
        return intent_data
    
    def _fallback_intent_parsing(self, message: str) -> Dict:
        """Fallback with SIMPLE company name mapping - no complex regex"""
        
        message_lower = message.lower()
        
        # SIMPLE company mapping - let LLM handle complex cases
        simple_mappings = {
            'google': 'GOOGL', 'tesla': 'TSLA', 'apple': 'AAPL', 
            'microsoft': 'MSFT', 'amazon': 'AMZN', 'meta': 'META',
            'facebook': 'META', 'nvidia': 'NVDA', 'netflix': 'NFLX',
            'silver etf': 'SLV', 'silver': 'SLV'
        }
        
        symbols = []
        for company, symbol in simple_mappings.items():
            if company in message_lower:
                symbols.append(symbol)
        
        # Enhanced intent detection with fundamental analysis support
        if any(word in message_lower for word in ['fundamental', 'fundamentals', 'valuation', 'ratios', 'pe ratio', 'financial', 'earnings', 'revenue']):
            intent = "fundamental_analysis"
            required_tools = ["fundamental_analysis"]
        elif any(word in message_lower for word in ['complete', 'full', 'comprehensive', 'detailed', 'deep dive']):
            intent = "comprehensive_analysis"
            required_tools = ["technical_analysis", "news_sentiment", "fundamental_analysis"]
        elif any(word in message_lower for word in ['news', 'headlines', 'sentiment']):
            intent = "news"
            required_tools = ["news_sentiment"]
        elif any(word in message_lower for word in ['portfolio', 'positions', 'holdings']):
            intent = "portfolio"
            required_tools = ["portfolio_check"]
        elif any(word in message_lower for word in ['find', 'screen', 'search', 'discover']):
            intent = "screener"
            required_tools = ["stock_screener"]
        elif symbols or any(word in message_lower for word in ['good pick', 'investment', 'buy', 'sell', 'analyze']):
            intent = "analyze"
            required_tools = ["technical_analysis", "news_sentiment"]
        else:
            intent = "general"
            required_tools = []
        
        return {
            "intent": intent,
            "symbols": symbols,
            "confidence": 0.4,  # Lower confidence for fallback
            "requires_tools": required_tools,
            "fallback": True
        }
    
    def _detect_conversation_context(self, message: str, user_profile: Dict) -> Dict:
        """Detect the human conversation context and emotional subtext"""
        
        msg_lower = message.lower()
        
        # Detect if user actually greeted
        greeting_words = ['hey', 'hi', 'hello', 'yo', 'sup', 'what\'s up', 'howdy']
        user_greeted = any(msg_lower.startswith(word) or f" {word}" in f" {msg_lower}" for word in greeting_words)
        
        context = {
            "user_greeted_first": user_greeted,
            "is_direct_question": message.count('?') > 0 or any(word in msg_lower for word in ['analyze', 'analysis', 'assessment', 'thoughts', 'technical', 'what\'s', 'how\'s']),
            "is_celebrating": any(word in msg_lower for word in ['moon', 'rocket', '🚀', 'lfg', 'holy', 'damn']),
            "is_worried": any(word in msg_lower for word in ['dump', 'sell', 'scared', 'worried', 'shit', 'fuck']),
            "is_seeking_validation": message.count('?') > 0 and any(word in msg_lower for word in ['thoughts', 'think', 'should i', 'good pick']),
            "has_fomo": any(word in msg_lower for word in ['fomo', 'missing out', 'too late', 'should i buy']),
            "energy_indicators": {
                "high": message.count('!') + message.count('🚀') + len([w for w in msg_lower.split() if w.isupper()]),
                "curse_words": sum(1 for word in ['shit', 'fuck', 'damn', 'holy'] if word in msg_lower)
            }
        }
        
        return context

    async def generate_response(
        self, 
        user_message: str,
        intent_data: Dict,
        tool_results: Dict,
        user_phone: str,
        user_profile: Dict = None
    ) -> str:
        """Generate human-like response with rich conversation context"""
        
        # NEW: Get conversation context from database
        conversation_context = {}
        if self.database_service and user_phone:
            try:
                conversation_context = await self.database_service.get_conversation_context(user_phone)
            except Exception as e:
                logger.warning(f"Failed to get conversation context for response: {e}")
        
        # Get comprehensive personality context (existing functionality enhanced)
        personality_context = self._build_personality_context(user_profile, user_message, conversation_context)
        
        # Detect conversation context and emotional state
        conversation_state = self._detect_conversation_context(user_message, user_profile or {})
        
        # Build human-like analysis context
        analysis_context = self._build_human_analysis_context(tool_results, intent_data)
        
        # NEW: Enhanced response strategy with conversation continuity
        response_strategy = self._determine_enhanced_response_strategy(intent_data, conversation_state, user_profile, conversation_context)
        
        # Determine formality level from message content
        casual_indicators = ['hey', 'yo', 'sup', 'haha', 'lol', 'rn', 'gonna', 'wanna', 'throwing', 'bucks', 'few bucks', 'thinking about']
        formal_indicators = ['financial investment', 'analysis', 'recommendation', 'portfolio allocation', 'could you', 'would you']
        
        casual_score = sum(1 for indicator in casual_indicators if indicator in user_message.lower())
        formal_score = sum(1 for indicator in formal_indicators if indicator in user_message.lower())
        
        detected_formality = "CASUAL" if casual_score > formal_score else "PROFESSIONAL"
        
        # Determine analysis depth required
        analysis_depth_indicators = ['analytical assessment', 'technical perspective', 'comprehensive', 'detailed analysis', 'investment potential', 'deep dive']
        requires_deep_analysis = any(indicator in user_message.lower() for indicator in analysis_depth_indicators)
        
        # NEW: Enhanced prompt with conversation context
        prompt = f"""You're their trading partner in an ongoing conversation. Use conversation history for natural continuity.

PERSONALITY & RELATIONSHIP CONTEXT:
{personality_context}

THEIR CURRENT MESSAGE: "{user_message}"

MESSAGE ANALYSIS:
- Length: {len(user_message)} characters
- Style: {detected_formality}
- Depth Required: {"COMPREHENSIVE" if requires_deep_analysis else "STANDARD"}
- User Greeted: {conversation_state.get('user_greeted_first', False)}

CONVERSATION CONTINUITY:
{self._format_conversation_continuity(conversation_context)}

MARKET DATA:
{analysis_context}

RESPONSE STRATEGY: {response_strategy}

CRITICAL EMOJI RULES - NEVER VIOLATE:
🚫 **NEVER use ANY emojis if user used ZERO emojis**
🚫 **NEVER use ANY emojis if their message is over 100 characters**  
✅ **ONLY use emojis if: message under 100 chars AND user used emojis**

PERSONALITY MATCHING RULES:
- **Professional/Formal Input** → Professional response, no slang, proper grammar
- **Casual Input** → Casual response, contractions okay  
- **Their formality level**: {"PROFESSIONAL" if len(user_message) > 40 and any(word in user_message.lower() for word in ["investment", "financial", "analysis", "recommendation"]) else "CASUAL"}

HUMAN RESPONSE RULES:

1. **MATCH THEIR EXACT STYLE**: If formal → be formal, if casual → be casual
2. **NO EMOJIS UNLESS THEY USED THEM**: Zero tolerance emoji policy
3. **USE CONVERSATION HISTORY**: Reference previous discussions naturally
4. **NO AI ARTIFACTS**: Never say "here's the analysis" - just dive in naturally
5. **CONTEXTUAL GREETINGS**: Only greet if appropriate based on relationship and today's conversation

RESPONSE EXAMPLES BY DETECTED STYLE:

**COMPREHENSIVE ANALYSIS** (when they ask for "analytical assessment", "technical perspective"):
"GOOGL technical setup: $193.18, RSI 58 (neutral-bullish), MACD positive divergence, breaking above 20-day MA resistance at $191. Key levels: support $188, resistance $197. Target range $200-205. Mixed news sentiment but solid fundamentals. Risk/reward favors upside with stop below $188."

**CASUAL ANALYSIS** (when they ask simple questions):  
"GOOGL looking solid at $193, RSI neutral so room to run. breaking key resistance, next target $200"

**CONTINUING CONVERSATION** (when referencing previous discussion):
"GOOGL update: still holding above that $191 support we talked about. RSI cooled off to 58, so that overbought concern from yesterday is gone. Looking good for that $200 target."

**PROFESSIONAL DETAILED** (formal analytical requests):
"GOOGL exhibits constructive technical characteristics at $193.18. RSI positioning at 58 indicates balanced momentum with upside capacity. MACD crossover confirms bullish divergence. Price action above 20-day moving average establishes support at $191 level."

FOR THIS SPECIFIC MESSAGE:
- Response style: {detected_formality} with {"COMPREHENSIVE" if requires_deep_analysis else "STANDARD"} depth
- Use conversation context naturally for continuity
- Include specific levels, indicators, timeframes if analysis requested
- NO emojis unless user message is under 100 chars AND user used emojis

SMS OPTIMIZATION:
- Keep under 300 chars but pack maximum value
- Can split into 2 messages if needed
- Use abbreviations naturally (tho, rn, gonna, etc.) ONLY if user is casual
- Match their formality level exactly
- Reference conversation history naturally when relevant

Generate the perfect contextual human response:"""

        try:
            if hasattr(self.openai_client, 'client'):
                response = await self.openai_client.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,  # Higher temperature for more natural, human-like responses
                    max_tokens=250  # Increased for context-rich responses
                )
            else:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=250
                )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Apply human-like post-processing
            generated_response = self._humanize_response(generated_response, user_profile, conversation_state)
            
            # Clean and validate response
            generated_response = self._clean_response(generated_response)
            generated_response = self._validate_response(generated_response)
            
            # SMS optimization
            if len(generated_response) > 320:
                generated_response = self._split_for_sms(generated_response)
            
            # NEW: Save enhanced message with context to database
            if self.database_service and user_phone:
                try:
                    await self.database_service.save_enhanced_message(
                        phone_number=user_phone,
                        user_message=user_message,
                        bot_response=generated_response,
                        intent_data=intent_data,
                        symbols=intent_data.get("symbols", []),
                        context_used=conversation_context.get("context_summary", "")
                    )
                except Exception as e:
                    logger.warning(f"Failed to save enhanced message: {e}")
            
            logger.info(f"Generated context-rich response: {len(generated_response)} chars")
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_human_fallback(intent_data, tool_results, user_profile, conversation_state)
    
    # NEW: Enhanced context formatting methods
    def _format_conversation_continuity(self, conversation_context: Dict) -> str:
        """Format conversation continuity information for response generation"""
        if not conversation_context:
            return "- New conversation, respond appropriately for first interaction"
        
        continuity_lines = []
        
        # Today's session status
        today_session = conversation_context.get("today_session", {})
        if today_session.get("is_first_message_today", True):
            continuity_lines.append("- First message today - natural greeting may be appropriate")
        else:
            msg_count = today_session.get("message_count", 0)
            continuity_lines.append(f"- Continuing today's conversation ({msg_count} messages so far)")
            
            # Today's topics for continuity
            topics_today = today_session.get("topics_discussed", [])
            if topics_today:
                continuity_lines.append(f"- Topics discussed today: {', '.join(topics_today)}")
        
        # Recent conversation references
        conv_ctx = conversation_context.get("conversation_context", {})
        recent_symbols = conv_ctx.get("recent_symbols", [])
        if recent_symbols:
            continuity_lines.append(f"- Can reference recent symbols: {', '.join(recent_symbols[-3:])}")
        
        # Relationship stage affects conversation style
        relationship = conv_ctx.get("relationship_stage", "new")
        total_convos = conv_ctx.get("total_conversations", 0)
        continuity_lines.append(f"- Relationship level: {relationship} ({total_convos} total conversations)")
        
        return "\n".join(continuity_lines)
    
    def _build_personality_context(self, user_profile: Dict, user_message: str, conversation_context: Dict = None) -> str:
        """Build comprehensive personality context including conversation history"""
        
        if not user_profile:
            context_parts = ["New user - be friendly and welcoming"]
        else:
            comm_style = user_profile.get('communication_style', {})
            trading_style = user_profile.get('trading_personality', {})
            
            context_parts = [
                f"COMMUNICATION: {comm_style.get('formality', 'casual')}/{comm_style.get('energy', 'moderate')} energy",
                f"TRADING: {trading_style.get('experience_level', 'intermediate')} {trading_style.get('trading_style', 'swing')} trader"
            ]
        
        # Add conversation context if available
        if conversation_context:
            conv_ctx = conversation_context.get("conversation_context", {})
            
            # Relationship context
            relationship = conv_ctx.get("relationship_stage", "new")
            total_convos = conv_ctx.get("total_conversations", 0)
            context_parts.append(f"RELATIONSHIP: {relationship} ({total_convos} conversations)")
            
            # Recent context for natural references
            recent_symbols = conv_ctx.get("recent_symbols", [])
            if recent_symbols:
                context_parts.append(f"RECENT FOCUS: {', '.join(recent_symbols[-3:])}")
            
            # Conversation frequency affects response style
            frequency = conv_ctx.get("conversation_frequency", "occasional")
            context_parts.append(f"FREQUENCY: {frequency}")
        
        return "\n".join(context_parts)
    
    def _determine_enhanced_response_strategy(self, intent_data: Dict, conversation_state: Dict, 
                                            user_profile: Dict, conversation_context: Dict) -> str:
        """Determine response strategy with conversation context awareness"""
        
        emotional_state = intent_data.get('emotional_state', 'neutral')
        
        # Check conversation continuity
        today_session = conversation_context.get("today_session", {}) if conversation_context else {}
        is_first_today = today_session.get("is_first_message_today", True)
        relationship = conversation_context.get("conversation_context", {}).get("relationship_stage", "new") if conversation_context else "new"
        
        # Greeting logic based on context
        if conversation_state.get('user_greeted_first'):
            if is_first_today or relationship == "new":
                return "CONTEXTUAL GREETING - they greeted first, appropriate to greet back and continue"
            else:
                return "FRIENDLY ACKNOWLEDGMENT - brief greeting then dive into their request"
        
        # For direct questions/analysis requests - context-aware responses
        if conversation_state.get('is_direct_question'):
            if not is_first_today and today_session.get("message_count", 0) > 2:
                return "CONTINUING CONVERSATION - reference previous discussion, no greeting needed"
            else:
                return "DIRECT ANALYSIS - jump straight to analysis, use conversation history if relevant"
        
        # Emotional states with context awareness
        if conversation_state.get('is_celebrating'):
            return "CELEBRATION MODE - match excitement, reference their trading journey if established relationship"
        
        elif conversation_state.get('is_worried'):
            support_level = "deep reassurance" if relationship in ["building_trust", "established"] else "gentle support"
            return f"SUPPORT MODE - provide {support_level}, reference past successes if known"
        
        elif conversation_state.get('is_seeking_validation'):
            return "VALIDATION MODE - honest assessment, use relationship history to build confidence"
        
        elif conversation_state.get('has_fomo'):
            return "FOMO CHECK - rational perspective, reference their typical investment approach if known"
        
        else:
            continuity = "with conversation continuity" if not is_first_today else "as fresh interaction"
            return f"ANALYTICAL MODE - provide solid analysis {continuity}, no unnecessary greetings"
    
    def _build_human_analysis_context(self, tool_results: Dict, intent_data: Dict) -> str:
        """Format analysis data in human-digestible way"""
        
        if not tool_results:
            return "No market data available right now"
        
        context = ""
        
        # Technical analysis in human terms
        if "technical_analysis" in tool_results:
            ta_data = tool_results["technical_analysis"]
            if ta_data:
                for symbol, data in ta_data.items():
                    price_info = data.get('price', {})
                    indicators = data.get('indicators', {})
                    
                    current_price = price_info.get('current', 'N/A')
                    change_pct = price_info.get('change_percent', 0)
                    
                    # Humanize technical indicators
                    rsi = indicators.get('RSI', {}).get('value')
                    rsi_human = (
                        "overbought (might pullback)" if rsi and rsi > 70 else
                        "oversold (might bounce)" if rsi and rsi < 30 else
                        "neutral territory" if rsi else "RSI unavailable"
                    )
                    
                    context += f"{symbol}: ${current_price} ({change_pct:+.1f}%) - {rsi_human}\n"
        
        # News sentiment in human terms
        if "news_sentiment" in tool_results:
            news_data = tool_results["news_sentiment"]
            if news_data:
                for symbol, sentiment_info in news_data.items():
                    sentiment_score = sentiment_info.get('sentiment_score', 0)
                    sentiment_human = (
                        "bullish news flow" if sentiment_score > 0.3 else
                        "bearish headlines" if sentiment_score < -0.3 else
                        "mixed news sentiment"
                    )
                    context += f"{symbol} news: {sentiment_human}\n"
        
        # Handle unavailable data humanely
        if "technical_analysis_unavailable" in tool_results:
            context += "Market data's being weird rn, but I can still help\n"
        
        if "news_sentiment_unavailable" in tool_results:
            context += "News feeds are slow today\n"
        
        return context or "Limited data available but we can work with it"
    
    def _humanize_response(self, response: str, user_profile: Dict, conversation_context: Dict) -> str:
        """Apply final human touches to the response"""
        
        if not user_profile:
            return response
        
        style = user_profile.get('communication_style', {})
        
        # CRITICAL: Only add contractions if user is casual
        if style.get('formality') == 'casual':
            # Add natural contractions and flow
            response = response.replace(" is ", "'s ")
            response = response.replace(" are ", "'re ")
            response = response.replace(" would ", "'d ")
            response = response.replace(" will ", "'ll ")
            response = response.replace("though", "tho")
            response = response.replace("right now", "rn")
            response = response.replace("going to", "gonna")
        
        # Adjust punctuation for energy level
        if style.get('energy') == 'high' and not response.endswith('!'):
            if response.endswith('.'):
                response = response[:-1] + '!'
        
        # Add natural transition words occasionally ONLY for casual users
        if len(response) > 100 and style.get('formality') == 'casual':
            import random
            if random.random() < 0.3:  # 30% chance
                transition = random.choice(['btw', 'also', 'oh and'])
                # Find a good spot to insert it (after a comma or period)
                if ', ' in response:
                    response = response.replace(', ', f', {transition} ', 1)
        
        return response
    
    def _split_for_sms(self, response: str) -> str:
        """Split long responses naturally for SMS"""
        
        sentences = response.split('. ')
        if len(sentences) <= 1:
            return response[:317] + "..."
        
        # Find natural breaking point
        mid_point = len(sentences) // 2
        first_part = '. '.join(sentences[:mid_point])
        second_part = '. '.join(sentences[mid_point:])
        
        # Ensure both parts are SMS-friendly
        if len(first_part) <= 300 and len(second_part) <= 300:
            return f"{first_part}\n\n{second_part}"
        
        # Fallback: just truncate
        return response[:317] + "..."
    
    def _clean_response(self, response: str) -> str:
        """Clean LLM response artifacts that shouldn't go to users"""
        
        # Remove meta-instructions
        patterns_to_remove = [
            r"Certainly!.*?for the user:\s*",
            r"Here's the.*?response.*?:\s*",
            r"Based on.*?here's.*?:\s*",
            r"Given.*?here's.*?:\s*",
            r"I'll.*?response.*?:\s*",
            r"Let me.*?response.*?:\s*",
            r"Here's the tailored response.*?:\s*",
            r".*?tailored response.*?:\s*",
            r"Hey there!.*?\s*",
            r"Hello!.*?\s*",
            r"Hi!.*?\s*"
        ]
        
        for pattern in patterns_to_remove:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove quotes around the entire response
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Remove extra whitespace and newlines at start
        response = response.strip()
        
        return response
    
    def _validate_response(self, response: str) -> str:
        """Validate response doesn't contain artifacts"""
        
        # Check for common artifacts
        artifacts = [
            "here's the", "certainly", "based on", "given", 
            "let me", "i'll provide", "tailored response", "hey there", "hello", "hi"
        ]
        
        response_lower = response.lower()
        if any(artifact in response_lower for artifact in artifacts):
            logger.warning(f"Response contains artifacts: {response[:50]}...")
            # Apply cleaning
            response = self._clean_response(response)
        
        return response
    
    def _serialize_fundamental_data(self, fundamental_data: Dict) -> Dict:
        """Convert FundamentalAnalysisResult objects to JSON-serializable format"""
        serializable_data = {}
        
        for symbol, analysis_result in fundamental_data.items():
            try:
                if hasattr(analysis_result, 'symbol'):
                    # This is a FundamentalAnalysisResult object
                    serializable_data[symbol] = {
                        "symbol": analysis_result.symbol,
                        "overall_score": analysis_result.overall_score,
                        "financial_health": analysis_result.financial_health.value if hasattr(analysis_result.financial_health, 'value') else str(analysis_result.financial_health),
                        "current_price": analysis_result.current_price,
                        "strength_areas": analysis_result.strength_areas,
                        "concern_areas": analysis_result.concern_areas,
                        "bull_case": analysis_result.bull_case,
                        "bear_case": analysis_result.bear_case,
                        "data_completeness": analysis_result.data_completeness
                    }
                    
                    # Add ratios if available
                    if analysis_result.ratios:
                        serializable_data[symbol]["ratios"] = {
                            "pe_ratio": getattr(analysis_result.ratios, 'pe_ratio', None),
                            "roe": getattr(analysis_result.ratios, 'roe', None),
                            "debt_to_equity": getattr(analysis_result.ratios, 'debt_to_equity', None),
                            "current_ratio": getattr(analysis_result.ratios, 'current_ratio', None)
                        }
                    
                    # Add growth metrics if available
                    if analysis_result.growth:
                        serializable_data[symbol]["growth"] = {
                            "revenue_growth_1y": getattr(analysis_result.growth, 'revenue_growth_1y', None),
                            "earnings_growth_1y": getattr(analysis_result.growth, 'earnings_growth_1y', None)
                        }
                else:
                    # Already serializable
                    serializable_data[symbol] = analysis_result
            except Exception as e:
                # Fallback: just include basic info
                serializable_data[symbol] = {"error": f"Serialization failed: {str(e)}"}
        
        return serializable_data
    
    def _check_if_first_message_of_day(self, user_phone: str) -> bool:
        """Check if this is the first message from this user today"""
        if not user_phone or not self.personality_engine:
            return False
            
        profile = self.personality_engine.get_user_profile(user_phone)
        if not profile:
            return True
            
        # Simple check - if user has fewer than 3 total messages, consider it early interaction
        total_messages = profile.get('learning_data', {}).get('total_messages', 0)
        return total_messages < 3
    
    def _generate_human_fallback(self, intent_data: Dict, tool_results: Dict, user_profile: Dict, conversation_context: Dict) -> str:
        """Generate human-like fallback when AI fails"""
        
        style = user_profile.get('communication_style', {}) if user_profile else {}
        formality = style.get('formality', 'casual')
        energy = style.get('energy', 'moderate')
        
        # Emotional fallbacks based on detected state
        if conversation_context.get('is_worried'):
            if formality == 'casual':
                return "hey market data's being wonky rn but don't stress - we'll figure this out together"
            else:
                return "Market data is temporarily unavailable, but I'm here to help you work through your concerns."
        
        elif conversation_context.get('is_celebrating'):
            if formality == 'casual':
                return "yooo can't pull the exact numbers rn but sounds like you're crushing it!"
            else:
                return "Congratulations on your success! I'll have updated analysis for you shortly."
        
        elif intent_data.get("symbols"):
            symbol = intent_data["symbols"][0]
            if formality == 'casual' and energy == 'high':
                return f"yo {symbol} data's loading slow but I got you! gimme a sec to grab fresh numbers"
            else:
                return f"Retrieving latest {symbol} analysis - one moment please."
        
        else:
            if formality == 'casual':
                return "data's being slow rn but I'm still here to help! what's on your mind?"
            else:
                return "I'm experiencing some technical difficulties but I'm here to assist however I can."


# ===== ENHANCED PERSONALITY-AWARE RESPONSE GENERATOR =====

class PersonalityAwareResponseGenerator:
    """Generates responses that sound like a real human trading buddy"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
        # Human conversation templates by personality
        self.conversation_templates = {
            'casual_excited': [
                "yooo {symbol} is {action}! 🚀 {price_context} {analysis} {recommendation}",
                "damn {symbol} {action} hard! {price_context} {analysis} {recommendation}",
                "holy shit {symbol}! {price_context} {analysis} {recommendation}"
            ],
            'casual_chill': [
                "yo {symbol}'s {action}, {price_context} {analysis} {recommendation}",
                "{symbol} looking {status} rn. {price_context} {analysis} {recommendation}",
                "so {symbol} {action} today. {price_context} {analysis} {recommendation}"
            ],
            'professional': [
                "{symbol} is {action} at {price_context} {analysis} {recommendation}",
                "Regarding {symbol}: {price_context} {analysis} {recommendation}",
                "{symbol} analysis: {price_context} {analysis} {recommendation}"
            ]
        }
    
    async def generate_personality_matched_response(
        self, 
        user_message: str,
        analysis_data: Dict,
        user_profile: Dict,
        user_phone: str
    ) -> str:
        """Generate response that feels like texting your knowledgeable trading friend"""
        
        # Build comprehensive human context
        human_context = self._build_human_conversation_context(user_profile, user_message, analysis_data)
        
        prompt = f"""You're responding as their actual trading buddy - someone who knows them personally and talks exactly like they do.

{human_context}

THEIR MESSAGE: "{user_message}"

RESPOND LIKE A REAL HUMAN:

1. **NO CORPORATE SPEAK**: Don't say "analysis indicates" or "data suggests" - just tell them what's happening
2. **USE THEIR LANGUAGE**: If they say "yo" you say "yo", if they curse appropriately you can too
3. **BE CONVERSATIONAL**: Like you're continuing an ongoing text conversation with a friend
4. **SHOW PERSONALITY**: Have opinions, show excitement/concern, be relatable
5. **REFERENCE SHARED CONTEXT**: If you've talked before, act like it

RESPONSE EXAMPLES BY THEIR STYLE:

**Casual/Excited Friend**:
"yooo SLV absolutely sending it! 🚀 $21.47 but overbought af, RSI at 72. might need a breather soon. you holding or taking profits?"

**Chill/Casual Buddy**:
"SLV looking solid rn, sitting at $21.47 with room to run. RSI only at 55 so not overbought yet. mixed news tho, might see some chop"

**Professional Friend**:
"Silver ETF performing well at $21.47, technically sound with RSI at 55 indicating room for continued upside. However, mixed sentiment may introduce volatility."

**Worried/Supportive**:
"hey don't panic on SLV - yeah it's volatile but that's just silver being silver. fundamentals still solid, just market being market"

Generate their perfect response (under 320 chars):"""

        try:
            if hasattr(self.openai_client, 'chat'):
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,  # High temperature for natural human variation
                    max_tokens=200
                )
            else:
                response = await self.openai_client.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=200
                )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Apply human finishing touches
            generated_response = self._apply_human_finishing_touches(generated_response, user_profile)
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Personality response generation failed: {e}")
            return self._generate_human_fallback_response(user_profile, analysis_data)
    
    def _build_human_conversation_context(self, user_profile: Dict, user_message: str, analysis_data: Dict) -> str:
        """Build natural conversation context like a human would remember"""
        
        if not user_profile:
            return "This is a new friend - be welcoming but not overly eager"
        
        # Extract human-relevant details
        style = user_profile.get('communication_style', {})
        trading = user_profile.get('trading_personality', {})
        history = user_profile.get('learning_data', {})
        memory = user_profile.get('context_memory', {})
        
        # Build friendship context
        total_convos = history.get('total_messages', 0)
        friendship_level = (
            "just met them" if total_convos < 3 else
            "getting to know them" if total_convos < 15 else
            "good friends" if total_convos < 50 else
            "close trading buddies"
        )
        
        # Recent conversation memory
        recent_stocks = memory.get('last_discussed_stocks', [])
        conversation_history = f"Recently talked about: {', '.join(recent_stocks[:3])}" if recent_stocks else "First conversation about stocks"
        
        # Their personality traits (how they actually talk)
        personality_traits = f"""
FRIENDSHIP LEVEL: {friendship_level} ({total_convos} messages)
HOW THEY TALK: {style.get('formality', 'casual')} + {style.get('energy', 'moderate')} energy + {style.get('emoji_usage', 'some')} emojis
TRADING STYLE: {trading.get('experience_level', 'intermediate')} {trading.get('trading_style', 'swing')} trader, {trading.get('risk_tolerance', 'moderate')} risk
CONVERSATION HISTORY: {conversation_history}
THEIR WINS/LOSSES: {history.get('successful_trades_mentioned', 0)}W/{history.get('loss_trades_mentioned', 0)}L mentioned
THEIR CONCERNS: {', '.join(memory.get('concerns_expressed', [])[:2]) if memory.get('concerns_expressed') else 'None mentioned yet'}
"""
        
        # Market data in human context
        market_context = ""
        if analysis_data:
            market_context = f"MARKET DATA AVAILABLE:\n{self._humanize_market_data(analysis_data)}"
        else:
            market_context = "MARKET DATA: Limited/unavailable - acknowledge this naturally"
        
        return f"{personality_traits}\n{market_context}"
    
    def _humanize_market_data(self, analysis_data: Dict) -> str:
        """Convert technical data into human-friendly talking points"""
        
        human_points = []
        
        if "technical_analysis" in analysis_data:
            ta_data = analysis_data["technical_analysis"]
            for symbol, data in ta_data.items():
                price_info = data.get('price', {})
                indicators = data.get('indicators', {})
                
                # Price action in human terms
                current = price_info.get('current')
                change_pct = price_info.get('change_percent', 0)
                
                if current:
                    direction = "pumping" if change_pct > 2 else "climbing" if change_pct > 0 else "dipping" if change_pct < -2 else "flat"
                    human_points.append(f"{symbol} ${current} {direction} ({change_pct:+.1f}%)")
                
                # RSI in human terms
                rsi = indicators.get('RSI', {}).get('value')
                if rsi:
                    rsi_human = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
                    human_points.append(f"RSI {rsi:.0f} ({rsi_human})")
        
        if "news_sentiment" in analysis_data:
            news_data = analysis_data["news_sentiment"]
            for symbol, sentiment_info in news_data.items():
                sentiment = sentiment_info.get('sentiment_score', 0)
                news_human = "bullish news" if sentiment > 0.2 else "bearish headlines" if sentiment < -0.2 else "mixed news"
                human_points.append(f"{symbol} {news_human}")
        
        return " | ".join(human_points) if human_points else "Data looking thin today"
    
    def _apply_human_finishing_touches(self, response: str, user_profile: Dict) -> str:
        """Add natural human touches to make it feel authentic"""
        
        if not user_profile:
            return response
        
        style = user_profile.get('communication_style', {})
        
        # Natural text message abbreviations ONLY for casual users
        if style.get('formality') == 'casual':
            replacements = {
                'though': 'tho',
                'right now': 'rn', 
                'probably': 'prob',
                'definitely': 'def',
                'because': 'bc',
                'your': 'ur',
                'you are': "you're",
                'going to': 'gonna'
            }
            
            for formal, casual in replacements.items():
                response = response.replace(formal, casual)
        
        # Energy level adjustments
        if style.get('energy') == 'high':
            # Add emphasis occasionally
            if 'good' in response:
                response = response.replace('good', 'solid')
            if not response.endswith('!') and not response.endswith('?'):
                response += '!'
        
        # Remove any remaining AI artifacts
        ai_phrases = [
            "based on the data", "according to analysis", "the data shows",
            "analysis indicates", "technical analysis suggests"
        ]
        
        for phrase in ai_phrases:
            response = response.replace(phrase, "")
        
        return response.strip()
    
    def _generate_human_fallback_response(self, user_profile: Dict, analysis_data: Dict) -> str:
        """Human fallback when generation fails"""
        
        style = user_profile.get('communication_style', {}) if user_profile else {}
        formality = style.get('formality', 'casual')
        
        if formality == 'casual':
            fallbacks = [
                "yo data's being weird rn but I got you! what you thinking about?",
                "market feeds acting up but I'm still here to help! what's up?",
                "tech issues on my end but we can still chat - what's on your mind?"
            ]
        else:
            fallbacks = [
                "I'm experiencing some technical difficulties, but I'm here to help with your trading questions.",
                "Market data is temporarily unavailable, but I can still assist you.",
                "Having some connectivity issues but I'm available for any questions you have."
            ]
        
        import random
        return random.choice(fallbacks)


# ===== ENHANCED MESSAGE PROCESSOR WITH CONTEXT AWARENESS =====

class ComprehensiveMessageProcessor:
    """Processes messages like a human trading buddy with conversation memory"""
    
    def __init__(self, openai_client, ta_service, personality_engine, database_service=None, news_service=None, fundamental_tool=None):
        self.trading_agent = TradingAgent(openai_client, personality_engine, database_service)  # NEW: Pass database service
        self.tool_executor = ToolExecutor(ta_service, None, None, news_service, fundamental_tool)
        self.response_generator = PersonalityAwareResponseGenerator(openai_client)
        self.personality_engine = personality_engine
        self.database_service = database_service  # NEW: Store database service
    
    async def process_complete_message(self, message: str, user_phone: str) -> str:
        """Process message with rich conversation context"""
        
        try:
            logger.info(f"🤝 Context-rich processing: '{message}' from {user_phone}")
            
            # Step 1: Understand the human context and intent (now with conversation history)
            intent_data = await self.trading_agent.parse_intent(message, user_phone)
            
            # Step 2: Learn from this interaction (existing functionality)
            if self.personality_engine:
                self.personality_engine.learn_from_message(user_phone, message, intent_data)
            
            # Step 3: Get tools/analysis if needed
            tool_results = await self.tool_executor.execute_tools(intent_data, user_phone)
            
            # Step 4: Get their personality (existing functionality)
            user_profile = self.personality_engine.get_user_profile(user_phone) if self.personality_engine else {}
            
            # Step 5: Generate context-rich response
            response = await self.trading_agent.generate_response(
                user_message=message,
                intent_data=intent_data,
                tool_results=tool_results,
                user_phone=user_phone,
                user_profile=user_profile
            )
            
            logger.info(f"✅ Context-rich response generated: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"💥 Context-rich processing failed: {e}")
            # Even fallback should sound human
            return "yo something's wonky on my end rn but I'm still here! try me again in a sec?"


# ===== TOOL EXECUTOR (Same as before but with enhanced error handling) =====

class ToolExecutor:
    """Handles execution of various trading tools based on intent"""
    
    def __init__(self, ta_service, portfolio_service, screener_service=None, news_service=None, fundamental_tool=None):
        self.ta_service = ta_service
        self.portfolio_service = portfolio_service
        self.screener_service = screener_service
        self.news_service = news_service
        self.fundamental_tool = fundamental_tool
    
    async def execute_tools(self, intent_data: Dict, user_phone: str) -> Dict[str, Any]:
        """Execute required tools based on parsed intent"""
        
        results = {}
        required_tools = intent_data.get("requires_tools", [])
        
        # Execute tools in parallel when possible
        tasks = []
        
        if "technical_analysis" in required_tools and intent_data.get("symbols"):
            tasks.append(self._execute_technical_analysis(intent_data["symbols"]))
        
        if "portfolio_check" in required_tools:
            tasks.append(self._execute_portfolio_check(user_phone))
        
        if "stock_screener" in required_tools:
            tasks.append(self._execute_stock_screener(intent_data.get("parameters", {})))
        
        if "news_sentiment" in required_tools and intent_data.get("symbols"):
            tasks.append(self._execute_news_sentiment(intent_data["symbols"]))
        
        if "fundamental_analysis" in required_tools and intent_data.get("symbols"):
            tasks.append(self._execute_fundamental_analysis(intent_data["symbols"], user_phone))
        
        # Wait for all tools to complete
        if tasks:
            tool_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(tool_results):
                if isinstance(result, Exception):
                    logger.error(f"Tool execution failed: {result}")
                    results["error"] = str(result)
                else:
                    results.update(result)
        
        return results
    
    async def _execute_technical_analysis(self, symbols: List[str]) -> Dict:
        """Execute technical analysis for symbols"""
        try:
            if not self.ta_service:
                return {"technical_analysis_unavailable": True}
            
            ta_results = {}
            for symbol in symbols[:3]:  # Limit to 3 symbols max
                ta_data = await self.ta_service.analyze_symbol(symbol.upper())
                if ta_data:
                    ta_results[symbol] = ta_data
            
            return {"technical_analysis": ta_results} if ta_results else {"technical_analysis_unavailable": True}
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"technical_analysis_unavailable": True}
    
    async def _execute_portfolio_check(self, user_phone: str) -> Dict:
        """Execute portfolio check for user"""
        try:
            if not self.portfolio_service:
                return {"portfolio_unavailable": True}
            
            portfolio_data = await self.portfolio_service.get_user_portfolio(user_phone)
            return {"portfolio": portfolio_data} if portfolio_data else {"portfolio_unavailable": True}
            
        except Exception as e:
            logger.error(f"Portfolio check failed: {e}")
            return {"portfolio_unavailable": True}
    
    async def _execute_stock_screener(self, parameters: Dict) -> Dict:
        """Execute stock screening with parameters"""
        try:
            if not self.screener_service:
                return {"screener_unavailable": True}
            
            screening_results = await self.screener_service.screen_stocks(parameters)
            return {"screener_results": screening_results} if screening_results else {"screener_unavailable": True}
            
        except Exception as e:
            logger.error(f"Stock screening failed: {e}")
            return {"screener_unavailable": True}
    
    async def _execute_news_sentiment(self, symbols: List[str]) -> Dict:
        """Execute news sentiment analysis for symbols"""
        try:
            if not self.news_service:
                return {"news_sentiment_unavailable": True}
            
            news_results = {}
            for symbol in symbols[:3]:
                try:
                    news_data = await self.news_service.get_sentiment(symbol.upper())
                    if news_data and not news_data.get('error'):
                        news_results[symbol] = news_data
                except Exception as e:
                    logger.warning(f"News sentiment failed for {symbol}: {e}")
                    continue
            
            return {"news_sentiment": news_results} if news_results else {"news_sentiment_unavailable": True}
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {"news_sentiment_unavailable": True}
    
    async def _execute_fundamental_analysis(self, symbols: List[str], user_phone: str) -> Dict:
        """Execute fundamental analysis for symbols"""
        try:
            if not self.fundamental_tool:
                return {"fundamental_analysis_unavailable": True}
            
            fundamental_results = {}
            for symbol in symbols[:2]:
                try:
                    depth = "standard"
                    user_style = "casual"
                    
                    fund_result = await self.fundamental_tool.execute({
                        "symbol": symbol.upper(),
                        "depth": depth,
                        "user_style": user_style
                    })
                    
                    if fund_result.get("success"):
                        fundamental_results[symbol] = fund_result["analysis_result"]
                except Exception as e:
                    logger.warning(f"Fundamental analysis failed for {symbol}: {e}")
                    continue
            
            return {"fundamental_analysis": fundamental_results} if fundamental_results else {"fundamental_analysis_unavailable": True}
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed: {e}")
            return {"fundamental_analysis_unavailable": True}
