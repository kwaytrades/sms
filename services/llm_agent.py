# services/llm_agent.py - ENHANCED INTELLIGENT ORCHESTRATOR AGENT

import json
import asyncio
import re
from typing import Dict, List, Optional, Any
from loguru import logger
import openai
from datetime import datetime
from openai import AsyncOpenAI

class ComprehensiveOrchestrator:
    """
    Enhanced comprehensive orchestrator with intelligent engine selection,
    conversation context awareness, and improved fundamental analysis detection
    """
    
    def __init__(self, openai_client, personality_engine, cache_service=None):
        self.openai_client = openai_client
        self.personality_engine = personality_engine
        self.cache_service = cache_service
        
        # Enhanced keyword detection for better engine selection
        self.fundamental_keywords = [
            "fundamental", "fundamentals", "earnings", "revenue", "profit", "pe ratio", 
            "valuation", "balance sheet", "income statement", "financial health",
            "debt", "cash flow", "book value", "dividend", "growth rate", "eps",
            "financial", "ratios", "p/e", "debt ratio", "roa", "roe"
        ]
        
        self.technical_keywords = [
            "technical", "chart", "resistance", "support", "rsi", "macd", 
            "moving average", "breakout", "pattern", "trend", "momentum"
        ]
        
        self.news_keywords = [
            "news", "updates", "happened", "headlines", "sentiment", "market news"
        ]
        
    async def orchestrate(self, user_message: str, user_phone: str) -> Dict[str, Any]:
        """
        Enhanced orchestration with intelligent engine selection and context awareness
        """
        
        # Step 1: Gather enhanced user context from cache
        user_context = await self._gather_enhanced_user_context(user_phone)
        
        # Step 2: Analyze intent with full context awareness
        intent_analysis = await self._analyze_intent_with_context(user_message, user_context)
        
        # Step 3: Intelligent engine selection with explicit request detection
        engines_to_call = self._determine_engines_intelligent(intent_analysis, user_context, user_message)
        
        # Step 4: Create context-aware prompt instructions
        prompt_instructions = self._create_enhanced_prompt_instructions(intent_analysis, user_context, user_message)
        
        # Step 5: Format enhanced chat context with conversation flow
        chat_context = self._format_enhanced_chat_context(user_context, user_message)
        
        # Step 6: Create detailed response instructions
        response_instructions = self._create_response_instructions(
            intent_analysis, user_context, user_message
        )
        
        # Step 7: Package everything for response agent
        orchestration_result = {
            "intent_analysis": intent_analysis,
            "engines_to_call": engines_to_call,
            "user_context": user_context,
            "prompt_instructions": prompt_instructions,
            "chat_context": chat_context,
            "original_message": user_message,
            "response_instructions": response_instructions,
            "user_phone": user_phone,
            "timestamp": datetime.now().isoformat()
        }
        
        # Enhanced logging with engine selection reasoning
        engine_names = [e['engine'] for e in engines_to_call]
        engine_reasons = [e.get('reason', 'standard') for e in engines_to_call]
        logger.info(f"🎯 Enhanced Orchestration: Intent={intent_analysis.get('primary_intent')}, "
                   f"Engines={engine_names}, Reasons={engine_reasons}, "
                   f"Context symbols={user_context.get('recent_symbols', [])}")
        
        return orchestration_result
    
    async def _gather_enhanced_user_context(self, user_phone: str) -> Dict[str, Any]:
        """Enhanced context gathering with conversation threading and flow detection"""
        
        context = {
            "user_profile": {},
            "conversation_history": [],
            "recent_symbols": [],
            "personality_traits": {},
            "communication_preferences": {},
            "trading_context": {},
            "session_data": {},
            "conversation_thread": [],
            "last_message": None
        }
        
        # Get user profile from personality engine (in-memory, fast)
        if self.personality_engine:
            profile = self.personality_engine.get_user_profile(user_phone)
            
            context["user_profile"] = profile
            context["personality_traits"] = {
                "formality": profile.get("communication_style", {}).get("formality", "casual"),
                "energy": profile.get("communication_style", {}).get("energy", "moderate"),
                "technical_depth": profile.get("communication_style", {}).get("technical_depth", "medium"),
                "emoji_usage": profile.get("communication_style", {}).get("emoji_usage", "some"),
                "response_length": profile.get("communication_style", {}).get("message_length", "medium")
            }
            
            context["trading_context"] = {
                "experience_level": profile.get("trading_personality", {}).get("experience_level", "intermediate"),
                "risk_tolerance": profile.get("trading_personality", {}).get("risk_tolerance", "moderate"),
                "trading_style": profile.get("trading_personality", {}).get("trading_style", "swing"),
                "preferred_sectors": profile.get("trading_personality", {}).get("preferred_sectors", []),
                "common_symbols": profile.get("trading_personality", {}).get("common_symbols", [])
            }
            
            # Get recent symbols from personality engine context memory
            context_memory = profile.get("context_memory", {})
            context["recent_symbols"] = context_memory.get("last_discussed_stocks", [])[:5]
        
        # Enhanced cache retrieval with conversation threading
        if self.cache_service:
            try:
                # Get conversation thread (last 10 messages for better context)
                thread_key = f"conversation_thread:{user_phone}"
                conversation_thread = await self.cache_service.get_list(thread_key, limit=10)
                context["conversation_thread"] = conversation_thread or []
                
                # Get last message for continuity detection
                last_message_key = f"last_message:{user_phone}"
                last_message = await self.cache_service.get(last_message_key)
                context["last_message"] = last_message
                
                # Fast cache lookup for recent messages (backward compatibility)
                cache_key = f"recent_messages:{user_phone}"
                recent_messages = await self.cache_service.get_list(cache_key, limit=3)
                context["conversation_history"] = recent_messages or []
                
                # Get additional recent symbols from cache
                symbols_key = f"recent_symbols:{user_phone}"
                cached_symbols = await self.cache_service.get_list(symbols_key, limit=5)
                if cached_symbols:
                    # Merge with personality engine symbols, remove duplicates
                    all_symbols = context["recent_symbols"] + cached_symbols
                    context["recent_symbols"] = list(dict.fromkeys(all_symbols))[:5]
                
                # Get today's session data
                session_key = f"user_session:{user_phone}"
                session_data = await self.cache_service.get(session_key)
                context["session_data"] = session_data or {}
                
            except Exception as e:
                logger.warning(f"Enhanced context gathering error: {e}")
                # Graceful degradation - continue without cache data
        
        return context
    
    async def _analyze_intent_with_context(self, user_message: str, user_context: Dict) -> Dict[str, Any]:
        """Enhanced intent analysis with explicit fundamental analysis detection"""
        
        # Build enhanced context for LLM
        context_summary = self._build_enhanced_context_summary(user_context)
        
        # Enhanced orchestrator prompt with explicit fundamental analysis detection
        orchestrator_prompt = f"""You are an expert trading assistant orchestrator. Analyze the user's message and context to determine their intent and needs.

USER CONTEXT:
{context_summary}

USER MESSAGE: "{user_message}"

Analyze this message and return detailed intent analysis as JSON:

{{
    "primary_intent": "analyze|screener|portfolio|help|general|celebrate|worry_check|validation_seeking",
    "emotional_state": "excited|worried|frustrated|curious|celebrating|neutral|anxious",
    "symbols_mentioned": ["AAPL", "TSLA"],
    "urgency_level": "immediate|research|casual_inquiry",
    "complexity_required": "quick_check|detailed_analysis|comprehensive_research",
    "user_subtext": "what they really want beyond the surface request",
    "conversation_continuity": "new_topic|continuing_discussion|follow_up_question",
    "confidence_score": 0.85,
    "requires_greeting": true|false,
    "market_timing_relevant": true|false,
    "requires_fundamental_analysis": true|false,
    "requires_technical_analysis": true|false,
    "requires_news_sentiment": true|false
}}

INTENT CLASSIFICATION RULES:
- "analyze" = wants analysis of specific stocks/symbols
- "screener" = wants to find/discover stocks based on criteria  
- "portfolio" = asking about their holdings/positions
- "celebrate" = sharing wins, excited about gains
- "worry_check" = concerned about losses/positions
- "validation_seeking" = wants confirmation of their thesis
- "help" = asking how to use the system
- "general" = casual conversation, no specific trading request

EXPLICIT ANALYSIS TYPE DETECTION:
- Set "requires_fundamental_analysis": true if message contains: fundamental, fundamentals, earnings, revenue, valuation, pe ratio, financial health, debt, cash flow, book value, dividend, growth rate, eps, financial ratios, balance sheet, income statement
- Set "requires_technical_analysis": true if message contains: technical, chart, resistance, support, rsi, macd, moving average, breakout, pattern, trend, momentum, or any stock symbol analysis request
- Set "requires_news_sentiment": true if message contains: news, updates, happened, headlines, sentiment, market news, or user shows emotional concern/excitement

SYMBOL EXTRACTION RULES:
- Extract tickers: AAPL, TSLA, MSFT, etc.
- Map company names: "apple" → "AAPL", "tesla" → "TSLA", "google" → "GOOGL", "nvidia" → "NVDA"
- Handle ETFs: "silver etf" → "SLV", "spy" → "SPY"
- Extract ALL symbols mentioned, not just first one

EMOTIONAL STATE DETECTION:
- "excited" = 🚀, "moon", "LFG", multiple exclamation marks
- "worried" = "should I sell", "dump", "concerned", "scared"
- "frustrated" = "wtf", "ugh", "bleeding", negative language
- "celebrating" = "killing it", gains mentions, success language

CONVERSATION CONTINUITY:
- "continuing_discussion" = references previous conversation or uses contextual pronouns
- "follow_up_question" = asks for more details about recent topic
- "new_topic" = completely new subject

URGENCY ASSESSMENT:
- "immediate" = "now", "today", "urgent", "quick"
- "research" = "thinking about", "considering", detailed questions
- "casual_inquiry" = general interest, no time pressure"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": orchestrator_prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            intent_analysis = json.loads(response.choices[0].message.content)
            
            # Validate and enhance intent analysis
            intent_analysis = self._validate_enhanced_intent_analysis(intent_analysis, user_message)
            
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Enhanced intent analysis failed: {e}")
            return self._fallback_enhanced_intent_analysis(user_message)
    
    def _determine_engines_intelligent(self, intent_analysis: Dict, user_context: Dict, user_message: str) -> List[Dict[str, Any]]:
        """Intelligent engine selection with explicit request detection and conversation context"""
        
        engines = []
        primary_intent = intent_analysis.get("primary_intent", "general")
        symbols = intent_analysis.get("symbols_mentioned", [])
        complexity = intent_analysis.get("complexity_required", "detailed_analysis")
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        recent_symbols = user_context.get("recent_symbols", [])
        
        # Get effective symbols (explicit or contextual)
        effective_symbols = symbols if symbols else recent_symbols[:2]
        
        # Convert user message to lowercase for keyword detection
        user_message_lower = user_message.lower()
        
        # Log engine selection process for debugging
        logger.info(f"🤖 Engine Selection Debug:")
        logger.info(f"   Intent: {primary_intent}")
        logger.info(f"   Explicit Symbols: {symbols}")
        logger.info(f"   Context Symbols: {recent_symbols}")
        logger.info(f"   Effective Symbols: {effective_symbols}")
        logger.info(f"   User Message: '{user_message}'")
        
        # 1. EXPLICIT FUNDAMENTAL ANALYSIS DETECTION (HIGHEST PRIORITY)
        if (intent_analysis.get("requires_fundamental_analysis") or 
            any(keyword in user_message_lower for keyword in self.fundamental_keywords)):
            engines.append({
                "engine": "fundamental_analysis",
                "symbols": effective_symbols[:2],
                "priority": "high",
                "reason": "explicit_fundamental_request"
            })
            logger.info(f"   ✅ Added Fundamental Analysis: explicit request detected")
        
        # 2. TECHNICAL ANALYSIS FOR STOCK ANALYSIS
        if primary_intent == "analyze" and effective_symbols:
            engines.append({
                "engine": "technical_analysis", 
                "symbols": effective_symbols[:3],
                "priority": "high",
                "reason": "stock_analysis_with_symbols"
            })
            logger.info(f"   ✅ Added Technical Analysis: analyzing {effective_symbols}")
        
        # 3. NEWS SENTIMENT FOR EMOTIONAL STATES OR EXPLICIT NEWS REQUESTS
        should_add_news = (
            intent_analysis.get("requires_news_sentiment") or
            any(keyword in user_message_lower for keyword in self.news_keywords) or
            emotional_state in ["worried", "excited", "frustrated"] or
            complexity in ["detailed_analysis", "comprehensive_research"]
        )
        
        if should_add_news and effective_symbols:
            engines.append({
                "engine": "news_sentiment",
                "symbols": effective_symbols[:2], 
                "priority": "medium",
                "reason": "emotional_context_or_news_request"
            })
            logger.info(f"   ✅ Added News Sentiment: emotional state or explicit request")
        
        # 4. PORTFOLIO ANALYSIS
        if primary_intent == "portfolio":
            engines.append({
                "engine": "portfolio_analysis",
                "user_phone": user_context.get("user_phone", ""),
                "priority": "high",
                "reason": "portfolio_inquiry"
            })
            logger.info(f"   ✅ Added Portfolio Analysis: portfolio inquiry")
        
        # 5. STOCK SCREENER
        if primary_intent == "screener":
            engines.append({
                "engine": "stock_screener",
                "parameters": self._build_screener_parameters(intent_analysis, user_context),
                "priority": "high",
                "reason": "stock_discovery_request"
            })
            logger.info(f"   ✅ Added Stock Screener: discovery request")
        
        # 6. EMOTIONAL SUPPORT WORKFLOWS
        if primary_intent in ["worry_check", "validation_seeking"] and effective_symbols:
            # For emotional support, get current data + news
            if not any(e["engine"] == "technical_analysis" for e in engines):
                engines.append({
                    "engine": "technical_analysis",
                    "symbols": effective_symbols[:2],
                    "priority": "high",
                    "reason": "emotional_support_technical"
                })
            
            if not any(e["engine"] == "news_sentiment" for e in engines):
                engines.append({
                    "engine": "news_sentiment",
                    "symbols": effective_symbols[:2], 
                    "priority": "high",
                    "reason": "emotional_support_news"
                })
            logger.info(f"   ✅ Added Emotional Support Engines")
        
        # 7. CELEBRATION WORKFLOW
        if primary_intent == "celebrate" and effective_symbols:
            if not any(e["engine"] == "technical_analysis" for e in engines):
                engines.append({
                    "engine": "technical_analysis",
                    "symbols": effective_symbols[:1],
                    "priority": "medium",
                    "reason": "celebration_price_confirmation"
                })
            logger.info(f"   ✅ Added Celebration Engine")
        
        # Final logging
        engine_names = [e['engine'] for e in engines]
        logger.info(f"   🎯 Final Engine Selection: {engine_names}")
        
        return engines
    
    def _create_enhanced_prompt_instructions(self, intent_analysis: Dict, user_context: Dict, user_message: str) -> str:
        """Create enhanced, context-aware instructions for the response LLM"""
        
        primary_intent = intent_analysis.get("primary_intent", "general")
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        symbols = intent_analysis.get("symbols_mentioned", [])
        personality = user_context.get("personality_traits", {})
        recent_symbols = user_context.get("recent_symbols", [])
        last_message = user_context.get("last_message", {})
        conversation_continuity = intent_analysis.get("conversation_continuity", "new_topic")
        
        # Get effective symbols (explicit or contextual)
        effective_symbols = symbols if symbols else recent_symbols[:2]
        
        # Build core instruction based on intent and emotion
        if primary_intent == "analyze":
            if symbols:
                # Explicit symbols mentioned
                if emotional_state == "worried":
                    instruction = f"Provide reassuring analysis of {', '.join(symbols)} with clear support levels and risk assessment to address their concerns."
                elif emotional_state == "excited":
                    instruction = f"Give balanced perspective on {', '.join(symbols)} momentum while matching their enthusiasm with realistic price targets."
                else:
                    instruction = f"Analyze {', '.join(symbols)} with current price, key technical levels, and actionable trading insights."
            elif recent_symbols and conversation_continuity in ["continuing_discussion", "follow_up_question"]:
                # Conversation continuation with context
                context_stocks = ', '.join(recent_symbols[:2])
                instruction = f"Continue analyzing {context_stocks} based on their follow-up question"
                
                # Enhanced context detection based on user message content
                user_message_lower = user_message.lower()
                if any(word in user_message_lower for word in self.fundamental_keywords):
                    instruction += " about fundamentals. Focus heavily on fundamental analysis including financial metrics, valuation ratios, and company health."
                elif any(word in user_message_lower for word in self.technical_keywords):
                    instruction += " about technical analysis. Focus on chart patterns, indicators, and price action."
                elif any(word in user_message_lower for word in self.news_keywords):
                    instruction += " about recent news. Summarize relevant market developments and sentiment."
                else:
                    instruction += ". Provide comprehensive insights addressing their specific question."
            else:
                # No context available or new topic
                instruction = "Provide helpful market analysis and trading guidance addressing their question."
        
        elif primary_intent == "screener":
            instruction = "Present top stock picks with clear selection criteria and brief analysis of why they're attractive opportunities."
        
        elif primary_intent == "portfolio":
            instruction = "Review their portfolio positions with performance summary and any rebalancing suggestions needed."
        
        elif primary_intent == "celebrate":
            instruction = f"Celebrate their {', '.join(effective_symbols) if effective_symbols else 'trading'} success while providing smart guidance on profit-taking or continuation."
        
        elif primary_intent == "worry_check":
            instruction = f"Address their concerns about {', '.join(effective_symbols) if effective_symbols else 'their positions'} with supportive analysis and clear next steps."
        
        elif primary_intent == "validation_seeking":
            instruction = f"Validate or challenge their {', '.join(effective_symbols) if effective_symbols else 'investment'} thesis with honest analysis and risk considerations."
        
        else:
            instruction = "Provide helpful trading guidance that directly addresses their question with actionable insights."
        
        # Add conversation continuity context
        if conversation_continuity == "continuing_discussion" and last_message:
            last_topic = last_message.get("topic", "market analysis")
            instruction += f" This continues their previous discussion about {last_topic}."
        
        # Add style modifier
        style = personality.get("formality", "casual")
        if style == "casual":
            instruction += " Use casual, friendly tone matching their communication style."
        elif style == "professional":
            instruction += " Maintain professional tone with precise analysis."
        
        # Add explicit fundamental analysis focus if detected
        if intent_analysis.get("requires_fundamental_analysis"):
            instruction += " IMPORTANT: Include detailed fundamental analysis as explicitly requested."
        
        return instruction
    
    def _format_enhanced_chat_context(self, user_context: Dict, current_message: str) -> Dict[str, Any]:
        """Format enhanced chat history and context with conversation flow awareness"""
        
        conversation_history = user_context.get("conversation_history", [])
        conversation_thread = user_context.get("conversation_thread", [])
        recent_symbols = user_context.get("recent_symbols", [])
        personality = user_context.get("personality_traits", {})
        last_message = user_context.get("last_message", {})
        
        # Format recent conversation from thread (more comprehensive)
        recent_messages = []
        for msg in conversation_thread[-5:]:  # Last 5 exchanges from thread
            if msg.get("user_message"):
                recent_messages.append({
                    "role": "user",
                    "content": msg["user_message"][:100] + ("..." if len(msg["user_message"]) > 100 else ""),
                    "timestamp": msg.get("timestamp", "")
                })
            if msg.get("bot_response"):
                recent_messages.append({
                    "role": "assistant", 
                    "content": msg["bot_response"][:100] + ("..." if len(msg["bot_response"]) > 100 else ""),
                    "timestamp": msg.get("timestamp", "")
                })
        
        # Fallback to old conversation history if thread is empty
        if not recent_messages:
            for msg in conversation_history[-3:]:
                if msg.get("user_message"):
                    recent_messages.append({
                        "role": "user",
                        "content": msg["user_message"][:100] + ("..." if len(msg["user_message"]) > 100 else "")
                    })
                if msg.get("bot_response"):
                    recent_messages.append({
                        "role": "assistant", 
                        "content": msg["bot_response"][:100] + ("..." if len(msg["bot_response"]) > 100 else "")
                    })
        
        chat_context = {
            "current_message": current_message,
            "recent_conversation": recent_messages,
            "user_profile": {
                "communication_style": personality.get("formality", "casual"),
                "technical_depth": personality.get("technical_depth", "medium"),
                "energy_level": personality.get("energy", "moderate")
            },
            "recent_symbols_discussed": recent_symbols[:5],
            "conversation_continuity": len(recent_messages) > 0,
            "relationship_stage": "established" if len(recent_messages) > 5 else "building",
            "last_topic": last_message.get("topic", "") if last_message else "",
            "conversation_flow": "continuing" if last_message else "new"
        }
        
        return chat_context
    
    def _build_enhanced_context_summary(self, user_context: Dict) -> str:
        """Build enhanced context summary with conversation flow information"""
        
        personality = user_context.get("personality_traits", {})
        trading = user_context.get("trading_context", {})
        recent_symbols = user_context.get("recent_symbols", [])
        conversation_thread = user_context.get("conversation_thread", [])
        last_message = user_context.get("last_message", {})
        
        context_parts = []
        
        # User profile summary
        context_parts.append(f"User Profile: {personality.get('formality', 'casual')} communicator, {trading.get('experience_level', 'intermediate')} trader")
        
        # Recent conversation context
        if recent_symbols:
            context_parts.append(f"Recently discussed: {', '.join(recent_symbols[:3])}")
        
        # Conversation flow context
        if last_message:
            context_parts.append(f"Last message topic: {last_message.get('topic', 'general')}")
            context_parts.append(f"Last symbols: {last_message.get('symbols', [])}")
        
        # Conversation history summary
        if conversation_thread:
            recent_count = len(conversation_thread)
            context_parts.append(f"Conversation depth: {recent_count} recent exchanges")
        elif len(user_context.get("conversation_history", [])) > 0:
            context_parts.append("Has conversation history")
        else:
            context_parts.append("New conversation")
        
        return "\n".join(context_parts)
    
    def _validate_enhanced_intent_analysis(self, intent_analysis: Dict, user_message: str) -> Dict[str, Any]:
        """Enhanced validation with explicit analysis type detection"""
        
        # Ensure required fields exist
        if "primary_intent" not in intent_analysis:
            intent_analysis["primary_intent"] = "general"
        
        if "symbols_mentioned" not in intent_analysis:
            intent_analysis["symbols_mentioned"] = []
        
        if "confidence_score" not in intent_analysis:
            intent_analysis["confidence_score"] = 0.5
        
        # Ensure analysis type flags exist
        if "requires_fundamental_analysis" not in intent_analysis:
            intent_analysis["requires_fundamental_analysis"] = False
        
        if "requires_technical_analysis" not in intent_analysis:
            intent_analysis["requires_technical_analysis"] = False
        
        if "requires_news_sentiment" not in intent_analysis:
            intent_analysis["requires_news_sentiment"] = False
        
        # Clean and validate symbols
        if intent_analysis["symbols_mentioned"]:
            cleaned_symbols = []
            for symbol in intent_analysis["symbols_mentioned"]:
                if isinstance(symbol, str) and 1 <= len(symbol) <= 5 and symbol.isalpha():
                    cleaned_symbols.append(symbol.upper())
            intent_analysis["symbols_mentioned"] = list(dict.fromkeys(cleaned_symbols))
        
        # Auto-correct intent if we have symbols but intent is general
        if intent_analysis["symbols_mentioned"] and intent_analysis["primary_intent"] == "general":
            intent_analysis["primary_intent"] = "analyze"
        
        # Double-check explicit analysis requirements with keyword detection
        user_message_lower = user_message.lower()
        
        # Override fundamental analysis flag if keywords detected
        if any(keyword in user_message_lower for keyword in self.fundamental_keywords):
            intent_analysis["requires_fundamental_analysis"] = True
            
        # Override technical analysis flag if keywords detected
        if any(keyword in user_message_lower for keyword in self.technical_keywords):
            intent_analysis["requires_technical_analysis"] = True
            
        # Override news sentiment flag if keywords detected
        if any(keyword in user_message_lower for keyword in self.news_keywords):
            intent_analysis["requires_news_sentiment"] = True
        
        return intent_analysis
    
    def _fallback_enhanced_intent_analysis(self, user_message: str) -> Dict[str, Any]:
        """Enhanced fallback intent analysis with keyword detection"""
        
        message_lower = user_message.lower()
        
        # Simple symbol extraction with enhanced mapping
        symbols = []
        symbol_mappings = {
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
            'google': 'GOOGL', 'amazon': 'AMZN', 'meta': 'META',
            'nvidia': 'NVDA', 'silver etf': 'SLV', 'spy': 'SPY',
            'amd': 'AMD', 'intel': 'INTC', 'netflix': 'NFLX'
        }
        
        for company, symbol in symbol_mappings.items():
            if company in message_lower:
                symbols.append(symbol)
        
        # Extract potential ticker symbols (2-5 uppercase letters)
        import re
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, user_message)
        for ticker in potential_tickers:
            if ticker not in ['SMS', 'API', 'USA', 'NYSE', 'NASDAQ']:  # Exclude common non-ticker words
                symbols.append(ticker)
        
        # Simple intent detection
        if any(word in message_lower for word in ['find', 'screen', 'search', 'top', 'best']):
            primary_intent = "screener"
        elif any(word in message_lower for word in ['portfolio', 'positions', 'holdings']):
            primary_intent = "portfolio"
        elif symbols or any(word in message_lower for word in ['analyze', 'analysis', 'thoughts', 'what about', 'how is']):
            primary_intent = "analyze"
        else:
            primary_intent = "general"
        
        # Detect analysis requirements
        requires_fundamental = any(keyword in message_lower for keyword in self.fundamental_keywords)
        requires_technical = any(keyword in message_lower for keyword in self.technical_keywords) or bool(symbols)
        requires_news = any(keyword in message_lower for keyword in self.news_keywords)
        
        return {
            "primary_intent": primary_intent,
            "emotional_state": "neutral",
            "symbols_mentioned": list(dict.fromkeys(symbols)),
            "urgency_level": "casual_inquiry",
            "complexity_required": "detailed_analysis",
            "user_subtext": "fallback analysis",
            "conversation_continuity": "new_topic",
            "confidence_score": 0.3,
            "requires_greeting": False,
            "market_timing_relevant": bool(symbols),
            "requires_fundamental_analysis": requires_fundamental,
            "requires_technical_analysis": requires_technical,
            "requires_news_sentiment": requires_news
        }
    
    def _create_prompt_instructions(self, intent_analysis: Dict, user_context: Dict) -> str:
        """Legacy method - redirects to enhanced version"""
        return self._create_enhanced_prompt_instructions(intent_analysis, user_context, "")
    
    def _format_chat_context(self, user_context: Dict, current_message: str) -> Dict[str, Any]:
        """Legacy method - redirects to enhanced version"""
        return self._format_enhanced_chat_context(user_context, current_message)
    
    def _create_response_instructions(self, intent_analysis: Dict, user_context: Dict, user_message: str) -> Dict[str, Any]:
        """Create detailed response instructions for internal use"""
        
        personality = user_context.get("personality_traits", {})
        trading_context = user_context.get("trading_context", {})
        
        instructions = {
            "response_strategy": self._determine_response_strategy(intent_analysis, personality),
            "communication_style": {
                "formality": personality.get("formality", "casual"),
                "energy_level": personality.get("energy", "moderate"), 
                "technical_depth": personality.get("technical_depth", "medium"),
                "use_emojis": self._should_use_emojis(user_message, personality),
                "response_length": personality.get("response_length", "medium")
            },
            "content_guidelines": self._build_content_guidelines(intent_analysis, trading_context),
            "context_integration": self._build_context_integration_instructions(user_context),
            "emotional_tone": self._determine_emotional_tone(intent_analysis),
            "character_limit": 320,  # SMS limit
            "key_focus_areas": self._identify_key_focus_areas(intent_analysis, user_context)
        }
        
        return instructions
    
    def _build_screener_parameters(self, intent_analysis: Dict, user_context: Dict) -> Dict[str, Any]:
        """Build parameters for stock screener based on intent and user context"""
        
        # Default screener parameters
        parameters = {
            "exchange": "us",
            "limit": 10,
            "sort": "refund_5d_p.desc",
            "filters": [
                ["exchange", "=", "us"],
                ["market_capitalization", ">", 2000000000],
                ["refund_5d_p", ">", 0]
            ]
        }
        
        return parameters
    
    def _determine_response_strategy(self, intent_analysis: Dict, personality: Dict) -> str:
        """Determine the overall response strategy"""
        
        primary_intent = intent_analysis.get("primary_intent", "general")
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        
        if emotional_state == "worried":
            return "supportive_analysis"
        elif emotional_state == "excited" or primary_intent == "celebrate":
            return "celebratory_validation"
        elif primary_intent == "analyze":
            return "analytical_insights"
        elif primary_intent == "screener":
            return "discovery_focused"
        elif primary_intent == "portfolio":
            return "portfolio_review"
        else:
            return "conversational_helpful"
    
    def _should_use_emojis(self, user_message: str, personality: Dict) -> bool:
        """Determine if response should include emojis"""
        
        # Never use emojis if user didn't use any
        user_has_emojis = any(ord(char) > 127 for char in user_message)  # Simple emoji detection
        
        # Never use emojis if message is long (professional context)
        message_is_long = len(user_message) > 100
        
        # Check user's emoji preference
        emoji_preference = personality.get("emoji_usage", "some")
        
        return user_has_emojis and not message_is_long and emoji_preference in ["some", "lots"]
    
    def _build_content_guidelines(self, intent_analysis: Dict, trading_context: Dict) -> Dict[str, Any]:
        """Build content guidelines for response generation"""
        
        guidelines = {
            "include_price_data": intent_analysis.get("primary_intent") in ["analyze", "celebrate", "worry_check"],
            "include_technical_levels": intent_analysis.get("complexity_required") in ["detailed_analysis", "comprehensive_research"],
            "include_news_context": intent_analysis.get("primary_intent") in ["analyze", "worry_check", "validation_seeking"],
            "focus_on_risk": intent_analysis.get("emotional_state") == "worried",
            "validate_thesis": intent_analysis.get("primary_intent") == "validation_seeking",
            "educational_tone": trading_context.get("experience_level") == "beginner"
        }
        
        return guidelines
    
    def _build_context_integration_instructions(self, user_context: Dict) -> Dict[str, Any]:
        """Build instructions for integrating user context into response"""
        
        recent_symbols = user_context.get("recent_symbols", [])
        conversation_history = user_context.get("conversation_history", [])
        
        return {
            "reference_recent_symbols": recent_symbols[:3] if recent_symbols else [],
            "continuation_context": len(conversation_history) > 0,
            "relationship_level": "established" if len(conversation_history) > 5 else "building",
            "first_interaction_today": len(conversation_history) == 0
        }
    
    def _determine_emotional_tone(self, intent_analysis: Dict) -> str:
        """Determine the emotional tone for the response"""
        
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        primary_intent = intent_analysis.get("primary_intent", "general")
        
        if emotional_state == "worried":
            return "reassuring"
        elif emotional_state == "excited" or primary_intent == "celebrate":
            return "enthusiastic"
        elif emotional_state == "frustrated":
            return "patient_supportive"
        elif primary_intent == "validation_seeking":
            return "confident_analytical"
        else:
            return "friendly_professional"
    
    def _identify_key_focus_areas(self, intent_analysis: Dict, user_context: Dict) -> List[str]:
        """Identify key areas the response should focus on"""
        
        focus_areas = []
        
        primary_intent = intent_analysis.get("primary_intent", "general")
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        complexity = intent_analysis.get("complexity_required", "detailed_analysis")
        
        if primary_intent == "analyze":
            focus_areas.extend(["current_price", "key_levels", "momentum"])
            
            if complexity == "comprehensive_research":
                focus_areas.extend(["fundamentals", "news_impact", "risk_factors"])
        
        if emotional_state == "worried":
            focus_areas.extend(["risk_assessment", "support_levels", "reassurance"])
        
        if emotional_state == "excited":
            focus_areas.extend(["momentum_confirmation", "profit_targets", "realistic_expectations"])
        
        if intent_analysis.get("market_timing_relevant"):
            focus_areas.append("entry_exit_timing")
        
        return focus_areas


# ===== ENHANCED TOOL EXECUTOR =====

class ToolExecutor:
    """Enhanced tool executor with better error handling and logging"""
    
    def __init__(self, ta_service, portfolio_service=None, screener_service=None, news_service=None, fundamental_tool=None):
        self.ta_service = ta_service
        self.portfolio_service = portfolio_service
        self.screener_service = screener_service
        self.news_service = news_service
        self.fundamental_tool = fundamental_tool
    
    async def execute_engines(self, engines_to_call: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute engines based on orchestrator instructions with enhanced logging"""
        
        results = {}
        tasks = []
        
        logger.info(f"🔧 Executing {len(engines_to_call)} engines in parallel")
        
        # Build execution tasks
        for engine_config in engines_to_call:
            engine_name = engine_config.get("engine")
            priority = engine_config.get("priority", "medium")
            reason = engine_config.get("reason", "standard")
            
            logger.info(f"   📋 {engine_name} (priority: {priority}, reason: {reason})")
            
            if engine_name == "technical_analysis":
                symbols = engine_config.get("symbols", [])
                if symbols:
                    tasks.append(self._execute_technical_analysis(symbols))
            
            elif engine_name == "news_sentiment":
                symbols = engine_config.get("symbols", [])
                if symbols:
                    tasks.append(self._execute_news_sentiment(symbols))
            
            elif engine_name == "fundamental_analysis":
                symbols = engine_config.get("symbols", [])
                if symbols:
                    tasks.append(self._execute_fundamental_analysis(symbols))
            
            elif engine_name == "stock_screener":
                parameters = engine_config.get("parameters", {})
                tasks.append(self._execute_stock_screener(parameters))
            
            elif engine_name == "portfolio_analysis":
                user_phone = engine_config.get("user_phone")
                if user_phone:
                    tasks.append(self._execute_portfolio_analysis(user_phone))
        
        # Execute all tasks in parallel
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results with enhanced error reporting
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    engine_name = engines_to_call[i].get("engine", f"engine_{i}")
                    logger.error(f"❌ {engine_name} execution failed: {result}")
                    results[f"{engine_name}_error"] = str(result)
                else:
                    results.update(result)
                    # Log successful execution
                    for key in result.keys():
                        if not key.endswith("_unavailable"):
                            logger.info(f"   ✅ {key}: {len(result[key]) if isinstance(result[key], dict) else 'completed'}")
        
        logger.info(f"🔧 Engine execution complete: {len(results)} result sets")
        return results
    
    async def _execute_technical_analysis(self, symbols: List[str]) -> Dict:
        """Execute technical analysis for symbols"""
        try:
            if not self.ta_service:
                return {"technical_analysis_unavailable": True}
            
            ta_results = {}
            for symbol in symbols:
                ta_data = await self.ta_service.analyze_symbol(symbol.upper())
                if ta_data:
                    ta_results[symbol] = ta_data
            
            return {"technical_analysis": ta_results} if ta_results else {"technical_analysis_unavailable": True}
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"technical_analysis_unavailable": True}
    
    async def _execute_news_sentiment(self, symbols: List[str]) -> Dict:
        """Execute news sentiment analysis for symbols"""
        try:
            if not self.news_service:
                return {"news_sentiment_unavailable": True}
            
            news_results = {}
            for symbol in symbols:
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
    
    async def _execute_fundamental_analysis(self, symbols: List[str]) -> Dict:
        """Execute fundamental analysis for symbols"""
        try:
            if not self.fundamental_tool:
                return {"fundamental_analysis_unavailable": True}
            
            fundamental_results = {}
            for symbol in symbols:
                try:
                    fund_result = await self.fundamental_tool.execute({
                        "symbol": symbol.upper(),
                        "depth": "standard",
                        "user_style": "casual"
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
    
    async def _execute_portfolio_analysis(self, user_phone: str) -> Dict:
        """Execute portfolio analysis for user"""
        try:
            if not self.portfolio_service:
                return {"portfolio_unavailable": True}
            
            portfolio_data = await self.portfolio_service.get_user_portfolio(user_phone)
            return {"portfolio": portfolio_data} if portfolio_data else {"portfolio_unavailable": True}
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {"portfolio_unavailable": True}


# ===== ENHANCED RESPONSE AGENT =====

class ResponseAgent:
    """Enhanced response agent with better data formatting and context integration"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def generate_response(self, orchestration_result: Dict, engine_results: Dict) -> str:
        """Generate response based on orchestrator instructions and engine results"""
        
        try:
            # Extract orchestration components
            prompt_instructions = orchestration_result.get("prompt_instructions", "")
            chat_context = orchestration_result.get("chat_context", {})
            original_message = orchestration_result.get("original_message", "")
            response_instructions = orchestration_result.get("response_instructions", {})
            
            # Build enhanced response prompt
            response_prompt = self._build_enhanced_response_prompt(
                prompt_instructions, chat_context, original_message, 
                engine_results, response_instructions
            )
            
            # Generate response
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Apply final processing
            final_response = self._process_final_response(generated_response, response_instructions)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Having some technical issues right now, but I'm here to help! Try me again in a moment."
    
    def _build_enhanced_response_prompt(self, prompt_instructions: str, chat_context: Dict, 
                                      original_message: str, engine_results: Dict, 
                                      response_instructions: Dict) -> str:
        """Build enhanced response generation prompt with better context integration"""
        
        # Format engine results for the prompt
        engine_data = self._format_enhanced_engine_results(engine_results, original_message)
        
        # Extract communication style
        communication_style = response_instructions.get("communication_style", {})
        
        # Build conversation context string
        conversation_context = self._format_conversation_context_string(chat_context)
        
        prompt = f"""ORCHESTRATOR INSTRUCTIONS: {prompt_instructions}

USER'S MESSAGE: "{original_message}"

CONVERSATION CONTEXT:
{conversation_context}

MARKET DATA & ANALYSIS:
{engine_data}

RESPONSE REQUIREMENTS:
- Style: {communication_style.get('formality', 'casual')}
- Technical depth: {communication_style.get('technical_depth', 'medium')}
- Use emojis: {communication_style.get('use_emojis', False)}
- Max length: {response_instructions.get('character_limit', 320)} characters
- Tone: {response_instructions.get('emotional_tone', 'friendly')}

IMPORTANT GUIDELINES:
- Address their specific question directly
- Use market data to support your points
- Match their communication style exactly
- Keep response focused and actionable
- Don't include meta-commentary or phrases like "Here's the analysis"

Generate the perfect response:"""

        return prompt
    
    def _format_enhanced_engine_results(self, engine_results: Dict, original_message: str) -> str:
        """Enhanced engine results formatting with context awareness"""
        
        if not engine_results:
            return "No market data available"
        
        formatted_results = []
        
        # Technical analysis with enhanced formatting
        if "technical_analysis" in engine_results:
            ta_data = engine_results["technical_analysis"]
            for symbol, data in ta_data.items():
                price_info = data.get('price', {})
                indicators = data.get('indicators', {})
                
                current = price_info.get('current', 'N/A')
                change_pct = price_info.get('change_percent', 0)
                rsi = indicators.get('RSI', {}).get('value', 'N/A')
                
                # Add support/resistance if available
                support = indicators.get('support_level', 'N/A')
                resistance = indicators.get('resistance_level', 'N/A')
                
                ta_summary = f"{symbol}: ${current} ({change_pct:+.1f}%), RSI: {rsi}"
                if support != 'N/A' and resistance != 'N/A':
                    ta_summary += f", Support: ${support}, Resistance: ${resistance}"
                
                formatted_results.append(ta_summary)
        
        # News sentiment with enhanced formatting
        if "news_sentiment" in engine_results:
            news_data = engine_results["news_sentiment"]
            for symbol, sentiment_info in news_data.items():
                news_sentiment = sentiment_info.get('news_sentiment', {})
                sentiment = news_sentiment.get('sentiment', 'neutral')
                impact_score = news_sentiment.get('impact_score', 0)
                summary = news_sentiment.get('summary', 'No significant news')
                
                formatted_results.append(f"{symbol} News: {sentiment.upper()} sentiment ({impact_score:.1f} impact) - {summary[:100]}")
        
        # Fundamental analysis with enhanced formatting
        if "fundamental_analysis" in engine_results:
            fund_data = engine_results["fundamental_analysis"]
            for symbol, fund_info in fund_data.items():
                if hasattr(fund_info, 'overall_score'):
                    score = fund_info.overall_score
                    health = fund_info.financial_health.value if hasattr(fund_info, 'financial_health') else 'N/A'
                    formatted_results.append(f"{symbol} Fundamentals: {score:.0f}/100 ({health})")
                else:
                    formatted_results.append(f"{symbol} Fundamentals: Analysis available")
        
        # Handle unavailable services
        unavailable_services = []
        for key in engine_results.keys():
            if key.endswith('_unavailable'):
                service_name = key.replace('_unavailable', '').replace('_', ' ').title()
                unavailable_services.append(service_name)
        
        if unavailable_services:
            formatted_results.append(f"Note: {', '.join(unavailable_services)} temporarily unavailable")
        
        return "\n".join(formatted_results) if formatted_results else "Market data processing..."
    
    def _format_conversation_context_string(self, chat_context: Dict) -> str:
        """Format conversation context into a readable string"""
        
        context_parts = []
        
        # Relationship and continuity
        relationship = chat_context.get('relationship_stage', 'new')
        continuity = chat_context.get('conversation_continuity', False)
        
        if continuity:
            context_parts.append("Continuing conversation")
            if chat_context.get('last_topic'):
                context_parts.append(f"Previous topic: {chat_context['last_topic']}")
        else:
            context_parts.append("New conversation")
        
        # User communication style
        user_profile = chat_context.get('user_profile', {})
        style = user_profile.get('communication_style', 'casual')
        tech_depth = user_profile.get('technical_depth', 'medium')
        context_parts.append(f"User style: {style}, technical depth: {tech_depth}")
        
        # Recent symbols discussed
        recent_symbols = chat_context.get('recent_symbols_discussed', [])
        if recent_symbols:
            context_parts.append(f"Recent symbols: {', '.join(recent_symbols[:3])}")
        
        return " | ".join(context_parts)
    
    def _format_recent_messages(self, recent_messages: List[Dict]) -> str:
        """Format recent conversation for prompt"""
        
        if not recent_messages:
            return "No recent conversation"
        
        formatted = []
        for msg in recent_messages[-4:]:  # Last 4 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            
            formatted_msg = f"{role}: {content}"
            if timestamp:
                formatted_msg += f" ({timestamp[:10]})"  # Just date part
            
            formatted.append(formatted_msg)
        
        return "\n".join(formatted)
    
    def _process_final_response(self, response: str, instructions: Dict) -> str:
        """Enhanced final response processing"""
        
        # Remove any LLM artifacts and meta-commentary
        response = response.strip()
        
        # Remove common LLM artifacts
        artifacts_to_remove = [
            "Here's the analysis:",
            "Here's your response:",
            "Based on the data:",
            "According to the information:",
            "Certainly!",
            "Here you go:",
            "Let me provide:",
        ]
        
        for artifact in artifacts_to_remove:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
        
        # Remove quotes if they wrap the entire response
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Ensure character limit
        max_chars = instructions.get('character_limit', 320)
        if len(response) > max_chars:
            response = response[:max_chars-3] + "..."
        
        # Ensure response isn't empty
        if not response:
            response = "I'm processing your request. Please try again in a moment."
        
        return response


# ===== ENHANCED COMPREHENSIVE MESSAGE PROCESSOR =====

class ComprehensiveMessageProcessor:
    """Enhanced main processor with conversation threading and improved caching"""
    
    def __init__(self, openai_client, ta_service, personality_engine, 
                 cache_service=None, news_service=None, fundamental_tool=None, 
                 portfolio_service=None, screener_service=None):
        
        self.orchestrator = ComprehensiveOrchestrator(
            openai_client, personality_engine, cache_service
        )
        
        self.tool_executor = ToolExecutor(
            ta_service, portfolio_service, screener_service, 
            news_service, fundamental_tool
        )
        
        self.response_agent = ResponseAgent(openai_client)
        self.personality_engine = personality_engine
        self.cache_service = cache_service
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """Enhanced message processing with conversation threading and improved caching"""
        
        try:
            logger.info(f"🎯 Enhanced orchestrated processing: '{message}' from {user_phone}")
            
            # Step 1: Enhanced orchestration with conversation context
            orchestration_result = await self.orchestrator.orchestrate(message, user_phone)
            
            # Step 2: Execute engines based on intelligent selection
            engines_to_call = orchestration_result.get("engines_to_call", [])
            engine_results = await self.tool_executor.execute_engines(engines_to_call)
            
            # Step 3: Learn from interaction
            if self.personality_engine:
                intent_data = orchestration_result.get("intent_analysis", {})
                self.personality_engine.learn_from_message(user_phone, message, intent_data)
            
            # Step 4: Generate enhanced response
            response = await self.response_agent.generate_response(
                orchestration_result, engine_results
            )
            
            # Step 5: Enhanced conversation caching and threading
            if self.cache_service:
                await self._cache_enhanced_conversation(
                    user_phone, message, response, orchestration_result
                )
            
            logger.info(f"✅ Enhanced orchestrated response: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"💥 Enhanced orchestrated processing failed: {e}")
            return "I'm having some technical difficulties, but I'm here to help! Please try again."
    
    async def _cache_enhanced_conversation(self, user_phone: str, message: str, 
                                         response: str, orchestration_result: Dict):
        """Enhanced conversation caching with threading and context"""
        
        try:
            timestamp = datetime.now().isoformat()
            intent_analysis = orchestration_result.get("intent_analysis", {})
            
            # Cache conversation thread entry
            thread_key = f"conversation_thread:{user_phone}"
            conversation_entry = {
                "timestamp": timestamp,
                "user_message": message,
                "bot_response": response,
                "intent": intent_analysis.get("primary_intent", "general"),
                "symbols": intent_analysis.get("symbols_mentioned", []),
                "emotional_state": intent_analysis.get("emotional_state", "neutral"),
                "engines_used": [e["engine"] for e in orchestration_result.get("engines_to_call", [])]
            }
            await self.cache_service.add_to_list(thread_key, conversation_entry, max_length=20)
            
            # Cache last message for continuity
            last_message_key = f"last_message:{user_phone}"
            last_message_data = {
                "message": message,
                "topic": intent_analysis.get("primary_intent", "general"),
                "symbols": intent_analysis.get("symbols_mentioned", []),
                "timestamp": timestamp
            }
            await self.cache_service.set(last_message_key, last_message_data, ttl=3600)
            
            # Cache recent messages (backward compatibility)
            await self.cache_service.add_to_list(
                f"recent_messages:{user_phone}",
                {
                    "user_message": message, 
                    "bot_response": response,
                    "timestamp": timestamp
                },
                max_length=5
            )
            
            # Cache symbols mentioned
            symbols = intent_analysis.get("symbols_mentioned", [])
            if symbols:
                await self.cache_service.add_to_list(
                    f"recent_symbols:{user_phone}",
                    symbols,
                    max_length=10
                )
                
            # Update session data
            session_key = f"user_session:{user_phone}"
            session_data = await self.cache_service.get(session_key) or {}
            session_data.update({
                "last_message_time": timestamp,
                "message_count": session_data.get("message_count", 0) + 1,
                "last_intent": intent_analysis.get("primary_intent", "general"),
                "total_symbols_discussed": len(set(session_data.get("all_symbols", []) + symbols))
            })
            session_data["all_symbols"] = list(set(session_data.get("all_symbols", []) + symbols))[:20]
            await self.cache_service.set(session_key, session_data, ttl=86400)
            
            logger.info(f"💾 Enhanced conversation cached successfully")
            
        except Exception as e:
            logger.warning(f"Enhanced conversation caching failed: {e}")


# ===== BACKWARD COMPATIBILITY =====

class TradingAgent:
    """Backward compatibility wrapper for the old TradingAgent interface"""
    
    def __init__(self, openai_client, personality_engine):
        self.openai_client = openai_client
        self.personality_engine = personality_engine
        
        # Initialize the enhanced orchestrator internally
        self.orchestrator = ComprehensiveOrchestrator(
            openai_client, personality_engine, cache_service=None
        )
        
    async def parse_intent(self, message: str, user_phone: str = None) -> Dict[str, Any]:
        """Legacy interface - parse user intent"""
        try:
            # Use enhanced orchestrator for intent analysis
            orchestration_result = await self.orchestrator.orchestrate(message, user_phone or "unknown")
            intent_analysis = orchestration_result.get("intent_analysis", {})
            
            # Convert to old format expected by other services
            return {
                "intent": intent_analysis.get("primary_intent", "general"),
                "symbols": intent_analysis.get("symbols_mentioned", []),
                "parameters": {
                    "urgency": intent_analysis.get("urgency_level", "medium"),
                    "complexity": intent_analysis.get("complexity_required", "medium"),
                    "emotion": intent_analysis.get("emotional_state", "neutral")
                },
                "requires_tools": [engine["engine"] for engine in orchestration_result.get("engines_to_call", [])],
                "confidence": intent_analysis.get("confidence_score", 0.5),
                "user_emotion": intent_analysis.get("emotional_state", "neutral"),
                "urgency": intent_analysis.get("urgency_level", "medium")
            }
        except Exception as e:
            logger.error(f"Legacy parse_intent failed: {e}")
            return {
                "intent": "general",
                "symbols": [],
                "parameters": {},
                "requires_tools": [],
                "confidence": 0.3,
                "user_emotion": "neutral",
                "urgency": "medium"
            }

# Export both old and new classes for maximum compatibility
__all__ = [
    'TradingAgent',  # Legacy compatibility
    'ToolExecutor',  # Enhanced tool executor
    'ComprehensiveOrchestrator',  # Enhanced orchestrator
    'ComprehensiveMessageProcessor',  # Enhanced main processor
    'ResponseAgent'  # Enhanced response agent
]
