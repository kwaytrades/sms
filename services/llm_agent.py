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
        Enhanced orchestration with complete context JSON structure for AI-powered response
        """
        
        # Step 1: Gather enhanced user context
        user_context = await self._gather_enhanced_user_context(user_phone)
        
        # Step 2: Analyze intent with full context awareness
        intent_analysis = await self._analyze_intent_with_context(user_message, user_context)
        
        # Step 3: Intelligent engine selection
        engines_to_call = self._determine_engines_intelligent(intent_analysis, user_context, user_message)
        
        # Step 4: Build complete context JSON with exact structure required
        complete_context = {
            "user_context": {
                "communication_style": user_context.get("personality_traits", {}).get("formality", "casual"),
                "technical_expertise": user_context.get("personality_traits", {}).get("technical_depth", "intermediate"),
                "emotional_state": intent_analysis.get("emotional_state", "curious"),
                "conversation_history": [msg.get("content", "")[:100] for msg in user_context.get("conversation_thread", [])[-3:]],
                "trading_personality": user_context.get("trading_context", {}).get("trading_style", "swing_trader"),
                "relationship_stage": "established" if len(user_context.get("conversation_thread", [])) > 5 else "building"
            },
            "intent_analysis": {
                "primary_intent": intent_analysis.get("primary_intent", "analyze"),
                "specific_request": intent_analysis.get("user_subtext", "analysis"),
                "urgency": intent_analysis.get("urgency_level", "research"),
                "depth_required": intent_analysis.get("complexity_required", "comprehensive")
            },
            "market_intelligence": {
                # This will be populated with engine results later
            },
            "response_strategy": {
                "focus_areas": self._determine_focus_areas(intent_analysis, user_context),
                "tone": self._determine_response_tone(intent_analysis),
                "include_actionable_insights": True,
                "address_specific_concerns": self._identify_concerns(intent_analysis, user_message)
            }
        }
        
        # Package orchestration result
        orchestration_result = {
            "complete_context": complete_context,
            "engines_to_call": engines_to_call,
            "original_message": user_message,
            "user_phone": user_phone,
            "timestamp": datetime.now().isoformat()
        }
        
        # Enhanced logging with engine selection reasoning
        engine_names = [e['engine'] for e in engines_to_call]
        logger.info(f"🎯 Enhanced Orchestration: Intent={intent_analysis.get('primary_intent')}, "
                   f"Engines={engine_names}, Symbols={intent_analysis.get('symbols_mentioned', [])}")
        
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
                "technical_depth": profile.get("communication_style", {}).get("technical_depth", "intermediate"),
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

Return JSON only."""

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
        
        # Final logging
        engine_names = [e['engine'] for e in engines]
        logger.info(f"   🎯 Final Engine Selection: {engine_names}")
        
        return engines
    
    def _determine_focus_areas(self, intent_analysis: Dict, user_context: Dict) -> List[str]:
        """Determine focus areas for response strategy"""
        focus_areas = []
        
        if intent_analysis.get("requires_fundamental_analysis"):
            focus_areas.extend(["fundamental_deep_dive", "valuation_concerns", "growth_analysis"])
        
        if intent_analysis.get("requires_technical_analysis"):
            focus_areas.extend(["technical_levels", "momentum_analysis", "entry_exit_points"])
        
        if intent_analysis.get("requires_news_sentiment"):
            focus_areas.extend(["news_impact", "sentiment_analysis", "market_reaction"])
        
        if intent_analysis.get("emotional_state") == "worried":
            focus_areas.extend(["risk_assessment", "support_levels", "reassurance"])
        
        return focus_areas if focus_areas else ["general_analysis"]
    
    def _determine_response_tone(self, intent_analysis: Dict) -> str:
        """Determine response tone based on intent and emotion"""
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        
        if emotional_state == "worried":
            return "reassuring_analytical"
        elif emotional_state == "excited":
            return "balanced_enthusiasm"
        elif emotional_state == "frustrated":
            return "patient_supportive"
        else:
            return "analytical_but_accessible"
    
    def _identify_concerns(self, intent_analysis: Dict, user_message: str) -> List[str]:
        """Identify specific concerns to address"""
        concerns = []
        user_message_lower = user_message.lower()
        
        if "why" in user_message_lower:
            concerns.append("explain_reasoning")
        
        if any(word in user_message_lower for word in ["score", "rating", "low", "high"]):
            concerns.append("explain_scoring")
        
        if intent_analysis.get("requires_fundamental_analysis"):
            concerns.append("fundamental_metrics_importance")
        
        return concerns if concerns else ["provide_actionable_insights"]
    
    def _build_enhanced_context_summary(self, user_context: Dict) -> str:
        """Build enhanced context summary with conversation flow information"""
        personality = user_context.get("personality_traits", {})
        trading = user_context.get("trading_context", {})
        recent_symbols = user_context.get("recent_symbols", [])
        conversation_thread = user_context.get("conversation_thread", [])
        
        context_parts = []
        
        # User profile summary
        context_parts.append(f"User Profile: {personality.get('formality', 'casual')} communicator, {trading.get('experience_level', 'intermediate')} trader")
        
        # Recent conversation context
        if recent_symbols:
            context_parts.append(f"Recently discussed: {', '.join(recent_symbols[:3])}")
        
        # Conversation history summary
        if conversation_thread:
            recent_count = len(conversation_thread)
            context_parts.append(f"Conversation depth: {recent_count} recent exchanges")
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


class ResponseAgent:
    """Enhanced response agent with AI-powered intelligent synthesis"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def generate_response(self, orchestration_result: Dict, engine_results: Dict) -> str:
        """Generate response using AI-powered intelligent synthesis"""
        
        try:
            # Extract complete context from orchestration
            complete_context = orchestration_result.get("complete_context", {})
            
            # Merge engine results into market_intelligence section
            complete_context = self._merge_engine_results_into_context(complete_context, engine_results)
            
            # Generate intelligent response using complete context
            response = await self.generate_intelligent_response(complete_context)
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Having some technical issues right now, but I'm here to help! Try me again in a moment."
    
    def _merge_engine_results_into_context(self, complete_context: Dict, engine_results: Dict) -> Dict:
        """Merge engine results into market_intelligence section using symbol names as keys"""
        
        market_intelligence = {}
        
        # Process technical analysis results
        if "technical_analysis" in engine_results:
            ta_data = engine_results["technical_analysis"]
            for symbol, data in ta_data.items():
                if symbol not in market_intelligence:
                    market_intelligence[symbol] = {}
                
                # Transform to match JSON structure
                price_info = data.get('price', {})
                indicators = data.get('indicators', {})
                
                market_intelligence[symbol]["current_price"] = price_info.get('current', 0)
                market_intelligence[symbol]["technical_analysis"] = {
                    "trend": "bullish_short_term" if price_info.get('change_percent', 0) > 0 else "bearish_short_term",
                    "support_levels": [indicators.get('support_level', price_info.get('current', 0) * 0.95)],
                    "resistance_levels": [indicators.get('resistance_level', price_info.get('current', 0) * 1.05)],
                    "rsi": indicators.get('RSI', {}).get('value', 50),
                    "key_insight": f"Price at ${price_info.get('current', 0)}, RSI: {indicators.get('RSI', {}).get('value', 50)}"
                }
        
        # Process fundamental analysis results
        if "fundamental_analysis" in engine_results:
            fund_data = engine_results["fundamental_analysis"]
            for symbol, data in fund_data.items():
                if symbol not in market_intelligence:
                    market_intelligence[symbol] = {}
                
                # Transform fundamental data to match JSON structure
                if hasattr(data, 'overall_score'):
                    score = data.overall_score
                    health = data.financial_health.value if hasattr(data, 'financial_health') else 'unknown'
                    
                    market_intelligence[symbol]["fundamental_analysis"] = {
                        "score": score,
                        "health": health,
                        "key_metrics": {
                            "pe_ratio": getattr(data, 'pe_ratio', 'N/A'),
                            "debt_to_equity": getattr(data, 'debt_to_equity', 'N/A'),
                            "revenue_growth": getattr(data, 'revenue_growth', 'N/A'),
                            "profit_margin": getattr(data, 'profit_margin', 'N/A')
                        },
                        "strengths": getattr(data, 'strengths', []),
                        "concerns": getattr(data, 'concerns', []),
                        "narrative": f"Fundamental score: {score}/100 ({health})"
                    }
        
        # Process news sentiment results
        if "news_sentiment" in engine_results:
            news_data = engine_results["news_sentiment"]
            for symbol, data in news_data.items():
                if symbol not in market_intelligence:
                    market_intelligence[symbol] = {}
                
                # Transform news data to match JSON structure
                news_sentiment = data.get('news_sentiment', {})
                sentiment = news_sentiment.get('sentiment', 'neutral')
                impact_score = news_sentiment.get('impact_score', 0.5)
                summary = news_sentiment.get('summary', 'No significant news')
                
                market_intelligence[symbol]["news_sentiment"] = {
                    "sentiment": sentiment,
                    "impact_score": impact_score,
                    "key_themes": [summary[:50]],
                    "recent_headlines": [summary[:100]]
                }
        
        # Update complete context with market intelligence
        complete_context["market_intelligence"] = market_intelligence
        
        return complete_context
    
    async def generate_intelligent_response(self, complete_context: Dict) -> str:
        """Use AI to synthesize complete context into intelligent, personalized response"""
        
        intelligent_prompt = f"""You are an expert trading advisor with deep market knowledge. Analyze the complete context below and generate the perfect personalized response.

COMPLETE CONTEXT:
{json.dumps(complete_context, indent=2)}

YOUR TASK:
1. Synthesize ALL available data into coherent insights
2. Address their specific request with appropriate depth  
3. Connect technical, fundamental, and sentiment data into a narrative
4. Match their exact communication style and expertise level
5. Provide actionable insights, not just data dumps
6. Explain WHY things are happening, not just WHAT is happening

RESPONSE PRINCIPLES:
- Be a knowledgeable advisor, not a data reporter
- Connect the dots between different analysis types
- Address concerns and opportunities specifically
- Use their preferred communication style naturally
- Keep insights actionable and relevant to their trading style
- Stay under 320 characters for SMS

Generate the perfect response that shows deep understanding of both the market data and this specific user:"""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": intelligent_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return self._process_final_response(response.choices[0].message.content, {"character_limit": 320})
    
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


class ComprehensiveMessageProcessor:
    """Enhanced main processor with AI-powered response generation"""
    
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
        """Enhanced message processing with AI-powered intelligent synthesis"""
        
        try:
            logger.info(f"🎯 Enhanced AI-powered processing: '{message}' from {user_phone}")
            
            # Step 1: Enhanced orchestration with complete context building
            orchestration_result = await self.orchestrator.orchestrate(message, user_phone)
            
            # Step 2: Execute engines based on intelligent selection
            engines_to_call = orchestration_result.get("engines_to_call", [])
            engine_results = await self.tool_executor.execute_engines(engines_to_call)
            
            # Step 3: Learn from interaction
            if self.personality_engine:
                intent_data = orchestration_result.get("complete_context", {}).get("intent_analysis", {})
                self.personality_engine.learn_from_message(user_phone, message, intent_data)
            
            # Step 4: Generate AI-powered intelligent response
            response = await self.response_agent.generate_response(
                orchestration_result, engine_results
            )
            
            # Step 5: Enhanced conversation caching
            if self.cache_service:
                await self._cache_enhanced_conversation(
                    user_phone, message, response, orchestration_result
                )
            
            logger.info(f"✅ AI-powered response generated: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"💥 AI-powered processing failed: {e}")
            return "I'm having some technical difficulties, but I'm here to help! Please try again."
    
    async def _cache_enhanced_conversation(self, user_phone: str, message: str, 
                                         response: str, orchestration_result: Dict):
        """Enhanced conversation caching with threading and context"""
        
        try:
            timestamp = datetime.now().isoformat()
            complete_context = orchestration_result.get("complete_context", {})
            intent_analysis = complete_context.get("intent_analysis", {})
            
            # Cache conversation thread entry
            thread_key = f"conversation_thread:{user_phone}"
            conversation_entry = {
                "timestamp": timestamp,
                "user_message": message,
                "bot_response": response,
                "intent": intent_analysis.get("primary_intent", "general"),
                "symbols": [],  # Will be extracted from market_intelligence
                "emotional_state": complete_context.get("user_context", {}).get("emotional_state", "neutral"),
                "engines_used": [e["engine"] for e in orchestration_result.get("engines_to_call", [])]
            }
            
            # Extract symbols from market intelligence
            market_intel = complete_context.get("market_intelligence", {})
            conversation_entry["symbols"] = list(market_intel.keys())
            
            await self.cache_service.add_to_list(thread_key, conversation_entry, max_length=20)
            
            # Cache last message for continuity
            last_message_key = f"last_message:{user_phone}"
            last_message_data = {
                "message": message,
                "topic": intent_analysis.get("primary_intent", "general"),
                "symbols": list(market_intel.keys()),
                "timestamp": timestamp
            }
            await self.cache_service.set(last_message_key, last_message_data, ttl=3600)
            
            logger.info(f"💾 Enhanced conversation cached successfully")
            
        except Exception as e:
            logger.warning(f"Enhanced conversation caching failed: {e}")


# Backward compatibility wrapper
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
            complete_context = orchestration_result.get("complete_context", {})
            intent_analysis = complete_context.get("intent_analysis", {})
            
            # Convert to old format expected by other services
            return {
                "intent": intent_analysis.get("primary_intent", "general"),
                "symbols": list(complete_context.get("market_intelligence", {}).keys()),
                "parameters": {
                    "urgency": intent_analysis.get("urgency", "medium"),
                    "complexity": intent_analysis.get("depth_required", "medium"),
                    "emotion": complete_context.get("user_context", {}).get("emotional_state", "neutral")
                },
                "requires_tools": [engine["engine"] for engine in orchestration_result.get("engines_to_call", [])],
                "confidence": 0.8,
                "user_emotion": complete_context.get("user_context", {}).get("emotional_state", "neutral"),
                "urgency": intent_analysis.get("urgency", "medium")
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
