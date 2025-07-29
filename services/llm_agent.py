# services/llm_agent.py - CLEANED VERSION (Fixed Syntax)

import json
import asyncio
import re
from typing import Dict, List, Optional, Any
from loguru import logger
import openai
from datetime import datetime
from openai import AsyncOpenAI

# ===== COMPREHENSIVE ORCHESTRATOR =====

class ComprehensiveOrchestrator:
    """Advanced orchestrator that analyzes intent, selects engines, and creates structured instructions"""
    
    def __init__(self, openai_client, personality_engine, cache_service=None):
        self.openai_client = openai_client
        self.personality_engine = personality_engine
        self.cache_service = cache_service
        
    async def orchestrate(self, user_message: str, user_phone: str) -> Dict[str, Any]:
        """Main orchestration method"""
        
        # Step 1: Gather user context from cache
        user_context = await self._gather_user_context_from_cache(user_phone)
        
        # Step 2: Analyze intent with context
        intent_analysis = await self._analyze_intent_with_context(user_message, user_context)
        
        # Pass original message context to engine selection  
        intent_analysis["original_message"] = user_message
        
        # Step 3: Determine engines to call based on intent
        engines_to_call = self._determine_engines(intent_analysis, user_context)
        
        # Step 4: Create clean prompt instructions for response LLM
        prompt_instructions = self._create_prompt_instructions(intent_analysis, user_context)
        
        # Step 5: Format chat history/context for response LLM
        chat_context = self._format_chat_context(user_context, user_message)
        
        # Step 6: Create detailed response instructions
        response_instructions = self._create_response_instructions(intent_analysis, user_context, user_message)
        
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
        
        logger.info(f"🎯 Orchestration complete: Intent={intent_analysis.get('primary_intent')}, Engines={len(engines_to_call)}, Context symbols={user_context.get('recent_symbols', [])}, Instructions='{prompt_instructions[:50]}...'")
        
        return orchestration_result
    
    async def _gather_user_context_from_cache(self, user_phone: str) -> Dict[str, Any]:
        """Gather comprehensive user context from cache"""
        
        context = {
            "user_profile": {},
            "conversation_history": [],
            "recent_symbols": [],
            "personality_traits": {},
            "communication_preferences": {},
            "trading_context": {},
            "session_data": {}
        }
        
        # Get user profile from personality engine
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
            
            # Get recent symbols from personality engine
            context_memory = profile.get("context_memory", {})
            context["recent_symbols"] = context_memory.get("last_discussed_stocks", [])[:5]
        
        # Get recent conversation history from cache
        if self.cache_service:
            try:
                cache_key = f"recent_messages:{user_phone}"
                recent_messages = await self.cache_service.get_list(cache_key, limit=3)
                context["conversation_history"] = recent_messages or []
                
                symbols_key = f"recent_symbols:{user_phone}"
                cached_symbols = await self.cache_service.get_list(symbols_key, limit=5)
                if cached_symbols:
                    all_symbols = context["recent_symbols"] + cached_symbols
                    context["recent_symbols"] = list(dict.fromkeys(all_symbols))[:5]
                
                session_key = f"user_session:{user_phone}"
                session_data = await self.cache_service.get(session_key)
                context["session_data"] = session_data or {}
                
            except Exception as e:
                logger.warning(f"Failed to get cache data: {e}")
        
        return context
    
    async def _analyze_intent_with_context(self, user_message: str, user_context: Dict) -> Dict[str, Any]:
        """Analyze user intent with full context awareness"""
        
        context_summary = self._build_context_summary(user_context)
        
        orchestrator_prompt = f"""You are an expert trading assistant orchestrator. Analyze the user's message and context.

USER CONTEXT:
{context_summary}

USER MESSAGE: "{user_message}"

Return detailed intent analysis as JSON:

{{
    "primary_intent": "analyze|screener|portfolio|help|general",
    "emotional_state": "excited|worried|frustrated|curious|neutral",
    "symbols_mentioned": ["AAPL"],
    "urgency_level": "immediate|research|casual_inquiry",
    "complexity_required": "quick_check|detailed_analysis|comprehensive_research",
    "confidence_score": 0.85
}}

INTENT RULES:
- "analyze" = wants analysis of stocks
- "screener" = wants to find stocks
- "portfolio" = asking about holdings
- "help" = asking how to use system
- "general" = casual conversation

SYMBOL EXTRACTION:
- Extract tickers: AAPL, TSLA, MSFT
- Map companies: "apple" → "AAPL", "tesla" → "TSLA"
- Extract ALL symbols mentioned"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": orchestrator_prompt}],
                temperature=0.1,
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            
            intent_analysis = json.loads(response.choices[0].message.content)
            intent_analysis = self._validate_intent_analysis(intent_analysis, user_message)
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return self._fallback_intent_analysis(user_message)
    
    def _determine_engines(self, intent_analysis: Dict, user_context: Dict) -> List[Dict[str, Any]]:
        """Determine which engines to call based on intent and context"""
        
        engines = []
        primary_intent = intent_analysis.get("primary_intent", "general")
        symbols = intent_analysis.get("symbols_mentioned", [])
        complexity = intent_analysis.get("complexity_required", "detailed_analysis")
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        recent_symbols = user_context.get("recent_symbols", [])
        
        # Smart engine selection logic
        if primary_intent == "analyze":
            if symbols:
                # Specific symbols mentioned
                engines.append({
                    "engine": "technical_analysis",
                    "symbols": symbols[:3],
                    "priority": "high"
                })
                
                if complexity in ["detailed_analysis", "comprehensive_research"]:
                    engines.append({
                        "engine": "news_sentiment", 
                        "symbols": symbols[:3],
                        "priority": "medium"
                    })
                
                if complexity == "comprehensive_research":
                    engines.append({
                        "engine": "fundamental_analysis",
                        "symbols": symbols[:2],
                        "priority": "low"
                    })
            else:
                # Use conversation context
                contextual_symbols = recent_symbols[:2] if recent_symbols else ["SPY"]
                
                if recent_symbols:
                    logger.info(f"🧠 Using conversation context: {contextual_symbols} (from recent: {recent_symbols})")
                else:
                    logger.info(f"🔍 No context symbols, using market overview: {contextual_symbols}")
                
                engines.append({
                    "engine": "technical_analysis",
                    "symbols": contextual_symbols,
                    "priority": "high"
                })
                
                # Check if they're asking about fundamentals
                original_message = intent_analysis.get("original_message", "").lower()
                if any(word in original_message for word in ["fundamental", "earnings", "revenue", "valuation", "pe", "financial"]):
                    engines.append({
                        "engine": "fundamental_analysis", 
                        "symbols": contextual_symbols,
                        "priority": "high"
                    })
                
                # Add news for follow-up questions
                if recent_symbols and complexity in ["detailed_analysis", "comprehensive_research"]:
                    engines.append({
                        "engine": "news_sentiment",
                        "symbols": contextual_symbols,
                        "priority": "medium"
                    })
        
        elif primary_intent == "screener":
            engines.append({
                "engine": "stock_screener",
                "parameters": self._build_screener_parameters(intent_analysis, user_context),
                "priority": "high"
            })
        
        elif primary_intent == "portfolio":
            engines.append({
                "engine": "portfolio_analysis",
                "user_phone": user_context.get("user_phone"),
                "priority": "high"
            })
        
        return engines
    
    def _create_prompt_instructions(self, intent_analysis: Dict, user_context: Dict) -> str:
        """Create clean instructions for the response LLM"""
        
        primary_intent = intent_analysis.get("primary_intent", "general")
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        symbols = intent_analysis.get("symbols_mentioned", [])
        personality = user_context.get("personality_traits", {})
        recent_symbols = user_context.get("recent_symbols", [])
        
        # Use contextual symbols if no explicit symbols mentioned
        effective_symbols = symbols if symbols else recent_symbols[:2]
        
        # Build instruction based on intent
        if primary_intent == "analyze":
            if symbols:
                # Explicit symbols mentioned
                if emotional_state == "worried":
                    instruction = f"Provide reassuring analysis of {', '.join(symbols)} with clear support levels and risk assessment."
                elif emotional_state == "excited":
                    instruction = f"Give balanced perspective on {', '.join(symbols)} momentum with realistic price targets."
                else:
                    instruction = f"Analyze {', '.join(symbols)} with current price, key technical levels, and actionable insights."
            elif recent_symbols:
                # Use conversation context
                context_stocks = ', '.join(recent_symbols[:2])
                instruction = f"Continue analyzing {context_stocks} based on their follow-up question about"
                
                # Add specific context
                original_message = intent_analysis.get("original_message", "").lower()
                if any(word in original_message for word in ["fundamental", "earnings", "revenue", "valuation"]):
                    instruction += " fundamentals. Provide fundamental analysis insights."
                elif any(word in original_message for word in ["technical", "chart", "resistance", "support"]):
                    instruction += " technical analysis. Focus on chart patterns and indicators."
                elif any(word in original_message for word in ["news", "updates", "happened"]):
                    instruction += " recent news. Summarize relevant market developments."
                else:
                    instruction += " their specific question. Provide comprehensive insights."
            else:
                instruction = "Provide helpful market analysis and trading guidance."
        
        elif primary_intent == "screener":
            instruction = "Present top stock picks with clear selection criteria and brief analysis."
        
        elif primary_intent == "portfolio":
            instruction = "Review their portfolio positions with performance summary and suggestions."
        
        else:
            instruction = "Provide helpful trading guidance that directly addresses their question."
        
        # Add style modifier
        style = personality.get("formality", "casual")
        if style == "casual":
            instruction += " Use casual, friendly tone."
        elif style == "professional":
            instruction += " Maintain professional tone with precise analysis."
        
        return instruction
    
    def _format_chat_context(self, user_context: Dict, current_message: str) -> Dict[str, Any]:
        """Format chat history and context"""
        
        conversation_history = user_context.get("conversation_history", [])
        recent_symbols = user_context.get("recent_symbols", [])
        personality = user_context.get("personality_traits", {})
        
        # Format recent conversation
        recent_messages = []
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
        
        return {
            "current_message": current_message,
            "recent_conversation": recent_messages,
            "user_profile": {
                "communication_style": personality.get("formality", "casual"),
                "technical_depth": personality.get("technical_depth", "medium"),
                "energy_level": personality.get("energy", "moderate")
            },
            "recent_symbols_discussed": recent_symbols[:5],
            "conversation_continuity": len(conversation_history) > 0,
            "relationship_stage": "established" if len(conversation_history) > 5 else "building"
        }
    
    def _create_response_instructions(self, intent_analysis: Dict, user_context: Dict, user_message: str) -> Dict[str, Any]:
        """Create detailed response instructions"""
        
        personality = user_context.get("personality_traits", {})
        
        return {
            "communication_style": {
                "formality": personality.get("formality", "casual"),
                "energy_level": personality.get("energy", "moderate"), 
                "technical_depth": personality.get("technical_depth", "medium"),
                "use_emojis": self._should_use_emojis(user_message, personality),
                "response_length": personality.get("response_length", "medium")
            },
            "emotional_tone": self._determine_emotional_tone(intent_analysis),
            "character_limit": 320
        }
    
    def _build_context_summary(self, user_context: Dict) -> str:
        """Build context summary for orchestrator prompt"""
        
        personality = user_context.get("personality_traits", {})
        trading = user_context.get("trading_context", {})
        recent_symbols = user_context.get("recent_symbols", [])
        conversation_history = user_context.get("conversation_history", [])
        
        context_parts = []
        
        context_parts.append(f"User Profile: {personality.get('formality', 'casual')} communicator, {trading.get('experience_level', 'intermediate')} trader")
        
        if recent_symbols:
            context_parts.append(f"Recently discussed: {', '.join(recent_symbols[:3])}")
        
        if conversation_history:
            last_msg = conversation_history[0] if conversation_history else {}
            context_parts.append(f"Last conversation: {last_msg.get('user_message', '')[:50]}...")
        else:
            context_parts.append("New conversation")
        
        return "\n".join(context_parts)
    
    def _validate_intent_analysis(self, intent_analysis: Dict, user_message: str) -> Dict[str, Any]:
        """Validate and enhance intent analysis results"""
        
        if "primary_intent" not in intent_analysis:
            intent_analysis["primary_intent"] = "general"
        
        if "symbols_mentioned" not in intent_analysis:
            intent_analysis["symbols_mentioned"] = []
        
        if "confidence_score" not in intent_analysis:
            intent_analysis["confidence_score"] = 0.5
        
        # Clean symbols
        if intent_analysis["symbols_mentioned"]:
            cleaned_symbols = []
            for symbol in intent_analysis["symbols_mentioned"]:
                if isinstance(symbol, str) and 1 <= len(symbol) <= 5 and symbol.isalpha():
                    cleaned_symbols.append(symbol.upper())
            intent_analysis["symbols_mentioned"] = list(dict.fromkeys(cleaned_symbols))
        
        # Auto-correct intent
        if intent_analysis["symbols_mentioned"] and intent_analysis["primary_intent"] == "general":
            intent_analysis["primary_intent"] = "analyze"
        
        return intent_analysis
    
    def _fallback_intent_analysis(self, user_message: str) -> Dict[str, Any]:
        """Fallback intent analysis when LLM fails"""
        
        message_lower = user_message.lower()
        
        # Simple symbol extraction
        symbols = []
        symbol_mappings = {
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
            'google': 'GOOGL', 'amazon': 'AMZN', 'meta': 'META',
            'nvidia': 'NVDA', 'silver etf': 'SLV', 'spy': 'SPY'
        }
        
        for company, symbol in symbol_mappings.items():
            if company in message_lower:
                symbols.append(symbol)
        
        # Simple intent detection
        if any(word in message_lower for word in ['find', 'screen', 'search', 'top', 'best']):
            primary_intent = "screener"
        elif any(word in message_lower for word in ['portfolio', 'positions', 'holdings']):
            primary_intent = "portfolio"
        elif symbols or any(word in message_lower for word in ['analyze', 'analysis', 'thoughts']):
            primary_intent = "analyze"
        else:
            primary_intent = "general"
        
        return {
            "primary_intent": primary_intent,
            "emotional_state": "neutral",
            "symbols_mentioned": symbols,
            "urgency_level": "casual_inquiry",
            "complexity_required": "detailed_analysis",
            "confidence_score": 0.3
        }
    
    def _build_screener_parameters(self, intent_analysis: Dict, user_context: Dict) -> Dict[str, Any]:
        """Build screener parameters"""
        return {
            "exchange": "us",
            "limit": 10,
            "sort": "refund_5d_p.desc",
            "filters": [
                ["exchange", "=", "us"],
                ["market_capitalization", ">", 2000000000],
                ["refund_5d_p", ">", 0]
            ]
        }
    
    def _should_use_emojis(self, user_message: str, personality: Dict) -> bool:
        """Determine if response should include emojis"""
        user_has_emojis = any(ord(char) > 127 for char in user_message)
        message_is_long = len(user_message) > 100
        emoji_preference = personality.get("emoji_usage", "some")
        return user_has_emojis and not message_is_long and emoji_preference in ["some", "lots"]
    
    def _determine_emotional_tone(self, intent_analysis: Dict) -> str:
        """Determine emotional tone for response"""
        emotional_state = intent_analysis.get("emotional_state", "neutral")
        primary_intent = intent_analysis.get("primary_intent", "general")
        
        if emotional_state == "worried":
            return "reassuring"
        elif emotional_state == "excited":
            return "enthusiastic"
        elif emotional_state == "frustrated":
            return "patient_supportive"
        else:
            return "friendly_professional"


# ===== TOOL EXECUTOR =====

class ToolExecutor:
    """Tool executor that works with orchestrator instructions"""
    
    def __init__(self, ta_service, portfolio_service=None, screener_service=None, news_service=None, fundamental_tool=None):
        self.ta_service = ta_service
        self.portfolio_service = portfolio_service
        self.screener_service = screener_service
        self.news_service = news_service
        self.fundamental_tool = fundamental_tool
    
    async def execute_engines(self, engines_to_call: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute engines based on orchestrator instructions"""
        
        results = {}
        tasks = []
        
        # Build execution tasks
        for engine_config in engines_to_call:
            engine_name = engine_config.get("engine")
            
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
            
            # Combine results
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    logger.error(f"Engine execution failed: {result}")
                    results[f"engine_error_{i}"] = str(result)
                else:
                    results.update(result)
        
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


# ===== RESPONSE AGENT =====

class ResponseAgent:
    """Generates responses based on orchestrator instructions and engine results"""
    
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
            
            # Build response prompt
            response_prompt = self._build_response_prompt(
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
    
    def _build_response_prompt(self, prompt_instructions: str, chat_context: Dict, 
                             original_message: str, engine_results: Dict, 
                             response_instructions: Dict) -> str:
        """Build the response generation prompt"""
        
        # Format engine results
        engine_data = self._format_engine_results(engine_results)
        
        # Extract communication style
        communication_style = response_instructions.get("communication_style", {})
        
        prompt = f"""INSTRUCTIONS: {prompt_instructions}

USER'S MESSAGE: "{original_message}"

MARKET DATA:
{engine_data}

STYLE REQUIREMENTS:
- Style: {communication_style.get('formality', 'casual')}
- Technical depth: {communication_style.get('technical_depth', 'medium')}
- Use emojis: {communication_style.get('use_emojis', False)}
- Max length: {response_instructions.get('character_limit', 320)} characters

Generate the perfect response:"""

        return prompt
    
    def _format_engine_results(self, engine_results: Dict) -> str:
        """Format engine results for response prompt"""
        
        if not engine_results:
            return "No market data available"
        
        formatted_results = []
        
        # Technical analysis
        if "technical_analysis" in engine_results:
            ta_data = engine_results["technical_analysis"]
            for symbol, data in ta_data.items():
                price_info = data.get('price', {})
                indicators = data.get('indicators', {})
                
                current = price_info.get('current', 'N/A')
                change_pct = price_info.get('change_percent', 0)
                rsi = indicators.get('RSI', {}).get('value', 'N/A')
                
                formatted_results.append(f"{symbol}: ${current} ({change_pct:+.1f}%), RSI: {rsi}")
        
        # News sentiment
        if "news_sentiment" in engine_results:
            news_data = engine_results["news_sentiment"]
            for symbol, sentiment_info in news_data.items():
                sentiment = sentiment_info.get('sentiment_score', 0)
                formatted_results.append(f"{symbol} news sentiment: {sentiment:.2f}")
        
        # Handle unavailable services
        unavailable_services = [key for key in engine_results.keys() if key.endswith('_unavailable')]
        if unavailable_services:
            formatted_results.append(f"Some data unavailable: {', '.join(unavailable_services)}")
        
        return "\n".join(formatted_results) if formatted_results else "Limited data available"
    
    def _process_final_response(self, response: str, instructions: Dict) -> str:
        """Apply final processing to the response"""
        
        # Remove any LLM artifacts
        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Ensure character limit
        max_chars = instructions.get('character_limit', 320)
        if len(response) > max_chars:
            response = response[:max_chars-3] + "..."
        
        return response


# ===== MAIN COMPREHENSIVE MESSAGE PROCESSOR =====

class ComprehensiveMessageProcessor:
    """Main processor that orchestrates the complete message handling flow"""
    
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
        """Final message processing flow with cache integration"""
        
        try:
            logger.info(f"🎯 Final orchestrated processing: '{message}' from {user_phone}")
            
            # Step 1: Orchestrate
            orchestration_result = await self.orchestrator.orchestrate(message, user_phone)
            
            # Step 2: Execute engines
            engines_to_call = orchestration_result.get("engines_to_call", [])
            engine_results = await self.tool_executor.execute_engines(engines_to_call)
            
            # Step 3: Learn from interaction
            if self.personality_engine:
                intent_data = orchestration_result.get("intent_analysis", {})
                self.personality_engine.learn_from_message(user_phone, message, intent_data)
            
            # Step 4: Generate response
            response = await self.response_agent.generate_response(
                orchestration_result, engine_results
            )
            
            # Step 5: Cache this interaction
            if self.cache_service:
                try:
                    # Cache recent message
                    await self.cache_service.add_to_list(
                        f"recent_messages:{user_phone}",
                        {
                            "user_message": message, 
                            "bot_response": response,
                            "timestamp": datetime.now().isoformat()
                        },
                        max_length=5
                    )
                    
                    # Cache symbols mentioned
                    symbols = orchestration_result.get("intent_analysis", {}).get("symbols_mentioned", [])
                    if symbols:
                        await self.cache_service.add_to_list(
                            f"recent_symbols:{user_phone}",
                            symbols,
                            max_length=10
                        )
                        
                    # Update session data
                    session_key = f"user_session:{user_phone}"
                    session_data = await self.cache_service.get(session_key) or {}
                    session_data["last_message_time"] = datetime.now().isoformat()
                    session_data["message_count"] = session_data.get("message_count", 0) + 1
                    await self.cache_service.set(session_key, session_data, ttl=86400)
                    
                except Exception as e:
                    logger.warning(f"Failed to cache interaction: {e}")
            
            logger.info(f"✅ Final orchestrated response generated: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"💥 Final orchestrated processing failed: {e}")
            return "I'm having some technical difficulties, but I'm here to help! Please try again."
