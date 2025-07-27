# ===== Intelligent LLM Orchestrator =====

import json
from typing import Dict, List, Any
from dataclasses import dataclass
from loguru import logger

@dataclass
class WorkflowPlan:
    primary_intent: str
    workflows: List[Dict[str, Any]]
    entities: Dict[str, Any]
    user_context: Dict[str, Any]
    execution_order: List[str]

# ===== COST-OPTIMIZED LLM ARCHITECTURE =====

"""
Cost Analysis:
- Orchestrator (gpt-4o-mini): ~$0.0001 per message (structured analysis)
- Response Generator (gpt-4o): ~$0.003 per message (creative output)
- Total: ~$0.0031 per message

VS Previous Approach:
- Intent Parsing (gpt-4o-mini): ~$0.0001
- Response Generation (gpt-4o): ~$0.003  
- Total: ~$0.0031 per message

VS All GPT-4o Approach:
- Orchestrator (gpt-4o): ~$0.005
- Response Generator (gpt-4o): ~$0.003
- Total: ~$0.008 per message

Savings: 60% cost reduction while maintaining quality!
"""

class IntelligentOrchestrator:
    """LLM-powered orchestrator that decides workflows and extracts everything"""
    
    def __init__(self, openai_service, personality_engine):
        self.openai_service = openai_service
        self.personality_engine = personality_engine
        
        # Available workflows
        self.available_workflows = {
            "stock_analysis": "Analyze specific stocks with TA, news, sentiment",
            "portfolio_review": "Check current portfolio performance and allocation", 
            "market_overview": "General market conditions and sentiment",
            "stock_discovery": "Find stocks based on criteria",
            "options_analysis": "Analyze options strategies and Greeks",
            "earnings_calendar": "Check upcoming earnings and events",
            "sector_analysis": "Analyze specific sectors or industries",
            "risk_assessment": "Evaluate portfolio or position risk",
            "price_alerts": "Set up or manage price alerts",
            "educational": "Explain trading concepts or provide guidance",
            "account_management": "Subscription, limits, preferences",
            "general_chat": "Casual conversation or unclear intent"
        }
    
    async def orchestrate_message(self, message: str, user_phone: str) -> WorkflowPlan:
        """Master orchestration function - decides everything"""
        
        # Get user context for better orchestration
        user_profile = self.personality_engine.get_user_profile(user_phone)
        
        orchestration_prompt = f"""You are an intelligent trading bot orchestrator. Analyze this message and create a complete execution plan.

USER MESSAGE: "{message}"

USER CONTEXT:
- Experience Level: {user_profile.get('trading_personality', {}).get('experience_level', 'intermediate')}
- Communication Style: {user_profile.get('communication_style', {}).get('formality', 'casual')}
- Recent Stocks: {user_profile.get('context_memory', {}).get('last_discussed_stocks', [])}
- Trading Style: {user_profile.get('trading_personality', {}).get('trading_style', 'swing')}

AVAILABLE WORKFLOWS:
{json.dumps(self.available_workflows, indent=2)}

EXTRACT AND PLAN:
1. **Primary Intent**: What's the main thing the user wants?
2. **Stock Symbols**: All stocks mentioned (tickers OR company names)
3. **Company Name Mapping**: Convert company names to tickers
4. **Secondary Intents**: Additional things they want
5. **User Sentiment**: Are they bullish/bearish/uncertain/excited/worried?
6. **Investment Context**: Buying/selling/holding/researching?
7. **Urgency Level**: Do they need this info now or just curious?
8. **Workflow Sequence**: What order should workflows execute?

EXAMPLE OUTPUTS:

Message: "yo check my portfolio and what's TSLA doing?"
Output: {{
  "primary_intent": "portfolio_review",
  "workflows": [
    {{"name": "portfolio_review", "priority": 1, "params": {{}}}},
    {{"name": "stock_analysis", "priority": 2, "params": {{"symbols": ["TSLA"]}}}}
  ],
  "entities": {{
    "symbols": ["TSLA"],
    "company_names_found": [],
    "user_sentiment": "neutral",
    "investment_context": "monitoring",
    "urgency": "medium"
  }},
  "user_context": {{
    "wants_portfolio_first": true,
    "interested_in_specific_stocks": ["TSLA"],
    "likely_follow_up": "position_sizing"
  }},
  "execution_order": ["portfolio_review", "stock_analysis"]
}}

Message: "Tesla earnings coming up, should I buy calls?"
Output: {{
  "primary_intent": "options_analysis", 
  "workflows": [
    {{"name": "stock_analysis", "priority": 1, "params": {{"symbols": ["TSLA"], "focus": "earnings"}}}},
    {{"name": "earnings_calendar", "priority": 2, "params": {{"symbols": ["TSLA"]}}}},
    {{"name": "options_analysis", "priority": 3, "params": {{"symbols": ["TSLA"], "strategy": "calls"}}}}
  ],
  "entities": {{
    "symbols": ["TSLA"],
    "company_names_found": ["Tesla"],
    "user_sentiment": "bullish_curious",
    "investment_context": "considering_options_purchase",
    "urgency": "high",
    "options_interest": "calls"
  }},
  "user_context": {{
    "earnings_focused": true,
    "options_trader": true,
    "seeking_advice": true
  }},
  "execution_order": ["stock_analysis", "earnings_calendar", "options_analysis"]
}}

Message: "find me some cheap growth stocks under $50"
Output: {{
  "primary_intent": "stock_discovery",
  "workflows": [
    {{"name": "stock_discovery", "priority": 1, "params": {{"criteria": {{"max_price": 50, "category": "growth", "price_range": "under_50"}}}}}}
  ],
  "entities": {{
    "symbols": [],
    "screening_criteria": {{"max_price": 50, "category": "growth"}},
    "user_sentiment": "hunting",
    "investment_context": "research_phase",
    "urgency": "low"
  }},
  "user_context": {{
    "price_conscious": true,
    "growth_focused": true,
    "discovery_mode": true
  }},
  "execution_order": ["stock_discovery"]
}}

Now analyze the user's message and return ONLY the JSON plan:"""

        try:
            response = await self.openai_service.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for structured analysis
                messages=[{"role": "user", "content": orchestration_prompt}],
                temperature=0.1,  # Very low temperature for consistent planning
                max_tokens=600,   # Reduced tokens for structured output
                response_format={"type": "json_object"}
            )
            
            plan_data = json.loads(response.choices[0].message.content)
            
            # Validate and enhance the plan
            validated_plan = self._validate_and_enhance_plan(plan_data, message)
            
            logger.info(f"🎯 Orchestration Plan: {validated_plan.primary_intent} | Workflows: {len(validated_plan.workflows)} | Symbols: {validated_plan.entities.get('symbols', [])}")
            
            return validated_plan
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return self._fallback_orchestration(message)
    
    def _validate_and_enhance_plan(self, plan_data: Dict, original_message: str) -> WorkflowPlan:
        """Validate LLM plan and add safety checks"""
        
        # Ensure required fields exist
        primary_intent = plan_data.get("primary_intent", "general_chat")
        workflows = plan_data.get("workflows", [])
        entities = plan_data.get("entities", {})
        user_context = plan_data.get("user_context", {})
        execution_order = plan_data.get("execution_order", [])
        
        # Validate workflow names
        valid_workflows = []
        for workflow in workflows:
            if workflow.get("name") in self.available_workflows:
                valid_workflows.append(workflow)
            else:
                logger.warning(f"Invalid workflow: {workflow.get('name')}")
        
        # Ensure symbols are properly formatted
        symbols = entities.get("symbols", [])
        if symbols:
            symbols = [s.upper().strip() for s in symbols if isinstance(s, str)]
            entities["symbols"] = symbols
        
        # Add company name to symbol conversion
        entities = self._enhance_symbol_extraction(entities, original_message)
        
        # Limit number of workflows to prevent overload
        if len(valid_workflows) > 3:
            logger.warning("Too many workflows planned, limiting to top 3")
            valid_workflows = sorted(valid_workflows, key=lambda x: x.get("priority", 999))[:3]
        
        return WorkflowPlan(
            primary_intent=primary_intent,
            workflows=valid_workflows,
            entities=entities,
            user_context=user_context,
            execution_order=execution_order[:3]  # Limit execution steps
        )
    
    def create_response_prompt(self, plan: WorkflowPlan, workflow_results: Dict, original_message: str, user_phone: str) -> str:
        """Create the prompt for the response generator LLM"""
        
        user_profile = self.personality_engine.get_user_profile(user_phone)
        
        # Build personality context
        personality_context = self._build_personality_context(user_profile)
        
        # Build workflow results summary
        results_context = self._build_results_context(workflow_results, plan)
        
        # Build conversation context
        conversation_context = self._build_conversation_context(plan, original_message)
        
        response_prompt = f"""You are a hyper-personalized SMS trading assistant. Generate a response that perfectly matches this user's style and provides actionable insights.

ORIGINAL MESSAGE: "{original_message}"

USER PERSONALITY PROFILE:
{personality_context}

ORCHESTRATION ANALYSIS:
Primary Intent: {plan.primary_intent}
User Sentiment: {plan.entities.get('user_sentiment', 'neutral')}
Investment Context: {plan.entities.get('investment_context', 'research')}
Urgency Level: {plan.entities.get('urgency', 'medium')}
Symbols Discussed: {plan.entities.get('symbols', [])}

WORKFLOW RESULTS:
{results_context}

CONVERSATION CONTEXT:
{conversation_context}

RESPONSE GUIDELINES:
1. Match their communication style exactly (formality: {user_profile.get('communication_style', {}).get('formality', 'casual')})
2. Use their preferred energy level ({user_profile.get('communication_style', {}).get('energy', 'moderate')})
3. Include appropriate emojis ({user_profile.get('communication_style', {}).get('emoji_usage', 'some')})
4. Match their technical depth ({user_profile.get('communication_style', {}).get('technical_depth', 'medium')})
5. Consider their trading experience ({user_profile.get('trading_personality', {}).get('experience_level', 'intermediate')})
6. Address their specific concerns and context
7. Provide actionable insights based on the workflow results
8. Keep SMS-friendly (under 320 characters total, can split into 2 messages if needed)

RESPONSE STRATEGY:
- If multiple workflows executed, prioritize the most important results
- If data unavailable, acknowledge honestly but stay helpful  
- If user seems worried/uncertain, provide reassuring guidance
- If user seems excited/bullish, match their energy but add appropriate caution
- Reference their past trading patterns or preferences when relevant

Generate the perfect personalized response now:"""

        return response_prompt
    
    def _build_personality_context(self, user_profile: Dict) -> str:
        """Build personality context for prompt"""
        
        if not user_profile:
            return "New user - no personality data yet. Use friendly, moderate tone."
        
        comm_style = user_profile.get('communication_style', {})
        trading_style = user_profile.get('trading_personality', {})
        learning_data = user_profile.get('learning_data', {})
        context_memory = user_profile.get('context_memory', {})
        
        return f"""Communication Style:
- Formality: {comm_style.get('formality', 'casual')} 
- Energy Level: {comm_style.get('energy', 'moderate')}
- Emoji Usage: {comm_style.get('emoji_usage', 'some')}
- Message Length Preference: {comm_style.get('message_length', 'medium')}
- Technical Depth: {comm_style.get('technical_depth', 'medium')}

Trading Personality:
- Experience Level: {trading_style.get('experience_level', 'intermediate')}
- Risk Tolerance: {trading_style.get('risk_tolerance', 'moderate')}
- Trading Style: {trading_style.get('trading_style', 'swing')}
- Common Stocks: {', '.join(trading_style.get('common_symbols', [])[:5])}
- Win/Loss Mentions: {learning_data.get('successful_trades_mentioned', 0)}W/{learning_data.get('loss_trades_mentioned', 0)}L

Recent Context:
- Last Discussed Stocks: {', '.join(context_memory.get('last_discussed_stocks', []))}
- Total Conversations: {learning_data.get('total_messages', 0)}"""
    
    def _build_results_context(self, workflow_results: Dict, plan: WorkflowPlan) -> str:
        """Build workflow results context for prompt"""
        
        if not workflow_results.get('workflow_results'):
            return "No workflow results available - acknowledge service unavailability."
        
        results_summary = []
        
        for workflow_name, result in workflow_results.get('workflow_results', {}).items():
            if 'error' in result:
                results_summary.append(f"❌ {workflow_name}: {result['error']}")
            else:
                results_summary.append(f"✅ {workflow_name}: {self._summarize_workflow_result(workflow_name, result)}")
        
        return '\n'.join(results_summary) if results_summary else "No actionable results from workflows."
    
    def _summarize_workflow_result(self, workflow_name: str, result: Dict) -> str:
        """Summarize individual workflow results"""
        
        if workflow_name == "stock_analysis":
            symbols = result.get('symbols_analyzed', [])
            ta_data = result.get('technical_analysis', {})
            
            summaries = []
            for symbol in symbols:
                if symbol in ta_data:
                    data = ta_data[symbol]
                    price_info = data.get('price', {})
                    if price_info:
                        price = price_info.get('current', 'N/A')
                        change = price_info.get('change_percent', 0)
                        summaries.append(f"{symbol}: ${price} ({change:+.1f}%)")
                    
            return f"Analyzed {', '.join(summaries)}" if summaries else f"Analysis attempted for {', '.join(symbols)}"
        
        elif workflow_name == "portfolio_review":
            portfolio = result.get('portfolio', {})
            if portfolio:
                return f"Portfolio data available with {len(portfolio.get('positions', []))} positions"
            else:
                return "Portfolio data unavailable"
        
        elif workflow_name == "stock_discovery":
            results = result.get('results', [])
            criteria = result.get('screening_criteria', {})
            return f"Found {len(results)} stocks matching criteria: {criteria}"
        
        else:
            return f"Completed with {len(result)} data points"
    
    def _build_conversation_context(self, plan: WorkflowPlan, original_message: str) -> str:
        """Build conversation context for prompt"""
        
        context_items = []
        
        # Add intent context
        if plan.primary_intent == "stock_analysis":
            context_items.append("User wants stock analysis - provide specific actionable insights")
        elif plan.primary_intent == "portfolio_review":
            context_items.append("User checking portfolio - focus on performance and recommendations")
        elif plan.primary_intent == "options_analysis":
            context_items.append("User interested in options - provide strategy guidance and risk awareness")
        
        # Add sentiment context
        user_sentiment = plan.entities.get('user_sentiment', '')
        if 'worried' in user_sentiment or 'concerned' in user_sentiment:
            context_items.append("User seems worried - provide reassuring but honest guidance")
        elif 'excited' in user_sentiment or 'bullish' in user_sentiment:
            context_items.append("User seems excited - match energy but add appropriate caution")
        
        # Add urgency context
        urgency = plan.entities.get('urgency', 'medium')
        if urgency == 'high':
            context_items.append("High urgency - user needs timely actionable information")
        elif urgency == 'low':
            context_items.append("Low urgency - user is researching, provide educational context")
        
        # Add investment context
        investment_context = plan.entities.get('investment_context', '')
        if 'buying' in investment_context or 'considering' in investment_context:
            context_items.append("User considering purchase - provide entry points and risk assessment")
        elif 'selling' in investment_context:
            context_items.append("User considering sale - provide exit strategy guidance")
        
# ===== RESPONSE GENERATOR =====

class ResponseGenerator:
    """Final LLM that generates personalized responses using orchestrator's prompt"""
    
    def __init__(self, openai_service):
        self.openai_service = openai_service
    
    async def generate_response(self, response_prompt: str) -> str:
        """Generate final response using orchestrator's prepared prompt"""
        
        try:
            response = await self.openai_service.client.chat.completions.create(
                model="gpt-4o",     # Premium model for creative response generation
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.7,    # Higher creativity for response generation
                max_tokens=250      # Adequate for SMS responses
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Ensure SMS compatibility
            if len(generated_response) > 320:
                # Try to split into 2 messages
                sentences = generated_response.split('. ')
                if len(sentences) > 1:
                    mid_point = len(sentences) // 2
                    first_half = '. '.join(sentences[:mid_point]) + '.'
                    second_half = '. '.join(sentences[mid_point:])
                    
                    if len(first_half) <= 160 and len(second_half) <= 160:
                        generated_response = f"{first_half}\n\n{second_half}"
                    else:
                        generated_response = generated_response[:317] + "..."
                else:
                    generated_response = generated_response[:317] + "..."
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Sorry, I'm having trouble right now. Please try again in a moment! 🔧"

# ===== COMPLETE ORCHESTRATED SMS PROCESSOR =====

class OrchestratedSMSProcessor:
    """Complete SMS processing with LLM orchestration"""
    
    def __init__(self, openai_service, ta_service, portfolio_service, personality_engine):
        self.orchestrator = IntelligentOrchestrator(openai_service, personality_engine)
        self.workflow_executor = WorkflowExecutor(ta_service, portfolio_service)
        self.response_generator = ResponseGenerator(openai_service)
        self.personality_engine = personality_engine
    
    async def process_sms(self, message: str, user_phone: str) -> str:
        """Complete orchestrated SMS processing"""
        
        logger.info(f"🎭 Processing SMS with full orchestration: '{message}'")
        
        try:
            # Step 1: LLM Orchestrator plans everything
            plan = await self.orchestrator.orchestrate_message(message, user_phone)
            
            # Step 2: Execute planned workflows
            workflow_results = await self.workflow_executor.execute_plan(plan, user_phone)
            
            # Step 3: Learn from interaction
            self.personality_engine.learn_from_message(user_phone, message, {
                "intent": plan.primary_intent,
                "symbols": plan.entities.get("symbols", []),
                "sentiment": plan.entities.get("user_sentiment", "neutral"),
                "context": plan.entities.get("investment_context", "general")
            })
            
            # Step 4: Orchestrator creates response prompt
            response_prompt = self.orchestrator.create_response_prompt(
                plan=plan,
                workflow_results=workflow_results,
                original_message=message,
                user_phone=user_phone
            )
            
            # Step 5: Response generator creates final response
            final_response = await self.response_generator.generate_response(response_prompt)
            
            logger.info(f"✅ Orchestrated response generated: {len(final_response)} chars")
            
            return final_response
            
        except Exception as e:
            logger.error(f"💥 Orchestrated SMS processing failed: {e}")
            return "Sorry, I'm having technical issues. Please try again in a moment! 🔧"
    
    def _enhance_symbol_extraction(self, entities: Dict, message: str) -> Dict:
        """Enhance symbol extraction with company name mapping"""
        
        # Company name mappings (comprehensive)
        company_mappings = {
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT', 'amazon': 'AMZN',
            'google': 'GOOGL', 'alphabet': 'GOOGL', 'meta': 'META', 'facebook': 'META',
            'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
            'verizon': 'VZ', 'at&t': 'T', 'att': 'T', 'comcast': 'CMCSA', 't-mobile': 'TMUS',
            'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'goldman': 'GS', 'goldman sachs': 'GS',
            'bank of america': 'BAC', 'wells fargo': 'WFC', 'morgan stanley': 'MS',
            'walmart': 'WMT', 'target': 'TGT', 'home depot': 'HD', 'starbucks': 'SBUX',
            'boeing': 'BA', 'caterpillar': 'CAT', 'general electric': 'GE',
            'johnson & johnson': 'JNJ', 'pfizer': 'PFE', 'merck': 'MRK',
            'exxon': 'XOM', 'chevron': 'CVX', 'coinbase': 'COIN', 'paypal': 'PYPL'
        }
        
        message_lower = message.lower()
        additional_symbols = []
        
        for company, symbol in company_mappings.items():
            if company in message_lower:
                additional_symbols.append(symbol)
        
        # Merge with existing symbols
        existing_symbols = entities.get("symbols", [])
        all_symbols = list(set(existing_symbols + additional_symbols))
        
        entities["symbols"] = all_symbols
        entities["company_names_detected"] = additional_symbols
        
        return entities
    
    def _fallback_orchestration(self, message: str) -> WorkflowPlan:
        """Fallback if LLM orchestration fails"""
        
        message_lower = message.lower()
        
        # Simple intent detection
        if any(word in message_lower for word in ['portfolio', 'positions']):
            primary_intent = "portfolio_review"
            workflows = [{"name": "portfolio_review", "priority": 1, "params": {}}]
        elif any(word in message_lower for word in ['find', 'discover', 'screen']):
            primary_intent = "stock_discovery" 
            workflows = [{"name": "stock_discovery", "priority": 1, "params": {}}]
        else:
            primary_intent = "general_chat"
            workflows = [{"name": "general_chat", "priority": 1, "params": {}}]
        
        return WorkflowPlan(
            primary_intent=primary_intent,
            workflows=workflows,
            entities={"symbols": [], "fallback": True},
            user_context={"fallback_mode": True},
            execution_order=[w["name"] for w in workflows]
        )

# ===== WORKFLOW EXECUTION ENGINE =====

class WorkflowExecutor:
    """Executes the workflows planned by the orchestrator"""
    
    def __init__(self, ta_service, portfolio_service, news_service=None):
        self.ta_service = ta_service
        self.portfolio_service = portfolio_service  
        self.news_service = news_service
        
        # Map workflow names to execution functions
        self.workflow_handlers = {
            "stock_analysis": self._execute_stock_analysis,
            "portfolio_review": self._execute_portfolio_review,
            "stock_discovery": self._execute_stock_discovery,
            "options_analysis": self._execute_options_analysis,
            "earnings_calendar": self._execute_earnings_calendar,
            "general_chat": self._execute_general_chat
        }
    
    async def execute_plan(self, plan: WorkflowPlan, user_phone: str) -> Dict[str, Any]:
        """Execute all workflows in the plan"""
        
        results = {
            "plan_summary": {
                "primary_intent": plan.primary_intent,
                "workflows_executed": [],
                "entities_processed": plan.entities
            },
            "workflow_results": {},
            "execution_order": plan.execution_order
        }
        
        # Execute workflows in priority order
        for workflow in sorted(plan.workflows, key=lambda x: x.get("priority", 999)):
            workflow_name = workflow["name"]
            workflow_params = workflow.get("params", {})
            
            if workflow_name in self.workflow_handlers:
                try:
                    logger.info(f"🔄 Executing workflow: {workflow_name}")
                    
                    result = await self.workflow_handlers[workflow_name](
                        params=workflow_params,
                        entities=plan.entities,
                        user_phone=user_phone
                    )
                    
                    results["workflow_results"][workflow_name] = result
                    results["plan_summary"]["workflows_executed"].append(workflow_name)
                    
                    logger.info(f"✅ Completed workflow: {workflow_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Workflow {workflow_name} failed: {e}")
                    results["workflow_results"][workflow_name] = {"error": str(e)}
            else:
                logger.warning(f"Unknown workflow: {workflow_name}")
        
        return results
    
    async def _execute_stock_analysis(self, params: Dict, entities: Dict, user_phone: str) -> Dict:
        """Execute stock analysis workflow"""
        
        symbols = entities.get("symbols", [])
        if not symbols:
            return {"error": "No symbols to analyze"}
        
        analysis_results = {}
        
        for symbol in symbols[:3]:  # Limit to 3 symbols
            if self.ta_service:
                ta_data = await self.ta_service.analyze_symbol(symbol)
                if ta_data:
                    analysis_results[symbol] = ta_data
        
        return {
            "symbols_analyzed": list(analysis_results.keys()),
            "technical_analysis": analysis_results,
            "focus": params.get("focus", "general")
        }
    
    async def _execute_portfolio_review(self, params: Dict, entities: Dict, user_phone: str) -> Dict:
        """Execute portfolio review workflow"""
        
        if not self.portfolio_service:
            return {"error": "Portfolio service not available"}
        
        try:
            portfolio_data = await self.portfolio_service.get_user_portfolio(user_phone)
            return {
                "portfolio": portfolio_data,
                "review_type": params.get("review_type", "summary")
            }
        except Exception as e:
            return {"error": f"Portfolio fetch failed: {str(e)}"}
    
    async def _execute_stock_discovery(self, params: Dict, entities: Dict, user_phone: str) -> Dict:
        """Execute stock discovery/screening workflow"""
        
        criteria = params.get("criteria", {})
        
        # Mock screener results for now
        return {
            "screening_criteria": criteria,
            "results": ["AAPL", "MSFT", "GOOGL"],  # Mock results
            "screener_type": "basic"
        }
    
    async def _execute_options_analysis(self, params: Dict, entities: Dict, user_phone: str) -> Dict:
        """Execute options analysis workflow"""
        
        symbols = entities.get("symbols", [])
        strategy = params.get("strategy", "calls")
        
        return {
            "symbols": symbols,
            "strategy": strategy,
            "analysis": "Options analysis coming soon",
            "options_available": False
        }
    
    async def _execute_earnings_calendar(self, params: Dict, entities: Dict, user_phone: str) -> Dict:
        """Execute earnings calendar workflow"""
        
        symbols = entities.get("symbols", [])
        
        return {
            "symbols": symbols,
            "upcoming_earnings": "Earnings calendar coming soon",
            "calendar_available": False
        }
    
    async def _execute_general_chat(self, params: Dict, entities: Dict, user_phone: str) -> Dict:
        """Handle general conversation"""
        
        return {
            "chat_response": "I'm here to help with trading questions!",
            "suggestions": ["Ask about a stock", "Check your portfolio", "Find new stocks"]
        }
