# services/gemini_service.py
import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import google.generativeai as genai
from loguru import logger
import hashlib
from datetime import datetime, timezone

@dataclass
class PersonalityAnalysisResult:
    """Structured result from Gemini personality analysis"""
    communication_insights: Dict[str, Any]
    trading_insights: Dict[str, Any]
    emotional_state: Dict[str, Any]
    service_needs: Dict[str, Any]
    sales_opportunity: Dict[str, Any]
    intent_analysis: Dict[str, Any]
    confidence_score: float
    analysis_timestamp: str
    processing_time_ms: int


class GeminiPersonalityService:
    """
    Gemini 1.5 Flash-powered semantic personality analysis service
    Replaces regex-based detection with intelligent LLM analysis
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initialize Gemini service with API configuration"""
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.analysis_count = 0
        
        # Pricing (per 1M tokens)
        self.input_cost_per_million = 0.075  # $0.075 per 1M input tokens
        self.output_cost_per_million = 0.30  # $0.30 per 1M output tokens
        
        # Cache for analysis results (prevent duplicate analysis)
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        logger.info(f"🤖 Gemini Personality Service initialized with {model_name}")
    
    def _get_cache_key(self, message: str, context: Dict = None) -> str:
        """Generate cache key for message analysis"""
        content = f"{message}_{json.dumps(context or {}, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        
        timestamp = cache_entry.get('timestamp', 0)
        return (time.time() - timestamp) < self.cache_ttl
    
    async def analyze_personality_semantic(
        self, 
        message: str, 
        context: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> PersonalityAnalysisResult:
        """
        Perform comprehensive semantic personality analysis using Gemini
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(message, context)
        if use_cache and cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.debug("📋 Using cached personality analysis")
                return cache_entry['result']
        
        try:
            # Generate comprehensive analysis prompt
            analysis_prompt = self._build_analysis_prompt(message, context)
            
            # Call Gemini API
            response = await self._call_gemini_async(analysis_prompt)
            
            # Parse structured response
            analysis_result = self._parse_gemini_response(response, start_time)
            
            # Cache result
            if use_cache:
                self.analysis_cache[cache_key] = {
                    'result': analysis_result,
                    'timestamp': time.time()
                }
            
            # Update usage tracking
            self._track_usage(response)
            
            logger.info(f"✅ Semantic personality analysis completed in {analysis_result.processing_time_ms}ms")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Gemini personality analysis failed: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(message, start_time)
    
    def _build_analysis_prompt(self, message: str, context: Dict = None) -> str:
        """Build comprehensive personality analysis prompt for Gemini"""
        
        prompt = f"""
You are an expert personality analyst for a trading SMS bot. Analyze this user message and provide comprehensive personality insights in JSON format.

USER MESSAGE: "{message}"

CONTEXT: {json.dumps(context or {}, indent=2)}

Provide analysis in this EXACT JSON structure:

{{
  "communication_insights": {{
    "formality_score": 0.0-1.0,
    "energy_level": "low|moderate|high",
    "technical_depth": "basic|intermediate|advanced",
    "emoji_usage": 0-10,
    "message_length_preference": "short|medium|long",
    "urgency_level": 0.0-1.0,
    "question_type": "statement|yes_no|open_ended|clarification",
    "communication_tone": "positive|negative|neutral",
    "politeness_level": "low|moderate|high",
    "directness_level": "low|moderate|high"
  }},
  "trading_insights": {{
    "symbols_mentioned": ["AAPL", "TSLA"],
    "trading_action": "buy|sell|hold|research|unclear",
    "action_confidence": 0.0-1.0,
    "risk_tolerance": "conservative|moderate|aggressive",
    "time_horizon": "scalping|day_trading|swing|position|unknown",
    "position_size_hints": "small|medium|large|unknown",
    "profit_loss_mention": {{
      "type": "profit|loss|neutral",
      "sharing_openness": true|false
    }},
    "sector_mentions": ["technology", "healthcare"],
    "analysis_request": "technical|fundamental|news|options|general",
    "research_orientation": 0.0-1.0,
    "momentum_focus": 0.0-1.0,
    "news_sensitivity": 0.0-1.0
  }},
  "emotional_state": {{
    "primary_emotion": "excitement|anxiety|frustration|satisfaction|confusion|neutral",
    "emotional_intensity": 0.0-1.0,
    "specific_emotions": {{
      "excitement": 0.0-1.0,
      "anxiety": 0.0-1.0,
      "frustration": 0.0-1.0,
      "confidence": 0.0-1.0,
      "greed": 0.0-1.0,
      "fear": 0.0-1.0
    }},
    "emotional_triggers": {{
      "fomo_indicators": 0.0-1.0,
      "euphoria_indicators": 0.0-1.0,
      "panic_indicators": 0.0-1.0,
      "regret_indicators": 0.0-1.0
    }},
    "support_needed": "emotional_reassurance|problem_solving|education_and_clarity|standard_guidance"
  }},
  "service_needs": {{
    "service_type": "complaint|technical_issue|billing_question|feature_request|praise|none",
    "urgency_level": 0.0-1.0,
    "escalation_risk": "low|medium|high",
    "follow_up_needed": true|false
  }},
  "sales_opportunity": {{
    "sales_readiness_score": 0.0-1.0,
    "current_sales_stage": "unqualified|awareness|consideration|evaluation|ready_to_buy",
    "interest_level": 0.0-1.0,
    "budget_indicators": {{
      "status": "has_budget|no_budget|flexible|unknown",
      "price_sensitivity": "low|medium|high"
    }},
    "timeline_urgency": "immediate|soon|later|unspecified",
    "decision_authority": "decision_maker|influencer|no_authority|unknown",
    "objections": ["price", "features", "timing", "trust"]
  }},
  "intent_analysis": {{
    "primary_intent": "stock_analysis|portfolio_help|market_news|education|account_help|general_chat",
    "analysis_type": "technical|fundamental|news|options|general",
    "complexity_level": "low|medium|high",
    "follow_up_likelihood": 0.0-1.0,
    "requires_tools": ["technical_analysis", "news_sentiment", "fundamental_analysis"]
  }}
}}

ANALYSIS GUIDELINES:
1. Extract ALL ticker symbols mentioned (AAPL, Tesla → TSLA, Apple → AAPL)
2. Detect trading emotions (FOMO, greed, fear, anxiety)
3. Assess technical sophistication level
4. Identify service/support needs
5. Score sales opportunities
6. Predict follow-up likelihood
7. Classify communication style precisely

Respond ONLY with valid JSON. No additional text."""

        return prompt
    
    async def _call_gemini_async(self, prompt: str) -> Any:
        """Make async call to Gemini API"""
        loop = asyncio.get_event_loop()
        
        def sync_generate():
            return self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent analysis
                    max_output_tokens=2048,
                    response_mime_type="application/json"
                )
            )
        
        return await loop.run_in_executor(None, sync_generate)
    
    def _parse_gemini_response(self, response: Any, start_time: float) -> PersonalityAnalysisResult:
        """Parse Gemini response into structured result"""
        try:
            # Extract JSON content
            response_text = response.text.strip()
            
            # Handle potential JSON wrapper
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            analysis_data = json.loads(response_text)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate confidence score based on completeness
            confidence_score = self._calculate_confidence_score(analysis_data)
            
            return PersonalityAnalysisResult(
                communication_insights=analysis_data.get('communication_insights', {}),
                trading_insights=analysis_data.get('trading_insights', {}),
                emotional_state=analysis_data.get('emotional_state', {}),
                service_needs=analysis_data.get('service_needs', {}),
                sales_opportunity=analysis_data.get('sales_opportunity', {}),
                intent_analysis=analysis_data.get('intent_analysis', {}),
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=processing_time_ms
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"❌ Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response: {response.text}")
            return self._create_fallback_analysis("", start_time)
    
    def _calculate_confidence_score(self, analysis_data: Dict) -> float:
        """Calculate confidence score based on analysis completeness"""
        expected_sections = [
            'communication_insights', 'trading_insights', 'emotional_state',
            'service_needs', 'sales_opportunity', 'intent_analysis'
        ]
        
        present_sections = sum(1 for section in expected_sections if section in analysis_data)
        completeness_score = present_sections / len(expected_sections)
        
        # Check for specific high-confidence indicators
        high_confidence_indicators = 0
        
        # Trading symbols detected
        if analysis_data.get('trading_insights', {}).get('symbols_mentioned'):
            high_confidence_indicators += 1
        
        # Clear emotional state
        emotional_intensity = analysis_data.get('emotional_state', {}).get('emotional_intensity', 0)
        if emotional_intensity > 0.5:
            high_confidence_indicators += 1
        
        # Specific trading action
        trading_action = analysis_data.get('trading_insights', {}).get('trading_action', 'unclear')
        if trading_action != 'unclear':
            high_confidence_indicators += 1
        
        # Technical depth detected
        tech_depth = analysis_data.get('communication_insights', {}).get('technical_depth', 'basic')
        if tech_depth in ['intermediate', 'advanced']:
            high_confidence_indicators += 1
        
        confidence_boost = min(0.3, high_confidence_indicators * 0.075)
        final_confidence = min(1.0, completeness_score + confidence_boost)
        
        return final_confidence
    
    def _create_fallback_analysis(self, message: str, start_time: float) -> PersonalityAnalysisResult:
        """Create fallback analysis when Gemini fails"""
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return PersonalityAnalysisResult(
            communication_insights={
                "formality_score": 0.5,
                "energy_level": "moderate",
                "technical_depth": "basic",
                "emoji_usage": 0,
                "message_length_preference": "medium",
                "urgency_level": 0.0,
                "question_type": "statement",
                "communication_tone": "neutral",
                "politeness_level": "moderate",
                "directness_level": "moderate"
            },
            trading_insights={
                "symbols_mentioned": [],
                "trading_action": "unclear",
                "action_confidence": 0.0,
                "risk_tolerance": "moderate",
                "time_horizon": "unknown",
                "position_size_hints": "unknown",
                "profit_loss_mention": {"type": "neutral", "sharing_openness": False},
                "sector_mentions": [],
                "analysis_request": "general",
                "research_orientation": 0.5,
                "momentum_focus": 0.5,
                "news_sensitivity": 0.5
            },
            emotional_state={
                "primary_emotion": "neutral",
                "emotional_intensity": 0.0,
                "specific_emotions": {
                    "excitement": 0.0, "anxiety": 0.0, "frustration": 0.0,
                    "confidence": 0.5, "greed": 0.0, "fear": 0.0
                },
                "emotional_triggers": {
                    "fomo_indicators": 0.0, "euphoria_indicators": 0.0,
                    "panic_indicators": 0.0, "regret_indicators": 0.0
                },
                "support_needed": "standard_guidance"
            },
            service_needs={
                "service_type": "none",
                "urgency_level": 0.0,
                "escalation_risk": "low",
                "follow_up_needed": False
            },
            sales_opportunity={
                "sales_readiness_score": 0.5,
                "current_sales_stage": "unqualified",
                "interest_level": 0.5,
                "budget_indicators": {"status": "unknown", "price_sensitivity": "medium"},
                "timeline_urgency": "unspecified",
                "decision_authority": "unknown",
                "objections": []
            },
            intent_analysis={
                "primary_intent": "general_chat",
                "analysis_type": "general",
                "complexity_level": "low",
                "follow_up_likelihood": 0.5,
                "requires_tools": []
            },
            confidence_score=0.2,  # Low confidence for fallback
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=processing_time_ms
        )
    
    def _track_usage(self, response: Any) -> None:
        """Track API usage and costs"""
        try:
            # Get usage metadata from response
            usage = response.usage_metadata
            
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            total_tokens = usage.total_token_count
            
            # Calculate costs
            input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
            output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
            total_cost = input_cost + output_cost
            
            # Update tracking
            self.total_tokens_used += total_tokens
            self.total_cost += total_cost
            self.analysis_count += 1
            
            logger.debug(f"💰 Analysis cost: ${total_cost:.4f} ({total_tokens} tokens)")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not track usage: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        avg_cost_per_analysis = self.total_cost / max(1, self.analysis_count)
        
        return {
            "total_analyses": self.analysis_count,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": round(self.total_cost, 4),
            "average_cost_per_analysis": round(avg_cost_per_analysis, 4),
            "cache_hit_rate": len(self.analysis_cache) / max(1, self.analysis_count),
            "model_name": self.model_name
        }
    
    def clear_cache(self) -> int:
        """Clear analysis cache and return number of items cleared"""
        cache_size = len(self.analysis_cache)
        self.analysis_cache.clear()
        logger.info(f"🧹 Cleared {cache_size} cached analyses")
        return cache_size
    
    async def batch_analyze(
        self, 
        messages: List[Dict[str, Any]], 
        max_concurrent: int = 5
    ) -> List[PersonalityAnalysisResult]:
        """Batch analyze multiple messages with concurrency control"""
        
        async def analyze_single(msg_data: Dict) -> PersonalityAnalysisResult:
            return await self.analyze_personality_semantic(
                msg_data['message'], 
                msg_data.get('context'),
                use_cache=True
            )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def controlled_analyze(msg_data: Dict) -> PersonalityAnalysisResult:
            async with semaphore:
                return await analyze_single(msg_data)
        
        # Execute batch analysis
        tasks = [controlled_analyze(msg_data) for msg_data in messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Batch analysis failed for message {i}: {result}")
                # Create fallback for failed analysis
                fallback = self._create_fallback_analysis(messages[i]['message'], time.time())
                valid_results.append(fallback)
            else:
                valid_results.append(result)
        
        logger.info(f"✅ Completed batch analysis of {len(messages)} messages")
        return valid_results


# Utility functions for easy integration
async def create_gemini_service(api_key: str) -> GeminiPersonalityService:
    """Factory function to create configured Gemini service"""
    return GeminiPersonalityService(api_key)


def convert_to_message_analysis(result: PersonalityAnalysisResult) -> Dict[str, Any]:
    """Convert GeminiPersonalityService result to MessageAnalysis format"""
    return {
        "communication_insights": result.communication_insights,
        "trading_insights": result.trading_insights,
        "emotional_state": result.emotional_state,
        "service_needs": result.service_needs,
        "sales_opportunity": result.sales_opportunity,
        "intent_analysis": result.intent_analysis,
        "confidence_score": result.confidence_score,
        "analysis_timestamp": result.analysis_timestamp,
        "processing_time_ms": result.processing_time_ms
    }
