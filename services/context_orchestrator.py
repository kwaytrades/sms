"""
Context Orchestrator - 3-Layer Memory Retrieval System
Handles intelligent context assembly for SMS Trading Bot conversational awareness
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from services.memory_manager import MemoryManager, AgentType, MessageDirection, EmotionalState, EmotionType

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ContextRelevance:
    """Context piece with relevance scoring"""
    content: str
    source: str  # "immediate", "semantic", "summary"
    relevance_score: float
    timestamp: datetime
    topics: List[str]
    context_type: str  # "conversation", "trading", "emotional", "user_profile"
    metadata: Dict[str, Any] = None

@dataclass
class SymbolInference:
    """Symbol inference from context"""
    symbol: str
    confidence: float
    source: str  # "explicit", "company_name", "context_inference"
    reasoning: str

@dataclass
class ConversationThread:
    """Conversation thread tracking"""
    thread_id: str
    current_topic: str
    active_symbols: List[str]
    emotional_state: str
    last_updated: datetime
    message_count: int

@dataclass
class StructuredContext:
    """Final structured context for LLM"""
    # Layer 1: Immediate Context (Last 3 exchanges)
    immediate_context: List[Dict[str, Any]]
    conversation_thread: ConversationThread
    
    # Layer 2: Semantic Context (Relevant memories)
    semantic_memories: List[ContextRelevance]
    
    # Layer 3: Summary Context (User patterns)
    user_profile: Dict[str, Any]
    conversation_summaries: List[Dict[str, Any]]
    
    # Context Intelligence
    inferred_symbols: List[SymbolInference]
    context_confidence: float
    missing_context_flags: List[str]
    
    # Metadata
    retrieval_time_ms: float
    layer_weights: Dict[str, float]
    compression_applied: bool

class ContextOrchestrator:
    """
    Intelligent 3-layer context retrieval and assembly system
    Transforms raw memory into structured, actionable context for LLM
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
        # Context configuration
        self.max_immediate_messages = 6  # 3 exchanges (user+bot pairs)
        self.max_semantic_memories = 5
        self.max_conversation_summaries = 3
        self.context_relevance_threshold = 0.3
        
        # Symbol inference patterns
        self.stock_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        self.company_mappings = {
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT', 'google': 'GOOGL',
            'nvidia': 'NVDA', 'amazon': 'AMZN', 'meta': 'META', 'netflix': 'NFLX',
            'amd': 'AMD', 'intel': 'INTC', 'spotify': 'SPOT', 'uber': 'UBER',
            'disney': 'DIS', 'boeing': 'BA', 'goldman': 'GS', 'jpmorgan': 'JPM'
        }
        
        # Context caching
        self.context_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.retrieval_stats = defaultdict(list)
        
    async def get_structured_context(
        self,
        user_id: str,
        current_message: str,
        intent: Dict[str, Any] = None,
        agent_type: AgentType = AgentType.TRADING
    ) -> StructuredContext:
        """
        Main entry point - Get fully structured 3-layer context
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = f"context:{user_id}:{hash(current_message[:50])}:{agent_type.value}"
            cached_context = await self._get_cached_context(cache_key)
            if cached_context:
                logger.info(f"Context cache hit for user {user_id}")
                return cached_context
            
            # Layer 1: Immediate Context
            immediate_task = self._get_immediate_context(user_id, agent_type)
            
            # Layer 2: Semantic Context  
            semantic_task = self._get_semantic_context(user_id, current_message, agent_type)
            
            # Layer 3: Summary Context
            summary_task = self._get_summary_context(user_id, agent_type)
            
            # Execute all layers in parallel
            immediate_context, semantic_memories, (user_profile, summaries) = await asyncio.gather(
                immediate_task, semantic_task, summary_task,
                return_exceptions=True
            )
            
            # Handle exceptions gracefully
            if isinstance(immediate_context, Exception):
                logger.warning(f"Immediate context failed: {immediate_context}")
                immediate_context = []
            
            if isinstance(semantic_memories, Exception):
                logger.warning(f"Semantic context failed: {semantic_memories}")
                semantic_memories = []
            
            if isinstance((user_profile, summaries), Exception):
                logger.warning(f"Summary context failed: {(user_profile, summaries)}")
                user_profile, summaries = {}, []
            
            # Context Intelligence Processing
            conversation_thread = self._extract_conversation_thread(immediate_context)
            inferred_symbols = await self._infer_symbols_from_context(
                current_message, immediate_context, semantic_memories
            )
            
            # Calculate context confidence and missing flags
            context_confidence = self._calculate_context_confidence(
                immediate_context, semantic_memories, user_profile
            )
            missing_flags = self._identify_missing_context(
                current_message, immediate_context, semantic_memories
            )
            
            # Apply intelligent compression if needed
            compression_applied = False
            if self._needs_compression(immediate_context, semantic_memories, summaries):
                immediate_context, semantic_memories, summaries = self._compress_context(
                    immediate_context, semantic_memories, summaries
                )
                compression_applied = True
            
            # Calculate layer weights based on context quality
            layer_weights = self._calculate_layer_weights(
                immediate_context, semantic_memories, user_profile
            )
            
            # Assembly structured context
            structured_context = StructuredContext(
                immediate_context=immediate_context,
                conversation_thread=conversation_thread,
                semantic_memories=semantic_memories,
                user_profile=user_profile,
                conversation_summaries=summaries,
                inferred_symbols=inferred_symbols,
                context_confidence=context_confidence,
                missing_context_flags=missing_flags,
                retrieval_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                layer_weights=layer_weights,
                compression_applied=compression_applied
            )
            
            # Cache the result
            await self._cache_context(cache_key, structured_context)
            
            # Log performance metrics
            retrieval_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.retrieval_stats[user_id].append(retrieval_time)
            logger.info(f"Context retrieved for {user_id} in {retrieval_time:.1f}ms")
            
            return structured_context
            
        except Exception as e:
            logger.exception(f"Context orchestration failed for user {user_id}: {e}")
            # Return minimal context on failure
            return self._create_fallback_context(current_message, user_id, agent_type)
    
    async def _get_immediate_context(self, user_id: str, agent_type: AgentType) -> List[Dict[str, Any]]:
        """Layer 1: Get last 3 message exchanges with conversation flow analysis"""
        try:
            # Get raw STM messages
            stm_messages = await self.memory_manager._get_short_term_memory(user_id, agent_type)
            
            if not stm_messages:
                return []
            
            # Process into conversation pairs and add flow analysis
            processed_messages = []
            for i, msg in enumerate(stm_messages[:self.max_immediate_messages]):
                # Add conversation flow context
                msg['position_in_conversation'] = i
                msg['is_recent'] = i < 3
                
                # Add topic continuity analysis
                if i > 0:
                    prev_msg = stm_messages[i-1]
                    msg['topic_continuation'] = self._analyze_topic_continuation(prev_msg, msg)
                else:
                    msg['topic_continuation'] = 'conversation_start'
                
                # Extract immediate context symbols
                msg['mentioned_symbols'] = self._extract_symbols_from_text(msg.get('content', ''))
                
                processed_messages.append(msg)
            
            logger.debug(f"Retrieved {len(processed_messages)} immediate context messages")
            return processed_messages
            
        except Exception as e:
            logger.exception(f"Immediate context retrieval failed: {e}")
            return []
    
    async def _get_semantic_context(
        self, 
        user_id: str, 
        current_message: str, 
        agent_type: AgentType
    ) -> List[ContextRelevance]:
        """Layer 2: Get semantically relevant memories with advanced scoring"""
        try:
            # Extract query context for semantic search
            query_symbols = self._extract_symbols_from_text(current_message)
            query_topics = self._extract_trading_topics(current_message)
            
            # Build enhanced search query
            search_query = current_message
            if query_symbols:
                search_query += f" {' '.join(query_symbols)}"
            if query_topics:
                search_query += f" {' '.join(query_topics)}"
            
            # Get vector search results
            vector_results = await self.memory_manager._enhanced_vector_search(
                user_id, search_query, agent_type, top_k=self.max_semantic_memories * 2
            )
            
            # Convert to ContextRelevance objects with enhanced scoring
            semantic_memories = []
            for result in vector_results:
                relevance_score = self._calculate_semantic_relevance(
                    result, current_message, query_symbols, query_topics
                )
                
                if relevance_score >= self.context_relevance_threshold:
                    memory = ContextRelevance(
                        content=result.get('content', ''),
                        source="semantic",
                        relevance_score=relevance_score,
                        timestamp=datetime.fromisoformat(result.get('timestamp', datetime.utcnow().isoformat())),
                        topics=result.get('topics', []),
                        context_type=result.get('memory_type', 'conversation'),
                        metadata=result
                    )
                    semantic_memories.append(memory)
            
            # Sort by relevance and limit results
            semantic_memories.sort(key=lambda x: x.relevance_score, reverse=True)
            semantic_memories = semantic_memories[:self.max_semantic_memories]
            
            logger.debug(f"Retrieved {len(semantic_memories)} semantic memories")
            return semantic_memories
            
        except Exception as e:
            logger.exception(f"Semantic context retrieval failed: {e}")
            return []
    
    async def _get_summary_context(
        self, 
        user_id: str, 
        agent_type: AgentType
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Layer 3: Get user profile and conversation summaries"""
        try:
            # Get user profile and conversation summaries in parallel
            profile_task = self.memory_manager._get_user_profile(user_id)
            summaries_task = self.memory_manager._get_conversation_summaries(
                user_id, agent_type, limit=self.max_conversation_summaries
            )
            
            user_profile, summaries = await asyncio.gather(profile_task, summaries_task)
            
            # Enhance user profile with trading intelligence
            if user_profile:
                user_profile = self._enhance_user_profile(user_profile)
            
            # Enhance summaries with relevance scoring
            enhanced_summaries = []
            for summary in summaries:
                # Add recency weight
                summary_date = summary.get('timestamp')
                if isinstance(summary_date, str):
                    summary_date = datetime.fromisoformat(summary_date)
                
                days_old = (datetime.utcnow() - summary_date).days if summary_date else 999
                recency_weight = max(0.1, 1.0 - (days_old * 0.1))
                
                summary['recency_weight'] = recency_weight
                summary['importance_adjusted'] = summary.get('importance_score', 0.5) * recency_weight
                
                enhanced_summaries.append(summary)
            
            # Sort by adjusted importance
            enhanced_summaries.sort(key=lambda x: x.get('importance_adjusted', 0), reverse=True)
            
            logger.debug(f"Retrieved user profile and {len(enhanced_summaries)} summaries")
            return user_profile, enhanced_summaries
            
        except Exception as e:
            logger.exception(f"Summary context retrieval failed: {e}")
            return {}, []
    
    def _extract_conversation_thread(self, immediate_context: List[Dict[str, Any]]) -> ConversationThread:
        """Extract and analyze conversation thread from immediate context"""
        if not immediate_context:
            return ConversationThread(
                thread_id="new_conversation",
                current_topic="general",
                active_symbols=[],
                emotional_state="neutral",
                last_updated=datetime.utcnow(),
                message_count=0
            )
        
        # Extract active symbols from recent messages
        all_symbols = []
        topics = []
        emotions = []
        
        for msg in immediate_context[:3]:  # Last 3 messages
            all_symbols.extend(msg.get('mentioned_symbols', []))
            if msg.get('topics'):
                topics.extend(msg['topics'])
            
            # Extract emotional state
            emotional_state = msg.get('emotional_state', {})
            if emotional_state:
                emotions.append(emotional_state.get('emotion_type', 'neutral'))
        
        # Determine current topic
        if topics:
            topic_counts = Counter(topics)
            current_topic = topic_counts.most_common(1)[0][0]
        else:
            current_topic = "general"
        
        # Determine emotional state
        if emotions:
            emotion_counts = Counter(emotions)
            current_emotion = emotion_counts.most_common(1)[0][0]
        else:
            current_emotion = "neutral"
        
        # Get unique active symbols
        active_symbols = list(set(all_symbols))
        
        return ConversationThread(
            thread_id=f"thread_{hash(''.join(active_symbols))}"[:12],
            current_topic=current_topic,
            active_symbols=active_symbols,
            emotional_state=current_emotion,
            last_updated=datetime.utcnow(),
            message_count=len(immediate_context)
        )
    
    async def _infer_symbols_from_context(
        self,
        current_message: str,
        immediate_context: List[Dict[str, Any]],
        semantic_memories: List[ContextRelevance]
    ) -> List[SymbolInference]:
        """Intelligent symbol inference from multi-layer context"""
        inferred_symbols = []
        
        # 1. Explicit symbols in current message
        explicit_symbols = self._extract_symbols_from_text(current_message)
        for symbol in explicit_symbols:
            inferred_symbols.append(SymbolInference(
                symbol=symbol,
                confidence=0.95,
                source="explicit",
                reasoning=f"Symbol '{symbol}' explicitly mentioned in message"
            ))
        
        # 2. Company name mapping
        message_lower = current_message.lower()
        for company, symbol in self.company_mappings.items():
            if company in message_lower and symbol not in explicit_symbols:
                inferred_symbols.append(SymbolInference(
                    symbol=symbol,
                    confidence=0.90,
                    source="company_name",
                    reasoning=f"Company name '{company}' mapped to {symbol}"
                ))
        
        # 3. Context inference from recent conversation
        if not explicit_symbols and immediate_context:
            # Look for recent symbols in conversation
            recent_symbols = []
            for msg in immediate_context[:3]:  # Last 3 messages
                recent_symbols.extend(msg.get('mentioned_symbols', []))
            
            if recent_symbols:
                # Use most frequently mentioned symbol
                symbol_counts = Counter(recent_symbols)
                most_common_symbol = symbol_counts.most_common(1)[0][0]
                
                # Check if current message could be referring to this symbol
                if self._message_could_reference_symbol(current_message, most_common_symbol):
                    inferred_symbols.append(SymbolInference(
                        symbol=most_common_symbol,
                        confidence=0.75,
                        source="context_inference",
                        reasoning=f"Recent conversation focused on {most_common_symbol}, message context suggests continuation"
                    ))
        
        # 4. Semantic memory inference
        if not inferred_symbols and semantic_memories:
            # Check semantic memories for relevant symbols
            symbol_relevance = defaultdict(float)
            for memory in semantic_memories[:3]:  # Top 3 memories
                for topic in memory.topics:
                    if self.stock_pattern.match(topic) and len(topic) <= 5:
                        symbol_relevance[topic] += memory.relevance_score
            
            if symbol_relevance:
                best_symbol = max(symbol_relevance.items(), key=lambda x: x[1])
                if best_symbol[1] > 0.5:  # Relevance threshold
                    inferred_symbols.append(SymbolInference(
                        symbol=best_symbol[0],
                        confidence=0.60,
                        source="semantic_memory",
                        reasoning=f"Semantic analysis suggests {best_symbol[0]} based on historical context"
                    ))
        
        # Remove duplicates and sort by confidence
        unique_symbols = {}
        for inference in inferred_symbols:
            if inference.symbol not in unique_symbols or inference.confidence > unique_symbols[inference.symbol].confidence:
                unique_symbols[inference.symbol] = inference
        
        result = list(unique_symbols.values())
        result.sort(key=lambda x: x.confidence, reverse=True)
        
        return result
    
    def _calculate_context_confidence(
        self,
        immediate_context: List[Dict[str, Any]],
        semantic_memories: List[ContextRelevance],
        user_profile: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in context quality"""
        confidence_factors = []
        
        # Immediate context confidence
        if immediate_context:
            immediate_confidence = min(len(immediate_context) / 6, 1.0)  # 6 = 3 exchanges
            confidence_factors.append(immediate_confidence * 0.4)
        
        # Semantic memory confidence
        if semantic_memories:
            avg_relevance = sum(m.relevance_score for m in semantic_memories) / len(semantic_memories)
            semantic_confidence = avg_relevance
            confidence_factors.append(semantic_confidence * 0.3)
        
        # User profile confidence
        if user_profile:
            profile_completeness = len(user_profile) / 20  # Assume 20 key fields
            profile_confidence = min(profile_completeness, 1.0)
            confidence_factors.append(profile_confidence * 0.3)
        
        return sum(confidence_factors) if confidence_factors else 0.2
    
    def _identify_missing_context(
        self,
        current_message: str,
        immediate_context: List[Dict[str, Any]],
        semantic_memories: List[ContextRelevance]
    ) -> List[str]:
        """Identify what context is missing for better inference"""
        missing_flags = []
        
        # Check for price references without symbols
        if any(word in current_message.lower() for word in ['$', 'price', 'support', 'resistance', 'target']):
            symbols_found = bool(self._extract_symbols_from_text(current_message))
            recent_symbols = bool(any(msg.get('mentioned_symbols') for msg in immediate_context[:3]))
            
            if not symbols_found and not recent_symbols:
                missing_flags.append("symbol_reference_unclear")
        
        # Check for pronoun references without context
        if any(word in current_message.lower() for word in ['it', 'this', 'that', 'they']):
            if len(immediate_context) < 2:
                missing_flags.append("pronoun_without_context")
        
        # Check for conversation continuity
        if not immediate_context:
            missing_flags.append("no_conversation_history")
        elif len(immediate_context) < 3:
            missing_flags.append("limited_conversation_history")
        
        # Check for user profile completeness
        if not semantic_memories:
            missing_flags.append("no_historical_context")
        
        return missing_flags
    
    # Helper methods
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract stock symbols from text with filtering"""
        if not text:
            return []
        
        # Find potential symbols
        potential_symbols = self.stock_pattern.findall(text.upper())
        
        # Filter out common false positives
        false_positives = {'YO', 'THE', 'AND', 'FOR', 'YOU', 'ARE', 'BUT', 'NOT', 'CAN', 'ALL', 'ANY', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW'}
        
        valid_symbols = [symbol for symbol in potential_symbols 
                        if symbol not in false_positives and 1 <= len(symbol) <= 5]
        
        return list(set(valid_symbols))
    
    def _extract_trading_topics(self, text: str) -> List[str]:
        """Extract trading-related topics from text"""
        trading_terms = {
            'earnings', 'dividend', 'split', 'merger', 'acquisition', 'ipo', 'buyback',
            'guidance', 'revenue', 'profit', 'loss', 'beat', 'miss', 'outlook',
            'bull', 'bear', 'rally', 'dip', 'crash', 'correction', 'breakout',
            'support', 'resistance', 'trend', 'momentum', 'volume', 'volatility'
        }
        
        text_lower = text.lower()
        found_topics = [term for term in trading_terms if term in text_lower]
        return found_topics
    
    def _analyze_topic_continuation(self, prev_msg: Dict, current_msg: Dict) -> str:
        """Analyze if current message continues the previous topic"""
        prev_topics = set(prev_msg.get('topics', []))
        current_topics = set(current_msg.get('topics', []))
        
        if prev_topics & current_topics:  # Intersection
            return 'topic_continuation'
        elif prev_topics and current_topics:
            return 'topic_shift'
        else:
            return 'topic_unclear'
    
    def _calculate_semantic_relevance(
        self,
        result: Dict,
        current_message: str,
        query_symbols: List[str],
        query_topics: List[str]
    ) -> float:
        """Calculate enhanced semantic relevance score"""
        base_score = result.get('score', 0.0)
        
        # Symbol matching bonus
        result_topics = result.get('topics', [])
        symbol_matches = len(set(query_symbols) & set(result_topics))
        symbol_bonus = symbol_matches * 0.2
        
        # Topic matching bonus
        topic_matches = len(set(query_topics) & set(result_topics))
        topic_bonus = topic_matches * 0.1
        
        # Recency bonus
        timestamp = result.get('timestamp')
        if timestamp:
            try:
                result_time = datetime.fromisoformat(timestamp)
                days_old = (datetime.utcnow() - result_time).days
                recency_bonus = max(0, 0.2 - (days_old * 0.02))
            except:
                recency_bonus = 0
        else:
            recency_bonus = 0
        
        # Emotional relevance bonus
        emotional_weight = result.get('emotional_weight', 1.0)
        emotional_bonus = (emotional_weight - 1.0) * 0.1
        
        total_score = base_score + symbol_bonus + topic_bonus + recency_bonus + emotional_bonus
        return min(total_score, 1.0)
    
    def _message_could_reference_symbol(self, message: str, symbol: str) -> bool:
        """Check if message could be referencing a specific symbol"""
        # Look for price-related words, question words, or trading actions
        reference_indicators = [
            'price', 'support', 'resistance', 'target', 'buy', 'sell', 'hold',
            'what', 'how', 'should', 'think', 'opinion', 'analysis', '$'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in reference_indicators)
    
    def _enhance_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance user profile with derived insights"""
        enhanced_profile = user_profile.copy()
        
        # Add trading sophistication score
        risk_tolerance = user_profile.get('risk_tolerance', 'medium')
        trading_style = user_profile.get('trading_style', 'swing')
        confidence_score = user_profile.get('confidence_score', 0.5)
        
        sophistication_map = {
            'day': 0.8, 'swing': 0.6, 'long': 0.4
        }
        sophistication = sophistication_map.get(trading_style, 0.5)
        
        enhanced_profile['trading_sophistication'] = min(sophistication + confidence_score, 1.0)
        
        # Add communication preferences summary
        enhanced_profile['response_style_preference'] = {
            'technical_depth': 'high' if sophistication > 0.7 else 'medium',
            'explanation_level': 'detailed' if confidence_score < 0.5 else 'concise',
            'risk_warnings': 'high' if risk_tolerance == 'low' else 'standard'
        }
        
        return enhanced_profile
    
    def _needs_compression(
        self,
        immediate_context: List[Dict],
        semantic_memories: List[ContextRelevance],
        summaries: List[Dict]
    ) -> bool:
        """Check if context needs compression"""
        # Rough token estimation
        estimated_tokens = 0
        estimated_tokens += len(str(immediate_context)) // 4  # Rough token conversion
        estimated_tokens += len(str(semantic_memories)) // 4
        estimated_tokens += len(str(summaries)) // 4
        
        return estimated_tokens > 1500  # Conservative limit for context
    
    def _compress_context(
        self,
        immediate_context: List[Dict],
        semantic_memories: List[ContextRelevance],
        summaries: List[Dict]
    ) -> Tuple[List[Dict], List[ContextRelevance], List[Dict]]:
        """Compress context while preserving important information"""
        # Keep only essential fields in immediate context
        compressed_immediate = []
        for msg in immediate_context[:4]:  # Limit to 4 messages
            compressed_msg = {
                'content': msg.get('content', ''),
                'direction': msg.get('direction'),
                'timestamp': msg.get('timestamp'),
                'mentioned_symbols': msg.get('mentioned_symbols', []),
                'topics': msg.get('topics', [])[:3]  # Limit topics
            }
            compressed_immediate.append(compressed_msg)
        
        # Keep top semantic memories
        compressed_semantic = semantic_memories[:3]
        
        # Keep most recent summaries
        compressed_summaries = summaries[:2]
        
        return compressed_immediate, compressed_semantic, compressed_summaries
    
    def _calculate_layer_weights(
        self,
        immediate_context: List[Dict],
        semantic_memories: List[ContextRelevance],
        user_profile: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate weights for each context layer based on quality"""
        weights = {}
        
        # Immediate context weight
        immediate_quality = min(len(immediate_context) / 6, 1.0)
        weights['immediate'] = 0.4 + (immediate_quality * 0.2)
        
        # Semantic context weight
        if semantic_memories:
            avg_relevance = sum(m.relevance_score for m in semantic_memories) / len(semantic_memories)
            weights['semantic'] = 0.3 + (avg_relevance * 0.2)
        else:
            weights['semantic'] = 0.1
        
        # Summary context weight
        profile_completeness = len(user_profile) / 20 if user_profile else 0
        weights['summary'] = 0.3 + (profile_completeness * 0.1)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {'immediate': 0.4, 'semantic': 0.3, 'summary': 0.3}
        
        return weights
    
    def _create_fallback_context(
        self,
        current_message: str,
        user_id: str,
        agent_type: AgentType
    ) -> StructuredContext:
        """Create minimal fallback context when retrieval fails"""
        return StructuredContext(
            immediate_context=[],
            conversation_thread=ConversationThread(
                thread_id="fallback",
                current_topic="general",
                active_symbols=self._extract_symbols_from_text(current_message),
                emotional_state="neutral",
                last_updated=datetime.utcnow(),
                message_count=0
            ),
            semantic_memories=[],
            user_profile={},
            conversation_summaries=[],
            inferred_symbols=[],
            context_confidence=0.1,
            missing_context_flags=["context_retrieval_failed"],
            retrieval_time_ms=0.0,
            layer_weights={'immediate': 0.4, 'semantic': 0.3, 'summary': 0.3},
            compression_applied=False
        )
    
    async def _get_cached_context(self, cache_key: str) -> Optional[StructuredContext]:
        """Get cached context if valid"""
        cached_data = self.context_cache.get(cache_key)
        if cached_data:
            cache_time = cached_data.get('cached_at', datetime.min)
            if (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl:
                return cached_data.get('context')
        return None
    
    async def _cache_context(self, cache_key: str, context: StructuredContext):
        """Cache context for performance"""
        self.context_cache[cache_key] = {
            'context': context,
            'cached_at': datetime.utcnow()
        }
        
        # Limit cache size
        if len(self.context_cache) > 100:
            # Remove oldest entries
            sorted_keys = sorted(self.context_cache.keys(),
                               key=lambda k: self.context_cache[k]['cached_at'])
            for key in sorted_keys[:50]:
                del self.context_cache[key]
    
    def format_context_for_llm(self, context: StructuredContext) -> str:
        """Format structured context for LLM consumption"""
        formatted_parts = []
        
        # Add conversation thread summary
        thread = context.conversation_thread
        if thread.active_symbols or thread.current_topic != "general":
            formatted_parts.append(f"CONVERSATION CONTEXT:")
            formatted_parts.append(f"- Current topic: {thread.current_topic}")
            formatted_parts.append(f"- Active symbols: {', '.join(thread.active_symbols) if thread.active_symbols else 'None'}")
            formatted_parts.append(f"- User mood: {thread.emotional_state}")
        
        # Add recent conversation
        if context.immediate_context:
            formatted_parts.append(f"\nRECENT CONVERSATION:")
            for msg in context.immediate_context[:4]:
                direction = "You" if msg.get('direction') == 'bot' else "User"
                content = msg.get('content', '')[:100]  # Truncate long messages
                formatted_parts.append(f"- {direction}: {content}")
        
        # Add relevant memories
        if context.semantic_memories:
            formatted_parts.append(f"\nRELEVANT HISTORY:")
            for memory in context.semantic_memories[:3]:
                content = memory.content[:80]  # Truncate
                formatted_parts.append(f"- {content} (relevance: {memory.relevance_score:.1f})")
        
        # Add user profile insights
        if context.user_profile:
            profile = context.user_profile
            style = profile.get('response_style_preference', {})
            formatted_parts.append(f"\nUSER PROFILE:")
            formatted_parts.append(f"- Trading style: {profile.get('trading_style', 'unknown')}")
            formatted_parts.append(f"- Risk tolerance: {profile.get('risk_tolerance', 'medium')}")
            formatted_parts.append(f"- Technical depth: {style.get('technical_depth', 'medium')}")
        
        # Add symbol inferences
        if context.inferred_symbols:
            formatted_parts.append(f"\nSYMBOL CONTEXT:")
            for inference in context.inferred_symbols[:2]:
                formatted_parts.append(f"- {inference.symbol}: {inference.reasoning} (confidence: {inference.confidence:.1f})")
        
        # Add context instructions
        if context.missing_context_flags:
            formatted_parts.append(f"\nCONTEXT NOTES:")
            for flag in context.missing_context_flags[:3]:
                if flag == "symbol_reference_unclear":
                    formatted_parts.append("- User may be referring to a stock without specifying symbol")
                elif flag == "no_conversation_history":
                    formatted_parts.append("- This appears to be the start of a new conversation")
        
        return "\n".join(formatted_parts)
    
    async def get_performance_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        if user_id and user_id in self.retrieval_stats:
            user_times = self.retrieval_stats[user_id]
            return {
                'user_id': user_id,
                'avg_retrieval_time_ms': sum(user_times) / len(user_times),
                'retrieval_count': len(user_times),
                'cache_size': len(self.context_cache)
            }
        else:
            all_times = [time for times in self.retrieval_stats.values() for time in times]
            return {
                'global_avg_retrieval_time_ms': sum(all_times) / len(all_times) if all_times else 0,
                'total_retrievals': len(all_times),
                'active_users': len(self.retrieval_stats),
                'cache_size': len(self.context_cache)
            }
