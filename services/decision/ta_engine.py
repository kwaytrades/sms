# ta_engine.py
"""
Decision Engine - Advanced Trading Analysis System
Combines CheatCode 4/4 signals with contextual market factors for intelligent trading decisions
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

from cheatcode_engine import CheatCodeSignal, SignalStrength, TrendDirection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionFactor(Enum):
    """Decision factor categories"""
    TECHNICAL = "technical"
    PRICE_POSITION = "price_position"
    VOLUME = "volume"
    MARKET_CONTEXT = "market_context"
    TIME_BASED = "time_based"
    RISK_MANAGEMENT = "risk_management"

class DecisionReason(Enum):
    """Reasons for decision adjustments"""
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    NEAR_RESISTANCE = "near_resistance"
    NEAR_SUPPORT = "near_support"
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DECLINE = "volume_decline"
    MARKET_HEADWIND = "market_headwind"
    MARKET_TAILWIND = "market_tailwind"
    EARNINGS_RISK = "earnings_risk"
    OPTIONS_EXPIRY = "options_expiry"
    SECTOR_WEAKNESS = "sector_weakness"
    HIGH_VOLATILITY = "high_volatility"
    MOMENTUM_DIVERGENCE = "momentum_divergence"
    RISK_TOO_HIGH = "risk_too_high"

@dataclass
class DecisionFactorScore:
    """Individual decision factor with score and reasoning"""
    factor_type: DecisionFactor
    reason: DecisionReason
    score: float  # -2.0 to +2.0
    description: str
    confidence: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'factor_type': self.factor_type.value,
            'reason': self.reason.value,
            'score': self.score,
            'description': self.description,
            'confidence': self.confidence
        }

@dataclass
class DecisionContext:
    """Market and stock context for decision making"""
    symbol: str
    current_price: float
    
    # Technical indicators
    rsi_14: float
    rsi_2: float
    bb_position: float  # 0-1 where 0.5 is middle of BB
    macd_signal: float
    volume_ratio: float  # Current volume / 20-day avg
    
    # Price levels
    resistance_distance: float  # % to nearest resistance
    support_distance: float     # % to nearest support
    ma_20_distance: float      # % from 20-day MA
    ma_50_distance: float      # % from 50-day MA
    
    # Market context
    market_trend: TrendDirection  # SPY trend
    sector_trend: TrendDirection  # Sector trend
    vix_level: float
    market_correlation: float
    
    # Time-based factors
    days_to_earnings: Optional[int]
    is_options_expiry_week: bool
    is_end_of_quarter: bool
    market_session: str  # "pre", "regular", "after"
    
    # Volume analysis
    volume_trend_5d: float      # 5-day volume trend
    volume_spike_detected: bool
    avg_volume_20d: float
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['market_trend'] = self.market_trend.value
        data['sector_trend'] = self.sector_trend.value
        return data

@dataclass
class EnhancedTradingDecision:
    """Enhanced trading decision combining CheatCode + Decision Engine"""
    
    # Original CheatCode signal
    cheatcode_signal: CheatCodeSignal
    
    # Decision engine analysis
    decision_factors: List[DecisionFactorScore]
    context: DecisionContext
    
    # Enhanced recommendation
    base_score: float           # CheatCode 0-4 score
    adjustment_score: float     # Decision factors -2 to +2
    final_score: float          # Combined score
    final_signal: SignalStrength
    
    # Enhanced trading advice
    recommended_action: str
    reasoning: str
    risk_assessment: str
    timing_advice: str
    position_sizing_adjustment: float  # Multiplier 0.5-1.5
    
    # Specific guidance
    wait_conditions: List[str]  # What to wait for before entering
    exit_conditions: List[str]  # What to watch for exits
    risk_factors: List[str]     # Current risk considerations
    
    # Confidence and alerts
    decision_confidence: float
    alerts_to_set: List[str]
    
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.cheatcode_signal.symbol,
            'timestamp': self.timestamp.isoformat(),
            'cheatcode_analysis': {
                'base_score': self.base_score,
                'signal_strength': self.cheatcode_signal.signal_strength.value,
                'confidence': self.cheatcode_signal.confidence,
                'component_scores': {
                    'cloud': self.cheatcode_signal.cloud_score,
                    'swing': self.cheatcode_signal.swing_score,
                    'squeeze': self.cheatcode_signal.squeeze_score,
                    'pattern': self.cheatcode_signal.pattern_score
                }
            },
            'decision_engine': {
                'adjustment_score': self.adjustment_score,
                'final_score': self.final_score,
                'final_signal': self.final_signal.value,
                'decision_confidence': self.decision_confidence,
                'factors': [factor.to_dict() for factor in self.decision_factors]
            },
            'trading_recommendation': {
                'action': self.recommended_action,
                'entry_price': self.cheatcode_signal.entry_price,
                'stop_loss': self.cheatcode_signal.stop_loss,
                'take_profit': self.cheatcode_signal.take_profit,
                'position_size_pct': self.cheatcode_signal.position_size_pct * self.position_sizing_adjustment,
                'position_sizing_adjustment': self.position_sizing_adjustment
            },
            'analysis': {
                'reasoning': self.reasoning,
                'risk_assessment': self.risk_assessment,
                'timing_advice': self.timing_advice,
                'wait_conditions': self.wait_conditions,
                'exit_conditions': self.exit_conditions,
                'risk_factors': self.risk_factors,
                'alerts_to_set': self.alerts_to_set
            },
            'market_context': self.context.to_dict()
        }

class DecisionEngine:
    """
    Advanced Decision Engine that enhances CheatCode signals with contextual analysis
    """
    
    def __init__(self):
        # Technical thresholds
        self.rsi_overbought = 70
        self.rsi_very_overbought = 80
        self.rsi_oversold = 30
        self.rsi_very_oversold = 20
        
        # Distance thresholds (%)
        self.resistance_threshold = 0.02  # 2%
        self.support_threshold = 0.02     # 2%
        
        # Volume thresholds
        self.volume_spike_threshold = 2.0   # 2x average
        self.volume_low_threshold = 0.5     # 0.5x average
        
        # Market context thresholds
        self.high_vix_threshold = 25
        self.very_high_vix_threshold = 35
        
    async def enhance_signal(self, cheatcode_signal: CheatCodeSignal, 
                           market_data: pd.DataFrame, 
                           context: DecisionContext) -> EnhancedTradingDecision:
        """
        Main enhancement function - takes CheatCode signal and adds decision engine analysis
        """
        try:
            logger.info(f"Enhancing signal for {cheatcode_signal.symbol}")
            
            # Calculate all decision factors
            decision_factors = await self._calculate_decision_factors(cheatcode_signal, market_data, context)
            
            # Calculate adjustment score
            adjustment_score = sum(factor.score for factor in decision_factors)
            adjustment_score = max(-2.0, min(2.0, adjustment_score))  # Clamp to -2 to +2
            
            # Calculate final score and signal
            base_score = cheatcode_signal.total_score
            final_score = base_score + adjustment_score
            final_signal = self._determine_final_signal(final_score, cheatcode_signal.signal_strength, decision_factors)
            
            # Generate enhanced recommendations
            recommended_action = self._generate_action_recommendation(final_signal, decision_factors)
            reasoning = self._generate_reasoning(cheatcode_signal, decision_factors, adjustment_score)
            risk_assessment = self._assess_risk(decision_factors, context)
            timing_advice = self._generate_timing_advice(decision_factors, context)
            
            # Position sizing adjustment
            position_sizing_adjustment = self._calculate_position_adjustment(decision_factors, context)
            
            # Generate specific guidance
            wait_conditions = self._identify_wait_conditions(decision_factors, context)
            exit_conditions = self._identify_exit_conditions(decision_factors, context)
            risk_factors = self._identify_risk_factors(decision_factors, context)
            alerts_to_set = self._generate_alerts(decision_factors, context)
            
            # Calculate decision confidence
            decision_confidence = self._calculate_decision_confidence(cheatcode_signal, decision_factors, adjustment_score)
            
            # Create enhanced decision
            enhanced_decision = EnhancedTradingDecision(
                cheatcode_signal=cheatcode_signal,
                decision_factors=decision_factors,
                context=context,
                base_score=base_score,
                adjustment_score=adjustment_score,
                final_score=final_score,
                final_signal=final_signal,
                recommended_action=recommended_action,
                reasoning=reasoning,
                risk_assessment=risk_assessment,
                timing_advice=timing_advice,
                position_sizing_adjustment=position_sizing_adjustment,
                wait_conditions=wait_conditions,
                exit_conditions=exit_conditions,
                risk_factors=risk_factors,
                decision_confidence=decision_confidence,
                alerts_to_set=alerts_to_set,
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"Enhanced decision for {cheatcode_signal.symbol}: {final_signal.value} "
                       f"(Base: {base_score:.1f}, Adj: {adjustment_score:+.1f}, Final: {final_score:.1f})")
            
            return enhanced_decision
            
        except Exception as e:
            logger.error(f"Failed to enhance signal for {cheatcode_signal.symbol}: {e}")
            raise
    
    async def _calculate_decision_factors(self, signal: CheatCodeSignal, 
                                        data: pd.DataFrame, 
                                        context: DecisionContext) -> List[DecisionFactorScore]:
        """Calculate all decision factors"""
        factors = []
        
        # Technical factors
        factors.extend(self._analyze_technical_factors(context))
        
        # Price position factors  
        factors.extend(self._analyze_price_position_factors(context))
        
        # Volume factors
        factors.extend(self._analyze_volume_factors(context))
        
        # Market context factors
        factors.extend(self._analyze_market_context_factors(context))
        
        # Time-based factors
        factors.extend(self._analyze_time_based_factors(context))
        
        # Risk management factors
        factors.extend(self._analyze_risk_factors(signal, context))
        
        return factors
    
    def _analyze_technical_factors(self, context: DecisionContext) -> List[DecisionFactorScore]:
        """Analyze technical indicator factors"""
        factors = []
        
        # RSI analysis
        if context.rsi_14 > self.rsi_very_overbought:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.TECHNICAL,
                reason=DecisionReason.OVERBOUGHT,
                score=-1.0,
                description=f"Very overbought RSI at {context.rsi_14:.1f}",
                confidence=0.9
            ))
        elif context.rsi_14 > self.rsi_overbought:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.TECHNICAL,
                reason=DecisionReason.OVERBOUGHT,
                score=-0.5,
                description=f"Overbought RSI at {context.rsi_14:.1f}",
                confidence=0.8
            ))
        elif context.rsi_14 < self.rsi_very_oversold:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.TECHNICAL,
                reason=DecisionReason.OVERSOLD,
                score=1.0,
                description=f"Very oversold RSI at {context.rsi_14:.1f}",
                confidence=0.9
            ))
        elif context.rsi_14 < self.rsi_oversold:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.TECHNICAL,
                reason=DecisionReason.OVERSOLD,
                score=0.5,
                description=f"Oversold RSI at {context.rsi_14:.1f}",
                confidence=0.8
            ))
        
        # MACD divergence
        if abs(context.macd_signal) > 0.5:
            score = 0.3 if context.macd_signal > 0 else -0.3
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.TECHNICAL,
                reason=DecisionReason.MOMENTUM_DIVERGENCE,
                score=score,
                description=f"MACD showing {'bullish' if score > 0 else 'bearish'} momentum",
                confidence=0.6
            ))
        
        return factors
    
    def _analyze_price_position_factors(self, context: DecisionContext) -> List[DecisionFactorScore]:
        """Analyze price position relative to key levels"""
        factors = []
        
        # Resistance proximity
        if context.resistance_distance <= self.resistance_threshold:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.PRICE_POSITION,
                reason=DecisionReason.NEAR_RESISTANCE,
                score=-0.5,
                description=f"Within {context.resistance_distance*100:.1f}% of resistance",
                confidence=0.8
            ))
        
        # Support proximity
        if context.support_distance <= self.support_threshold:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.PRICE_POSITION,
                reason=DecisionReason.NEAR_SUPPORT,
                score=0.5,
                description=f"Within {context.support_distance*100:.1f}% of support",
                confidence=0.8
            ))
        
        # Moving average analysis
        if context.ma_20_distance > 0.05:  # 5% above 20MA
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.PRICE_POSITION,
                reason=DecisionReason.OVERBOUGHT,
                score=-0.3,
                description=f"{context.ma_20_distance*100:.1f}% above 20-day MA",
                confidence=0.6
            ))
        elif context.ma_20_distance < -0.05:  # 5% below 20MA
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.PRICE_POSITION,
                reason=DecisionReason.OVERSOLD,
                score=0.3,
                description=f"{abs(context.ma_20_distance)*100:.1f}% below 20-day MA",
                confidence=0.6
            ))
        
        return factors
    
    def _analyze_volume_factors(self, context: DecisionContext) -> List[DecisionFactorScore]:
        """Analyze volume-related factors"""
        factors = []
        
        # Volume spike
        if context.volume_spike_detected or context.volume_ratio > self.volume_spike_threshold:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.VOLUME,
                reason=DecisionReason.VOLUME_SPIKE,
                score=0.8,
                description=f"Volume spike: {context.volume_ratio:.1f}x average",
                confidence=0.9
            ))
        
        # Low volume
        elif context.volume_ratio < self.volume_low_threshold:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.VOLUME,
                reason=DecisionReason.VOLUME_DECLINE,
                score=-0.3,
                description=f"Low volume: {context.volume_ratio:.1f}x average",
                confidence=0.7
            ))
        
        # Volume trend
        if context.volume_trend_5d < -0.2:  # 20% decline over 5 days
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.VOLUME,
                reason=DecisionReason.VOLUME_DECLINE,
                score=-0.5,
                description="Declining volume trend over 5 days",
                confidence=0.7
            ))
        elif context.volume_trend_5d > 0.2:  # 20% increase over 5 days
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.VOLUME,
                reason=DecisionReason.VOLUME_SPIKE,
                score=0.5,
                description="Increasing volume trend over 5 days",
                confidence=0.7
            ))
        
        return factors
    
    def _analyze_market_context_factors(self, context: DecisionContext) -> List[DecisionFactorScore]:
        """Analyze broader market context"""
        factors = []
        
        # Market trend
        if context.market_trend == TrendDirection.BULLISH:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.MARKET_CONTEXT,
                reason=DecisionReason.MARKET_TAILWIND,
                score=0.3,
                description="Bullish market trend (SPY)",
                confidence=0.8
            ))
        elif context.market_trend == TrendDirection.BEARISH:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.MARKET_CONTEXT,
                reason=DecisionReason.MARKET_HEADWIND,
                score=-0.3,
                description="Bearish market trend (SPY)",
                confidence=0.8
            ))
        
        # Sector trend
        if context.sector_trend == TrendDirection.BEARISH:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.MARKET_CONTEXT,
                reason=DecisionReason.SECTOR_WEAKNESS,
                score=-0.5,
                description="Bearish sector trend",
                confidence=0.7
            ))
        
        # VIX analysis
        if context.vix_level > self.very_high_vix_threshold:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.MARKET_CONTEXT,
                reason=DecisionReason.HIGH_VOLATILITY,
                score=-0.4,
                description=f"Very high market fear (VIX: {context.vix_level:.1f})",
                confidence=0.8
            ))
        elif context.vix_level > self.high_vix_threshold:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.MARKET_CONTEXT,
                reason=DecisionReason.HIGH_VOLATILITY,
                score=-0.2,
                description=f"Elevated market fear (VIX: {context.vix_level:.1f})",
                confidence=0.7
            ))
        
        return factors
    
    def _analyze_time_based_factors(self, context: DecisionContext) -> List[DecisionFactorScore]:
        """Analyze time-based risk factors"""
        factors = []
        
        # Earnings proximity
        if context.days_to_earnings is not None and context.days_to_earnings <= 7:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.TIME_BASED,
                reason=DecisionReason.EARNINGS_RISK,
                score=-0.3,
                description=f"Earnings in {context.days_to_earnings} days",
                confidence=0.8
            ))
        
        # Options expiration
        if context.is_options_expiry_week:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.TIME_BASED,
                reason=DecisionReason.OPTIONS_EXPIRY,
                score=-0.2,
                description="Options expiration week - increased volatility",
                confidence=0.6
            ))
        
        # Market session
        if context.market_session in ["pre", "after"]:
            factors.append(DecisionFactorScore(
                factor_type=DecisionFactor.TIME_BASED,
                reason=DecisionReason.HIGH_VOLATILITY,
                score=-0.1,
                description=f"Extended hours trading ({context.market_session}-market)",
                confidence=0.5
            ))
        
        return factors
    
    def _analyze_risk_factors(self, signal: CheatCodeSignal, context: DecisionContext) -> List[DecisionFactorScore]:
        """Analyze overall risk management factors"""
        factors = []
        
        # Risk/reward analysis
        if signal.stop_loss > 0 and signal.entry_price > 0:
            risk_amount = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            reward_amount = abs(signal.take_profit - signal.entry_price) / signal.entry_price
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            if risk_reward_ratio < 1.5:  # Poor risk/reward
                factors.append(DecisionFactorScore(
                    factor_type=DecisionFactor.RISK_MANAGEMENT,
                    reason=DecisionReason.RISK_TOO_HIGH,
                    score=-0.4,
                    description=f"Poor risk/reward ratio: {risk_reward_ratio:.1f}:1",
                    confidence=0.9
                ))
            elif risk_reward_ratio > 3.0:  # Excellent risk/reward
                factors.append(DecisionFactorScore(
                    factor_type=DecisionFactor.RISK_MANAGEMENT,
                    reason=DecisionReason.RISK_TOO_HIGH,
                    score=0.3,
                    description=f"Excellent risk/reward ratio: {risk_reward_ratio:.1f}:1",
                    confidence=0.8
                ))
        
        return factors
    
    def _determine_final_signal(self, final_score: float, 
                              base_signal: SignalStrength, 
                              factors: List[DecisionFactorScore]) -> SignalStrength:
        """Determine final signal strength based on combined analysis"""
        
        # Major negative factors that override strong signals
        major_negatives = [f for f in factors if f.score <= -0.8]
        
        # Strong positive factors
        strong_positives = [f for f in factors if f.score >= 0.8]
        
        if final_score >= 4.0 and not major_negatives:
            return SignalStrength.STRONG_BUY
        elif final_score >= 3.0:
            return SignalStrength.BUY if not major_negatives else SignalStrength.WEAK_BUY
        elif final_score >= 2.0:
            return SignalStrength.WEAK_BUY
        elif final_score <= -4.0 and not strong_positives:
            return SignalStrength.STRONG_SELL
        elif final_score <= -3.0:
            return SignalStrength.SELL if not strong_positives else SignalStrength.WEAK_SELL
        elif final_score <= -2.0:
            return SignalStrength.WEAK_SELL
        else:
            return SignalStrength.HOLD
    
    def _generate_action_recommendation(self, signal: SignalStrength, 
                                      factors: List[DecisionFactorScore]) -> str:
        """Generate specific action recommendation"""
        
        if signal == SignalStrength.STRONG_BUY:
            return "Strong Buy - Enter position immediately"
        elif signal == SignalStrength.BUY:
            return "Buy - Good entry opportunity" 
        elif signal == SignalStrength.WEAK_BUY:
            wait_factors = [f for f in factors if f.score < -0.3]
            if wait_factors:
                return "Weak Buy - Consider waiting for better entry"
            else:
                return "Weak Buy - Small position acceptable"
        elif signal == SignalStrength.STRONG_SELL:
            return "Strong Sell - Exit/Short position immediately"
        elif signal == SignalStrength.SELL:
            return "Sell - Good exit opportunity"
        elif signal == SignalStrength.WEAK_SELL:
            return "Weak Sell - Consider reducing position"
        else:
            return "Hold - Wait for clearer signals"
    
    def _generate_reasoning(self, cheatcode_signal: CheatCodeSignal, 
                          factors: List[DecisionFactorScore], 
                          adjustment: float) -> str:
        """Generate detailed reasoning for the decision"""
        
        base_strength = cheatcode_signal.signal_strength.value
        
        reasoning = f"CheatCode 4/4 analysis shows {base_strength} "
        reasoning += f"({cheatcode_signal.total_score:.1f}/4 score). "
        
        if adjustment > 0.5:
            reasoning += f"Decision engine adds +{adjustment:.1f} due to: "
            positive_factors = [f for f in factors if f.score > 0]
            reasoning += ", ".join([f.description for f in positive_factors[:3]])
        elif adjustment < -0.5:
            reasoning += f"Decision engine subtracts {adjustment:.1f} due to: "
            negative_factors = [f for f in factors if f.score < 0]
            reasoning += ", ".join([f.description for f in negative_factors[:3]])
        else:
            reasoning += "Decision engine confirms signal with minor adjustments."
        
        return reasoning
    
    def _assess_risk(self, factors: List[DecisionFactorScore], context: DecisionContext) -> str:
        """Assess overall risk level"""
        
        risk_factors = [f for f in factors if f.score < -0.3]
        
        if len(risk_factors) >= 3:
            return "HIGH RISK - Multiple negative factors present"
        elif len(risk_factors) == 2:
            return "MODERATE RISK - Some caution warranted"
        elif len(risk_factors) == 1:
            return "LOW-MODERATE RISK - Minor concerns"
        else:
            return "LOW RISK - Favorable conditions"
    
    def _generate_timing_advice(self, factors: List[DecisionFactorScore], 
                              context: DecisionContext) -> str:
        """Generate timing-specific advice"""
        
        timing_factors = [f for f in factors if f.factor_type == DecisionFactor.TIME_BASED]
        technical_factors = [f for f in factors if f.reason in [DecisionReason.OVERBOUGHT, DecisionReason.NEAR_RESISTANCE]]
        
        if any(f.reason == DecisionReason.EARNINGS_RISK for f in timing_factors):
            return "Consider waiting until after earnings for clearer direction"
        elif technical_factors:
            return "Wait for pullback to better entry level"
        elif context.market_session != "regular":
            return "Consider waiting for regular market hours"
        else:
            return "Good timing for entry"
    
    def _calculate_position_adjustment(self, factors: List[DecisionFactorScore], 
                                     context: DecisionContext) -> float:
        """Calculate position sizing adjustment multiplier"""
        
        risk_score = sum(f.score for f in factors if f.score < 0)
        opportunity_score = sum(f.score for f in factors if f.score > 0)
        
        # Base adjustment
        adjustment = 1.0
        
        # Reduce size for high risk
        if risk_score < -1.5:
            adjustment *= 0.5
        elif risk_score < -0.8:
            adjustment *= 0.75
        
        # Increase size for high opportunity (with limits)
        if opportunity_score > 1.5:
            adjustment *= 1.3
        elif opportunity_score > 0.8:
            adjustment *= 1.15
        
        # Volatility adjustment
        if context.vix_level > 30:
            adjustment *= 0.8
        
        return max(0.25, min(1.5, adjustment))
    
    def _identify_wait_conditions(self, factors: List[DecisionFactorScore], 
                                context: DecisionContext) -> List[str]:
        """Identify specific conditions to wait for"""
        conditions = []
        
        for factor in factors:
            if factor.reason == DecisionReason.OVERBOUGHT and factor.score < -0.5:
                conditions.append(f"RSI to cool below {self.rsi_overbought}")
            elif factor.reason == DecisionReason.NEAR_RESISTANCE:
                conditions.append("Breakout above resistance with volume")
            elif factor.reason == DecisionReason.EARNINGS_RISK:
                conditions.append("Wait until after earnings announcement")
            elif factor.reason == DecisionReason.VOLUME_DECLINE:
                conditions.append("Volume increase to confirm move")
        
        return conditions
    
    def _identify_exit_conditions(self, factors: List[DecisionFactorScore], 
                                context: DecisionContext) -> List[str]:
        """Identify exit conditions to watch for"""
        conditions = []
        
        # Standard exit conditions
        conditions.append("Stop loss hit")
        conditions.append("Take profit target reached")
        
        # Dynamic exit conditions based on analysis
        if context.rsi_14 > 75:
            conditions.append("RSI reaches extremely overbought (>80)")
        
        if any(f.reason == DecisionReason.MARKET_HEADWIND for f in factors):
            conditions.append("Market trend deteriorates further")
        
        conditions.append("Volume dries up significantly")
        conditions.append("Break below key support level")
        
        return conditions
    
    def _identify_risk_factors(self, factors: List[DecisionFactorScore], 
                             context: DecisionContext) -> List[str]:
        """Identify current risk factors"""
        risks = []
        
        for factor in factors:
            if factor.score < -0.3:
                risks.append(factor.description)
        
        # Add general risk factors
        if context.vix_level > 25:
            risks.append(f"Elevated market volatility (VIX: {context.vix_level:.1f})")
        
        if context.days_to_earnings and context.days_to_earnings <= 3:
            risks.append("Earnings volatility risk")
        
        return risks[:5]  # Limit to top 5 risks
    
    def _generate_alerts(self, factors: List[DecisionFactorScore], 
                        context: DecisionContext) -> List[str]:
        """Generate useful alerts to set"""
        alerts = []
        
        # Price-based alerts
        resistance_distance = context.resistance_distance
        if resistance_distance <= 0.05:  # Within 5% of resistance
            alerts.append(f"Alert when price breaks above resistance")
        
        support_distance = context.support_distance  
        if support_distance <= 0.05:  # Within 5% of support
            alerts.append(f"Alert if price breaks below support")
        
        # Technical alerts
        if context.rsi_14 > 65:
            alerts.append("Alert when RSI drops below 60")
        elif context.rsi_14 < 35:
            alerts.append("Alert when RSI rises above 40")
        
        # Volume alerts
        if context.volume_ratio < 0.7:
            alerts.append("Alert on volume spike (>1.5x average)")
        
        return alerts
    
    def _calculate_decision_confidence(self, cheatcode_signal: CheatCodeSignal, 
                                     factors: List[DecisionFactorScore], 
                                     adjustment: float) -> float:
        """Calculate confidence in the enhanced decision"""
        
        # Base confidence from CheatCode
        base_confidence = cheatcode_signal.confidence
        
        # Factor alignment
        factor_scores = [f.score for f in factors]
        factor_alignment = 1.0 - (np.std(factor_scores) / 2.0) if factor_scores else 1.0
        
        # Adjustment magnitude (smaller adjustments = higher confidence)
        adjustment_confidence = 1.0 - min(abs(adjustment) / 2.0, 0.5)
        
        # Combined confidence
        decision_confidence = (base_confidence * 0.4 + 
                             factor_alignment * 0.3 + 
                             adjustment_confidence * 0.3)
        
        return max(0.1, min(1.0, decision_confidence))

# FastAPI integration
from fastapi import APIRouter, HTTPException

decision_router = APIRouter(prefix="/decision", tags=["Decision Engine"])

# Global decision engine instance
decision_engine = DecisionEngine()

@decision_router.post("/enhance/{symbol}")
async def enhance_signal_endpoint(symbol: str, request_data: Dict[str, Any]):
    """
    Enhance a CheatCode signal with Decision Engine analysis
    
    Body should contain:
    {
        "cheatcode_signal": {...},  // CheatCode signal object
        "market_data": {...},       // OHLCV DataFrame data  
        "context": {...}            // DecisionContext object
    }
    """
    try:
        # Parse request data
        if not all(key in request_data for key in ["cheatcode_signal", "market_data", "context"]):
            raise HTTPException(status_code=400, detail="Missing required data in request")
        
        # Convert data back to objects (simplified for example)
        cheatcode_data = request_data["cheatcode_signal"]
        market_data = pd.DataFrame(request_data["market_data"])
        context_data = request_data["context"]
        
        # Create context object (simplified - you'd need proper conversion)
        context = DecisionContext(
            symbol=symbol,
            current_price=context_data.get("current_price", 100),
            rsi_14=context_data.get("rsi_14", 50),
            rsi_2=context_data.get("rsi_2", 50),
            bb_position=context_data.get("bb_position", 0.5),
            macd_signal=context_data.get("macd_signal", 0),
            volume_ratio=context_data.get("volume_ratio", 1.0),
            resistance_distance=context_data.get("resistance_distance", 0.1),
            support_distance=context_data.get("support_distance", 0.1),
            ma_20_distance=context_data.get("ma_20_distance", 0),
            ma_50_distance=context_data.get("ma_50_distance", 0),
            market_trend=TrendDirection(context_data.get("market_trend", 0)),
            sector_trend=TrendDirection(context_data.get("sector_trend", 0)),
            vix_level=context_data.get("vix_level", 20),
            market_correlation=context_data.get("market_correlation", 0.5),
            days_to_earnings=context_data.get("days_to_earnings"),
            is_options_expiry_week=context_data.get("is_options_expiry_week", False),
            is_end_of_quarter=context_data.get("is_end_of_quarter", False),
            market_session=context_data.get("market_session", "regular"),
            volume_trend_5d=context_data.get("volume_trend_5d", 0),
            volume_spike_detected=context_data.get("volume_spike_detected", False),
            avg_volume_20d=context_data.get("avg_volume_20d", 1000000)
        )
        
        # Create mock CheatCode signal (you'd parse this properly)
        # This is simplified for the example
        
        # For demo purposes, return structure
        return {
            "success": True,
            "symbol": symbol,
            "message": "Decision engine enhancement endpoint ready",
            "note": "Full implementation requires proper object conversion"
        }
        
    except Exception as e:
        logger.error(f"Enhancement failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@decision_router.get("/test/{symbol}")
async def test_decision_engine(symbol: str):
    """Test endpoint showing decision engine capabilities"""
    
    return {
        "success": True,
        "symbol": symbol,
        "decision_engine": {
            "factors_analyzed": [factor.value for factor in DecisionFactor],
            "adjustment_range": "-2.0 to +2.0",
            "position_sizing_adjustment": "0.5x to 1.5x",
            "sample_factors": {
                "technical": ["RSI overbought (-0.5)", "MACD bullish (+0.3)"],
                "price_position": ["Near resistance (-0.5)", "Above 20MA (+0.3)"],
                "volume": ["Volume spike (+0.8)", "Low volume (-0.3)"],
                "market_context": ["Bullish market (+0.3)", "High VIX (-0.4)"],
                "time_based": ["Pre-earnings (-0.3)", "Options expiry (-0.2)"],
                "risk_management": ["Poor R:R (-0.4)", "Good R:R (+0.3)"]
            }
        },
        "example_enhancement": {
            "cheatcode_base": "3.2/4 STRONG_BUY",
            "decision_adjustment": "-1.0 (overbought + near resistance)",
            "final_result": "2.2/4 BUY with wait for pullback advice"
        }
    }

@decision_router.get("/factors")
async def get_decision_factors():
    """Get all decision factors and their descriptions"""
    
    return {
        "decision_factors": {
            "technical": {
                "rsi_overbought": "RSI > 70 (-0.5), RSI > 80 (-1.0)",
                "rsi_oversold": "RSI < 30 (+0.5), RSI < 20 (+1.0)",
                "macd_signal": "MACD momentum confirmation (±0.3)",
                "momentum_divergence": "Technical momentum signals (±0.3)"
            },
            "price_position": {
                "near_resistance": "Within 2% of resistance (-0.5)",
                "near_support": "Within 2% of support (+0.5)",
                "ma_distance": "Distance from key moving averages (±0.3)"
            },
            "volume": {
                "volume_spike": "2x+ average volume (+0.8)",
                "volume_decline": "Low volume confirmation (-0.3 to -0.5)",
                "volume_trend": "5-day volume trend (±0.5)"
            },
            "market_context": {
                "market_trend": "SPY trend direction (±0.3)",
                "sector_trend": "Sector performance (±0.5)",
                "vix_level": "Market fear gauge (-0.2 to -0.4)",
                "correlation": "Market correlation factors"
            },
            "time_based": {
                "earnings_risk": "Within 7 days of earnings (-0.3)",
                "options_expiry": "Options expiration week (-0.2)",
                "market_session": "Extended hours trading (-0.1)"
            },
            "risk_management": {
                "risk_reward": "Risk/reward ratio analysis (±0.4)",
                "position_sizing": "Dynamic position adjustment (0.5x-1.5x)"
            }
        },
        "scoring": {
            "range": "-2.0 to +2.0 adjustment to CheatCode base score",
            "final_signals": [signal.value for signal in SignalStrength],
            "confidence": "Combined CheatCode + Decision Engine confidence"
        }
    }

# Export for main application
__all__ = ['DecisionEngine', 'DecisionContext', 'EnhancedTradingDecision', 'decision_router']
