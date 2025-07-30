# cheatcode_engine.py
"""
CheatCode Algo Engine - 4/4 Confirmation Trading Signal System
Converts TradingView Pine Script to Python for production trading signals
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
from fastapi import APIRouter, HTTPException
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength classification"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class TrendDirection(Enum):
    """Trend direction"""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

@dataclass
class CheatCodeSignal:
    """Complete CheatCode signal with 4/4 confirmation"""
    symbol: str
    timestamp: datetime
    
    # Individual component scores (0-1)
    cloud_score: float
    swing_score: float
    squeeze_score: float
    pattern_score: float
    
    # Component directions
    cloud_trend: TrendDirection
    swing_trend: TrendDirection
    squeeze_trend: TrendDirection
    pattern_trend: TrendDirection
    
    # Overall assessment
    total_score: float  # 0-4 scale
    signal_strength: SignalStrength
    confidence: float  # 0-1
    
    # Trading recommendations
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    
    # Technical levels
    support_levels: List[float]
    resistance_levels: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['signal_strength'] = self.signal_strength.value
        data['cloud_trend'] = self.cloud_trend.value
        data['swing_trend'] = self.swing_trend.value
        data['squeeze_trend'] = self.squeeze_trend.value
        data['pattern_trend'] = self.pattern_trend.value
        return data

class CheatCodeEngine:
    """
    Main CheatCode trading engine implementing 4/4 confirmation system
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
        # Cloud (SuperTrend) parameters
        self.cloud_period = 20
        self.cloud_multiplier = 2.0
        
        # Swing parameters
        self.swing_lookback = 3
        self.swing_smooth1 = 20
        self.swing_smooth2 = 10
        self.swing_signal_length = 3
        self.swing_ob_level = 40
        self.swing_os_level = -25
        
        # Squeeze parameters
        self.squeeze_lookback = 10
        self.squeeze_signal_smooth = 3
        self.squeeze_bb_length = 10
        self.squeeze_bb_mult = 1.0
        
        # Pattern recognition parameters
        self.pattern_pivot_period = 3
        self.pattern_strength_threshold = 0.6
        
    async def analyze_symbol(self, symbol: str, data: pd.DataFrame) -> CheatCodeSignal:
        """
        Main analysis function - generates complete CheatCode signal
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            CheatCodeSignal with 4/4 confirmation analysis
        """
        try:
            # Ensure we have enough data
            if len(data) < 50:
                raise ValueError(f"Insufficient data for {symbol}: need at least 50 bars")
            
            # Calculate each component
            cloud_signal = self._calculate_cloud_signal(data)
            swing_signal = self._calculate_swing_signal(data)
            squeeze_signal = self._calculate_squeeze_signal(data)
            pattern_signal = self._calculate_pattern_signal(data)
            
            # Calculate support/resistance levels
            support_levels, resistance_levels = self._calculate_sr_levels(data)
            
            # Generate overall signal
            total_score = cloud_signal['score'] + swing_signal['score'] + squeeze_signal['score'] + pattern_signal['score']
            signal_strength = self._determine_signal_strength(total_score, cloud_signal, swing_signal, squeeze_signal, pattern_signal)
            confidence = self._calculate_confidence(cloud_signal, swing_signal, squeeze_signal, pattern_signal)
            
            # Generate trading recommendations
            current_price = float(data['close'].iloc[-1])
            entry_price, stop_loss, take_profit, position_size = self._generate_trade_recommendations(
                current_price, signal_strength, confidence, support_levels, resistance_levels
            )
            
            # Create complete signal
            signal = CheatCodeSignal(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                cloud_score=cloud_signal['score'],
                swing_score=swing_signal['score'],
                squeeze_score=squeeze_signal['score'],
                pattern_score=pattern_signal['score'],
                cloud_trend=TrendDirection(cloud_signal['direction']),
                swing_trend=TrendDirection(swing_signal['direction']),
                squeeze_trend=TrendDirection(squeeze_signal['direction']),
                pattern_trend=TrendDirection(pattern_signal['direction']),
                total_score=total_score,
                signal_strength=signal_strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_pct=position_size,
                support_levels=support_levels,
                resistance_levels=resistance_levels
            )
            
            # Cache the signal
            if self.redis_client:
                await self._cache_signal(symbol, signal)
            
            logger.info(f"Generated CheatCode signal for {symbol}: {signal_strength.value} ({total_score:.1f}/4)")
            return signal
            
        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {e}")
            raise
    
    def _calculate_cloud_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SuperTrend cloud signal"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate ATR
            tr = np.maximum(high - low, 
                           np.maximum(np.abs(high - np.roll(close, 1)),
                                    np.abs(low - np.roll(close, 1))))
            atr = self._ema(tr, self.cloud_period)
            
            # Calculate SuperTrend
            hl2 = (high + low) / 2
            upper_band = hl2 + (self.cloud_multiplier * atr)
            lower_band = hl2 - (self.cloud_multiplier * atr)
            
            # SuperTrend logic
            supertrend = np.zeros_like(close)
            trend = np.ones_like(close)
            
            for i in range(1, len(close)):
                # Basic SuperTrend calculation
                if close[i] <= lower_band[i-1]:
                    trend[i] = -1
                elif close[i] >= upper_band[i-1]:
                    trend[i] = 1
                else:
                    trend[i] = trend[i-1]
                
                if trend[i] == 1:
                    supertrend[i] = lower_band[i]
                else:
                    supertrend[i] = upper_band[i]
            
            # Current signal
            current_trend = trend[-1]
            current_price = close[-1]
            current_supertrend = supertrend[-1]
            
            # Calculate score based on price distance from SuperTrend
            distance = abs(current_price - current_supertrend) / current_supertrend
            score = min(distance * 10, 1.0)  # Normalize to 0-1
            
            # Trend confirmation
            trend_strength = np.sum(trend[-5:] == current_trend) / 5  # Last 5 bars consistency
            score *= trend_strength
            
            return {
                'score': score,
                'direction': int(current_trend),
                'supertrend_level': current_supertrend,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            logger.error(f"Cloud signal calculation failed: {e}")
            return {'score': 0, 'direction': 0, 'supertrend_level': 0, 'trend_strength': 0}
    
    def _calculate_swing_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate CC Swing oscillator signal"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate highest/lowest over lookback period
            hih = self._rolling_max(high, self.swing_lookback)
            lil = self._rolling_min(low, self.swing_lookback)
            
            # Swing calculation
            numb = self._rma(self._rma(close - 0.5 * (hih + lil), self.swing_smooth1), self.swing_smooth2)
            denb = 0.5 * self._rma(self._rma(hih - lil, self.swing_smooth1), self.swing_smooth2)
            
            # Avoid division by zero
            denb = np.where(denb == 0, 1e-8, denb)
            swing = 100 * numb / denb
            signal = self._rma(swing, self.swing_signal_length)
            
            # Current values
            current_swing = swing[-1]
            current_signal = signal[-1]
            
            # Determine trend and score
            if current_swing > current_signal:
                direction = 1
                if current_swing > self.swing_ob_level:
                    score = min((current_swing - self.swing_ob_level) / (75 - self.swing_ob_level), 1.0)
                else:
                    score = max(current_swing / self.swing_ob_level, 0) * 0.7
            else:
                direction = -1
                if current_swing < self.swing_os_level:
                    score = min((self.swing_os_level - current_swing) / (75 - abs(self.swing_os_level)), 1.0)
                else:
                    score = max(abs(current_swing) / abs(self.swing_os_level), 0) * 0.7
            
            # Crossover confirmation
            crossover_strength = 1.0 if (current_swing > current_signal and swing[-2] <= signal[-2]) else \
                               1.0 if (current_swing < current_signal and swing[-2] >= signal[-2]) else 0.5
            
            score *= crossover_strength
            
            return {
                'score': score,
                'direction': direction,
                'swing_value': current_swing,
                'signal_value': current_signal,
                'crossover_strength': crossover_strength
            }
            
        except Exception as e:
            logger.error(f"Swing signal calculation failed: {e}")
            return {'score': 0, 'direction': 0, 'swing_value': 0, 'signal_value': 0, 'crossover_strength': 0}
    
    def _calculate_squeeze_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate CCA Squeeze momentum signal"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Source calculation
            source = (high + low + close * 2) / 4
            
            # Moving average and standard deviation
            ma = self._ema(source, self.squeeze_lookback)
            std = self._rolling_std(source, self.squeeze_lookback)
            
            # Squeeze oscillator
            squeeze_raw = (source - ma) * 100 / np.where(std == 0, 1, std)
            squeeze = self._ema(squeeze_raw, self.squeeze_signal_smooth)
            
            # Double smooth if needed
            squeeze_smooth = self._ema(squeeze, self.squeeze_signal_smooth)
            
            # Transform to 0-100 scale
            oscillator = (self._ema(squeeze_smooth, self.squeeze_lookback) + 100) / 2 - 4
            
            # Current values
            current_osc = oscillator[-1]
            prev_osc = oscillator[-2] if len(oscillator) > 1 else current_osc
            
            # Determine trend and score
            if current_osc > 50:
                direction = 1
                momentum_strength = (current_osc - 50) / 50
                trend_strength = 1 if current_osc > prev_osc else 0.7
            else:
                direction = -1
                momentum_strength = (50 - current_osc) / 50
                trend_strength = 1 if current_osc < prev_osc else 0.7
            
            score = momentum_strength * trend_strength
            
            # Bollinger Band squeeze detection
            bb_basis = self._sma(oscillator, self.squeeze_bb_length)
            bb_dev = self.squeeze_bb_mult * self._rolling_std(oscillator, self.squeeze_bb_length)
            bb_upper = bb_basis + bb_dev
            bb_lower = bb_basis - bb_dev
            
            # Squeeze breakout bonus
            if current_osc > bb_upper[-1] or current_osc < bb_lower[-1]:
                score *= 1.3  # Breakout bonus
            
            return {
                'score': min(score, 1.0),
                'direction': direction,
                'oscillator_value': current_osc,
                'momentum_strength': momentum_strength,
                'bb_upper': bb_upper[-1],
                'bb_lower': bb_lower[-1]
            }
            
        except Exception as e:
            logger.error(f"Squeeze signal calculation failed: {e}")
            return {'score': 0, 'direction': 0, 'oscillator_value': 50, 'momentum_strength': 0, 'bb_upper': 0, 'bb_lower': 0}
    
    def _calculate_pattern_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate pattern recognition signal"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values if 'volume' in data.columns else np.ones_like(close)
            
            # Pivot point detection
            pivot_highs = self._find_pivot_highs(high, self.pattern_pivot_period)
            pivot_lows = self._find_pivot_lows(low, self.pattern_pivot_period)
            
            # Support/Resistance breaks
            recent_resistance = np.max(high[-20:]) if len(high) >= 20 else high[-1]
            recent_support = np.min(low[-20:]) if len(low) >= 20 else low[-1]
            
            current_price = close[-1]
            prev_price = close[-2] if len(close) > 1 else current_price
            
            # Volume confirmation
            avg_volume = np.mean(volume[-10:]) if len(volume) >= 10 else volume[-1]
            current_volume = volume[-1]
            volume_strength = min(current_volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
            
            # Pattern detection
            if current_price > recent_resistance and prev_price <= recent_resistance:
                # Bullish breakout
                direction = 1
                breakout_strength = (current_price - recent_resistance) / recent_resistance
                score = min(breakout_strength * 10, 1.0) * volume_strength
            elif current_price < recent_support and prev_price >= recent_support:
                # Bearish breakdown
                direction = -1
                breakdown_strength = (recent_support - current_price) / recent_support
                score = min(breakdown_strength * 10, 1.0) * volume_strength
            else:
                # No clear pattern
                direction = 0
                # Check for trend continuation
                price_momentum = (current_price - close[-5]) / close[-5] if len(close) >= 5 else 0
                if abs(price_momentum) > 0.02:  # 2% move
                    direction = 1 if price_momentum > 0 else -1
                    score = min(abs(price_momentum) * 5, 0.6)  # Lower score for continuation
                else:
                    score = 0.1  # Minimal score for sideways movement
            
            return {
                'score': min(score, 1.0),
                'direction': direction,
                'recent_resistance': recent_resistance,
                'recent_support': recent_support,
                'volume_strength': volume_strength,
                'pivot_highs': pivot_highs[-5:].tolist() if len(pivot_highs) > 0 else [],
                'pivot_lows': pivot_lows[-5:].tolist() if len(pivot_lows) > 0 else []
            }
            
        except Exception as e:
            logger.error(f"Pattern signal calculation failed: {e}")
            return {'score': 0, 'direction': 0, 'recent_resistance': 0, 'recent_support': 0, 'volume_strength': 1, 'pivot_highs': [], 'pivot_lows': []}
    
    def _determine_signal_strength(self, total_score: float, cloud: Dict, swing: Dict, squeeze: Dict, pattern: Dict) -> SignalStrength:
        """Determine overall signal strength based on 4/4 confirmation"""
        
        # Count confirmations
        bullish_confirmations = sum([
            cloud['direction'] == 1,
            swing['direction'] == 1,
            squeeze['direction'] == 1,
            pattern['direction'] == 1
        ])
        
        bearish_confirmations = sum([
            cloud['direction'] == -1,
            swing['direction'] == -1,
            squeeze['direction'] == -1,
            pattern['direction'] == -1
        ])
        
        # Determine signal strength
        if bullish_confirmations >= 3 and total_score >= 3.0:
            return SignalStrength.STRONG_BUY
        elif bullish_confirmations >= 3 or total_score >= 2.5:
            return SignalStrength.BUY
        elif bullish_confirmations >= 2 or total_score >= 1.5:
            return SignalStrength.WEAK_BUY
        elif bearish_confirmations >= 3 and total_score >= 3.0:
            return SignalStrength.STRONG_SELL
        elif bearish_confirmations >= 3 or total_score >= 2.5:
            return SignalStrength.SELL
        elif bearish_confirmations >= 2 or total_score >= 1.5:
            return SignalStrength.WEAK_SELL
        else:
            return SignalStrength.HOLD
    
    def _calculate_confidence(self, cloud: Dict, swing: Dict, squeeze: Dict, pattern: Dict) -> float:
        """Calculate overall confidence in the signal"""
        
        # Individual component strengths
        component_strengths = [cloud['score'], swing['score'], squeeze['score'], pattern['score']]
        
        # Direction alignment
        directions = [cloud['direction'], swing['direction'], squeeze['direction'], pattern['direction']]
        alignment = len(set(d for d in directions if d != 0))
        alignment_score = 1.0 if alignment == 1 else 0.7 if alignment == 2 else 0.4
        
        # Average component strength
        avg_strength = np.mean(component_strengths)
        
        # Confidence calculation
        confidence = avg_strength * alignment_score
        
        # Bonus for 4/4 confirmation
        if all(d != 0 and d == directions[0] for d in directions):
            confidence *= 1.2
        
        return min(confidence, 1.0)
    
    def _generate_trade_recommendations(self, current_price: float, signal_strength: SignalStrength, 
                                      confidence: float, support_levels: List[float], 
                                      resistance_levels: List[float]) -> Tuple[float, float, float, float]:
        """Generate specific trading recommendations"""
        
        # Base position sizing based on confidence
        base_position = 0.02 if confidence > 0.8 else 0.015 if confidence > 0.6 else 0.01
        
        if signal_strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            # Long position
            entry_price = current_price
            
            # Stop loss: nearest support or 3% below entry
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.97)
                stop_loss = max(nearest_support, current_price * 0.97)
            else:
                stop_loss = current_price * 0.97
            
            # Take profit: nearest resistance or 6% above entry
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.06)
                take_profit = min(nearest_resistance, current_price * 1.06)
            else:
                take_profit = current_price * 1.06
            
            # Position sizing
            risk_per_trade = abs(entry_price - stop_loss) / entry_price
            position_size = min(base_position / risk_per_trade, 0.1) if risk_per_trade > 0 else base_position
            
        elif signal_strength in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            # Short position
            entry_price = current_price
            
            # Stop loss: nearest resistance or 3% above entry
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.03)
                stop_loss = min(nearest_resistance, current_price * 1.03)
            else:
                stop_loss = current_price * 1.03
            
            # Take profit: nearest support or 6% below entry
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.94)
                take_profit = max(nearest_support, current_price * 0.94)
            else:
                take_profit = current_price * 0.94
            
            # Position sizing
            risk_per_trade = abs(stop_loss - entry_price) / entry_price
            position_size = min(base_position / risk_per_trade, 0.1) if risk_per_trade > 0 else base_position
            
        else:
            # Hold or weak signals
            entry_price = current_price
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.02
            position_size = base_position * 0.5
        
        return entry_price, stop_loss, take_profit, position_size
    
    def _calculate_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        high = data['high'].values
        low = data['low'].values
        
        # Find pivot points
        pivot_highs = self._find_pivot_highs(high, 5)
        pivot_lows = self._find_pivot_lows(low, 5)
        
        # Get recent levels
        resistance_levels = pivot_highs[-10:].tolist() if len(pivot_highs) > 0 else []
        support_levels = pivot_lows[-10:].tolist() if len(pivot_lows) > 0 else []
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        support_levels = sorted(list(set(support_levels)))
        
        return support_levels, resistance_levels
    
    # Technical indicator helper functions
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return np.convolve(data, np.ones(period), 'valid') / period
    
    def _rma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Rolling Moving Average (Wilder's smoothing)"""
        alpha = 1 / period
        rma = np.zeros_like(data)
        rma[0] = data[0]
        
        for i in range(1, len(data)):
            rma[i] = alpha * data[i] + (1 - alpha) * rma[i-1]
        
        return rma
    
    def _rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling maximum"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            result[i] = np.max(data[start_idx:i+1])
        return result
    
    def _rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling minimum"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            result[i] = np.min(data[start_idx:i+1])
        return result
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            result[i] = np.std(data[start_idx:i+1])
        return result
    
    def _find_pivot_highs(self, data: np.ndarray, period: int) -> np.ndarray:
        """Find pivot high points"""
        pivots = []
        for i in range(period, len(data) - period):
            if all(data[i] > data[i-j] for j in range(1, period+1)) and \
               all(data[i] > data[i+j] for j in range(1, period+1)):
                pivots.append(data[i])
        return np.array(pivots)
    
    def _find_pivot_lows(self, data: np.ndarray, period: int) -> np.ndarray:
        """Find pivot low points"""
        pivots = []
        for i in range(period, len(data) - period):
            if all(data[i] < data[i-j] for j in range(1, period+1)) and \
               all(data[i] < data[i+j] for j in range(1, period+1)):
                pivots.append(data[i])
        return np.array(pivots)
    
    async def _cache_signal(self, symbol: str, signal: CheatCodeSignal):
        """Cache signal in Redis"""
        try:
            if self.redis_client:
                cache_key = f"cheatcode_signal:{symbol}"
                await self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour TTL
                    json.dumps(signal.to_dict())
                )
        except Exception as e:
            logger.warning(f"Failed to cache signal for {symbol}: {e}")

# FastAPI Router for testing endpoints
router = APIRouter(prefix="/cheatcode", tags=["CheatCode Engine"])

# Global engine instance
cheatcode_engine = CheatCodeEngine()

@router.post("/analyze/{symbol}")
async def analyze_symbol_endpoint(symbol: str, market_data: Dict[str, Any]):
    """
    Analyze a symbol using CheatCode 4/4 confirmation system
    
    Body should contain:
    {
        "ohlcv": [
            {"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000000},
            ...
        ]
    }
    """
    try:
        # Convert market data to DataFrame
        if "ohlcv" not in market_data:
            raise HTTPException(status_code=400, detail="Missing 'ohlcv' data in request body")
        
        df = pd.DataFrame(market_data["ohlcv"])
        required_columns = ['open', 'high', 'low', 'close']
        
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"Missing required columns: {required_columns}")
        
        # Analyze symbol
        signal = await cheatcode_engine.analyze_symbol(symbol.upper(), df)
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "signal": signal.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/test/{symbol}")
async def test_with_sample_data(symbol: str):
    """
    Test endpoint with sample market data
    """
    try:
        # Generate sample OHLCV data for testing
        np.random.seed(42)  # For reproducible results
        
        # Create 100 bars of sample data
        num_bars = 100
        base_price = 100
        
        sample_data = []
        current_price = base_price
        
        for i in range(num_bars):
            # Random walk with slight upward bias
            price_change = np.random.normal(0.001, 0.02)  # 0.1% avg gain, 2% volatility
            current_price *= (1 + price_change)
            
            # Generate OHLC
            open_price = current_price
            high = open_price * (1 + abs(np.random.normal(0, 0.015)))
            low = open_price * (1 - abs(np.random.normal(0, 0.015)))
            close = low + (high - low) * np.random.random()
            volume = np.random.randint(500000, 2000000)
            
            sample_data.append({
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume
            })
            
            current_price = close
        
        # Analyze with sample data
        df = pd.DataFrame(sample_data)
        signal = await cheatcode_engine.analyze_symbol(symbol.upper(), df)
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "signal": signal.to_dict(),
            "note": "This analysis uses randomly generated sample data for testing purposes"
        }
        
    except Exception as e:
        logger.error(f"Test analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Test analysis failed: {str(e)}")

@router.get("/signals/summary")
async def get_signals_summary():
    """
    Get summary of CheatCode engine capabilities and signal types
    """
    return {
        "engine": "CheatCode 4/4 Confirmation System",
        "components": {
            "cloud": "SuperTrend multi-timeframe analysis",
            "swing": "CC Swing momentum oscillator",
            "squeeze": "CCA Squeeze volatility indicator", 
            "pattern": "Chart pattern and breakout detection"
        },
        "signal_strengths": [e.value for e in SignalStrength],
        "scoring": {
            "total_score": "0-4 scale (sum of individual component scores)",
            "confidence": "0-1 scale based on component alignment",
            "components": "Each component scored 0-1 based on strength"
        },
        "trading_recommendations": {
            "entry_price": "Recommended entry level",
            "stop_loss": "Risk management level",
            "take_profit": "Profit target level",
            "position_size_pct": "Recommended position size as % of portfolio"
        }
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "CheatCode 4/4 Confirmation System",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Export for use in main application
__all__ = ['CheatCodeEngine', 'CheatCodeSignal', 'SignalStrength', 'router']
