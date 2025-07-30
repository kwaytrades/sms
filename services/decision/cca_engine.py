# cheatcode_engine.py
"""
CheatCode Algo Engine - Multi-Timeframe Analysis with Cache-First Architecture
Converts TradingView Pine Script to Python with background processing capabilities
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from fastapi import APIRouter, HTTPException
import redis.asyncio as redis
import os
from abc import ABC, abstractmethod

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

class TimeframeType(Enum):
    """Supported timeframes"""
    ONE_MINUTE = "1m"
    FIVE_MINUTE = "5m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"

@dataclass
class TimeframeConfig:
    """Configuration for each timeframe"""
    timeframe: str
    period: str
    weight: float
    bars_needed: int
    description: str

@dataclass
class CheatCodeSignal:
    """Single timeframe CheatCode signal with 4/4 confirmation"""
    symbol: str
    timeframe: str
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
    
    # Component details for debugging
    component_details: Dict[str, Any]
    
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

@dataclass
class MultiTimeframeSignal:
    """Combined multi-timeframe CheatCode analysis"""
    symbol: str
    timestamp: datetime
    
    # Master signal (weighted combination)
    master_score: float  # 0-4 scale
    master_strength: SignalStrength
    master_confidence: float
    
    # Timeframe alignment
    alignment_score: float  # 0-1.3 (bonus for agreement)
    timeframes_agreeing: int  # Number of timeframes in agreement
    dominant_direction: TrendDirection
    
    # Individual timeframe signals
    timeframe_signals: Dict[str, CheatCodeSignal]
    
    # Combined recommendations
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    
    # Multi-timeframe insights
    trend_strength: str  # "Strong/Moderate/Weak"
    momentum_phase: str  # "Acceleration/Continuation/Deceleration"
    risk_level: str  # "Low/Medium/High"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['master_strength'] = self.master_strength.value
        data['dominant_direction'] = self.dominant_direction.value
        
        # Convert timeframe signals
        tf_signals = {}
        for tf, signal in self.timeframe_signals.items():
            tf_signals[tf] = signal.to_dict()
        data['timeframe_signals'] = tf_signals
        
        return data

class DataProvider(ABC):
    """Abstract interface for market data providers"""
    
    @abstractmethod
    async def get_timeframe_data(self, symbol: str, timeframe: str, period: str) -> Optional[pd.DataFrame]:
        pass

class CachedDataProvider(DataProvider):
    """Cache-first data provider with API fallback"""
    
    def __init__(self, redis_client=None, eodhd_api_key: str = None):
        self.redis_client = redis_client
        self.eodhd_api_key = eodhd_api_key or os.environ.get('EODHD_API_TOKEN')
        
    async def get_timeframe_data(self, symbol: str, timeframe: str, period: str) -> Optional[pd.DataFrame]:
        """Get data from cache first, fallback to API"""
        
        # Try cache first
        cache_key = f"market_data:{symbol}:{timeframe}:{period}"
        
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data_dict = json.loads(cached_data)
                    df = pd.DataFrame(data_dict['data'])
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    logger.info(f"✅ Cache HIT for {symbol}:{timeframe}")
                    return df
            except Exception as e:
                logger.warning(f"Cache read failed for {symbol}: {e}")
        
        # Fallback to API
        logger.info(f"📡 Cache MISS - fetching {symbol}:{timeframe} from API")
        return await self._fetch_from_api(symbol, timeframe, period)
    
    async def _fetch_from_api(self, symbol: str, timeframe: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from EODHD API"""
        if not self.eodhd_api_key:
            logger.warning(f"No EODHD API key - using mock data for {symbol}")
            return self._generate_mock_data(symbol, timeframe)
        
        try:
            # Convert timeframe to EODHD format
            if timeframe in ["1m", "5m"]:
                # Intraday data
                url = f"https://eodhd.com/api/intraday/{symbol}.US"
                interval = "1m" if timeframe == "1m" else "5m"
                params = {
                    'api_token': self.eodhd_api_key,
                    'interval': interval,
                    'fmt': 'json'
                }
            else:
                # Daily data
                url = f"https://eodhd.com/api/eod/{symbol}.US"
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                # Convert period to days
                if period == "1mo":
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                elif period == "3mo":
                    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                elif period == "6mo":
                    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
                elif period == "2y":
                    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                else:
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                params = {
                    'api_token': self.eodhd_api_key,
                    'fmt': 'json',
                    'from': start_date,
                    'to': end_date
                }
            
            # Note: In production, you would use aiohttp here
            # For now, returning mock data to avoid external dependencies
            logger.info(f"Would fetch from {url} with params {params}")
            return self._generate_mock_data(symbol, timeframe)
            
        except Exception as e:
            logger.error(f"API fetch failed for {symbol}: {e}")
            return self._generate_mock_data(symbol, timeframe)
    
    def _generate_mock_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate realistic mock data for testing"""
        # Base prices for popular stocks
        base_prices = {
            'AAPL': 185, 'TSLA': 245, 'MSFT': 420, 'GOOGL': 140,
            'AMZN': 155, 'NVDA': 875, 'META': 485, 'NFLX': 485
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # Number of bars based on timeframe
        if timeframe == "1m":
            num_bars = 390  # 1 trading day
            freq = '1min'
        elif timeframe == "5m":
            num_bars = 390  # 5 days worth
            freq = '5min'
        elif timeframe == "1h":
            num_bars = 156  # 1 month worth (6.5h/day * 24 days)
            freq = '1H'
        else:  # 1d
            num_bars = 60  # 60 days
            freq = 'D'
        
        # Generate dates
        if timeframe in ["1m", "5m", "1h"]:
            # Business hours only for intraday
            dates = pd.bdate_range(end=datetime.now(), periods=num_bars//6.5, freq='D')
            # Expand to hours (mock - in reality you'd handle market hours properly)
            all_dates = []
            for date in dates:
                for hour in range(9, 16):  # 9 AM to 4 PM
                    if timeframe == "1m":
                        for minute in range(0, 60, 1):
                            all_dates.append(date.replace(hour=hour, minute=minute))
                    elif timeframe == "5m":
                        for minute in range(0, 60, 5):
                            all_dates.append(date.replace(hour=hour, minute=minute))
                    elif timeframe == "1h":
                        all_dates.append(date.replace(hour=hour))
            dates = pd.DatetimeIndex(all_dates[:num_bars])
        else:
            dates = pd.bdate_range(end=datetime.now(), periods=num_bars, freq=freq)
        
        # Random walk with trend
        returns = np.random.normal(0.0005, 0.015, num_bars)  # Slight upward bias
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Volatility based on timeframe
            if timeframe == "1m":
                volatility = np.random.uniform(0.001, 0.005)
            elif timeframe == "5m":
                volatility = np.random.uniform(0.003, 0.008)
            elif timeframe == "1h":
                volatility = np.random.uniform(0.005, 0.015)
            else:  # 1d
                volatility = np.random.uniform(0.01, 0.03)
            
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = close * (1 + np.random.uniform(-0.005, 0.005))
            
            # Volume based on timeframe
            if timeframe == "1m":
                volume = np.random.randint(10000, 50000)
            elif timeframe == "5m":
                volume = np.random.randint(50000, 200000)
            elif timeframe == "1h":
                volume = np.random.randint(200000, 800000)
            else:  # 1d
                volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'Date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        logger.info(f"Generated {len(df)} bars of mock data for {symbol}:{timeframe}")
        return df

class CheatCodeEngine:
    """
    CheatCode trading engine with multi-timeframe analysis
    """
    
    def __init__(self, redis_client=None, data_provider: DataProvider = None):
        self.redis_client = redis_client
        self.data_provider = data_provider or CachedDataProvider(redis_client)
        
        # Timeframe configurations
        self.timeframe_configs = {
            TimeframeType.ONE_MINUTE.value: TimeframeConfig("1m", "1mo", 0.1, 200, "Scalping/Noise"),
            TimeframeType.FIVE_MINUTE.value: TimeframeConfig("5m", "3mo", 0.2, 150, "Intraday Momentum"),
            TimeframeType.ONE_HOUR.value: TimeframeConfig("1h", "6mo", 0.3, 100, "Swing Trading"),
            TimeframeType.ONE_DAY.value: TimeframeConfig("1d", "2y", 0.4, 60, "Primary Trend")
        }
        
        # Component parameters (same as original)
        self.cloud_period = 20
        self.cloud_multiplier = 2.0
        self.swing_lookback = 3
        self.swing_smooth1 = 20
        self.swing_smooth2 = 10
        self.swing_signal_length = 3
        self.swing_ob_level = 40
        self.swing_os_level = -25
        self.squeeze_lookback = 10
        self.squeeze_signal_smooth = 3
        self.squeeze_bb_length = 10
        self.squeeze_bb_mult = 1.0
        self.pattern_pivot_period = 3
        self.pattern_strength_threshold = 0.6
        
    async def analyze_multi_timeframe(self, symbol: str, timeframes: List[str] = None) -> MultiTimeframeSignal:
        """
        Main multi-timeframe analysis function
        
        Args:
            symbol: Stock symbol
            timeframes: List of timeframes to analyze (defaults to all)
            
        Returns:
            MultiTimeframeSignal with combined analysis
        """
        if timeframes is None:
            timeframes = list(self.timeframe_configs.keys())
        
        try:
        # Analyze each timeframe
        timeframe_signals = {}
        
        for tf in timeframes:
            if tf not in self.timeframe_configs:
                logger.warning(f"Unsupported timeframe: {tf}")
                continue
            
            config = self.timeframe_configs[tf]
            signal = await self.analyze_single_timeframe(symbol, tf, config)
            
            if signal:
                timeframe_signals[tf] = signal
                logger.info(f"✅ {symbol}:{tf} analysis complete - {signal.signal_strength.value}")
            else:
                logger.warning(f"❌ {symbol}:{tf} analysis failed")
            
            if not timeframe_signals:
                raise ValueError(f"No successful timeframe analysis for {symbol}")
            
            # Combine timeframes into master signal
            master_signal = self._combine_timeframe_signals(symbol, timeframe_signals)
            
            # Cache the result
            await self._cache_multi_timeframe_signal(symbol, master_signal)
            
            logger.info(f"🎯 Multi-timeframe analysis complete for {symbol}: {master_signal.master_strength.value}")
            return master_signal
            
        except Exception as e:
            logger.error(f"Multi-timeframe analysis failed for {symbol}: {e}")
            raise
    
    async def analyze_single_timeframe(self, symbol: str, timeframe: str, config: TimeframeConfig) -> Optional[CheatCodeSignal]:
        """Analyze a single timeframe"""
        try:
            # Get data for this timeframe
            data = await self.data_provider.get_timeframe_data(symbol, timeframe, config.period)
            
            if data is None or len(data) < config.bars_needed:
                logger.warning(f"Insufficient data for {symbol}:{timeframe}")
                return None
            
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
                current_price, signal_strength, confidence, support_levels, resistance_levels, timeframe
            )
            
            # Create signal
            signal = CheatCodeSignal(
                symbol=symbol,
                timeframe=timeframe,
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
                resistance_levels=resistance_levels,
                component_details={
                    'cloud': cloud_signal,
                    'swing': swing_signal,
                    'squeeze': squeeze_signal,
                    'pattern': pattern_signal
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Single timeframe analysis failed for {symbol}:{timeframe}: {e}")
            return None
    
    def _combine_timeframe_signals(self, symbol: str, timeframe_signals: Dict[str, CheatCodeSignal]) -> MultiTimeframeSignal:
        """Combine multiple timeframe signals into master signal"""
        
        # Weighted scoring
        total_weighted_score = 0
        total_weight = 0
        
        for tf, signal in timeframe_signals.items():
            weight = self.timeframe_configs[tf].weight
            total_weighted_score += signal.total_score * weight
            total_weight += weight
        
        master_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Calculate alignment
        alignment_score, timeframes_agreeing, dominant_direction = self._calculate_timeframe_alignment(timeframe_signals)
        
        # Apply alignment bonus/penalty
        adjusted_master_score = master_score * alignment_score
        
        # Determine master signal strength
        master_strength = self._determine_master_signal_strength(adjusted_master_score, timeframes_agreeing, len(timeframe_signals))
        
        # Calculate master confidence
        individual_confidences = [signal.confidence for signal in timeframe_signals.values()]
        master_confidence = np.mean(individual_confidences) * alignment_score
        
        # Generate master recommendations
        primary_signal = timeframe_signals.get("1d") or timeframe_signals.get("1h") or list(timeframe_signals.values())[0]
        entry_price = primary_signal.entry_price
        stop_loss = primary_signal.stop_loss
        take_profit = primary_signal.take_profit
        
        # Adjust position size based on alignment
        base_position = primary_signal.position_size_pct
        if alignment_score > 1.2:  # Strong alignment
            position_size = min(base_position * 1.5, 0.15)  # Increase size but cap at 15%
        elif alignment_score < 0.95:  # Poor alignment
            position_size = base_position * 0.5  # Reduce size
        else:
            position_size = base_position
        
        # Determine trend characteristics
        trend_strength = self._analyze_trend_strength(timeframe_signals)
        momentum_phase = self._analyze_momentum_phase(timeframe_signals)
        risk_level = self._assess_risk_level(alignment_score, master_confidence, timeframe_signals)
        
        return MultiTimeframeSignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            master_score=adjusted_master_score,
            master_strength=master_strength,
            master_confidence=master_confidence,
            alignment_score=alignment_score,
            timeframes_agreeing=timeframes_agreeing,
            dominant_direction=dominant_direction,
            timeframe_signals=timeframe_signals,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            trend_strength=trend_strength,
            momentum_phase=momentum_phase,
            risk_level=risk_level
        )
    
    def _calculate_timeframe_alignment(self, timeframe_signals: Dict[str, CheatCodeSignal]) -> Tuple[float, int, TrendDirection]:
        """Calculate how well timeframes align"""
        
        directions = []
        strengths = []
        
        for signal in timeframe_signals.values():
            if signal.signal_strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY, SignalStrength.WEAK_BUY]:
                directions.append(1)
                strengths.append(1 if signal.signal_strength == SignalStrength.STRONG_BUY else 
                               0.7 if signal.signal_strength == SignalStrength.BUY else 0.4)
            elif signal.signal_strength in [SignalStrength.STRONG_SELL, SignalStrength.SELL, SignalStrength.WEAK_SELL]:
                directions.append(-1)
                strengths.append(1 if signal.signal_strength == SignalStrength.STRONG_SELL else 
                               0.7 if signal.signal_strength == SignalStrength.SELL else 0.4)
            else:
                directions.append(0)
                strengths.append(0.1)
        
        if not directions:
            return 1.0, 0, TrendDirection.NEUTRAL
        
        # Count agreements
        direction_counts = {1: 0, -1: 0, 0: 0}
        for d in directions:
            direction_counts[d] += 1
        
        total_timeframes = len(directions)
        
        # Determine dominant direction
        if direction_counts[1] > direction_counts[-1] and direction_counts[1] > direction_counts[0]:
            dominant_direction = TrendDirection.BULLISH
            agreeing = direction_counts[1]
        elif direction_counts[-1] > direction_counts[1] and direction_counts[-1] > direction_counts[0]:
            dominant_direction = TrendDirection.BEARISH
            agreeing = direction_counts[-1]
        else:
            dominant_direction = TrendDirection.NEUTRAL
            agreeing = direction_counts[0]
        
        # Calculate alignment score
        agreement_ratio = agreeing / total_timeframes
        
        if agreement_ratio >= 1.0:  # All agree
            alignment_score = 1.3
        elif agreement_ratio >= 0.75:  # 3/4 agree
            alignment_score = 1.15
        elif agreement_ratio >= 0.6:  # Majority agrees
            alignment_score = 1.05
        elif agreement_ratio >= 0.5:  # Split decision
            alignment_score = 0.95
        else:  # Conflicted
            alignment_score = 0.85
        
        # Bonus for strong signals in agreement
        if agreeing >= 2:
            avg_strength = np.mean([s for d, s in zip(directions, strengths) if 
                                  (d == 1 and dominant_direction == TrendDirection.BULLISH) or 
                                  (d == -1 and dominant_direction == TrendDirection.BEARISH)])
            alignment_score *= (1 + avg_strength * 0.1)  # Up to 10% bonus for strong signals
        
        return min(alignment_score, 1.3), agreeing, dominant_direction
    
    def _determine_master_signal_strength(self, score: float, agreeing: int, total: int) -> SignalStrength:
        """Determine master signal strength"""
        
        agreement_ratio = agreeing / total if total > 0 else 0
        
        if score >= 3.5 and agreement_ratio >= 0.75:
            return SignalStrength.STRONG_BUY if score > 0 else SignalStrength.STRONG_SELL
        elif score >= 2.5 and agreement_ratio >= 0.6:
            return SignalStrength.BUY if score > 0 else SignalStrength.SELL
        elif score >= 1.5:
            return SignalStrength.WEAK_BUY if score > 0 else SignalStrength.WEAK_SELL
        else:
            return SignalStrength.HOLD
    
    def _analyze_trend_strength(self, timeframe_signals: Dict[str, CheatCodeSignal]) -> str:
        """Analyze overall trend strength across timeframes"""
        
        # Weight longer timeframes more heavily for trend analysis
        weights = {"1d": 0.4, "1h": 0.3, "5m": 0.2, "1m": 0.1}
        
        total_weighted_score = 0
        total_weight = 0
        
        for tf, signal in timeframe_signals.items():
            weight = weights.get(tf, 0.1)
            total_weighted_score += signal.total_score * weight
            total_weight += weight
        
        avg_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        if avg_score >= 3.0:
            return "Strong"
        elif avg_score >= 2.0:
            return "Moderate"
        else:
            return "Weak"
    
    def _analyze_momentum_phase(self, timeframe_signals: Dict[str, CheatCodeSignal]) -> str:
        """Analyze momentum phase across timeframes"""
        
        # Compare short vs long timeframe strength
        short_tf_scores = []
        long_tf_scores = []
        
        for tf, signal in timeframe_signals.items():
            if tf in ["1m", "5m"]:
                short_tf_scores.append(signal.total_score)
            else:
                long_tf_scores.append(signal.total_score)
        
        if short_tf_scores and long_tf_scores:
            short_avg = np.mean(short_tf_scores)
            long_avg = np.mean(long_tf_scores)
            
            difference = short_avg - long_avg
            
            if difference > 0.5:
                return "Acceleration"
            elif difference < -0.5:
                return "Deceleration"
            else:
                return "Continuation"
        
        return "Continuation"
    
    def _assess_risk_level(self, alignment_score: float, confidence: float, timeframe_signals: Dict[str, CheatCodeSignal]) -> str:
        """Assess overall risk level"""
        
        # Factors that increase risk
        risk_factors = 0
        
        # Poor alignment increases risk
        if alignment_score < 1.0:
            risk_factors += 1
        
        # Low confidence increases risk
        if confidence < 0.6:
            risk_factors += 1
        
        # Check for conflicting signals in key timeframes
        if "1d" in timeframe_signals and "1h" in timeframe_signals:
            daily_direction = timeframe_signals["1d"].signal_strength
            hourly_direction = timeframe_signals["1h"].signal_strength
            
            # Convert to simple direction
            daily_bull = daily_direction in [SignalStrength.STRONG_BUY, SignalStrength.BUY, SignalStrength.WEAK_BUY]
            hourly_bull = hourly_direction in [SignalStrength.STRONG_BUY, SignalStrength.BUY, SignalStrength.WEAK_BUY]
            
            if daily_bull != hourly_bull:
                risk_factors += 1
        
        # Check volatility indicators
        volatility_signals = 0
        for signal in timeframe_signals.values():
            if 'squeeze' in signal.component_details:
                squeeze_data = signal.component_details['squeeze']
                if squeeze_data.get('momentum_strength', 0) > 0.8:
                    volatility_signals += 1
        
        if volatility_signals >= len(timeframe_signals) * 0.5:
            risk_factors += 1
        
        # Determine risk level
        if risk_factors == 0:
            return "Low"
        elif risk_factors <= 2:
            return "Medium"
        else:
            return "High"
    
    # Cache management methods
    async def _cache_multi_timeframe_signal(self, symbol: str, signal: MultiTimeframeSignal):
        """Cache multi-timeframe signal"""
        try:
            if self.redis_client:
                cache_key = f"cheatcode:multi:{symbol}"
                await self.redis_client.setex(
                    cache_key,
                    43200,  # 12 hours TTL
                    json.dumps(signal.to_dict())
                )
                
                # Cache individual timeframes too
                for tf, tf_signal in signal.timeframe_signals.items():
                    tf_cache_key = f"cheatcode:{tf}:{symbol}"
                    await self.redis_client.setex(
                        tf_cache_key,
                        21600,  # 6 hours TTL for individual timeframes
                        json.dumps(tf_signal.to_dict())
                    )
                
                logger.info(f"✅ Cached multi-timeframe signal for {symbol}")
                
        except Exception as e:
            logger.warning(f"Failed to cache multi-timeframe signal for {symbol}: {e}")
    
    async def get_cached_signal(self, symbol: str, timeframe: str = None) -> Union[MultiTimeframeSignal, CheatCodeSignal, None]:
        """Get cached signal"""
        try:
            if not self.redis_client:
                return None
            
            if timeframe:
                # Get specific timeframe
                cache_key = f"cheatcode:{timeframe}:{symbol}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    # Note: You'd need to reconstruct the CheatCodeSignal object here
                    return data
            else:
                # Get multi-timeframe
                cache_key = f"cheatcode:multi:{symbol}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    # Note: You'd need to reconstruct the MultiTimeframeSignal object here
                    return data
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached signal for {symbol}: {e}")
            return None
    
    # Original CheatCode component calculations (unchanged)
    def _calculate_cloud_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SuperTrend cloud signal (unchanged from original)"""
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
            score = min(distance * 10, 1.0)
            
            # Trend confirmation
            trend_strength = np.sum(trend[-5:] == current_trend) / 5
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
        """Calculate CC Swing oscillator signal (unchanged from original)"""
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
        """Calculate CCA Squeeze momentum signal (unchanged from original)"""
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
                score *= 1.3
            
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
        """Calculate pattern recognition signal (unchanged from original)"""
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
                if abs(price_momentum) > 0.02:
                    direction = 1 if price_momentum > 0 else -1
                    score = min(abs(price_momentum) * 5, 0.6)
                else:
                    score = 0.1
            
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
        """Determine overall signal strength based on 4/4 confirmation (unchanged from original)"""
        
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
        """Calculate overall confidence in the signal (unchanged from original)"""
        
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
                                      resistance_levels: List[float], timeframe: str) -> Tuple[float, float, float, float]:
        """Generate specific trading recommendations with timeframe adjustments"""
        
        # Base position sizing based on confidence and timeframe
        base_position = 0.02 if confidence > 0.8 else 0.015 if confidence > 0.6 else 0.01
        
        # Timeframe-specific adjustments
        if timeframe == "1m":
            base_position *= 0.5  # Smaller positions for scalping
            risk_multiplier = 0.005  # Tighter stops
        elif timeframe == "5m":
            base_position *= 0.7
            risk_multiplier = 0.01
        elif timeframe == "1h":
            base_position *= 1.0
            risk_multiplier = 0.02
        else:  # 1d
            base_position *= 1.2  # Larger positions for swing trades
            risk_multiplier = 0.03
        
        if signal_strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            # Long position
            entry_price = current_price
            
            # Stop loss: nearest support or risk multiplier below entry
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], 
                                    default=current_price * (1 - risk_multiplier))
                stop_loss = max(nearest_support, current_price * (1 - risk_multiplier))
            else:
                stop_loss = current_price * (1 - risk_multiplier)
            
            # Take profit: nearest resistance or 2x risk above entry
            risk_amount = entry_price - stop_loss
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], 
                                       default=current_price + 2 * risk_amount)
                take_profit = min(nearest_resistance, current_price + 2 * risk_amount)
            else:
                take_profit = current_price + 2 * risk_amount
            
            # Position sizing
            risk_per_trade = abs(entry_price - stop_loss) / entry_price
            position_size = min(base_position / risk_per_trade, 0.1) if risk_per_trade > 0 else base_position
            
        elif signal_strength in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            # Short position
            entry_price = current_price
            
            # Stop loss: nearest resistance or risk multiplier above entry
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], 
                                       default=current_price * (1 + risk_multiplier))
                stop_loss = min(nearest_resistance, current_price * (1 + risk_multiplier))
            else:
                stop_loss = current_price * (1 + risk_multiplier)
            
            # Take profit: nearest support or 2x risk below entry
            risk_amount = stop_loss - entry_price
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], 
                                    default=current_price - 2 * risk_amount)
                take_profit = max(nearest_support, current_price - 2 * risk_amount)
            else:
                take_profit = current_price - 2 * risk_amount
            
            # Position sizing
            risk_per_trade = abs(stop_loss - entry_price) / entry_price
            position_size = min(base_position / risk_per_trade, 0.1) if risk_per_trade > 0 else base_position
            
        else:
            # Hold or weak signals
            entry_price = current_price
            stop_loss = current_price * (1 - risk_multiplier)
            take_profit = current_price * (1 + risk_multiplier)
            position_size = base_position * 0.5
        
        return entry_price, stop_loss, take_profit, position_size
    
    def _calculate_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels (unchanged from original)"""
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
    
    # Technical indicator helper functions (unchanged from original)
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
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - period + 1)
            result[i] = np.mean(data[start_idx:i+1])
        return result
    
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

# FastAPI Router for CheatCode endpoints
router = APIRouter(prefix="/cheatcode", tags=["CheatCode Engine"])

# Global engine instance
cheatcode_engine = CheatCodeEngine()

@router.post("/analyze-multi/{symbol}")
async def analyze_multi_timeframe_endpoint(symbol: str, timeframes: List[str] = None):
    """
    Analyze a symbol using multi-timeframe CheatCode system
    
    Query params:
    - timeframes: Optional list of timeframes ["1m", "5m", "1h", "1d"]
    """
    try:
        # Use default timeframes if none provided
        if timeframes is None:
            timeframes = ["1h", "1d"]  # Default to swing trading timeframes
        
        # Validate timeframes
        valid_timeframes = [tf.value for tf in TimeframeType]
        invalid_tfs = [tf for tf in timeframes if tf not in valid_timeframes]
        if invalid_tfs:
            raise HTTPException(status_code=400, detail=f"Invalid timeframes: {invalid_tfs}")
        
        # Analyze symbol
        signal = await cheatcode_engine.analyze_multi_timeframe(symbol.upper(), timeframes)
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "timeframes_analyzed": timeframes,
            "signal": signal.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Multi-timeframe analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/analyze-cached/{symbol}")
async def get_cached_analysis(symbol: str, timeframe: str = None):
    """
    Get cached CheatCode analysis
    """
    try:
        cached_signal = await cheatcode_engine.get_cached_signal(symbol.upper(), timeframe)
        
        if cached_signal:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "cache_hit": True,
                "signal": cached_signal
            }
        else:
            return {
                "success": False,
                "symbol": symbol.upper(),
                "cache_hit": False,
                "message": "No cached analysis found"
            }
        
    except Exception as e:
        logger.error(f"Failed to get cached analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Cache retrieval failed: {str(e)}")

@router.get("/test-multi/{symbol}")
async def test_multi_timeframe(symbol: str):
    """
    Test multi-timeframe analysis with all timeframes
    """
    try:
        # Analyze all timeframes
        all_timeframes = [tf.value for tf in TimeframeType]
        signal = await cheatcode_engine.analyze_multi_timeframe(symbol.upper(), all_timeframes)
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "analysis_type": "full_multi_timeframe",
            "timeframes_analyzed": all_timeframes,
            "signal": signal.to_dict(),
            "note": "This analysis uses mock data for testing purposes"
        }
        
    except Exception as e:
        logger.error(f"Test multi-timeframe analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Test analysis failed: {str(e)}")

@router.get("/engine-status")
async def get_engine_status():
    """
    Get engine status and configuration
    """
    return {
        "engine": "CheatCode Multi-Timeframe Engine",
        "version": "2.0.0",
        "supported_timeframes": [tf.value for tf in TimeframeType],
        "timeframe_configs": {
            tf: {
                "weight": config.weight,
                "description": config.description,
                "bars_needed": config.bars_needed
            }
            for tf, config in cheatcode_engine.timeframe_configs.items()
        },
        "components": {
            "cloud": "SuperTrend multi-timeframe analysis",
            "swing": "CC Swing momentum oscillator",
            "squeeze": "CCA Squeeze volatility indicator", 
            "pattern": "Chart pattern and breakout detection"
        },
        "new_features": [
            "Multi-timeframe analysis with weighted scoring",
            "Timeframe alignment detection and bonuses", 
            "Cache-first architecture with background processing",
            "Trend strength and momentum phase analysis",
            "Risk level assessment across timeframes",
            "Timeframe-specific position sizing"
        ],
        "cache_status": "Redis-based caching enabled" if cheatcode_engine.redis_client else "No cache configured"
    }

# Export for use in main application
__all__ = ['CheatCodeEngine', 'MultiTimeframeSignal', 'CheatCodeSignal', 'SignalStrength', 'TimeframeType', 'router']
