# technical_analysis_engine.py
"""
Technical Analysis Engine - Multi-Timeframe Analysis with Cache-First Architecture
Professional-grade technical analysis with market hours awareness and background processing
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from fastapi import APIRouter, HTTPException
import redis.asyncio as redis
import os
import pytz
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalStrength(Enum):
    """Technical signal strength classification"""
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    SLIGHTLY_BULLISH = "SLIGHTLY_BULLISH"
    NEUTRAL = "NEUTRAL"
    SLIGHTLY_BEARISH = "SLIGHTLY_BEARISH"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"

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

class MarketSession(Enum):
    """Market session types"""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"

@dataclass
class TimeframeConfig:
    """Configuration for each timeframe"""
    timeframe: str
    period: str
    weight: float
    bars_needed: int
    description: str
    update_frequency: str  # How often to update during market hours

@dataclass
class TechnicalIndicators:
    """Technical indicators for a single timeframe"""
    # Trend indicators
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    
    # Momentum indicators
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    
    # Volatility indicators
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_percent: float
    atr: float
    
    # Volume indicators
    volume_sma: float
    volume_ratio: float
    
    # Support/Resistance
    support_levels: List[float]
    resistance_levels: List[float]

@dataclass
class TechnicalSignal:
    """Single timeframe technical analysis signal"""
    symbol: str
    timeframe: str
    timestamp: datetime
    current_price: float
    
    # Technical indicators
    indicators: TechnicalIndicators
    
    # Signal scores (0-1 each)
    trend_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float
    
    # Overall assessment
    total_score: float  # 0-4 scale
    technical_strength: TechnicalStrength
    trend_direction: TrendDirection
    confidence: float  # 0-1
    
    # Key insights
    trend_analysis: str
    momentum_analysis: str
    volume_analysis: str
    key_levels: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['technical_strength'] = self.technical_strength.value
        data['trend_direction'] = self.trend_direction.value
        return data

@dataclass
class MultiTimeframeTechnicalSignal:
    """Combined multi-timeframe technical analysis"""
    symbol: str
    timestamp: datetime
    current_price: float
    
    # Master signal (weighted combination)
    master_score: float  # 0-4 scale
    master_strength: TechnicalStrength
    master_confidence: float
    
    # Timeframe alignment
    alignment_score: float  # 0-1.3 (bonus for agreement)
    timeframes_agreeing: int
    dominant_trend: TrendDirection
    
    # Individual timeframe signals
    timeframe_signals: Dict[str, TechnicalSignal]
    
    # Multi-timeframe insights
    trend_strength: str  # "Strong/Moderate/Weak"
    momentum_phase: str  # "Acceleration/Continuation/Deceleration"
    volatility_regime: str  # "Low/Normal/High"
    volume_profile: str  # "Strong/Normal/Weak"
    
    # Key technical levels across timeframes
    major_support: float
    major_resistance: float
    next_support: float
    next_resistance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['master_strength'] = self.master_strength.value
        data['dominant_trend'] = self.dominant_trend.value
        
        # Convert timeframe signals
        tf_signals = {}
        for tf, signal in self.timeframe_signals.items():
            tf_signals[tf] = signal.to_dict()
        data['timeframe_signals'] = tf_signals
        
        return data

class MarketHours:
    """Market hours utility for US markets"""
    
    def __init__(self):
        self.eastern_tz = pytz.timezone('US/Eastern')
        
    def get_current_market_session(self) -> MarketSession:
        """Get current market session"""
        now_et = datetime.now(self.eastern_tz)
        current_time = now_et.time()
        
        # Check if it's a weekday
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return MarketSession.CLOSED
        
        # Market hours (Eastern Time)
        pre_market_start = time(4, 0)  # 4:00 AM
        regular_start = time(9, 30)    # 9:30 AM
        regular_end = time(16, 0)      # 4:00 PM
        after_hours_end = time(20, 0)  # 8:00 PM
        
        if pre_market_start <= current_time < regular_start:
            return MarketSession.PRE_MARKET
        elif regular_start <= current_time < regular_end:
            return MarketSession.REGULAR
        elif regular_end <= current_time < after_hours_end:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED
    
    def is_market_hours(self, include_extended: bool = True) -> bool:
        """Check if market is open"""
        session = self.get_current_market_session()
        
        if include_extended:
            return session in [MarketSession.PRE_MARKET, MarketSession.REGULAR, MarketSession.AFTER_HOURS]
        else:
            return session == MarketSession.REGULAR
    
    def should_update_timeframe(self, timeframe: str) -> bool:
        """Determine if timeframe should be updated based on market hours"""
        session = self.get_current_market_session()
        
        # Only update intraday timeframes during market hours
        if timeframe in ["1m", "5m"] and session != MarketSession.REGULAR:
            return False
        
        # Update hourly during extended hours
        if timeframe == "1h" and session == MarketSession.CLOSED:
            return False
        
        # Daily can be updated anytime (for international markets, after-hours, etc.)
        return True

class CachedTechnicalDataProvider:
    """Cache-first data provider with market hours awareness"""
    
    def __init__(self, redis_client=None, eodhd_api_key: str = None):
        self.redis_client = redis_client
        self.eodhd_api_key = eodhd_api_key or os.environ.get('EODHD_API_TOKEN')
        self.market_hours = MarketHours()
        
    async def get_timeframe_data(self, symbol: str, timeframe: str, period: str) -> Optional[pd.DataFrame]:
        """Get data from cache first, fallback to API only during market hours"""
        
        # Check if we should fetch data for this timeframe
        if not self.market_hours.should_update_timeframe(timeframe):
            logger.info(f"Market closed for {timeframe} updates - using cache only for {symbol}")
        
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
        
        # Only fetch from API during appropriate market hours
        if self.market_hours.should_update_timeframe(timeframe):
            logger.info(f"📡 Cache MISS - fetching {symbol}:{timeframe} from API (market hours: {self.market_hours.get_current_market_session().value})")
            return await self._fetch_from_api(symbol, timeframe, period)
        else:
            logger.info(f"⏰ Market closed for {timeframe} - returning mock data for {symbol}")
            return self._generate_mock_data(symbol, timeframe)
    
    async def _fetch_from_api(self, symbol: str, timeframe: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from EODHD API with market hours check"""
        if not self.eodhd_api_key:
            logger.warning(f"No EODHD API key - using mock data for {symbol}")
            return self._generate_mock_data(symbol, timeframe)
        
        # Additional market hours check for intraday data
        session = self.market_hours.get_current_market_session()
        if timeframe in ["1m", "5m"] and session not in [MarketSession.REGULAR, MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS]:
            logger.info(f"Skipping {timeframe} API call - market closed")
            return self._generate_mock_data(symbol, timeframe)
        
        try:
            # Convert timeframe to EODHD format
            if timeframe in ["1m", "5m"]:
                # Intraday data - only during market hours
                url = f"https://eodhd.com/api/intraday/{symbol}.US"
                interval = "1m" if timeframe == "1m" else "5m"
                params = {
                    'api_token': self.eodhd_api_key,
                    'interval': interval,
                    'fmt': 'json'
                }
            else:
                # Daily data - can be fetched anytime
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
            logger.info(f"Would fetch from {url} with params {params} (session: {session.value})")
            return self._generate_mock_data(symbol, timeframe)
            
        except Exception as e:
            logger.error(f"API fetch failed for {symbol}: {e}")
            return self._generate_mock_data(symbol, timeframe)
    
    def _generate_mock_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate realistic mock data with market hours consideration"""
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
        
        # Generate dates with market hours consideration
        if timeframe in ["1m", "5m", "1h"]:
            # Business hours only for intraday
            dates = pd.bdate_range(end=datetime.now(), periods=max(1, num_bars//40), freq='D')
            all_dates = []
            
            for date in dates:
                # Only generate data for market hours (9:30 AM - 4:00 PM ET)
                market_start = 9.5  # 9:30 AM
                market_end = 16.0   # 4:00 PM
                
                if timeframe == "1m":
                    # Generate minute-by-minute data during market hours
                    minutes_in_session = int((market_end - market_start) * 60)
                    for minute in range(minutes_in_session):
                        hour = int(market_start + minute / 60)
                        minute_part = int((market_start + minute / 60 - hour) * 60)
                        all_dates.append(date.replace(hour=hour, minute=minute_part))
                elif timeframe == "5m":
                    # Generate 5-minute bars during market hours
                    for minutes in range(0, int((market_end - market_start) * 60), 5):
                        hour = int(market_start + minutes / 60)
                        minute_part = int((market_start + minutes / 60 - hour) * 60)
                        all_dates.append(date.replace(hour=hour, minute=minute_part))
                elif timeframe == "1h":
                    # Generate hourly bars during market hours
                    for hour in range(int(market_start), int(market_end)):
                        all_dates.append(date.replace(hour=hour, minute=30))
            
            dates = pd.DatetimeIndex(all_dates[:num_bars])
        else:
            dates = pd.bdate_range(end=datetime.now(), periods=num_bars, freq=freq)
        
        # Random walk with trend
        returns = np.random.normal(0.0005, 0.015, num_bars)  # Slight upward bias
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data with realistic intraday patterns
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Volatility based on timeframe and time of day
            if timeframe == "1m":
                # Higher volatility at open/close
                hour = date.hour
                if hour in [9, 15]:  # Market open/close hours
                    volatility = np.random.uniform(0.002, 0.008)
                else:
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
            
            # Volume based on timeframe and market session
            if timeframe == "1m":
                base_volume = 20000
                # Higher volume at market open/close
                hour = date.hour
                if hour == 9:  # Market open
                    volume_multiplier = np.random.uniform(3, 5)
                elif hour == 15:  # Market close
                    volume_multiplier = np.random.uniform(2, 3)
                else:
                    volume_multiplier = np.random.uniform(0.5, 1.5)
                volume = int(base_volume * volume_multiplier)
            elif timeframe == "5m":
                volume = np.random.randint(100000, 400000)
            elif timeframe == "1h":
                volume = np.random.randint(500000, 1500000)
            else:  # 1d
                volume = np.random.randint(2000000, 8000000)
            
            data.append({
                'Date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        # Add market session info for intraday timeframes
        session = self.market_hours.get_current_market_session()
        logger.info(f"Generated {len(df)} bars of mock data for {symbol}:{timeframe} (session: {session.value})")
        return df

class TechnicalAnalysisEngine:
    """
    Technical Analysis engine with multi-timeframe analysis and market hours awareness
    """
    
    def __init__(self, redis_client=None, data_provider=None):
        self.redis_client = redis_client
        self.data_provider = data_provider or CachedTechnicalDataProvider(redis_client)
        self.market_hours = MarketHours()
        
        # Timeframe configurations with update frequencies
        self.timeframe_configs = {
            TimeframeType.ONE_MINUTE.value: TimeframeConfig("1m", "1mo", 0.1, 100, "Scalping/Real-time", "1min"),
            TimeframeType.FIVE_MINUTE.value: TimeframeConfig("5m", "3mo", 0.2, 80, "Intraday Trading", "5min"),
            TimeframeType.ONE_HOUR.value: TimeframeConfig("1h", "6mo", 0.3, 50, "Swing Trading", "1hour"),
            TimeframeType.ONE_DAY.value: TimeframeConfig("1d", "2y", 0.4, 30, "Position Trading", "daily")
        }
        
    async def analyze_multi_timeframe_technical(self, symbol: str, timeframes: List[str] = None) -> MultiTimeframeTechnicalSignal:
        """
        Main multi-timeframe technical analysis function
        
        Args:
            symbol: Stock symbol
            timeframes: List of timeframes to analyze (defaults to market-appropriate timeframes)
            
        Returns:
            MultiTimeframeTechnicalSignal with combined analysis
        """
        if timeframes is None:
            # Default timeframes based on market session
            session = self.market_hours.get_current_market_session()
            if session == MarketSession.REGULAR:
                timeframes = ["1m", "5m", "1h", "1d"]  # All timeframes during regular hours
            elif session in [MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS]:
                timeframes = ["1h", "1d"]  # Extended hours - focus on longer timeframes
            else:  # Market closed
                timeframes = ["1d"]  # Only daily analysis when market is closed
        
        try:
            # Analyze each timeframe
            timeframe_signals = {}
            
            for tf in timeframes:
                if tf not in self.timeframe_configs:
                    logger.warning(f"Unsupported timeframe: {tf}")
                    continue
                
                config = self.timeframe_configs[tf]
                signal = await self.analyze_single_timeframe_technical(symbol, tf, config)
                
                if signal:
                    timeframe_signals[tf] = signal
                    logger.info(f"✅ {symbol}:{tf} technical analysis complete - {signal.technical_strength.value}")
                else:
                    logger.warning(f"❌ {symbol}:{tf} technical analysis failed")
            
            if not timeframe_signals:
                raise ValueError(f"No successful technical analysis for {symbol}")
            
            # Combine timeframes into master signal
            master_signal = self._combine_timeframe_technical_signals(symbol, timeframe_signals)
            
            # Cache the result
            await self._cache_multi_timeframe_technical_signal(symbol, master_signal)
            
            logger.info(f"🎯 Multi-timeframe technical analysis complete for {symbol}: {master_signal.master_strength.value}")
            return master_signal
            
        except Exception as e:
            logger.error(f"Multi-timeframe technical analysis failed for {symbol}: {e}")
            raise
    
    async def analyze_single_timeframe_technical(self, symbol: str, timeframe: str, config: TimeframeConfig) -> Optional[TechnicalSignal]:
        """Analyze a single timeframe for technical indicators"""
        try:
            # Get data for this timeframe
            data = await self.data_provider.get_timeframe_data(symbol, timeframe, config.period)
            
            if data is None or len(data) < config.bars_needed:
                logger.warning(f"Insufficient data for {symbol}:{timeframe}")
                return None
            
            current_price = float(data['close'].iloc[-1])
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(data)
            
            # Calculate individual scores
            trend_score = self._calculate_trend_score(data, indicators)
            momentum_score = self._calculate_momentum_score(data, indicators)
            volatility_score = self._calculate_volatility_score(data, indicators)
            volume_score = self._calculate_volume_score(data, indicators)
            
            # Overall technical assessment
            total_score = trend_score + momentum_score + volatility_score + volume_score
            technical_strength = self._determine_technical_strength(total_score, trend_score, momentum_score)
            trend_direction = self._determine_trend_direction(indicators)
            confidence = self._calculate_technical_confidence(trend_score, momentum_score, volatility_score, volume_score)
            
            # Generate insights
            trend_analysis = self._analyze_trend(data, indicators, timeframe)
            momentum_analysis = self._analyze_momentum(data, indicators, timeframe)
            volume_analysis = self._analyze_volume(data, indicators, timeframe)
            key_levels = self._identify_key_levels(data, indicators)
            
            # Create signal
            signal = TechnicalSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.utcnow(),
                current_price=current_price,
                indicators=indicators,
                trend_score=trend_score,
                momentum_score=momentum_score,
                volatility_score=volatility_score,
                volume_score=volume_score,
                total_score=total_score,
                technical_strength=technical_strength,
                trend_direction=trend_direction,
                confidence=confidence,
                trend_analysis=trend_analysis,
                momentum_analysis=momentum_analysis,
                volume_analysis=volume_analysis,
                key_levels=key_levels
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Single timeframe technical analysis failed for {symbol}:{timeframe}: {e}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        
        # Trend indicators
        sma_20 = self._sma(close, 20)[-1] if len(close) >= 20 else close[-1]
        sma_50 = self._sma(close, 50)[-1] if len(close) >= 50 else close[-1]
        sma_200 = self._sma(close, 200)[-1] if len(close) >= 200 else close[-1]
        ema_12 = self._ema(close, 12)[-1] if len(close) >= 12 else close[-1]
        ema_26 = self._ema(close, 26)[-1] if len(close) >= 26 else close[-1]
        
        # Momentum indicators
        rsi = self._calculate_rsi(close, 14)[-1] if len(close) >= 14 else 50
        macd_line, macd_signal, macd_histogram = self._calculate_macd(close)
        
        # Volatility indicators
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
        bb_percent = ((close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100) if bb_upper[-1] != bb_lower[-1] else 50
        atr = self._calculate_atr(high, low, close, 14)[-1] if len(close) >= 14 else 0
        
        # Volume indicators
        volume_sma = self._sma(volume, 20)[-1] if len(volume) >= 20 else volume[-1]
        volume_ratio = volume[-1] / volume_sma if volume_sma > 0 else 1.0
        
        # Support/Resistance levels
        support_levels, resistance_levels = self._calculate_support_resistance(high, low, close)
        
        return TechnicalIndicators(
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            rsi=rsi,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_upper=bb_upper[-1],
            bb_middle=bb_middle[-1],
            bb_lower=bb_lower[-1],
            bb_percent=bb_percent,
            atr=atr,
            volume_sma=volume_sma,
            volume_ratio=volume_ratio,
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )
    
    def _calculate_trend_score(self, data: pd.DataFrame, indicators: TechnicalIndicators) -> float:
        """Calculate trend strength score (0-1)"""
        close = data['close'].values
        current_price = close[-1]
        
        score = 0.0
        
        # Moving average alignment (40% of trend score)
        ma_score = 0.0
        if current_price > indicators.sma_20:
            ma_score += 0.25
        if current_price > indicators.sma_50:
            ma_score += 0.25
        if current_price > indicators.sma_200:
            ma_score += 0.25
        if indicators.sma_20 > indicators.sma_50:
            ma_score += 0.25
        
        score += ma_score * 0.4
        
        # EMA relationship (30% of trend score)
        ema_score = 0.0
        if indicators.ema_12 > indicators.ema_26:
            ema_score += 0.5
        if current_price > indicators.ema_12:
            ema_score += 0.5
        
        score += ema_score * 0.3
        
        # Price trend (30% of trend score)
        if len(close) >= 10:
            recent_trend = (close[-1] - close[-10]) / close[-10]
            if recent_trend > 0.02:  # +2% trend
                score += 0.3
            elif recent_trend > 0:
                score += 0.15
            elif recent_trend < -0.02:  # -2% trend
                score += 0.0  # Bearish trend gets 0 points in trend score
            else:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_momentum_score(self, data: pd.DataFrame, indicators: TechnicalIndicators) -> float:
        """Calculate momentum score (0-1)"""
        score = 0.0
        
        # RSI momentum (50% of momentum score)
        if 30 < indicators.rsi < 70:  # Healthy momentum
            if indicators.rsi > 50:
                score += 0.5 * (indicators.rsi - 50) / 20  # Scale from 50-70
            else:
                score += 0.25 * indicators.rsi / 50  # Partial credit below 50
        elif indicators.rsi >= 70:  # Overbought but still bullish
            score += 0.4
        
        # MACD momentum (50% of momentum score)
        if indicators.macd_line > indicators.macd_signal:
            score += 0.25
        if indicators.macd_histogram > 0:
            score += 0.25
        
        return min(score, 1.0)
    
    def _calculate_volatility_score(self, data: pd.DataFrame, indicators: TechnicalIndicators) -> float:
        """Calculate volatility score (0-1) - higher is better for trading"""
        score = 0.0
        
        # Bollinger Band position (60% of volatility score)
        if 20 < indicators.bb_percent < 80:  # Good trading range
            score += 0.6
        elif indicators.bb_percent >= 80:  # Near upper band
            score += 0.4
        elif indicators.bb_percent <= 20:  # Near lower band
            score += 0.3
        
        # ATR relative to price (40% of volatility score)
        close = data['close'].values
        atr_percent = (indicators.atr / close[-1]) * 100
        if 1.0 < atr_percent < 3.0:  # Ideal volatility for trading
            score += 0.4
        elif atr_percent >= 3.0:  # High volatility
            score += 0.2
        else:  # Low volatility
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_volume_score(self, data: pd.DataFrame, indicators: TechnicalIndicators) -> float:
        """Calculate volume score (0-1)"""
        score = 0.0
        
        # Volume confirmation (100% of volume score)
        if indicators.volume_ratio > 1.5:  # High volume
            score += 1.0
        elif indicators.volume_ratio > 1.2:  # Above average volume
            score += 0.7
        elif indicators.volume_ratio > 0.8:  # Normal volume
            score += 0.5
        else:  # Low volume
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_technical_strength(self, total_score: float, trend_score: float, momentum_score: float) -> TechnicalStrength:
        """Determine overall technical strength"""
        # Weight trend and momentum more heavily
        weighted_score = (trend_score * 0.4) + (momentum_score * 0.4) + (total_score * 0.2)
        
        if weighted_score >= 0.85:
            return TechnicalStrength.VERY_BULLISH
        elif weighted_score >= 0.7:
            return TechnicalStrength.BULLISH
        elif weighted_score >= 0.55:
            return TechnicalStrength.SLIGHTLY_BULLISH
        elif weighted_score >= 0.45:
            return TechnicalStrength.NEUTRAL
        elif weighted_score >= 0.3:
            return TechnicalStrength.SLIGHTLY_BEARISH
        elif weighted_score >= 0.15:
            return TechnicalStrength.BEARISH
        else:
            return TechnicalStrength.VERY_BEARISH
    
    def _determine_trend_direction(self, indicators: TechnicalIndicators) -> TrendDirection:
        """Determine trend direction based on moving averages"""
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA relationship
        if indicators.ema_12 > indicators.ema_26:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # SMA alignment
        if indicators.sma_20 > indicators.sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # MACD
        if indicators.macd_line > indicators.macd_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return TrendDirection.BULLISH
        elif bearish_signals > bullish_signals:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL
    
    def _calculate_technical_confidence(self, trend_score: float, momentum_score: float, 
                                      volatility_score: float, volume_score: float) -> float:
        """Calculate confidence in technical analysis"""
        # Higher confidence when all indicators align
        scores = [trend_score, momentum_score, volatility_score, volume_score]
        
        # Average score
        avg_score = np.mean(scores)
        
        # Consistency bonus (lower standard deviation = higher confidence)
        consistency = 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.5
        
        # Volume confirmation bonus
        volume_bonus = min(volume_score * 0.2, 0.2)
        
        confidence = avg_score * consistency + volume_bonus
        return min(confidence, 1.0)
    
    def _analyze_trend(self, data: pd.DataFrame, indicators: TechnicalIndicators, timeframe: str) -> str:
        """Generate trend analysis text"""
        close = data['close'].values
        current_price = close[-1]
        
        if current_price > indicators.sma_20 > indicators.sma_50 > indicators.sma_200:
            return f"Strong bullish trend across all timeframes. Price above all major moving averages."
        elif current_price > indicators.sma_20 > indicators.sma_50:
            return f"Bullish short-to-medium term trend. Price above 20 and 50 SMAs."
        elif current_price < indicators.sma_20 < indicators.sma_50 < indicators.sma_200:
            return f"Strong bearish trend. Price below all major moving averages."
        elif indicators.sma_20 > indicators.sma_50:
            return f"Mixed trend. Short-term bullish but longer-term concerns."
        else:
            return f"Sideways/consolidating trend. Price range-bound between moving averages."
    
    def _analyze_momentum(self, data: pd.DataFrame, indicators: TechnicalIndicators, timeframe: str) -> str:
        """Generate momentum analysis text"""
        if indicators.rsi > 70:
            return f"Overbought momentum (RSI: {indicators.rsi:.1f}). MACD {'bullish' if indicators.macd_line > indicators.macd_signal else 'bearish'}."
        elif indicators.rsi < 30:
            return f"Oversold momentum (RSI: {indicators.rsi:.1f}). Potential reversal setup."
        elif 40 < indicators.rsi < 60:
            return f"Neutral momentum (RSI: {indicators.rsi:.1f}). MACD showing {'bullish' if indicators.macd_line > indicators.macd_signal else 'bearish'} divergence."
        else:
            return f"{'Bullish' if indicators.rsi > 50 else 'Bearish'} momentum (RSI: {indicators.rsi:.1f}). Trend continuation likely."
    
    def _analyze_volume(self, data: pd.DataFrame, indicators: TechnicalIndicators, timeframe: str) -> str:
        """Generate volume analysis text"""
        if indicators.volume_ratio > 2.0:
            return f"Exceptionally high volume ({indicators.volume_ratio:.1f}x average). Strong institutional interest."
        elif indicators.volume_ratio > 1.5:
            return f"High volume confirmation ({indicators.volume_ratio:.1f}x average). Move well-supported."
        elif indicators.volume_ratio > 1.2:
            return f"Above-average volume ({indicators.volume_ratio:.1f}x average). Decent participation."
        elif indicators.volume_ratio > 0.8:
            return f"Normal volume levels ({indicators.volume_ratio:.1f}x average). Standard trading activity."
        else:
            return f"Below-average volume ({indicators.volume_ratio:.1f}x average). Lack of conviction in move."
    
    def _identify_key_levels(self, data: pd.DataFrame, indicators: TechnicalIndicators) -> Dict[str, float]:
        """Identify key technical levels"""
        close = data['close'].values
        current_price = close[-1]
        
        # Find nearest support and resistance
        support_levels = [s for s in indicators.support_levels if s < current_price]
        resistance_levels = [r for r in indicators.resistance_levels if r > current_price]
        
        nearest_support = max(support_levels) if support_levels else current_price * 0.95
        nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
        
        return {
            "current_price": current_price,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "sma_20": indicators.sma_20,
            "sma_50": indicators.sma_50,
            "bb_upper": indicators.bb_upper,
            "bb_lower": indicators.bb_lower
        }
    
    def _combine_timeframe_technical_signals(self, symbol: str, timeframe_signals: Dict[str, TechnicalSignal]) -> MultiTimeframeTechnicalSignal:
        """Combine multiple timeframe technical signals"""
        
        current_price = list(timeframe_signals.values())[0].current_price
        
        # Weighted scoring
        total_weighted_score = 0
        total_weight = 0
        
        for tf, signal in timeframe_signals.items():
            weight = self.timeframe_configs[tf].weight
            total_weighted_score += signal.total_score * weight
            total_weight += weight
        
        master_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Calculate alignment
        alignment_score, timeframes_agreeing, dominant_trend = self._calculate_technical_alignment(timeframe_signals)
        
        # Apply alignment bonus/penalty
        adjusted_master_score = master_score * alignment_score
        
        # Determine master strength
        master_strength = self._determine_master_technical_strength(adjusted_master_score, timeframes_agreeing, len(timeframe_signals))
        
        # Calculate master confidence
        individual_confidences = [signal.confidence for signal in timeframe_signals.values()]
        master_confidence = np.mean(individual_confidences) * alignment_score
        
        # Analyze cross-timeframe characteristics
        trend_strength = self._analyze_cross_timeframe_trend_strength(timeframe_signals)
        momentum_phase = self._analyze_cross_timeframe_momentum_phase(timeframe_signals)
        volatility_regime = self._analyze_volatility_regime(timeframe_signals)
        volume_profile = self._analyze_volume_profile(timeframe_signals)
        
        # Identify major levels across timeframes
        major_support, major_resistance = self._identify_major_levels(timeframe_signals)
        next_support, next_resistance = self._identify_next_levels(timeframe_signals, current_price)
        
        return MultiTimeframeTechnicalSignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            current_price=current_price,
            master_score=adjusted_master_score,
            master_strength=master_strength,
            master_confidence=master_confidence,
            alignment_score=alignment_score,
            timeframes_agreeing=timeframes_agreeing,
            dominant_trend=dominant_trend,
            timeframe_signals=timeframe_signals,
            trend_strength=trend_strength,
            momentum_phase=momentum_phase,
            volatility_regime=volatility_regime,
            volume_profile=volume_profile,
            major_support=major_support,
            major_resistance=major_resistance,
            next_support=next_support,
            next_resistance=next_resistance
        )
    
    def _calculate_technical_alignment(self, timeframe_signals: Dict[str, TechnicalSignal]) -> Tuple[float, int, TrendDirection]:
        """Calculate technical alignment across timeframes"""
        
        trends = []
        strengths = []
        
        for signal in timeframe_signals.values():
            trends.append(signal.trend_direction.value)
            
            # Convert technical strength to numeric value
            strength_map = {
                TechnicalStrength.VERY_BULLISH: 1.0,
                TechnicalStrength.BULLISH: 0.7,
                TechnicalStrength.SLIGHTLY_BULLISH: 0.4,
                TechnicalStrength.NEUTRAL: 0.0,
                TechnicalStrength.SLIGHTLY_BEARISH: -0.4,
                TechnicalStrength.BEARISH: -0.7,
                TechnicalStrength.VERY_BEARISH: -1.0
            }
            strengths.append(strength_map[signal.technical_strength])
        
        if not trends:
            return 1.0, 0, TrendDirection.NEUTRAL
        
        # Count trend agreements
        bullish_count = sum(1 for t in trends if t == 1)
        bearish_count = sum(1 for t in trends if t == -1)
        neutral_count = sum(1 for t in trends if t == 0)
        
        total_timeframes = len(trends)
        
        # Determine dominant trend
        if bullish_count > bearish_count and bullish_count > neutral_count:
            dominant_trend = TrendDirection.BULLISH
            agreeing = bullish_count
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            dominant_trend = TrendDirection.BEARISH
            agreeing = bearish_count
        else:
            dominant_trend = TrendDirection.NEUTRAL
            agreeing = neutral_count
        
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
        
        return min(alignment_score, 1.3), agreeing, dominant_trend
    
    def _determine_master_technical_strength(self, score: float, agreeing: int, total: int) -> TechnicalStrength:
        """Determine master technical strength"""
        
        agreement_ratio = agreeing / total if total > 0 else 0
        
        if score >= 3.5 and agreement_ratio >= 0.75:
            return TechnicalStrength.VERY_BULLISH
        elif score >= 2.8 and agreement_ratio >= 0.6:
            return TechnicalStrength.BULLISH
        elif score >= 2.2:
            return TechnicalStrength.SLIGHTLY_BULLISH
        elif score >= 1.8:
            return TechnicalStrength.NEUTRAL
        elif score >= 1.2:
            return TechnicalStrength.SLIGHTLY_BEARISH
        elif score >= 0.6:
            return TechnicalStrength.BEARISH
        else:
            return TechnicalStrength.VERY_BEARISH
    
    def _analyze_cross_timeframe_trend_strength(self, timeframe_signals: Dict[str, TechnicalSignal]) -> str:
        """Analyze trend strength across timeframes"""
        
        # Weight longer timeframes more heavily
        weights = {"1d": 0.4, "1h": 0.3, "5m": 0.2, "1m": 0.1}
        
        total_weighted_score = 0
        total_weight = 0
        
        for tf, signal in timeframe_signals.items():
            weight = weights.get(tf, 0.1)
            total_weighted_score += signal.trend_score * weight
            total_weight += weight
        
        avg_trend_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        if avg_trend_score >= 0.75:
            return "Strong"
        elif avg_trend_score >= 0.5:
            return "Moderate"
        else:
            return "Weak"
    
    def _analyze_cross_timeframe_momentum_phase(self, timeframe_signals: Dict[str, TechnicalSignal]) -> str:
        """Analyze momentum phase across timeframes"""
        
        # Compare short vs long timeframe momentum
        short_tf_momentum = []
        long_tf_momentum = []
        
        for tf, signal in timeframe_signals.items():
            if tf in ["1m", "5m"]:
                short_tf_momentum.append(signal.momentum_score)
            else:
                long_tf_momentum.append(signal.momentum_score)
        
        if short_tf_momentum and long_tf_momentum:
            short_avg = np.mean(short_tf_momentum)
            long_avg = np.mean(long_tf_momentum)
            
            difference = short_avg - long_avg
            
            if difference > 0.2:
                return "Acceleration"
            elif difference < -0.2:
                return "Deceleration"
            else:
                return "Continuation"
        
        return "Continuation"
    
    def _analyze_volatility_regime(self, timeframe_signals: Dict[str, TechnicalSignal]) -> str:
        """Analyze volatility regime"""
        
        volatility_scores = [signal.volatility_score for signal in timeframe_signals.values()]
        avg_volatility = np.mean(volatility_scores)
        
        if avg_volatility >= 0.7:
            return "High"
        elif avg_volatility >= 0.4:
            return "Normal"
        else:
            return "Low"
    
    def _analyze_volume_profile(self, timeframe_signals: Dict[str, TechnicalSignal]) -> str:
        """Analyze volume profile"""
        
        volume_scores = [signal.volume_score for signal in timeframe_signals.values()]
        avg_volume = np.mean(volume_scores)
        
        if avg_volume >= 0.7:
            return "Strong"
        elif avg_volume >= 0.4:
            return "Normal"
        else:
            return "Weak"
    
    def _identify_major_levels(self, timeframe_signals: Dict[str, TechnicalSignal]) -> Tuple[float, float]:
        """Identify major support and resistance across timeframes"""
        
        all_support = []
        all_resistance = []
        
        for signal in timeframe_signals.values():
            all_support.extend(signal.indicators.support_levels)
            all_resistance.extend(signal.indicators.resistance_levels)
        
        # Find most significant levels (those that appear across multiple timeframes)
        major_support = np.median(all_support) if all_support else 0
        major_resistance = np.median(all_resistance) if all_resistance else 0
        
        return major_support, major_resistance
    
    def _identify_next_levels(self, timeframe_signals: Dict[str, TechnicalSignal], current_price: float) -> Tuple[float, float]:
        """Identify next support and resistance levels"""
        
        all_support = []
        all_resistance = []
        
        for signal in timeframe_signals.values():
            # Add moving averages as dynamic support/resistance
            all_support.extend([signal.indicators.sma_20, signal.indicators.sma_50, signal.indicators.bb_lower])
            all_resistance.extend([signal.indicators.sma_20, signal.indicators.sma_50, signal.indicators.bb_upper])
            all_support.extend(signal.indicators.support_levels)
            all_resistance.extend(signal.indicators.resistance_levels)
        
        # Find nearest levels
        support_below = [s for s in all_support if s < current_price]
        resistance_above = [r for r in all_resistance if r > current_price]
        
        next_support = max(support_below) if support_below else current_price * 0.95
        next_resistance = min(resistance_above) if resistance_above else current_price * 1.05
        
        return next_support, next_resistance
    
    # Cache management methods
    async def _cache_multi_timeframe_technical_signal(self, symbol: str, signal: MultiTimeframeTechnicalSignal):
        """Cache multi-timeframe technical signal with market hours awareness"""
        try:
            if self.redis_client:
                # Determine TTL based on market session
                session = self.market_hours.get_current_market_session()
                
                if session == MarketSession.REGULAR:
                    ttl = 300  # 5 minutes during regular hours
                elif session in [MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS]:
                    ttl = 900  # 15 minutes during extended hours
                else:  # Market closed
                    ttl = 3600  # 1 hour when market is closed
                
                cache_key = f"technical:multi:{symbol}"
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(signal.to_dict())
                )
                
                # Cache individual timeframes with appropriate TTLs
                for tf, tf_signal in signal.timeframe_signals.items():
                    tf_ttl = ttl // 2 if tf in ["1m", "5m"] else ttl  # Shorter TTL for intraday
                    tf_cache_key = f"technical:{tf}:{symbol}"
                    await self.redis_client.setex(
                        tf_cache_key,
                        tf_ttl,
                        json.dumps(tf_signal.to_dict())
                    )
                
                session_name = session.value
                logger.info(f"✅ Cached multi-timeframe technical signal for {symbol} (session: {session_name}, TTL: {ttl}s)")
                
        except Exception as e:
            logger.warning(f"Failed to cache multi-timeframe technical signal for {symbol}: {e}")
    
    async def get_cached_technical_signal(self, symbol: str, timeframe: str = None) -> Union[MultiTimeframeTechnicalSignal, TechnicalSignal, None]:
        """Get cached technical signal"""
        try:
            if not self.redis_client:
                return None
            
            if timeframe:
                # Get specific timeframe
                cache_key = f"technical:{timeframe}:{symbol}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return data
            else:
                # Get multi-timeframe
                cache_key = f"technical:multi:{symbol}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return data
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached technical signal for {symbol}: {e}")
            return None
    
    # Technical indicator calculation methods
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - period + 1)
            result[i] = np.mean(data[start_idx:i+1])
        return result
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Relative Strength Index"""
        if len(close) < period + 1:
            return np.full_like(close, 50)
        
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(close)
        avg_losses = np.zeros_like(close)
        
        # Initial averages
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # Subsequent averages using Wilder's smoothing
        for i in range(period + 1, len(close)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, close: np.ndarray) -> Tuple[float, float, float]:
        """MACD calculation"""
        if len(close) < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = self._ema(close, 12)
        ema_26 = self._ema(close, 26)
        macd_line = ema_12 - ema_26
        
        if len(macd_line) >= 9:
            macd_signal = self._ema(macd_line, 9)
            macd_histogram = macd_line - macd_signal
            return macd_line[-1], macd_signal[-1], macd_histogram[-1]
        
        return macd_line[-1], 0.0, 0.0
    
    def _calculate_bollinger_bands(self, close: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands calculation"""
        sma = self._sma(close, period)
        
        rolling_std = np.zeros_like(close)
        for i in range(len(close)):
            start_idx = max(0, i - period + 1)
            rolling_std[i] = np.std(close[start_idx:i+1])
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Average True Range"""
        if len(close) < 2:
            return np.zeros_like(close)
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        atr = self._sma(tr, period)
        return atr
    
    def _calculate_support_resistance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        
        # Find pivot highs and lows
        def find_pivots(data, window=5):
            pivots = []
            for i in range(window, len(data) - window):
                if all(data[i] > data[i-j] for j in range(1, window+1)) and \
                   all(data[i] > data[i+j] for j in range(1, window+1)):
                    pivots.append(data[i])
                elif all(data[i] < data[i-j] for j in range(1, window+1)) and \
                     all(data[i] < data[i+j] for j in range(1, window+1)):
                    pivots.append(data[i])
            return pivots
        
        # Get pivot points from highs and lows
        resistance_candidates = find_pivots(high)
        support_candidates = find_pivots(low)
        
        # Filter and sort
        current_price = close[-1]
        resistance_levels = sorted([r for r in resistance_candidates if r > current_price])[:5]
        support_levels = sorted([s for s in support_candidates if s < current_price], reverse=True)[:5]
        
        return support_levels, resistance_levels

# FastAPI Router for Technical Analysis endpoints
router = APIRouter(prefix="/technical", tags=["Technical Analysis Engine"])

# Global engine instance
technical_engine = TechnicalAnalysisEngine()

@router.post("/analyze-multi/{symbol}")
async def analyze_multi_timeframe_technical_endpoint(symbol: str, timeframes: List[str] = None):
    """
    Analyze a symbol using multi-timeframe technical analysis
    
    Query params:
    - timeframes: Optional list of timeframes ["1m", "5m", "1h", "1d"]
    """
    try:
        # Market hours check
        market_session = technical_engine.market_hours.get_current_market_session()
        
        # Use market-appropriate timeframes if none provided
        if timeframes is None:
            if market_session == MarketSession.REGULAR:
                timeframes = ["1m", "5m", "1h", "1d"]
            elif market_session in [MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS]:
                timeframes = ["1h", "1d"]
            else:
                timeframes = ["1d"]
        
        # Validate timeframes
        valid_timeframes = [tf.value for tf in TimeframeType]
        invalid_tfs = [tf for tf in timeframes if tf not in valid_timeframes]
        if invalid_tfs:
            raise HTTPException(status_code=400, detail=f"Invalid timeframes: {invalid_tfs}")
        
        # Analyze symbol
        signal = await technical_engine.analyze_multi_timeframe_technical(symbol.upper(), timeframes)
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "market_session": market_session.value,
            "timeframes_analyzed": timeframes,
            "signal": signal.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Multi-timeframe technical analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/analyze-cached/{symbol}")
async def get_cached_technical_analysis(symbol: str, timeframe: str = None):
    """
    Get cached technical analysis
    """
    try:
        cached_signal = await technical_engine.get_cached_technical_signal(symbol.upper(), timeframe)
        
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
                "message": "No cached technical analysis found"
            }
        
    except Exception as e:
        logger.error(f"Failed to get cached technical analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Cache retrieval failed: {str(e)}")

@router.get("/market-status")
async def get_market_status():
    """
    Get current market status and session
    """
    market_hours = MarketHours()
    session = market_hours.get_current_market_session()
    is_open = market_hours.is_market_hours(include_extended=True)
    is_regular_hours = market_hours.is_market_hours(include_extended=False)
    
    # Get timezone info
    eastern_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern_tz)
    
    return {
        "market_session": session.value,
        "is_market_open": is_open,
        "is_regular_hours": is_regular_hours,
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "recommended_timeframes": {
            "regular_hours": ["1m", "5m", "1h", "1d"],
            "extended_hours": ["1h", "1d"],
            "market_closed": ["1d"]
        },
        "next_session_info": {
            "pre_market": "4:00 AM - 9:30 AM ET",
            "regular": "9:30 AM - 4:00 PM ET", 
            "after_hours": "4:00 PM - 8:00 PM ET"
        }
    }

@router.get("/test-multi/{symbol}")
async def test_multi_timeframe_technical(symbol: str):
    """
    Test multi-timeframe technical analysis with market-appropriate timeframes
    """
    try:
        # Use market-appropriate timeframes
        market_session = technical_engine.market_hours.get_current_market_session()
        
        if market_session == MarketSession.REGULAR:
            timeframes = ["1m", "5m", "1h", "1d"]
        elif market_session in [MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS]:
            timeframes = ["1h", "1d"]
        else:
            timeframes = ["1d"]
        
        signal = await technical_engine.analyze_multi_timeframe_technical(symbol.upper(), timeframes)
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "analysis_type": "full_multi_timeframe_technical",
            "market_session": market_session.value,
            "timeframes_analyzed": timeframes,
            "signal": signal.to_dict(),
            "note": "This analysis uses market-hours-appropriate timeframes and mock data for testing"
        }
        
    except Exception as e:
        logger.error(f"Test multi-timeframe technical analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Test analysis failed: {str(e)}")

@router.get("/engine-status")
async def get_technical_engine_status():
    """
    Get technical analysis engine status and configuration
    """
    market_hours = MarketHours()
    session = market_hours.get_current_market_session()
    
    return {
        "engine": "Technical Analysis Multi-Timeframe Engine",
        "version": "2.0.0",
        "market_hours_aware": True,
        "current_market_session": session.value,
        "supported_timeframes": [tf.value for tf in TimeframeType],
        "timeframe_configs": {
            tf: {
                "weight": config.weight,
                "description": config.description,
                "bars_needed": config.bars_needed,
                "update_frequency": config.update_frequency
            }
            for tf, config in technical_engine.timeframe_configs.items()
        },
        "technical_indicators": {
            "trend": ["SMA 20/50/200", "EMA 12/26", "Moving Average Alignment"],
            "momentum": ["RSI", "MACD Line/Signal/Histogram"],
            "volatility": ["Bollinger Bands", "ATR", "BB Percent"],
            "volume": ["Volume SMA", "Volume Ratio", "Volume Confirmation"],
            "support_resistance": ["Pivot Points", "Dynamic S/R Levels"]
        },
        "signal_strengths": [strength.value for strength in TechnicalStrength],
        "market_hours_logic": {
            "regular_hours": "All timeframes active, 5-minute cache TTL",
            "extended_hours": "Hourly and daily only, 15-minute cache TTL", 
            "market_closed": "Daily analysis only, 1-hour cache TTL",
            "api_calls": "Only made during appropriate market sessions"
        },
        "new_features": [
            "Market hours awareness for API calls and caching",
            "Multi-timeframe technical analysis with weighted scoring",
            "Dynamic cache TTL based on market session",
            "Cross-timeframe trend and momentum analysis",
            "Volatility regime and volume profile assessment",
            "Major support/resistance level identification"
        ],
        "cache_status": "Redis-based caching enabled" if technical_engine.redis_client else "No cache configured"
    }

@router.get("/indicators/{symbol}")
async def get_technical_indicators_detail(symbol: str, timeframe: str = "1d"):
    """
    Get detailed technical indicators for a specific symbol and timeframe
    """
    try:
        if timeframe not in [tf.value for tf in TimeframeType]:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
        
        # Check if timeframe is appropriate for current market session
        market_hours = MarketHours()
        if not market_hours.should_update_timeframe(timeframe):
            logger.warning(f"Timeframe {timeframe} not recommended during current market session")
        
        config = technical_engine.timeframe_configs[timeframe]
        signal = await technical_engine.analyze_single_timeframe_technical(symbol.upper(), timeframe, config)
        
        if not signal:
            raise HTTPException(status_code=404, detail=f"Could not analyze {symbol} for {timeframe}")
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "current_price": signal.current_price,
            "technical_strength": signal.technical_strength.value,
            "trend_direction": signal.trend_direction.value,
            "confidence": signal.confidence,
            "detailed_indicators": {
                "trend_indicators": {
                    "sma_20": signal.indicators.sma_20,
                    "sma_50": signal.indicators.sma_50,
                    "sma_200": signal.indicators.sma_200,
                    "ema_12": signal.indicators.ema_12,
                    "ema_26": signal.indicators.ema_26
                },
                "momentum_indicators": {
                    "rsi": signal.indicators.rsi,
                    "macd_line": signal.indicators.macd_line,
                    "macd_signal": signal.indicators.macd_signal,
                    "macd_histogram": signal.indicators.macd_histogram
                },
                "volatility_indicators": {
                    "bollinger_upper": signal.indicators.bb_upper,
                    "bollinger_middle": signal.indicators.bb_middle,
                    "bollinger_lower": signal.indicators.bb_lower,
                    "bollinger_percent": signal.indicators.bb_percent,
                    "atr": signal.indicators.atr
                },
                "volume_indicators": {
                    "volume_sma": signal.indicators.volume_sma,
                    "volume_ratio": signal.indicators.volume_ratio
                }
            },
            "analysis": {
                "trend_analysis": signal.trend_analysis,
                "momentum_analysis": signal.momentum_analysis,
                "volume_analysis": signal.volume_analysis
            },
            "key_levels": signal.key_levels,
            "scores": {
                "trend_score": signal.trend_score,
                "momentum_score": signal.momentum_score,
                "volatility_score": signal.volatility_score,
                "volume_score": signal.volume_score,
                "total_score": signal.total_score
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get technical indicators for {symbol}:{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=f"Indicator analysis failed: {str(e)}")

# Export for use in main application
__all__ = ['TechnicalAnalysisEngine', 'MultiTimeframeTechnicalSignal', 'TechnicalSignal', 'TechnicalStrength', 'TimeframeType', 'MarketHours', 'router']
