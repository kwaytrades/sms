# services/technical_analysis.py - Unified Technical Analysis Service
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
import holidays
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import json
import re

from config import settings, POPULAR_TICKERS, is_popular_ticker

class TechnicalAnalysisService:
    def __init__(self):
        self.session = None
        self.market_scheduler = MarketScheduler()
        self.cache_manager = None  # Will be set by database service
        
    async def initialize(self, cache_manager):
        """Initialize with cache manager from database service"""
        self.cache_manager = cache_manager
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        logger.info("✅ Technical Analysis Service initialized")
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
    
    async def analyze_symbol(self, symbol: str, interval: str = "1d", period: str = "1mo") -> Dict[str, Any]:
        """Complete technical analysis for a symbol with caching"""
        try:
            # Validate inputs
            symbol = self._validate_symbol(symbol)
            interval = self._validate_interval(interval)
            period = self._validate_period(period)
            
            # Check cache first
            cache_key = f"ta:{symbol}:{interval}:{period}"
            cached_result = await self._get_cached_data(cache_key)
            
            if cached_result:
                cached_result["cache_status"] = "hit"
                logger.info(f"📈 Cache HIT for {symbol}")
                return cached_result
            
            logger.info(f"📈 Cache MISS for {symbol} - fetching fresh data")
            
            # Fetch fresh market data
            df = await self._fetch_market_data(symbol, period)
            
            if df.empty:
                raise ValueError(f"No market data available for {symbol}")
            
            # Calculate all technical indicators
            indicators = await self._calculate_all_indicators(df)
            
            # Find support/resistance levels
            levels = await self._find_support_resistance(df)
            
            # Detect price gaps
            gaps = await self._detect_gaps(df)
            
            # Generate trading signals
            signals = await self._generate_signals(df, indicators, levels)
            
            # Build response
            latest = df.iloc[-1]
            result = {
                "symbol": symbol,
                "interval": interval,
                "period": period,
                "timestamp": datetime.utcnow().isoformat(),
                "data_source": "eodhd",
                "cache_status": "miss",
                "data_points": len(df),
                
                # Price data
                "current_price": round(float(latest['Close']), 2),
                "price_change": {
                    "amount": round(float(latest['Close'] - df['Close'].iloc[-2]), 2) if len(df) > 1 else 0,
                    "percent": round(float((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100), 2) if len(df) > 1 else 0
                },
                "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
                
                # Technical analysis
                "indicators": indicators,
                "support_levels": levels.get("support", []),
                "resistance_levels": levels.get("resistance", []),
                "gaps": gaps,
                "signals": signals,
                
                # Metadata
                "is_popular": is_popular_ticker(symbol),
                "market_hours": self.market_scheduler.is_market_hours()
            }
            
            # Cache the result
            await self._cache_data(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Technical analysis failed for {symbol}: {e}")
            return {
                "error": f"Technical analysis failed for {symbol}: {str(e)}",
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_trading_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Get just the trading signals for a symbol"""
        try:
            analysis = await self.analyze_symbol(symbol)
            if "error" in analysis:
                return [{"type": "error", "message": analysis["error"]}]
            
            return analysis.get("signals", [])
            
        except Exception as e:
            logger.error(f"❌ Get signals failed for {symbol}: {e}")
            return [{"type": "error", "message": str(e)}]
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.cache_manager:
                return {"error": "Cache manager not available"}
            
            # Get cache statistics
            stats = {
                "cache_backend": "Redis",
                "market_hours": self.market_scheduler.is_market_hours(),
                "popular_tickers_count": len(POPULAR_TICKERS),
                "cache_settings": {
                    "popular_ttl": settings.cache_popular_ttl,
                    "ondemand_ttl": settings.cache_ondemand_ttl,
                    "afterhours_ttl": settings.cache_afterhours_ttl
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Get cache stats failed: {e}")
            return {"error": str(e)}
    
    async def clear_cache(self) -> bool:
        """Clear all technical analysis cache"""
        try:
            if not self.cache_manager:
                return False
            
            # This would clear all TA cache keys
            # Implementation depends on your cache manager
            return True
            
        except Exception as e:
            logger.error(f"❌ Clear cache failed: {e}")
            return False
    
    async def invalidate_cache(self, symbol: str) -> bool:
        """Invalidate cache for specific symbol"""
        try:
            if not self.cache_manager:
                return False
            
            symbol = symbol.upper()
            # Clear all cache entries for this symbol
            cache_patterns = [
                f"ta:{symbol}:1d:1mo",
                f"ta:{symbol}:1d:3mo",
                f"ta:{symbol}:1d:6mo",
                f"ta:{symbol}:1d:1y"
            ]
            
            for pattern in cache_patterns:
                await self.cache_manager.delete(pattern)
            
            logger.info(f"🗑️ Cache invalidated for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Invalidate cache failed for {symbol}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "service": "Technical Analysis",
            "version": "3.0.0",
            "market_hours": self.market_scheduler.is_market_hours(),
            "next_market_open": self.market_scheduler.next_market_open().isoformat(),
            "supported_indicators": [
                "RSI", "MACD", "EMA", "Bollinger Bands", 
                "VWAP", "ATR", "Support/Resistance", "Gap Analysis"
            ],
            "data_source": "EODHD",
            "cache_enabled": self.cache_manager is not None
        }
    
    # Private methods
    
    def _validate_symbol(self, symbol: str) -> str:
        """Validate and clean symbol"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.strip().upper()
        
        if not re.match(r'^[A-Z0-9.\-]+$', symbol):
            raise ValueError("Symbol contains invalid characters")
        
        if len(symbol) > 10:
            raise ValueError("Symbol too long")
        
        return symbol
    
    def _validate_interval(self, interval: str) -> str:
        """Validate interval parameter"""
        valid_intervals = ['1d', '1h', '4h']
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Supported: {valid_intervals}")
        return interval
    
    def _validate_period(self, period: str) -> str:
        """Validate period parameter"""
        valid_periods = ['1mo', '3mo', '6mo', '1y', '2y']
        if period not in valid_periods:
            raise ValueError(f"Invalid period. Supported: {valid_periods}")
        return period
    
    async def _fetch_market_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch market data from EODHD"""
        try:
            if not settings.eodhd_api_key:
                raise ValueError("EODHD API key not configured")
            
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            period_days = {'1mo': 35, '3mo': 95, '6mo': 185, '1y': 370, '2y': 735}
            days = period_days.get(period, 35)
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"https://eodhd.com/api/eod/{symbol}.US"
            params = {
                'api_token': settings.eodhd_api_key,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if not data:
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    df_data = []
                    for item in data:
                        try:
                            df_data.append({
                                'Open': float(item['open']),
                                'High': float(item['high']),
                                'Low': float(item['low']),
                                'Close': float(item['adjusted_close']),
                                'Volume': int(item['volume']) if item['volume'] else 0
                            })
                        except (KeyError, ValueError, TypeError):
                            continue
                    
                    if not df_data:
                        return pd.DataFrame()
                    
                    dates = pd.to_datetime([item['date'] for item in data[:len(df_data)]])
                    df = pd.DataFrame(df_data, index=dates)
                    df.sort_index(inplace=True)
                    
                    logger.info(f"📊 Fetched {len(df)} data points for {symbol}")
                    return df
                else:
                    logger.error(f"❌ EODHD API error {response.status} for {symbol}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"❌ Fetch market data failed for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        try:
            if len(df) < 14:  # Minimum data for most indicators
                return indicators
            
            # RSI
            indicators['rsi'] = await self._calculate_rsi(df['Close'])
            
            # MACD
            indicators['macd'] = await self._calculate_macd(df['Close'])
            
            # EMA
            indicators['ema'] = await self._calculate_emas(df['Close'])
            
            # Bollinger Bands
            indicators['bollinger_bands'] = await self._calculate_bollinger_bands(df['Close'])
            
            # VWAP
            indicators['vwap'] = await self._calculate_vwap(df)
            
            # ATR
            indicators['atr'] = await self._calculate_atr(df)
            
        except Exception as e:
            logger.error(f"❌ Calculate indicators failed: {e}")
        
        return indicators
    
    async def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Dict[str, Any]:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = float(rsi.iloc[-1])
            
            return {
                "value": round(current_rsi, 2),
                "signal": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral",
                "interpretation": self._interpret_rsi(current_rsi)
            }
            
        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {e}")
            return {}
    
    async def _calculate_macd(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate MACD indicator"""
        try:
            ema_12 = prices.ewm(span=12, adjust=False).mean()
            ema_26 = prices.ewm(span=26, adjust=False).mean()
            
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return {
                "macd": round(float(macd_line.iloc[-1]), 4),
                "signal": round(float(signal_line.iloc[-1]), 4),
                "histogram": round(float(histogram.iloc[-1]), 4),
                "trend": "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"
            }
            
        except Exception as e:
            logger.error(f"❌ MACD calculation failed: {e}")
            return {}
    
    async def _calculate_emas(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate various EMA periods"""
        try:
            emas = {}
            for period in [20, 50, 200]:
                if len(prices) >= period:
                    ema = prices.ewm(span=period, adjust=False).mean()
                    emas[f"ema_{period}"] = round(float(ema.iloc[-1]), 2)
            
            return emas
            
        except Exception as e:
            logger.error(f"❌ EMA calculation failed: {e}")
            return {}
    
    async def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            
            upper = rolling_mean + (rolling_std * std_dev)
            lower = rolling_mean - (rolling_std * std_dev)
            
            current_price = prices.iloc[-1]
            bb_position = ((current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])) * 100
            
            return {
                "upper": round(float(upper.iloc[-1]), 2),
                "middle": round(float(rolling_mean.iloc[-1]), 2),
                "lower": round(float(lower.iloc[-1]), 2),
                "bb_position": round(bb_position, 2),
                "signal": "overbought" if bb_position > 80 else "oversold" if bb_position < 20 else "neutral"
            }
            
        except Exception as e:
            logger.error(f"❌ Bollinger Bands calculation failed: {e}")
            return {}
    
    async def _calculate_vwap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate VWAP"""
        try:
            if df['Volume'].sum() == 0:
                return {}
            
            volume_sum = df['Volume'].cumsum()
            price_volume_sum = (df['Close'] * df['Volume']).cumsum()
            vwap = price_volume_sum / volume_sum
            
            current_price = df['Close'].iloc[-1]
            vwap_value = float(vwap.iloc[-1])
            
            return {
                "value": round(vwap_value, 2),
                "signal": "bullish" if current_price > vwap_value else "bearish"
            }
            
        except Exception as e:
            logger.error(f"❌ VWAP calculation failed: {e}")
            return {}
    
    async def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """Calculate Average True Range"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            atr_value = float(atr.iloc[-1])
            current_price = df['Close'].iloc[-1]
            
            return {
                "value": round(atr_value, 4),
                "volatility": "high" if atr_value > current_price * 0.02 else "low"
            }
            
        except Exception as e:
            logger.error(f"❌ ATR calculation failed: {e}")
            return {}
    
    async def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels"""
        try:
            if len(df) < 10:
                return {"support": [], "resistance": []}
            
            # Simple pivot point method
            window = min(5, len(df) // 4)
            current_price = df['Close'].iloc[-1]
            
            # Find pivot highs (resistance)
            pivot_highs = []
            for i in range(window, len(df) - window):
                if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, window+1)):
                    pivot_highs.append(df['High'].iloc[i])
            
            # Find pivot lows (support)
            pivot_lows = []
            for i in range(window, len(df) - window):
                if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, window+1)):
                    pivot_lows.append(df['Low'].iloc[i])
            
            # Get nearest levels
            resistance_levels = sorted([r for r in pivot_highs if r > current_price])[:3]
            support_levels = sorted([s for s in pivot_lows if s < current_price], reverse=True)[:3]
            
            return {
                "support": [round(float(s), 2) for s in support_levels],
                "resistance": [round(float(r), 2) for r in resistance_levels]
            }
            
        except Exception as e:
            logger.error(f"❌ Support/resistance calculation failed: {e}")
            return {"support": [], "resistance": []}
    
    async def _detect_gaps(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect price gaps"""
        try:
            if len(df) < 2:
                return []
            
            gaps = []
            prev_close = df['Close'].shift(1)
            gap_percent = (df['Open'] - prev_close) / prev_close * 100
            
            significant_gaps = gap_percent[abs(gap_percent) > 1.0].dropna()
            
            for idx, gap_pct in significant_gaps.tail(5).items():
                gaps.append({
                    "date": idx.strftime('%Y-%m-%d'),
                    "type": "gap_up" if gap_pct > 0 else "gap_down",
                    "gap_percent": round(float(gap_pct), 2),
                    "prev_close": round(float(prev_close.loc[idx]), 2),
                    "open": round(float(df['Open'].loc[idx]), 2)
                })
            
            return gaps
            
        except Exception as e:
            logger.error(f"❌ Gap detection failed: {e}")
            return []
    
    async def _generate_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict) -> List[Dict[str, str]]:
        """Generate trading signals"""
        signals = []
        
        try:
            current_price = df['Close'].iloc[-1]
            
            # RSI signals
            rsi_data = indicators.get('rsi', {})
            if rsi_data:
                rsi_val = rsi_data.get('value', 50)
                if rsi_val > 70:
                    signals.append({
                        "type": "warning",
                        "indicator": "RSI",
                        "message": f"Overbought (RSI: {rsi_val})",
                        "strength": "strong" if rsi_val > 80 else "moderate"
                    })
                elif rsi_val < 30:
                    signals.append({
                        "type": "opportunity",
                        "indicator": "RSI", 
                        "message": f"Oversold (RSI: {rsi_val})",
                        "strength": "strong" if rsi_val < 20 else "moderate"
                    })
            
            # MACD signals
            macd_data = indicators.get('macd', {})
            if macd_data and macd_data.get('trend'):
                if macd_data['trend'] == 'bullish':
                    signals.append({
                        "type": "bullish",
                        "indicator": "MACD",
                        "message": "Bullish momentum confirmed",
                        "strength": "moderate"
                    })
                else:
                    signals.append({
                        "type": "bearish",
                        "indicator": "MACD",
                        "message": "Bearish momentum confirmed", 
                        "strength": "moderate"
                    })
            
            # Support/Resistance signals
            for support in levels.get("support", []):
                if abs(current_price - support) / support * 100 < 2:  # Within 2%
                    signals.append({
                        "type": "opportunity",
                        "indicator": "Support",
                        "message": f"Near support at ${support}",
                        "strength": "moderate"
                    })
            
            for resistance in levels.get("resistance", []):
                if abs(current_price - resistance) / resistance * 100 < 2:  # Within 2%
                    signals.append({
                        "type": "warning",
                        "indicator": "Resistance",
                        "message": f"Near resistance at ${resistance}",
                        "strength": "moderate"
                    })
            
            if not signals:
                signals.append({
                    "type": "neutral",
                    "indicator": "Overall",
                    "message": "No strong signals detected",
                    "strength": "neutral"
                })
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            signals.append({
                "type": "error",
                "indicator": "System",
                "message": "Error generating signals",
                "strength": "unknown"
            })
        
        return signals
    
    def _interpret_rsi(self, rsi_value: float) -> str:
        """Provide RSI interpretation"""
        if rsi_value > 80:
            return "Extremely overbought - strong sell signal"
        elif rsi_value > 70:
            return "Overbought - consider selling"
        elif rsi_value < 20:
            return "Extremely oversold - strong buy signal"
        elif rsi_value < 30:
            return "Oversold - consider buying"
        else:
            return "Neutral territory"
    
    async def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        try:
            if not self.cache_manager:
                return None
            
            # This would use your Redis cache manager
            # Implementation depends on your cache manager interface
            return None  # Placeholder
            
        except Exception as e:
            logger.error(f"❌ Cache get failed: {e}")
            return None
    
    async def _cache_data(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """Cache data with appropriate TTL"""
        try:
            if not self.cache_manager:
                return False
            
            # Determine TTL based on symbol popularity and market hours
            symbol = data.get("symbol", "")
            
            if not self.market_scheduler.is_market_hours():
                ttl = settings.cache_afterhours_ttl
            elif is_popular_ticker(symbol):
                ttl = settings.cache_popular_ttl
            else:
                ttl = settings.cache_ondemand_ttl
            
            # This would use your Redis cache manager
            # Implementation depends on your cache manager interface
            return True  # Placeholder
            
        except Exception as e:
            logger.error(f"❌ Cache set failed: {e}")
            return False

class MarketScheduler:
    """Helper class to manage market hours and scheduling"""
    
    def __init__(self):
        self.timezone = pytz.timezone('US/Eastern')
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        self.holidays = holidays.UnitedStates()
    
    def is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.timezone)
        
        if now.weekday() >= 5:  # Weekend
            return False
        
        if now.date() in self.holidays:  # Holiday
            return False
        
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close
    
    def next_market_open(self) -> datetime:
        """Calculate next market opening time"""
        now = datetime.now(self.timezone)
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if now.time() > self.market_close:
            next_open = next_open + timedelta(days=1)
        
        while (next_open.weekday() >= 5 or next_open.date() in self.holidays):
            next_open = next_open + timedelta(days=1)
        
        return next_open
