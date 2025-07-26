# services/technical_analysis.py - Integrated Technical Analysis Service
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import json
import os

class TechnicalAnalysisService:
    def __init__(self):
        self.eodhd_api_key = os.getenv('EODHD_API_KEY')
        self.session = None
        self.cache_manager = None
        
        if not self.eodhd_api_key:
            logger.warning("⚠️ EODHD_API_KEY not found - technical analysis will use mock data")
    
    async def initialize(self, cache_manager=None):
        """Initialize with cache manager"""
        self.cache_manager = cache_manager
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        logger.info("✅ Technical Analysis Service initialized")
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
    
    async def analyze_symbol(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Complete technical analysis for a symbol with caching"""
        try:
            symbol = symbol.upper().strip()
            
            # Check cache first
            cache_key = f"ta_analysis:{symbol}:{period}"
            cached_result = await self._get_cached_data(cache_key)
            
            if cached_result:
                cached_result["cache_status"] = "hit"
                cached_result["source"] = "cache"
                logger.info(f"📈 TA Cache HIT for {symbol}")
                return cached_result
            
            logger.info(f"📈 TA Cache MISS for {symbol} - fetching fresh data")
            
            # Fetch fresh market data
            df = await self._fetch_market_data(symbol, period)
            
            if df is None or df.empty:
                return await self._get_fallback_data(symbol)
            
            # Calculate all technical indicators
            result = await self._calculate_technical_indicators(symbol, df)
            
            # Cache the result
            await self._cache_result(cache_key, result, symbol)
            
            result["cache_status"] = "miss"
            result["source"] = "fresh"
            result["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"✅ TA analysis completed for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"❌ TA analysis failed for {symbol}: {e}")
            return await self._get_fallback_data(symbol)
    
    async def _fetch_market_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch market data from EODHD API"""
        if not self.eodhd_api_key:
            logger.warning(f"No EODHD API key - using mock data for {symbol}")
            return self._generate_mock_data(symbol)
        
        try:
            # Convert period to EODHD format
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            if period == "1mo":
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif period == "3mo":
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            elif period == "6mo":
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            else:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            url = f"https://eodhd.com/api/eod/{symbol}.US"
            params = {
                'api_token': self.eodhd_api_key,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if not data:
                        logger.warning(f"No data returned for {symbol}")
                        return None
                    
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(data)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df = df.astype(float)
                    
                    logger.info(f"✅ Fetched {len(df)} days of data for {symbol}")
                    return df
                else:
                    logger.error(f"EODHD API error {response.status} for {symbol}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _generate_mock_data(self, symbol: str) -> pd.DataFrame:
        """Generate realistic mock data for testing"""
        # Base prices for popular stocks
        base_prices = {
            'AAPL': 185, 'TSLA': 245, 'MSFT': 420, 'GOOGL': 140,
            'AMZN': 155, 'NVDA': 875, 'META': 485, 'NFLX': 485
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # Generate 30 days of mock data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Random walk with trend
        returns = np.random.normal(0.001, 0.02, 30)  # Slight upward bias
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.01, 0.03)
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = close * (1 + np.random.uniform(-0.005, 0.005))
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Adjusted_close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        logger.info(f"Generated mock data for {symbol}")
        return df
    
    async def _calculate_technical_indicators(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        
        # Current price info
        current_price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
        
        # Technical indicators
        technical = {}
        
        # RSI (14-period)
        technical['rsi'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(df['Close'])
        technical['macd'] = {
            'line': float(macd_line.iloc[-1]) if not macd_line.empty else 0,
            'signal': float(macd_signal.iloc[-1]) if not macd_signal.empty else 0,
            'histogram': float(macd_histogram.iloc[-1]) if not macd_histogram.empty else 0
        }
        
        # Moving averages
        technical['sma_20'] = float(df['Close'].rolling(20).mean().iloc[-1])
        technical['sma_50'] = float(df['Close'].rolling(50).mean().iloc[-1]) if len(df) >= 50 else technical['sma_20']
        technical['ema_12'] = float(df['Close'].ewm(span=12).mean().iloc[-1])
        technical['ema_26'] = float(df['Close'].ewm(span=26).mean().iloc[-1])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'])
        technical['bollinger'] = {
            'upper': float(bb_upper.iloc[-1]),
            'middle': float(bb_middle.iloc[-1]),
            'lower': float(bb_lower.iloc[-1])
        }
        
        # Support and Resistance
        support, resistance = self._calculate_support_resistance(df)
        technical['support'] = support
        technical['resistance'] = resistance
        
        # Volume analysis
        avg_volume = int(df['Volume'].rolling(20).mean().iloc[-1])
        current_volume = int(df['Volume'].iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Generate trading signals
        signals = self._generate_signals(df, technical)
        
        return {
            'symbol': symbol,
            'price': {
                'current': current_price,
                'change': change,
                'change_percent': change_percent,
                'previous_close': prev_close
            },
            'technical_indicators': technical,
            'volume': {
                'current': current_volume,
                'average': avg_volume,
                'ratio': volume_ratio
            },
            'signals': signals,
            'data_points': len(df),
            'is_popular': symbol in ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI for insufficient data
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            # Return zeros for insufficient data
            return pd.Series([0]), pd.Series([0]), pd.Series([0])
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices.iloc[-1]
            return pd.Series([current_price * 1.02]), pd.Series([current_price]), pd.Series([current_price * 0.98])
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        if len(df) < 5:
            current_price = df['Close'].iloc[-1]
            return float(current_price * 0.95), float(current_price * 1.05)
        
        # Simple method: recent lows and highs
        recent_data = df.tail(20)  # Last 20 days
        
        support = float(recent_data['Low'].min())
        resistance = float(recent_data['High'].max())
        
        return support, resistance
    
    def _generate_signals(self, df: pd.DataFrame, technical: Dict) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators"""
        signals = {
            'overall': 'neutral',
            'strength': 0,
            'reasons': []
        }
        
        score = 0
        reasons = []
        
        # RSI signals
        rsi = technical.get('rsi', 50)
        if rsi > 70:
            score -= 1
            reasons.append('RSI overbought (>70)')
        elif rsi < 30:
            score += 1
            reasons.append('RSI oversold (<30)')
        elif 40 <= rsi <= 60:
            reasons.append('RSI neutral')
        
        # MACD signals
        macd = technical.get('macd', {})
        if macd.get('line', 0) > macd.get('signal', 0):
            score += 1
            reasons.append('MACD bullish crossover')
        elif macd.get('line', 0) < macd.get('signal', 0):
            score -= 1
            reasons.append('MACD bearish crossover')
        
        # Moving average signals
        current_price = df['Close'].iloc[-1]
        sma_20 = technical.get('sma_20', current_price)
        
        if current_price > sma_20:
            score += 1
            reasons.append('Price above 20-day SMA')
        else:
            score -= 1
            reasons.append('Price below 20-day SMA')
        
        # Determine overall signal
        if score >= 2:
            signals['overall'] = 'bullish'
        elif score <= -2:
            signals['overall'] = 'bearish'
        else:
            signals['overall'] = 'neutral'
        
        signals['strength'] = abs(score)
        signals['reasons'] = reasons
        
        return signals
    
    async def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Get data from cache if available"""
        if not self.cache_manager:
            return None
        
        try:
            # Simple in-memory cache fallback
            if not hasattr(self, '_memory_cache'):
                self._memory_cache = {}
            
            if cache_key in self._memory_cache:
                cached_data, timestamp = self._memory_cache[cache_key]
                # Check if cache is still valid (5 minutes for now)
                if datetime.now() - timestamp < timedelta(minutes=5):
                    return cached_data
                else:
                    del self._memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: Dict, symbol: str):
        """Cache the analysis result"""
        try:
            if not hasattr(self, '_memory_cache'):
                self._memory_cache = {}
            
            # Simple in-memory cache
            self._memory_cache[cache_key] = (result.copy(), datetime.now())
            
            # Keep cache size manageable
            if len(self._memory_cache) > 100:
                # Remove oldest entries
                oldest_key = min(self._memory_cache.keys(), key=lambda k: self._memory_cache[k][1])
                del self._memory_cache[oldest_key]
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Return fallback data when analysis fails"""
        # Basic fallback data with mock values
        fallback_prices = {
            'AAPL': 185.50, 'TSLA': 242.75, 'MSFT': 420.00, 'GOOGL': 140.00,
            'AMZN': 155.00, 'NVDA': 875.00, 'META': 485.00, 'NFLX': 485.00
        }
        
        price = fallback_prices.get(symbol, 100.00)
        
        return {
            'symbol': symbol,
            'price': {
                'current': price,
                'change': 0.50,
                'change_percent': 0.5,
                'previous_close': price - 0.50
            },
            'technical_indicators': {
                'rsi': 65.0,
                'macd': {'line': 0.1, 'signal': 0.05, 'histogram': 0.05},
                'sma_20': price * 0.99,
                'sma_50': price * 0.98,
                'ema_12': price * 1.01,
                'ema_26': price * 0.99,
                'bollinger': {
                    'upper': price * 1.05,
                    'middle': price,
                    'lower': price * 0.95
                },
                'support': price * 0.95,
                'resistance': price * 1.05
            },
            'volume': {
                'current': 1500000,
                'average': 1200000,
                'ratio': 1.25
            },
            'signals': {
                'overall': 'neutral',
                'strength': 1,
                'reasons': ['Using fallback data - limited analysis']
            },
            'source': 'fallback',
            'cache_status': 'miss',
            'timestamp': datetime.now().isoformat(),
            'data_points': 30,
            'is_popular': symbol in ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
        }
