# services/technical_analysis.py - CORRECTED VERSION

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from loguru import logger
import os

# Import EODHD official library
try:
    from eodhd import APIClient
    EODHD_AVAILABLE = True
except ImportError:
    EODHD_AVAILABLE = False
    logger.warning("EODHD library not installed. Install with: pip install eodhd")

class TechnicalAnalysisService:
    def __init__(self):
        self.eodhd_api_key = os.getenv('EODHD_API_KEY')
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize EODHD client
        if EODHD_AVAILABLE and self.eodhd_api_key:
            try:
                self.api_client = APIClient(self.eodhd_api_key)
                logger.info("✅ EODHD API client initialized")
            except Exception as e:
                logger.error(f"❌ EODHD client initialization failed: {e}")
                self.api_client = None
        else:
            self.api_client = None
            logger.warning("❌ EODHD API client not available")
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Main analysis function using EODHD library"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_ttl:
                    logger.info(f"📈 TA Cache HIT for {symbol}")
                    return cached_data['data']
            
            logger.info(f"📈 TA Cache MISS for {symbol} - fetching fresh data")
            
            # Fetch data using EODHD library (FIXED: removed await)
            market_data = self._fetch_market_data_eodhd(symbol)
            
            if market_data is None or market_data.empty:
                logger.error(f"❌ No market data for {symbol}")
                return {
                    'symbol': symbol,
                    'error': 'No market data available',
                    'message': f'Unable to fetch data for {symbol}. Please verify the symbol.',
                    'source': 'eodhd_library',
                    'cache_status': 'miss'
                }
            
            # Calculate technical indicators
            analysis = self._calculate_technical_indicators(market_data, symbol)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': analysis,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ TA analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ TA analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'message': f'Analysis failed for {symbol}. Please try again.',
                'source': 'eodhd_library',
                'cache_status': 'error'
            }
    
    def _fetch_market_data_eodhd(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data using EODHD Python library (SYNCHRONOUS)"""
        if not self.api_client:
            logger.error("EODHD API client not available")
            return None
        
        try:
            # Calculate date range (last 30 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Add .US suffix if not present (EODHD format)
            if '.' not in symbol:
                symbol_with_exchange = f"{symbol}.US"
            else:
                symbol_with_exchange = symbol
            
            logger.info(f"🔍 Fetching EODHD data for {symbol_with_exchange}")
            
            # Use EODHD library to get historical data (SYNCHRONOUS CALL)
            data = self.api_client.get_eod_historical_stock_market_data(
                symbol=symbol_with_exchange,
                period='d',  # daily
                from_date=start_date,
                to_date=end_date,
                order='a'  # ascending
            )
            
            if not data:
                logger.warning(f"No data returned for {symbol_with_exchange}")
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            
            # FIXED: Check for correct column names that EODHD actually returns
            logger.info(f"EODHD returned columns: {df.columns.tolist()}")
            
            # EODHD typically returns: ['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns for {symbol}: {missing_columns}")
                logger.info(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Set date as index and convert to datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            if df.empty:
                logger.error(f"No valid data after cleaning for {symbol}")
                return None
            
            logger.info(f"✅ Fetched {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ EODHD data fetch failed for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate technical indicators from price data"""
        try:
            if df.empty or len(df) < 14:
                return {
                    'symbol': symbol,
                    'error': 'Insufficient data for technical analysis',
                    'message': f'Need at least 14 days of data for {symbol}. Got {len(df)} days.',
                    'source': 'eodhd_library'
                }
            
            # Current price info
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            # Calculate RSI
            rsi = self._calculate_rsi(df['close'])
            
            # Calculate MACD
            macd_data = self._calculate_macd(df['close'])
            
            # Calculate moving averages
            ma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            # Price change
            price_change = latest['close'] - previous['close']
            price_change_pct = (price_change / previous['close']) * 100
            
            # Generate trading signal
            signal = self._generate_trading_signal(rsi, macd_data, latest['close'], ma_20, ma_50)
            
            return {
                'symbol': symbol,
                'price': {
                    'current': round(float(latest['close']), 2),
                    'open': round(float(latest['open']), 2),
                    'high': round(float(latest['high']), 2),
                    'low': round(float(latest['low']), 2),
                    'volume': int(latest['volume']) if not pd.isna(latest['volume']) else 0,
                    'change': round(float(price_change), 2),
                    'change_percent': round(float(price_change_pct), 2)
                },
                'technical_indicators': {
                    'rsi': round(float(rsi), 2) if not pd.isna(rsi) else None,
                    'macd': {
                        'macd': round(float(macd_data['macd']), 2) if not pd.isna(macd_data['macd']) else None,
                        'signal': round(float(macd_data['signal']), 2) if not pd.isna(macd_data['signal']) else None,
                        'histogram': round(float(macd_data['histogram']), 2) if not pd.isna(macd_data['histogram']) else None
                    },
                    'moving_averages': {
                        'ma_20': round(float(ma_20), 2) if ma_20 and not pd.isna(ma_20) else None,
                        'ma_50': round(float(ma_50), 2) if ma_50 and not pd.isna(ma_50) else None
                    }
                },
                'trading_signal': signal,
                'source': 'eodhd_library',
                'cache_status': 'fresh',
                'timestamp': datetime.now().isoformat(),
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"❌ Technical calculation failed: {e}")
            return {
                'symbol': symbol,
                'error': f'Technical calculation failed: {str(e)}',
                'source': 'eodhd_library'
            }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return np.nan
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal_line.iloc[-1], 
                'histogram': histogram.iloc[-1]
            }
        except:
            return {
                'macd': np.nan,
                'signal': np.nan,
                'histogram': np.nan
            }
    
    def _generate_trading_signal(self, rsi: float, macd: Dict, current_price: float, ma_20: float, ma_50: float) -> Dict[str, Any]:
        """Generate basic trading signal"""
        try:
            signals = []
            overall_signal = "neutral"
            
            # RSI signals
            if not pd.isna(rsi):
                if rsi > 70:
                    signals.append("RSI overbought (>70)")
                elif rsi < 30:
                    signals.append("RSI oversold (<30)")
            
            # MACD signals
            if not pd.isna(macd['histogram']):
                if macd['histogram'] > 0:
                    signals.append("MACD bullish")
                else:
                    signals.append("MACD bearish")
            
            # Moving average signals
            if ma_20 and ma_50 and not pd.isna(ma_20) and not pd.isna(ma_50):
                if ma_20 > ma_50:
                    signals.append("MA bullish trend")
                else:
                    signals.append("MA bearish trend")
            
            # Determine overall signal
            bullish_count = sum(1 for s in signals if 'bullish' in s or 'oversold' in s)
            bearish_count = sum(1 for s in signals if 'bearish' in s or 'overbought' in s)
            
            if bullish_count > bearish_count:
                overall_signal = "bullish"
            elif bearish_count > bullish_count:
                overall_signal = "bearish"
            
            return {
                "overall": overall_signal,
                "signals": signals,
                "strength": abs(bullish_count - bearish_count)
            }
        except:
            return {
                "overall": "neutral",
                "signals": ["Unable to generate signals"],
                "strength": 0
            }
    
    async def close(self):
        """Cleanup method"""
        self.cache.clear()
        logger.info("✅ Technical Analysis service closed")
