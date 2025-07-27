# Alternative: Keep using direct API calls (more reliable)

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from loguru import logger
import os

class TechnicalAnalysisService:
    def __init__(self):
        self.eodhd_api_key = os.getenv('EODHD_API_KEY')
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info(f"✅ TA Service initialized with API key: {'Set' if self.eodhd_api_key else 'Not Set'}")
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Main analysis function using direct EODHD API"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_ttl:
                    logger.info(f"📈 TA Cache HIT for {symbol}")
                    return cached_data['data']
            
            logger.info(f"📈 TA Cache MISS for {symbol} - fetching fresh data")
            
            # Fetch data using direct API
            market_data = self._fetch_market_data_direct(symbol)
            
            if market_data is None or market_data.empty:
                logger.error(f"❌ No market data for {symbol}")
                return {
                    'symbol': symbol,
                    'error': 'No market data available',
                    'message': f'Unable to fetch data for {symbol}. Symbol may not be available or markets may be closed.',
                    'source': 'eodhd_direct_api',
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
                'message': f'Technical error analyzing {symbol}. Please try again.',
                'source': 'eodhd_direct_api',
                'cache_status': 'error'
            }
    
    def _fetch_market_data_direct(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data using direct EODHD API calls"""
        if not self.eodhd_api_key:
            logger.error("EODHD API key not available")
            return None
        
        try:
            # Calculate date range (last 30 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Add .US suffix if not present
            if '.' not in symbol:
                symbol_with_exchange = f"{symbol}.US"
            else:
                symbol_with_exchange = symbol
            
            logger.info(f"🔍 Fetching EODHD data for {symbol_with_exchange}")
            
            # Direct API call
            url = f"https://eodhd.com/api/eod/{symbol_with_exchange}"
            params = {
                'api_token': self.eodhd_api_key,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"EODHD API error {response.status_code} for {symbol}: {response.text}")
                return None
            
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response for {symbol}: {e}")
                return None
            
            if not data or not isinstance(data, list):
                logger.warning(f"No valid data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Check required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                # Try with adjusted_close
                if 'adjusted_close' in df.columns:
                    df['close'] = df['adjusted_close']
                else:
                    logger.error(f"Missing required columns for {symbol}. Available: {df.columns.tolist()}")
                    return None
            
            # Process DataFrame
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove NaN rows
            df = df.dropna()
            
            if df.empty:
                logger.error(f"No valid data after cleaning for {symbol}")
                return None
            
            logger.info(f"✅ Fetched {len(df)} days of data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error fetching data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching data for {symbol}: {e}")
            return None
    
    # ... rest of the methods stay the same (RSI, MACD, etc.)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate technical indicators from price data"""
        try:
            if df.empty or len(df) < 14:
                return {
                    'symbol': symbol,
                    'error': 'Insufficient data for technical analysis',
                    'source': 'eodhd_direct_api'
                }
            
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            # Calculate indicators
            rsi = self._calculate_rsi(df['close'])
            macd_data = self._calculate_macd(df['close'])
            
            # Moving averages
            ma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            # Price change
            price_change = latest['close'] - previous['close']
            price_change_pct = (price_change / previous['close']) * 100
            
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
                'source': 'eodhd_direct_api',
                'cache_status': 'fresh',
                'timestamp': datetime.now().isoformat(),
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"❌ Technical calculation failed: {e}")
            return {
                'symbol': symbol,
                'error': f'Technical calculation failed: {str(e)}',
                'source': 'eodhd_direct_api'
            }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI"""
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
    
    async def close(self):
        """Cleanup method"""
        self.cache.clear()
        logger.info("✅ Technical Analysis service closed")
