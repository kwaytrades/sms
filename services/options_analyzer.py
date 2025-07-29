# services/options_analyzer.py
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import math

logger = logging.getLogger(__name__)

class OptionsAnalyzer:
    """Complete options analysis service with EODHD integration"""
    
    def __init__(self, eodhd_api_key: str, redis_client=None):
        self.api_key = eodhd_api_key
        self.redis = redis_client
        self.base_url = "https://eodhd.com/api"
        
    async def analyze_options_chain(self, symbol: str, user_profile: dict = None) -> Dict[str, Any]:
        """Complete options chain analysis with personalized insights"""
        try:
            # Get options data
            options_data = await self._fetch_options_chain(symbol)
            if not options_data:
                return {"error": "Options data not available"}
            
            # Perform comprehensive analysis
            analysis_tasks = [
                self._analyze_unusual_activity(options_data),
                self._calculate_sentiment_indicators(options_data),
                self._identify_key_levels(options_data),
                self._generate_strategy_suggestions(options_data, user_profile)
            ]
            
            unusual_activity, sentiment, key_levels, strategies = await asyncio.gather(*analysis_tasks)
            
            # Get current stock price for context
            stock_price = await self._get_current_price(symbol)
            
            return {
                "symbol": symbol,
                "current_price": stock_price,
                "analysis_time": datetime.now().isoformat(),
                "unusual_activity": unusual_activity,
                "sentiment_indicators": sentiment,
                "key_levels": key_levels,
                "suggested_strategies": strategies,
                "options_flow": await self._analyze_options_flow(options_data),
                "gamma_exposure": await self._calculate_gamma_exposure(options_data),
                "summary": self._generate_options_summary(unusual_activity, sentiment, key_levels, user_profile)
            }
            
        except Exception as e:
            logger.error(f"❌ Options analysis error for {symbol}: {e}")
            return {"error": f"Options analysis failed: {str(e)}"}
    
    async def _fetch_options_chain(self, symbol: str) -> Dict:
        """Fetch complete options chain from EODHD"""
        cache_key = f"options_chain:{symbol}"
        
        # Check cache (5-minute TTL for options)
        if self.redis:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return eval(cached_data)  # In production, use json.loads
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get options chain
                options_url = f"{self.base_url}/options/{symbol}.US"
                params = {
                    "api_token": self.api_key,
                    "fmt": "json"
                }
                
                async with session.get(options_url, params=params) as response:
                    if response.status == 200:
                        options_data = await response.json()
                        
                        # Cache the result
                        if self.redis:
                            await self.redis.setex(cache_key, 300, str(options_data))
                        
                        return options_data
                    else:
                        logger.error(f"❌ EODHD options API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Error fetching options for {symbol}: {e}")
            return None
    
    async def _analyze_unusual_activity(self, options_data: Dict) -> Dict:
        """Detect unusual options activity patterns"""
        try:
            unusual_calls = []
            unusual_puts = []
            
            if 'data' not in options_data:
                return {"calls": [], "puts": [], "total_unusual": 0}
            
            for option in options_data['data']:
                if not option.get('volume') or not option.get('openInterest'):
                    continue
                
                volume = option.get('volume', 0)
                open_interest = option.get('openInterest', 1)
                vol_oi_ratio = volume / max(open_interest, 1)
                
                # Flag unusual activity (volume > 2x open interest)
                if vol_oi_ratio > 2.0 and volume > 100:
                    option_info = {
                        "strike": option.get('strike'),
                        "expiration": option.get('expirationDate'),
                        "type": option.get('type'),
                        "volume": volume,
                        "open_interest": open_interest,
                        "vol_oi_ratio": round(vol_oi_ratio, 2),
                        "premium": option.get('lastPrice', 0)
                    }
                    
                    if option.get('type') == 'call':
                        unusual_calls.append(option_info)
                    else:
                        unusual_puts.append(option_info)
            
            # Sort by volume
            unusual_calls.sort(key=lambda x: x['volume'], reverse=True)
            unusual_puts.sort(key=lambda x: x['volume'], reverse=True)
            
            return {
                "calls": unusual_calls[:5],  # Top 5
                "puts": unusual_puts[:5],    # Top 5
                "total_unusual": len(unusual_calls) + len(unusual_puts)
            }
            
        except Exception as e:
            logger.error(f"❌ Unusual activity analysis error: {e}")
            return {"calls": [], "puts": [], "total_unusual": 0}
    
    async def _calculate_sentiment_indicators(self, options_data: Dict) -> Dict:
        """Calculate put/call ratio and other sentiment indicators"""
        try:
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0
            total_put_oi = 0
            
            if 'data' not in options_data:
                return {"put_call_ratio": 0, "sentiment": "neutral"}
            
            for option in options_data['data']:
                volume = option.get('volume', 0)
                oi = option.get('openInterest', 0)
                
                if option.get('type') == 'call':
                    total_call_volume += volume
                    total_call_oi += oi
                else:
                    total_put_volume += volume
                    total_put_oi += oi
            
            # Calculate ratios
            pc_volume_ratio = total_put_volume / max(total_call_volume, 1)
            pc_oi_ratio = total_put_oi / max(total_call_oi, 1)
            
            # Determine sentiment
            if pc_volume_ratio < 0.7:
                sentiment = "bullish"
            elif pc_volume_ratio > 1.3:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return {
                "put_call_volume_ratio": round(pc_volume_ratio, 3),
                "put_call_oi_ratio": round(pc_oi_ratio, 3),
                "total_call_volume": total_call_volume,
                "total_put_volume": total_put_volume,
                "sentiment": sentiment
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment calculation error: {e}")
            return {"put_call_ratio": 0, "sentiment": "neutral"}
    
    async def _identify_key_levels(self, options_data: Dict) -> Dict:
        """Identify key support/resistance levels from options"""
        try:
            strike_volume = {}
            strike_oi = {}
            
            if 'data' not in options_data:
                return {"resistance_levels": [], "support_levels": []}
            
            # Aggregate volume and OI by strike
            for option in options_data['data']:
                strike = option.get('strike')
                if not strike:
                    continue
                
                volume = option.get('volume', 0)
                oi = option.get('openInterest', 0)
                
                if strike not in strike_volume:
                    strike_volume[strike] = 0
                    strike_oi[strike] = 0
                
                strike_volume[strike] += volume
                strike_oi[strike] += oi
            
            # Find high-volume/OI strikes
            significant_strikes = []
            for strike in strike_volume:
                total_activity = strike_volume[strike] + (strike_oi[strike] * 0.5)
                if total_activity > 500:  # Threshold for significance
                    significant_strikes.append({
                        "strike": strike,
                        "total_activity": total_activity,
                        "volume": strike_volume[strike],
                        "open_interest": strike_oi[strike]
                    })
            
            # Sort by activity
            significant_strikes.sort(key=lambda x: x['total_activity'], reverse=True)
            
            return {
                "key_levels": significant_strikes[:8],
                "max_pain": self._calculate_max_pain(options_data),
                "gamma_wall": self._find_gamma_wall(options_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Key levels analysis error: {e}")
            return {"resistance_levels": [], "support_levels": []}
    
    async def _generate_strategy_suggestions(self, options_data: Dict, user_profile: dict = None) -> List[Dict]:
        """Generate personalized options strategy suggestions"""
        try:
            strategies = []
            
            # Default risk tolerance
            risk_tolerance = user_profile.get('risk_tolerance', 'medium') if user_profile else 'medium'
            experience = user_profile.get('trading_experience', 'intermediate') if user_profile else 'intermediate'
            
            # Basic strategies based on profile
            if experience == 'beginner':
                strategies.extend([
                    {
                        "name": "Covered Call",
                        "description": "Generate income on existing stock position",
                        "risk_level": "low",
                        "complexity": "beginner"
                    },
                    {
                        "name": "Cash Secured Put",
                        "description": "Generate income while waiting to buy stock",
                        "risk_level": "medium",
                        "complexity": "beginner"
                    }
                ])
            
            elif experience in ['intermediate', 'advanced']:
                strategies.extend([
                    {
                        "name": "Iron Condor",
                        "description": "Profit from low volatility",
                        "risk_level": "medium",
                        "complexity": "intermediate"
                    },
                    {
                        "name": "Straddle",
                        "description": "Profit from high volatility (either direction)",
                        "risk_level": "high",
                        "complexity": "intermediate"
                    }
                ])
            
            return strategies[:3]  # Limit to top 3 suggestions
            
        except Exception as e:
            logger.error(f"❌ Strategy suggestion error: {e}")
            return []
    
    async def _analyze_options_flow(self, options_data: Dict) -> Dict:
        """Analyze options flow for directional bias"""
        try:
            flow_analysis = {
                "bullish_flow": 0,
                "bearish_flow": 0,
                "large_trades": [],
                "unusual_spreads": []
            }
            
            if 'data' not in options_data:
                return flow_analysis
            
            for option in options_data['data']:
                volume = option.get('volume', 0)
                if volume > 1000:  # Large trade threshold
                    trade_info = {
                        "strike": option.get('strike'),
                        "type": option.get('type'),
                        "volume": volume,
                        "premium": option.get('lastPrice', 0)
                    }
                    flow_analysis["large_trades"].append(trade_info)
                    
                    # Classify as bullish/bearish
                    if option.get('type') == 'call':
                        flow_analysis["bullish_flow"] += volume
                    else:
                        flow_analysis["bearish_flow"] += volume
            
            return flow_analysis
            
        except Exception as e:
            logger.error(f"❌ Options flow analysis error: {e}")
            return {"bullish_flow": 0, "bearish_flow": 0, "large_trades": []}
    
    async def _calculate_gamma_exposure(self, options_data: Dict) -> Dict:
        """Calculate gamma exposure levels (simplified)"""
        try:
            # This is a simplified gamma calculation
            # In production, you'd need more sophisticated Greeks calculation
            
            total_gamma_exposure = 0
            gamma_by_strike = {}
            
            if 'data' not in options_data:
                return {"total_gamma": 0, "gamma_levels": []}
            
            for option in options_data['data']:
                # Simplified gamma approximation
                # Real implementation would use Black-Scholes
                strike = option.get('strike', 0)
                volume = option.get('volume', 0)
                
                if strike and volume:
                    # Approximate gamma (peaks at ATM)
                    approx_gamma = volume * 0.01  # Simplified
                    total_gamma_exposure += approx_gamma
                    
                    if strike in gamma_by_strike:
                        gamma_by_strike[strike] += approx_gamma
                    else:
                        gamma_by_strike[strike] = approx_gamma
            
            return {
                "total_gamma": round(total_gamma_exposure, 2),
                "gamma_by_strike": dict(sorted(gamma_by_strike.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
        except Exception as e:
            logger.error(f"❌ Gamma exposure calculation error: {e}")
            return {"total_gamma": 0, "gamma_levels": []}
    
    def _generate_options_summary(self, unusual_activity: Dict, sentiment: Dict, key_levels: Dict, user_profile: dict = None) -> str:
        """Generate human-readable options summary"""
        try:
            summary_parts = []
            
            # Sentiment
            sentiment_text = sentiment.get('sentiment', 'neutral').title()
            pc_ratio = sentiment.get('put_call_volume_ratio', 0)
            summary_parts.append(f"Options sentiment: {sentiment_text} (P/C: {pc_ratio})")
            
            # Unusual activity
            total_unusual = unusual_activity.get('total_unusual', 0)
            if total_unusual > 0:
                summary_parts.append(f"{total_unusual} contracts with unusual activity")
            
            # Key levels
            key_strikes = key_levels.get('key_levels', [])
            if key_strikes:
                top_strike = key_strikes[0]['strike']
                summary_parts.append(f"Heavy activity at ${top_strike} strike")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"❌ Summary generation error: {e}")
            return "Options analysis completed"
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current stock price"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/real-time/{symbol}.US"
                params = {"api_token": self.api_key, "fmt": "json"}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('close', 0)
                    return 0
        except:
            return 0
    
    def _calculate_max_pain(self, options_data: Dict) -> float:
        """Calculate max pain level (simplified)"""
        # Simplified max pain calculation
        # Real implementation would be more sophisticated
        try:
            if 'data' not in options_data:
                return 0
                
            strikes = {}
            for option in options_data['data']:
                strike = option.get('strike')
                oi = option.get('openInterest', 0)
                if strike and oi:
                    if strike not in strikes:
                        strikes[strike] = 0
                    strikes[strike] += oi
            
            if strikes:
                max_oi_strike = max(strikes.items(), key=lambda x: x[1])
                return max_oi_strike[0]
            return 0
        except:
            return 0
    
    def _find_gamma_wall(self, options_data: Dict) -> float:
        """Find potential gamma wall (simplified)"""
        # Simplified gamma wall detection
        try:
            call_strikes = {}
            if 'data' not in options_data:
                return 0
                
            for option in options_data['data']:
                if option.get('type') == 'call':
                    strike = option.get('strike')
                    oi = option.get('openInterest', 0)
                    if strike and oi:
                        if strike not in call_strikes:
                            call_strikes[strike] = 0
                        call_strikes[strike] += oi
            
            if call_strikes:
                gamma_wall = max(call_strikes.items(), key=lambda x: x[1])
                return gamma_wall[0]
            return 0
        except:
            return 0
