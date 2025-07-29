# services/fundamental_analysis.py - MODIFIED for cache-first hybrid approach (RAW DATA ONLY)

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisDepth(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

class FinancialHealth(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DISTRESSED = "distressed"

@dataclass
class FinancialRatios:
    """Core financial ratios structure"""
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None

@dataclass
class GrowthMetrics:
    """Growth and trend analysis"""
    revenue_growth_1y: Optional[float] = None
    revenue_growth_3y: Optional[float] = None
    earnings_growth_1y: Optional[float] = None
    eps_growth_1y: Optional[float] = None

@dataclass
class FundamentalAnalysisResult:
    """Complete fundamental analysis result - RAW DATA ONLY"""
    symbol: str
    analysis_timestamp: datetime
    current_price: float
    ratios: FinancialRatios = None
    growth: GrowthMetrics = None
    financial_health: FinancialHealth = FinancialHealth.FAIR
    overall_score: float = 50.0
    strength_areas: List[str] = None
    concern_areas: List[str] = None
    bull_case: str = "Analysis pending"
    bear_case: str = "Analysis pending"
    data_completeness: float = 0.0
    last_quarter_date: Optional[str] = None
    data_source: str = "unknown"
    response_time_ms: float = 0.0
    cache_hit: bool = False
    
    def __post_init__(self):
        """Initialize default values after creation"""
        if self.ratios is None:
            self.ratios = FinancialRatios()
        if self.growth is None:
            self.growth = GrowthMetrics()
        if self.strength_areas is None:
            self.strength_areas = []
        if self.concern_areas is None:
            self.concern_areas = []

class HybridFundamentalAnalysisService:
    """
    Hybrid Fundamental Analysis Service - Cache First with Live API Fallback:
    1. Check background cache (MongoDB/Redis) - <100ms
    2. If cache miss, use live EODHD API - 3-5s
    3. Cache the live result for future requests
    4. Returns RAW fundamental data only (no SMS formatting)
    """
    
    def __init__(self, mongodb_url: str = None, redis_url: str = None):
        # API configuration
        self.eodhd_api_key = os.getenv('EODHD_API_KEY')
        self.base_url = "https://eodhd.com/api"
        
        # Background cache connections
        self.mongodb_url = mongodb_url or os.getenv('MONGODB_URL')
        self.redis_url = redis_url or os.getenv('REDIS_URL')
        self.mongo_client = None
        self.db = None
        self.redis_client = None
        
        # Memory cache (from your original logic)
        self.memory_cache = {}
        self.memory_cache_ttl = 7 * 24 * 3600  # 1 week
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "background_cache_hits": 0,
            "memory_cache_hits": 0,
            "live_api_calls": 0,
            "avg_response_time_ms": 0
        }
        
        logger.info("✅ Hybrid Fundamental Analysis Service initialized")
        logger.info(f"   EODHD API: {'Set' if self.eodhd_api_key else 'Not Set'}")
        logger.info(f"   Background Cache: {'Enabled' if self.mongodb_url else 'Disabled'}")
    
    async def initialize(self):
        """Initialize background cache connections"""
        try:
            if self.mongodb_url and self.redis_url:
                # MongoDB connection
                self.mongo_client = AsyncIOMotorClient(self.mongodb_url)
                self.db = self.mongo_client.trading_bot
                
                # Redis connection
                self.redis_client = aioredis.from_url(self.redis_url)
                
                # Test connections
                await self.db.command("ping")
                await self.redis_client.ping()
                
                logger.info("✅ Background cache connections established")
            else:
                logger.info("📡 Running in API-only mode (no background cache)")
                
        except Exception as e:
            logger.warning(f"⚠️ Background cache unavailable, using API-only mode: {e}")
            self.mongo_client = None
            self.db = None
            self.redis_client = None
    
    async def analyze(self, symbol: str, analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD) -> FundamentalAnalysisResult:
        """
        Main analysis method - Cache first, API fallback (RAW DATA ONLY)
        
        Order of operations:
        1. Check background cache (Redis/MongoDB) - ~50ms
        2. Check memory cache - ~1ms  
        3. Make live API call - ~3-5s
        4. Cache result in both systems
        """
        start_time = datetime.now()
        symbol = symbol.upper()
        
        try:
            self.stats["total_requests"] += 1
            
            # Step 1: Check background cache first (fastest for popular stocks)
            if self.redis_client or self.db:
                background_result = await self._get_from_background_cache(symbol)
                if background_result:
                    self.stats["background_cache_hits"] += 1
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_avg_response_time(response_time)
                    
                    logger.info(f"⚡ Background cache HIT for {symbol} in {response_time:.1f}ms")
                    
                    # Add metadata
                    background_result.data_source = "background_cache"
                    background_result.response_time_ms = round(response_time, 1)
                    background_result.cache_hit = True
                    
                    return background_result
            
            # Step 2: Check memory cache (existing logic)
            memory_result = self._get_from_memory_cache(symbol, analysis_depth)
            if memory_result:
                self.stats["memory_cache_hits"] += 1
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                logger.info(f"💾 Memory cache HIT for {symbol} in {response_time:.1f}ms")
                
                # Add metadata
                memory_result.data_source = "memory_cache"
                memory_result.response_time_ms = round(response_time, 1)
                memory_result.cache_hit = True
                
                return memory_result
            
            # Step 3: Cache miss - use live API (your existing logic)
            self.stats["live_api_calls"] += 1
            
            logger.info(f"📡 Cache MISS for {symbol} - fetching from live API")
            
            live_result = await self._analyze_with_live_api(symbol, analysis_depth)
            
            if live_result and not hasattr(live_result, 'error'):
                # Step 4: Cache the result in both systems
                await self._cache_live_result(symbol, live_result)
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                # Add metadata
                live_result.data_source = "live_api"
                live_result.response_time_ms = round(response_time, 1)
                live_result.cache_hit = False
                
                logger.info(f"✅ Live API success for {symbol} in {response_time:.1f}ms")
            else:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"❌ Live API failed for {symbol}")
            
            return live_result
            
        except Exception as e:
            logger.error(f"❌ Fundamental analysis failed for {symbol}: {e}")
            return self._create_minimal_result(symbol, f"Analysis error: {str(e)}")
    
    async def _get_from_background_cache(self, symbol: str) -> Optional[FundamentalAnalysisResult]:
        """Get fundamental data from background cache (Redis/MongoDB)"""
        try:
            # Try Redis first (fastest)
            if self.redis_client:
                redis_data = await self._get_redis_fundamental_data(symbol)
                if redis_data:
                    return self._format_background_cache_result(symbol, redis_data, "redis")
            
            # Try MongoDB (daily pipeline data)
            if self.db:
                mongodb_data = await self._get_mongodb_fundamental_data(symbol)
                if mongodb_data:
                    # Promote to Redis for next time
                    if self.redis_client:
                        await self._promote_to_redis_cache(symbol, mongodb_data)
                    
                    return self._format_background_cache_result(symbol, mongodb_data, "mongodb")
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Background cache error for {symbol}: {e}")
            return None
    
    async def _get_redis_fundamental_data(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data from Redis cache"""
        try:
            # Get fundamental data from Redis hash
            fundamental_data = await self.redis_client.hgetall(f"stock:{symbol}:fundamental")
            basic_data = await self.redis_client.hgetall(f"stock:{symbol}:basic")
            
            if fundamental_data and basic_data:
                # Convert Redis data back to proper format
                converted_fund = {}
                for key, value in fundamental_data.items():
                    key = key.decode() if isinstance(key, bytes) else key
                    value = value.decode() if isinstance(value, bytes) else value
                    
                    if key in ['pe', 'pb', 'roe', 'debt_to_equity', 'dividend_yield', 'eps', 
                              'revenue_growth', 'profit_margin', 'market_cap', 'beta']:
                        try:
                            converted_fund[key] = float(value)
                        except (ValueError, TypeError):
                            converted_fund[key] = value
                    else:
                        converted_fund[key] = value
                
                converted_basic = {}
                for key, value in basic_data.items():
                    key = key.decode() if isinstance(key, bytes) else key
                    value = value.decode() if isinstance(value, bytes) else value
                    
                    if key in ['price', 'market_cap']:
                        try:
                            converted_basic[key] = float(value)
                        except (ValueError, TypeError):
                            converted_basic[key] = value
                    else:
                        converted_basic[key] = value
                
                return {"fundamental": converted_fund, "basic": converted_basic}
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Redis get failed for {symbol}: {e}")
            return None
    
    async def _get_mongodb_fundamental_data(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data from MongoDB (daily pipeline)"""
        try:
            stock_data = await self.db.stocks.find_one(
                {"symbol": symbol},
                {"fundamental": 1, "basic": 1, "last_updated": 1}
            )
            
            if stock_data and stock_data.get("fundamental") and stock_data.get("basic"):
                # Check data freshness (within 2 days)
                last_updated = stock_data.get("last_updated", datetime.now())
                if isinstance(last_updated, str):
                    try:
                        last_updated = datetime.fromisoformat(last_updated)
                    except ValueError:
                        last_updated = datetime.now()
                
                age = datetime.now() - last_updated
                if age < timedelta(days=2):
                    return {
                        "fundamental": stock_data["fundamental"],
                        "basic": stock_data["basic"]
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB get failed for {symbol}: {e}")
            return None
    
    def _format_background_cache_result(self, symbol: str, cache_data: Dict, cache_source: str) -> FundamentalAnalysisResult:
        """Format background cache data to match analysis result format"""
        try:
            fundamental = cache_data.get("fundamental", {})
            basic = cache_data.get("basic", {})
            
            # Create ratios from cached data
            ratios = FinancialRatios(
                pe_ratio=fundamental.get("pe", None),
                pb_ratio=fundamental.get("pb", None),
                roe=fundamental.get("roe", None),
                debt_to_equity=fundamental.get("debt_to_equity", None),
                gross_margin=fundamental.get("profit_margin", None),  # Use profit_margin if available
                net_margin=fundamental.get("profit_margin", None)
            )
            
            # Create growth metrics from cached data
            growth = GrowthMetrics(
                revenue_growth_1y=fundamental.get("revenue_growth", None),
                eps_growth_1y=fundamental.get("eps_growth", None)
            )
            
            # Assess financial health from cached data
            financial_health = self._assess_health_from_cache(fundamental)
            
            # Calculate score from cached data
            overall_score = self._calculate_score_from_cache(fundamental)
            
            return FundamentalAnalysisResult(
                symbol=symbol,
                analysis_timestamp=datetime.now(),
                current_price=float(basic.get("price", 0)),
                ratios=ratios,
                growth=growth,
                financial_health=financial_health,
                overall_score=overall_score,
                strength_areas=["Cached analysis"],
                concern_areas=["Limited cache data"],
                bull_case="Based on cached fundamental metrics",
                bear_case="Cache data may be outdated",
                data_completeness=70.0,  # Assume reasonable completeness for cached data
                last_quarter_date="Unknown",
                data_source=f"background_cache_{cache_source}",
                cache_hit=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error formatting cache data for {symbol}: {e}")
            return self._create_minimal_result(symbol, "Cache formatting error")
    
    def _assess_health_from_cache(self, fundamental: Dict) -> FinancialHealth:
        """Assess financial health from cached data"""
        try:
            score = 0
            factors = 0
            
            # ROE assessment
            roe = fundamental.get("roe")
            if roe and roe > 0:
                factors += 1
                if roe > 20:
                    score += 4
                elif roe > 15:
                    score += 3
                elif roe > 10:
                    score += 2
                else:
                    score += 1
            
            # Debt assessment
            debt_to_equity = fundamental.get("debt_to_equity")
            if debt_to_equity is not None:
                factors += 1
                if debt_to_equity < 0.3:
                    score += 3
                elif debt_to_equity < 0.6:
                    score += 2
                elif debt_to_equity < 1.0:
                    score += 1
            
            # Growth assessment
            revenue_growth = fundamental.get("revenue_growth")
            if revenue_growth is not None:
                factors += 1
                if revenue_growth > 15:
                    score += 3
                elif revenue_growth > 5:
                    score += 2
                elif revenue_growth > 0:
                    score += 1
            
            # Calculate final health rating
            if factors == 0:
                return FinancialHealth.FAIR
            
            avg_score = score / factors
            
            if avg_score >= 3.5:
                return FinancialHealth.EXCELLENT
            elif avg_score >= 2.5:
                return FinancialHealth.GOOD
            elif avg_score >= 1.5:
                return FinancialHealth.FAIR
            elif avg_score >= 0.5:
                return FinancialHealth.POOR
            else:
                return FinancialHealth.DISTRESSED
                
        except Exception:
            return FinancialHealth.FAIR
    
    def _calculate_score_from_cache(self, fundamental: Dict) -> float:
        """Calculate score from cached fundamental data"""
        try:
            total_score = 0
            components = 0
            
            # ROE component
            roe = fundamental.get("roe")
            if roe and roe > 0:
                components += 1
                total_score += min(roe, 25)
            
            # Growth component
            revenue_growth = fundamental.get("revenue_growth")
            if revenue_growth is not None:
                components += 1
                growth_score = max(0, min(revenue_growth + 10, 25))
                total_score += growth_score
            
            # Margin component
            profit_margin = fundamental.get("profit_margin")
            if profit_margin and profit_margin > 0:
                components += 1
                total_score += min(profit_margin, 25)
            
            if components > 0:
                return min(total_score / components * 4, 100)
            else:
                return 50.0
                
        except Exception:
            return 50.0
    
    def _get_from_memory_cache(self, symbol: str, analysis_depth: AnalysisDepth) -> Optional[FundamentalAnalysisResult]:
        """Get from existing memory cache (your original logic)"""
        try:
            cache_key = f"fundamental_analysis:{symbol}:{analysis_depth.value}"
            
            if cache_key in self.memory_cache:
                cached_data = self.memory_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.memory_cache_ttl:
                    return cached_data['data']
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Memory cache error: {e}")
            return None
    
    async def _analyze_with_live_api(self, symbol: str, analysis_depth: AnalysisDepth) -> FundamentalAnalysisResult:
        """Use your existing live API analysis logic"""
        try:
            logger.info(f"🔍 Starting live fundamental analysis for {symbol}")
            
            # Fetch data with timeout and error handling (your existing logic)
            try:
                data = await asyncio.wait_for(
                    self._fetch_all_data(symbol), 
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Data fetch timeout for {symbol}")
                return self._create_minimal_result(symbol, "Data fetch timeout")
            except Exception as e:
                logger.error(f"❌ Data fetch failed for {symbol}: {e}")
                return self._create_minimal_result(symbol, "Data unavailable")
            
            # Perform analysis with the fetched data (your existing logic)
            result = await self._perform_robust_analysis(symbol, data)
            
            # Cache result in memory (your existing logic)
            cache_key = f"fundamental_analysis:{symbol}:{analysis_depth.value}"
            self.memory_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Live API analysis failed for {symbol}: {e}")
            return self._create_minimal_result(symbol, f"Analysis error: {str(e)}")
    
    async def _cache_live_result(self, symbol: str, result: FundamentalAnalysisResult):
        """Cache live API result in background cache for future requests"""
        try:
            if not result or hasattr(result, 'error'):
                return
            
            # Prepare cache data structure
            cache_data = {
                "symbol": symbol,
                "last_updated": datetime.now(),
                "basic": {
                    "price": result.current_price,
                    "market_cap": getattr(result.ratios, 'market_cap', 0) if result.ratios else 0,
                    "sector": "Unknown",
                    "exchange": "US"
                },
                "fundamental": {
                    "pe": result.ratios.pe_ratio if result.ratios else None,
                    "pb": result.ratios.pb_ratio if result.ratios else None,
                    "roe": result.ratios.roe if result.ratios else None,
                    "debt_to_equity": result.ratios.debt_to_equity if result.ratios else None,
                    "dividend_yield": 0,
                    "eps": 0,
                    "revenue_growth": result.growth.revenue_growth_1y if result.growth else None,
                    "profit_margin": result.ratios.net_margin if result.ratios else None,
                    "market_cap": result.current_price * 1000000,  # Estimate
                    "beta": 1.0
                },
                "api_cached": True,  # Mark as API-cached data
                "cache_source": "live_fundamental_api"
            }
            
            # Store in MongoDB
            if self.db:
                await self.db.stocks.replace_one(
                    {"symbol": symbol},
                    cache_data,
                    upsert=True
                )
            
            # Store in Redis
            if self.redis_client:
                await self._store_in_redis_cache(symbol, cache_data)
            
            logger.info(f"✅ Cached live fundamental result for {symbol}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to cache live fundamental result for {symbol}: {e}")
    
    async def _promote_to_redis_cache(self, symbol: str, mongodb_data: Dict):
        """Promote MongoDB data to Redis for faster future access"""
        try:
            if not self.redis_client:
                return
            
            await self._store_in_redis_cache(symbol, {
                "basic": mongodb_data.get("basic", {}),
                "fundamental": mongodb_data.get("fundamental", {})
            })
            
            logger.info(f"✅ Promoted {symbol} fundamental data to Redis cache")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis promotion failed for {symbol}: {e}")
    
    async def _store_in_redis_cache(self, symbol: str, cache_data: Dict):
        """Store data in Redis cache"""
        try:
            pipe = self.redis_client.pipeline()
            
            # Store basic data
            if "basic" in cache_data:
                basic_data = {k: str(v) for k, v in cache_data["basic"].items()}
                pipe.hset(f"stock:{symbol}:basic", mapping=basic_data)
                pipe.expire(f"stock:{symbol}:basic", 7200)  # 2 hours for API-cached data
            
            # Store fundamental data  
            if "fundamental" in cache_data:
                fund_data = {k: str(v) for k, v in cache_data["fundamental"].items() if v is not None}
                pipe.hset(f"stock:{symbol}:fundamental", mapping=fund_data)
                pipe.expire(f"stock:{symbol}:fundamental", 7200)  # 2 hours for API-cached data
            
            # Store timestamp
            pipe.set(f"stock:{symbol}:last_updated", datetime.now().isoformat())
            pipe.expire(f"stock:{symbol}:last_updated", 7200)
            
            await pipe.execute()
            
        except Exception as e:
            logger.warning(f"⚠️ Redis storage failed for {symbol}: {e}")
    
    def _update_avg_response_time(self, response_time_ms: float):
        """Update running average response time"""
        current_avg = self.stats["avg_response_time_ms"]
        total_requests = self.stats["total_requests"]
        
        self.stats["avg_response_time_ms"] = (
            (current_avg * (total_requests - 1) + response_time_ms) / total_requests
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        total_requests = self.stats["total_requests"]
        
        return {
            "service_mode": "hybrid_cache_first",
            "performance_stats": self.stats,
            "cache_efficiency": {
                "background_cache_hit_rate": round((self.stats["background_cache_hits"] / max(total_requests, 1)) * 100, 2),
                "memory_cache_hit_rate": round((self.stats["memory_cache_hits"] / max(total_requests, 1)) * 100, 2),
                "overall_cache_hit_rate": round(((self.stats["background_cache_hits"] + self.stats["memory_cache_hits"]) / max(total_requests, 1)) * 100, 2),
                "live_api_usage_rate": round((self.stats["live_api_calls"] / max(total_requests, 1)) * 100, 2)
            },
            "response_times": {
                "background_cache": "~50ms",
                "memory_cache": "~1ms", 
                "live_api": "~3-5s",
                "avg_response_time_ms": round(self.stats["avg_response_time_ms"], 1)
            },
            "capabilities": {
                "cached_stocks": "800+ popular stocks (instant)",
                "live_api_stocks": "Any stock symbol (3-5s)",
                "auto_caching": "Live results cached for future requests",
                "background_cache": "Enabled" if self.db else "Disabled"
            },
            "data_format": "raw_fundamental_data_only",
            "timestamp": datetime.now().isoformat()
        }
    
    # === YOUR EXISTING METHODS (unchanged) ===
    
    async def _fetch_all_data(self, symbol: str) -> Dict:
        """Your existing data fetching logic - UNCHANGED"""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
            try:
                # Fetch fundamentals data
                fundamentals_url = f"{self.base_url}/fundamentals/{symbol}?api_token={self.eodhd_api_key}"
                
                async with session.get(fundamentals_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Successfully fetched fundamentals for {symbol}")
                        return data
                    else:
                        logger.warning(f"⚠️ EODHD API returned status {response.status} for {symbol}")
                        return {}
                        
            except Exception as e:
                logger.error(f"❌ Error fetching data for {symbol}: {e}")
                return {}
    
    async def _perform_robust_analysis(self, symbol: str, data: Dict) -> FundamentalAnalysisResult:
        """Your existing analysis logic - UNCHANGED"""
        # Safe data extraction
        highlights = data.get("Highlights", {}) or {}
        valuation = data.get("Valuation", {}) or {}
        
        # Extract current price safely
        current_price = self._safe_float(highlights.get("MarketCapitalization", 0)) / max(self._safe_float(highlights.get("SharesOutstanding", 1)), 1)
        if not current_price:
            current_price = self._safe_float(data.get("General", {}).get("CurrPrice", 0))
        if not current_price:
            current_price = 100.0  # Fallback price for calculation purposes
        
        # Calculate ratios safely
        ratios = self._calculate_ratios_safe(highlights, valuation)
        
        # Calculate growth metrics safely  
        growth = self._calculate_growth_safe(data)
        
        # Assess financial health
        financial_health = self._assess_health_safe(ratios, growth)
        
        # Calculate composite score
        overall_score = self._calculate_score_safe(ratios, growth)
        
        # Identify strengths and concerns
        strengths, concerns = self._identify_areas_safe(ratios, growth)
        
        # Generate investment thesis
        bull_case, bear_case = self._generate_thesis_safe(ratios, growth, financial_health)
        
        # Calculate data completeness
        data_completeness = self._calculate_completeness_safe(highlights, valuation)
        
        return FundamentalAnalysisResult(
            symbol=symbol,
            analysis_timestamp=datetime.now(),
            current_price=current_price,
            ratios=ratios,
            growth=growth,
            financial_health=financial_health,
            overall_score=overall_score,
            strength_areas=strengths,
            concern_areas=concerns,
            bull_case=bull_case,
            bear_case=bear_case,
            data_completeness=data_completeness,
            last_quarter_date=self._extract_quarter_date_safe(data)
        )
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Your existing safe float conversion - UNCHANGED"""
        try:
            if value is None or value == "" or value == "None":
                return default
            if isinstance(value, str):
                # Handle percentage strings
                if value.endswith('%'):
                    return float(value[:-1])
                # Handle "N/A" or similar
                if value.lower() in ['n/a', 'na', '-', 'null']:
                    return default
            return float(value)
        except (TypeError, ValueError, AttributeError):
            return default
    
    def _calculate_ratios_safe(self, highlights: Dict, valuation: Dict) -> FinancialRatios:
        """Your existing ratios calculation - UNCHANGED"""
        return FinancialRatios(
            pe_ratio=self._safe_float(highlights.get("PERatio")),
            peg_ratio=self._safe_float(highlights.get("PEGRatio")),
            pb_ratio=self._safe_float(highlights.get("PriceBookMRQ")),
            ps_ratio=self._safe_float(highlights.get("PriceSalesTTM")),
            roe=self._safe_float(highlights.get("ReturnOnEquityTTM")),
            roa=self._safe_float(highlights.get("ReturnOnAssetsTTM")),
            gross_margin=self._safe_float(highlights.get("GrossProfitMargin")),
            operating_margin=self._safe_float(highlights.get("OperatingMarginTTM")),
            net_margin=self._safe_float(highlights.get("ProfitMargin")),
            current_ratio=self._safe_float(valuation.get("CurrentRatio")),
            debt_to_equity=self._safe_float(valuation.get("DebtEquityRatio"))
        )
    
    def _calculate_growth_safe(self, data: Dict) -> GrowthMetrics:
        """Your existing growth calculation - UNCHANGED"""
        highlights = data.get("Highlights", {}) or {}
        
        return GrowthMetrics(
            revenue_growth_1y=self._safe_float(highlights.get("RevenueGrowthTTM")),
            earnings_growth_1y=self._safe_float(highlights.get("EarningsGrowthTTM")),
            eps_growth_1y=self._safe_float(highlights.get("EPSGrowthTTM"))
        )
    
    def _assess_health_safe(self, ratios: FinancialRatios, growth: GrowthMetrics) -> FinancialHealth:
        """Your existing health assessment - UNCHANGED"""
        score = 0
        factors = 0
        
        # ROE assessment
        if ratios.roe and ratios.roe > 0:
            factors += 1
            if ratios.roe > 20:
                score += 4
            elif ratios.roe > 15:
                score += 3
            elif ratios.roe > 10:
                score += 2
            else:
                score += 1
        
        # Current ratio assessment
        if ratios.current_ratio and ratios.current_ratio > 0:
            factors += 1
            if ratios.current_ratio > 2:
                score += 3
            elif ratios.current_ratio > 1.5:
                score += 2
            elif ratios.current_ratio > 1:
                score += 1
        
        # Debt assessment
        if ratios.debt_to_equity is not None:
            factors += 1
            if ratios.debt_to_equity < 0.3:
                score += 3
            elif ratios.debt_to_equity < 0.6:
                score += 2
            elif ratios.debt_to_equity < 1.0:
                score += 1
        
        # Growth assessment
        if growth.revenue_growth_1y is not None:
            factors += 1
            if growth.revenue_growth_1y > 15:
                score += 3
            elif growth.revenue_growth_1y > 5:
                score += 2
            elif growth.revenue_growth_1y > 0:
                score += 1
        
        # Calculate final health rating
        if factors == 0:
            return FinancialHealth.FAIR
        
        avg_score = score / factors
        
        if avg_score >= 3.5:
            return FinancialHealth.EXCELLENT
        elif avg_score >= 2.5:
            return FinancialHealth.GOOD
        elif avg_score >= 1.5:
            return FinancialHealth.FAIR
        elif avg_score >= 0.5:
            return FinancialHealth.POOR
        else:
            return FinancialHealth.DISTRESSED
    
    def _calculate_score_safe(self, ratios: FinancialRatios, growth: GrowthMetrics) -> float:
        """Your existing score calculation - UNCHANGED"""
        total_score = 0
        components = 0
        
        # ROE component (0-25 points)
        if ratios.roe and ratios.roe > 0:
            components += 1
            total_score += min(ratios.roe, 25)
        
        # Growth component (0-25 points)
        if growth.revenue_growth_1y is not None:
            components += 1
            growth_score = max(0, min(growth.revenue_growth_1y + 10, 25))
            total_score += growth_score
        
        # Margin component (0-25 points)
        if ratios.net_margin and ratios.net_margin > 0:
            components += 1
            total_score += min(ratios.net_margin, 25)
        
        # Liquidity component (0-25 points)
        if ratios.current_ratio and ratios.current_ratio > 0:
            components += 1
            liquidity_score = min(ratios.current_ratio * 12.5, 25)
            total_score += liquidity_score
        
        if components > 0:
            return min(total_score / components * 4, 100)  # Scale to 0-100
        else:
            return 50.0  # Default if no data
    
    def _identify_areas_safe(self, ratios: FinancialRatios, growth: GrowthMetrics) -> Tuple[List[str], List[str]]:
        """Your existing areas identification - UNCHANGED"""
        strengths = []
        concerns = []
        
        # ROE analysis
        if ratios.roe:
            if ratios.roe > 20:
                strengths.append("Excellent ROE")
            elif ratios.roe < 5:
                concerns.append("Low ROE")
        
        # Growth analysis
        if growth.revenue_growth_1y is not None:
            if growth.revenue_growth_1y > 15:
                strengths.append("Strong growth")
            elif growth.revenue_growth_1y < 0:
                concerns.append("Declining revenue")
        
        # Debt analysis
        if ratios.debt_to_equity is not None:
            if ratios.debt_to_equity < 0.3:
                strengths.append("Low debt")
            elif ratios.debt_to_equity > 1.5:
                concerns.append("High debt")
        
        # Margin analysis
        if ratios.net_margin:
            if ratios.net_margin > 15:
                strengths.append("High margins")
            elif ratios.net_margin < 3:
                concerns.append("Low margins")
        
        # Valuation analysis
        if ratios.pe_ratio:
            if 10 <= ratios.pe_ratio <= 20:
                strengths.append("Fair valuation")
            elif ratios.pe_ratio > 40:
                concerns.append("High valuation")
        
        # Ensure we have at least something
        if not strengths and not concerns:
            strengths = ["Analysis available"]
            concerns = ["Limited data"]
        
        return strengths, concerns
    
    def _generate_thesis_safe(self, ratios: FinancialRatios, growth: GrowthMetrics, health: FinancialHealth) -> Tuple[str, str]:
        """Your existing thesis generation - UNCHANGED"""
        bull_points = []
        bear_points = []
        
        # Bull case factors
        if growth.revenue_growth_1y and growth.revenue_growth_1y > 10:
            bull_points.append(f"Strong {growth.revenue_growth_1y:.1f}% revenue growth")
        
        if ratios.roe and ratios.roe > 15:
            bull_points.append(f"Solid {ratios.roe:.1f}% ROE")
        
        if ratios.debt_to_equity is not None and ratios.debt_to_equity < 0.5:
            bull_points.append("Conservative debt levels")
        
        if health in [FinancialHealth.EXCELLENT, FinancialHealth.GOOD]:
            bull_points.append("Strong financial position")
        
        # Bear case factors
        if growth.revenue_growth_1y is not None and growth.revenue_growth_1y < 0:
            bear_points.append("Revenue declining")
        
        if ratios.pe_ratio and ratios.pe_ratio > 30:
            bear_points.append(f"High {ratios.pe_ratio:.1f}x valuation")
        
        if ratios.debt_to_equity and ratios.debt_to_equity > 1.0:
            bear_points.append("High debt burden")
        
        if ratios.current_ratio and ratios.current_ratio < 1.2:
            bear_points.append("Liquidity concerns")
        
        # Format cases
        bull_case = "; ".join(bull_points) if bull_points else "Financial metrics appear stable"
        bear_case = "; ".join(bear_points) if bear_points else "No major red flags identified"
        
        return bull_case, bear_case
    
    def _calculate_completeness_safe(self, highlights: Dict, valuation: Dict) -> float:
        """Your existing completeness calculation - UNCHANGED"""
        expected_fields = ["PERatio", "ReturnOnEquityTTM", "ProfitMargin", "RevenueGrowthTTM"]
        available_count = 0
        
        for field in expected_fields:
            if highlights.get(field) is not None and highlights.get(field) != "":
                available_count += 1
        
        return (available_count / len(expected_fields)) * 100
    
    def _extract_quarter_date_safe(self, data: Dict) -> Optional[str]:
        """Your existing quarter date extraction - UNCHANGED"""
        try:
            general = data.get("General", {})
            return general.get("LastSplitDate", "Unknown")
        except Exception:
            return "Unknown"
    
    def _create_minimal_result(self, symbol: str, error_message: str) -> FundamentalAnalysisResult:
        """Your existing minimal result creation - UNCHANGED"""
        return FundamentalAnalysisResult(
            symbol=symbol,
            analysis_timestamp=datetime.now(),
            current_price=0.0,
            financial_health=FinancialHealth.FAIR,
            overall_score=50.0,
            strength_areas=["Data unavailable"],
            concern_areas=["Analysis incomplete"],
            bull_case=f"Unable to complete analysis: {error_message}",
            bear_case="Insufficient data for risk assessment",
            data_completeness=0.0,
            last_quarter_date="Unknown"
        )
    
    async def close(self):
        """Cleanup method"""
        self.memory_cache.clear()
        
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("✅ Hybrid Fundamental Analysis service closed")


# Example usage in your main app
async def setup_hybrid_fundamental_service():
    """Setup the hybrid fundamental service"""
    fundamental_service = HybridFundamentalAnalysisService(
        mongodb_url=os.getenv('MONGODB_URL'),
        redis_url=os.getenv('REDIS_URL')
    )
    
    await fundamental_service.initialize()
    return fundamental_service

# Usage example - RAW DATA ONLY
async def test_hybrid_fundamental_service():
    service = await setup_hybrid_fundamental_service()
    
    # This returns RAW fundamental data (no SMS formatting)
    result = await service.analyze("AAPL")
    
    print(f"Symbol: {result.symbol}")
    print(f"Financial Health: {result.financial_health.value}")
    print(f"Overall Score: {result.overall_score}")
    print(f"PE Ratio: {result.ratios.pe_ratio}")
    print(f"ROE: {result.ratios.roe}")
    print(f"Revenue Growth: {result.growth.revenue_growth_1y}")
    print(f"Data Source: {result.data_source}")
    print(f"Response Time: {result.response_time_ms}ms")
    print(f"Cache Hit: {result.cache_hit}")


# Export for compatibility - RAW DATA ONLY
__all__ = ['HybridFundamentalAnalysisService', 'FundamentalAnalysisResult', 
           'FinancialRatios', 'GrowthMetrics', 'AnalysisDepth', 'FinancialHealth']
