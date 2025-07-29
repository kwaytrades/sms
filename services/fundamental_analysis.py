# services/fundamental_analysis.py - FIXED VERSION
"""
SMS Trading Bot - Fundamental Analysis Engine (ROBUST ERROR HANDLING)
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from enum import Enum

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
    """Complete fundamental analysis result"""
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

class FundamentalAnalysisEngine:
    """Robust Fundamental Analysis Engine with comprehensive error handling"""
    
    def __init__(self, eodhd_api_key: str, redis_client=None):
        self.eodhd_api_key = eodhd_api_key
        self.redis_client = redis_client
        self.base_url = "https://eodhd.com/api"
        self.cache_ttl = 7 * 24 * 3600  # 1 week
    
    async def analyze(self, symbol: str, analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD) -> FundamentalAnalysisResult:
        """Main analysis method with comprehensive error handling"""
        try:
            logger.info(f"🔍 Starting fundamental analysis for {symbol}")
            
            # Check cache first (with proper error handling)
            cache_key = f"fundamental_analysis:{symbol}:{analysis_depth.value}"
            cached_result = await self._get_cached_result_safe(cache_key)
            if cached_result:
                logger.info(f"✅ Returning cached fundamental analysis for {symbol}")
                return cached_result
            
            # Fetch data with timeout and error handling
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
            
            # Perform analysis with the fetched data
            result = await self._perform_robust_analysis(symbol, data)
            
            # Cache result safely
            await self._cache_result_safe(cache_key, result)
            
            logger.info(f"✅ Completed fundamental analysis for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"💥 Fundamental analysis failed for {symbol}: {e}")
            return self._create_minimal_result(symbol, f"Analysis error: {str(e)}")
    
    async def _fetch_all_data(self, symbol: str) -> Dict:
        """Fetch all required data with proper error handling"""
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
        """Perform analysis with robust error handling"""
        
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
        """Safely convert value to float with default"""
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
        """Calculate ratios with comprehensive safety checks"""
        
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
        """Calculate growth metrics safely"""
        
        highlights = data.get("Highlights", {}) or {}
        
        return GrowthMetrics(
            revenue_growth_1y=self._safe_float(highlights.get("RevenueGrowthTTM")),
            earnings_growth_1y=self._safe_float(highlights.get("EarningsGrowthTTM")),
            eps_growth_1y=self._safe_float(highlights.get("EPSGrowthTTM"))
        )
    
    def _assess_health_safe(self, ratios: FinancialRatios, growth: GrowthMetrics) -> FinancialHealth:
        """Assess financial health with safety checks"""
        
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
        """Calculate composite score safely"""
        
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
        """Identify strengths and concerns safely"""
        
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
        """Generate investment thesis safely"""
        
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
        """Calculate data completeness safely"""
        
        expected_fields = ["PERatio", "ReturnOnEquityTTM", "ProfitMargin", "RevenueGrowthTTM"]
        available_count = 0
        
        for field in expected_fields:
            if highlights.get(field) is not None and highlights.get(field) != "":
                available_count += 1
        
        return (available_count / len(expected_fields)) * 100
    
    def _extract_quarter_date_safe(self, data: Dict) -> Optional[str]:
        """Extract last quarter date safely"""
        try:
            general = data.get("General", {})
            return general.get("LastSplitDate", "Unknown")
        except Exception:
            return "Unknown"
    
    def _create_minimal_result(self, symbol: str, error_message: str) -> FundamentalAnalysisResult:
        """Create minimal result when data is unavailable"""
        
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
    
    async def _get_cached_result_safe(self, cache_key: str) -> Optional[FundamentalAnalysisResult]:
        """Safely retrieve cached result"""
        try:
            if not self.redis_client:
                return None
            
            # Handle both sync and async Redis clients
            if hasattr(self.redis_client, 'get'):
                cached_data = self.redis_client.get(cache_key)
            else:
                cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                if isinstance(cached_data, bytes):
                    cached_data = cached_data.decode('utf-8')
                
                data = json.loads(cached_data)
                return self._reconstruct_result_from_cache(data)
            
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def _cache_result_safe(self, cache_key: str, result: FundamentalAnalysisResult):
        """Safely cache result"""
        try:
            if not self.redis_client:
                return
            
            result_dict = {
                "symbol": result.symbol,
                "analysis_timestamp": result.analysis_timestamp.isoformat(),
                "current_price": result.current_price,
                "financial_health": result.financial_health.value,
                "overall_score": result.overall_score,
                "strength_areas": result.strength_areas,
                "concern_areas": result.concern_areas,
                "bull_case": result.bull_case,
                "bear_case": result.bear_case,
                "data_completeness": result.data_completeness
            }
            
            # Handle both sync and async Redis clients
            if hasattr(self.redis_client, 'setex'):
                self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result_dict))
            else:
                await self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result_dict))
                
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _reconstruct_result_from_cache(self, data: Dict) -> FundamentalAnalysisResult:
        """Reconstruct result object from cached data"""
        
        return FundamentalAnalysisResult(
            symbol=data.get("symbol", ""),
            analysis_timestamp=datetime.fromisoformat(data.get("analysis_timestamp", datetime.now().isoformat())),
            current_price=data.get("current_price", 0.0),
            financial_health=FinancialHealth(data.get("financial_health", "fair")),
            overall_score=data.get("overall_score", 50.0),
            strength_areas=data.get("strength_areas", []),
            concern_areas=data.get("concern_areas", []),
            bull_case=data.get("bull_case", ""),
            bear_case=data.get("bear_case", ""),
            data_completeness=data.get("data_completeness", 0.0)
        )

class FundamentalAnalysisTool:
    """Integration wrapper for the conversation agent"""
    
    def __init__(self, eodhd_api_key: str, redis_client=None):
        self.engine = FundamentalAnalysisEngine(eodhd_api_key, redis_client)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fundamental analysis with robust error handling"""
        try:
            symbol = parameters.get("symbol", "").upper().strip()
            depth_str = parameters.get("depth", "standard")
            user_style = parameters.get("user_style", "professional")
            
            if not symbol:
                return {
                    "success": False,
                    "error": "Symbol required",
                    "sms_response": "Please specify a stock symbol for fundamental analysis."
                }
            
            logger.info(f"🔍 Executing fundamental analysis for {symbol}")
            
            # Convert depth string to enum safely
            try:
                depth = AnalysisDepth(depth_str)
            except ValueError:
                depth = AnalysisDepth.STANDARD
            
            # Perform analysis
            result = await self.engine.analyze(symbol, depth)
            
            # Generate response based on analysis
            if result.data_completeness > 30:
                # We have enough data for a real analysis
                sms_response = self._format_real_analysis(result, user_style)
            else:
                # Limited data available
                sms_response = f"{symbol} fundamental data limited. Basic metrics: Health rated {result.financial_health.value}, analysis score {result.overall_score:.0f}/100. Consider technical analysis for trading insights."
            
            return {
                "success": True,
                "analysis_result": result,
                "sms_response": sms_response,
                "metadata": {
                    "symbol": symbol,
                    "overall_score": result.overall_score,
                    "financial_health": result.financial_health.value,
                    "data_completeness": result.data_completeness
                }
            }
            
        except Exception as e:
            logger.error(f"💥 Fundamental analysis tool failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "sms_response": f"{parameters.get('symbol', 'Stock')} fundamental analysis temporarily unavailable."
            }
    
    def _format_real_analysis(self, result: FundamentalAnalysisResult, user_style: str) -> str:
        """Format analysis into professional response"""
        
        # Build response components
        health_desc = {
            FinancialHealth.EXCELLENT: "excellent",
            FinancialHealth.GOOD: "strong", 
            FinancialHealth.FAIR: "fair",
            FinancialHealth.POOR: "weak",
            FinancialHealth.DISTRESSED: "concerning"
        }
        
        response_parts = []
        
        # Opening with key metrics
        response_parts.append(f"{result.symbol} fundamentals show {health_desc.get(result.financial_health, 'fair')} financial health (score: {result.overall_score:.0f}/100).")
        
        # Key ratios if available
        if result.ratios.pe_ratio and result.ratios.pe_ratio > 0:
            pe_desc = "expensive" if result.ratios.pe_ratio > 25 else "reasonable" if result.ratios.pe_ratio > 15 else "attractive"
            response_parts.append(f"P/E ratio of {result.ratios.pe_ratio:.1f} appears {pe_desc}.")
        
        # Growth info
        if result.growth.revenue_growth_1y is not None:
            if result.growth.revenue_growth_1y > 0:
                response_parts.append(f"Revenue growing {result.growth.revenue_growth_1y:.1f}% annually.")
            else:
                response_parts.append(f"Revenue declining {abs(result.growth.revenue_growth_1y):.1f}%.")
        
        # Key strength or concern
        if result.strength_areas:
            response_parts.append(f"Strength: {result.strength_areas[0].lower()}.")
        
        if result.concern_areas and len(response_parts) < 4:
            response_parts.append(f"Concern: {result.concern_areas[0].lower()}.")
        
        # Join and ensure SMS length
        response = " ".join(response_parts)
        if len(response) > 280:
            response = response[:277] + "..."
        
        return response

# Export for compatibility
__all__ = ['FundamentalAnalysisEngine', 'FundamentalAnalysisTool', 'AnalysisDepth', 'FinancialHealth']
