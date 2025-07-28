# services/analysis/fundamental_analysis.py
"""
SMS Trading Bot - Fundamental Analysis Engine
Comprehensive financial statement analysis and valuation metrics system
Integrates with existing ToolExecutor pattern for seamless orchestration
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import redis
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisDepth(Enum):
    """Analysis depth levels for different user tiers"""
    BASIC = "basic"           # Free tier - key ratios only
    STANDARD = "standard"     # Paid tier - full ratios + trends
    COMPREHENSIVE = "comprehensive"  # Pro tier - everything + forecasts

class FinancialHealth(Enum):
    """Overall financial health classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DISTRESSED = "distressed"

@dataclass
class FinancialRatios:
    """Core financial ratios structure"""
    # Valuation Ratios
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    
    # Profitability Ratios
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    roic: Optional[float] = None  # Return on Invested Capital
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    
    # Liquidity Ratios
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    
    # Leverage Ratios
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    # Efficiency Ratios
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None

@dataclass
class GrowthMetrics:
    """Growth and trend analysis"""
    revenue_growth_1y: Optional[float] = None
    revenue_growth_3y: Optional[float] = None
    earnings_growth_1y: Optional[float] = None
    earnings_growth_3y: Optional[float] = None
    eps_growth_1y: Optional[float] = None
    book_value_growth: Optional[float] = None
    free_cash_flow_growth: Optional[float] = None

@dataclass
class ValuationAnalysis:
    """Comprehensive valuation assessment"""
    intrinsic_value_estimate: Optional[float] = None
    discount_to_fair_value: Optional[float] = None
    peer_comparison_percentile: Optional[int] = None
    sector_relative_pe: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

@dataclass
class FundamentalAnalysisResult:
    """Complete fundamental analysis result"""
    symbol: str
    analysis_timestamp: datetime
    current_price: float
    
    # Core Analysis Components
    ratios: FinancialRatios
    growth: GrowthMetrics
    valuation: ValuationAnalysis
    
    # Summary Assessments
    financial_health: FinancialHealth
    overall_score: float  # 0-100 composite score
    strength_areas: List[str]
    concern_areas: List[str]
    
    # Investment Thesis
    bull_case: str
    bear_case: str
    analyst_consensus: Optional[str] = None
    
    # Data Quality
    data_completeness: float  # Percentage of available data
    last_quarter_date: Optional[str] = None
    
    def to_sms_summary(self, user_style: str = "casual") -> str:
        """Convert to SMS-friendly format based on user personality"""
        if user_style == "technical":
            return self._technical_summary()
        elif user_style == "casual":
            return self._casual_summary()
        else:
            return self._standard_summary()
    
    def _casual_summary(self) -> str:
        """Casual, approachable summary"""
        health_emoji = {
            FinancialHealth.EXCELLENT: "💪",
            FinancialHealth.GOOD: "👍",
            FinancialHealth.FAIR: "🤔",
            FinancialHealth.POOR: "😬",
            FinancialHealth.DISTRESSED: "🚨"
        }
        
        emoji = health_emoji.get(self.financial_health, "📊")
        
        summary = f"{emoji} {self.symbol} Fundamentals:\n"
        summary += f"Health: {self.financial_health.value.title()} ({self.overall_score:.0f}/100)\n"
        
        if self.ratios.pe_ratio:
            pe_assessment = "expensive" if self.ratios.pe_ratio > 25 else "reasonable" if self.ratios.pe_ratio > 15 else "cheap"
            summary += f"P/E: {self.ratios.pe_ratio:.1f} ({pe_assessment})\n"
        
        if self.growth.revenue_growth_1y:
            growth_trend = "growing fast! 🚀" if self.growth.revenue_growth_1y > 20 else "growing steady" if self.growth.revenue_growth_1y > 5 else "struggling"
            summary += f"Revenue: {growth_trend} ({self.growth.revenue_growth_1y:+.1f}%)\n"
        
        if self.strength_areas:
            summary += f"Strengths: {', '.join(self.strength_areas[:2])}\n"
        
        if self.concern_areas:
            summary += f"Concerns: {', '.join(self.concern_areas[:2])}"
        
        return summary
    
    def _technical_summary(self) -> str:
        """Technical, detailed summary"""
        summary = f"📊 {self.symbol} Fundamental Analysis:\n"
        summary += f"Score: {self.overall_score:.0f}/100 | Health: {self.financial_health.value.upper()}\n"
        
        # Key ratios
        if self.ratios.pe_ratio and self.ratios.roe:
            summary += f"P/E: {self.ratios.pe_ratio:.1f} | ROE: {self.ratios.roe:.1f}%\n"
        
        if self.ratios.debt_to_equity:
            summary += f"D/E: {self.ratios.debt_to_equity:.2f} | "
        
        if self.ratios.current_ratio:
            summary += f"Current Ratio: {self.ratios.current_ratio:.2f}\n"
        
        # Growth metrics
        if self.growth.revenue_growth_1y and self.growth.earnings_growth_1y:
            summary += f"Growth: Rev {self.growth.revenue_growth_1y:+.1f}% | EPS {self.growth.earnings_growth_1y:+.1f}%\n"
        
        # Valuation
        if self.valuation.intrinsic_value_estimate:
            discount = ((self.current_price - self.valuation.intrinsic_value_estimate) / self.valuation.intrinsic_value_estimate) * 100
            summary += f"Fair Value: ${self.valuation.intrinsic_value_estimate:.2f} ({discount:+.1f}%)"
        
        return summary

class FundamentalAnalysisEngine:
    """
    Advanced Fundamental Analysis Engine
    Integrates with existing SMS bot architecture and ToolExecutor pattern
    """
    
    def __init__(self, eodhd_api_key: str, redis_client: redis.Redis):
        self.eodhd_api_key = eodhd_api_key
        self.redis_client = redis_client
        self.base_url = "https://eodhd.com/api"
        
        # Cache settings - fundamental data changes very slowly (quarterly/yearly reports)
        self.cache_ttl = {
            "ratios": 7 * 24 * 3600,  # 1 week
            "financials": 7 * 24 * 3600,  # 1 week
            "analysis": 7 * 24 * 3600,  # 1 week
        }
        
        # Analysis scoring weights
        self.scoring_weights = {
            "profitability": 0.30,  # ROE, margins, growth
            "valuation": 0.25,      # P/E, PEG, relative valuation
            "financial_health": 0.25,  # Debt, liquidity, coverage
            "growth": 0.20          # Revenue/earnings growth trends
        }
    
    async def analyze(self, symbol: str, analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD) -> FundamentalAnalysisResult:
        """
        Main analysis method - follows ToolExecutor integration pattern
        """
        try:
            logger.info(f"Starting fundamental analysis for {symbol} at {analysis_depth.value} depth")
            
            # Check cache first
            cache_key = f"fundamental_analysis:{symbol}:{analysis_depth.value}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Returning cached fundamental analysis for {symbol}")
                return cached_result
            
            # Fetch required data in parallel
            financial_data, market_data, ratios_data = await asyncio.gather(
                self._fetch_financial_statements(symbol),
                self._fetch_market_data(symbol),
                self._fetch_key_ratios(symbol),
                return_exceptions=True
            )
            
            # Handle data fetch failures gracefully
            if isinstance(financial_data, Exception):
                logger.warning(f"Financial data fetch failed for {symbol}: {financial_data}")
                financial_data = {}
            
            if isinstance(market_data, Exception):
                logger.warning(f"Market data fetch failed for {symbol}: {market_data}")
                market_data = {}
            
            if isinstance(ratios_data, Exception):
                logger.warning(f"Ratios data fetch failed for {symbol}: {ratios_data}")
                ratios_data = {}
            
            # Perform comprehensive analysis
            analysis_result = await self._perform_analysis(
                symbol, financial_data, market_data, ratios_data, analysis_depth
            )
            
            # Cache the result
            await self._cache_result(cache_key, analysis_result)
            
            logger.info(f"Completed fundamental analysis for {symbol}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {str(e)}")
            # Return minimal analysis with error indication
            return FundamentalAnalysisResult(
                symbol=symbol,
                analysis_timestamp=datetime.now(),
                current_price=0.0,
                ratios=FinancialRatios(),
                growth=GrowthMetrics(),
                valuation=ValuationAnalysis(),
                financial_health=FinancialHealth.FAIR,
                overall_score=50.0,
                strength_areas=["Data unavailable"],
                concern_areas=["Analysis incomplete"],
                bull_case="Insufficient data for analysis",
                bear_case="Insufficient data for analysis",
                data_completeness=0.0
            )
    
    async def _fetch_financial_statements(self, symbol: str) -> Dict:
        """Fetch latest financial statements from EODHD"""
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch income statement, balance sheet, and cash flow
                endpoints = {
                    "income": f"{self.base_url}/fundamentals/{symbol}?api_token={self.eodhd_api_key}&filter=Financials::Income_Statement",
                    "balance": f"{self.base_url}/fundamentals/{symbol}?api_token={self.eodhd_api_key}&filter=Financials::Balance_Sheet",
                    "cashflow": f"{self.base_url}/fundamentals/{symbol}?api_token={self.eodhd_api_key}&filter=Financials::Cash_Flow"
                }
                
                tasks = [session.get(url) for url in endpoints.values()]
                responses = await asyncio.gather(*tasks)
                
                financial_data = {}
                for key, response in zip(endpoints.keys(), responses):
                    if response.status == 200:
                        data = await response.json()
                        financial_data[key] = data
                    else:
                        logger.warning(f"Failed to fetch {key} statement for {symbol}: {response.status}")
                        financial_data[key] = {}
                
                return financial_data
                
        except Exception as e:
            logger.error(f"Error fetching financial statements for {symbol}: {str(e)}")
            return {}
    
    async def _fetch_market_data(self, symbol: str) -> Dict:
        """Fetch current market data and historical prices"""
        try:
            async with aiohttp.ClientSession() as session:
                # Current market data
                url = f"{self.base_url}/real-time/{symbol}?api_token={self.eodhd_api_key}&fmt=json"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Failed to fetch market data for {symbol}: {response.status}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return {}
    
    async def _fetch_key_ratios(self, symbol: str) -> Dict:
        """Fetch pre-calculated key ratios from EODHD"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/fundamentals/{symbol}?api_token={self.eodhd_api_key}&filter=Valuation,Highlights"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Failed to fetch ratios for {symbol}: {response.status}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Error fetching ratios for {symbol}: {str(e)}")
            return {}
    
    async def _perform_analysis(
        self, 
        symbol: str, 
        financial_data: Dict, 
        market_data: Dict, 
        ratios_data: Dict,
        analysis_depth: AnalysisDepth
    ) -> FundamentalAnalysisResult:
        """Perform comprehensive fundamental analysis"""
        
        # Extract current price
        current_price = float(market_data.get("close", 0)) or float(market_data.get("price", 0))
        
        # Calculate financial ratios
        ratios = await self._calculate_ratios(financial_data, ratios_data, current_price)
        
        # Calculate growth metrics
        growth = await self._calculate_growth_metrics(financial_data)
        
        # Perform valuation analysis
        valuation = await self._perform_valuation_analysis(ratios, growth, current_price, symbol)
        
        # Assess financial health
        financial_health = self._assess_financial_health(ratios, growth)
        
        # Calculate composite score
        overall_score = self._calculate_composite_score(ratios, growth, valuation)
        
        # Identify strengths and concerns
        strength_areas, concern_areas = self._identify_key_areas(ratios, growth, valuation)
        
        # Generate investment thesis
        bull_case, bear_case = self._generate_investment_thesis(ratios, growth, valuation, financial_health)
        
        # Calculate data completeness
        data_completeness = self._calculate_data_completeness(financial_data, ratios_data)
        
        return FundamentalAnalysisResult(
            symbol=symbol,
            analysis_timestamp=datetime.now(),
            current_price=current_price,
            ratios=ratios,
            growth=growth,
            valuation=valuation,
            financial_health=financial_health,
            overall_score=overall_score,
            strength_areas=strength_areas,
            concern_areas=concern_areas,
            bull_case=bull_case,
            bear_case=bear_case,
            data_completeness=data_completeness,
            last_quarter_date=self._extract_last_quarter_date(financial_data)
        )
    
    async def _calculate_ratios(self, financial_data: Dict, ratios_data: Dict, current_price: float) -> FinancialRatios:
        """Calculate comprehensive financial ratios"""
        
        # Extract data from various sources
        highlights = ratios_data.get("Highlights", {})
        valuation = ratios_data.get("Valuation", {})
        
        # Get latest financial statement data
        latest_income = self._get_latest_period_data(financial_data.get("income", {}))
        latest_balance = self._get_latest_period_data(financial_data.get("balance", {}))
        
        # Calculate ratios with safe division
        def safe_divide(numerator, denominator, default=None):
            try:
                if denominator and float(denominator) != 0:
                    return float(numerator) / float(denominator)
                return default
            except (TypeError, ValueError, ZeroDivisionError):
                return default
        
        return FinancialRatios(
            # Valuation ratios from API
            pe_ratio=self._safe_float(highlights.get("PERatio")),
            peg_ratio=self._safe_float(highlights.get("PEGRatio")),
            pb_ratio=self._safe_float(highlights.get("PriceBookMRQ")),
            ps_ratio=self._safe_float(highlights.get("PriceSalesTTM")),
            ev_ebitda=self._safe_float(valuation.get("EnterpriseValueEbitda")),
            
            # Profitability ratios
            roe=self._safe_float(highlights.get("ReturnOnEquityTTM")),
            roa=self._safe_float(highlights.get("ReturnOnAssetsTTM")),
            gross_margin=self._safe_float(highlights.get("GrossProfitMargin")),
            operating_margin=self._safe_float(highlights.get("OperatingMarginTTM")),
            net_margin=self._safe_float(highlights.get("ProfitMargin")),
            
            # Calculate liquidity ratios from balance sheet
            current_ratio=safe_divide(
                latest_balance.get("totalCurrentAssets"),
                latest_balance.get("totalCurrentLiabilities")
            ),
            quick_ratio=safe_divide(
                float(latest_balance.get("totalCurrentAssets", 0)) - float(latest_balance.get("inventory", 0)),
                latest_balance.get("totalCurrentLiabilities")
            ),
            
            # Leverage ratios
            debt_to_equity=safe_divide(
                latest_balance.get("totalDebt"),
                latest_balance.get("totalStockholderEquity")
            ),
            debt_to_assets=safe_divide(
                latest_balance.get("totalDebt"),
                latest_balance.get("totalAssets")
            ),
            
            # Efficiency ratios
            asset_turnover=safe_divide(
                latest_income.get("totalRevenue"),
                latest_balance.get("totalAssets")
            )
        )
    
    async def _calculate_growth_metrics(self, financial_data: Dict) -> GrowthMetrics:
        """Calculate growth trends from historical financial data"""
        
        income_data = financial_data.get("income", {})
        if not income_data:
            return GrowthMetrics()
        
        # Get historical data points
        periods = self._get_historical_periods(income_data, periods=4)  # Last 4 quarters/years
        
        if len(periods) < 2:
            return GrowthMetrics()
        
        def calculate_growth_rate(current, previous):
            """Calculate growth rate between two periods"""
            try:
                if previous and float(previous) != 0:
                    return ((float(current) - float(previous)) / float(previous)) * 100
                return None
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        
        # Calculate various growth metrics
        latest = periods[0]
        year_ago = periods[-1] if len(periods) >= 4 else periods[-1]
        
        return GrowthMetrics(
            revenue_growth_1y=calculate_growth_rate(
                latest.get("totalRevenue"),
                year_ago.get("totalRevenue")
            ),
            earnings_growth_1y=calculate_growth_rate(
                latest.get("netIncome"),
                year_ago.get("netIncome")
            ),
            eps_growth_1y=calculate_growth_rate(
                latest.get("eps"),
                year_ago.get("eps")
            )
        )
    
    async def _perform_valuation_analysis(self, ratios: FinancialRatios, growth: GrowthMetrics, current_price: float, symbol: str) -> ValuationAnalysis:
        """Perform valuation analysis and estimate fair value"""
        
        # Simple DCF-based intrinsic value estimation
        intrinsic_value = None
        if ratios.pe_ratio and growth.earnings_growth_1y:
            # Simplified PEG-based valuation
            fair_pe = min(max(growth.earnings_growth_1y, 10), 30)  # Cap between 10-30
            if fair_pe and ratios.pe_ratio:
                intrinsic_value = current_price * (fair_pe / ratios.pe_ratio)
        
        # Calculate discount to fair value
        discount_to_fair_value = None
        if intrinsic_value and current_price:
            discount_to_fair_value = ((current_price - intrinsic_value) / intrinsic_value) * 100
        
        return ValuationAnalysis(
            intrinsic_value_estimate=intrinsic_value,
            discount_to_fair_value=discount_to_fair_value,
            dividend_yield=self._safe_float(ratios.pe_ratio) if hasattr(ratios, 'dividend_yield') else None
        )
    
    def _assess_financial_health(self, ratios: FinancialRatios, growth: GrowthMetrics) -> FinancialHealth:
        """Assess overall financial health based on key metrics"""
        
        score = 0
        max_score = 0
        
        # Profitability assessment
        if ratios.roe:
            max_score += 25
            if ratios.roe > 20:
                score += 25
            elif ratios.roe > 15:
                score += 20
            elif ratios.roe > 10:
                score += 15
            elif ratios.roe > 5:
                score += 10
        
        # Liquidity assessment
        if ratios.current_ratio:
            max_score += 20
            if ratios.current_ratio > 2.0:
                score += 20
            elif ratios.current_ratio > 1.5:
                score += 15
            elif ratios.current_ratio > 1.0:
                score += 10
        
        # Leverage assessment
        if ratios.debt_to_equity:
            max_score += 20
            if ratios.debt_to_equity < 0.3:
                score += 20
            elif ratios.debt_to_equity < 0.6:
                score += 15
            elif ratios.debt_to_equity < 1.0:
                score += 10
            elif ratios.debt_to_equity < 2.0:
                score += 5
        
        # Growth assessment
        if growth.revenue_growth_1y:
            max_score += 20
            if growth.revenue_growth_1y > 20:
                score += 20
            elif growth.revenue_growth_1y > 10:
                score += 15
            elif growth.revenue_growth_1y > 5:
                score += 10
            elif growth.revenue_growth_1y > 0:
                score += 5
        
        # Valuation assessment
        if ratios.pe_ratio:
            max_score += 15
            if 10 <= ratios.pe_ratio <= 20:
                score += 15
            elif 8 <= ratios.pe_ratio <= 25:
                score += 10
            elif ratios.pe_ratio <= 30:
                score += 5
        
        # Calculate final health score
        if max_score > 0:
            health_percentage = (score / max_score) * 100
        else:
            health_percentage = 50  # Default if no data
        
        # Map to health categories
        if health_percentage >= 80:
            return FinancialHealth.EXCELLENT
        elif health_percentage >= 65:
            return FinancialHealth.GOOD
        elif health_percentage >= 50:
            return FinancialHealth.FAIR
        elif health_percentage >= 35:
            return FinancialHealth.POOR
        else:
            return FinancialHealth.DISTRESSED
    
    def _calculate_composite_score(self, ratios: FinancialRatios, growth: GrowthMetrics, valuation: ValuationAnalysis) -> float:
        """Calculate overall composite score (0-100)"""
        
        total_score = 0
        total_weight = 0
        
        # Profitability score (30% weight)
        if ratios.roe and ratios.net_margin:
            prof_score = min((ratios.roe / 20) * 50 + (ratios.net_margin / 20) * 50, 100)
            total_score += prof_score * self.scoring_weights["profitability"]
            total_weight += self.scoring_weights["profitability"]
        
        # Valuation score (25% weight)
        if ratios.pe_ratio:
            val_score = max(0, 100 - abs(ratios.pe_ratio - 18) * 3)  # Penalty for deviation from 18 P/E
            total_score += val_score * self.scoring_weights["valuation"]
            total_weight += self.scoring_weights["valuation"]
        
        # Financial health score (25% weight)
        if ratios.current_ratio and ratios.debt_to_equity:
            health_score = min(ratios.current_ratio * 30, 60) + max(0, 40 - ratios.debt_to_equity * 20)
            total_score += health_score * self.scoring_weights["financial_health"]
            total_weight += self.scoring_weights["financial_health"]
        
        # Growth score (20% weight)
        if growth.revenue_growth_1y:
            growth_score = min(max(growth.revenue_growth_1y * 2 + 50, 0), 100)
            total_score += growth_score * self.scoring_weights["growth"]
            total_weight += self.scoring_weights["growth"]
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 50.0  # Default score if no data
    
    def _identify_key_areas(self, ratios: FinancialRatios, growth: GrowthMetrics, valuation: ValuationAnalysis) -> Tuple[List[str], List[str]]:
        """Identify key strength and concern areas"""
        
        strengths = []
        concerns = []
        
        # Analyze profitability
        if ratios.roe and ratios.roe > 15:
            strengths.append("High ROE")
        elif ratios.roe and ratios.roe < 8:
            concerns.append("Low ROE")
        
        if ratios.net_margin and ratios.net_margin > 15:
            strengths.append("High margins")
        elif ratios.net_margin and ratios.net_margin < 5:
            concerns.append("Low margins")
        
        # Analyze growth
        if growth.revenue_growth_1y and growth.revenue_growth_1y > 15:
            strengths.append("Strong growth")
        elif growth.revenue_growth_1y and growth.revenue_growth_1y < 0:
            concerns.append("Declining revenue")
        
        # Analyze liquidity
        if ratios.current_ratio and ratios.current_ratio > 2:
            strengths.append("Strong liquidity")
        elif ratios.current_ratio and ratios.current_ratio < 1:
            concerns.append("Liquidity issues")
        
        # Analyze leverage
        if ratios.debt_to_equity and ratios.debt_to_equity < 0.3:
            strengths.append("Low debt")
        elif ratios.debt_to_equity and ratios.debt_to_equity > 1.5:
            concerns.append("High debt")
        
        # Analyze valuation
        if ratios.pe_ratio and 10 <= ratios.pe_ratio <= 18:
            strengths.append("Fair valuation")
        elif ratios.pe_ratio and ratios.pe_ratio > 30:
            concerns.append("High valuation")
        
        return strengths, concerns
    
    def _generate_investment_thesis(self, ratios: FinancialRatios, growth: GrowthMetrics, valuation: ValuationAnalysis, health: FinancialHealth) -> Tuple[str, str]:
        """Generate bull and bear investment cases"""
        
        bull_points = []
        bear_points = []
        
        # Build bull case
        if growth.revenue_growth_1y and growth.revenue_growth_1y > 10:
            bull_points.append(f"Strong {growth.revenue_growth_1y:.1f}% revenue growth")
        
        if ratios.roe and ratios.roe > 15:
            bull_points.append(f"Excellent {ratios.roe:.1f}% ROE")
        
        if ratios.debt_to_equity and ratios.debt_to_equity < 0.5:
            bull_points.append("Conservative debt levels")
        
        if health in [FinancialHealth.EXCELLENT, FinancialHealth.GOOD]:
            bull_points.append("Strong financial position")
        
        # Build bear case
        if growth.revenue_growth_1y and growth.revenue_growth_1y < 0:
            bear_points.append("Declining revenue trend")
        
        if ratios.pe_ratio and ratios.pe_ratio > 25:
            bear_points.append(f"High {ratios.pe_ratio:.1f}x P/E valuation")
        
        if ratios.debt_to_equity and ratios.debt_to_equity > 1.0:
            bear_points.append("High debt burden")
        
        if ratios.current_ratio and ratios.current_ratio < 1.2:
            bear_points.append("Potential liquidity concerns")
        
        # Format cases
        bull_case = "; ".join(bull_points) if bull_points else "Limited positive catalysts identified"
        bear_case = "; ".join(bear_points) if bear_points else "Limited major risks identified"
        
        return bull_case, bear_case
    
    # Helper methods
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        try:
            if value is not None and value != "":
                return float(value)
            return None
        except (TypeError, ValueError):
            return None
    
    def _get_latest_period_data(self, financial_data: Dict) -> Dict:
        """Extract latest period data from financial statements"""
        if not financial_data:
            return {}
        
        # EODHD typically returns data with quarterly/yearly structure
        # Look for the most recent data point
        if isinstance(financial_data, dict):
            for key in ["quarterly", "yearly"]:
                if key in financial_data and financial_data[key]:
                    # Return the first (most recent) period
                    periods = financial_data[key]
                    if isinstance(periods, dict):
                        # Get most recent date
                        latest_date = max(periods.keys())
                        return periods[latest_date]
                    elif isinstance(periods, list) and len(periods) > 0:
                        return periods[0]
        
        return {}
    
    def _get_historical_periods(self, financial_data: Dict, periods: int = 4) -> List[Dict]:
        """Get historical periods for trend analysis"""
        if not financial_data:
            return []
        
        historical_data = []
        
        # Look for quarterly data first, then yearly
        for timeframe in ["quarterly", "yearly"]:
            if timeframe in financial_data and financial_data[timeframe]:
                data = financial_data[timeframe]
                
                if isinstance(data, dict):
                    # Sort by date and take most recent periods
                    sorted_dates = sorted(data.keys(), reverse=True)
                    for date in sorted_dates[:periods]:
                        historical_data.append(data[date])
                elif isinstance(data, list):
                    historical_data = data[:periods]
                
                if len(historical_data) >= 2:
                    break
        
        return historical_data
    
    def _extract_last_quarter_date(self, financial_data: Dict) -> Optional[str]:
        """Extract the date of the last reported quarter"""
        try:
            income_data = financial_data.get("income", {})
            if "quarterly" in income_data and income_data["quarterly"]:
                if isinstance(income_data["quarterly"], dict):
                    return max(income_data["quarterly"].keys())
                elif isinstance(income_data["quarterly"], list) and len(income_data["quarterly"]) > 0:
                    # Look for date field in the first item
                    first_period = income_data["quarterly"][0]
                    return first_period.get("date", "Unknown")
            return None
        except Exception:
            return None
    
    def _calculate_data_completeness(self, financial_data: Dict, ratios_data: Dict) -> float:
        """Calculate percentage of available vs expected data"""
        expected_fields = 20  # Expected number of key data points
        available_fields = 0
        
        # Check financial data availability
        if financial_data.get("income"):
            available_fields += 5
        if financial_data.get("balance"):
            available_fields += 5
        if financial_data.get("cashflow"):
            available_fields += 3
        
        # Check ratios data availability
        if ratios_data.get("Highlights"):
            available_fields += 4
        if ratios_data.get("Valuation"):
            available_fields += 3
        
        return min((available_fields / expected_fields) * 100, 100)
    
    async def _get_cached_result(self, cache_key: str) -> Optional[FundamentalAnalysisResult]:
        """Retrieve cached analysis result"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                # Reconstruct the result object
                return FundamentalAnalysisResult(**data)
            return None
        except Exception as e:
            logger.warning(f"Failed to retrieve cached result: {str(e)}")
            return None
    
    async def _cache_result(self, cache_key: str, result: FundamentalAnalysisResult):
        """Cache analysis result"""
        try:
            # Convert result to dict for caching
            result_dict = {
                "symbol": result.symbol,
                "analysis_timestamp": result.analysis_timestamp.isoformat(),
                "current_price": result.current_price,
                "ratios": result.ratios.__dict__,
                "growth": result.growth.__dict__,
                "valuation": result.valuation.__dict__,
                "financial_health": result.financial_health.value,
                "overall_score": result.overall_score,
                "strength_areas": result.strength_areas,
                "concern_areas": result.concern_areas,
                "bull_case": result.bull_case,
                "bear_case": result.bear_case,
                "data_completeness": result.data_completeness,
                "last_quarter_date": result.last_quarter_date
            }
            
            self.redis_client.setex(
                cache_key,
                self.cache_ttl["analysis"],
                json.dumps(result_dict, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")

# Integration class for ToolExecutor pattern
class FundamentalAnalysisTool:
    """
    Integration wrapper for the ToolExecutor pattern
    Provides the standard interface expected by the SMS bot system
    """
    
    def __init__(self, eodhd_api_key: str, redis_client: redis.Redis):
        self.engine = FundamentalAnalysisEngine(eodhd_api_key, redis_client)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard ToolExecutor interface for SMS bot integration
        
        Expected parameters:
        - symbol: str - Stock symbol to analyze
        - depth: str - Analysis depth ("basic", "standard", "comprehensive")
        - user_style: str - User communication style for response formatting
        """
        try:
            symbol = parameters.get("symbol", "").upper()
            depth_str = parameters.get("depth", "standard")
            user_style = parameters.get("user_style", "casual")
            
            if not symbol:
                return {
                    "success": False,
                    "error": "Symbol parameter required",
                    "sms_response": "❌ Please specify a stock symbol for fundamental analysis"
                }
            
            # Convert depth string to enum
            try:
                depth = AnalysisDepth(depth_str)
            except ValueError:
                depth = AnalysisDepth.STANDARD
            
            # Perform analysis
            result = await self.engine.analyze(symbol, depth)
            
            # Generate SMS-friendly response
            sms_response = result.to_sms_summary(user_style)
            
            return {
                "success": True,
                "analysis_result": result,
                "sms_response": sms_response,
                "metadata": {
                    "symbol": symbol,
                    "analysis_depth": depth.value,
                    "overall_score": result.overall_score,
                    "financial_health": result.financial_health.value,
                    "data_completeness": result.data_completeness
                }
            }
            
        except Exception as e:
            logger.error(f"Fundamental analysis tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "sms_response": f"📊 {parameters.get('symbol', 'Analysis')} fundamental data temporarily unavailable. Try again shortly."
            }

# Example usage for testing
async def example_usage():
    """Example of how to use the Fundamental Analysis Engine"""
    import os
    
    # Initialize Redis client (adjust connection as needed)
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # Initialize the engine
    engine = FundamentalAnalysisEngine(
        eodhd_api_key=os.getenv("EODHD_API_KEY"),
        redis_client=redis_client
    )
    
    # Analyze a stock
    result = await engine.analyze("AAPL", AnalysisDepth.COMPREHENSIVE)
    
    # Print results
    print("=== Fundamental Analysis Result ===")
    print(f"Symbol: {result.symbol}")
    print(f"Overall Score: {result.overall_score:.1f}/100")
    print(f"Financial Health: {result.financial_health.value}")
    print(f"Strengths: {', '.join(result.strength_areas)}")
    print(f"Concerns: {', '.join(result.concern_areas)}")
    print("\n=== SMS Summary (Casual) ===")
    print(result.to_sms_summary("casual"))
    print("\n=== SMS Summary (Technical) ===")
    print(result.to_sms_summary("technical"))

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
