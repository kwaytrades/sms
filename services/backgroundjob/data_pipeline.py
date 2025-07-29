async def get_stock_data_with_fallback(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get stock data with fallback API call if not in universe"""
        try:
            symbol = symbol.upper()
            
            # Check if symbol is in our cached universe first
            if symbol in self.stock_universe and not force_refresh:
                # Get from cache/MongoDB
                cached_data = await self._get_cached_stock_data(symbol)
                if cached_data:
                    logger.info(f"✅ Returning cached data for {symbol}")
                    return {
                        "symbol": symbol,
                        "data_source": "cached",
                        "data": cached_data,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Symbol not in universe or force refresh - make API call
            logger.info(f"📡 Fetching on-demand data for {symbol}")
            fresh_data = await self._fetch_single_stock_data(symbol)
            
            if fresh_data.get("success"):
                # Cache the result for future use
                await self._cache_single_stock_data(symbol, fresh_data)
                
                # Consider adding to permanent universe if requested frequently
                await self._track_symbol_request(symbol)
                
                return {
                    "symbol": symbol,
                    "data_source": "on_demand_api",
                    "data": fresh_data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "symbol": symbol,
                    "error": f"Failed to fetch data for {symbol}",
                    "data_source": "api_failed"
                }
                
        except Exception as e:
            logger.error(f"❌ Fallback data fetch failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "data_source": "error"
            }
    
async def _get_cached_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get cached stock data from Redis/MongoDB"""
        try:
            # Try Redis first (fastest)
            if self.redis_client:
                basic_data = await self.redis_client.hgetall(f"stock:{symbol}:basic")
                technical_data = await self.redis_client.hgetall(f"stock:{symbol}:technical")
                fundamental_data = await self.redis_client.hgetall(f"stock:{symbol}:fundamental")
                
                if basic_data:
                    # Convert Redis data back to proper format
                    return {
                        "basic": {k: float(v) if k in ['price', 'volume', 'market_cap', 'change_1d'] else v 
                                 for k, v in dict(basic_data).items()},
                        "technical": {k: float(v) if k in ['rsi', 'sma_20', 'sma_50', 'sma_200', 'volatility'] 
                                     else json.loads(v) if k in ['support_levels', 'resistance_levels'] 
                                     else v for k, v in dict(technical_data).items()},
                        "fundamental": {k: float(v) if k in ['pe', 'pb', 'roe', 'debt_to_equity', 'dividend_yield'] 
                                       else v for k, v in dict(fundamental_data).items()},
                        "source": "redis_cache"
                    }
            
            # Try MongoDB as fallback
            if self.db:
                mongo_data = await self.db.stocks.find_one({"symbol": symbol})
                if mongo_data:
                    return {
                        "basic": mongo_data.get("basic", {}),
                        "technical": mongo_data.get("technical", {}),
                        "fundamental": mongo_data.get("fundamental", {}),
                        "source": "mongodb_cache"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting cached data for {symbol}: {e}")
            return None
    
    async def _fetch_single_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch complete data for a single stock via API"""
        try:
            # Fetch all data types for the single symbol
            basic_data = await self._fetch_single_basic_data(symbol)
            technical_data = await self._calculate_single_technical_data(symbol)
            fundamental_data = await self._fetch_single_fundamental_data(symbol)
            
            if basic_data or technical_data or fundamental_data:
                return {
                    "success": True,
                    "basic": basic_data or {},
                    "technical": technical_data or {},
                    "fundamental": fundamental_data or {},
                    "screening_tags": self._generate_screening_tags(
                        basic_data or {}, 
                        technical_data or {}, 
                        fundamental_data or {}
                    )
                }
            else:
                return {"success": False, "error": "No data available"}
                
        except Exception as e:
            logger.error(f"❌ Single stock fetch failed for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fetch_single_basic_data(self, symbol: str) -> Dict:
        """Fetch basic market data for single symbol"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://eodhd.com/api/real-time/{symbol}.US"
                params = {
                    "api_token": self.eodhd_api_key,
                    "fmt": "json"
                }
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "price": float(data.get("close", 0)),
                            "volume": int(data.get("volume", 0)),
                            "market_cap": float(data.get("market_cap", 0)),
                            "change_1d": float(data.get("change_p", 0)),
                            "high": float(data.get("high", 0)),
                            "low": float(data.get("low", 0)),
                            "open": float(data.get("open", 0)),
                            "previous_close": float(data.get("previous_close", 0)),
                            "sector": "Unknown",  # Would need separate call
                            "exchange": "NASDAQ"
                        }
            return {}
        except Exception as e:
            logger.error(f"❌ Basic data fetch failed for {symbol}: {e}")
            return {}
    
    async def _calculate_single_technical_data(self, symbol: str) -> Dict:
        """Calculate technical indicators for single symbol"""
        try:
            # Get historical data for calculations
            async with aiohttp.ClientSession() as session:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                
                url = f"https://eodhd.com/api/eod/{symbol}.US"
                params = {
                    'api_token': self.eodhd_api_key,
                    'fmt': 'json',
                    'from': start_date,
                    'to': end_date
                }
                
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 20:
                            return self._calculate_indicators_from_ohlc(data)
            
            return self._get_default_technical_data()
        except Exception as e:
            logger.error(f"❌ Technical calculation failed for {symbol}: {e}")
            return self._get_default_technical_data()
    
    async def _fetch_single_fundamental_data(self, symbol: str) -> Dict:
        """Fetch fundamental data for single symbol"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://eodhd.com/api/fundamentals/{symbol}.US"
                params = {
                    "api_token": self.eodhd_api_key,
                    "fmt": "json"
                }
                
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        highlights = data.get("Highlights", {})
                        valuation = data.get("Valuation", {})
                        
                        return {
                            "pe": float(highlights.get("PERatio", 0)),
                            "pb": float(highlights.get("PriceBookMRQ", 0)),
                            "roe": float(highlights.get("ReturnOnEquityTTM", 0)),
                            "debt_to_equity": float(highlights.get("DebtToEquityMRQ", 0)),
                            "dividend_yield": float(highlights.get("DividendYield", 0)),
                            "eps": float(highlights.get("EarningsShare", 0)),
                            "revenue_growth": float(highlights.get("RevenueGrowthTTM", 0)),
                            "profit_margin": float(highlights.get("ProfitMargin", 0)),
                            "market_cap": float(highlights.get("MarketCapitalization", 0)),
                            "beta": float(highlights.get("Beta", 1.0)),
                            "price_to_sales": float(valuation.get("TrailingPS", 0)),
                            "enterprise_value": float(valuation.get("EnterpriseValue", 0)),
                            "forward_pe": float(valuation.get("ForwardPE", 0))
                        }
            
            return self._get_default_fundamental_data()
        except Exception as e:
            logger.error(f"❌ Fundamental fetch failed for {symbol}: {e}")
            return self._get_default_fundamental_data()
    
    async def _cache_single_stock_data(self, symbol: str, data: Dict):
        """Cache single stock data for future use"""
        try:
            # Store in MongoDB
            stock_record = {
                "symbol": symbol,
                "last_updated": datetime.now(),
                "basic": data.get("basic", {}),
                "technical": data.get("technical", {}),
                "fundamental": data.get("fundamental", {}),
                "screening_tags": data.get("screening_tags", []),
                "on_demand": True  # Flag to indicate this was fetched on-demand
            }
            
            await self.db.stocks.replace_one(
                {"symbol": symbol},
                stock_record,
                upsert=True
            )
            
            # Store in Redis with shorter TTL (2 hours for on-demand)
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                
                if data.get("basic"):
                    pipe.hset(f"stock:{symbol}:basic", mapping={k: str(v) for k, v in data["basic"].items()})
                    pipe.expire(f"stock:{symbol}:basic", 7200)  # 2 hours
                
                if data.get("technical"):
                    tech_data = data["technical"].copy()
                    if "support_levels" in tech_data:
                        tech_data["support_levels"] = json.dumps(tech_data["support_levels"])
                    if "resistance_levels" in tech_data:
                        tech_data["resistance_levels"] = json.dumps(tech_data["resistance_levels"])
                    
                    pipe.hset(f"stock:{symbol}:technical", mapping={k: str(v) for k, v in tech_data.items()})
                    pipe.expire(f"stock:{symbol}:technical", 7200)  # 2 hours
                
                if data.get("fundamental"):
                    pipe.hset(f"stock:{symbol}:fundamental", mapping={k: str(v) for k, v in data["fundamental"].items()})
                    pipe.expire(f"stock:{symbol}:fundamental", 7200)  # 2 hours
                
                await pipe.execute()
            
            logger.info(f"✅ Cached on-demand data for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache data for {symbol}: {e}")
    
    async def _track_symbol_request(self, symbol: str):
        """Track symbol requests to identify popular symbols for permanent addition"""
        try:
            if self.redis_client:
                # Increment request counter
                count = await self.redis_client.incr(f"symbol_requests:{symbol}")
                await self.redis_client.expire(f"symbol_requests:{symbol}", 2592000)  # 30 days
                
                # If requested frequently, consider adding to permanent universe
                if count >= 5:  # Requested 5+ times
                    logger.info(f"🔥 {symbol} requested {count} times - consider adding to permanent universe")
                    
                    # Add to pending additions list
                    await self.redis_client.sadd("pending_universe_additions", symbol)
        except Exception as e:
            logger.error(f"❌ Error tracking symbol request for {symbol}: {e}")
    
    async def get_crypto_data_with_fallback(self, symbol: str) -> Dict[str, Any]:
        """Get cryptocurrency data (all cryptos supported via API)"""
        try:
            symbol = symbol.upper()
            
            # Check if in our crypto universe first
            if symbol in self.crypto_universe:
                # Get from cache if available
                cached_crypto = await self._get_cached_crypto_data(symbol)
                if cached_crypto:
                    return {
                        "symbol": symbol,
                        "data_source": "cached_crypto",
                        "data": cached_crypto,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Fetch from crypto API (CoinGecko or similar)
            crypto_data = await self._fetch_crypto_data(symbol)
            
            if crypto_data.get("success"):
                # Cache for future use
                await self._cache_crypto_data(symbol, crypto_data)
                
                return {
                    "symbol": symbol,
                    "data_source": "crypto_api",
                    "data": crypto_data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "symbol": symbol,
                    "error": f"Cryptocurrency {symbol} not found",
                    "data_source": "crypto_api_failed"
                }
                
        except Exception as e:
            logger.error(f"❌ Crypto data fetch failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "data_source": "error"
            }
    
    async def _get_cached_crypto_data(self, symbol: str) -> Optional[Dict]:
        """Get cached cryptocurrency data"""
        try:
            if self.redis_client:
                cached_json = await self.redis_client.get(f"crypto:{symbol}")
                if cached_json:
                    return json.loads(cached_json)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting cached crypto data for {symbol}: {e}")
            return None
    
    async def _fetch_crypto_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch cryptocurrency data from free API (CoinGecko)"""
        try:
            # Use CoinGecko free API (no key required)
            async with aiohttp.ClientSession() as session:
                # First, get coin ID from symbol
                url = "https://api.coingecko.com/api/v3/coins/list"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        coins_list = await response.json()
                        coin_id = None
                        
                        for coin in coins_list:
                            if coin.get("symbol", "").upper() == symbol:
                                coin_id = coin.get("id")
                                break
                        
                        if not coin_id:
                            return {"success": False, "error": f"Coin {symbol} not found"}
                        
                        # Get detailed coin data
                        detail_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                        async with session.get(detail_url, timeout=10) as detail_response:
                            if detail_response.status == 200:
                                data = await detail_response.json()
                                market_data = data.get("market_data", {})
                                
                                return {
                                    "success": True,
                                    "name": data.get("name", ""),
                                    "symbol": symbol,
                                    "price_usd": market_data.get("current_price", {}).get("usd", 0),
                                    "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                                    "volume_24h": market_data.get("total_volume", {}).get("usd", 0),
                                    "change_24h": market_data.get("price_change_percentage_24h", 0),
                                    "change_7d": market_data.get("price_change_percentage_7d", 0),
                                    "change_30d": market_data.get("price_change_percentage_30d", 0),
                                    "rank": data.get("market_cap_rank", 0),
                                    "circulating_supply": market_data.get("circulating_supply", 0),
                                    "total_supply": market_data.get("total_supply", 0),
                                    "max_supply": market_data.get("max_supply", 0),
                                    "ath": market_data.get("ath", {}).get("usd", 0),
                                    "atl": market_data.get("atl", {}).get("usd", 0)
                                }
            
            return {"success": False, "error": "API request failed"}
        except Exception as e:
            logger.error(f"❌ Crypto API request failed for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _cache_crypto_data(self, symbol: str, data: Dict):
        """Cache cryptocurrency data"""
        try:
            if self.redis_client:
                # Cache with 5-minute TTL for crypto (volatile)
                await self.redis_client.setex(
                    f"crypto:{symbol}",
                    300,  # 5 minutes
                    json.dumps(data)
                )
                logger.info(f"✅ Cached crypto data for {symbol}")
        except Exception as e:
            logger.error(f"❌ Failed to cache crypto data for {symbol}: {e}")
    
    async def get_pending_universe_additions(self) -> List[str]:
        """Get list of symbols that should be added to permanent universe"""
        try:
            if self.redis_client:
                pending = await self.redis_client.smembers("pending_universe_additions")
                return list(pending) if pending else []
            return []
        except Exception as e:
            logger.error(f"❌ Error getting pending additions: {e}")
            return []# services/backgroundjob/data_pipeline.py
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis
import schedule
import time
from threading import Thread

logger = logging.getLogger(__name__)

class BackgroundDataPipeline:
    """Background job system for daily stock data refresh and weekly cleanup"""
    
    def __init__(self, mongodb_url: str, redis_url: str, eodhd_api_key: str, ta_service_url: str):
        self.mongodb_url = mongodb_url
        self.redis_url = redis_url
        self.eodhd_api_key = eodhd_api_key
        self.ta_service_url = ta_service_url
        
        # Database connections
        self.mongo_client = None
        self.db = None
        self.redis_client = None
        
        # Global crypto markets access
        self.crypto_universe = self._build_crypto_universe()
        
        # Combined universe for total tracking
        self.total_universe_size = len(self.stock_universe) + len(self.crypto_universe)
        
        # Job status tracking
        self.job_status = {
            "last_daily_run": None,
            "last_weekly_cleanup": None,
            "daily_job_running": False,
            "cleanup_job_running": False,
            "total_stocks_cached": 0,
            "last_error": None,
            "stocks_processed": {
                "basic_data": 0,
                "technical_data": 0,
                "fundamental_data": 0
            }
        }
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(self.mongodb_url)
            self.db = self.mongo_client.trading_bot
            
            # Redis connection
            self.redis_client = aioredis.from_url(self.redis_url)
            
            # Test connections
            await self.db.command("ping")
            await self.redis_client.ping()
            
            logger.info("✅ Background pipeline initialized - DB connections ready")
            
            # Create indexes for efficient querying
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"❌ Background pipeline initialization failed: {e}")
            raise
    
    def _build_stock_universe(self) -> List[str]:
        """Build comprehensive stock universe - 1000+ stocks with emphasis on Tech/Quantum/Crypto"""
        return [
            # === QUANTUM COMPUTING PURE PLAYS ===
            "IONQ",     # IonQ - Leading quantum computing
            "RGTI",     # Rigetti Computing - Quantum cloud services
            "QBTS",     # D-Wave Quantum - Quantum annealing
            "QUBT",     # Quantum Computing Inc - Quantum software
            "ARQQ",     # Arqit Quantum - Quantum encryption
            "QTUM",     # Defiance Quantum ETF
            "QNGY",     # Quantum Energy Partners
            
            # === CRYPTO & BLOCKCHAIN COMPLETE ===
            # Major Crypto Exchanges
            "COIN",     # Coinbase
            "HOOD",     # Robinhood (crypto enabled)
            
            # Bitcoin Mining (All Major Players)
            "MARA",     # Marathon Digital Holdings
            "RIOT",     # Riot Platforms
            "CLSK",     # CleanSpark
            "HUT",      # Hut 8 Mining Corp
            "BITF",     # Bitfarms
            "BTBT",     # Bit Digital
            "ANY",      # Sphere 3D (Bitcoin mining)
            "CAN",      # Canaan Inc (mining hardware)
            "EBON",     # Ebang International
            "SOS",      # SOS Limited
            "GREE",     # Greenidge Generation
            "SPRT",     # Support.com/Greenidge
            "WULF",     # TeraWulf Inc
            "CORZ",     # Core Scientific
            "IREN",     # Iris Energy
            "CIFR",     # Cipher Mining
            "GRIID",    # Griid Infrastructure
            
            # Crypto Financial Services
            "SQ",       # Block (Square) - Bitcoin treasury
            "PYPL",     # PayPal - Crypto services
            "SOFI",     # SoFi - Crypto trading
            "SI",       # Silvergate Capital (crypto banking)
            "COIN",     # Coinbase
            "MSTR",     # MicroStrategy - Bitcoin treasury
            
            # === AI & MACHINE LEARNING (EXPANDED) ===
            # AI Pure Plays
            "AI",       # C3.ai
            "BBAI",     # BigBear.ai
            "SOUN",     # SoundHound AI
            "UPST",     # Upstart (AI lending)
            "PATH",     # UiPath (RPA)
            "AMBA",     # Ambarella (AI vision)
            "SMCI",     # Super Micro Computer
            "VEEV",     # Veeva Systems
            
            # AI Chips & Hardware  
            "NVDA",     # NVIDIA
            "AMD",      # Advanced Micro Devices
            "INTC",     # Intel
            "QCOM",     # Qualcomm
            "AVGO",     # Broadcom
            "MRVL",     # Marvell Technology
            "TXN",      # Texas Instruments
            "ADI",      # Analog Devices
            "XLNX",     # Xilinx (now AMD)
            "LRCX",     # Lam Research
            "AMAT",     # Applied Materials
            "KLAC",     # KLA Corporation
            
            # === MEGA CAP TECH (FAANG+) ===
            "AAPL",     # Apple
            "MSFT",     # Microsoft
            "GOOGL",    # Alphabet Class A
            "GOOG",     # Alphabet Class C
            "AMZN",     # Amazon
            "META",     # Meta Platforms
            "TSLA",     # Tesla
            "NFLX",     # Netflix
            "ORCL",     # Oracle
            "CRM",      # Salesforce
            "ADBE",     # Adobe
            "NOW",      # ServiceNow
            "INTU",     # Intuit
            "IBM",      # IBM
            
            # === CLOUD & ENTERPRISE SOFTWARE ===
            "SNOW",     # Snowflake
            "DDOG",     # Datadog
            "CRWD",     # CrowdStrike
            "NET",      # Cloudflare
            "ZS",       # Zscaler
            "OKTA",     # Okta
            "PANW",     # Palo Alto Networks
            "FTNT",     # Fortinet
            "CYBR",     # CyberArk
            "S",        # SentinelOne
            "TENB",     # Tenable
            "RPD",      # Rapid7
            "QLYS",     # Qualys
            "PING",     # Ping Identity
            "CHKP",     # Check Point Software
            
            # === SAAS & PRODUCTIVITY ===
            "TEAM",     # Atlassian
            "WDAY",     # Workday
            "DOCU",     # DocuSign
            "ZM",       # Zoom
            "SLACK",    # Slack (now part of CRM)
            "BOX",      # Box
            "DBX",      # Dropbox
            "FIVN",     # Five9
            "TWLO",     # Twilio
            "SEND",     # SendGrid (now part of TWLO)
            "PLAN",     # Anaplan
            "SUMO",     # Sumo Logic
            "NEWR",     # New Relic
            "SPLK",     # Splunk
            "ESTC",     # Elastic
            "MDB",      # MongoDB
            
            # === SEMICONDUCTOR ECOSYSTEM ===
            "TSM",      # Taiwan Semiconductor
            "ASML",     # ASML Holding
            "MU",       # Micron Technology
            "WDC",      # Western Digital
            "STX",      # Seagate Technology
            "NXPI",     # NXP Semiconductors
            "MCHP",     # Microchip Technology
            "ON",       # ON Semiconductor
            "SWKS",     # Skyworks Solutions
            "QRVO",     # Qorvo
            "CRUS",     # Cirrus Logic
            "MPWR",     # Monolithic Power Systems
            "POWER",    # Tower Semiconductor
            
            # === ELECTRIC VEHICLES & CLEAN ENERGY ===
            # EV Manufacturers
            "TSLA",     # Tesla
            "RIVN",     # Rivian
            "LCID",     # Lucid Motors
            "NIO",      # NIO
            "XPEV",     # XPeng
            "LI",       # Li Auto
            "FSR",      # Fisker
            "NKLA",     # Nikola
            "RIDE",     # Lordstown Motors
            "WKHS",     # Workhorse Group
            "GOEV",     # Canoo
            "ARVL",     # Arrival
            "MULN",     # Mullen Automotive
            "SOLO",     # ElectraMeccanica
            "AYRO",     # AYRO Inc
            
            # EV Charging & Infrastructure
            "CHPT",     # ChargePoint
            "BLNK",     # Blink Charging
            "EVGO",     # EVgo Inc
            "DCFC",     # Tritium DCFC
            "WBX",      # Wallbox
            "FRSX",     # Foresight Autonomous
            "PTRA",     # Proterra
            
            # Solar & Clean Energy
            "ENPH",     # Enphase Energy
            "SEDG",     # SolarEdge Technologies
            "RUN",      # Sunrun
            "NOVA",     # Sunnova Energy
            "SPWR",     # SunPower
            "CSIQ",     # Canadian Solar
            "JKS",      # JinkoSolar
            "DQ",       # Daqo New Energy
            "FSLR",     # First Solar
            "MAXN",     # Maxeon Solar Technologies
            "ARRY",     # Array Technologies
            
            # Energy Storage & Batteries
            "BE",       # Bloom Energy
            "FCEL",     # FuelCell Energy
            "PLUG",     # Plug Power
            "HYLN",     # Hyliion
            "QS",       # QuantumScape
            "STEM",     # Stem Inc
            "NEE",      # NextEra Energy
            "NFE",      # New Fortress Energy
            
            # === FINTECH & DIGITAL PAYMENTS ===
            "V",        # Visa
            "MA",       # Mastercard
            "SQ",       # Block (Square)
            "PYPL",     # PayPal
            "AFRM",     # Affirm
            "UPST",     # Upstart
            "SOFI",     # SoFi Technologies
            "LC",       # LendingClub
            "HOOD",     # Robinhood
            "COIN",     # Coinbase
            "NU",       # Nu Holdings
            "PAGS",     # PagSeguro Digital
            "STNE",     # StoneCo
            "MELI",     # MercadoLibre
            
            # === GAMING & ENTERTAINMENT ===
            "RBLX",     # Roblox
            "UNITY",    # Unity Software
            "EA",       # Electronic Arts
            "ATVI",     # Activision Blizzard
            "TTWO",     # Take-Two Interactive
            "ZNGA",     # Zynga
            "NTES",     # NetEase
            "BILI",     # Bilibili
            "HUYA",     # Huya Inc
            "DOYU",     # DouYu International
            
            # === BIOTECH & GENOMICS ===
            "MRNA",     # Moderna
            "BNTX",     # BioNTech
            "NVAX",     # Novavax
            "GILD",     # Gilead Sciences
            "BIIB",     # Biogen
            "VRTX",     # Vertex Pharmaceuticals
            "REGN",     # Regeneron
            "BMRN",     # BioMarin Pharmaceutical
            "SRPT",     # Sarepta Therapeutics
            "RARE",     # Ultragenyx Pharmaceutical
            "FOLD",     # Amicus Therapeutics
            "BLUE",     # bluebird bio
            "EDIT",     # Editas Medicine
            "NTLA",     # Intellia Therapeutics
            "CRSP",     # CRISPR Therapeutics
            "BEAM",     # Beam Therapeutics
            "PACB",     # Pacific Biosciences
            "ILMN",     # Illumina
            "10X",      # 10x Genomics
            
            # === TRADITIONAL TECH ===
            "CSCO",     # Cisco Systems
            "ORCL",     # Oracle
            "IBM",      # IBM
            "HPQ",      # HP Inc
            "HPE",      # Hewlett Packard Enterprise
            "DELL",     # Dell Technologies
            "VMW",      # VMware
            "NTAP",     # NetApp
            "WDC",      # Western Digital
            "STX",      # Seagate Technology
            
            # === FINANCIAL SERVICES (TOP TIER) ===
            "BRK.B",    # Berkshire Hathaway
            "JPM",      # JPMorgan Chase
            "BAC",      # Bank of America
            "WFC",      # Wells Fargo
            "GS",       # Goldman Sachs
            "MS",       # Morgan Stanley
            "C",        # Citigroup
            "AXP",      # American Express
            "SPGI",     # S&P Global
            "BLK",      # BlackRock
            "SCHW",     # Charles Schwab
            "USB",      # U.S. Bancorp
            "PNC",      # PNC Financial
            "COF",      # Capital One
            "TFC",      # Truist Financial
            "CME",      # CME Group
            "ICE",      # Intercontinental Exchange
            "MCO",      # Moody's Corporation
            "MMC",      # Marsh & McLennan
            "AON",      # Aon
            
            # === HEALTHCARE & PHARMA (MAJOR) ===
            "UNH",      # UnitedHealth Group
            "JNJ",      # Johnson & Johnson
            "PFE",      # Pfizer
            "ABBV",     # AbbVie
            "LLY",      # Eli Lilly
            "MRK",      # Merck
            "TMO",      # Thermo Fisher Scientific
            "ABT",      # Abbott Laboratories
            "DHR",      # Danaher
            "CVS",      # CVS Health
            "BMY",      # Bristol Myers Squibb
            "AMGN",     # Amgen
            "MDT",      # Medtronic
            "CI",       # Cigna
            "HUM",      # Humana
            "ZTS",      # Zoetis
            "ISRG",     # Intuitive Surgical
            "SYK",      # Stryker
            "BSX",      # Boston Scientific
            "EW",       # Edwards Lifesciences
            
            # === CONSUMER BRANDS ===
            "HD",       # Home Depot
            "MCD",      # McDonald's
            "DIS",      # Walt Disney
            "NKE",      # Nike
            "SBUX",     # Starbucks
            "LOW",      # Lowe's
            "TJX",      # TJX Companies
            "BKNG",     # Booking Holdings
            "CMG",      # Chipotle Mexican Grill
            "MAR",      # Marriott International
            "HLT",      # Hilton Worldwide
            "GM",       # General Motors
            "F",        # Ford Motor
            "CCL",      # Carnival Corporation
            "RCL",      # Royal Caribbean
            "NCLH",     # Norwegian Cruise Line
            "DAL",      # Delta Air Lines
            "UAL",      # United Airlines
            "AAL",      # American Airlines
            "LUV",      # Southwest Airlines
            "ABNB",     # Airbnb
            "UBER",     # Uber Technologies
            "LYFT",     # Lyft
            "DASH",     # DoorDash
            
            # === E-COMMERCE & MARKETPLACE ===
            "AMZN",     # Amazon
            "SHOP",     # Shopify
            "ETSY",     # Etsy
            "CHWY",     # Chewy
            "W",        # Wayfair
            "OSTK",     # Overstock.com
            "CVNA",     # Carvana
            "VIPS",     # Vipshop Holdings
            "JD",       # JD.com
            "PDD",      # PDD Holdings
            "BABA",     # Alibaba Group
            "SE",       # Sea Limited
            
            # === SOCIAL MEDIA & COMMUNICATION ===
            "META",     # Meta Platforms
            "SNAP",     # Snap Inc
            "PINS",     # Pinterest
            "TWTR",     # Twitter (if still public)
            "ROKU",     # Roku
            "SPOT",     # Spotify
            "YELP",     # Yelp
            "BMBL",     # Bumble
            "MTCH",     # Match Group
            "IAC",      # IAC/InterActiveCorp
            "T",        # AT&T
            "VZ",       # Verizon
            "CMCSA",    # Comcast
            "CHTR",     # Charter Communications
            "DIS",      # Walt Disney
            "PARA",     # Paramount Global
            "WBD",      # Warner Bros. Discovery
            "LYV",      # Live Nation Entertainment
            
            # === MEME STOCKS & RETAIL FAVORITES ===
            "GME",      # GameStop
            "AMC",      # AMC Entertainment
            "BB",       # BlackBerry
            "NOK",      # Nokia
            "WISH",     # ContextLogic (Wish)
            "CLOV",     # Clover Health
            "SPCE",     # Virgin Galactic
            "PLTR",     # Palantir Technologies
            "HOOD",     # Robinhood Markets
            
            # === ENERGY & TRADITIONAL ===
            "XOM",      # Exxon Mobil
            "CVX",      # Chevron
            "COP",      # ConocoPhillips
            "EOG",      # EOG Resources
            "SLB",      # Schlumberger
            "MPC",      # Marathon Petroleum
            "VLO",      # Valero Energy
            "PSX",      # Phillips 66
            "PXD",      # Pioneer Natural Resources
            "KMI",      # Kinder Morgan
            "WMB",      # Williams Companies
            "OKE",      # ONEOK
            "EPD",      # Enterprise Products Partners
            "ET",       # Energy Transfer
            "MPLX",     # MPLX LP
            "BKR",      # Baker Hughes
            "HAL",      # Halliburton
            "DVN",      # Devon Energy
            "FANG",     # Diamondback Energy
            "MRO",      # Marathon Oil
            
            # === INDUSTRIALS & AEROSPACE ===
            "BA",       # Boeing
            "CAT",      # Caterpillar
            "HON",      # Honeywell
            "UPS",      # United Parcel Service
            "RTX",      # Raytheon Technologies
            "LMT",      # Lockheed Martin
            "GE",       # General Electric
            "MMM",      # 3M Company
            "DE",       # Deere & Company
            "EMR",      # Emerson Electric
            "ITW",      # Illinois Tool Works
            "PH",       # Parker-Hannifin
            "ETN",      # Eaton Corporation
            "ROK",      # Rockwell Automation
            "DOV",      # Dover Corporation
            "XYL",      # Xylem
            "PCAR",     # PACCAR
            "CMI",      # Cummins
            "IR",       # Ingersoll Rand
            "OTIS",     # Otis Worldwide
            
            # === MATERIALS & CHEMICALS ===
            "LIN",      # Linde
            "APD",      # Air Products and Chemicals
            "ECL",      # Ecolab
            "SHW",      # Sherwin-Williams
            "DD",       # DuPont
            "DOW",      # Dow Inc
            "NEM",      # Newmont Corporation
            "FCX",      # Freeport-McMoRan
            "FMC",      # FMC Corporation
            "ALB",      # Albemarle Corporation
            "CE",       # Celanese Corporation
            "MLM",      # Martin Marietta Materials
            "VMC",      # Vulcan Materials
            "NUE",      # Nucor Corporation
            "STLD",     # Steel Dynamics
            "X",        # United States Steel
            "CLF",      # Cleveland-Cliffs
            "AA",       # Alcoa Corporation
            
            # === UTILITIES & INFRASTRUCTURE ===
            "NEE",      # NextEra Energy
            "DUK",      # Duke Energy
            "SO",       # Southern Company
            "D",        # Dominion Energy
            "EXC",      # Exelon Corporation
            "XEL",      # Xcel Energy
            "SRE",      # Sempra Energy
            "PEG",      # Public Service Enterprise Group
            "ES",       # Eversource Energy
            "AWK",      # American Water Works
            "AEP",      # American Electric Power
            "EIX",      # Edison International
            "WEC",      # WEC Energy Group
            "DTE",      # DTE Energy
            "ED",       # Consolidated Edison
            "ETR",      # Entergy Corporation
            "FE",       # FirstEnergy Corp
            "CNP",      # CenterPoint Energy
            "NI",       # NiSource
            "LNT",      # Alliant Energy
            
            # === CONSUMER STAPLES ===
            "PG",       # Procter & Gamble
            "KO",       # Coca-Cola
            "PEP",      # PepsiCo
            "WMT",      # Walmart
            "COST",     # Costco
            "MDLZ",     # Mondelez International
            "CL",       # Colgate-Palmolive
            "KMB",      # Kimberly-Clark
            "GIS",      # General Mills
            "K",        # Kellogg Company
            "SYY",      # Sysco Corporation
            "ADM",      # Archer-Daniels-Midland
            "TSN",      # Tyson Foods
            "CAG",      # Conagra Brands
            "CPB",      # Campbell Soup
            "HRL",      # Hormel Foods
            "MKC",      # McCormick & Company
            "SJM",      # Smucker Company
        ]
    
    def _build_crypto_universe(self) -> List[str]:
        """Build comprehensive cryptocurrency universe - Top 500+ cryptos"""
        return [
            # === TOP 50 CRYPTOCURRENCIES BY MARKET CAP ===
            "BTC",      # Bitcoin
            "ETH",      # Ethereum
            "USDT",     # Tether
            "BNB",      # Binance Coin
            "SOL",      # Solana
            "USDC",     # USD Coin
            "XRP",      # Ripple
            "STETH",    # Lido Staked Ether
            "TON",      # Toncoin
            "DOGE",     # Dogecoin
            "ADA",      # Cardano
            "AVAX",     # Avalanche
            "SHIB",     # Shiba Inu
            "TRX",      # TRON
            "DOT",      # Polkadot
            "LINK",     # Chainlink
            "MATIC",    # Polygon
            "WBTC",     # Wrapped Bitcoin
            "ICP",      # Internet Computer
            "BCH",      # Bitcoin Cash
            "NEAR",     # NEAR Protocol
            "UNI",      # Uniswap
            "LTC",      # Litecoin
            "LEO",      # UNUS SED LEO
            "DAI",      # Dai
            "FET",      # Fetch.ai
            "APT",      # Aptos
            "ETC",      # Ethereum Classic
            "XMR",      # Monero
            "RENDER",   # Render Token
            "HBAR",     # Hedera
            "STX",      # Stacks
            "IMX",      # Immutable X
            "CRO",      # Cronos
            "VET",      # VeChain
            "MNT",      # Mantle
            "INJ",      # Injective
            "ARB",      # Arbitrum
            "OP",       # Optimism
            "ATOM",     # Cosmos
            "FDUSD",    # First Digital USD
            "TAO",      # Bittensor
            "AAVE",     # Aave
            "GRT",      # The Graph
            "MKR",      # Maker
            "ALGO",     # Algorand
            "WIF",      # dogwifhat
            "BONK",     # Bonk
            "PEPE",     # Pepe
            "FLOKI",    # FLOKI
            
            # === MAJOR ALTCOINS (51-150) ===
            "FIL",      # Filecoin
            "LDO",      # Lido DAO
            "THETA",    # THETA
            "FLOW",     # Flow
            "XLM",      # Stellar
            "SAND",     # The Sandbox
            "MANA",     # Decentraland
            "AXS",      # Axie Infinity
            "CHZ",      # Chiliz
            "EGLD",     # MultiversX
            "KLAY",     # Klaytn
            "RUNE",     # THORChain
            "FTM",      # Fantom
            "XTZ",      # Tezos
            "GALA",     # Gala
            "APE",      # ApeCoin
            "MINA",     # Mina
            "NEO",      # Neo
            "EOS",      # EOS
            "IOTA",     # IOTA
            "QNT",      # Quant
            "ZIL",      # Zilliqa
            "ENJ",      # Enjin Coin
            "LRC",      # Loopring
            "BAT",      # Basic Attention Token
            "ZEC",      # Zcash
            "COMP",     # Compound
            "YFI",      # yearn.finance
            "SNX",      # Synthetix
            "1INCH",    # 1inch
            "CRV",      # Curve DAO Token
            "SUSHI",    # SushiSwap
            "UMA",      # UMA
            "REN",      # Ren
            "KNC",      # Kyber Network Crystal
            "ZRX",      # 0x
            "REP",      # Augur
            "STORJ",    # Storj
            "NMR",      # Numeraire
            "MLN",      # Melon
            "ANT",      # Aragon
            "GNT",      # Golem (Legacy)
            "BAL",      # Balancer
            "BAND",     # Band Protocol
            "OCEAN",    # Ocean Protocol
            "OMG",      # OMG Network
            "ICX",      # ICON
            "QTUM",     # Qtum
            "LSK",      # Lisk
            "STRAT",    # Stratis
            
            # === DEFI TOKENS (100+) ===
            "UNI",      # Uniswap
            "AAVE",     # Aave
            "COMP",     # Compound
            "YFI",      # yearn.finance
            "SNX",      # Synthetix
            "MKR",      # Maker
            "CRV",      # Curve DAO Token
            "SUSHI",    # SushiSwap
            "1INCH",    # 1inch
            "BAL",      # Balancer
            "UMA",      # UMA
            "REN",      # Ren
            "KNC",      # Kyber Network Crystal
            "ZRX",      # 0x
            "REP",      # Augur
            "NMR",      # Numeraire
            "MLN",      # Melon
            "ANT",      # Aragon
            "BAND",     # Band Protocol
            "OCEAN",    # Ocean Protocol
            "ALPHA",    # Alpha Finance Lab
            "DYDX",     # dYdX
            "ENS",      # Ethereum Name Service
            "LOOKS",    # LooksRare
            "CVX",      # Convex Finance
            "FXS",      # Frax Share
            "ALCX",     # Alchemix
            "BADGER",   # Badger DAO
            "FARM",     # Harvest Finance
            "CREAM",    # Cream Finance
            "PICKLE",   # Pickle Finance
            "BOBA",     # Boba Network
            "PERP",     # Perpetual Protocol
            "GMX",      # GMX
            "JOE",      # TraderJoe
            "TIME",     # Wonderland
            "SPELL",    # Spell Token
            "ICE",      # IceToken
            "TOMB",     # Tomb Finance
            "SPIRIT",   # SpiritSwap
            "BOO",      # Spookyswap
            "LQTY",     # Liquity
            "LUSD",     # Liquity USD
            "RAI",      # Rai Reflex Index
            "OHM",      # Olympus
            "KLIMA",    # Klima DAO
            "BTRFLY",   # Redacted Cartel
            "FPIS",     # Frax Price Index Share
            "CRV",      # Curve DAO Token
            "CVX",      # Convex Finance
            "LDO",      # Lido DAO
            "RPL",      # Rocket Pool
            "ANKR",     # Ankr
            
            # === GAMING & METAVERSE (50+) ===
            "AXS",      # Axie Infinity
            "SAND",     # The Sandbox
            "MANA",     # Decentraland
            "ENJ",      # Enjin Coin
            "GALA",     # Gala
            "APE",      # ApeCoin
            "CHZ",      # Chiliz
            "FLOW",     # Flow
            "IMX",      # Immutable X
            "WAX",      # WAX
            "ALICE",    # MyNeighborAlice
            "TLM",      # Alien Worlds
            "SLP",      # Smooth Love Potion
            "ILV",      # Illuvium
            "STAR",     # StarAtlas
            "POLIS",    # Star Atlas DAO
            "YGG",      # Yield Guild Games
            "GHST",     # Aavegotchi
            "REVV",     # REVV
            "PLA",      # PlayDapp
            "TOWER",    # Tower
            "UFO",      # UFO Gaming
            "GODS",     # Gods Unchained
            "SKILL",    # CryptoBlades
            "DG",       # Decentral Games
            "NFTX",     # NFTX
            "RARI",     # Rarible
            "SUPER",    # SuperFarm
            "DEGO",     # Dego Finance
            "DOSE",     # Dose Token
            "NAKA",     # Nakamoto Games
            "CWAR",     # Cryowar
            "SIPHER",   # Sipher
            "MONI",     # Monsta Infinite
            "HERO",     # Step Hero
            "JEWEL",    # DeFi Kingdoms
            "CRYSTAL",  # CrystalVerse
            "REALM",    # Realm
            "GAM",      # Gamium
            "GAFI",     # GameFi
            "CREO",     # Creo Engine
            "VRA",      # Verasity
            "ERN",      # Ethernity Chain
            "WHALE",    # Whale
            "NFTB",     # NFTb
            "RFOX",     # RedFOX Labs
            "OVR",      # Ovr
            "MOBOX",    # Mobox
            "HIGH",     # Highstreet
            "DPET",     # My DeFi Pet
            "BYG",      # Baby Doge Game
            
            # === LAYER 1 BLOCKCHAINS (30+) ===
            "SOL",      # Solana
            "ADA",      # Cardano
            "AVAX",     # Avalanche
            "DOT",      # Polkadot
            "NEAR",     # NEAR Protocol
            "ATOM",     # Cosmos
            "ALGO",     # Algorand
            "XTZ",      # Tezos
            "EGLD",     # MultiversX
            "HBAR",     # Hedera
            "FTM",      # Fantom
            "ONE",      # Harmony
            "ZIL",      # Zilliqa
            "VET",      # VeChain
            "ICX",      # ICON
            "QTUM",     # Qtum
            "LSK",      # Lisk
            "WAVES",    # Waves
            "KSM",      # Kusama
            "ROSE",     # Oasis Network
            "CKB",      # Nervos Network
            "CELO",     # Celo
            "KAVA",     # Kava
            "SECRET",   # Secret
            "SCRT",     # Secret Network
            "OSMO",     # Osmosis
            "JUNO",     # Juno Network
            "EVMOS",    # Evmos
            "REGEN",    # Regen Network
            "ROWAN",    # Sifchain
            "IOV",      # Starname
            "AKT",      # Akash Network
            "DVPN",     # Sentinel
            "NGM",      # e-Money
            "XPRT",     # Persistence
            "IRIS",     # IRISnet
            "CRE",      # Crescent Network
            "CMDX",     # Comdex
            "HUAHUA",   # Chihuahua
            "BTSG",     # BitSong
            
            # === LAYER 2 & SCALING (20+) ===
            "MATIC",    # Polygon
            "ARB",      # Arbitrum
            "OP",       # Optimism
            "LRC",      # Loopring
            "IMX",      # Immutable X
            "BOBA",     # Boba Network
            "METIS",    # Metis
            "SKL",      # SKALE Network
            "OMG",      # OMG Network
            "RDN",      # Raiden Network Token
            "CELR",     # Celer Network
            "HOT",      # Holo
            "IOTX",     # IoTeX
            "ZK",       # zkSync
            "STRK",     # StarkNet
            "MINA",     # Mina
            "COTI",     # COTI
            "LTO",      # LTO Network
            "DUSK",     # Dusk Network
            "BEAM",     # Beam
            "GRIN",     # Grin
            
            # === MEME COINS & COMMUNITY (50+) ===
            "DOGE",     # Dogecoin
            "SHIB",     # Shiba Inu
            "PEPE",     # Pepe
            "FLOKI",    # FLOKI
            "WIF",      # dogwifhat
            "BONK",     # Bonk
            "BABYDOGE", # Baby Doge Coin
            "ELON",     # Dogelon Mars
            "AKITA",    # Akita Inu
            "KISHU",    # Kishu Inu
            "HOGE",     # Hoge Finance
            "HOKK",     # Hokkaido Inu
            "CATGIRL",  # Catgirl
            "CATE",     # CateCoin
            "SAITAMA",  # Saitama
            "LUFFY",    # Luffy
            "GOKU",     # Goku
            "KUMA",     # Kuma Inu
            "PITBULL",  # Pitbull
            "ELONGATE", # ElonGate
            "SAFEMOON", # SafeMoon
            "CUMROCKET",# CumRocket
            "ASS",      # Australian Safe Shepherd
            "PUSSY",    # PussyDAO
            "BOOBA",    # Booba Token
            "MILF",     # MILF Token
            "PORN",     # PornRocket
            "XXX",      # XXXNifty
            "NSFW",     # Pleasure Coin
            "TABOO",    # TABOO Token
            "SPANK",    # SpankChain
            "CUM",      # Cuminu
            "TITS",     # TitsCoin
            "DRUGS",    # Drugs Token
            "WEED",     # WeedCash
            "METH",     # Meth Token
            "CRACK",    # CrackToken
            "BEER",     # BeerCoin
            "WINE",     # WineChain
            "PIZZA",    # PizzaCoin
            "BURGER",   # BurgerCities
            "TACO",     # TacoCoin
            "SUSHI",    # SushiSwap
            "CAKE",     # PancakeSwap
            "DONUT",    # Donut
            "BANANA",   # ApeSwap Finance
            "CHERRY",   # Cherry Network
            "APPLE",    # Apple Network
            "LEMON",    # Lemon
            "ORANGE",   # Orange
            "GRAPE",    # Grape Protocol
            "BERRY",    # Berry Data
            "HONEY",    # Honey
            "SUGAR",    # Sugar Kingdom Odyssey
            "SALT",     # Salt
            
            # === AI & DATA TOKENS (30+) ===
            "FET",      # Fetch.ai
            "OCEAN",    # Ocean Protocol
            "SingularityNET", # SingularityNET (AGIX)
            "RNDR",     # Render Token
            "GRT",      # The Graph
            "LINK",     # Chainlink
            "BAND",     # Band Protocol
            "API3",     # API3
            "TRB",      # Tellor
            "NMR",      # Numeraire
            "MLN",      # Melon
            "ARKM",     # Arkham
            "CTI",      # ClinTex
            "MDT",      # Measurable Data Token
            "DATA",     # Data Economy Index
            "DIA",      # DIA
            "DOS",      # DOS Network
            "NEST",     # NEST Protocol
            "RAZOR",    # Razor Network
            "FLUX",     # Flux
            "RLC",      # iExec RLC
            "PHALA",    # Phala Network
            "AI",       # SingularityNET
            "AIOZ",     # AIOZ Network
            "ALEPH",    # Aleph.im
            "CUDOS",    # Cudos
            "CARTESI",  # Cartesi
            "FORTE",    # Forte
            "EFFECT",   # Effect Network
            "DEEP",     # Deep Brain Chain
            
            # === PRIVACY COINS (15+) ===
            "XMR",      # Monero
            "ZEC",      # Zcash
            "DASH",     # Dash
            "FIRO",     # Firo
            "BEAM",     # Beam
            "GRIN",     # Grin
            "HAVEN",    # Haven Protocol
            "ARRR",     # Pirate Chain
            "DERO",     # Dero
            "RYO",      # Ryo Currency
            "TURTLE",   # TurtleCoin
            "SUMO",     # Sumokoin
            "AEON",     # Aeon
            "LOKI",     # Loki
            "OXEN",     # Oxen
            
            # === ENTERPRISE & INSTITUTIONAL (20+) ===
            "XRP",      # Ripple
            "XLM",      # Stellar
            "HBAR",     # Hedera
            "IOTA",     # IOTA
            "VET",      # VeChain
            "QNT",      # Quant
            "LCX",      # LCX
            "POWR",     # Power Ledger
            "WTC",      # Waltonchain
            "AMB",      # Ambrosus
            "TEL",      # Telcoin
            "TFUEL",    # Theta Fuel
            "HOT",      # Holo
            "REQ",      # Request Network
            "ORN",      # Orion Protocol
            "POLY",     # Polymath
            "STO",      # StoreCoin
            "SWTH",     # Switcheo
            "ZCN",      # 0Chain
            "POLS",     # Polkastarter
        ]
    
    async def _create_indexes(self):
        """Create MongoDB indexes for efficient screening queries"""
        try:
            # Compound indexes for common screening patterns
            await self.db.stocks.create_index([
                ("basic.market_cap", -1),
                ("basic.volume", -1)
            ])
            
            await self.db.stocks.create_index([
                ("technical.rsi", 1),
                ("basic.sector", 1)
            ])
            
            await self.db.stocks.create_index([
                ("fundamental.pe", 1),
                ("fundamental.roe", -1)
            ])
            
            await self.db.stocks.create_index([
                ("basic.sector", 1),
                ("basic.market_cap", -1)
            ])
            
            await self.db.stocks.create_index([
                ("screening_tags", 1)
            ])
            
            # Single field indexes
            await self.db.stocks.create_index("symbol")
            await self.db.stocks.create_index("last_updated")
            await self.db.stocks.create_index("basic.price")
            
            logger.info("✅ MongoDB indexes created")
            
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")
    
    async def daily_data_refresh_job(self):
        """Main daily job - refresh all stock data at 6 AM"""
        if self.job_status["daily_job_running"]:
            logger.warning("⚠️ Daily job already running, skipping")
            return
        
        start_time = time.time()
        self.job_status["daily_job_running"] = True
        self.job_status["last_error"] = None
        self.job_status["stocks_processed"] = {"basic_data": 0, "technical_data": 0, "fundamental_data": 0}
        
        try:
            logger.info(f"🚀 Starting daily data refresh for {len(self.stock_universe)} stocks")
            
            # Step 1: Fetch basic market data (bulk API call)
            basic_data = await self._fetch_basic_market_data()
            self.job_status["stocks_processed"]["basic_data"] = len(basic_data)
            logger.info(f"✅ Fetched basic data for {len(basic_data)} stocks")
            
            # Step 2: Fetch technical analysis data (batch calls)
            technical_data = await self._fetch_technical_data()
            self.job_status["stocks_processed"]["technical_data"] = len(technical_data)
            logger.info(f"✅ Fetched technical data for {len(technical_data)} stocks")
            
            # Step 3: Fetch fundamental data (individual calls - slower)
            fundamental_data = await self._fetch_fundamental_data()
            self.job_status["stocks_processed"]["fundamental_data"] = len(fundamental_data)
            logger.info(f"✅ Fetched fundamental data for {len(fundamental_data)} stocks")
            
            # Step 4: Combine data and store
            combined_data = await self._combine_stock_data(basic_data, technical_data, fundamental_data)
            
            # Step 5: Store in MongoDB
            await self._store_in_mongodb(combined_data)
            
            # Step 6: Cache in Redis for fast access
            await self._cache_in_redis(combined_data)
            
            # Update job status
            duration = time.time() - start_time
            self.job_status.update({
                "last_daily_run": datetime.now().isoformat(),
                "total_stocks_cached": len(combined_data),
                "daily_job_running": False,
                "last_duration_seconds": round(duration, 2)
            })
            
            logger.info(f"✅ Daily data refresh completed in {duration:.2f}s - {len(combined_data)} stocks updated")
            
        except Exception as e:
            self.job_status["last_error"] = str(e)
            self.job_status["daily_job_running"] = False
            logger.error(f"❌ Daily data refresh failed: {e}")
            raise
    
    async def weekly_cleanup_job(self):
        """Weekly cleanup job - clear old data at 2 AM Sunday"""
        if self.job_status["cleanup_job_running"]:
            logger.warning("⚠️ Cleanup job already running, skipping")
            return
        
        start_time = time.time()
        self.job_status["cleanup_job_running"] = True
        
        try:
            logger.info("🧹 Starting weekly cache cleanup")
            
            # Step 1: Clear Redis stock data
            redis_pattern = "stock:*"
            redis_keys = []
            async for key in self.redis_client.scan_iter(match=redis_pattern):
                redis_keys.append(key)
            
            if redis_keys:
                await self.redis_client.delete(*redis_keys)
                logger.info(f"✅ Cleared {len(redis_keys)} Redis keys")
            
            # Step 2: Clear MongoDB stocks collection
            result = await self.db.stocks.delete_many({})
            logger.info(f"✅ Cleared {result.deleted_count} MongoDB documents")
            
            # Step 3: Clear derived caches
            derived_keys = [
                "sector_performance", "market_context", "screening_stats",
                "popular_stocks", "market_movers", "news:*"
            ]
            
            for pattern in derived_keys:
                if "*" in pattern:
                    # Handle pattern keys
                    keys_to_delete = []
                    async for key in self.redis_client.scan_iter(match=pattern):
                        keys_to_delete.append(key)
                    if keys_to_delete:
                        await self.redis_client.delete(*keys_to_delete)
                else:
                    # Handle exact keys
                    await self.redis_client.delete(pattern)
            
            # Step 4: Recreate indexes (they're cleared with collection)
            await self._create_indexes()
            
            # Update job status
            duration = time.time() - start_time
            self.job_status.update({
                "last_weekly_cleanup": datetime.now().isoformat(),
                "cleanup_job_running": False,
                "cleanup_duration_seconds": round(duration, 2)
            })
            
            logger.info(f"✅ Weekly cleanup completed in {duration:.2f}s")
            
        except Exception as e:
            self.job_status["last_error"] = str(e)
            self.job_status["cleanup_job_running"] = False
            logger.error(f"❌ Weekly cleanup failed: {e}")
            raise
    
    async def _fetch_basic_market_data(self) -> Dict[str, Dict]:
        """Fetch basic market data from EODHD for all stocks"""
        basic_data = {}
        
        try:
            # Use EODHD's real-time bulk endpoint
            async with aiohttp.ClientSession() as session:
                # Process in smaller batches to avoid API limits
                batch_size = 50
                for i in range(0, len(self.stock_universe), batch_size):
                    batch = self.stock_universe[i:i + batch_size]
                    symbols_str = ",".join([f"{symbol}.US" for symbol in batch])
                    
                    url = "https://eodhd.com/api/real-time/bulk"
                    params = {
                        "api_token": self.eodhd_api_key,
                        "symbols": symbols_str,
                        "fmt": "json"
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data:
                                if isinstance(item, dict):
                                    symbol = item.get("code", "").replace(".US", "")
                                    if symbol in self.stock_universe:
                                        basic_data[symbol] = {
                                            "price": float(item.get("close", 0)),
                                            "volume": int(item.get("volume", 0)),
                                            "market_cap": float(item.get("market_cap", 0)),
                                            "change_1d": float(item.get("change_p", 0)),
                                            "high": float(item.get("high", 0)),
                                            "low": float(item.get("low", 0)),
                                            "open": float(item.get("open", 0)),
                                            "previous_close": float(item.get("previous_close", 0)),
                                            "sector": "Technology",  # Default - would need separate call for sectors
                                            "exchange": "NASDAQ"
                                        }
                        else:
                            logger.warning(f"⚠️ EODHD basic data API error: {response.status} for batch starting with {batch[0]}")
                    
                    # Small delay between batches to respect rate limits
                    await asyncio.sleep(0.1)
            
            return basic_data
            
        except Exception as e:
            logger.error(f"❌ Basic market data fetch failed: {e}")
            return {}
    
    async def _fetch_technical_data(self) -> Dict[str, Dict]:
        """Fetch technical analysis data from your TA service or calculate from EODHD"""
        technical_data = {}
        
        try:
            # Option 1: Use your existing TA service
            if self.ta_service_url and self.ta_service_url != "http://localhost:8001":
                async with aiohttp.ClientSession() as session:
                    for symbol in self.stock_universe:
                        try:
                            url = f"{self.ta_service_url}/analyze/{symbol}"
                            params = {"period": "1mo"}
                            
                            async with session.get(url, params=params, timeout=10) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    
                                    # Extract key technical indicators
                                    technical_data[symbol] = {
                                        "rsi": data.get("rsi", 50),
                                        "macd_signal": data.get("macd_signal", "neutral"),
                                        "sma_20": data.get("sma_20", 0),
                                        "sma_50": data.get("sma_50", 0),
                                        "sma_200": data.get("sma_200", 0),
                                        "support_levels": data.get("support_levels", []),
                                        "resistance_levels": data.get("resistance_levels", []),
                                        "trend": data.get("trend", "neutral"),
                                        "volatility": data.get("volatility", 0),
                                        "bollinger_bands": data.get("bollinger_bands", {}),
                                        "volume_sma": data.get("volume_sma", 0)
                                    }
                                else:
                                    # Set default values for failed requests
                                    technical_data[symbol] = self._get_default_technical_data()
                        except Exception as e:
                            logger.warning(f"⚠️ TA data failed for {symbol}: {e}")
                            technical_data[symbol] = self._get_default_technical_data()
                        
                        # Small delay to avoid overwhelming TA service
                        await asyncio.sleep(0.05)
            
            else:
                # Option 2: Generate basic technical data from EODHD historical data
                technical_data = await self._calculate_basic_technical_indicators()
            
            return technical_data
            
        except Exception as e:
            logger.error(f"❌ Technical data fetch failed: {e}")
            return {}
    
    def _get_default_technical_data(self) -> Dict:
        """Get default technical data for failed requests"""
        return {
            "rsi": 50,
            "macd_signal": "neutral",
            "trend": "neutral",
            "sma_20": 0,
            "sma_50": 0,
            "sma_200": 0,
            "support_levels": [],
            "resistance_levels": [],
            "volatility": 0,
            "bollinger_bands": {},
            "volume_sma": 0
        }
    
    async def _calculate_basic_technical_indicators(self) -> Dict[str, Dict]:
        """Calculate basic technical indicators from EODHD historical data"""
        technical_data = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in self.stock_universe:
                    try:
                        # Get 3 months of historical data for calculations
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                        
                        url = f"https://eodhd.com/api/eod/{symbol}.US"
                        params = {
                            'api_token': self.eodhd_api_key,
                            'fmt': 'json',
                            'from': start_date,
                            'to': end_date
                        }
                        
                        async with session.get(url, params=params, timeout=15) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                if data and len(data) > 20:  # Need enough data for calculations
                                    technical_data[symbol] = self._calculate_indicators_from_ohlc(data)
                                else:
                                    technical_data[symbol] = self._get_default_technical_data()
                            else:
                                technical_data[symbol] = self._get_default_technical_data()
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Technical calculation failed for {symbol}: {e}")
                        technical_data[symbol] = self._get_default_technical_data()
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
            
            return technical_data
            
        except Exception as e:
            logger.error(f"❌ Technical calculation failed: {e}")
            return {}
    
    def _calculate_indicators_from_ohlc(self, ohlc_data: List[Dict]) -> Dict:
        """Calculate basic technical indicators from OHLC data"""
        try:
            if not ohlc_data or len(ohlc_data) < 20:
                return self._get_default_technical_data()
            
            # Extract closing prices and volumes
            closes = [float(d.get('adjusted_close', d.get('close', 0))) for d in ohlc_data]
            volumes = [int(d.get('volume', 0)) for d in ohlc_data]
            highs = [float(d.get('high', 0)) for d in ohlc_data]
            lows = [float(d.get('low', 0)) for d in ohlc_data]
            
            # Simple RSI calculation (14-period)
            rsi = self._calculate_rsi(closes, 14)
            
            # Simple Moving Averages
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else 0
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else 0
            sma_200 = sum(closes) / len(closes) if len(closes) >= 200 else 0
            
            # Simple trend detection
            trend = "bullish" if closes[-1] > sma_20 > sma_50 else "bearish" if closes[-1] < sma_20 < sma_50 else "neutral"
            
            # Basic support/resistance (recent highs/lows)
            recent_data = ohlc_data[-20:]  # Last 20 days
            resistance_levels = sorted([float(d['high']) for d in recent_data], reverse=True)[:3]
            support_levels = sorted([float(d['low']) for d in recent_data])[:3]
            
            # Simple volatility (standard deviation of recent closes)
            recent_closes = closes[-20:]
            if len(recent_closes) > 1:
                avg = sum(recent_closes) / len(recent_closes)
                volatility = (sum((x - avg) ** 2 for x in recent_closes) / len(recent_closes)) ** 0.5
            else:
                volatility = 0
            
            return {
                "rsi": round(rsi, 2),
                "macd_signal": "bullish" if trend == "bullish" else "bearish" if trend == "bearish" else "neutral",
                "trend": trend,
                "sma_20": round(sma_20, 2),
                "sma_50": round(sma_50, 2),
                "sma_200": round(sma_200, 2),
                "support_levels": [round(x, 2) for x in support_levels],
                "resistance_levels": [round(x, 2) for x in resistance_levels],
                "volatility": round(volatility, 2),
                "volume_sma": sum(volumes[-20:]) // 20 if len(volumes) >= 20 else 0,
                "bollinger_bands": {}
            }
            
        except Exception as e:
            logger.error(f"❌ Indicator calculation failed: {e}")
            return self._get_default_technical_data()
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Default neutral RSI
            
            # Calculate price changes
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            # Separate gains and losses
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            # Calculate average gains and losses
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return max(0, min(100, rsi))  # Clamp between 0 and 100
            
        except Exception:
            return 50.0  # Default neutral RSI on error
    
    async def _fetch_fundamental_data(self) -> Dict[str, Dict]:
        """Fetch fundamental data from EODHD"""
        fundamental_data = {}
        
        try:
            # Fundamental data requires individual API calls
            async with aiohttp.ClientSession() as session:
                for symbol in self.stock_universe:
                    try:
                        url = f"https://eodhd.com/api/fundamentals/{symbol}.US"
                        params = {
                            "api_token": self.eodhd_api_key,
                            "fmt": "json"
                        }
                        
                        async with session.get(url, params=params, timeout=15) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                # Extract key fundamental metrics
                                highlights = data.get("Highlights", {})
                                valuation = data.get("Valuation", {})
                                financials = data.get("Financials", {})
                                
                                fundamental_data[symbol] = {
                                    "pe": float(highlights.get("PERatio", 0)),
                                    "pb": float(highlights.get("PriceBookMRQ", 0)),
                                    "roe": float(highlights.get("ReturnOnEquityTTM", 0)),
                                    "debt_to_equity": float(highlights.get("DebtToEquityMRQ", 0)),
                                    "dividend_yield": float(highlights.get("DividendYield", 0)),
                                    "eps": float(highlights.get("EarningsShare", 0)),
                                    "revenue_growth": float(highlights.get("RevenueGrowthTTM", 0)),
                                    "profit_margin": float(highlights.get("ProfitMargin", 0)),
                                    "market_cap": float(highlights.get("MarketCapitalization", 0)),
                                    "beta": float(highlights.get("Beta", 1.0)),
                                    "price_to_sales": float(valuation.get("TrailingPS", 0)),
                                    "enterprise_value": float(valuation.get("EnterpriseValue", 0)),
                                    "forward_pe": float(valuation.get("ForwardPE", 0))
                                }
                            else:
                                # Set default values for failed requests
                                fundamental_data[symbol] = self._get_default_fundamental_data()
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Fundamental data failed for {symbol}: {e}")
                        fundamental_data[symbol] = self._get_default_fundamental_data()
                    
                    # Rate limiting - fundamental calls are expensive
                    await asyncio.sleep(0.2)
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"❌ Fundamental data fetch failed: {e}")
            return {}
    
    def _get_default_fundamental_data(self) -> Dict:
        """Get default fundamental data for failed requests"""
        return {
            "pe": 0, "pb": 0, "roe": 0, "debt_to_equity": 0,
            "dividend_yield": 0, "eps": 0, "revenue_growth": 0,
            "profit_margin": 0, "market_cap": 0, "beta": 1.0,
            "price_to_sales": 0, "enterprise_value": 0, "forward_pe": 0
        }
    
    async def _combine_stock_data(self, basic: Dict, technical: Dict, fundamental: Dict) -> List[Dict]:
        """Combine all data sources into unified stock records"""
        combined_data = []
        
        for symbol in self.stock_universe:
            stock_record = {
                "symbol": symbol,
                "last_updated": datetime.now(),
                "basic": basic.get(symbol, {}),
                "technical": technical.get(symbol, {}),
                "fundamental": fundamental.get(symbol, {}),
                "screening_tags": self._generate_screening_tags(
                    basic.get(symbol, {}),
                    technical.get(symbol, {}),
                    fundamental.get(symbol, {})
                )
            }
            combined_data.append(stock_record)
        
        return combined_data
    
    def _generate_screening_tags(self, basic: Dict, technical: Dict, fundamental: Dict) -> List[str]:
        """Generate screening tags for fast filtering"""
        tags = []
        
        # Market cap tags
        market_cap = basic.get("market_cap", 0)
        if market_cap > 200000000000:  # $200B+
            tags.append("mega_cap")
        elif market_cap > 10000000000:  # $10B+
            tags.append("large_cap")
        elif market_cap > 2000000000:   # $2B+
            tags.append("mid_cap")
        else:
            tags.append("small_cap")
        
        # Technical tags
        rsi = technical.get("rsi", 50)
        if rsi > 70:
            tags.append("overbought")
        elif rsi < 30:
            tags.append("oversold")
        else:
            tags.append("neutral_rsi")
        
        # Trend tags
        trend = technical.get("trend", "neutral")
        tags.append(f"trend_{trend}")
        
        # Fundamental tags
        pe = fundamental.get("pe", 0)
        if 0 < pe < 15:
            tags.append("value")
        elif pe > 30:
            tags.append("growth")
        
        # Price movement tags
        change_1d = basic.get("change_1d", 0)
        if change_1d > 5:
            tags.append("strong_mover_up")
        elif change_1d > 2:
            tags.append("mover_up")
        elif change_1d < -5:
            tags.append("strong_mover_down")
        elif change_1d < -2:
            tags.append("mover_down")
        
        # Volume tags
        volume = basic.get("volume", 0)
        if volume > 10000000:  # 10M+ volume
            tags.append("high_volume")
        elif volume > 1000000:  # 1M+ volume
            tags.append("normal_volume")
        else:
            tags.append("low_volume")
        
        # Dividend tags
        dividend_yield = fundamental.get("dividend_yield", 0)
        if dividend_yield > 3:
            tags.append("high_dividend")
        elif dividend_yield > 0:
            tags.append("dividend_paying")
        
        return tags
    
    async def _store_in_mongodb(self, stock_data: List[Dict]):
        """Store combined data in MongoDB"""
        try:
            # Use upsert to replace existing documents
            for stock in stock_data:
                await self.db.stocks.replace_one(
                    {"symbol": stock["symbol"]},
                    stock,
                    upsert=True
                )
            
            logger.info(f"✅ Stored {len(stock_data)} stocks in MongoDB")
            
        except Exception as e:
            logger.error(f"❌ MongoDB storage failed: {e}")
            raise
    
    async def _cache_in_redis(self, stock_data: List[Dict]):
        """Cache data in Redis for fast access"""
        try:
            pipe = self.redis_client.pipeline()
            
            for stock in stock_data:
                symbol = stock["symbol"]
                
                # Cache each data type separately for flexible querying
                if stock["basic"]:
                    pipe.hset(f"stock:{symbol}:basic", mapping={k: str(v) for k, v in stock["basic"].items()})
                    pipe.expire(f"stock:{symbol}:basic", 604800)  # 7 days
                
                if stock["technical"]:
                    # Convert lists to strings for Redis storage
                    tech_data = stock["technical"].copy()
                    if "support_levels" in tech_data:
                        tech_data["support_levels"] = json.dumps(tech_data["support_levels"])
                    if "resistance_levels" in tech_data:
                        tech_data["resistance_levels"] = json.dumps(tech_data["resistance_levels"])
                    if "bollinger_bands" in tech_data:
                        tech_data["bollinger_bands"] = json.dumps(tech_data["bollinger_bands"])
                    
                    pipe.hset(f"stock:{symbol}:technical", mapping={k: str(v) for k, v in tech_data.items()})
                    pipe.expire(f"stock:{symbol}:technical", 604800)  # 7 days
                
                if stock["fundamental"]:
                    pipe.hset(f"stock:{symbol}:fundamental", mapping={k: str(v) for k, v in stock["fundamental"].items()})
                    pipe.expire(f"stock:{symbol}:fundamental", 604800)  # 7 days
                
                # Cache metadata
                pipe.set(f"stock:{symbol}:last_updated", stock["last_updated"].isoformat())
                pipe.expire(f"stock:{symbol}:last_updated", 604800)
                
                if stock["screening_tags"]:
                    pipe.sadd(f"stock:{symbol}:tags", *stock["screening_tags"])
                    pipe.expire(f"stock:{symbol}:tags", 604800)
            
            await pipe.execute()
            logger.info(f"✅ Cached {len(stock_data)} stocks in Redis")
            
        except Exception as e:
            logger.error(f"❌ Redis caching failed: {e}")
            raise
    
    def start_scheduler(self):
        """Start the background job scheduler"""
        # Schedule daily job at 6 AM ET
        schedule.every().day.at("06:00").do(self._run_async_job, self.daily_data_refresh_job)
        
        # Schedule weekly cleanup at 2 AM Sunday
        schedule.every().sunday.at("02:00").do(self._run_async_job, self.weekly_cleanup_job)
        
        logger.info("📅 Background job scheduler started")
        logger.info("   - Daily data refresh: 6:00 AM ET")
        logger.info("   - Weekly cleanup: Sunday 2:00 AM ET")
        
        # Run scheduler in background thread
        scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _run_async_job(self, job_func):
        """Wrapper to run async jobs from scheduler"""
        try:
            # Create new event loop for background thread
            import asyncio
            
            def run_job():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(job_func())
                finally:
                    loop.close()
            
            import threading
            job_thread = threading.Thread(target=run_job)
            job_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Scheduled job failed: {e}")
    
    async def get_job_status(self) -> Dict[str, Any]:
        """Get current job status for monitoring"""
        return {
            "pipeline_status": "active",
            "job_status": self.job_status,
            "next_daily_job": "06:00 AM ET daily",
            "next_weekly_cleanup": "Sunday 02:00 AM ET",
            "stock_universe_size": len(self.stock_universe),
            "crypto_universe_size": len(self.crypto_universe),
            "total_universe_size": self.total_universe_size,
            "database_connected": bool(self.db),
            "redis_connected": bool(self.redis_client),
            "stock_universe_preview": self.stock_universe[:10],
            "crypto_universe_preview": self.crypto_universe[:10],
            "features": {
                "on_demand_stock_fetching": True,
                "comprehensive_crypto_support": True,
                "quantum_computing_stocks": True,
                "meme_crypto_coverage": True,
                "ai_ml_stock_focus": True,
                "defi_token_coverage": True,
                "fallback_api_calls": True
            }
        }
    
    async def force_run_daily_job(self):
        """Manually trigger daily job (for testing/emergency)"""
        logger.info("🔧 Manually triggered daily data refresh")
        await self.daily_data_refresh_job()
    
    async def force_run_cleanup(self):
        """Manually trigger cleanup job (for testing/emergency)"""
        logger.info("🔧 Manually triggered weekly cleanup")
        await self.weekly_cleanup_job()
    
    async def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("✅ Background pipeline connections closed")
