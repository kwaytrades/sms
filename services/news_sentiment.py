# services/news_sentiment.py

import os
import json
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from loguru import logger
import redis

class NewsSentimentService:
    def __init__(self, redis_client=None, openai_service=None):
        self.marketaux_api_key = os.getenv('MARKETAUX_API_KEY')
        self.redis_client = redis_client
        self.openai_service = openai_service
        
        # Cache settings - 1 hour for regular, real-time for alerts
        self.cache_ttl_hourly = 3600  # 1 hour
        self.cache_ttl_realtime = 60  # 1 minute for breaking news
        
        logger.info(f"✅ News Sentiment Service initialized with MarketAux API: {'Set' if self.marketaux_api_key else 'Not Set'}")
        logger.info(f"✅ Redis available: {self.redis_client is not None}")
        logger.info(f"✅ OpenAI service available: {self.openai_service is not None}")
    
    async def get_sentiment(self, symbol: str, mode: str = "cached") -> Optional[Dict[str, Any]]:
        """
        Main sentiment analysis function
        mode: "cached" for SMS responses (1hr cache), "realtime" for alerts
        """
        try:
            symbol = symbol.upper()
            
            # Check cache first (except for realtime mode)
            if mode == "cached":
                cached_sentiment = await self._get_cached_sentiment(symbol)
                if cached_sentiment:
                    logger.info(f"📰 News Sentiment Cache HIT for {symbol}")
                    return cached_sentiment
            
            logger.info(f"📰 News Sentiment Cache MISS for {symbol} - fetching fresh data (mode: {mode})")
            
            # Fetch fresh news data
            news_data = await self._fetch_marketaux_news(symbol)
            
            if not news_data or not news_data.get('articles'):
                return self._create_no_news_response(symbol)
            
            # Analyze sentiment using GPT-4o-mini
            sentiment_analysis = await self._analyze_sentiment_batch(symbol, news_data['articles'])
            
            # Cache the result (except realtime mode)
            if mode == "cached":
                await self._cache_sentiment(symbol, sentiment_analysis)
            
            logger.info(f"✅ News sentiment analysis completed for {symbol}")
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"❌ News sentiment analysis failed for {symbol}: {e}")
            return self._create_error_response(symbol, str(e))
    
    async def _get_cached_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached sentiment data"""
        if not self.redis_client:
            return None
        
        try:
            # Use hourly time bucket for cache key
            hour_bucket = datetime.now().strftime('%Y-%m-%d-%H')
            cache_key = f"news_sentiment:{symbol}:{hour_bucket}"
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Cache retrieval error for {symbol}: {e}")
            return None
    
    async def _cache_sentiment(self, symbol: str, sentiment_data: Dict[str, Any]) -> None:
        """Cache sentiment data"""
        if not self.redis_client:
            return
        
        try:
            hour_bucket = datetime.now().strftime('%Y-%m-%d-%H')
            cache_key = f"news_sentiment:{symbol}:{hour_bucket}"
            
            # Add cache metadata
            sentiment_data['cache_status'] = 'fresh'
            sentiment_data['cached_at'] = datetime.now().isoformat()
            
            self.redis_client.setex(
                cache_key,
                self.cache_ttl_hourly,
                json.dumps(sentiment_data, default=str)
            )
            
            logger.info(f"📰 Cached sentiment for {symbol} (TTL: {self.cache_ttl_hourly}s)")
        except Exception as e:
            logger.error(f"Cache storage error for {symbol}: {e}")
    
    async def _fetch_marketaux_news(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch news from MarketAux API with broader timeframe and relevance ranking"""
        if not self.marketaux_api_key:
            logger.error("MarketAux API key not available")
            return None
        
        try:
            # MarketAux API endpoint
            url = "https://api.marketaux.com/v1/news/all"
            
            params = {
                'api_token': self.marketaux_api_key,
                'symbols': symbol,
                'filter_entities': 'true',
                'must_have_entities': 'true',  # Ensure articles mention the symbol
                'language': 'en',
                'limit': 25,  # Get more articles to choose from
                'published_after': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),  # Last 30 days
                'sort': 'entity_sentiment_score',  # Sort by sentiment relevance
                'sort_order': 'desc'  # Most impactful first
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    response_text = await response.text()
                    
                    if response.status == 400:
                        logger.error(f"MarketAux 400 Bad Request for {symbol}: {response_text}")
                        return None
                    elif response.status == 401:
                        logger.error(f"MarketAux 401 Unauthorized - Check API key")
                        return None
                    elif response.status == 403:
                        logger.error(f"MarketAux 403 Forbidden - Check plan permissions")
                        return None
                    elif response.status != 200:
                        logger.error(f"MarketAux API error {response.status} for {symbol}: {response_text}")
                        return None
                    
                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON response for {symbol}: {e}")
                        return None
                    
                    if not data.get('data'):
                        logger.warning(f"No news data returned for {symbol} in last 30 days")
                        return None
                    
                    articles = data['data']
                    logger.info(f"✅ Fetched {len(articles)} news articles for {symbol}")
                    
                    # Rank articles by relevance and recency
                    ranked_articles = self._rank_articles_by_importance(articles, symbol)
                    
                    logger.info(f"📊 Ranked top {len(ranked_articles)} most relevant articles for {symbol}")
                    return {'articles': ranked_articles}
                    
        except asyncio.TimeoutError:
            logger.error(f"MarketAux API timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"MarketAux API error for {symbol}: {e}")
            return None
    
    def _rank_articles_by_importance(self, articles: List[Dict], symbol: str) -> List[Dict]:
        """Rank articles by relevance, sentiment impact, and recency"""
        
        def calculate_importance_score(article: Dict) -> float:
            """Calculate importance score for ranking articles"""
            score = 0.0
            
            # 1. Recency score (0-30 points) - newer is better
            try:
                pub_date = datetime.fromisoformat(article.get('published_at', '').replace('Z', '+00:00'))
                days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                recency_score = max(0, 30 - days_old)  # Full points for today, decreases daily
                score += recency_score
            except:
                score += 0  # No points if date parsing fails
            
            # 2. Entity sentiment score (0-25 points) - strong sentiment is important
            entities = article.get('entities', [])
            symbol_entities = [e for e in entities if e.get('symbol') == symbol]
            if symbol_entities:
                sentiment_score = abs(symbol_entities[0].get('sentiment_score', 0)) * 25
                score += sentiment_score
            
            # 3. Entity match score (0-20 points) - how relevant is the mention
            if symbol_entities:
                match_score = symbol_entities[0].get('match_score', 0) * 20
                score += match_score
            
            # 4. Title relevance (0-15 points) - symbol in title is more important
            title = article.get('title', '').upper()
            if symbol in title:
                score += 15
            elif any(word in title for word in [symbol.lower(), 'earnings', 'revenue', 'profit']):
                score += 10
            
            # 5. Source credibility (0-10 points) - trusted sources get boost
            source = article.get('source', '').lower()
            trusted_sources = ['reuters', 'bloomberg', 'wsj', 'cnbc', 'marketwatch', 'yahoo', 'sec']
            if any(trusted in source for trusted in trusted_sources):
                score += 10
            elif any(domain in source for domain in ['finance', 'business', 'market']):
                score += 5
            
            return score
        
        # Calculate scores and sort
        scored_articles = []
        for article in articles:
            importance_score = calculate_importance_score(article)
            article['_importance_score'] = importance_score
            scored_articles.append(article)
        
        # Sort by importance score (highest first) and return top 10
        ranked_articles = sorted(scored_articles, key=lambda x: x['_importance_score'], reverse=True)[:10]
        
        # Log top articles for debugging
        if ranked_articles:
            logger.info(f"🔝 Top article for {symbol}: {ranked_articles[0].get('title', 'No title')[:100]} (score: {ranked_articles[0]['_importance_score']:.1f})")
        
        return ranked_articles
    
    async def _analyze_sentiment_batch(self, symbol: str, articles: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment using GPT-4o-mini batch processing"""
        if not self.openai_service:
            return self._create_fallback_sentiment(symbol, articles)
        
        try:
            # Use top 5 most important articles for analysis
            top_articles = articles[:5]
            
            # Create batch prompt with ranked articles
            articles_text = "\n\n".join([
                f"Article {i+1} (Importance Score: {article.get('_importance_score', 0):.1f}):\n"
                f"Headline: {article.get('title', 'N/A')}\n"
                f"Summary: {article.get('description', 'N/A')[:300]}...\n"
                f"Published: {article.get('published_at', 'N/A')}\n"
                f"Source: {article.get('source', 'N/A')}"
                for i, article in enumerate(top_articles)
            ])
            
            prompt = f"""
            Analyze the following ranked financial news articles for {symbol} and provide sentiment analysis.
            These articles are ranked by importance, relevance, and recency:

            {articles_text}

            Return JSON response with this exact format:
            {{
                "sentiment": "bullish" | "bearish" | "neutral",
                "impact_score": 0.1-1.0,
                "confidence": 0.1-1.0,
                "key_drivers": ["reason1", "reason2", "reason3"],
                "summary": "Brief explanation in 1-2 sentences focusing on most important news"
            }}

            Guidelines:
            - sentiment: bullish (positive outlook), bearish (negative outlook), neutral (mixed/unclear)
            - impact_score: Expected price impact (0.1=minor, 0.5=moderate, 1.0=major market-moving news)
            - confidence: How reliable is this sentiment (0.1=low, 1.0=high confidence)
            - key_drivers: Main reasons driving the sentiment (prioritize recent/high-impact events)
            - summary: Concise explanation focusing on the most important developments
            - Weight more recent and higher-scored articles more heavily in your analysis
            """
            
            # Use GPT-4o-mini for cost efficiency
            response = await self.openai_service.get_completion(
                prompt,
                max_tokens=300,
                temperature=0.3,
                model="gpt-4o-mini"
            )
            
            # Parse JSON response
            sentiment_data = json.loads(response)
            
            # Create full response
            return {
                'symbol': symbol,
                'news_sentiment': {
                    'sentiment': sentiment_data.get('sentiment', 'neutral'),
                    'impact_score': float(sentiment_data.get('impact_score', 0.5)),
                    'confidence': float(sentiment_data.get('confidence', 0.5)),
                    'key_drivers': sentiment_data.get('key_drivers', []),
                    'summary': sentiment_data.get('summary', 'Mixed market sentiment'),
                    'article_count': len(top_articles),
                    'analysis_period': '30_days',
                    'top_headline': top_articles[0].get('title', 'N/A') if top_articles else 'N/A'
                },
                'headlines': [article.get('title', 'N/A') for article in top_articles[:3]],
                'source': 'marketaux_api',
                'timestamp': datetime.now().isoformat(),
                'analysis_mode': 'gpt4o_mini'
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response for {symbol}: {e}")
            return self._create_fallback_sentiment(symbol, articles)
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self._create_fallback_sentiment(symbol, articles)
    
    def _create_fallback_sentiment(self, symbol: str, articles: List[Dict]) -> Dict[str, Any]:
        """Create basic sentiment analysis when GPT is unavailable"""
        # Simple keyword-based sentiment
        positive_keywords = ['beat', 'exceed', 'growth', 'profit', 'surge', 'bullish', 'upgrade', 'strong']
        negative_keywords = ['miss', 'decline', 'loss', 'drop', 'bearish', 'downgrade', 'weak', 'concern']
        
        article_text = ' '.join([
            f"{article.get('title', '')} {article.get('description', '')}"
            for article in articles[:5]
        ]).lower()
        
        positive_count = sum(1 for word in positive_keywords if word in article_text)
        negative_count = sum(1 for word in negative_keywords if word in article_text)
        
        if positive_count > negative_count:
            sentiment = 'bullish'
            impact_score = min(0.8, 0.3 + (positive_count * 0.1))
        elif negative_count > positive_count:
            sentiment = 'bearish'
            impact_score = min(0.8, 0.3 + (negative_count * 0.1))
        else:
            sentiment = 'neutral'
            impact_score = 0.3
        
        return {
            'symbol': symbol,
            'news_sentiment': {
                'sentiment': sentiment,
                'impact_score': impact_score,
                'confidence': 0.6,  # Lower confidence for fallback
                'key_drivers': ['Market news analysis'],
                'summary': f'Recent news shows {sentiment} sentiment for {symbol}',
                'article_count': len(articles)
            },
            'headlines': [article.get('title', 'N/A') for article in articles[:3]],
            'source': 'marketaux_api',
            'timestamp': datetime.now().isoformat(),
            'analysis_mode': 'keyword_fallback'
        }
    
    def _create_no_news_response(self, symbol: str) -> Dict[str, Any]:
        """Response when no news is available"""
        return {
            'symbol': symbol,
            'news_sentiment': {
                'sentiment': 'neutral',
                'impact_score': 0.0,
                'confidence': 0.0,
                'key_drivers': [],
                'summary': f'No recent news found for {symbol}',
                'article_count': 0
            },
            'headlines': [],
            'source': 'marketaux_api',
            'timestamp': datetime.now().isoformat(),
            'analysis_mode': 'no_news'
        }
    
    def _create_error_response(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """Response when analysis fails"""
        return {
            'symbol': symbol,
            'error': error_msg,
            'message': f'News sentiment unavailable for {symbol}. Please try again.',
            'source': 'marketaux_api',
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_breaking_news(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get breaking news for multiple symbols (for alert system)"""
        breaking_news = []
        
        for symbol in symbols:
            try:
                # Always use realtime mode for breaking news
                sentiment_data = await self.get_sentiment(symbol, mode="realtime")
                
                if sentiment_data and not sentiment_data.get('error'):
                    # Check if this is "breaking" (high impact + recent)
                    impact_score = sentiment_data.get('news_sentiment', {}).get('impact_score', 0)
                    if impact_score >= 0.7:  # High impact threshold
                        breaking_news.append(sentiment_data)
                        
            except Exception as e:
                logger.error(f"Breaking news check failed for {symbol}: {e}")
                continue
        
        return breaking_news
    
    async def close(self):
        """Cleanup method"""
        logger.info("✅ News Sentiment service closed")
