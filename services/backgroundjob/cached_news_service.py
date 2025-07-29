# services/backgroundjob/cached_news_service.py
# Stub file - redirects to the actual news sentiment service

try:
    from ..news_sentiment import NewsSentimentService
    CachedNewsService = NewsSentimentService
    CachedNewsSentimentService = NewsSentimentService
    NEWS_SERVICE_AVAILABLE = True
except ImportError:
    # Fallback if news service not available
    class CachedNewsService:
        def __init__(self, *args, **kwargs):
            pass
        
        async def get_sentiment(self, symbol):
            return {"sentiment": "neutral", "score": 0.5, "source": "unavailable"}
    
    class CachedNewsSentimentService:
        def __init__(self, *args, **kwargs):
            pass
        
        async def get_sentiment(self, symbol):
            return {"sentiment": "neutral", "score": 0.5, "source": "unavailable"}
    
    NEWS_SERVICE_AVAILABLE = False

__all__ = ['CachedNewsService', 'CachedNewsSentimentService', 'NEWS_SERVICE_AVAILABLE']
