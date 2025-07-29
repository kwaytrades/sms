# services/backgroundjob/cached_fundamental_service.py
# Stub file - redirects to the actual fundamental analysis service

from ..fundamental_analysis import FAEngine

# Re-export with expected names for backward compatibility
CachedFundamentalService = FAEngine
FundamentalAnalysisService = FAEngine

__all__ = ['CachedFundamentalService', 'FundamentalAnalysisService', 'FAEngine']
