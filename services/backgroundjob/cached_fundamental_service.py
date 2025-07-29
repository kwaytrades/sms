# services/backgroundjob/cached_fundamental_service.py
# Stub file - redirects to the actual fundamental analysis service

from ..fundamental_analysis import FAEngine

# Re-export with all expected names for backward compatibility
CachedFundamentalService = FAEngine
CachedFundamentalAnalysisService = FAEngine
FundamentalAnalysisService = FAEngine
FundamentalAnalysisTool = FAEngine

__all__ = ['CachedFundamentalService', 'CachedFundamentalAnalysisService', 'FundamentalAnalysisService', 'FundamentalAnalysisTool', 'FAEngine']
