# services/backgroundjob/cached_technical_service.py
# Stub file - redirects to the actual technical analysis service

from ..technical_analysis import TAEngine

# Re-export with all expected names for backward compatibility
CachedTechnicalService = TAEngine
CachedTechnicalAnalysisService = TAEngine
TechnicalAnalysisService = TAEngine
TAService = TAEngine

__all__ = ['CachedTechnicalService', 'CachedTechnicalAnalysisService', 'TechnicalAnalysisService', 'TAEngine', 'TAService']
