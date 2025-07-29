# services/backgroundjob/cached_technical_service.py
# Stub file - redirects to the actual technical analysis service

from ..technical_analysis import TAEngine

# Re-export with expected names for backward compatibility
CachedTechnicalService = TAEngine
TechnicalAnalysisService = TAEngine

__all__ = ['CachedTechnicalService', 'TechnicalAnalysisService', 'TAEngine']
