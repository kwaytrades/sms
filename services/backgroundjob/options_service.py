# services/backgroundjob/options_service.py
# Stub file - redirects to the actual options service

from ..options_service import OptionsEngine

# Re-export with the expected class name for backward compatibility
OptionsAnalyzer = OptionsEngine

__all__ = ['OptionsAnalyzer', 'OptionsEngine']
