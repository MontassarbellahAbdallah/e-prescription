"""
Exceptions personnalisées pour l'analyseur d'interactions médicamenteuses
"""

class MedInteractionAnalyzerError(Exception):
    """Exception de base pour l'application"""
    pass

class ConfigurationError(MedInteractionAnalyzerError):
    """Erreur de configuration"""
    pass

class APIKeyError(MedInteractionAnalyzerError):
    """Erreur liée aux clés API"""
    pass

class PDFProcessingError(MedInteractionAnalyzerError):
    """Erreur lors du traitement des PDF"""
    pass

class VectorStoreError(MedInteractionAnalyzerError):
    """Erreur liée au vector store"""
    pass

class LLMAnalysisError(MedInteractionAnalyzerError):
    """Erreur lors de l'analyse LLM"""
    pass

class CacheError(MedInteractionAnalyzerError):
    """Erreur liée au cache"""
    pass

class ValidationError(MedInteractionAnalyzerError):
    """Erreur de validation des données"""
    pass

class ExportError(MedInteractionAnalyzerError):
    """Erreur lors de l'export"""
    pass
