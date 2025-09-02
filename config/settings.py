"""
Configuration centralisée de l'application
"""
import os
from typing import Optional, List
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class Settings:
    """Configuration centralisée de l'application"""
    
    # API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Cache configuration
    CACHE_DURATION_HOURS: int = int(os.getenv("CACHE_DURATION_HOURS", "24"))
    CACHE_DIR: str = "cache"
    
    # PDF processing limits
    MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
    MIN_PAGE_CONTENT_LENGTH: int = int(os.getenv("MIN_PAGE_CONTENT_LENGTH", "1"))
    
    # Text chunking parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # Model configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    
    # Search parameters
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv("DEFAULT_SEARCH_RESULTS", "50"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "50"))
    
    # Paths
    DATA_DIR: str = "Data"
    GUIDELINES_DIR: str = os.path.join(DATA_DIR, "guidelines")
    FAISS_INDEX_DIR: str = "faiss_index_chat"
    
    # UI Configuration
    PAGE_TITLE: str = "Analyseur d'Interactions Médicamenteuses"
    PAGE_ICON: str = "💊"
    LAYOUT: str = "wide"
    
    @classmethod
    def get_api_keys(cls) -> List[str]:
        """Retourne la liste des clés API Google"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY non trouvée dans les variables d'environnement")
        
        keys = [k.strip() for k in cls.GOOGLE_API_KEY.split(",") if k.strip()]
        if not keys:
            raise ValueError("Aucune clé API valide trouvée")
        
        return keys
    
    @classmethod
    def validate_configuration(cls) -> bool:
        """Valide la configuration"""
        try:
            # Vérifier les clés API
            cls.get_api_keys()
            
            # Vérifier les valeurs numériques
            assert cls.CACHE_DURATION_HOURS > 0, "CACHE_DURATION_HOURS doit être positif"
            assert cls.MAX_PDF_SIZE_MB > 0, "MAX_PDF_SIZE_MB doit être positif"
            assert cls.CHUNK_SIZE > 0, "CHUNK_SIZE doit être positif"
            assert 0 <= cls.LLM_TEMPERATURE <= 1, "LLM_TEMPERATURE doit être entre 0 et 1"
            
            return True
        except Exception as e:
            print(f"Erreur de configuration: {e}")
            return False

# Instance globale des paramètres
settings = Settings()

# Niveaux d'interaction pour classification
INTERACTION_LEVELS = {
    'major': 'Major',
    'majeur': 'Major',
    'élevé': 'Major',
    'moderate': 'Moderate',
    'modérée': 'Moderate',
    'modéré': 'Moderate',
    'moyen': 'Moderate',
    'minor': 'Minor',
    'mineure': 'Minor',
    'mineur': 'Minor',
    'faible': 'Minor'
}

# Couleurs pour l'interface
LEVEL_COLORS = {
    'Major': '#DC3545',      # Rouge
    'Moderate': '#FD7E14',   # Orange
    'Minor': '#28A745',      # Vert
    'Aucune': '#6C757D'      # Gris
}

# Formats d'export supportés
EXPORT_FORMATS = ['CSV', 'Excel']

# Séparateurs pour le text splitting
TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
