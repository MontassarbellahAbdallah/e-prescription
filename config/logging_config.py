"""
Configuration du logging pour l'application
"""
import logging
import sys
from typing import Optional
from config.settings import settings

def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure le système de logging pour l'application
    
    Args:
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Format personnalisé pour les messages
    
    Returns:
        Logger configuré
    """
    # Utiliser le niveau depuis les settings si pas spécifié
    if level is None:
        level = settings.LOG_LEVEL
    
    # Format par défaut
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configuration du logger principal
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', encoding='utf-8')
        ]
    )
    
    # Créer le logger principal
    logger = logging.getLogger('MedInteractionAnalyzer')
    
    # Réduire le niveau de logging pour certaines bibliothèques
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger.info("Système de logging initialisé")
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger avec un nom spécifique
    
    Args:
        name: Nom du logger (généralement __name__)
    
    Returns:
        Logger configuré
    """
    return logging.getLogger(f'MedInteractionAnalyzer.{name}')

# Logger principal de l'application
main_logger = setup_logging()
