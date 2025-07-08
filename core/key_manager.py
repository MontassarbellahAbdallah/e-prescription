"""
Gestionnaire avancé des clés API Gemini avec monitoring et rotation
"""
import itertools
import time
from typing import Callable, Any, Dict, List
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

class GeminiKeyManager:
    """
    Gestionnaire des clés API Gemini avec rotation automatique et monitoring
    
    Fonctionnalités:
    - Rotation automatique des clés en cas de quota dépassé
    - Monitoring de l'utilisation des clés
    - Retry intelligent avec délais
    - Validation des clés au démarrage
    """
    
    def __init__(self):
        """Initialise le gestionnaire avec les clés API depuis la configuration"""
        self._keys = settings.get_api_keys()
        logger.info(f"Initialized with {len(self._keys)} API keys")
        
        # Validation et test de la première clé
        self._validate_first_key()
        
        # Configuration de la rotation
        self._cycle = itertools.cycle(self._keys)
        self.current_key = None
        self.current_key_index = 0
        
        # Statistiques d'utilisation
        self.usage_stats = {
            i: {'calls': 0, 'errors': 0, 'last_used': None}
            for i in range(len(self._keys))
        }
        
        # Première rotation
        self.rotate()
    
    def _validate_first_key(self) -> None:
        """
        Valide et teste la première clé API
        
        Raises:
            ValueError: Si la clé est invalide
        """
        try:
            # Configurer avec la première clé
            genai.configure(api_key=self._keys[0])
            
            # Test simple avec le modèle
            model = genai.GenerativeModel(settings.LLM_MODEL)
            response = model.generate_content("test", stream=False)
            
            if response and response.text:
                logger.info("API key validation successful")
            else:
                logger.warning("API key validation returned empty response")
                
        except Exception as e:
            logger.warning(f"API key validation failed: {e}")
            # Ne pas lever l'exception, juste logger l'avertissement
    
    def rotate(self) -> None:
        """
        Passe à la clé API suivante et configure le SDK
        """
        previous_index = self.current_key_index
        self.current_key = next(self._cycle)
        
        # Trouver l'index de la clé actuelle
        self.current_key_index = self._keys.index(self.current_key)
        
        # Configurer le SDK
        genai.configure(api_key=self.current_key)
        
        # Mettre à jour les stats
        self.usage_stats[self.current_key_index]['last_used'] = time.time()
        
        logger.info(f"Rotated from key {previous_index} to key {self.current_key_index}")
    
    def wrap_quota(self, func: Callable) -> Callable:
        """
        Décorateur pour gérer automatiquement les quotas et erreurs API
        
        Args:
            func: Fonction à wrapper (généralement model.invoke)
            
        Returns:
            Fonction wrappée avec gestion d'erreurs
        """
        def _inner(*args, **kwargs) -> Any:
            max_retries = len(self._keys)
            tried = 0
            start_time = time.time()
            
            while tried < max_retries:
                try:
                    # Incrémenter les stats
                    self.usage_stats[self.current_key_index]['calls'] += 1
                    
                    # Appeler la fonction
                    result = func(*args, **kwargs)
                    
                    # Logger le succès
                    execution_time = time.time() - start_time
                    logger.info(
                        f"LLM call successful in {execution_time:.2f}s "
                        f"(key {self.current_key_index}, attempt {tried + 1})"
                    )
                    
                    return result
                    
                except ResourceExhausted as e:
                    # Quota dépassé - rotation nécessaire
                    self.usage_stats[self.current_key_index]['errors'] += 1
                    logger.warning(
                        f"Quota exceeded for key {self.current_key_index}: {e}"
                    )
                    
                    tried += 1
                    if tried < max_retries:
                        self.rotate()
                        time.sleep(1)  # Délai avant retry
                    
                except Exception as e:
                    # Autres erreurs
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['quota', 'rate', 'limit']):
                        # Erreur liée au quota
                        self.usage_stats[self.current_key_index]['errors'] += 1
                        logger.warning(
                            f"Quota-related error detected for key {self.current_key_index}: {e}"
                        )
                        
                        tried += 1
                        if tried < max_retries:
                            self.rotate()
                            time.sleep(1)
                    else:
                        # Erreur non liée au quota - propager
                        logger.error(f"Non-quota LLM error: {e}")
                        raise
            
            # Toutes les clés ont échoué
            raise RuntimeError(
                f"Quota exceeded for all {len(self._keys)} API keys. "
                "Please wait or add more keys."
            )
        
        return _inner
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'utilisation des clés
        
        Returns:
            Dictionnaire avec les statistiques détaillées
        """
        total_calls = sum(stats['calls'] for stats in self.usage_stats.values())
        total_errors = sum(stats['errors'] for stats in self.usage_stats.values())
        
        # Calculer le taux de succès
        success_rate = 0.0
        if total_calls > 0:
            success_rate = ((total_calls - total_errors) / total_calls) * 100
        
        # Statistiques par clé
        key_stats = {}
        for i, stats in self.usage_stats.items():
            key_success_rate = 0.0
            if stats['calls'] > 0:
                key_success_rate = ((stats['calls'] - stats['errors']) / stats['calls']) * 100
            
            key_stats[f'key_{i}'] = {
                'calls': stats['calls'],
                'errors': stats['errors'],
                'success_rate': round(key_success_rate, 1),
                'last_used': stats['last_used'],
                'is_current': i == self.current_key_index
            }
        
        return {
            'total_keys': len(self._keys),
            'current_key_index': self.current_key_index,
            'total_calls': total_calls,
            'total_errors': total_errors,
            'overall_success_rate': round(success_rate, 1),
            'keys': key_stats
        }
    
    def reset_stats(self) -> None:
        """Remet à zéro les statistiques d'utilisation"""
        self.usage_stats = {
            i: {'calls': 0, 'errors': 0, 'last_used': None}
            for i in range(len(self._keys))
        }
        logger.info("Usage statistics reset")
    
    def force_rotate_to_key(self, key_index: int) -> bool:
        """
        Force la rotation vers une clé spécifique
        
        Args:
            key_index: Index de la clé (0 à len(keys)-1)
            
        Returns:
            True si la rotation a réussi, False sinon
        """
        try:
            if 0 <= key_index < len(self._keys):
                # Recréer le cycle pour pointer vers la bonne clé
                remaining_keys = self._keys[key_index:] + self._keys[:key_index]
                self._cycle = itertools.cycle(remaining_keys)
                
                # Effectuer la rotation
                self.rotate()
                return True
            else:
                logger.error(f"Invalid key index: {key_index}")
                return False
                
        except Exception as e:
            logger.error(f"Error forcing rotation to key {key_index}: {e}")
            return False
    
    def test_all_keys(self) -> Dict[int, bool]:
        """
        Teste toutes les clés API et retourne leur statut
        
        Returns:
            Dictionnaire {index: is_valid} pour chaque clé
        """
        results = {}
        original_key_index = self.current_key_index
        
        for i, key in enumerate(self._keys):
            try:
                # Configurer temporairement avec cette clé
                genai.configure(api_key=key)
                
                # Test simple
                model = genai.GenerativeModel(settings.LLM_MODEL)
                response = model.generate_content("test", stream=False)
                
                results[i] = bool(response and response.text)
                logger.info(f"Key {i} test: {'PASS' if results[i] else 'FAIL'}")
                
            except Exception as e:
                results[i] = False
                logger.warning(f"Key {i} test failed: {e}")
        
        # Restaurer la clé originale
        self.force_rotate_to_key(original_key_index)
        
        return results

# Instance globale du gestionnaire de clés (singleton pattern)
_key_manager_instance = None

def get_key_manager() -> GeminiKeyManager:
    """
    Retourne l'instance globale du gestionnaire de clés (singleton)
    
    Returns:
        Instance du GeminiKeyManager
    """
    global _key_manager_instance
    if _key_manager_instance is None:
        _key_manager_instance = GeminiKeyManager()
    return _key_manager_instance
