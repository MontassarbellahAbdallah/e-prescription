"""
Gestionnaire de cache pour optimiser les analyses répétitives
"""
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

class CacheManager:
    """
    Gestionnaire de cache pour optimiser les performances de l'application
    
    Fonctionnalités:
    - Cache persistant sur disque
    - Expiration automatique des données
    - Clés de cache sécurisées (hash MD5)
    - Gestion d'erreurs robuste
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialise le gestionnaire de cache
        
        Args:
            cache_dir: Répertoire de cache (utilise settings par défaut)
        """
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.cache_duration = timedelta(hours=settings.CACHE_DURATION_HOURS)
        
        # Créer le répertoire de cache s'il n'existe pas
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Cache directory initialized: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory {self.cache_dir}: {e}")
            raise
    
    def _get_cache_key(self, data: str, prefix: str = "") -> str:
        """
        Génère une clé de cache unique et sécurisée avec préfixe
        
        Args:
            data: Données à hasher
            prefix: Préfixe pour éviter les collisions (ex: 'drugs_', 'search_', 'interaction_')
            
        Returns:
            Clé de cache (hash MD5 avec préfixe)
        """
        if not isinstance(data, str):
            data = str(data)
        
        # Ajouter le préfixe aux données avant le hash pour éviter les collisions
        prefixed_data = f"{prefix}:{data}" if prefix else data
        hash_key = hashlib.md5(prefixed_data.encode('utf-8')).hexdigest()
        
        # Retourner la clé avec préfixe pour faciliter l'identification
        return f"{prefix}{hash_key}" if prefix else hash_key
    
    def _get_cache_file_path(self, key: str) -> str:
        """
        Retourne le chemin complet du fichier de cache
        
        Args:
            key: Clé de cache
            
        Returns:
            Chemin complet du fichier
        """
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """
        Vérifie si le cache est encore valide
        
        Args:
            timestamp: Timestamp de création du cache
            
        Returns:
            True si le cache est valide, False sinon
        """
        return datetime.now() - timestamp < self.cache_duration
    
    def get(self, key: str, prefix: str = "") -> Optional[Any]:
        """
        Récupère une valeur du cache si elle existe et est valide
        
        Args:
            key: Clé de cache (sera hashée automatiquement)
            prefix: Préfixe pour identifier le type de cache
            
        Returns:
            Données du cache ou None si inexistant/expiré
        """
        try:
            # Générer la clé de cache avec préfixe
            cache_key = self._get_cache_key(key, prefix)
            cache_file = self._get_cache_file_path(cache_key)
            
            # Vérifier l'existence du fichier
            if not os.path.exists(cache_file):
                logger.debug(f"Cache miss - file not found: {prefix}{cache_key[:8]}...")
                return None
            
            # Charger les données
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Vérifier la validité du cache
            if not self._is_cache_valid(cached_data['timestamp']):
                logger.debug(f"Cache expired for key: {prefix}{cache_key[:8]}...")
                # Supprimer le fichier expiré
                self._remove_cache_file(cache_file)
                return None
            
            logger.info(f"Cache hit for key: {prefix}{cache_key[:8]}...")
            return cached_data['data']
            
        except Exception as e:
            logger.warning(f"Error reading cache for key {prefix}:{key[:50]}: {e}")
            return None
    
    def set(self, key: str, data: Any, prefix: str = "") -> bool:
        """
        Sauvegarde une valeur dans le cache
        
        Args:
            key: Clé de cache (sera hashée automatiquement)
            data: Données à sauvegarder
            prefix: Préfixe pour identifier le type de cache
            
        Returns:
            True si sauvegarde réussie, False sinon
        """
        try:
            # Générer la clé de cache avec préfixe
            cache_key = self._get_cache_key(key, prefix)
            cache_file = self._get_cache_file_path(cache_key)
            
            # Préparer les données avec timestamp et métadonnées
            cached_data = {
                'data': data,
                'timestamp': datetime.now(),
                'key': key,  # Pour debug
                'prefix': prefix,  # Pour identification
                'cache_type': prefix or 'generic'
            }
            
            # Sauvegarder
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            logger.info(f"Cache saved for key: {prefix}{cache_key[:8]}... (type: {prefix or 'generic'})")
            return True
            
        except Exception as e:
            logger.warning(f"Error writing cache for key {prefix}:{key[:50]}: {e}")
            return False
    
    def _remove_cache_file(self, file_path: str) -> None:
        """
        Supprime un fichier de cache en toute sécurité
        
        Args:
            file_path: Chemin du fichier à supprimer
        """
        try:
            os.remove(file_path)
            logger.debug(f"Removed expired cache file: {os.path.basename(file_path)}")
        except Exception as e:
            logger.warning(f"Failed to remove cache file {file_path}: {e}")
    
    def clear(self, key: Optional[str] = None) -> int:
        """
        Supprime les entrées du cache
        
        Args:
            key: Clé spécifique à supprimer (None = tout supprimer)
            
        Returns:
            Nombre de fichiers supprimés
        """
        try:
            if key is not None:
                # Supprimer une clé spécifique
                cache_key = self._get_cache_key(key)
                cache_file = self._get_cache_file_path(cache_key)
                if os.path.exists(cache_file):
                    self._remove_cache_file(cache_file)
                    return 1
                return 0
            else:
                # Supprimer tout le cache
                if not os.path.exists(self.cache_dir):
                    return 0
                
                removed_count = 0
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        file_path = os.path.join(self.cache_dir, filename)
                        self._remove_cache_file(file_path)
                        removed_count += 1
                
                logger.info(f"Cleared {removed_count} cache files")
                return removed_count
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def cleanup_expired(self) -> int:
        """
        Supprime tous les fichiers de cache expirés
        
        Returns:
            Nombre de fichiers supprimés
        """
        try:
            if not os.path.exists(self.cache_dir):
                return 0
            
            removed_count = 0
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.pkl'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if not self._is_cache_valid(cached_data['timestamp']):
                        self._remove_cache_file(file_path)
                        removed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error checking cache file {filename}: {e}")
                    # Supprimer les fichiers corrompus
                    self._remove_cache_file(file_path)
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired cache files")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """
        Retourne les statistiques du cache
        
        Returns:
            Dictionnaire avec les statistiques
        """
        try:
            if not os.path.exists(self.cache_dir):
                return {
                    'total_files': 0,
                    'total_size_mb': 0,
                    'valid_files': 0,
                    'expired_files': 0
                }
            
            total_files = 0
            total_size = 0
            valid_files = 0
            expired_files = 0
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.pkl'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                total_files += 1
                total_size += os.path.getsize(file_path)
                
                try:
                    with open(file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if self._is_cache_valid(cached_data['timestamp']):
                        valid_files += 1
                    else:
                        expired_files += 1
                        
                except Exception:
                    expired_files += 1
            
            return {
                'total_files': total_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'valid_files': valid_files,
                'expired_files': expired_files,
                'cache_dir': self.cache_dir,
                'cache_duration_hours': settings.CACHE_DURATION_HOURS
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}

# Instance globale du cache (singleton pattern)
_cache_instance = None

def get_cache_manager() -> CacheManager:
    """
    Retourne l'instance globale du gestionnaire de cache (singleton)
    
    Returns:
        Instance du CacheManager
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance
