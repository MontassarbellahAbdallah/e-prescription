"""
Gestionnaire des embeddings pour la recherche sémantique
"""
from typing import List, Dict, Optional, Any, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from config.settings import settings
from config.logging_config import get_logger
from core.exceptions import VectorStoreError
from core.cache_manager import get_cache_manager
from utils.helpers import measure_execution_time

logger = get_logger(__name__)

class EmbeddingManager:
    """
    Gestionnaire centralisé des embeddings
    
    Fonctionnalités:
    - Gestion unifiée des embeddings Google
    - Cache des embeddings calculés
    - Métriques de similarité
    - Optimisation des performances
    """
    
    def __init__(self, model_name: str = None, use_cache: bool = True):
        """
        Initialise le gestionnaire d'embeddings
        
        Args:
            model_name: Nom du modèle d'embedding (utilise settings par défaut)
            use_cache: Utiliser le cache pour les embeddings
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.model_name)
        self.cache_manager = get_cache_manager() if use_cache else None
        
        # Statistiques
        self.stats = {
            'embeddings_created': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'similarity_calculations': 0
        }
    
    @measure_execution_time
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Crée les embeddings pour une liste de documents
        
        Args:
            texts: Liste des textes à embedder
            
        Returns:
            Liste des vecteurs d'embedding
            
        Raises:
            VectorStoreError: Si la création d'embeddings échoue
        """
        if not texts:
            return []
        
        try:
            logger.info(f"Creating embeddings for {len(texts)} documents")
            
            # Créer les embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Mettre à jour les statistiques
            self.stats['embeddings_created'] += len(texts)
            
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to create embeddings: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    @measure_execution_time
    def embed_query(self, text: str) -> List[float]:
        """
        Crée l'embedding pour une requête
        
        Args:
            text: Texte de la requête
            
        Returns:
            Vecteur d'embedding de la requête
        """
        if not text or not text.strip():
            return []
        
        # Vérifier le cache
        cache_key = f"query_embed_{text}"
        if self.cache_manager:
            cached_embedding = self.cache_manager.get(cache_key)
            if cached_embedding is not None:
                self.stats['cache_hits'] += 1
                return cached_embedding
            self.stats['cache_misses'] += 1
        
        try:
            # Créer l'embedding
            embedding = self.embeddings.embed_query(text)
            
            # Mettre en cache
            if self.cache_manager:
                self.cache_manager.set(cache_key, embedding)
            
            self.stats['embeddings_created'] += 1
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to create query embedding: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings
        
        Args:
            embedding1: Premier vecteur d'embedding
            embedding2: Deuxième vecteur d'embedding
            
        Returns:
            Score de similarité (0-1, plus proche de 1 = plus similaire)
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        try:
            # Convertir en arrays numpy
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculer la similarité cosinus
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            self.stats['similarity_calculations'] += 1
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        document_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Trouve les documents les plus similaires à une requête
        
        Args:
            query_embedding: Embedding de la requête
            document_embeddings: Liste des embeddings de documents
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de tuples (index, score_de_similarité) triés par similarité
        """
        if not query_embedding or not document_embeddings:
            return []
        
        similarities = []
        
        for i, doc_embedding in enumerate(document_embeddings):
            similarity = self.calculate_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def batch_similarity(
        self, 
        queries: List[str], 
        documents: List[str]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Calcule les similarités pour plusieurs requêtes et documents
        
        Args:
            queries: Liste des requêtes
            documents: Liste des documents
            
        Returns:
            Dictionnaire {requête: [(document, similarité), ...]}
        """
        results = {}
        
        # Créer les embeddings
        query_embeddings = [self.embed_query(q) for q in queries]
        doc_embeddings = self.embed_documents(documents)
        
        for i, query in enumerate(queries):
            query_embedding = query_embeddings[i]
            similarities = []
            
            for j, doc in enumerate(documents):
                doc_embedding = doc_embeddings[j]
                similarity = self.calculate_similarity(query_embedding, doc_embedding)
                similarities.append((doc, similarity))
            
            # Trier par similarité
            similarities.sort(key=lambda x: x[1], reverse=True)
            results[query] = similarities
        
        return results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques des embeddings
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = self.stats.copy()
        
        # Calculer le taux de cache hit
        total_queries = stats['cache_hits'] + stats['cache_misses']
        if total_queries > 0:
            stats['cache_hit_rate'] = (stats['cache_hits'] / total_queries) * 100
        else:
            stats['cache_hit_rate'] = 0
        
        stats['model_name'] = self.model_name
        
        return stats


