"""
Processeur RAG optimisé pour la gestion du Vector Store et la recherche
"""
import os
import pickle
import sys
import hashlib
import json
import glob
import shutil
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import re

from config.settings import settings
from config.logging_config import get_logger
from core.exceptions import VectorStoreError
from core.cache_manager import get_cache_manager
from data.pdf_processor import PDFProcessor, PDFTextChunker
from utils.helpers import measure_execution_time, format_file_size

logger = get_logger(__name__)

class RAGProcessor:
    """
    Processeur RAG optimisé pour la recherche documentaire
    
    Fonctionnalités:
    - Création et gestion du Vector Store FAISS
    - Recherche sémantique avec scores
    - Cache intégré pour les recherches
    - Sauvegarde et chargement automatiques
    - Métadonnées enrichies
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialise le processeur RAG
        
        Args:
            cache_enabled: Activer le cache pour les recherches
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.vector_store = None
        self.document_metadata = {}
        self.cache_manager = get_cache_manager() if cache_enabled else None
        
        # Optimisations pour _save_stats
        self._last_stats_hash = None
        self._stats_save_count = 0
        
        # Statistiques
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'index_size_mb': 0,
            'last_updated': None,
            'search_count': 0
        }
    
    @measure_execution_time
    def create_vector_store_sequential(self, pdf_paths: List[str], force_rebuild: bool = False, progress_callback=None) -> bool:
        """
        Crée le Vector Store en mode séquentiel avec sauvegarde après chaque PDF
        
        Args:
            pdf_paths: Liste des chemins vers les fichiers PDF
            force_rebuild: Forcer la reconstruction même si l'index existe
            progress_callback: Fonction de callback pour la progression
            
        Returns:
            True si la création a réussi
            
        Raises:
            VectorStoreError: Si la création échoue
        """
        try:
            # Vérifier si l'index existe déjà
            if not force_rebuild and self._index_exists():
                logger.info("Vector store already exists, loading instead of creating")
                return self.load_vector_store()
            
            if not pdf_paths:
                raise VectorStoreError("No PDF files provided for vector store creation")
            
            logger.info(f"Creating vector store sequentially from {len(pdf_paths)} PDF files")
            
            # Nettoyer l'index existant si reconstruction forcée
            if force_rebuild:
                self.clear_index()
            
            # Initialiser le processeur PDF avec métadonnées enrichies
            processor = PDFProcessor(validate_files=True, parallel_processing=False)
            
            # Variables pour accumuler les résultats
            combined_chunks = []
            combined_metadata = {}
            chunk_offset = 0
            successful_pdfs = 0
            failed_pdfs = 0
            
            # Traiter chaque PDF séquentiellement
            for i, pdf_path in enumerate(pdf_paths, 1):
                if progress_callback:
                    progress_callback(i, len(pdf_paths), f"Processing {os.path.basename(pdf_path)}")
                
                try:
                    logger.info(f"Processing PDF {i}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
                    
                    # Extraire avec métadonnées enrichies
                    text, metadata, doc_stats = processor.extract_with_enriched_metadata(pdf_path)
                    
                    if text.strip():
                        # Créer les chunks
                        chunker = PDFTextChunker()
                        chunks, chunk_metadata = chunker.create_chunks_with_metadata(text, metadata)
                        
                        if chunks:
                            # Ajuster les IDs des chunks pour éviter les conflits
                            adjusted_chunks = []
                            adjusted_metadata = {}
                            
                            for j, chunk in enumerate(chunks):
                                new_chunk_id = chunk_offset + j
                                adjusted_chunks.append(chunk)
                                
                                # Utiliser les métadonnées enrichies du chunk
                                original_meta = chunk_metadata.get(j, {})
                                adjusted_metadata[new_chunk_id] = original_meta
                            
                            # Ajouter aux résultats combinés
                            combined_chunks.extend(adjusted_chunks)
                            combined_metadata.update(adjusted_metadata)
                            chunk_offset += len(chunks)
                            
                            successful_pdfs += 1
                            
                            logger.info(
                                f"✅ PDF {i}/{len(pdf_paths)} processed: {len(chunks)} chunks, "
                                f"{doc_stats.get('total_drug_mentions', 0)} drug mentions"
                            )
                            
                            # Sauvegarde intermédiaire après chaque PDF réussi
                            self._save_intermediate_progress(combined_chunks, combined_metadata, i, len(pdf_paths))
                        else:
                            logger.warning(f"No chunks created from {os.path.basename(pdf_path)}")
                    else:
                        logger.warning(f"No text extracted from {os.path.basename(pdf_path)}")
                        
                except Exception as e:
                    failed_pdfs += 1
                    logger.error(f"⚠️ Problem with PDF {i}/{len(pdf_paths)} [{os.path.basename(pdf_path)}]: {e}")
                    # Continuer avec le PDF suivant
                    continue
            
            # Créer le Vector Store final avec tous les chunks
            if combined_chunks:
                success = self._create_faiss_index(combined_chunks, combined_metadata)
                
                if success:
                    # Mettre à jour les statistiques finales
                    self.stats.update({
                        'total_documents': successful_pdfs,
                        'total_chunks': len(combined_chunks),
                        'last_updated': datetime.now().isoformat(),
                        'search_count': 0,
                        'failed_documents': failed_pdfs
                    })
                    
                    logger.info(
                        f"✅ Vector store created successfully: "
                        f"{successful_pdfs}/{len(pdf_paths)} documents processed, "
                        f"{len(combined_chunks)} total chunks"
                    )
                    
                    return True
                else:
                    raise VectorStoreError("Failed to create FAISS index from processed chunks")
            else:
                raise VectorStoreError("No valid chunks extracted from any PDF files")
                
        except Exception as e:
            error_msg = f"Failed to create vector store sequentially: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def _save_intermediate_progress(self, chunks: List[str], metadata: Dict, current_pdf: int, total_pdfs: int) -> None:
        """
        Sauvegarde intermédiaire après chaque PDF
        
        Args:
            chunks: Chunks accumulés jusqu'à maintenant
            metadata: Métadonnées accumulées
            current_pdf: PDF actuel
            total_pdfs: Total de PDFs
        """
        try:
            # Créer le répertoire d'index si nécessaire
            os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
            
            # Sauvegarder les métadonnées intermédiaires
            metadata_file = os.path.join(settings.FAISS_INDEX_DIR, "metadata_progress.pkl")
            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f)
            
            # Sauvegarder les stats de progression
            progress_stats = {
                'processed_pdfs': current_pdf,
                'total_pdfs': total_pdfs,
                'total_chunks_so_far': len(chunks),
                'timestamp': datetime.now().isoformat()
            }
            
            progress_file = os.path.join(settings.FAISS_INDEX_DIR, "progress.pkl")
            with open(progress_file, "wb") as f:
                pickle.dump(progress_stats, f)
            
            logger.debug(f"Intermediate progress saved: {current_pdf}/{total_pdfs} PDFs, {len(chunks)} chunks")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate progress: {e}")
    
    @measure_execution_time
    def create_vector_store_from_pdfs(self, pdf_paths: List[str], force_rebuild: bool = False) -> bool:
        """
        Crée le Vector Store à partir de fichiers PDF (utilise maintenant le mode séquentiel)
        
        Args:
            pdf_paths: Liste des chemins vers les fichiers PDF
            force_rebuild: Forcer la reconstruction même si l'index existe
            
        Returns:
            True si la création a réussi
            
        Raises:
            VectorStoreError: Si la création échoue
        """
        # Utiliser maintenant seulement le mode séquentiel avec métadonnées enrichies
        return self.create_vector_store_sequential(pdf_paths, force_rebuild)
    
    def _create_faiss_index(self, text_chunks: List[str], chunk_metadata: Dict) -> bool:
        """
        Crée l'index FAISS avec les chunks de texte (métadonnées unifiées)
        
        Args:
            text_chunks: Liste des chunks de texte (déjà avec marqueurs DOC_META)
            chunk_metadata: Métadonnées associées aux chunks
            
        Returns:
            True si la création a réussi
        """
        try:
            # NE PAS ajouter de nouveaux marqueurs - utiliser les DOC_META existants
            texts = text_chunks.copy()  # Utiliser directement les chunks avec leurs marqueurs DOC_META
            
            # Créer le répertoire d'index
            os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
            
            # Créer le Vector Store FAISS
            logger.info(f"Creating FAISS embeddings from {len(texts)} chunks...")
            self.vector_store = FAISS.from_texts(texts, embedding=self.embeddings)
            
            # Sauvegarder l'index
            self.vector_store.save_local(settings.FAISS_INDEX_DIR)
            logger.info(f"FAISS index saved to {settings.FAISS_INDEX_DIR}")
            
            # Sauvegarder les métadonnées
            self.document_metadata = chunk_metadata
            metadata_file = os.path.join(settings.FAISS_INDEX_DIR, "metadata.pkl")
            with open(metadata_file, "wb") as f:
                pickle.dump(chunk_metadata, f)
            
            # Sauvegarder les statistiques
            self._save_stats(force=True)
            
            logger.info(f"Metadata saved: {len(chunk_metadata)} chunks with unified DOC_META system")
            return True
            
        except Exception as e:
            error_msg = f"Error creating FAISS index: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    @measure_execution_time
    def load_vector_store(self) -> bool:
        """
        Charge le Vector Store existant (amélioré pour le rechargement)
        
        Returns:
            True si le chargement a réussi
        """
        try:
            if not self._index_exists():
                logger.warning("No existing vector store found - files missing")
                return False
            
            logger.info("Loading existing FAISS vector store...")
            
            # Charger l'index FAISS
            self.vector_store = FAISS.load_local(
                settings.FAISS_INDEX_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully")
            
            # Charger les métadonnées
            self._load_metadata()
            
            # Charger les statistiques
            self._load_stats()
            
            # CORRECTION: Recalculer les stats à partir des métadonnées chargées
            if self.document_metadata:
                unique_docs = set()
                total_chunks = len(self.document_metadata)
                
                for metadata in self.document_metadata.values():
                    doc_name = metadata.get('document', 'Unknown')
                    if doc_name and doc_name != 'Unknown':
                        unique_docs.add(doc_name)
                
                # Mettre à jour les stats avec les vraies valeurs
                self.stats.update({
                    'total_documents': len(unique_docs),
                    'total_chunks': total_chunks
                })
                
                # Sauvegarder les stats recalculées
                self._save_stats(force=True)
                
                logger.info(f"Stats recalculated: {len(unique_docs)} docs, {total_chunks} chunks")
            
            # Vérifier la cohérence
            if self.vector_store and len(self.document_metadata) > 0:
                logger.info(
                    f"✅ Vector store loaded successfully: {self.stats['total_chunks']} chunks from "
                    f"{self.stats['total_documents']} documents"
                )
                
                # Vérification rapide des métadonnées
                self.quick_metadata_check()
                
                return True
            else:
                logger.warning("Vector store loaded but metadata is empty or inconsistent")
                # Essayer de reconstruire les métadonnées si possible
                if self.vector_store:
                    logger.info("Attempting to reconstruct basic metadata...")
                    self._reconstruct_basic_metadata()
                    return True
                return False
                
        except Exception as e:
            error_msg = f"Failed to load vector store: {e}"
            logger.error(error_msg, exc_info=True)
            
            # Nettoyer en cas d'échec
            self.vector_store = None
            self.document_metadata = {}
            return False
    
    def _load_metadata(self) -> None:
        """Charge les métadonnées depuis le fichier"""
        metadata_file = os.path.join(settings.FAISS_INDEX_DIR, "metadata.pkl")
        
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, "rb") as f:
                    self.document_metadata = pickle.load(f)
                logger.info(f"Metadata loaded: {len(self.document_metadata)} chunks")
            else:
                logger.warning("No metadata file found")
                self.document_metadata = {}
        except Exception as e:
            logger.warning(f"Error loading metadata: {e}")
            self.document_metadata = {}
    
    def _load_stats(self) -> None:
        """Charge les statistiques depuis le fichier"""
        try:
            stats_file = os.path.join(settings.FAISS_INDEX_DIR, "stats.pkl")
            if os.path.exists(stats_file):
                with open(stats_file, "rb") as f:
                    loaded_stats = pickle.load(f)
                    self.stats.update(loaded_stats)
                logger.info("Stats loaded from file")
            else:
                logger.info("No stats file found")
        except Exception as e:
            logger.warning(f"Error loading stats: {e}")
    
    def _save_stats(self, force: bool = False) -> bool:
        """Version simplifiée mais robuste"""
        try:
            # Validation basique
            if not isinstance(self.stats, dict):
                return False
            
            # Enrichir les stats
            enriched_stats = self.stats.copy()
            enriched_stats['save_timestamp'] = datetime.now().isoformat()
            
            # Sauvegarde atomique
            stats_file = os.path.join(settings.FAISS_INDEX_DIR, "stats.pkl")
            temp_file = f"{stats_file}.tmp"
            
            with open(temp_file, "wb") as f:
                pickle.dump(enriched_stats, f)
            
            # Déplacement atomique
            os.replace(temp_file, stats_file)  # Plus simple que votre logique Windows/Unix
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
            return False
    
    def _reconstruct_basic_metadata(self) -> None:
        """Reconstruit des métadonnées de base si elles sont manquantes"""
        try:
            if not self.vector_store:
                return
            
            logger.info("Reconstructing basic metadata from vector store...")
            
            # Faire une recherche de test pour obtenir des documents
            test_docs = self.vector_store.similarity_search("test", k=10)
            
            reconstructed_meta = {}
            
            for i, doc in enumerate(test_docs):
                # Essayer d'extraire les informations DOC_META existantes
                content = doc.page_content
                
                # Chercher les marqueurs DOC_META
                import re
                match = re.search(r'\[DOC_META_(\d+)\]', content)
                
                if match:
                    chunk_id = int(match.group(1))
                    reconstructed_meta[chunk_id] = {
                        'document': 'Document reconstruit',
                        'page': 'Page inconnue',
                        'text_length': len(content),
                        'extraction_time': datetime.now().isoformat(),
                        'file_path': 'Chemin inconnu',
                        'file_size': 0
                    }
            
            if reconstructed_meta:
                self.document_metadata = reconstructed_meta
                
                # Recalculer les stats
                self.stats.update({
                    'total_documents': len(set(meta.get('document', 'Unknown') for meta in reconstructed_meta.values())),
                    'total_chunks': len(reconstructed_meta)
                })
                
                logger.info(f"Reconstructed metadata for {len(reconstructed_meta)} chunks")
            
        except Exception as e:
            logger.warning(f"Failed to reconstruct metadata: {e}")
    
    def _index_exists(self) -> bool:
        """Vérifie si l'index FAISS existe"""
        required_files = [
            os.path.join(settings.FAISS_INDEX_DIR, "index.faiss"),
            os.path.join(settings.FAISS_INDEX_DIR, "index.pkl")
        ]
        return all(os.path.exists(f) for f in required_files)
    
    @measure_execution_time
    def search_documents_with_sources(
        self, 
        query: str, 
        k: int = None,
        score_threshold: float = None
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Recherche sémantique avec sources et scores (logging amélioré)
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats (utilise settings par défaut)
            score_threshold: Seuil de score minimal
            
        Returns:
            Tuple (documents, informations_sources)
        """
        logger.info(f"Starting RAG search: query='{query[:100]}...', k={k}, threshold={score_threshold}")
        
        if not self.vector_store:
            logger.error("No vector store available for search - RAG not initialized")
            return [], []
        
        if k is None:
            k = settings.DEFAULT_SEARCH_RESULTS
            logger.debug(f"Using default search results count: {k}")
        
        # Vérifier le cache avec préfixe spécifique
        cache_key = f"search_{query}_{k}_{score_threshold}"
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key, prefix="search_")
            if cached_result:
                logger.info(f"Search cache hit for: {query[:50]}... - returning {len(cached_result.get('docs', []))} cached results")
                return cached_result['docs'], cached_result['sources']
        
        try:
            logger.debug(f"Performing FAISS similarity search with {k} results")
            # Recherche avec scores
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"FAISS search completed: {len(docs_with_scores)} raw results found")
            
            docs = []
            sources_info = []
            filtered_count = 0
            
            for i, (doc, score) in enumerate(docs_with_scores):
                logger.debug(f"Processing result {i+1}/{len(docs_with_scores)}: score={score:.4f}")
                
                # Filtrer par score si spécifié
                if score_threshold is not None and score > score_threshold:
                    filtered_count += 1
                    logger.debug(f"Filtered out result {i+1} due to low score: {score:.4f} > {score_threshold}")
                    continue
                
                # Extraire et nettoyer le contenu
                logger.debug(f"Extracting chunk info for result {i+1}")
                chunk_info = self._extract_chunk_info(doc)
                cleaned_content = self._clean_document_content(doc.page_content)
                
                logger.debug(f"Result {i+1} metadata: doc='{chunk_info['document']}', page={chunk_info['page']}, chunk_id={chunk_info['chunk_id']}")
                
                # Créer le document nettoyé
                clean_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
                docs.append(clean_doc)
                
                # Informations sur la source
                source_info = {
                    'document': chunk_info['document'],
                    'page': chunk_info['page'],
                    'relevance_score': float(score),
                    'content_preview': self._create_content_preview(cleaned_content),
                    'full_content': cleaned_content,
                    'chunk_info': chunk_info
                }
                sources_info.append(source_info)
            
            # Mettre en cache avec préfixe spécifique
            if self.cache_manager:
                result = {'docs': docs, 'sources': sources_info}
                self.cache_manager.set(cache_key, result, prefix="search_")
                logger.debug(f"Search results cached with key: search_{cache_key[:20]}...")
            
            # Mettre à jour les statistiques et sauvegarder si nécessaire
            self.stats['search_count'] += 1
            
            # Sauvegarder les stats tous les 10 recherches (optimisation)
            if self.stats['search_count'] % 10 == 0:
                self._save_stats()  # Sauvegarde normale (pas forcée)
            
            logger.info(f"Search completed successfully: {len(docs)} final results (filtered: {filtered_count}) for '{query[:50]}...'")
            return docs, sources_info
            
        except Exception as e:
            error_msg = f"Search failed for query '{query[:50]}...': {e}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg)
    
    def search_with_detailed_sources(
        self, 
        query: str, 
        k: int = None,
        score_threshold: float = None
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Recherche avec sources détaillées et citations exactes
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats
            score_threshold: Seuil de score minimal
            
        Returns:
            Tuple (documents, informations_sources_détaillées)
        """
        # Utiliser la recherche standard puis enrichir les sources
        docs, sources_info = self.search_documents_with_sources(query, k, score_threshold)
        
        # Enrichir avec des citations exactes et un contexte détaillé
        detailed_sources = []
        
        for i, (doc, source) in enumerate(zip(docs, sources_info)):
            # Extraire la citation exacte
            exact_quote = self._extract_exact_quote(doc.page_content, query)
            
            # Enrichir les informations de source
            detailed_source = {
                **source,  # Garder toutes les infos existantes
                'exact_quote': exact_quote,
                'citation_context': self._get_citation_context(doc.page_content, exact_quote),
                'academic_citation': self._format_academic_citation(source),
                'relevance_explanation': self._explain_relevance(doc.page_content, query),
                'document_section': source.get('chunk_info', {}).get('metadata', {}).get('section_title', 'Section inconnue'),
                'guideline_type': source.get('chunk_info', {}).get('metadata', {}).get('guideline_type', 'Type inconnu'),
                'full_content': doc.page_content  # Ajouter le contenu complet
            }
            
            detailed_sources.append(detailed_source)
        
        return docs, detailed_sources
    
    def _extract_exact_quote(self, content: str, query: str) -> str:
        """
        Extrait la phrase exacte qui contient l'information pertinente
        
        Args:
            content: Contenu du document
            query: Requête de recherche
            
        Returns:
            Citation exacte
        """
        import re
        
        # Nettoyer le contenu
        clean_content = self._clean_document_content(content)
        
        # Diviser en phrases
        sentences = re.split(r'[.!?]+', clean_content)
        
        # Extraire les mots-clés de la requête
        query_words = [word.lower().strip() for word in re.findall(r'\b\w+\b', query) if len(word) > 2]
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Ignorer les phrases trop courtes
                continue
                
            sentence_lower = sentence.lower()
            
            # Compter les mots-clés présents
            score = sum(1 for word in query_words if word in sentence_lower)
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence[:200] + "..." if len(best_sentence) > 200 else best_sentence
    
    def _get_citation_context(self, content: str, exact_quote: str) -> str:
        """
        Obtient le contexte autour de la citation (100 caractères avant/après)
        
        Args:
            content: Contenu complet
            exact_quote: Citation exacte
            
        Returns:
            Contexte avec citation mise en évidence
        """
        if not exact_quote or exact_quote not in content:
            return content[:300] + "..." if len(content) > 300 else content
        
        # Trouver la position de la citation
        quote_start = content.find(exact_quote)
        
        # Extraire 100 caractères avant et après
        context_start = max(0, quote_start - 100)
        context_end = min(len(content), quote_start + len(exact_quote) + 100)
        
        context = content[context_start:context_end]
        
        # Ajouter des indicateurs si le contexte est tronqué
        if context_start > 0:
            context = "..." + context
        if context_end < len(content):
            context = context + "..."
        
        return context
    
    def _format_academic_citation(self, source: Dict) -> str:
        """
        Formate une citation académique médicale
        
        Args:
            source: Informations sur la source
            
        Returns:
            Citation formatée
        """
        document = source.get('document', 'Document inconnu')
        page = source.get('page', 'Page inconnue')
        
        # Vérifier si on a des vraies informations
        if document == 'Document inconnu' or page == 'Page inconnue':
            # Essayer de récupérer depuis chunk_info
            chunk_info = source.get('chunk_info', {})
            if chunk_info:
                metadata = chunk_info.get('metadata', {})
                document = metadata.get('source_file', metadata.get('document', document))
                page = metadata.get('page', page)
        
        # Déterminer le type de document et formater en conséquence
        document_lower = str(document).lower()
        if 'beers' in document_lower:
            return f"American Geriatrics Society Beers Criteria (Page {page})"
        elif 'stopp' in document_lower or 'start' in document_lower:
            return f"STOPP/START Criteria (Page {page})"
        elif 'laroche' in document_lower:
            return f"Liste de Laroche (Page {page})"
        elif 'priscus' in document_lower:
            return f"PRISCUS List (Page {page})"
        else:
            return f"{document} (Page {page})"
    
    def _explain_relevance(self, content: str, query: str) -> str:
        """
        Explique pourquoi ce contenu est pertinent pour la requête
        
        Args:
            content: Contenu du document
            query: Requête de recherche
            
        Returns:
            Explication de la pertinence
        """
        import re
        
        # Extraire les mots-clés de la requête
        query_words = [word.lower().strip() for word in re.findall(r'\b\w+\b', query) if len(word) > 2]
        content_lower = content.lower()
        
        # Compter les correspondances
        matches = [word for word in query_words if word in content_lower]
        
        if len(matches) > 0:
            return f"Contient les termes : {', '.join(matches[:3])}"
        else:
            return "Pertinence sémantique détectée par l'IA"
    
    def debug_search_metadata(self, query: str, k: int = 3) -> Dict:
        """
        Méthode de debug pour vérifier les métadonnées
        
        Args:
            query: Requête de test
            k: Nombre de résultats
            
        Returns:
            Informations de debug
        """
        logger.info(f"DEBUG: Starting metadata debug for query: {query}")
        
        if not self.vector_store:
            return {'error': 'No vector store available'}
        
        # Faire une recherche standard
        docs, sources_info = self.search_documents_with_sources(query, k)
        
        debug_info = {
            'total_metadata_chunks': len(self.document_metadata),
            'search_results_count': len(docs),
            'sources_info_count': len(sources_info),
            'sample_metadata': {},
            'sample_source_info': {},
            'vector_store_status': 'loaded' if self.vector_store else 'not_loaded'
        }
        
        # Échantillon de métadonnées
        if self.document_metadata:
            sample_key = list(self.document_metadata.keys())[0]
            debug_info['sample_metadata'] = self.document_metadata[sample_key]
        
        # Échantillon d'info de source
        if sources_info:
            debug_info['sample_source_info'] = sources_info[0]
        
        logger.info(f"DEBUG results: {debug_info}")
        return debug_info
    
    def quick_metadata_check(self) -> None:
        """
        Vérification rapide des métadonnées pour debug
        """
        logger.info(f"=== QUICK METADATA CHECK ===")
        logger.info(f"Total metadata chunks: {len(self.document_metadata)}")
        logger.info(f"Vector store loaded: {self.vector_store is not None}")
        
        if self.document_metadata:
            # Analyser les 3 premiers chunks
            for i, (chunk_id, metadata) in enumerate(list(self.document_metadata.items())[:3]):
                logger.info(f"Chunk {chunk_id}: {metadata.get('source_file', metadata.get('document', 'NO_NAME'))} - Page {metadata.get('page', 'NO_PAGE')}")
        else:
            logger.warning("NO METADATA FOUND - Vector store may need to be rebuilt with sequential method")
    
    def _extract_chunk_info(self, doc: Document) -> Dict:
        """
        Extrait les informations du chunk depuis le document (système DOC_META unifié)
        
        Args:
            doc: Document FAISS
            
        Returns:
            Informations du chunk
        """
        # Chercher l'ID du chunk dans le contenu avec le bon pattern DOC_META
        match = re.search(r"\[DOC_META_(\d+)\]", doc.page_content)
        
        if match:
            chunk_id = int(match.group(1))
            chunk_metadata = self.document_metadata.get(chunk_id, {})
            
            logger.debug(f"Found DOC_META chunk_id {chunk_id} with metadata: {chunk_metadata.get('document', 'Unknown')}")
            
            # Utiliser les nouvelles clés de métadonnées enrichies si disponibles
            document_name = (
                chunk_metadata.get('source_file') or 
                chunk_metadata.get('document') or 
                'Document inconnu'
            )
            
            page_info = chunk_metadata.get('page', 'Page inconnue')
            
            return {
                'chunk_id': chunk_id,
                'document': document_name,
                'page': page_info,
                'text_length': chunk_metadata.get('text_length', len(doc.page_content)),
                'file_path': chunk_metadata.get('file_path', ''),
                'extraction_time': chunk_metadata.get('extraction_time', ''),
                'file_size': chunk_metadata.get('file_size', 0),
                'metadata': chunk_metadata  # Garder toutes les métadonnées enrichies
            }
        
        # Fallback: essayer de deviner à partir du contenu
        logger.warning(f"No DOC_META marker found in content: {doc.page_content[:100]}...")
        
        # Essayer de détecter le type de document depuis le contenu
        content_lower = doc.page_content.lower()
        detected_doc = 'Document inconnu'
        detected_page = 'Page inconnue'
        
        # Détection basique depuis le contenu
        if 'stopp' in content_lower or 'start' in content_lower:
            detected_doc = 'STOPP/START Criteria'
        elif 'beers' in content_lower:
            detected_doc = 'AGS Beers Criteria'
        elif 'laroche' in content_lower:
            detected_doc = 'Liste de Laroche'
        elif 'priscus' in content_lower:
            detected_doc = 'PRISCUS List'
        
        # Essayer de détecter un numéro de page
        page_match = re.search(r'page\s+(\d+)', content_lower)
        if page_match:
            detected_page = page_match.group(1)
        
        return {
            'chunk_id': None,
            'document': detected_doc,
            'page': detected_page,
            'text_length': len(doc.page_content),
            'file_path': '',
            'extraction_time': '',
            'file_size': 0,
            'metadata': {
                'detected_from_content': True,
                'detection_method': 'fallback_content_analysis'
            }
        }
    
    def _clean_document_content(self, content: str) -> str:
        """
        Nettoie le contenu du document en supprimant UNIQUEMENT les marqueurs (pas le contenu)
        
        Args:
            content: Contenu brut avec marqueurs DOC_META
            
        Returns:
            Contenu nettoyé sans marqueurs mais avec le texte intégral
        """
        if not content:
            return ""
        
        # Supprimer UNIQUEMENT les marqueurs DOC_META (garder le contenu)
        # Pattern: [DOC_META_\d+]contenu[/DOC_META_\d+] -> contenu
        cleaned = re.sub(r'\[DOC_META_\d+\]', '', content)
        cleaned = re.sub(r'\[/DOC_META_\d+\]', '', cleaned)
        
        # Supprimer les anciens marqueurs INDEX si présents (legacy)
        cleaned = re.sub(r'\[INDEX_\d+\]', '', cleaned)
        
        # Nettoyer les espaces multiples créés par la suppression des marqueurs
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _create_content_preview(self, content: str, max_length: int = 200) -> str:
        """
        Crée un aperçu du contenu
        
        Args:
            content: Contenu complet
            max_length: Longueur maximale de l'aperçu
            
        Returns:
            Aperçu du contenu
        """
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du Vector Store
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = self.stats.copy()
        
        # Ajouter des informations sur la taille de l'index
        if self._index_exists():
            try:
                index_files = [
                    os.path.join(settings.FAISS_INDEX_DIR, "index.faiss"),
                    os.path.join(settings.FAISS_INDEX_DIR, "index.pkl"),
                    os.path.join(settings.FAISS_INDEX_DIR, "metadata.pkl")
                ]
                
                total_size = sum(
                    os.path.getsize(f) for f in index_files if os.path.exists(f)
                )
                stats['index_size_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception:
                stats['index_size_mb'] = 0
        
        stats['vector_store_loaded'] = self.vector_store is not None
        stats['metadata_count'] = len(self.document_metadata)
        
        return stats
    

    
    def _load_stats(self) -> None:
        """Charge les statistiques et les recalcule si nécessaire"""
        try:
            stats_file = os.path.join(settings.FAISS_INDEX_DIR, "stats.pkl")
            if os.path.exists(stats_file):
                with open(stats_file, "rb") as f:
                    loaded_stats = pickle.load(f)
                    self.stats.update(loaded_stats)
                logger.info("Stats loaded from file")
            else:
                logger.info("No stats file found - will recalculate")
        except Exception as e:
            logger.warning(f"Error loading stats: {e}")
        
        # IMPORTANT: Recalculer les stats depuis les métadonnées si elles sont disponibles
        if len(self.document_metadata) > 0:
            logger.info("Recalculating stats from metadata...")
            
            # Calculer le nombre de documents uniques
            unique_docs = set()
            total_chars = 0
            
            for chunk_metadata in self.document_metadata.values():
                doc_name = chunk_metadata.get('document', 'Unknown')
                unique_docs.add(doc_name)
                total_chars += chunk_metadata.get('text_length', 0)
            
            # Mettre à jour les stats
            self.stats.update({
                'total_documents': len(unique_docs),
                'total_chunks': len(self.document_metadata),
                'total_chars': total_chars,
                'last_updated': datetime.now().isoformat()
            })
            
            logger.info(f"Stats recalculated: {self.stats['total_documents']} docs, {self.stats['total_chunks']} chunks")
            
            # Sauvegarder les stats mises à jour
            self._save_stats(force=True)
    
    def recalculate_stats(self) -> bool:
        """Force le recalcul des statistiques depuis les métadonnées"""
        try:
            if len(self.document_metadata) == 0:
                logger.warning("No metadata available for stats calculation")
                return False
            
            logger.info("Force recalculating statistics...")
            
            # Calculer le nombre de documents uniques
            unique_docs = set()
            total_chars = 0
            pages_count = 0
            
            for chunk_metadata in self.document_metadata.values():
                doc_name = chunk_metadata.get('document', 'Unknown')
                unique_docs.add(doc_name)
                total_chars += chunk_metadata.get('text_length', 0)
                if chunk_metadata.get('page'):
                    pages_count += 1
            
            # Mettre à jour les stats
            self.stats.update({
                'total_documents': len(unique_docs),
                'total_chunks': len(self.document_metadata),
                'total_chars': total_chars,
                'total_pages': pages_count,
                'last_updated': datetime.now().isoformat(),
                'search_count': self.stats.get('search_count', 0)  # Préserver le compteur de recherches
            })
            
            # Calculer la taille de l'index
            try:
                index_files = [
                    os.path.join(settings.FAISS_INDEX_DIR, "index.faiss"),
                    os.path.join(settings.FAISS_INDEX_DIR, "index.pkl"),
                    os.path.join(settings.FAISS_INDEX_DIR, "metadata.pkl")
                ]
                
                total_size = sum(
                    os.path.getsize(f) for f in index_files if os.path.exists(f)
                )
                self.stats['index_size_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception:
                self.stats['index_size_mb'] = 0
            
            # Sauvegarder
            self._save_stats(force=True)
            
            logger.info(
                f"Stats recalculated successfully: {self.stats['total_documents']} documents, "
                f"{self.stats['total_chunks']} chunks, {self.stats['index_size_mb']} MB"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recalculating stats: {e}")
            return False
    
    def rebuild_index(self, pdf_paths: List[str]) -> bool:
        """
        Reconstruit complètement l'index du Vector Store
        
        Args:
            pdf_paths: Liste des chemins PDF
            
        Returns:
            True si la reconstruction a réussi
        """
        logger.info("Rebuilding vector store index...")
        
        # Supprimer l'ancien index
        self.clear_index()
        
        # Créer le nouvel index
        return self.create_vector_store_from_pdfs(pdf_paths, force_rebuild=True)
    
    def clear_index(self) -> bool:
        """
        Supprime l'index existant
        
        Returns:
            True si la suppression a réussi
        """
        try:
            if os.path.exists(settings.FAISS_INDEX_DIR):
                import shutil
                shutil.rmtree(settings.FAISS_INDEX_DIR)
                logger.info("Vector store index cleared")
            
            # Réinitialiser les variables
            self.vector_store = None
            self.document_metadata = {}
            self.stats = {
                'total_documents': 0,
                'total_chunks': 0,
                'index_size_mb': 0,
                'last_updated': None,
                'search_count': 0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False
    
    def get_document_list(self) -> List[Dict]:
        """
        Retourne la liste des documents indexés
        
        Returns:
            Liste des documents avec leurs métadonnées
        """
        documents = {}
        
        for chunk_id, metadata in self.document_metadata.items():
            doc_name = metadata.get('document', 'Unknown')
            
            if doc_name not in documents:
                documents[doc_name] = {
                    'name': doc_name,
                    'path': metadata.get('file_path', ''),
                    'pages': set(),
                    'chunks': 0,
                    'total_chars': 0
                }
            
            documents[doc_name]['pages'].add(metadata.get('page', 0))
            documents[doc_name]['chunks'] += 1
            documents[doc_name]['total_chars'] += metadata.get('text_length', 0)
        
        # Convertir en liste et formater
        result = []
        for doc_info in documents.values():
            doc_info['pages'] = len(doc_info['pages'])
            result.append(doc_info)
        
        return sorted(result, key=lambda x: x['name'])
    
    def diagnose_rag_system(self) -> Dict[str, Any]:
        """
        Diagnostic complet du système RAG pour débogage
        
        Returns:
            Rapport de diagnostic détaillé
        """
        logger.info("Starting RAG system diagnosis...")
        
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'vector_store_status': 'not_initialized',
            'index_files': {},
            'metadata_status': {},
            'cache_status': {},
            'document_analysis': {},
            'issues_found': [],
            'recommendations': []
        }
        
        # 1. Vérifier le statut du vector store
        if self.vector_store:
            diagnosis['vector_store_status'] = 'initialized'
        else:
            diagnosis['vector_store_status'] = 'not_initialized'
            diagnosis['issues_found'].append("Vector store not initialized")
            diagnosis['recommendations'].append("Run load_vector_store() or create_vector_store_from_pdfs()")
        
        # 2. Vérifier les fichiers d'index
        index_files = [
            ('index.faiss', 'FAISS index file'),
            ('index.pkl', 'FAISS metadata file'),
            ('metadata.pkl', 'Custom metadata file'),
            ('stats.pkl', 'Statistics file')
        ]
        
        for filename, description in index_files:
            filepath = os.path.join(settings.FAISS_INDEX_DIR, filename)
            diagnosis['index_files'][filename] = {
                'exists': os.path.exists(filepath),
                'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                'description': description
            }
            
            if not os.path.exists(filepath):
                diagnosis['issues_found'].append(f"Missing {description}: {filename}")
        
        # 3. Analyser les métadonnées
        diagnosis['metadata_status'] = {
            'total_chunks': len(self.document_metadata),
            'chunks_with_doc_info': 0,
            'chunks_with_page_info': 0,
            'unique_documents': set(),
            'doc_meta_pattern_found': 0
        }
        
        for chunk_id, metadata in self.document_metadata.items():
            if metadata.get('document'):
                diagnosis['metadata_status']['chunks_with_doc_info'] += 1
                diagnosis['metadata_status']['unique_documents'].add(metadata['document'])
            
            if metadata.get('page'):
                diagnosis['metadata_status']['chunks_with_page_info'] += 1
        
        diagnosis['metadata_status']['unique_documents'] = len(diagnosis['metadata_status']['unique_documents'])
        
        # 4. Analyser le cache
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            diagnosis['cache_status'] = cache_stats
        else:
            diagnosis['cache_status'] = {'error': 'Cache manager not available'}
            diagnosis['issues_found'].append("Cache manager not initialized")
        
        # 5. Test de recherche simple
        try:
            if self.vector_store:
                test_results = self.vector_store.similarity_search("test", k=1)
                diagnosis['search_test'] = {
                    'success': True,
                    'results_count': len(test_results),
                    'sample_content_length': len(test_results[0].page_content) if test_results else 0
                }
                
                # Vérifier les marqueurs DOC_META dans les résultats
                if test_results:
                    sample_content = test_results[0].page_content
                    has_doc_meta = '[DOC_META_' in sample_content
                    diagnosis['search_test']['has_doc_meta_markers'] = has_doc_meta
                    
                    if not has_doc_meta:
                        diagnosis['issues_found'].append("No DOC_META markers found in search results")
                        diagnosis['recommendations'].append("Rebuild vector store with proper metadata markers")
            else:
                diagnosis['search_test'] = {'success': False, 'error': 'Vector store not available'}
        except Exception as e:
            diagnosis['search_test'] = {'success': False, 'error': str(e)}
            diagnosis['issues_found'].append(f"Search test failed: {e}")
        
        # 6. Recommandations basées sur l'analyse
        if len(diagnosis['issues_found']) == 0:
            diagnosis['recommendations'].append("RAG system appears to be functioning correctly")
        else:
            if diagnosis['vector_store_status'] == 'not_initialized':
                diagnosis['recommendations'].append("Initialize the vector store first")
            
            if diagnosis['metadata_status']['total_chunks'] == 0:
                diagnosis['recommendations'].append("No metadata found - rebuild the index from PDF files")
            
            if not diagnosis['index_files']['index.faiss']['exists']:
                diagnosis['recommendations'].append("FAISS index missing - run create_vector_store_from_pdfs()")
        
        # 7. Score de santé global
        total_checks = 6
        issues_count = len(diagnosis['issues_found'])
        health_score = max(0, (total_checks - issues_count) / total_checks * 100)
        diagnosis['health_score'] = round(health_score, 1)
        
        logger.info(f"RAG diagnosis completed: {issues_count} issues found, health score: {health_score}%")
        
        return diagnosis
    
    # Méthodes auxiliaires pour _save_stats optimisé
    
    def _validate_stats(self) -> bool:
        """
        Valide les données statistiques avant sauvegarde
        
        Returns:
            True si les stats sont valides
        """
        try:
            # Vérifications de base
            if not isinstance(self.stats, dict):
                logger.error("Stats is not a dictionary")
                return False
            
            # Vérifier les champs obligatoires
            required_fields = ['total_documents', 'total_chunks', 'search_count']
            for field in required_fields:
                if field not in self.stats:
                    logger.error(f"Missing required field in stats: {field}")
                    return False
                
                if not isinstance(self.stats[field], (int, float)):
                    logger.error(f"Invalid type for {field}: expected number, got {type(self.stats[field])}")
                    return False
                
                if self.stats[field] < 0:
                    logger.error(f"Negative value for {field}: {self.stats[field]}")
                    return False
            
            # Vérifications de cohérence
            if self.stats['total_chunks'] > 0 and self.stats['total_documents'] == 0:
                logger.warning("Found chunks but no documents - potential inconsistency")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating stats: {e}")
            return False
    
    def _calculate_stats_hash(self) -> str:
        """
        Calcule un hash des statistiques pour détecter les changements
        
        Returns:
            Hash MD5 des statistiques
        """
        try:
            import hashlib
            import json
            
            # Créer une version sérialisable des stats
            serializable_stats = {}
            for key, value in self.stats.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    serializable_stats[key] = value
                else:
                    serializable_stats[key] = str(value)
            
            # Trier les clés pour un hash cohérent
            stats_json = json.dumps(serializable_stats, sort_keys=True)
            return hashlib.md5(stats_json.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Error calculating stats hash: {e}")
            return str(datetime.now().timestamp())  # Fallback
    
    def _enrich_stats_for_save(self) -> Dict:
        """
        Enrichit les statistiques avec des métadonnées de sauvegarde
        
        Returns:
            Statistiques enrichies
        """
        enriched = self.stats.copy()
        
        # Ajouter des métadonnées de sauvegarde
        enriched.update({
            'save_timestamp': datetime.now().isoformat(),
            'save_count': getattr(self, '_stats_save_count', 0) + 1,
            'system_info': {
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                'os_name': os.name,
                'faiss_dir': settings.FAISS_INDEX_DIR
            },
            'integrity_hash': self._calculate_stats_hash()
        })
        
        # Mettre à jour le compteur
        self._stats_save_count = enriched['save_count']
        
        return enriched
    
    def _create_stats_backup(self, stats_file: str) -> bool:
        """
        Crée un backup du fichier de statistiques
        
        Args:
            stats_file: Chemin vers le fichier de stats
            
        Returns:
            True si le backup a été créé
        """
        try:
            if not os.path.exists(stats_file):
                return False
            
            # Créer le nom du backup avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{stats_file}.backup_{timestamp}"
            
            # Copier le fichier
            import shutil
            shutil.copy2(stats_file, backup_file)
            
            # Nettoyer les anciens backups (garder seulement les 5 plus récents)
            self._cleanup_old_backups(stats_file)
            
            logger.debug(f"Stats backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to create stats backup: {e}")
            return False
    
    def _cleanup_old_backups(self, stats_file: str, keep_count: int = 5) -> None:
        """
        Nettoie les anciens backups en gardant seulement les plus récents
        
        Args:
            stats_file: Fichier de stats principal
            keep_count: Nombre de backups à conserver
        """
        try:
            import glob
            
            backup_pattern = f"{stats_file}.backup_*"
            backup_files = glob.glob(backup_pattern)
            
            if len(backup_files) <= keep_count:
                return
            
            # Trier par date de modification (plus récent en premier)
            backup_files.sort(key=os.path.getmtime, reverse=True)
            
            # Supprimer les anciens
            for old_backup in backup_files[keep_count:]:
                try:
                    os.remove(old_backup)
                    logger.debug(f"Removed old backup: {old_backup}")
                except Exception as e:
                    logger.warning(f"Failed to remove old backup {old_backup}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning up old backups: {e}")
    
    def _verify_stats_file(self, file_path: str) -> bool:
        """
        Vérifie l'intégrité du fichier de statistiques
        
        Args:
            file_path: Chemin vers le fichier à vérifier
            
        Returns:
            True si le fichier est valide
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Vérifier que le fichier peut être lu
            with open(file_path, "rb") as f:
                loaded_stats = pickle.load(f)
            
            # Vérifier que c'est un dictionnaire
            if not isinstance(loaded_stats, dict):
                logger.error("Loaded stats is not a dictionary")
                return False
            
            # Vérifier les champs essentiels
            essential_fields = ['total_documents', 'total_chunks']
            for field in essential_fields:
                if field not in loaded_stats:
                    logger.error(f"Missing essential field in saved stats: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Stats file verification failed: {e}")
            return False


