"""
Application principale - Analyseur de prescription Médicamenteuses
"""
import streamlit as st
import sys
import os
import glob
import time
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire racine au path Python
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

# Imports de l'architecture modulaire
from config.settings import settings
from config.logging_config import get_logger
from core.cache_manager import get_cache_manager
from core.key_manager import get_key_manager
from core.exceptions import MedInteractionAnalyzerError
from data.rag_processor import RAGProcessor
from data.pdf_processor import PDFProcessor
from ai.llm_analyzer import LLMAnalyzer
from ui.styles import apply_custom_css, create_main_header, setup_page_config, sidebar_info
from ui.pages.analysis_page import render_analysis_page
from ui.pages.search_page import render_search_page
from ui.pages.history_page import render_history_page

logger = get_logger(__name__)

class MedInteractionApp:
    """Application principale d'analyse d'interactions médicamenteuses"""
    
    def __init__(self):
        """Initialise l'application"""
        self.initialized = False
        self.setup_page()
        self.initialize_system()
    
    def setup_page(self):
        """Configure la page Streamlit"""
        setup_page_config()
        apply_custom_css()
        
        # En-tête principal
        create_main_header(
            "Analyseur des préscriptions Médicamenteuses",
            "Analyse intelligente des préscriptions médicamenteuses basée sur l'IA"
        )
    
    def initialize_system(self):
        """Initialise tous les composants du système"""
        if 'system_initialized' in st.session_state and st.session_state.system_initialized:
            self.initialized = True
            return
        
        try:
            with st.spinner("Initialisation du système..."):
                # Validation de la configuration
                if not settings.validate_configuration():
                    st.error("Configuration invalide. Vérifiez vos variables d'environnement.")
                    st.stop()
                
                # Initialisation des composants core
                self._initialize_core_components()
                
                # Initialisation des composants IA
                self._initialize_ai_components()
                
                # Initialisation de la base documentaire
                self._initialize_document_base()
                
                # Marquer comme initialisé
                st.session_state.system_initialized = True
                self.initialized = True
                
                logger.info("Système initialisé avec succès")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            st.error(f"Erreur d'initialisation: {e}")
            st.stop()
    
    def _initialize_core_components(self):
        """Initialise les composants core"""
        # Cache manager
        if 'cache_manager' not in st.session_state:
            st.session_state.cache_manager = get_cache_manager()
            logger.info("Cache manager initialisé")
        
        # Key manager
        if 'key_manager' not in st.session_state:
            st.session_state.key_manager = get_key_manager()
            logger.info("Key manager initialisé")
    
    def _initialize_ai_components(self):
        """Initialise les composants IA"""
        # LLM Analyzer
        if 'llm_analyzer' not in st.session_state:
            st.session_state.llm_analyzer = LLMAnalyzer(use_cache=True)
            logger.info("LLM Analyzer initialisé")
    
    def _initialize_document_base(self):
        """Initialise la base documentaire avec rechargement automatique"""
        # RAG Processor
        if 'rag_processor' not in st.session_state:
            st.session_state.rag_processor = RAGProcessor(cache_enabled=True)
            
            # Charger ou construire l'index
            self._setup_vector_store()
            
            logger.info("RAG Processor initialisé")
        else:
            # Si le RAG processor existe mais que le vector store n'est pas chargé
            if not st.session_state.rag_processor.vector_store:
                logger.info("RAG Processor existe mais vector store non chargé - rechargement...")
                self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Configure le vector store avec les PDF disponibles (amélioré)"""
        try:
            # Créer le dossier guidelines s'il n'existe pas
            os.makedirs(settings.GUIDELINES_DIR, exist_ok=True)
            
            # Chercher les fichiers PDF
            pdf_files = glob.glob(os.path.join(settings.GUIDELINES_DIR, "*.pdf"))
            
            if not pdf_files:
                st.warning(
                    f"Aucun fichier PDF trouvé dans {settings.GUIDELINES_DIR}. "
                    "Placez vos documents de référence dans ce dossier pour activer la recherche documentaire."
                )
                return
            
            logger.info(f"Found {len(pdf_files)} PDF files for vector store setup")
            
            # Vérifier si l'index existe déjà
            if self._vector_store_exists():
                logger.info("Vector store files exist - attempting to load...")
                # Charger l'index existant
                success = st.session_state.rag_processor.load_vector_store()
                if success:
                    stats = st.session_state.rag_processor.get_stats()
                    logger.info(f"Vector store loaded successfully: {stats['total_documents']} docs, {stats['total_chunks']} chunks")
                    
                    # Afficher le message de succès seulement lors de la première initialisation
                    if 'vector_store_message_shown' not in st.session_state:
                        st.success(
                            f"Base documentaire chargée: {stats['total_documents']} documents, "
                            f"{stats['total_chunks']} chunks"
                        )
                        st.session_state.vector_store_message_shown = True
                else:
                    logger.warning("Failed to load existing vector store - rebuilding...")
                    st.warning("Erreur lors du chargement de l'index, reconstruction nécessaire")
                    self._build_vector_store(pdf_files)
            else:
                # Construire un nouvel index
                logger.info("No existing vector store found - building new one...")
                st.info(f"Première indexation de {len(pdf_files)} document(s) PDF...")
                self._build_vector_store(pdf_files)
                
        except Exception as e:
            logger.error(f"Erreur setup vector store: {e}", exc_info=True)
            st.error(f"Erreur lors de la configuration de la base documentaire: {e}")
    
    def _vector_store_exists(self) -> bool:
        """Vérifie si le vector store existe"""
        index_files = [
            os.path.join(settings.FAISS_INDEX_DIR, "index.faiss"),
            os.path.join(settings.FAISS_INDEX_DIR, "index.pkl")
        ]
        return all(os.path.exists(f) for f in index_files)
    
    def _build_vector_store(self, pdf_files: list):
        """Construit le vector store à partir des PDF"""
        try:
            with st.spinner("Indexation des documents en cours..."):
                success = st.session_state.rag_processor.create_vector_store_from_pdfs(
                    pdf_files, force_rebuild=True
                )
                
                if success:
                    stats = st.session_state.rag_processor.get_stats()
                    st.success(
                        f"Indexation terminée: {stats['total_documents']} documents, "
                        f"{stats['total_chunks']} chunks créés"
                    )
                else:
                    st.error("Échec de l'indexation")
                    
        except Exception as e:
            logger.error(f"Erreur construction vector store: {e}")
            st.error(f"Erreur lors de l'indexation: {e}")
    
    def render_sidebar(self):
        """Affiche la sidebar avec les informations système"""
        with st.sidebar:
            # Bouton Diagnostic RAG
            # if st.button("Diagnostic RAG", use_container_width=True, help="Diagnostiquer le système RAG pour identifier les problèmes"):
            #     self._handle_rag_diagnosis()
            
            #st.markdown("---")
            
            # Navigation
            st.subheader("Navigation")
            
            # Initialiser la page active
            if 'active_page' not in st.session_state:
                st.session_state.active_page = "analysis"
            
            # Boutons de navigation
            if st.button("Analyseur des préscriptions", use_container_width=True):
                st.session_state.active_page = "analysis"
                st.rerun()
            
            # if st.button("Recherche documentaire", use_container_width=True):
            #     st.session_state.active_page = "search"
            #     st.rerun()
            
            if st.button("Historique", use_container_width=True):
                st.session_state.active_page = "history"
                st.rerun()
            
            # Bouton Vider cache
            # if st.button("Vider cache", use_container_width=True, help="Vider le cache d'extraction pour forcer une nouvelle analyse"):
            #     self._handle_clear_cache()
            
            st.markdown("---")
            
            # Statistiques système
            # self._render_system_stats()
            
            # Aide et documentation
            self._render_help_section()
    
    # def _render_system_stats(self):
    #     """Affiche les statistiques système dans la sidebar"""
    #     status_html = self._get_system_status_html()
    #     sidebar_info("Statut Système", status_html)
    
    def _get_system_status_html(self) -> str:
        """Génère le HTML pour le statut système"""
        status_items = []
        
        # Statut des composants
        if hasattr(st.session_state, 'llm_analyzer'):
            status_items.append("LLM Analyzer: Actif")
        else:
            status_items.append("LLM Analyzer: Inactif")
        
        if hasattr(st.session_state, 'rag_processor') and st.session_state.rag_processor.vector_store:
            stats = st.session_state.rag_processor.get_stats()
            status_items.append(f"Base documentaire: {stats['total_documents']} docs")
        else:
            status_items.append("Base documentaire: Non disponible")
        
        # Statistiques d'usage
        analysis_count = len(st.session_state.get('analysis_history', []))
        search_count = len(st.session_state.get('search_history', []))
        
        status_items.extend([
            f"Analyses: {analysis_count}",
            f"Recherches: {search_count}"
        ])
        
        return "<br>".join(status_items)
    
    def _handle_clear_cache(self):
        """Gère la demande de vidage du cache"""
        try:
            if 'llm_analyzer' in st.session_state and st.session_state.llm_analyzer:
                cleared = st.session_state.llm_analyzer.clear_drug_extraction_cache()
                if cleared:
                    st.success(f"Cache vidé ({cleared} entrées)")
                else:
                    st.info("Cache déjà vide")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("LLM Analyzer non disponible")
        except Exception as e:
            logger.error(f"Erreur lors du vidage du cache: {e}")
            st.error(f"Erreur lors du vidage du cache: {e}")
    
    def _handle_rag_diagnosis(self):
        """Gère le diagnostic du système RAG"""
        try:
            if 'rag_processor' in st.session_state and st.session_state.rag_processor:
                with st.spinner("Diagnostic du système RAG en cours..."):
                    diagnosis = st.session_state.rag_processor.diagnose_rag_system()
                
                # Afficher les résultats
                st.subheader("Rapport de diagnostic RAG")
                
                # Score de santé
                health_score = diagnosis['health_score']
                if health_score >= 80:
                    st.success(f"Système RAG en bonne santé: {health_score}%")
                elif health_score >= 60:
                    st.warning(f"Système RAG avec problèmes mineurs: {health_score}%")
                else:
                    st.error(f"Système RAG avec problèmes majeurs: {health_score}%")
                
                # Statut des composants
                st.markdown("### Statut des composants")
                col1, col2 = st.columns(2)
                
                with col1:
                    vector_status = diagnosis['vector_store_status']
                    if vector_status == 'initialized':
                        st.success("Vector Store: Initialisé")
                    else:
                        st.error("Vector Store: Non initialisé")
                    
                    # Métadonnées
                    meta_status = diagnosis['metadata_status']
                    st.info(f"Métadonnées: {meta_status['total_chunks']} chunks")
                    st.info(f"Documents: {meta_status['unique_documents']} uniques")
                
                with col2:
                    # Fichiers d'index
                    st.markdown("Fichiers d'index:")
                    for filename, info in diagnosis['index_files'].items():
                        status = "✅" if info['exists'] else "❌"
                        size_mb = info['size'] / (1024*1024) if info['size'] > 0 else 0
                        st.markdown(f"{status} {filename} ({size_mb:.1f} MB)")
                
                # Problèmes identifiés
                if diagnosis['issues_found']:
                    st.markdown("### Problèmes identifiés")
                    for issue in diagnosis['issues_found']:
                        st.error(f"• {issue}")
                
                # Recommandations
                if diagnosis['recommendations']:
                    st.markdown("### Recommandations")
                    for rec in diagnosis['recommendations']:
                        st.info(f"• {rec}")
                
                # Test de recherche
                if 'search_test' in diagnosis:
                    search_test = diagnosis['search_test']
                    st.markdown("### Test de recherche")
                    if search_test['success']:
                        st.success(f"Test réussi: {search_test['results_count']} résultats")
                        if 'has_doc_meta_markers' in search_test:
                            if search_test['has_doc_meta_markers']:
                                st.success("Marqueurs DOC_META détectés")
                            else:
                                st.error("Marqueurs DOC_META manquants")
                    else:
                        st.error(f"Test échoué: {search_test.get('error', 'Erreur inconnue')}")
                
                # Bouton pour forcer la reconstruction
                if health_score < 60:
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Forcer la reconstruction du Vector Store", type="secondary"):
                            self._force_rebuild_vector_store()
                    with col2:
                        if st.button("Recalculer les statistiques", type="secondary"):
                            self._recalculate_rag_stats()
                
            else:
                st.error("RAG Processor non disponible")
                
        except Exception as e:
            logger.error(f"Erreur lors du diagnostic RAG: {e}")
            st.error(f"Erreur lors du diagnostic: {e}")
    
    def _force_rebuild_vector_store(self):
        """Force la reconstruction du vector store avec nouvelle méthode séquentielle"""
        try:
            # Chercher les fichiers PDF
            pdf_files = glob.glob(os.path.join(settings.GUIDELINES_DIR, "*.pdf"))
            
            if not pdf_files:
                st.error("Aucun fichier PDF trouvé pour la reconstruction")
                return
            
            st.info(f"Début de la reconstruction avec {len(pdf_files)} PDFs...")
            
            # Callback de progression
            progress_container = st.empty()
            
            def progress_callback(current, total, description):
                percentage = (current / total) * 100
                progress_text = f"PDF {current}/{total} ({percentage:.1f}%): {os.path.basename(description)}"
                progress_container.info(progress_text)
            
            with st.spinner("Reconstruction séquentielle du Vector Store en cours..."):
                # Utiliser la nouvelle méthode séquentielle
                success = st.session_state.rag_processor.create_vector_store_sequential(
                    pdf_files, 
                    force_rebuild=True,
                    progress_callback=progress_callback
                )
                
                progress_container.empty()  # Nettoyer l'affichage de progression
                
                if success:
                    stats = st.session_state.rag_processor.get_stats()
                    st.success(
                        f"Reconstruction réussie avec nouvelle méthode !\n"
                        f"Documents: {stats['total_documents']}\n"
                        f"Chunks: {stats['total_chunks']}\n"
                        f"Taille: {stats['index_size_mb']} MB"
                    )
                    
                    # Test de recherche pour vérifier les métadonnées enrichies
                    try:
                        docs, sources = st.session_state.rag_processor.search_with_detailed_sources(
                            "diabetes medication", k=3
                        )
                        
                        if sources:
                            st.info(
                                f"🔍 Test de recherche: {len(sources)} sources trouvées\n"
                                f"Exemple: {sources[0].get('academic_citation', 'Citation non disponible')}"
                            )
                    except Exception as e:
                        st.warning(f"Test de recherche échoué: {e}")
                    
                    time.sleep(3)
                    st.rerun()
                else:
                    st.error("Échec de la reconstruction")
                    
        except Exception as e:
            logger.error(f"Erreur lors de la reconstruction séquentielle: {e}")
            st.error(f"Erreur lors de la reconstruction: {e}")
    
    def _render_help_section(self):
        """Affiche la section d'aide"""
        help_content = """
        Guide d'utilisation:
        
        1. Analyse: Saisissez une prescription médicamenteuse pour analyse
        3. Historique: Consultez vos analyses précédentes
        
        Niveaux d'interaction:
        - Major: Éviter l'association
        - Moderate: Surveillance requise
        - Minor: Généralement acceptable
        - aucun: Pas d'interaction
        
        Support: Cette application est un outil d'aide à la décision.
        Consultez toujours un professionnel de santé.
        """
        sidebar_info(help_content)
    
    def render_main_content(self):
        """Affiche le contenu principal selon la page active"""
        page = st.session_state.get('active_page', 'analysis')
        
        try:
            if page == "analysis":
                render_analysis_page()
            elif page == "search":
                render_search_page()
            elif page == "history":
                render_history_page()
            else:
                st.error(f"Page inconnue: {page}")
                st.session_state.active_page = "analysis"
                st.rerun()
                
        except Exception as e:
            logger.error(f"Erreur rendu page {page}: {e}")
            st.error(f"Erreur lors de l'affichage de la page: {e}")
    
    def render_footer(self):
        """Affiche le footer de l'application"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(
                """
                <div style="text-align: center; color: #666; padding: 1rem;">
                    <p><em>Outil d'aide à la décision - Consultez toujours un professionnel de santé</em></p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def handle_special_cases(self):
        """Gère les cas spéciaux et les redirections"""
        # Gestion de la question pré-remplie pour réanalyse
        if 'reanalyze_question' in st.session_state:
            st.session_state.active_page = "analysis"
            # La question sera utilisée dans analysis_page.py
        
        # Gestion de la requête de recherche pré-remplie
        if 'auto_search_query' in st.session_state:
            st.session_state.active_page = "search"
            # La requête sera utilisée dans search_page.py
    
    def run(self):
        """Lance l'application principale"""
        if not self.initialized:
            st.error("Système non initialisé")
            return
        
        try:
            # Gestion des cas spéciaux
            self.handle_special_cases()
            
            # Affichage de l'interface
            self.render_sidebar()
            self.render_main_content()
            self.render_footer()
            
        except Exception as e:
            logger.error(f"Erreur application principale: {e}")
            st.error(f"Erreur inattendue: {e}")
            
            # Bouton de réinitialisation en cas d'erreur
            if st.button("Réinitialiser l'application"):
                # Nettoyer le cache de session
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

def main():
    """Fonction principale"""
    try:
        # Créer et lancer l'application
        app = MedInteractionApp()
        app.run()
        
    except Exception as e:
        # Gestion d'erreur de dernier recours
        st.error(f"Erreur critique: {e}")
        st.markdown(
            """
            ### Aide au dépannage
            
            1. Vérifiez vos variables d'environnement (notamment `GOOGLE_API_KEY`)
            2. Assurez-vous que les dépendances sont installées (`pip install -r requirements.txt`)
            3. Vérifiez les permissions du dossier `Data/guidelines/`
            4. Redémarrez l'application si nécessaire
            
            Si le problème persiste, consultez les logs dans `app.log`
            """
        )
        
        # Bouton de diagnostic
        if st.button("🔧 Diagnostic système"):
            st.subheader("📋 Informations de diagnostic")
            
            # Vérifications de base
            diagnostics = []
            
            # Variables d'environnement
            api_key = os.getenv("GOOGLE_API_KEY")
            diagnostics.append(f"GOOGLE_API_KEY: {'Définie' if api_key else 'Manquante'}")
            
            # Dossiers
            data_dir = Path(settings.DATA_DIR)
            guidelines_dir = Path(settings.GUIDELINES_DIR)
            diagnostics.append(f"Dossier Data: {'Existe' if data_dir.exists() else 'Manquant'}")
            diagnostics.append(f"Dossier Guidelines: {'Existe' if guidelines_dir.exists() else 'Manquant'}")
            
            # Fichiers PDF
            if guidelines_dir.exists():
                pdf_count = len(list(guidelines_dir.glob("*.pdf")))
                diagnostics.append(f"Fichiers PDF: {pdf_count} trouvé(s)")
            else:
                diagnostics.append("Fichiers PDF: Dossier guidelines inexistant")
            
            # Permissions d'écriture
            try:
                test_file = Path("test_write.tmp")
                test_file.write_text("test")
                test_file.unlink()
                diagnostics.append("Permissions écriture: OK")
            except Exception:
                diagnostics.append("Permissions écriture: Problème")
            
            # Affichage
            for diagnostic in diagnostics:
                st.markdown(f"- {diagnostic}")

if __name__ == "__main__":
    main()
