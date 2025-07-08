"""
Page de recherche documentaire avancée
"""
import streamlit as st
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from config.logging_config import get_logger
from data.rag_processor import RAGProcessor
from data.validators import DataValidator
from ai.llm_analyzer import LLMAnalyzer
from ui.styles import (
    create_status_message, create_content_card, create_metric_card
)
from utils.helpers import truncate_text, format_timestamp

logger = get_logger(__name__)

class SearchPage:
    """Page de recherche documentaire avancée"""
    
    def __init__(self):
        self.rag_processor = None
        self.llm_analyzer = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialise les composants nécessaires"""
        try:
            if 'rag_processor' in st.session_state:
                self.rag_processor = st.session_state.rag_processor
            else:
                self.rag_processor = RAGProcessor(cache_enabled=True)
                self.rag_processor.load_vector_store()
                st.session_state.rag_processor = self.rag_processor
            
            if 'llm_analyzer' in st.session_state:
                self.llm_analyzer = st.session_state.llm_analyzer
            else:
                self.llm_analyzer = LLMAnalyzer(use_cache=True)
                st.session_state.llm_analyzer = self.llm_analyzer
                
        except Exception as e:
            logger.error(f"Error initializing search page components: {e}")
            st.error(f"Erreur d'initialisation: {e}")
    
    def render(self):
        """Affiche la page de recherche"""
        st.subheader("🔍 Recherche documentaire avancée")
        
        if not self.rag_processor or not self.rag_processor.vector_store:
            self._render_no_vector_store_message()
            return
        
        self._render_vector_store_stats()
        
        tab1, tab2, tab3 = st.tabs(["🔍 Recherche générale", "💊 Recherche par médicament", "📊 Exploration"])
        
        with tab1:
            self._render_general_search()
        
        with tab2:
            self._render_drug_search()
        
        with tab3:
            self._render_exploration_section()
    
    def _render_no_vector_store_message(self):
        """Affiche un message quand le vector store n'est pas disponible"""
        create_status_message(
            "Aucune base documentaire disponible. "
            "Veuillez d'abord indexer des documents.",
            "warning"
        )
        
        st.markdown("### 📚 Pour commencer")
        st.markdown("""
        1. Placez vos fichiers PDF dans le dossier `Data/guidelines/`
        2. Relancez l'application pour indexer automatiquement les documents
        3. Revenez sur cette page pour effectuer des recherches
        """)
    
    def _render_vector_store_stats(self):
        """Affiche les statistiques du vector store"""
        stats = self.rag_processor.get_stats()
        
        st.markdown("### 📊 Statistiques de la base documentaire")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Documents indexés", str(stats.get('total_documents', 0)))
        
        with col2:
            create_metric_card("Chunks de texte", str(stats.get('total_chunks', 0)))
        
        with col3:
            create_metric_card("Taille index", f"{stats.get('index_size_mb', 0)} MB")
        
        with col4:
            create_metric_card("Recherches effectuées", str(stats.get('search_count', 0)))
    
    def _render_general_search(self):
        """Interface de recherche générale"""
        st.markdown("### 🔍 Recherche dans tous les documents")
        
        search_query = st.text_area(
            "Requête de recherche:",
            placeholder="Ex: interaction warfarine, cytochrome P450, contre-indications...",
            height=100
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            num_results = st.slider("Nombre de résultats", 1, 20, 5)
        
        with col2:
            include_score = st.checkbox("Afficher scores", True)
        
        with col3:
            detailed_view = st.checkbox("Vue détaillée", False)
        
        with col4:
            min_score = st.slider("Score minimum", 0.0, 2.0, 0.0, 0.1)
        
        if st.button("🔍 Rechercher", type="primary") and search_query.strip():
            self._execute_general_search(search_query, num_results, include_score, detailed_view, min_score)
    
    def _execute_general_search(self, query: str, num_results: int, include_score: bool, detailed_view: bool, min_score: float):
        """Exécute la recherche générale"""
        is_valid, cleaned_query = DataValidator.validate_search_query(query)
        if not is_valid:
            st.error("❌ Requête invalide. Veuillez saisir au moins 3 caractères.")
            return
        
        with st.spinner("🔍 Recherche en cours..."):
            try:
                docs, sources_info = self.rag_processor.search_documents_with_sources(
                    cleaned_query, k=num_results, score_threshold=min_score if min_score > 0 else None
                )
                
                if docs:
                    st.success(f"✅ {len(docs)} résultat(s) trouvé(s)")
                    self._save_search_to_history(cleaned_query, len(docs))
                    self._display_search_results(docs, sources_info, include_score, detailed_view)
                else:
                    st.warning("🔍 Aucun résultat trouvé pour cette requête")
                    
            except Exception as e:
                logger.error(f"Search error: {e}")
                st.error(f"❌ Erreur lors de la recherche: {e}")
    
    def _display_search_results(self, docs: List, sources_info: List[Dict], include_score: bool, detailed_view: bool):
        """Affiche les résultats de recherche"""
        st.markdown("---")
        st.markdown("### 📄 Résultats de recherche")
        
        for i, (doc, source) in enumerate(zip(docs, sources_info), 1):
            header = f"📄 Résultat {i}: {source['document']} (Page {source['page']})"
            
            if include_score:
                score = source['relevance_score']
                score_color = "🟢" if score < 0.5 else "🟡" if score < 1.0 else "🔴"
                header += f" - Score: {score_color} {score:.3f}"
            
            with st.expander(header, expanded=i <= 3):
                content_key = 'full_content' if detailed_view else 'content_preview'
                content = source.get(content_key, '')
                
                if content:
                    create_content_card("Contenu", content)
    
    def _render_drug_search(self):
        """Interface de recherche par médicament"""
        st.markdown("### 💊 Recherche spécialisée par médicament")
        
        drug_name = st.text_input(
            "Nom du médicament:",
            placeholder="Ex: warfarine, aspirine, oméprazole..."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_type = st.selectbox(
                "Type de recherche:",
                [
                    "Informations générales",
                    "Interactions médicamenteuses", 
                    "Contre-indications",
                    "Effets secondaires",
                    "Posologie",
                    "Mécanisme d'action"
                ]
            )
        
        with col2:
            num_results = st.slider("Nombre de résultats", 1, 15, 5, key="drug_search_results")
        
        if st.button("🔍 Rechercher médicament", type="primary") and drug_name.strip():
            self._execute_drug_search(drug_name, search_type, num_results)
    
    def _execute_drug_search(self, drug_name: str, search_type: str, num_results: int):
        """Exécute la recherche par médicament"""
        search_queries = {
            "Informations générales": f"{drug_name}",
            "Interactions médicamenteuses": f"{drug_name} interaction médicamenteuse",
            "Contre-indications": f"{drug_name} contre-indication",
            "Effets secondaires": f"{drug_name} effet secondaire indésirable",
            "Posologie": f"{drug_name} posologie dose",
            "Mécanisme d'action": f"{drug_name} mécanisme action pharmacologie"
        }
        
        query = search_queries[search_type]
        
        with st.spinner(f"🔍 Recherche d'informations sur {drug_name}..."):
            try:
                docs, sources_info = self.rag_processor.search_documents_with_sources(
                    query, k=num_results
                )
                
                if docs:
                    st.success(f"✅ {len(docs)} information(s) trouvée(s) sur {drug_name}")
                    self._display_drug_search_results(drug_name, search_type, docs, sources_info)
                else:
                    st.warning(f"🔍 Aucune information trouvée sur {drug_name}")
                    
            except Exception as e:
                logger.error(f"Drug search error: {e}")
                st.error(f"❌ Erreur lors de la recherche: {e}")
    
    def _display_drug_search_results(self, drug_name: str, search_type: str, docs: List, sources_info: List[Dict]):
        """Affiche les résultats de recherche spécialisée pour un médicament"""
        st.markdown("---")
        st.markdown(f"### 💊 {search_type} - {drug_name}")
        
        for i, (doc, source) in enumerate(zip(docs, sources_info), 1):
            with st.expander(f"📄 Source {i}: {source['document']} (Page {source['page']})", expanded=i == 1):
                content = source.get('content_preview', '')
                if content:
                    st.markdown(f"> {content}")
    
    def _render_exploration_section(self):
        """Section d'exploration de la base documentaire"""
        st.markdown("### 📊 Exploration de la base documentaire")
        
        # Statistiques avancées
        documents = self.rag_processor.get_document_list()
        
        if documents:
            import pandas as pd
            df = pd.DataFrame(documents)
            
            st.dataframe(
                df,
                column_config={
                    "name": st.column_config.TextColumn("Document", width="large"),
                    "pages": st.column_config.NumberColumn("Pages", width="small"),
                    "chunks": st.column_config.NumberColumn("Chunks", width="small"),
                    "total_chars": st.column_config.NumberColumn("Caractères", width="medium")
                },
                use_container_width=True
            )
        else:
            st.info("Aucun document disponible")
        
        # Recherches récentes
        search_history = st.session_state.get('search_history', [])
        
        if search_history:
            st.markdown("#### 🔥 Recherches récentes")
            for i, search in enumerate(search_history[-5:], 1):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{search['query']}**")
                
                with col2:
                    st.write(f"{search['results_count']} résultats")
                
                with col3:
                    if st.button("🔄", key=f"repeat_search_{i}", help="Répéter cette recherche"):
                        self._execute_general_search(search['query'], 5, True, False, 0.0)
    
    def _save_search_to_history(self, query: str, results_count: int):
        """Sauvegarde une recherche dans l'historique"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        search_entry = {
            'query': query,
            'results_count': results_count,
            'timestamp': datetime.now(),
            'page': 'search'
        }
        
        st.session_state.search_history.append(search_entry)
        
        if len(st.session_state.search_history) > 50:
            st.session_state.search_history = st.session_state.search_history[-50:]

def render_search_page():
    """Fonction utilitaire pour afficher la page de recherche"""
    page = SearchPage()
    page.render()

search_page = SearchPage()
