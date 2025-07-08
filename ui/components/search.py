"""
Composants de recherche avancée pour l'interface utilisateur
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from data.rag_processor import RAGProcessor
from data.validators import DataValidator
from utils.helpers import truncate_text, format_timestamp
from ui.styles import create_status_message, create_content_card

class AdvancedSearch:
    """Interface de recherche avancée dans les documents"""
    
    def __init__(self, rag_processor: RAGProcessor):
        """
        Initialise l'interface de recherche
        
        Args:
            rag_processor: Processeur RAG pour la recherche
        """
        self.rag_processor = rag_processor
        self.search_history = []
    
    def render_search_interface(self):
        """
        Affiche l'interface de recherche complète
        """
        st.subheader("🔍 Recherche avancée dans les documents")
        
        # Contrôles de recherche
        search_query, search_options = self._render_search_controls()
        
        # Exécuter la recherche
        if st.button("🔍 Rechercher", type="primary") and search_query:
            self._execute_search(search_query, search_options)
    
    def _render_search_controls(self) -> Tuple[str, Dict]:
        """
        Affiche les contrôles de recherche
        
        Returns:
            Tuple (requête, options)
        """
        # Zone de saisie principale
        search_query = st.text_area(
            "Requête de recherche:",
            placeholder="Ex: interaction warfarine, cytochrome P450, contre-indications...",
            height=100
        )
        
        # Options avancées
        with st.expander("⚙️ Options avancées"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_results = st.slider("Nombre de résultats", 1, 20, 5)
            
            with col2:
                include_score = st.checkbox("Afficher scores", True)
            
            with col3:
                detailed_view = st.checkbox("Vue détaillée", False)
        
        search_options = {
            'num_results': num_results,
            'include_score': include_score,
            'detailed_view': detailed_view
        }
        
        return search_query, search_options
    
    def _execute_search(self, query: str, options: Dict):
        """
        Exécute la recherche avec les options spécifiées
        """
        is_valid, cleaned_query = DataValidator.validate_search_query(query)
        
        if not is_valid:
            st.error("Requête invalide. Veuillez saisir au moins 3 caractères.")
            return
        
        with st.spinner("Recherche en cours..."):
            try:
                docs, sources_info = self.rag_processor.search_documents_with_sources(
                    cleaned_query, k=options['num_results']
                )
                
                self._display_search_results(docs, sources_info, options)
                
            except Exception as e:
                st.error(f"Erreur lors de la recherche: {str(e)}")
    
    def _display_search_results(self, docs: List, sources_info: List[Dict], options: Dict):
        """
        Affiche les résultats de recherche
        """
        if not docs:
            st.warning("Aucun résultat trouvé.")
            return
        
        st.success(f"✅ {len(docs)} résultat(s) trouvé(s)")
        
        for i, (doc, source) in enumerate(zip(docs, sources_info), 1):
            with st.expander(f"📄 Résultat {i}: {source['document']} (Page {source['page']})"):
                if options['include_score']:
                    st.write(f"**Score:** {source['relevance_score']:.3f}")
                
                content_key = 'full_content' if options['detailed_view'] else 'content_preview'
                content = source.get(content_key, '')
                
                if content:
                    st.markdown(f"> {content}")

class DrugSearch:
    """Interface de recherche spécialisée pour les médicaments"""
    
    def __init__(self, rag_processor: RAGProcessor):
        self.rag_processor = rag_processor
    
    def render_drug_search_interface(self):
        """Affiche l'interface de recherche de médicaments"""
        st.subheader("💊 Recherche par médicament")
        
        drug_name = st.text_input("Nom du médicament:", placeholder="Ex: warfarine, aspirine...")
        
        if st.button("🔍 Rechercher informations", type="primary") and drug_name:
            self._execute_drug_search(drug_name)
    
    def _execute_drug_search(self, drug_name: str):
        """Exécute la recherche pour un médicament spécifique"""
        with st.spinner(f"Recherche d'informations sur {drug_name}..."):
            try:
                query = f"{drug_name} interaction contre-indication effet"
                docs, sources_info = self.rag_processor.search_documents_with_sources(query, k=5)
                
                if docs:
                    st.success(f"✅ Informations trouvées sur {drug_name}")
                    for i, (doc, source) in enumerate(zip(docs, sources_info), 1):
                        with st.expander(f"{source['document']} - Page {source['page']}"):
                            content = source.get('content_preview', '')
                            if content:
                                st.markdown(f"> {content}")
                else:
                    st.warning("Aucune information trouvée.")
                    
            except Exception as e:
                st.error(f"Erreur: {e}")

# Fonctions utilitaires
def render_advanced_search(rag_processor: RAGProcessor):
    """Fonction utilitaire pour afficher la recherche avancée"""
    search_interface = AdvancedSearch(rag_processor)
    search_interface.render_search_interface()

def render_drug_search(rag_processor: RAGProcessor):
    """Fonction utilitaire pour afficher la recherche de médicaments"""
    drug_search = DrugSearch(rag_processor)
    drug_search.render_drug_search_interface()
