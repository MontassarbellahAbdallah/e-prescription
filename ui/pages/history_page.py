"""
Page d'historique des analyses et recherches
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from config.logging_config import get_logger
from ui.components.tables import display_interactions_table
from ui.components.charts import display_interaction_charts
from ui.styles import create_metric_card, create_status_message
from utils.helpers import format_timestamp, truncate_text

logger = get_logger(__name__)

class HistoryPage:
    """Page d'historique des analyses et recherches"""
    
    def __init__(self):
        self.analysis_history = self._get_analysis_history()
        self.search_history = self._get_search_history()
    
    def _get_analysis_history(self) -> List[Dict]:
        """Récupère l'historique des analyses"""
        return st.session_state.get('analysis_history', [])
    
    def _get_search_history(self) -> List[Dict]:
        """Récupère l'historique des recherches"""
        return st.session_state.get('search_history', [])
    
    def render(self):
        """Affiche la page d'historique"""
        st.subheader("Historique des activités")
        
        # Statistiques générales
        self._render_overview_stats()
        
        # Onglets pour différents types d'historique
        tab1, tab2 = st.tabs([
            "Analyses d'interactions", 
            "autres"
        ])
        
        with tab1:
            self._render_analysis_history()
        
        # with tab2:
        #     self._render_search_history()
        
        # with tab2:
        #     self._render_management_section()
    
    def _render_overview_stats(self):
        """Affiche les statistiques générales"""
        st.markdown("### Vue d'ensemble")
        
        col1= st.columns(1)[0]
        
        with col1:
            create_metric_card("Analyses effectuées", str(len(self.analysis_history)))
        
    
    def _render_analysis_history(self):
        """Affiche l'historique des analyses"""
        st.markdown("### Historique des analyses d'interactions")
        
        if not self.analysis_history:
            create_status_message(
                "Aucune analyse dans l'historique. "
                "Commencez par analyser des interactions dans l'onglet principal.",
                "info"
            )
            return
        
        # Afficher les analyses (limiter à 10 plus récentes)
        recent_analyses = sorted(
            self.analysis_history, 
            key=lambda x: x.get('timestamp', datetime.min), 
            reverse=True
        )[:10]
        
        for i, analysis in enumerate(recent_analyses, 1):
            self._render_analysis_entry(analysis, i)
    
    def _render_analysis_entry(self, analysis: Dict, index: int):
        """Affiche une entrée d'analyse"""
        timestamp = analysis.get('timestamp', datetime.now())
        question = analysis.get('question', 'Question inconnue')
        drugs = analysis.get('drugs', [])
        stats = analysis.get('stats', {})
        interactions = analysis.get('interactions', [])
        
        time_str = format_timestamp(timestamp)
        header = f"Analyse {index} - {time_str}"
        
        major_count = stats.get('major', 0)
        if major_count > 0:
            header += f" {major_count} Major"
        
        with st.expander(header, expanded=index <= 2):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Médicaments:** {', '.join(drugs) if drugs else 'Aucun'}")
            
            with col2:
                if stats:
                    st.metric("Combinaisons", stats.get('total_combinations', 0))
                    
                    col_major, col_moderate, col_minor = st.columns(3)
                    with col_major:
                        st.metric("Major", stats.get('major', 0))
                    with col_moderate:
                        st.metric("Moderate", stats.get('moderate', 0))
                    with col_minor:
                        st.metric("Minor", stats.get('minor', 0))
            
    
    def _prepare_analysis_export(self, analysis: Dict) -> str:
        """Prépare les données d'une analyse pour l'export CSV"""
        try:
            interactions = analysis.get('interactions', [])
            if not interactions:
                return ""
            
            df = pd.DataFrame(interactions)
            return df.to_csv(index=False, encoding='utf-8')
        except Exception as e:
            logger.error(f"Error preparing analysis export: {e}")
            return ""
    
    def _render_search_history(self):
        """Affiche l'historique des recherches"""
        st.markdown("### Historique des recherches documentaires")
        
        if not self.search_history:
            create_status_message(
                "Aucune recherche dans l'historique. "
                "Effectuez des recherches dans l'onglet de recherche documentaire.",
                "info"
            )
            return
        
        recent_searches = sorted(
            self.search_history, 
            key=lambda x: x.get('timestamp', datetime.min), 
            reverse=True
        )[:20]
        
        for i, search in enumerate(recent_searches, 1):
            self._render_search_entry(search, i)
    
    def _render_search_entry(self, search: Dict, index: int):
        """Affiche une entrée de recherche"""
        timestamp = search.get('timestamp', datetime.now())
        query = search.get('query', 'Requête inconnue')
        results_count = search.get('results_count', 0)
        
        time_str = format_timestamp(timestamp)
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.write(f"**{query}**")
            st.caption(time_str)
        
        with col2:
            st.metric("Résultats", results_count)
        
        with col3:
            if st.button(key=f"repeat_search_{index}", help="Répéter cette recherche"):
                st.session_state['auto_search_query'] = query
                st.session_state.active_page = "search"
                st.rerun()
        
        with col4:
            if st.button(key=f"delete_search_{index}", help="Supprimer cette recherche"):
                st.session_state.search_history.remove(search)
                st.rerun()
    
    
    def _export_full_history(self):
        """Exporte tout l'historique"""
        try:
            full_history = {
                'analyses': self.analysis_history,
                'searches': self.search_history,
                'export_timestamp': datetime.now().isoformat(),
                'total_analyses': len(self.analysis_history),
                'total_searches': len(self.search_history)
            }
            
            import json
            json_data = json.dumps(full_history, indent=2, default=str, ensure_ascii=False)
            
            st.download_button(
                label="Télécharger l'historique complet (JSON)",
                data=json_data,
                file_name=f"historique_complet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("Export prêt au téléchargement")
            
        except Exception as e:
            logger.error(f"Error exporting full history: {e}")
            st.error(f"Erreur lors de l'export: {e}")
    
    def _cleanup_old_history(self):
        """Nettoie l'historique ancien (plus de 30 jours)"""
        cutoff_date = datetime.now() - timedelta(days=30)
        cleaned_count = 0
        
        original_count = len(st.session_state.get('analysis_history', []))
        st.session_state.analysis_history = [
            analysis for analysis in st.session_state.get('analysis_history', [])
            if analysis.get('timestamp', datetime.min) >= cutoff_date
        ]
        cleaned_count += original_count - len(st.session_state.analysis_history)
        
        original_count = len(st.session_state.get('search_history', []))
        st.session_state.search_history = [
            search for search in st.session_state.get('search_history', [])
            if search.get('timestamp', datetime.min) >= cutoff_date
        ]
        cleaned_count += original_count - len(st.session_state.search_history)
        
        if cleaned_count > 0:
            st.success(f"{cleaned_count} entrées anciennes supprimées")
            st.rerun()
        else:
            st.info("Aucune entrée ancienne à supprimer")
    
    def _clear_all_history(self):
        """Vide tout l'historique avec confirmation"""
        st.markdown("#### Confirmation requise")
        st.warning(
            "Cette action supprimera définitivement tout l'historique "
            "(analyses et recherches). Cette action ne peut pas être annulée."
        )
        
        confirm_text = st.text_input(
            "Tapez 'SUPPRIMER' pour confirmer:",
            key="confirm_clear_history"
        )
        
        if confirm_text == "SUPPRIMER":
            if st.button("Confirmer la suppression", type="primary"):
                st.session_state.analysis_history = []
                st.session_state.search_history = []
                
                st.success("Tout l'historique a été supprimé")
                st.rerun()

def render_history_page():
    """Fonction utilitaire pour afficher la page d'historique"""
    page = HistoryPage()
    page.render()

history_page = HistoryPage()
