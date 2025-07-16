"""
Composants de tableaux pour l'affichage des interactions médicamenteuses
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Any
from utils.constants import LEVEL_COLORS
from utils.helpers import truncate_text, format_timestamp

class InteractionTable:
    """Gestionnaire de tableaux d'interactions simplifié"""
    
    def __init__(self, interactions: List[Dict]):
        """
        Initialise le gestionnaire de tableau
        
        Args:
            interactions: Liste des interactions à afficher
        """
        self.interactions = interactions
        self.df = pd.DataFrame(interactions) if interactions else pd.DataFrame()
    
    def display_interactive_table(self, enable_filters: bool = True):
        """
        Affiche un tableau simple sans filtres
        
        Args:
            enable_filters: Paramètre conservé pour compatibilité (non utilisé)
        """
        if self.df.empty:
            st.warning("Aucune interaction à afficher")
            return
        
        # Affichage du tableau stylé directement
        self._render_styled_table(self.df)
    
    def _render_styled_table(self, df: pd.DataFrame):
        """Affiche le tableau avec style personnalisé et responsive - Thème sombre"""
        st.subheader("Interactions détectées")
        
        # Préparer les données pour l'affichage
        display_df = df.copy()
        
        # Tronquer les explications longues
        # display_df['Explanation'] = display_df['Explanation'].apply(
        #     lambda x: truncate_text(str(x), max_length=500)
        # )
        display_df['Explanation'] = display_df['Explanation'].astype(str)

        
        # Fonction de style pour les niveaux (adaptée au thème sombre)
        def highlight_level(val):
            if val in LEVEL_COLORS:
                color = LEVEL_COLORS[val]
                # Styles adaptés au thème sombre
                return f'background-color: {color}; color: white; font-weight: bold; text-align: center; border-radius: 4px; padding: 6px 8px; display: inline-block; margin: 2px;'
            return ''
        
        # Appliquer le style
        styled_df = display_df.style.applymap(
            highlight_level, 
            subset=['Level']
        )
        
        # Calcul du nombre d'éléments pour la configuration
        num_rows = len(display_df)
        
        # Configuration des colonnes responsive
        # Calculer les largeurs optimales selon le contenu

        column_config = {
            "Drug1": "Médicament 1",      # Format simple = auto-sizing
            "Drug2": "Médicament 2",      # Format simple = auto-sizing  
            "Level": "Niveau",            # Format simple = auto-sizing
            "Explanation": st.column_config.TextColumn(
                "Explication",
                width="large",
                help="Explication détaillée de l'interaction"
            )  # Format simple = auto-sizing
        }
        
        # Message informatif sur le nombre d'éléments
        if num_rows > 10:
            st.info(f"{num_rows} interactions détectées - Tableau complet affiché")
        elif num_rows > 0:
            st.info(f"{num_rows} interaction(s) détectée(s)")
        
        # Afficher le tableau avec configuration responsive et thème sombre
        st.dataframe(
            styled_df,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
             
        )
        
        

class HistoryTable:
    """Gestionnaire de tableaux pour l'historique des analyses"""
    
    def __init__(self, history: List[Dict]):
        """
        Initialise le gestionnaire d'historique
        
        Args:
            history: Liste des analyses dans l'historique
        """
        self.history = history
        self.df = pd.DataFrame(history) if history else pd.DataFrame()
    
    def display_history_table(self, max_entries: int = 10):
        """
        Affiche le tableau d'historique
        
        Args:
            max_entries: Nombre maximum d'entrées à afficher
        """
        if self.df.empty:
            st.info("📝 Aucune analyse dans l'historique")
            return
        
        st.subheader("📊 Historique des analyses")
        
        # Limiter le nombre d'entrées et trier par date
        history_to_show = self.history[-max_entries:]  # Prendre les plus récentes
        history_to_show.reverse()  # Plus récentes en premier
        
        for i, analysis in enumerate(history_to_show, 1):
            timestamp = analysis.get('timestamp')
            question = analysis.get('question', 'Question inconnue')
            drugs = analysis.get('drugs', [])
            stats = analysis.get('stats', {})
            
            # Formater le timestamp
            time_str = format_timestamp(timestamp) if timestamp else 'Date inconnue'
            
            with st.expander(f"📋 Analyse {i} - {time_str}"):
                # Informations de base
                st.write(f"**Question:** {question}")
                st.write(f"**Médicaments:** {', '.join(drugs) if drugs else 'Aucun'}")
                
                # Statistiques si disponibles
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Combinaisons", stats.get('total_combinations', 0))
                    
                    with col2:
                        st.metric("Major", stats.get('major', 0))
                    
                    with col3:
                        st.metric("Moderate", stats.get('moderate', 0))
                    
                    with col4:
                        st.metric("Minor", stats.get('minor', 0))
                
                # Boutons d'action
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(f"🔄 Réanalyser", key=f"reanalyze_{i}"):
                        st.session_state['reanalyze_question'] = question
                        st.rerun()
                
                with col2:
                    if 'interactions' in analysis:
                        # Créer les données d'export
                        export_data = self._prepare_export_data(analysis)
                        if export_data:
                            st.download_button(
                                label="📄 Exporter CSV",
                                data=export_data,
                                file_name=f"analyse_{time_str.replace('/', '_').replace(' ', '_').replace(':', '_')}.csv",
                                mime="text/csv",
                                key=f"export_{i}"
                            )
    
    def _prepare_export_data(self, analysis: Dict) -> str:
        """
        Prépare les données d'une analyse pour l'export CSV
        
        Args:
            analysis: Données de l'analyse
            
        Returns:
            Données CSV formatées
        """
        try:
            interactions = analysis.get('interactions', [])
            if not interactions:
                return ""
            
            df = pd.DataFrame(interactions)
            return df.to_csv(index=False, encoding='utf-8')
        except Exception:
            return ""

# Fonctions utilitaires pour les tableaux
def display_interactions_table(interactions: List[Dict], enable_filters: bool = True):
    """
    Fonction utilitaire pour afficher un tableau d'interactions
    
    Args:
        interactions: Liste des interactions
        enable_filters: Activer les filtres (paramètre conservé pour compatibilité)
    """
    table = InteractionTable(interactions)
    table.display_interactive_table(enable_filters)

def display_history_table(history: List[Dict], max_entries: int = 10):
    """
    Fonction utilitaire pour afficher l'historique
    
    Args:
        history: Historique des analyses
        max_entries: Nombre maximum d'entrées
    """
    table = HistoryTable(history)
    table.display_history_table(max_entries)

def create_comparison_table(data: Dict[str, List], title: str = "Comparaison"):
    """
    Crée un tableau de comparaison
    
    Args:
        data: Données à comparer
        title: Titre du tableau
    """
    st.subheader(title)
    
    df = pd.DataFrame(data)
    
    # Style personnalisé pour la comparaison
    def highlight_comparison(val):
        if isinstance(val, str):
            if '✅' in val:  # Checkmark
                return 'background-color: #d4edda; color: #155724;'
            elif '❌' in val:  # Cross
                return 'background-color: #f8d7da; color: #721c24;'
            elif '⚠️' in val:  # Warning
                return 'background-color: #fff3cd; color: #856404;'
        return ''
    
    styled_df = df.style.applymap(highlight_comparison)
    
    st.dataframe(
        styled_df,
        use_container_width=True
    )
 
