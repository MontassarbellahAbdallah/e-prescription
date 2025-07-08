"""
Composants d'export pour les données d'interactions
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from io import BytesIO

from config.logging_config import get_logger
from ui.styles import create_status_message

logger = get_logger(__name__)

class DataExporter:
    """Gestionnaire d'export de données"""
    
    def __init__(self, interactions: List[Dict], stats: Dict[str, Any]):
        """
        Initialise l'exporteur de données
        
        Args:
            interactions: Liste des interactions à exporter
            stats: Statistiques associées
        """
        self.interactions = interactions
        self.stats = stats
        self.df = pd.DataFrame(interactions) if interactions else pd.DataFrame()
    
    def create_csv_export(self, include_metadata: bool = True) -> str:
        """
        Crée un export CSV des données
        
        Args:
            include_metadata: Inclure les métadonnées dans l'export
            
        Returns:
            Données CSV formatées
        """
        try:
            if self.df.empty:
                return ""
            
            export_df = self.df.copy()
            
            if include_metadata:
                # Ajouter des métadonnées en en-tête
                metadata_rows = [
                    {"Drug1": "=== MÉTADONNÉES ===", "Drug2": "", "Level": "", "Explanation": ""},
                    {"Drug1": "Date d'export", "Drug2": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Level": "", "Explanation": ""},
                    {"Drug1": "Total interactions", "Drug2": str(len(self.interactions)), "Level": "", "Explanation": ""},
                    {"Drug1": "Interactions Major", "Drug2": str(self.stats.get('major', 0)), "Level": "", "Explanation": ""},
                    {"Drug1": "Interactions Moderate", "Drug2": str(self.stats.get('moderate', 0)), "Level": "", "Explanation": ""},
                    {"Drug1": "Interactions Minor", "Drug2": str(self.stats.get('minor', 0)), "Level": "", "Explanation": ""},
                    {"Drug1": "=== DONNÉES ===", "Drug2": "", "Level": "", "Explanation": ""}
                ]
                
                metadata_df = pd.DataFrame(metadata_rows)
                export_df = pd.concat([metadata_df, export_df], ignore_index=True)
            
            return export_df.to_csv(index=False, encoding='utf-8')
            
        except Exception as e:
            logger.error(f"Error creating CSV export: {e}")
            return ""
    
    def create_excel_export(self, include_charts: bool = False) -> Optional[bytes]:
        """Crée un export Excel avec plusieurs feuilles"""
        try:
            if self.df.empty:
                return None
            
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Feuille principale avec les interactions
                self.df.to_excel(writer, sheet_name='Interactions', index=False)
                
                # Feuille avec les statistiques
                stats_data = [
                    ["Métrique", "Valeur"],
                    ["Date d'export", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    ["Total interactions", len(self.interactions)],
                    ["Interactions Major", self.stats.get('major', 0)],
                    ["Interactions Moderate", self.stats.get('moderate', 0)],
                    ["Interactions Minor", self.stats.get('minor', 0)],
                    ["Aucune interaction", self.stats.get('none', 0)],
                    ["Erreurs", self.stats.get('errors', 0)]
                ]
                
                stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
                stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
                
                # Feuille avec répartition par niveau
                if not self.df.empty:
                    level_counts = self.df['Level'].value_counts().reset_index()
                    level_counts.columns = ['Niveau', 'Nombre']
                    level_counts.to_excel(writer, sheet_name='Répartition', index=False)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating Excel export: {e}")
            return None
    
    def create_json_export(self, detailed: bool = True) -> str:
        """Crée un export JSON structuré"""
        try:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_interactions': len(self.interactions),
                    'statistics': self.stats
                },
                'interactions': self.interactions
            }
            
            if detailed:
                # Ajouter des analyses supplémentaires
                if not self.df.empty:
                    export_data['analysis'] = {
                        'level_distribution': self.df['Level'].value_counts().to_dict(),
                        'unique_drugs': list(set(list(self.df['Drug1']) + list(self.df['Drug2']))),
                        'drug_count': len(set(list(self.df['Drug1']) + list(self.df['Drug2'])))
                    }
            
            return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error creating JSON export: {e}")
            return ""

def create_export_section(interactions: List[Dict], stats: Dict[str, Any]):
    """Crée une section d'export complète"""
    if not interactions:
        create_status_message("Aucune donnée à exporter", "warning")
        return
    
    st.markdown("---")
    st.subheader("📥 Export des résultats")
    
    exporter = DataExporter(interactions, stats)
    
    # Options d'export
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Options d'export:**")
        include_metadata = st.checkbox("Inclure les métadonnées", value=True)
        include_charts = st.checkbox("Inclure les graphiques (Excel)", value=False)
    
    with col2:
        st.markdown("**Formats disponibles:**")
        export_format = st.selectbox(
            "Format d'export:",
            ["CSV", "Excel", "JSON"]
        )
    
    # Nom de fichier personnalisé
    default_filename = f"interactions_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    custom_filename = st.text_input(
        "Nom du fichier (sans extension):",
        value=default_filename
    )
    
    # Boutons d'export
    st.markdown("**Télécharger:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export CSV
        csv_data = exporter.create_csv_export(include_metadata)
        if csv_data:
            st.download_button(
                label="📄 Télécharger CSV",
                data=csv_data,
                file_name=f"{custom_filename}.csv",
                mime="text/csv",
                help="Format CSV compatible avec Excel et autres tableurs"
            )
    
    with col2:
        # Export Excel
        excel_data = exporter.create_excel_export(include_charts)
        if excel_data:
            st.download_button(
                label="📊 Télécharger Excel",
                data=excel_data,
                file_name=f"{custom_filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Format Excel avec plusieurs feuilles"
            )
    
    with col3:
        # Export JSON
        json_data = exporter.create_json_export(detailed=True)
        if json_data:
            st.download_button(
                label="🔧 Télécharger JSON",
                data=json_data,
                file_name=f"{custom_filename}.json",
                mime="application/json",
                help="Format JSON pour intégration avec d'autres systèmes"
            )
    
    # Aperçu des données
    with st.expander("👁️ Aperçu des données à exporter"):
        st.markdown(f"**Nombre total d'interactions:** {len(interactions)}")
        
        # Tableau de prévisualisation
        if len(interactions) > 0:
            preview_df = pd.DataFrame(interactions).head(5)
            st.dataframe(preview_df, use_container_width=True)
            
            if len(interactions) > 5:
                st.caption(f"Aperçu des 5 premières interactions sur {len(interactions)} au total")
        
        # Statistiques de prévisualisation
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Major", stats.get('major', 0))
        with col2:
            st.metric("Moderate", stats.get('moderate', 0))
        with col3:
            st.metric("Minor", stats.get('minor', 0))
        with col4:
            st.metric("Aucune", stats.get('none', 0))

def create_quick_export_buttons(interactions: List[Dict], stats: Dict[str, Any], prefix: str = ""):
    """Crée des boutons d'export rapide"""
    if not interactions:
        return
    
    exporter = DataExporter(interactions, stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = exporter.create_csv_export(include_metadata=True)
        if csv_data:
            st.download_button(
                label="📄 Export CSV",
                data=csv_data,
                file_name=f"{prefix}interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"quick_csv_{prefix}"
            )
    
    with col2:
        excel_data = exporter.create_excel_export()
        if excel_data:
            st.download_button(
                label="📊 Export Excel",
                data=excel_data,
                file_name=f"{prefix}interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"quick_excel_{prefix}"
            )

def export_analysis_summary(analyses: List[Dict]) -> str:
    """Exporte un résumé de plusieurs analyses"""
    try:
        summary_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_analyses': len(analyses),
                'date_range': {
                    'oldest': min(a.get('timestamp', datetime.now()) for a in analyses).isoformat() if analyses else None,
                    'newest': max(a.get('timestamp', datetime.now()) for a in analyses).isoformat() if analyses else None
                }
            },
            'summary_statistics': {
                'total_drug_combinations': sum(a.get('stats', {}).get('total_combinations', 0) for a in analyses),
                'total_major_interactions': sum(a.get('stats', {}).get('major', 0) for a in analyses),
                'total_moderate_interactions': sum(a.get('stats', {}).get('moderate', 0) for a in analyses),
                'total_minor_interactions': sum(a.get('stats', {}).get('minor', 0) for a in analyses)
            },
            'analyses': analyses
        }
        
        return json.dumps(summary_data, indent=2, ensure_ascii=False, default=str)
        
    except Exception as e:
        logger.error(f"Error creating analysis summary export: {e}")
        return ""

# Fonctions utilitaires pour l'export
def validate_filename(filename: str) -> str:
    """Valide et nettoie un nom de fichier"""
    import re
    # Supprimer les caractères invalides
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limiter la longueur
    if len(cleaned) > 100:
        cleaned = cleaned[:100]
    
    return cleaned or "export"

def get_file_size_human_readable(data: str) -> str:
    """Retourne la taille d'un fichier en format lisible"""
    size_bytes = len(data.encode('utf-8'))
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
