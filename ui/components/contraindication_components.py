"""
Composants UI pour l'analyse de contre-indications
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional
from config.logging_config import get_logger
from ui.styles import create_metric_card, create_status_message

logger = get_logger(__name__)

def display_contraindication_analysis_section(contraindication_result: Dict):
    """
    Affiche la section complète d'analyse de contre-indications
    
    Args:
        contraindication_result: Résultat de l'analyse de contre-indications
    """
    if not contraindication_result or 'contraindication_analysis' not in contraindication_result:
        st.warning("Aucune donnée de contre-indications disponible")
        return
    
    contraindication_data = contraindication_result['contraindication_analysis']
    stats = contraindication_result['stats']
    context_used = contraindication_result.get('context_used', False)
    
    # En-tête de section
    st.markdown("### Contre-indications")
    
    # Information sur l'utilisation de la base vectorielle
    # if context_used:
    #     st.info("Analyse basée sur la base de données vectorielle (FAISS) et la prescription")
    # else:
    #     st.warning("Analyse basée uniquement sur les connaissances générales - Base de données vectorielle non disponible")
    
    # Vérifier s'il y a des contre-indications
    if stats['total_contraindications'] == 0:
        # Vérifier s'il y a des médicaments sans contre-indication
        if stats.get('aucune_contre_indication_count', 0) > 0:
            create_status_message(
                f"Aucune contre-indication détectée pour {stats['aucune_contre_indication_count']} médicament(s)",
                "success"
            )
            _display_safe_medications(contraindication_data.get('aucune_contre_indication', []))
        elif stats.get('donnees_insuffisantes_count', 0) > 0:
            create_status_message(
                f"Données insuffisantes dans la base de connaissances pour {stats['donnees_insuffisantes_count']} médicament(s)",
                "info"
            )
            _display_insufficient_data_medications(contraindication_data.get('donnees_insuffisantes', []))
        else:
            create_status_message(
                "La base de connaissances ne contient pas d'informations sur les contre-indications pour cette prescription",
                "info"
            )
        return
    
    # Alerte si contre-indications critiques
    if stats.get('has_critical_contraindications', False):
        create_status_message(
            f"{stats['contre_indications_absolues_count']} contre-indication(s) absolue(s) détectée(s) - PRESCRIPTION DANGEREUSE",
            "error"
        )
    
    if stats.get('contre_indications_relatives_count', 0) > 0:
        create_status_message(
            f"{stats['contre_indications_relatives_count']} contre-indication(s) relative(s) détectée(s) - SURVEILLANCE REQUISE",
            "warning"
        )
    
    # Métriques de contre-indications
    display_contraindication_metrics(stats)
    
    # Graphiques de contre-indications
    display_contraindication_charts(contraindication_data, stats)
    
    # Tableau détaillé
    display_contraindication_table(contraindication_data)
    
    # Recommandations
    display_contraindication_recommendations(contraindication_data, stats)

def _display_safe_medications(safe_medications: List[Dict]):
    """
    Affiche les médicaments sans contre-indication
    
    Args:
        safe_medications: Liste des médicaments sans contre-indication
    """
    if not safe_medications:
        return
    
    with st.expander("Voir les médicaments sans contre-indication", expanded=False):
        for item in safe_medications:
            medicament = item.get('medicament', 'Inconnu')
            commentaire = item.get('commentaire', 'Aucune contre-indication identifiée dans la base de connaissances')
            
            st.markdown(f"**{medicament}**: {commentaire}")

def _display_insufficient_data_medications(insufficient_data_medications: List[Dict]):
    """
    Affiche les médicaments avec données insuffisantes
    
    Args:
        insufficient_data_medications: Liste des médicaments avec données insuffisantes
    """
    if not insufficient_data_medications:
        return
    
    with st.expander("Voir les médicaments avec données insuffisantes", expanded=False):
        for item in insufficient_data_medications:
            medicament = item.get('medicament', 'Inconnu')
            raison = item.get('raison', 'La base de connaissances ne contient pas d\'informations suffisantes sur ce médicament')
            
            st.markdown(f"**{medicament}**: {raison}")

def display_contraindication_metrics(stats: Dict):
    """
    Affiche les métriques de contre-indications
    
    Args:
        stats: Statistiques de contre-indications
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_meds = stats.get('total_medications', 0)
        create_metric_card(
            "Médicaments analysés", 
            str(total_meds)
        )
    
    with col2:
        absolues_count = stats.get('contre_indications_absolues_count', 0)
        create_metric_card(
            "Contre-indications absolues", 
            str(absolues_count)
        )
    
    with col3:
        relatives_count = stats.get('contre_indications_relatives_count', 0)
        create_metric_card(
            "Contre-indications relatives", 
            str(relatives_count)
        )
    
    with col4:
        safe_count = stats.get('aucune_contre_indication_count', 0)
        create_metric_card(
            "Médicaments sûrs", 
            str(safe_count)
        )
    
    with col5:
        safety_level = stats.get('prescription_safety_level', 'UNKNOWN')
        create_metric_card(
            "Niveau de sécurité", 
            safety_level
        )

def display_contraindication_charts(contraindication_data: Dict, stats: Dict):
    """
    Affiche les graphiques d'analyse de contre-indications
    
    Args:
        contraindication_data: Données d'analyse de contre-indications
        stats: Statistiques calculées
    """
    # Vérifier s'il y a des données à afficher
    if stats.get('total_contraindications', 0) == 0 and stats.get('total_medications', 0) == 0:
        st.info("Aucune donnée à visualiser")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en camembert : répartition des types de contre-indications
        fig_pie = create_contraindication_pie_chart(stats)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Graphique de répartition non disponible")
    
    with col2:
        # Graphique en barres : niveau de sécurité
        fig_bar = create_safety_level_chart(stats)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Graphique de sécurité non disponible")

def create_contraindication_pie_chart(stats: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en camembert pour les types de contre-indications
    
    Args:
        stats: Statistiques de contre-indications
        
    Returns:
        Figure Plotly ou None
    """
    try:
        # Préparer les données
        labels = []
        values = []
        colors = []
        
        absolues_count = stats.get('contre_indications_absolues_count', 0)
        relatives_count = stats.get('contre_indications_relatives_count', 0)
        safe_count = stats.get('aucune_contre_indication_count', 0)
        insufficient_count = stats.get('donnees_insuffisantes_count', 0)
        
        if absolues_count > 0:
            labels.append('Contre-indications absolues')
            values.append(absolues_count)
            colors.append('#DC3545')  # Rouge
        
        if relatives_count > 0:
            labels.append('Contre-indications relatives') 
            values.append(relatives_count)
            colors.append('#FD7E14')  # Orange
        
        if safe_count > 0:
            labels.append('Aucune contre-indication')
            values.append(safe_count)
            colors.append('#28A745')  # Vert
        
        if insufficient_count > 0:
            labels.append('Données insuffisantes')
            values.append(insufficient_count)
            colors.append('#6C757D')  # Gris
        
        if not values:
            return None
        
        # Créer le graphique
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.3,
            textinfo='label+value+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Répartition des contre-indications",
            showlegend=True,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating contraindication pie chart: {e}")
        return None

def create_safety_level_chart(stats: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en barres pour le niveau de sécurité
    
    Args:
        stats: Statistiques de contre-indications
        
    Returns:
        Figure Plotly ou None
    """
    try:
        safety_level = stats.get('prescription_safety_level', 'UNKNOWN')
        
        # Données pour le graphique
        levels = ['SAFE', 'CAUTION', 'CRITICAL']
        values = [0, 0, 0]
        colors = ['#28A745', '#FD7E14', '#DC3545']
        
        if safety_level == 'SAFE':
            values[0] = 1
        elif safety_level == 'CAUTION':
            values[1] = 1
        elif safety_level == 'CRITICAL':
            values[2] = 1
        
        # Créer le graphique
        fig = go.Figure(data=[go.Bar(
            x=levels,
            y=values,
            marker_color=colors,
            text=['Sûr', 'Prudence', 'Critique'],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Niveau de sécurité de la prescription",
            xaxis_title="Niveau",
            yaxis_title="Status",
            yaxis=dict(range=[0, 1.2]),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating safety level chart: {e}")
        return None

def display_contraindication_table(contraindication_data: Dict):
    """
    Affiche le tableau détaillé des contre-indications
    
    Args:
        contraindication_data: Données d'analyse de contre-indications
    """
    st.markdown("#### Détail des contre-indications")
    
    # Préparer les données pour le tableau
    table_data = []
    
    # Ajouter les contre-indications absolues
    for item in contraindication_data.get('contre_indications_absolues', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Absolue',
            'Condition/Pathologie': item.get('condition', 'Non spécifiée'),
            'Mécanisme': item.get('mecanisme', 'Non spécifié'),
            'Conséquences': item.get('consequences', 'Risque élevé'),
            'Recommandation': item.get('recommandation', 'Éviter absolument'),
            'Source': item.get('source', 'Base de connaissances')
        })
    
    # Ajouter les contre-indications relatives
    for item in contraindication_data.get('contre_indications_relatives', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Relative',
            'Condition/Pathologie': item.get('condition', 'Non spécifiée'),
            'Mécanisme': item.get('mecanisme', 'Non spécifié'),
            'Conséquences': item.get('consequences', 'Risque modéré'),
            'Recommandation': item.get('recommandation', 'Surveillance renforcée'),
            'Source': item.get('source', 'Base de connaissances')
        })
    
    if not table_data:
        st.info("Aucune contre-indication détectée")
        return
    
    # Créer le DataFrame
    df = pd.DataFrame(table_data)
    
    # Ajouter un filtre par type si plusieurs types présents
    types_disponibles = df['Type'].unique().tolist()
    if len(types_disponibles) > 1:
        col1, col2 = st.columns([3, 1])
        with col2:
            filtre_type = st.selectbox(
                "Filtrer par type:",
                ['Tous'] + types_disponibles,
                key="contraindication_filter"
            )
        
        if filtre_type != 'Tous':
            df = df[df['Type'] == filtre_type]
    
    # Afficher le tableau
    st.dataframe(df, use_container_width=True)
    
    # Informations supplémentaires
    st.caption(f"Total: {len(table_data)} contre-indication(s) détectée(s)")

def display_contraindication_recommendations(contraindication_data: Dict, stats: Dict):
    """
    Affiche les recommandations pour les contre-indications
    
    Args:
        contraindication_data: Données d'analyse de contre-indications
        stats: Statistiques de contre-indications
    """
    #st.markdown("#### Recommandations")
    
    # Recommandations basées sur les contre-indications absolues
    absolues = contraindication_data.get('contre_indications_absolues', [])
    if absolues:
        st.markdown("##### Actions urgentes (Contre-indications absolues):")
        for item in absolues:
            medicament = item.get('medicament', 'Inconnu')
            recommandation = item.get('recommandation', 'Arrêt immédiat du médicament')
            
            st.error(f"**{medicament}**: {recommandation}")
    
    # Recommandations basées sur les contre-indications relatives
    relatives = contraindication_data.get('contre_indications_relatives', [])
    if relatives:
        st.markdown("##### Surveillance requise (Contre-indications relatives):")
        for item in relatives:
            medicament = item.get('medicament', 'Inconnu')
            recommandation = item.get('recommandation', 'Surveillance renforcée')
            
            st.warning(f"**{medicament}**: {recommandation}")
    
    # Recommandations générales
    _display_general_contraindication_recommendations(stats)

def _display_general_contraindication_recommendations(stats: Dict):
    """Affiche les recommandations générales pour les contre-indications"""
    st.markdown("##### Recommandations générales:")
    
    if stats.get('has_critical_contraindications', False):
        st.markdown("""
        **URGENCE - Contre-indications absolues détectées:**
        - **Arrêt immédiat** des médicaments contre-indiqués
        - **Contact immédiat** avec le prescripteur
        - **Réévaluation complète** de la prescription
        - **Surveillance médicale** urgente du patient
        - **Documentation** de tous les changements
        """)
    elif stats.get('contre_indications_relatives_count', 0) > 0:
        st.markdown("""
        **PRUDENCE - Contre-indications relatives détectées:**
        - **Évaluation bénéfice/risque** approfondie
        - **Surveillance clinique renforcée** 
        - **Ajustement posologique** si nécessaire
        - **Information du patient** sur les signes d'alerte
        - **Suivi régulier** des paramètres biologiques
        """)
    else:
        st.markdown("""
        **PRESCRIPTION SÛRE:**
        - **Poursuite** du traitement selon prescription
        - **Surveillance standard** du patient
        - **Réévaluation périodique** selon protocole
        """)

def get_contraindication_summary_for_overview(contraindication_result: Dict) -> Dict:
    """
    Retourne un résumé de l'analyse de contre-indications pour la vue d'ensemble
    
    Args:
        contraindication_result: Résultat de l'analyse de contre-indications
        
    Returns:
        Dictionnaire avec résumé pour vue d'ensemble
    """
    if not contraindication_result or 'stats' not in contraindication_result:
        return {
            'status': 'no_data',
            'message': 'Pas de données de contre-indications',
            'color': 'secondary',
            'count': 0
        }
    
    stats = contraindication_result['stats']
    total_contraindications = stats.get('total_contraindications', 0)
    has_critical = stats.get('has_critical_contraindications', False)
    safety_level = stats.get('prescription_safety_level', 'UNKNOWN')
    
    if total_contraindications == 0:
        return {
            'status': 'safe',
            'message': 'Aucune contre-indication',
            'color': 'success',
            'count': 0
        }
    elif has_critical:
        return {
            'status': 'critical',
            'message': f"{total_contraindications} contre-indication(s) absolue(s)",
            'color': 'error',
            'count': total_contraindications
        }
    else:
        return {
            'status': 'warning',
            'message': f"{total_contraindications} contre-indication(s) relative(s)",
            'color': 'warning', 
            'count': total_contraindications
        }
