"""
Composants UI pour l'analyse de redondance thérapeutique
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
from config.logging_config import get_logger
from ui.styles import create_metric_card, create_status_message

logger = get_logger(__name__)

def display_redundancy_analysis_section(redundancy_result: Dict):
    """
    Affiche la section complète d'analyse de redondance thérapeutique
    
    Args:
        redundancy_result: Résultat de l'analyse de redondance
    """
    if not redundancy_result or 'redundancy_analysis' not in redundancy_result:
        st.warning("Aucune donnée de redondance thérapeutique disponible")
        return
    
    redundancy_data = redundancy_result['redundancy_analysis']
    stats = redundancy_result['stats']
    context_used = redundancy_result.get('context_used', False)
    
    # En-tête de section
    st.markdown("### 🔄 Redondance thérapeutique")
    
    # Vérifier s'il y a des redondances
    if stats['total_redundancies'] == 0:
        # Vérifier s'il y a des médicaments sans redondance
        if stats.get('aucune_redondance_count', 0) > 0:
            create_status_message(
                f"✅ Aucune redondance thérapeutique détectée pour {stats['aucune_redondance_count']} médicament(s) - Prescription optimisée",
                "success"
            )
            _display_unique_medications(redundancy_data.get('aucune_redondance', []))
        else:
            create_status_message(
                "❓ Aucune information de redondance disponible dans l'analyse",
                "info"
            )
        return
    
    # Alerte si redondances critiques
    if stats.get('has_critical_redundancies', False):
        create_status_message(
            f"🚨 {stats['redondance_directe_count']} redondance(s) directe(s) détectée(s) - OPTIMISATION URGENTE REQUISE",
            "error"
        )
    
    if stats.get('redondance_classe_count', 0) > 0:
        create_status_message(
            f"⚠️ {stats['redondance_classe_count']} redondance(s) de classe détectée(s) - RÉVISION RECOMMANDÉE",
            "warning"
        )
    
    if stats.get('redondance_fonctionnelle_count', 0) > 0:
        create_status_message(
            f"💡 {stats['redondance_fonctionnelle_count']} redondance(s) fonctionnelle(s) - OPTIMISATION POSSIBLE",
            "info"
        )
    
    # Métriques de redondance
    display_redundancy_metrics(stats)
    
    # Graphiques de redondance
    display_redundancy_charts(redundancy_data, stats)
    
    # Tableau détaillé
    display_redundancy_table(redundancy_data)
    
    # Recommandations
    display_redundancy_recommendations(redundancy_data, stats)

def _display_unique_medications(unique_medications: List[Dict]):
    """
    Affiche les médicaments sans redondance
    
    Args:
        unique_medications: Liste des médicaments sans redondance
    """
    if not unique_medications:
        return
    
    with st.expander("Voir les médicaments uniques", expanded=False):
        for item in unique_medications:
            medicament = item.get('medicament', 'Inconnu')
            commentaire = item.get('commentaire', 'Médicament unique dans sa classe/fonction thérapeutique')
            
            st.markdown(f"✅ **{medicament}**: {commentaire}")

def display_redundancy_metrics(stats: Dict):
    """
    Affiche les métriques de redondance
    
    Args:
        stats: Statistiques de redondance
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_meds = stats.get('total_medications', 0)
        create_metric_card(
            "Médicaments analysés", 
            str(total_meds)
        )
    
    with col2:
        directe_count = stats.get('redondance_directe_count', 0)
        create_metric_card(
            "Redondances directes", 
            str(directe_count)
        )
    
    with col3:
        classe_count = stats.get('redondance_classe_count', 0)
        create_metric_card(
            "Redondances de classe", 
            str(classe_count)
        )
    
    with col4:
        fonctionnelle_count = stats.get('redondance_fonctionnelle_count', 0)
        create_metric_card(
            "Redondances fonctionnelles", 
            str(fonctionnelle_count)
        )
    
    with col5:
        optimization_potential = stats.get('prescription_optimization_potential', 'UNKNOWN')
        create_metric_card(
            "Potentiel d'optimisation", 
            optimization_potential
        )

def display_redundancy_charts(redundancy_data: Dict, stats: Dict):
    """
    Affiche les graphiques d'analyse de redondance
    
    Args:
        redundancy_data: Données d'analyse de redondance
        stats: Statistiques calculées
    """
    # Vérifier s'il y a des données à afficher
    if stats.get('total_redundancies', 0) == 0 and stats.get('total_medications', 0) == 0:
        st.info("Aucune donnée à visualiser")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en camembert : répartition des types de redondances
        fig_pie = create_redundancy_pie_chart(stats)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Graphique de répartition non disponible")
    
    with col2:
        # Graphique en barres : potentiel d'optimisation
        fig_bar = create_optimization_potential_chart(stats)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Graphique d'optimisation non disponible")

def create_redundancy_pie_chart(stats: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en camembert pour les types de redondances
    
    Args:
        stats: Statistiques de redondance
        
    Returns:
        Figure Plotly ou None
    """
    try:
        # Préparer les données
        labels = []
        values = []
        colors = []
        
        directe_count = stats.get('redondance_directe_count', 0)
        classe_count = stats.get('redondance_classe_count', 0)
        fonctionnelle_count = stats.get('redondance_fonctionnelle_count', 0)
        unique_count = stats.get('aucune_redondance_count', 0)
        
        if directe_count > 0:
            labels.append('Redondances directes')
            values.append(directe_count)
            colors.append('#DC3545')  # Rouge
        
        if classe_count > 0:
            labels.append('Redondances de classe') 
            values.append(classe_count)
            colors.append('#FD7E14')  # Orange
        
        if fonctionnelle_count > 0:
            labels.append('Redondances fonctionnelles')
            values.append(fonctionnelle_count)
            colors.append('#17A2B8')  # Bleu clair
        
        if unique_count > 0:
            labels.append('Médicaments uniques')
            values.append(unique_count)
            colors.append('#28A745')  # Vert
        
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
            title="Répartition des redondances thérapeutiques",
            showlegend=True,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating redundancy pie chart: {e}")
        return None

def create_optimization_potential_chart(stats: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en barres pour le potentiel d'optimisation
    
    Args:
        stats: Statistiques de redondance
        
    Returns:
        Figure Plotly ou None
    """
    try:
        optimization_potential = stats.get('prescription_optimization_potential', 'UNKNOWN')
        
        # Données pour le graphique
        levels = ['LOW', 'MEDIUM', 'HIGH']
        values = [0, 0, 0]
        colors = ['#28A745', '#FD7E14', '#DC3545']
        
        if optimization_potential == 'LOW':
            values[0] = 1
        elif optimization_potential == 'MEDIUM':
            values[1] = 1
        elif optimization_potential == 'HIGH':
            values[2] = 1
        
        # Créer le graphique
        fig = go.Figure(data=[go.Bar(
            x=levels,
            y=values,
            marker_color=colors,
            text=['Faible', 'Moyen', 'Élevé'],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Potentiel d'optimisation de la prescription",
            xaxis_title="Niveau",
            yaxis_title="Status",
            yaxis=dict(range=[0, 1.2]),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating optimization potential chart: {e}")
        return None

def display_redundancy_table(redundancy_data: Dict):
    """
    Affiche le tableau détaillé des redondances
    
    Args:
        redundancy_data: Données d'analyse de redondance
    """
    st.markdown("#### Détail des redondances thérapeutiques")
    
    # Préparer les données pour le tableau
    table_data = []
    
    # Ajouter les redondances directes
    for item in redundancy_data.get('redondance_directe', []):
        table_data.append({
            'Classe thérapeutique': item.get('classe_therapeutique', 'Inconnue'),
            'Médicaments redondants': ', '.join(item.get('medicaments', [])),
            'Type de redondance': 'Redondance directe',
            'Gravité': 'Élevée',
            'Mécanisme': item.get('mecanisme', 'Même molécule active'),
            'Risque': item.get('risque', 'Surdosage, effets cumulés'),
            'Recommandation': item.get('recommandation', 'Éliminer les doublons'),
            'Source': item.get('source', 'Base de connaissances')
        })
    
    # Ajouter les redondances de classe
    for item in redundancy_data.get('redondance_classe', []):
        table_data.append({
            'Classe thérapeutique': item.get('classe_therapeutique', 'Inconnue'),
            'Médicaments redondants': ', '.join(item.get('medicaments', [])),
            'Type de redondance': 'Redondance de classe',
            'Gravité': 'Modérée',
            'Mécanisme': item.get('mecanisme', 'Même classe thérapeutique'),
            'Risque': item.get('risque', 'Effets additifs, interactions'),
            'Recommandation': item.get('recommandation', 'Évaluer la nécessité'),
            'Source': item.get('source', 'Base de connaissances')
        })
    
    # Ajouter les redondances fonctionnelles
    for item in redundancy_data.get('redondance_fonctionnelle', []):
        table_data.append({
            'Classe thérapeutique': item.get('classe_therapeutique', 'Inconnue'),
            'Médicaments redondants': ', '.join(item.get('medicaments', [])),
            'Type de redondance': 'Redondance fonctionnelle',
            'Gravité': 'Faible à Modérée',
            'Mécanisme': item.get('mecanisme', 'Effet thérapeutique similaire'),
            'Risque': item.get('risque', 'Complexification du traitement'),
            'Recommandation': item.get('recommandation', 'Optimiser la stratégie'),
            'Source': item.get('source', 'Base de connaissances')
        })
    
    if not table_data:
        st.info("Aucune redondance thérapeutique détectée")
        return
    
    # Créer le DataFrame
    df = pd.DataFrame(table_data)
    
    # Ajouter un filtre par type si plusieurs types présents
    types_disponibles = df['Type de redondance'].unique().tolist()
    if len(types_disponibles) > 1:
        col1, col2 = st.columns([3, 1])
        with col2:
            filtre_type = st.selectbox(
                "Filtrer par type:",
                ['Tous'] + types_disponibles,
                key="redundancy_filter"
            )
        
        if filtre_type != 'Tous':
            df = df[df['Type de redondance'] == filtre_type]
    
    # Afficher le tableau
    st.dataframe(df, use_container_width=True)
    
    # Informations supplémentaires
    st.caption(f"Total: {len(table_data)} redondance(s) thérapeutique(s) détectée(s)")

def display_redundancy_recommendations(redundancy_data: Dict, stats: Dict):
    """
    Affiche les recommandations pour les redondances
    
    Args:
        redundancy_data: Données d'analyse de redondance
        stats: Statistiques de redondance
    """
    st.markdown("#### 💡 Recommandations d'optimisation")
    
    # Recommandations basées sur les redondances directes
    directes = redundancy_data.get('redondance_directe', [])
    if directes:
        st.markdown("##### 🚨 Actions urgentes (Redondances directes):")
        for item in directes:
            medicaments = ', '.join(item.get('medicaments', []))
            recommandation = item.get('recommandation', 'Éliminer les doublons, ajuster la posologie')
            
            st.error(f"🚨 **{medicaments}**: {recommandation}")
    
    # Recommandations basées sur les redondances de classe
    classe = redundancy_data.get('redondance_classe', [])
    if classe:
        st.markdown("##### ⚠️ Révision recommandée (Redondances de classe):")
        for item in classe:
            medicaments = ', '.join(item.get('medicaments', []))
            recommandation = item.get('recommandation', 'Évaluer la nécessité, choisir un représentant')
            
            st.warning(f"⚠️ **{medicaments}**: {recommandation}")
    
    # Recommandations basées sur les redondances fonctionnelles
    fonctionnelles = redundancy_data.get('redondance_fonctionnelle', [])
    if fonctionnelles:
        st.markdown("##### 💡 Optimisation possible (Redondances fonctionnelles):")
        for item in fonctionnelles:
            medicaments = ', '.join(item.get('medicaments', []))
            recommandation = item.get('recommandation', 'Optimiser la stratégie thérapeutique')
            
            st.info(f"💡 **{medicaments}**: {recommandation}")
    
    # Recommandations générales
    _display_general_redundancy_recommendations(stats)

def _display_general_redundancy_recommendations(stats: Dict):
    """Affiche les recommandations générales pour les redondances"""
    st.markdown("##### Recommandations générales:")
    
    optimization_potential = stats.get('prescription_optimization_potential', 'LOW')
    
    if optimization_potential == 'HIGH':
        st.markdown("""
        **🚨 OPTIMISATION URGENTE - Redondances critiques détectées:**
        - **Élimination immédiate** des doublons médicamenteux
        - **Révision complète** de la stratégie thérapeutique
        - **Ajustement posologique** après suppression des redondances
        - **Surveillance médicale** renforcée pendant la transition
        - **Documentation** de tous les changements effectués
        """)
    elif optimization_potential == 'MEDIUM':
        st.markdown("""
        **⚠️ RÉVISION RECOMMANDÉE - Redondances modérées détectées:**
        - **Évaluation bénéfice/risque** de chaque association
        - **Simplification** de la prescription si possible
        - **Choix du médicament** le plus approprié par classe
        - **Information du patient** sur les modifications
        - **Suivi clinique** pour s'assurer de l'efficacité maintenue
        """)
    else:
        st.markdown("""
        **✅ PRESCRIPTION OPTIMISÉE:**
        - **Aucune redondance** thérapeutique majeure détectée
        - **Poursuite** du traitement selon prescription
        - **Réévaluation périodique** de l'optimisation thérapeutique
        - **Surveillance standard** des effets thérapeutiques
        """)

def get_redundancy_summary_for_overview(redundancy_result: Dict) -> Dict:
    """
    Retourne un résumé de l'analyse de redondance pour la vue d'ensemble
    
    Args:
        redundancy_result: Résultat de l'analyse de redondance
        
    Returns:
        Dictionnaire avec résumé pour vue d'ensemble
    """
    if not redundancy_result or 'stats' not in redundancy_result:
        return {
            'status': 'no_data',
            'message': 'Pas de données de redondance',
            'color': 'secondary',
            'icon': '❓',
            'count': 0
        }
    
    stats = redundancy_result['stats']
    total_redundancies = stats.get('total_redundancies', 0)
    has_critical = stats.get('has_critical_redundancies', False)
    optimization_potential = stats.get('prescription_optimization_potential', 'UNKNOWN')
    
    if total_redundancies == 0:
        return {
            'status': 'optimized',
            'message': 'Prescription optimisée',
            'color': 'success',
            'icon': '✅',
            'count': 0
        }
    elif has_critical:
        return {
            'status': 'critical',
            'message': f"{total_redundancies} redondance(s) critique(s)",
            'color': 'error',
            'icon': '🚨',
            'count': total_redundancies
        }
    elif optimization_potential == 'MEDIUM':
        return {
            'status': 'medium',
            'message': f"{total_redundancies} redondance(s) modérée(s)",
            'color': 'warning', 
            'icon': '⚠️',
            'count': total_redundancies
        }
    else:
        return {
            'status': 'low',
            'message': f"{total_redundancies} redondance(s) mineure(s)",
            'color': 'info', 
            'icon': '💡',
            'count': total_redundancies
        }

def create_redundancy_metrics_for_overview(redundancy_result: Dict) -> Dict:
    """
    Crée les métriques de redondance pour la vue d'ensemble globale
    
    Args:
        redundancy_result: Résultat de l'analyse de redondance
        
    Returns:
        Dictionnaire avec les métriques formatées
    """
    if not redundancy_result or 'stats' not in redundancy_result:
        return {
            'redondance_directe': 0,
            'redondance_classe': 0,
            'redondance_fonctionnelle': 0,
            'total_redundancies': 0,
            'has_critical': False,
            'optimization_potential': 'UNKNOWN'
        }
    
    stats = redundancy_result['stats']
    
    return {
        'redondance_directe': stats.get('redondance_directe_count', 0),
        'redondance_classe': stats.get('redondance_classe_count', 0),
        'redondance_fonctionnelle': stats.get('redondance_fonctionnelle_count', 0),
        'total_redundancies': stats.get('total_redundancies', 0),
        'has_critical': stats.get('has_critical_redundancies', False),
        'optimization_potential': stats.get('prescription_optimization_potential', 'UNKNOWN')
    }
