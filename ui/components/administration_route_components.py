"""
Composants UI pour l'analyse des voies d'administration
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config.logging_config import get_logger
from ui.styles import create_metric_card, create_status_message

logger = get_logger(__name__)

def display_administration_route_section(route_result: Dict):
    """
    Affiche la section complète d'analyse des voies d'administration
    
    Args:
        route_result: Résultat de l'analyse des voies d'administration
    """
    if not route_result or 'administration_route_analysis' not in route_result:
        st.warning("Aucune donnée sur les voies d'administration disponible")
        return
    
    route_data = route_result['administration_route_analysis']
    stats = route_result['stats']
    
    # En-tête de section
    st.markdown("### 💉 Voie d'administration inappropriée")
    
    # Vérifier s'il y a des problèmes
    if stats['total_issues'] == 0:
        create_status_message(
            "✅ Aucun problème de voie d'administration détecté dans cette prescription",
            "success"
        )
        
        # Afficher quand même les médicaments avec voie appropriée s'il y en a
        if stats.get('voie_appropriee_count', 0) > 0:
            st.info(f"📋 {stats['voie_appropriee_count']} médicament(s) avec voie d'administration appropriée")
            _display_appropriate_routes(route_data.get('voie_appropriee', []))
        return
    
    # Alerte si problèmes critiques
    if stats.get('has_critical_issues', False):
        create_status_message(
            f"🚨 {stats['total_issues']} problème(s) de voie d'administration détecté(s) - Gravité élevée présente",
            "error"
        )
    else:
        create_status_message(
            f"⚠️ {stats['total_issues']} problème(s) de voie d'administration détecté(s)",
            "warning"
        )
    
    # Métriques des voies d'administration
    display_route_metrics(stats)
    
    # Graphiques des voies d'administration
    display_route_charts(route_data, stats)
    
    # Tableau détaillé
    display_route_table(route_data)
    
    # Chronologie d'administration (spécifique aux voies d'administration)
    
    # Recommandations
    display_route_recommendations(route_data)

def _display_appropriate_routes(appropriate_routes: List[Dict]):
    """
    Affiche les médicaments avec voie d'administration appropriée
    
    Args:
        appropriate_routes: Liste des médicaments avec voie appropriée
    """
    if not appropriate_routes:
        return
    
    with st.expander("Voir les voies d'administration appropriées", expanded=False):
        for item in appropriate_routes:
            medicament = item.get('medicament', 'Inconnu')
            voie = item.get('voie_prescrite', 'Non spécifiée')
            commentaire = item.get('commentaire', 'Voie appropriée')
            
            st.markdown(f"✅ **{medicament}** ({voie}): {commentaire}")

def display_route_metrics(stats: Dict):
    """
    Affiche les métriques des voies d'administration
    
    Args:
        stats: Statistiques des voies d'administration
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_meds = stats.get('total_medications', 0)
        create_metric_card(
            "Médicaments analysés", 
            str(total_meds)
        )
    
    with col2:
        inappropriee_count = stats.get('voie_inappropriee_count', 0)
        create_metric_card(
            "Voies inappropriées", 
            str(inappropriee_count)
        )
    
    with col3:
        incompatible_count = stats.get('voie_incompatible_count', 0)
        create_metric_card(
            "Incompatibilités", 
            str(incompatible_count)
        )
    
    with col4:
        risquee_count = stats.get('voie_risquee_count', 0)
        create_metric_card(
            "Voies risquées", 
            str(risquee_count)
        )
    
    with col5:
        total_issues = stats.get('total_issues', 0)
        create_metric_card(
            "Total problèmes", 
            str(total_issues)
        )

def display_route_charts(route_data: Dict, stats: Dict):
    """
    Affiche les graphiques d'analyse des voies d'administration
    
    Args:
        route_data: Données d'analyse des voies d'administration
        stats: Statistiques calculées
    """
    # Vérifier s'il y a des données à afficher
    if stats.get('total_issues', 0) == 0 and stats.get('voie_appropriee_count', 0) == 0:
        st.info("Aucune donnée à visualiser")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en camembert : répartition des types de problèmes
        fig_pie = create_route_pie_chart(stats)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Graphique de répartition non disponible")
    
    with col2:
        # Graphique en barres : répartition par gravité
        fig_bar = create_route_severity_chart(stats)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Graphique de gravité non disponible")

def create_route_pie_chart(stats: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en camembert pour les types de problèmes de voies d'administration
    
    Args:
        stats: Statistiques des voies d'administration
        
    Returns:
        Figure Plotly ou None
    """
    try:
        # Préparer les données
        labels = []
        values = []
        colors = []
        
        inappropriee_count = stats.get('voie_inappropriee_count', 0)
        incompatible_count = stats.get('voie_incompatible_count', 0)
        risquee_count = stats.get('voie_risquee_count', 0)
        appropriee_count = stats.get('voie_appropriee_count', 0)
        
        if inappropriee_count > 0:
            labels.append('Voie inappropriée')
            values.append(inappropriee_count)
            colors.append('#DC3545')  # Rouge
        
        if incompatible_count > 0:
            labels.append('Incompatibilité') 
            values.append(incompatible_count)
            colors.append('#E83E8C')  # Rose
        
        if risquee_count > 0:
            labels.append('Voie risquée')
            values.append(risquee_count)
            colors.append('#FD7E14')  # Orange
        
        if appropriee_count > 0:
            labels.append('Voie appropriée')
            values.append(appropriee_count)
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
            title="Répartition des problèmes de voies d'administration",
            showlegend=True,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating route pie chart: {e}")
        return None

def create_route_severity_chart(stats: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en barres pour la gravité des problèmes
    
    Args:
        stats: Statistiques des voies d'administration
        
    Returns:
        Figure Plotly ou None
    """
    try:
        gravite_data = stats.get('gravite_repartition', {})
        
        # S'assurer que toutes les clés existent
        gravite_data = {
            'Faible': gravite_data.get('Faible', 0),
            'Modérée': gravite_data.get('Modérée', 0),
            'Élevée': gravite_data.get('Élevée', 0)
        }
        
        # Filtrer les valeurs non nulles
        labels = [k for k, v in gravite_data.items() if v > 0]
        values = [v for v in gravite_data.values() if v > 0]
        
        if not values:
            return None
        
        # Couleurs selon la gravité
        color_map = {
            'Faible': '#28A745',
            'Modérée': '#FD7E14', 
            'Élevée': '#DC3545'
        }
        colors = [color_map.get(label, '#6C757D') for label in labels]
        
        # Créer le graphique
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Gravité des problèmes de voies d'administration",
            xaxis_title="Niveau de gravité",
            yaxis_title="Nombre de problèmes",
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating route severity chart: {e}")
        return None

def display_route_table(route_data: Dict):
    """
    Affiche le tableau détaillé des problèmes de voies d'administration
    
    Args:
        route_data: Données d'analyse des voies d'administration
    """
    st.markdown("#### Détail des problèmes de voies d'administration")
    
    # Préparer les données pour le tableau
    table_data = []
    
    # Ajouter les voies inappropriées
    for item in route_data.get('voie_inappropriee', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Voie inappropriée',
            'Voie prescrite': item.get('voie_prescrite', 'Non spécifiée'),
            'Voie recommandée': item.get('voie_recommandee', 'Non spécifiée'),
            'Gravité': item.get('gravite', 'Faible'),
            'Justification': item.get('justification', 'Non spécifiée'),
            'Recommandation': item.get('recommandation', '')
        })
    
    # Ajouter les voies incompatibles
    for item in route_data.get('voie_incompatible', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Incompatibilité',
            'Voie prescrite': item.get('voie_prescrite', 'Non spécifiée'),
            'Voie recommandée': item.get('voie_recommandee', 'Non spécifiée'),
            'Gravité': item.get('gravite', 'Faible'),
            'Justification': item.get('justification', 'Non spécifiée'),
            'Recommandation': item.get('recommandation', '')
        })
    
    # Ajouter les voies risquées
    for item in route_data.get('voie_risquee', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Voie risquée',
            'Voie prescrite': item.get('voie_prescrite', 'Non spécifiée'),
            'Voie recommandée': item.get('voie_recommandee', 'Non spécifiée'),
            'Gravité': item.get('gravite', 'Faible'),
            'Justification': item.get('justification', 'Non spécifiée'),
            'Recommandation': item.get('recommandation', '')
        })
    
    if not table_data:
        st.info("Aucun problème de voie d'administration détecté")
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
                key="route_filter"
            )
        
        if filtre_type != 'Tous':
            df = df[df['Type'] == filtre_type]
    
    # Afficher le tableau
    st.dataframe(df, use_container_width=True)
    
    # Informations supplémentaires
    st.caption(f"Total: {len(table_data)} problème(s) de voie d'administration")

def display_route_recommendations(route_data: Dict):
    """
    Affiche les recommandations pour les problèmes de voies d'administration
    
    Args:
        route_data: Données d'analyse des voies d'administration
    """
    st.markdown("#### 💡 Recommandations")
    
    # Collecter toutes les recommandations
    recommendations = []
    
    # Recommandations pour voies inappropriées
    for item in route_data.get('voie_inappropriee', []):
        if item.get('recommandation'):
            recommendations.append({
                'type': 'Voie inappropriée',
                'medicament': item.get('medicament', 'Inconnu'),
                'recommandation': item.get('recommandation'),
                'gravite': item.get('gravite', 'Faible')
            })
    
    # Recommandations pour incompatibilités
    for item in route_data.get('voie_incompatible', []):
        if item.get('recommandation'):
            recommendations.append({
                'type': 'Incompatibilité',
                'medicament': item.get('medicament', 'Inconnu'),
                'recommandation': item.get('recommandation'),
                'gravite': item.get('gravite', 'Faible')
            })
    
    # Recommandations pour voies risquées
    for item in route_data.get('voie_risquee', []):
        if item.get('recommandation'):
            recommendations.append({
                'type': 'Voie risquée',
                'medicament': item.get('medicament', 'Inconnu'),
                'recommandation': item.get('recommandation'),
                'gravite': item.get('gravite', 'Faible')
            })
    
    if not recommendations:
        st.info("Aucune recommandation spécifique pour les voies d'administration")
        _display_general_route_recommendations()
        return
    
    # Trier par gravité (Élevée en premier)
    severity_order = {'Élevée': 3, 'Modérée': 2, 'Faible': 1}
    recommendations.sort(key=lambda x: severity_order.get(x['gravite'], 0), reverse=True)
    
    # Afficher les recommandations spécifiques
    st.markdown("##### Recommandations spécifiques:")
    for i, rec in enumerate(recommendations, 1):
        gravite = rec['gravite']
        medicament = rec['medicament']
        type_prob = rec['type']
        recommandation = rec['recommandation']
        
        if gravite == 'Élevée':
            st.error(f"🚨 **{medicament}** ({type_prob}): {recommandation}")
        elif gravite == 'Modérée':
            st.warning(f"⚠️ **{medicament}** ({type_prob}): {recommandation}")
        else:
            st.info(f"💡 **{medicament}** ({type_prob}): {recommandation}")
    
    # Recommandations générales
    _display_general_route_recommendations()

def _display_general_route_recommendations():
    """Affiche les recommandations générales pour les voies d'administration"""
    st.markdown("##### Recommandations générales:")
    st.markdown("""
    - **Validation pharmaceutique** avant administration par voie intraveineuse
    - **Respect des débits** d'administration recommandés
    - **Formation du personnel** aux bonnes pratiques d'administration
    - **Surveillance étroite** lors des administrations par voies risquées
    - **Suivi des paramètres cliniques** pendant et après l'administration
    - **Documentation complète** des modalités d'administration dans le dossier patient
    - **Matériel adapté** selon la voie d'administration (filtres, pompes, etc.)
    """)

def get_administration_route_summary_for_overview(route_result: Dict) -> Dict:
    """
    Retourne un résumé de l'analyse des voies d'administration pour la vue d'ensemble
    
    Args:
        route_result: Résultat de l'analyse des voies d'administration
        
    Returns:
        Dictionnaire avec résumé pour vue d'ensemble
    """
    if not route_result or 'stats' not in route_result:
        return {
            'status': 'no_data',
            'message': 'Pas de données de voies d\'administration',
            'color': 'secondary',
            'icon': '❓',
            'count': 0
        }
    
    stats = route_result['stats']
    total_issues = stats.get('total_issues', 0)
    has_critical = stats.get('has_critical_issues', False)
    total_meds = stats.get('total_medications', 0)
    
    if total_issues == 0:
        return {
            'status': 'ok',
            'message': 'Voies appropriées',
            'color': 'success',
            'icon': '✅',
            'count': total_meds
        }
    elif has_critical:
        return {
            'status': 'critical',
            'message': f"{total_issues} problème(s) critique(s)",
            'color': 'error',
            'icon': '🚨',
            'count': total_issues
        }
    else:
        return {
            'status': 'warning',
            'message': f"{total_issues} problème(s) de voie",
            'color': 'warning', 
            'icon': '⚠️',
            'count': total_issues
        }

def create_administration_route_metrics_for_overview(route_result: Dict) -> Dict:
    """
    Crée les métriques de voies d'administration pour la vue d'ensemble globale
    
    Args:
        route_result: Résultat de l'analyse des voies d'administration
        
    Returns:
        Dictionnaire avec les métriques formatées
    """
    if not route_result or 'stats' not in route_result:
        return {
            'voie_inappropriee': 0,
            'voie_incompatible': 0,
            'voie_risquee': 0,
            'total_issues': 0,
            'has_critical': False
        }
    
    stats = route_result['stats']
    
    return {
        'voie_inappropriee': stats.get('voie_inappropriee_count', 0),
        'voie_incompatible': stats.get('voie_incompatible_count', 0),
        'voie_risquee': stats.get('voie_risquee_count', 0),
        'total_issues': stats.get('total_issues', 0),
        'has_critical': stats.get('has_critical_issues', False),
        'appropriate': stats.get('voie_appropriee_count', 0)
    }
