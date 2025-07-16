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
    
    
    # Métriques des voies d'administration
    #display_route_metrics(stats)
    
    # Graphiques des voies d'administration
    #display_route_charts(route_data, stats)
    
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
            
            st.markdown(f"**{medicament}** ({voie}): {commentaire}")

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

def display_route_recommendations(route_data: Dict):
    """
    Affiche les recommandations pour les problèmes de voies d'administration
    
    Args:
        route_data: Données d'analyse des voies d'administration
    """
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
            st.error(f"**{medicament}** ({type_prob}): {recommandation}")
        elif gravite == 'Modérée':
            st.warning(f"**{medicament}** ({type_prob}): {recommandation}")
        else:
            st.info(f"**{medicament}** ({type_prob}): {recommandation}")
    
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
            'count': total_meds
        }
    elif has_critical:
        return {
            'status': 'critical',
            'message': f"{total_issues} problème(s) critique(s)",
            'color': 'error',
            'count': total_issues
        }
    else:
        return {
            'status': 'warning',
            'message': f"{total_issues} problème(s) de voie",
            'color': 'warning', 
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
