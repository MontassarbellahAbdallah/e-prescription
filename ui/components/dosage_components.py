"""
Composants UI pour l'analyse de dosage
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional
from config.logging_config import get_logger
from ui.styles import create_metric_card, create_status_message

logger = get_logger(__name__)

def display_dosage_analysis_section(dosage_result: Dict):
    """
    Affiche la section complète d'analyse de dosage
    
    Args:
        dosage_result: Résultat de l'analyse de dosage
    """
    if not dosage_result or 'dosage_analysis' not in dosage_result:
        st.warning("Aucune donnée de dosage disponible")
        return
    
    dosage_data = dosage_result['dosage_analysis']
    stats = dosage_result['stats']
    
    # En-tête de section
    st.markdown("### Dosage inadapté")
    
    # Vérifier s'il y a des problèmes
    if stats['total_issues'] == 0:
        create_status_message(
            "Aucun problème de dosage détecté dans cette prescription",
            "success"
        )
        
        # Afficher quand même les médicaments avec dosage approprié s'il y en a
        if stats.get('dosage_approprie_count', 0) > 0:
            st.info(f"{stats['dosage_approprie_count']} médicament(s) avec dosage approprié")
            _display_appropriate_dosages(dosage_data.get('dosage_approprie', []))
        return
    
    # Alerte si problèmes critiques
    if stats.get('has_critical_issues', False):
        create_status_message(
            f"{stats['total_issues']} problème(s) de dosage détecté(s) - Gravité élevée présente",
            "error"
        )
    else:
        create_status_message(
            f"{stats['total_issues']} problème(s) de dosage détecté(s)",
            "warning"
        )
    
    # Métriques de dosage
    display_dosage_metrics(stats)
    
    # Graphiques de dosage
    display_dosage_charts(dosage_data, stats)
    
    # Tableau détaillé
    display_dosage_table(dosage_data)
    
    # Recommandations
    display_dosage_recommendations(dosage_data)

def _display_appropriate_dosages(appropriate_dosages: List[Dict]):
    """
    Affiche les médicaments avec dosage approprié
    
    Args:
        appropriate_dosages: Liste des médicaments avec dosage approprié
    """
    if not appropriate_dosages:
        return
    
    with st.expander("Voir les dosages appropriés", expanded=False):
        for item in appropriate_dosages:
            medicament = item.get('medicament', 'Inconnu')
            dose = item.get('dose_prescrite', 'Non spécifiée')
            commentaire = item.get('commentaire', 'Dosage approprié')
            
            st.markdown(f"**{medicament}** ({dose}): {commentaire}")

def display_dosage_metrics(stats: Dict):
    """
    Affiche les métriques de dosage
    
    Args:
        stats: Statistiques de dosage
    """
    col1, col2, col3, col4= st.columns(4)
    
    with col1:
        total_meds = stats.get('total_medications', 0)
        create_metric_card(
            "Nombre des Médicaments", 
            str(total_meds)
        )
    
    with col2:
        surdosage_count = stats.get('surdosage_count', 0)
        create_metric_card(
            "Surdosages possible", 
            str(surdosage_count),
            #delta="Critique" if surdosage_count > 0 else None,
            #delta_color="error" if surdosage_count > 0 else "normal"
        )
    
    with col3:
        sous_dosage_count = stats.get('sous_dosage_count', 0)
        create_metric_card(
            "Sous-dosages possible", 
            str(sous_dosage_count),
            #delta="Attention" if sous_dosage_count > 0 else None,
            #delta_color="warning" if sous_dosage_count > 0 else "normal"
        )
    
    with col4:
        appropriate_count = stats.get('dosage_approprie_count', 0)
        create_metric_card(
            "Dosages appropriés", 
            str(appropriate_count),
            #delta_color="success"
        )
    

def display_dosage_charts(dosage_data: Dict, stats: Dict):
    """
    Affiche les graphiques d'analyse de dosage
    
    Args:
        dosage_data: Données d'analyse de dosage
        stats: Statistiques calculées
    """
    # Vérifier s'il y a des données à afficher
    if stats.get('total_issues', 0) == 0 and stats.get('dosage_approprie_count', 0) == 0:
        st.info("Aucune donnée à visualiser")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en camembert : répartition des types de problèmes
        fig_pie = create_dosage_pie_chart(stats)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Graphique de répartition non disponible")
    

def create_dosage_pie_chart(stats: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en camembert pour les types de problèmes de dosage
    
    Args:
        stats: Statistiques de dosage
        
    Returns:
        Figure Plotly ou None
    """
    try:
        # Préparer les données
        labels = []
        values = []
        colors = []
        
        surdosage_count = stats.get('surdosage_count', 0)
        sous_dosage_count = stats.get('sous_dosage_count', 0)
        appropriate_count = stats.get('dosage_approprie_count', 0)
        
        if surdosage_count > 0:
            labels.append('Surdosage')
            values.append(surdosage_count)
            colors.append('#DC3545')  # Rouge
        
        if sous_dosage_count > 0:
            labels.append('Sous-dosage') 
            values.append(sous_dosage_count)
            colors.append('#FD7E14')  # Orange
        
        if appropriate_count > 0:
            labels.append('Dosage approprié')
            values.append(appropriate_count)
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
            title="Répartition des problèmes de dosage",
            showlegend=True,
            #height=300,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating dosage pie chart: {e}")
        return None

def display_dosage_table(dosage_data: Dict):
    """
    Affiche le tableau détaillé des problèmes de dosage
    
    Args:
        dosage_data: Données d'analyse de dosage
    """
    st.markdown("#### Détail des problèmes de dosage")
    
    # Préparer les données pour le tableau
    table_data = []
    
    # Ajouter les surdosages
    for item in dosage_data.get('surdosage', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Surdosage',
            'Dose prescrite': item.get('dose_prescrite', 'Non spécifiée'),
            'Dose recommandée': item.get('dose_recommandee', 'Non spécifiée'),
            'Gravité': item.get('gravite', 'Faible'),
            'Facteur de risque': item.get('facteur_risque', 'Non spécifié'),
            'Explication': item.get('explication', ''),
            'Recommandation': item.get('recommandation', '')
        })
    
    # Ajouter les sous-dosages
    for item in dosage_data.get('sous_dosage', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Sous-dosage',
            'Dose prescrite': item.get('dose_prescrite', 'Non spécifiée'),
            'Dose recommandée': item.get('dose_recommandee', 'Non spécifiée'),
            'Gravité': item.get('gravite', 'Faible'),
            'Facteur de risque': item.get('facteur_risque', 'Non spécifié'),
            'Explication': item.get('explication', ''),
            'Recommandation': item.get('recommandation', '')
        })
    
    if not table_data:
        st.info("Aucun problème de dosage détecté")
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
                key="dosage_filter"
            )
        
        if filtre_type != 'Tous':
            df = df[df['Type'] == filtre_type]
    
    # Afficher le tableau SANS STYLE - Plus simple et toujours lisible
    st.dataframe(df, use_container_width=True)
    
    # Informations supplémentaires
    st.caption(f"Total: {len(table_data)} problème(s) de dosage")

def display_dosage_recommendations(dosage_data: Dict):
    """
    Affiche les recommandations pour les problèmes de dosage
    
    Args:
        dosage_data: Données d'analyse de dosage
    """
    #st.markdown("#### Recommandations")
    
    # Collecter toutes les recommandations
    recommendations = []
    
    # Recommandations pour surdosages
    for item in dosage_data.get('surdosage', []):
        if item.get('recommandation'):
            recommendations.append({
                'type': 'Surdosage',
                'medicament': item.get('medicament', 'Inconnu'),
                'recommandation': item.get('recommandation'),
                'gravite': item.get('gravite', 'Faible')
            })
    
    # Recommandations pour sous-dosages  
    for item in dosage_data.get('sous_dosage', []):
        if item.get('recommandation'):
            recommendations.append({
                'type': 'Sous-dosage',
                'medicament': item.get('medicament', 'Inconnu'),
                'recommandation': item.get('recommandation'),
                'gravite': item.get('gravite', 'Faible')
            })
    
    if not recommendations:
        st.info("Aucune recommandation spécifique pour le dosage")
        _display_general_dosage_recommendations()
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
            st.info(f"  **{medicament}** ({type_prob}): {recommandation}")
    
    # Recommandations générales
    _display_general_dosage_recommendations()

def _display_general_dosage_recommendations():
    """Affiche les recommandations générales pour le dosage"""
    st.markdown("##### Recommandations générales:")
    st.markdown("""
    - **Surveillance étroite** recommandée pour tous les ajustements de dosage
    - **Réévaluation clinique** dans les 24-48h après modification
    - **Monitoring des paramètres biologiques** selon les médicaments concernés
    - **Information du patient** sur les signes d'alerte à surveiller
    - **Documentation** de tout changement dans le dossier patient
    - **Consultation spécialisée** si ajustements complexes nécessaires
    """)

def get_dosage_summary_for_overview(dosage_result: Dict) -> Dict:
    """
    Retourne un résumé de l'analyse de dosage pour la vue d'ensemble
    
    Args:
        dosage_result: Résultat de l'analyse de dosage
        
    Returns:
        Dictionnaire avec résumé pour vue d'ensemble
    """
    if not dosage_result or 'stats' not in dosage_result:
        return {
            'status': 'no_data',
            'message': 'Pas de données de dosage',
            'color': 'secondary',
            'count': 0
        }
    
    stats = dosage_result['stats']
    total_issues = stats.get('total_issues', 0)
    has_critical = stats.get('has_critical_issues', False)
    total_meds = stats.get('total_medications', 0)
    
    if total_issues == 0:
        return {
            'status': 'ok',
            'message': 'Dosages appropriés',
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
            'message': f"{total_issues} problème(s) de dosage",
            'color': 'warning', 
            'count': total_issues
        }

def create_dosage_metrics_for_overview(dosage_result: Dict) -> Dict:
    """
    Crée les métriques de dosage pour la vue d'ensemble globale
    
    Args:
        dosage_result: Résultat de l'analyse de dosage
        
    Returns:
        Dictionnaire avec les métriques formatées
    """
    if not dosage_result or 'stats' not in dosage_result:
        return {
            'surdosage': 0,
            'sous_dosage': 0,
            'total_issues': 0,
            'has_critical': False
        }
    
    stats = dosage_result['stats']
    
    return {
        'surdosage': stats.get('surdosage_count', 0),
        'sous_dosage': stats.get('sous_dosage_count', 0),
        'total_issues': stats.get('total_issues', 0),
        'has_critical': stats.get('has_critical_issues', False),
        'appropriate': stats.get('dosage_approprie_count', 0)
    }