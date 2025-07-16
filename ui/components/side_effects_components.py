"""
Composants UI pour l'analyse d'effets secondaires
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional
from config.logging_config import get_logger

logger = get_logger(__name__)

def display_side_effects_analysis_section(side_effects_result: Dict):
    """
    Affiche la section complète d'analyse d'effets secondaires
    
    Args:
        side_effects_result: Résultat de l'analyse d'effets secondaires
    """
    if not side_effects_result or 'side_effects_analysis' not in side_effects_result:
        st.warning("Aucune donnée d'effets secondaires disponible")
        return
    
    side_effects_data = side_effects_result['side_effects_analysis']
    stats = side_effects_result['stats']
    
    # En-tête de section
    # st.markdown("### Effets secondaires")
    
    # # Vérifier s'il y a des effets secondaires
    # if stats['total_side_effects'] == 0 and stats['effets_cumules_count'] == 0 and stats['risques_graves_count'] == 0:
    #     create_status_message(
    #         "Aucun effet secondaire significatif détecté pour cette prescription",
    #         "success"
    #     )
    #     return
    
    # # Alerte si effets critiques
    # if stats.get('has_critical_effects', False):
    #     create_status_message(
    #         f"Effets secondaires détectés - Surveillance nécessaire",
    #         "error"
    #     )
    # else:
    #     create_status_message(
    #         f"Effets secondaires détectés - Surveillance recommandée",
    #         "warning"
    #     )
    
    # Métriques d'effets secondaires

    
    # Tableau détaillé
    display_side_effects_table(side_effects_data)
    
    # Recommandations
    display_side_effects_recommendations(side_effects_data)


def create_side_effects_stacked_area_chart(side_effects_data: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en aires empilées pour la distribution des effets par gravité
    
    Args:
        side_effects_data: Données d'effets secondaires
        
    Returns:
        Figure Plotly ou None
    """
    try:
        effets_individuels = side_effects_data.get('effets_individuels', [])
        
        if not effets_individuels:
            return None
        
        # Préparer les données
        medications = []
        gravite_counts = {}
        
        for effet in effets_individuels:
            med = effet.get('medicament', 'Inconnu')
            gravite = effet.get('gravite', 'Faible')
            
            if med not in medications:
                medications.append(med)
            
            if med not in gravite_counts:
                gravite_counts[med] = {'Faible': 0, 'Modérée': 0, 'Élevée': 0}
            
            # Compter les effets (pas seulement 1 par médicament)
            nb_effets = len(effet.get('effets', []))
            gravite_counts[med][gravite] += nb_effets
        
        # Créer le graphique
        fig = go.Figure()
        
        colors = {'Faible': '#28A745', 'Modérée': '#FD7E14', 'Élevée': '#DC3545'}
        
        for gravite in ['Faible', 'Modérée', 'Élevée']:
            values = [gravite_counts[med][gravite] for med in medications]
            fig.add_trace(go.Scatter(
                x=medications,
                y=values,
                mode='lines',
                stackgroup='one',
                name=gravite,
                fill='tonexty',
                line=dict(color=colors[gravite])
            ))
        
        fig.update_layout(
            title="Distribution des Effets par Gravité",
            xaxis_title="Médicaments",
            yaxis_title="Nombre d'effets secondaires",
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating stacked area chart: {e}")
        return None

def create_side_effects_radar_chart(side_effects_data: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique radar pour visualiser les effets par système
    
    Args:
        side_effects_data: Données d'effets secondaires
        
    Returns:
        Figure Plotly ou None
    """
    try:
        effets_individuels = side_effects_data.get('effets_individuels', [])
        
        if not effets_individuels:
            return None
        
        # Analyser les systèmes affectés
        systems_data = {}
        
        for effet in effets_individuels:
            medicament = effet.get('medicament', 'Inconnu')
            systeme = effet.get('systeme_affecte', 'Non spécifié')
            gravite = effet.get('gravite', 'Faible')
            
            if medicament not in systems_data:
                systems_data[medicament] = {}
            
            if systeme not in systems_data[medicament]:
                systems_data[medicament][systeme] = 0
            
            # Pondérer par gravité
            weight = {'Faible': 1, 'Modérée': 2, 'Élevée': 3}.get(gravite, 1)
            systems_data[medicament][systeme] += weight
        
        # Créer le graphique radar
        fig = go.Figure()
        
        # Tous les systèmes possibles
        all_systems = set()
        for med_data in systems_data.values():
            all_systems.update(med_data.keys())
        all_systems = sorted(list(all_systems))
        
        # Couleurs pour les médicaments
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (medicament, system_scores) in enumerate(systems_data.items()):
            values = [system_scores.get(sys, 0) for sys in all_systems]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=all_systems,
                fill='toself',
                name=medicament,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title="Radar des Effets par Système",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max(system_scores.values()) for system_scores in systems_data.values()] + [1])]
                )
            ),
            showlegend=True,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        return None

def create_side_effects_horizontal_bar_chart(side_effects_data: Dict, stats: Dict) -> Optional[go.Figure]:
    """
    Crée un graphique en barres horizontales avec annotations
    
    Args:
        side_effects_data: Données d'effets secondaires
        stats: Statistiques calculées
        
    Returns:
        Figure Plotly ou None
    """
    try:
        effets_individuels = side_effects_data.get('effets_individuels', [])
        
        if not effets_individuels:
            return None
        
        # Préparer les données pour le graphique
        medications_data = {}
        
        for effet in effets_individuels:
            med = effet.get('medicament', 'Inconnu')
            gravite = effet.get('gravite', 'Faible')
            nb_effets = len(effet.get('effets', []))
            
            if med not in medications_data:
                medications_data[med] = {'Faible': 0, 'Modérée': 0, 'Élevée': 0, 'total': 0}
            
            medications_data[med][gravite] += nb_effets
            medications_data[med]['total'] += nb_effets
        
        # Trier par nombre total d'effets
        sorted_meds = sorted(medications_data.items(), key=lambda x: x[1]['total'], reverse=True)
        
        # Créer le graphique
        fig = go.Figure()
        
        colors = {'Faible': '#28A745', 'Modérée': '#FD7E14', 'Élevée': '#DC3545'}
        
        for gravite in ['Élevée', 'Modérée', 'Faible']:  # Ordre inverse pour l'empilement
            values = [data[gravite] for med, data in sorted_meds]
            medications = [med for med, data in sorted_meds]
            
            fig.add_trace(go.Bar(
                y=medications,
                x=values,
                name=gravite,
                orientation='h',
                marker_color=colors[gravite]
            ))
        
        # Ajouter annotations pour les effets les plus graves
        annotations = []
        for med, data in sorted_meds[:3]:  # Top 3 médicaments
            if data['Élevée'] > 0:
                annotations.append(dict(
                    x=data['total'],
                    y=med,
                    text=f"⚠️ {data['Élevée']} effet(s) grave(s)",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red"
                ))
        
        fig.update_layout(
            title="Nombre d'Effets Secondaires par Médicament",
            xaxis_title="Nombre d'effets secondaires",
            yaxis_title="Médicaments",
            barmode='stack',
            annotations=annotations,
            margin=dict(t=50, b=50, l=150, r=50)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating horizontal bar chart: {e}")
        return None

def display_side_effects_table(side_effects_data: Dict):
    """
    Affiche le tableau détaillé des effets secondaires
    
    Args:
        side_effects_data: Données d'analyse d'effets secondaires
    """
    st.markdown("#### Détail des effets secondaires")
    
    # Préparer les données pour le tableau
    table_data = []
    
    # Ajouter les effets individuels
    for item in side_effects_data.get('effets_individuels', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Effet individuel',
            'Effets': ', '.join(item.get('effets', [])),
            'Gravité': item.get('gravite', 'Faible'),
            'Fréquence': item.get('frequence', 'Non spécifiée'),
            'Système affecté': item.get('systeme_affecte', 'Non spécifié'),
            'Surveillance': item.get('surveillance', 'Standard'),
            'Source': item.get('source', 'Base de connaissances')
        })
    
    # Ajouter les effets cumulés
    for item in side_effects_data.get('effets_cumules', []):
        table_data.append({
            'Médicament': ', '.join(item.get('medicaments', [])),
            'Type': 'Effet cumulé',
            'Effets': item.get('effet_combine', 'Non spécifié'),
            'Gravité': item.get('gravite', 'Faible'),
            'Fréquence': 'Combinaison',
            'Système affecté': 'Multiple',
            'Surveillance': item.get('recommandation', 'Surveillance standard'),
            'Source': 'Analyse combinée'
        })
    
    # Ajouter les risques graves
    for item in side_effects_data.get('risques_graves', []):
        table_data.append({
            'Médicament': item.get('medicament', 'Inconnu'),
            'Type': 'Risque grave',
            'Effets': item.get('effet', 'Non spécifié'),
            'Gravité': 'Élevée',
            'Fréquence': 'Critique',
            'Système affecté': 'Critique',
            'Surveillance': item.get('monitoring', 'Surveillance étroite'),
            'Source': 'Analyse de risque'
        })
    
    if not table_data:
        st.info("Aucun effet secondaire détecté")
        return
    
    # Créer le DataFrame
    df = pd.DataFrame(table_data)
    
    # Afficher le tableau
    st.dataframe(df, use_container_width=True)
    
    # Informations supplémentaires
    st.caption(f"Total: {len(table_data)} effet(s) secondaire(s)")

def display_side_effects_recommendations(side_effects_data: Dict):
    """
    Affiche les recommandations pour les effets secondaires
    
    Args:
        side_effects_data: Données d'analyse d'effets secondaires
    """
    #st.markdown("#### Recommandations")
    
    # Collecter toutes les recommandations
    recommendations = []
    
    # Recommandations pour effets individuels
    for item in side_effects_data.get('effets_individuels', []):
        surveillance = item.get('surveillance')
        if surveillance and surveillance != 'Standard':
            recommendations.append({
                'type': 'Surveillance',
                'medicament': item.get('medicament', 'Inconnu'),
                'recommandation': surveillance,
                'gravite': item.get('gravite', 'Faible')
            })
    
    # Recommandations pour effets cumulés
    for item in side_effects_data.get('effets_cumules', []):
        recommandation = item.get('recommandation')
        if recommandation:
            recommendations.append({
                'type': 'Effet cumulé',
                'medicament': ', '.join(item.get('medicaments', [])),
                'recommandation': recommandation,
                'gravite': item.get('gravite', 'Faible')
            })
    
    # Recommandations pour risques graves
    for item in side_effects_data.get('risques_graves', []):
        monitoring = item.get('monitoring')
        signes_alerte = item.get('signes_alerte')
        if monitoring or signes_alerte:
            rec_text = f"Monitoring: {monitoring}"
            if signes_alerte:
                rec_text += f" | Signes d'alerte: {signes_alerte}"
            recommendations.append({
                'type': 'Risque grave',
                'medicament': item.get('medicament', 'Inconnu'),
                'recommandation': rec_text,
                'gravite': 'Élevée'
            })
    
    if not recommendations:
        st.info("Aucune recommandation spécifique pour les effets secondaires")
        _display_general_side_effects_recommendations()
        return
    
    # Trier par gravité (Élevée en premier)
    severity_order = {'Élevée': 3, 'Modérée': 2, 'Faible': 1}
    recommendations.sort(key=lambda x: severity_order.get(x['gravite'], 0), reverse=True)
    
    # Afficher les recommandations spécifiques
    st.markdown("##### Recommandations spécifiques:")
    for i, rec in enumerate(recommendations, 1):
        gravite = rec['gravite']
        medicament = rec['medicament']
        type_rec = rec['type']
        recommandation = rec['recommandation']
        
        if gravite == 'Élevée':
            st.error(f"**{medicament}** ({type_rec}): {recommandation}")
        elif gravite == 'Modérée':
            st.warning(f"**{medicament}** ({type_rec}): {recommandation}")
        else:
            st.info(f"**{medicament}** ({type_rec}): {recommandation}")
    
    # Recommandations générales
    _display_general_side_effects_recommendations()

def _display_general_side_effects_recommendations():
    """Affiche les recommandations générales pour les effets secondaires"""
    st.markdown("##### Recommandations générales:")
    st.markdown("""
    - **Surveillance clinique** régulière pour détecter l'apparition d'effets secondaires
    - **Information du patient** sur les effets secondaires possibles et signes d'alerte
    - **Monitoring biologique** selon les médicaments (fonction rénale, hépatique, etc.)
    - **Ajustement posologique** si effets secondaires dose-dépendants
    - **Arrêt immédiat** en cas d'effet secondaire grave ou d'allergie
    - **Documentation** de tout effet secondaire dans le dossier patient
    - **Déclaration de pharmacovigilance** pour les effets secondaires graves ou inattendus
    """)

def get_side_effects_summary_for_overview(side_effects_result: Dict) -> Dict:
    """
    Retourne un résumé de l'analyse d'effets secondaires pour la vue d'ensemble
    
    Args:
        side_effects_result: Résultat de l'analyse d'effets secondaires
        
    Returns:
        Dictionnaire avec résumé pour vue d'ensemble
    """
    if not side_effects_result or 'stats' not in side_effects_result:
        return {
            'status': 'no_data',
            'message': 'Pas de données d\'effets secondaires',
            'color': 'secondary',
            'count': 0
        }
    
    stats = side_effects_result['stats']
    total_effects = stats.get('total_side_effects', 0)
    cumulative_effects = stats.get('effets_cumules_count', 0)
    serious_effects = stats.get('risques_graves_count', 0)
    has_critical = stats.get('has_critical_effects', False)
    
    total_all_effects = total_effects + cumulative_effects + serious_effects
    
    if total_all_effects == 0:
        return {
            'status': 'ok',
            'message': 'Aucun effet secondaire significatif',
            'color': 'success',
            'count': 0
        }
    elif has_critical or serious_effects > 0:
        return {
            'status': 'critical',
            'message': f"{total_all_effects} effet(s) - Surveillance nécessaire",
            'color': 'error',
            'count': total_all_effects
        }
    else:
        return {
            'status': 'warning',
            'message': f"{total_all_effects} effet(s) secondaire(s)",
            'color': 'warning',
            'count': total_all_effects
        }

def create_side_effects_metrics_for_overview(side_effects_result: Dict) -> Dict:
    """
    Crée les métriques d'effets secondaires pour la vue d'ensemble globale
    
    Args:
        side_effects_result: Résultat de l'analyse d'effets secondaires
        
    Returns:
        Dictionnaire avec les métriques formatées
    """
    if not side_effects_result or 'stats' not in side_effects_result:
        return {
            'total_effects': 0,
            'cumulative_effects': 0,
            'serious_effects': 0,
            'has_critical': False
        }
    
    stats = side_effects_result['stats']
    
    return {
        'total_effects': stats.get('total_side_effects', 0),
        'cumulative_effects': stats.get('effets_cumules_count', 0),
        'serious_effects': stats.get('risques_graves_count', 0),
        'has_critical': stats.get('has_critical_effects', False)
    }
