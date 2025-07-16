"""
Composants de graphiques pour la visualisation des données d'interactions
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any
from utils.constants import LEVEL_COLORS, PLOTLY_CONFIG

class InteractionCharts:
    """Gestionnaire de graphiques pour les interactions médicamenteuses"""
    
    def __init__(self, interactions: List[Dict]):
        """
        Initialise le gestionnaire de graphiques
        
        Args:
            interactions: Liste des interactions à visualiser
        """
        self.interactions = interactions
        self.df = pd.DataFrame(interactions) if interactions else pd.DataFrame()
    
    def create_level_distribution_pie(self, title: str = "Distribution des niveaux d'interaction") -> go.Figure:
        """
        Crée un graphique en secteurs de la distribution des niveaux
        
        Args:
            title: Titre du graphique
            
        Returns:
            Figure Plotly
        """
        if self.df.empty:
            return self._create_empty_chart("Aucune donnée disponible")
        
        # Compter les niveaux existants
        level_counts = self.df['Level'].value_counts()
        
        # S'assurer que toutes les classes sont représentées
        all_levels = ['Major', 'Moderate', 'Minor', 'Aucune']
        complete_counts = {}
        
        for level in all_levels:
            complete_counts[level] = level_counts.get(level, 0)
        
        # Filtrer les niveaux avec des valeurs > 0 pour l'affichage
        display_levels = {k: v for k, v in complete_counts.items() if v > 0}
        
        if not display_levels:
            return self._create_empty_chart("Aucune interaction détectée")
        
        # Préparer les données
        labels = list(display_levels.keys())
        values = list(display_levels.values())
        colors = [LEVEL_COLORS.get(level, '#6C757D') for level in labels]
        
        # Créer le graphique
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent+value',
            texttemplate='<b>%{label}</b><br>%{value} (%{percent})',
            hovertemplate='<b>%{label}</b><br>Nombre: %{value}<br>Pourcentage: %{percent}<extra></extra>',
            hole=0.4  # Donut chart
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2E86AB'}
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            height=400,
            margin=dict(t=80, b=80, l=20, r=20),
            font=dict(size=12)
        )
        
        return fig
    
    def create_level_distribution_bar(self, title: str = "Nombre d'interactions par niveau") -> go.Figure:
        """
        Crée un graphique en barres de la distribution des niveaux
        
        Args:
            title: Titre du graphique
            
        Returns:
            Figure Plotly
        """
        if self.df.empty:
            return self._create_empty_chart("Aucune donnée disponible")
        
        # Compter les niveaux existants
        level_counts = self.df['Level'].value_counts()
        
        # Ordonner selon la gravité et inclure toutes les classes
        level_order = ['Major', 'Moderate', 'Minor', 'Aucune']
        
        # Créer un dictionnaire complet avec toutes les classes
        complete_counts = {}
        for level in level_order:
            complete_counts[level] = level_counts.get(level, 0)
        
        # Préparer les données pour l'affichage (toutes les classes, même avec 0)
        ordered_levels = level_order
        ordered_counts = [complete_counts[level] for level in ordered_levels]
        colors = [LEVEL_COLORS.get(level, '#6C757D') for level in ordered_levels]
        
        # Créer le graphique
        fig = go.Figure(data=[
            go.Bar(
                x=ordered_levels,
                y=ordered_counts,
                marker_color=colors,
                text=ordered_counts,
                textposition='auto',
                texttemplate='<b>%{text}</b>',
                hovertemplate='<b>%{x}</b><br>Nombre: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2E86AB'}
            },
            xaxis_title="Niveau d'interaction",
            yaxis_title="Nombre d'interactions",
            showlegend=False,
            height=400,
            margin=dict(t=80, b=80, l=60, r=20),
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_drug_frequency_chart(self, top_n: int = 15) -> go.Figure:
        """
        Crée un graphique de fréquence des médicaments
        
        Args:
            top_n: Nombre de médicaments à afficher
            
        Returns:
            Figure Plotly
        """
        if self.df.empty:
            return self._create_empty_chart("Aucune donnée disponible")
        
        # Compter les occurrences de chaque médicament
        all_drugs = list(self.df['Drug1']) + list(self.df['Drug2'])
        drug_counts = pd.Series(all_drugs).value_counts().head(top_n)
        
        # Créer le graphique horizontal
        fig = go.Figure(data=[
            go.Bar(
                x=drug_counts.values,
                y=drug_counts.index,
                orientation='h',
                marker_color='#667eea',
                text=drug_counts.values,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Interactions: %{x}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': f"Top {top_n} médicaments par nombre d'interactions",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2E86AB'}
            },
            xaxis_title="Nombre d'interactions",
            yaxis_title="Médicaments",
            height=max(400, top_n * 25),
            margin=dict(t=80, b=80, l=150, r=20),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Crée un graphique vide avec un message
        
        Args:
            message: Message à afficher
            
        Returns:
            Figure Plotly vide
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray"),
            showarrow=False
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        
        return fig

class StatisticsCharts:
    """Gestionnaire de graphiques pour les statistiques générales"""
    
    def __init__(self, stats: Dict[str, Any]):
        """
        Initialise le gestionnaire de graphiques statistiques
        
        Args:
            stats: Dictionnaire des statistiques
        """
        self.stats = stats
    
    def create_analysis_summary_chart(self) -> go.Figure:
        """
        Crée un graphique de résumé d'analyse
        
        Returns:
            Figure Plotly
        """
        if not self.stats:
            return self._create_empty_chart("Aucune statistique disponible")
        
        # Métriques principales
        metrics = {
            'Médicaments analysés': self.stats.get('total_drugs', 0),
            'Combinaisons testées': self.stats.get('total_combinations', 0),
            'Interactions Major': self.stats.get('major', 0),
            'Interactions Moderate': self.stats.get('moderate', 0),
            'Interactions Minor': self.stats.get('minor', 0)
        }
        
        # Couleurs pour chaque métrique
        colors = ['#667eea', '#764ba2', '#dc3545', '#ffc107', '#28a745']
        
        # Créer le graphique en barres
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=colors,
                text=list(metrics.values()),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Valeur: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': "Résumé de l'analyse",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2E86AB'}
            },
            xaxis_title="Métriques",
            yaxis_title="Valeurs",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Crée un graphique vide avec un message
        
        Args:
            message: Message à afficher
            
        Returns:
            Figure Plotly vide
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray"),
            showarrow=False
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        
        return fig

# Fonctions utilitaires pour les graphiques
def display_interaction_charts(interactions: List[Dict]):
    """
    Affiche les graphiques d'interactions
    
    Args:
        interactions: Liste des interactions
    """
    if not interactions:
        st.warning("📈 Aucune donnée d'interaction pour les graphiques")
        return
    
    charts = InteractionCharts(interactions)
    
    # Graphiques en colonnes
    with st.container():
        st.plotly_chart(
            charts.create_level_distribution_bar(),
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

def display_statistics_charts(stats: Dict[str, Any]):
    """
    Affiche les graphiques de statistiques
    
    Args:
        stats: Dictionnaire des statistiques
    """
    if not stats:
        st.warning("📈 Aucune statistique pour les graphiques")
        return
    
    charts = StatisticsCharts(stats)
    
    st.plotly_chart(
        charts.create_analysis_summary_chart(),
        use_container_width=True,
        config=PLOTLY_CONFIG
    )
