"""
Page principale d'analyse des interactions médicamenteuses
"""
import streamlit as st
import time
from typing import List, Dict, Optional
from datetime import datetime

from config.logging_config import get_logger
from ai.llm_analyzer import LLMAnalyzer
from data.rag_processor import RAGProcessor
from data.validators import DataValidator
from ui.components.tables import display_interactions_table
from ui.components.charts import display_interaction_charts, display_statistics_charts
# from ui.components.export import create_export_section  # Supprimé
from ui.styles import (
    create_main_header, create_status_message, create_progress_bar,
    create_metric_card, create_interaction_card
)
from utils.helpers import estimate_analysis_time
from ui.components.dosage_components import (
    display_dosage_analysis_section, 
    get_dosage_summary_for_overview
)
from ui.components.contraindication_components import (
    display_contraindication_analysis_section,
    get_contraindication_summary_for_overview
)
from ui.components.redundancy_components import (
    display_redundancy_analysis_section,
    get_redundancy_summary_for_overview
)

logger = get_logger(__name__)

class AnalysisPage:
    """Page d'analyse des interactions médicamenteuses"""
    
    def __init__(self):
        self.llm_analyzer = None
        self.rag_processor = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialise les composants nécessaires"""
        try:
            # Vérifier si les composants sont déjà initialisés dans la session
            if 'llm_analyzer' in st.session_state:
                self.llm_analyzer = st.session_state.llm_analyzer
            else:
                self.llm_analyzer = LLMAnalyzer(use_cache=True)
                st.session_state.llm_analyzer = self.llm_analyzer
            
            if 'rag_processor' in st.session_state:
                self.rag_processor = st.session_state.rag_processor
            else:
                self.rag_processor = RAGProcessor(cache_enabled=True)
                # Essayer de charger le vector store existant
                self.rag_processor.load_vector_store()
                st.session_state.rag_processor = self.rag_processor
                
        except Exception as e:
            logger.error(f"Error initializing analysis page components: {e}")
            st.error(f"Erreur d'initialisation: {e}")
    
    def render(self):
        """Affiche la page d'analyse de prescription"""
        
        # Zone de saisie principale
        self._render_input_section()
        
        # Zone de résultats si une analyse a été effectuée
        if 'current_analysis' in st.session_state:
            self._render_results_section()
    
    def _render_input_section(self):
        """Affiche la section de saisie"""
        st.markdown("### Analysez votre prescription médicale")
        
        # Zone de texte pour la question
        user_question = st.text_area(
            "Prescription à analyser:",
            placeholder="Copiez-collez votre prescription ici...",
            height=300,
            max_chars=20000,  # Limitation à 20 000 caractères
            help="Copiez-collez votre prescription complète - L'IA extraira automatiquement les molécules actives et analysera les interactions (max 20 000 caractères)"
        )
        
        # Affichage du nombre de caractères
        char_count = len(user_question) if user_question else 0
        if char_count > 8000:
            st.warning(f"{char_count}/10 000 caractères - Prescription très longue, considérez la diviser")
        elif char_count > 5000:
            st.info(f"{char_count}/10 000 caractères - Prescription longue")
        elif char_count > 0:
            st.caption(f"{char_count}/10 000 caractères")
        
        # Options d'analyse
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            analyze_button = st.button(
                "Analyser la prescription", 
                type="primary",
                disabled=not user_question.strip(),
                help="Lancer l'analyse complète de la prescription médicale"
            )
        
        with col2:
            use_context = st.checkbox(
                "Guidelines médicales", 
                value=True,  # Déjà activé
                help="Utiliser les guidelines et documents de référence médicaux pour enrichir l'analyse"
            )
        
        with col3:
            detailed_analysis = st.checkbox(
                "Rapport détaillé",
                value=False,  # Déjà activé
                help="Générer un rapport détaillé avec recommandations et sources"
            )
        
        # Traitement de la demande d'analyse
        if analyze_button and user_question.strip():
            self._process_analysis_request(user_question, use_context, detailed_analysis)
    
    def _process_analysis_request(self, question: str, use_context: bool, detailed_analysis: bool):
        """Traite une demande d'analyse"""
        try:
            # Validation de la question
            is_valid, cleaned_question = DataValidator.validate_search_query(question)
            if not is_valid:
                st.error("Question invalide. Veuillez saisir au moins 3 caractères.")
                return
            
            # Étape 1: Extraction des médicaments
            with st.spinner("Extraction des médicaments en cours..."):
                drugs = self.llm_analyzer.extract_drug_names(cleaned_question)
            
            if not drugs:
                create_status_message(
                    "Aucun médicament identifié dans votre question. "
                    "Veuillez reformuler en mentionnant explicitement les médicaments.",
                    "warning"
                )
                return
            
            # Afficher les médicaments trouvés
            st.success(f"Médicaments identifiés: {', '.join(drugs)}")
            
            # Étape 2: Recherche de contexte documentaire si demandé
            context_docs = []
            sources_info = []
            if use_context and self.rag_processor.vector_store:
                with st.spinner("Recherche du contexte documentaire..."):
                    context_docs, sources_info = self.rag_processor.search_with_detailed_sources(
                        cleaned_question, k=5
                    )
            
            # Étape 3: Analyse des interactions
            if len(drugs) >= 2:
                self._analyze_drug_interactions(drugs, context_docs, cleaned_question, detailed_analysis, sources_info)
            elif len(drugs) == 1:
                self._handle_single_drug(drugs[0], use_context, cleaned_question, sources_info)
            
        except Exception as e:
            logger.error(f"Error processing analysis request: {e}")
            st.error(f"Erreur lors de l'analyse: {e}")
    
    def _analyze_drug_interactions(self, drugs: List[str], context_docs: List, question: str, detailed_analysis: bool, sources_info: List[Dict]):
        """Analyse les interactions entre les médicaments ET le dosage"""
        # Estimation du temps (maintenant pour interactions + dosage)
        combinations_count = len(drugs) * (len(drugs) - 1) // 2
        estimated_time = estimate_analysis_time(combinations_count)
        
        # En-tête d'analyse avec informations
        st.markdown("### Analyse complète en cours")
        st.info(f"Analyse de {combinations_count} combinaisons + dosages (temps estimé: {estimated_time})")
        
        # Conteneur pour la barre de progression
        progress_container = st.empty()
        
        # NOUVEAU: Utiliser l'analyse complète au lieu de juste les interactions
        with st.spinner("Analyse des interactions et dosages en cours..."):
            try:
                # Analyse complète (interactions + dosage + contre-indications + redondances)
                # context_docs est passé pour enrichir TOUTES les analyses
                complete_result = self.llm_analyzer.analyze_prescription_complete(question, context_docs)
                
                # Nettoyer la barre de progression
                progress_container.empty()
                
                if complete_result:
                    # Sauvegarder les résultats dans la session avec le nouveau format
                    analysis_data = {
                        'timestamp': datetime.now(),
                        'question': question,
                        'drugs': complete_result['drugs'],
                        'patient_info': complete_result['patient_info'],
                        'interactions': complete_result.get('interactions'),
                        'dosage': complete_result.get('dosage'),
                        'contraindications': complete_result.get('contraindications'),
                        'redundancy': complete_result.get('redundancy'),  # NOUVEAU: Ajout redondance
                        'context_used': len(context_docs) > 0,
                        'analysis_type': 'complete'
                    }
                    
                    st.session_state.current_analysis = analysis_data
                    
                    # Ajouter à l'historique
                    if 'analysis_history' not in st.session_state:
                        st.session_state.analysis_history = []
                    st.session_state.analysis_history.append(analysis_data)
                    
                    # Mémoriser si une explication détaillée est demandée
                    if detailed_analysis and context_docs:
                        st.session_state.detailed_explanation_needed = {
                            'question': question,
                            'context_docs': context_docs,
                            'sources_info': sources_info
                        }
                else:
                    st.error("Aucune analyse effectuée")
                    
            except Exception as e:
                progress_container.empty()
                logger.error(f"Error in complete analysis: {e}")
                st.error(f"Erreur lors de l'analyse complète: {e}")

                
    
    def _handle_single_drug(self, drug: str, use_context: bool, question: str, sources_info: List[Dict]):
        """Gère le cas d'un seul médicament"""
        st.warning(
            f"Un seul médicament identifié: {drug}. "
            "L'analyse d'interactions nécessite au moins 2 médicaments."
        )
        
        # Recherche documentaire sur le médicament unique
        if use_context and self.rag_processor.vector_store and sources_info:
            with st.spinner("Recherche d'informations..."):
                # Utiliser les sources déjà récupérées
                if sources_info:
                    st.info(f"Informations trouvées sur {drug} dans les guidelines médicales")
    
    def _calculate_global_risk_score(self, analysis: Dict) -> Dict:
        """Calcule un score de risque global basé sur toutes les analyses"""
        
        risk_factors = []
        
        # Facteurs d'interaction
        interactions_data = analysis.get('interactions')
        if interactions_data:
            stats = interactions_data['stats']
            if stats['major'] > 0:
                risk_factors.append('interactions_major')
            elif stats['moderate'] > 0:
                risk_factors.append('interactions_moderate')
        
        # Facteurs de dosage
        dosage_data = analysis.get('dosage')
        if dosage_data:
            if dosage_data['stats']['has_critical_issues']:
                risk_factors.append('dosage_critical')
            elif dosage_data['stats']['total_issues'] > 0:
                risk_factors.append('dosage_moderate')
        
        # Facteurs de contre-indications
        contraindication_data = analysis.get('contraindications')
        if contraindication_data:
            if contraindication_data['stats']['has_critical_contraindications']:
                risk_factors.append('contraindication_critical')
            elif contraindication_data['stats']['total_contraindications'] > 0:
                risk_factors.append('contraindication_moderate')
        
        # NOUVEAU: Facteurs de redondance thérapeutique
        redundancy_data = analysis.get('redundancy')
        if redundancy_data:
            if redundancy_data['stats']['has_critical_redundancies']:
                risk_factors.append('redundancy_critical')
            elif redundancy_data['stats']['total_redundancies'] > 0:
                risk_factors.append('redundancy_moderate')
        
        # Évaluation globale
        if ('interactions_major' in risk_factors or 
            'dosage_critical' in risk_factors or 
            'contraindication_critical' in risk_factors or
            'redundancy_critical' in risk_factors):
            return {
                'level': 'ÉLEVÉ',
                'description': 'Révision urgente',
                'color': 'error'
            }
        elif ('interactions_moderate' in risk_factors or 
              'dosage_moderate' in risk_factors or 
              'contraindication_moderate' in risk_factors or
              'redundancy_moderate' in risk_factors):
            return {
                'level': 'MODÉRÉ', 
                'description': 'Surveillance requise',
                'color': 'warning'
            }
        else:
            return {
                'level': 'FAIBLE',
                'description': 'Prescription acceptable',
                'color': 'success'
            }
        
    def _render_global_overview(self, analysis: Dict):
        """Affiche la vue d'ensemble globale avec toutes les métriques"""
        st.markdown("#### 📊 Vue d'ensemble de l'analyse")
        
        # Métriques globales - 6 colonnes principales + 1 pour risque global
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            create_metric_card("Médicaments", str(len(analysis['drugs'])))
        
        with col2:
            # Métriques interactions
            interactions_data = analysis.get('interactions')
            if interactions_data:
                major_count = interactions_data['stats']['major']
                create_metric_card("Interactions Major", str(major_count))
            else:
                create_metric_card("Interactions Major", "N/A")
        
        with col3:
            # Métriques dosage
            dosage_data = analysis.get('dosage')
            if dosage_data:
                dosage_issues = dosage_data['stats']['total_issues']
                create_metric_card("Problèmes dosage", str(dosage_issues))
            else:
                create_metric_card("Problèmes dosage", "N/A")
        
        with col4:
            # Métriques contre-indications
            contraindication_data = analysis.get('contraindications')
            if contraindication_data:
                contraindication_count = contraindication_data['stats']['total_contraindications']
                create_metric_card("Contre-indications", str(contraindication_count))
            else:
                create_metric_card("Contre-indications", "N/A")
        
        with col5:
            # NOUVEAU: Métriques redondances
            redundancy_data = analysis.get('redundancy')
            if redundancy_data:
                redundancy_count = redundancy_data['stats']['total_redundancies']
                create_metric_card("Redondances", str(redundancy_count))
            else:
                create_metric_card("Redondances", "N/A")
        
        with col6:
            # Score de risque global
            risk_score = self._calculate_global_risk_score(analysis)
            create_metric_card("Risque global", risk_score['level'])
    
    def _render_prescription_evaluation(self, analysis: Dict):
        """Affiche l'évaluation globale de la prescription"""
        st.markdown("---")
        st.markdown("### 🏥 Évaluation globale de la prescription")
        
        risk_score = self._calculate_global_risk_score(analysis)
        
        if risk_score['level'] == 'ÉLEVÉ':
            st.error(f"🔴 **PRESCRIPTION À RISQUE ÉLEVÉ** - {risk_score['description']}")
        elif risk_score['level'] == 'MODÉRÉ':
            st.warning(f"🟡 **PRESCRIPTION À RISQUE MODÉRÉ** - {risk_score['description']}")
        else:
            st.success(f"🟢 **PRESCRIPTION À RISQUE FAIBLE** - {risk_score['description']}")
        
        # Recommandations générales
        st.markdown("#### Recommandations générales:")
        
        recommendations = []
        
        # Recommandations basées sur les interactions
        interactions_data = analysis.get('interactions')
        if interactions_data and interactions_data['stats']['major'] > 0:
            recommendations.append("⚠️ **Interactions majeures détectées** - Éviter les associations ou surveillance très étroite")
        
        # Recommandations basées sur le dosage
        dosage_data = analysis.get('dosage')
        if dosage_data and dosage_data['stats']['has_critical_issues']:
            recommendations.append("⚠️ **Problèmes de dosage critiques** - Ajustements immédiats nécessaires")
        
        # Recommandations basées sur les contre-indications
        contraindication_data = analysis.get('contraindications')
        if contraindication_data and contraindication_data['stats']['has_critical_contraindications']:
            recommendations.append("🚨 **Contre-indications absolues détectées** - Arrêt immédiat des médicaments concernés")
        elif contraindication_data and contraindication_data['stats']['total_contraindications'] > 0:
            recommendations.append("⚠️ **Contre-indications relatives** - Surveillance renforcée nécessaire")
        
        # NOUVEAU: Recommandations basées sur les redondances
        redundancy_data = analysis.get('redundancy')
        if redundancy_data and redundancy_data['stats']['has_critical_redundancies']:
            recommendations.append("🚨 **Redondances critiques détectées** - Optimisation urgente de la prescription")
        elif redundancy_data and redundancy_data['stats']['total_redundancies'] > 0:
            recommendations.append("💡 **Redondances détectées** - Optimisation thérapeutique recommandée")
        
        # Recommandations générales
        recommendations.extend([
            "📋 **Révision pharmaceutique** recommandée dans les 24h",
            "👨‍⚕️ **Consultation médicale** si modification nécessaire", 
            "📊 **Surveillance clinique** renforcée pendant 48-72h",
            "🏥 **Documentation** de toute modification dans le dossier patient"
        ])
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    def _render_complete_analysis_results(self, analysis: Dict):
        """Affiche les résultats d'une analyse complète"""
        
        # 1. Vue d'ensemble globale avec toutes les métriques
        self._render_global_overview(analysis)
        
        # 2. Organisation par onglets pour chaque section
        tab1, tab2, tab3, tab4 = st.tabs(["Interactions médicamenteuses", "Dosage", "Contre-indications", "Redondance thérapeutique"])
        
        with tab1:
            # Section interactions (existante)
            interactions_data = analysis.get('interactions')
            if interactions_data:
                interactions = interactions_data['interactions']
                stats = interactions_data['stats']
                
                # Métriques interactions
                self._render_metrics(stats)
                
                # Graphiques interactions
                st.subheader("Visualisations des interactions")
                display_interaction_charts(interactions)
                
                # Tableau interactions
                display_interactions_table(interactions, enable_filters=True)
            else:
                st.info("Analyse des interactions non disponible (moins de 2 médicaments)")
        
        with tab2:
            # Section dosage
            dosage_data = analysis.get('dosage')
            if dosage_data:
                display_dosage_analysis_section(dosage_data)
            else:
                st.warning("Données de dosage non disponibles")
        
        with tab3:
            # Section contre-indications
            contraindication_data = analysis.get('contraindications')
            if contraindication_data:
                display_contraindication_analysis_section(contraindication_data)
            else:
                st.warning("Données de contre-indications non disponibles")
        
        with tab4:
            # NOUVELLE SECTION: Redondance thérapeutique
            redundancy_data = analysis.get('redundancy')
            if redundancy_data:
                display_redundancy_analysis_section(redundancy_data)
            else:
                st.warning("Données de redondance thérapeutique non disponibles")
        
        # 3. Évaluation globale de la prescription
        self._render_prescription_evaluation(analysis)
        
        # 4. Explication détaillée à la fin si demandée
        if 'detailed_explanation_needed' in st.session_state:
            explanation_data = st.session_state.detailed_explanation_needed
            
            st.markdown("### Rapport détaillé")
            
            with st.spinner("Génération du rapport détaillé..."):
                detailed_explanation = self.llm_analyzer.get_detailed_explanation_with_sources(
                    explanation_data['question'],
                    explanation_data['context_docs'],
                    explanation_data.get('sources_info', [])
                )
            
            st.markdown(detailed_explanation)
            
            del st.session_state.detailed_explanation_needed
    
    def _render_legacy_analysis_results(self, analysis: Dict):
        """Affiche les résultats d'une analyse legacy (compatibilité)"""
        # Code existant pour les analyses qui n'ont que les interactions
        interactions = analysis['interactions']
        stats = analysis['stats']
        
        # Évaluation du risque (code existant)
        major_count = stats['major']
        moderate_count = stats['moderate']

        if major_count > 0 or moderate_count > 0:
            st.error("🔴 **PRESCRIPTION PORTEUSE DE RISQUE**")
            if major_count > 0:
                st.write(f"⚠️ {major_count} interaction(s) majeure(s) détectée(s)")
            if moderate_count > 0:
                st.write(f"⚠️ {moderate_count} interaction(s) modérée(s) détectée(s)")
        else:
            st.success("✅ **PRESCRIPTION SAINE**")
            st.write("Aucune interaction majeure ou modérée détectée")

        st.markdown("---")
        
        # Métriques principales (code existant)
        self._render_metrics(stats)
        
        # Graphiques (code existant)
        st.subheader("Visualisations")
        display_interaction_charts(interactions)
        
        # Tableau des interactions (code existant)
        display_interactions_table(interactions, enable_filters=True)
    
    def _render_results_section(self):
        """Affiche la section des résultats avec toutes les analyses"""
        if 'current_analysis' not in st.session_state:
            return
        
        analysis = st.session_state.current_analysis
        
        st.markdown("---")
        st.subheader("Résultats de l'analyse")
        
        # Vérifier le type d'analyse
        analysis_type = analysis.get('analysis_type', 'legacy')
        
        if analysis_type == 'complete':
            # NOUVELLE LOGIQUE: Analyse complète avec interactions + dosage + contre-indications + redondances
            self._render_complete_analysis_results(analysis)
        else:
            # ANCIENNE LOGIQUE: Compatibilité avec les analyses existantes
            self._render_legacy_analysis_results(analysis)
    
    def _render_metrics(self, stats: Dict):
        """Affiche les métriques principales"""
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            create_metric_card("Médicaments", str(stats['total_drugs']))
        
        with col2:
            create_metric_card("Combinaisons", str(stats['total_combinations']))
        
        with col3:
            major_count = stats['major']
            create_metric_card("Major", str(major_count))
        
        with col4:
            create_metric_card("Moderate", str(stats['moderate']))
        
        with col5:
            create_metric_card("Minor", str(stats['minor']))
        
        with col6:
            create_metric_card("Aucune", str(stats.get('aucune', 0)))
        
        # Métriques supplémentaires
        if 'analysis_time' in stats:
            col1, col2 = st.columns(2)
            with col1:
                create_metric_card("Temps d'analyse", f"{stats['analysis_time']:.1f}s")
            with col2:
                avg_time = stats.get('avg_time_per_combination', 0)
                create_metric_card("Temps moyen/combinaison", f"{avg_time:.2f}s")

def render_analysis_page():
    """Fonction utilitaire pour afficher la page d'analyse"""
    page = AnalysisPage()
    page.render()

# Interface de compatibilité pour l'importation
analysis_page = AnalysisPage()
