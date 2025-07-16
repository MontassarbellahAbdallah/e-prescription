"""
Analyseur d'effets secondaires pour détecter les risques liés aux effets indésirables
"""
import json
import re
from typing import List, Dict, Tuple, Optional
from config.logging_config import get_logger
from core.key_manager import get_key_manager
from core.cache_manager import get_cache_manager
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from utils.constants import PROMPT_TEMPLATES
from utils.helpers import measure_execution_time

logger = get_logger(__name__)

class SideEffectsAnalyzer:
    """
    Analyseur spécialisé pour les effets secondaires médicamenteux
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialise l'analyseur d'effets secondaires
        
        Args:
            use_cache: Utiliser le cache pour les analyses
        """
        self.key_manager = get_key_manager()
        self.cache_manager = get_cache_manager() if use_cache else None
        self.model = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
    
    @property
    def cached_invoke(self):
        """Wrapper pour les appels LLM avec gestion de quota"""
        return self.key_manager.wrap_quota(self.model.invoke)
    
    @measure_execution_time
    def analyze_side_effects(self, prescription: str, patient_info: str = "", context_docs: List[str] = None) -> Dict:
        """
        Analyse les effets secondaires de la prescription
        
        Args:
            prescription: Texte de la prescription
            patient_info: Informations sur le patient (âge, pathologies, etc.)
            context_docs: Documents de contexte de la base vectorielle
            
        Returns:
            Dictionnaire avec analyse d'effets secondaires structurée
        """
        # Créer une clé de cache
        cache_key = f"side_effects_{prescription}_{patient_info}_{len(context_docs or [])}"
        
        # Vérifier le cache
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key, prefix="side_effects_")
            if cached_result is not None:
                logger.info("Side effects analysis cache hit")
                return cached_result
        
        try:
            # Préparer le contexte
            context = self._prepare_context(context_docs or [])
            
            # Préparer le prompt
            prompt = PROMPT_TEMPLATES['side_effects_analysis'].format(
                prescription=prescription,
                patient_info=patient_info or "Informations patient non spécifiées",
                context=context
            )
            
            # Appeler le LLM
            response = self.cached_invoke(prompt)
            content = response.content.strip()
            
            # Parser la réponse JSON
            side_effects_data = self._parse_side_effects_response(content)
            
            # Calculer les statistiques
            stats = self._calculate_side_effects_stats(side_effects_data)
            
            result = {
                'side_effects_analysis': side_effects_data,
                'stats': stats,
                'raw_response': content,
                'context_used': len(context_docs or []) > 0
            }
            
            # Mettre en cache
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, prefix="side_effects_")
            
            logger.info(f"Side effects analysis completed: {stats['total_side_effects']} effects found")
            return result
            
        except Exception as e:
            logger.error(f"Side effects analysis failed: {e}")
            return self._get_empty_side_effects_result(str(e))
    
    def _prepare_context(self, context_docs: List[str]) -> str:
        """
        Prépare le contexte à partir des documents
        
        Args:
            context_docs: Liste des documents de contexte
            
        Returns:
            Contexte formaté
        """
        if not context_docs:
            return "Aucune documentation spécifique disponible. Base-toi sur tes connaissances médicales générales pour les effets secondaires."
        
        context = "Documentation médicale de référence sur les effets secondaires:\n\n"
        for i, doc in enumerate(context_docs[:10], 1):
            context += f"Source {i}:\n{doc}\n\n"
        
        return context
    
    def _parse_side_effects_response(self, content: str) -> Dict:
        """
        Parse la réponse JSON du LLM
        
        Args:
            content: Contenu de la réponse
            
        Returns:
            Données d'effets secondaires parsées
        """
        try:
            # Chercher le JSON dans la réponse
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                if 'side_effects_analysis' in data:
                    return data['side_effects_analysis']
                else:
                    return data
            else:
                logger.warning("No JSON found in side effects response")
                return self._get_empty_side_effects_analysis()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in side effects response: {e}")
            return self._manual_parse_side_effects(content)
        except Exception as e:
            logger.error(f"Error parsing side effects response: {e}")
            return self._get_empty_side_effects_analysis()
    
    def _manual_parse_side_effects(self, content: str) -> Dict:
        """
        Parse manuel si JSON échoue
        
        Args:
            content: Contenu à parser
            
        Returns:
            Données d'effets secondaires parsées manuellement
        """
        result = {
            'effets_individuels': [],
            'effets_cumules': [],
            'risques_graves': []
        }
        
        # Parse basique pour extraire les informations d'effets secondaires
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip().lower()
            
            if any(keyword in line for keyword in ['effet individuel', 'effets individuels']):
                current_section = 'effets_individuels'
            elif any(keyword in line for keyword in ['effet cumulé', 'effets cumulés']):
                current_section = 'effets_cumules'
            elif any(keyword in line for keyword in ['risque grave', 'risques graves']):
                current_section = 'risques_graves'
        
        logger.warning("Used manual parsing for side effects analysis")
        return result
    
    def _calculate_side_effects_stats(self, side_effects_data: Dict) -> Dict:
        """
        Calcule les statistiques d'effets secondaires
        
        Args:
            side_effects_data: Données d'effets secondaires analysées
            
        Returns:
            Statistiques calculées
        """
        stats = {
            'total_medications': 0,
            'total_side_effects': 0,
            'effets_par_gravite': {'Faible': 0, 'Modérée': 0, 'Élevée': 0},
            'effets_cumules_count': len(side_effects_data.get('effets_cumules', [])),
            'risques_graves_count': len(side_effects_data.get('risques_graves', [])),
            'has_critical_effects': False,
            'systemes_affectes': {}
        }
        
        # Analyser les effets individuels
        effets_individuels = side_effects_data.get('effets_individuels', [])
        medications = set()
        
        for effet in effets_individuels:
            # Compter les médicaments
            medicament = effet.get('medicament', 'Inconnu')
            medications.add(medicament)
            
            # Compter les effets par gravité
            gravite = effet.get('gravite', 'Faible')
            if gravite in stats['effets_par_gravite']:
                stats['effets_par_gravite'][gravite] += 1
            
            # Détecter les effets critiques
            if gravite == 'Élevée':
                stats['has_critical_effects'] = True
            
            # Compter les systèmes affectés
            systeme = effet.get('systeme_affecte', 'Non spécifié')
            if systeme in stats['systemes_affectes']:
                stats['systemes_affectes'][systeme] += 1
            else:
                stats['systemes_affectes'][systeme] = 1
        
        # Vérifier les risques graves
        if stats['risques_graves_count'] > 0:
            stats['has_critical_effects'] = True
        
        # Totaux
        stats['total_medications'] = len(medications)
        stats['total_side_effects'] = len(effets_individuels)
        
        return stats
    
    def _get_empty_side_effects_analysis(self) -> Dict:
        """Retourne une structure d'effets secondaires vide"""
        return {
            'effets_individuels': [],
            'effets_cumules': [],
            'risques_graves': []
        }
    
    def _get_empty_side_effects_result(self, error_msg: str = "") -> Dict:
        """Retourne un résultat d'effets secondaires vide avec erreur"""
        return {
            'side_effects_analysis': self._get_empty_side_effects_analysis(),
            'stats': {
                'total_medications': 0,
                'total_side_effects': 0,
                'effets_par_gravite': {'Faible': 0, 'Modérée': 0, 'Élevée': 0},
                'effets_cumules_count': 0,
                'risques_graves_count': 0,
                'has_critical_effects': False,
                'systemes_affectes': {},
                'error': error_msg
            },
            'raw_response': "",
            'context_used': False
        }
    
    def format_side_effects_for_display(self, side_effects_analysis: Dict) -> List[Dict]:
        """
        Formate les données d'effets secondaires pour l'affichage dans les tableaux
        
        Args:
            side_effects_analysis: Données d'analyse d'effets secondaires
            
        Returns:
            Liste de dictionnaires formatés pour les tableaux
        """
        formatted_data = []
        
        # Traiter les effets individuels
        for item in side_effects_analysis.get('effets_individuels', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Effet individuel',
                'Effets': ', '.join(item.get('effets', [])),
                'Gravité': item.get('gravite', 'Faible'),
                'Fréquence': item.get('frequence', 'Non spécifiée'),
                'Système affecté': item.get('systeme_affecte', 'Non spécifié'),
                'Surveillance': item.get('surveillance', 'Standard'),
                'Source': item.get('source', 'Base de connaissances'),
                'Couleur': self._get_severity_color(item.get('gravite', 'Faible'))
            })
        
        # Traiter les effets cumulés
        for item in side_effects_analysis.get('effets_cumules', []):
            formatted_data.append({
                'Médicament': ', '.join(item.get('medicaments', [])),
                'Type': 'Effet cumulé',
                'Effets': item.get('effet_combine', 'Non spécifié'),
                'Gravité': item.get('gravite', 'Faible'),
                'Fréquence': 'Combinaison',
                'Système affecté': 'Multiple',
                'Surveillance': item.get('recommandation', 'Surveillance standard'),
                'Source': 'Analyse combinée',
                'Couleur': self._get_severity_color(item.get('gravite', 'Faible'))
            })
        
        # Traiter les risques graves
        for item in side_effects_analysis.get('risques_graves', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Risque grave',
                'Effets': item.get('effet', 'Non spécifié'),
                'Gravité': 'Élevée',
                'Fréquence': 'Critique',
                'Système affecté': 'Critique',
                'Surveillance': item.get('monitoring', 'Surveillance étroite'),
                'Source': 'Analyse de risque',
                'Couleur': self._get_severity_color('Élevée')
            })
        
        return formatted_data
    
    def _get_severity_color(self, gravite: str) -> str:
        """
        Retourne la couleur associée à la gravité
        
        Args:
            gravite: Niveau de gravité
            
        Returns:
            Code couleur hexadécimal
        """
        colors = {
            'Faible': '#28A745',    # Vert
            'Modérée': '#FD7E14',   # Orange  
            'Élevée': '#DC3545'     # Rouge
        }
        return colors.get(gravite, '#6C757D')  # Gris par défaut
