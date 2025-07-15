"""
Analyseur de voies d'administration pour détecter les problèmes liés aux modes d'administration des médicaments
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

class AdministrationRouteAnalyzer:
    """
    Analyseur spécialisé pour les problèmes liés aux voies d'administration
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialise l'analyseur de voies d'administration
        
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
    def analyze_administration_routes(self, prescription: str, patient_info: str = "", context_docs: List[str] = None) -> Dict:
        """
        Analyse les voies d'administration de la prescription
        
        Args:
            prescription: Texte de la prescription
            patient_info: Informations sur le patient (âge, poids, etc.)
            context_docs: Documents de contexte de la base vectorielle
            
        Returns:
            Dictionnaire avec analyse de voies d'administration structurée
        """
        # Créer une clé de cache
        cache_key = f"admin_route_{prescription}_{patient_info}_{len(context_docs or [])}"
        
        # Vérifier le cache
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key, prefix="admin_route_")
            if cached_result is not None:
                logger.info("Administration route analysis cache hit")
                return cached_result
        
        try:
            # Préparer le contexte
            context = self._prepare_context(context_docs or [])
            
            # Préparer le prompt
            prompt = PROMPT_TEMPLATES['administration_route_analysis'].format(
                prescription=prescription,
                patient_info=patient_info or "Informations patient non spécifiées",
                context=context
            )
            
            # Appeler le LLM
            response = self.cached_invoke(prompt)
            content = response.content.strip()
            
            # Parser la réponse JSON
            route_data = self._parse_route_response(content)
            
            # Calculer les statistiques
            stats = self._calculate_route_stats(route_data)
            
            result = {
                'administration_route_analysis': route_data,
                'stats': stats,
                'raw_response': content,
                'context_used': len(context_docs or []) > 0
            }
            
            # Mettre en cache
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, prefix="admin_route_")
            
            logger.info(f"Administration route analysis completed: {stats['total_issues']} issues found")
            return result
            
        except Exception as e:
            logger.error(f"Administration route analysis failed: {e}")
            return self._get_empty_route_result(str(e))
    
    def _prepare_context(self, context_docs: List[str]) -> str:
        """
        Prépare le contexte à partir des documents
        
        Args:
            context_docs: Liste des documents de contexte
            
        Returns:
            Contexte formaté
        """
        if not context_docs:
            return "Aucune documentation spécifique disponible. Base-toi sur tes connaissances médicales générales pour les voies d'administration standards."
        
        context = "Documentation médicale de référence sur les voies d'administration:\n\n"
        for i, doc in enumerate(context_docs[:10], 1):
            context += f"Source {i}:\n{doc}\n\n"
        
        return context
    
    def _parse_route_response(self, content: str) -> Dict:
        """
        Parse la réponse JSON du LLM
        
        Args:
            content: Contenu de la réponse
            
        Returns:
            Données de voies d'administration parsées
        """
        try:
            # Chercher le JSON dans la réponse
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                if 'administration_route_analysis' in data:
                    return data['administration_route_analysis']
                else:
                    return data
            else:
                logger.warning("No JSON found in administration route response")
                return self._get_empty_route_analysis()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in administration route response: {e}")
            # Essayer de parser manuellement
            return self._manual_parse_route(content)
        except Exception as e:
            logger.error(f"Error parsing administration route response: {e}")
            return self._get_empty_route_analysis()
    
    def _manual_parse_route(self, content: str) -> Dict:
        """
        Parse manuel si JSON échoue
        
        Args:
            content: Contenu à parser
            
        Returns:
            Données de voies d'administration parsées manuellement
        """
        # Parse basique pour extraire les informations de voies d'administration
        result = {
            'voie_inappropriee': [],
            'voie_incompatible': [],
            'voie_risquee': [],
            'voie_appropriee': []
        }
        
        # Chercher les mentions dans le texte
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip().lower()
            
            if any(keyword in line for keyword in ['inappropriée', 'inadaptée', 'incorrecte']):
                current_section = 'voie_inappropriee'
            elif any(keyword in line for keyword in ['incompatible', 'incompatibilité']):
                current_section = 'voie_incompatible'
            elif any(keyword in line for keyword in ['risquée', 'dangereuse', 'surveillance']):
                current_section = 'voie_risquee'
            elif any(keyword in line for keyword in ['appropriée', 'correcte', 'adaptée']):
                current_section = 'voie_appropriee'
        
        logger.warning("Used manual parsing for administration route analysis")
        return result
    
    def _calculate_route_stats(self, route_data: Dict) -> Dict:
        """
        Calcule les statistiques des voies d'administration
        
        Args:
            route_data: Données de voies d'administration analysées
            
        Returns:
            Statistiques calculées
        """
        stats = {
            'total_medications': 0,
            'voie_inappropriee_count': len(route_data.get('voie_inappropriee', [])),
            'voie_incompatible_count': len(route_data.get('voie_incompatible', [])),
            'voie_risquee_count': len(route_data.get('voie_risquee', [])),
            'voie_appropriee_count': len(route_data.get('voie_appropriee', [])),
            'total_issues': 0,
            'gravite_repartition': {'Faible': 0, 'Modérée': 0, 'Élevée': 0},
            'has_critical_issues': False
        }
        
        # Compter les problèmes par gravité
        all_issues = (
            route_data.get('voie_inappropriee', []) + 
            route_data.get('voie_incompatible', []) + 
            route_data.get('voie_risquee', [])
        )
        
        for issue in all_issues:
            gravite = issue.get('gravite', 'Faible')
            if gravite in stats['gravite_repartition']:
                stats['gravite_repartition'][gravite] += 1
            
            if gravite == 'Élevée':
                stats['has_critical_issues'] = True
        
        stats['total_issues'] = len(all_issues)
        stats['total_medications'] = stats['total_issues'] + stats['voie_appropriee_count']
        
        return stats
    
    def _get_empty_route_analysis(self) -> Dict:
        """Retourne une structure de voies d'administration vide"""
        return {
            'voie_inappropriee': [],
            'voie_incompatible': [],
            'voie_risquee': [],
            'voie_appropriee': []
        }
    
    def _get_empty_route_result(self, error_msg: str = "") -> Dict:
        """Retourne un résultat de voies d'administration vide avec erreur"""
        return {
            'administration_route_analysis': self._get_empty_route_analysis(),
            'stats': {
                'total_medications': 0,
                'voie_inappropriee_count': 0,
                'voie_incompatible_count': 0,
                'voie_risquee_count': 0,
                'voie_appropriee_count': 0,
                'total_issues': 0,
                'gravite_repartition': {'Faible': 0, 'Modérée': 0, 'Élevée': 0},
                'has_critical_issues': False,
                'error': error_msg
            },
            'raw_response': "",
            'context_used': False
        }
    
    def format_route_for_display(self, route_analysis: Dict) -> List[Dict]:
        """
        Formate les données de voies d'administration pour l'affichage dans les tableaux
        
        Args:
            route_analysis: Données d'analyse de voies d'administration
            
        Returns:
            Liste de dictionnaires formatés pour les tableaux
        """
        formatted_data = []
        
        # Traiter les voies inappropriées
        for item in route_analysis.get('voie_inappropriee', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Voie inappropriée',
                'Voie prescrite': item.get('voie_prescrite', 'Non spécifiée'),
                'Voie recommandée': item.get('voie_recommandee', 'Non spécifiée'),
                'Gravité': item.get('gravite', 'Faible'),
                'Justification': item.get('justification', 'Non spécifiée'),
                'Explication': item.get('explication', 'Aucune explication'),
                'Recommandation': item.get('recommandation', 'Aucune recommandation'),
                'Couleur': self._get_severity_color(item.get('gravite', 'Faible'))
            })
        
        # Traiter les voies incompatibles
        for item in route_analysis.get('voie_incompatible', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Voie incompatible',
                'Voie prescrite': item.get('voie_prescrite', 'Non spécifiée'),
                'Voie recommandée': item.get('voie_recommandee', 'Non spécifiée'),
                'Gravité': item.get('gravite', 'Faible'),
                'Justification': item.get('justification', 'Non spécifiée'),
                'Explication': item.get('explication', 'Aucune explication'),
                'Recommandation': item.get('recommandation', 'Aucune recommandation'),
                'Couleur': self._get_severity_color(item.get('gravite', 'Faible'))
            })
        
        # Traiter les voies risquées
        for item in route_analysis.get('voie_risquee', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Voie risquée',
                'Voie prescrite': item.get('voie_prescrite', 'Non spécifiée'),
                'Voie recommandée': item.get('voie_recommandee', 'Non spécifiée'),
                'Gravité': item.get('gravite', 'Faible'),
                'Justification': item.get('justification', 'Non spécifiée'),
                'Explication': item.get('explication', 'Aucune explication'),
                'Recommandation': item.get('recommandation', 'Aucune recommandation'),
                'Couleur': self._get_severity_color(item.get('gravite', 'Faible'))
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
