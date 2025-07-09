"""
Analyseur de dosage pour détecter les surdosages et sous-dosages
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

class DosageAnalyzer:
    """
    Analyseur spécialisé pour les problèmes de dosage
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialise l'analyseur de dosage
        
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
    def analyze_dosage(self, prescription: str, patient_info: str = "", context_docs: List[str] = None) -> Dict:
        """
        Analyse les dosages de la prescription
        
        Args:
            prescription: Texte de la prescription
            patient_info: Informations sur le patient (âge, poids, etc.)
            context_docs: Documents de contexte de la base vectorielle
            
        Returns:
            Dictionnaire avec analyse de dosage structurée
        """
        # Créer une clé de cache
        cache_key = f"dosage_{prescription}_{patient_info}_{len(context_docs or [])}"
        
        # Vérifier le cache
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key, prefix="dosage_")
            if cached_result is not None:
                logger.info("Dosage analysis cache hit")
                return cached_result
        
        try:
            # Préparer le contexte
            context = self._prepare_context(context_docs or [])
            
            # Préparer le prompt
            prompt = PROMPT_TEMPLATES['dosage_analysis'].format(
                prescription=prescription,
                patient_info=patient_info or "Informations patient non spécifiées",
                context=context
            )
            
            # Appeler le LLM
            response = self.cached_invoke(prompt)
            content = response.content.strip()
            
            # Parser la réponse JSON
            dosage_data = self._parse_dosage_response(content)
            
            # Calculer les statistiques
            stats = self._calculate_dosage_stats(dosage_data)
            
            result = {
                'dosage_analysis': dosage_data,
                'stats': stats,
                'raw_response': content,
                'context_used': len(context_docs or []) > 0
            }
            
            # Mettre en cache
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, prefix="dosage_")
            
            logger.info(f"Dosage analysis completed: {stats['total_issues']} issues found")
            return result
            
        except Exception as e:
            logger.error(f"Dosage analysis failed: {e}")
            return self._get_empty_dosage_result(str(e))
    
    def _prepare_context(self, context_docs: List[str]) -> str:
        """
        Prépare le contexte à partir des documents
        
        Args:
            context_docs: Liste des documents de contexte
            
        Returns:
            Contexte formaté
        """
        if not context_docs:
            return "Aucune documentation spécifique disponible. Base-toi sur tes connaissances médicales générales pour les dosages standards."
        
        context = "Documentation médicale de référence sur les dosages:\n\n"
        for i, doc in enumerate(context_docs[:10], 1):  # Limiter à 3 documents
            context += f"Source {i}:\n{doc}\n\n"
        
        return context
    
    def _parse_dosage_response(self, content: str) -> Dict:
        """
        Parse la réponse JSON du LLM
        
        Args:
            content: Contenu de la réponse
            
        Returns:
            Données de dosage parsées
        """
        try:
            # Chercher le JSON dans la réponse
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                if 'dosage_analysis' in data:
                    return data['dosage_analysis']
                else:
                    return data
            else:
                logger.warning("No JSON found in dosage response")
                return self._get_empty_dosage_analysis()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in dosage response: {e}")
            # Essayer de parser manuellement
            return self._manual_parse_dosage(content)
        except Exception as e:
            logger.error(f"Error parsing dosage response: {e}")
            return self._get_empty_dosage_analysis()
    
    def _manual_parse_dosage(self, content: str) -> Dict:
        """
        Parse manuel si JSON échoue
        
        Args:
            content: Contenu à parser
            
        Returns:
            Données de dosage parsées manuellement
        """
        # Parse basique pour extraire les informations de dosage
        result = {
            'surdosage': [],
            'sous_dosage': [],
            'dosage_approprie': []
        }
        
        # Chercher les mentions de surdosage/sous-dosage dans le texte
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip().lower()
            
            if any(keyword in line for keyword in ['surdosage', 'dose élevée', 'dose excessive']):
                current_section = 'surdosage'
            elif any(keyword in line for keyword in ['sous-dosage', 'dose faible', 'dose insuffisante']):
                current_section = 'sous_dosage'
            elif any(keyword in line for keyword in ['approprié', 'correct', 'adapté']):
                current_section = 'dosage_approprie'
        
        logger.warning("Used manual parsing for dosage analysis")
        return result
    
    def _calculate_dosage_stats(self, dosage_data: Dict) -> Dict:
        """
        Calcule les statistiques de dosage
        
        Args:
            dosage_data: Données de dosage analysées
            
        Returns:
            Statistiques calculées
        """
        stats = {
            'total_medications': 0,
            'surdosage_count': len(dosage_data.get('surdosage', [])),
            'sous_dosage_count': len(dosage_data.get('sous_dosage', [])),
            'dosage_approprie_count': len(dosage_data.get('dosage_approprie', [])),
            'total_issues': 0,
            'gravite_repartition': {'Faible': 0, 'Modérée': 0, 'Élevée': 0},
            'has_critical_issues': False
        }
        
        # Compter les problèmes par gravité
        all_issues = dosage_data.get('surdosage', []) + dosage_data.get('sous_dosage', [])
        
        for issue in all_issues:
            gravite = issue.get('gravite', 'Faible')
            if gravite in stats['gravite_repartition']:
                stats['gravite_repartition'][gravite] += 1
            
            if gravite == 'Élevée':
                stats['has_critical_issues'] = True
        
        stats['total_issues'] = len(all_issues)
        stats['total_medications'] = stats['total_issues'] + stats['dosage_approprie_count']
        
        return stats
    
    def _get_empty_dosage_analysis(self) -> Dict:
        """Retourne une structure de dosage vide"""
        return {
            'surdosage': [],
            'sous_dosage': [],
            'dosage_approprie': []
        }
    
    def _get_empty_dosage_result(self, error_msg: str = "") -> Dict:
        """Retourne un résultat de dosage vide avec erreur"""
        return {
            'dosage_analysis': self._get_empty_dosage_analysis(),
            'stats': {
                'total_medications': 0,
                'surdosage_count': 0,
                'sous_dosage_count': 0,
                'dosage_approprie_count': 0,
                'total_issues': 0,
                'gravite_repartition': {'Faible': 0, 'Modérée': 0, 'Élevée': 0},
                'has_critical_issues': False,
                'error': error_msg
            },
            'raw_response': "",
            'context_used': False
        }
    
    def format_dosage_for_display(self, dosage_analysis: Dict) -> List[Dict]:
        """
        Formate les données de dosage pour l'affichage dans les tableaux
        
        Args:
            dosage_analysis: Données d'analyse de dosage
            
        Returns:
            Liste de dictionnaires formatés pour les tableaux
        """
        formatted_data = []
        
        # Traiter les surdosages
        for item in dosage_analysis.get('surdosage', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Surdosage',
                'Dose prescrite': item.get('dose_prescrite', 'Non spécifiée'),
                'Dose recommandée': item.get('dose_recommandee', 'Non spécifiée'),
                'Gravité': item.get('gravite', 'Faible'),
                'Facteur de risque': item.get('facteur_risque', 'Non spécifié'),
                'Explication': item.get('explication', 'Aucune explication'),
                'Recommandation': item.get('recommandation', 'Aucune recommandation'),
                'Source': item.get('source', 'Base de connaissances'),
                'Couleur': self._get_severity_color(item.get('gravite', 'Faible'))
            })
        
        # Traiter les sous-dosages
        for item in dosage_analysis.get('sous_dosage', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Sous-dosage',
                'Dose prescrite': item.get('dose_prescrite', 'Non spécifiée'),
                'Dose recommandée': item.get('dose_recommandee', 'Non spécifiée'),
                'Gravité': item.get('gravite', 'Faible'),
                'Facteur de risque': item.get('facteur_risque', 'Non spécifié'),
                'Explication': item.get('explication', 'Aucune explication'),
                'Recommandation': item.get('recommandation', 'Aucune recommandation'),
                'Source': item.get('source', 'Base de connaissances'),
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