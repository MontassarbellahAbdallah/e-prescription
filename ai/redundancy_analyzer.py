"""
Analyseur de redondance thérapeutique pour détecter les redondances médicamenteuses
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

class RedundancyAnalyzer:
    """
    Analyseur spécialisé pour la redondance thérapeutique
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialise l'analyseur de redondance thérapeutique
        
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
    def analyze_redundancy(self, prescription: str, patient_info: str = "", context_docs: List[str] = None) -> Dict:
        """
        Analyse les redondances thérapeutiques de la prescription
        
        Args:
            prescription: Texte de la prescription
            patient_info: Informations sur le patient
            context_docs: Documents de contexte de la base vectorielle
            
        Returns:
            Dictionnaire avec analyse de redondance structurée
        """
        # Créer une clé de cache
        cache_key = f"redundancy_{prescription}_{patient_info}_{len(context_docs or [])}"
        
        # Vérifier le cache
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key, prefix="redundancy_")
            if cached_result is not None:
                logger.info("Redundancy analysis cache hit")
                return cached_result
        
        try:
            # Préparer le contexte
            context = self._prepare_context(context_docs or [])
            
            # Préparer le prompt
            prompt = PROMPT_TEMPLATES['redundancy_analysis'].format(
                prescription=prescription,
                patient_info=patient_info or "Informations patient non spécifiées",
                context=context
            )
            
            # Appeler le LLM
            response = self.cached_invoke(prompt)
            content = response.content.strip()
            
            # Parser la réponse JSON
            redundancy_data = self._parse_redundancy_response(content)
            
            # Calculer les statistiques
            stats = self._calculate_redundancy_stats(redundancy_data)
            
            result = {
                'redundancy_analysis': redundancy_data,
                'stats': stats,
                'raw_response': content,
                'context_used': len(context_docs or []) > 0
            }
            
            # Mettre en cache
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, prefix="redundancy_")
            
            logger.info(f"Redundancy analysis completed: {stats['total_redundancies']} redundancies found")
            return result
            
        except Exception as e:
            logger.error(f"Redundancy analysis failed: {e}")
            return self._get_empty_redundancy_result(str(e))
    
    def _prepare_context(self, context_docs: List[str]) -> str:
        """
        Prépare le contexte à partir des documents
        
        Args:
            context_docs: Liste des documents de contexte
            
        Returns:
            Contexte formaté
        """
        if not context_docs:
            return "Aucune documentation spécifique disponible. Base-toi sur tes connaissances médicales générales sur les classes thérapeutiques."
        
        context = "Documentation médicale de référence sur les classes thérapeutiques et redondances:\n\n"
        for i, doc in enumerate(context_docs[:5], 1):  # Limiter à 5 documents
            context += f"Source {i}:\n{doc}\n\n"
        
        return context
    
    def _parse_redundancy_response(self, content: str) -> Dict:
        """
        Parse la réponse JSON du LLM
        
        Args:
            content: Contenu de la réponse
            
        Returns:
            Données de redondance parsées
        """
        try:
            # Chercher le JSON dans la réponse
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                if 'redundancy_analysis' in data:
                    return data['redundancy_analysis']
                else:
                    return data
            else:
                logger.warning("No JSON found in redundancy response")
                return self._get_empty_redundancy_analysis()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in redundancy response: {e}")
            # Essayer de parser manuellement
            return self._manual_parse_redundancy(content)
        except Exception as e:
            logger.error(f"Error parsing redundancy response: {e}")
            return self._get_empty_redundancy_analysis()
    
    def _manual_parse_redundancy(self, content: str) -> Dict:
        """
        Parse manuel si JSON échoue
        
        Args:
            content: Contenu à parser
            
        Returns:
            Données de redondance parsées manuellement
        """
        result = {
            'redondance_directe': [],
            'redondance_classe': [],
            'redondance_fonctionnelle': [],
            'aucune_redondance': []
        }
        
        # Parse basique pour extraire les informations de redondance
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip().lower()
            
            if any(keyword in line for keyword in ['redondance directe', 'même molécule', 'doublon']):
                current_section = 'redondance_directe'
            elif any(keyword in line for keyword in ['même classe', 'classe thérapeutique', 'redondance de classe']):
                current_section = 'redondance_classe'
            elif any(keyword in line for keyword in ['redondance fonctionnelle', 'même effet', 'effet similaire']):
                current_section = 'redondance_fonctionnelle'
            elif any(keyword in line for keyword in ['aucune redondance', 'pas de redondance']):
                current_section = 'aucune_redondance'
        
        logger.warning("Used manual parsing for redundancy analysis")
        return result
    
    def _calculate_redundancy_stats(self, redundancy_data: Dict) -> Dict:
        """
        Calcule les statistiques de redondance
        
        Args:
            redundancy_data: Données de redondance analysées
            
        Returns:
            Statistiques calculées
        """
        stats = {
            'total_medications': 0,
            'redondance_directe_count': len(redundancy_data.get('redondance_directe', [])),
            'redondance_classe_count': len(redundancy_data.get('redondance_classe', [])),
            'redondance_fonctionnelle_count': len(redundancy_data.get('redondance_fonctionnelle', [])),
            'aucune_redondance_count': len(redundancy_data.get('aucune_redondance', [])),
            'total_redundancies': 0,
            'gravite_repartition': {'Directe': 0, 'Classe': 0, 'Fonctionnelle': 0},
            'has_critical_redundancies': False,
            'prescription_optimization_potential': 'LOW'
        }
        
        # Compter les redondances par type
        stats['gravite_repartition']['Directe'] = stats['redondance_directe_count']
        stats['gravite_repartition']['Classe'] = stats['redondance_classe_count']
        stats['gravite_repartition']['Fonctionnelle'] = stats['redondance_fonctionnelle_count']
        
        stats['total_redundancies'] = (
            stats['redondance_directe_count'] + 
            stats['redondance_classe_count'] + 
            stats['redondance_fonctionnelle_count']
        )
        
        if stats['redondance_directe_count'] > 0:
            stats['has_critical_redundancies'] = True
            stats['prescription_optimization_potential'] = 'HIGH'
        elif stats['redondance_classe_count'] > 0:
            stats['prescription_optimization_potential'] = 'MEDIUM'
        
        stats['total_medications'] = (
            stats['total_redundancies'] + 
            stats['aucune_redondance_count']
        )
        
        return stats
    
    def _get_empty_redundancy_analysis(self) -> Dict:
        """Retourne une structure de redondance vide"""
        return {
            'redondance_directe': [],
            'redondance_classe': [],
            'redondance_fonctionnelle': [],
            'aucune_redondance': []
        }
    
    def _get_empty_redundancy_result(self, error_msg: str = "") -> Dict:
        """Retourne un résultat de redondance vide avec erreur"""
        return {
            'redundancy_analysis': self._get_empty_redundancy_analysis(),
            'stats': {
                'total_medications': 0,
                'redondance_directe_count': 0,
                'redondance_classe_count': 0,
                'redondance_fonctionnelle_count': 0,
                'aucune_redondance_count': 0,
                'total_redundancies': 0,
                'gravite_repartition': {'Directe': 0, 'Classe': 0, 'Fonctionnelle': 0},
                'has_critical_redundancies': False,
                'prescription_optimization_potential': 'UNKNOWN',
                'error': error_msg
            },
            'raw_response': "",
            'context_used': False
        }
    
    def format_redundancy_for_display(self, redundancy_analysis: Dict) -> List[Dict]:
        """
        Formate les données de redondance pour l'affichage dans les tableaux
        
        Args:
            redundancy_analysis: Données d'analyse de redondance
            
        Returns:
            Liste de dictionnaires formatés pour les tableaux
        """
        formatted_data = []
        
        # Traiter les redondances directes
        for item in redundancy_analysis.get('redondance_directe', []):
            formatted_data.append({
                'Classe thérapeutique': item.get('classe_therapeutique', 'Inconnue'),
                'Médicaments redondants': ', '.join(item.get('medicaments', [])),
                'Type de redondance': 'Redondance directe',
                'Gravité': 'Élevée',
                'Mécanisme': item.get('mecanisme', 'Même molécule active prescrite plusieurs fois'),
                'Risque': item.get('risque', 'Surdosage, effets indésirables cumulés'),
                'Recommandation': item.get('recommandation', 'Éliminer les doublons'),
                'Source': item.get('source', 'Base de connaissances'),
                'Couleur': '#DC3545'  # Rouge
            })
        
        # Traiter les redondances de classe
        for item in redundancy_analysis.get('redondance_classe', []):
            formatted_data.append({
                'Classe thérapeutique': item.get('classe_therapeutique', 'Inconnue'),
                'Médicaments redondants': ', '.join(item.get('medicaments', [])),
                'Type de redondance': 'Redondance de classe',
                'Gravité': 'Modérée',
                'Mécanisme': item.get('mecanisme', 'Même classe thérapeutique'),
                'Risque': item.get('risque', 'Effets additifs, interactions'),
                'Recommandation': item.get('recommandation', 'Évaluer la nécessité, ajuster les doses'),
                'Source': item.get('source', 'Base de connaissances'),
                'Couleur': '#FD7E14'  # Orange
            })
        
        # Traiter les redondances fonctionnelles
        for item in redundancy_analysis.get('redondance_fonctionnelle', []):
            formatted_data.append({
                'Classe thérapeutique': item.get('classe_therapeutique', 'Inconnue'),
                'Médicaments redondants': ', '.join(item.get('medicaments', [])),
                'Type de redondance': 'Redondance fonctionnelle',
                'Gravité': 'Faible à Modérée',
                'Mécanisme': item.get('mecanisme', 'Effet thérapeutique similaire'),
                'Risque': item.get('risque', 'Effet cumulé non nécessaire'),
                'Recommandation': item.get('recommandation', 'Optimiser la stratégie thérapeutique'),
                'Source': item.get('source', 'Base de connaissances'),
                'Couleur': '#28A745'  # Vert
            })
        
        return formatted_data
