"""
Analyseur de contre-indications pour détecter les contre-indications médicamenteuses
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

class ContraindicationAnalyzer:
    """
    Analyseur spécialisé pour les contre-indications médicamenteuses
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialise l'analyseur de contre-indications
        
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
    def analyze_contraindications(self, prescription: str, patient_info: str = "", context_docs: List[str] = None) -> Dict:
        """
        Analyse les contre-indications de la prescription
        
        Args:
            prescription: Texte de la prescription
            patient_info: Informations sur le patient (âge, pathologies, etc.)
            context_docs: Documents de contexte de la base vectorielle
            
        Returns:
            Dictionnaire avec analyse de contre-indications structurée
        """
        # Créer une clé de cache
        cache_key = f"contraindication_{prescription}_{patient_info}_{len(context_docs or [])}"
        
        # Vérifier le cache
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key, prefix="contraindication_")
            if cached_result is not None:
                logger.info("Contraindication analysis cache hit")
                return cached_result
        
        try:
            # Préparer le contexte
            context = self._prepare_context(context_docs or [])
            
            # Préparer le prompt
            prompt = PROMPT_TEMPLATES['contraindication_analysis'].format(
                prescription=prescription,
                patient_info=patient_info or "Informations patient non spécifiées",
                context=context
            )
            
            # Appeler le LLM
            response = self.cached_invoke(prompt)
            content = response.content.strip()
            
            # Parser la réponse JSON
            contraindication_data = self._parse_contraindication_response(content)
            
            # Calculer les statistiques
            stats = self._calculate_contraindication_stats(contraindication_data)
            
            result = {
                'contraindication_analysis': contraindication_data,
                'stats': stats,
                'raw_response': content,
                'context_used': len(context_docs or []) > 0
            }
            
            # Mettre en cache
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, prefix="contraindication_")
            
            logger.info(f"Contraindication analysis completed: {stats['total_contraindications']} contraindications found")
            return result
            
        except Exception as e:
            logger.error(f"Contraindication analysis failed: {e}")
            return self._get_empty_contraindication_result(str(e))
    
    def _prepare_context(self, context_docs: List[str]) -> str:
        """
        Prépare le contexte à partir des documents
        
        Args:
            context_docs: Liste des documents de contexte
            
        Returns:
            Contexte formaté
        """
        if not context_docs:
            return "Aucune documentation spécifique disponible. Base-toi sur tes connaissances médicales générales."
        
        context = "Documentation médicale de référence:\\n\\n"
        for i, doc in enumerate(context_docs[:3], 1):  # Limiter à 3 documents
            context += f"Source {i}:\\n{doc}\\n\\n"
        
        return context
    
    def _parse_contraindication_response(self, content: str) -> Dict:
        """
        Parse la réponse JSON du LLM
        
        Args:
            content: Contenu de la réponse
            
        Returns:
            Données de contre-indications parsées
        """
        try:
            # Chercher le JSON dans la réponse
            json_match = re.search(r'\\{.*\\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                if 'contraindication_analysis' in data:
                    return data['contraindication_analysis']
                else:
                    return data
            else:
                logger.warning("No JSON found in contraindication response")
                return self._get_empty_contraindication_analysis()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in contraindication response: {e}")
            # Essayer de parser manuellement
            return self._manual_parse_contraindication(content)
        except Exception as e:
            logger.error(f"Error parsing contraindication response: {e}")
            return self._get_empty_contraindication_analysis()
    
    def _manual_parse_contraindication(self, content: str) -> Dict:
        """
        Parse manuel si JSON échoue
        
        Args:
            content: Contenu à parser
            
        Returns:
            Données de contre-indications parsées manuellement
        """
        result = {
            'contre_indications_absolues': [],
            'contre_indications_relatives': [],
            'aucune_contre_indication': [],
            'donnees_insuffisantes': []
        }
        
        # Parse basique pour extraire les informations
        lines = content.split('\\n')
        current_section = None
        
        for line in lines:
            line = line.strip().lower()
            
            if any(keyword in line for keyword in ['contre-indication absolue', 'interdiction', 'formellement contre-indiqué']):
                current_section = 'contre_indications_absolues'
            elif any(keyword in line for keyword in ['contre-indication relative', 'prudence', 'surveillance']):
                current_section = 'contre_indications_relatives'
            elif any(keyword in line for keyword in ['aucune contre-indication', 'pas de contre-indication']):
                current_section = 'aucune_contre_indication'
            elif any(keyword in line for keyword in ['données insuffisantes', 'information manquante']):
                current_section = 'donnees_insuffisantes'
        
        logger.warning("Used manual parsing for contraindication analysis")
        return result
    
    def _calculate_contraindication_stats(self, contraindication_data: Dict) -> Dict:
        """
        Calcule les statistiques de contre-indications
        
        Args:
            contraindication_data: Données de contre-indications analysées
            
        Returns:
            Statistiques calculées
        """
        stats = {
            'total_medications': 0,
            'contre_indications_absolues_count': len(contraindication_data.get('contre_indications_absolues', [])),
            'contre_indications_relatives_count': len(contraindication_data.get('contre_indications_relatives', [])),
            'aucune_contre_indication_count': len(contraindication_data.get('aucune_contre_indication', [])),
            'donnees_insuffisantes_count': len(contraindication_data.get('donnees_insuffisantes', [])),
            'total_contraindications': 0,
            'gravite_repartition': {'Absolue': 0, 'Relative': 0},
            'has_critical_contraindications': False,
            'prescription_safety_level': 'SAFE'
        }
        
        # Compter les contre-indications par gravité
        stats['gravite_repartition']['Absolue'] = stats['contre_indications_absolues_count']
        stats['gravite_repartition']['Relative'] = stats['contre_indications_relatives_count']
        
        stats['total_contraindications'] = stats['contre_indications_absolues_count'] + stats['contre_indications_relatives_count']
        
        if stats['contre_indications_absolues_count'] > 0:
            stats['has_critical_contraindications'] = True
            stats['prescription_safety_level'] = 'CRITICAL'
        elif stats['contre_indications_relatives_count'] > 0:
            stats['prescription_safety_level'] = 'CAUTION'
        
        stats['total_medications'] = (
            stats['contre_indications_absolues_count'] + 
            stats['contre_indications_relatives_count'] +
            stats['aucune_contre_indication_count'] +
            stats['donnees_insuffisantes_count']
        )
        
        return stats
    
    def _get_empty_contraindication_analysis(self) -> Dict:
        """Retourne une structure de contre-indications vide"""
        return {
            'contre_indications_absolues': [],
            'contre_indications_relatives': [],
            'aucune_contre_indication': [],
            'donnees_insuffisantes': []
        }
    
    def _get_empty_contraindication_result(self, error_msg: str = "") -> Dict:
        """Retourne un résultat de contre-indications vide avec erreur"""
        return {
            'contraindication_analysis': self._get_empty_contraindication_analysis(),
            'stats': {
                'total_medications': 0,
                'contre_indications_absolues_count': 0,
                'contre_indications_relatives_count': 0,
                'aucune_contre_indication_count': 0,
                'donnees_insuffisantes_count': 0,
                'total_contraindications': 0,
                'gravite_repartition': {'Absolue': 0, 'Relative': 0},
                'has_critical_contraindications': False,
                'prescription_safety_level': 'UNKNOWN',
                'error': error_msg
            },
            'raw_response': "",
            'context_used': False
        }
    
    def format_contraindication_for_display(self, contraindication_analysis: Dict) -> List[Dict]:
        """
        Formate les données de contre-indications pour l'affichage dans les tableaux
        
        Args:
            contraindication_analysis: Données d'analyse de contre-indications
            
        Returns:
            Liste de dictionnaires formatés pour les tableaux
        """
        formatted_data = []
        
        # Traiter les contre-indications absolues
        for item in contraindication_analysis.get('contre_indications_absolues', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Contre-indication absolue',
                'Condition/Pathologie': item.get('condition', 'Non spécifiée'),
                'Gravité': 'Absolue',
                'Mécanisme': item.get('mecanisme', 'Non spécifié'),
                'Conséquences': item.get('consequences', 'Risque élevé'),
                'Recommandation': item.get('recommandation', 'Éviter absolument'),
                'Source': item.get('source', 'Base de connaissances'),
                'Couleur': '#DC3545'  # Rouge
            })
        
        # Traiter les contre-indications relatives
        for item in contraindication_analysis.get('contre_indications_relatives', []):
            formatted_data.append({
                'Médicament': item.get('medicament', 'Inconnu'),
                'Type': 'Contre-indication relative',
                'Condition/Pathologie': item.get('condition', 'Non spécifiée'),
                'Gravité': 'Relative',
                'Mécanisme': item.get('mecanisme', 'Non spécifié'),
                'Conséquences': item.get('consequences', 'Risque modéré'),
                'Recommandation': item.get('recommandation', 'Surveillance renforcée'),
                'Source': item.get('source', 'Base de connaissances'),
                'Couleur': '#FD7E14'  # Orange
            })
        
        return formatted_data
