

"""
Fonctions utilitaires réutilisables pour l'application
"""
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from config.logging_config import get_logger
from utils.constants import LEVEL_COLORS, INTERACTION_LEVELS

logger = get_logger(__name__)

def format_timestamp(timestamp: datetime = None) -> str:
    """
    Formate un timestamp en chaîne lisible
    
    Args:
        timestamp: Timestamp à formater (utilise datetime.now() si None)
        
    Returns:
        Chaîne formatée (ex: "21/06/2025 14:30")
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%d/%m/%Y %H:%M")

def format_file_size(size_bytes: int) -> str:
    """
    Formate une taille de fichier en unités lisibles
    
    Args:
        size_bytes: Taille en octets
        
    Returns:
        Chaîne formatée (ex: "2.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1
    
    return f"{size_bytes:.1f} {units[unit_index]}"

def classify_interaction_level(level_text: str) -> str:
    """
    Classifie le niveau d'interaction à partir du texte de réponse LLM
    
    Args:
        level_text: Texte du niveau d'interaction
        
    Returns:
        Niveau standardisé ('Major', 'Moderate', 'Minor', 'Aucune')
    """
    level_lower = level_text.lower().strip()
    return INTERACTION_LEVELS.get(level_lower, 'Aucune')

def get_level_color(level: str) -> str:
    """
    Retourne la couleur associée à un niveau d'interaction
    
    Args:
        level: Niveau d'interaction
        
    Returns:
        Code couleur hexadécimal
    """
    return LEVEL_COLORS.get(level, LEVEL_COLORS['Aucune'])

def sanitize_filename(filename: str) -> str:
    """
    Nettoie un nom de fichier pour qu'il soit valide
    
    Args:
        filename: Nom de fichier à nettoyer
        
    Returns:
        Nom de fichier nettoyé
    """
    # Caractères interdits dans les noms de fichiers
    invalid_chars = '<>:"/\\|?*'
    
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Supprimer les espaces multiples et en début/fin
    filename = ' '.join(filename.split())
    
    # Limiter la longueur
    # if len(filename) > 200:
    #     filename = filename[:200] + "..."
    
    return filename

def validate_pdf_extension(filename: str) -> bool:
    """
    Vérifie si un fichier a une extension PDF valide
    
    Args:
        filename: Nom du fichier
        
    Returns:
        True si l'extension est valide
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() == '.pdf'

def calculate_combinations_count(n: int) -> int:
    """
    Calcule le nombre de combinaisons C(n,2)
    
    Args:
        n: Nombre d'éléments
        
    Returns:
        Nombre de combinaisons possibles
    """
    if n < 2:
        return 0
    return (n * (n - 1)) // 2

def estimate_analysis_time(num_combinations: int, avg_time_per_combination: float = 2.0) -> str:
    """
    Estime le temps d'analyse basé sur le nombre de combinaisons
    
    Args:
        num_combinations: Nombre de combinaisons à analyser
        avg_time_per_combination: Temps moyen par combinaison (secondes)
        
    Returns:
        Estimation formatée (ex: "2 min 30 sec")
    """
    total_seconds = int(num_combinations * avg_time_per_combination)
    
    if total_seconds < 60:
        return f"{total_seconds} sec"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes} min {seconds} sec" if seconds > 0 else f"{minutes} min"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}min"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Tronque un texte s'il dépasse une longueur maximale
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter si tronqué
        
    Returns:
        Texte tronqué
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_drug_pair_key(drug1: str, drug2: str) -> str:
    """
    Crée une clé unique pour une paire de médicaments (ordre indépendant)
    
    Args:
        drug1: Premier médicament
        drug2: Deuxième médicament
        
    Returns:
        Clé unique pour la paire
    """
    # Trier pour avoir un ordre consistant
    sorted_drugs = sorted([drug1.strip().lower(), drug2.strip().lower()])
    return f"{sorted_drugs[0]}_{sorted_drugs[1]}"

def clean_drug_name(drug_name: str) -> str:
    """
    Nettoie minimalement un nom de médicament en préservant la casse originale
    
    Args:
        drug_name: Nom du médicament
        
    Returns:
        Nom nettoyé mais avec casse préservée
    """
    # Supprimer les espaces en début/fin
    cleaned = drug_name.strip()
    
    # Supprimer les caractères spéciaux courants
    cleaned = cleaned.replace('®', '').replace('™', '').replace('©', '')
    
    # Supprimer les espaces multiples
    cleaned = ' '.join(cleaned.split())
    
    # PRÉSERVER LA CASSE ORIGINALE - ne pas modifier
    return cleaned

def parse_llm_response(response_text: str) -> Dict[str, str]:
    """
    Parse une réponse LLM structurée avec format NIVEAU:/EXPLICATION:
    
    Args:
        response_text: Texte de réponse du LLM
        
    Returns:
        Dictionnaire avec 'level' et 'explanation'
    """
    result = {
        'level': 'Aucune',
        'explanation': 'Aucune information disponible'
    }
    
    try:
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("NIVEAU:"):
                level_text = line.replace("NIVEAU:", "").strip()
                result['level'] = classify_interaction_level(level_text)
            elif line.startswith("EXPLICATION:"):
                result['explanation'] = line.replace("EXPLICATION:", "").strip()
    
    except Exception as e:
        logger.warning(f"Error parsing LLM response: {e}")
    
    return result

def create_export_filename(base_name: str, file_format: str, timestamp: datetime = None) -> str:
    """
    Crée un nom de fichier pour l'export
    
    Args:
        base_name: Nom de base
        file_format: Format du fichier ('csv' ou 'excel')
        timestamp: Timestamp (utilise datetime.now() si None)
        
    Returns:
        Nom de fichier complet
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    extension = '.xlsx' if file_format.lower() == 'excel' else '.csv'
    
    filename = f"{base_name}_{timestamp_str}{extension}"
    return sanitize_filename(filename)

def measure_execution_time(func):
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction
    
    Args:
        func: Fonction à mesurer
        
    Returns:
        Fonction wrappée qui log le temps d'exécution
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Division sécurisée qui évite la division par zéro
    
    Args:
        numerator: Numérateur
        denominator: Dénominateur
        default: Valeur par défaut si division par zéro
        
    Returns:
        Résultat de la division ou valeur par défaut
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Formate un pourcentage avec un nombre de décimales spécifié
    
    Args:
        value: Valeur à formater (0-100)
        decimal_places: Nombre de décimales
        
    Returns:
        Pourcentage formaté (ex: "85.5%")
    """
    try:
        return f"{value:.{decimal_places}f}%"
    except (TypeError, ValueError):
        return "0.0%"

def is_valid_drug_list(drugs: List[str]) -> bool:
    """
    Vérifie si une liste de médicaments est valide pour l'analyse
    
    Args:
        drugs: Liste des médicaments
        
    Returns:
        True si la liste est valide
    """
    if not drugs or len(drugs) < 2:
        return False
    
    # Vérifier que tous les éléments sont des chaînes non vides
    for drug in drugs:
        if not isinstance(drug, str) or not drug.strip():
            return False
    
    return True

def create_progress_message(current: int, total: int, item_name: str = "item") -> str:
    """
    Crée un message de progression
    
    Args:
        current: Élément actuel
        total: Total d'éléments
        item_name: Nom du type d'élément
        
    Returns:
        Message de progression formaté
    """
    percentage = safe_divide(current, total, 0) * 100
    return f"Traitement {item_name} {current}/{total} ({percentage:.1f}%)"



def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Divise une liste en chunks de taille spécifiée
    
    Args:
        lst: Liste à diviser
        chunk_size: Taille de chaque chunk
        
    Returns:
        Liste de chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionne plusieurs dictionnaires (les derniers écrasent les premiers)
    
    Args:
        *dicts: Dictionnaires à fusionner
        
    Returns:
        Dictionnaire fusionné
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def extract_exact_quote_with_context(text: str, query_words: List[str], context_chars: int = 100) -> Dict[str, str]:
    """
    Extrait une citation exacte avec son contexte pour traçabilité
    
    Args:
        text: Texte source
        query_words: Mots-clés de la requête
        context_chars: Nombre de caractères de contexte avant/après
        
    Returns:
        Dictionnaire avec citation et contexte
    """
    import re
    
    # Diviser en phrases
    sentences = re.split(r'[.!?]+', text)
    
    best_sentence = ""
    best_score = 0
    best_position = 0
    
    # Trouver la meilleure phrase
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
            
        sentence_lower = sentence.lower()
        score = sum(1 for word in query_words if word.lower() in sentence_lower)
        
        if score > best_score:
            best_score = score
            best_sentence = sentence
            best_position = text.find(sentence)
    
    if not best_sentence:
        return {
            'exact_quote': text[:200] + "..." if len(text) > 200 else text,
            'context': text[:500] + "..." if len(text) > 500 else text
        }
    
    # Extraire le contexte
    context_start = max(0, best_position - context_chars)
    context_end = min(len(text), best_position + len(best_sentence) + context_chars)
    
    context = text[context_start:context_end]
    if context_start > 0:
        context = "..." + context
    if context_end < len(text):
        context = context + "..."
    
    return {
        'exact_quote': best_sentence[:200] + "..." if len(best_sentence) > 200 else best_sentence,
        'context': context
    }

def format_academic_medical_citation(document_name: str, page: str, section: str = None) -> str:
    """
    Formate une citation académique médicale
    
    Args:
        document_name: Nom du document
        page: Numéro de page
        section: Section optionnelle
        
    Returns:
        Citation formatée
    """
    # Déterminer le type de guideline
    doc_lower = document_name.lower()
    
    if 'beers' in doc_lower:
        base_citation = "American Geriatrics Society Beers Criteria"
    elif 'stopp' in doc_lower or 'start' in doc_lower:
        base_citation = "STOPP/START Criteria"
    elif 'laroche' in doc_lower:
        base_citation = "Liste de Laroche"
    elif 'priscus' in doc_lower:
        base_citation = "PRISCUS List"
    elif 'onc' in doc_lower:
        base_citation = "ONC High-Priority Drug-Drug Interactions"
    else:
        base_citation = document_name
    
    # Construire la citation
    citation = f"{base_citation} (Page {page})"
    if section:
        citation += f", Section: {section}"
    
    return citation

def calculate_metadata_confidence(text_length: int, drug_mentions: int, section_type: str) -> float:
    """
    Calcule un score de confiance pour les métadonnées extraites
    
    Args:
        text_length: Longueur du texte
        drug_mentions: Nombre de mentions de molécules
        section_type: Type de section
        
    Returns:
        Score de confiance entre 0 et 1
    """
    score = 0.0
    
    # Facteur longueur (30%)
    length_score = min(text_length / 1000, 1.0)
    score += length_score * 0.3
    
    # Facteur molécules (40%)
    drug_score = min(drug_mentions / 5, 1.0)
    score += drug_score * 0.4
    
    # Facteur type de section (30%)
    section_scores = {
        'interaction': 1.0,
        'contraindication': 0.9,
        'warning': 0.8,
        'indication': 0.7,
        'dosage': 0.6,
        'adverse_effect': 0.5,
        'general': 0.3
    }
    section_score = section_scores.get(section_type, 0.3)
    score += section_score * 0.3
    
    return round(min(score, 1.0), 3)

def detect_guideline_type_from_content(text: str, filename: str = "") -> str:
    """
    Détecte le type de guideline depuis le contenu et/ou le nom de fichier
    
    Args:
        text: Contenu du document
        filename: Nom de fichier optionnel
        
    Returns:
        Type de guideline détecté
    """
    combined_text = (filename + " " + text).lower()
    
    # Patterns de détection améliorés
    if any(pattern in combined_text for pattern in ['stopp', 'start', 'screening tool']):
        return 'STOPP/START'
    elif any(pattern in combined_text for pattern in ['beers', 'ags', 'american geriatrics']):
        return 'BEERS_CRITERIA'
    elif 'laroche' in combined_text:
        return 'LAROCHE_LIST'
    elif 'priscus' in combined_text:
        return 'PRISCUS_LIST'
    elif any(pattern in combined_text for pattern in ['onc', 'oncology', 'drug-drug interaction']):
        return 'ONC_DDI'
    else:
        return 'UNKNOWN_GUIDELINE'

def create_progress_indicator(current: int, total: int, description: str = "") -> str:
    """
    Crée un indicateur de progression pour le traitement séquentiel
    
    Args:
        current: Élément actuel
        total: Total d'éléments
        description: Description optionnelle
        
    Returns:
        Chaîne d'indicateur de progression
    """
    percentage = (current / total) * 100 if total > 0 else 0
    progress_bar = "█" * int(percentage // 10) + "░" * (10 - int(percentage // 10))
    
    base_indicator = f"[{progress_bar}] {current}/{total} ({percentage:.1f}%)"
    
    if description:
        return f"{base_indicator} - {description}"
    
    return base_indicator
