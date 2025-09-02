"""
Validators pour la validation des données et fichiers
"""
import os
import pdfplumber
from typing import List, Dict, Tuple, Optional
from config.settings import settings
from config.logging_config import get_logger
from core.exceptions import ValidationError, PDFProcessingError
from utils.helpers import format_file_size

logger = get_logger(__name__)

class PDFValidator:
    """Validateur pour les fichiers PDF"""
    
    @staticmethod
    def validate_file_exists(pdf_path: str) -> bool:
        """
        Vérifie qu'un fichier PDF existe
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            True si le fichier existe
            
        Raises:
            ValidationError: Si le fichier n'existe pas
        """
        if not os.path.exists(pdf_path):
            raise ValidationError(f"Fichier PDF non trouvé: {pdf_path}")
        return True
    
    @staticmethod
    def validate_file_size(pdf_path: str) -> bool:
        """
        Vérifie que la taille du fichier PDF est acceptable
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            True si la taille est acceptable
            
        Raises:
            ValidationError: Si le fichier est trop volumineux ou vide
        """
        try:
            file_size = os.path.getsize(pdf_path)
            
            # Fichier vide
            if file_size == 0:
                raise ValidationError(f"Fichier PDF vide: {os.path.basename(pdf_path)}")
            
            # Fichier trop volumineux
            max_size = settings.MAX_PDF_SIZE_MB * 1024 * 1024  # Convertir en octets
            if file_size > max_size:
                raise ValidationError(
                    f"Fichier PDF trop volumineux: {os.path.basename(pdf_path)} "
                    f"({format_file_size(file_size)} > {settings.MAX_PDF_SIZE_MB} MB)"
                )
            
            logger.info(f"PDF size validation passed: {os.path.basename(pdf_path)} ({format_file_size(file_size)})")
            return True
            
        except OSError as e:
            raise ValidationError(f"Erreur d'accès au fichier {pdf_path}: {e}")
    
    @staticmethod
    def validate_pdf_readable(pdf_path: str) -> bool:
        """
        Vérifie qu'un fichier PDF peut être lu et n'est pas corrompu
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            True si le PDF est lisible
            
        Raises:
            PDFProcessingError: Si le PDF ne peut pas être lu
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Vérifier qu'il y a au moins une page
                if len(pdf.pages) == 0:
                    raise PDFProcessingError(f"PDF sans pages: {os.path.basename(pdf_path)}")
                
                # Tester la lecture de la première page
                first_page = pdf.pages[0]
                test_text = first_page.extract_text()
                
                # Le texte peut être None ou vide pour des PDF d'images
                logger.info(f"PDF readability test passed: {os.path.basename(pdf_path)} ({len(pdf.pages)} pages)")
                return True
                
        except Exception as e:
            raise PDFProcessingError(f"Impossible de lire le PDF {os.path.basename(pdf_path)}: {e}")
    
    @staticmethod
    def validate_pdf_has_content(pdf_path: str) -> Tuple[bool, Dict]:
        """
        Vérifie qu'un PDF contient du texte extractible
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Tuple (is_valid, stats) avec les statistiques du contenu
            
        Raises:
            PDFProcessingError: Si le PDF ne peut pas être analysé
        """
        try:
            stats = {
                'total_pages': 0,
                'pages_with_text': 0,
                'total_chars': 0,
                'avg_chars_per_page': 0,
                'has_sufficient_content': False
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                stats['total_pages'] = len(pdf.pages)
                
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) >= settings.MIN_PAGE_CONTENT_LENGTH:
                            stats['pages_with_text'] += 1
                            stats['total_chars'] += len(page_text.strip())
                    except Exception as e:
                        logger.warning(f"Erreur extraction page dans {os.path.basename(pdf_path)}: {e}")
                        continue
                
                # Calculer les moyennes
                if stats['pages_with_text'] > 0:
                    stats['avg_chars_per_page'] = stats['total_chars'] / stats['pages_with_text']
                
                # Déterminer si le contenu est suffisant
                stats['has_sufficient_content'] = (
                    stats['pages_with_text'] > 0 and 
                    stats['total_chars'] > 100  # Au moins 100 caractères au total
                )
                
                logger.info(
                    f"PDF content analysis: {os.path.basename(pdf_path)} - "
                    f"{stats['pages_with_text']}/{stats['total_pages']} pages with text, "
                    f"{stats['total_chars']} total chars"
                )
                
                return stats['has_sufficient_content'], stats
                
        except Exception as e:
            raise PDFProcessingError(f"Erreur analyse contenu PDF {os.path.basename(pdf_path)}: {e}")
    
    @classmethod
    def validate_pdf_file(cls, pdf_path: str) -> Tuple[bool, Dict]:
        """
        Validation complète d'un fichier PDF
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Tuple (is_valid, validation_report)
        """
        report = {
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path),
            'file_size': 0,
            'exists': False,
            'readable': False,
            'has_content': False,
            'validation_errors': [],
            'content_stats': {}
        }
        
        try:
            # 1. Vérifier l'existence
            cls.validate_file_exists(pdf_path)
            report['exists'] = True
            report['file_size'] = os.path.getsize(pdf_path)
            
            # 2. Vérifier la taille
            cls.validate_file_size(pdf_path)
            
            # 3. Vérifier la lisibilité
            cls.validate_pdf_readable(pdf_path)
            report['readable'] = True
            
            # 4. Vérifier le contenu
            has_content, content_stats = cls.validate_pdf_has_content(pdf_path)
            report['has_content'] = has_content
            report['content_stats'] = content_stats
            
            is_valid = report['exists'] and report['readable'] and report['has_content']
            
            if is_valid:
                logger.info(f"PDF validation successful: {os.path.basename(pdf_path)}")
            else:
                logger.warning(f"PDF validation failed: {os.path.basename(pdf_path)}")
            
            return is_valid, report
            
        except (ValidationError, PDFProcessingError) as e:
            report['validation_errors'].append(str(e))
            logger.error(f"PDF validation error: {e}")
            return False, report
        
        except Exception as e:
            error_msg = f"Erreur inattendue lors de la validation: {e}"
            report['validation_errors'].append(error_msg)
            logger.error(error_msg)
            return False, report

class DataValidator:
    """Validateur pour les données de l'application"""
    
    @staticmethod
    def validate_drug_list(drugs: List[str]) -> Tuple[bool, List[str]]:
        """
        Valide une liste de médicaments
        
        Args:
            drugs: Liste des noms de médicaments
            
        Returns:
            Tuple (is_valid, cleaned_drugs)
        """
        if not drugs:
            return False, []
        
        cleaned_drugs = []
        
        for drug in drugs:
            if not isinstance(drug, str):
                continue
            
            # Nettoyer le nom du médicament
            cleaned = drug.strip()
            if len(cleaned) < 2:  # Nom trop court
                continue
            
            if len(cleaned) > 100:  # Nom trop long
                cleaned = cleaned[:100]
            
            cleaned_drugs.append(cleaned)
        
        # Supprimer les doublons en préservant l'ordre
        seen = set()
        unique_drugs = []
        for drug in cleaned_drugs:
            if drug.lower() not in seen:
                seen.add(drug.lower())
                unique_drugs.append(drug)
        
        is_valid = len(unique_drugs) >= 2
        return is_valid, unique_drugs
    
    @staticmethod
    def validate_search_query(query: str) -> Tuple[bool, str]:
        """
        Valide une requête de recherche
        
        Args:
            query: Requête de recherche
            
        Returns:
            Tuple (is_valid, cleaned_query)
        """
        if not isinstance(query, str):
            return False, ""
        
        cleaned = query.strip()
        
        # Vérifications de base
        if len(cleaned) < 3:
            return False, cleaned
        
        if len(cleaned) > 200000:
            cleaned = cleaned[:200000]
        
        return True, cleaned
    
    @staticmethod
    def validate_export_format(format_name: str) -> bool:
        """
        Valide un format d'export
        
        Args:
            format_name: Nom du format ('csv' ou 'excel')
            
        Returns:
            True si le format est supporté
        """
        return format_name.lower() in ['csv', 'excel']

class ConfigValidator:
    """Validateur pour la configuration"""
    
    @staticmethod
    def validate_api_keys() -> Tuple[bool, List[str]]:
        """
        Valide les clés API Google
        
        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []
        
        try:
            keys = settings.get_api_keys()
            
            if not keys:
                errors.append("Aucune clé API trouvée")
                return False, errors
            
            # Vérifier le format des clés (basique)
            for i, key in enumerate(keys):
                if not isinstance(key, str) or len(key) < 20:
                    errors.append(f"Clé API {i+1} semble invalide (trop courte)")
            
            if errors:
                return False, errors
            
            logger.info(f"API keys validation passed: {len(keys)} keys found")
            return True, []
            
        except Exception as e:
            errors.append(f"Erreur validation clés API: {e}")
            return False, errors
    
    @staticmethod
    def validate_directories() -> Tuple[bool, List[str]]:
        """
        Valide l'existence des répertoires requis
        
        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []
        required_dirs = [
            settings.DATA_DIR,
            settings.GUIDELINES_DIR,
            settings.CACHE_DIR
        ]
        
        for dir_path in required_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                if not os.path.exists(dir_path):
                    errors.append(f"Impossible de créer le répertoire: {dir_path}")
                elif not os.access(dir_path, os.W_OK):
                    errors.append(f"Pas de permission d'écriture: {dir_path}")
            except Exception as e:
                errors.append(f"Erreur avec le répertoire {dir_path}: {e}")
        
        return len(errors) == 0, errors

def validate_pdf_batch(pdf_paths: List[str]) -> Dict[str, any]:
    """
    Valide un lot de fichiers PDF
    
    Args:
        pdf_paths: Liste des chemins vers les fichiers PDF
        
    Returns:
        Rapport de validation du lot
    """
    report = {
        'total_files': len(pdf_paths),
        'valid_files': 0,
        'invalid_files': 0,
        'validation_details': {},
        'summary': {
            'total_size': 0,
            'total_pages': 0,
            'total_chars': 0
        }
    }
    
    for pdf_path in pdf_paths:
        is_valid, file_report = PDFValidator.validate_pdf_file(pdf_path)
        
        report['validation_details'][pdf_path] = file_report
        
        if is_valid:
            report['valid_files'] += 1
            report['summary']['total_size'] += file_report['file_size']
            if 'content_stats' in file_report:
                stats = file_report['content_stats']
                report['summary']['total_pages'] += stats.get('total_pages', 0)
                report['summary']['total_chars'] += stats.get('total_chars', 0)
        else:
            report['invalid_files'] += 1
    
    logger.info(
        f"PDF batch validation completed: {report['valid_files']}/{report['total_files']} valid files"
    )
    
    return report
