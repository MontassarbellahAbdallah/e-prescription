"""
Processeur PDF optimisé pour l'extraction de texte avec métadonnées
"""
import os
import pdfplumber
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import settings
from config.logging_config import get_logger
from core.exceptions import PDFProcessingError
from data.validators import PDFValidator, validate_pdf_batch, DataValidator
from utils.helpers import format_file_size, format_timestamp, measure_execution_time

logger = get_logger(__name__)

class PDFProcessor:
    """
    Processeur PDF robuste avec extraction de métadonnées
    
    Fonctionnalités:
    - Extraction de texte avec métadonnées détaillées
    - Validation automatique des fichiers
    - Traitement parallèle (optionnel)
    - Gestion d'erreurs avancée
    - Statistiques de traitement
    """
    
    def __init__(self, validate_files: bool = True, parallel_processing: bool = False):
        """
        Initialise le processeur PDF
        
        Args:
            validate_files: Valider les fichiers avant traitement
            parallel_processing: Utiliser le traitement parallèle
        """
        self.validate_files = validate_files
        self.parallel_processing = parallel_processing
        self.processing_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_pages': 0,
            'total_characters': 0,
            'processing_time': 0.0
        }
    
    @measure_execution_time
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict, Dict]:
        """
        Extrait le texte d'un seul fichier PDF avec métadonnées
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Tuple (texte_extrait, métadonnées, statistiques_document)
            
        Raises:
            PDFProcessingError: Si l'extraction échoue
        """
        if self.validate_files:
            is_valid, validation_report = PDFValidator.validate_pdf_file(pdf_path)
            if not is_valid:
                errors = validation_report.get('validation_errors', ['Validation failed'])
                raise PDFProcessingError(f"PDF invalid: {'; '.join(errors)}")
        
        try:
            text_content = ""
            metadata = {}
            doc_name = os.path.basename(pdf_path)
            chunk_counter = 0
            page_count = 0
            total_chars = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF: {doc_name} ({len(pdf.pages)} pages)")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extraction améliorée avec options
                        page_text = page.extract_text(
                            x_tolerance=2,  # Tolérance horizontale
                            y_tolerance=3,  # Tolérance verticale
                            layout=True,    # Préserver la mise en page
                            x_density=7.25, # Densité pour grouper les caractères
                            y_density=13    # Densité pour les lignes
                        )
                        
                        if page_text and len(page_text.strip()) >= settings.MIN_PAGE_CONTENT_LENGTH:
                            # Nettoyer le texte
                            cleaned_text = self._clean_extracted_text(page_text)
                            
                            # Créer les métadonnées pour cette page
                            page_metadata = {
                                'document': doc_name,
                                'page': page_num,
                                'text_length': len(cleaned_text),
                                'extraction_time': datetime.now().isoformat(),
                                'file_path': pdf_path,
                                'file_size': os.path.getsize(pdf_path)
                            }
                            
                            # Ajouter les métadonnées au dictionnaire global
                            metadata[chunk_counter] = page_metadata
                            
                            # Ajouter le texte avec marqueurs de métadonnées
                            text_content += f"\n[DOC_META_{chunk_counter}]{cleaned_text}[/DOC_META_{chunk_counter}]\n"
                            
                            chunk_counter += 1
                            page_count += 1
                            total_chars += len(cleaned_text)
                        
                        else:
                            logger.debug(f"Page {page_num} skipped (insufficient content)")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {doc_name}: {e}")
                        continue
                
                # Statistiques du document
                doc_stats = {
                    'total_pages': len(pdf.pages),
                    'processed_pages': page_count,
                    'total_characters': total_chars,
                    'file_size': os.path.getsize(pdf_path),
                    'processing_time': datetime.now().isoformat()
                }
                
                if page_count > 0:
                    logger.info(
                        f"✅ {doc_name}: {page_count}/{len(pdf.pages)} pages processed, "
                        f"{total_chars} characters extracted"
                    )
                else:
                    logger.warning(f"⚠️ {doc_name}: No readable content found")
                
                return text_content, metadata, doc_stats
                
        except Exception as e:
            error_msg = f"Critical error processing {os.path.basename(pdf_path)}: {e}"
            logger.error(error_msg)
            raise PDFProcessingError(error_msg)
    
    def extract_with_enriched_metadata(self, pdf_path: str) -> Tuple[str, Dict, Dict]:
        """
        Extrait le texte d'un PDF avec métadonnées enrichies pour traçabilité
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Tuple (texte_extrait, métadonnées_enrichies, statistiques_document)
        """
        if self.validate_files:
            is_valid, validation_report = PDFValidator.validate_pdf_file(pdf_path)
            if not is_valid:
                errors = validation_report.get('validation_errors', ['Validation failed'])
                raise PDFProcessingError(f"PDF invalid: {'; '.join(errors)}")
        
        try:
            text_content = ""
            metadata = {}
            doc_name = os.path.basename(pdf_path)
            chunk_counter = 0
            page_count = 0
            total_chars = 0
            
            # Déterminer le type de guideline depuis le nom de fichier
            guideline_type = self._detect_guideline_type(doc_name)
            
            # Charger la liste des molécules pour la détection
            molecule_list = self._load_molecule_list_for_detection()
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF with enriched metadata: {doc_name} ({len(pdf.pages)} pages)")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extraction améliorée avec options
                        page_text = page.extract_text(
                            x_tolerance=2,  # Tolérance horizontale
                            y_tolerance=3,  # Tolérance verticale
                            layout=True,    # Préserver la mise en page
                            x_density=7.25, # Densité pour grouper les caractères
                            y_density=13    # Densité pour les lignes
                        )
                        
                        if page_text and len(page_text.strip()) >= settings.MIN_PAGE_CONTENT_LENGTH:
                            # Nettoyer le texte
                            cleaned_text = self._clean_extracted_text(page_text)
                            
                            # Détection automatique des sections
                            section_info = self._detect_sections(cleaned_text, page_num)
                            
                            # Détection des mentions de molécules
                            drug_mentions = self._detect_drug_mentions(cleaned_text, molecule_list)
                            
                            # Mots-clés d'interaction (basés sur les molécules détectées)
                            interaction_keywords = drug_mentions  # Les molécules sont les mots-clés
                            
                            # Score de confiance basé sur la qualité du texte et des détections
                            confidence_score = self._calculate_confidence_score(
                                cleaned_text, drug_mentions, section_info
                            )
                            
                            # Créer les métadonnées enrichies pour cette page
                            page_metadata = {
                                'source_file': doc_name,
                                'page': page_num,
                                'section_title': section_info['title'],
                                'section_type': section_info['type'],
                                'drug_mentions': drug_mentions,
                                'interaction_keywords': interaction_keywords,
                                'guideline_type': guideline_type,
                                'confidence_score': confidence_score,
                                'text_length': len(cleaned_text),
                                'extraction_time': datetime.now().isoformat(),
                                'file_path': pdf_path,
                                'file_size': os.path.getsize(pdf_path),
                                'exact_quote_context': self._prepare_quote_context(cleaned_text)
                            }
                            
                            # Ajouter les métadonnées au dictionnaire global
                            metadata[chunk_counter] = page_metadata
                            
                            # Ajouter le texte avec marqueurs de métadonnées
                            text_content += f"\n[DOC_META_{chunk_counter}]{cleaned_text}[/DOC_META_{chunk_counter}]\n"
                            
                            chunk_counter += 1
                            page_count += 1
                            total_chars += len(cleaned_text)
                        
                        else:
                            logger.debug(f"Page {page_num} skipped (insufficient content)")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {doc_name}: {e}")
                        continue
                
                # Statistiques du document
                doc_stats = {
                    'total_pages': len(pdf.pages),
                    'processed_pages': page_count,
                    'total_characters': total_chars,
                    'file_size': os.path.getsize(pdf_path),
                    'processing_time': datetime.now().isoformat(),
                    'guideline_type': guideline_type,
                    'total_drug_mentions': sum(len(meta.get('drug_mentions', [])) for meta in metadata.values()),
                    'avg_confidence_score': sum(meta.get('confidence_score', 0) for meta in metadata.values()) / max(len(metadata), 1)
                }
                
                if page_count > 0:
                    logger.info(
                        f"✅ {doc_name}: {page_count}/{len(pdf.pages)} pages processed, "
                        f"{total_chars} characters extracted, "
                        f"{doc_stats['total_drug_mentions']} drug mentions found"
                    )
                else:
                    logger.warning(f"⚠️ {doc_name}: No readable content found")
                
                return text_content, metadata, doc_stats
                
        except Exception as e:
            error_msg = f"Critical error processing {os.path.basename(pdf_path)}: {e}"
            logger.error(error_msg)
            raise PDFProcessingError(error_msg)
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Nettoie le texte extrait du PDF de manière plus robuste
        
        Args:
            text: Texte brut extrait
            
        Returns:
            Texte nettoyé et bien structuré
        """
        if not text:
            return ""
        
        import re
        
        # Supprimer les caractères de contrôle mais garder les sauts de ligne
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Corriger les mots coupés (ex: "word-\nword" -> "wordword")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Normaliser les espaces mais garder la structure
        text = re.sub(r'[ \t]+', ' ', text)  # Espaces multiples -> un seul
        
        # Garder max 2 sauts de ligne consécutifs
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Nettoyer les tirets de bullet points mal formatés
        text = re.sub(r'^\s*[-•·]\s*', '• ', text, flags=re.MULTILINE)
        
        # Corriger les numérotations cassées
        text = re.sub(r'(\d+)\s*\.\s*([A-Z])', r'\1. \2', text)
        
        return text.strip()
    
    def _detect_guideline_type(self, filename: str) -> str:
        """
        Détermine le type de guideline depuis le nom de fichier
        
        Args:
            filename: Nom du fichier PDF
            
        Returns:
            Type de guideline détecté
        """
        filename_lower = filename.lower()
        
        # Patterns de détection
        if 'stopp' in filename_lower or 'start' in filename_lower:
            return 'STOPP/START'
        elif 'beers' in filename_lower or 'ags' in filename_lower:
            return 'BEERS_CRITERIA'
        elif 'laroche' in filename_lower:
            return 'LAROCHE_LIST'
        elif 'priscus' in filename_lower:
            return 'PRISCUS_LIST'
        elif 'onc' in filename_lower or 'ddi' in filename_lower:
            return 'ONC_DDI'
        else:
            return 'UNKNOWN_GUIDELINE'
    
    def _load_molecule_list_for_detection(self) -> set:
        """
        Charge la liste des molécules depuis le CSV pour la détection
        
        Returns:
            Set des noms de molécules (normalisés)
        """
        import pandas as pd
        
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "mimic_molecule", "molecule_unique_mimic.csv"
        )
        
        molecules = set()
        
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                if 'Molécule' in df.columns:
                    for molecule in df['Molécule'].dropna():
                        clean_name = molecule.strip().lower()
                        if clean_name:
                            molecules.add(clean_name)
                    
                    logger.debug(f"Loaded {len(molecules)} molecules for detection")
                else:
                    logger.warning(f"Column 'Molécule' not found in CSV")
            else:
                logger.warning(f"Molecule CSV not found at: {csv_path}")
                
        except Exception as e:
            logger.error(f"Error loading molecule list for detection: {e}")
            
        return molecules
    
    def _detect_sections(self, text: str, page_num: int) -> Dict[str, str]:
        """
        Détecte automatiquement les sections dans le texte
        
        Args:
            text: Texte de la page
            page_num: Numéro de page
            
        Returns:
            Informations sur la section
        """
        import re
        
        # Patterns pour détecter les sections
        section_patterns = {
            'interaction': r'\b(interaction|drug.{0,10}interaction|contraindication)\b',
            'dosage': r'\b(dosage|dose|posologie|administration)\b',
            'indication': r'\b(indication|treatment|traitement|thérapie)\b',
            'warning': r'\b(warning|attention|avertissement|précaution)\b',
            'contraindication': r'\b(contraindication|contre.indication)\b',
            'adverse_effect': r'\b(adverse|effect|side.effect|effet.indésirable)\b'
        }
        
        # Chercher les titres potentiels (lignes courtes en majuscules ou avec formatting spécial)
        lines = text.split('\n')
        potential_titles = []
        
        for line in lines[:10]:  # Analyser les 10 premières lignes
            line = line.strip()
            if len(line) < 100 and len(line) > 5:  # Titre potentiel
                if line.isupper() or line.count(' ') < 5:  # Titre en majuscules ou court
                    potential_titles.append(line)
        
        # Déterminer le type de section
        text_lower = text.lower()
        section_type = 'general'
        
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                section_type = section_name
                break
        
        # Titre de section (prendre le premier titre potentiel ou générer)
        section_title = potential_titles[0] if potential_titles else f"Page {page_num}"
        
        return {
            'title': section_title[:100],  # Limiter la longueur
            'type': section_type
        }
    
    def _detect_drug_mentions(self, text: str, molecule_list: set) -> List[str]:
        """
        Détecte les mentions de molécules dans le texte
        
        Args:
            text: Texte à analyser
            molecule_list: Liste des molécules connues
            
        Returns:
            Liste des molécules détectées
        """
        import re
        
        found_molecules = set()
        text_lower = text.lower()
        
        # Recherche exacte dans la liste CSV
        for molecule in molecule_list:
            molecule_clean = molecule.strip()
            if not molecule_clean:
                continue
                
            # Pattern de correspondance exacte (mot entier)
            pattern = r'\b' + re.escape(molecule_clean) + r'\b'
            if re.search(pattern, text_lower):
                # Retrouver la forme originale
                found_molecules.add(molecule_clean.title())  # Mettre en forme title case
        
        return sorted(list(found_molecules))
    
    def _calculate_confidence_score(self, text: str, drug_mentions: List[str], section_info: Dict) -> float:
        """
        Calcule un score de confiance pour le contenu extrait
        
        Args:
            text: Texte extrait
            drug_mentions: Molécules détectées
            section_info: Informations sur la section
            
        Returns:
            Score de confiance entre 0 et 1
        """
        score = 0.0
        
        # Facteur 1: Longueur du texte (plus c'est long, plus c'est fiable)
        length_score = min(len(text) / 1000, 1.0)  # Normaliser à 1000 caractères
        score += length_score * 0.3
        
        # Facteur 2: Nombre de molécules détectées
        drug_score = min(len(drug_mentions) / 5, 1.0)  # Normaliser à 5 molécules
        score += drug_score * 0.4
        
        # Facteur 3: Type de section (certaines sections sont plus fiables)
        section_scores = {
            'interaction': 1.0,
            'contraindication': 0.9,
            'warning': 0.8,
            'indication': 0.7,
            'dosage': 0.6,
            'adverse_effect': 0.5,
            'general': 0.3
        }
        section_score = section_scores.get(section_info.get('type', 'general'), 0.3)
        score += section_score * 0.3
        
        return round(min(score, 1.0), 3)  # Assurer que le score reste entre 0 et 1
    
    def _prepare_quote_context(self, text: str) -> Dict[str, str]:
        """
        Prépare le contexte pour les citations exactes (100 caractères avant/après)
        
        Args:
            text: Texte complet
            
        Returns:
            Contexte préparé pour les citations
        """
        # Pour l'instant, retourner le texte complet
        # L'extraction de citation exacte se fera lors de la recherche
        return {
            'full_text': text,
            'prepared_for_citation': True
        }
    
    @measure_execution_time
    def extract_text_from_multiple_pdfs(self, pdf_paths: List[str]) -> Tuple[str, Dict, Dict]:
        """
        Extrait le texte de plusieurs fichiers PDF
        
        Args:
            pdf_paths: Liste des chemins vers les fichiers PDF
            
        Returns:
            Tuple (texte_combiné, métadonnées_combinées, statistiques_globales)
        """
        if not pdf_paths:
            logger.warning("No PDF files provided for processing")
            return "", {}, self.processing_stats
        
        # Validation par lot si demandée
        if self.validate_files:
            validation_report = validate_pdf_batch(pdf_paths)
            valid_files = [
                path for path, details in validation_report['validation_details'].items()
                if not details['validation_errors']
            ]
            
            if not valid_files:
                raise PDFProcessingError("No valid PDF files found after validation")
            
            logger.info(f"Validation passed: {len(valid_files)}/{len(pdf_paths)} files valid")
            pdf_paths = valid_files
        
        # Réinitialiser les statistiques
        self.processing_stats = {
            'total_files': len(pdf_paths),
            'successful_files': 0,
            'failed_files': 0,
            'total_pages': 0,
            'total_characters': 0,
            'processing_time': 0.0
        }
        
        combined_text = ""
        combined_metadata = {}
        chunk_offset = 0
        
        if self.parallel_processing and len(pdf_paths) > 1:
            # Traitement parallèle (garde la méthode normale)
            results = self._process_pdfs_parallel(pdf_paths)
        else:
            # Traitement séquentiel avec métadonnées enrichies
            results = self._process_pdfs_sequential_enriched(pdf_paths)
        
        # Combiner les résultats
        for pdf_path, result in results.items():
            if result['success']:
                text, metadata, doc_stats = result['data']
                
                # Ajouter le texte
                combined_text += text
                
                # Réajuster les clés des métadonnées pour éviter les conflits
                for original_key, meta_data in metadata.items():
                    new_key = chunk_offset + original_key
                    combined_metadata[new_key] = meta_data
                
                chunk_offset += len(metadata)
                
                # Mettre à jour les statistiques
                self.processing_stats['successful_files'] += 1
                self.processing_stats['total_pages'] += doc_stats['processed_pages']
                self.processing_stats['total_characters'] += doc_stats['total_characters']
            
            else:
                self.processing_stats['failed_files'] += 1
                logger.error(f"Failed to process {os.path.basename(pdf_path)}: {result['error']}")
        
        logger.info(
            f"PDF processing completed: {self.processing_stats['successful_files']}/{self.processing_stats['total_files']} files successful, "
            f"{self.processing_stats['total_pages']} pages, {self.processing_stats['total_characters']} characters"
        )
        
        return combined_text, combined_metadata, self.processing_stats
    
    def _process_pdfs_sequential(self, pdf_paths: List[str]) -> Dict[str, Dict]:
        """Traitement séquentiel des PDF"""
        results = {}
        
        for pdf_path in pdf_paths:
            try:
                text, metadata, doc_stats = self.extract_text_from_pdf(pdf_path)
                results[pdf_path] = {
                    'success': True,
                    'data': (text, metadata, doc_stats)
                }
            except Exception as e:
                results[pdf_path] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def _process_pdfs_sequential_enriched(self, pdf_paths: List[str]) -> Dict[str, Dict]:
        """Traitement séquentiel avec métadonnées enrichies"""
        results = {}
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                logger.info(f"Processing PDF {i}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
                text, metadata, doc_stats = self.extract_with_enriched_metadata(pdf_path)
                results[pdf_path] = {
                    'success': True,
                    'data': (text, metadata, doc_stats)
                }
                logger.info(f"✅ PDF {i}/{len(pdf_paths)} completed successfully")
            except Exception as e:
                logger.error(f"⚠️ Problem with PDF {i}/{len(pdf_paths)} [{os.path.basename(pdf_path)}]: {e}")
                results[pdf_path] = {
                    'success': False,
                    'error': str(e)
                }
                # Continuer avec le PDF suivant (comme demandé)
                continue
        
        return results
    
    def _process_pdfs_parallel(self, pdf_paths: List[str], max_workers: int = 3) -> Dict[str, Dict]:
        """Traitement parallèle des PDF"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumettre les tâches
            future_to_path = {
                executor.submit(self.extract_text_from_pdf, pdf_path): pdf_path
                for pdf_path in pdf_paths
            }
            
            # Collecter les résultats
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    text, metadata, doc_stats = future.result()
                    results[pdf_path] = {
                        'success': True,
                        'data': (text, metadata, doc_stats)
                    }
                except Exception as e:
                    results[pdf_path] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return results
    
    def get_processing_summary(self) -> str:
        """
        Retourne un résumé du traitement
        
        Returns:
            Résumé formaté du traitement
        """
        stats = self.processing_stats
        
        summary = f"""
📊 Résumé du traitement PDF:
• Fichiers traités: {stats['successful_files']}/{stats['total_files']}
• Pages traitées: {stats['total_pages']}
• Caractères extraits: {stats['total_characters']:,}
• Fichiers en erreur: {stats['failed_files']}
"""
        
        if stats['total_characters'] > 0:
            avg_chars_per_page = stats['total_characters'] / max(stats['total_pages'], 1)
            summary += f"• Moyenne par page: {avg_chars_per_page:.0f} caractères\n"
        
        return summary.strip()

class PDFTextChunker:
    """
    Gestionnaire de chunking pour le texte extrait des PDF
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialise le chunker
        
        Args:
            chunk_size: Taille des chunks (utilise settings par défaut)
            chunk_overlap: Chevauchement entre chunks (utilise settings par défaut)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    def create_chunks_with_metadata(self, text: str, metadata: Dict) -> Tuple[List[str], Dict]:
        """
        Crée des chunks de texte en préservant les métadonnées
        
        Args:
            text: Texte à chunker
            metadata: Métadonnées associées
            
        Returns:
            Tuple (liste_des_chunks, métadonnées_des_chunks)
        """
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return [], {}
        
        # Import des outils de chunking
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from utils.constants import TEXT_SEPARATORS
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=TEXT_SEPARATORS
        )
        
        chunks = text_splitter.split_text(text)
        chunk_metadata = {}
        
        logger.info(f"Text chunking: {len(text)} chars → {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            # Trouver les métadonnées associées à ce chunk
            best_match = self._find_chunk_metadata(chunk, metadata)
            
            chunk_metadata[i] = best_match or {
                'document': 'Document inconnu',
                'page': 'Page inconnue',
                'text_length': len(chunk),
                'chunk_index': i,
                'chunk_size': self.chunk_size
            }
        
        logger.info(f"Chunking completed: {len(chunks)} chunks created")
        return chunks, chunk_metadata
    
    def _find_chunk_metadata(self, chunk: str, metadata: Dict) -> Optional[Dict]:
        """
        Trouve les métadonnées correspondant à un chunk
        
        Args:
            chunk: Chunk de texte
            metadata: Métadonnées disponibles
            
        Returns:
            Métadonnées correspondantes ou None
        """
        import re
        
        # Chercher les marqueurs de métadonnées dans le chunk
        pattern = r'\[DOC_META_(\d+)\]'
        matches = re.findall(pattern, chunk)
        
        if matches:
            # Utiliser le premier marqueur trouvé
            meta_id = int(matches[0])
            return metadata.get(meta_id)
        
        return None

# Fonctions utilitaires pour compatibilité
def get_pdf_text_with_metadata_robust(pdf_paths: List[str]) -> Tuple[str, Dict]:
    """
    Fonction de compatibilité avec l'ancienne interface
    
    Args:
        pdf_paths: Liste des chemins PDF
        
    Returns:
        Tuple (texte, métadonnées)
    """
    processor = PDFProcessor(validate_files=True, parallel_processing=False)
    text, metadata, _ = processor.extract_text_from_multiple_pdfs(pdf_paths)
    return text, metadata

def get_text_chunks_with_metadata(text: str, metadata: Dict) -> Tuple[List[str], Dict]:
    """
    Fonction de compatibilité pour le chunking
    
    Args:
        text: Texte à chunker
        metadata: Métadonnées
        
    Returns:
        Tuple (chunks, métadonnées_chunks)
    """
    chunker = PDFTextChunker()
    return chunker.create_chunks_with_metadata(text, metadata)
