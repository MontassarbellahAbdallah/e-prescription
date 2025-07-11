"""
Analyseur LLM optimisé pour l'extraction de médicaments et l'analyse d'interactions
"""
import time
import os
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from itertools import combinations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from ai.dosage_analyzer import DosageAnalyzer
from ai.contraindication_analyzer import ContraindicationAnalyzer
from ai.redundancy_analyzer import RedundancyAnalyzer
from config.settings import settings
from config.logging_config import get_logger
from core.exceptions import LLMAnalysisError
from core.key_manager import get_key_manager
from core.cache_manager import get_cache_manager
from utils.constants import PROMPT_TEMPLATES, INTERACTION_LEVELS
from utils.helpers import (
    classify_interaction_level, 
    parse_llm_response, 
    clean_drug_name,
    estimate_analysis_time,
    measure_execution_time,
)

logger = get_logger(__name__)

class DrugExtractor:
    """
    Extracteur de noms de médicaments à partir de texte naturel
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialise l'extracteur de médicaments
        
        Args:
            use_cache: Utiliser le cache pour les extractions
        """
        self.key_manager = get_key_manager()
        self.cache_manager = get_cache_manager() if use_cache else None
        self.model = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
        
        # Charger la liste des molécules depuis le CSV
        self.molecule_list = self._load_molecule_list()
        logger.info(f"Loaded {len(self.molecule_list)} molecules from CSV for CSV-only extraction method")
    
    def _load_molecule_list(self) -> set:
        """
        Charge la liste des molécules depuis le fichier CSV
        
        Returns:
            Set des noms de molécules (normalisés en minuscules)
        """
        import pandas as pd
        import os
        
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  # Remonter de ai/ vers racine
            "data", "mimic_molecule", "molecule_unique_mimic.csv"
        )
        
        molecules = set()
        
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Normaliser les noms de molécules
                if 'Molécule' in df.columns:
                    for molecule in df['Molécule'].dropna():
                        # Ajouter le nom original (nettoyé)
                        clean_name = clean_drug_name(str(molecule))
                        if clean_name:
                            molecules.add(clean_name.lower())
                            
                            # Ajouter aussi des variations communes
                            # Ex: "Sodium chloride" -> ["sodium chloride", "nacl"]
                            variations = self._get_molecule_variations(clean_name)
                            molecules.update(v.lower() for v in variations)
                    
                    logger.info(f"Successfully loaded {len(molecules)} molecule names and variations for CSV-only method")
                else:
                    logger.warning(f"Column 'Molécule' not found in CSV. Available columns: {df.columns.tolist()}")
            else:
                logger.warning(f"Molecule CSV not found at: {csv_path}")
                
        except Exception as e:
            logger.error(f"Error loading molecule list: {e}")
            
        return molecules
    
    def _get_molecule_variations(self, molecule_name: str) -> list:
        """
        Génère des variations communes d'un nom de molécule
        
        Args:
            molecule_name: Nom de base de la molécule
            
        Returns:
            Liste des variations possibles
        """
        variations = [molecule_name]
        name_lower = molecule_name.lower()
        
        # Variations communes pour certaines molécules
        common_variations = {
            'acetylsalicylic acid': ['aspirin', 'asa'],
            'sodium chloride': ['nacl', 'saline', 'normal saline'],
            'potassium chloride': ['kcl'],
            'magnesium sulfate': ['mgso4', 'epsom salt'],
            'calcium chloride': ['cacl2'],
            'sodium bicarbonate': ['nahco3', 'bicarbonate'],
            'morphine sulfate': ['morphine'],
            'heparin sodium': ['heparin'],
            'insulin': ['insulin human', 'regular insulin']
        }
        
        # Chercher des correspondances
        for key, vars_list in common_variations.items():
            if key in name_lower or name_lower in key:
                variations.extend(vars_list)
        
        # Variations automatiques
        # Retirer "sulfate", "sodium", etc.
        words = name_lower.split()
        if len(words) > 1:
            # Version sans suffixes courants
            filtered_words = [w for w in words if w not in ['sodium', 'sulfate', 'chloride', 'hydrochloride']]
            if filtered_words and len(filtered_words) < len(words):
                variations.append(' '.join(filtered_words))
        
        return list(set(variations))  # Éliminer les doublons
    
    @property
    def cached_invoke(self):
        """Wrapper pour les appels LLM avec gestion de quota"""
        return self.key_manager.wrap_quota(self.model.invoke)
    
    def _extract_from_csv_only(self, question: str) -> List[str]:
        """
        Extraction UNIQUEMENT depuis le fichier CSV (méthode 1)
        CORRESPONDANCE EXACTE SEULEMENT (pas de correspondance partielle)
        
        Args:
            question: Texte de la prescription
            
        Returns:
            Liste des molécules trouvées dans le CSV avec correspondance exacte uniquement
        """
        import re
        
        found_molecules = set()
        question_lower = question.lower()
        
        logger.debug(f"CSV-ONLY extraction (EXACT match only) analyzing text: {question_lower[:200]}...")
        
        # Recherche dans la liste CSV avec CORRESPONDANCE EXACTE SEULEMENT
        for molecule in self.molecule_list:
            molecule_clean = molecule.strip()
            if not molecule_clean:
                continue
                
            # SEULEMENT Pattern de correspondance exacte (mot entier)
            pattern = r'\b' + re.escape(molecule_clean) + r'\b'
            if re.search(pattern, question_lower):
                original_form = self._get_original_form(molecule_clean)
                found_molecules.add(original_form)
                logger.debug(f"CSV-ONLY match (EXACT): '{molecule_clean}' -> '{original_form}'")
        
        result = sorted(list(found_molecules))
        logger.info(f"CSV-ONLY extraction (EXACT only) result: {len(result)} molecules found - {result}")
        return result
    
    def _extract_with_llm_only(self, question: str) -> List[str]:
        """
        Extraction UNIQUEMENT avec LLM (méthode 2)
        Utilise SEULEMENT un prompt simple, sans détection de patterns spécialisés
        
        Args:
            question: Question à analyser
            
        Returns:
            Liste des médicaments extraits par LLM uniquement avec prompt simple
        """
        logger.info("Using LLM-ONLY extraction (simple prompt, no special patterns)")
        
        # Utiliser SEULEMENT l'extraction LLM standard avec prompt simple
        return self._extract_with_llm_standard(question)
    
    def _get_original_form(self, molecule_lower: str) -> str:
        """
        Retrouve la forme originale d'une molécule depuis le CSV
        
        Args:
            molecule_lower: Nom de molécule en minuscules
            
        Returns:
            Forme originale avec la casse correcte
        """
        import pandas as pd
        import os
        
        try:
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "mimic_molecule", "molecule_unique_mimic.csv"
            )
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'Molécule' in df.columns:
                    for original in df['Molécule'].dropna():
                        clean_original = clean_drug_name(str(original))
                        if clean_original and clean_original.lower() == molecule_lower:
                            return clean_original
        except Exception as e:
            logger.debug(f"Error getting original form for {molecule_lower}: {e}")
        
        # Fallback: capitaliser le nom
        return ' '.join(word.capitalize() for word in molecule_lower.split())
    
    @measure_execution_time
    def extract_drug_names(self, question: str) -> List[str]:
        """
        Extrait les noms de médicaments avec SEULEMENT deux méthodes:
        - MÉTHODE 1: CSV seul (vérification dans le fichier CSV uniquement)
        - MÉTHODE 2: LLM seul (extraction par l'IA uniquement)
        
        Args:
            question: Question contenant potentiellement des médicaments
            
        Returns:
            Liste des noms de médicaments extraits
            
        Raises:
            LLMAnalysisError: Si l'extraction échoue
        """
        if not question or not question.strip():
            return []
        
        # Vérifier le cache
        cache_key = f"drugs_{question}"
        if self.cache_manager:
            cached_drugs = self.cache_manager.get(cache_key, prefix="drugs_")
            if cached_drugs is not None:
                logger.info(f"Drug extraction cache hit for: {question[:50]}...")
                return cached_drugs
        
        try:
            logger.info(f"Starting SIMPLE drug extraction for: {question[:100]}...")
            
            # MÉTHODE 1: Essayer d'abord le CSV SEUL
            csv_drugs = self._extract_from_csv_only(question)
            
            if csv_drugs:
                # CSV a trouvé des molécules -> UTILISER UNIQUEMENT ÇA
                logger.info(f"CSV extraction successful: {len(csv_drugs)} drugs found - {csv_drugs}")
                final_drugs = csv_drugs
                method = 'csv_only'
            else:
                # CSV n'a rien trouvé -> UTILISER LLM SEUL
                logger.info("No molecules found in CSV, using LLM extraction")
                llm_drugs = self._extract_with_llm_only(question)
                final_drugs = llm_drugs
                method = 'llm_only'
            
            # Nettoyer les résultats finaux
            final_drugs = [clean_drug_name(drug) for drug in final_drugs if drug and len(drug.strip()) > 1]
            final_drugs = sorted(list(set(final_drugs)))  # Dédupliquer
            
            # Statistiques simples
            logger.info(
                f"SIMPLE extraction completed: {len(final_drugs)} drugs found using {method} - {final_drugs}"
            )
            
            # Mettre en cache
            if self.cache_manager:
                self.cache_manager.set(cache_key, final_drugs, prefix="drugs_")
            
            return final_drugs
            
        except Exception as e:
            error_msg = f"Drug extraction failed: {e}"
            logger.error(error_msg)
            raise LLMAnalysisError(error_msg)
    

    
    def _extract_with_llm_standard(self, question: str) -> List[str]:
        """
        Extraction LLM standard avec prompt simple
        Utilise UNIQUEMENT un prompt simple sans détection de patterns spécialisés
        
        Args:
            question: Question à analyser
            
        Returns:
            Liste des médicaments extraits par LLM avec prompt simple
        """
        logger.info("Using simple LLM extraction (no special patterns)")
        
        # Si le texte est très long (> 3000 caractères), le diviser en chunks
        if len(question) > 3000:
            logger.info(f"Long text detected ({len(question)} chars), splitting into chunks")
            return self._extract_from_long_text(question)
        else:
            # Traitement normal avec prompt simple
            prompt = PROMPT_TEMPLATES['drug_extraction_simple'].format(question=question)
            response = self.cached_invoke(prompt)
            content = response.content.strip()
            
            if content.upper() == "AUCUN":
                return []
            else:
                raw_drugs = [drug.strip() for drug in content.split(",") if drug.strip()]
                drugs = [clean_drug_name(drug) for drug in raw_drugs if len(drug.strip()) > 1]
                return sorted(list(set(drugs)))
    
    def _extract_from_long_text(self, question: str) -> List[str]:
        """
        Extrait les médicaments d'un texte long en le divisant en chunks
        Utilise UNIQUEMENT un prompt LLM simple (pas de patterns spécialisés)
        
        Args:
            question: Texte long à analyser
            
        Returns:
            Liste combinée des médicaments extraits par LLM avec prompt simple
        """
        chunk_size = 2500  # Taille sécurisée pour le LLM
        chunks = []
        
        # Diviser le texte en chunks en essayant de respecter les lignes
        lines = question.split('\n')
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk + line + '\n') > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks for simple LLM analysis")
        
        # Extraire les médicaments de chaque chunk avec prompt simple
        all_drugs = set()
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars) with simple LLM")
            
            try:
                prompt = PROMPT_TEMPLATES['drug_extraction_simple'].format(question=chunk)
                response = self.cached_invoke(prompt)
                content = response.content.strip()
                
                if content.upper() != "AUCUN":
                    raw_drugs = [drug.strip() for drug in content.split(",") if drug.strip()]
                    chunk_drugs = [clean_drug_name(drug) for drug in raw_drugs if len(drug.strip()) > 1]
                    all_drugs.update(chunk_drugs)
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk {i+1}: {e}")
                continue
        
        final_drugs = sorted(list(all_drugs))
        logger.info(f"Combined simple LLM extraction from {len(chunks)} chunks: {len(final_drugs)} unique drugs found")
        
        return final_drugs
    
    def extract_drug_names_batch(self, questions: List[str]) -> Dict[str, List[str]]:
        """
        Extrait les médicaments de plusieurs questions
        
        Args:
            questions: Liste de questions
            
        Returns:
            Dictionnaire {question: [médicaments]}
        """
        results = {}
        
        for question in questions:
            try:
                drugs = self.extract_drug_names(question)
                results[question] = drugs
            except Exception as e:
                logger.error(f"Failed to extract drugs from '{question[:50]}...': {e}")
                results[question] = []
        
        return results

class InteractionAnalyzer:
    """
    Analyseur d'interactions médicamenteuses
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialise l'analyseur d'interactions
        
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
    def analyze_single_interaction(
        self, 
        drug1: str, 
        drug2: str, 
        context_docs: Optional[List[Document]] = None
    ) -> Dict[str, str]:
        """
        Analyse l'interaction entre deux médicaments
        
        Args:
            drug1: Premier médicament
            drug2: Deuxième médicament
            context_docs: Documents de contexte pour enrichir l'analyse
            
        Returns:
            Dictionnaire avec 'level' et 'explanation'
        """
        # Créer une clé de cache basée sur la paire de médicaments
        drug_pair = tuple(sorted([drug1.lower(), drug2.lower()]))
        cache_key = f"interaction_{drug_pair[0]}_{drug_pair[1]}"
        
        # Vérifier le cache avec préfixe spécifique pour les interactions
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key, prefix="interaction_")
            if cached_result is not None:
                logger.info(f"Interaction cache hit for: {drug1} + {drug2}")
                return cached_result
        
        try:
            # Préparer le contexte
            context_text = ""
            if context_docs:
                context_text = "\n".join([
                    doc.page_content[:500] for doc in context_docs[:3]  # Limiter le contexte
                ])
            
            # Préparer le prompt
            prompt = PROMPT_TEMPLATES['interaction_analysis'].format(
                drug1=drug1,
                drug2=drug2,
                context=context_text
            )
            
            # Appeler le LLM
            response = self.cached_invoke(prompt)
            content = response.content.strip()
            
            # Parser la réponse
            result = parse_llm_response(content)
            
            # Normaliser le niveau
            result['level'] = classify_interaction_level(result['level'])
            
            # Mettre en cache avec préfixe spécifique
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, prefix="interaction_")
            
            logger.debug(f"Interaction analyzed: {drug1} + {drug2} = {result['level']}")
            return result
            
        except Exception as e:
            error_msg = f"Interaction analysis failed for {drug1} + {drug2}: {e}"
            logger.error(error_msg)
            return {
                'level': 'Erreur',
                'explanation': f'Erreur lors de l\'analyse: {str(e)[:100]}'
            }
    
    def analyze_all_combinations(
        self, 
        drugs: List[str], 
        context_docs: Optional[List[Document]] = None,
        progress_callback = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Analyse toutes les combinaisons possibles de médicaments
        
        Args:
            drugs: Liste des médicaments
            context_docs: Documents de contexte
            progress_callback: Fonction de callback pour le progrès
            
        Returns:
            Tuple (résultats_interactions, statistiques)
        """
        if len(drugs) < 2:
            return [], {
                'total_drugs': len(drugs),
                'total_combinations': 0,
                'major': 0,
                'moderate': 0,
                'minor': 0,
                'aucune': 0,  # Changé de 'none' à 'aucune'
                'errors': 0
            }
        
        # Générer toutes les combinaisons
        combinations_list = list(combinations(sorted(drugs), 2))
        
        logger.info(f"Starting interaction analysis: {len(combinations_list)} combinations to analyze")
        
        # Initialiser les résultats
        interactions_results = []
        stats = {'major': 0, 'moderate': 0, 'minor': 0, 'none': 0, 'errors': 0}
        
        start_time = time.time()
        
        for i, (drug1, drug2) in enumerate(combinations_list):
            # Callback de progression
            if progress_callback:
                progress_callback(i + 1, len(combinations_list), drug1, drug2)
            
            # Analyser l'interaction
            interaction_result = self.analyze_single_interaction(drug1, drug2, context_docs)
            
            # Classer le niveau
            level = interaction_result['level']
            if level == 'Major':
                stats['major'] += 1
            elif level == 'Moderate':
                stats['moderate'] += 1
            elif level == 'Minor':
                stats['minor'] += 1
            elif level == 'Erreur':
                stats['errors'] += 1
            else:
                stats['none'] += 1
            
            # Ajouter aux résultats
            interactions_results.append({
                'Drug1': drug1,
                'Drug2': drug2,
                'Level': level,
                'Explanation': interaction_result['explanation']
            })
        
        # Statistiques finales
        total_time = time.time() - start_time
        summary_stats = {
            'total_drugs': len(drugs),
            'total_combinations': len(combinations_list),
            'major': stats['major'],
            'moderate': stats['moderate'],
            'minor': stats['minor'],
            'aucune': stats['none'],  # Changé de 'none' à 'aucune'
            'errors': stats['errors'],
            'analysis_time': total_time,
            'avg_time_per_combination': total_time / len(combinations_list) if combinations_list else 0
        }
        
        logger.info(
            f"Interaction analysis completed in {total_time:.2f}s: "
            f"{stats['major']} major, {stats['moderate']} moderate, "
            f"{stats['minor']} minor, {stats['none']} none, {stats['errors']} errors"
        )
        
        return interactions_results, summary_stats

class DetailedExplainer:
    """
    Générateur d'explications détaillées avec sources
    """
    
    def __init__(self):
        """Initialise l'explicateur détaillé"""
        self.key_manager = get_key_manager()
        self.model = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
    
    @property
    def cached_invoke(self):
        """Wrapper pour les appels LLM avec gestion de quota"""
        return self.key_manager.wrap_quota(self.model.invoke)
    
    def get_detailed_explanation_with_sources(
        self, 
        query: str, 
        docs: List[Document], 
        sources_info: List[Dict]
    ) -> str:
        """
        Génère une explication détaillée avec sources intégrées
        
        Args:
            query: Question de l'utilisateur
            docs: Documents de contexte
            sources_info: Informations sur les sources
            
        Returns:
            Explication détaillée avec sources intégrées
        """
        if not docs:
            return "Aucune information supplémentaire trouvée dans les documents."
        
        try:
            # Préparer le contexte avec sources
            context_with_sources = self._prepare_context_with_sources(docs, sources_info)
            
            # Créer le prompt enrichi avec sources
            prompt = PromptTemplate(
                template=PROMPT_TEMPLATES['detailed_explanation_with_sources'],
                input_variables=["context", "question"]
            )
            
            # Créer la chaîne de QA
            chain = load_qa_chain(self.model, chain_type="stuff", prompt=prompt)
            
            # Générer la réponse
            response = chain({
                "input_documents": docs[:3],  # Limiter le nombre de documents
                "question": query
            }, return_only_outputs=True)
            
            explanation = response["output_text"]
            
            # Ajouter les notifications de sources (bulles cliquables)
            explanation_with_sources = self._add_source_notifications(explanation, sources_info)
            
            return explanation_with_sources
            
        except Exception as e:
            error_msg = f"Failed to generate detailed explanation with sources: {e}"
            logger.error(error_msg)
            return f"Erreur lors de la génération de l'explication: {str(e)[:100]}"
    
    def _prepare_context_with_sources(self, docs: List[Document], sources_info: List[Dict]) -> str:
        """
        Prépare le contexte en incluant les informations de sources
        
        Args:
            docs: Documents
            sources_info: Informations sur les sources
            
        Returns:
            Contexte enrichi avec sources
        """
        context_parts = []
        
        for i, (doc, source) in enumerate(zip(docs[:3], sources_info[:3])):
            # Formater chaque source avec ses métadonnées
            source_header = f"\n--- SOURCE {i+1}: {source.get('academic_citation', source.get('document', 'Source inconnue'))} ---\n"
            
            # Ajouter les métadonnées enrichies si disponibles
            if source.get('exact_quote'):
                source_header += f"Citation exacte: \"{source['exact_quote']}\"\n"
            
            if source.get('document_section'):
                source_header += f"Section: {source['document_section']}\n"
            
            if source.get('guideline_type'):
                source_header += f"Type de guideline: {source['guideline_type']}\n"
            
            source_header += "Contenu:\n"
            
            context_parts.append(source_header + doc.page_content[:500])
        
        return "\n".join(context_parts)
    
    def _add_source_notifications(self, explanation: str, sources_info: List[Dict]) -> str:
        """
        Ajoute des notifications de sources (bulles cliquables) dans l'explication
        
        Args:
            explanation: Explication générée
            sources_info: Informations sur les sources
            
        Returns:
            Explication avec notifications de sources
        """
        # Ajouter des références de sources dans le texte
        enhanced_explanation = explanation
        
        # Ajouter une section avec les bulles de sources à la fin
        if sources_info:
            enhanced_explanation += "\n\n📝 **Sources consultées** (cliquez pour détails):\n"
            
            for i, source in enumerate(sources_info[:3], 1):
                citation = source.get('academic_citation', source.get('document', f'Source {i}'))
                relevance = source.get('relevance_score', 0)
                
                # Créer une "bulle" cliquable (pour Streamlit)
                source_bubble = f"🟡 **Source {i}**: {citation} (Score: {relevance:.2f})"
                enhanced_explanation += f"\n{source_bubble}"
        
        return enhanced_explanation
    
    def _create_sources_summary(self, sources_info: List[Dict]) -> str:
        """
        Crée un résumé des sources consultées
        
        Args:
            sources_info: Informations sur les sources
            
        Returns:
            Résumé formaté des sources
        """
        if not sources_info:
            return ""
        
        sources_summary = "\n\n---\n**Sources consultées:**\n"
        
        for i, source in enumerate(sources_info, 1):
            sources_summary += (
                f"{i}. {source['document']} (Page {source['page']}) - "
                f"Score: {source['relevance_score']:.3f}\n"
            )
        
        return sources_summary

class LLMAnalyzer:
    """
    Analyseur LLM complet combinant extraction, interactions et dosage
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialise l'analyseur LLM complet
        
        Args:
            use_cache: Utiliser le cache pour toutes les opérations
        """
        self.drug_extractor = DrugExtractor(use_cache)
        self.interaction_analyzer = InteractionAnalyzer(use_cache)
        self.detailed_explainer = DetailedExplainer()
        
        # NOUVEAU: Ajouter l'analyseur de dosage
        self.dosage_analyzer = DosageAnalyzer(use_cache)
        
        # NOUVEAU: Ajouter l'analyseur de contre-indications
        self.contraindication_analyzer = ContraindicationAnalyzer(use_cache)
        
        # NOUVEAU: Ajouter l'analyseur de redondance thérapeutique
        self.redundancy_analyzer = RedundancyAnalyzer(use_cache)
        
        # Statistiques d'utilisation
        self.usage_stats = {
            'extractions': 0,
            'interactions_analyzed': 0,
            'explanations_generated': 0,
            'dosage_analyses': 0,  # NOUVEAU
            'contraindication_analyses': 0,  # NOUVEAU
            'redundancy_analyses': 0,  # NOUVEAU
            'total_llm_calls': 0,
            'cache_hits': 0
        }
    
    # 3. Ajouter la méthode d'analyse de dosage
    
    def analyze_dosage(self, prescription: str, patient_info: str = "", context_docs=None) -> Dict:
        """
        Analyse les dosages de la prescription avec contexte vectoriel (interface simplifiée)
        
        Args:
            prescription: Texte de la prescription
            patient_info: Informations sur le patient
            context_docs: Documents de contexte de la base vectorielle
            
        Returns:
            Résultat de l'analyse de dosage
        """
        self.usage_stats['dosage_analyses'] += 1
        return self.dosage_analyzer.analyze_dosage(prescription, patient_info, context_docs)
    
    def analyze_contraindications(self, prescription: str, patient_info: str = "", context_docs=None) -> Dict:
        """
        Analyse les contre-indications de la prescription (interface simplifiée)
        
        Args:
            prescription: Texte de la prescription
            patient_info: Informations sur le patient
            context_docs: Documents de contexte de la base vectorielle
            
        Returns:
            Résultat de l'analyse de contre-indications
        """
        self.usage_stats['contraindication_analyses'] += 1
        return self.contraindication_analyzer.analyze_contraindications(prescription, patient_info, context_docs)
    
    def analyze_redundancy(self, prescription: str, patient_info: str = "", context_docs=None) -> Dict:
        """
        Analyse les redondances thérapeutiques de la prescription (interface simplifiée)
        
        Args:
            prescription: Texte de la prescription
            patient_info: Informations sur le patient
            context_docs: Documents de contexte de la base vectorielle
            
        Returns:
            Résultat de l'analyse de redondance
        """
        self.usage_stats['redundancy_analyses'] += 1
        return self.redundancy_analyzer.analyze_redundancy(prescription, patient_info, context_docs)
    
    # 4. Modifier la méthode get_usage_statistics pour inclure le dosage
    
    def get_usage_statistics(self) -> Dict:
        """
        Retourne les statistiques d'utilisation (mise à jour avec dosage)
        
        Returns:
            Statistiques d'utilisation de l'analyseur
        """
        # Ajouter les stats du gestionnaire de clés
        key_manager_stats = self.drug_extractor.key_manager.get_usage_stats()
        
        combined_stats = {
            'analyzer_stats': self.usage_stats,
            'key_manager_stats': key_manager_stats,
            'total_api_calls': key_manager_stats['total_calls'],
            'success_rate': key_manager_stats['overall_success_rate']
        }
        
        return combined_stats
    
    # 5. Ajouter une méthode pour extraire les informations patient
    
    def extract_patient_info(self, prescription: str) -> str:
        """
        Extrait les informations patient depuis la prescription
        
        Args:
            prescription: Texte de la prescription
            
        Returns:
            Informations patient formatées
        """
        import re
        
        patient_info = []
        
        # Chercher l'âge
        age_patterns = [
            r'(\d+)\s*ans?',
            r'âge[:\s]*(\d+)',
            r'patient[:\s,]*[MF],?\s*(\d+)\s*ans?'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, prescription, re.IGNORECASE)
            if match:
                age = match.group(1)
                patient_info.append(f"Âge: {age} ans")
                break
        
        # Chercher le sexe
        sex_patterns = [
            r'patient[:\s]*([MF])',
            r'([MF]),?\s*\d+\s*ans?'
        ]
        
        for pattern in sex_patterns:
            match = re.search(pattern, prescription, re.IGNORECASE)
            if match:
                sex = match.group(1).upper()
                sex_full = "Homme" if sex == "M" else "Femme"
                patient_info.append(f"Sexe: {sex_full}")
                break
        
        # Chercher des conditions médicales mentionnées
        conditions = []
        condition_keywords = [
            'diabète', 'hypertension', 'insuffisance', 'cardiopathie', 
            'BPCO', 'asthme', 'dépression', 'anxiété', 'douleur'
        ]
        
        for keyword in condition_keywords:
            if keyword.lower() in prescription.lower():
                conditions.append(keyword)
        
        if conditions:
            patient_info.append(f"Conditions: {', '.join(conditions)}")
        
        return " | ".join(patient_info) if patient_info else "Informations patient non spécifiées"
    
    # 6. Ajouter une méthode d'analyse complète
    
    def extract_drug_names(self, question: str) -> List[str]:
        """
        Extrait les noms de médicaments (interface simplifiée)
        
        Args:
            question: Question contenant des médicaments
            
        Returns:
            Liste des médicaments extraits
        """
        self.usage_stats['extractions'] += 1
        return self.drug_extractor.extract_drug_names(question)
    
    def analyze_single_interaction(
        self, 
        drug1: str, 
        drug2: str, 
        context_docs: Optional[List[Document]] = None
    ) -> Dict[str, str]:
        """
        Analyse une interaction entre deux médicaments (interface simplifiée)
        
        Args:
            drug1: Premier médicament
            drug2: Deuxième médicament
            context_docs: Documents de contexte
            
        Returns:
            Résultat de l'analyse
        """
        self.usage_stats['interactions_analyzed'] += 1
        return self.interaction_analyzer.analyze_single_interaction(drug1, drug2, context_docs)
    
    def analyze_all_combinations(
        self, 
        drugs: List[str], 
        context_docs: Optional[List[Document]] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Analyse toutes les combinaisons de médicaments (interface simplifiée)
        
        Args:
            drugs: Liste des médicaments
            context_docs: Documents de contexte
            
        Returns:
            Tuple (résultats, statistiques)
        """
        # Estimer le temps
        combinations_count = len(drugs) * (len(drugs) - 1) // 2 if len(drugs) > 1 else 0
        estimated_time = estimate_analysis_time(combinations_count)
        
        logger.info(f"Starting comprehensive analysis: {combinations_count} combinations, estimated time: {estimated_time}")
        
        return self.interaction_analyzer.analyze_all_combinations(drugs, context_docs)
    
    def get_detailed_explanation_with_sources(
        self, 
        query: str, 
        docs: List[Document], 
        sources_info: List[Dict]
    ) -> str:
        """
        Génère une explication détaillée (interface simplifiée)
        
        Args:
            query: Question de l'utilisateur
            docs: Documents de contexte
            sources_info: Informations sur les sources
            
        Returns:
            Explication détaillée
        """
        self.usage_stats['explanations_generated'] += 1
        return self.detailed_explainer.get_detailed_explanation_with_sources(query, docs, sources_info)
    
    def clear_cache(self) -> bool:
        """
        Vide le cache de l'analyseur
        
        Returns:
            True si le cache a été vidé
        """
        cache_cleared = 0
        
        if self.drug_extractor.cache_manager:
            cache_cleared += self.drug_extractor.cache_manager.clear()
        
        if self.interaction_analyzer.cache_manager:
            cache_cleared += self.interaction_analyzer.cache_manager.clear()
        
        logger.info(f"Cache cleared: {cache_cleared} entries removed")
        return cache_cleared > 0
    
    def clear_drug_extraction_cache(self) -> int:
        """
        Vide uniquement le cache d'extraction de médicaments (amélioré avec préfixes)
        
        Returns:
            Nombre d'entrées supprimées
        """
        cache_cleared = 0
        
        if self.drug_extractor.cache_manager:
            cache_manager = self.drug_extractor.cache_manager
            
            try:
                # Chercher tous les fichiers de cache qui commencent par "drugs_"
                cache_dir = cache_manager.cache_dir
                if not os.path.exists(cache_dir):
                    logger.info("No cache directory found")
                    return 0
                
                import glob
                cache_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
                
                for cache_file in cache_files:
                    try:
                        import pickle
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        
                        # Vérifier si c'est un cache d'extraction de médicaments
                        cache_type = cached_data.get('cache_type', '')
                        prefix = cached_data.get('prefix', '')
                        original_key = cached_data.get('key', '')
                        
                        # Supprimer si c'est un cache de type "drugs_" ou contient "drugs_" dans la clé
                        if (cache_type == 'drugs_' or 
                            prefix == 'drugs_' or 
                            'drugs_' in original_key or
                            original_key.startswith('drugs_')):
                            
                            os.remove(cache_file)
                            cache_cleared += 1
                            logger.debug(f"Removed drug extraction cache: {os.path.basename(cache_file)}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing cache file {cache_file}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error clearing drug extraction cache: {e}")
        
        logger.info(f"Drug extraction cache cleared: {cache_cleared} entries removed")
        return cache_cleared
    
    def analyze_prescription_complete(self, prescription: str, context_docs=None) -> Dict:
        """
        Analyse complète d'une prescription (interactions + dosage + contre-indications)
        
        Args:
            prescription: Texte de la prescription
            context_docs: Documents de contexte pour enrichir l'analyse
            
        Returns:
            Résultat d'analyse complète
        """
        logger.info("Starting complete prescription analysis (interactions + dosage + contraindications)")
        
        try:
            # 1. Extraction des médicaments
            drugs = self.extract_drug_names(prescription)
            
            # 2. Extraction des informations patient
            patient_info = self.extract_patient_info(prescription)
            
            # 3. Analyse des interactions (si au moins 2 médicaments)
            interactions_result = None
            interactions_stats = None
            
            if len(drugs) >= 2:
                interactions, interactions_stats = self.analyze_all_combinations(drugs, context_docs)
                interactions_result = {
                    'interactions': interactions,
                    'stats': interactions_stats
                }
            
            # 4. Analyse du dosage (avec contexte vectoriel)
            dosage_result = self.analyze_dosage(prescription, patient_info, context_docs)
            
            # 5. NOUVEAU: Analyse des contre-indications
            contraindication_result = self.analyze_contraindications(prescription, patient_info, context_docs)
            
            # 6. NOUVEAU: Analyse des redondances thérapeutiques
            redundancy_result = self.analyze_redundancy(prescription, patient_info, context_docs)
            
            # 7. Résultat combiné
            complete_result = {
                'drugs': drugs,
                'patient_info': patient_info,
                'interactions': interactions_result,
                'dosage': dosage_result,
                'contraindications': contraindication_result,  # NOUVEAU
                'redundancy': redundancy_result,  # NOUVEAU
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'complete'
            }
            
            logger.info(f"Complete analysis finished: {len(drugs)} drugs, "
                       f"{interactions_stats['total_combinations'] if interactions_stats else 0} interactions, "
                       f"{dosage_result['stats']['total_issues']} dosage issues, "
                       f"{contraindication_result['stats']['total_contraindications']} contraindications, "
                       f"{redundancy_result['stats']['total_redundancies']} redundancies")
            
            return complete_result
            
        except Exception as e:
            logger.error(f"Complete prescription analysis failed: {e}")
            raise


