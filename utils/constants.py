"""
Constantes utilisées dans l'application
"""

# Niveaux d'interaction pour classification
INTERACTION_LEVELS = {
    'major': 'Major',
    'majeur': 'Major',
    'élevé': 'Major',
    'severe': 'Major',
    'high': 'Major',
    
    'moderate': 'Moderate',
    'modérée': 'Moderate',
    'modéré': 'Moderate',
    'moyen': 'Moderate',
    'medium': 'Moderate',
    
    'minor': 'Minor',
    'mineure': 'Minor',
    'mineur': 'Minor',
    'faible': 'Minor',
    'low': 'Minor',
    'weak': 'Minor'
}

# Couleurs pour l'interface (Bootstrap colors)
LEVEL_COLORS = {
    'Major': '#DC3545',      # Rouge Bootstrap (danger)
    'Moderate': '#FD7E14',   # Orange Bootstrap (warning)
    'Minor': '#28A745',      # Vert Bootstrap (success)
    'Aucune': '#6C757D',     # Gris Bootstrap (secondary)
    'Erreur': '#E83E8C'      # Rose Bootstrap (pour les erreurs)
}

# Icônes pour les niveaux
LEVEL_ICONS = {
    'Major': '🔴',
    'Moderate': '🟡',
    'Minor': '🟢',
    'Aucune': '⚪',
    'Erreur': '❌'
}

# Formats d'export supportés
EXPORT_FORMATS = {
    'csv': {
        'extension': '.csv',
        'mime_type': 'text/csv',
        'description': 'CSV (Comma Separated Values)'
    },
    'excel': {
        'extension': '.xlsx',
        'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'description': 'Excel Workbook'
    }
}

# Séparateurs pour le text splitting
TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Extensions de fichiers PDF supportées
PDF_EXTENSIONS = ['.pdf', '.PDF']

# Taille maximale des fichiers PDF (en octets)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Messages d'erreur standards
ERROR_MESSAGES = {
    'no_drugs_found': "Aucun médicament identifié dans votre question.",
    'insufficient_drugs': "L'analyse d'interactions nécessite au moins 2 médicaments.",
    'pdf_processing_failed': "Erreur lors du traitement des fichiers PDF.",
    'vector_store_error': "Erreur lors de la création ou du chargement de l'index.",
    'llm_analysis_failed': "Erreur lors de l'analyse par l'IA.",
    'export_failed': "Erreur lors de l'export des résultats.",
    'api_quota_exceeded': "Quota API dépassé. Veuillez réessayer plus tard.",
    'invalid_configuration': "Configuration invalide. Vérifiez vos paramètres."
}

# Messages de succès standards
SUCCESS_MESSAGES = {
    'drugs_extracted': "Médicaments identifiés avec succès",
    'analysis_completed': "Analyse des interactions terminée",
    'export_ready': "Export prêt au téléchargement",
    'cache_cleared': "Cache vidé avec succès",
    #'vector_store_loaded': "Base documentaire chargée",
    'pdf_processed': "Documents PDF traités avec succès"
}

# Configuration Streamlit par défaut
STREAMLIT_CONFIG = {
    'page_title': "Analyseur d'Interactions Médicamenteuses",
    'page_icon': "💊",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# CSS classes pour les styles
CSS_CLASSES = {
    'main_header': 'main-header',
    'metric_card': 'metric-card',
    'interaction_major': 'interaction-major',
    'interaction_moderate': 'interaction-moderate',
    'interaction_minor': 'interaction-minor',
    'error_message': 'error-message',
    'success_message': 'success-message'
}

# Durées de timeout (en secondes)
TIMEOUTS = {
    'llm_request': 30,
    'pdf_processing': 120,
    'vector_store_creation': 300,
    'search_request': 10
}

# Limites de l'application
LIMITS = {
    'max_drugs_per_analysis': 20,
    'max_combinations': 190,  # C(20,2) = 190
    'max_pdf_files': 50,
    'max_search_results': 20,
    'max_history_entries': 100
}

# Configuration des graphiques Plotly
PLOTLY_CONFIG = {
    'displayModeBar': False,
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'interaction_analysis',
        'height': 500,
        'width': 700,
        'scale': 1
    }
}

# Templates de prompts
PROMPT_TEMPLATES = {
    'drug_extraction_simple': 
    """
        Tu es un expert pharmaceutique. Analyse cette question et extrait UNIQUEMENT les noms des médicaments/substances actives mentionnés.
        
        Question: "{question}"
        
        Réponds UNIQUEMENT avec une liste de noms séparés par des virgules, sans explication.
        Si aucun médicament n'est trouvé, réponds "AUCUN".
        
        Exemple: "Aspirine, Warfarine"
        """,
    
    # """
    # Extrait tous les noms de médicaments/molécules actives d'un texte médical.
    
    # Texte: "{question}"
    
    # Instructions:
    # - Identifie tous les noms de médicaments ou molécules actives mentionnés
    # - Donne uniquement les noms des principes actifs (pas les marques commerciales)
    # - Ignore les dosages, formes galéniques, et fréquences
    # - Sépare les résultats par des virgules
    # - Si aucun médicament trouvé, réponds "AUCUN"
    
    # Exemples:
    # Texte: "Patient prend Aspirin 100mg, Metformin 850mg et Omeprazole 20mg"
    # Réponse: Aspirin, Metformin, Omeprazole
    
    # Texte: "Prescription: Lisinopril 10mg 1x/jour, Atorvastatin 20mg le soir"
    # Réponse: Lisinopril, Atorvastatin
    
    # Molécules:
    # """,
    
    'interaction_analysis': """
    Tu es un pharmacien clinique expert. Analyse l'interaction potentielle entre ces deux médicaments: {drug1} et {drug2}.
    
    Contexte médical:
    {context}
    
    Réponds au format exact:
    NIVEAU: [Major/Moderate/Minor/Aucune]
    EXPLICATION: [Explication détaillée]
    """,
    
    # 'detailed_explanation_with_sources': """
    # Tu es un pharmacien clinique expert. Base-toi sur le contexte médical fourni avec sources pour répondre à la question de manière professionnelle et structurée.
    
    # Contexte médical avec sources détaillées:
    # {context}
    
    # Question: {question}
    
    # Instructions:
    # - Intègre naturellement les références aux sources dans ta réponse
    # - Utilise le format "Selon [Source]" ou "D'après [Guidelines]" pour citer
    # - Structure ta réponse avec:
    #   1. tous les interactions entre les tous les molecules de la prescription
    #   2. les autres facteurs de risque s'il y en a(Contre-indications, Surdosage ou sous-dosage,Redondance thérapeutique,Durée de traitement excessive ou insuffisante, Les effets secondaires, etc.)
    #   3. Mécanismes d'action concernés
    #   4. Recommandations pratiques
    #   5. Surveillance nécessaire
    
    # Réponse avec sources intégrées:
    # """,

    #1. Résumé des interactions principales
    # 'detailed_explanation': """
    # Basé sur le contexte médical fourni, réponds à la question de manière professionnelle et structurée.
    
    # Contexte médical:
    # {context}
    
    # Question: {question}
    
    # Structure ta réponse avec:
    # 1. Résumé des interactions principales
    # 2. Mécanismes d'action concernés
    # 3. Recommandations pratiques
    # 4. Surveillance nécessaire
    
    # Réponse:
    # """

    ################################################################################
    'detailed_explanation_with_sources': """
    Tu es un pharmacien clinique expert. Base-toi sur le contexte médical fourni avec sources pour répondre à la question de manière professionnelle et structurée.
    
    Contexte médical avec sources détaillées:
    {context}
    
    Question: {question}
    
    Instructions:
    - Intègre naturellement les références aux sources dans ta réponse
    - Utilise le format "Selon [Source]" ou "D'après [Guidelines]" pour citer
    - Structure ta réponse avec:
      1. Interactions médicamenteuses (format existant)
      2. Dosage inadapté (NOUVEAU)
      3. Autres facteurs de risque (le reste)
      4. Mécanismes d'action concernés
      5. Recommandations pratiques
      6. Surveillance nécessaire
    
    Pour la section "Dosage inadapté", analyse spécifiquement:
    - Surdosage potentiel (doses trop élevées, accumulation, interactions augmentant l'effet)
    - Sous-dosage potentiel (doses insuffisantes, efficacité compromise)
    - Ajustements nécessaires selon l'âge, le poids, la fonction rénale/hépatique
    
    Réponse avec sources intégrées:
    """,
    # Nouveau prompt spécialisé pour l'analyse de dosage
    'dosage_analysis': """
    Tu es un pharmacien expert. Analyse les dosages de cette prescription médicale en te basant sur le contexte médical fourni.
    
    Prescription: {prescription}
    Informations patient: {patient_info}
    
    Contexte médical de référence:
    {context}
    
    Pour chaque médicament, évalue:
    1. La dose prescrite vs dose recommandée (utilise les références du contexte si disponibles)
    2. Les facteurs d'ajustement (âge, poids, fonction rénale/hépatique)
    3. Les risques de surdosage ou sous-dosage selon les guidelines
    4. Les interactions pouvant modifier l'effet des doses
    5. Les recommandations spécifiques pour les populations à risque (personnes âgées, insuffisance rénale/hépatique)
    
    Réponds au format JSON suivant:
    {{
        "dosage_analysis": {{
            "surdosage": [
                {{
                    "medicament": "nom du médicament",
                    "dose_prescrite": "dose avec unité",
                    "dose_recommandee": "dose recommandée avec unité",
                    "facteur_risque": "âge/interactions/fonction organique",
                    "gravite": "Faible/Modérée/Élevée",
                    "explication": "explication détaillée (cite les sources du contexte si utilisées)",
                    "recommandation": "action recommandée",
                    "source": "source de l'information (contexte médical ou connaissances générales)"
                }}
            ],
            "sous_dosage": [
                {{
                    "medicament": "nom du médicament",
                    "dose_prescrite": "dose avec unité", 
                    "dose_recommandee": "dose recommandée avec unité",
                    "facteur_risque": "raison du sous-dosage",
                    "gravite": "Faible/Modérée/Élevée",
                    "explication": "explication détaillée (cite les sources du contexte si utilisées)",
                    "recommandation": "action recommandée",
                    "source": "source de l'information (contexte médical ou connaissances générales)"
                }}
            ],
            "dosage_approprie": [
                {{
                    "medicament": "nom du médicament",
                    "dose_prescrite": "dose avec unité",
                    "commentaire": "dosage approprié pour ce patient",
                    "source": "source de la validation (contexte médical ou connaissances générales)"
                }}
            ]
        }}
    }}
    
    Si aucun problème de dosage détecté, retourne des listes vides pour surdosage et sous_dosage.
    """,
    
    # Nouveau prompt spécialisé pour l'analyse de contre-indications
    'contraindication_analysis': """
    Tu es un pharmacien expert spécialisé dans les contre-indications médicamenteuses. Analyse cette prescription pour détecter les contre-indications.
    
    Prescription: {prescription}
    Informations patient: {patient_info}
    
    Contexte médical de référence:
    {context}
    
    Pour chaque médicament, évalue:
    1. Les contre-indications absolues (interdiction formelle)
    2. Les contre-indications relatives (prudence, surveillance)
    3. Les pathologies du patient mentionnées
    4. Les interactions avec d'autres conditions
    
    IMPORTANT: Si la base de connaissances ne contient pas d'informations sur un médicament spécifique, indique-le clairement dans "donnees_insuffisantes".
    
    Réponds au format JSON suivant:
    {{
        "contraindication_analysis": {{
            "contre_indications_absolues": [
                {{
                    "medicament": "nom du médicament",
                    "condition": "pathologie/condition contre-indiquée",
                    "mecanisme": "mécanisme de la contre-indication",
                    "consequences": "conséquences possibles",
                    "recommandation": "action recommandée",
                    "source": "source de l'information (base vectorielle ou connaissances générales)"
                }}
            ],
            "contre_indications_relatives": [
                {{
                    "medicament": "nom du médicament",
                    "condition": "pathologie/condition nécessitant prudence", 
                    "mecanisme": "mécanisme de précaution",
                    "consequences": "risques potentiels",
                    "recommandation": "mesures de surveillance",
                    "source": "source de l'information"
                }}
            ],
            "aucune_contre_indication": [
                {{
                    "medicament": "nom du médicament",
                    "commentaire": "aucune contre-indication identifiée"
                }}
            ],
            "donnees_insuffisantes": [
                {{
                    "medicament": "nom du médicament",
                    "raison": "la base de connaissances ne contient pas d'informations suffisantes sur ce médicament"
                }}
            ]
        }}
    }}
    
    Si aucune contre-indication n'est trouvée dans la base de connaissances, utilise la section "donnees_insuffisantes" pour éviter les hallucinations.
    """,
    
    # Nouveau prompt spécialisé pour l'analyse de redondance thérapeutique
    'redundancy_analysis': """
    Tu es un pharmacien expert spécialisé dans l'optimisation thérapeutique. Analyse cette prescription pour détecter les redondances thérapeutiques.
    
    Prescription: {prescription}
    Informations patient: {patient_info}
    
    Contexte médical de référence:
    {context}
    
    Pour chaque médicament, évalue:
    1. Les redondances directes (même molécule prescrite plusieurs fois)
    2. Les redondances de classe (médicaments de même classe thérapeutique)
    3. Les redondances fonctionnelles (même effet thérapeutique par mécanismes différents)
    4. L'optimisation possible de la stratégie thérapeutique
    
    Base-toi sur les guidelines du contexte médical pour identifier les associations inappropriées ou redondantes.
    
    Réponds au format JSON suivant:
    {{
        "redundancy_analysis": {{
            "redondance_directe": [
                {{
                    "classe_therapeutique": "classe concernée",
                    "medicaments": ["médicament1", "médicament2"],
                    "mecanisme": "même principe actif prescrit plusieurs fois",
                    "risque": "surdosage, effets indésirables cumulés",
                    "recommandation": "éliminer les doublons, ajuster la posologie",
                    "source": "source de l'information (contexte médical ou connaissances générales)"
                }}
            ],
            "redondance_classe": [
                {{
                    "classe_therapeutique": "classe concernée",
                    "medicaments": ["médicament1", "médicament2"],
                    "mecanisme": "même classe thérapeutique, effets similaires",
                    "risque": "effets additifs, interactions potentielles",
                    "recommandation": "évaluer la nécessité, choisir un seul représentant",
                    "source": "source de l'information"
                }}
            ],
            "redondance_fonctionnelle": [
                {{
                    "classe_therapeutique": "classes concernées",
                    "medicaments": ["médicament1", "médicament2"],
                    "mecanisme": "effet thérapeutique similaire par mécanismes différents",
                    "risque": "effet cumulé non nécessaire, complexification du traitement",
                    "recommandation": "optimiser la stratégie thérapeutique, simplifier si possible",
                    "source": "source de l'information"
                }}
            ],
            "aucune_redondance": [
                {{
                    "medicament": "nom du médicament",
                    "commentaire": "médicament unique dans sa classe/fonction thérapeutique"
                }}
            ]
        }}
    }}
    
    Si aucune redondance n'est détectée, utilise la section "aucune_redondance" pour chaque médicament analysé.
    """,
    
    # Nouveau prompt spécialisé pour l'analyse des voies d'administration
    'administration_route_analysis': """
    Tu es un pharmacien expert spécialisé dans les voies d'administration médicamenteuses. Analyse cette prescription pour détecter les problèmes liés aux voies d'administration.
    
    Prescription: {prescription}
    Informations patient: {patient_info}
    
    Contexte médical de référence:
    {context}
    
    Pour chaque médicament, évalue:
    1. La pertinence de la voie d'administration prescrite
    2. Les risques spécifiques liés à la voie choisie
    3. Les incompatibilités entre voie et médicament/formulation
    4. Le timing et la fréquence d'administration par rapport à la voie
    5. L'adéquation de la voie avec l'état du patient (âge, comorbidités)
    
    Base-toi sur les guidelines du contexte médical et sur les bonnes pratiques d'administration.
    
    Réponds au format JSON suivant:
    {{
        "administration_route_analysis": {{
            "voie_inappropriee": [
                {{
                    "medicament": "nom du médicament",
                    "voie_prescrite": "voie prescrite",
                    "voie_recommandee": "voie recommandée",
                    "justification": "raison de l'inadéquation",
                    "gravite": "Faible/Modérée/Élevée",
                    "explication": "explication détaillée (cite les sources du contexte si utilisées)",
                    "timing_administration": "timing problématique si pertinent",
                    "frequence": "fréquence d'administration si pertinent",
                    "recommandation": "action recommandée",
                    "source": "source de l'information (contexte médical ou connaissances générales)"
                }}
            ],
            "voie_incompatible": [
                {{
                    "medicament": "nom du médicament",
                    "voie_prescrite": "voie prescrite",
                    "voie_recommandee": "voie recommandée",
                    "justification": "raison de l'incompatibilité",
                    "gravite": "Faible/Modérée/Élevée",
                    "explication": "explication détaillée (cite les sources du contexte si utilisées)",
                    "timing_administration": "timing problématique si pertinent",
                    "frequence": "fréquence d'administration si pertinent",
                    "recommandation": "action recommandée",
                    "source": "source de l'information (contexte médical ou connaissances générales)"
                }}
            ],
            "voie_risquee": [
                {{
                    "medicament": "nom du médicament",
                    "voie_prescrite": "voie prescrite",
                    "voie_recommandee": "même voie avec précautions ou alternative",
                    "justification": "nature du risque",
                    "gravite": "Faible/Modérée/Élevée",
                    "explication": "explication détaillée des risques",
                    "timing_administration": "timing recommandé si pertinent",
                    "frequence": "fréquence d'administration si pertinent",
                    "recommandation": "précautions à prendre",
                    "source": "source de l'information (contexte médical ou connaissances générales)"
                }}
            ],
            "voie_appropriee": [
                {{
                    "medicament": "nom du médicament",
                    "voie_prescrite": "voie prescrite",
                    "timing_administration": "timing si spécifié",
                    "frequence": "fréquence si spécifiée",
                    "commentaire": "voie d'administration appropriée pour ce médicament et ce patient",
                    "source": "source de la validation (contexte médical ou connaissances générales)"
                }}
            ]
        }}
    }}
    
    Si aucun problème de voie d'administration n'est détecté pour un médicament, ajoute-le dans la section "voie_appropriee".
    """
}
