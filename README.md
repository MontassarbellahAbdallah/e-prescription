# 💊 Analyse de prescription médicale

## 🎯 Vue d'ensemble

Application Streamlit professionnelle pour l'analyse de prescription médicale, basée sur l'IA Gemini et la technologie RAG (Retrieval-Augmented Generation).

### ⚙️ **Utilisation de la Vector Store Database**

## **✅ OUI, le système utilise intensivement la base vectorielle pour :**

### **1. Détection des Interactions**
- La base vectorielle **FAISS** contient 12 documents PDF de guidelines médicales
- Pour l'analyse d'interactions, le système :
  ```python
  # Dans llm_analyzer.py
  def analyze_single_interaction(self, drug1, drug2, context_docs):
      # context_docs provient de la recherche vectorielle
      context_text = "\n".join([doc.page_content[:500] for doc in context_docs[:3]])
  ```

### **2. Analyse de Dosage**
- **✅ TOTALEMENT** : Le dosage utilise maintenant la base vectorielle pour enrichir l'analyse
- Le système recherche dans les guidelines pour les dosages recommandés selon l'âge et pathologies
  ```python
  # Dans dosage_analyzer.py
  def analyze_dosage(self, prescription, patient_info, context_docs):
      context = self._prepare_context(context_docs or [])
      prompt = PROMPT_TEMPLATES['dosage_analysis'].format(
          context=context  # Utilise les docs de la base vectorielle
      )
  ```

### **3. Contre-indications**
- **✅ FORTEMENT** : Les contre-indications utilisent directement la base vectorielle
  ```python
  # Dans contraindication_analyzer.py
  def analyze_contraindications(self, prescription, patient_info, context_docs):
      context = self._prepare_context(context_docs or [])
      prompt = PROMPT_TEMPLATES['contraindication_analysis'].format(
          context=context  # Utilise les docs de la base vectorielle
      )
  ```

### **4. Analyse Complète Intégrée**
- Le système effectue une **analyse complète** qui utilise la base vectorielle pour toutes les analyses :
  ```python
  # Dans llm_analyzer.py
  def analyze_prescription_complete(self, prescription, context_docs):
      # Toutes les analyses utilisent context_docs de la base vectorielle
      interactions = self.analyze_all_combinations(drugs, context_docs)
      dosage = self.analyze_dosage(prescription, patient_info, context_docs)
      contraindications = self.analyze_contraindications(prescription, patient_info, context_docs)
  ```

### ✨ Fonctionnalités principales

- 🔬 **Analyse d'interactions** : Extraction automatique de médicaments et analyse des interactions
- 🔍 **Recherche documentaire** : Interface de recherche avancée dans la base de connaissances
- 📊 **Historique complet** : Suivi des analyses et recherches avec export
- 🧠 **IA intégrée** : Utilisation de Google Gemini pour l'analyse sémantique
- 📚 **RAG optimisé** : Recherche vectorielle avec FAISS pour contextualiser les réponses
- 💾 **Cache intelligent** : System de cache pour optimiser les performances
- 🔄 **Gestion multi-clés** : Rotation automatique des clés API pour éviter les quotas

## 🏗️ Architecture Modulaire

### Structure du projet
```
prescription_chatbot_modular/
├── 📁 config/                 # Configuration centralisée
│   ├── settings.py           # Paramètres de l'application
│   └── logging_config.py     # Configuration des logs
├── 📁 core/                  # Services centraux
│   ├── cache_manager.py      # Gestionnaire de cache
│   ├── key_manager.py        # Gestionnaire de clés API
│   └── exceptions.py         # Exceptions personnalisées
├── 📁 data/                  # Traitement des données
│   ├── pdf_processor.py      # Extraction de texte PDF
│   ├── rag_processor.py      # Processeur RAG avec FAISS
│   └── validators.py         # Validateurs de données
├── 📁 ai/                    # Intelligence artificielle
│   ├── llm_analyzer.py       # Analyseur LLM (Gemini)
│   └── embeddings.py         # Gestionnaire d'embeddings
├── 📁 ui/                    # Interface utilisateur
│   ├── 📁 components/        # Composants réutilisables
│   │   ├── tables.py         # Tableaux d'interactions
│   │   ├── charts.py         # Graphiques statistiques
│   │   ├── search.py         # Interface de recherche
│   │   └── export.py         # Fonctions d'export
│   ├── 📁 pages/            # Pages de l'application
│   │   ├── analysis_page.py  # Page d'analyse principale
│   │   ├── search_page.py    # Page de recherche
│   │   └── history_page.py   # Page d'historique
│   └── styles.py            # Styles CSS et helpers UI
├── 📁 utils/                # Utilitaires
│   ├── constants.py         # Constantes de l'application
│   └── helpers.py           # Fonctions utilitaires
├── 📁 Data/                 # Données de l'application
│   └── guidelines/          # Documents PDF de référence
├── main.py                  # Point d'entrée principal
├── requirements.txt         # Dépendances Python
└── README.md               # Cette documentation
```

## 🚀 Installation et Configuration

### Prérequis

- Python 3.8+
- Clé API Google (Gemini)
- Documents PDF de référence (guides médicaux, monographies, etc.)

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Configuration des variables d'environnement

Créez un fichier `.env` à la racine du projet :

```env
# Clé API Google (obligatoire)
GOOGLE_API_KEY=votre_cle_api_ici

# Configuration optionnelle
LOG_LEVEL=INFO
CACHE_DURATION_HOURS=24
MAX_PDF_SIZE_MB=50
CHUNK_SIZE=800
CHUNK_OVERLAP=150
EMBEDDING_MODEL=models/embedding-001
LLM_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.0
DEFAULT_SEARCH_RESULTS=5
```

### 3. Préparation des documents

Placez vos documents PDF de référence dans le dossier `Data/guidelines/`:

```bash
mkdir -p Data/guidelines
# Copiez vos fichiers PDF médicaux ici
```

### 4. Lancement de l'application

```bash
streamlit run main.py
```

## 📋 Guide d'utilisation

### Page d'Analyse d'Interactions

1. **Saisie de la prescription** :
   ```
   Exemple : "Voici une prescription medicatmenteuse pour un patient agé:
    Patient : F, 82 ans
    Prescription du 2190-07-19 :
    • MAIN Oxycodone-Acetaminophen ,molécule :(Oxycodone) ,(5mg/325mg Tablet) ,dose : 1-2 TAB ,forme : 1-2 TAB ,voie : PO
    • MAIN Amiodarone HCl ,molécule :(Amiodarone) ,(150mg/3mL Vial) ,dose : 150 mg ,forme : 1 VIAL ,voie : IV
    • MAIN Magnesium Sulfate ,molécule :(Magnesium sulfate) ,(1g/2mL Vial) ,dose : 2 gm ,forme : 4 ml ,voie : IV

    Question : cette prescription médicamenteuse est-elle porteuse de risque ou saine?
   ```

2. **Extraction automatique** : L'IA identifie les médicaments mentionnés

3. **Analyse des combinaisons** : Toutes les paires possibles sont analysées

4. **Résultats** : Visualisation des interactions avec niveaux de gravité

### Page de Recherche Documentaire

1. **Recherche générale** : Requêtes en langage naturel dans tous les documents
2. **Recherche par médicament** : Recherche spécialisée avec types prédéfinis
3. **Exploration** : Statistiques et analyse de la base documentaire

### Page d'Historique

1. **Analyses précédentes** : Consultation et réanalyse
2. **Recherches effectuées** : Historique des requêtes
3. **Gestion** : Export et nettoyage de l'historique

## ⚙️ Configuration Avancée

### Gestion Multi-Clés API

Pour éviter les limitations de quota, vous pouvez configurer plusieurs clés :

```env
GOOGLE_API_KEY=cle1,cle2,cle3
```

### Personnalisation du Cache

```env
CACHE_DURATION_HOURS=48  # Cache plus long
CACHE_DIR=custom_cache   # Dossier personnalisé
```

### Optimisation des Chunks

```env
CHUNK_SIZE=1200         # Chunks plus larges
CHUNK_OVERLAP=200       # Plus de recouvrement
```

## 🔧 Développement et Extension

### Ajout d'un nouveau composant UI

1. Créez le fichier dans `ui/components/`
2. Suivez les patterns existants (logging, error handling)
3. Importez dans les pages appropriées

### Ajout d'un nouveau processeur de données

1. Créez le fichier dans `data/`
2. Héritez des classes de base si applicable
3. Ajoutez les tests unitaires

### Modification des prompts IA

Les templates de prompts sont dans `utils/constants.py` :

```python
PROMPT_TEMPLATES = {
    'drug_extraction': "Votre template personnalisé...",
    'interaction_analysis': "Votre template personnalisé..."
}
```

## 📊 Monitoring et Logs

### Fichiers de log

- `app.log` : Logs principaux de l'application
- Console Streamlit : Erreurs et warnings en temps réel

### Métriques disponibles

- Nombre d'analyses effectuées
- Temps de réponse des requêtes
- Taux de cache hit/miss
- Utilisation des clés API

## 🚨 Dépannage

### Problèmes courants

**Erreur : "GOOGLE_API_KEY non trouvée"**
```bash
# Vérifiez le fichier .env
cat .env | grep GOOGLE_API_KEY
```

**Erreur : "Aucun document indexé"**
```bash
# Vérifiez les fichiers PDF
ls -la Data/guidelines/*.pdf
```

**Performance lente**
```bash
# Videz le cache
rm -rf cache/*
# Ou utilisez l'interface de gestion
```

### Diagnostic système

Utilisez le bouton "🔧 Diagnostic système" en cas d'erreur pour obtenir un rapport détaillé.

## 🏥 Avertissements Médicaux

⚠️ **IMPORTANT** : Cette application est un **outil d'aide à la décision** uniquement.

- Ne remplace pas l'avis d'un professionnel de santé
- Les résultats doivent être validés par un pharmacien ou médecin
- Utilisez uniquement des sources documentaires fiables
- Mettez à jour régulièrement la base de connaissances

## 📈 Évolutions Prévues

- [ ] Support d'autres formats de documents (Word, Excel)
- [ ] Interface API REST
- [ ] Authentification utilisateur
- [ ] Base de données persistante
- [ ] Notifications par email
- [ ] Intégration avec systèmes hospitaliers
 
**Technologies**: Python, Streamlit, Google Gemini, FAISS, LangChain