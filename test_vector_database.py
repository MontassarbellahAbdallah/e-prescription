#!/usr/bin/env python3
"""
Script de diagnostic pour tester la base vectorielle FAISS
"""
import sys
import os

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vector_database():
    """Test de la base de données vectorielle"""
    
    print("🔍 Diagnostic de la base vectorielle FAISS")
    print("=" * 50)
    
    try:
        from data.rag_processor import RAGProcessor
        
        # Initialiser le RAG processor
        rag_processor = RAGProcessor(cache_enabled=True)
        print("✅ RAGProcessor initialisé")
        
        # Charger le vector store
        if rag_processor.load_vector_store():
            print("✅ Vector store chargé avec succès")
        else:
            print("❌ Impossible de charger le vector store")
            return False
        
        # Test de recherche avec des termes spécifiques aux guidelines
        test_queries = [
            "benzodiazépine personnes âgées",
            "diazépam STOPP",
            "médicaments inappropriés",
            "anticholinergiques",
            "liste de Beers",
            "STOPP START critères",
            "contre-indication âgé",
            "PIM potentially inappropriate medications"
        ]
        
        print("\n🔍 Test de recherche dans la base vectorielle:")
        print("-" * 50)
        
        for query in test_queries:
            try:
                # Recherche avec sources détaillées
                context_docs, sources_info = rag_processor.search_with_detailed_sources(query, k=3)
                
                print(f"\n📋 Requête: '{query}'")
                print(f"   Résultats trouvés: {len(context_docs)}")
                
                if context_docs:
                    for i, (doc, source) in enumerate(zip(context_docs[:2], sources_info[:2])):
                        print(f"   📄 Document {i+1}: {source.get('document', 'Inconnu')}")
                        print(f"      Score: {source.get('relevance_score', 0):.3f}")
                        print(f"      Contenu: {doc.page_content[:100]}...")
                else:
                    print("   ❌ Aucun résultat trouvé")
                    
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
        
        # Test spécifique pour contre-indications
        print("\n🎯 Test spécifique: contre-indications")
        print("-" * 50)
        
        contraindication_query = "benzodiazépine diazépam personnes âgées contre-indication STOPP"
        context_docs, sources_info = rag_processor.search_with_detailed_sources(contraindication_query, k=5)
        
        print(f"Requête test: '{contraindication_query}'")
        print(f"Résultats: {len(context_docs)} documents trouvés")
        
        if context_docs:
            print("✅ La base contient des informations sur les contre-indications")
            for i, source in enumerate(sources_info[:3]):
                print(f"   {i+1}. {source.get('document', 'Inconnu')} (Score: {source.get('relevance_score', 0):.3f})")
        else:
            print("❌ Aucune information sur les contre-indications trouvée")
            print("   Possible cause: Index FAISS vide ou non créé")
        
        return len(context_docs) > 0
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def suggest_prescription_examples():
    """Suggère des exemples basés sur les documents disponibles"""
    
    print("\n💡 Suggestions d'exemples basés sur vos documents:")
    print("=" * 50)
    
    examples = [
        {
            "title": "Benzodiazépines chez personnes âgées (STOPP)",
            "prescription": """Patient: Homme, 78 ans, démence légère, chutes récentes

Prescription:
- Diazépam 5mg le soir
- Lorazépam 1mg si besoin
- Zolpidem 10mg le soir

Antécédents: Démence, chutes récurrentes"""
        },
        {
            "title": "Anticholinergiques (liste de Beers)",
            "prescription": """Patient: Femme, 82 ans, glaucome, prostate hypertrophiée

Prescription:
- Oxybutynine 5mg 2x/jour
- Diphenhydramine 25mg le soir
- Amitriptyline 25mg le soir

Antécédents: Glaucome, hypertrophie prostatique"""
        },
        {
            "title": "AINS chez personnes âgées (STOPP)",
            "prescription": """Patient: Homme, 75 ans, insuffisance cardiaque, ulcère gastrique

Prescription:
- Ibuprofène 400mg 3x/jour
- Indométacine 25mg 2x/jour
- Diclofénac gel

Antécédents: Insuffisance cardiaque, antécédents d'ulcère"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📋 Exemple {i}: {example['title']}")
        print("-" * 30)
        print(example['prescription'])

if __name__ == "__main__":
    print("🚀 Diagnostic de la base vectorielle pour contre-indications")
    print("=" * 60)
    
    # Test de la base vectorielle
    vector_db_works = test_vector_database()
    
    if vector_db_works:
        print("\n🎉 La base vectorielle fonctionne !")
        suggest_prescription_examples()
    else:
        print("\n⚠️ Problème avec la base vectorielle")
        print("Solutions possibles:")
        print("1. Vérifier que l'index FAISS a été créé")
        print("2. Relancer le traitement des PDFs")
        print("3. Vérifier les documents dans data/guidelines/")
    
    print("\n" + "=" * 60)
    print("Diagnostic terminé")
