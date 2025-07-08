"""
Script de test rapide pour vérifier l'implémentation du dosage
"""

def test_imports():
    """Test des imports"""
    print("🧪 Test des imports...")
    
    try:
        # Test 1: Prompt templates
        from utils.constants import PROMPT_TEMPLATES
        assert 'dosage_analysis' in PROMPT_TEMPLATES
        print("✅ Templates de prompts OK")
        
        # Test 2: DosageAnalyzer
        from ai.dosage_analyzer import DosageAnalyzer
        analyzer = DosageAnalyzer(use_cache=False)
        print("✅ DosageAnalyzer OK")
        
        # Test 3: Composants UI
        from ui.components.dosage_components import display_dosage_analysis_section
        print("✅ Composants UI OK")
        
        # Test 4: LLMAnalyzer avec dosage
        from ai.llm_analyzer import LLMAnalyzer
        llm = LLMAnalyzer(use_cache=False)
        assert hasattr(llm, 'dosage_analyzer')
        assert hasattr(llm, 'analyze_dosage')
        assert hasattr(llm, 'analyze_prescription_complete')
        print("✅ LLMAnalyzer intégré OK")
        
        # Test 5: Analysis page
        from ui.pages.analysis_page import AnalysisPage
        print("✅ Analysis page OK")
        
        print("\n🎉 TOUS LES TESTS PASSENT ! L'implémentation est correcte.")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_basic_functionality():
    """Test basique sans appel API"""
    print("\n🧪 Test de fonctionnalité basique...")
    
    try:
        from ai.dosage_analyzer import DosageAnalyzer
        
        analyzer = DosageAnalyzer(use_cache=False)
        
        # Test des méthodes utilitaires
        empty_result = analyzer._get_empty_dosage_result("Test")
        assert 'dosage_analysis' in empty_result
        assert 'stats' in empty_result
        
        # Test de formatage
        test_data = {
            'surdosage': [{
                'medicament': 'TestMed',
                'dose_prescrite': '100mg',
                'gravite': 'Élevée'
            }],
            'sous_dosage': [],
            'dosage_approprie': []
        }
        
        formatted = analyzer.format_dosage_for_display(test_data)
        assert len(formatted) == 1
        assert formatted[0]['Type'] == 'Surdosage'
        
        print("✅ Fonctionnalité basique OK")
        return True
        
    except Exception as e:
        print(f"❌ Erreur fonctionnalité: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Test de l'implémentation 'Dosage inadapté'")
    print("=" * 50)
    
    import_success = test_imports()
    
    if import_success:
        functionality_success = test_basic_functionality()
        
        if functionality_success:
            print("\n🎯 RÉSULTAT FINAL:")
            print("✅ L'implémentation est PRÊTE !")
            print("🚀 Vous pouvez maintenant tester avec une vraie prescription.")
        else:
            print("\n⚠️ Problème de fonctionnalité détecté")
    else:
        print("\n❌ Problème d'imports détecté")
