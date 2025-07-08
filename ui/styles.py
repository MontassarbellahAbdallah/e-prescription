"""
Styles CSS et helpers pour l'interface utilisateur Streamlit
"""
import streamlit as st
from typing import Dict, Any, Optional
from utils.constants import LEVEL_COLORS, CSS_CLASSES

# CSS principal pour l'application
MAIN_CSS = """
<style>
/* Variables CSS */
:root {
    --primary-color: #7B50DF;
    --secondary-color: #F46056;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
    --info-color: #17a2b8;
    --light-color: #f8f9fa;
    --dark-color: #343a40;
    --border-radius: 8px;
    --box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    --transition: all 0.3s ease;
}

/* En-tête principal */
.main-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    padding: 2rem;
    border-radius: var(--border-radius);
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: var(--box-shadow);
    animation: fadeInDown 0.8s ease-out;
}

.main-header h1 {
    margin: 0 0 0.5rem 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    margin: 0;
    font-size: 1.2rem;
    opacity: 0.9;
}

/* Cartes métriques */
.metric-card {
    background: var(--light-color);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    border-left: 4px solid var(--primary-color);
    margin: 0.5rem 0;
    box-shadow: var(--box-shadow);
    transition: var(--transition);
    animation: fadeInUp 0.6s ease-out;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.metric-card h3 {
    color: var(--primary-color);
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
    font-weight: 600;
}

.metric-card .metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--dark-color);
    margin: 0;
}

.metric-card .metric-delta {
    font-size: 0.9rem;
    margin-top: 0.25rem;
}

/* Interactions par niveau */
.interaction-major {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border-left: 5px solid var(--danger-color);
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    animation: slideInLeft 0.5s ease-out;
}

.interaction-moderate {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border-left: 5px solid var(--warning-color);
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    animation: slideInLeft 0.5s ease-out;
}

.interaction-minor {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    border-left: 5px solid var(--success-color);
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    animation: slideInLeft 0.5s ease-out;
}

.interaction-none {
    background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
    border-left: 5px solid #9e9e9e;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    animation: slideInLeft 0.5s ease-out;
}

/* Messages d'état */
.success-message {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    padding: 1rem;
    border-radius: var(--border-radius);
    border: 1px solid #c3e6cb;
    margin: 1rem 0;
    box-shadow: var(--box-shadow);
    animation: fadeIn 0.5s ease-out;
}

.error-message {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    color: #721c24;
    padding: 1rem;
    border-radius: var(--border-radius);
    border: 1px solid #f5c6cb;
    margin: 1rem 0;
    box-shadow: var(--box-shadow);
    animation: fadeIn 0.5s ease-out;
}

.warning-message {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    color: #856404;
    padding: 1rem;
    border-radius: var(--border-radius);
    border: 1px solid #ffeaa7;
    margin: 1rem 0;
    box-shadow: var(--box-shadow);
    animation: fadeIn 0.5s ease-out;
}

.info-message {
    background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    color: #0c5460;
    padding: 1rem;
    border-radius: var(--border-radius);
    border: 1px solid #bee5eb;
    margin: 1rem 0;
    box-shadow: var(--box-shadow);
    animation: fadeIn 0.5s ease-out;
}

/* Boutons personnalisés */
.custom-button {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: var(--border-radius);
    font-weight: 600;
    text-decoration: none;
    display: inline-block;
    transition: var(--transition);
    box-shadow: var(--box-shadow);
    cursor: pointer;
}

.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    color: white;
    text-decoration: none;
}

.custom-button:active {
    transform: translateY(0);
}

/* Cartes de contenu */
.content-card {
    background: white;
    padding: 1.5rem;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin: 1rem 0;
    border: 1px solid #e9ecef;
    animation: fadeInUp 0.6s ease-out;
}

.content-card h3 {
    color: var(--primary-color);
    margin-top: 0;
    margin-bottom: 1rem;
    font-weight: 600;
}

/* Barres de progression personnalisées */
.progress-container {
    background: #e9ecef;
    border-radius: 10px;
    padding: 3px;
    margin: 1rem 0;
}

.progress-bar {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    height: 20px;
    border-radius: 8px;
    transition: width 0.5s ease-out;
    position: relative;
    overflow: hidden;
}

.progress-bar::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    right: 0;
    background-image: linear-gradient(
        -45deg,
        rgba(255, 255, 255, .2) 25%,
        transparent 25%,
        transparent 50%,
        rgba(255, 255, 255, .2) 50%,
        rgba(255, 255, 255, .2) 75%,
        transparent 75%,
        transparent
    );
    background-size: 50px 50px;
    animation: move 2s linear infinite;
}

/* Tableaux stylés */
.custom-table {
    border-collapse: collapse;
    width: 100%;
    height: 100%;
    margin: 1rem 0;
    background: white;
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--box-shadow);
}

.custom-table th {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    padding: 1rem;
    text-align: left;
    font-weight: 600;
}

.custom-table td {
    padding: 1rem;
    border-bottom: 1px solid #e9ecef;
    transition: background-color 0.2s ease;
}

.custom-table tr:hover td {
    background-color: #f8f9fa;
}

.custom-table tr:last-child td {
    border-bottom: none;
}

/* Badges de niveau */
.level-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: inline-block;
    margin: 0.25rem;
}

.level-badge.major {
    background: var(--danger-color);
    color: white;
}

.level-badge.moderate {
    background: var(--warning-color);
    color: #333;
}

.level-badge.minor {
    background: var(--success-color);
    color: white;
}

.level-badge.none {
    background: #6c757d;
    color: white;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes move {
    0% {
        background-position: 0 0;
    }
    100% {
        background-position: 50px 50px;
    }
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header {
        padding: 1rem;
    }
    
    .main-header h1 {
        font-size: 2rem;
    }
    
    .main-header p {
        font-size: 1rem;
    }
    
    .metric-card {
        padding: 1rem;
    }
    
    .custom-table th,
    .custom-table td {
        padding: 0.5rem;
        font-size: 0.9rem;
    }
}

/* Sidebar customization */
.css-1d391kg {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom selectbox styling */
.stSelectbox > div > div {
    border-radius: var(--border-radius);
    border: 2px solid #e9ecef;
    transition: var(--transition);
}

.stSelectbox > div > div:focus-within {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
}

/* Custom text input styling */
.stTextInput > div > div > input {
    border-radius: var(--border-radius);
    border: 2px solid #e9ecef;
    transition: var(--transition);
}

.stTextInput > div > div > input:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
}

/* Custom button styling */
.stButton > button {
    border-radius: var(--border-radius);
    border: 2px solid var(--primary-color);
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    font-weight: 600;
    transition: var(--transition);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Loading spinner customization */
.stSpinner > div {
    border-top-color: var(--primary-color) !important;
}
</style>
"""

def apply_custom_css():
    """Applique les styles CSS personnalisés à l'application Streamlit"""
    st.markdown(MAIN_CSS, unsafe_allow_html=True)

def create_main_header(title: str, subtitle: str = None):
    """
    Crée un en-tête principal stylé
    
    Args:
        title: Titre principal
        subtitle: Sous-titre optionnel
    """
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """
    Crée une carte métrique stylée
    
    Args:
        title: Titre de la métrique
        value: Valeur principale
        delta: Variation optionnelle
        delta_color: Couleur du delta (success, warning, error, normal)
    """
    delta_colors = {
        "success": "var(--success-color)",
        "warning": "var(--warning-color)", 
        "error": "var(--danger-color)",
        "normal": "var(--dark-color)"
    }
    
    delta_html = ""
    if delta:
        color = delta_colors.get(delta_color, delta_colors["normal"])
        delta_html = f'<div class="metric-delta" style="color: {color};">{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_interaction_card(drug1: str, drug2: str, level: str, explanation: str):
    """
    Crée une carte d'interaction stylée selon le niveau
    
    Args:
        drug1: Premier médicament
        drug2: Deuxième médicament  
        level: Niveau d'interaction
        explanation: Explication de l'interaction
    """
    level_lower = level.lower()
    css_class = f"interaction-{level_lower}" if level_lower in ['major', 'moderate', 'minor'] else "interaction-none"
    
    icon_map = {
        'major': '🔴',
        'moderate': '🟡', 
        'minor': '🟢',
        'aucune': '⚪'
    }
    
    icon = icon_map.get(level_lower, '⚪')
    
    st.markdown(f"""
    <div class="{css_class}">
        <h4>{icon} {drug1} + {drug2}</h4>
        <div class="level-badge {level_lower}">{level}</div>
        <p style="margin-top: 0.5rem; margin-bottom: 0;">{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

def create_status_message(message: str, message_type: str = "info"):
    """
    Crée un message d'état stylé
    
    Args:
        message: Texte du message
        message_type: Type de message (success, error, warning, info)
    """
    css_class = f"{message_type}-message"
    
    icon_map = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    }
    
    icon = icon_map.get(message_type, 'ℹ️')
    
    st.markdown(f"""
    <div class="{css_class}">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)

def create_progress_bar(current: int, total: int, title: str = "Progression"):
    """
    Crée une barre de progression stylée
    
    Args:
        current: Valeur actuelle
        total: Valeur totale
        title: Titre de la progression
    """
    percentage = (current / total * 100) if total > 0 else 0
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600;">{title}</span>
            <span>{current}/{total} ({percentage:.1f}%)</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {percentage}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_level_badge(level: str) -> str:
    """
    Crée un badge HTML pour un niveau d'interaction
    
    Args:
        level: Niveau d'interaction
        
    Returns:
        Code HTML du badge
    """
    level_lower = level.lower()
    return f'<span class="level-badge {level_lower}">{level}</span>'

def create_content_card(title: str, content: str, icon: str = "📄"):
    """
    Crée une carte de contenu stylée
    
    Args:
        title: Titre de la carte
        content: Contenu de la carte  
        icon: Icône optionnelle
    """
    st.markdown(f"""
    <div class="content-card">
        <h3>{icon} {title}</h3>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def format_interaction_level(level: str) -> str:
    """
    Formate un niveau d'interaction avec couleur et icône
    
    Args:
        level: Niveau d'interaction brut
        
    Returns:
        Niveau formaté avec HTML
    """
    level_clean = level.strip().title()
    color = LEVEL_COLORS.get(level_clean, LEVEL_COLORS['Aucune'])
    
    icon_map = {
        'Major': '🔴',
        'Moderate': '🟡',
        'Minor': '🟢', 
        'Aucune': '⚪'
    }
    
    icon = icon_map.get(level_clean, '⚪')
    
    return f"""
    <span style="color: {color}; font-weight: bold;">
        {icon} {level_clean}
    </span>
    """

def create_stats_cards(stats: Dict[str, Any]):
    """
    Crée une série de cartes statistiques
    
    Args:
        stats: Dictionnaire des statistiques à afficher
    """
    if not stats:
        return
    
    # Créer les colonnes dynamiquement
    num_stats = len(stats)
    cols = st.columns(min(num_stats, 4))  # Maximum 4 colonnes
    
    for i, (key, value) in enumerate(stats.items()):
        with cols[i % 4]:
            # Formater la clé (remplacer underscores par espaces, capitaliser)
            display_key = key.replace('_', ' ').title()
            
            # Formater la valeur
            if isinstance(value, float):
                display_value = f"{value:.1f}"
            elif isinstance(value, int):
                display_value = f"{value:,}"
            else:
                display_value = str(value)
            
            create_metric_card(display_key, display_value)

# def sidebar_info(title: str, content: str):
#     """
#     Ajoute une section d'information dans la sidebar
    
#     Args:
#         title: Titre de la section
#         content: Contenu en markdown
#     """
#     st.sidebar.markdown(f"""
#     <div style="background: black; padding: 1rem; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
#         <h4 style="color: var(--primary-color); margin-top: 0;">{title}</h4>
#         {content}
#     </div>
#     """, unsafe_allow_html=True)
def sidebar_info(content: str):
    """
    Ajoute une section d'information dans la sidebar
    
    Args:
        title: Titre de la section
        content: Contenu en markdown
    """
    st.sidebar.markdown(f"""
        {content}""", unsafe_allow_html=True)

def create_download_button(data, filename: str, mime_type: str, label: str):
    """
    Crée un bouton de téléchargement stylé
    
    Args:
        data: Données à télécharger
        filename: Nom du fichier
        mime_type: Type MIME
        label: Texte du bouton
    """
    return st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type,
        help=f"Télécharger {filename}"
    )

# Configuration globale de la page
def setup_page_config():
    """Configure la page Streamlit avec les paramètres optimaux"""
    st.set_page_config(
        page_title="Analyseur d'Interactions Médicamenteuses",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': """
            # Analyseur d'Interactions Médicamenteuses
            
            Application développée avec une architecture modulaire
            pour l'analyse intelligente des interactions médicamenteuses.
            
            **Version**: 2.0 Modulaire
            **Technologies**: Streamlit, LangChain, FAISS, Google Gemini
            """
        }
    )
