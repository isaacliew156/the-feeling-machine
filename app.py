"""
The Feeling Machine - AI Emotion Analysis App
Compare BERT, CNN+GloVe, and Traditional ML models for emotion classification
Advanced emotion understanding through AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import nltk
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data for Traditional ML preprocessing"""
    nltk_packages = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger']
    for package in nltk_packages:
        try:
            # Check different potential paths for the packages
            if package in ['punkt', 'averaged_perceptron_tagger']:
                path = f'tokenizers/{package}'
            elif package == 'stopwords':
                path = f'corpora/{package}'
            elif package == 'wordnet':
                path = f'corpora/{package}'
            elif package == 'vader_lexicon':
                path = f'vader_lexicon'
            else:
                path = package
                
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                st.warning(f"Could not download NLTK package {package}: {e}")

# Initialize NLTK (run once)
download_nltk_data()

# Import custom classes for pickle compatibility
try:
    # Import from utils folder
    from utils.preprocessing import TextPreprocessor, Config
    from utils.feature_engineering import FeatureEngineer, FeatureExtractor
    from utils.translator import get_translator, translate_text, TranslationResult
    from utils.languages import get_text, get_emotion_label, get_available_languages, LANGUAGES
    logger.info("Successfully imported custom classes from utils")
except ImportError as e:
    logger.error(f"Failed to import custom classes: {e}")
    # These classes might be needed for pickle compatibility
    # Create placeholder classes if not available
    class TextPreprocessor: pass
    class Config: pass
    class FeatureEngineer: pass
    class FeatureExtractor: pass

# Translation helper functions
def t(key: str, **kwargs) -> str:
    """
    Convenience function to get translated text for current UI language
    
    Args:
        key: Translation key
        **kwargs: Format parameters
        
    Returns:
        Translated text in current UI language
    """
    current_lang = st.session_state.get('ui_language', 'en')
    return get_text(key, current_lang, **kwargs)

def te(emotion: str) -> str:
    """
    Convenience function to get translated emotion label for current UI language
    
    Args:
        emotion: Emotion key
        
    Returns:
        Translated emotion label in current UI language  
    """
    current_lang = st.session_state.get('ui_language', 'en')
    return get_emotion_label(emotion, current_lang)

# Import our model loaders
from model_loaders import (
    # Model loaders
    get_bert_loader, get_embedding_loader, get_traditional_ml_loader, get_ensemble_loader,
    # Convenience functions
    predict_bert, predict_embedding, predict_traditional_ml, predict_ensemble,
    load_ensemble_models, get_ensemble_info,
    # Utils
    EMOTION_LABELS, calculate_model_agreement, format_prediction_time,
    get_model_summary_stats, PredictionResult
)

# Import new UI components
try:
    from ui_components import (
        get_global_styles, get_emotion_colors, apply_custom_css,
        create_tab_navigation, display_prediction_results, create_comparison_chart
    )
    from ui_components.styles import ColorScheme, get_emotion_color_mapping
    from ui_components.navigation import (
        create_quick_analysis_tab, create_model_settings_tab, 
        create_batch_analysis_tab, apply_tab_styles
    )
    from ui_components.results_display import display_prediction_results as enhanced_display_results
    UI_COMPONENTS_AVAILABLE = True
    logger.info("Successfully imported enhanced UI components")
except ImportError as e:
    logger.warning(f"Enhanced UI components not available: {e}")
    UI_COMPONENTS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="The Feeling Machine - AI Emotion Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimalist White Theme CSS  
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 🎨 Elegant Gradient Color Variables */
    :root {
        --bg-primary: linear-gradient(135deg, #f8faff 0%, #e8f2ff 30%, #f1f8ff 70%, #ffffff 100%);
        --bg-secondary: rgba(248, 250, 255, 0.6);
        --bg-card: rgba(255, 255, 255, 0.95);
        --border-light: #e5e7eb;
        --border-medium: #d1d5db;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-tertiary: #9ca3af;
        --accent-blue: #2563eb;
        --accent-blue-light: #eff6ff;
        --hover-bg: #f3f4f6;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }

    /* 🌟 Global Styling */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-sizing: border-box;
    }

    .stApp {
        background: var(--bg-primary);
        background-attachment: fixed;
        color: var(--text-primary);
        line-height: 1.6;
        padding: 0 2rem;
        min-height: 100vh;
        position: relative;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(37, 99, 235, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(59, 130, 246, 0.04) 0%, transparent 50%);
        pointer-events: none;
        z-index: -2;
    }
    
    /* ✨ 浮动几何图形层 */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            /* 主要大圆形 - 蓝色系 */
            radial-gradient(circle 120px at 15% 25%, rgba(59, 130, 246, 0.12) 0%, rgba(59, 130, 246, 0.03) 50%, transparent 70%),
            /* 右上中圆形 - 紫色系 */
            radial-gradient(circle 80px at 85% 15%, rgba(167, 139, 250, 0.10) 0%, rgba(167, 139, 250, 0.02) 50%, transparent 70%),
            /* 右下小圆形 - 绿色系 */
            radial-gradient(circle 50px at 80% 80%, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.02) 50%, transparent 70%),
            /* 左下椭圆形 - 橙色系 */
            radial-gradient(ellipse 140px 90px at 20% 85%, rgba(251, 146, 60, 0.06) 0%, rgba(251, 146, 60, 0.01) 50%, transparent 70%),
            /* 右中小椭圆 - 粉色系 */
            radial-gradient(ellipse 70px 45px at 90% 45%, rgba(236, 72, 153, 0.05) 0%, rgba(236, 72, 153, 0.01) 50%, transparent 70%),
            /* 中央微小圆 - 青色系 */
            radial-gradient(circle 35px at 50% 40%, rgba(6, 182, 212, 0.07) 0%, rgba(6, 182, 212, 0.01) 50%, transparent 70%),
            /* 左中椭圆 - 靛蓝系 */
            radial-gradient(ellipse 90px 60px at 10% 50%, rgba(99, 102, 241, 0.05) 0%, rgba(99, 102, 241, 0.01) 50%, transparent 70%);
        animation: floatingShapes 28s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
        opacity: 0.7;
    }
    
    /* 🌊 高级浮动动画关键帧 */
    @keyframes floatingShapes {
        0% {
            transform: translateY(0px) translateX(0px) rotate(0deg) scale(1);
            filter: blur(0px);
            opacity: 0.7;
        }
        20% {
            transform: translateY(-15px) translateX(8px) rotate(3deg) scale(1.02);
            filter: blur(0.3px);
            opacity: 0.5;
        }
        40% {
            transform: translateY(-25px) translateX(5px) rotate(-2deg) scale(0.98);
            filter: blur(0.8px);
            opacity: 0.8;
        }
        60% {
            transform: translateY(-30px) translateX(-3px) rotate(4deg) scale(1.05);
            filter: blur(1.2px);
            opacity: 0.4;
        }
        80% {
            transform: translateY(-18px) translateX(-8px) rotate(-1deg) scale(0.96);
            filter: blur(0.6px);
            opacity: 0.6;
        }
        100% {
            transform: translateY(0px) translateX(0px) rotate(0deg) scale(1);
            filter: blur(0px);
            opacity: 0.7;
        }
    }

    /* 🎭 脉动动画效果 */
    @keyframes pulse {
        0%, 100% {
            opacity: 0.3;
            transform: scale(1);
        }
        50% {
            opacity: 0.6;
            transform: scale(1.05);
        }
    }
    
    /* 🌀 旋转脉动动画 */
    @keyframes rotateAndPulse {
        0% {
            transform: rotate(0deg) scale(1);
            opacity: 0.2;
        }
        25% {
            transform: rotate(90deg) scale(1.1);
            opacity: 0.4;
        }
        50% {
            transform: rotate(180deg) scale(1);
            opacity: 0.3;
        }
        75% {
            transform: rotate(270deg) scale(0.9);
            opacity: 0.5;
        }
        100% {
            transform: rotate(360deg) scale(1);
            opacity: 0.2;
        }
    }

    /* 📐 Better Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
        position: relative;
    }

    /* ✨ Section Spacing */
    h2, h3 {
        margin-top: 3rem;
        margin-bottom: 1.5rem;
    }

    .stMarkdown p {
        margin-bottom: 1.2rem;
    }

    /* 🚫 Hide Streamlit Elements */
    .css-1d391kg { display: none; }
    #MainMenu { visibility: hidden; }
    .stDeployButton { display: none !important; }
    footer { visibility: hidden; }
    .stActionButton { display: none; }
    header[data-testid="stHeader"] { display: none !important; }
    .stApp > header { display: none; }
    .viewerBadge_container__1QSob { display: none !important; }
    .viewerBadge_link__1S137 { display: none !important; }

    /* 🏷️ Elegant Glass Cards */
    .simple-card {
        background: var(--bg-card);
        border: 1px solid rgba(229, 231, 235, 0.4);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.08),
            0 1px 3px rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .simple-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
    }

    .simple-card:hover {
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.12),
            0 2px 8px rgba(59, 130, 246, 0.08);
        border-color: rgba(59, 130, 246, 0.2);
        transform: translateY(-4px);
    }

    /* 🔘 Clean Buttons */
    .stButton > button {
        background: var(--accent-blue);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-sm);
        letter-spacing: -0.01em;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }

    /* 📝 Enhanced Input Elements */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.8);
        border: 1.5px solid rgba(229, 231, 235, 0.5);
        border-radius: 16px;
        color: var(--text-primary);
        padding: 18px 22px;
        font-size: 0.95rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: inherit;
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(37, 99, 235, 0.6);
        background: rgba(255, 255, 255, 0.95);
        outline: none;
        box-shadow: 
            0 0 0 4px rgba(37, 99, 235, 0.1),
            0 4px 20px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }

    /* 🏷️ Simple Emotion Tags */
    .emotion-high { 
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        padding: 6px 12px;
        border-radius: 20px;
        margin: 2px 4px;
        display: inline-block;
        font-weight: 500;
        font-size: 0.85rem;
    }

    .emotion-medium { 
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fed7aa;
        padding: 6px 12px;
        border-radius: 20px;
        margin: 2px 4px;
        display: inline-block;
        font-weight: 500;
        font-size: 0.85rem;
    }

    .emotion-low { 
        background: #f0fdf4;
        color: #16a34a;
        border: 1px solid #bbf7d0;
        padding: 6px 12px;
        border-radius: 20px;
        margin: 2px 4px;
        display: inline-block;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* 📊 Enhanced Metrics */
    .stMetric {
        background: var(--bg-card);
        border: 1px solid rgba(229, 231, 235, 0.4);
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 2px 12px rgba(0, 0, 0, 0.06),
            0 1px 3px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(8px);
    }

    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 6px 25px rgba(0, 0, 0, 0.1),
            0 2px 8px rgba(59, 130, 246, 0.06);
        border-color: rgba(59, 130, 246, 0.2);
    }

    /* 🎚️ Clean Slider */
    .stSlider .st-bf {
        background: var(--border-light);
        border-radius: 4px;
        height: 6px;
    }
    
    .stSlider .st-bh {
        background: var(--accent-blue);
        border-radius: 4px;
        height: 6px;
    }

    /* 📱 Enhanced Radio Buttons */
    .stRadio > div {
        background: var(--bg-secondary);
        border: 1px solid rgba(229, 231, 235, 0.3);
        border-radius: 12px;
        padding: 16px;
        display: flex;
        gap: 18px;
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .stRadio label {
        color: var(--text-primary);
        font-weight: 500;
        cursor: pointer;
        padding: 8px 12px;
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    .stRadio label:hover {
        background: var(--hover-bg);
        color: var(--accent-blue);
    }

    /* 📈 Simple Progress Bar */
    .stProgress .st-bo {
        background: var(--border-light);
        border-radius: 4px;
        height: 8px;
    }

    .stProgress .st-bp {
        background: var(--accent-blue);
        border-radius: 4px;
        height: 8px;
    }

    /* 🎭 Clean Headers */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1.5rem;
        line-height: 1.2;
    }

    h1 { 
        font-size: 3rem; 
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.5rem; }

    /* 🔔 Clean Alert Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        border-left: 4px solid;
        padding: 16px 20px;
        margin: 16px 0;
        font-weight: 500;
    }

    .stSuccess {
        background: #f0fdf4;
        border-left-color: #16a34a;
        color: #15803d;
    }

    .stError {
        background: #fef2f2;
        border-left-color: #dc2626;
        color: #dc2626;
    }

    .stInfo {
        background: #eff6ff;
        border-left-color: var(--accent-blue);
        color: var(--accent-blue);
    }

    .stWarning {
        background: #fffbeb;
        border-left-color: #d97706;
        color: #d97706;
    }

    /* 🖱️ Clean Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { 
        background: var(--bg-secondary); 
    }
    ::-webkit-scrollbar-thumb { 
        background: var(--border-medium); 
        border-radius: 4px; 
    }
    ::-webkit-scrollbar-thumb:hover { 
        background: var(--text-tertiary); 
    }

    /* ✨ Advanced Animations */
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* 🚀 Custom Loading Animation */
    .custom-loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 3rem;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 50%, #f0f9ff 100%);
        border-radius: 20px;
        border: 2px solid var(--accent-blue);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }

    .custom-loading::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(37, 99, 235, 0.1), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .emotion-particles {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 1rem 0;
    }

    .particle {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: bounce 1.5s infinite ease-in-out;
    }

    .particle:nth-child(1) { background: #ef4444; animation-delay: -0.32s; }
    .particle:nth-child(2) { background: #f97316; animation-delay: -0.16s; }
    .particle:nth-child(3) { background: #eab308; animation-delay: 0s; }
    .particle:nth-child(4) { background: #22c55e; animation-delay: 0.16s; }
    .particle:nth-child(5) { background: #3b82f6; animation-delay: 0.32s; }
    .particle:nth-child(6) { background: #8b5cf6; animation-delay: 0.48s; }

    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.8) translateY(0); opacity: 0.5; }
        40% { transform: scale(1.2) translateY(-10px); opacity: 1; }
    }

    .progress-steps {
        width: 100%;
        max-width: 400px;
        margin: 1.5rem 0;
    }

    .step {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.7);
        border-left: 4px solid var(--border-light);
        transition: all 0.8s ease;
        opacity: 0.4;
        transform: translateX(-10px);
    }

    .step.active {
        opacity: 1;
        transform: translateX(0);
        border-left-color: var(--accent-blue);
        background: rgba(37, 99, 235, 0.1);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }

    .step.completed {
        opacity: 0.7;
        border-left-color: #22c55e;
        background: rgba(34, 197, 94, 0.1);
    }

    /* 🎯 Enhanced Button Animations */
    .stButton > button {
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transition: width 0.6s, height 0.6s;
        transform: translate(-50%, -50%);
        z-index: 0;
    }

    .stButton > button:active::before {
        width: 300px;
        height: 300px;
    }

    /* Pulse animation for primary button */
    .stButton > button[kind="primary"] {
        animation: buttonPulse 2s infinite;
        box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.7);
    }

    @keyframes buttonPulse {
        0% {
            box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(37, 99, 235, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(37, 99, 235, 0);
        }
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #1d4ed8, #2563eb, #3b82f6);
        background-size: 200% 200%;
        animation: gradientShift 2s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 🎊 Confetti Animation */
    .confetti {
        position: fixed;
        width: 10px;
        height: 10px;
        background: #f0f;
        animation: confetti-fall 3s linear infinite;
        z-index: 1000;
    }

    .confetti:nth-child(1) { left: 10%; animation-delay: 0s; background: #ff6b6b; }
    .confetti:nth-child(2) { left: 20%; animation-delay: 0.2s; background: #4ecdc4; }
    .confetti:nth-child(3) { left: 30%; animation-delay: 0.4s; background: #45b7d1; }
    .confetti:nth-child(4) { left: 40%; animation-delay: 0.6s; background: #96ceb4; }
    .confetti:nth-child(5) { left: 50%; animation-delay: 0.8s; background: #ffeaa7; }
    .confetti:nth-child(6) { left: 60%; animation-delay: 1s; background: #fab1a0; }
    .confetti:nth-child(7) { left: 70%; animation-delay: 1.2s; background: #e17055; }
    .confetti:nth-child(8) { left: 80%; animation-delay: 1.4s; background: #a29bfe; }
    .confetti:nth-child(9) { left: 90%; animation-delay: 1.6s; background: #fd79a8; }

    @keyframes confetti-fall {
        0% {
            transform: translateY(-100vh) rotate(0deg);
            opacity: 1;
        }
        100% {
            transform: translateY(100vh) rotate(720deg);
            opacity: 0;
        }
    }

    /* 💫 Number Count Animation */
    .animated-number {
        font-weight: bold;
        color: var(--accent-blue);
        transition: all 0.3s ease;
    }

    /* 🌟 Enhanced Cards with Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(31, 41, 55, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(31, 41, 55, 0.15);
    }

    /* 🔗 Clean Expander */
    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        color: var(--text-primary);
        font-weight: 600;
        padding: 16px 20px;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--hover-bg);
        border-color: var(--border-medium);
    }

    /* 📋 Clean File Uploader */
    .stFileUploader {
        background: var(--bg-secondary);
        border: 2px dashed var(--border-medium);
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-blue);
        background: var(--accent-blue-light);
    }

    /* 📊 Clean Selectbox */
    .stSelectbox select {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        color: var(--text-primary);
        padding: 8px 12px;
    }

    /* 🔘 Toggle */
    .stCheckbox {
        padding: 8px 0;
    }
    
    .stCheckbox label {
        color: var(--text-primary);
        font-weight: 500;
    }

    /* 🎯 Clean Multiselect */
    .stMultiSelect > div > div {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: 8px;
    }

    .stMultiSelect > div > div > div {
        color: var(--text-primary);
        font-weight: 500;
    }

    /* Override red selection styles */
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--accent-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        margin: 2px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    .stMultiSelect [data-baseweb="tag"] [data-baseweb="tag-primary-action"] {
        color: white !important;
    }

    .stMultiSelect [data-baseweb="tag"] [data-baseweb="tag-close"] {
        color: rgba(255, 255, 255, 0.8) !important;
        margin-left: 4px !important;
    }

    .stMultiSelect [data-baseweb="tag"] [data-baseweb="tag-close"]:hover {
        color: white !important;
        background: rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Add JavaScript for enhanced interactions
st.markdown("""
<script>
    // Custom loading animation controller
    function showCustomLoading() {
        const loadingHTML = `
            <div class="custom-loading" id="customLoading">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎭</div>
                <div class="emotion-particles">
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                </div>
                <div class="progress-steps">
                    <div class="step active" id="step1">
                        <strong>🔍 Analyzing text structure...</strong>
                    </div>
                    <div class="step" id="step2">
                        <strong>🧠 Processing emotions...</strong>
                    </div>
                    <div class="step" id="step3">
                        <strong>✨ Generating results...</strong>
                    </div>
                </div>
                <div style="color: var(--text-secondary); text-align: center; font-size: 0.9rem;">
                    Our AI models are working their magic...
                </div>
            </div>
        `;
        
        return loadingHTML;
    }
    
    // Progress through loading steps
    function animateLoadingSteps() {
        setTimeout(() => {
            const step1 = document.getElementById('step1');
            const step2 = document.getElementById('step2');
            if (step1 && step2) {
                step1.classList.remove('active');
                step1.classList.add('completed');
                step2.classList.add('active');
            }
        }, 1500);
        
        setTimeout(() => {
            const step2 = document.getElementById('step2');
            const step3 = document.getElementById('step3');
            if (step2 && step3) {
                step2.classList.remove('active');
                step2.classList.add('completed');
                step3.classList.add('active');
            }
        }, 3000);
        
        setTimeout(() => {
            const step3 = document.getElementById('step3');
            if (step3) {
                step3.classList.remove('active');
                step3.classList.add('completed');
            }
        }, 4000);
    }
    
    // Confetti celebration
    function triggerConfetti() {
        const confettiContainer = document.createElement('div');
        confettiContainer.id = 'confettiContainer';
        confettiContainer.style.position = 'fixed';
        confettiContainer.style.top = '0';
        confettiContainer.style.left = '0';
        confettiContainer.style.width = '100%';
        confettiContainer.style.height = '100%';
        confettiContainer.style.pointerEvents = 'none';
        confettiContainer.style.zIndex = '9999';
        
        for (let i = 0; i < 50; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.left = Math.random() * 100 + '%';
            confetti.style.animationDelay = Math.random() * 3 + 's';
            confetti.style.background = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#fab1a0'][Math.floor(Math.random() * 6)];
            confettiContainer.appendChild(confetti);
        }
        
        document.body.appendChild(confettiContainer);
        
        setTimeout(() => {
            document.body.removeChild(confettiContainer);
        }, 4000);
    }
    
    // Number counting animation
    function animateNumber(element, endValue, duration = 1000) {
        const startValue = 0;
        const startTime = performance.now();
        
        function updateNumber(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = 1 - Math.pow(1 - progress, 3); // Ease out cubic
            
            const currentValue = startValue + (endValue - startValue) * easeProgress;
            element.textContent = Math.round(currentValue) + '%';
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            }
        }
        
        requestAnimationFrame(updateNumber);
    }
    
</script>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = {
            'bert': False,
            'embedding': False, 
            'traditional_ml': False,
            'ensemble': False
        }
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    
    if 'ensemble_strategy' not in st.session_state:
        st.session_state.ensemble_strategy = 'adaptive_cascade'
    
    # Translation-related session state
    if 'enable_translation' not in st.session_state:
        st.session_state.enable_translation = True
        
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        
    if 'last_translation' not in st.session_state:
        st.session_state.last_translation = None
    
    # UI Language settings
    if 'ui_language' not in st.session_state:
        st.session_state.ui_language = 'en'
    
    if 'ensemble_loader' not in st.session_state:
        st.session_state.ensemble_loader = None

@st.cache_data
def get_example_texts() -> Dict[str, List[str]]:
    """Get categorized example texts for testing"""
    return {
        "Positive Emotions": [
            "I'm so excited about my new job! This is going to be amazing!",
            "Thank you so much for your help, I really appreciate it!",
            "I love spending time with my family, they mean everything to me.",
            "What a beautiful sunset! Nature always fills me with wonder."
        ],
        "Negative Emotions": [
            "I can't believe you would do something so disappointing.",
            "This traffic is making me so angry and frustrated!",
            "I feel so sad and lonely since my friend moved away.",
            "I'm terrified about the upcoming presentation tomorrow."
        ],
        "Complex Emotions": [
            "I'm nervous but also excited about starting university next month.",
            "That movie was confusing but I think I understand the message now.",
            "I feel guilty for not calling my parents more often.",
            "I'm surprised by how much I enjoyed that book, didn't expect that."
        ],
        "Neutral/Mixed": [
            "The weather today is okay, nothing special really.",
            "I went to the store and bought some groceries for dinner.",
            "The meeting was informative but quite long and detailed.",
            "I'm not sure what to think about the new policy changes."
        ]
    }

def load_selected_models(selected_models: List[str]) -> Dict[str, bool]:
    """Load selected models and return loading status"""
    loading_status = {}
    
    with st.spinner("Loading selected models..."):
        progress_bar = st.progress(0)
        
        for i, model in enumerate(selected_models):
            st.write(f"Loading {model}...")
            
            try:
                if model == "BERT" and not st.session_state.models_loaded['bert']:
                    bert_loader = get_bert_loader()
                    success = bert_loader.load_model()
                    st.session_state.models_loaded['bert'] = success
                    loading_status['BERT'] = success
                    
                elif model == "CNN + GloVe" and not st.session_state.models_loaded['embedding']:
                    embedding_loader = get_embedding_loader()
                    if embedding_loader.is_available():
                        success = embedding_loader.load_model()
                        st.session_state.models_loaded['embedding'] = success
                        loading_status['CNN + GloVe'] = success
                    else:
                        st.error("TensorFlow not available for CNN + GloVe model")
                        loading_status['CNN + GloVe'] = False
                        
                elif model == "Traditional ML" and not st.session_state.models_loaded['traditional_ml']:
                    traditional_loader = get_traditional_ml_loader()
                    success = traditional_loader.load_model()
                    st.session_state.models_loaded['traditional_ml'] = success
                    loading_status['Traditional ML'] = success
                    
                elif model == "Ensemble" and not st.session_state.models_loaded['ensemble']:
                    # Ensemble requires individual models to be loaded first
                    available_models = []
                    if st.session_state.models_loaded['bert']:
                        available_models.append('BERT')
                    if st.session_state.models_loaded['embedding']:
                        available_models.append('CNN + GloVe')
                    if st.session_state.models_loaded['traditional_ml']:
                        available_models.append('Traditional ML')
                    
                    if len(available_models) >= 2:
                        ensemble_loader = get_ensemble_loader()
                        ensemble_loader.available_models = available_models
                        success_dict = ensemble_loader.load_models()
                        success = any(success_dict.values())
                        st.session_state.models_loaded['ensemble'] = success
                        st.session_state.ensemble_loader = ensemble_loader
                        loading_status['Ensemble'] = success
                        if success:
                            st.success(f"Ensemble initialized with {len(available_models)} models: {', '.join(available_models)}")
                    else:
                        st.error("Ensemble requires at least 2 individual models to be loaded first")
                        loading_status['Ensemble'] = False
                        
                else:
                    # Model already loaded
                    if model == "BERT":
                        loading_status['BERT'] = st.session_state.models_loaded['bert']
                    elif model == "CNN + GloVe":
                        loading_status['CNN + GloVe'] = st.session_state.models_loaded['embedding']
                    elif model == "Traditional ML":
                        loading_status['Traditional ML'] = st.session_state.models_loaded['traditional_ml']
                    elif model == "Ensemble":
                        loading_status['Ensemble'] = st.session_state.models_loaded['ensemble']
                
            except Exception as e:
                st.error(f"Failed to load {model}: {str(e)}")
                loading_status[model] = False
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        progress_bar.empty()
    
    return loading_status

def predict_with_models(text: str, selected_models: List[str], 
                       threshold: float, use_optimal: bool) -> Dict[str, PredictionResult]:
    """Predict emotions using selected models"""
    results = {}
    
    for model in selected_models:
        try:
            if model == "BERT" and st.session_state.models_loaded['bert']:
                result = predict_bert(text, threshold, use_optimal)
                results['BERT'] = result
                
            elif model == "CNN + GloVe" and st.session_state.models_loaded['embedding']:
                result = predict_embedding(text, threshold, use_optimal)
                results['CNN + GloVe'] = result
                
            elif model == "Traditional ML" and st.session_state.models_loaded['traditional_ml']:
                result = predict_traditional_ml(text, threshold, use_optimal)
                results['Traditional ML'] = result
                
            elif model == "Ensemble" and st.session_state.models_loaded['ensemble']:
                if st.session_state.ensemble_loader:
                    strategy = st.session_state.ensemble_strategy
                    result = st.session_state.ensemble_loader.predict(
                        text, strategy=strategy, threshold=threshold, use_optimal=use_optimal
                    )
                    results['Ensemble'] = result
                else:
                    st.error("Ensemble loader not initialized")
                
        except Exception as e:
            st.error(f"Prediction failed for {model}: {str(e)}")
            
    return results

def create_radar_chart(results: Dict[str, PredictionResult], title: str = "Model Comparison"):
    """Create radar chart comparing emotion predictions"""
    if not results:
        return None
    
    # Get top emotions across all models
    all_emotions = set()
    for result in results.values():
        all_emotions.update(result.emotion_scores.keys())
    
    # Select top emotions by maximum score across models
    emotion_max_scores = {}
    for emotion in all_emotions:
        max_score = max(result.emotion_scores.get(emotion, 0) for result in results.values())
        emotion_max_scores[emotion] = max_score
    
    top_emotions = sorted(emotion_max_scores.items(), key=lambda x: x[1], reverse=True)[:12]
    selected_emotions = [emotion for emotion, _ in top_emotions]
    
    fig = go.Figure()
    
    colors = ['#2563eb', '#7c3aed', '#059669', '#dc2626', '#d97706', '#0891b2']
    
    for i, (model_name, result) in enumerate(results.items()):
        scores = [result.emotion_scores.get(emotion, 0) for emotion in selected_emotions]
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=selected_emotions,
            fill='toself',
            name=model_name,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10, color='#6b7280'),
                gridcolor='#e5e7eb',
                linecolor='#d1d5db'
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='#6b7280'),
                gridcolor='#e5e7eb',
                linecolor='#d1d5db'
            ),
            bgcolor='#ffffff'
        ),
        showlegend=True,
        title=dict(
            text=title, 
            x=0.5, 
            y=0.95,  # Position title higher to avoid legend overlap
            xanchor='center',
            yanchor='top',
            font=dict(size=18, color='#1f2937', family="Inter")
        ),
        height=450,  # Match bar chart height
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,  # Slightly lower to give space for title
            xanchor="center",  # Center instead of right
            x=0.5,   # Center position
            font=dict(color='#1f2937', family="Inter")
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color='#1f2937', family="Inter"),
        margin=dict(t=80, b=50, l=50, r=50)  # Increased top margin for title
    )
    
    return fig

def create_comparison_bar_chart(results: Dict[str, PredictionResult]):
    """Create bar chart comparing top emotions across models"""
    if not results:
        return None
    
    # Get top 8 emotions across all models
    all_scores = {}
    for model_name, result in results.items():
        for emotion, score in result.emotion_scores.items():
            if emotion not in all_scores:
                all_scores[emotion] = []
            all_scores[emotion].append((model_name, score))
    
    # Calculate average scores and get top emotions
    avg_scores = {emotion: np.mean([score for _, score in scores]) 
                 for emotion, scores in all_scores.items()}
    
    top_emotions = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:8]
    selected_emotions = [emotion for emotion, _ in top_emotions]
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=1,
        vertical_spacing=0.08
    )
    
    colors = {'BERT': '#2563eb', 'CNN + GloVe': '#7c3aed', 'Traditional ML': '#059669', 'Ensemble': '#dc2626'}
    
    for model_name, result in results.items():
        scores = [result.emotion_scores.get(emotion, 0) for emotion in selected_emotions]
        
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=selected_emotions,
                y=scores,
                marker_color=colors.get(model_name, '#96CEB4'),
                opacity=0.8
            )
        )
    
    fig.update_layout(
        barmode='group',
        height=450,  # Increased height for better spacing
        xaxis_title="Emotions",
        yaxis_title="Confidence Score",
        title=dict(
            text="",
            y=0.95,  # Move title up
            x=0.5,   # Center title
            xanchor='center',
            yanchor='top',
            font=dict(color='#1f2937', family="Inter")
        ),
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=1.02,  # Position legend below title
            xanchor="center", 
            x=0.5,   # Center legend
            font=dict(color='#1f2937', family="Inter")
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(color='#6b7280', family="Inter"),
            title=dict(font=dict(color='#1f2937', family="Inter")),
            gridcolor='#f3f4f6',
            zerolinecolor='#e5e7eb'
        ),
        yaxis=dict(
            tickfont=dict(color='#6b7280', family="Inter"),
            title=dict(font=dict(color='#1f2937', family="Inter")),
            gridcolor='#f3f4f6',
            zerolinecolor='#e5e7eb'
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color='#1f2937', family="Inter"),
        margin=dict(t=80, b=50, l=50, r=50),  # Increased top margin for title
        annotations=[
            dict(
                text="Top Emotions by Model",
                x=0.5,
                y=1.05,  # Lower position to avoid being cut off
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                font=dict(size=16, color='#1f2937', family="Inter", weight="bold"),
                showarrow=False
            )
        ]
    )
    
    return fig

def display_prediction_results(results: Dict[str, PredictionResult], text: str):
    """Display prediction results in a formatted way with animations"""
    if not results:
        st.warning("No prediction results to display")
        return
    
    st.subheader(t('results_title'))
    
    # Display input text
    st.markdown(f"**{t('input_text')}** *{text[:200]}{'...' if len(text) > 200 else ''}*")
    
    # Create columns for each model
    cols = st.columns(len(results))
    
    for i, (model_name, result) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"### {model_name}")
            
            # Performance metrics with confidence display
            confidence = result.get_confidence()
            confidence_percent = int(confidence * 100)
            
            # Create confidence display with elegant styling
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0; padding: 0.5rem; 
                       background: rgba(37, 99, 235, 0.05); 
                       border-radius: 12px; border: 1px solid rgba(37, 99, 235, 0.1);">
                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">{t('max_confidence')}</div>
                <div style="font-size: 2rem; font-weight: bold; color: var(--accent-blue); 
                           text-shadow: 0 1px 3px rgba(37, 99, 235, 0.3);">
                    {confidence_percent}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if result.prediction_time:
                st.metric(t('prediction_time'), format_prediction_time(result.prediction_time))
            
            # Top emotions with fade-in animation
            top_emotions = result.get_top_emotions(5)
            
            st.markdown(f"**{t('top_emotions')}**")
            st.markdown(f"""
            <div class="emotions-container" id="emotions_{i}">
            """, unsafe_allow_html=True)
            
            for j, (emotion, score) in enumerate(top_emotions):
                if score > 0.7:
                    css_class = "emotion-high"
                elif score > 0.4:
                    css_class = "emotion-medium"
                else:
                    css_class = "emotion-low"
                
                st.markdown(f"""
                <div class="{css_class}" style="opacity: 0; transform: translateX(-20px); transition: all 0.5s ease; margin: 4px 2px; animation: slideInLeft 0.6s ease forwards {j * 0.1}s;">
                    {te(emotion)}: {score:.1%}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Predicted emotions (above threshold)
            if result.predicted_emotions:
                translated_emotions = [te(emotion) for emotion in result.predicted_emotions]
                st.markdown(f"**{t('predicted')}:** {', '.join(translated_emotions)}")
            else:
                st.markdown(f"**{t('no_results')}**")
    
    # Add slide-in animation CSS
    st.markdown("""
    <style>
        @keyframes slideInLeft {
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .emotions-container > div {
            display: inline-block;
            margin: 2px 4px;
        }
    </style>
    """, unsafe_allow_html=True)

def create_model_agreement_analysis(results: Dict[str, PredictionResult]):
    """Create model agreement analysis"""
    if len(results) < 2:
        return None
    
    agreement = calculate_model_agreement(list(results.values()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Agreement", f"{agreement['agreement_percentage']:.1f}%")
        st.metric("Common Emotions", len(agreement['common_emotions']))
    
    with col2:
        st.metric("Total Unique Emotions", agreement['total_unique_emotions'])
        if agreement['common_emotions']:
            st.markdown("""
            <div style="margin: 1rem 0;">
                <div style="margin-bottom: 0.75rem; font-weight: 600; color: var(--text-primary); font-size: 0.95rem;">
                    🤝 Common Emotions:
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
            """, unsafe_allow_html=True)
            
            # Display each common emotion as a beautiful tag
            for emotion in agreement['common_emotions']:
                st.markdown(f"""
                <div class="emotion-high" style="display: inline-block; margin: 2px; 
                           background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%);
                           color: #1e40af; border: 1px solid #93c5fd; 
                           padding: 6px 12px; border-radius: 16px; font-weight: 500; 
                           font-size: 0.85rem; box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);">
                    ✨ {emotion}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

def create_model_selection():
    """Create clean model selection interface"""
    st.markdown(f"### {t('choose_models')}")
    st.markdown(t('choose_models_desc'))
    
    # Model selection with multiselect
    available_models = ["BERT", "CNN + GloVe", "Traditional ML", "Ensemble"]
    selected_models = st.multiselect(
        t('select_models'),
        available_models,
        default=["BERT", "Traditional ML"],
        help="Choose which models to load and compare",
        key="model_selector"
    )
    
    if selected_models:
        st.markdown("---")
        
        # Show simple info cards for selected models
        models_info = {
            "BERT": {"icon": "🧠", "desc": "Transformer-based • High accuracy • Context understanding"},
            "CNN + GloVe": {"icon": "🌐", "desc": "Neural network • Fast processing • Good for short texts"},
            "Traditional ML": {"icon": "📊", "desc": "Classical ML • Lightweight • Interpretable results"},
            "Ensemble": {"icon": "🎭", "desc": "Combined models • Highest accuracy • Smart fusion"}
        }
        
        cols = st.columns(len(selected_models))
        for i, model in enumerate(selected_models):
            with cols[i]:
                if model in models_info:
                    info = models_info[model]
                    # Enhanced card with glassmorphism and 3D effects
                    st.markdown(f"""
                    <div class="glass-card model-card" style="
                        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
                        backdrop-filter: blur(15px);
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        border-radius: 16px;
                        padding: 1.25rem;
                        text-align: center;
                        box-shadow: 0 8px 32px rgba(31, 41, 55, 0.12), 0 2px 16px rgba(31, 41, 55, 0.08);
                        margin-bottom: 1rem;
                        position: relative;
                        overflow: hidden;
                        transform: perspective(1000px) rotateX(0deg) rotateY(0deg);
                        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                    " onmouseover="
                        this.style.transform = 'perspective(1000px) rotateX(-5deg) rotateY(5deg) translateY(-8px)';
                        this.style.boxShadow = '0 20px 40px rgba(31, 41, 55, 0.2), 0 8px 24px rgba(31, 41, 55, 0.12)';
                    " onmouseout="
                        this.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)';
                        this.style.boxShadow = '0 8px 32px rgba(31, 41, 55, 0.12), 0 2px 16px rgba(31, 41, 55, 0.08)';
                    ">
                        <div style="
                            position: absolute;
                            top: 0;
                            left: 0;
                            right: 0;
                            height: 3px;
                            background: linear-gradient(90deg, #2563eb, #7c3aed, #059669, #dc2626);
                            background-size: 300% 100%;
                            animation: gradientMove 4s ease infinite;
                        "></div>
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 0.75rem;">
                            <div style="
                                font-size: 2rem;
                                filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
                                margin-bottom: 0.25rem;
                            ">{info['icon']}</div>
                            <div style="
                                font-weight: 700;
                                background: linear-gradient(135deg, #1f2937 0%, #2563eb 100%);
                                -webkit-background-clip: text;
                                -webkit-text-fill-color: transparent;
                                background-clip: text;
                                font-size: 1rem;
                                letter-spacing: -0.025em;
                            ">{model}</div>
                            <div style="
                                color: var(--text-secondary);
                                font-size: 0.8rem;
                                line-height: 1.4;
                                font-weight: 500;
                            ">{info['desc']}</div>
                        </div>
                    </div>
                    
                    <style>
                        @keyframes gradientMove {{
                            0% {{ background-position: 0% 50%; }}
                            50% {{ background-position: 100% 50%; }}
                            100% {{ background-position: 0% 50%; }}
                        }}
                        
                        .model-card:hover::before {{
                            content: '';
                            position: absolute;
                            top: -50%;
                            left: -50%;
                            width: 200%;
                            height: 200%;
                            background: radial-gradient(circle, rgba(37, 99, 235, 0.1) 0%, transparent 50%);
                            animation: rotate 10s linear infinite;
                            pointer-events: none;
                        }}
                        
                        @keyframes rotate {{
                            from {{ transform: rotate(0deg); }}
                            to {{ transform: rotate(360deg); }}
                        }}
                    </style>
                    """, unsafe_allow_html=True)
    
    return selected_models

def create_settings_panel():
    """Create the model settings panel as a collapsible card"""
    with st.expander(t('settings'), expanded=True):
        # Model selection
        selected_models = create_model_selection()
        
        if not selected_models:
            st.warning("Please select at least one model")
            return None, None, None
        
        # Load models and settings in a clean layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Load models button
            if st.button("🚀 Load Selected Models", type="primary", use_container_width=True):
                loading_status = load_selected_models(selected_models)
                
                # Show simple loading result
                success_count = sum(loading_status.values())
                total_count = len(loading_status)
                if success_count == total_count:
                    st.success(f"✅ All {success_count} models loaded successfully!")
                elif success_count > 0:
                    st.warning(f"⚠️ {success_count}/{total_count} models loaded")
                else:
                    st.error("❌ No models could be loaded")
        
        with col2:
            # Simple threshold setting
            use_optimal = st.toggle("🎯 Use Optimal Thresholds", value=True, 
                                   help="Recommended: Use model-optimized thresholds")
            
            if not use_optimal:
                threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            else:
                threshold = 0.5
        
        # Translation settings
        st.markdown("---")
        st.markdown(f"**{t('translation_settings')}**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            enable_translation = st.toggle(t('auto_translate'), value=True,
                                         help=t('auto_translate_help'))
            
            if enable_translation:
                # API key input
                api_key_input = st.text_input(
                    t('gemini_api_key'), 
                    value=st.session_state.get('gemini_api_key', ''),
                    type="password",
                    help=t('api_key_help'),
                    placeholder="AIza..."
                )
                
                if api_key_input:
                    st.session_state.gemini_api_key = api_key_input
                    
                    # Test API key
                    translator = get_translator()
                    if translator.is_available():
                        st.success(t('api_configured'))
                        
                        # Cache stats
                        cache_stats = translator.get_cache_stats()
                        if cache_stats['cache_size'] > 0:
                            st.info(t('cache_stats', count=cache_stats['cache_size']))
                    else:
                        st.error(t('api_not_available'))
        
        with col2:
            if enable_translation and st.session_state.get('gemini_api_key'):
                if st.button(t('clear_cache'), help="Clear translation cache"):
                    translator = get_translator()
                    translator.clear_cache()
                    st.success(t('cache_cleared'))
                    st.info(t('restart_tip'))
        
        st.session_state.enable_translation = enable_translation
        
        # Language settings
        st.markdown("---")
        st.markdown(f"**{t('language_settings')}**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # UI Language selector
            available_languages = get_available_languages()
            language_options = list(available_languages.keys())
            current_lang_index = language_options.index(st.session_state.ui_language) if st.session_state.ui_language in language_options else 0
            
            selected_language = st.selectbox(
                t('ui_language'),
                options=language_options,
                format_func=lambda x: f"{available_languages[x]} ({x.upper()})",
                index=current_lang_index,
                help=t('ui_language_help')
            )
            
            # Update session state if language changed
            if selected_language != st.session_state.ui_language:
                st.session_state.ui_language = selected_language
                st.rerun()  # Refresh to apply new language
        
        with col2:
            # Language info
            current_lang_name = available_languages.get(st.session_state.ui_language, 'English')
            st.metric("📍 Current Language", current_lang_name)
        
        # Ensemble settings (only if needed)
        if "Ensemble" in selected_models:
            st.markdown("---")
            st.markdown("**🧠 Ensemble Strategy**")
            
            col1, col2 = st.columns(2)
            with col1:
                ensemble_strategies = ['adaptive_cascade', 'weighted_average', 'majority_voting']
                strategy_names = {'adaptive_cascade': '🎯 Smart Routing', 
                                'weighted_average': '⚖️ Weighted Average', 
                                'majority_voting': '🗳️ Majority Vote'}
                
                selected_strategy = st.radio(
                    "Choose strategy:",
                    ensemble_strategies,
                    format_func=lambda x: strategy_names[x],
                    horizontal=True,
                    index=0
                )
                st.session_state.ensemble_strategy = selected_strategy
            
            with col2:
                st.info("💡 Smart Routing adapts to different text types automatically")
    
    return selected_models, threshold, use_optimal

def main():
    """Main application function with enhanced UI"""
    initialize_session_state()
    
    # Apply enhanced CSS if available
    if UI_COMPONENTS_AVAILABLE:
        apply_custom_css()
        apply_tab_styles()
    
    # 🌟 Enhanced Header
    if UI_COMPONENTS_AVAILABLE:
        st.markdown(f"""
        <div class="main-header fade-in">
            <h1 class="main-title">{t('title')}</h1>
            <p class="main-subtitle">{t('subtitle')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback header for legacy mode
        st.markdown(f"""
        <div class="fade-in" style="text-align: center; padding: 3rem 0 2rem 0;">
            <h1 style="font-size: 4rem; font-weight: 700; margin-bottom: 1rem; 
                       background: linear-gradient(135deg, #1f2937 0%, #2563eb 50%, #6366f1 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       background-clip: text; letter-spacing: -0.02em;">
                {t('title')}
            </h1>
            <p style="font-size: 1.2rem; color: var(--text-secondary); font-weight: 400; margin: 0;">
                {t('subtitle')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tab navigation
    if UI_COMPONENTS_AVAILABLE:
        # Enhanced tab navigation
        tab_config, tabs = create_tab_navigation()
        
        with tabs[0]:  # Quick Analysis
            render_quick_analysis_tab()
            
        with tabs[1]:  # Model Settings
            render_model_settings_tab()
            
        with tabs[2]:  # Batch Analysis
            render_batch_analysis_tab()
    else:
            # Legacy mode - use expander navigation
        st.info("⚠️ Running in legacy mode. Enhanced UI components not available.")
        render_legacy_interface()
    
    # Footer
    render_footer()


def render_quick_analysis_tab():
    """Render the Quick Analysis tab content"""
    st.markdown(f"""
    <div class="tab-header">
        <h2>🔍 {t('quick_analysis', default='Quick Analysis')}</h2>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            {t('quick_analysis_desc', default='Analyze emotions in text using state-of-the-art AI models')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get model settings from session state or show settings prompt
    if not _check_model_settings():
        st.warning("⚙️ Please configure models in the Model Settings tab first.")
        return
    
    selected_models = st.session_state.get('selected_models', [])
    threshold = st.session_state.get('threshold', 0.5)
    use_optimal = st.session_state.get('use_optimal', True)
    
    # Quick Analysis Content
    render_text_analysis_section(selected_models, threshold, use_optimal)


def render_model_settings_tab():
    """Render the Model Settings tab content"""
    st.markdown(f"""
    <div class="tab-header">
        <h2>⚙️ {t('model_settings', default='Model Settings')}</h2>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            {t('model_settings_desc', default='Configure AI models, thresholds, and translation settings')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render settings panel
    settings_result = create_settings_panel()
    if settings_result:
        selected_models, threshold, use_optimal = settings_result
        # Store in session state for other tabs
        st.session_state.selected_models = selected_models
        st.session_state.threshold = threshold
        st.session_state.use_optimal = use_optimal
        
        if selected_models:
            st.success(f"✅ {len(selected_models)} model(s) configured: {', '.join(selected_models)}")


def render_batch_analysis_tab():
    """Render the Batch Analysis tab content"""
    st.markdown(f"""
    <div class="tab-header">
        <h2>📊 {t('batch_analysis', default='Batch Analysis')}</h2>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            {t('batch_analysis_desc', default='Upload and process multiple texts from CSV files for comprehensive analysis')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models are configured
    if not _check_model_settings():
        st.warning("⚙️ Please configure models in the Model Settings tab first.")
        return
    
    # Render batch analysis content
    render_batch_analysis_section()



def _check_model_settings():
    """Check if model settings are configured"""
    return bool(st.session_state.get('selected_models', []))


def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: var(--text-secondary);">
        <p>🎭 <strong>The Feeling Machine</strong> | Where Technology Meets Emotion</p>
        <p style="font-size: 0.9rem; opacity: 0.8; font-style: italic;">
            Crafted with ❤️ for Understanding Human Hearts • Advanced AI Emotion Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_text_analysis_section(selected_models, threshold, use_optimal):
    """Render the main text analysis section"""
    input_method = st.radio(
        t('choose_input_method'),
        [t('type_custom_text'), t('select_example_text')],
        horizontal=True
    )
    
    if input_method == t('select_example_text'):
        example_texts = get_example_texts()
        col_cat, col_ex = st.columns(2)
        with col_cat:
            category = st.selectbox("Category:", list(example_texts.keys()))
        with col_ex:
            example_options = example_texts[category]
            selected_example = st.selectbox("Example:", example_options)
        text_input = st.text_area(t('enter_text'), value=selected_example, height=120, key="example_text")
    else:
        text_input = st.text_area(t('enter_text'), 
                                 placeholder=t('text_placeholder'), 
                                 height=120, key="custom_text")
    
    # Character count and input enhancements
    char_count = len(text_input)
    word_count = len(text_input.split()) if text_input.strip() else 0
    
    # Input statistics with improved styling
    col_stats, col_shortcut = st.columns([1, 1])
    with col_stats:
        if char_count > 0:
            color = "#22c55e" if char_count <= 500 else "#f59e0b" if char_count <= 1000 else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 1rem; padding: 0.5rem 1rem; 
                       background: rgba(243, 244, 246, 0.5); border-radius: 8px; margin: 0.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.1rem;">📝</span>
                    <span style="color: {color}; font-weight: 600;">{char_count}</span>
                    <span style="color: var(--text-secondary); font-size: 0.9rem;">{t('characters')}</span>
                </div>
                <div style="height: 20px; width: 1px; background: var(--border-light);"></div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: var(--accent-blue); font-weight: 600;">{word_count}</span>
                    <span style="color: var(--text-secondary); font-size: 0.9rem;">{t('words')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_shortcut:
        # Always show the tip for better UX
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.25rem; 
                   background: rgba(239, 246, 255, 0.85); 
                   border-radius: 12px; margin: 0.75rem 0; 
                   border: 1px solid rgba(59, 130, 246, 0.25);
                   box-shadow: 0 2px 8px rgba(59, 130, 246, 0.08),
                              0 1px 3px rgba(59, 130, 246, 0.04);
                   backdrop-filter: blur(12px);
                   position: relative;
                   overflow: hidden;">
            <span style="font-size: 1rem; filter: drop-shadow(0 1px 2px rgba(37, 99, 235, 0.1));">💡</span>
            <span style="color: #1e40af; font-size: 0.85rem; font-weight: 500; 
                        line-height: 1.4;">
{t('tip')}
            </span>
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; 
                       background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.4), transparent);"></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a subtle info box below the text area
    emotions_detected_text = t('emotions_detected').replace('🎭', '').replace('\n', ' ')
    st.markdown(f"""
    <div style="background: var(--bg-secondary); border-left: 4px solid var(--accent-blue); 
               padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">🎭</span>
            <span style="font-weight: 600; color: var(--accent-blue);">{emotions_detected_text}</span>
            <span style="color: var(--text-secondary);">•</span>
            <span style="color: var(--text-secondary); font-size: 0.9rem;">
                {t('emotions_desc')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis button - centered layout
    st.markdown("---")
    
    
    
    # Center the analyze button with improved spacing
    col_space_left, col_btn, col_space_right = st.columns([1, 2, 1])
    with col_btn:
        analyze_clicked = st.button(t('analyze_button'), 
                                  type="primary", 
                                  disabled=not text_input.strip(), 
                                  use_container_width=True)
    
    if analyze_clicked:
        # Check if any models are loaded
        any_loaded = any([
            st.session_state.models_loaded.get('bert', False) and 'BERT' in selected_models,
            st.session_state.models_loaded.get('embedding', False) and 'CNN + GloVe' in selected_models,
            st.session_state.models_loaded.get('traditional_ml', False) and 'Traditional ML' in selected_models,
            st.session_state.models_loaded.get('ensemble', False) and 'Ensemble' in selected_models
        ])
        
        if not any_loaded:
            st.error("Please load at least one model first using the settings panel above")
        else:
            # Show step-by-step loading animation with actual processing
            loading_placeholder = st.empty()
            
            # Step 1: Text Analysis
            loading_placeholder.markdown("""
            <div class="custom-loading">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎭</div>
                <div class="emotion-particles">
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                </div>
                <div class="progress-steps">
                    <div class="step active">
                        <strong>🔍 Analyzing text structure...</strong>
                    </div>
                    <div class="step">
                        <strong>🧠 Processing emotions...</strong>
                    </div>
                    <div class="step">
                        <strong>✨ Generating results...</strong>
                    </div>
                </div>
                <div style="color: var(--text-secondary); text-align: center; font-size: 0.9rem;">
                    Preparing text for analysis...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Actually do text preprocessing
            time.sleep(1.5)
            
            # Translation processing if enabled
            translation_result = None
            if st.session_state.get('enable_translation', False):
                translator = get_translator()
                if translator.is_available():
                    translation_result = translator.translate_to_english(text_input.strip())
                    processed_text = translation_result.translated_text
                else:
                    processed_text = text_input.strip()
                    st.warning("⚠️ Translation enabled but API key not available. Using original text.")
            else:
                processed_text = text_input.strip()
            
            # Store translation result in session state for display
            if translation_result:
                st.session_state.last_translation = translation_result
            
            # Step 2: Model Processing
            loading_placeholder.markdown("""
            <div class="custom-loading">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎭</div>
                <div class="emotion-particles">
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                </div>
                <div class="progress-steps">
                    <div class="step completed">
                        <strong>🔍 Analyzing text structure...</strong>
                    </div>
                    <div class="step active">
                        <strong>🧠 Processing emotions...</strong>
                    </div>
                    <div class="step">
                        <strong>✨ Generating results...</strong>
                    </div>
                </div>
                <div style="color: var(--text-secondary); text-align: center; font-size: 0.9rem;">
                    Running AI models on your text...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Actually run the models
            results = predict_with_models(processed_text, selected_models, threshold, use_optimal)
            
            # Step 3: Finalizing Results
            loading_placeholder.markdown("""
            <div class="custom-loading">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎭</div>
                <div class="emotion-particles">
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                </div>
                <div class="progress-steps">
                    <div class="step completed">
                        <strong>🔍 Analyzing text structure...</strong>
                    </div>
                    <div class="step completed">
                        <strong>🧠 Processing emotions...</strong>
                    </div>
                    <div class="step active">
                        <strong>✨ Generating results...</strong>
                    </div>
                </div>
                <div style="color: var(--text-secondary); text-align: center; font-size: 0.9rem;">
                    Finalizing emotion analysis...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Brief pause to show final step
            time.sleep(1.0)
            
            # Clear the loading animation
            loading_placeholder.empty()
            
            if results:
                # Trigger confetti celebration
                st.markdown("""
                <script>
                    // Trigger confetti
                    setTimeout(() => {
                        const confettiContainer = document.createElement('div');
                        confettiContainer.id = 'confettiContainer';
                        confettiContainer.style.position = 'fixed';
                        confettiContainer.style.top = '0';
                        confettiContainer.style.left = '0';
                        confettiContainer.style.width = '100%';
                        confettiContainer.style.height = '100%';
                        confettiContainer.style.pointerEvents = 'none';
                        confettiContainer.style.zIndex = '9999';
                        
                        for (let i = 0; i < 30; i++) {
                            const confetti = document.createElement('div');
                            confetti.className = 'confetti';
                            confetti.style.left = Math.random() * 100 + '%';
                            confetti.style.animationDelay = Math.random() * 2 + 's';
                            const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#fab1a0'];
                            confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
                            confettiContainer.appendChild(confetti);
                        }
                        
                        document.body.appendChild(confettiContainer);
                        
                        setTimeout(() => {
                            if (document.body.contains(confettiContainer)) {
                                document.body.removeChild(confettiContainer);
                            }
                        }, 3000);
                    }, 500);
                </script>
                """, unsafe_allow_html=True)
                
                # Store in session state
                st.session_state.comparison_results = results
                st.session_state.prediction_history.append({
                    'text': text_input[:100] + '...' if len(text_input) > 100 else text_input,
                    'results': results,
                    'timestamp': time.time()
                })
                
                # Display translation info if available
                if st.session_state.get('last_translation'):
                    translation_info = st.session_state.last_translation
                    st.markdown("---")
                    st.markdown(f"### {t('language_detection')}")
                    
                    # Always show language detection
                    if translation_info.detected_language:
                        lang_name = {
                            'zh': 'Chinese', 'es': 'Spanish', 'fr': 'French', 
                            'de': 'German', 'ja': 'Japanese', 'ko': 'Korean',
                            'ru': 'Russian', 'pt': 'Portuguese', 'it': 'Italian',
                            'ar': 'Arabic', 'hi': 'Hindi', 'th': 'Thai',
                            'ms': 'Malay', 'id': 'Indonesian', 'vi': 'Vietnamese',
                            'tl': 'Filipino', 'my': 'Burmese', 'km': 'Khmer', 'en': 'English'
                        }.get(translation_info.detected_language, translation_info.detected_language.upper())
                        
                        col_lang, col_status = st.columns([1, 1])
                        with col_lang:
                            st.metric(t('detected_language'), lang_name)
                        with col_status:
                            if translation_info.was_translated:
                                st.metric(t('status'), t('translated'))
                            else:
                                st.metric(t('status'), t('no_translation'))
                    
                    # Show original vs translated text only if translated
                    if translation_info.was_translated:
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            with st.expander(t('original_vs_translated'), expanded=False):
                                st.markdown(f"**{t('original_text')}**")
                                st.info(translation_info.original_text)
                                st.markdown(f"**{t('translated_text')}**")
                                st.success(translation_info.translated_text)
                        
                        with col2:
                            # Translation quality and errors
                            st.metric(t('translation_quality'), f"{translation_info.confidence:.1%}")
                            
                            if translation_info.error_message:
                                st.error(f"⚠️ {translation_info.error_message}")
                    
                    # Show error message even if not translated
                    elif translation_info.error_message:
                        st.error(f"⚠️ {translation_info.error_message}")
                
                # Display results
                st.markdown("---")
                display_prediction_results(results, text_input)
                
                # Visualizations
                if len(results) > 0:
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 1.5rem 0; padding: 1rem; 
                               background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                               border-radius: 12px; border-left: 4px solid var(--accent-blue);">
                        <h3 style="margin: 0; color: #1e293b; font-weight: 600;">
                            📊 Model Comparison Charts
                        </h3>
                        <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.9rem;">
                            Compare emotion detection across different AI models
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Better column spacing
                    col1, col2 = st.columns([1, 1], gap="medium")
                    
                    with col1:
                        st.markdown("""
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <h4 style="margin: 0; color: #374151; font-weight: 500;">
                                📊 Emotion Radar
                            </h4>
                            <p style="margin: 0.25rem 0 0 0; color: #6b7280; font-size: 0.85rem;">
                                Multi-dimensional emotion comparison
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        radar_fig = create_radar_chart(results)
                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <h4 style="margin: 0; color: #374151; font-weight: 500;">
                                📈 Confidence Levels
                            </h4>
                            <p style="margin: 0.25rem 0 0 0; color: #6b7280; font-size: 0.85rem;">
                                Top emotions by confidence score
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        bar_fig = create_comparison_bar_chart(results)
                        if bar_fig:
                            st.plotly_chart(bar_fig, use_container_width=True)
                    
                    # Model agreement analysis
                    if len(results) > 1:
                        st.markdown("---")
                        st.subheader("🤝 Model Agreement")
                        create_model_agreement_analysis(results)
    
    # 📈 Recent Activity (Show recent predictions for convenience)
    if st.session_state.get('prediction_history', []):
        st.markdown("---")
        st.markdown("### 📈 Recent Activity")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{len(st.session_state.prediction_history)}** predictions made")
        with col2:
            if st.button("🗑️ Clear", type="secondary", key="clear_recent_activity"):
                st.session_state.prediction_history = []
                st.rerun()
        
        # Show last 3 predictions
        for i, item in enumerate(reversed(st.session_state.prediction_history[-3:])):
            with st.expander(f"🔸 {item['text'][:50]}{'...' if len(item['text']) > 50 else ''}", expanded=False):
                timestamp = datetime.fromtimestamp(item.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"📅 {timestamp}")
                
                for model_name, result in item['results'].items():
                    if result.predicted_emotions:
                        translated_emotions = [te(emotion) for emotion in result.predicted_emotions]
                        confidence = result.get_confidence()
                        st.markdown(f"**{model_name}** (Confidence: {confidence:.1%}): {', '.join(translated_emotions)}")
    
    
def render_batch_analysis_section():
    """Render the batch analysis section content"""
    st.markdown("### Upload multiple texts for batch processing")
    st.info("💡 Perfect for analyzing datasets, social media posts, or survey responses")
    
    # Usage Guide
    with st.expander("📋 File Format Guide", expanded=False):
        st.markdown("""
        **Simple CSV Format - Only `text` column required!**
        
        Your CSV file just needs one column named **`text`** containing the texts to analyze.
        
        **Simplest format (recommended):**
        ```
        text
        "I love this new product!"
        "Feeling stressed about the deadline"
        "Great job everyone, well done!"
        ```
        
        **Extended format (optional extra columns):**
        ```
        text,category,source
        "I love this new product!","review","website"
        "Feeling stressed about the deadline","personal","diary"
        "Great job everyone, well done!","work","email"
        ```
        
        **Requirements:**
        - ✅ File format: `.csv`
        - ✅ **Only required column: `text`** (case-sensitive)
        - ✅ Additional columns: Completely optional (ignored during analysis)
        - ✅ Text encoding: UTF-8 (recommended)
        - ✅ Max file size: 200MB
        - ✅ **Processes all texts in your file**
        
        **Tips for better results:**
        - Keep texts under 512 characters for optimal performance
        - Remove special characters that might cause encoding issues
        - One text per row, avoid line breaks within text cells
        """)
        
        # Sample CSV download
        sample_data = {
            'text': [
                "I'm so excited about this opportunity!",
                "This is really frustrating me.",
                "I feel nervous about the presentation tomorrow.",
                "What a wonderful surprise, thank you!",
                "I'm disappointed with the service quality."
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_sample,
            file_name="emotion_analysis_sample.csv",
            mime="text/csv",
            help="Download this sample file to see the expected format",
            key="sample_csv_download_batch_tab"
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Select a CSV file with a 'text' column containing texts to analyze",
        key="csv_uploader_batch_tab"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV file must contain a 'text' column")
            else:
                st.success(f"✅ Loaded {len(df)} texts")
                st.dataframe(df.head(3), use_container_width=True)
                
                if st.button("🚀 Process All Texts", type="primary", key="process_batch_tab"):
                    # Get model settings
                    selected_models = st.session_state.get('selected_models', [])
                    threshold = st.session_state.get('threshold', 0.5)
                    use_optimal = st.session_state.get('use_optimal_threshold', True)
                    
                    if not selected_models:
                        st.error("⚙️ Please configure models in the Model Settings tab first.")
                    else:
                        # Process all texts
                        texts_to_process = df['text'].tolist()
                        total_texts = len(texts_to_process)
                        
                        st.info(f"🔄 Processing {total_texts} texts with {len(selected_models)} model(s)...")
                        
                        # Create progress bar and results storage
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        batch_results = []
                        
                        # Process each text
                        for i, text in enumerate(texts_to_process):
                            # Update progress
                            progress = (i + 1) / total_texts
                            progress_bar.progress(progress)
                            status_text.text(f"Processing text {i + 1} of {total_texts}...")
                            
                            # Handle translation if enabled
                            processed_text = text.strip()
                            translation_result = None
                            if st.session_state.get('enable_translation', False):
                                translator = get_translator()
                                if translator.is_available():
                                    translation_result = translator.translate_to_english(processed_text)
                                    processed_text = translation_result.translated_text
                            
                            # Predict emotions
                            try:
                                results = predict_with_models(processed_text, selected_models, threshold, use_optimal)
                                batch_results.append({
                                    'original_text': text,
                                    'processed_text': processed_text,
                                    'translation_result': translation_result,
                                    'results': results,
                                    'timestamp': time.time()
                                })
                            except Exception as e:
                                st.error(f"Error processing text {i + 1}: {str(e)}")
                                batch_results.append({
                                    'original_text': text,
                                    'processed_text': processed_text,
                                    'translation_result': translation_result,
                                    'results': {},
                                    'error': str(e),
                                    'timestamp': time.time()
                                })
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Store results in session state for download
                        st.session_state.batch_results = batch_results
                        
                        # Display results summary
                        st.success(f"✅ Completed processing {total_texts} texts!")
                        
                        # Results display with comprehensive summary
                        display_batch_summary(batch_results, selected_models)
                        display_batch_results(batch_results, selected_models)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Add download results button if batch results exist
    if st.session_state.get('batch_results'):
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        # Create downloadable CSV
        results_data = []
        for item in st.session_state.batch_results:
            row = {
                'text': item['original_text'],
                'processed_text': item['processed_text']
            }
            
            # Add model predictions
            for model_name, result in item['results'].items():
                row[f'{model_name}_emotions'] = ', '.join(result.predicted_emotions)
                row[f'{model_name}_confidence'] = result.get_confidence()
            
            # Add translation info if available
            if item.get('translation_result'):
                row['original_language'] = item['translation_result'].detected_language
                row['translated'] = 'Yes'
            else:
                row['translated'] = 'No'
            
            # Add timestamp
            row['processed_at'] = datetime.fromtimestamp(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        csv_results = results_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Batch Results",
            data=csv_results,
            file_name=f"emotion_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the complete batch analysis results",
            key="download_batch_results"
        )


def display_batch_summary(batch_results, selected_models):
    """Display comprehensive summary dashboard for batch results"""
    if not batch_results:
        return
    
    # Filter successful results
    successful_results = [item for item in batch_results if item['results']]
    
    if not successful_results:
        return
    
    st.markdown("### 📊 Comprehensive Analysis Summary")
    st.markdown("*Quick insights for decision makers*")
    
    # Collect all emotions and statistics
    all_emotions = []
    emotion_counts = {}
    model_consistency = {}
    confidence_scores = []
    
    for item in successful_results:
        item_emotions = []
        for model_name, result in item['results'].items():
            if result.predicted_emotions:
                # Collect emotions
                for emotion in result.predicted_emotions:
                    all_emotions.append(emotion)
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    item_emotions.extend(result.predicted_emotions)
                
                # Collect confidence scores
                confidence_scores.append(result.get_confidence())
        
        # Track model consistency (how many models agree on emotions)
        unique_emotions = set(item_emotions)
        model_consistency[len(unique_emotions)] = model_consistency.get(len(unique_emotions), 0) + 1
    
    # === 1. Key Insights Cards ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1]) if emotion_counts else ("neutral", 0)
        st.metric(
            "🎭 Most Common Emotion", 
            te(most_common_emotion[0]).title(),
            f"{most_common_emotion[1]} occurrences"
        )
    
    with col2:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric(
            "🎯 Avg Confidence", 
            f"{avg_confidence:.1%}",
            "Model certainty"
        )
    
    with col3:
        unique_emotions_found = len(emotion_counts)
        st.metric(
            "🌈 Emotion Diversity",
            f"{unique_emotions_found}",
            "Different emotions detected"
        )
    
    with col4:
        total_emotion_instances = len(all_emotions)
        avg_emotions_per_text = total_emotion_instances / len(successful_results) if successful_results else 0
        st.metric(
            "📈 Complexity",
            f"{avg_emotions_per_text:.1f}",
            "Emotions per text"
        )
    
    # === 2. Emotion Distribution ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if emotion_counts:
            st.markdown("#### 🎭 Emotion Distribution")
            
            try:
                # Create bar chart
                sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
                emotions = [te(emotion).title() for emotion, _ in sorted_emotions[:10]]  # Top 10
                counts = [count for _, count in sorted_emotions[:10]]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=emotions,
                        y=counts,
                        marker_color='rgba(55, 128, 191, 0.7)',
                        text=counts,
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title="Top 10 Emotions Detected",
                    xaxis_title="Emotions",
                    yaxis_title="Frequency",
                    height=400,
                    showlegend=False,
                    xaxis=dict(tickangle=45)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                # Fallback to simple text display if plotly fails
                st.warning(f"Chart rendering failed: {str(e)}")
                st.markdown("**Top 10 Emotions (Text View):**")
                sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
                for i, (emotion, count) in enumerate(sorted_emotions[:10], 1):
                    st.write(f"{i}. **{te(emotion).title()}**: {count}")
    
    with col2:
        st.markdown("#### 📊 Top Emotions")
        
        # Top emotions list with percentages
        total_detections = sum(emotion_counts.values())
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            st.write(f"**{te(emotion).title()}:** {count} ({percentage:.1f}%)")
    
    # === 3. Model Consistency Analysis ===
    if len(selected_models) > 1:
        st.markdown("#### 🤝 Model Agreement Analysis")
        
        # Calculate agreement statistics
        agreements = []
        disagreements = []
        
        for item in successful_results:
            model_emotions = {}
            for model_name, result in item['results'].items():
                model_emotions[model_name] = set(result.predicted_emotions)
            
            if len(model_emotions) > 1:
                models = list(model_emotions.keys())
                common_emotions = set.intersection(*model_emotions.values()) if model_emotions else set()
                all_emotions_item = set.union(*model_emotions.values()) if model_emotions else set()
                
                if len(all_emotions_item) > 0:
                    agreement_score = len(common_emotions) / len(all_emotions_item)
                    agreements.append(agreement_score)
                    
                    if agreement_score < 0.5:  # Less than 50% agreement
                        disagreements.append(item['original_text'][:50] + "...")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_agreement = sum(agreements) / len(agreements) if agreements else 1.0
            st.metric("🎯 Model Agreement", f"{avg_agreement:.1%}")
        
        with col2:
            high_agreement = sum(1 for score in agreements if score > 0.7)
            st.metric("✅ High Agreement", f"{high_agreement}/{len(agreements)}")
        
        with col3:
            low_agreement = len(disagreements)
            st.metric("⚠️ Need Review", f"{low_agreement}")
        
        if disagreements and low_agreement > 0:
            with st.expander(f"⚠️ Texts with Low Model Agreement ({low_agreement} texts)"):
                for text in disagreements[:5]:  # Show first 5
                    st.write(f"• {text}")
    
    # === 4. Confidence Distribution ===
    if confidence_scores:
        st.markdown("#### 🎯 Confidence Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                # Confidence histogram
                fig = go.Figure(data=[
                    go.Histogram(
                        x=confidence_scores,
                        nbinsx=20,
                        marker_color='rgba(76, 175, 80, 0.7)'
                    )
                ])
                fig.update_layout(
                    title="Model Confidence Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Number of Texts",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                # Fallback to text summary if histogram fails
                st.warning(f"Chart rendering failed: {str(e)}")
                st.markdown("**Confidence Distribution (Text View):**")
                avg_conf = sum(confidence_scores) / len(confidence_scores)
                min_conf = min(confidence_scores)
                max_conf = max(confidence_scores)
                st.write(f"• Average: {avg_conf:.2%}")
                st.write(f"• Range: {min_conf:.2%} - {max_conf:.2%}")
                st.write(f"• Total samples: {len(confidence_scores)}")
        
        with col2:
            # Confidence categories
            high_conf = sum(1 for score in confidence_scores if score > 0.8)
            med_conf = sum(1 for score in confidence_scores if 0.5 <= score <= 0.8)
            low_conf = sum(1 for score in confidence_scores if score < 0.5)
            
            st.write("**Confidence Levels:**")
            st.write(f"🟢 High (>80%): {high_conf}")
            st.write(f"🟡 Medium (50-80%): {med_conf}")
            st.write(f"🔴 Low (<50%): {low_conf}")
            
            if low_conf > 0:
                st.warning(f"⚠️ {low_conf} texts have low confidence scores and may need manual review.")
    
    # === 5. Executive Summary ===
    st.markdown("#### 📋 Executive Summary")
    
    # Generate insights
    insights = []
    
    if emotion_counts:
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        dominant_pct = (dominant_emotion[1] / total_detections * 100) if total_detections > 0 else 0
        insights.append(f"🎭 **Primary Emotion:** {te(dominant_emotion[0]).title()} dominates with {dominant_pct:.1f}% of all detections")
    
    if confidence_scores:
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        if avg_conf > 0.8:
            insights.append(f"🎯 **High Reliability:** Average model confidence is {avg_conf:.1%} - results are highly trustworthy")
        elif avg_conf < 0.6:
            insights.append(f"⚠️ **Review Needed:** Average confidence is {avg_conf:.1%} - some results may need manual verification")
    
    if len(selected_models) > 1 and agreements:
        avg_agreement = sum(agreements) / len(agreements)
        if avg_agreement > 0.8:
            insights.append(f"🤝 **Strong Consensus:** Models agree {avg_agreement:.1%} of the time - consistent results")
        elif avg_agreement < 0.6:
            insights.append(f"🤔 **Mixed Signals:** Models agree only {avg_agreement:.1%} of the time - results may vary by approach")
    
    if unique_emotions_found >= 15:
        insights.append(f"🌈 **Rich Emotional Landscape:** {unique_emotions_found} different emotions detected - diverse content")
    elif unique_emotions_found <= 5:
        insights.append(f"🎯 **Focused Content:** Only {unique_emotions_found} emotions detected - consistent emotional tone")
    
    for insight in insights:
        st.write(insight)
    
    if not insights:
        st.info("📊 Processing completed successfully. Review the detailed statistics above for insights.")

def display_batch_results(batch_results, selected_models):
    """Display batch processing results"""
    if not batch_results:
        return
    
    st.markdown("### 📊 Batch Analysis Results")
    
    # Calculate summary statistics
    total_processed = len(batch_results)
    successful_predictions = sum(1 for item in batch_results if item['results'])
    errors = total_processed - successful_predictions
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Texts", total_processed)
    with col2:
        st.metric("Successfully Processed", successful_predictions)
    with col3:
        st.metric("Errors", errors)
    with col4:
        accuracy_rate = (successful_predictions / total_processed * 100) if total_processed > 0 else 0
        st.metric("Success Rate", f"{accuracy_rate:.1f}%")
    
    # Display sample results
    if successful_predictions > 0:
        st.markdown("#### 📋 Sample Results")
        
        # Show first few results
        display_count = min(5, successful_predictions)
        
        for i, item in enumerate([r for r in batch_results if r['results']][:display_count]):
            with st.expander(f"Text {i+1}: {item['original_text'][:100]}{'...' if len(item['original_text']) > 100 else ''}"): 
                
                # Translation info
                if item.get('translation_result'):
                    st.info(f"🌐 Translated from {item['translation_result'].detected_language}: {item['processed_text']}")
                
                # Show results for each model
                for model_name, result in item['results'].items():
                    if result.predicted_emotions:
                        emotions_display = ', '.join([te(emotion) for emotion in result.predicted_emotions])
                        confidence = result.get_confidence()
                        st.markdown(f"**{model_name}** (Confidence: {confidence:.1%}): {emotions_display}")
    
    # Show errors if any
    if errors > 0:
        st.markdown("#### ⚠️ Processing Errors")
        error_items = [r for r in batch_results if 'error' in r]
        
        for i, item in enumerate(error_items[:3]):  # Show first 3 errors
            with st.expander(f"Error {i+1}: {item['original_text'][:50]}{'...' if len(item['original_text']) > 50 else ''}"):
                st.error(f"Error: {item['error']}")
        
        if len(error_items) > 3:
            st.caption(f"... and {len(error_items) - 3} more errors")



def render_legacy_interface():
    """Render the legacy expander-based interface for backward compatibility"""
    st.warning("⚠️ Enhanced UI components not available. Using legacy interface.")
    
    # Settings panel
    settings_result = create_settings_panel()
    if settings_result is None:
        st.stop()
    
    selected_models, threshold, use_optimal = settings_result
    
    # Text analysis (existing implementation)
    render_text_analysis_section(selected_models, threshold, use_optimal)
    


if __name__ == "__main__":
    main()