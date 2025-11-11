"""
Tab-based Navigation System for GoEmotions App

This module provides a modern tab navigation interface to replace the
previous expander-based navigation, improving user experience and
functionality discoverability.
"""

import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
from utils.languages import get_text


def t(key: str, **kwargs) -> str:
    """Translation helper function"""
    current_lang = st.session_state.get('ui_language', 'en')
    return get_text(key, current_lang, **kwargs)


def create_tab_navigation() -> Tuple[str, Dict[str, Any]]:
    """
    Create the main tab navigation interface
    
    Returns:
        Tuple containing (active_tab_name, tab_config)
    """
    
    # Define tab configuration
    tab_config = {
        'quick_analysis': {
            'label': f"🔍 {t('quick_analysis', default='Quick Analysis')}",
            'description': t('quick_analysis_desc', default='Analyze single texts with AI models'),
            'icon': '🔍'
        },
        'model_settings': {
            'label': f"⚙️ {t('model_settings', default='Model Settings')}",  
            'description': t('model_settings_desc', default='Configure AI models and parameters'),
            'icon': '⚙️'
        },
        'batch_analysis': {
            'label': f"📊 {t('batch_analysis', default='Batch Analysis')}",
            'description': t('batch_analysis_desc', default='Process multiple texts from CSV files'),
            'icon': '📊'
        }
    }
    
    # Create tabs
    tab_labels = [config['label'] for config in tab_config.values()]
    tabs = st.tabs(tab_labels)
    
    # Store active tab in session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'quick_analysis'
    
    return tab_config, tabs


def display_tab_content(tab_name: str, tab_container, content_func):
    """
    Display content for a specific tab with consistent styling
    
    Args:
        tab_name: Name of the tab
        tab_container: Streamlit tab container 
        content_func: Function to render tab content
    """
    with tab_container:
        # Add tab-specific styling
        st.markdown(f"""
        <div class="tab-content tab-{tab_name}">
        """, unsafe_allow_html=True)
        
        # Execute content function
        content_func()
        
        st.markdown("</div>", unsafe_allow_html=True)


def create_quick_analysis_tab():
    """Content for the Quick Analysis tab"""
    
    # Tab header with description
    st.markdown(f"""
    <div class="tab-header">
        <h2>🔍 {t('quick_analysis', default='Quick Analysis')}</h2>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            {t('quick_analysis_desc', default='Analyze emotions in text using state-of-the-art AI models')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # This content will be populated by the main app
    st.info(t('quick_analysis_content_placeholder', 
             default='Quick analysis content will be rendered here'))


def create_model_settings_tab():
    """Content for the Model Settings tab"""
    
    st.markdown(f"""
    <div class="tab-header">
        <h2>⚙️ {t('model_settings', default='Model Settings')}</h2>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            {t('model_settings_desc', default='Configure AI models, thresholds, and translation settings')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection and configuration will be handled by main app
    st.info(t('model_settings_content_placeholder',
             default='Model settings content will be rendered here'))


def create_batch_analysis_tab():
    """Content for the Batch Analysis tab"""
    
    st.markdown(f"""
    <div class="tab-header">
        <h2>📊 {t('batch_analysis', default='Batch Analysis')}</h2>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            {t('batch_analysis_desc', default='Upload and process multiple texts from CSV files for comprehensive analysis')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Batch analysis content will be handled by main app
    st.info(t('batch_analysis_content_placeholder',
             default='Batch analysis content will be rendered here'))




def get_tab_styles() -> str:
    """
    Get additional CSS styles specific to tab navigation
    """
    return """
    <style>
    /* Tab Content Styling */
    .tab-content {
        padding: 1rem 0;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .tab-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-radius: 12px;
        border: 1px solid var(--border-light);
    }
    
    .tab-header h2 {
        margin: 0 0 0.5rem 0;
        color: var(--text-primary);
        font-weight: 700;
    }
    
    /* Tab-specific styling */
    .tab-quick_analysis {
        /* Quick analysis specific styles */
    }
    
    .tab-model_settings {
        /* Model settings specific styles */
    }
    
    .tab-batch_analysis {
        /* Batch analysis specific styles */  
    }
    
    
    /* Enhanced tab hover effects */
    .stTabs [data-baseweb="tab"]:hover::before {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 2px;
        background: var(--accent-blue);
        transition: width 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        width: 80%;
    }
    
    /* Active tab indicator */
    .stTabs [data-baseweb="tab"][aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        border-radius: 2px;
        box-shadow: 0 2px 4px rgba(29, 78, 216, 0.3);
    }
    
    /* Mobile responsive tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            min-width: auto;
            flex: 1;
            text-align: center;
            padding: 8px 12px;
            font-size: 0.85rem;
        }
        
        .tab-header {
            padding: 0.75rem;
            margin-bottom: 1.5rem;
        }
        
        .tab-header h2 {
            font-size: 1.5rem;
        }
    }
    
    /* Smooth content transitions */
    @keyframes fadeIn {
        0% {
            opacity: 0;
            transform: translateY(10px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading state for tab content */
    .tab-loading {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
        color: var(--text-muted);
    }
    
    .tab-loading::before {
        content: '⏳';
        font-size: 2rem;
        margin-right: 1rem;
        animation: spin 2s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """


def apply_tab_styles():
    """Apply tab-specific CSS styles"""
    st.markdown(get_tab_styles(), unsafe_allow_html=True)


def get_breadcrumb_navigation(active_tab: str) -> str:
    """
    Create breadcrumb navigation for better UX
    """
    tab_names = {
        'quick_analysis': t('quick_analysis', default='Quick Analysis'),
        'model_settings': t('model_settings', default='Model Settings'),
        'batch_analysis': t('batch_analysis', default='Batch Analysis')
    }
    
    current_tab_name = tab_names.get(active_tab, 'Unknown')
    
    return f"""
    <div class="breadcrumb-nav" style="
        margin: 0 0 1rem 0;
        padding: 0.5rem 1rem;
        background: var(--bg-secondary);
        border-radius: 8px;
        font-size: 0.9rem;
        color: var(--text-secondary);
        border-left: 4px solid var(--accent-blue);
    ">
        <span style="color: var(--text-muted);">📍</span>
        <span style="margin: 0 0.5rem;">GoEmotions</span>
        <span style="color: var(--text-muted);">›</span>
        <span style="margin: 0 0.5rem; color: var(--accent-blue); font-weight: 600;">
            {current_tab_name}
        </span>
    </div>
    """


# Helper functions for tab state management
def get_active_tab() -> str:
    """Get the currently active tab"""
    return st.session_state.get('active_tab', 'quick_analysis')


def set_active_tab(tab_name: str):
    """Set the active tab"""
    st.session_state.active_tab = tab_name


def is_tab_visited(tab_name: str) -> bool:
    """Check if a tab has been visited"""
    visited_tabs = st.session_state.get('visited_tabs', set())
    return tab_name in visited_tabs


def mark_tab_visited(tab_name: str):
    """Mark a tab as visited"""
    if 'visited_tabs' not in st.session_state:
        st.session_state.visited_tabs = set()
    st.session_state.visited_tabs.add(tab_name)