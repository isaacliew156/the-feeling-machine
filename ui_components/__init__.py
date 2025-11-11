"""
UI Components Package for GoEmotions NLP Project

This package contains modular UI components for the Streamlit application,
providing better code organization, maintainability, and reusability.

Components:
- styles: Centralized CSS and styling management
- navigation: Tab-based navigation system
- model_settings: Model configuration and settings panel
- analysis_tab: Main text analysis interface
- batch_tab: Batch processing functionality
- history_tab: Prediction history display
- results_display: Results visualization components
"""

import logging

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "GoEmotions Project"

# Import main components for easy access
try:
    from .styles import get_global_styles, get_emotion_colors, apply_custom_css
    from .navigation import create_tab_navigation
    from .results_display import display_prediction_results
    
    # Create placeholder for create_comparison_chart if not available
    try:
        from .results_display import create_enhanced_comparison_chart as create_comparison_chart
    except ImportError:
        create_comparison_chart = None
        
    __all__ = [
        'get_global_styles',
        'get_emotion_colors', 
        'apply_custom_css',
        'create_tab_navigation',
        'display_prediction_results',
        'create_comparison_chart'
    ]
except ImportError as e:
    # Fallback if components can't be imported
    logger.warning(f"UI components could not be imported: {e}")
    __all__ = []