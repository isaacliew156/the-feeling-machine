"""
Results Display Components for GoEmotions App

This module provides enhanced result visualization components with
improved accessibility, better color schemes, and interactive features.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
import time
from datetime import datetime

# Import from local modules
from model_loaders.utils import PredictionResult, format_prediction_time, calculate_model_agreement
from utils.languages import get_text, get_emotion_label
from .styles import get_emotion_color_mapping, ColorScheme


def t(key: str, **kwargs) -> str:
    """Translation helper function"""
    current_lang = st.session_state.get('ui_language', 'en')
    return get_text(key, current_lang, **kwargs)


def te(emotion: str) -> str:
    """Emotion translation helper function"""
    current_lang = st.session_state.get('ui_language', 'en')
    return get_emotion_label(emotion, current_lang)


def display_prediction_results(results: Dict[str, PredictionResult], text: str, 
                             use_enhanced_display: bool = True):
    """
    Display prediction results with enhanced styling and accessibility
    
    Args:
        results: Dictionary of model name -> PredictionResult
        text: Original input text
        use_enhanced_display: Whether to use the new enhanced display format
    """
    if not results:
        st.warning(t('no_results', default="No prediction results to display"))
        return

    if use_enhanced_display:
        _display_enhanced_results(results, text)
    else:
        _display_legacy_results(results, text)


def _display_enhanced_results(results: Dict[str, PredictionResult], text: str):
    """Enhanced results display with improved UX and accessibility"""
    
    # Results header with input text preview
    st.markdown(f"""
    <div class="results-header" style="
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        box-shadow: var(--shadow-sm);
    ">
        <h2 style="
            color: var(--text-primary);
            margin: 0 0 1rem 0;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            ✨ {t('results_title', default='Analysis Results')}
        </h2>
        <div style="
            background: var(--bg-primary);
            border: 1px solid var(--border-light);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        ">
            <strong style="color: var(--text-primary);">{t('analyzed_text', default='Analyzed Text')}:</strong>
            <div style="
                margin-top: 0.5rem;
                color: var(--text-secondary);
                font-style: italic;
                line-height: 1.6;
                max-height: 4rem;
                overflow: hidden;
                text-overflow: ellipsis;
            ">
                "{text[:200]}{'...' if len(text) > 200 else ''}"
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model results in enhanced cards
    if len(results) == 1:
        # Single model - full width display
        _display_single_model_result(results)
    else:
        # Multiple models - responsive grid
        _display_multiple_model_results(results)
    
    # Model agreement analysis for multiple models
    if len(results) > 1:
        st.markdown("---")
        _display_model_agreement_analysis(results)
    
    # Interactive comparison chart
    st.markdown("---")
    comparison_chart = create_enhanced_comparison_chart(results)
    if comparison_chart:
        st.plotly_chart(comparison_chart, use_container_width=True, config={'displayModeBar': False})


def _display_single_model_result(results: Dict[str, PredictionResult]):
    """Display single model result with full-width layout"""
    model_name, result = next(iter(results.items()))
    
    # Main result card
    st.markdown(f"""
    <div class="single-model-result" style="
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
    ">
        <h3 style="
            color: var(--text-primary);
            margin: 0 0 1.5rem 0;
            font-weight: 700;
            font-size: 1.5rem;
            text-align: center;
        ">
            🤖 {model_name}
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = result.get_confidence()
        st.metric(
            label="🎯 " + t('confidence', default='Confidence'),
            value=f"{confidence:.1%}",
            help=t('confidence_help', default='Highest emotion score from the model')
        )
    
    with col2:
        if result.prediction_time:
            st.metric(
                label="⚡ " + t('prediction_time', default='Processing Time'),
                value=format_prediction_time(result.prediction_time),
                help=t('prediction_time_help', default='Time taken to analyze the text')
            )
    
    with col3:
        predicted_count = len(result.predicted_emotions)
        st.metric(
            label="🎭 " + t('emotions_detected', default='Emotions Detected'),
            value=str(predicted_count),
            help=t('emotions_detected_help', default='Number of emotions above threshold')
        )
    
    # Emotion display
    _display_emotion_results(result, model_name)


def _display_multiple_model_results(results: Dict[str, PredictionResult]):
    """Display multiple model results in responsive grid"""
    
    # Determine layout based on number of models
    num_models = len(results)
    if num_models == 2:
        cols = st.columns(2)
    elif num_models == 3:
        cols = st.columns(3)
    else:
        cols = st.columns(min(4, num_models))
    
    for i, (model_name, result) in enumerate(results.items()):
        with cols[i % len(cols)]:
            _display_model_card(model_name, result)


def _display_model_card(model_name: str, result: PredictionResult):
    """Display individual model result card"""
    confidence = result.get_confidence()
    confidence_color = _get_confidence_color(confidence)
    
    st.markdown(f"""
    <div class="model-card" style="
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, {confidence_color}, var(--accent-purple));
        "></div>
        
        <h4 style="
            color: var(--text-primary);
            margin: 0 0 1rem 0;
            font-weight: 600;
            font-size: 1.1rem;
            text-align: center;
        ">
            {model_name}
        </h4>
        
        <div style="
            text-align: center;
            margin: 1rem 0;
            padding: 1rem;
            background: rgba(29, 78, 216, 0.05);
            border-radius: 8px;
            border: 1px solid rgba(29, 78, 216, 0.1);
        ">
            <div style="
                font-size: 0.85rem;
                color: var(--text-secondary);
                margin-bottom: 0.5rem;
            ">
                {t('max_confidence', default='Max Confidence')}
            </div>
            <div style="
                font-size: 2rem;
                font-weight: bold;
                color: {confidence_color};
                text-shadow: 0 1px 3px rgba(29, 78, 216, 0.3);
            ">
                {confidence:.0%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing time
    if result.prediction_time:
        st.caption(f"⚡ {format_prediction_time(result.prediction_time)}")
    
    # Emotion results
    _display_emotion_results(result, model_name, compact=True)


def _display_emotion_results(result: PredictionResult, model_name: str, compact: bool = False):
    """Display emotion results with enhanced styling"""
    
    top_emotions = result.get_top_emotions(5 if not compact else 3)
    emotion_colors = get_emotion_color_mapping()
    
    if not compact:
        st.markdown(f"**{t('top_emotions', default='Top Emotions')}:**")
    
    # Display emotion tags
    emotions_html = []
    for emotion, score in top_emotions:
        if score > 0.01:  # Only show emotions with meaningful scores
            # Get emotion-specific colors
            emotion_lower = emotion.lower()
            if emotion_lower in emotion_colors:
                colors = emotion_colors[emotion_lower]
                bg_color = colors['bg']
                text_color = colors['text']
                border_color = colors['border']
            else:
                # Fallback to score-based colors
                if score > 0.7:
                    bg_color = ColorScheme.EMOTION_HIGH_BG
                    text_color = ColorScheme.EMOTION_HIGH_TEXT
                    border_color = ColorScheme.EMOTION_HIGH_BORDER
                elif score > 0.4:
                    bg_color = ColorScheme.EMOTION_MEDIUM_BG
                    text_color = ColorScheme.EMOTION_MEDIUM_TEXT
                    border_color = ColorScheme.EMOTION_MEDIUM_BORDER
                else:
                    bg_color = ColorScheme.EMOTION_LOW_BG
                    text_color = ColorScheme.EMOTION_LOW_TEXT
                    border_color = ColorScheme.EMOTION_LOW_BORDER
            
            emotions_html.append(f"""
            <div class="emotion-tag" style="
                background: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                padding: 6px 12px;
                border-radius: 20px;
                margin: 3px;
                display: inline-block;
                font-weight: 600;
                font-size: {'0.8rem' if compact else '0.9rem'};
                transition: all 0.3s ease;
                cursor: default;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0, 0, 0, 0.15)';"
               onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 1px 3px rgba(0, 0, 0, 0.1)';">
                {te(emotion)}: {score:.1%}
            </div>
            """)
    
    st.markdown(f"""
    <div class="emotions-container" style="
        margin: 1rem 0;
        line-height: 2.5;
    ">
        {''.join(emotions_html)}
    </div>
    """, unsafe_allow_html=True)
    
    # Predicted emotions summary
    if result.predicted_emotions:
        predicted_translated = [te(emotion) for emotion in result.predicted_emotions]
        if not compact:
            st.markdown(f"**{t('predicted', default='Predicted')}:** {', '.join(predicted_translated)}")
        else:
            st.caption(f"✨ {', '.join(predicted_translated[:2])}{'...' if len(predicted_translated) > 2 else ''}")
    else:
        if not compact:
            st.markdown(f"**{t('no_results', default='No emotions detected above threshold')}**")


def _display_model_agreement_analysis(results: Dict[str, PredictionResult]):
    """Display model agreement analysis with enhanced visuals"""
    
    st.markdown(f"""
    <h3 style="
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        font-weight: 700;
        text-align: center;
    ">
        🤝 {t('model_agreement', default='Model Agreement Analysis')}
    </h3>
    """, unsafe_allow_html=True)
    
    agreement = calculate_model_agreement(list(results.values()))
    
    # Agreement metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        agreement_score = agreement['agreement_percentage']
        agreement_color = _get_agreement_color(agreement_score)
        st.markdown(f"""
        <div style="
            text-align: center;
            padding: 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
        ">
            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">
                {t('agreement_score', default='Agreement Score')}
            </div>
            <div style="
                font-size: 2rem;
                font-weight: bold;
                color: {agreement_color};
            ">
                {agreement_score:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            label="🎯 " + t('common_emotions', default='Common Emotions'),
            value=len(agreement['common_emotions']),
            help=t('common_emotions_help', default='Emotions detected by all models')
        )
    
    with col3:
        st.metric(
            label="🌟 " + t('total_unique', default='Total Unique'),
            value=agreement['total_unique_emotions'],
            help=t('total_unique_help', default='Total unique emotions detected across all models')
        )
    
    # Common emotions display
    if agreement['common_emotions']:
        st.markdown(f"**{t('emotions_all_models_agree', default='Emotions all models agree on')}:**")
        common_emotions_html = []
        for emotion in agreement['common_emotions']:
            common_emotions_html.append(f"""
            <div class="common-emotion" style="
                background: linear-gradient(135deg, var(--accent-green), #059669);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                margin: 4px;
                display: inline-block;
                font-weight: 600;
                font-size: 0.9rem;
                box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3);
            ">
                ✨ {te(emotion)}
            </div>
            """)
        
        st.markdown(f"""
        <div class="common-emotions-container" style="
            margin: 1rem 0;
            padding: 1rem;
            background: rgba(5, 150, 105, 0.05);
            border-radius: 12px;
            border: 1px solid rgba(5, 150, 105, 0.2);
            line-height: 2.5;
        ">
            {''.join(common_emotions_html)}
        </div>
        """, unsafe_allow_html=True)


def create_enhanced_comparison_chart(results: Dict[str, PredictionResult]) -> go.Figure:
    """
    Create an enhanced comparison chart with better accessibility and interactivity
    """
    if not results:
        return None
    
    # Collect top emotions from all models
    all_emotions = set()
    model_scores = {}
    
    for model_name, result in results.items():
        top_emotions = result.get_top_emotions(8)  # Get more emotions for comparison
        model_scores[model_name] = dict(top_emotions)
        all_emotions.update([emotion for emotion, score in top_emotions if score > 0.05])
    
    if not all_emotions:
        return None
    
    # Sort emotions by maximum score across models
    emotion_max_scores = {}
    for emotion in all_emotions:
        max_score = max([model_scores[model].get(emotion, 0) for model in model_scores])
        emotion_max_scores[emotion] = max_score
    
    sorted_emotions = sorted(emotion_max_scores.keys(), 
                           key=lambda x: emotion_max_scores[x], 
                           reverse=True)[:10]  # Top 10 emotions
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[t('emotion_comparison', default='Emotion Scores by Model')]
    )
    
    # Color palette for models
    model_colors = [
        ColorScheme.ACCENT_BLUE,
        ColorScheme.ACCENT_GREEN,
        ColorScheme.ACCENT_PURPLE,
        ColorScheme.ACCENT_ORANGE,
        '#e11d48',  # Red
        '#7c3aed',  # Violet
        '#0891b2',  # Cyan
        '#ca8a04'   # Yellow
    ]
    
    # Add bars for each model
    for i, (model_name, scores) in enumerate(model_scores.items()):
        emotion_scores = [scores.get(emotion, 0) for emotion in sorted_emotions]
        translated_emotions = [te(emotion) for emotion in sorted_emotions]
        
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=translated_emotions,
                y=emotion_scores,
                marker=dict(
                    color=model_colors[i % len(model_colors)],
                    opacity=0.8,
                    line=dict(color='rgba(0,0,0,0.1)', width=1)
                ),
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Emotion: %{x}<br>"
                    "Score: %{y:.1%}<br>"
                    "<extra></extra>"
                ),
                texttemplate='%{y:.1%}',
                textposition='outside',
                textfont=dict(size=10, color=ColorScheme.TEXT_PRIMARY)
            )
        )
    
    # Update layout with improved accessibility
    fig.update_layout(
        title=dict(
            text=t('model_emotion_comparison', default='Model Emotion Comparison'),
            x=0.5,
            font=dict(size=18, color=ColorScheme.TEXT_PRIMARY, family="Inter", weight="bold")
        ),
        xaxis=dict(
            title=t('emotions', default='Emotions'),
            tickfont=dict(color=ColorScheme.TEXT_SECONDARY, family="Inter", size=11),
            title_font=dict(color=ColorScheme.TEXT_PRIMARY, family="Inter", size=13),
            gridcolor='#f3f4f6',
            showline=True,
            linecolor='#e5e7eb',
            tickangle=45
        ),
        yaxis=dict(
            title=t('confidence_score', default='Confidence Score'),
            tickfont=dict(color=ColorScheme.TEXT_SECONDARY, family="Inter", size=11),
            title_font=dict(color=ColorScheme.TEXT_PRIMARY, family="Inter", size=13),
            gridcolor='#f3f4f6',
            showline=True,
            linecolor='#e5e7eb',
            tickformat='.0%'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=ColorScheme.TEXT_PRIMARY, family="Inter"),
        margin=dict(t=60, b=80, l=60, r=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(color=ColorScheme.TEXT_PRIMARY)
        ),
        height=500,
        hovermode='x unified'
    )
    
    return fig


def _display_legacy_results(results: Dict[str, PredictionResult], text: str):
    """
    Legacy results display for backward compatibility
    """
    st.subheader(t('results_title', default='Results'))
    st.markdown(f"**{t('input_text', default='Input text')}:** *{text[:200]}{'...' if len(text) > 200 else ''}*")
    
    # Simple columns layout
    cols = st.columns(len(results))
    
    for i, (model_name, result) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"### {model_name}")
            
            confidence = result.get_confidence()
            confidence_percent = int(confidence * 100)
            
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0; padding: 0.5rem; 
                       background: rgba(37, 99, 235, 0.05); 
                       border-radius: 12px; border: 1px solid rgba(37, 99, 235, 0.1);">
                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                    {t('max_confidence', default='Max Confidence')}
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: var(--accent-blue); 
                           text-shadow: 0 1px 3px rgba(37, 99, 235, 0.3);">
                    {confidence_percent}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if result.prediction_time:
                st.metric(t('prediction_time', default='Prediction time'), 
                         format_prediction_time(result.prediction_time))
            
            # Simple emotion display
            top_emotions = result.get_top_emotions(5)
            st.markdown(f"**{t('top_emotions', default='Top emotions')}:**")
            
            for j, (emotion, score) in enumerate(top_emotions):
                if score > 0.7:
                    css_class = "emotion-high"
                elif score > 0.4:
                    css_class = "emotion-medium"
                else:
                    css_class = "emotion-low"
                
                st.markdown(f"""
                <div class="{css_class}" style="
                    margin: 4px 2px; 
                    animation: slideInLeft 0.6s ease forwards {j * 0.1}s;
                ">
                    {te(emotion)}: {score:.1%}
                </div>
                """, unsafe_allow_html=True)
            
            if result.predicted_emotions:
                translated_emotions = [te(emotion) for emotion in result.predicted_emotions]
                st.markdown(f"**{t('predicted', default='Predicted')}:** {', '.join(translated_emotions)}")
            else:
                st.markdown(f"**{t('no_results', default='No emotions above threshold')}**")


# Helper functions
def _get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return ColorScheme.PRIMARY_SUCCESS
    elif confidence >= 0.6:
        return ColorScheme.PRIMARY_WARNING
    else:
        return ColorScheme.PRIMARY_ERROR


def _get_agreement_color(agreement_score: float) -> str:
    """Get color based on agreement score"""
    if agreement_score >= 80:
        return ColorScheme.PRIMARY_SUCCESS
    elif agreement_score >= 60:
        return ColorScheme.PRIMARY_WARNING
    else:
        return ColorScheme.PRIMARY_ERROR