"""
Emotion Hierarchy and Correlation Analysis for GoEmotions Dataset
Defines hierarchical relationships and correlation patterns between emotions
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from .utils import EMOTION_LABELS

class EmotionHierarchy:
    """
    Manages hierarchical emotion relationships and correlations
    Based on GoEmotions dataset analysis and psychological emotion models
    """
    
    def __init__(self):
        # Define hierarchical emotion groups
        self.emotion_hierarchy = {
            'negative': {
                'anger_group': ['anger', 'annoyance', 'disapproval', 'disgust'],
                'sadness_group': ['sadness', 'disappointment', 'grief', 'remorse'],
                'fear_group': ['fear', 'nervousness'],
                'shame_group': ['embarrassment']
            },
            'positive': {
                'joy_group': ['joy', 'amusement', 'excitement', 'pride'],
                'love_group': ['love', 'caring', 'gratitude'],
                'approval_group': ['approval', 'admiration', 'optimism'],
                'relief_group': ['relief']
            },
            'ambiguous': {
                'surprise_group': ['surprise'],
                'confusion_group': ['confusion', 'realization'],
                'curiosity_group': ['curiosity', 'desire'],
                'neutral_group': ['neutral']
            }
        }
        
        # Emotion correlation matrix based on common co-occurrence patterns
        self.emotion_correlations = {
            'joy': {'amusement': 0.8, 'excitement': 0.7, 'love': 0.6, 'pride': 0.5, 'optimism': 0.6},
            'anger': {'annoyance': 0.9, 'disapproval': 0.8, 'disgust': 0.7},
            'sadness': {'disappointment': 0.8, 'grief': 0.9, 'remorse': 0.6},
            'fear': {'nervousness': 0.8, 'surprise': 0.4},
            'love': {'caring': 0.8, 'gratitude': 0.7, 'admiration': 0.6},
            'surprise': {'confusion': 0.5, 'realization': 0.6, 'curiosity': 0.4},
            'excitement': {'joy': 0.7, 'amusement': 0.6, 'pride': 0.5},
            'disgust': {'anger': 0.7, 'disapproval': 0.6, 'annoyance': 0.5},
            'approval': {'admiration': 0.7, 'optimism': 0.6, 'gratitude': 0.5},
            'caring': {'love': 0.8, 'gratitude': 0.6, 'approval': 0.4},
            'curiosity': {'confusion': 0.5, 'surprise': 0.4, 'realization': 0.5},
            'amusement': {'joy': 0.8, 'excitement': 0.6},
            'annoyance': {'anger': 0.9, 'disapproval': 0.7, 'disgust': 0.5},
            'embarrassment': {'sadness': 0.4, 'fear': 0.3, 'nervousness': 0.5},
            'gratitude': {'love': 0.7, 'caring': 0.6, 'approval': 0.5, 'admiration': 0.4},
            'optimism': {'joy': 0.6, 'approval': 0.6, 'excitement': 0.4},
            'pride': {'joy': 0.5, 'excitement': 0.5, 'approval': 0.4},
            'disappointment': {'sadness': 0.8, 'disapproval': 0.4},
            'disapproval': {'anger': 0.8, 'annoyance': 0.7, 'disappointment': 0.4},
            'nervousness': {'fear': 0.8, 'embarrassment': 0.5},
            'realization': {'surprise': 0.6, 'confusion': 0.5, 'curiosity': 0.5},
            'admiration': {'love': 0.6, 'gratitude': 0.4, 'approval': 0.7},
            'desire': {'love': 0.3, 'curiosity': 0.4},
            'confusion': {'surprise': 0.5, 'curiosity': 0.5, 'realization': 0.5},
            'grief': {'sadness': 0.9, 'remorse': 0.5},
            'remorse': {'sadness': 0.6, 'grief': 0.5, 'embarrassment': 0.4},
            'relief': {'joy': 0.4, 'gratitude': 0.3},
            'neutral': {}  # Neutral has weak correlations with most emotions
        }
        
        # Conflicting emotions (mutually exclusive or very unlikely to co-occur)
        self.emotion_conflicts = {
            'joy': ['sadness', 'anger', 'fear', 'disgust', 'disappointment'],
            'sadness': ['joy', 'amusement', 'excitement', 'pride'],
            'anger': ['joy', 'love', 'gratitude', 'caring'],
            'fear': ['joy', 'excitement', 'pride', 'optimism'],
            'love': ['anger', 'disgust', 'fear'],
            'disgust': ['love', 'caring', 'gratitude', 'admiration'],
            'excitement': ['sadness', 'fear', 'disappointment', 'grief'],
            'pride': ['sadness', 'embarrassment', 'remorse', 'fear']
        }
        
        # Build reverse lookup maps
        self._build_emotion_to_group_map()
        self._build_group_to_category_map()
    
    def _build_emotion_to_group_map(self):
        """Build mapping from emotion to its group"""
        self.emotion_to_group = {}
        self.emotion_to_category = {}
        
        for category, groups in self.emotion_hierarchy.items():
            for group_name, emotions in groups.items():
                for emotion in emotions:
                    self.emotion_to_group[emotion] = group_name
                    self.emotion_to_category[emotion] = category
    
    def _build_group_to_category_map(self):
        """Build mapping from group to category"""
        self.group_to_category = {}
        for category, groups in self.emotion_hierarchy.items():
            for group_name in groups.keys():
                self.group_to_category[group_name] = category
    
    def get_emotion_category(self, emotion: str) -> Optional[str]:
        """Get the top-level category (positive/negative/ambiguous) for an emotion"""
        return self.emotion_to_category.get(emotion)
    
    def get_emotion_group(self, emotion: str) -> Optional[str]:
        """Get the specific group for an emotion"""
        return self.emotion_to_group.get(emotion)
    
    def get_related_emotions(self, emotion: str, threshold: float = 0.5) -> List[str]:
        """Get emotions that are positively correlated with the given emotion"""
        if emotion not in self.emotion_correlations:
            return []
        
        related = []
        for related_emotion, correlation in self.emotion_correlations[emotion].items():
            if correlation >= threshold:
                related.append(related_emotion)
        
        return related
    
    def get_conflicting_emotions(self, emotion: str) -> List[str]:
        """Get emotions that conflict with the given emotion"""
        return self.emotion_conflicts.get(emotion, [])
    
    def apply_correlation_boost(self, emotion_scores: Dict[str, float], 
                               boost_factor: float = 0.1) -> Dict[str, float]:
        """
        Apply correlation-based boosting to emotion scores
        If one emotion has a high score, boost its correlated emotions
        """
        boosted_scores = emotion_scores.copy()
        
        for emotion, score in emotion_scores.items():
            if score > 0.6:  # Only boost for relatively confident predictions
                related_emotions = self.get_related_emotions(emotion, threshold=0.5)
                
                for related_emotion in related_emotions:
                    if related_emotion in boosted_scores:
                        correlation_strength = self.emotion_correlations[emotion].get(related_emotion, 0)
                        boost_amount = score * correlation_strength * boost_factor
                        boosted_scores[related_emotion] += boost_amount
                        # Cap at 1.0
                        boosted_scores[related_emotion] = min(boosted_scores[related_emotion], 1.0)
        
        return boosted_scores
    
    def apply_conflict_suppression(self, emotion_scores: Dict[str, float], 
                                  suppression_factor: float = 0.2) -> Dict[str, float]:
        """
        Apply conflict-based suppression to emotion scores
        If one emotion has a high score, suppress its conflicting emotions
        """
        suppressed_scores = emotion_scores.copy()
        
        for emotion, score in emotion_scores.items():
            if score > 0.7:  # Only suppress for very confident predictions
                conflicting_emotions = self.get_conflicting_emotions(emotion)
                
                for conflicting_emotion in conflicting_emotions:
                    if conflicting_emotion in suppressed_scores:
                        suppression_amount = score * suppression_factor
                        suppressed_scores[conflicting_emotion] *= (1 - suppression_amount)
                        # Floor at 0.0
                        suppressed_scores[conflicting_emotion] = max(suppressed_scores[conflicting_emotion], 0.0)
        
        return suppressed_scores
    
    def get_category_distribution(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate the distribution of scores across emotion categories
        Useful for high-level emotion analysis
        """
        category_scores = {'positive': 0.0, 'negative': 0.0, 'ambiguous': 0.0}
        category_counts = {'positive': 0, 'negative': 0, 'ambiguous': 0}
        
        for emotion, score in emotion_scores.items():
            category = self.get_emotion_category(emotion)
            if category:
                category_scores[category] += score
                category_counts[category] += 1
        
        # Average scores per category
        for category in category_scores:
            if category_counts[category] > 0:
                category_scores[category] /= category_counts[category]
        
        return category_scores
    
    def suggest_threshold_adjustments(self, emotion_scores: Dict[str, float], 
                                    base_threshold: float = 0.5) -> Dict[str, float]:
        """
        Suggest threshold adjustments based on emotion relationships
        Some emotions naturally have lower/higher expression thresholds
        """
        threshold_adjustments = {}
        
        # Emotions that tend to be expressed more subtly (lower threshold)
        subtle_emotions = ['realization', 'curiosity', 'confusion', 'nervousness', 'caring']
        
        # Emotions that require stronger signal (higher threshold)
        strong_emotions = ['anger', 'love', 'excitement', 'grief', 'disgust']
        
        for emotion in EMOTION_LABELS:
            if emotion in subtle_emotions:
                threshold_adjustments[emotion] = base_threshold * 0.8
            elif emotion in strong_emotions:
                threshold_adjustments[emotion] = base_threshold * 1.2
            else:
                threshold_adjustments[emotion] = base_threshold
        
        return threshold_adjustments
    
    def analyze_emotion_coherence(self, predicted_emotions: List[str]) -> Dict[str, float]:
        """
        Analyze the coherence of predicted emotions
        Returns metrics about how well the predictions align with emotion theory
        """
        if not predicted_emotions:
            return {'coherence_score': 1.0, 'conflict_count': 0, 'category_diversity': 0}
        
        # Check for conflicts
        conflict_count = 0
        total_pairs = 0
        
        for i, emotion1 in enumerate(predicted_emotions):
            for emotion2 in predicted_emotions[i+1:]:
                total_pairs += 1
                if emotion2 in self.get_conflicting_emotions(emotion1):
                    conflict_count += 1
        
        # Calculate category diversity
        categories = set()
        for emotion in predicted_emotions:
            category = self.get_emotion_category(emotion)
            if category:
                categories.add(category)
        
        category_diversity = len(categories)
        
        # Coherence score (0-1, higher is more coherent)
        if total_pairs > 0:
            coherence_score = 1.0 - (conflict_count / total_pairs)
        else:
            coherence_score = 1.0
        
        return {
            'coherence_score': coherence_score,
            'conflict_count': conflict_count,
            'category_diversity': category_diversity,
            'dominant_category': max(categories) if categories else None
        }

# Global instance for easy access
emotion_hierarchy = EmotionHierarchy()