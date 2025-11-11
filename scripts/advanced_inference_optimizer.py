"""
Advanced Inference Optimizer for CNN+GloVe Model
Implements isotonic regression, pattern rules, dynamic thresholding, and UI rescaling
"""

import re
import numpy as np
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Optional sklearn import
try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ sklearn not available - using fallback calibration methods")
    SKLEARN_AVAILABLE = False

class AdvancedInferenceOptimizer:
    """
    Comprehensive inference optimization for emotion classification
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Core components
        self.isotonic_calibrators = {}
        self.pattern_rules = self._init_pattern_rules()
        self.emotion_priors = self._compute_emotion_priors()
        
        # Model parameters
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral'
        ]
        
        # Emotion categories
        self.negative_emotions = {
            'anger', 'disgust', 'fear', 'sadness', 'disappointment', 'annoyance', 
            'disapproval', 'grief', 'remorse', 'nervousness', 'embarrassment'
        }
        self.positive_emotions = {
            'joy', 'love', 'admiration', 'gratitude', 'excitement', 'pride', 
            'amusement', 'approval', 'caring', 'optimism', 'relief'
        }
        
        print("✅ Advanced Inference Optimizer initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return config.get('INFERENCE_OPTIMIZATION', {})
        
        # Default configuration
        return {
            "method": "advanced",
            "isotonic_regression": True,
            "pattern_rules": True,
            "dynamic_threshold": True,
            "ui_rescaling": True,
            "base_threshold": 0.15,
            "confidence_targets": {
                "positive": [0.7, 0.9],
                "negative": [0.6, 0.85],
                "neutral": [0.5, 0.7]
            }
        }
    
    def _init_pattern_rules(self) -> Dict:
        """Initialize pattern-based adjustment rules"""
        return {
            'rhetorical_question': {
                'pattern': re.compile(r'(why|how|what|where|when|who).*\?(?!.*because|.*answer)', re.IGNORECASE),
                'description': 'Rhetorical questions expressing confusion or curiosity',
                'adjustments': {
                    'confusion': 1.8,
                    'curiosity': 2.0,
                    'realization': 1.3,
                    'neutral': 0.6
                }
            },
            'strong_negative': {
                'pattern': re.compile(r'(hate|terrible|worst|awful|disgusting|horrible|stupid|dumb)', re.IGNORECASE),
                'description': 'Strong negative expressions',
                'adjustments': {
                    'disgust': 2.2,
                    'anger': 2.0,
                    'annoyance': 1.8,
                    'disapproval': 1.5,
                    'neutral': 0.3
                }
            },
            'strong_positive': {
                'pattern': re.compile(r'(love|amazing|wonderful|fantastic|excellent|perfect|awesome|brilliant)', re.IGNORECASE),
                'description': 'Strong positive expressions',
                'adjustments': {
                    'admiration': 2.2,
                    'joy': 2.0,
                    'excitement': 1.8,
                    'love': 1.9,
                    'neutral': 0.3
                }
            },
            'gratitude_expressions': {
                'pattern': re.compile(r'(thank|appreciate|grateful|thanks)', re.IGNORECASE),
                'description': 'Expressions of gratitude',
                'adjustments': {
                    'gratitude': 2.5,
                    'appreciation': 2.0,
                    'caring': 1.3,
                    'neutral': 0.4
                }
            },
            'fear_anxiety': {
                'pattern': re.compile(r'(scared|afraid|terrified|anxious|worried|nervous)', re.IGNORECASE),
                'description': 'Fear and anxiety expressions',
                'adjustments': {
                    'fear': 2.3,
                    'nervousness': 2.0,
                    'disappointment': 1.2,
                    'neutral': 0.3
                }
            },
            'surprise_shock': {
                'pattern': re.compile(r'(surprised|shocked|unexpected|wow|omg|unbelievable)', re.IGNORECASE),
                'description': 'Surprise and shock expressions',
                'adjustments': {
                    'surprise': 2.4,
                    'realization': 1.6,
                    'excitement': 1.3,
                    'neutral': 0.4
                }
            }
        }
    
    def _compute_emotion_priors(self) -> Dict:
        """Compute emotion prior probabilities based on dataset statistics"""
        # Based on GoEmotions dataset statistics
        return {
            'neutral': 0.27,
            'approval': 0.13,
            'admiration': 0.08,
            'annoyance': 0.07,
            'joy': 0.06,
            'gratitude': 0.05,
            'disapproval': 0.05,
            'disappointment': 0.04,
            'love': 0.04,
            'realization': 0.04,
            'optimism': 0.03,
            'sadness': 0.03,
            'anger': 0.03,
            'confusion': 0.03,
            'caring': 0.03,
            'excitement': 0.02,
            'surprise': 0.02,
            'curiosity': 0.02,
            'amusement': 0.02,
            'desire': 0.02,
            'disgust': 0.02,
            'fear': 0.01,
            'embarrassment': 0.01,
            'nervousness': 0.01,
            'pride': 0.01,
            'grief': 0.01,
            'relief': 0.01,
            'remorse': 0.01
        }
    
    def train_isotonic_calibrator(self, validation_probs: np.ndarray, validation_labels: np.ndarray):
        """Train isotonic regression calibrators for each emotion"""
        if not SKLEARN_AVAILABLE:
            print("❌ sklearn not available - cannot train isotonic calibrators")
            print("💡 Install sklearn: pip install scikit-learn")
            return
        
        print("🔧 Training isotonic regression calibrators...")
        
        for i, emotion in enumerate(self.emotion_labels):
            try:
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.fit(validation_probs[:, i], validation_labels[:, i])
                self.isotonic_calibrators[emotion] = iso_reg
                print(f"  ✅ {emotion}: calibrator trained")
            except Exception as e:
                print(f"  ⚠️ {emotion}: failed to train calibrator - {str(e)}")
        
        print(f"✅ Isotonic calibrators trained for {len(self.isotonic_calibrators)} emotions")
    
    def apply_isotonic_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply isotonic regression calibration"""
        if not self.isotonic_calibrators:
            # Fallback: Platt scaling approximation
            return self._platt_scaling_fallback(raw_probs)
        
        calibrated = np.zeros_like(raw_probs)
        for i, emotion in enumerate(self.emotion_labels):
            if emotion in self.isotonic_calibrators:
                try:
                    calibrated[i] = self.isotonic_calibrators[emotion].transform([raw_probs[i]])[0]
                except:
                    # Fallback for individual emotion
                    calibrated[i] = self._platt_scaling_single(raw_probs[i])
            else:
                calibrated[i] = self._platt_scaling_single(raw_probs[i])
        
        return calibrated
    
    def _platt_scaling_fallback(self, raw_probs: np.ndarray) -> np.ndarray:
        """Fallback Platt scaling when isotonic regression unavailable"""
        # Gentler scaling - avoid over-boosting
        calibrated = np.zeros_like(raw_probs)
        
        for i, prob in enumerate(raw_probs):
            if prob <= 0.001:
                calibrated[i] = 0.02
            elif prob >= 0.999:
                calibrated[i] = 0.95
            else:
                # Gentle sigmoid boost
                logit = np.log(prob / (1 - prob))
                # Less aggressive scaling
                scaled_logit = 1.5 * logit + 0.3  # Reduced from 2.5, -0.8
                calibrated[i] = 1 / (1 + np.exp(-scaled_logit))
        
        return np.clip(calibrated, 0.01, 0.95)
    
    def _platt_scaling_single(self, prob: float) -> float:
        """Single probability Platt scaling"""
        if prob <= 0:
            return 0.01
        if prob >= 1:
            return 0.99
        
        logit = np.log(prob / (1 - prob))
        scaled_logit = 2.5 * logit - 0.8
        return 1 / (1 + np.exp(-scaled_logit))
    
    def apply_pattern_rules(self, text: str, probs: np.ndarray) -> np.ndarray:
        """Apply pattern-based adjustments"""
        if not self.config.get('pattern_rules', True):
            return probs
        
        adjusted_probs = probs.copy()
        applied_rules = []
        
        for rule_name, rule_config in self.pattern_rules.items():
            if rule_config['pattern'].search(text):
                applied_rules.append(rule_name)
                
                for emotion, multiplier in rule_config['adjustments'].items():
                    if emotion in self.emotion_labels:
                        emotion_idx = self.emotion_labels.index(emotion)
                        adjusted_probs[emotion_idx] *= multiplier
        
        # Clip to valid probability range
        adjusted_probs = np.clip(adjusted_probs, 0.001, 0.999)
        
        if applied_rules:
            print(f"  📝 Applied rules: {', '.join(applied_rules)}")
        
        return adjusted_probs
    
    def compute_dynamic_threshold(self, probs: np.ndarray, text: str) -> float:
        """Compute dynamic threshold based on text and probability characteristics"""
        if not self.config.get('dynamic_threshold', True):
            return self.config.get('base_threshold', 0.15)
        
        base_threshold = self.config.get('base_threshold', 0.15)
        
        # Factor 1: Text length adjustment
        text_length = len(text.split())
        if text_length < 5:
            length_factor = 1.3  # Short texts need higher threshold
        elif text_length > 20:
            length_factor = 0.7  # Long texts can use lower threshold
        else:
            length_factor = 1.0
        
        # Factor 2: Confidence distribution
        max_prob = np.max(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-7))
        normalized_entropy = entropy / np.log(len(probs))  # Normalize by max entropy
        
        if max_prob < 0.2:  # Very uncertain
            confidence_factor = 0.6
        elif max_prob > 0.8:  # Very confident
            confidence_factor = 1.4
        else:
            confidence_factor = 1.0
        
        # Factor 3: Entropy-based adjustment
        if normalized_entropy > 0.9:  # Very uniform distribution
            entropy_factor = 0.8
        elif normalized_entropy < 0.3:  # Very concentrated
            entropy_factor = 1.2
        else:
            entropy_factor = 1.0
        
        # Compute final threshold
        dynamic_threshold = base_threshold * length_factor * confidence_factor * entropy_factor
        final_threshold = np.clip(dynamic_threshold, 0.08, 0.35)
        
        print(f"  🎯 Dynamic threshold: {final_threshold:.3f} (base={base_threshold:.3f}, factors: length={length_factor:.2f}, conf={confidence_factor:.2f}, entropy={entropy_factor:.2f})")
        
        return final_threshold
    
    def rescale_for_ui(self, probs: np.ndarray) -> np.ndarray:
        """Rescale probabilities for better UI display"""
        if not self.config.get('ui_rescaling', True):
            return probs
        
        # More conservative UI rescaling
        display_probs = probs.copy()
        
        # Step 1: Gentle power transformation
        max_prob = np.max(probs)
        
        if max_prob < 0.1:
            # Very low confidence - moderate boost
            power_factor = 0.8
        elif max_prob < 0.3:
            # Low-medium confidence - gentle boost
            power_factor = 0.9
        elif max_prob > 0.7:
            # Already high confidence - minimal change
            power_factor = 1.0
        else:
            # Medium confidence - slight boost
            power_factor = 0.95
        
        display_probs = np.power(display_probs, power_factor)
        
        # Step 2: Category-specific adjustments
        for i, emotion in enumerate(self.emotion_labels):
            if emotion in self.positive_emotions:
                display_probs[i] *= 1.1  # 10% boost for positive emotions
            elif emotion in self.negative_emotions:
                display_probs[i] *= 1.15  # 15% boost for negative emotions
        
        # Step 3: Ensure reasonable display range
        display_probs = np.clip(display_probs, 0.01, 0.95)
        
        # Step 4: Make sure top emotion is visually distinct
        top_idx = np.argmax(display_probs)
        if display_probs[top_idx] < 0.3:
            # Boost top emotion for visibility
            display_probs[top_idx] = min(display_probs[top_idx] * 1.3, 0.85)
        
        return display_probs
    
    def optimize_predictions(self, text: str, raw_probs: np.ndarray) -> Dict[str, Any]:
        """
        Main optimization pipeline
        """
        print(f"🔍 Optimizing predictions for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Step 1: Isotonic regression calibration
        calibrated_probs = self.apply_isotonic_calibration(raw_probs)
        print(f"  📊 Isotonic calibration: {np.max(raw_probs):.3f} → {np.max(calibrated_probs):.3f}")
        
        # Step 2: Pattern-based adjustments
        pattern_adjusted = self.apply_pattern_rules(text, calibrated_probs)
        print(f"  🎯 Pattern adjustment: {np.max(calibrated_probs):.3f} → {np.max(pattern_adjusted):.3f}")
        
        # Step 3: Dynamic threshold computation
        dynamic_threshold = self.compute_dynamic_threshold(pattern_adjusted, text)
        
        # Step 4: UI confidence rescaling
        display_probs = self.rescale_for_ui(pattern_adjusted)
        print(f"  📱 UI rescaling: {np.max(pattern_adjusted):.3f} → {np.max(display_probs):.3f}")
        
        # Create results
        emotion_scores = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotion_labels, pattern_adjusted)
        }
        
        display_scores = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotion_labels, display_probs)
        }
        
        # Apply dynamic threshold for predictions
        predicted_emotions = [
            emotion for emotion, score in emotion_scores.items() 
            if score > dynamic_threshold
        ]
        
        return {
            'optimized_scores': emotion_scores,
            'display_scores': display_scores,
            'predicted_emotions': predicted_emotions,
            'dynamic_threshold': dynamic_threshold,
            'max_confidence': float(np.max(pattern_adjusted)),
            'display_confidence': float(np.max(display_probs)),
            'optimization_steps': {
                'raw_max': float(np.max(raw_probs)),
                'calibrated_max': float(np.max(calibrated_probs)),
                'pattern_adjusted_max': float(np.max(pattern_adjusted)),
                'display_max': float(np.max(display_probs))
            }
        }
    
    def save_calibrators(self, save_path: str):
        """Save trained isotonic calibrators"""
        if self.isotonic_calibrators:
            with open(save_path, 'wb') as f:
                pickle.dump(self.isotonic_calibrators, f)
            print(f"✅ Calibrators saved to: {save_path}")
    
    def load_calibrators(self, load_path: str):
        """Load trained isotonic calibrators"""
        if os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    self.isotonic_calibrators = pickle.load(f)
                print(f"✅ Calibrators loaded from: {load_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load calibrators: {str(e)}")
                return False
        return False

# Utility functions for integration
def create_optimizer(config_path: Optional[str] = None) -> AdvancedInferenceOptimizer:
    """Factory function to create optimizer"""
    return AdvancedInferenceOptimizer(config_path)

if __name__ == "__main__":
    # Test the optimizer
    optimizer = AdvancedInferenceOptimizer()
    
    # Test cases
    test_cases = [
        ("This is amazing! I love it so much!", np.array([0.02, 0.01, 0.03, 0.04, 0.08, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.01, 0.543, 0.01, 0.05, 0.01, 0.46, 0.07, 0.01, 0.09, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.02])),
        ("Why does this always happen to me?", np.array([0.02, 0.01, 0.08, 0.12, 0.05, 0.02, 0.15, 0.08, 0.01, 0.06, 0.04, 0.02, 0.01, 0.02, 0.03, 0.02, 0.01, 0.02, 0.01, 0.04, 0.03, 0.01, 0.04, 0.01, 0.01, 0.08, 0.02, 0.35])),
        ("I hate this stupid thing!", np.array([0.01, 0.01, 0.243, 0.274, 0.02, 0.01, 0.02, 0.01, 0.01, 0.068, 0.08, 0.311, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.01, 0.05]))
    ]
    
    print("🧪 Testing Advanced Inference Optimizer")
    print("=" * 80)
    
    for text, raw_probs in test_cases:
        print(f"\n📝 Text: '{text}'")
        result = optimizer.optimize_predictions(text, raw_probs)
        
        print(f"   📊 Max confidence: {result['max_confidence']:.1%}")
        print(f"   📱 Display confidence: {result['display_confidence']:.1%}")
        print(f"   🎯 Dynamic threshold: {result['dynamic_threshold']:.3f}")
        print(f"   ✅ Predicted: {', '.join(result['predicted_emotions']) if result['predicted_emotions'] else 'None'}")
        
        # Show top 3 optimized emotions
        top_emotions = sorted(result['optimized_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   🏆 Top 3: {', '.join([f'{e}({s:.1%})' for e, s in top_emotions])}")
        print("-" * 60)