"""
Shared utilities for model loading and prediction
"""

import os
import json
import pickle
import logging
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle optional imports
try:
    import streamlit as st
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

def load_pickle_with_fallback(file_path: str):
    """Load pickle file with fallback strategies for compatibility"""
    try:
        # First try standard pickle load
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Standard pickle load failed: {e}")
        
        # Try with custom unpickler
        try:
            with open(file_path, 'rb') as f:
                return CustomUnpickler(f).load()
        except Exception as e2:
            logger.error(f"Custom unpickler failed: {e2}")
            
            # Last resort: try joblib if available
            try:
                import joblib
                return joblib.load(file_path)
            except ImportError:
                logger.error("joblib not available for fallback")
            except Exception as e3:
                logger.error(f"joblib load failed: {e3}")
        
        return None

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle class path changes and missing modules"""
    
    def find_class(self, module, name):
        # Handle numpy version compatibility (numpy 1.x -> 2.x migration)
        if module == 'numpy.core' or module == 'numpy._core':
            try:
                import numpy
                
                # Special handling for common numpy functions
                if name == '_reconstruct':
                    # Try to get the actual _reconstruct function
                    try:
                        if hasattr(numpy, '_core'):
                            # numpy 2.x
                            from numpy._core.multiarray import _reconstruct
                            return _reconstruct
                        elif hasattr(numpy, 'core'):
                            # numpy 1.x
                            from numpy.core.multiarray import _reconstruct
                            return _reconstruct
                    except ImportError:
                        pass
                
                # Try both module paths
                if hasattr(numpy, '_core'):
                    # numpy 2.x - try _core first
                    try:
                        return super().find_class('numpy._core', name)
                    except:
                        try:
                            return super().find_class('numpy.core', name)
                        except:
                            pass
                elif hasattr(numpy, 'core'):
                    # numpy 1.x - try core first
                    try:
                        return super().find_class('numpy.core', name)
                    except:
                        try:
                            return super().find_class('numpy._core', name)
                        except:
                            pass
            except ImportError:
                pass
        
        # Handle other numpy submodule redirections
        numpy_module_mappings = {
            'numpy.core.multiarray': ['numpy._core.multiarray', 'numpy.core.multiarray'],
            'numpy._core.multiarray': ['numpy._core.multiarray', 'numpy.core.multiarray'],
            'numpy.core.umath': ['numpy._core.umath', 'numpy.core.umath'],
            'numpy._core.umath': ['numpy._core.umath', 'numpy.core.umath'],
            'numpy.core.numeric': ['numpy._core.numeric', 'numpy.core.numeric'],
            'numpy._core.numeric': ['numpy._core.numeric', 'numpy.core.numeric']
        }
        
        if module in numpy_module_mappings:
            for alt_module in numpy_module_mappings[module]:
                try:
                    return super().find_class(alt_module, name)
                except:
                    continue
        
        # Handle class path remapping for compatibility
        if module == '__main__':
            # Map to the matching classes in utils
            if name == 'TextPreprocessor':
                from utils.preprocessing import TextPreprocessor
                return TextPreprocessor
            elif name == 'Config':
                from utils.preprocessing import Config
                return Config
            elif name == 'FeatureEngineer':
                from utils.feature_engineering import FeatureEngineer
                return FeatureEngineer
            elif name == 'FeatureExtractor':
                from utils.feature_engineering import FeatureExtractor
                return FeatureExtractor
        
        # Handle keras module remapping
        if 'keras.preprocessing.text' in module:
            try:
                from tensorflow.keras.preprocessing import text as keras_text
                return getattr(keras_text, name)
            except ImportError:
                pass
        
        # Default behavior
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError) as e:
            # If class not found, create a placeholder
            logger.warning(f"Class {module}.{name} not found, creating placeholder")
            
            class PlaceholderClass:
                """Flexible placeholder class that accepts any arguments"""
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs
                
                def __call__(self, *args, **kwargs):
                    # Make it callable
                    return self.__class__(*args, **kwargs)
                
                def __getattr__(self, name):
                    # Return another placeholder for any attribute access
                    return PlaceholderClass()
                
                def __repr__(self):
                    return f"PlaceholderClass({module}.{name})"
            
            return PlaceholderClass

# Emotion labels for GoEmotions dataset
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

class ModelPerformance:
    """Container for model performance metrics"""
    
    def __init__(self, model_name: str, f1_macro: float, f1_micro: float, 
                 precision: float = None, recall: float = None, training_time: float = None):
        self.model_name = model_name
        self.f1_macro = f1_macro
        self.f1_micro = f1_micro
        self.precision = precision
        self.recall = recall
        self.training_time = training_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display"""
        return {
            "Model": self.model_name,
            "F1 Macro": f"{self.f1_macro:.4f}" if self.f1_macro else "N/A",
            "F1 Micro": f"{self.f1_micro:.4f}" if self.f1_micro else "N/A",
            "Precision": f"{self.precision:.4f}" if self.precision else "N/A",
            "Recall": f"{self.recall:.4f}" if self.recall else "N/A",
            "Training Time": f"{self.training_time:.2f}s" if self.training_time else "N/A"
        }

class PredictionResult:
    """Container for model prediction results"""
    
    def __init__(self, model_name: str, emotion_scores: Dict[str, float], 
                 predicted_emotions: List[str], prediction_time: float = None):
        self.model_name = model_name
        self.emotion_scores = emotion_scores
        self.predicted_emotions = predicted_emotions
        self.prediction_time = prediction_time
        self.timestamp = datetime.now()
    
    def get_top_emotions(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N emotions by score"""
        sorted_emotions = sorted(self.emotion_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        return sorted_emotions[:n]
    
    def get_confidence(self) -> float:
        """Get maximum confidence score"""
        return max(self.emotion_scores.values()) if self.emotion_scores else 0.0

def get_project_root() -> str:
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def load_json_config(file_path: str) -> Dict[str, Any]:
    """Load JSON configuration file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {str(e)}")
        return {}

def safe_pickle_load(file_path: str) -> Any:
    """Safely load pickle file with custom unpickler and compatibility fixes"""
    
    # Strategy 1: Try CustomUnpickler first
    try:
        logger.info(f"Trying CustomUnpickler for {file_path}")
        with open(file_path, 'rb') as f:
            return CustomUnpickler(f).load()
    except Exception as e:
        logger.warning(f"CustomUnpickler failed for {file_path}: {str(e)}")
    
    # Strategy 2: Try normal pickle loading
    try:
        logger.info(f"Trying standard pickle.load for {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if 'keras.preprocessing.text' in str(e):
            logger.warning("Keras preprocessing module not found, trying compatibility fix...")
            try:
                # Try to fix keras preprocessing imports
                import types
                
                # Create a mock keras.preprocessing.text module
                if 'keras' not in sys.modules:
                    sys.modules['keras'] = types.ModuleType('keras')
                if 'keras.preprocessing' not in sys.modules:
                    sys.modules['keras.preprocessing'] = types.ModuleType('keras.preprocessing')
                
                # Try to import from tensorflow.keras instead
                try:
                    from tensorflow.keras.preprocessing import text as keras_text
                    sys.modules['keras.preprocessing.text'] = keras_text
                    logger.info("Using tensorflow.keras.preprocessing.text as fallback")
                except ImportError:
                    # Create a minimal mock if tensorflow version doesn't have it
                    mock_module = types.ModuleType('keras.preprocessing.text')
                    sys.modules['keras.preprocessing.text'] = mock_module
                    logger.info("Created mock keras.preprocessing.text module")
                
                # Try loading again with CustomUnpickler
                with open(file_path, 'rb') as f:
                    return CustomUnpickler(f).load()
                    
            except Exception as fix_error:
                logger.error(f"Keras compatibility fix failed: {str(fix_error)}")
                return None
        else:
            logger.error(f"Failed to load pickle file {file_path}: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Standard pickle.load failed for {file_path}: {str(e)}")
        return None

def apply_threshold(scores: Dict[str, float], threshold: float = 0.5, 
                   optimal_thresholds: Dict[str, float] = None, 
                   use_optimal: bool = True) -> List[str]:
    """Apply thresholds to get predicted emotions"""
    predicted_emotions = []
    
    for emotion, score in scores.items():
        if use_optimal and optimal_thresholds and emotion in optimal_thresholds:
            thresh = optimal_thresholds[emotion]
        else:
            thresh = threshold
            
        if score >= thresh:
            predicted_emotions.append(emotion)
    
    return predicted_emotions

def format_prediction_time(time_seconds: float) -> str:
    """Format prediction time for display"""
    if time_seconds < 0.001:
        return f"{time_seconds*1000000:.1f}μs"
    elif time_seconds < 1.0:
        return f"{time_seconds*1000:.1f}ms"
    else:
        return f"{time_seconds:.2f}s"

def calculate_model_agreement(results: List[PredictionResult]) -> Dict[str, Any]:
    """Calculate agreement between models"""
    if len(results) < 2:
        return {"agreement_score": 1.0, "common_emotions": []}
    
    # Get all predicted emotions from all models
    all_predictions = [set(r.predicted_emotions) for r in results]
    
    # Calculate intersection and union
    common_emotions = set.intersection(*all_predictions)
    all_emotions = set.union(*all_predictions)
    
    # Calculate Jaccard similarity (intersection over union)
    if len(all_emotions) == 0:
        agreement_score = 1.0
    else:
        agreement_score = len(common_emotions) / len(all_emotions)
    
    return {
        "agreement_score": agreement_score,
        "common_emotions": list(common_emotions),
        "total_unique_emotions": len(all_emotions),
        "agreement_percentage": agreement_score * 100
    }

def normalize_emotion_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize emotion scores to sum to 1"""
    total = sum(scores.values())
    if total == 0:
        return scores
    return {emotion: score / total for emotion, score in scores.items()}

def get_model_summary_stats(results: List[PredictionResult]) -> Dict[str, Any]:
    """Get summary statistics for a list of prediction results"""
    if not results:
        return {}
    
    prediction_times = [r.prediction_time for r in results if r.prediction_time]
    emotion_counts = {}
    
    for result in results:
        for emotion in result.predicted_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return {
        "total_predictions": len(results),
        "avg_prediction_time": np.mean(prediction_times) if prediction_times else 0,
        "most_common_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None,
        "avg_emotions_per_text": np.mean([len(r.predicted_emotions) for r in results]),
        "avg_confidence": np.mean([r.get_confidence() for r in results])
    }

class TextPreprocessor:
    """Simple text preprocessor for all models"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove very short texts
        if len(text.strip()) < 2:
            return ""
            
        return text.strip()
    
    @staticmethod
    def validate_input(text: str) -> bool:
        """Validate input text"""
        return isinstance(text, str) and len(text.strip()) >= 2

def create_error_result(model_name: str, error_msg: str) -> PredictionResult:
    """Create error result for failed predictions"""
    return PredictionResult(
        model_name=model_name,
        emotion_scores={emotion: 0.0 for emotion in EMOTION_LABELS},
        predicted_emotions=[],
        prediction_time=0.0
    )