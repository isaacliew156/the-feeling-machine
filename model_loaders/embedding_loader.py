"""
Word Embedding (CNN + GloVe) Model Loader for GoEmotions Classification
"""

import os
import json
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Handle optional imports
try:
    import streamlit as st
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from .utils import (
    ModelPerformance, PredictionResult, EMOTION_LABELS, 
    get_project_root, apply_threshold, create_error_result,
    TextPreprocessor, logger, load_json_config, safe_pickle_load
)

# Advanced inference optimizer - will be imported when needed
OPTIMIZER_AVAILABLE = False

class WordEmbeddingModelLoader:
    """Word Embedding (CNN + GloVe) model loader and predictor"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.performance_metrics = None
        self.max_sequence_length = 50
        self.optimal_threshold = 0.15
        
        # Advanced inference optimizer
        self.inference_optimizer = None
        self.use_advanced_optimization = True
        
    def load_model(_self) -> bool:
        """Load Word Embedding model and tokenizer with caching"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available for Word Embedding model")
            if ST_AVAILABLE:
                st.error("TensorFlow is required for Word Embedding model but not installed")
            return False
        
        try:
            project_root = get_project_root()
            model_dir = os.path.join(project_root, "models", "word_embedding")
            
            # Check if model files exist
            model_path = os.path.join(model_dir, "best_embedding_model")
            config_path = os.path.join(model_dir, "config.json")
            tokenizer_path = os.path.join(model_dir, "tokenizer.pickle")
            
            if not os.path.exists(model_path):
                logger.error(f"Word Embedding model not found: {model_path}")
                return False
                
            if not os.path.exists(tokenizer_path):
                logger.error(f"Word Embedding tokenizer not found: {tokenizer_path}")
                return False
            
            # Load configuration
            logger.info(f"Loading config from: {config_path}")
            _self.config = load_json_config(config_path)
            if not _self.config:
                logger.error("Failed to load Word Embedding model config")
                return False
            
            _self.max_sequence_length = _self.config.get('MAX_SEQUENCE_LENGTH', 50)
            _self.optimal_threshold = _self.config.get('BEST_THRESHOLD', 0.15)
            
            logger.info(f"Word Embedding config loaded: max_seq_len={_self.max_sequence_length}, threshold={_self.optimal_threshold}")
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from: {tokenizer_path}")
            _self.tokenizer = safe_pickle_load(tokenizer_path)
            if _self.tokenizer is None:
                logger.error("Failed to load Word Embedding tokenizer")
                return False
            
            logger.info("Word Embedding tokenizer loaded successfully")
            
            # Load model
            logger.info(f"Loading TensorFlow model from: {model_path}")
            try:
                # Try standard loading first
                _self.model = tf.keras.models.load_model(model_path)
                logger.info("Word Embedding model loaded successfully (standard method)")
                
            except Exception as model_load_error:
                error_str = str(model_load_error)
                if "Keras 3" in error_str and "SavedModel" in error_str:
                    logger.warning("Keras 3 SavedModel compatibility issue detected, trying TFSMLayer...")
                    try:
                        # Use TFSMLayer for Keras 3 compatibility
                        tfsm_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                        
                        # Create a wrapper model
                        input_shape = (_self.max_sequence_length,)  # From config
                        inputs = tf.keras.Input(shape=input_shape, dtype=tf.int32)
                        outputs = tfsm_layer(inputs)
                        
                        _self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
                        logger.info("Word Embedding model loaded successfully (TFSMLayer method)")
                        
                    except Exception as tfsm_error:
                        logger.error(f"TFSMLayer loading also failed: {str(tfsm_error)}")
                        logger.info("Trying alternative loading approaches...")
                        
                        # Try with different call endpoints
                        alternative_endpoints = ['predict', 'inference', '__call__']
                        for endpoint in alternative_endpoints:
                            try:
                                logger.info(f"Trying endpoint: {endpoint}")
                                tfsm_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint=endpoint)
                                
                                input_shape = (_self.max_sequence_length,)
                                inputs = tf.keras.Input(shape=input_shape, dtype=tf.int32)
                                outputs = tfsm_layer(inputs)
                                
                                _self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
                                logger.info(f"Model loaded successfully with endpoint: {endpoint}")
                                break
                                
                            except Exception as alt_error:
                                logger.debug(f"Endpoint {endpoint} failed: {str(alt_error)}")
                                continue
                        else:
                            # If all endpoints fail, try direct TensorFlow loading
                            logger.info("Trying direct TensorFlow SavedModel loading...")
                            try:
                                import tensorflow.compat.v1 as tf_v1
                                tf_v1.disable_v2_behavior()
                                
                                # Load as TF SavedModel
                                imported = tf.saved_model.load(model_path)
                                
                                # Create a callable wrapper
                                _self.model = imported
                                logger.info("Model loaded using TensorFlow SavedModel (TF v1 compat)")
                                
                            except Exception as tf_error:
                                logger.error(f"All loading methods failed. Final error: {str(tf_error)}")
                                raise model_load_error
                else:
                    logger.error(f"TensorFlow model loading failed: {error_str}")
                    raise model_load_error
            
            # Print model info if available
            if hasattr(_self.model, 'input_shape'):
                logger.info(f"Model input shape: {_self.model.input_shape}")
            if hasattr(_self.model, 'output_shape'):
                logger.info(f"Model output shape: {_self.model.output_shape}")
            elif hasattr(_self.model, 'signatures'):
                logger.info(f"Model signatures: {list(_self.model.signatures.keys())}")
            
            # Load performance metrics
            _self._load_performance_metrics()
            
            # Initialize advanced inference optimizer
            if _self.use_advanced_optimization:
                try:
                    # Dynamic import of advanced optimizer
                    import sys
                    project_root = get_project_root()
                    scripts_path = os.path.join(project_root, 'scripts')
                    if scripts_path not in sys.path:
                        sys.path.append(scripts_path)
                    
                    from advanced_inference_optimizer import AdvancedInferenceOptimizer
                    
                    config_path = os.path.join(project_root, "models", "word_embedding", "config.json")
                    _self.inference_optimizer = AdvancedInferenceOptimizer(config_path)
                    
                    # Try to load pre-trained calibrators
                    calibrator_path = os.path.join(project_root, "models", "word_embedding", "isotonic_calibrators.pkl")
                    _self.inference_optimizer.load_calibrators(calibrator_path)
                    
                    logger.info("✅ Advanced inference optimizer loaded successfully")
                    
                except Exception as opt_error:
                    logger.warning(f"Failed to initialize optimizer: {str(opt_error)}")
                    _self.inference_optimizer = None
                    _self.use_advanced_optimization = False
            else:
                logger.info("Using fallback confidence calibration")
                _self.use_advanced_optimization = False
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load Word Embedding model: {error_msg}")
            
            # Provide more specific error information
            if "No module named 'tensorflow'" in error_msg:
                detailed_msg = "TensorFlow is required but not installed. Install with: pip install tensorflow"
            elif "SavedModel" in error_msg:
                detailed_msg = f"TensorFlow SavedModel loading failed: {error_msg}"
            elif "pickle" in error_msg.lower() or "tokenizer" in error_msg.lower():
                detailed_msg = f"Tokenizer loading failed: {error_msg}"
            else:
                detailed_msg = f"General loading error: {error_msg}"
            
            if ST_AVAILABLE:
                st.error(f"Failed to load Word Embedding model: {detailed_msg}")
            
            return False
    
    def _load_performance_metrics(self):
        """Load performance metrics from results file"""
        try:
            project_root = get_project_root()
            results_file = os.path.join(project_root, "results", "word_embedding_results", "embedding_results.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract performance metrics
                perf = results.get('test_performance', {})
                
                self.performance_metrics = ModelPerformance(
                    model_name="CNN + GloVe",
                    f1_macro=perf.get('f1_macro', 0.315),
                    f1_micro=perf.get('f1_micro', 0.405),
                    precision=perf.get('precision', 0.281),
                    recall=perf.get('recall', 0.401),
                    training_time=None  # Not available in results
                )
                
                logger.info(f"Word Embedding performance metrics loaded: F1-Macro={self.performance_metrics.f1_macro:.4f}")
            else:
                logger.warning(f"Word Embedding results file not found: {results_file}")
                # Use default metrics
                self.performance_metrics = ModelPerformance(
                    model_name="CNN + GloVe",
                    f1_macro=0.315,
                    f1_micro=0.405,
                    precision=0.281,
                    recall=0.401
                )
                
        except Exception as e:
            logger.error(f"Failed to load Word Embedding performance metrics: {str(e)}")
            # Use default metrics
            self.performance_metrics = ModelPerformance(
                model_name="CNN + GloVe",
                f1_macro=0.315,
                f1_micro=0.405,
                precision=0.281,
                recall=0.401
            )
    
    def preprocess_text(self, text: str) -> Optional[np.ndarray]:
        """Preprocess text for Word Embedding model"""
        if not TextPreprocessor.validate_input(text):
            return None
        
        # Clean text
        cleaned_text = TextPreprocessor.clean_text(text)
        
        # Tokenize using the loaded tokenizer
        try:
            # Convert text to sequences
            sequences = self.tokenizer.texts_to_sequences([cleaned_text])
            
            # Pad sequences
            padded_sequences = pad_sequences(
                sequences, 
                maxlen=self.max_sequence_length,
                padding='post',
                truncating='post'
            )
            
            return padded_sequences
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            return None
    
    def predict_single(self, text: str, threshold: float = 0.5, 
                      use_optimal: bool = True) -> PredictionResult:
        """Predict emotions for single text"""
        start_time = time.time()
        
        try:
            # Check if model is loaded
            if self.model is None or self.tokenizer is None:
                if not self.load_model():
                    return create_error_result("CNN + GloVe", "Model not loaded")
            
            # Preprocess text
            processed_input = self.preprocess_text(text)
            if processed_input is None:
                return create_error_result("CNN + GloVe", "Text preprocessing failed")
            
            # Make prediction - handle different model types
            try:
                if hasattr(self.model, 'predict'):
                    # Standard Keras model
                    predictions = self.model.predict(processed_input, verbose=0)
                elif hasattr(self.model, 'signatures'):
                    # TensorFlow SavedModel
                    signature_key = list(self.model.signatures.keys())[0]
                    infer = self.model.signatures[signature_key]
                    predictions = infer(tf.constant(processed_input, dtype=tf.int32))
                    
                    # Extract predictions from signature output
                    if isinstance(predictions, dict):
                        # Find the output tensor
                        output_keys = list(predictions.keys())
                        predictions = predictions[output_keys[0]].numpy()
                    else:
                        predictions = predictions.numpy()
                elif callable(self.model):
                    # Direct callable model
                    predictions = self.model(tf.constant(processed_input, dtype=tf.int32))
                    if hasattr(predictions, 'numpy'):
                        predictions = predictions.numpy()
                else:
                    raise ValueError("Unknown model type")
                
                # Handle prediction output
                if len(predictions.shape) > 1:
                    probabilities = predictions[0]  # First (and only) sample
                else:
                    probabilities = predictions
                    
            except Exception as pred_error:
                logger.error(f"Prediction failed: {str(pred_error)}")
                raise pred_error
            
            # Ensure we have the right number of emotions
            if len(probabilities) != len(EMOTION_LABELS):
                logger.warning(f"Prediction shape mismatch: got {len(probabilities)}, expected {len(EMOTION_LABELS)}")
                # Pad or truncate as needed
                if len(probabilities) < len(EMOTION_LABELS):
                    probabilities = np.pad(probabilities, (0, len(EMOTION_LABELS) - len(probabilities)))
                else:
                    probabilities = probabilities[:len(EMOTION_LABELS)]
            
            # Advanced inference optimization
            if self.use_advanced_optimization and self.inference_optimizer:
                # Use advanced optimizer
                optimization_result = self.inference_optimizer.optimize_predictions(text, probabilities)
                optimized_scores_dict = optimization_result['optimized_scores']
                display_scores_dict = optimization_result['display_scores']
                dynamic_threshold = optimization_result['dynamic_threshold']
                
                # Convert dicts back to arrays for compatibility
                probabilities = np.array([optimized_scores_dict[emotion] for emotion in EMOTION_LABELS])
                display_probabilities = np.array([display_scores_dict[emotion] for emotion in EMOTION_LABELS])
                
                logger.info(f"Advanced optimization applied: max_conf={optimization_result['max_confidence']:.3f}, display_conf={optimization_result['display_confidence']:.3f}")
                
            else:
                # Fallback: Simple linear scaling 
                logger.info("Using fallback linear scaling")
                negative_emotions = {'anger', 'disgust', 'fear', 'sadness', 'disappointment', 'annoyance', 
                                   'disapproval', 'grief', 'remorse', 'nervousness', 'embarrassment'}
                positive_emotions = {'joy', 'love', 'admiration', 'gratitude', 'excitement', 'pride', 
                                   'amusement', 'approval', 'caring', 'optimism', 'relief'}
                
                for i, emotion in enumerate(EMOTION_LABELS):
                    prob = probabilities[i]
                    
                    # Determine scaling factor based on emotion category
                    if emotion in negative_emotions:
                        scale_factor = 2.8  # Aggressive boost for negative emotions
                    elif emotion in positive_emotions:
                        scale_factor = 1.8  # Moderate boost for positive emotions  
                    else:
                        scale_factor = 1.5  # Conservative boost for neutral
                    
                    # Apply linear scaling with cap
                    scaled_prob = prob * scale_factor
                    probabilities[i] = min(scaled_prob, 0.95)  # Cap at 95%
                
                display_probabilities = probabilities
                dynamic_threshold = self.optimal_threshold
            
            # Create emotion scores dictionary (for internal logic)
            emotion_scores = {
                emotion: float(prob) 
                for emotion, prob in zip(EMOTION_LABELS, probabilities)
            }
            
            # Create display scores dictionary (for UI)
            display_emotion_scores = {
                emotion: float(prob) 
                for emotion, prob in zip(EMOTION_LABELS, display_probabilities)
            }
            
            # Apply thresholds using dynamic threshold if available
            if self.use_advanced_optimization and self.inference_optimizer and 'dynamic_threshold' in locals():
                # Use dynamic threshold
                predicted_emotions = [
                    emotion for emotion, score in emotion_scores.items() 
                    if score > dynamic_threshold
                ]
                logger.info(f"Dynamic threshold applied: {dynamic_threshold:.3f}")
            else:
                # Use traditional threshold
                optimal_thresholds = {emotion: self.optimal_threshold for emotion in EMOTION_LABELS}
                predicted_emotions = apply_threshold(
                    emotion_scores, threshold, optimal_thresholds, use_optimal
                )
            
            prediction_time = time.time() - start_time
            
            return PredictionResult(
                model_name="CNN + GloVe",
                emotion_scores=display_emotion_scores,  # Use display scores for UI
                predicted_emotions=predicted_emotions,
                prediction_time=prediction_time
            )
            
        except Exception as e:
            logger.error(f"Word Embedding prediction failed: {str(e)}")
            return create_error_result("CNN + GloVe", f"Prediction failed: {str(e)}")
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5, 
                     use_optimal: bool = True) -> List[PredictionResult]:
        """Predict emotions for multiple texts"""
        results = []
        
        # For efficiency, we could batch process, but for now do individual predictions
        for text in texts:
            result = self.predict_single(text, threshold, use_optimal)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.performance_metrics is None:
            self._load_performance_metrics()
        
        info = {
            "name": "CNN + GloVe Embeddings",
            "type": "Convolutional Neural Network",
            "architecture": "CNN with Word Embeddings",
            "max_sequence_length": self.max_sequence_length,
            "optimal_threshold": self.optimal_threshold,
            "performance": self.performance_metrics.to_dict() if self.performance_metrics else {}
        }
        
        # Add config details if available
        if self.config:
            info.update({
                "embedding_dim": self.config.get('EMBEDDING_DIM', 300),
                "vocab_size": self.config.get('MAX_VOCAB_SIZE', 20000),
                "cnn_filters": self.config.get('CNN_FILTERS', [128, 128, 128]),
                "cnn_kernel_sizes": self.config.get('CNN_KERNEL_SIZES', [2, 3, 4]),
                "hidden_dim": self.config.get('HIDDEN_DIM', 256)
            })
        
        return info
    
    def is_available(self) -> bool:
        """Check if Word Embedding dependencies are available"""
        return TF_AVAILABLE
    
    def get_required_packages(self) -> List[str]:
        """Get list of required packages"""
        return ["tensorflow", "numpy"]

# Global instance
_embedding_loader = None

def get_embedding_loader() -> WordEmbeddingModelLoader:
    """Get singleton Word Embedding loader instance"""
    global _embedding_loader
    if _embedding_loader is None:
        _embedding_loader = WordEmbeddingModelLoader()
    return _embedding_loader

# Convenience functions
def predict_embedding(text: str, threshold: float = 0.5, use_optimal: bool = True) -> PredictionResult:
    """Convenience function for Word Embedding prediction"""
    loader = get_embedding_loader()
    return loader.predict_single(text, threshold, use_optimal)

def load_embedding_model() -> bool:
    """Convenience function to load Word Embedding model"""
    loader = get_embedding_loader()
    return loader.load_model()

def get_embedding_info() -> Dict[str, Any]:
    """Convenience function to get Word Embedding model info"""
    loader = get_embedding_loader()
    return loader.get_model_info()