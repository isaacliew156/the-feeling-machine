"""
Traditional ML Model Loader for GoEmotions Classification
"""

import os
import json
import time
import pickle
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Handle optional imports
try:
    import streamlit as st
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from .utils import (
    ModelPerformance, PredictionResult, EMOTION_LABELS, 
    get_project_root, apply_threshold, create_error_result,
    TextPreprocessor, logger, load_json_config, load_pickle_with_fallback
)

class TraditionalMLModelLoader:
    """Traditional ML model loader and predictor"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_extractor = None
        self.metadata = None
        self.optimal_thresholds = None
        self.performance_metrics = None
        
    def load_model(_self) -> bool:
        """Load Traditional ML model and components with caching"""
        try:
            project_root = get_project_root()
            model_dir = os.path.join(project_root, "models", "traditional_ml")
            
            # Check if model files exist
            metadata_path = os.path.join(model_dir, "metadata.json")
            preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
            feature_extractor_path = os.path.join(model_dir, "feature_extractor.pkl")
            
            if not os.path.exists(metadata_path):
                logger.error(f"Traditional ML metadata not found: {metadata_path}")
                return False
                
            if not os.path.exists(preprocessor_path):
                logger.error(f"Traditional ML preprocessor not found: {preprocessor_path}")
                return False
                
            if not os.path.exists(feature_extractor_path):
                logger.error(f"Traditional ML feature extractor not found: {feature_extractor_path}")
                return False
            
            # Load metadata
            logger.info("Loading Traditional ML metadata...")
            _self.metadata = load_json_config(metadata_path)
            if not _self.metadata:
                logger.error("Failed to load Traditional ML metadata")
                return False
            
            # Get optimal thresholds
            _self.optimal_thresholds = _self.metadata.get('optimal_thresholds', {})
            
            # Ensure NLTK data is available
            _self._ensure_nltk_data()
            
            # Load preprocessor with multiple fallback strategies
            logger.info("Loading Traditional ML preprocessor...")
            _self.preprocessor = _self._load_pickle_with_fallback(preprocessor_path, "preprocessor")
            if _self.preprocessor is None:
                logger.error("Failed to load Traditional ML preprocessor")
                return False
            logger.info(f"Preprocessor loaded: {type(_self.preprocessor).__name__}")
            
            # Config should now match from pickle file
            logger.info("Preprocessor config loaded from pickle")
            
            # Load feature extractor with multiple fallback strategies
            logger.info("Loading Traditional ML feature extractor...")
            _self.feature_extractor = _self._load_pickle_with_fallback(feature_extractor_path, "feature extractor")
            if _self.feature_extractor is None:
                logger.error("Failed to load Traditional ML feature extractor")
                return False
            logger.info(f"Feature extractor loaded: {type(_self.feature_extractor).__name__}")
            
            # Config should now match from pickle file
            logger.info("Feature extractor config loaded from pickle")
            
            # Load best model
            best_model_name = _self.metadata['best_model']['name']
            safe_name = best_model_name.lower().replace(' ', '_')
            model_path = os.path.join(model_dir, f'{safe_name}_model.pkl')
            
            if not os.path.exists(model_path):
                logger.error(f"Traditional ML model not found: {model_path}")
                return False
            
            logger.info(f"Loading Traditional ML model: {best_model_name}")
            _self.model = _self._load_pickle_with_fallback(model_path, f"model ({best_model_name})")
            if _self.model is None:
                logger.error(f"Failed to load Traditional ML model: {best_model_name}")
                return False
            logger.info(f"Model loaded: {type(_self.model).__name__}")
            
            logger.info("Traditional ML model loaded successfully")
            
            # Load performance metrics
            _self._load_performance_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Traditional ML model: {str(e)}")
            if ST_AVAILABLE:
                st.error(f"Failed to load Traditional ML model: {str(e)}")
            return False
    
    def _load_pickle_with_fallback(self, file_path: str, description: str) -> Any:
        """Load pickle file with multiple fallback strategies"""
        strategies = [
            ("joblib.load", lambda path: joblib.load(path)),
            ("pickle.load", lambda path: self._load_with_pickle(path)),
            ("load_pickle_with_fallback", lambda path: load_pickle_with_fallback(path))
        ]
        
        for strategy_name, load_func in strategies:
            try:
                logger.info(f"Trying {strategy_name} for {description}...")
                obj = load_func(file_path)
                if obj is not None:
                    logger.info(f"Successfully loaded {description} using {strategy_name}")
                    return obj
            except Exception as e:
                logger.warning(f"{strategy_name} failed for {description}: {e}")
                continue
        
        logger.error(f"All loading strategies failed for {description}")
        return None
    
    def _load_with_pickle(self, file_path: str) -> Any:
        """Load file with standard pickle"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available"""
        try:
            import nltk
            
            # Check and download required NLTK data
            required_data = [
                ('stopwords', 'corpora/stopwords'),
                ('punkt', 'tokenizers/punkt')
            ]
            
            for data_name, data_path in required_data:
                try:
                    nltk.data.find(data_path)
                    logger.info(f"NLTK {data_name} already available")
                except LookupError:
                    logger.info(f"Downloading NLTK {data_name}...")
                    nltk.download(data_name, quiet=True)
                    
        except Exception as e:
            logger.warning(f"Could not ensure NLTK data: {e}")
    
    
    def _load_performance_metrics(self):
        """Load performance metrics from metadata"""
        try:
            if self.metadata and 'best_model' in self.metadata:
                best_model = self.metadata['best_model']
                metrics = best_model.get('metrics', {})
                
                self.performance_metrics = ModelPerformance(
                    model_name=best_model.get('name', 'Traditional ML'),
                    f1_macro=metrics.get('f1_macro', 0.308),
                    f1_micro=metrics.get('f1_micro', 0.4),
                    precision=metrics.get('precision', 0.32),
                    recall=metrics.get('recall', 0.38),
                    training_time=best_model.get('training_time', None)
                )
                
                logger.info(f"Traditional ML performance metrics loaded: F1-Macro={self.performance_metrics.f1_macro:.4f}")
            else:
                # Use default metrics
                self.performance_metrics = ModelPerformance(
                    model_name="Traditional ML",
                    f1_macro=0.308,
                    f1_micro=0.4,
                    precision=0.32,
                    recall=0.38
                )
                
        except Exception as e:
            logger.error(f"Failed to load Traditional ML performance metrics: {str(e)}")
            # Use default metrics
            self.performance_metrics = ModelPerformance(
                model_name="Traditional ML",
                f1_macro=0.308,
                f1_micro=0.4,
                precision=0.32,
                recall=0.38
            )
    
    def predict_single(self, text: str, threshold: float = 0.5, 
                      use_optimal: bool = True) -> PredictionResult:
        """Predict emotions for single text"""
        start_time = time.time()
        
        try:
            # Check if model is loaded
            if self.model is None or self.preprocessor is None or self.feature_extractor is None:
                if not self.load_model():
                    return create_error_result("Traditional ML", "Model not loaded")
            
            # No longer need config initialization - classes match training structure
            
            # Validate input
            if not TextPreprocessor.validate_input(text):
                return create_error_result("Traditional ML", "Invalid input text")
            
            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)
            if not processed_text:
                return create_error_result("Traditional ML", "Text preprocessing failed")
            
            # Extract features
            features = self.feature_extractor.transform([text], [processed_text])
            
            # Check if NB model needs special features
            model_name = self.metadata['best_model']['name'].lower()
            if 'nb' in model_name:
                if hasattr(self.feature_extractor, 'get_nb_features'):
                    features = self.feature_extractor.get_nb_features(features)
            
            # Get predictions
            predictions = self.model.predict(features)
            
            # Handle sparse matrix output
            if hasattr(predictions, 'toarray'):
                predictions = predictions.toarray()
            
            # Extract first row (single prediction)
            if len(predictions.shape) > 1:
                predictions = predictions[0]
            
            # Get probabilities
            probabilities = self._get_model_probabilities(features, predictions)
            
            # Create emotion scores dictionary
            emotion_columns = self.metadata['dataset_info']['emotion_columns']
            emotion_scores = {
                emotion: float(prob) 
                for emotion, prob in zip(emotion_columns, probabilities)
            }
            
            # Apply thresholds
            predicted_emotions = apply_threshold(
                emotion_scores, threshold, self.optimal_thresholds, use_optimal
            )
            
            prediction_time = time.time() - start_time
            
            return PredictionResult(
                model_name="Traditional ML",
                emotion_scores=emotion_scores,
                predicted_emotions=predicted_emotions,
                prediction_time=prediction_time
            )
            
        except Exception as e:
            logger.error(f"Traditional ML prediction failed: {str(e)}")
            return create_error_result("Traditional ML", f"Prediction failed: {str(e)}")
    
    def _get_model_probabilities(self, features, predictions) -> np.ndarray:
        """Extract probabilities from model with error handling"""
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(features)
                
                # Handle sparse matrix output
                if hasattr(probabilities, 'toarray'):
                    probabilities = probabilities.toarray()
                
                # Handle different probability formats
                if isinstance(probabilities, list):
                    # Multi-class case - extract positive class probabilities
                    probabilities = np.array([
                        p[0, 1] if p.shape[1] > 1 else p[0, 0] 
                        for p in probabilities
                    ])
                else:
                    # Single prediction case
                    if len(probabilities.shape) > 1:
                        probabilities = probabilities[0]
                
                return probabilities
                
            except Exception as e:
                logger.warning(f"Failed to get probabilities: {str(e)}, using binary predictions")
                return predictions.astype(float)
        else:
            # Model doesn't support probabilities
            return predictions.astype(float)
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5, 
                     use_optimal: bool = True) -> List[PredictionResult]:
        """Predict emotions for multiple texts"""
        results = []
        
        for text in texts:
            result = self.predict_single(text, threshold, use_optimal)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.performance_metrics is None:
            self._load_performance_metrics()
        
        info = {
            "name": "Traditional ML",
            "type": "Machine Learning Pipeline",
            "algorithm": self.metadata['best_model']['name'] if self.metadata else "Unknown",
            "features": "TF-IDF + Linguistic Features",
            "optimal_thresholds_available": bool(self.optimal_thresholds),
            "performance": self.performance_metrics.to_dict() if self.performance_metrics else {}
        }
        
        if self.metadata:
            dataset_info = self.metadata.get('dataset_info', {})
            info.update({
                "training_samples": dataset_info.get('train_samples', 'Unknown'),
                "test_samples": dataset_info.get('test_samples', 'Unknown'),
                "num_emotions": dataset_info.get('num_emotions', len(EMOTION_LABELS))
            })
        
        return info
    
    def is_available(self) -> bool:
        """Check if Traditional ML dependencies are available"""
        try:
            import sklearn
            import joblib
            import pickle
            return True
        except ImportError:
            return False
    
    def get_required_packages(self) -> List[str]:
        """Get list of required packages"""
        return ["scikit-learn", "joblib", "numpy", "scipy", "nltk"]

# Global instance
_traditional_ml_loader = None

def get_traditional_ml_loader() -> TraditionalMLModelLoader:
    """Get singleton Traditional ML loader instance"""
    global _traditional_ml_loader
    if _traditional_ml_loader is None:
        _traditional_ml_loader = TraditionalMLModelLoader()
    return _traditional_ml_loader

# Convenience functions
def predict_traditional_ml(text: str, threshold: float = 0.5, use_optimal: bool = True) -> PredictionResult:
    """Convenience function for Traditional ML prediction"""
    loader = get_traditional_ml_loader()
    return loader.predict_single(text, threshold, use_optimal)

def load_traditional_ml_model() -> bool:
    """Convenience function to load Traditional ML model"""
    loader = get_traditional_ml_loader()
    return loader.load_model()

def get_traditional_ml_info() -> Dict[str, Any]:
    """Convenience function to get Traditional ML model info"""
    loader = get_traditional_ml_loader()
    return loader.get_model_info()