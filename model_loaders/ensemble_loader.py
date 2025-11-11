"""
Ensemble Emotion Classifier for GoEmotions Multi-Model System
Combines BERT, CNN+GloVe, and Traditional ML models using advanced ensemble strategies
"""

import numpy as np
import re
import time
import logging
import emoji
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import statistics

# Import our model loaders and utilities
from .bert_loader import get_bert_loader
from .embedding_loader import get_embedding_loader  
from .traditional_ml_loader import get_traditional_ml_loader
from .utils import (
    PredictionResult, EMOTION_LABELS, ModelPerformance,
    apply_threshold, create_error_result, format_prediction_time,
    TextPreprocessor, logger
)
from .emotion_hierarchy import emotion_hierarchy
from .calibration import ModelSpecificCalibrator, create_default_calibrators

class EnsembleEmotionClassifier:
    """
    Advanced ensemble classifier combining multiple emotion recognition models
    
    Features:
    - Adaptive Cascade: Routes texts based on characteristics
    - Multiple ensemble strategies: voting, weighted, stacking, hierarchical
    - Confidence calibration: Platt scaling for score normalization
    - Emotion hierarchy: Leverages psychological emotion relationships
    - Edge case handling: Robust to various input types
    - Performance optimization: Caching, parallel processing, early stopping
    """
    
    def __init__(self, models_to_load: List[str] = None):
        """
        Initialize ensemble classifier
        
        Args:
            models_to_load: List of models to load ['BERT', 'CNN + GloVe', 'Traditional ML']
                           If None, loads all available models
        """
        self.available_models = models_to_load or ['BERT', 'CNN + GloVe', 'Traditional ML']
        self.loaded_models = {}
        self.model_loaders = {}
        self.model_performance = {}
        
        # Ensemble configuration
        self.ensemble_strategies = [
            'majority_voting', 'weighted_average', 'adaptive_cascade', 
            'stacking', 'hierarchical', 'confidence_weighted'
        ]
        self.default_strategy = 'adaptive_cascade'
        
        # Calibration
        self.calibrator = None
        self.use_calibration = True
        
        # Cascade configuration
        self.cascade_config = {
            'short_text_threshold': 15,      # Words
            'high_confidence_threshold': 0.8, # Early stopping threshold
            'negation_words': {
                'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody', 
                'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'seldom',
                "n't", "don't", "won't", "can't", "shouldn't", "wouldn't"
            },
            'sarcasm_markers': {
                'totally', 'absolutely', 'fantastic', 'brilliant', 'wonderful',
                'amazing', 'perfect', 'great', 'awesome', 'incredible',
                'obviously', 'clearly', 'sure', 'right', 'yeah right'
            }
        }
        
        # Performance tracking
        self.prediction_cache = {}
        self.performance_stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'average_latency': [],
            'strategy_usage': {strategy: 0 for strategy in self.ensemble_strategies}
        }
        
        # Model weights (will be updated based on performance)
        self.model_weights = {
            'BERT': 0.4,           # Best F1 macro
            'CNN + GloVe': 0.35,   # Good balance
            'Traditional ML': 0.25  # Fastest, good for simple cases
        }
        
        logger.info(f"Initialized EnsembleEmotionClassifier with models: {self.available_models}")
    
    def load_models(self) -> Dict[str, bool]:
        """
        Load specified models and return loading status
        
        Returns:
            Dictionary mapping model names to loading success status
        """
        loading_status = {}
        
        for model_name in self.available_models:
            try:
                start_time = time.time()
                
                if model_name == 'BERT':
                    loader = get_bert_loader()
                    success = loader.load_model()
                    if success:
                        self.model_loaders['BERT'] = loader
                        self.model_performance['BERT'] = loader.get_model_info().get('performance', {})
                    
                elif model_name == 'CNN + GloVe':
                    loader = get_embedding_loader()
                    success = loader.load_model()
                    if success:
                        self.model_loaders['CNN + GloVe'] = loader
                        self.model_performance['CNN + GloVe'] = loader.get_model_info().get('performance', {})
                    
                elif model_name == 'Traditional ML':
                    loader = get_traditional_ml_loader()
                    success = loader.load_model()
                    if success:
                        self.model_loaders['Traditional ML'] = loader
                        self.model_performance['Traditional ML'] = loader.get_model_info().get('performance', {})
                
                loading_time = time.time() - start_time
                loading_status[model_name] = success
                
                if success:
                    self.loaded_models[model_name] = True
                    logger.info(f"✅ Loaded {model_name} in {loading_time:.2f}s")
                else:
                    logger.error(f"❌ Failed to load {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading {model_name}: {str(e)}")
                loading_status[model_name] = False
        
        # Initialize calibrator after models are loaded
        if self.loaded_models:
            self._initialize_calibrator()
        
        return loading_status
    
    def _initialize_calibrator(self):
        """Initialize calibration system"""
        try:
            from .utils import get_project_root
            project_root = get_project_root()
            self.calibrator = create_default_calibrators(project_root)
            logger.info("Initialized calibration system")
        except Exception as e:
            logger.warning(f"Failed to initialize calibrator: {e}")
            self.use_calibration = False
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded"""
        return self.loaded_models.get(model_name, False)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of successfully loaded models"""
        return [model for model, loaded in self.loaded_models.items() if loaded]
    
    @lru_cache(maxsize=100)
    def _analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """
        Analyze text characteristics for adaptive routing
        Uses LRU cache for performance
        """
        # Basic metrics
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Pattern detection
        has_negation = any(neg_word in text.lower() for neg_word in self.cascade_config['negation_words'])
        has_sarcasm_markers = any(marker in text.lower() for marker in self.cascade_config['sarcasm_markers'])
        
        # Emoji analysis
        emoji_count = len([c for c in text if c in emoji.EMOJI_DATA])
        
        # Complexity indicators
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_word_ratio = len(set(words)) / max(len(words), 1)
        
        # Language detection (simple heuristic)
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        is_likely_english = english_chars / max(len([c for c in text if c.isalpha()]), 1) > 0.8
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_ratio': caps_ratio,
            'has_negation': has_negation,
            'has_sarcasm_markers': has_sarcasm_markers,
            'emoji_count': emoji_count,
            'avg_word_length': avg_word_length,
            'unique_word_ratio': unique_word_ratio,
            'is_likely_english': is_likely_english,
            'is_short': word_count < self.cascade_config['short_text_threshold'],
            'is_complex': avg_word_length > 6 or unique_word_ratio < 0.5,
            'has_strong_emotion_markers': exclamation_count > 1 or caps_ratio > 0.3
        }
    
    def _predict_single_model(self, model_name: str, text: str, 
                             threshold: float = 0.5, use_optimal: bool = True) -> Optional[PredictionResult]:
        """
        Predict using a single model with error handling
        """
        try:
            if model_name not in self.model_loaders:
                return None
            
            loader = self.model_loaders[model_name]
            result = loader.predict_single(text, threshold, use_optimal)
            
            # Apply calibration if available
            if self.use_calibration and self.calibrator:
                calibrated_scores = self.calibrator.calibrate_model_scores(
                    model_name, result.emotion_scores
                )
                
                # Recalculate predicted emotions with calibrated scores
                predicted_emotions = apply_threshold(
                    calibrated_scores, threshold, 
                    getattr(loader, 'optimal_thresholds', None), use_optimal
                )
                
                # Create new result with calibrated scores
                result = PredictionResult(
                    model_name=result.model_name,
                    emotion_scores=calibrated_scores,
                    predicted_emotions=predicted_emotions,
                    prediction_time=result.prediction_time
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {str(e)}")
            return create_error_result(model_name, str(e))
    
    def _adaptive_cascade_strategy(self, text: str, threshold: float = 0.5, 
                                  use_optimal: bool = True) -> PredictionResult:
        """
        Adaptive cascade ensemble strategy
        Routes texts based on characteristics for optimal speed/accuracy trade-off
        """
        start_time = time.time()
        characteristics = self._analyze_text_characteristics(text)
        
        # Strategy 1: Fast path for simple, short texts without complexity
        if (characteristics['is_short'] and 
            not characteristics['has_negation'] and 
            not characteristics['has_sarcasm_markers'] and
            characteristics['is_likely_english'] and
            'Traditional ML' in self.loaded_models):
            
            result = self._predict_single_model('Traditional ML', text, threshold, use_optimal)
            if result and result.get_confidence() > self.cascade_config['high_confidence_threshold']:
                prediction_time = time.time() - start_time
                result.prediction_time = prediction_time
                logger.debug(f"Cascade: Fast path used (Traditional ML), confidence: {result.get_confidence():.3f}")
                return result
        
        # Strategy 2: BERT for sarcasm, negation, or complex language
        if (characteristics['has_sarcasm_markers'] or 
            characteristics['has_negation'] or
            characteristics['is_complex']) and 'BERT' in self.loaded_models:
            
            result = self._predict_single_model('BERT', text, threshold, use_optimal)
            if result and result.get_confidence() > 0.7:  # Slightly lower threshold for BERT
                prediction_time = time.time() - start_time
                result.prediction_time = prediction_time
                logger.debug(f"Cascade: BERT priority used, confidence: {result.get_confidence():.3f}")
                return result
        
        # Strategy 3: Full ensemble for everything else
        return self._weighted_average_strategy(text, threshold, use_optimal, cascade_mode=True)
    
    def _weighted_average_strategy(self, text: str, threshold: float = 0.5, 
                                  use_optimal: bool = True, cascade_mode: bool = False) -> PredictionResult:
        """
        Weighted average ensemble strategy
        Combines predictions using model performance-based weights
        """
        start_time = time.time()
        model_results = {}
        
        # Get predictions from all loaded models
        if cascade_mode:
            # In cascade mode, use parallel execution for remaining models
            available_models = [m for m in self.loaded_models.keys() if self.loaded_models[m]]
        else:
            available_models = [m for m in self.loaded_models.keys() if self.loaded_models[m]]
        
        # Parallel prediction execution
        with ThreadPoolExecutor(max_workers=min(3, len(available_models))) as executor:
            future_to_model = {
                executor.submit(self._predict_single_model, model, text, threshold, use_optimal): model
                for model in available_models
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        model_results[model_name] = result
                except Exception as e:
                    logger.error(f"Parallel prediction failed for {model_name}: {e}")
        
        if not model_results:
            return create_error_result("Ensemble", "All model predictions failed")
        
        # Combine scores using weighted average
        combined_scores = {}
        total_weight = 0
        
        for emotion in EMOTION_LABELS:
            weighted_sum = 0
            weight_sum = 0
            
            for model_name, result in model_results.items():
                if emotion in result.emotion_scores:
                    weight = self.model_weights.get(model_name, 1.0)
                    weighted_sum += result.emotion_scores[emotion] * weight
                    weight_sum += weight
            
            combined_scores[emotion] = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Apply emotion hierarchy correlations
        combined_scores = emotion_hierarchy.apply_correlation_boost(combined_scores, boost_factor=0.05)
        combined_scores = emotion_hierarchy.apply_conflict_suppression(combined_scores, suppression_factor=0.1)
        
        # Get predicted emotions
        predicted_emotions = apply_threshold(combined_scores, threshold)
        
        prediction_time = time.time() - start_time
        
        # Create ensemble result
        return PredictionResult(
            model_name="Ensemble (Weighted)",
            emotion_scores=combined_scores,
            predicted_emotions=predicted_emotions,
            prediction_time=prediction_time
        )
    
    def _majority_voting_strategy(self, text: str, threshold: float = 0.5, 
                                 use_optimal: bool = True) -> PredictionResult:
        """
        Majority voting ensemble strategy
        Each model votes for emotions above threshold, majority wins
        """
        start_time = time.time()
        model_results = {}
        
        # Get predictions from all models
        for model_name in self.loaded_models:
            if self.loaded_models[model_name]:
                result = self._predict_single_model(model_name, text, threshold, use_optimal)
                if result:
                    model_results[model_name] = result
        
        if not model_results:
            return create_error_result("Ensemble", "All model predictions failed")
        
        # Count votes for each emotion
        emotion_votes = {emotion: 0 for emotion in EMOTION_LABELS}
        emotion_scores_sum = {emotion: 0.0 for emotion in EMOTION_LABELS}
        
        for result in model_results.values():
            # Vote based on predicted emotions
            for emotion in result.predicted_emotions:
                emotion_votes[emotion] += 1
            
            # Sum scores for averaging
            for emotion, score in result.emotion_scores.items():
                emotion_scores_sum[emotion] += score
        
        # Determine majority threshold
        num_models = len(model_results)
        majority_threshold = num_models // 2 + 1
        
        # Select emotions with majority votes
        predicted_emotions = [
            emotion for emotion, votes in emotion_votes.items() 
            if votes >= majority_threshold
        ]
        
        # Average scores
        avg_scores = {
            emotion: score_sum / num_models 
            for emotion, score_sum in emotion_scores_sum.items()
        }
        
        prediction_time = time.time() - start_time
        
        return PredictionResult(
            model_name="Ensemble (Majority)",
            emotion_scores=avg_scores,
            predicted_emotions=predicted_emotions,
            prediction_time=prediction_time
        )
    
    def _confidence_weighted_strategy(self, text: str, threshold: float = 0.5, 
                                     use_optimal: bool = True) -> PredictionResult:
        """
        Confidence-weighted ensemble strategy
        Weights models based on their confidence for this specific prediction
        """
        start_time = time.time()
        model_results = {}
        
        # Get predictions from all models
        for model_name in self.loaded_models:
            if self.loaded_models[model_name]:
                result = self._predict_single_model(model_name, text, threshold, use_optimal)
                if result:
                    model_results[model_name] = result
        
        if not model_results:
            return create_error_result("Ensemble", "All model predictions failed")
        
        # Calculate dynamic weights based on confidence
        model_confidences = {}
        for model_name, result in model_results.items():
            confidence = result.get_confidence()
            model_confidences[model_name] = confidence
        
        # Normalize confidences to weights
        total_confidence = sum(model_confidences.values())
        if total_confidence == 0:
            # Fallback to equal weights
            weights = {model: 1.0 / len(model_results) for model in model_results}
        else:
            weights = {model: conf / total_confidence for model, conf in model_confidences.items()}
        
        # Combine scores using confidence weights
        combined_scores = {}
        for emotion in EMOTION_LABELS:
            weighted_sum = 0
            for model_name, result in model_results.items():
                if emotion in result.emotion_scores:
                    weighted_sum += result.emotion_scores[emotion] * weights[model_name]
            combined_scores[emotion] = weighted_sum
        
        # Get predicted emotions
        predicted_emotions = apply_threshold(combined_scores, threshold)
        
        prediction_time = time.time() - start_time
        
        return PredictionResult(
            model_name="Ensemble (Confidence)",
            emotion_scores=combined_scores,
            predicted_emotions=predicted_emotions,
            prediction_time=prediction_time
        )
    
    def _hierarchical_strategy(self, text: str, threshold: float = 0.5, 
                              use_optimal: bool = True) -> PredictionResult:
        """
        Hierarchical ensemble strategy
        First determines emotion category, then specific emotions within that category
        """
        start_time = time.time()
        
        # Get predictions from all models
        model_results = {}
        for model_name in self.loaded_models:
            if self.loaded_models[model_name]:
                result = self._predict_single_model(model_name, text, threshold, use_optimal)
                if result:
                    model_results[model_name] = result
        
        if not model_results:
            return create_error_result("Ensemble", "All model predictions failed")
        
        # Step 1: Determine dominant emotion category
        category_scores = {}
        for result in model_results.values():
            cat_dist = emotion_hierarchy.get_category_distribution(result.emotion_scores)
            for category, score in cat_dist.items():
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)
        
        # Average category scores
        avg_category_scores = {
            cat: np.mean(scores) for cat, scores in category_scores.items()
        }
        
        # Find dominant category
        dominant_category = max(avg_category_scores.items(), key=lambda x: x[1])[0]
        
        # Step 2: Focus on emotions in dominant category
        category_emotions = []
        for group_emotions in emotion_hierarchy.emotion_hierarchy[dominant_category].values():
            category_emotions.extend(group_emotions)
        
        # Combine scores with focus on dominant category
        combined_scores = {}
        for emotion in EMOTION_LABELS:
            scores = [result.emotion_scores.get(emotion, 0) for result in model_results.values()]
            avg_score = np.mean(scores)
            
            # Boost emotions in dominant category
            if emotion in category_emotions:
                avg_score *= 1.2  # 20% boost
            
            combined_scores[emotion] = min(avg_score, 1.0)  # Cap at 1.0
        
        # Get predicted emotions
        predicted_emotions = apply_threshold(combined_scores, threshold)
        
        prediction_time = time.time() - start_time
        
        return PredictionResult(
            model_name=f"Ensemble (Hierarchical-{dominant_category})",
            emotion_scores=combined_scores,
            predicted_emotions=predicted_emotions,
            prediction_time=prediction_time
        )
    
    def predict(self, text: str, strategy: str = None, threshold: float = 0.5, 
                use_optimal: bool = True, use_cache: bool = True) -> PredictionResult:
        """
        Main prediction method with ensemble strategy selection
        
        Args:
            text: Input text to analyze
            strategy: Ensemble strategy to use
            threshold: Emotion threshold
            use_optimal: Use model-specific optimal thresholds
            use_cache: Enable prediction caching
            
        Returns:
            PredictionResult with ensemble prediction
        """
        # Input validation and edge case handling
        if not self._validate_input(text):
            return self._handle_edge_case(text)
        
        # Check cache first
        cache_key = f"{text}_{strategy}_{threshold}_{use_optimal}"
        if use_cache and cache_key in self.prediction_cache:
            self.performance_stats['cache_hits'] += 1
            return self.prediction_cache[cache_key]
        
        # Select strategy
        strategy = strategy or self.default_strategy
        if strategy not in self.ensemble_strategies:
            logger.warning(f"Unknown strategy '{strategy}', falling back to '{self.default_strategy}'")
            strategy = self.default_strategy
        
        # Execute prediction
        start_time = time.time()
        
        try:
            if strategy == 'adaptive_cascade':
                result = self._adaptive_cascade_strategy(text, threshold, use_optimal)
            elif strategy == 'weighted_average':
                result = self._weighted_average_strategy(text, threshold, use_optimal)
            elif strategy == 'majority_voting':
                result = self._majority_voting_strategy(text, threshold, use_optimal)
            elif strategy == 'confidence_weighted':
                result = self._confidence_weighted_strategy(text, threshold, use_optimal)
            elif strategy == 'hierarchical':
                result = self._hierarchical_strategy(text, threshold, use_optimal)
            else:
                # Fallback to weighted average
                result = self._weighted_average_strategy(text, threshold, use_optimal)
            
            # Update performance stats
            prediction_time = time.time() - start_time
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['average_latency'].append(prediction_time)
            self.performance_stats['strategy_usage'][strategy] += 1
            
            # Cache result
            if use_cache:
                self.prediction_cache[cache_key] = result
                # Keep cache size manageable
                if len(self.prediction_cache) > 100:
                    # Remove oldest entry
                    oldest_key = next(iter(self.prediction_cache))
                    del self.prediction_cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            return create_error_result("Ensemble", f"Prediction failed: {str(e)}")
    
    def _validate_input(self, text: str) -> bool:
        """Validate input text"""
        if not isinstance(text, str):
            return False
        if len(text.strip()) == 0:
            return False
        if len(text) > 10000:  # Very long text
            return False
        return True
    
    def _handle_edge_case(self, text: str) -> PredictionResult:
        """Handle edge cases like empty text, very long text, etc."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            # Empty text -> neutral
            scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
            scores['neutral'] = 0.8
            return PredictionResult(
                model_name="Ensemble (Edge-Case)",
                emotion_scores=scores,
                predicted_emotions=['neutral'],
                prediction_time=0.001
            )
        
        if len(text) > 10000:
            # Very long text -> process first 1000 characters
            truncated_text = text[:1000]
            logger.warning(f"Text too long ({len(text)} chars), truncated to 1000 chars")
            return self.predict(truncated_text, use_cache=False)
        
        # Default fallback
        return create_error_result("Ensemble", "Invalid input")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble system information"""
        return {
            "name": "Ensemble Emotion Classifier",
            "type": "Multi-Model Ensemble",
            "loaded_models": list(self.loaded_models.keys()),
            "available_strategies": self.ensemble_strategies,
            "default_strategy": self.default_strategy,
            "model_weights": self.model_weights,
            "performance_stats": self.performance_stats.copy(),
            "calibration_enabled": self.use_calibration,
            "cache_size": len(self.prediction_cache)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats['average_latency']:
            stats['avg_latency_ms'] = np.mean(stats['average_latency']) * 1000
            stats['median_latency_ms'] = np.median(stats['average_latency']) * 1000
            stats['p95_latency_ms'] = np.percentile(stats['average_latency'], 95) * 1000
        
        if stats['total_predictions'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_predictions']
        
        return stats
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """Update model weights for ensemble"""
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.model_weights.update({
                model: weight / total_weight 
                for model, weight in new_weights.items()
            })
            logger.info(f"Updated model weights: {self.model_weights}")


# Global ensemble instance
_ensemble_loader = None

def get_ensemble_loader() -> EnsembleEmotionClassifier:
    """Get singleton ensemble loader instance"""
    global _ensemble_loader
    if _ensemble_loader is None:
        _ensemble_loader = EnsembleEmotionClassifier()
    return _ensemble_loader

def predict_ensemble(text: str, strategy: str = 'adaptive_cascade', 
                    threshold: float = 0.5, use_optimal: bool = True) -> PredictionResult:
    """Convenience function for ensemble prediction"""
    loader = get_ensemble_loader()
    return loader.predict(text, strategy, threshold, use_optimal)

def load_ensemble_models(models: List[str] = None) -> Dict[str, bool]:
    """Convenience function to load ensemble models"""
    loader = get_ensemble_loader()
    if models:
        loader.available_models = models
    return loader.load_models()

def get_ensemble_info() -> Dict[str, Any]:
    """Convenience function to get ensemble info"""
    loader = get_ensemble_loader()
    return loader.get_model_info()