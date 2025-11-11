"""
Confidence Calibration for Multi-Model Ensemble
Implements Platt Scaling and other calibration methods to normalize prediction confidence
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)

class PlattScaling:
    """
    Platt Scaling calibration for binary classification probabilities
    Maps classifier outputs to calibrated probabilities using sigmoid function
    """
    
    def __init__(self):
        self.calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.is_fitted = False
        
    def fit(self, scores: np.ndarray, true_labels: np.ndarray) -> 'PlattScaling':
        """
        Fit Platt scaling parameters
        
        Args:
            scores: Raw classifier scores (N,)
            true_labels: Binary true labels (N,) 
            
        Returns:
            Self for method chaining
        """
        # Reshape for sklearn
        scores_reshaped = scores.reshape(-1, 1)
        
        # Fit logistic regression: P(y=1|score) = 1/(1 + exp(-(A*score + B)))
        self.calibrator.fit(scores_reshaped, true_labels)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to get calibrated probabilities
        
        Args:
            scores: Raw classifier scores
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            # If not fitted, return sigmoid of raw scores as approximation
            return 1 / (1 + np.exp(-scores))
        
        scores_reshaped = scores.reshape(-1, 1)
        # Get probability of positive class
        calibrated_probs = self.calibrator.predict_proba(scores_reshaped)[:, 1]
        
        return calibrated_probs
    
    def get_parameters(self) -> Tuple[float, float]:
        """Get the fitted A and B parameters"""
        if not self.is_fitted:
            return 1.0, 0.0
        
        A = self.calibrator.coef_[0][0]
        B = self.calibrator.intercept_[0]
        return A, B


class IsotonicCalibration:
    """
    Isotonic regression calibration - non-parametric method
    Good for non-sigmoid-shaped reliability diagrams
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
    
    def fit(self, scores: np.ndarray, true_labels: np.ndarray) -> 'IsotonicCalibration':
        """Fit isotonic regression calibrator"""
        self.calibrator.fit(scores, true_labels)
        self.is_fitted = True
        return self
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration"""
        if not self.is_fitted:
            # Fallback to identity function
            return np.clip(scores, 0, 1)
        
        return self.calibrator.predict(scores)


class MultiLabelCalibrator:
    """
    Calibrator for multi-label emotion classification
    Maintains separate calibrators for each emotion
    """
    
    def __init__(self, emotions: List[str], method: str = 'platt'):
        """
        Initialize multi-label calibrator
        
        Args:
            emotions: List of emotion labels
            method: 'platt' or 'isotonic'
        """
        self.emotions = emotions
        self.method = method
        self.calibrators = {}
        self.is_fitted = False
        
        # Initialize calibrators for each emotion
        for emotion in emotions:
            if method == 'platt':
                self.calibrators[emotion] = PlattScaling()
            elif method == 'isotonic':
                self.calibrators[emotion] = IsotonicCalibration()
            else:
                raise ValueError(f"Unknown calibration method: {method}")
    
    def fit(self, emotion_scores: Dict[str, np.ndarray], 
            true_labels: Dict[str, np.ndarray]) -> 'MultiLabelCalibrator':
        """
        Fit calibrators for all emotions
        
        Args:
            emotion_scores: Dict mapping emotion -> raw scores array
            true_labels: Dict mapping emotion -> binary labels array
        """
        for emotion in self.emotions:
            if emotion in emotion_scores and emotion in true_labels:
                scores = emotion_scores[emotion]
                labels = true_labels[emotion]
                
                # Only fit if we have enough samples and both classes
                if len(scores) >= 10 and len(np.unique(labels)) == 2:
                    try:
                        self.calibrators[emotion].fit(scores, labels)
                        logger.info(f"Fitted calibrator for {emotion}")
                    except Exception as e:
                        logger.warning(f"Failed to fit calibrator for {emotion}: {e}")
                else:
                    logger.warning(f"Insufficient data for {emotion} calibration")
        
        self.is_fitted = True
        return self
    
    def calibrate_scores(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply calibration to emotion scores
        
        Args:
            emotion_scores: Raw emotion scores
            
        Returns:
            Calibrated emotion scores
        """
        calibrated_scores = {}
        
        for emotion, score in emotion_scores.items():
            if emotion in self.calibrators:
                try:
                    calibrated_score = self.calibrators[emotion].predict_proba(np.array([score]))[0]
                    calibrated_scores[emotion] = float(calibrated_score)
                except Exception as e:
                    logger.warning(f"Calibration failed for {emotion}: {e}")
                    calibrated_scores[emotion] = score
            else:
                calibrated_scores[emotion] = score
        
        return calibrated_scores
    
    def save(self, filepath: str):
        """Save calibrator to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'MultiLabelCalibrator':
        """Load calibrator from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ModelSpecificCalibrator:
    """
    Manages calibration for multiple models
    Each model gets its own calibrator for each emotion
    """
    
    def __init__(self, model_names: List[str], emotions: List[str], method: str = 'platt'):
        self.model_names = model_names
        self.emotions = emotions
        self.method = method
        self.model_calibrators = {}
        
        # Initialize calibrator for each model
        for model_name in model_names:
            self.model_calibrators[model_name] = MultiLabelCalibrator(emotions, method)
    
    def fit_model(self, model_name: str, emotion_scores: Dict[str, np.ndarray],
                  true_labels: Dict[str, np.ndarray]):
        """Fit calibrator for a specific model"""
        if model_name in self.model_calibrators:
            self.model_calibrators[model_name].fit(emotion_scores, true_labels)
            logger.info(f"Fitted calibrator for model: {model_name}")
    
    def calibrate_model_scores(self, model_name: str, 
                              emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Calibrate scores for a specific model"""
        if model_name in self.model_calibrators:
            return self.model_calibrators[model_name].calibrate_scores(emotion_scores)
        else:
            logger.warning(f"No calibrator found for model: {model_name}")
            return emotion_scores
    
    def get_calibration_stats(self, model_name: str) -> Dict[str, Any]:
        """Get calibration statistics for a model"""
        if model_name not in self.model_calibrators:
            return {}
        
        stats = {}
        calibrator = self.model_calibrators[model_name]
        
        for emotion in self.emotions:
            if emotion in calibrator.calibrators:
                emotion_calibrator = calibrator.calibrators[emotion]
                if hasattr(emotion_calibrator, 'get_parameters'):
                    A, B = emotion_calibrator.get_parameters()
                    stats[emotion] = {'A': A, 'B': B, 'is_fitted': emotion_calibrator.is_fitted}
                else:
                    stats[emotion] = {'is_fitted': emotion_calibrator.is_fitted}
        
        return stats


class TemperatureScaling:
    """
    Temperature scaling for neural network calibration
    Scales logits by a learned temperature parameter
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, true_labels: np.ndarray) -> 'TemperatureScaling':
        """
        Fit temperature parameter using cross-entropy loss
        
        Args:
            logits: Raw logits from neural network
            true_labels: True binary labels
        """
        from scipy.optimize import minimize_scalar
        
        def negative_log_likelihood(temp):
            scaled_logits = logits / temp
            # Convert to probabilities
            probs = 1 / (1 + np.exp(-scaled_logits))
            # Avoid log(0)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            # Negative log likelihood
            nll = -np.mean(true_labels * np.log(probs) + (1 - true_labels) * np.log(1 - probs))
            return nll
        
        # Optimize temperature
        result = minimize_scalar(negative_log_likelihood, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        
        logger.info(f"Fitted temperature scaling with T={self.temperature:.3f}")
        return self
    
    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits"""
        if not self.is_fitted:
            return logits
        
        return logits / self.temperature


def create_default_calibrators(project_root: str) -> ModelSpecificCalibrator:
    """
    Create default calibrators for the three main models
    Returns pre-configured calibrator that can be fitted with validation data
    """
    from .utils import EMOTION_LABELS
    
    model_names = ['BERT', 'CNN + GloVe', 'Traditional ML']
    calibrator = ModelSpecificCalibrator(model_names, EMOTION_LABELS, method='platt')
    
    # Try to load existing calibrators
    calibrator_path = os.path.join(project_root, 'models', 'calibrators.pkl')
    if os.path.exists(calibrator_path):
        try:
            loaded_calibrator = ModelSpecificCalibrator.load(calibrator_path)
            logger.info("Loaded existing calibrators")
            return loaded_calibrator
        except Exception as e:
            logger.warning(f"Failed to load existing calibrators: {e}")
    
    logger.info("Created new default calibrators")
    return calibrator


def analyze_calibration_quality(predicted_probs: np.ndarray, 
                               true_labels: np.ndarray, 
                               n_bins: int = 10) -> Dict[str, float]:
    """
    Analyze calibration quality using reliability diagrams
    
    Args:
        predicted_probs: Predicted probabilities
        true_labels: True binary labels
        n_bins: Number of bins for reliability diagram
        
    Returns:
        Dictionary with calibration metrics
    """
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0  # Maximum Calibration Error
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            
            # Calibration error for this bin
            bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += bin_error * prop_in_bin
            mce = max(mce, bin_error)
    
    # Brier Score
    brier_score = np.mean((predicted_probs - true_labels) ** 2)
    
    return {
        'ece': ece,           # Expected Calibration Error
        'mce': mce,           # Maximum Calibration Error  
        'brier_score': brier_score  # Brier Score
    }