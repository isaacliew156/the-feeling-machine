"""
Model Loaders Package for GoEmotions Multi-Model Comparison
"""

import logging

logger = logging.getLogger(__name__)

from .utils import (
    ModelPerformance, 
    PredictionResult, 
    EMOTION_LABELS,
    apply_threshold,
    calculate_model_agreement,
    get_project_root,
    TextPreprocessor,
    create_error_result,
    format_prediction_time,
    get_model_summary_stats
)

from .bert_loader import (
    BERTModelLoader,
    get_bert_loader,
    predict_bert,
    load_bert_model,
    get_bert_info
)

from .embedding_loader import (
    WordEmbeddingModelLoader,
    get_embedding_loader,
    predict_embedding,
    load_embedding_model,
    get_embedding_info
)

try:
    from .traditional_ml_loader import (
        TraditionalMLModelLoader,
        get_traditional_ml_loader,
        predict_traditional_ml,
        load_traditional_ml_model,
        get_traditional_ml_info
    )
    TRADITIONAL_ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Traditional ML loader not available: {str(e)}")
    TRADITIONAL_ML_AVAILABLE = False

try:
    from .ensemble_loader import (
        EnsembleEmotionClassifier,
        get_ensemble_loader,
        predict_ensemble,
        load_ensemble_models,
        get_ensemble_info
    )
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Ensemble loader not available: {str(e)}")
    ENSEMBLE_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Claude Assistant"

__all__ = [
    # Utils
    "ModelPerformance",
    "PredictionResult", 
    "EMOTION_LABELS",
    "apply_threshold",
    "calculate_model_agreement",
    "get_project_root",
    "TextPreprocessor",
    "create_error_result",
    "format_prediction_time",
    "get_model_summary_stats",
    
    # BERT
    "BERTModelLoader",
    "get_bert_loader",
    "predict_bert",
    "load_bert_model",
    "get_bert_info",
    
    # Word Embedding
    "WordEmbeddingModelLoader",
    "get_embedding_loader", 
    "predict_embedding",
    "load_embedding_model",
    "get_embedding_info",
    
    # Traditional ML
    "TraditionalMLModelLoader",
    "get_traditional_ml_loader",
    "predict_traditional_ml",
    "load_traditional_ml_model",
    "get_traditional_ml_info",
    
    # Ensemble
    "EnsembleEmotionClassifier",
    "get_ensemble_loader",
    "predict_ensemble",
    "load_ensemble_models",
    "get_ensemble_info"
]