"""
BERT Model Loader for GoEmotions Classification
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Handle optional imports
try:
    import streamlit as st
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# Handle optional imports
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .utils import (
    ModelPerformance, PredictionResult, EMOTION_LABELS, 
    get_project_root, apply_threshold, create_error_result,
    TextPreprocessor, logger
)

class BERTModelLoader:
    """BERT model loader and predictor"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 128
        self.model_path = None
        self.performance_metrics = None
        
    def load_model(_self) -> bool:
        """Load BERT model and tokenizer with caching"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch and transformers not available for BERT model")
            if ST_AVAILABLE:
                st.error("PyTorch and transformers are required for BERT model but not installed")
            return False
            
        try:
            project_root = get_project_root()
            model_dir = os.path.join(project_root, "models", "bert")
            
            # Check if model files exist
            model_file = os.path.join(model_dir, "best_bert_model.pt")
            tokenizer_dir = os.path.join(model_dir, "tokenizer")
            
            if not os.path.exists(model_file):
                logger.error(f"BERT model file not found: {model_file}")
                return False
                
            if not os.path.exists(tokenizer_dir):
                logger.error(f"BERT tokenizer directory not found: {tokenizer_dir}")
                return False
            
            # Load tokenizer
            logger.info("Loading BERT tokenizer...")
            _self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
            logger.info("BERT tokenizer loaded successfully")
            
            # Load model
            logger.info("Loading BERT model...")
            
            # Initialize model with correct architecture
            _self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=len(EMOTION_LABELS),
                problem_type="multi_label_classification"
            )
            
            # Load trained weights
            checkpoint = torch.load(model_file, map_location=_self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint contains training info (epoch, optimizer, etc.)
                model_state_dict = checkpoint['model_state_dict']
                logger.info(f"Loading from checkpoint with epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                # Direct model state dict
                model_state_dict = checkpoint
            
            # Fix missing position_ids for compatibility
            if 'bert.embeddings.position_ids' not in model_state_dict:
                logger.info("Adding missing position_ids for compatibility")
                # Create position_ids tensor with the standard size (512)
                model_state_dict['bert.embeddings.position_ids'] = torch.arange(512).expand((1, -1))
            
            _self.model.load_state_dict(model_state_dict)
            _self.model.to(_self.device)
            _self.model.eval()
            
            logger.info(f"BERT model loaded successfully on device: {_self.device}")
            
            # Load performance metrics
            _self._load_performance_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {str(e)}")
            if ST_AVAILABLE:
                st.error(f"Failed to load BERT model: {str(e)}")
            return False
    
    def _load_performance_metrics(self):
        """Load performance metrics from results file"""
        try:
            project_root = get_project_root()
            results_file = os.path.join(project_root, "results", "bert_results", "bert_final_results.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract optimized performance metrics
                perf = results.get('test_performance', {}).get('optimized_threshold', {})
                
                self.performance_metrics = ModelPerformance(
                    model_name="BERT",
                    f1_macro=perf.get('f1_macro', 0.373),
                    f1_micro=results.get('test_performance', {}).get('default_threshold', {}).get('f1_micro', 0.392),
                    precision=perf.get('precision', 0.332),
                    recall=perf.get('recall', 0.475),
                    training_time=None  # Not available in results
                )
                
                # Store optimal threshold
                self.optimal_threshold = perf.get('threshold', 0.65)
                
                logger.info(f"BERT performance metrics loaded: F1-Macro={self.performance_metrics.f1_macro:.4f}")
            else:
                logger.warning(f"BERT results file not found: {results_file}")
                # Use default metrics
                self.performance_metrics = ModelPerformance(
                    model_name="BERT",
                    f1_macro=0.373,
                    f1_micro=0.392,
                    precision=0.332,
                    recall=0.475
                )
                self.optimal_threshold = 0.65
                
        except Exception as e:
            logger.error(f"Failed to load BERT performance metrics: {str(e)}")
            # Use default metrics
            self.performance_metrics = ModelPerformance(
                model_name="BERT",
                f1_macro=0.373,
                f1_micro=0.392,
                precision=0.332,
                recall=0.475
            )
            self.optimal_threshold = 0.65
    
    def preprocess_text(self, text: str):
        """Preprocess text for BERT model"""
        if not TORCH_AVAILABLE or not TextPreprocessor.validate_input(text):
            return None
        
        # Clean and prepare text
        cleaned_text = TextPreprocessor.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict_single(self, text: str, threshold: float = 0.5, 
                      use_optimal: bool = True) -> PredictionResult:
        """Predict emotions for single text"""
        start_time = time.time()
        
        try:
            # Check if model is loaded
            if self.model is None or self.tokenizer is None:
                if not self.load_model():
                    return create_error_result("BERT", "Model not loaded")
            
            # Preprocess text
            inputs = self.preprocess_text(text)
            if inputs is None:
                return create_error_result("BERT", "Invalid input text")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Create emotion scores dictionary
            emotion_scores = {
                emotion: float(prob) 
                for emotion, prob in zip(EMOTION_LABELS, probabilities)
            }
            
            # Apply thresholds
            optimal_thresholds = {emotion: self.optimal_threshold for emotion in EMOTION_LABELS}
            predicted_emotions = apply_threshold(
                emotion_scores, threshold, optimal_thresholds, use_optimal
            )
            
            prediction_time = time.time() - start_time
            
            return PredictionResult(
                model_name="BERT",
                emotion_scores=emotion_scores,
                predicted_emotions=predicted_emotions,
                prediction_time=prediction_time
            )
            
        except Exception as e:
            logger.error(f"BERT prediction failed: {str(e)}")
            return create_error_result("BERT", f"Prediction failed: {str(e)}")
    
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
        
        return {
            "name": "BERT (bert-base-uncased)",
            "type": "Transformer (BERT)",
            "parameters": "110M",
            "max_sequence_length": self.max_length,
            "device": str(self.device),
            "optimal_threshold": getattr(self, 'optimal_threshold', 0.65),
            "performance": self.performance_metrics.to_dict() if self.performance_metrics else {}
        }
    
    def is_available(self) -> bool:
        """Check if BERT dependencies are available"""
        return TORCH_AVAILABLE
    
    def get_required_packages(self) -> List[str]:
        """Get list of required packages"""
        return ["torch", "transformers", "numpy"]

# Global instance
_bert_loader = None

def get_bert_loader() -> BERTModelLoader:
    """Get singleton BERT loader instance"""
    global _bert_loader
    if _bert_loader is None:
        _bert_loader = BERTModelLoader()
    return _bert_loader

# Convenience functions
def predict_bert(text: str, threshold: float = 0.5, use_optimal: bool = True) -> PredictionResult:
    """Convenience function for BERT prediction"""
    loader = get_bert_loader()
    return loader.predict_single(text, threshold, use_optimal)

def load_bert_model() -> bool:
    """Convenience function to load BERT model"""
    loader = get_bert_loader()
    return loader.load_model()

def get_bert_info() -> Dict[str, Any]:
    """Convenience function to get BERT model info"""
    loader = get_bert_loader()
    return loader.get_model_info()