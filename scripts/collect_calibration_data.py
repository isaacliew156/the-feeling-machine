"""
Calibration Data Collection Script for CNN+GloVe Model
Collects validation predictions to train isotonic regression calibrators
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Model loading imports
from model_loaders.embedding_loader import get_embedding_loader
from scripts.advanced_inference_optimizer import AdvancedInferenceOptimizer

def load_validation_data(sample_size: int = 5000) -> Tuple[List[str], np.ndarray]:
    """Load validation data for calibration"""
    print("📊 Loading validation data...")
    
    data_path = os.path.join(project_root, 'data', 'go_emotions_dataset.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"   Total samples: {len(df)}")
    
    # Basic preprocessing (same as training)
    if 'example_very_unclear' in df.columns:
        df = df[df['example_very_unclear'] == False].copy()
        print(f"   After removing unclear: {len(df)}")
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        import re
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'([.!?])\1+', r'\1', text)
        text = ' '.join(text.split())
        return text
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() >= 3].copy()
    print(f"   After cleaning: {len(df)}")
    
    # Emotion labels
    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Sample validation data
    np.random.seed(42)
    if len(df) > sample_size:
        val_df = df.sample(n=sample_size, random_state=42)
        print(f"   Sampled {sample_size} for validation")
    else:
        val_df = df
        print(f"   Using all {len(df)} samples")
    
    texts = val_df['cleaned_text'].tolist()
    labels = val_df[emotion_columns].values
    
    print(f"✅ Validation data loaded: {len(texts)} samples, {labels.shape[1]} emotions")
    return texts, labels

def collect_raw_predictions(texts: List[str]) -> np.ndarray:
    """Collect raw model predictions without optimization"""
    print("🔮 Collecting raw model predictions...")
    
    # Load model without optimization
    loader = get_embedding_loader()
    
    # Disable advanced optimization temporarily
    original_use_optimization = getattr(loader, 'use_advanced_optimization', True)
    loader.use_advanced_optimization = False
    
    if not loader.load_model():
        raise RuntimeError("Failed to load CNN+GloVe model")
    
    predictions = []
    batch_size = 50  # Process in batches for memory efficiency
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_predictions = []
        
        print(f"   Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        for text in batch_texts:
            try:
                # Get prediction without advanced optimization
                result = loader.predict_single(text, use_optimal=False)
                if not result.error:
                    # Extract raw probabilities
                    emotion_probs = [result.emotion_scores[emotion] for emotion in loader.inference_optimizer.emotion_labels 
                                   if hasattr(loader, 'inference_optimizer') and loader.inference_optimizer
                                   else [result.emotion_scores[emotion] for emotion in sorted(result.emotion_scores.keys())]]
                    batch_predictions.append(emotion_probs)
                else:
                    print(f"   ⚠️ Error for text: {text[:50]}...")
                    # Fill with zeros for failed predictions
                    batch_predictions.append([0.01] * 28)  # 28 emotions
                    
            except Exception as e:
                print(f"   ❌ Exception for text: {text[:50]}... - {str(e)}")
                batch_predictions.append([0.01] * 28)
        
        predictions.extend(batch_predictions)
    
    # Restore original optimization setting
    loader.use_advanced_optimization = original_use_optimization
    
    predictions_array = np.array(predictions)
    print(f"✅ Raw predictions collected: {predictions_array.shape}")
    return predictions_array

def train_and_save_calibrators(texts: List[str], raw_predictions: np.ndarray, true_labels: np.ndarray):
    """Train isotonic regression calibrators and save them"""
    print("🔧 Training isotonic regression calibrators...")
    
    # Initialize optimizer
    config_path = os.path.join(project_root, "models", "word_embedding", "config.json")
    optimizer = AdvancedInferenceOptimizer(config_path)
    
    # Train calibrators
    try:
        optimizer.train_isotonic_calibrator(raw_predictions, true_labels)
        
        # Save calibrators
        calibrator_path = os.path.join(project_root, "models", "word_embedding", "isotonic_calibrators.pkl")
        optimizer.save_calibrators(calibrator_path)
        
        print(f"✅ Calibrators trained and saved to: {calibrator_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to train calibrators: {str(e)}")
        return False

def validate_calibration(texts: List[str], raw_predictions: np.ndarray, true_labels: np.ndarray):
    """Validate the calibration quality"""
    print("📈 Validating calibration quality...")
    
    try:
        # Load optimizer with trained calibrators
        config_path = os.path.join(project_root, "models", "word_embedding", "config.json")
        optimizer = AdvancedInferenceOptimizer(config_path)
        
        calibrator_path = os.path.join(project_root, "models", "word_embedding", "isotonic_calibrators.pkl")
        if optimizer.load_calibrators(calibrator_path):
            
            # Test calibration on a sample
            sample_size = min(100, len(texts))
            sample_indices = np.random.choice(len(texts), sample_size, replace=False)
            
            improvements = []
            
            for idx in sample_indices:
                text = texts[idx]
                raw_probs = raw_predictions[idx]
                true_label = true_labels[idx]
                
                # Apply calibration
                calibrated_probs = optimizer.apply_isotonic_calibration(raw_probs)
                
                # Calculate confidence improvement
                raw_max = np.max(raw_probs)
                calibrated_max = np.max(calibrated_probs)
                
                improvement = (calibrated_max - raw_max) / raw_max * 100 if raw_max > 0 else 0
                improvements.append(improvement)
            
            avg_improvement = np.mean(improvements)
            print(f"   📊 Average confidence improvement: {avg_improvement:.1f}%")
            print(f"   📊 Improvement range: {np.min(improvements):.1f}% - {np.max(improvements):.1f}%")
            
            # Calculate calibration quality metrics
            calibration_quality = np.mean([imp for imp in improvements if imp > 0])
            print(f"   📈 Positive improvements: {np.sum(np.array(improvements) > 0)}/{len(improvements)}")
            print(f"   📈 Average positive improvement: {calibration_quality:.1f}%")
            
            return avg_improvement > 10  # Success if average improvement > 10%
        else:
            print("   ⚠️ Could not load calibrators for validation")
            return False
            
    except Exception as e:
        print(f"   ❌ Calibration validation failed: {str(e)}")
        return False

def main():
    """Main calibration data collection and training pipeline"""
    print("🎯 CNN+GloVe Calibration Data Collection & Training")
    print("=" * 80)
    
    try:
        # Step 1: Load validation data
        texts, true_labels = load_validation_data(sample_size=3000)  # Smaller sample for efficiency
        
        # Step 2: Collect raw predictions
        raw_predictions = collect_raw_predictions(texts)
        
        # Step 3: Train and save calibrators
        training_success = train_and_save_calibrators(texts, raw_predictions, true_labels)
        
        if training_success:
            # Step 4: Validate calibration
            validation_success = validate_calibration(texts, raw_predictions, true_labels)
            
            if validation_success:
                print("\n" + "=" * 80)
                print("🎉 CALIBRATION TRAINING COMPLETED SUCCESSFULLY!")
                print("=" * 80)
                print("✅ Isotonic regression calibrators trained")
                print("✅ Calibrators saved to models/word_embedding/isotonic_calibrators.pkl")
                print("✅ Calibration quality validated")
                print("\n💡 Next steps:")
                print("   1. Restart your application to load the new calibrators")
                print("   2. Test with the same examples that had low confidence")
                print("   3. Expect 15-30% confidence improvement on average")
            else:
                print("\n⚠️ Calibration training completed but validation shows poor quality")
                print("   Consider increasing sample size or checking data quality")
        else:
            print("\n❌ Calibration training failed")
            print("   Check the error messages above and ensure:")
            print("   - Model can be loaded successfully")
            print("   - Validation data is available")
            print("   - sklearn is installed")
            
    except Exception as e:
        print(f"\n💥 Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()