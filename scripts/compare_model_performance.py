"""
Evaluation script for the FIXED CNN+GloVe model
Compares fixed model with original to validate improvements
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ModelEvaluator:
    """Evaluates and compares original vs fixed CNN+GloVe models"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.original_model_dir = os.path.join(self.project_root, 'models', 'word_embedding')
        self.fixed_model_dir = os.path.join(self.project_root, 'models', 'word_embedding_fixed')
        self.results_dir = os.path.join(self.project_root, 'results', 'model_comparison')
        
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_test_data(self):
        """Load and prepare test data"""
        print("Loading test data...")
        data_path = os.path.join(self.project_root, 'data', 'go_emotions_dataset.csv')
        df = pd.read_csv(data_path)
        
        # Same preprocessing as training
        if 'example_very_unclear' in df.columns:
            df = df[df['example_very_unclear'] == False].copy()
        
        def clean_text(text):
            if not isinstance(text, str):
                return ""
            import re
            text = text.lower()
            text = re.sub(r'http\\S+|www.\\S+', '', text)
            text = re.sub(r'@\\w+', '', text)
            text = re.sub(r'#(\\w+)', r'\\1', text)
            text = re.sub(r'([.!?])\\1+', r'\\1', text)
            text = ' '.join(text.split())
            return text
        
        df['cleaned_text'] = df['text'].apply(clean_text)
        df = df[df['cleaned_text'].str.len() >= 3].copy()
        
        emotion_columns = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral'
        ]
        
        # Take a sample for evaluation
        test_df = df.sample(n=5000, random_state=42)  # Sample for faster evaluation
        
        return test_df['cleaned_text'].values, test_df[emotion_columns].values, emotion_columns
    
    def load_model_and_tokenizer(self, model_dir, model_name):
        """Load model and tokenizer from directory"""
        try:
            print(f"Loading {model_name} model...")
            
            # Load model
            model_path = os.path.join(model_dir, 'best_embedding_model' if 'original' in model_name else 'best_fixed_model')
            model = tf.keras.models.load_model(model_path)
            
            # Load tokenizer
            tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Load config
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"✅ {model_name} model loaded successfully")
            return model, tokenizer, config
            
        except Exception as e:
            print(f"❌ Failed to load {model_name} model: {str(e)}")
            return None, None, None
    
    def preprocess_texts(self, texts, tokenizer, max_length):
        """Preprocess texts for model prediction"""
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    def evaluate_model(self, model, X_preprocessed, y_true, model_name, config):
        """Evaluate model and return metrics"""
        print(f"\\nEvaluating {model_name}...")
        
        # Get predictions
        y_pred_proba = model.predict(X_preprocessed, batch_size=64, verbose=1)
        
        # Test multiple thresholds
        thresholds = np.arange(0.1, 0.8, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_true, y_pred, average='macro')
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            threshold_results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Get final predictions with best threshold
        y_pred_best = (y_pred_proba > best_threshold).astype(int)
        
        # Calculate confidence statistics
        confidence_stats = {
            'mean_confidence': float(np.mean(np.max(y_pred_proba, axis=1))),
            'median_confidence': float(np.median(np.max(y_pred_proba, axis=1))),
            'std_confidence': float(np.std(np.max(y_pred_proba, axis=1))),
            'min_confidence': float(np.min(np.max(y_pred_proba, axis=1))),
            'max_confidence': float(np.max(np.max(y_pred_proba, axis=1))),
            'confidence_range_20_30': float(np.mean((np.max(y_pred_proba, axis=1) >= 0.2) & 
                                                  (np.max(y_pred_proba, axis=1) <= 0.3))),
            'confidence_above_50': float(np.mean(np.max(y_pred_proba, axis=1) > 0.5))
        }
        
        results = {
            'model_name': model_name,
            'best_threshold': float(best_threshold),
            'f1_macro': float(best_f1),
            'precision': float(precision_score(y_true, y_pred_best, average='macro', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_best, average='macro', zero_division=0)),
            'confidence_stats': confidence_stats,
            'threshold_results': threshold_results
        }
        
        return results, y_pred_proba
    
    def compare_models(self):
        """Compare original and fixed models"""
        print("=" * 80)
        print("CNN+GLOVE MODEL COMPARISON")
        print("=" * 80)
        
        # Load test data
        X_text, y_true, emotion_labels = self.load_test_data()
        print(f"✅ Test data loaded: {len(X_text)} samples")
        
        results = {}
        predictions = {}
        
        # Evaluate original model
        orig_model, orig_tokenizer, orig_config = self.load_model_and_tokenizer(
            self.original_model_dir, "Original"
        )
        
        if orig_model is not None:
            X_orig = self.preprocess_texts(X_text, orig_tokenizer, 
                                          orig_config.get('MAX_SEQUENCE_LENGTH', 50))
            results['original'], predictions['original'] = self.evaluate_model(
                orig_model, X_orig, y_true, "Original", orig_config
            )
        
        # Evaluate fixed model (if exists)
        if os.path.exists(self.fixed_model_dir):
            fixed_model, fixed_tokenizer, fixed_config = self.load_model_and_tokenizer(
                self.fixed_model_dir, "Fixed"
            )
            
            if fixed_model is not None:
                X_fixed = self.preprocess_texts(X_text, fixed_tokenizer,
                                               fixed_config.get('MAX_SEQUENCE_LENGTH', 50))
                results['fixed'], predictions['fixed'] = self.evaluate_model(
                    fixed_model, X_fixed, y_true, "Fixed", fixed_config
                )
        
        # Generate comparison report
        self.generate_comparison_report(results, emotion_labels, y_true, predictions)
        
        return results
    
    def generate_comparison_report(self, results, emotion_labels, y_true, predictions):
        """Generate comprehensive comparison report"""
        print("\\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        
        # Overall performance comparison
        print("\\n📊 OVERALL PERFORMANCE:")
        print("-" * 40)
        
        for model_name, result in results.items():
            print(f"\\n{model_name.upper()} MODEL:")
            print(f"  F1 Macro:        {result['f1_macro']:.4f}")
            print(f"  Precision:       {result['precision']:.4f}")
            print(f"  Recall:          {result['recall']:.4f}")
            print(f"  Best Threshold:  {result['best_threshold']:.2f}")
        
        # Confidence analysis
        print("\\n🎯 CONFIDENCE ANALYSIS:")
        print("-" * 40)
        
        for model_name, result in results.items():
            conf = result['confidence_stats']
            print(f"\\n{model_name.upper()} MODEL:")
            print(f"  Mean Confidence:     {conf['mean_confidence']:.3f}")
            print(f"  Median Confidence:   {conf['median_confidence']:.3f}")
            print(f"  Confidence Range:    {conf['min_confidence']:.3f} - {conf['max_confidence']:.3f}")
            print(f"  In 20-30% Range:     {conf['confidence_range_20_30']:.1%}")
            print(f"  Above 50%:           {conf['confidence_above_50']:.1%}")
        
        # Improvement analysis
        if 'original' in results and 'fixed' in results:
            orig = results['original']
            fixed = results['fixed']
            
            print("\\n📈 IMPROVEMENTS:")
            print("-" * 40)
            
            f1_improvement = (fixed['f1_macro'] - orig['f1_macro']) / orig['f1_macro'] * 100
            conf_improvement = (fixed['confidence_stats']['mean_confidence'] - 
                              orig['confidence_stats']['mean_confidence']) / \
                              orig['confidence_stats']['mean_confidence'] * 100
            
            print(f"  F1 Macro:        {f1_improvement:+.1f}%")
            print(f"  Mean Confidence: {conf_improvement:+.1f}%")
            print(f"  Threshold:       {orig['best_threshold']:.2f} → {fixed['best_threshold']:.2f}")
            
            # High confidence predictions
            orig_high_conf = orig['confidence_stats']['confidence_above_50']
            fixed_high_conf = fixed['confidence_stats']['confidence_above_50']
            print(f"  High Confidence: {orig_high_conf:.1%} → {fixed_high_conf:.1%}")
        
        # Save detailed results
        self.save_results(results)
        
        # Create visualizations
        self.create_visualizations(results)
    
    def save_results(self, results):
        """Save comparison results to JSON"""
        results_path = os.path.join(self.results_dir, 'model_comparison_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n✅ Results saved: {results_path}")
    
    def create_visualizations(self, results):
        """Create comparison visualizations"""
        if len(results) < 2:
            print("⚠️ Need both models for visualization")
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. F1 Score comparison
        models = list(results.keys())
        f1_scores = [results[model]['f1_macro'] for model in models]
        
        bars = axes[0, 0].bar(models, f1_scores, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
        axes[0, 0].set_title('F1 Macro Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_ylim(0, max(f1_scores) * 1.1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Confidence distribution comparison
        for i, (model_name, result) in enumerate(results.items()):
            conf_stats = result['confidence_stats']
            axes[0, 1].bar(i, conf_stats['mean_confidence'], 
                          color=['#ff7f0e', '#2ca02c'][i], alpha=0.8, 
                          label=f'{model_name} (μ={conf_stats["mean_confidence"]:.3f})')
        
        axes[0, 1].set_title('Mean Confidence Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Mean Confidence')
        axes[0, 1].set_xticks(range(len(results)))
        axes[0, 1].set_xticklabels(list(results.keys()))
        axes[0, 1].legend()
        
        # 3. Threshold comparison
        thresholds = [results[model]['best_threshold'] for model in models]
        bars = axes[1, 0].bar(models, thresholds, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
        axes[1, 0].set_title('Optimal Threshold Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Threshold')
        axes[1, 0].set_ylim(0, max(thresholds) * 1.1)
        
        for bar, threshold in zip(bars, thresholds):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{threshold:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Confidence ranges comparison
        ranges_data = []
        for model_name, result in results.items():
            conf_stats = result['confidence_stats']
            ranges_data.append([
                conf_stats['confidence_range_20_30'] * 100,  # 20-30% range
                conf_stats['confidence_above_50'] * 100      # Above 50%
            ])
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, [r[0] for r in ranges_data], width, 
                      label='20-30% Range', color='#ff7f0e', alpha=0.8)
        axes[1, 1].bar(x + width/2, [r[1] for r in ranges_data], width,
                      label='Above 50%', color='#2ca02c', alpha=0.8)
        
        axes[1, 1].set_title('Confidence Range Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Percentage of Predictions (%)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\\n✅ Visualizations saved: {plot_path}")

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    results = evaluator.compare_models()
    
    print("\\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\\n🎯 Key Findings:")
    
    if 'original' in results and 'fixed' in results:
        orig_conf = results['original']['confidence_stats']['mean_confidence']
        fixed_conf = results['fixed']['confidence_stats']['mean_confidence']
        
        orig_f1 = results['original']['f1_macro']
        fixed_f1 = results['fixed']['f1_macro']
        
        print(f"\\n1. Confidence Improvement: {orig_conf:.3f} → {fixed_conf:.3f}")
        print(f"2. F1 Score Change: {orig_f1:.4f} → {fixed_f1:.4f}")
        print(f"3. Original 20-30% problem: {results['original']['confidence_stats']['confidence_range_20_30']:.1%} in range")
        print(f"4. Fixed model improvement: {results['fixed']['confidence_stats']['confidence_above_50']:.1%} above 50%")
        
        if fixed_conf > orig_conf:
            print("\\n✅ SUCCESS: Fixed model shows improved confidence levels!")
        else:
            print("\\n⚠️ Further tuning needed to improve confidence")
    
    return results

if __name__ == "__main__":
    results = main()