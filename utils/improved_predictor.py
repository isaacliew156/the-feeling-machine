
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle
import re
import os

class ImprovedWordEmbeddingPredictor:
    """
    Improved predictor for CNN+GloVe model with confidence fixes
    """
    
    def __init__(self, model_path, tokenizer_path, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # FIXED: Better thresholds for different emotion types
        self.emotion_thresholds = {
            # High-frequency emotions (need higher thresholds)
            'neutral': 0.4,
            'approval': 0.3,
            'admiration': 0.3,
            'annoyance': 0.25,
            
            # Medium-frequency emotions
            'joy': 0.25,
            'sadness': 0.25,
            'anger': 0.25,
            'love': 0.2,
            'gratitude': 0.2,
            'optimism': 0.25,
            'disappointment': 0.2,
            
            # Low-frequency emotions (need lower thresholds)
            'grief': 0.1,
            'pride': 0.1,
            'relief': 0.1,
            'remorse': 0.15,
            'nervousness': 0.15,
            'embarrassment': 0.15,
        }
        
        # Default threshold for emotions not specified above
        self.default_threshold = 0.2
        
        # Temperature scaling parameter
        self.temperature = 0.7  # Lower = more confident

        self.emotion_labels = self.config['EMOTION_COLUMNS']

    def apply_temperature_scaling(self, probabilities):
        """Apply temperature scaling to improve confidence calibration"""
        epsilon = 1e-7
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        
        # Convert to logits
        logits = np.log(probabilities / (1 - probabilities))
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Convert back to probabilities
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        
        return scaled_probs
    
    def get_threshold_for_emotion(self, emotion):
        """Get optimal threshold for specific emotion"""
        return self.emotion_thresholds.get(emotion, self.default_threshold)
    
    def predict(self, text):
        """Predict emotions with improved confidence handling"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.config['MAX_SEQUENCE_LENGTH'])
        
        # Predict
        raw_probabilities = self.model.predict(padded, verbose=0)[0]
        
        # Apply temperature scaling for better confidence
        probabilities = self.apply_temperature_scaling(raw_probabilities)
        
        # Apply emotion-specific thresholds
        predicted_emotions = []
        emotion_scores = {}
        
        for i, (emotion, prob) in enumerate(zip(self.emotion_labels, probabilities)):
            emotion_scores[emotion] = float(prob)
            threshold = self.get_threshold_for_emotion(emotion)
            
            if prob > threshold:
                predicted_emotions.append(emotion)
        
        # Get confidence score (max probability)
        confidence = float(np.max(probabilities))
        
        # Get top 5 emotions
        top_emotions = sorted(
            zip(self.emotion_labels, probabilities),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'text': text,
            'predicted_emotions': predicted_emotions,
            'confidence': confidence,
            'emotion_scores': emotion_scores,
            'top_5_emotions': [(emotion, float(score)) for emotion, score in top_emotions],
            'temperature_used': self.temperature
        }
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'([.!?])\1+', r'\1', text)
        text = ' '.join(text.split())
        return text
    
    def batch_predict(self, texts):
        """Predict emotions for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# Usage example:
if __name__ == "__main__":
    # Initialize improved predictor
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "models", "word_embedding", "best_embedding_model")
    tokenizer_path = os.path.join(project_root, "models", "word_embedding", "tokenizer.pickle")
    config_path = os.path.join(project_root, "models", "word_embedding", "config.json")
    
    predictor = ImprovedWordEmbeddingPredictor(model_path, tokenizer_path, config_path)
    
    # Test examples
    test_texts = [
        "I absolutely love this! This is amazing!",
        "This is terrible and very disappointing.",
        "Thank you so much for your help, I really appreciate it!",
        "I'm really confused about this whole situation.",
        "That's hilarious! I can't stop laughing!"
    ]
    
    print("\n🎯 Testing Improved Predictions:")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: '{text}'")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Predicted: {', '.join(result['predicted_emotions']) if result['predicted_emotions'] else 'neutral'}")
        print(f"Top 3: {', '.join([f'{e}({s:.2f})' for e, s in result['top_5_emotions'][:3]])}")
