"""
Feature engineering utilities for Traditional ML models
Exact copy from training notebook to ensure pickle compatibility
"""

import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Ensure NLTK data is available
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Random seed
RANDOM_SEED = 42


class FeatureEngineer:
    """
    Extract linguistic and sentiment features for emotion detection
    """
    
    def __init__(self):
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Emotion lexicons
        self.emotion_lexicons = {
            'joy': ['happy', 'joy', 'joyful', 'cheerful', 'delighted', 'pleased', 'glad', 'elated'],
            'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'sorrowful', 'gloomy', 'melancholy'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'enraged', 'outraged'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'startled'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled'],
            'love': ['love', 'adore', 'cherish', 'affection', 'fond', 'devoted'],
            'gratitude': ['grateful', 'thankful', 'appreciate', 'gratitude', 'thanks']
        }
        
        # Intensifiers and diminishers
        self.intensifiers = {
            'very', 'really', 'extremely', 'absolutely', 'completely',
            'totally', 'utterly', 'quite', 'remarkably', 'exceptionally'
        }
        
        self.diminishers = {
            'slightly', 'somewhat', 'rather', 'fairly', 'a bit',
            'a little', 'kind of', 'sort of', 'moderately'
        }
        
    def extract_features(self, text, processed_text):
        """
        Extract comprehensive features from text
        
        Args:
            text: Original text
            processed_text: Preprocessed text
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Basic statistics (10 features)
        features.extend(self._extract_basic_stats(text, processed_text))
        
        # Punctuation features (10 features)
        features.extend(self._extract_punctuation_features(text))
        
        # Sentiment features (10 features)
        features.extend(self._extract_sentiment_features(text))
        
        # Emotion lexicon features (16 features)
        features.extend(self._extract_emotion_features(processed_text))
        
        # Linguistic features (10 features)
        features.extend(self._extract_linguistic_features(text, processed_text))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_basic_stats(self, text, processed_text):
        """Extract basic text statistics"""
        tokens = processed_text.split()
        original_tokens = text.split()
        
        features = [
            len(text),                                    # Character count
            len(tokens),                                  # Token count
            len(set(tokens)),                            # Unique token count
            len(set(tokens)) / max(len(tokens), 1),      # Vocabulary richness
            np.mean([len(t) for t in tokens]) if tokens else 0,  # Avg token length
            np.std([len(t) for t in tokens]) if len(tokens) > 1 else 0,  # Std token length
            sum(1 for t in tokens if len(t) > 6),        # Long word count
            sum(1 for t in tokens if len(t) <= 3),       # Short word count
            len(original_tokens) - len(tokens),           # Tokens removed
            text.count(' ') / max(len(text), 1)          # Space ratio
        ]
        
        return features
    
    def _extract_punctuation_features(self, text):
        """Extract punctuation-based features"""
        features = [
            text.count('!'),                              # Exclamation marks
            text.count('?'),                              # Question marks
            text.count('.'),                              # Periods
            text.count(','),                              # Commas
            text.count('...'),                            # Ellipsis
            text.count('!!!') + text.count('???'),        # Multiple punctuation
            text.count('?!') + text.count('!?'),          # Mixed punctuation
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
            len(re.findall(r'[A-Z]{2,}', text)),          # All caps words
            len(re.findall(r'[!?]{2,}', text))           # Repeated punctuation
        ]
        
        return features
    
    def _extract_sentiment_features(self, text):
        """Extract sentiment scores"""
        # VADER sentiment scores
        scores = self.sia.polarity_scores(text)
        
        features = [
            scores['pos'],                                # Positive score
            scores['neg'],                                # Negative score
            scores['neu'],                                # Neutral score
            scores['compound'],                           # Compound score
            abs(scores['compound']),                      # Absolute sentiment
            scores['pos'] - scores['neg'],                # Sentiment difference
            max(scores['pos'], scores['neg']),            # Dominant sentiment
            1 if scores['compound'] > 0.5 else 0,         # Strong positive
            1 if scores['compound'] < -0.5 else 0,        # Strong negative
            1 if abs(scores['compound']) < 0.1 else 0    # Neutral indicator
        ]
        
        return features
    
    def _extract_emotion_features(self, processed_text):
        """Extract emotion lexicon features"""
        tokens = set(processed_text.lower().split())
        features = []
        
        # Count emotion words for each category
        for emotion, words in self.emotion_lexicons.items():
            word_set = set(words)
            count = len(tokens & word_set)
            ratio = count / max(len(tokens), 1)
            features.extend([count, ratio])
        
        return features
    
    def _extract_linguistic_features(self, text, processed_text):
        """Extract linguistic patterns"""
        tokens = processed_text.split()
        
        # Negation detection
        negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none'}
        negation_count = sum(1 for token in tokens if token in negation_words)
        
        # Intensifier and diminisher counts
        intensifier_count = sum(1 for token in tokens if token in self.intensifiers)
        diminisher_count = sum(1 for token in tokens if token in self.diminishers)
        
        # Personal pronouns
        first_person = sum(1 for token in tokens if token in {'i', 'me', 'my', 'myself', 'mine'})
        second_person = sum(1 for token in tokens if token in {'you', 'your', 'yours', 'yourself'})
        third_person = sum(1 for token in tokens if token in {'he', 'she', 'they', 'them', 'his', 'her', 'their'})
        
        features = [
            negation_count,                               # Negation count
            negation_count / max(len(tokens), 1),         # Negation ratio
            intensifier_count,                            # Intensifier count
            diminisher_count,                             # Diminisher count
            first_person,                                 # First person pronouns
            second_person,                                # Second person pronouns
            third_person,                                 # Third person pronouns
            len(re.findall(r'\b(but|however|although|though)\b', text.lower())),  # Contrast words
            len(re.findall(r'\b(because|since|therefore|thus)\b', text.lower())), # Causal words
            len(re.findall(r'\b(if|unless|whether)\b', text.lower()))            # Conditional words
        ]
        
        return features


class FeatureExtractor:
    """
    Combine multiple feature extraction methods
    """
    
    def __init__(self, config):
        self.config = config
        self.vectorizers = {}
        self.feature_engineer = FeatureEngineer()
        # Use MinMaxScaler to avoid negative values for NB models
        self.scaler = MinMaxScaler()  
        self.svd = None
        self.is_fitted = False
        
    def fit_transform(self, texts, processed_texts):
        """
        Fit and transform features
        
        Args:
            texts: Original texts
            processed_texts: Preprocessed texts
            
        Returns:
            Combined feature matrix
        """
        print("\n🔄 Extracting features...")
        all_features = []
        
        # 1. TF-IDF Features
        print("  📊 Extracting TF-IDF features...")
        tfidf_vectorizer = TfidfVectorizer(**self.config.TFIDF_PARAMS)
        tfidf_features = tfidf_vectorizer.fit_transform(processed_texts)
        self.vectorizers['tfidf'] = tfidf_vectorizer
        all_features.append(tfidf_features)
        print(f"     Shape: {tfidf_features.shape}")
        
        # 2. Character N-gram Features (if enabled)
        if self.config.USE_CHAR_NGRAMS:
            print("  📊 Extracting character n-gram features...")
            char_vectorizer = TfidfVectorizer(**self.config.CHAR_PARAMS)
            char_features = char_vectorizer.fit_transform(processed_texts)
            self.vectorizers['char'] = char_vectorizer
            all_features.append(char_features)
            print(f"     Shape: {char_features.shape}")
        
        # 3. Count Vectorizer (for comparison)
        print("  📊 Extracting count features...")
        count_vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        count_features = count_vectorizer.fit_transform(processed_texts)
        self.vectorizers['count'] = count_vectorizer
        all_features.append(count_features)
        print(f"     Shape: {count_features.shape}")
        
        # Store NB-safe feature indices (before adding linguistic features)
        current_idx = 0
        self.nb_feature_indices = {}
        self.nb_feature_indices['tfidf'] = (current_idx, current_idx + tfidf_features.shape[1])
        current_idx += tfidf_features.shape[1]
        
        if self.config.USE_CHAR_NGRAMS:
            self.nb_feature_indices['char'] = (current_idx, current_idx + char_features.shape[1])
            current_idx += char_features.shape[1]
            
        self.nb_feature_indices['count'] = (current_idx, current_idx + count_features.shape[1])
        current_idx += count_features.shape[1]
        
        self.nb_end_idx = current_idx  # End index for NB-safe features
        
        # 4. Linguistic Features (if enabled)
        if self.config.USE_LINGUISTIC_FEATURES:
            print("  📊 Extracting linguistic features...")
            linguistic_features = []
            
            for text, proc_text in tqdm(zip(texts, processed_texts), 
                                       total=len(texts),
                                       desc="     Processing"):
                features = self.feature_engineer.extract_features(text, proc_text)
                linguistic_features.append(features)
            
            linguistic_features = np.array(linguistic_features)
            
            # Check for negative values before scaling
            min_val = linguistic_features.min()
            if min_val < 0:
                print(f"     ⚠️ Found negative values (min: {min_val:.4f}), applying MinMaxScaler")
            
            # Scale linguistic features to [0, 1] range
            linguistic_features = self.scaler.fit_transform(linguistic_features)
            all_features.append(linguistic_features)
            print(f"     Shape: {linguistic_features.shape}")
        
        # Combine all features
        print("\n  🔗 Combining all features...")
        
        # Convert dense arrays to sparse for consistency
        sparse_features = []
        for feat in all_features:
            if isinstance(feat, np.ndarray):
                sparse_features.append(csr_matrix(feat))
            else:
                sparse_features.append(feat)
        
        # Horizontally stack all features
        combined_features = hstack(sparse_features)
        
        # Apply dimensionality reduction if enabled
        if hasattr(self.config, 'USE_DIMENSIONALITY_REDUCTION') and self.config.USE_DIMENSIONALITY_REDUCTION:
            print("\n  📉 Applying dimensionality reduction...")
            n_components = min(self.config.SVD_COMPONENTS, combined_features.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
            combined_features = self.svd.fit_transform(combined_features)
            print(f"     Reduced to {combined_features.shape[1]} dimensions")
            print(f"     Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
            # Convert back to sparse
            combined_features = csr_matrix(combined_features)
        
        # Print feature matrix statistics
        print(f"\n✅ Total feature dimensions: {combined_features.shape}")
        print(f"   Samples: {combined_features.shape[0]:,}")
        print(f"   Features: {combined_features.shape[1]:,}")
        
        # Check for negative values
        if hasattr(combined_features, 'data'):
            has_negative = (combined_features.data < 0).any()
            min_val = combined_features.data.min() if len(combined_features.data) > 0 else 0
        else:
            has_negative = (combined_features < 0).any()
            min_val = combined_features.min()
            
        print(f"   Min value: {min_val:.4f}")
        print(f"   Has negative values: {has_negative}")
        print(f"   Sparsity: {1 - (combined_features.nnz / (combined_features.shape[0] * combined_features.shape[1])):.2%}")
        
        self.is_fitted = True
        return combined_features
    
    def transform(self, texts, processed_texts):
        """
        Transform new data using fitted extractors
        
        Args:
            texts: Original texts
            processed_texts: Preprocessed texts
            
        Returns:
            Combined feature matrix
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        all_features = []
        
        # Transform using fitted vectorizers
        tfidf_features = self.vectorizers['tfidf'].transform(processed_texts)
        all_features.append(tfidf_features)
        
        if 'char' in self.vectorizers:
            char_features = self.vectorizers['char'].transform(processed_texts)
            all_features.append(char_features)
        
        count_features = self.vectorizers['count'].transform(processed_texts)
        all_features.append(count_features)
        
        if self.config.USE_LINGUISTIC_FEATURES:
            linguistic_features = []
            for text, proc_text in zip(texts, processed_texts):
                features = self.feature_engineer.extract_features(text, proc_text)
                linguistic_features.append(features)
            
            linguistic_features = np.array(linguistic_features)
            linguistic_features = self.scaler.transform(linguistic_features)
            all_features.append(linguistic_features)
        
        # Combine features
        sparse_features = []
        for feat in all_features:
            if isinstance(feat, np.ndarray):
                sparse_features.append(csr_matrix(feat))
            else:
                sparse_features.append(feat)
        
        combined_features = hstack(sparse_features)
        
        # Apply dimensionality reduction if fitted
        if self.svd is not None:
            combined_features = self.svd.transform(combined_features)
            combined_features = csr_matrix(combined_features)
        
        return combined_features
    
    def get_nb_features(self, X):
        """
        Extract only non-negative features suitable for Naive Bayes
        
        Args:
            X: Full feature matrix
            
        Returns:
            Feature matrix with only TF-IDF and count features
        """
        # If dimensionality reduction was applied, we can't extract NB features
        if self.svd is not None:
            raise ValueError("Cannot extract NB-specific features after dimensionality reduction")
        
        # Extract only the non-negative features
        nb_features = X[:, :self.nb_end_idx]
        
        # Verify no negative values
        if hasattr(nb_features, 'data'):
            min_val = nb_features.data.min() if len(nb_features.data) > 0 else 0
        else:
            min_val = nb_features.min()
            
        assert min_val >= 0, f"NB features contain negative values! Min: {min_val}"
        
        return nb_features