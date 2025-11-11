"""
Text preprocessing utilities for Traditional ML models
Exact copy from training notebook to ensure pickle compatibility
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import os

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    """Download required NLTK data"""
    required = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'omw-1.4']
    for package in required:
        try:
            if package == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package in ['stopwords', 'wordnet']:
                nltk.data.find(f'corpora/{package}')
            elif package == 'vader_lexicon':
                nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download(package, quiet=True)

ensure_nltk_data()

# Random seed for reproducibility
RANDOM_SEED = 42

class Config:
    """Configuration settings for Traditional ML pipeline"""
    
    # Project structure
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    DATA_PATH = os.path.join(DATA_DIR, 'go_emotions_dataset.csv')
    
    # Model output paths
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'traditional_ml')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'traditional_ml_results')
    
    # Data parameters
    USE_FULL_DATASET = True
    SAMPLE_SIZE = None
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    
    # Text preprocessing parameters
    MIN_TEXT_LENGTH = 2
    MAX_TEXT_LENGTH = 512
    REMOVE_STOPWORDS = True
    USE_LEMMATIZATION = True
    USE_STEMMING = False
    REMOVE_NUMBERS = True
    
    # Feature extraction parameters
    TFIDF_PARAMS = {
        'max_features': 10000,
        'ngram_range': (1, 3),
        'min_df': 2,
        'max_df': 0.95,
        'use_idf': True,
        'smooth_idf': True,
        'sublinear_tf': True,
        'norm': 'l2'
    }
    
    # Character n-gram parameters
    CHAR_PARAMS = {
        'analyzer': 'char',
        'ngram_range': (3, 5),
        'max_features': 2000,
        'min_df': 2
    }
    
    # Model training parameters
    USE_CLASS_WEIGHTS = True
    USE_MULTIOUTPUT = True
    CV_FOLDS = 5
    CHECK_CONVERGENCE = True
    
    # Feature engineering flags
    USE_LINGUISTIC_FEATURES = True
    USE_SENTIMENT_FEATURES = True
    USE_CHAR_NGRAMS = True
    
    # Performance optimization
    N_JOBS = -1
    VERBOSE = 1
    
    # Emotion categories
    EMOTION_COLUMNS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]


class TextPreprocessor:
    """
    Comprehensive text preprocessing for emotion classification
    """
    
    def __init__(self, config):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Load stopwords and keep emotion-relevant ones
        self.stop_words = set(stopwords.words('english'))
        self.emotion_words = {
            'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody', 'nothing',
            'nowhere', 'hardly', 'scarcely', 'barely', 'seldom',
            'love', 'hate', 'like', 'dislike', 'happy', 'sad', 'angry',
            'afraid', 'surprised', 'disgusted', 'trust', 'anticipate',
            'very', 'really', 'so', 'too', 'quite', 'extremely'
        }
        self.stop_words -= self.emotion_words
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile all regex patterns"""
        self.patterns = {
            'url': re.compile(r'https?://\S+|www\.\S+'),
            'email': re.compile(r'\S+@\S+'),
            'mention': re.compile(r'@\w+'),
            'hashtag': re.compile(r'#(\w+)'),
            'number': re.compile(r'\b\d+\b'),
            'special_chars': re.compile(r'[^a-zA-Z0-9\s\.\,\!\?\'\-]'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'multiple_spaces': re.compile(r'\s+'),
            'reddit_quotes': re.compile(r'&gt;|&lt;|&amp;'),
            'subreddit': re.compile(r'/?r/\w+'),
            'reddit_user': re.compile(r'/?u/\w+')
        }
        
        # Contraction mapping
        self.contractions = {
            "won't": "will not", "wouldn't": "would not", "can't": "cannot",
            "couldn't": "could not", "shouldn't": "should not", "mustn't": "must not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "doesn't": "does not", "don't": "do not", "didn't": "did not",
            "ain't": "is not", "let's": "let us", "i'm": "i am",
            "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have",
            "they've": "they have", "i'd": "i would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would",
            "they'd": "they would", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will"
        }
    
    def preprocess(self, text):
        """
        Main preprocessing pipeline
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle Reddit-specific content
        text = self._handle_reddit_content(text)
        
        # Expand contractions
        text = self._expand_contractions(text)
        
        # Remove URLs, emails, mentions
        text = self.patterns['url'].sub(' ', text)
        text = self.patterns['email'].sub(' ', text)
        text = self.patterns['mention'].sub(' ', text)
        
        # Handle hashtags (keep the word)
        text = self.patterns['hashtag'].sub(r'\1', text)
        
        # Remove numbers (optional - might be useful for some emotions)
        if self.config.REMOVE_NUMBERS:
            text = self.patterns['number'].sub(' ', text)
        
        # Handle repeated characters (e.g., 'sooooo' -> 'soo')
        text = self.patterns['repeated_chars'].sub(r'\1\1', text)
        
        # Remove special characters but keep basic punctuation
        text = self.patterns['special_chars'].sub(' ', text)
        
        # Normalize whitespace
        text = self.patterns['multiple_spaces'].sub(' ', text).strip()
        
        # Tokenize
        tokens = self._tokenize_text(text)
        
        # Apply length constraints
        if len(tokens) < self.config.MIN_TEXT_LENGTH:
            return ""
        
        # Rejoin tokens
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def _handle_reddit_content(self, text):
        """Handle Reddit-specific content"""
        # Remove [NAME], [RELIGION], etc. placeholders
        text = re.sub(r'\[[\w\s]+\]', '', text)
        
        # Handle Reddit quotes
        text = self.patterns['reddit_quotes'].sub(' ', text)
        
        # Handle subreddit mentions
        text = self.patterns['subreddit'].sub(' ', text)
        
        # Handle user mentions
        text = self.patterns['reddit_user'].sub(' ', text)
        
        # Handle /s, /jk, etc.
        text = re.sub(r'\/[a-z]+\b', '', text)
        
        return text
    
    def _expand_contractions(self, text):
        """Expand contractions"""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def _tokenize_text(self, text):
        """Tokenize and process individual tokens"""
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split
            tokens = text.split()
        
        processed_tokens = []
        for token in tokens:
            # Skip very short tokens
            if len(token) < 2 and token not in ['i', '!', '?']:
                continue
            
            # Skip stopwords (if enabled)
            if self.config.REMOVE_STOPWORDS and token in self.stop_words:
                continue
            
            # Apply lemmatization or stemming
            if self.config.USE_LEMMATIZATION:
                token = self.lemmatizer.lemmatize(token, pos='v')
                token = self.lemmatizer.lemmatize(token, pos='n')
            elif self.config.USE_STEMMING:
                token = self.stemmer.stem(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    @staticmethod
    def validate_input(text):
        """Validate input text"""
        return isinstance(text, str) and len(text.strip()) >= 2