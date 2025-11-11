"""
Fixed CNN+GloVe Model for GoEmotions Classification
Addresses low confidence issues (20-30%) through:
1. Class weight balancing
2. Focal Loss implementation
3. Reduced regularization
4. Improved training strategy
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import pickle
import re
from datetime import datetime
from tqdm import tqdm

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("=" * 60)
print("FIXED CNN+GLOVE MODEL FOR GOEMOTIONS")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

class FixedConfig:
    """FIXED Configuration addressing low confidence issues"""
    
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'go_emotions_dataset.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'word_embedding_fixed')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'word_embedding_fixed')
    EMBEDDING_PATH = os.path.join(PROJECT_ROOT, 'embeddings', 'glove.6B.300d.txt')
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Data parameters
    MAX_SEQUENCE_LENGTH = 50
    MAX_VOCAB_SIZE = 20000
    TEST_SPLIT = 0.15
    VAL_SPLIT = 0.15
    
    # Embedding parameters
    EMBEDDING_DIM = 300
    EMBEDDING_TRAINABLE = True
    
    # FIXED Model architecture - Reduced regularization
    CNN_FILTERS = [200, 200, 200]  # Increased from 128
    CNN_KERNEL_SIZES = [3, 5, 7]  # Increased from [2,3,4]
    HIDDEN_DIM = 512  # Increased from 256
    DROPOUT_RATE = 0.3  # Reduced from 0.5
    SPATIAL_DROPOUT = 0.2  # Reduced from 0.3
    L2_REG = 1e-5  # Reduced from 1e-4
    
    # FIXED Training parameters
    BATCH_SIZE = 32  # Reduced for better convergence
    EPOCHS = 50  # Increased from 30
    LEARNING_RATE = 0.0005  # Reduced initial LR
    PATIENCE = 8  # Increased patience
    
    # Label smoothing
    LABEL_SMOOTHING = 0.05
    
    # Focal loss parameters
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Emotions
    EMOTION_COLUMNS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]

config = FixedConfig()

# FIXED: Focal Loss Implementation
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance
    Helps model focus on hard examples and improves confidence
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) \
               -(1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        
        return tf.reduce_mean(loss)
    return loss_fn

# FIXED: Label smoothing
def smooth_labels(y, smooth_factor=0.05):
    """Apply label smoothing to reduce overconfidence"""
    return y * (1 - smooth_factor) + smooth_factor / len(config.EMOTION_COLUMNS)

# FIXED: Improved CNN Architecture
def build_fixed_cnn_model(vocab_size, embedding_matrix, config):
    """
    FIXED CNN architecture addressing low confidence issues:
    1. Larger kernels for better feature extraction
    2. Reduced regularization
    3. Batch normalization
    4. Attention pooling instead of GlobalMax
    """
    inputs = layers.Input(shape=(config.MAX_SEQUENCE_LENGTH,))
    
    # Embedding layer
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=config.EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=config.MAX_SEQUENCE_LENGTH,
        trainable=config.EMBEDDING_TRAINABLE
    )(inputs)
    
    # FIXED: Reduced spatial dropout
    embedding = layers.SpatialDropout1D(config.SPATIAL_DROPOUT)(embedding)
    
    # FIXED: Multiple CNN branches with batch normalization
    conv_layers = []
    for filters, kernel_size in zip(config.CNN_FILTERS, config.CNN_KERNEL_SIZES):
        # Conv1D with batch norm
        conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='valid',
            kernel_regularizer=tf.keras.regularizers.l2(config.L2_REG)
        )(embedding)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        
        # FIXED: Attention-based pooling instead of GlobalMax
        attention = layers.Dense(1, activation='tanh')(conv)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(filters)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Weighted average pooling
        merged = layers.Multiply()([conv, attention])
        pooled = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(merged)
        
        conv_layers.append(pooled)
    
    # Concatenate CNN features
    if len(conv_layers) > 1:
        merged = layers.concatenate(conv_layers)
    else:
        merged = conv_layers[0]
    
    # FIXED: Improved dense layers
    dense = layers.Dense(
        config.HIDDEN_DIM,
        kernel_regularizer=tf.keras.regularizers.l2(config.L2_REG)
    )(merged)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Activation('relu')(dense)
    dense = layers.Dropout(config.DROPOUT_RATE)(dense)
    
    # Second dense layer
    dense = layers.Dense(256)(dense)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Activation('relu')(dense)
    dense = layers.Dropout(config.DROPOUT_RATE)(dense)
    
    # FIXED: Output layer with better initialization
    outputs = layers.Dense(
        len(config.EMOTION_COLUMNS), 
        activation='sigmoid',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )(dense)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Fixed_CNN_GloVe')
    return model

# FIXED: Learning rate schedule
def create_lr_schedule(initial_lr=0.0005):
    """Cosine decay with restarts for better convergence"""
    return tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.1
    )

# FIXED: Class weight computation
def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset"""
    class_weights = {}
    for i, emotion in enumerate(config.EMOTION_COLUMNS):
        pos_samples = np.sum(y_train[:, i])
        neg_samples = len(y_train) - pos_samples
        
        if pos_samples > 0:
            # Higher weight for rarer classes
            weight = neg_samples / pos_samples
            # Cap the weight to prevent extreme values
            class_weights[i] = min(weight, 10.0)
        else:
            class_weights[i] = 1.0
    
    return class_weights

# Data loading function
def load_and_preprocess_data():
    """Load and preprocess data with improved cleaning"""
    print("\n" + "=" * 60)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 60)
    
    df = pd.read_csv(config.DATA_PATH)
    print(f"✅ Loaded {len(df):,} samples")
    
    # Remove unclear examples
    if 'example_very_unclear' in df.columns:
        df = df[df['example_very_unclear'] == False].copy()
        print(f"✅ Clean samples: {len(df):,}")
    
    # FIXED: Improved text cleaning
    def clean_text(text):
        """Enhanced preprocessing while preserving emotion signals"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs but keep emotional indicators
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Handle mentions and hashtags more carefully
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        
        # Preserve important punctuation for emotions
        text = re.sub(r'([.!?])\1+', r'\1', text)  # Normalize repeated punctuation
        
        # Clean whitespace
        text = ' '.join(text.split())
        return text
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() >= 3].copy()  # Minimum 3 chars
    
    return df

# Main training function
def train_fixed_model():
    """Main training pipeline with all fixes applied"""
    # Load data
    df = load_and_preprocess_data()
    
    X_text = df['cleaned_text'].values
    y = df[config.EMOTION_COLUMNS].values
    
    print(f"✅ Final dataset: {len(df):,} samples")
    print(f"✅ Label distribution: {y.sum(axis=1).mean():.2f} labels per sample")
    
    # Tokenization
    print("\n" + "=" * 60)
    print("TOKENIZATION")
    print("=" * 60)
    
    tokenizer = Tokenizer(num_words=config.MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_text)
    
    sequences = tokenizer.texts_to_sequences(X_text)
    X = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH, 
                      padding='post', truncating='post')
    
    word_index = tokenizer.word_index
    vocab_size = min(len(word_index) + 1, config.MAX_VOCAB_SIZE)
    
    print(f"✅ Vocabulary size: {vocab_size:,}")
    print(f"✅ Sequence shape: {X.shape}")
    
    # Data splitting
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT, random_state=RANDOM_SEED, 
        stratify=y[:, -1]  # Stratify by neutral class
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SPLIT/(1-config.TEST_SPLIT),
        random_state=RANDOM_SEED, stratify=y_temp[:, -1]
    )
    
    print(f"\n✅ Train: {X_train.shape[0]:,} samples")
    print(f"✅ Val: {X_val.shape[0]:,} samples")
    print(f"✅ Test: {X_test.shape[0]:,} samples")
    
    # FIXED: Apply label smoothing
    y_train_smooth = smooth_labels(y_train, config.LABEL_SMOOTHING)
    y_val_smooth = smooth_labels(y_val, config.LABEL_SMOOTHING)
    
    # FIXED: Compute class weights
    print("\n" + "=" * 60)
    print("CLASS WEIGHT COMPUTATION")
    print("=" * 60)
    
    class_weights = compute_class_weights(y_train)
    print("✅ Class weights computed")
    
    # Show most imbalanced classes
    sorted_weights = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
    print("\n📊 Most imbalanced emotions:")
    for i, weight in sorted_weights[:5]:
        emotion = config.EMOTION_COLUMNS[i]
        support = np.sum(y_train[:, i])
        print(f"  {emotion:15s}: weight={weight:.2f}, support={support}")
    
    # Load embeddings (same as before)
    print("\n" + "=" * 60)
    print("LOADING EMBEDDINGS")
    print("=" * 60)
    
    def load_glove_embeddings(path, word_index, embedding_dim, vocab_size):
        embeddings_index = {}
        
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading GloVe"):
                    values = line.split()
                    word = values[0]
                    try:
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs
                    except:
                        continue
            print(f"✅ Loaded {len(embeddings_index):,} word vectors")
        else:
            print("⚠️ GloVe file not found, using random initialization")
            return np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        
        # Create embedding matrix
        embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        words_found = 0
        
        for word, i in word_index.items():
            if i >= vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                words_found += 1
        
        coverage = words_found/min(len(word_index), vocab_size)*100
        print(f"✅ Coverage: {words_found}/{min(len(word_index), vocab_size)} ({coverage:.1f}%)")
        
        return embedding_matrix
    
    embedding_matrix = load_glove_embeddings(
        config.EMBEDDING_PATH, word_index, config.EMBEDDING_DIM, vocab_size
    )
    
    # Build FIXED model
    print("\n" + "=" * 60)
    print("FIXED MODEL ARCHITECTURE")
    print("=" * 60)
    
    model = build_fixed_cnn_model(vocab_size, embedding_matrix, config)
    
    # FIXED: Compile with focal loss and improved optimizer
    lr_schedule = create_lr_schedule(config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
        loss=focal_loss(gamma=config.FOCAL_GAMMA, alpha=config.FOCAL_ALPHA),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    print(model.summary())
    
    # FIXED: Training with all improvements
    print("\n" + "=" * 60)
    print("FIXED TRAINING")
    print("=" * 60)
    
    checkpoint_path = os.path.join(config.MODEL_DIR, 'best_fixed_model')
    
    callbacks_list = [
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_format='tf',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"Training FIXED model for up to {config.EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train_smooth,
        validation_data=(X_val, y_val_smooth),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks_list,
        class_weight=class_weights,  # FIXED: Add class weights
        verbose=1
    )
    
    # Save artifacts
    tokenizer_path = os.path.join(config.MODEL_DIR, 'tokenizer.pickle')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save config
    config_dict = {
        'EMOTION_COLUMNS': config.EMOTION_COLUMNS,
        'MAX_SEQUENCE_LENGTH': config.MAX_SEQUENCE_LENGTH,
        'MAX_VOCAB_SIZE': config.MAX_VOCAB_SIZE,
        'EMBEDDING_DIM': config.EMBEDDING_DIM,
        'CNN_FILTERS': config.CNN_FILTERS,
        'CNN_KERNEL_SIZES': config.CNN_KERNEL_SIZES,
        'HIDDEN_DIM': config.HIDDEN_DIM,
        'DROPOUT_RATE': config.DROPOUT_RATE,
        'BATCH_SIZE': config.BATCH_SIZE,
        'LEARNING_RATE': config.LEARNING_RATE,
        'EPOCHS': config.EPOCHS,
        'FOCAL_GAMMA': config.FOCAL_GAMMA,
        'FOCAL_ALPHA': config.FOCAL_ALPHA,
        'LABEL_SMOOTHING': config.LABEL_SMOOTHING
    }
    
    config_path = os.path.join(config.MODEL_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("\n✅ FIXED model training complete!")
    print(f"✅ Model saved: {checkpoint_path}")
    print(f"✅ Tokenizer saved: {tokenizer_path}")
    print(f"✅ Config saved: {config_path}")
    
    return history, model, (X_test, y_test), tokenizer

if __name__ == "__main__":
    print("Starting FIXED CNN+GloVe training...")
    history, model, test_data, tokenizer = train_fixed_model()
    print("\n🎉 FIXED training pipeline completed!")