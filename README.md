# The Feeling Machine: A Comparative Study of Multi-Model Emotion Classification

A comprehensive emotion classification system comparing three distinct NLP approaches on the GoEmotions dataset. This project implements Traditional Machine Learning, CNN with GloVe embeddings, and BERT transformer models for multi-label emotion classification across 28 fine-grained emotion categories.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Transformers](https://img.shields.io/badge/transformers-4.21+-orange.svg)

## Overview

This project addresses the challenge of fine-grained emotion detection in user-generated text, moving beyond traditional sentiment polarity (positive/negative/neutral) to capture nuanced emotional complexity. The system implements and evaluates three distinct approaches to multi-label emotion classification, providing insights into the trade-offs between model complexity, accuracy, and computational efficiency.

**Project Details:**
- Programme: RSWY3S1
- Course: BMCS2003 Artificial Intelligence (202505 Session, Year 2025/26)
- Team Members: Liew Yi Shen (24WMR01484), Lim Huan Qian (24WMR12198)

## Core Capabilities

**Multi-Model Architecture**
- Three distinct approaches: Traditional ML (Multinomial Naive Bayes), Deep Learning (CNN+GloVe), and Transformer-based (BERT)
- Comparative performance analysis across multiple evaluation metrics
- Parallel inference supporting real-time model comparison

**Emotion Detection**
- 28 emotion categories: 27 distinct emotions plus neutral
- Multi-label classification supporting simultaneous emotion detection
- Fine-grained taxonomy including admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, and neutral

**Interactive Platform**
- Streamlit-based web interface for real-time emotion analysis
- Batch processing with CSV upload capability
- Comprehensive visualization of emotion distributions and model predictions
- Confidence scoring and model agreement analysis

## Model Specifications

### BERT (Transformer-based)
**Architecture:** Fine-tuned bert-base-uncased with custom classification head
- 12 transformer layers, 768 hidden dimensions, 12 attention heads
- 109.5M parameters
- Classification head: Linear layer (768→28) with sigmoid activation
- Training: 3 epochs, AdamW optimizer (lr=2e-5), FP16 mixed precision

**Performance:**
- F1-Macro: 0.3726
- F1-Micro: 0.3897
- Precision (Macro): 0.3317
- Recall (Macro): 0.4839
- Inference Time: ~100ms per text
- Optimal Threshold: 0.65

**Strengths:** Superior contextual understanding, best performance on complex emotional expressions

### CNN + GloVe (Deep Learning)
**Architecture:** Multi-kernel Convolutional Neural Network with pre-trained word embeddings
- 300-dimensional GloVe embeddings (6B tokens, 87.6% vocabulary coverage)
- Parallel 1D convolutions (kernel sizes: 2, 3, 4; 128 filters each)
- 6.48M total parameters
- Sequence length: 50 tokens

**Performance:**
- F1-Macro: 0.3099
- F1-Micro: 0.4075
- Precision (Macro): 0.2883
- Recall (Macro): 0.3925
- Inference Time: ~20ms per text
- Optimal Threshold: 0.15

**Strengths:** Balance between accuracy and efficiency, handles variable text lengths effectively

### Traditional ML (Classical Approach)
**Architecture:** Multinomial Naive Bayes with engineered features
- 17,056-dimensional feature space
  - TF-IDF vectors: 10,000 word n-grams + 2,000 character n-grams
  - Count vectors: 5,000 word/bigram frequencies
  - Linguistic features: 56 hand-crafted features (sentiment scores, punctuation, emotion lexicon, pronoun usage)
- OneVsRestClassifier for multi-label classification

**Performance:**
- F1-Macro: 0.3081
- F1-Micro: 0.3589
- Precision (Macro): 0.2584
- Recall (Macro): 0.3964
- Inference Time: ~5ms per text

**Strengths:** Fastest inference, interpretable features, minimal resource requirements

## Installation

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for BERT)
- CUDA-capable GPU (optional, improves BERT inference speed)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GoEmotions_NLP_Project.git
cd GoEmotions_NLP_Project
```

2. Create virtual environment:
```bash
python -m venv goemotions_env

# Windows
.\goemotions_env\Scripts\activate

# Linux/Mac
source goemotions_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download GloVe embeddings (required for CNN model):
```bash
# Download glove.6B.300d.txt from Stanford NLP
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings/
```

5. Model files:
The trained models are not included in the repository due to file size constraints (~1.7GB total).

**Option A:** Train models using provided Jupyter notebooks in `notebooks/` directory
- `01_Traditional_ML_Pipeline.ipynb`
- `02_Word_Embedding_Pipeline.ipynb`
- `03_BERT_Pipeline_GoEmotions.ipynb`

**Option B:** Contact repository maintainers for pre-trained model files

6. Environment variables (optional for translation features):
```bash
cp .env.example .env
# Add Gemini API key if using translation: https://makersuite.google.com/app/apikey
```

### Running the Application

```bash
streamlit run app.py
```

Access the interface at `http://localhost:8501`

## Usage

### Single Text Analysis
1. Enter text in the input field
2. Select models for comparison
3. View emotion predictions with confidence scores
4. Compare results across different architectures

### Batch Processing
1. Prepare CSV file with `text` column
2. Upload via batch processing interface
3. Process multiple texts simultaneously
4. Download comprehensive analysis results

### Configuration
- Adjust confidence thresholds in model settings
- Enable/disable translation for non-English text
- Configure batch processing parameters
- View model loading status

## Project Structure

```
GoEmotions_NLP_Project/
├── app.py                              # Streamlit application
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment specification
│
├── data/
│   └── go_emotions_dataset.csv         # GoEmotions dataset
│
├── embeddings/
│   └── glove.6B.300d.txt               # Pre-trained GloVe embeddings
│
├── models/
│   ├── bert/                           # BERT model files and tokenizer
│   ├── traditional_ml/                 # Serialized classical ML models
│   └── word_embedding/                 # CNN+GloVe model and configuration
│
├── model_loaders/
│   ├── bert_loader.py                  # BERT model initialization
│   ├── embedding_loader.py             # CNN model loader
│   ├── traditional_ml_loader.py        # Classical ML loader
│   ├── calibration.py                  # Threshold optimization
│   └── utils.py                        # Shared utilities
│
├── notebooks/
│   ├── 01_Traditional_ML_Pipeline.ipynb
│   ├── 02_Word_Embedding_Pipeline.ipynb
│   └── 03_BERT_Pipeline_GoEmotions.ipynb
│
├── results/
│   ├── bert_results/
│   ├── traditional_ml_results/
│   └── word_embedding_results/
│
├── ui_components/
│   ├── navigation.py                   # Interface navigation system
│   ├── results_display.py              # Result visualization components
│   └── styles.py                       # WCAG-compliant styling
│
└── utils/
    ├── preprocessing.py                # Text preprocessing pipeline
    ├── translator.py                   # Multi-language support
    └── feature_engineering.py          # Feature extraction for Traditional ML
```

## Dataset Information

**GoEmotions Dataset** (Demszky et al., 2020)
- 207,814 manually annotated Reddit comments (2005-2019)
- 28 emotion categories (27 emotions + neutral)
- Multi-label annotations (average 1.20 labels per comment)
- Train/Validation/Test split: 70%/15%/15%

**Data Distribution:**
- Training: 145,469 samples
- Validation: 31,172 samples
- Test: 31,173 samples

**Class Imbalance:**
- Most frequent: neutral (26.6%, 55,298 samples)
- Least frequent: grief (0.04%, 88 samples in test set)
- Imbalance ratio: 82:1 (neutral:grief)

## Performance Analysis

### Comparative Results (Test Set: 31,173 samples)

| Metric | Traditional ML | CNN+GloVe | BERT |
|--------|---------------|-----------|------|
| F1-Macro | 0.3081 | 0.3099 | **0.3726** |
| F1-Micro | 0.3589 | 0.4075 | 0.3897 |
| Precision (Macro) | 0.2584 | 0.2883 | 0.3317 |
| Recall (Macro) | 0.3964 | 0.3925 | 0.4839 |
| Training Time | 40 sec | ~7 min | ~78 min |
| Inference Time | 5ms | 20ms | 100ms |

### Threshold Optimization Impact

| Model | Default F1-Macro | Optimal Threshold | Optimized F1-Macro | Improvement |
|-------|-----------------|-------------------|-------------------|-------------|
| CNN+GloVe | 0.1461 | 0.15 | 0.3099 | +112.1% |
| BERT | 0.3486 | 0.65 | 0.3726 | +6.9% |

### Per-Emotion Performance (BERT, Top 5 and Bottom 5)

**Best Performing:**
1. Gratitude: 0.768 (1,728 samples)
2. Amusement: 0.626 (1,433 samples)
3. Love: 0.611 (1,242 samples)
4. Admiration: 0.544 (2,605 samples)
5. Remorse: 0.529 (379 samples)

**Most Challenging:**
24. Pride: 0.206 (207 samples)
25. Realization: 0.202 (1,347 samples)
26. Relief: 0.179 (195 samples)
27. Nervousness: 0.173 (274 samples)
28. Grief: 0.062 (88 samples)

## Technical Architecture

### Model Loading System
- Lazy loading: models initialized only when required
- Streamlit caching for efficient state management
- NumPy version compatibility handling for legacy model files
- Graceful error handling with informative feedback

### Preprocessing Pipeline
- Text normalization (lowercasing, URL/mention removal)
- Tokenization adapted per model architecture
- Reddit-specific artifact handling
- Text length reduced by 45.1% while preserving emotional signals

### Batch Processing
- Progress tracking for large datasets
- Memory-efficient streaming for CSV files
- Comprehensive result aggregation and visualization
- Multi-format export capability

## Key Findings

1. **Model Performance:** BERT achieves 20.9% improvement over Traditional ML in F1-Macro, confirming the value of contextual embeddings for complex emotion detection

2. **Threshold Calibration:** Critical for multi-label classification; CNN model improved 112.1% with optimal threshold optimization

3. **Linguistic Distinctiveness:** Emotions with unique lexical markers (gratitude, love) outperform abstract emotions (realization) regardless of sample size

4. **Efficiency Trade-offs:** Traditional ML achieves 82.7% of BERT's performance at 5% of computational cost, suitable for real-time applications

5. **Class Imbalance Challenge:** Rare emotions remain difficult to detect despite architectural sophistication and weighted loss functions

## Limitations and Future Work

### Current Limitations
- Sub-0.4 F1-Macro scores indicate fundamental challenges in 28-category fine-grained classification
- Severe class imbalance limits rare emotion detection despite mitigation strategies
- Reddit-specific language patterns may not generalize to other domains
- BERT's 100ms inference time limits real-time deployment scenarios

### Future Directions
- Investigate lighter transformer architectures (DistilBERT, ALBERT) for improved efficiency
- Explore synthetic data generation for rare emotion categories
- Implement per-emotion threshold optimization rather than global calibration
- Develop hierarchical classification grouping related emotions
- Apply model quantization for faster BERT inference
- Investigate cross-lingual models using multilingual transformers

## Troubleshooting

### Common Issues

**NLTK Data Missing:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Memory Constraints:**
- Load models individually rather than simultaneously
- Reduce batch processing size
- Use Traditional ML model for resource-constrained environments

**GPU Support:**
- BERT automatically utilizes GPU when available
- Force CPU usage: set `device = torch.device('cpu')` in model loader

**Translation Features:**
- Requires Gemini API key (optional)
- Application functions without translation; improves non-English text analysis

## Applications

- Customer feedback analysis and support ticket classification
- Social media monitoring and brand sentiment tracking
- Mental health support systems (early distress pattern identification)
- Educational platforms (student engagement and confusion detection)
- Content moderation (identifying harmful or toxic content)
- Human-computer interaction research

## References

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 4040–4054. https://doi.org/10.18653/v1/2020.acl-main.372

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint*. https://arxiv.org/abs/1810.04805

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of EMNLP*, 1532–1543. https://aclanthology.org/D14-1162

## Acknowledgments

- Google Research for the GoEmotions dataset
- Hugging Face for transformers library and model hub
- Stanford NLP Group for GloVe word embeddings
- Streamlit for web application framework
- PyTorch and scikit-learn communities

## License

This project is licensed under the MIT License.

## Contact

For questions or collaboration opportunities, please use GitHub Issues for bug reports and feature requests.

---

*BMCS2003 Artificial Intelligence - 202505 Session, Year 2025/26*
