# GoEmotions Multi-Model Emotion Classifier

A comprehensive emotion classification system that compares three different deep learning approaches on the GoEmotions dataset. This project implements and evaluates BERT, CNN+GloVe, and Traditional ML models for multi-label emotion classification with a professional Streamlit web interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Transformers](https://img.shields.io/badge/transformers-4.21+-orange.svg)

## ✨ Features

### 🚀 Core Functionality
- **Multi-Model Support**: Compare predictions from BERT, CNN+GloVe, and Traditional ML models
- **Interactive Web Interface**: Modern Streamlit app with intuitive tab navigation
- **Real-time Predictions**: Get instant emotion predictions from your text
- **Batch Processing**: Analyze multiple texts simultaneously with comprehensive summary dashboard
- **Multilingual Support**: Built-in translation for non-English text analysis
- **Professional Demo Pages**: Ready-to-use HTML demo interfaces for presentations

### 📊 Advanced Analytics
- **Comprehensive Summary Dashboard**: Key metrics, emotion distribution, and model agreement analysis
- **Confidence Analysis**: Risk assessment with automatic flagging of low-confidence predictions
- **Model Agreement Visualization**: Trust indicators when multiple models agree/disagree
- **Executive Summary**: Auto-generated business insights for decision makers
- **Performance Metrics**: Detailed comparisons between models with interactive charts

### 🎭 Emotion Detection
The system classifies text into **28 emotion categories**:
- **Positive**: admiration, amusement, approval, caring, excitement, gratitude, joy, love, optimism, pride, relief
- **Negative**: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
- **Complex**: confusion, curiosity, desire, realization, surprise
- **Neutral**: neutral

## 🤖 Models

### 1. BERT (Transformer-based)
- **Architecture**: Fine-tuned BERT base model  
- **Strengths**: Best for capturing contextual nuances and semantic relationships
- **Performance**: F1 Macro: ~0.52, F1 Micro: ~0.64
- **Inference Time**: ~100ms per text

### 2. CNN + GloVe (Word Embeddings)
- **Architecture**: Convolutional Neural Network with pre-trained GloVe embeddings
- **Strengths**: Good balance between speed and accuracy, handles various text lengths well
- **Performance**: F1 Macro: ~0.47, F1 Micro: ~0.60  
- **Inference Time**: ~20ms per text

### 3. Traditional ML (Classical Approaches)
- **Architecture**: Ensemble of classical ML algorithms (SVM, Logistic Regression, Random Forest)
- **Strengths**: Fastest inference time, interpretable features, low resource requirements
- **Performance**: F1 Macro: ~0.45, F1 Micro: ~0.58
- **Inference Time**: ~5ms per text

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (recommended)
- CUDA-capable GPU (optional, for faster BERT inference)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/GoEmotions_NLP_Project.git
cd GoEmotions_NLP_Project
```

2. **Create and activate virtual environment:**
```bash
python -m venv goemotions_env

# Windows
.\goemotions_env\Scripts\activate

# Linux/Mac  
source goemotions_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables (optional for translation):**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Gemini API key
# Get key from: https://makersuite.google.com/app/apikey
```

5. **Download model files** (not included in repository due to size):

⚠️ **Important**: Model files (~1.7GB) are not included in this repository.

**Option A: Train your own models**
- Use the provided Jupyter notebooks in `notebooks/` directory
- Follow training pipelines: 01_Traditional_ML → 02_Word_Embedding → 03_BERT

**Option B: Download pre-trained models** (if available)
- Contact repository maintainer for model files
- Or download from [provide your link here - Google Drive, Hugging Face, etc.]
- Extract to `models/` directory maintaining the structure

6. **Download GloVe embeddings** (if using CNN+GloVe model):
```bash
# Download glove.6B.300d.txt from: https://nlp.stanford.edu/projects/glove/
# Place in embeddings/ folder
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings/
```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Usage Guide

### 1. Quick Analysis
- Enter any text in the input box
- Select which models to use for prediction
- Get instant emotion analysis with confidence scores
- Compare results across different AI models

### 2. Batch Processing  
- Upload CSV files with a `text` column
- Process hundreds of texts simultaneously
- Get comprehensive summary dashboard with:
  - Key metrics (most common emotion, average confidence)
  - Interactive emotion distribution charts
  - Model agreement analysis
  - Executive summary with business insights

### 3. Model Settings
- Configure AI models and thresholds
- Enable/disable translation for non-English text
- Adjust confidence thresholds for predictions
- View model loading status and performance

## 🎪 Demo Features

### Professional Demo Pages
- **English Demo**: CloudAI Pro product review interface (`assets/demo_review_page.html`)
- **Malay Demo**: SmartLearn Pro education platform (`assets/demo_review_page_malay.html`)
- **Complete Workflow**: From data collection → CSV export → AI analysis → business insights

### Demo Presentation Workflow
1. **Show realistic product review interface** with varied customer feedback
2. **Add new review live** during presentation to demonstrate interactivity
3. **Export CSV data** from the demo page
4. **Import to Streamlit** batch analysis tab
5. **Display AI analysis results** with professional dashboard
6. **Present business insights** from executive summary

Perfect for demonstrating practical AI applications to stakeholders, teachers, or clients.

## 📂 Project Structure

```
GoEmotions_NLP_Project/
├── README.md                     # This file
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore patterns
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── assets/                       # Demo files and assets
│   ├── demo_review_page.html     # English demo interface
│   └── demo_review_page_malay.html  # Malay demo interface
├── data/                         # Dataset and test data
│   ├── go_emotions_dataset.csv
│   └── test_batch.csv            # Sample data for testing
├── docs/                         # Technical documentation
│   ├── CNN_GLOVE_CONFIDENCE.md   # Model technical details
│   └── DEPLOYMENT_NOTES.md       # Deployment instructions
├── embeddings/                   # GloVe embeddings
│   └── glove.6B.300d.txt
├── models/                       # Trained models
│   ├── bert/                     # BERT model files
│   ├── traditional_ml/           # Classical ML models
│   └── word_embedding/           # CNN+GloVe model
├── model_loaders/                # Model loading utilities
│   ├── bert_loader.py
│   ├── embedding_loader.py
│   ├── ensemble_loader.py
│   ├── traditional_ml_loader.py
│   ├── calibration.py
│   ├── emotion_hierarchy.py
│   └── utils.py
├── notebooks/                    # Training notebooks
│   ├── 01_Traditional_ML_Pipeline.ipynb
│   ├── 02_Word_Embedding_Pipeline.ipynb
│   └── 03_BERT_Pipeline_GoEmotions.ipynb
├── results/                      # Model evaluation results
│   ├── bert_results/
│   ├── traditional_ml_results/
│   └── word_embedding_results/
├── scripts/                      # Utility scripts
│   ├── 02_Word_Embedding_Pipeline.py
│   ├── advanced_inference_optimizer.py
│   ├── collect_calibration_data.py
│   └── compare_model_performance.py
├── tests/                        # Unit and integration tests
│   └── README.md
├── ui_components/                # Modular UI components
│   ├── navigation.py             # Tab navigation system
│   ├── results_display.py        # Enhanced result displays
│   └── styles.py                 # WCAG-compliant styling
└── utils/                        # Utility functions
    ├── feature_engineering.py
    ├── improved_predictor.py
    ├── preprocessing.py
    ├── translator.py             # Multi-language support
    └── languages.py
```

## 🏗️ Technical Architecture

### UI Architecture (Post-Refactor)
- **Modern Tab Navigation** replacing vertical expanders for better UX
- **WCAG AA Compliant Colors** ensuring accessibility for all users
- **Responsive Design** optimized for both desktop and mobile
- **Modular Components** with organized codebase in `ui_components/`

### Model Loading System
- **Lazy Loading**: Models loaded only when needed
- **Error Handling**: Graceful fallbacks and user-friendly error messages
- **Caching**: Efficient model state management with Streamlit caching
- **Compatibility**: NumPy version migration handling for older model files

### Batch Processing Pipeline
- **Progress Tracking**: Real-time progress bars for large datasets
- **Memory Efficiency**: Streaming processing for large CSV files
- **Result Aggregation**: Comprehensive analytics and visualization
- **Export Functionality**: Results downloadable in multiple formats

## 📊 Performance Benchmarks

| Model | F1 Macro | F1 Micro | Precision | Recall | Inference Time | Memory Usage |
|-------|----------|----------|-----------|---------|----------------|--------------|
| BERT | 0.52 | 0.64 | 0.58 | 0.61 | ~100ms | ~2GB |
| CNN+GloVe | 0.47 | 0.60 | 0.53 | 0.56 | ~20ms | ~500MB |
| Traditional ML | 0.45 | 0.58 | 0.51 | 0.54 | ~5ms | ~100MB |

**Note**: Benchmarks measured on Intel i7 CPU with 16GB RAM. GPU acceleration available for BERT model.

## 🔧 Advanced Configuration

### Environment Variables
Create a `.env` file for configuration (copy from `.env.example`):
```bash
# Gemini API for translation (optional)
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here
```

The translation feature is optional - the app works without it, but translation improves emotion analysis for non-English text.

### Custom Model Integration
The system supports loading custom trained models:
```python
# Add your model to model_loaders/
class CustomModelLoader:
    def load_model(self):
        # Your model loading logic here
        pass
```

## 🧪 Training New Models

Use the provided Jupyter notebooks to retrain models with your own data:

1. **Traditional ML**: `notebooks/01_Traditional_ML_Pipeline.ipynb`
   - Feature extraction and classical ML training
   - Hyperparameter tuning and cross-validation

2. **CNN+GloVe**: `notebooks/02_Word_Embedding_Pipeline.ipynb`  
   - Word embedding model training
   - Architecture optimization

3. **BERT**: `notebooks/03_BERT_Pipeline_GoEmotions.ipynb`
   - Transformer fine-tuning
   - Advanced training techniques

## 🌍 Dataset Information

This project uses the **[GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions)** by Google Research:
- **58,009 Reddit comments** carefully curated and labeled
- **27 emotion categories + neutral** for comprehensive coverage  
- **High inter-annotator agreement** ensuring label quality
- **Diverse topics and contexts** for robust model training

## 🛠️ Troubleshooting

### Common Issues

**NLTK Data Error**
```bash
# The app will automatically download required packages on first run
# Manual download if needed:
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Memory Issues** 
- Reduce batch size in settings
- Load models one at a time
- Use Traditional ML model for resource-constrained environments

**GPU Support**
- BERT automatically uses GPU if available
- To force CPU usage, set `device = torch.device('cpu')` in model loader

**Translation Issues**
- Obtain Gemini API key from https://makersuite.google.com/app/apikey
- Copy `.env.example` to `.env` and add your API key
- Translation feature is optional - app works without it

## 📈 Business Applications

### Use Cases
- **Customer Service**: Analyze support tickets and feedback for emotion trends
- **Product Development**: Monitor user sentiment towards features and updates  
- **Marketing**: Understand emotional response to campaigns and content
- **HR Analytics**: Assess employee satisfaction and engagement
- **Social Media**: Track brand sentiment and public opinion
- **Content Moderation**: Identify potentially harmful or toxic content

### ROI Benefits
- **Time Savings**: Automate hours of manual sentiment analysis
- **Scalability**: Process thousands of texts in minutes
- **Accuracy**: AI models provide consistent, unbiased analysis
- **Insights**: Discover emotion patterns invisible to human reviewers

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Submit bug reports and feature requests via GitHub Issues
- Fork the repository and submit Pull Requests
- Improve documentation and add examples
- Share your trained models and improvements

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black . --line-length 100
isort . --profile black
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Research** for the GoEmotions dataset
- **Hugging Face** for the transformers library and model hub
- **Stanford NLP** for GloVe word embeddings
- **Streamlit** for the amazing web application framework
- **PyTorch** and **scikit-learn** for machine learning foundations

## 📬 Contact

For questions, feedback, or collaboration opportunities:
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and community support

---

<div align="center">

**Built with ❤️ for advancing emotion AI research and applications**

[🚀 **Try Live Demo**](https://your-demo-url.com) | [📚 **Read Docs**](docs/) | [🎯 **View Examples**](examples/)

</div>