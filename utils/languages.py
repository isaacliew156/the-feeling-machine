"""
Multi-language support for GoEmotions Emotion Analysis
Provides translations for UI text and emotion labels
"""

# Supported languages
LANGUAGES = {
    'en': 'English',
    'ms': 'Bahasa Melayu',
    'zh': '中文'
}

# Emotion label translations for all 28 emotions
EMOTION_TRANSLATIONS = {
    'en': {
        'admiration': 'Admiration',
        'amusement': 'Amusement', 
        'anger': 'Anger',
        'annoyance': 'Annoyance',
        'approval': 'Approval',
        'caring': 'Caring',
        'confusion': 'Confusion',
        'curiosity': 'Curiosity',
        'desire': 'Desire',
        'disappointment': 'Disappointment',
        'disapproval': 'Disapproval',
        'disgust': 'Disgust',
        'embarrassment': 'Embarrassment',
        'excitement': 'Excitement',
        'fear': 'Fear',
        'gratitude': 'Gratitude',
        'grief': 'Grief',
        'joy': 'Joy',
        'love': 'Love',
        'nervousness': 'Nervousness',
        'optimism': 'Optimism',
        'pride': 'Pride',
        'realization': 'Realization',
        'relief': 'Relief',
        'remorse': 'Remorse',
        'sadness': 'Sadness',
        'surprise': 'Surprise',
        'neutral': 'Neutral'
    },
    'ms': {
        'admiration': 'Kekaguman',
        'amusement': 'Hiburan',
        'anger': 'Marah',
        'annoyance': 'Jengkel',
        'approval': 'Persetujuan',
        'caring': 'Prihatin',
        'confusion': 'Keliru',
        'curiosity': 'Ingin Tahu',
        'desire': 'Keinginan',
        'disappointment': 'Kecewa',
        'disapproval': 'Tidak Setuju',
        'disgust': 'Jijik',
        'embarrassment': 'Malu',
        'excitement': 'Teruja',
        'fear': 'Takut',
        'gratitude': 'Syukur',
        'grief': 'Kesedihan',
        'joy': 'Gembira',
        'love': 'Cinta',
        'nervousness': 'Gugup',
        'optimism': 'Optimis',
        'pride': 'Bangga',
        'realization': 'Sedar',
        'relief': 'Lega',
        'remorse': 'Penyesalan',
        'sadness': 'Sedih',
        'surprise': 'Terkejut',
        'neutral': 'Neutral'
    },
    'zh': {
        'admiration': '钦佩',
        'amusement': '愉快',
        'anger': '愤怒',
        'annoyance': '烦恼',
        'approval': '赞同',
        'caring': '关心',
        'confusion': '困惑',
        'curiosity': '好奇',
        'desire': '渴望',
        'disappointment': '失望',
        'disapproval': '不赞同',
        'disgust': '厌恶',
        'embarrassment': '尴尬',
        'excitement': '兴奋',
        'fear': '恐惧',
        'gratitude': '感激',
        'grief': '悲伤',
        'joy': '喜悦',
        'love': '爱',
        'nervousness': '紧张',
        'optimism': '乐观',
        'pride': '自豪',
        'realization': '领悟',
        'relief': '宽慰',
        'remorse': '懊悔',
        'sadness': '悲伤',
        'surprise': '惊讶',
        'neutral': '中性'
    }
}

# UI text translations
UI_TRANSLATIONS = {
    'en': {
        # Main title and subtitle
        'title': 'The Feeling Machine',
        'subtitle': 'AI-powered emotion analysis that understands the nuances of human expression',
        
        # Input section
        'enter_text': '✨ Enter your text:',
        'text_placeholder': 'Type your text here and discover its emotional signature...',
        'characters': 'characters',
        'words': 'words',
        'tip': 'Tip: After typing, click elsewhere to enable the analyze button',
        'analyze_button': '🚀 ANALYZE EMOTIONS',
        
        # Settings
        'settings': '⚙️ Model Settings & Configuration',
        'choose_models': '🤖 Choose AI Models',
        'choose_models_desc': 'Select one or more models to compare their emotion predictions',
        'select_models': 'Select Models:',
        'load_models': '🚀 Load Selected Models',
        'optimal_thresholds': '🎯 Use Optimal Thresholds',
        'optimal_help': 'Recommended: Use model-optimized thresholds',
        'confidence_threshold': 'Confidence Threshold',
        
        # Text Analysis
        'text_analysis': '🔍 Text Analysis',
        'text_analysis_desc': 'Analyze emotions in text using state-of-the-art models',
        'choose_input_method': 'Choose input method:',
        'type_custom_text': 'Type custom text',
        'select_example_text': 'Select example text',
        
        # Batch Analysis
        'batch_analysis': 'Batch Analysis (Advanced)',
        
        # Navigation tabs
        'quick_analysis': 'Quick Analysis',
        'quick_analysis_desc': 'Analyze emotions in text using state-of-the-art AI models',
        'model_settings': 'Model Settings',
        'model_settings_desc': 'Configure AI models, thresholds, and translation settings',
        'batch_analysis_desc': 'Upload and process multiple texts from CSV files for comprehensive analysis',
        'history': 'History',
        'history_desc': 'View your recent predictions and analysis results',
        'no_history': 'No predictions yet. Analyze some texts to see your history here!',
        
        # Translation section
        'translation_settings': '🌐 Translation Settings',
        'auto_translate': '🔄 Auto-translate to English',
        'auto_translate_help': 'Automatically translate non-English text for better emotion analysis',
        'gemini_api_key': 'Gemini API Key:',
        'api_key_help': 'Enter your Google Gemini API key. Get one at: https://makersuite.google.com/app/apikey',
        'api_configured': '✅ API key configured successfully!',
        'api_not_available': '❌ API key not available',
        'clear_cache': '🗑️ Clear Cache',
        'cache_cleared': 'Cache cleared!',
        'cache_stats': '📊 Cache: {count} translations stored',
        'restart_tip': '💡 Restart if language detection seems incorrect',
        
        # Language section
        'language_settings': '🌐 Language Settings',
        'ui_language': '🎨 Interface Language',
        'ui_language_help': 'Select language for interface and emotion labels',
        
        # Results
        'results_title': '🎯 Prediction Results',
        'input_text': 'Input Text:',
        'max_confidence': 'Max Confidence',
        'prediction_time': 'Prediction Time',
        'top_emotions': 'Top 5 Emotions:',
        'predicted': 'Predicted:',
        'no_results': 'No prediction results to display',
        
        # Language detection
        'language_detection': '🌐 Language Detection & Translation',
        'detected_language': '🔍 Detected Language',
        'status': 'Status',
        'translated': 'Translated',
        'no_translation': 'No translation needed',
        'translation_quality': '🎯 Translation Quality',
        'original_vs_translated': '📝 Original vs Translated Text',
        'original_text': 'Original Text:',
        'translated_text': 'Translated Text:',
        
        # Model info
        'emotions_detected': '🎭\n28 Emotions',
        'emotions_desc': 'From admiration to neutral, our AI models detect the full spectrum of human emotions',
        
        # Error messages
        'models_not_loaded': 'Please select at least one model',
        'all_models_loaded': '✅ All {count} models loaded successfully!',
        'partial_models_loaded': '⚠️ {success}/{total} models loaded',
        'no_models_loaded': '❌ No models could be loaded',
        'translation_failed': '⚠️ Translation enabled but API key not available. Using original text.',
        'warning': '⚠️ {message}'
    },
    'ms': {
        # Main title and subtitle
        'title': 'Mesin Perasaan',
        'subtitle': 'Analisis emosi berkuasa AI yang memahami nuansa ekspresi manusia',
        
        # Input section
        'enter_text': '✨ Masukkan teks anda:',
        'text_placeholder': 'Taip teks anda di sini dan temui tandatangan emosi...',
        'characters': 'aksara',
        'words': 'perkataan',
        'tip': 'Petua: Selepas menaip, klik di tempat lain untuk membolehkan butang analisis',
        'analyze_button': '🚀 ANALISIS EMOSI',
        
        # Settings
        'settings': '⚙️ Tetapan Model & Konfigurasi',
        'choose_models': '🤖 Pilih Model AI',
        'choose_models_desc': 'Pilih satu atau lebih model untuk membandingkan ramalan emosi mereka',
        'select_models': 'Pilih Model:',
        'load_models': '🚀 Muatkan Model Terpilih',
        'optimal_thresholds': '🎯 Gunakan Ambang Optimum',
        'optimal_help': 'Disyorkan: Gunakan ambang yang dioptimumkan model',
        'confidence_threshold': 'Ambang Keyakinan',
        
        # Text Analysis
        'text_analysis': '🔍 Analisis Teks',
        'text_analysis_desc': 'Analisis emosi dalam teks menggunakan model terkini',
        'choose_input_method': 'Pilih kaedah input:',
        'type_custom_text': 'Taip teks khusus',
        'select_example_text': 'Pilih teks contoh',
        
        # Batch Analysis
        'batch_analysis': 'Analisis Berkelompok (Lanjutan)',
        
        # Navigation tabs
        'quick_analysis': 'Analisis Cepat',
        'quick_analysis_desc': 'Analisis emosi dalam teks menggunakan model AI terkini',
        'model_settings': 'Tetapan Model',
        'model_settings_desc': 'Konfigurasi model AI, ambang, dan tetapan terjemahan',
        'batch_analysis_desc': 'Muat naik dan proses berbilang teks dari fail CSV untuk analisis menyeluruh',
        'history': 'Sejarah',
        'history_desc': 'Lihat ramalan dan hasil analisis terkini anda',
        'no_history': 'Tiada ramalan lagi. Analisis beberapa teks untuk melihat sejarah di sini!',
        
        # Translation section
        'translation_settings': '🌐 Tetapan Terjemahan',
        'auto_translate': '🔄 Auto-terjemah ke Bahasa Inggeris',
        'auto_translate_help': 'Terjemahkan teks bukan Bahasa Inggeris secara automatik untuk analisis emosi yang lebih baik',
        'gemini_api_key': 'Kunci API Gemini:',
        'api_key_help': 'Masukkan kunci API Google Gemini anda. Dapatkan di: https://makersuite.google.com/app/apikey',
        'api_configured': '✅ Kunci API dikonfigurasi dengan jayanya!',
        'api_not_available': '❌ Kunci API tidak tersedia',
        'clear_cache': '🗑️ Kosongkan Cache',
        'cache_cleared': 'Cache dikosongkan!',
        'cache_stats': '📊 Cache: {count} terjemahan tersimpan',
        'restart_tip': '💡 Mula semula jika pengesanan bahasa nampak salah',
        
        # Language section
        'language_settings': '🌐 Tetapan Bahasa',
        'ui_language': '🎨 Bahasa Antara Muka',
        'ui_language_help': 'Pilih bahasa untuk antara muka dan label emosi',
        
        # Results
        'results_title': '🎯 Keputusan Ramalan',
        'input_text': 'Teks Input:',
        'max_confidence': 'Keyakinan Maksimum',
        'prediction_time': 'Masa Ramalan',
        'top_emotions': '5 Emosi Teratas:',
        'predicted': 'Diramal:',
        'no_results': 'Tiada keputusan ramalan untuk dipaparkan',
        
        # Language detection
        'language_detection': '🌐 Pengesanan Bahasa & Terjemahan',
        'detected_language': '🔍 Bahasa Dikesan',
        'status': 'Status',
        'translated': 'Diterjemahkan',
        'no_translation': 'Tiada terjemahan diperlukan',
        'translation_quality': '🎯 Kualiti Terjemahan',
        'original_vs_translated': '📝 Teks Asal vs Terjemahan',
        'original_text': 'Teks Asal:',
        'translated_text': 'Teks Terjemahan:',
        
        # Model info
        'emotions_detected': '🎭\n28 Emosi',
        'emotions_desc': 'Dari kekaguman hingga neutral, model AI kami mengesan spektrum penuh emosi manusia',
        
        # Error messages
        'models_not_loaded': 'Sila pilih sekurang-kurangnya satu model',
        'all_models_loaded': '✅ Semua {count} model dimuatkan dengan jayanya!',
        'partial_models_loaded': '⚠️ {success}/{total} model dimuatkan',
        'no_models_loaded': '❌ Tiada model dapat dimuatkan',
        'translation_failed': '⚠️ Terjemahan dibolehkan tetapi kunci API tidak tersedia. Menggunakan teks asal.',
        'warning': '⚠️ {message}'
    },
    'zh': {
        # Main title and subtitle
        'title': '情感机器',
        'subtitle': 'AI驱动的情感分析，理解人类表达的细微差别',
        
        # Input section
        'enter_text': '✨ 输入文本：',
        'text_placeholder': '在此输入文本，发现其情感特征...',
        'characters': '字符',
        'words': '单词',
        'tip': '提示：输入后，点击其他地方启用分析按钮',
        'analyze_button': '🚀 分析情感',
        
        # Settings
        'settings': '⚙️ 模型设置与配置',
        'choose_models': '🤖 选择AI模型',
        'choose_models_desc': '选择一个或多个模型来比较它们的情感预测',
        'select_models': '选择模型：',
        'load_models': '🚀 加载选定模型',
        'optimal_thresholds': '🎯 使用最优阈值',
        'optimal_help': '推荐：使用模型优化的阈值',
        'confidence_threshold': '置信度阈值',
        
        # Text Analysis
        'text_analysis': '🔍 文本分析',
        'text_analysis_desc': '使用最先进的模型分析文本中的情感',
        'choose_input_method': '选择输入方式：',
        'type_custom_text': '输入自定义文本',
        'select_example_text': '选择示例文本',
        
        # Batch Analysis
        'batch_analysis': '批量分析（高级）',
        
        # Navigation tabs
        'quick_analysis': '快速分析',
        'quick_analysis_desc': '使用最先进的AI模型分析文本中的情感',
        'model_settings': '模型设置',
        'model_settings_desc': '配置AI模型、阈值和翻译设置',
        'batch_analysis_desc': '上传和处理CSV文件中的多个文本进行综合分析',
        'history': '历史记录',
        'history_desc': '查看您最近的预测和分析结果',
        'no_history': '还没有预测记录。分析一些文本以在此查看您的历史记录！',
        
        # Translation section
        'translation_settings': '🌐 翻译设置',
        'auto_translate': '🔄 自动翻译成英文',
        'auto_translate_help': '自动翻译非英文文本以获得更好的情感分析效果',
        'gemini_api_key': 'Gemini API密钥：',
        'api_key_help': '输入您的Google Gemini API密钥。获取地址：https://makersuite.google.com/app/apikey',
        'api_configured': '✅ API密钥配置成功！',
        'api_not_available': '❌ API密钥不可用',
        'clear_cache': '🗑️ 清理缓存',
        'cache_cleared': '缓存已清理！',
        'cache_stats': '📊 缓存：已存储{count}个翻译',
        'restart_tip': '💡 如果语言检测似乎不正确，请重启',
        
        # Language section
        'language_settings': '🌐 语言设置',
        'ui_language': '🎨 界面语言',
        'ui_language_help': '选择界面和情感标签的语言',
        
        # Results
        'results_title': '🎯 预测结果',
        'input_text': '输入文本：',
        'max_confidence': '最大置信度',
        'prediction_time': '预测时间',
        'top_emotions': '前5个情感：',
        'predicted': '预测：',
        'no_results': '无预测结果可显示',
        
        # Language detection
        'language_detection': '🌐 语言检测与翻译',
        'detected_language': '🔍 检测到的语言',
        'status': '状态',
        'translated': '已翻译',
        'no_translation': '无需翻译',
        'translation_quality': '🎯 翻译质量',
        'original_vs_translated': '📝 原文与译文对比',
        'original_text': '原文：',
        'translated_text': '译文：',
        
        # Model info
        'emotions_detected': '🎭\n28种情感',
        'emotions_desc': '从钦佩到中性，我们的AI模型检测人类情感的完整光谱',
        
        # Error messages
        'models_not_loaded': '请至少选择一个模型',
        'all_models_loaded': '✅ 所有{count}个模型加载成功！',
        'partial_models_loaded': '⚠️ {success}/{total}个模型已加载',
        'no_models_loaded': '❌ 无法加载任何模型',
        'translation_failed': '⚠️ 已启用翻译但API密钥不可用。使用原文。',
        'warning': '⚠️ {message}'
    }
}

# Helper functions
def get_text(key: str, lang: str = 'en', **kwargs) -> str:
    """
    Get translated UI text
    
    Args:
        key: Translation key
        lang: Language code ('en', 'ms', 'zh')
        **kwargs: Format parameters for the text
        
    Returns:
        Translated text, fallback to English if key not found
    """
    try:
        text = UI_TRANSLATIONS[lang].get(key, UI_TRANSLATIONS['en'].get(key, key))
        if kwargs:
            return text.format(**kwargs)
        return text
    except (KeyError, ValueError):
        return key

def get_emotion_label(emotion: str, lang: str = 'en') -> str:
    """
    Get translated emotion label
    
    Args:
        emotion: Emotion key (e.g., 'anger', 'joy')
        lang: Language code ('en', 'ms', 'zh')
        
    Returns:
        Translated emotion label, fallback to English if not found
    """
    return EMOTION_TRANSLATIONS.get(lang, {}).get(emotion, 
           EMOTION_TRANSLATIONS['en'].get(emotion, emotion))

def get_available_languages() -> dict:
    """Get available languages dictionary"""
    return LANGUAGES.copy()

def is_supported_language(lang: str) -> bool:
    """Check if language is supported"""
    return lang in LANGUAGES