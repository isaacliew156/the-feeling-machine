"""
Gemini API Translation Module for GoEmotions Project
Provides translation functionality to convert non-English text to English
for better emotion analysis performance.
"""

import requests
import json
import time
import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass
import streamlit as st
import logging

# Setup logging
logger = logging.getLogger(__name__)

try:
    from langdetect import detect, DetectorFactory
    # Set seed for consistent language detection
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available. Language detection will be disabled.")

@dataclass
class TranslationResult:
    """Result of a translation operation"""
    original_text: str
    translated_text: str
    detected_language: Optional[str]
    confidence: float
    was_translated: bool
    error_message: Optional[str] = None

class GeminiTranslator:
    """
    Gemini API translator for converting text to English
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the translator
        
        Args:
            api_key: Gemini API key. If None, will try to get from session state or env
        """
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Cache for translations to avoid repeated API calls
        self._translation_cache: Dict[str, TranslationResult] = {}
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum 1 second between requests
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from various sources"""
        if self.api_key:
            return self.api_key
        
        # Try to get from Streamlit session state
        if 'gemini_api_key' in st.session_state:
            return st.session_state.gemini_api_key
        
        # Try to get from secrets (if using Streamlit Cloud)
        try:
            return st.secrets.get("GEMINI_API_KEY")
        except:
            pass
        
        return None
    
    def _detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the input text
        
        Args:
            text: Input text to detect language for
            
        Returns:
            Language code (e.g., 'en', 'zh', 'es') or None if detection fails
        """
        if not LANGDETECT_AVAILABLE:
            return None
            
        try:
            # Clean text for better detection
            clean_text = text.strip()
            if len(clean_text) < 3:
                return None
            
            # Check for Malay/Indonesian indicators first
            malay_indicators = [
                'kamu', 'saya', 'aku', 'dia', 'kita', 'mereka', 'ini', 'itu', 'adalah', 'dan', 
                'atau', 'tidak', 'bukan', 'sudah', 'akan', 'sedang', 'sangat', 'sekali',
                'juga', 'hanya', 'masih', 'belum', 'pernah', 'selalu', 'kadang',
                'baik', 'buruk', 'bagus', 'jelek', 'cantik', 'ganteng', 'jahat',
                'senang', 'sedih', 'marah', 'takut', 'cinta', 'sayang', 'rindu', 'kangen',
                'la', 'lah', 'nya', 'kan', 'dong', 'sih', 'kok', 'deh'  # particles
            ]
            
            text_lower = clean_text.lower()
            words = text_lower.split()
            
            # If contains Malay indicators, force Malay detection
            malay_word_count = sum(1 for word in words if any(indicator in word for indicator in malay_indicators))
            if malay_word_count > 0:
                # Determine if it's more Indonesian or Malay based on context
                # Default to Malay for Malaysian context
                return 'ms'
            
            detected = detect(clean_text)
            return detected
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None
    
    def _is_english(self, text: str) -> bool:
        """
        Check if text is likely English
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be English
        """
        # Check for common Malay/Indonesian words that might be misdetected as English
        malay_indicators = [
            'kamu', 'saya', 'aku', 'dia', 'kita', 'mereka', 'ini', 'itu', 'adalah', 'dan', 
            'atau', 'tidak', 'bukan', 'sudah', 'akan', 'sedang', 'sangat', 'sekali',
            'juga', 'hanya', 'masih', 'sudah', 'belum', 'pernah', 'selalu', 'kadang',
            'baik', 'buruk', 'bagus', 'jelek', 'cantik', 'ganteng', 'jahat', 'baik',
            'senang', 'sedih', 'marah', 'takut', 'cinta', 'sayang', 'rindu', 'kangen',
            'la', 'lah', 'nya', 'kan', 'dong', 'sih', 'kok', 'deh'  # common particles
        ]
        
        # Convert to lowercase for checking
        text_lower = text.lower()
        words = text_lower.split()
        
        # If contains Malay indicators, definitely not English
        malay_word_count = sum(1 for word in words if any(indicator in word for indicator in malay_indicators))
        if malay_word_count > 0:
            return False
        
        try:
            # Check if mostly ASCII
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
            if ascii_ratio < 0.8:
                return False
            
            # Use language detection if available
            if LANGDETECT_AVAILABLE:
                detected_lang = self._detect_language(text)
                # Be more strict about English detection
                if detected_lang in ['ms', 'id', 'tl']:  # Malay, Indonesian, Filipino
                    return False
                return detected_lang == 'en' if detected_lang else True
            
            # Fallback: assume ASCII text is English only if no Malay indicators
            return ascii_ratio > 0.9
            
        except Exception:
            return False
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _call_gemini_api(self, text: str) -> Optional[str]:
        """
        Make API call to Gemini for translation
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text or None if failed
        """
        api_key = self._get_api_key()
        if not api_key:
            logger.error("No API key available for Gemini translation")
            return None
        
        # Rate limiting
        self._rate_limit()
        
        # Prepare the request
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': api_key
        }
        
        # Create a focused translation prompt
        prompt = f"""Translate the following text to English. If the text is already in English, return it unchanged. Only return the translation, no explanations or additional text:

{text}"""
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract translated text from response
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            translated = parts[0]['text'].strip()
                            return translated
                
                logger.error(f"Unexpected API response format: {result}")
                return None
                
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during translation: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during translation: {e}")
            return None
    
    def translate_to_english(self, text: str) -> TranslationResult:
        """
        Translate text to English if it's not already in English
        
        Args:
            text: Input text to translate
            
        Returns:
            TranslationResult with translation details
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                detected_language=None,
                confidence=0.0,
                was_translated=False,
                error_message="Empty text provided"
            )
        
        original_text = text.strip()
        
        # Check cache first
        cache_key = self._get_cache_key(original_text)
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]
        
        # Detect language
        detected_lang = self._detect_language(original_text)
        
        # Check if text is already English
        if self._is_english(original_text):
            result = TranslationResult(
                original_text=original_text,
                translated_text=original_text,
                detected_language=detected_lang or 'en',
                confidence=1.0,
                was_translated=False
            )
            self._translation_cache[cache_key] = result
            return result
        
        # Translate using Gemini API
        translated = self._call_gemini_api(original_text)
        
        if translated:
            result = TranslationResult(
                original_text=original_text,
                translated_text=translated,
                detected_language=detected_lang,
                confidence=0.9,  # High confidence for successful API translation
                was_translated=True
            )
        else:
            # Fallback - return original text if translation fails
            result = TranslationResult(
                original_text=original_text,
                translated_text=original_text,
                detected_language=detected_lang,
                confidence=0.0,
                was_translated=False,
                error_message="Translation API failed, using original text"
            )
        
        # Cache the result
        self._translation_cache[cache_key] = result
        return result
    
    def is_available(self) -> bool:
        """Check if translator is properly configured and available"""
        return self._get_api_key() is not None
    
    def clear_cache(self):
        """Clear the translation cache"""
        self._translation_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._translation_cache),
            'cached_translations': sum(1 for r in self._translation_cache.values() if r.was_translated),
            'cached_english': sum(1 for r in self._translation_cache.values() if not r.was_translated)
        }

# Global translator instance
_translator = None

def get_translator() -> GeminiTranslator:
    """Get the global translator instance"""
    global _translator
    if _translator is None:
        _translator = GeminiTranslator()
    return _translator

def translate_text(text: str) -> TranslationResult:
    """
    Convenience function to translate text using the global translator
    
    Args:
        text: Text to translate
        
    Returns:
        TranslationResult with translation details
    """
    translator = get_translator()
    return translator.translate_to_english(text)