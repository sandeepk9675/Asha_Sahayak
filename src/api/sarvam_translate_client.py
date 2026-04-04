"""
Sarvam Translate API client for ASHA-Sahayak.
Provides translation between 22 Indian languages and English.
"""

import os
import requests
from typing import Optional


def _get_api_key() -> str:
    """Get Sarvam API key."""
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                key = spark._jvm.com.databricks.dbutils_v1.DBUtilsHolder.dbutils().secrets().get("asha-sahayak", "sarvam-api-key")
        except Exception:
            pass
    return key


SARVAM_BASE_URL = "https://api.sarvam.ai"

# Language code mapping for Sarvam Translate
LANGUAGE_MAP = {
    "hi": "hi-IN",   # Hindi
    "ta": "ta-IN",   # Tamil
    "te": "te-IN",   # Telugu
    "kn": "kn-IN",   # Kannada
    "ml": "ml-IN",   # Malayalam
    "bn": "bn-IN",   # Bengali
    "mr": "mr-IN",   # Marathi
    "gu": "gu-IN",   # Gujarati
    "pa": "pa-IN",   # Punjabi
    "or": "or-IN",   # Odia
    "as": "as-IN",   # Assamese
    "ur": "ur-IN",   # Urdu
    "en": "en-IN",   # English
}


def translate(
    text: str,
    source_lang: str = "hi",
    target_lang: str = "en",
) -> str:
    """
    Translate text between Indian languages using Sarvam Translate API.
    
    Args:
        text: Input text to translate
        source_lang: Source language code (e.g., "hi", "ta", "en")
        target_lang: Target language code
        
    Returns:
        Translated text
    """
    if source_lang == target_lang:
        return text
    
    api_key = _get_api_key()
    if not api_key:
        # Fallback: return original text with a marker
        return f"[Translation unavailable] {text}"
    
    source_code = LANGUAGE_MAP.get(source_lang, f"{source_lang}-IN")
    target_code = LANGUAGE_MAP.get(target_lang, f"{target_lang}-IN")
    
    try:
        response = requests.post(
            f"{SARVAM_BASE_URL}/api/v1/translate",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": text,
                "source_language_code": source_code,
                "target_language_code": target_code,
                "model": "mayura:v1",
                "enable_preprocessing": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("translated_text", text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def translate_to_english(text: str, source_lang: str = "hi") -> str:
    """Convenience: translate any Indian language to English."""
    return translate(text, source_lang, "en")


def translate_from_english(text: str, target_lang: str = "hi") -> str:
    """Convenience: translate English to any Indian language."""
    return translate(text, "en", target_lang)


def detect_language(text: str) -> str:
    """
    Simple language detection heuristic.
    For production, use a proper detection API.
    """
    # Check for Devanagari (Hindi, Marathi, Sanskrit)
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hi"
    # Tamil
    if any('\u0B80' <= c <= '\u0BFF' for c in text):
        return "ta"
    # Telugu
    if any('\u0C00' <= c <= '\u0C7F' for c in text):
        return "te"
    # Kannada
    if any('\u0C80' <= c <= '\u0CFF' for c in text):
        return "kn"
    # Malayalam
    if any('\u0D00' <= c <= '\u0D7F' for c in text):
        return "ml"
    # Bengali
    if any('\u0980' <= c <= '\u09FF' for c in text):
        return "bn"
    # Gujarati
    if any('\u0A80' <= c <= '\u0AFF' for c in text):
        return "gu"
    # Default to English
    return "en"
