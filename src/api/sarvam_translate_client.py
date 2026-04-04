"""
Sarvam Translate API client for ASHA-Sahayak.
Provides translation between Indian languages and English.
"""

import os
import requests


def _get_api_key() -> str:
    """Get Sarvam API key."""
    key = (
        os.environ.get("SARVAM_API_KEY", "")
        or os.environ.get("sarvam_api_key", "")
    )

    if not key:
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark:
                key = spark._jvm.com.databricks.dbutils_v1.DBUtilsHolder.dbutils().secrets().get(
                    "asha-sahayak", "sarvam_api_key"
                )
        except Exception:
            pass

    return key


SARVAM_BASE_URL = "https://api.sarvam.ai"

LANGUAGE_MAP = {
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "or": "od-IN",
    "as": "as-IN",
    "ur": "ur-IN",
    "en": "en-IN",
}


def translate(
    text: str,
    source_lang: str = "hi",
    target_lang: str = "en",
) -> str:
    """
    Translate text between Indian languages using Sarvam Translate API.
    """
    if source_lang == target_lang:
        return text

    api_key = _get_api_key()
    if not api_key:
        print("Translation warning: SARVAM_API_KEY not found; returning original text.")
        return text

    source_code = LANGUAGE_MAP.get(source_lang, f"{source_lang}-IN")
    target_code = LANGUAGE_MAP.get(target_lang, f"{target_lang}-IN")

    try:
        response = requests.post(
            f"{SARVAM_BASE_URL}/translate",
            headers={
                "api-subscription-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "input": text,
                "source_language_code": source_code,
                "target_language_code": target_code,
                "model": "mayura:v1",
                "mode": "formal",
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
    return translate(text, source_lang, "en")


def translate_from_english(text: str, target_lang: str = "hi") -> str:
    return translate(text, "en", target_lang)


def detect_language(text: str) -> str:
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hi"
    if any('\u0B80' <= c <= '\u0BFF' for c in text):
        return "ta"
    if any('\u0C00' <= c <= '\u0C7F' for c in text):
        return "te"
    if any('\u0C80' <= c <= '\u0CFF' for c in text):
        return "kn"
    if any('\u0D00' <= c <= '\u0D7F' for c in text):
        return "ml"
    if any('\u0980' <= c <= '\u09FF' for c in text):
        return "bn"
    if any('\u0A80' <= c <= '\u0AFF' for c in text):
        return "gu"
    return "en"