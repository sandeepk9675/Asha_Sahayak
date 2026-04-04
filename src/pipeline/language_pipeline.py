"""
Language Pipeline for ASHA-Sahayak.
Handles: Audio → STT → Translation → LLM → Translation → Response
Supports text, voice and image inputs.
"""

import os
import json
import tempfile
from typing import Optional
from datetime import datetime

from src.api.sarvam_client import speech_to_text, extract_text_from_image
from src.api.sarvam_translate_client import (
    translate_to_english,
    translate_from_english,
    detect_language,
    LANGUAGE_MAP,
)
from src.pipeline.rag_pipeline import run_rag_pipeline


# Language code to STT language code mapping
STT_LANGUAGE_MAP = {
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "or": "or-IN",
    "en": "en-IN",
}


def process_text_input(
    spark,
    patient_id: str,
    text: str,
    language: str = "hi",
) -> dict:
    """
    Process text input from ASHA worker.
    
    Args:
        spark: SparkSession
        patient_id: Patient UUID
        text: Input text in any language
        language: Language code of input
        
    Returns:
        {
            "original_input": str,
            "translated_input": str,
            "ai_response": str,
            "translated_response": str,
            "health_updates": dict,
            "input_type": "TEXT",
        }
    """
    # Detect language if not specified
    if language == "auto":
        language = detect_language(text)
    
    # Translate to English
    if language != "en":
        english_input = translate_to_english(text, language)
    else:
        english_input = text
    
    # Run RAG pipeline
    rag_result = run_rag_pipeline(spark, patient_id, english_input, "TEXT")
    
    # Translate response back to ASHA's language
    if language != "en":
        translated_response = translate_from_english(rag_result["response"], language)
    else:
        translated_response = rag_result["response"]
    
    return {
        "original_input": text,
        "translated_input": english_input,
        "ai_response": rag_result["response"],
        "translated_response": translated_response,
        "health_updates": rag_result["health_updates"],
        "input_type": "TEXT",
    }


def process_audio_input(
    spark,
    patient_id: str,
    audio_file_path: str,
    language: str = "hi",
) -> dict:
    """
    Process voice/audio input from ASHA worker.
    Audio → STT → English → RAG → English → Local Language
    
    Args:
        spark: SparkSession
        patient_id: Patient UUID
        audio_file_path: Path to audio file
        language: Expected language of audio
        
    Returns:
        Same structure as process_text_input
    """
    # Step 1: Speech to Text
    stt_lang = STT_LANGUAGE_MAP.get(language, f"{language}-IN")
    transcribed_text = speech_to_text(audio_file_path, stt_lang)
    
    if transcribed_text.startswith("["):
        # Error from STT
        return {
            "original_input": transcribed_text,
            "translated_input": transcribed_text,
            "ai_response": "Sorry, I could not understand the audio. Please try again or type your message.",
            "translated_response": "माफ़ करें, मैं ऑडियो समझ नहीं पाई। कृपया दोबारा कोशिश करें या अपना संदेश टाइप करें।",
            "health_updates": {},
            "input_type": "AUDIO",
        }
    
    # Step 2: Process as text
    result = process_text_input(spark, patient_id, transcribed_text, language)
    result["input_type"] = "AUDIO"
    result["original_input"] = f"[Audio] {transcribed_text}"
    return result


def process_image_input(
    spark,
    patient_id: str,
    image_path: str,
    language: str = "hi",
    additional_text: str = "",
) -> dict:
    """
    Process image input (e.g., lab report photo).
    Image → OCR → Parse → RAG → Response
    
    Args:
        spark: SparkSession
        patient_id: Patient UUID
        image_path: Path to image file
        language: ASHA's language preference
        additional_text: Optional accompanying text message
        
    Returns:
        Same structure as process_text_input, with extracted text
    """
    # Step 1: Extract text from image via OCR/Vision
    extracted_text = extract_text_from_image(
        image_path,
        prompt=(
            "Extract all medical information from this report image. "
            "Include: patient name, hemoglobin (Hb), blood pressure (BP), "
            "weight, blood sugar, urine tests, and any other test results. "
            "Format as structured text."
        ),
    )
    
    # Step 2: Combine extracted text with any additional message
    combined_query = f"Lab report/medical document uploaded. Extracted content:\n{extracted_text}"
    if additional_text:
        combined_query += f"\n\nASHA's note: {additional_text}"
    
    combined_query += "\n\nPlease analyze this medical data, update the patient's health status, flag any concerns, and provide recommendations."
    
    # Step 3: Run RAG pipeline (already in English from OCR)
    rag_result = run_rag_pipeline(spark, patient_id, combined_query, "IMAGE")
    
    # Step 4: Translate response
    if language != "en":
        translated_response = translate_from_english(rag_result["response"], language)
    else:
        translated_response = rag_result["response"]
    
    return {
        "original_input": f"[Image] {extracted_text[:500]}",
        "translated_input": combined_query,
        "ai_response": rag_result["response"],
        "translated_response": translated_response,
        "health_updates": rag_result["health_updates"],
        "extracted_text": extracted_text,
        "input_type": "IMAGE",
    }


def get_supported_languages() -> dict:
    """Return dict of supported language codes and names."""
    return {
        "hi": "Hindi (हिन्दी)",
        "ta": "Tamil (தமிழ்)",
        "te": "Telugu (తెలుగు)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "ml": "Malayalam (മലയാളം)",
        "bn": "Bengali (বাংলা)",
        "mr": "Marathi (मराठी)",
        "gu": "Gujarati (ગુજરાતી)",
        "pa": "Punjabi (ਪੰਜਾਬੀ)",
        "or": "Odia (ଓଡ଼ିଆ)",
        "en": "English",
    }
