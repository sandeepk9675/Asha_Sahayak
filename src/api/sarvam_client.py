"""
Sarvam AI API client for ASHA-Sahayak.
Wraps Sarvam-m (LLM), Saarika (STT), and Mayura (Vision/OCR) APIs.
Falls back to Databricks Foundation Model API if Sarvam is unavailable.
"""

import os
import json
import base64
import requests
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Get Sarvam API key from env or Databricks secrets."""
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


def _get_databricks_token() -> str:
    """Get Databricks API token for Foundation Model fallback."""
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not token:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                token = spark._jvm.com.databricks.dbutils_v1.DBUtilsHolder.dbutils().secrets().get("asha-sahayak", "databricks-token")
        except Exception:
            pass
    return token


SARVAM_BASE_URL = "https://api.sarvam.ai"


# ---------------------------------------------------------------------------
# Sarvam-m LLM API
# ---------------------------------------------------------------------------

def chat_completion(
    messages: list[dict],
    model: str = "sarvam-m",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    Call Sarvam-m Chat Completion API.
    
    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
        model: Model name (default "sarvam-m")
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        
    Returns:
        Assistant response text
    """
    api_key = _get_api_key()
    
    if api_key:
        try:
            response = requests.post(
                f"{SARVAM_BASE_URL}/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Sarvam API error: {e}. Trying fallback...")
    
    # Fallback to Databricks Foundation Model API
    return _databricks_llm_fallback(messages, temperature, max_tokens)


def _databricks_llm_fallback(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Fallback to Databricks Foundation Model API (Meta Llama 3)."""
    token = _get_databricks_token()
    host = os.environ.get("DATABRICKS_HOST", "")
    
    if not token or not host:
        return "[Error: No LLM API available. Please configure SARVAM_API_KEY or DATABRICKS_TOKEN.]"
    
    try:
        response = requests.post(
            f"{host}/serving-endpoints/databricks-meta-llama-3-1-70b-instruct/invocations",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM Fallback Error: {e}]"


# ---------------------------------------------------------------------------
# Sarvam Saarika — Speech to Text
# ---------------------------------------------------------------------------

def speech_to_text(
    audio_file_path: str,
    language_code: str = "hi-IN",
) -> str:
    """
    Convert speech audio to text using Sarvam Saarika STT API.
    
    Args:
        audio_file_path: Path to audio file (WAV/MP3)
        language_code: BCP-47 language code (e.g., "hi-IN", "ta-IN", "kn-IN")
        
    Returns:
        Transcribed text
    """
    api_key = _get_api_key()
    if not api_key:
        return "[Error: SARVAM_API_KEY not configured for STT]"
    
    try:
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
        
        response = requests.post(
            f"{SARVAM_BASE_URL}/api/v1/speech-to-text",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            files={"file": ("audio.wav", audio_data, "audio/wav")},
            data={
                "language_code": language_code,
                "model": "saarika:v2",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("transcript", "")
    except Exception as e:
        return f"[STT Error: {e}]"


# ---------------------------------------------------------------------------
# Sarvam Mayura — Vision / OCR
# ---------------------------------------------------------------------------

def extract_text_from_image(
    image_path: str,
    prompt: str = "Extract all text from this medical report. Return structured data including patient name, hemoglobin, blood pressure, weight, and any test results.",
) -> str:
    """
    Use Sarvam Mayura Vision API to extract text from medical images.
    
    Args:
        image_path: Path to image file
        prompt: Instruction for the vision model
        
    Returns:
        Extracted text from the image
    """
    api_key = _get_api_key()
    if not api_key:
        return "[Error: SARVAM_API_KEY not configured for OCR]"
    
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine MIME type
        ext = image_path.lower().rsplit(".", 1)[-1]
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/jpeg")
        
        response = requests.post(
            f"{SARVAM_BASE_URL}/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sarvam-m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 2048,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[OCR Error: {e}]"


# ---------------------------------------------------------------------------
# Text-to-Speech (optional for future use)
# ---------------------------------------------------------------------------

def text_to_speech(
    text: str,
    language_code: str = "hi-IN",
    output_path: str = "/tmp/asha_tts_output.wav",
) -> Optional[str]:
    """
    Convert text to speech using Sarvam TTS API.
    
    Returns:
        Path to output audio file, or None on error.
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    
    try:
        response = requests.post(
            f"{SARVAM_BASE_URL}/api/v1/text-to-speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": text,
                "language_code": language_code,
                "model": "bulbul:v2",
                "speaker": "anila",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        if "audio" in data:
            audio_bytes = base64.b64decode(data["audio"])
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            return output_path
        return None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None
