"""
Embeddings client for ASHA-Sahayak.
Uses HuggingFace Inference API for multilingual-e5-small embeddings.
Falls back to a simple TF-IDF approach if API is unavailable.
"""

import os
import json
import numpy as np
import requests
from typing import Optional


def _get_hf_api_key() -> str:
    """Get HuggingFace API key."""
    key = os.environ.get("HF_API_KEY", "")
    if not key:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                key = spark._jvm.com.databricks.dbutils_v1.DBUtilsHolder.dbutils().secrets().get("asha-sahayak", "hf-api-key")
        except Exception:
            pass
    return key


HF_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL}"


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.
    
    Uses HuggingFace Inference API for multilingual-e5-small.
    Falls back to simple hash-based embeddings if API unavailable.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    api_key = _get_hf_api_key()
    
    if api_key:
        try:
            # Add "query: " prefix for e5 models
            prefixed = [f"query: {t}" for t in texts]
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": prefixed, "options": {"wait_for_model": True}},
                timeout=60,
            )
            response.raise_for_status()
            embeddings = np.array(response.json(), dtype=np.float32)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms
        except Exception as e:
            print(f"HF Embedding API error: {e}. Using fallback embeddings.")
    
    return _fallback_embeddings(texts)


def get_passage_embeddings(texts: list[str]) -> np.ndarray:
    """
    Get embeddings for passages (documents) — uses 'passage:' prefix for e5.
    """
    api_key = _get_hf_api_key()
    
    if api_key:
        try:
            prefixed = [f"passage: {t}" for t in texts]
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": prefixed, "options": {"wait_for_model": True}},
                timeout=60,
            )
            response.raise_for_status()
            embeddings = np.array(response.json(), dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms
        except Exception as e:
            print(f"HF Embedding API error: {e}. Using fallback.")
    
    return _fallback_embeddings(texts)


def _fallback_embeddings(texts: list[str], dim: int = 384) -> np.ndarray:
    """
    Simple fallback: generate deterministic pseudo-embeddings using hashing.
    NOT suitable for production — only for demo when API is unavailable.
    """
    import hashlib
    embeddings = []
    for text in texts:
        # Create a deterministic hash-based vector
        h = hashlib.sha384(text.encode("utf-8")).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        # Pad or truncate to dim
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)), mode="wrap")
        else:
            vec = vec[:dim]
        # Normalize
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        embeddings.append(vec)
    return np.array(embeddings, dtype=np.float32)
