"""
Chat Service for ASHA-Sahayak.
Manages conversations, history, and integrates with the language pipeline.
"""

import uuid
import json
from datetime import datetime
from typing import Optional

from src.utils.delta_utils import read_table, append_rows
from src.pipeline.language_pipeline import (
    process_text_input,
    process_audio_input,
    process_image_input,
)
from src.pipeline.risk_engine import assess_risk


def chat(
    spark,
    patient_id: str,
    message: str = "",
    audio_path: str = None,
    image_path: str = None,
    language: str = "hi",
    asha_id: str = "ASHA001",
) -> dict:
    """
    Main chat function. Handles text, audio, and image inputs.
    
    Args:
        spark: SparkSession
        patient_id: Patient UUID
        message: Text message (for text input)
        audio_path: Path to audio file (for voice input)
        image_path: Path to image file (for image input)
        language: Language code
        asha_id: ASHA worker ID
        
    Returns:
        {
            "response": str (in ASHA's language),
            "english_response": str,
            "health_updates": dict,
            "risk_alert": dict or None,
        }
    """
    # Determine input type and process
    if audio_path:
        result = process_audio_input(spark, patient_id, audio_path, language)
    elif image_path:
        result = process_image_input(spark, patient_id, image_path, language, message)
    else:
        result = process_text_input(spark, patient_id, message, language)
    
    # Store conversation in Delta Lake
    conversation = {
        "conversation_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "asha_id": asha_id,
        "timestamp": datetime.now(),
        "input_type": result.get("input_type", "TEXT"),
        "original_input": result.get("original_input", message),
        "translated_input": result.get("translated_input", message),
        "ai_response": result.get("ai_response", ""),
        "translated_response": result.get("translated_response", ""),
        "extracted_health_updates": json.dumps(result.get("health_updates", {})),
    }
    
    try:
        append_rows(spark, "conversations", [conversation])
    except Exception as e:
        print(f"Error storing conversation: {e}")
    
    # Run risk assessment if health updates were extracted
    risk_alert = None
    health_updates = result.get("health_updates", {})
    if health_updates:
        symptoms = health_updates.get("symptoms", []) + health_updates.get("risk_flags", [])
        if symptoms or health_updates.get("hemoglobin") or health_updates.get("bp_systolic"):
            try:
                ehr_data = {}
                if health_updates.get("hemoglobin"):
                    ehr_data["hemoglobin"] = health_updates["hemoglobin"]
                if health_updates.get("bp_systolic"):
                    ehr_data["bp_systolic"] = health_updates["bp_systolic"]
                if health_updates.get("bp_diastolic"):
                    ehr_data["bp_diastolic"] = health_updates["bp_diastolic"]
                
                risk_result = assess_risk(
                    spark, patient_id,
                    ehr_data=ehr_data if ehr_data else None,
                    conversation_symptoms=symptoms,
                )
                
                if risk_result["risk_level"] != "GREEN":
                    risk_alert = risk_result
            except Exception as e:
                print(f"Risk assessment error: {e}")
    
    return {
        "response": result.get("translated_response", result.get("ai_response", "")),
        "english_response": result.get("ai_response", ""),
        "health_updates": health_updates,
        "risk_alert": risk_alert,
    }


def get_chat_history(spark, patient_id: str, limit: int = 20) -> list:
    """Get conversation history for a patient."""
    conv_df = read_table(spark, "conversations")
    df = conv_df[conv_df["patient_id"] == patient_id].copy()
    df = df.sort_values("timestamp", ascending=False).head(limit)
    
    # Reverse so oldest shows first
    df = df.iloc[::-1]
    
    history = []
    for _, row in df.iterrows():
        history.append({
            "timestamp": str(row["timestamp"]),
            "input_type": row["input_type"],
            "user_message": row["original_input"],
            "ai_response": row["translated_response"] if row["translated_response"] else row["ai_response"],
        })
    
    return history


def get_chat_history_for_gradio(spark, patient_id: str) -> list:
    """
    Get chat history formatted for Gradio Chatbot component.
    Returns list of (user_msg, bot_msg) tuples.
    """
    history = get_chat_history(spark, patient_id)
    
    gradio_history = []
    for h in history:
        user_msg = h["user_message"]
        if h["input_type"] == "AUDIO":
            user_msg = f"🎤 {user_msg}"
        elif h["input_type"] == "IMAGE":
            user_msg = f"📷 {user_msg}"
        
        gradio_history.append({"role": "user", "content": user_msg})
        gradio_history.append({"role": "assistant", "content": h["ai_response"]})
    
    return gradio_history
