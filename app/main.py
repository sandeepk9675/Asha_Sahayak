"""
ASHA-Sahayak (आशा सहायक) — Gradio Frontend
Main application file for the AI-powered maternal health assistant.
Deploy as Databricks App or run locally.
"""

import os
import sys
import json
import tempfile
import uuid
from datetime import datetime, date, timedelta

import gradio as gr
import pandas as pd

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------
os.environ.setdefault("ASHA_DELTA_BASE", "/dbfs/asha_sahayak/delta")
os.environ.setdefault("ASHA_FAISS_PATH", "/dbfs/asha_sahayak/faiss")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Import Services
# ---------------------------------------------------------------------------
from src.utils.delta_utils import get_spark
from src.services.patient_service import (
    register_patient, get_patient, list_patients,
    search_patients, get_patients_dataframe,
)
from src.services.ehr_service import (
    upload_ehr_image, add_ehr_manual, get_patient_ehrs, get_ehrs_dataframe,
)
from src.services.chat_service import (
    chat, get_chat_history, get_chat_history_for_gradio,
)
from src.services.dashboard_service import (
    get_dashboard_data, get_dashboard_summary_text,
)
from src.pipeline.schedule_engine import generate_schedule, get_overdue_checkups
from src.pipeline.ration_engine import generate_ration_plan, get_village_ration_summary
from src.pipeline.risk_engine import get_patient_risk_summary
from src.pipeline.language_pipeline import get_supported_languages

# ---------------------------------------------------------------------------
# Initialize Spark
# ---------------------------------------------------------------------------
try:
    spark = get_spark()
    print("SparkSession initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize Spark: {e}")
    spark = None

# Current ASHA ID (in production, this would come from authentication)
CURRENT_ASHA_ID = "ASHA001"

# Supported languages
LANGUAGES = get_supported_languages()
LANG_CHOICES = [(v, k) for k, v in LANGUAGES.items()]


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def get_patient_choices():
    """Get patient list as dropdown choices."""
    try:
        patients = list_patients(spark, CURRENT_ASHA_ID)
        choices = [(f"{p['name']} ({p['village']}) - T{p['trimester']} W{p['gestational_weeks']} [{p['risk_status']}]", p['patient_id']) for p in patients]
        return choices
    except Exception as e:
        return [("Error loading patients", "")]


def risk_badge(status):
    """Return color-coded risk badge."""
    colors = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    return colors.get(status, "⚪")


# ===================================================================
# TAB 1: DASHBOARD
# ===================================================================

def refresh_dashboard():
    """Refresh the village dashboard."""
    try:
        summary = get_dashboard_summary_text(spark, CURRENT_ASHA_ID)
        data = get_dashboard_data(spark, CURRENT_ASHA_ID)
        
        # Alerts HTML
        alerts_html = ""
        for a in data.get("alerts", []):
            emoji = "🚨" if a.get("emergency") else "🔴"
            factors = ", ".join(a.get("risk_factors", [])[:2])
            alerts_html += f'<div style="padding:8px;margin:4px 0;background:#fee;border-left:4px solid red;border-radius:4px">{emoji} <b>{a["patient_name"]}</b> — {factors}</div>'
        
        if not alerts_html:
            alerts_html = '<div style="padding:8px;background:#efe;border-radius:4px">✅ No active alerts</div>'
        
        # Stats
        stats = data.get("village_stats", {})
        tri = data.get("trimester_distribution", {})
        stats_html = f"""
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin:12px 0">
            <div style="padding:12px 20px;background:#e8f5e9;border-radius:8px;text-align:center">
                <div style="font-size:24px;font-weight:bold">{stats.get('total_patients', 0)}</div>
                <div>Total</div>
            </div>
            <div style="padding:12px 20px;background:#e8f5e9;border-radius:8px;text-align:center">
                <div style="font-size:24px;font-weight:bold;color:green">🟢 {stats.get('green_count', 0)}</div>
                <div>Normal</div>
            </div>
            <div style="padding:12px 20px;background:#fff9c4;border-radius:8px;text-align:center">
                <div style="font-size:24px;font-weight:bold;color:orange">🟡 {stats.get('yellow_count', 0)}</div>
                <div>Watch</div>
            </div>
            <div style="padding:12px 20px;background:#ffebee;border-radius:8px;text-align:center">
                <div style="font-size:24px;font-weight:bold;color:red">🔴 {stats.get('red_count', 0)}</div>
                <div>High Risk</div>
            </div>
            <div style="padding:12px 20px;background:#e3f2fd;border-radius:8px;text-align:center">
                <div style="font-size:24px;font-weight:bold">{stats.get('overdue_checkups', 0)}</div>
                <div>Overdue</div>
            </div>
        </div>
        <div style="margin-top:8px">
            <b>By Trimester:</b> {' | '.join([f'{k}: {v}' for k, v in sorted(tri.items())])}
        </div>
        """
        
        # Upcoming deliveries
        upcoming = data.get("upcoming_deliveries", [])
        deliveries_html = ""
        if upcoming:
            deliveries_html = "<h4>🏥 Upcoming Deliveries (next 30 days)</h4>"
            for u in upcoming:
                deliveries_html += f'<div style="padding:4px 0">{risk_badge(u["risk_status"])} {u["patient_name"]} — EDD: {u["edd"]} ({u["days_until_edd"]} days)</div>'
        
        return alerts_html, stats_html, deliveries_html
    except Exception as e:
        error_msg = f'<div style="color:red">Error: {e}</div>'
        return error_msg, error_msg, error_msg


# ===================================================================
# TAB 2: PATIENT LIST & REGISTRATION
# ===================================================================

def load_patients_table():
    """Load patients as a formatted HTML table."""
    try:
        patients = list_patients(spark, CURRENT_ASHA_ID)
        if not patients:
            return "<p>No patients registered yet.</p>", gr.update(choices=[])
        
        html = '<table style="width:100%;border-collapse:collapse">'
        html += '<tr style="background:#1a73e8;color:white"><th style="padding:8px">Name</th><th>Age</th><th>Village</th><th>Trimester</th><th>Weeks</th><th>Risk</th><th>EDD</th></tr>'
        
        choices = []
        for i, p in enumerate(patients):
            bg = "#fff" if i % 2 == 0 else "#f5f5f5"
            html += f'<tr style="background:{bg}">'
            html += f'<td style="padding:6px 8px">{p["name"]}</td>'
            html += f'<td style="text-align:center">{p["age"]}</td>'
            html += f'<td>{p["village"]}</td>'
            html += f'<td style="text-align:center">T{p["trimester"]}</td>'
            html += f'<td style="text-align:center">{p["gestational_weeks"]}</td>'
            html += f'<td style="text-align:center">{risk_badge(p["risk_status"])} {p["risk_status"]}</td>'
            html += f'<td>{p["edd"]}</td>'
            html += '</tr>'
            choices.append((f"{p['name']} ({p['village']}) [{p['risk_status']}]", p['patient_id']))
        
        html += '</table>'
        return html, gr.update(choices=choices)
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>", gr.update(choices=[])


def register_new_patient(name, age, lmp_str, village, contact, language, blood_group, height, weight):
    """Register a new patient."""
    try:
        if not name or not lmp_str:
            return "❌ Name and LMP date are required."
        
        lmp_date = datetime.strptime(lmp_str, "%Y-%m-%d").date()
        
        result = register_patient(
            spark, name=name, age=int(age), lmp_date=lmp_date,
            village=village, contact=contact, language_preference=language,
            blood_group=blood_group, height_cm=float(height) if height else 0,
            pre_pregnancy_weight_kg=float(weight) if weight else 0,
            asha_id=CURRENT_ASHA_ID,
        )
        return result["message"]
    except Exception as e:
        return f"❌ Error: {e}"


# ===================================================================
# TAB 3: PATIENT PROFILE (Chat, EHR, Schedule, Ration)
# ===================================================================

def load_patient_profile(patient_id):
    """Load patient profile details."""
    if not patient_id:
        return "Select a patient first.", "", "", ""
    
    try:
        patient = get_patient(spark, patient_id)
        if not patient:
            return "Patient not found.", "", "", ""
        
        # Basic info
        info_html = f"""
        <div style="padding:16px;background:#f8f9fa;border-radius:8px">
            <h3>{risk_badge(patient['risk_status'])} {patient['name']}</h3>
            <table style="width:100%">
                <tr><td><b>Age:</b></td><td>{patient['age']} years</td><td><b>Village:</b></td><td>{patient['village']}</td></tr>
                <tr><td><b>Contact:</b></td><td>{patient['contact']}</td><td><b>Blood Group:</b></td><td>{patient['blood_group']}</td></tr>
                <tr><td><b>LMP:</b></td><td>{patient['lmp_date']}</td><td><b>EDD:</b></td><td>{patient['edd']}</td></tr>
                <tr><td><b>Gestational Age:</b></td><td>{patient['gestational_weeks']} weeks</td><td><b>Trimester:</b></td><td>T{patient['trimester']}</td></tr>
                <tr><td><b>Weeks Remaining:</b></td><td>{patient['weeks_remaining']}</td><td><b>Language:</b></td><td>{patient['language_preference']}</td></tr>
                <tr><td><b>Height:</b></td><td>{patient['height_cm']} cm</td><td><b>Pre-preg Weight:</b></td><td>{patient['pre_pregnancy_weight_kg']} kg</td></tr>
            </table>
        </div>
        """
        
        # Risk summary
        risk = get_patient_risk_summary(spark, patient_id)
        risk_html = f"""
        <div style="padding:12px;background:{'#ffebee' if risk['risk_level']=='RED' else '#fff9c4' if risk['risk_level']=='YELLOW' else '#e8f5e9'};border-radius:8px;margin-top:8px">
            <b>Risk Level: {risk_badge(risk['risk_level'])} {risk['risk_level']}</b>
            <p>{risk['recommended_action']}</p>
            {'<p><b>Factors:</b> ' + ', '.join(risk['risk_factors']) + '</p>' if risk['risk_factors'] else ''}
        </div>
        """
        
        # EHR summary
        ehrs = get_patient_ehrs(spark, patient_id)
        ehr_html = ""
        if ehrs:
            ehr_html = '<h4>Recent Health Records</h4><table style="width:100%;border-collapse:collapse">'
            ehr_html += '<tr style="background:#1a73e8;color:white"><th style="padding:6px">Date</th><th>Hb</th><th>BP</th><th>Weight</th><th>Sugar(F)</th></tr>'
            for ehr in ehrs[:5]:
                ehr_html += f'<tr><td style="padding:4px 6px">{ehr["visit_date"]}</td><td>{ehr["hemoglobin"]}</td><td>{ehr["bp"]}</td><td>{ehr["weight_kg"]}</td><td>{ehr["blood_sugar_fasting"]}</td></tr>'
            ehr_html += '</table>'
        
        return info_html, risk_html, ehr_html, patient.get("language_preference", "hi")
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>", "", "", "hi"


def chat_with_ai(patient_id, message, language, history):
    """Handle chat interaction."""
    if not patient_id:
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please select a patient first."})
        return history, ""
    
    if not message:
        return history or [], ""
    
    try:
        result = chat(
            spark, patient_id=patient_id, message=message,
            language=language, asha_id=CURRENT_ASHA_ID,
        )
        
        history = history or []
        history.append({"role": "user", "content": message})
        
        response = result["response"]
        
        # Add risk alert if present
        if result.get("risk_alert"):
            alert = result["risk_alert"]
            response += f"\n\n🚨 **ALERT: {alert['risk_level']}**\n"
            response += f"Risk factors: {', '.join(alert['risk_factors'])}\n"
            response += f"Action: {alert['recommended_action']}"
        
        history.append({"role": "assistant", "content": response})
        return history, ""
    except Exception as e:
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error: {e}"})
        return history, ""


def chat_with_audio(patient_id, audio, language, history):
    """Handle voice input."""
    if not patient_id:
        history = history or []
        history.append({"role": "assistant", "content": "⚠️ Please select a patient first."})
        return history
    
    if audio is None:
        return history or []
    
    try:
        result = chat(
            spark, patient_id=patient_id, audio_path=audio,
            language=language, asha_id=CURRENT_ASHA_ID,
        )
        
        history = history or []
        history.append({"role": "user", "content": f"🎤 {result.get('english_response', 'Voice message')}"}[:100])
        
        response = result["response"]
        if result.get("risk_alert"):
            alert = result["risk_alert"]
            response += f"\n\n🚨 **ALERT: {alert['risk_level']}**\n{alert['recommended_action']}"
        
        history.append({"role": "assistant", "content": response})
        return history
    except Exception as e:
        history = history or []
        history.append({"role": "assistant", "content": f"❌ Error: {e}"})
        return history


def chat_with_image(patient_id, image, message, language, history):
    """Handle image upload with optional text."""
    if not patient_id:
        history = history or []
        history.append({"role": "assistant", "content": "⚠️ Please select a patient first."})
        return history
    
    if image is None:
        return history or []
    
    try:
        # Save uploaded image to temp file
        temp_path = os.path.join(tempfile.gettempdir(), f"asha_upload_{uuid.uuid4().hex}.jpg")
        if isinstance(image, str):
            temp_path = image
        else:
            from PIL import Image
            img = Image.fromarray(image)
            img.save(temp_path)
        
        result = chat(
            spark, patient_id=patient_id, message=message or "",
            image_path=temp_path, language=language, asha_id=CURRENT_ASHA_ID,
        )
        
        history = history or []
        history.append({"role": "user", "content": f"📷 Image uploaded" + (f": {message}" if message else "")})
        
        response = result["response"]
        if result.get("risk_alert"):
            alert = result["risk_alert"]
            response += f"\n\n🚨 **ALERT: {alert['risk_level']}**\n{alert['recommended_action']}"
        
        history.append({"role": "assistant", "content": response})
        return history
    except Exception as e:
        history = history or []
        history.append({"role": "assistant", "content": f"❌ Error: {e}"})
        return history


def add_ehr_record(patient_id, hb, bp_sys, bp_dia, weight, sugar_f, sugar_pp, urine_alb, usg, medicines):
    """Add manual EHR record."""
    if not patient_id:
        return "⚠️ Select a patient first."
    
    try:
        result = add_ehr_manual(
            spark, patient_id=patient_id,
            hemoglobin=float(hb) if hb else None,
            bp_systolic=int(bp_sys) if bp_sys else None,
            bp_diastolic=int(bp_dia) if bp_dia else None,
            weight_kg=float(weight) if weight else None,
            blood_sugar_fasting=float(sugar_f) if sugar_f else None,
            blood_sugar_pp=float(sugar_pp) if sugar_pp else None,
            urine_albumin=urine_alb or "Normal",
            usg_findings=usg or "",
            prescribed_medicines=medicines or "",
        )
        return result["message"]
    except Exception as e:
        return f"❌ Error: {e}"


def generate_patient_schedule(patient_id):
    """Generate and display ANC schedule."""
    if not patient_id:
        return "<p>Select a patient first.</p>"
    
    try:
        schedules = generate_schedule(spark, patient_id)
        
        if not schedules:
            return "<p>No schedule generated. Check patient LMP date.</p>"
        
        html = '<h4>📅 ANC Checkup Schedule</h4>'
        html += '<table style="width:100%;border-collapse:collapse">'
        html += '<tr style="background:#1a73e8;color:white"><th style="padding:6px">#</th><th>Date</th><th>Type</th><th>Status</th><th>Tests Due</th></tr>'
        
        for s in schedules:
            status = s["status"]
            bg = "#e8f5e9" if status == "COMPLETED" else "#fff9c4" if status == "PENDING" else "#ffebee"
            status_icon = "✅" if status == "COMPLETED" else "⏳" if status == "PENDING" else "⚠️"
            tests = json.loads(s["tests_due"]) if isinstance(s["tests_due"], str) else s.get("tests_due", [])
            
            html += f'<tr style="background:{bg}">'
            html += f'<td style="padding:4px 8px">{s["visit_number"]}</td>'
            html += f'<td>{s["scheduled_date"]}</td>'
            html += f'<td>{s["visit_type"]}</td>'
            html += f'<td>{status_icon} {status}</td>'
            html += f'<td style="font-size:0.85em">{", ".join(tests[:3])}</td>'
            html += '</tr>'
        
        html += '</table>'
        return html
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>"


def generate_patient_ration(patient_id):
    """Generate and display ration plan."""
    if not patient_id:
        return "<p>Select a patient first.</p>"
    
    try:
        plan = generate_ration_plan(spark, patient_id, use_llm=True)
        
        if "error" in plan:
            return f"<p style='color:red'>{plan['error']}</p>"
        
        html = f"""
        <div style="padding:16px;background:#f8f9fa;border-radius:8px">
            <h4>🍚 Weekly Ration Plan — {plan['patient_name']}</h4>
            <p><b>Trimester:</b> T{plan['trimester']} (Week {plan.get('gestational_weeks', '?')}) | 
            <b>Daily Calories:</b> {plan['daily_calorie_target']} kcal | 
            <b>Protein:</b> {plan['protein_target_g']}g</p>
            {'<p style="color:red"><b>Conditions:</b> ' + ', '.join(plan['conditions']) + '</p>' if plan.get('conditions') else ''}
        """
        
        # Ration items table
        items = plan.get("ration_items", [])
        if items:
            html += '<h5>Daily Ration Items</h5>'
            html += '<table style="width:100%;border-collapse:collapse">'
            html += '<tr style="background:#4caf50;color:white"><th style="padding:6px">Item</th><th>Quantity</th><th>Frequency</th></tr>'
            for item in items:
                html += f'<tr><td style="padding:4px 8px">{item.get("item","")}</td><td>{item.get("quantity_g","")}g</td><td>{item.get("frequency","")}</td></tr>'
            html += '</table>'
        
        # Supplements
        supps = plan.get("supplements", [])
        if supps:
            html += '<h5>Supplements</h5>'
            html += '<table style="width:100%;border-collapse:collapse">'
            html += '<tr style="background:#2196f3;color:white"><th style="padding:6px">Name</th><th>Dosage</th><th>Frequency</th></tr>'
            for s in supps:
                html += f'<tr><td style="padding:4px 8px">{s.get("name","")}</td><td>{s.get("dosage","")}</td><td>{s.get("frequency","")}</td></tr>'
            html += '</table>'
        
        # Special notes
        if plan.get("special_notes"):
            html += f'<div style="margin-top:8px;padding:8px;background:#fff3e0;border-radius:4px"><b>Notes:</b> {plan["special_notes"]}</div>'
        
        html += '</div>'
        return html
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>"


# ===================================================================
# TAB 4: RATION DISTRIBUTION (Village-wide)
# ===================================================================

def load_village_rations():
    """Load village-wide ration summary."""
    try:
        summary = get_village_ration_summary(spark, CURRENT_ASHA_ID)
        
        if not summary:
            return "<p>No ration plans generated yet. Generate plans for individual patients first.</p>"
        
        html = '<h4>📋 Weekly Ration Distribution List</h4>'
        html += '<table style="width:100%;border-collapse:collapse">'
        html += '<tr style="background:#4caf50;color:white"><th style="padding:6px">Patient</th><th>T</th><th>Risk</th><th>Ration Items</th><th>Supplements</th><th>Notes</th></tr>'
        
        for i, s in enumerate(summary):
            bg = "#fff" if i % 2 == 0 else "#f5f5f5"
            items = ", ".join([f"{r['item']} ({r['quantity_g']}g)" for r in s["ration_items"][:4]]) if s["ration_items"] else "Not generated"
            supps = ", ".join([f"{r['name']}" for r in s["supplements"]]) if s["supplements"] else "-"
            
            html += f'<tr style="background:{bg}">'
            html += f'<td style="padding:4px 8px">{risk_badge(s["risk_status"])} {s["patient_name"]}</td>'
            html += f'<td style="text-align:center">T{s["trimester"]}</td>'
            html += f'<td style="text-align:center">{s["risk_status"]}</td>'
            html += f'<td style="font-size:0.85em">{items}</td>'
            html += f'<td style="font-size:0.85em">{supps}</td>'
            html += f'<td style="font-size:0.85em">{s["special_notes"][:60]}</td>'
            html += '</tr>'
        
        html += '</table>'
        return html
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>"


# ===================================================================
# BUILD GRADIO APP
# ===================================================================

def build_app():
    """Build the complete Gradio application."""
    
    with gr.Blocks(
        title="ASHA-Sahayak (आशा सहायक)",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        h1 { color: #1a73e8 !important; }
        .risk-red { background: #ffebee; border-left: 4px solid red; }
        .risk-green { background: #e8f5e9; border-left: 4px solid green; }
        """
    ) as app:
        
        # Header
        gr.Markdown(
            """
            # 🏥 ASHA-Sahayak (आशा सहायक)
            ### AI-Powered Multilingual Maternal Health Assistant
            ---
            """
        )
        
        # Language selector
        with gr.Row():
            language_selector = gr.Dropdown(
                choices=LANG_CHOICES,
                value="hi",
                label="🌐 Language / भाषा",
                scale=1,
            )
            gr.Markdown(f"**ASHA:** {CURRENT_ASHA_ID} | **Date:** {date.today()}")
        
        # ============================================================
        # TAB: DASHBOARD
        # ============================================================
        with gr.Tab("🏠 Dashboard"):
            gr.Markdown("### Village Overview")
            
            with gr.Row():
                dashboard_refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
            
            alerts_display = gr.HTML(label="Alerts")
            stats_display = gr.HTML(label="Statistics")
            deliveries_display = gr.HTML(label="Upcoming Deliveries")
            
            dashboard_refresh_btn.click(
                fn=refresh_dashboard,
                outputs=[alerts_display, stats_display, deliveries_display],
            )
        
        # ============================================================
        # TAB: PATIENTS
        # ============================================================
        with gr.Tab("👤 Patients"):
            with gr.Row():
                patients_refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
            
            patients_table_display = gr.HTML(label="Patient List")
            
            gr.Markdown("---\n### ➕ Register New Patient")
            
            with gr.Row():
                reg_name = gr.Textbox(label="Name", placeholder="Patient's full name")
                reg_age = gr.Number(label="Age", value=25, minimum=14, maximum=50)
                reg_lmp = gr.Textbox(label="LMP Date (YYYY-MM-DD)", placeholder="2026-01-15")
            
            with gr.Row():
                reg_village = gr.Textbox(label="Village", placeholder="Village name")
                reg_contact = gr.Textbox(label="Contact", placeholder="Phone number")
                reg_language = gr.Dropdown(
                    choices=LANG_CHOICES, value="hi", label="Language"
                )
            
            with gr.Row():
                reg_blood = gr.Textbox(label="Blood Group", placeholder="B+")
                reg_height = gr.Number(label="Height (cm)", value=155)
                reg_weight = gr.Number(label="Pre-pregnancy Weight (kg)", value=52)
            
            register_btn = gr.Button("✅ Register Patient", variant="primary")
            register_result = gr.Textbox(label="Result", interactive=False)
            
            # Patient selector for profile tab (hidden state)
            selected_patient_id = gr.State(value=None)
            patient_selector = gr.Dropdown(
                label="Select Patient for Profile",
                choices=[],
                interactive=True,
            )
            
            register_btn.click(
                fn=register_new_patient,
                inputs=[reg_name, reg_age, reg_lmp, reg_village, reg_contact, reg_language, reg_blood, reg_height, reg_weight],
                outputs=[register_result],
            )
            
            patients_refresh_btn.click(
                fn=load_patients_table,
                outputs=[patients_table_display, patient_selector],
            )
        
        # ============================================================
        # TAB: PATIENT PROFILE (Chat + EHR + Schedule + Ration)
        # ============================================================
        with gr.Tab("👩 Patient Profile"):
            
            # Patient selector
            profile_patient_selector = gr.Dropdown(
                label="Select Patient",
                choices=[],
                interactive=True,
            )
            load_profile_btn = gr.Button("📋 Load Profile", variant="primary")
            
            # Profile display
            profile_info = gr.HTML(label="Patient Info")
            profile_risk = gr.HTML(label="Risk Status")
            profile_ehrs = gr.HTML(label="Health Records")
            profile_lang = gr.State(value="hi")
            
            load_profile_btn.click(
                fn=load_patient_profile,
                inputs=[profile_patient_selector],
                outputs=[profile_info, profile_risk, profile_ehrs, profile_lang],
            )
            
            # Sub-tabs within profile
            with gr.Tab("💬 AI Chat"):
                chatbot = gr.Chatbot(
                    label="Chat with ASHA-Sahayak",
                    height=400,
                    type="messages",
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Type message",
                        placeholder="Ask about the patient's health...",
                        scale=4,
                    )
                    chat_send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    audio_input = gr.Audio(
                        label="🎤 Voice Input",
                        type="filepath",
                        scale=2,
                    )
                    audio_send_btn = gr.Button("🎤 Send Audio", scale=1)
                
                with gr.Row():
                    image_input = gr.Image(label="📷 Upload Image (Lab Report)", type="filepath", scale=2)
                    image_text = gr.Textbox(label="Note (optional)", placeholder="Additional context for the image", scale=1)
                    image_send_btn = gr.Button("📷 Send Image", scale=1)
                
                chat_send_btn.click(
                    fn=chat_with_ai,
                    inputs=[profile_patient_selector, chat_input, language_selector, chatbot],
                    outputs=[chatbot, chat_input],
                )
                
                chat_input.submit(
                    fn=chat_with_ai,
                    inputs=[profile_patient_selector, chat_input, language_selector, chatbot],
                    outputs=[chatbot, chat_input],
                )
                
                audio_send_btn.click(
                    fn=chat_with_audio,
                    inputs=[profile_patient_selector, audio_input, language_selector, chatbot],
                    outputs=[chatbot],
                )
                
                image_send_btn.click(
                    fn=chat_with_image,
                    inputs=[profile_patient_selector, image_input, image_text, language_selector, chatbot],
                    outputs=[chatbot],
                )
            
            with gr.Tab("📋 EHR / Lab Records"):
                gr.Markdown("### Add Health Record")
                
                with gr.Row():
                    ehr_hb = gr.Number(label="Hemoglobin (g/dL)", minimum=0, maximum=20)
                    ehr_bp_sys = gr.Number(label="BP Systolic", minimum=0, maximum=300)
                    ehr_bp_dia = gr.Number(label="BP Diastolic", minimum=0, maximum=200)
                    ehr_weight = gr.Number(label="Weight (kg)", minimum=0, maximum=200)
                
                with gr.Row():
                    ehr_sugar_f = gr.Number(label="Fasting Sugar (mg/dL)", minimum=0, maximum=500)
                    ehr_sugar_pp = gr.Number(label="PP Sugar (mg/dL)", minimum=0, maximum=500)
                    ehr_urine = gr.Dropdown(
                        choices=["Normal", "Trace", "+", "++", "+++"],
                        value="Normal",
                        label="Urine Albumin",
                    )
                
                with gr.Row():
                    ehr_usg = gr.Textbox(label="USG Findings", placeholder="Ultrasound findings...")
                    ehr_meds = gr.Textbox(label="Prescribed Medicines", placeholder="Medicines...")
                
                ehr_submit_btn = gr.Button("💾 Save EHR Record", variant="primary")
                ehr_result = gr.Textbox(label="Result", interactive=False)
                
                ehr_submit_btn.click(
                    fn=add_ehr_record,
                    inputs=[profile_patient_selector, ehr_hb, ehr_bp_sys, ehr_bp_dia, ehr_weight, ehr_sugar_f, ehr_sugar_pp, ehr_urine, ehr_usg, ehr_meds],
                    outputs=[ehr_result],
                )
            
            with gr.Tab("📅 Checkup Schedule"):
                schedule_btn = gr.Button("📅 Generate/View Schedule", variant="primary")
                schedule_display = gr.HTML(label="ANC Schedule")
                
                schedule_btn.click(
                    fn=generate_patient_schedule,
                    inputs=[profile_patient_selector],
                    outputs=[schedule_display],
                )
            
            with gr.Tab("🍚 Ration Plan"):
                ration_btn = gr.Button("🍚 Generate Ration Plan", variant="primary")
                ration_display = gr.HTML(label="Ration Plan")
                
                ration_btn.click(
                    fn=generate_patient_ration,
                    inputs=[profile_patient_selector],
                    outputs=[ration_display],
                )
        
        # ============================================================
        # TAB: RATION DISTRIBUTION
        # ============================================================
        with gr.Tab("🍚 Ration Distribution"):
            gr.Markdown("### Weekly Ration Distribution List (All Patients)")
            rations_refresh_btn = gr.Button("🔄 Load Ration Summary", variant="primary")
            village_rations_display = gr.HTML(label="Village Ration Summary")
            
            rations_refresh_btn.click(
                fn=load_village_rations,
                outputs=[village_rations_display],
            )
        
        # ============================================================
        # Auto-load on startup
        # ============================================================
        app.load(
            fn=lambda: load_patients_table()[1],
            outputs=[profile_patient_selector],
        )
    
    return app


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        show_error=True,
    )
