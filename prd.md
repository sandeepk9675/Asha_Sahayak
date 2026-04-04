# ASHA-Sahayak (आशा सहायक) — Product Requirements Document

> **AI-Powered Multilingual Maternal Health Assistant for ASHA Workers**
> Bharat Bricks Hacks 2026 — IISc Bengaluru | Track: Swatantra (Open / Any Indic AI Use Case)

---

## 1. Executive Summary

**Product Name:** ASHA-Sahayak (आशा सहायक)

**One-liner:** An AI-powered multilingual maternal health assistant for ASHA workers that manages pregnant women's profiles, analyzes EHRs via RAG, recommends nutrition rations, schedules checkups, detects emergencies, and provides a village-level dashboard — built entirely on Databricks using API-based model inference.

**Problem:** India's ~10 lakh ASHA workers are the frontline of maternal healthcare in rural areas. Each manages dozens of pregnant women using paper MCP cards, has no decision support for nutrition/ration allocation, struggles with scheduling, and cannot easily spot high-risk pregnancies. Language diversity (22+ languages) makes it worse.

**Solution:** ASHA-Sahayak digitizes and augments the ASHA workflow with AI — via API-based models — that understands pregnancy health data, speaks local languages, and delivers actionable recommendations: ration plans, checkup schedules, risk flags, and emergency alerts — all grounded in PMSMA/ICDS/Saksham guidelines via a RAG pipeline on Databricks.

**Key Constraint:** All AI model inference is done via **API calls** (Sarvam AI, Databricks Foundation Models, HuggingFace Inference API) — zero local model loading. This keeps Databricks Free Edition's ~15GB RAM free for data processing, RAG, and the application itself.

---

## 2. Hackathon Alignment

### Mandatory Requirements Compliance

| Requirement | How We Meet It |
|---|---|
| **Databricks as core** | Delta Lake for all patient data (7 versioned tables), PySpark for data processing & aggregation, FAISS on DBFS for vector search, MLflow for experiment tracking |
| **AI must be central** | RAG pipeline + LLM inference drives every feature: conversational AI, ration recommendation, risk detection, schedule generation |
| **Prefer India-built models** | Sarvam-m (LLM), Sarvam Translate (translation), Param-1 (fallback) — all Indian-built via Sarvam AI |
| **Working demo** | Gradio frontend deployed as Databricks App, reproducible from GitHub |
| **Databricks App or Notebook UI** | Gradio app deployed via Databricks Apps |

### Judging Criteria Mapping

| Criteria (Weight) | Our Approach |
|---|---|
| **Databricks Usage (30%)** | Delta Lake with time travel, PySpark batch processing, Spark SQL dashboards, FAISS on DBFS, MLflow logging, Databricks Apps deployment |
| **Accuracy & Effectiveness (25%)** | RAG grounded in official PMSMA/WHO/Saksham guidelines, rule-based risk engine with medical thresholds, structured ration recommendations |
| **Innovation (25%)** | First-of-kind ASHA worker AI assistant, voice-first Indic language interface, end-to-end maternal care workflow (not just chatbot) |
| **Presentation & Demo (20%)** | Full demo flow: register patient → upload EHR → chat in Hindi → get ration plan → see emergency alert → village dashboard |

### Free Edition Constraints Strategy

| Resource | Constraint | Our Strategy |
|---|---|---|
| Compute | CPU-only, no GPU | **All model inference via API** — no local model loading at all |
| Memory | ~15 GB RAM | RAM used only for Delta Lake, PySpark, FAISS index, and Gradio app |
| Storage | Limited DBFS | Delta tables are compact; FAISS index < 500MB; clean temp files |
| Model Serving | No dedicated endpoints | API-based inference (Sarvam/HF/Databricks APIs) + Databricks Apps for UI |

---

## 3. Target Users

| User | Role | Needs |
|------|------|-------|
| **ASHA Worker** (Primary) | Frontline health worker in rural India | Simple interface, voice input in local language, quick access to patient status, daily task list |
| **ANM / Medical Officer** (Secondary) | Supervises ASHA, reviews high-risk cases | Aggregated dashboard, emergency alerts, patient history summaries |
| **Pregnant Women** (Beneficiary) | Receive care via ASHA | Timely checkups, appropriate nutrition, risk detection |

---

## 4. Core Features

### F1: Patient Profile Management
- ASHA creates profiles: name, age, LMP (Last Menstrual Period), expected delivery date, village, contact, language preference
- Auto-calculates gestational age, trimester, weeks remaining from LMP
- Links to PMSMA MCP card data (Green/Red sticker system)
- Stored in Delta Lake with versioning (time travel for audit trail)

### F2: EHR Upload & Processing
- Upload EHR documents: images of lab reports, prescriptions, handwritten notes
- OCR via API (Sarvam Saarika vision API or Google Cloud Vision API) extracts text
- LLM parses extracted text into structured fields: Hemoglobin, BP, weight, blood group, urine tests, USG findings, HIV/VDRL status
- All EHRs stored in `ehr_records` Delta table with patient ID linkage and raw document path

### F3: Multilingual Conversational AI (Core AI)
- ASHA interacts via **voice (audio)**, **text**, or **image** in any Indian language
- **Speech-to-Text:** Sarvam AI STT API (supports Indian languages natively) or Whisper API
- **Translation:** Sarvam Translate API for 22 Indian languages
- **LLM Reasoning:** Sarvam-m API or Databricks Foundation Model API (Meta Llama) for response generation
- **RAG pipeline** assembles context: patient profile + previous conversations + EHR history + medical guidelines from FAISS vector store
- Responses translated back to ASHA's preferred language via Sarvam Translate API
- All conversations logged in Delta Lake with extracted health updates

### F4: RAG-Powered Ration Recommendation
- **Knowledge base:** ICDS/Poshan 2.0 nutrition guidelines, Saksham scheme norms, WHO maternal nutrition standards
- **Inputs:** patient's trimester, hemoglobin, BMI, weight gain trajectory, conditions (anemia, GDM, hypertension), medical history
- **Output:** Personalized weekly ration plan with:
  - Daily calorie/protein targets (2nd trimester +340 kcal, 3rd trimester +450 kcal)
  - Specific items mapped to Anganwadi availability: rice, dal, eggs, jaggery, groundnuts, milk, green leafy vegetables
  - Supplements: IFA tablets, calcium supplements, dosage and frequency
- Flags severe malnutrition for additional Take Home Ration (THR)

### F5: Checkup Schedule Planner
- Auto-generates ANC schedule per PMSMA guidelines:
  - Minimum 4 ANC visits across trimesters
  - PMSMA clinic visit on 9th of every month
  - USG in 2nd trimester
  - Lab investigations per visit type
- Tracks PENDING / COMPLETED / OVERDUE status
- Adapts frequency for high-risk pregnancies (more frequent visits per e-PMSMA norms)
- Displayed as calendar view in UI

### F6: Emergency Detection & Appointment Scheduling
- AI analyzes health updates from conversation + EHR data against danger signs
- **Risk Classification:** Normal (Green) → High Risk (Red) → Emergency
- If emergency detected: immediate alert + auto-creates appointment at nearest facility
- Shows available slots (simulated facility data for hackathon)
- Escalation: ASHA → ANM → PHC/CHC Medical Officer

### F7: Village Dashboard (Summarized View)
- **Today's View:** Who to visit, what checkups due, medicines to distribute
- **Weekly View:** Ration distribution list for all women, medicines inventory needed
- **Risk Overview:** Color-coded list — Red (emergency/high-risk) at top, Green (normal) at bottom
- **Statistics:** Total registered, by trimester, high-risk count, overdue checkups, upcoming deliveries
- Filterable by risk level, trimester, village/hamlet
- Powered by Spark SQL aggregation queries on Delta tables

---

## 5. Technical Architecture

### 5.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                 ASHA-Sahayak UI (Gradio on Databricks App)      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│   │🎤 Voice  │  │💬 Text   │  │📷 Image  │  │📊 Dashboard  │  │
│   │  Chat    │  │  Chat    │  │  Upload   │  │  (Village)   │  │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
└────────┼──────────────┼────────────┼────────────────┼──────────┘
         │              │            │                │
         ▼              ▼            ▼                │
┌────────────────────────────────────────────────── ──┼──────────┐
│                 API-Based Processing Layer           │          │
│                                                     │          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │          │
│  │ Sarvam STT  │  │ Sarvam       │  │ OCR API   │  │          │
│  │ API (Voice→ │  │ API (HF)     │  │ (Vision)  │  │          │
│  │ Text)       │  │ (Translation)│  │           │  │          │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  │          │
│         └────────────────┼─────────────────┘        │          │
│                          ▼                          │          │
│  ┌──────────────────────────────────────────────┐   │          │
│  │          RAG Pipeline (Databricks)            │   │          │
│  │                                               │   │          │
│  │  ┌──────────────┐    ┌─────────────────────┐ │   │          │
│  │  │ Query         │    │ FAISS Vector Search │ │   │          │
│  │  │ Expansion     │───▶│ (on DBFS)           │ │   │          │
│  │  └──────────────┘    └─────────┬───────────┘ │   │          │
│  │                                ▼             │   │          │
│  │  ┌──────────────────────────────────────┐    │   │          │
│  │  │ Context Assembly (PySpark):          │    │   │          │
│  │  │  • Patient Profile    (Delta Lake)   │    │   │          │
│  │  │  • EHR History        (Delta Lake)   │    │   │          │
│  │  │  • Past Conversations (Delta Lake)   │    │   │          │
│  │  │  • Medical Guidelines (FAISS RAG)    │    │   │          │
│  │  └──────────────┬───────────────────────┘    │   │          │
│  │                 ▼                            │   │          │
│  │  ┌──────────────────────────────────────┐    │   │          │
│  │  │ LLM Inference via API:               │    │   │          │
│  │  │  Sarvam-m API / Databricks FM API    │    │   │          │
│  │  │  (Llama 3 / Mistral hosted)          │    │   │          │
│  │  └──────────────┬───────────────────────┘    │   │          │
│  └─────────────────┼────────────────────────────┘   │          │
│                    ▼                                │          │
│  ┌─────────────────────────┐  ┌─────────────────────▼────────┐ │
│  │ Risk Assessment Engine  │  │ Dashboard Aggregation        │ │
│  │ (Rule-based, PySpark)   │  │ (Spark SQL on Delta Lake)    │ │
│  └────────────┬────────────┘  └──────────────────────────────┘ │
└───────────────┼────────────────────────────────────────────────┘
                ▼
┌────────────────────────────────────────────────────────────────┐
│               Databricks Lakehouse (Delta Lake)                 │
│                                                                 │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐  │
│  │patients_profiles│ │ehr_records     │ │conversations       │  │
│  └────────────────┘ └────────────────┘ └────────────────────┘  │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐  │
│  │checkup_schedules│ │ration_plans   │ │risk_assessments    │  │
│  └────────────────┘ └────────────────┘ └────────────────────┘  │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐  │
│  │appointments    │ │FAISS index     │ │MLflow experiments  │  │
│  │                │ │(vector store)  │ │& model registry    │  │
│  └────────────────┘ └────────────────┘ └────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 API-Based Model Architecture (No Local Models)

| Model | Provider / API | Purpose | Why This Choice |
|-------|---------------|---------|-----------------|
| **Sarvam-m** | Sarvam AI API (`api.sarvam.ai`) | Primary LLM for Indic multilingual reasoning & response generation | India-built, optimized for Indian languages, fast API |
| **Sarvam Saarika** | Sarvam AI API | Speech-to-Text for Indian language audio input | Native support for 10+ Indian languages, better than Whisper for Indic |
| **Sarvam Mayura** | Sarvam AI API | Vision model for OCR on handwritten EHR/lab reports | Handles Indian scripts and handwritten medical records |
| **Sarvam Translate** | Sarvam AI API (`api.sarvam.ai`) | Translation across 22 Indian languages ↔ English | India-built, unified Sarvam ecosystem, single API key for all services |
| **Llama 3 / Mistral** | Databricks Foundation Model API (fallback) | Fallback LLM if Sarvam API is unavailable | Available on Databricks Free Edition via Foundation Model API |
| **multilingual-e5-small** | HuggingFace Inference API (for embeddings) | Document & query embedding for FAISS vector search | Compact, Indic-aware, API-based so no local loading |

**API Key Management:** Stored securely via Databricks Secrets (`dbutils.secrets.get()`)

**Cost:** Most of these APIs have free tiers or hackathon credits sufficient for demo usage.

### 5.3 Data Model (Delta Lake — 7 Tables)

#### `patients_profiles`
| Column | Type | Description |
|--------|------|-------------|
| patient_id | STRING (UUID) | Primary key |
| asha_id | STRING | ASHA worker identifier |
| name | STRING | Patient name |
| age | INT | Patient age |
| village | STRING | Village/hamlet name |
| contact | STRING | Phone number |
| lmp_date | DATE | Last Menstrual Period |
| edd | DATE | Expected Delivery Date (auto: LMP + 280 days) |
| blood_group | STRING | Blood group |
| height_cm | FLOAT | Height in cm |
| pre_pregnancy_weight_kg | FLOAT | Pre-pregnancy weight |
| risk_status | STRING | GREEN / YELLOW / RED |
| language_preference | STRING | Preferred language code (hi, ta, te, kn, etc.) |
| registration_date | TIMESTAMP | When registered |
| last_updated | TIMESTAMP | Last modification |

#### `ehr_records`
| Column | Type | Description |
|--------|------|-------------|
| record_id | STRING (UUID) | Primary key |
| patient_id | STRING | FK → patients_profiles |
| visit_date | DATE | Date of visit/test |
| trimester | INT | 1, 2, or 3 |
| gestational_weeks | INT | Weeks of pregnancy |
| hemoglobin | FLOAT | Hb in g/dL |
| bp_systolic | INT | Systolic BP |
| bp_diastolic | INT | Diastolic BP |
| weight_kg | FLOAT | Current weight |
| urine_albumin | STRING | Normal/Trace/+/++/+++ |
| urine_sugar | STRING | Normal/+ to ++++ |
| blood_sugar_fasting | FLOAT | Fasting blood glucose mg/dL |
| blood_sugar_pp | FLOAT | Post-prandial glucose mg/dL |
| hiv_status | STRING | Reactive/Non-reactive |
| vdrl_status | STRING | Reactive/Non-reactive |
| malaria_status | STRING | Positive/Negative |
| usg_findings | STRING | Ultrasound report text |
| prescribed_medicines | STRING | Current prescriptions |
| raw_document_path | STRING | DBFS path to uploaded image/PDF |
| extracted_text | STRING | OCR-extracted text |
| created_at | TIMESTAMP | Record creation time |

#### `conversations`
| Column | Type | Description |
|--------|------|-------------|
| conversation_id | STRING (UUID) | Primary key |
| patient_id | STRING | FK → patients_profiles |
| asha_id | STRING | ASHA worker ID |
| timestamp | TIMESTAMP | When conversation happened |
| input_type | STRING | AUDIO / TEXT / IMAGE |
| original_input | STRING | Raw input text or audio file path |
| translated_input | STRING | English translation of input |
| ai_response | STRING | LLM response in English |
| translated_response | STRING | Response in ASHA's language |
| extracted_health_updates | STRING (JSON) | Structured health data extracted from conversation |

#### `checkup_schedules`
| Column | Type | Description |
|--------|------|-------------|
| schedule_id | STRING (UUID) | Primary key |
| patient_id | STRING | FK → patients_profiles |
| visit_number | INT | Visit sequence (1, 2, 3, 4+) |
| scheduled_date | DATE | Planned date |
| actual_date | DATE | When actually done (nullable) |
| visit_type | STRING | ROUTINE_ANC / PMSMA_9TH / USG / LAB / EMERGENCY |
| tests_due | STRING (JSON array) | Tests to be conducted |
| status | STRING | PENDING / COMPLETED / OVERDUE / CANCELLED |

#### `ration_plans`
| Column | Type | Description |
|--------|------|-------------|
| plan_id | STRING (UUID) | Primary key |
| patient_id | STRING | FK → patients_profiles |
| week_start_date | DATE | Week start |
| week_end_date | DATE | Week end |
| trimester | INT | Current trimester |
| daily_calorie_target | INT | Target kcal/day |
| protein_target_g | INT | Target protein g/day |
| ration_items | STRING (JSON) | `[{item, quantity_g, frequency}]` |
| supplements | STRING (JSON) | `[{name, dosage, frequency}]` |
| special_notes | STRING | e.g., "severely anemic, increase iron-rich foods" |
| generated_by_model | STRING | Which model/API generated this |

#### `risk_assessments`
| Column | Type | Description |
|--------|------|-------------|
| assessment_id | STRING (UUID) | Primary key |
| patient_id | STRING | FK → patients_profiles |
| assessment_date | TIMESTAMP | When assessed |
| risk_level | STRING | GREEN / YELLOW / RED |
| risk_factors | STRING (JSON array) | List of detected risk factors |
| recommended_action | STRING | What should be done |
| emergency_flag | BOOLEAN | Requires immediate attention? |
| auto_appointment_created | BOOLEAN | Was appointment auto-scheduled? |

#### `appointments`
| Column | Type | Description |
|--------|------|-------------|
| appointment_id | STRING (UUID) | Primary key |
| patient_id | STRING | FK → patients_profiles |
| facility_name | STRING | PHC/CHC/DH name |
| facility_type | STRING | PHC / CHC / DH |
| scheduled_datetime | TIMESTAMP | Appointment time |
| appointment_type | STRING | ROUTINE / EMERGENCY / SPECIALIST |
| status | STRING | SCHEDULED / COMPLETED / MISSED |
| notes | STRING | Additional notes |

### 5.4 RAG Pipeline Detail

**Knowledge Base Documents:**
- PMSMA operational framework & MCP card guidelines
- Safe Motherhood Booklet (government PDF)
- WHO ANC recommendations
- ICDS / Poshan 2.0 / Saksham scheme nutrition norms
- Common pregnancy complications & management protocols
- Drug safety during pregnancy reference

**Pipeline:**
1. **Ingestion (Notebook `03_ingest_knowledge_base.py`):**
   - Parse PDFs → extract text
   - Chunk: 512 tokens with 50-token overlap
   - Embed via `multilingual-e5-small` API (HuggingFace Inference)
   - Store vectors in FAISS index on DBFS
   - Store chunk metadata in Delta table

2. **Query Time (in `rag_pipeline.py`):**
   - Take user query (translated to English) + patient context summary
   - Expand query with medical synonyms (e.g., "anemia" → "low hemoglobin, iron deficiency")
   - Embed query via same embeddings API
   - FAISS top-5 retrieval
   - Fetch from Delta Lake: patient profile, last 5 conversations, latest 3 EHR records
   - Assemble prompt: system instructions + retrieved guidelines + patient context + user query
   - Call LLM API (Sarvam-m)
   - Post-process: extract structured health updates, risk flags from response

3. **Logging:**
   - MLflow logs each RAG call: query, retrieved chunks, context size, LLM response, latency

### 5.5 Risk Assessment Engine

**Rule-Based (Deterministic) — runs after every conversation and EHR upload:**

| Condition | Threshold | Classification | Action |
|-----------|-----------|----------------|--------|
| Severe anemia | Hb < 7 g/dL | 🔴 EMERGENCY | Immediate referral, auto-schedule appointment |
| Severe pre-eclampsia | BP > 160/110 | 🔴 EMERGENCY | Immediate referral |
| Vaginal bleeding | Reported in conversation | 🔴 EMERGENCY | Immediate referral |
| Convulsions / seizures | Reported | 🔴 EMERGENCY | Immediate referral |
| Reduced fetal movement | Reported | 🔴 EMERGENCY | Immediate referral |
| Severe headache + blurred vision | Reported | 🔴 EMERGENCY | Eclampsia warning, referral |
| Pregnancy-induced hypertension | BP > 140/90 | 🔴 HIGH RISK | Red sticker, increase visit frequency |
| Gestational diabetes | Fasting glucose > 126 mg/dL | 🔴 HIGH RISK | Red sticker, dietary plan |
| Moderate anemia | Hb 7–10 g/dL | 🟡 ELEVATED | Increase IFA dosage, diet plan |
| Adolescent pregnancy | Age < 18 | 🔴 HIGH RISK | Red sticker, specialist referral |
| Elderly primigravida | Age > 35 | 🔴 HIGH RISK | Red sticker, specialist referral |
| Previous C-section / stillbirth | History | 🔴 HIGH RISK | Red sticker, hospital delivery |
| Normal | All within range | 🟢 NORMAL | Green sticker, routine ANC |

### 5.6 Ration Recommendation Logic

**Calorie Targets by Trimester (per WHO/ICDS norms):**
| Trimester | Additional kcal/day | Protein g/day | Key Nutrients |
|-----------|-------------------|---------------|---------------|
| 1st | +0 | 55g | Folic acid, Iron |
| 2nd | +340 | 70g | Iron, Calcium, Protein |
| 3rd | +450 | 70g | Iron, Calcium, Protein, DHA |

**Condition-based Adjustments:**
- **Anemia (Hb < 11):** Double IFA tablets, emphasize: jaggery, green leafy vegetables, dates, beetroot
- **GDM:** Low glycemic index diet, reduce rice/sugar, increase dal/vegetables
- **Underweight (BMI < 18.5):** Add extra THR (Take Home Ration) from Anganwadi
- **Overweight (BMI > 25):** Reduce carbs, maintain protein

**Anganwadi-Available Items (Saksham mapped):**
Rice, wheat, dal (lentils), eggs, milk, groundnuts, jaggery, green leafy vegetables, seasonal fruits, soy chunks, fortified flour, IFA tablets (100mg iron + 500mcg folic acid), calcium supplements (500mg)

---

## 6. UI Design

### Design Principles
- **Voice-first:** Large microphone button, prominent on every screen
- **Icon-heavy, minimal text:** Designed for semi-literate rural health workers
- **Color-coded risk:** 🟢 Green (normal) | 🟡 Yellow (watch) | 🔴 Red (high-risk/emergency)
- **Language selector** at top of every screen
- **High contrast, large touch targets** for mobile/touchscreen use
- **Maximum 2 taps** to reach any feature

### Screen Flow

```
┌──────────────┐
│  🏠 HOME /    │
│  DASHBOARD    │──────────────────────────────────┐
│  • Today's    │                                  │
│    schedule   │                                  │
│  • Alerts     │                                  │
│  • Quick stats│                                  │
└──────┬───────┘                                   │
       │                                           │
       ▼                                           ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐
│ 👤 PATIENT   │  │ ➕ NEW       │  │ 📋 WEEKLY RATION PLAN    │
│ LIST         │  │ REGISTRATION │  │ • All women's rations    │
│ • Search bar │  │ • Name, age  │  │ • Medicines inventory    │
│ • Risk badges│  │ • LMP date   │  │ • Distribution checklist │
│ • Trimester  │  │ • Village    │  └──────────────────────────┘
└──────┬───────┘  │ • Language   │
       │          └──────────────┘
       ▼
┌──────────────────────────────────────────┐
│ 👩 PATIENT PROFILE                       │
│ ┌──────┬──────┬──────┬──────┬──────┐     │
│ │ Info │ Chat │ EHR  │Sched.│Ration│     │
│ └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘     │
│    │      │      │      │      │          │
│    ▼      ▼      ▼      ▼      ▼          │
│  Basic  🎤AI    📷Lab  📅ANC  🍚Weekly    │
│  Info   Chat   Upload Visits  Ration     │
│         Voice+       Calendar Plan       │
│         Text+                            │
│         Image                            │
└──────────────────────────────────────────┘
```

### Dashboard Wireframe
```
┌─────────────────────────────────────────────┐
│ 🏠 ASHA-Sahayak         [🌐 Hindi ▼]       │
├─────────────────────────────────────────────┤
│ 📅 Today: April 4, 2026                     │
│                                             │
│ ⚠️ ALERTS (2)                               │
│ ┌─────────────────────────────────────────┐ │
│ │ 🔴 Priya Devi — Hb dropped to 6.5      │ │
│ │ 🔴 Sita Kumari — BP 155/100 reported   │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ 📋 TODAY'S SCHEDULE (5 visits)              │
│ ┌─────────────────────────────────────────┐ │
│ │ 09:00 Meena — Routine ANC (Week 28)    │ │
│ │ 10:30 Lakshmi — Lab tests due          │ │
│ │ 11:00 Kavitha — PMSMA visit            │ │
│ │ 14:00 Radha — Ration delivery          │ │
│ │ 15:30 Anita — Follow-up (GDM)         │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ 📊 VILLAGE OVERVIEW                         │
│ Total: 23  🟢 15  🟡 5  🔴 3               │
│ T1: 4 | T2: 11 | T3: 8                     │
│ Overdue checkups: 3                         │
│                                             │
│ [👤 Patients] [➕ Register] [🍚 Rations]    │
└─────────────────────────────────────────────┘
```

---

## 7. Implementation Plan (12 Hours)

### Hour 0–2: Foundation (Team: All 4 members)

| Task | Owner | Details |
|------|-------|---------|
| Databricks workspace setup | Member 1 | Free Edition signup, invite team, install libraries |
| Delta Lake tables creation | Member 1 | Run notebook `01_setup_delta_tables.py` — all 7 tables |
| Seed patient data | Member 2 | Create 12-15 realistic patient profiles with varied trimesters, risk levels |
| API keys & secrets | Member 2 | Set up Sarvam AI, HuggingFace API keys in Databricks Secrets |
| RAG knowledge base ingestion | Member 3 | Parse PMSMA guidelines, Safe Motherhood booklet, Saksham norms → chunk → embed via API → FAISS on DBFS |
| MLflow experiment setup | Member 4 | Create experiment, define logging schema |
| Gradio app scaffold | Member 4 | Basic Gradio app structure with tabs |

### Hour 2–6: Core AI Pipeline (Split into 2 parallel tracks)

**Track A: Backend AI (Members 1 & 2)**
| Task | Hours | Details |
|------|-------|---------|
| Language pipeline | 2–3 | Wire up Sarvam STT API + Sarvam Translate API for audio→text→English pipeline |
| RAG pipeline | 3–4 | Query expansion + FAISS search + Delta Lake context retrieval + prompt assembly |
| LLM inference endpoint | 3–4 | Sarvam-m API integration, system prompt with pregnancy domain knowledge |
| Conversational AI function | 4–6 | `chat(patient_id, input, input_type, language)` → response + extracted data |

**Track B: Feature Engines + UI (Members 3 & 4)**
| Task | Hours | Details |
|------|-------|---------|
| Risk assessment engine | 2–3 | Rule-based engine from EHR data + conversation NLU |
| Ration recommendation | 3–4 | RAG + LLM to generate personalized ration plans, store in Delta |
| Schedule engine | 2–3 | Auto-generate ANC schedule from LMP, PMSMA norms |
| Gradio UI: Patient CRUD | 3–5 | Registration form, patient list, profile view |
| Gradio UI: Chat interface | 4–6 | Voice button, text input, image upload, chat history |

### Hour 6–10: Integration & Dashboard

| Task | Owner | Details |
|------|-------|---------|
| Integrate AI pipeline → UI | Members 1 & 4 | Connect backend APIs to Gradio interface |
| EHR upload + OCR flow | Member 2 | Image upload → Sarvam Mayura API → parse → Delta |
| Village dashboard | Member 3 | Spark SQL aggregations → Gradio dashboard tab |
| Emergency flow | Member 1 | Risk detection → alert → auto-appointment creation |
| Ration planner view | Member 3 | Weekly ration table for all patients |
| Schedule view | Member 4 | Calendar-style checkup schedule |

### Hour 10–12: Polish & Submission

| Task | Owner | Details |
|------|-------|---------|
| End-to-end testing | All | 3 full demo scenarios with different languages |
| Bug fixes | All | Fix integration issues |
| Record demo video (2 min) | Member 1 | Full flow: register → EHR → chat → ration → dashboard |
| Write README & architecture diagram | Member 2 | GitHub-ready documentation |
| Push to public GitHub | Member 3 | Clean repo, architecture diagram, run instructions |
| Prepare 5-min pitch | Member 4 | Problem → Architecture → Demo → Impact |
| Deploy as Databricks App | Member 1 | Final deployment verification |

---

## 8. Project Structure

```
asha-sahayak/
├── notebooks/
│   ├── 01_setup_delta_tables.py          # Create all 7 Delta Lake tables
│   ├── 02_seed_data.py                   # Load sample patient data
│   ├── 03_ingest_knowledge_base.py       # Parse docs → chunk → embed → FAISS
│   ├── 04_setup_mlflow.py               # MLflow experiment creation
│   └── 05_demo_scenarios.py             # End-to-end test scenarios
├── src/
│   ├── api/
│   │   ├── sarvam_client.py             # Sarvam AI API wrapper (STT, LLM, Vision)
│   │   ├── sarvam_translate_client.py   # Sarvam Translate API wrapper
│   │   └── embeddings_client.py         # multilingual-e5-small HF API wrapper
│   ├── pipeline/
│   │   ├── rag_pipeline.py              # RAG orchestration: retrieve + assemble + infer
│   │   ├── language_pipeline.py         # Audio → Text → English → Response → Local language
│   │   ├── risk_engine.py               # Rule-based risk assessment
│   │   ├── ration_engine.py             # Nutrition recommendation via RAG + LLM
│   │   └── schedule_engine.py           # ANC schedule auto-generation
│   ├── services/
│   │   ├── patient_service.py           # CRUD operations on patients_profiles Delta table
│   │   ├── ehr_service.py               # EHR upload → OCR → parse → store
│   │   ├── chat_service.py              # Conversation management & history
│   │   └── dashboard_service.py         # Spark SQL aggregation for village dashboard
│   └── utils/
│       └── delta_utils.py               # Delta Lake read/write helpers
├── app/
│   └── main.py                          # Gradio frontend (all screens)
├── data/
│   ├── knowledge_base/                  # PMSMA PDFs, Saksham guidelines, WHO ANC
│   │   ├── pmsma_guidelines.txt
│   │   ├── safe_motherhood_booklet.txt
│   │   ├── saksham_nutrition_norms.txt
│   │   └── who_anc_recommendations.txt
│   └── seed/                            # Sample patient CSVs
│       ├── patients.csv
│       ├── ehr_samples.csv
│       └── facilities.csv
├── README.md                            # What, why, architecture diagram, how to run, demo
├── architecture_diagram.png             # Visual architecture
└── requirements.txt                     # Python dependencies
```

---

## 9. Government Scheme Alignment

| Scheme | How ASHA-Sahayak Supports It |
|--------|------------------------------|
| **PMSMA** (Pradhan Mantri Surakshit Matritva Abhiyan) | Auto-schedules 9th-of-month PMSMA visits, Green/Red sticker risk classification, tracks HRP per e-PMSMA, generates ANC checkup plans |
| **Saksham / Poshan 2.0** | RAG-grounded ration recommendations per ICDS norms, maps to Anganwadi-available items, tracks THR distribution |
| **JSSK** (Janani Shishu Suraksha Karyakram) | Free transport flagging for difficult areas, facility delivery planning for high-risk cases |
| **e-PMSMA** | Name-based HRP line listing, individual HRP tracking, follow-up scheduling, ASHA incentive tracking support |

---

## 10. Verification & Demo Scenarios

### Scenario 1: New Patient Registration + First Chat (Hindi Voice)
1. ASHA taps "New Registration"
2. Enters: Meena Devi, age 24, LMP: Jan 15 2026, Village: Rampur, Language: Hindi
3. System auto-calculates: 11 weeks pregnant, 1st trimester, EDD: Oct 22 2026
4. ASHA taps Chat → records Hindi voice message: "Meena ko subah ulti ho rahi hai aur chakkar aa rahe hain" (Meena is having morning vomiting and dizziness)
5. System: STT → translates → RAG retrieves morning sickness guidelines → LLM responds with advice → translated back to Hindi
6. Extracted: symptom=nausea+dizziness, severity=mild → GREEN status maintained

### Scenario 2: EHR Upload + Emergency Detection
1. ASHA opens patient Priya Devi (32 weeks pregnant)
2. Uploads photo of lab report
3. System OCR extracts: Hb = 6.2 g/dL, BP = 165/110
4. Risk engine triggers: 🔴 EMERGENCY — Severe anemia + Severe pre-eclampsia
5. Dashboard shows red alert, auto-creates emergency appointment at District Hospital
6. ASHA sees: "IMMEDIATE REFERRAL REQUIRED — Take to District Hospital TODAY"

### Scenario 3: Village Dashboard + Ration Planning
1. ASHA opens Dashboard
2. Sees: 23 patients total, 3 high-risk (red), 5 elevated (yellow), 15 normal (green)
3. Today's schedule: 5 visits listed with times and visit types
4. Taps "Weekly Rations" → sees table: each woman's name, ration items, supplements
5. Aggregated view: total rice needed = 45kg, dal = 15kg, eggs = 200, IFA tablets = 160

---

## 11. Scope Boundaries

### In Scope
- Patient profile CRUD with auto-gestational calculations
- EHR upload with API-based OCR
- Multilingual conversational AI (voice + text + image) via APIs
- RAG-powered medical Q&A grounded in government guidelines
- Personalized ration recommendation
- Automated ANC checkup scheduling
- Rule-based emergency detection with auto-appointment
- Village-level aggregated dashboard
- All on Databricks Free Edition with API-based model inference

### Out of Scope
- Real SMS / push notification delivery
- Actual hospital appointment booking via real APIs
- Offline mode / PWA
- User authentication / multi-tenant (assume single ASHA user for demo)
- Real ABHA / Aadhaar integration
- Text-to-Speech (TTS) audio output (stretch goal only)
- Mobile native app (web-based Gradio only)
- ASHA incentive payment tracking
- Post-delivery / neonatal care tracking

---

## 12. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model Inference | **All via API** — no local models | Frees ~15GB RAM for data processing; avoids GPU dependency; faster iteration |
| LLM | **Sarvam-m API** (primary), Databricks FM API (fallback) | India-built, Indic-optimized; Databricks FM gives platform integration bonus |
| STT | **Sarvam Saarika API** | Native Indian language support, better than Whisper for Indic accents |
| Translation | **Sarvam Translate API** | 22 Indian languages, unified Sarvam ecosystem, single API key |
| OCR / Vision | **Sarvam Mayura API** | Handles Indian scripts and handwritten medical records |
| Embeddings | **multilingual-e5-small via HF API** | Indic-aware, compact, API-based |
| Vector Store | **FAISS on DBFS** | Zero external dependencies, well-supported in Databricks |
| Frontend | **Gradio** | Best for chat+audio+image UIs, native Databricks App deployment |
| Track | **Swatantra** | Healthcare triage assistant — strong social impact story |
| Data Storage | **Delta Lake (7 tables)** | Versioning, time travel, ACID — maximizes Databricks usage score |

---

## 13. Dependencies & Prerequisites

- [ ] Databricks Free Edition workspace created
- [ ] Sarvam AI API key (free tier available at `sarvam.ai`)
- [ ] HuggingFace API token (free tier)
- [ ] PMSMA guidelines PDF, Safe Motherhood booklet, Saksham nutrition norms (public documents)
- [ ] Python libraries: `gradio`, `faiss-cpu`, `langchain`, `pyspark`, `mlflow`, `requests`, `pandas`
- [ ] GitHub public repository

---

## 14. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sarvam API rate limits / downtime | Core features blocked | Fallback to Databricks Foundation Model API (Llama 3) + basic regex OCR |
| FAISS index too large for DBFS | RAG pipeline fails | Limit knowledge base to ~100 critical documents, compress index |
| Gradio audio recording issues in browser | Voice feature broken | Fallback to text-only input, pre-record demo audio files |
| Sarvam API rate limits / latency | Slow user experience | Cache common translations, batch translate, show loading indicator |
| 12-hour time constraint | Features incomplete | Priority order: F1 (profiles) → F3 (chat AI) → F6 (risk) → F7 (dashboard) → F4 (ration) → F5 (schedule) → F2 (EHR) |

---

*Last updated: April 4, 2026*
*Bharat Bricks Hacks 2026 — IISc Bengaluru — Track: Swatantra*