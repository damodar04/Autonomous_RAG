import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import duckdb
import pandas as pd
import plotly.graph_objects as go
import json
import datetime

# Load environment variables
load_dotenv()

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Clinical Trials Architect",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# PREMIUM LIGHT THEME CSS
# ----------------------------------------------------
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #F8F9FA !important;
        color: #212529 !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #1A1A1A !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    p, li, label, .stMarkdown, .stText {
        color: #4A4A4A !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Inputs & Text Areas */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
    }
    
    /* SelectBox Specific Fixes */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    /* Dropdown Menu */
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    li[data-baseweb="option"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* Selected Option in Dropdown */
    li[aria-selected="true"] {
        background-color: #E9ECEF !important;
        color: #000000 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #6C757D !important;
        border: none !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #007BFF !important;
        border-bottom: 3px solid #007BFF !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #007BFF, #0056b3) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.2) !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 123, 255, 0.3) !important;
    }

    /* Metrics & Cards */
    div[data-testid="stMetricValue"] {
        color: #007BFF !important;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #F0F0F0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        color: #212529 !important;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #4A4A4A !important;
    }
    
    /* Custom Containers */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if 'llm_config' not in st.session_state:
    st.session_state.llm_config = None
if 'knowledge_stores' not in st.session_state:
    st.session_state.knowledge_stores = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'mimic_db' not in st.session_state:
    st.session_state.mimic_db = None
if 'mimic_stats' not in st.session_state:
    st.session_state.mimic_stats = None
if 'last_generated_design' not in st.session_state:
    st.session_state.last_generated_design = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'compliance_result' not in st.session_state:
    st.session_state.compliance_result = None
if 'original_protocol' not in st.session_state:
    st.session_state.original_protocol = None

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.title("AI Clinical Trials Architect")
st.markdown("#### Multi-Agent Autonomous Protocol Designer")
st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR (same as original)
# ----------------------------------------------------
with st.sidebar:
    st.markdown("## 🧬 Control Center")
    st.caption("v2.1 • AI Director Active")
    st.divider()

    st.markdown("### 🔑 API Configuration")
    api_key = st.text_input("DeepSeek API Key", value=os.getenv("DEEPSEEK_API_KEY", ""), type="password")
    base_url = st.text_input("Base URL", value="https://api.deepseek.com")

    st.markdown("### 📂 Data Paths")
    pubmed_path = st.text_input("PubMed Data Path", value="./data/pubmed_articles")
    fda_path = st.text_input("FDA Guidelines Path", value="./data/fda_guidelines")
    ethics_path = st.text_input("Ethics Documents Path", value="./data/ethical_guidelines")
    mimic_path = st.text_input("MIMIC Database Path", value="./data/mimic_db")

    st.divider()

    # MODEL INITIALIZATION (unchanged)
    if st.button("🚀 Initialize System", use_container_width=True):
        try:
            with st.spinner("Initializing DeepSeek models..."):
                llm_config = {
                    "planner": ChatOpenAI(
                        model="deepseek-chat", temperature=0.0,
                        openai_api_key=api_key, openai_api_base=base_url),
                    "drafter": ChatOpenAI(
                        model="deepseek-chat", temperature=0.2,
                        openai_api_key=api_key, openai_api_base=base_url),
                    "sql_coder": ChatOpenAI(
                        model="deepseek-chat", temperature=0.0,
                        openai_api_key=api_key, openai_api_base=base_url),
                    "director": ChatOpenAI(
                        model="deepseek-reasoner", temperature=0.0,
                        openai_api_key=api_key, openai_api_base=base_url),
                }

                # Embeddings
                # Using LocalTFIDF to avoid TensorFlow/Keras dependency issues ('NoneType' object has no attribute 'cadam32bit_grad_fp32')
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    class LocalTFIDF:
                        def __init__(self):
                            self.vectorizer = None
                        def embed_documents(self, texts):
                            try:
                                # Handle empty or stop-word only texts
                                clean_texts = [t if t and t.strip() else "placeholder" for t in texts]
                                self.vectorizer = TfidfVectorizer(max_features=768, token_pattern=r"(?u)\b\w+\b")
                                X = self.vectorizer.fit_transform(clean_texts)
                                return X.toarray().tolist()
                            except ValueError:
                                # Fallback for empty vocabulary
                                return [[0.0] * 768 for _ in texts]
                        def embed_query(self, t):
                            if self.vectorizer is None:
                                return [0.0] * 768
                            try:
                                X = self.vectorizer.transform([t])
                                return X.toarray()[0].tolist()
                            except:
                                return [0.0] * 768
                    embedding_model = LocalTFIDF()
                except Exception as e:
                    st.error(f"Embedding init failed: {e}")
                    embedding_model = None

                llm_config["embedding_model"] = embedding_model
                st.session_state.llm_config = llm_config

            st.success("Models initialized successfully!")
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
    # ----------------------------------------------------
    # LOAD KNOWLEDGE BASE
    # ----------------------------------------------------
    if st.button("📚 Load Knowledge Base", use_container_width=True):
        if st.session_state.llm_config is None:
            st.warning("Please initialize the system first.")
        else:
            with st.spinner("Loading Knowledge Base..."):
                knowledge_stores = {}
                embedding_model = st.session_state.llm_config["embedding_model"]

                # -------- PUBMED --------
                if os.path.exists(pubmed_path):
                    pubmed_docs = []
                    for f in os.listdir(pubmed_path):
                        if f.endswith(".txt"):
                            try:
                                loader = TextLoader(os.path.join(pubmed_path, f), encoding="utf-8")
                                pubmed_docs.extend(loader.load())
                            except:
                                pass
                    if pubmed_docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        chunks = splitter.split_documents(pubmed_docs)
                        chunks = [c for c in chunks if c.page_content.strip()] # Filter empty
                        if chunks:
                            db = FAISS.from_documents(chunks, embedding_model)
                            knowledge_stores["pubmed_retriever"] = db.as_retriever(search_kwargs={"k": 3})
                            st.success(f"PubMed Loaded ✓  ({len(pubmed_docs)} docs)")
                    else:
                        st.warning("PubMed folder is empty.")

                # -------- FDA --------
                if os.path.exists(fda_path):
                    fda_docs = []
                    for f in os.listdir(fda_path):
                        file_path = os.path.join(fda_path, f)
                        if f.endswith(".txt"):
                            try:
                                loader = TextLoader(file_path, encoding="utf-8")
                                fda_docs.extend(loader.load())
                            except:
                                pass
                        if f.endswith(".pdf"):
                            try:
                                loader = PyPDFLoader(file_path)
                                fda_docs.extend(loader.load())
                            except:
                                pass
                    if fda_docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        chunks = splitter.split_documents(fda_docs)
                        chunks = [c for c in chunks if c.page_content.strip()] # Filter empty
                        if chunks:
                            db = FAISS.from_documents(chunks, embedding_model)
                            knowledge_stores["fda_retriever"] = db.as_retriever(search_kwargs={"k": 3})
                            st.success(f"FDA Guidelines Loaded ✓ ({len(fda_docs)} docs)")
                    else:
                        st.warning("No FDA documents found.")

                # -------- ETHICS --------
                if os.path.exists(ethics_path):
                    ethics_docs = []
                    for f in os.listdir(ethics_path):
                        if f.endswith(".txt"):
                            try:
                                loader = TextLoader(os.path.join(ethics_path, f), encoding="utf-8")
                                ethics_docs.extend(loader.load())
                            except:
                                pass
                    if ethics_docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = splitter.split_documents(ethics_docs)
                        chunks = [c for c in chunks if c.page_content.strip()] # Filter empty
                        if chunks:
                            db = FAISS.from_documents(chunks, embedding_model)
                            knowledge_stores["ethics_retriever"] = db.as_retriever(search_kwargs={"k": 2})
                            st.success(f"Ethics Documents Loaded ✓ ({len(ethics_docs)} docs)")
                    else:
                        st.warning("Ethics folder empty.")

                st.session_state.knowledge_stores = knowledge_stores
                st.success("Knowledge Base fully loaded!")

    # ----------------------------------------------------
    # LOAD MIMIC DB
    # ----------------------------------------------------
    if st.button("🏥 Load MIMIC Database", use_container_width=True):
        if st.session_state.llm_config is None:
            st.warning("Initialize the system first.")
        else:
            try:
                conn = duckdb.connect(":memory:")

                patients_file = os.path.join(mimic_path, "PATIENTS.csv")
                diagnoses_file = os.path.join(mimic_path, "DIAGNOSES_ICD.csv")

                if os.path.exists(patients_file):
                    conn.execute(f"CREATE TABLE patients AS SELECT * FROM read_csv_auto('{patients_file}')")
                if os.path.exists(diagnoses_file):
                    conn.execute(f"CREATE TABLE diagnoses_icd AS SELECT * FROM read_csv_auto('{diagnoses_file}')")

                stats = {
                    "total_patients": conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
                    "total_diagnoses": conn.execute("SELECT COUNT(*) FROM diagnoses_icd").fetchone()[0],
                    "gender_dist": conn.execute("SELECT GENDER, COUNT(*) AS cnt FROM patients GROUP BY GENDER").fetchdf(),
                    "top_diagnoses": conn.execute("SELECT ICD9_CODE, COUNT(*) AS cnt FROM diagnoses_icd GROUP BY ICD9_CODE ORDER BY cnt DESC LIMIT 10").fetchdf()
                }

                st.session_state.mimic_db = conn
                st.session_state.mimic_stats = stats

                st.success("MIMIC Database Loaded Successfully ✓")

            except Exception as e:
                st.error(f"Failed to load MIMIC: {str(e)}")

# ----------------------------------------------------
# TABS
# ----------------------------------------------------

tab1, tab2, tab_eval, tab3, tab4, tab5 = st.tabs([
    "💬 Chat Architect",
    "🏗️ Trial Designer",
    "⚖️ Evaluator",
    "🔍 Knowledge Base",
    "📊 MIMIC Analytics",
    "ℹ️ System Info"
])

# ----------------------------------------------------
# TAB 1 - CHAT ARCHITECT (unchanged functionality)
# ----------------------------------------------------
with tab1:
    st.header("Chat with AI Clinical Trials Architect")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask about clinical trial design…"):
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            try:
                llm = st.session_state.llm_config["planner"]
                response = llm.invoke(user_prompt)
                reply = response.content
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ----------------------------------------------------
# TAB 2 — NEW TRIAL DESIGNER (UPDATED)
# ----------------------------------------------------
with tab2:
    st.header("Clinical Trial Design Generator (Enhanced Controls)")

    colA, colB = st.columns(2)

    # LEFT SIDE — DISEASE & DRUG DETAILS
    with colA:
        st.subheader("🧪 Disease & Population")
        disease = st.text_input("Disease / Condition", "Type 2 Diabetes")
        target_population = st.text_area("Target Population",
            "Adults aged 30–65 with uncontrolled T2D (HbA1c ≥ 8%).")

        st.subheader("💊 Drug Information")
        drug_name = st.text_input("Drug Name", "Dapagliflozin")
        drug_class = st.text_input("Drug Class", "SGLT2 inhibitor")
        dose = st.number_input("Dose (mg)", 1, 500, 10)
        frequency = st.selectbox("Dose Frequency", ["Once Daily", "Twice Daily", "Weekly"])
        route = st.selectbox("Route", ["Oral", "IV", "Subcutaneous"])

        comparator = st.selectbox("Comparator Arm", ["Placebo", "Standard of Care", "Active Comparator"])

    # RIGHT SIDE — STUDY DESIGN
    with colB:
        st.subheader("🏗️ Study Design")
        study_type = st.selectbox("Study Type", [
            "Randomized Controlled Trial",
            "Open Label",
            "Crossover Study"
        ])
        phase = st.selectbox("Trial Phase", ["Phase I", "Phase II", "Phase III", "Phase IV"])
        duration = st.number_input("Duration (weeks)", 1, 520, 24)
        randomization = st.selectbox("Randomization Ratio", ["1:1", "2:1", "3:1"])
        blinding = st.selectbox("Blinding Type", ["Double-Blind", "Single-Blind", "Open Label"])

        st.subheader("📈 Endpoints")
        primary_ep = st.text_input("Primary Endpoint", "Change in HbA1c from baseline at Week 24")
        secondary_ep = st.text_area("Secondary Endpoints",
            "Fasting plasma glucose, weight, renal markers")
        safety_params = st.text_area("Safety Monitoring",
            "Adverse events, hypoglycemia, renal function")

    st.divider()

    # GENERATE BUTTON
    if st.button("Generate Protocol", use_container_width=True):
        if st.session_state.llm_config is None:
            st.warning("Initialize the system first!")
        else:
            llm = st.session_state.llm_config["director"]

            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            prompt = f"""
Generate a complete clinical trial protocol.

Date: {current_date}
Disease: {disease}
Target Population: {target_population}

Drug: {drug_name}
Class: {drug_class}
Dose: {dose} mg
Frequency: {frequency}
Route: {route}

Comparator: {comparator}

Study Type: {study_type}
Phase: {phase}
Duration: {duration} weeks
Randomization: {randomization}
Blinding: {blinding}

Primary Endpoint: {primary_ep}
Secondary Endpoints: {secondary_ep}
Safety: {safety_params}

Return a structured protocol with:
- Title
- Background
- Objectives
- Inclusion / Exclusion
- Endpoints
- Study Procedures
- Arms Description
- Statistical Plan
- Safety Monitoring

INSTRUCTIONS:
1. Use {current_date} as the protocol creation date.
2. Ensure all sections are fully detailed and comprehensive.
3. DO NOT use placeholders like "[Insert Date]" or "[Insert Name]". Fill in all details based on the provided parameters and medical knowledge.
4. Ensure the protocol is professional and ready for review.
"""

            with st.spinner("Generating protocol…"):
                try:
                    response = llm.invoke(prompt)
                    protocol = response.content
                    st.session_state.last_generated_design = protocol
                    st.session_state.original_protocol = protocol

                    st.subheader("📄 Generated Protocol")
                    st.markdown(protocol)

                    st.download_button(
                        "Download Protocol",
                        data=protocol,
                        file_name=f"{drug_name}_protocol.txt"
                    )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
# ----------------------------------------------------
# TAB 3 — EVALUATOR (ENHANCED)
# ----------------------------------------------------
with tab_eval:
    st.header("⚖️ Protocol Evaluator 360°")

    if not st.session_state.last_generated_design:
        st.info("⚠️ Generate a protocol first in the Trial Designer tab.")
        
        st.markdown("---")
        st.markdown("### 🧪 Quick Test")
        if st.button("🎲 Load Sample Protocol (Demo)", use_container_width=True):
            st.session_state.last_generated_design = """
# PROTOCOL TITLE: Phase III Study of Dapagliflozin in Heart Failure

## 1. OBJECTIVES
To evaluate the efficacy and safety of Dapagliflozin 10mg once daily in patients with heart failure and reduced ejection fraction (HFrEF).

## 2. STUDY DESIGN
- **Phase:** III
- **Design:** Randomized, Double-Blind, Placebo-Controlled
- **Duration:** 18 months
- **Arms:** 
  1. Dapagliflozin 10mg QD
  2. Placebo QD

## 3. POPULATION
- **Inclusion:** Adults >18 years, NYHA class II-IV, LVEF <= 40%, NT-proBNP >= 600 pg/mL.
- **Exclusion:** eGFR < 30 mL/min, SBP < 95 mmHg, Type 1 Diabetes.

## 4. ENDPOINTS
- **Primary:** Time to first occurrence of worsening heart failure or cardiovascular death.
- **Secondary:** Total number of HF hospitalizations, Change in KCCQ score at 8 months.

## 5. STATISTICAL PLAN
Cox proportional hazards model will be used for the primary endpoint. Sample size calculated to detect 15% risk reduction with 90% power.

## 6. SAFETY
Monitoring for volume depletion, renal function, and ketoacidosis.
"""
            st.rerun()
    else:
        st.subheader("Current Protocol")
        with st.expander("📄 View Protocol Text", expanded=False):
            st.markdown(st.session_state.last_generated_design)

        # ----------------------------------------------------
        # MANUAL IMPROVEMENT WORKFLOW
        # ----------------------------------------------------
        # ----------------------------------------------------
        # MANUAL IMPROVEMENT WORKFLOW
        # ----------------------------------------------------
        if st.session_state.evaluation_results:
            st.divider()
            st.markdown("### ✨ Protocol Refinement")
            st.info("Define your target score ranges and automatically improve the protocol.")
            
            # Get current scores to set as defaults
            current_scores = st.session_state.evaluation_results.get("scores", {})
            
            # Helper to safely get int score
            def get_score(key):
                val = current_scores.get(key, 5)
                try:
                    return int(float(val))
                except:
                    return 5

            s_rigor = get_score("Scientific Rigor")
            s_reg = get_score("Regulatory Compliance")
            s_ethics = get_score("Ethical Soundness")
            s_feas = get_score("Feasibility")
            s_patient = get_score("Patient Centricity")
            
            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
            
            # Range Sliders (Min, Max) - Default to (Current, 10)
            with col_t1: t_rigor = st.slider(f"Scientific Rigor (Curr: {s_rigor})", 0, 10, (s_rigor, 10), key="slider_rigor")
            with col_t2: t_reg = st.slider(f"Reg. Compliance (Curr: {s_reg})", 0, 10, (s_reg, 10), key="slider_reg")
            with col_t3: t_ethics = st.slider(f"Ethical Soundness (Curr: {s_ethics})", 0, 10, (s_ethics, 10), key="slider_ethics")
            with col_t4: t_feas = st.slider(f"Feasibility (Curr: {s_feas})", 0, 10, (s_feas, 10), key="slider_feas")
            with col_t5: t_patient = st.slider(f"Patient Centricity (Curr: {s_patient})", 0, 10, (s_patient, 10), key="slider_patient")
            
            if st.button("✨ Auto-Improve Protocol to Targets", use_container_width=True):
                if st.session_state.llm_config is None:
                    st.warning("Please initialize the system first.")
                else:
                    director = st.session_state.llm_config["director"]
                    current_protocol = st.session_state.last_generated_design
                    
                    # MERGE RECOMMENDATIONS
                    recs = st.session_state.evaluation_results.get("recommendations", [])
                    
                    # Check for Regulatory Audit results
                    reg_recs = []
                    if st.session_state.compliance_result:
                        reg_recs = st.session_state.compliance_result.get("recommendations", [])
                        if reg_recs:
                            st.info(f"Including {len(reg_recs)} regulatory recommendations in improvement plan.")
                    
                    all_recs = recs + reg_recs
                    
                    with st.spinner("Refining protocol to meet target score ranges..."):
                        try:
                            refine_prompt = f"""
                            You are an expert Clinical Trial Designer. Your goal is to IMPROVE the provided protocol to strictly meet the target score ranges.
                            
                            CURRENT PROTOCOL:
                            {current_protocol}
                            
                            CRITICAL FEEDBACK TO ADDRESS:
                            {json.dumps(all_recs)}
                            
                            TARGET SCORE RANGES (You MUST optimize for these):
                            - Scientific Rigor: {t_rigor[0]} - {t_rigor[1]} / 10
                            - Regulatory Compliance: {t_reg[0]} - {t_reg[1]} / 10
                            - Ethical Soundness: {t_ethics[0]} - {t_ethics[1]} / 10
                            - Feasibility: {t_feas[0]} - {t_feas[1]} / 10
                            - Patient Centricity: {t_patient[0]} - {t_patient[1]} / 10
                            
                            INSTRUCTIONS:
                            1. Analyze the current protocol and the feedback.
                            2. CHAIN OF THOUGHT: Briefly explain (in <thought> tags) what specific changes you will make to hit the target scores (e.g., "To increase Feasibility to 9, I will remove the weekly biopsy requirement").
                            3. Rewrite the FULL protocol.
                            4. Ensure the new protocol is coherent, professional, and explicitly addresses the weaknesses identified.
                            
                            Return the FULL improved protocol text.
                            """
                            refine_response = director.invoke(refine_prompt)
                            improved_protocol = refine_response.content
                            
                            # Clean up thought tags if present in final output (optional, but good for UX)
                            if "<thought>" in improved_protocol:
                                improved_protocol = improved_protocol.split("</thought>")[-1].strip()

                            st.session_state.last_generated_design = improved_protocol
                            st.success("Protocol updated successfully! Scroll up to 'View Protocol Text' to see changes.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Improvement failed: {str(e)}")

        # ----------------------------------------------------
        # PROTOCOL DIFF VIEWER
        # ----------------------------------------------------
        if st.session_state.original_protocol and st.session_state.last_generated_design != st.session_state.original_protocol:
            st.divider()
            st.markdown("### 🔄 Protocol Changes")
            with st.expander("View Differences (Original vs. New)", expanded=True):
                import difflib
                
                original = st.session_state.original_protocol.splitlines()
                modified = st.session_state.last_generated_design.splitlines()
                
                diff = difflib.ndiff(original, modified)
                
                diff_text = []
                for line in diff:
                    if line.startswith('- '):
                        diff_text.append(f":red[- {line[2:]}]")
                    elif line.startswith('+ '):
                        diff_text.append(f":green[+ {line[2:]}]")
                    elif line.startswith('? '):
                        continue
                    else:
                        diff_text.append(f"  {line[2:]}")
                
                st.markdown("  \n".join(diff_text))

        st.divider()
        if st.button("🚀 Run Comprehensive Evaluation", use_container_width=True):
            try:
                llm = st.session_state.llm_config["director"]

                eval_prompt = f"""
Evaluate the following clinical trial protocol.

PROTOCOL:
{st.session_state.last_generated_design}

Provide a detailed analysis in JSON format with the following structure:
{{
    "scores": {{
        "Scientific Rigor": <0-10>,
        "Regulatory Compliance": <0-10>,
        "Ethical Soundness": <0-10>,
        "Feasibility": <0-10>,
        "Patient Centricity": <0-10>
    }},
    "reasoning": {{
        "Scientific Rigor": "...",
        "Regulatory Compliance": "...",
        "Ethical Soundness": "...",
        "Feasibility": "...",
        "Patient Centricity": "..."
    }},
    "swot": {{
        "strengths": ["...", "..."],
        "weaknesses": ["...", "..."],
        "opportunities": ["...", "..."],
        "threats": ["...", "..."]
    }},
    "recommendations": ["...", "..."]
}}
"""

                with st.spinner("🔍 Analyzing protocol against global standards..."):
                    response = llm.invoke(eval_prompt)
                    text = response.content

                    # Clean JSON
                    if "```" in text:
                        text = text.split("```")[1]
                    if "json" in text:
                        text = text.replace("json", "")

                    result = json.loads(text)
                    st.session_state.evaluation_results = result

            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

        # If evaluation exists, display metrics
        if st.session_state.evaluation_results:
            res = st.session_state.evaluation_results
            scores = res["scores"]
            reasoning = res["reasoning"]
            swot = res.get("swot", {})
            recs = res.get("recommendations", [])

            # --- SECTION 1: SCORECARD ---
            st.markdown("### 📊 Performance Scorecard")
            
            # Calculate Average
            avg_score = sum(scores.values()) / len(scores)
            
            col_main, col_radar = st.columns([1, 1])
            
            with col_main:
                st.metric("Overall Quality Score", f"{avg_score:.1f}/10")
                st.progress(avg_score / 10)
                
                st.markdown("#### Dimension Breakdown")
                for metric, score in scores.items():
                    st.markdown(f"**{metric}**")
                    st.progress(score / 10)
                    st.caption(f"Score: {score}/10")

            with col_radar:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(scores.values()),
                    theta=list(scores.keys()),
                    fill='toself',
                    name="Scores",
                    line=dict(color='#00ADB5'),
                    fillcolor='rgba(0, 173, 181, 0.3)'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10], gridcolor='#444'),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FAFAFA'),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # --- SECTION 2: SWOT ANALYSIS ---
            st.markdown("### 🧭 SWOT Analysis")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ✅ Strengths")
                for s in swot.get("strengths", []):
                    st.success(f"- {s}")
                    
                st.markdown("#### ⚠️ Weaknesses")
                for w in swot.get("weaknesses", []):
                    st.error(f"- {w}")
            
            with c2:
                st.markdown("#### 💡 Opportunities")
                for o in swot.get("opportunities", []):
                    st.info(f"- {o}")
                    
                st.markdown("#### 🛡️ Threats")
                for t in swot.get("threats", []):
                    st.warning(f"- {t}")

            st.divider()

            # --- SECTION 3: RECOMMENDATIONS ---
            st.markdown("### 🚀 Actionable Recommendations")
            for i, rec in enumerate(recs, 1):
                st.markdown(f"""
                <div style="background-color:#262730; padding:15px; border-radius:8px; margin-bottom:10px; border-left: 4px solid #00ADB5;">
                    <strong>{i}.</strong> {rec}
                </div>
                """, unsafe_allow_html=True)

            # --- SECTION 4: DETAILED REASONING ---
            with st.expander("🧠 View Detailed Reasoning for Scores"):
                for m, r in reasoning.items():
                    st.markdown(f"**{m}**: {r}")
                    st.divider()



        # --- SECTION 5: REGULATORY COMPLIANCE DEEP DIVE ---
        st.divider()
        st.subheader("⚖️ Regulatory Compliance Deep Dive")
        
        reg_framework = st.selectbox(
            "Select Regulatory Framework for Compliance Check",
            ["WHO (International GCP)", "USA (FDA 21 CFR)", "India (CDSCO / ICMR)"]
        )
        
        if st.button(f"Check Compliance with {reg_framework}", use_container_width=True):
            if st.session_state.llm_config is None:
                st.warning("Please initialize the system first.")
            else:
                try:
                    llm = st.session_state.llm_config["director"]
                    
                    reg_prompt = f"""
                    Evaluate the following clinical trial protocol against {reg_framework} guidelines.
                    
                    PROTOCOL:
                    {st.session_state.last_generated_design}
                    
                    Provide a detailed compliance analysis in JSON format:
                    {{
                        "compliance_score": <0-100>,
                        "status": "<Compliant/Partial/Non-Compliant>",
                        "key_issues": ["...", "..."],
                        "missing_elements": ["...", "..."],
                        "recommendations": ["...", "..."]
                    }}
                    """
                    
                    with st.spinner(f"Auditing against {reg_framework} standards..."):
                        response = llm.invoke(reg_prompt)
                        text = response.content
                        
                        if "```" in text:
                            text = text.split("```")[1]
                        if "json" in text:
                            text = text.replace("json", "")
                            
                        st.session_state.compliance_result = json.loads(text)
                        
                except Exception as e:
                    st.error(f"Regulatory check failed: {str(e)}")

        # Display Compliance Results (Persistent)
        if st.session_state.compliance_result:
            reg_result = st.session_state.compliance_result
            score = reg_result["compliance_score"]
            status = reg_result["status"]
            
            st.markdown(f"### 📋 Audit Report: {reg_framework}")
            
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Compliance Score", f"{score}%")
                if score >= 80:
                    st.success(status)
                elif score >= 50:
                    st.warning(status)
                else:
                    st.error(status)
                    
            with c2:
                if reg_result["key_issues"]:
                    st.markdown("**Key Issues:**")
                    for issue in reg_result["key_issues"]:
                        st.error(f"- {issue}")
                
                if reg_result["missing_elements"]:
                    st.markdown("**Missing Elements:**")
                    for missing in reg_result["missing_elements"]:
                        st.warning(f"- {missing}")
                        
                st.markdown("**Recommendations:**")
                for rec in reg_result["recommendations"]:
                    st.info(f"- {rec}")

            # AUTO-FIX BUTTON
            st.divider()
            if st.button("✨ Auto-Fix Protocol with Recommendations", use_container_width=True):
                if st.session_state.llm_config is None:
                    st.warning("Initialize system first.")
                else:
                    with st.spinner("Refining protocol based on compliance audit..."):
                        try:
                            llm = st.session_state.llm_config["director"]
                            fix_prompt = f"""
                            Rewrite the following clinical trial protocol to address these regulatory compliance recommendations.
                            
                            ORIGINAL PROTOCOL:
                            {st.session_state.last_generated_design}
                            
                            COMPLIANCE RECOMMENDATIONS ({reg_framework}):
                            {json.dumps(reg_result["recommendations"])}
                            
                            Ensure the new protocol is fully compliant and maintains the original scientific intent.
                            Return the full updated protocol text.
                            """
                            response = llm.invoke(fix_prompt)
                            st.session_state.last_generated_design = response.content
                            st.success("Protocol updated! The 'Current Protocol' view has been refreshed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Auto-fix failed: {str(e)}")

# ----------------------------------------------------
# TAB 4 — KNOWLEDGE BASE SEARCH
# ----------------------------------------------------
with tab3:
    st.header("Knowledge Base Search")

    query = st.text_input("Search Query", "")
    colA, colB, colC = st.columns(3)
    with colA: do_pubmed = st.checkbox("PubMed", True)
    with colB: do_fda = st.checkbox("FDA Guidelines", True)
    with colC: do_ethics = st.checkbox("Ethics", True)

    if st.button("Search", use_container_width=True):
        if st.session_state.knowledge_stores is None:
            st.warning("Load the Knowledge Base first.")
        else:
            results = []

            if do_pubmed and "pubmed_retriever" in st.session_state.knowledge_stores:
                res = st.session_state.knowledge_stores["pubmed_retriever"].get_relevant_documents(query)
                results.append(("PubMed", res))

            if do_fda and "fda_retriever" in st.session_state.knowledge_stores:
                res = st.session_state.knowledge_stores["fda_retriever"].get_relevant_documents(query)
                results.append(("FDA", res))

            if do_ethics and "ethics_retriever" in st.session_state.knowledge_stores:
                res = st.session_state.knowledge_stores["ethics_retriever"].get_relevant_documents(query)
                results.append(("Ethics", res))

            for src, docs in results:
                st.subheader(f"{src} Results")
                for i, d in enumerate(docs, 1):
                    with st.expander(f"{src} Result {i}"):
                        st.write(d.page_content)

# ----------------------------------------------------
# TAB 5 — MIMIC ANALYTICS
# ----------------------------------------------------
with tab4:
    st.header("MIMIC Database Analytics")

    if not st.session_state.mimic_db:
        st.info("Load the MIMIC Database first.")
    else:
        stats = st.session_state.mimic_stats

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", stats["total_patients"])
        with col2:
            st.metric("Total Diagnoses", stats["total_diagnoses"])
        with col3:
            st.metric("Avg Diagnoses / Patient", 
                      round(stats["total_diagnoses"] / stats["total_patients"], 2))

        st.subheader("Gender Distribution")
        st.bar_chart(stats["gender_dist"].set_index("GENDER"))

        st.subheader("Top Diagnoses")
        st.dataframe(stats["top_diagnoses"])

        # SQL query box
        st.divider()
        st.subheader("Custom SQL Query on MIMIC")
        query_default = "SELECT * FROM patients LIMIT 5;"
        sql_query = st.text_area("Enter SQL Query:", query_default)

        if st.button("Run Query"):
            try:
                df = st.session_state.mimic_db.execute(sql_query).fetchdf()
                st.dataframe(df)
            except Exception as e:
                st.error(f"Query error: {str(e)}")

        # ----------------------------------------------------
        # ASK THE DATA (NL-to-SQL)
        # ----------------------------------------------------
        st.divider()
        st.subheader("🗣️ Ask the Data")
        st.info("Ask questions about patients in plain English (e.g., 'Show me all female patients over 50').")
        
        user_query = st.text_input("Enter your question:")
        
        if st.button("Analyze"):
            if st.session_state.llm_config is None:
                st.warning("Please initialize the system first.")
            elif not user_query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Translating question to SQL..."):
                    try:
                        sql_coder = st.session_state.llm_config["sql_coder"]
                        
                        # Fetch sample data for context
                        try:
                            patients_sample = st.session_state.mimic_db.execute("SELECT * FROM patients LIMIT 3").fetchdf().to_string(index=False)
                            diagnoses_sample = st.session_state.mimic_db.execute("SELECT * FROM diagnoses_icd LIMIT 3").fetchdf().to_string(index=False)
                        except:
                            patients_sample = "No sample available"
                            diagnoses_sample = "No sample available"

                        # Enhanced Schema context
                        schema_context = f"""
                        Table: patients
                        Columns: ROW_ID, SUBJECT_ID, GENDER, DOB (Date of Birth), DOD (Date of Death), EXPIRE_FLAG
                        Sample Data:
                        {patients_sample}
                        
                        Table: diagnoses_icd
                        Columns: ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
                        Sample Data:
                        {diagnoses_sample}
                        """
                        
                        prompt = f"""
                        You are a SQL expert. Convert the following natural language question into a DuckDB SQL query.
                        
                        Schema Context:
                        {schema_context}
                        
                        Question: {user_query}
                        
                        Rules:
                        1. Return ONLY the SQL query. No markdown, no explanations.
                        2. Use standard SQL syntax compatible with DuckDB.
                        3. For text matching, use ILIKE for case-insensitivity (e.g., GENDER ILIKE 'F').
                        4. Dates are in 'YYYY-MM-DD' format. Cast strings to DATE if needed (e.g. CAST('2020-01-01' AS DATE)).
                        5. If the question cannot be answered with the schema, return "SELECT 'Cannot answer' as error".
                        """
                        
                        response = sql_coder.invoke(prompt)
                        generated_sql = response.content.replace("```sql", "").replace("```", "").strip()
                        
                        with st.expander("View Generated SQL", expanded=True):
                            st.code(generated_sql, language="sql")
                        
                        # Execute
                        df_result = st.session_state.mimic_db.execute(generated_sql).fetchdf()
                        st.dataframe(df_result)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

# ----------------------------------------------------
# TAB 6 — SYSTEM INFO
# ----------------------------------------------------
with tab5:
    st.header("System Information")

    st.markdown("""
### About the AI Clinical Trials Architect

This system uses:
- DeepSeek LLMs (`deepseek-chat`, `deepseek-reasoner`)
- HuggingFace embeddings (MiniLM)
- FAISS vector store
- LangChain RAG pipeline  
- Streamlit UI framework
""")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align:center; color:#444; padding:10px'>
AI Clinical Trials Architect • Powered by DeepSeek & LangChain
</div>
""", unsafe_allow_html=True)
