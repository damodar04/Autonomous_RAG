🚀 AI Clinical Trials Architect
Autonomous Multi-Agent RAG System for Protocol Design & Feasibility Analysis

⭐ Overview

The AI Clinical Trials Architect is an end-to-end autonomous system that designs, evaluates, and optimizes clinical trial protocols using:

Autonomous RAG (Retrieval-Augmented Generation)

Multi-Agent collaboration

Real-world patient data (MIMIC-III)

DeepSeek LLMs

FAISS vector search

DuckDB SQL analytics

This system behaves like a virtual clinical research team:
Medical researcher, FDA specialist, ethics officer, cohort analyst, protocol writer, and even a director-level evaluator — all working together.

Also, it's polite. Usually.
(Unless you ask it to violate FDA guidelines — then it becomes a responsible adult.)

🎯 Why This Project Exists

Protocol development is traditionally:

❌ Slow
❌ Manual
❌ Fragmented across teams
❌ Prone to regulatory and ethical mistakes

Autonomous RAG makes everything:

✅ Faster
✅ Evidence-backed
✅ More accurate
✅ Self-improving

🧬 Key Features
🔹 1. Multi-Agent Architecture

Planner

Medical Research Retriever

Regulatory (FDA) Specialist

Ethics Specialist

SQL Cohort Analyst

Protocol Synthesizer

Director Reasoner (Evaluation)

🔹 2. RAG-Powered Knowledge Integration

PubMed literature

FDA guidelines (PDF + TXT)

Ethics / Belmont Report

Structured patient data (MIMIC)

🔹 3. Autonomous Evolution Loop

Each generated protocol is auto-evaluated across:

Scientific Rigor

Regulatory Compliance

Ethical Soundness

Feasibility

Patient Centricity

System then improves itself in the next iteration.

🔹 4. Clinical Trial Designer UI

Users can specify:

Drug Name

Dose (mg)

Frequency

Route

Comparator

Arms, Blinding, Randomization

Primary/Secondary Endpoints

Safety parameters

🔹 5. MIMIC-III Powered Feasibility

Real patient data → recruitment feasibility → more realistic trials.

🔹 6. Full Streamlit Interface

Dark theme removed.
Now clean, medical-grade white UI.

🧠 Architecture Diagram

Or ASCII view:

User
  ↓
Planner Agent
  ↓
──────────────────────────────────────────────
|  PubMed Retriever   → Evidence              |
|  FDA Retriever      → Compliance            |
|  Ethics Retriever   → Ethics                |
|  SQL Cohort Agent   → Real patient data     |
──────────────────────────────────────────────
                  ↓
          Synthesizer Agent
                  ↓
        ┌──────────────────────┐
        │  Protocol Draft      │
        └──────────────────────┘
                  ↓
          Evaluation Engine
                  ↓
         Diagnostic + SOP Fixer
                  ↓
   Self-Improved Protocol Next Round

⚙️ Technical Stack
Component	Tech
LLMs	DeepSeek Chat, DeepSeek Reasoner
Framework	LangChain + Multi-Agent LangGraph (optional)
Embeddings	HuggingFace MiniLM / TF-IDF fallback
Vector DB	FAISS
Backend DB	DuckDB (for MIMIC)
Frontend	Streamlit
Visualizations	Plotly
Knowledge Sources	PubMed, FDA, Ethics, MIMIC
📁 Project Structure
/project-root
│
├── clinical_trials_app.py     # Main Streamlit app
├── data/
│   ├── pubmed_articles/
│   ├── fda_guidelines/
│   ├── ethical_guidelines/
│   └── mimic_db/
│
├── embeddings/                # Vector store cache
├── docs/                      # Procedure docs, SOP
└── README.md                  # You are reading me

🛠️ Installation & Setup
1. Clone repo
git clone https://github.com/yourusername/clinical-trials-architect.git
cd clinical-trials-architect

2. Install dependencies
pip install -r requirements.txt

3. Add .env
DEEPSEEK_API_KEY=your_key_here

4. Run the app
streamlit run clinical_trials_app.py

🚀 How It Works
Step 1 — Initialize System

Loads models (DeepSeek Chat & Reasoner) + embeddings.

Step 2 — Load Knowledge Base

Indexes all PubMed, FDA, Ethics documents into FAISS.

Step 3 — Load MIMIC

Loads patient demographics + ICD diagnoses.

Step 4 — Design a Trial

Fill form → drug name → dose → endpoints → generate protocol.

Step 5 — Evaluation

System auto-grades your protocol and visualizes quality.

Step 6 — Feasibility Check

Filter real patients based on inclusion/exclusion.

😂 Why Autonomous RAG is Like a Big Brain Team

Think of it like the Avengers:

PubMed Agent = Doctor Strange (sees all knowledge)

FDA Agent = Captain America (follows rules)

Ethics Agent = Vision (moral compass)

SQL Agent = Iron Man (data & analytics)

Synthesizer = Nick Fury (brings it all together)

Director Agent = The One Above All

Together, they save clinical trials from becoming disasters.

🧪 Technical Deep Dive for Engineers
Vector Pipeline

Text → Chunk → Embedding → FAISS Index → Retriever

Autonomous Loop

LLM → Draft → Evaluate → Reflection → SOP Mutation → Re-run

Cohort SQL Generation

Natural language → SQL → DuckDB → DataFrame → Insights

LLM Routing

Planner → Domain Agent → Synthesizer → Evaluator → Director.

Optimizations

TF-IDF fallback for offline RAG

Chunk overlap tuning (100–150 chars)

Temperature controls for deterministic outputs

Error-handling for PDF fallbacks

🧭 Roadmap

 Dose recommendation engine

 Automatic sample size estimation

 Multi-country site feasibility

 Auto-generate CONSORT-compliant diagrams

 Real-time clinical trial monitoring

 Multi-agent LangGraph rewrite

🤝 Contributing

Pull requests welcome!
Especially if you’re from pharma, clinical ops, or AI engineering.

⭐ Like this project?

Give it a ⭐ on GitHub — it helps a lot.

📬 Need Help?

Open an Issue or email damodar.7974@gmail.com
