import duckdb
import os
import pandas as pd

# Mock Streamlit session state
class MockSessionState:
    def __init__(self):
        self.mimic_db = None

st_session_state = MockSessionState()

mimic_path = "./data/mimic_db"
patients_file = os.path.join(mimic_path, "PATIENTS.csv")
diagnoses_file = os.path.join(mimic_path, "DIAGNOSES_ICD.csv")

print("Initializing DB...")
conn = duckdb.connect(":memory:")
if os.path.exists(patients_file):
    conn.execute(f"CREATE TABLE patients AS SELECT * FROM read_csv_auto('{patients_file}')")
if os.path.exists(diagnoses_file):
    conn.execute(f"CREATE TABLE diagnoses_icd AS SELECT * FROM read_csv_auto('{diagnoses_file}')")

st_session_state.mimic_db = conn

print("\n--- Testing Schema Context Generation ---")
try:
    patients_sample = st_session_state.mimic_db.execute("SELECT * FROM patients LIMIT 3").fetchdf().to_string(index=False)
    diagnoses_sample = st_session_state.mimic_db.execute("SELECT * FROM diagnoses_icd LIMIT 3").fetchdf().to_string(index=False)
except Exception as e:
    print(f"Error fetching samples: {e}")
    patients_sample = "Error"
    diagnoses_sample = "Error"

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

print("Generated Schema Context:")
print(schema_context)

print("\n--- Testing Query Execution (Simulated) ---")
test_sql = "SELECT * FROM patients WHERE GENDER = 'F' LIMIT 2"
try:
    df_result = st_session_state.mimic_db.execute(test_sql).fetchdf()
    print("Query executed successfully.")
    print(df_result)
except Exception as e:
    print(f"Query execution failed: {e}")
