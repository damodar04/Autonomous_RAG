import datetime

# Simulate variables
disease = "Type 2 Diabetes"
target_population = "Adults"
drug_name = "Dapagliflozin"
drug_class = "SGLT2"
dose = 10
frequency = "Once Daily"
route = "Oral"
comparator = "Placebo"
study_type = "RCT"
phase = "Phase III"
duration = 24
randomization = "1:1"
blinding = "Double-Blind"
primary_ep = "HbA1c"
secondary_ep = "Weight"
safety_params = "AEs"

# Logic from App
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

print(f"Current Date: {current_date}")
print("Prompt constructed successfully.")
print("-" * 20)
print(prompt)
