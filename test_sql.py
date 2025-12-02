import duckdb

# Setup Mock DB
conn = duckdb.connect(":memory:")
conn.execute("CREATE TABLE patients (SUBJECT_ID INTEGER, GENDER VARCHAR, DOB DATE, DOD DATE, EXPIRE_FLAG INTEGER)")
conn.execute("INSERT INTO patients VALUES (1, 'F', '1950-01-01', NULL, 0), (2, 'M', '1960-01-01', NULL, 0), (3, 'F', '1940-01-01', '2020-01-01', 1)")

# Simulate User Query
user_query = "Show me all female patients"

# Simulate LLM Response (Mock)
generated_sql = "SELECT * FROM patients WHERE GENDER = 'F'"

print(f"User Query: {user_query}")
print(f"Generated SQL: {generated_sql}")

# Execute
try:
    df = conn.execute(generated_sql).fetchdf()
    print("\nResult DataFrame:")
    print(df)
except Exception as e:
    print(f"Error: {e}")
