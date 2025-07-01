# streamlit_app.py

import os
import re
import sqlite3

import streamlit as st
import pandas as pd
import google.generativeai as genai    # <-- correct import

# Configure your Gemini API key
api_key = st.secrets["gemini_api"]
genai.configure(api_key=api_key)

# 2) Define our system prompt
SQL_SYSTEM_PROMPT = """
You are an expert SQL assistant for a property management system. Here is the database schema:

-- tenants(id, timestamp, first_name, last_name, email, phone, date_of_birth, created_at)
-- properties(id, timestamp, name, address_line1, address_line2, city, state, postal_code, country, created_at)
-- units(id, timestamp, property_id, unit_number, floor, bedrooms, bathrooms, square_feet, status, created_at)
-- leases(id, timestamp, tenant_id, unit_id, start_date, end_date, rent_amount, security_deposit, status, created_at)
-- agents(id, timestamp, first_name, last_name, role, email, phone, created_at)
-- service_tickets(id, timestamp, lease_id, raised_by, assigned_to, category, subcategory, description, status, priority, created_at, updated_at)
-- ticket_comments(id, timestamp, ticket_id, author_id, author_type, comment_text, created_at)
-- ticket_conversations(id, timestamp, ticket_id, author_type, author_id, message_text, sent_at)
-- payments(id, timestamp, lease_id, payment_type, billing_period, due_date, amount, method, paid_on, reference_number, created_at)

Always generate valid SQLite SQL using the correct table and column names.
Respond only with the raw SQL—do NOT include markdown fences or annotations.
"""

# 3) Create our chat‑based SQL generator
sql_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction=SQL_SYSTEM_PROMPT
)

def nl_to_sql(question: str) -> str:
    """Convert a natural‑language question into raw SQL."""
    chat = sql_model.start_chat()
    resp = chat.send_message(question)
    return resp.text

def clean_sql(raw: str) -> str:
    """Strip out any ``` fences."""
    sql = re.sub(r"```[^\n]*\n", "", raw)
    sql = re.sub(r"\n```", "", sql)
    return sql.strip()

# 4) Streamlit UI
st.title("🏠 Property Management SQL Chatbot")

st.markdown(
    "Upload your `database.db` (SQLite) or use the default. "
    "Then ask any question about tenants, leases, properties, payments, or tickets."
)

# Database uploader
db_file = st.sidebar.file_uploader("Upload SQLite DB", type=["db","sqlite"])
if db_file:
    db_path = "/tmp/uploaded.db"
    with open(db_path, "wb") as f:
        f.write(db_file.getbuffer())
else:
    db_path = "database.db"  # must exist in repo

question = st.text_input("Ask a question about your property data:")

if st.button("Run Query") and question:
    # 1) Generate SQL
    with st.spinner("Generating SQL…"):
        raw_sql = nl_to_sql(question)
        sql = clean_sql(raw_sql)

    st.subheader("Generated SQL")
    st.code(sql, language="sql")

    # 2) Execute it
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()

        if df.empty:
            st.warning("No results returned.")
        else:
            st.subheader("Results")
            st.dataframe(df)

            # 3) Summarize in English
            summary_prompt = f"""
Here are the results of your query:

SQL:
{sql}

Results:
{df.to_dict(orient='records')}

Please summarize these results in plain English.
"""
            with st.spinner("Generating summary…"):
                ans_chat = sql_model.start_chat()
                ans_resp = ans_chat.send_message(summary_prompt)

            st.subheader("Summary")
            st.write(ans_resp.text)

    except Exception as e:
        st.error(f"Error executing SQL: {e}")
