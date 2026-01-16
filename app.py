import os
import io
import json
import re
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from PIL import Image
import PyPDF2 as pdf
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key missing. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)
st.set_page_config(page_title="AI Invoice Analyst", layout="wide", page_icon="🧾")

# Model priority list
MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemma-3-27b-it"
]

SAMPLE_DIR = "samples"
SAMPLE_CSV = os.path.join(SAMPLE_DIR, "sample_data.csv")
SAMPLE_IMG = os.path.join(SAMPLE_DIR, "sample_invoice.jpg")

# --- Utilities ---

def init_samples():
    if not os.path.exists(SAMPLE_DIR): os.makedirs(SAMPLE_DIR)
    if not os.path.exists(SAMPLE_CSV):
        data = []
        for i in range(1, 21):
            data.append({
                "Invoice_ID": f"INV-{1000+i}",
                "Date": (datetime(2025, 1, 1) + timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d"),
                "Vendor": random.choice(["Amazon", "Google", "Microsoft", "Uber"]),
                "Amount": round(random.uniform(50, 2000), 2),
                "Status": random.choice(["Paid", "Pending"])
            })
        pd.DataFrame(data).to_csv(SAMPLE_CSV, index=False)

def get_file_bytes(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f: return io.BytesIO(f.read())

def call_llm(role, content, prompt):
    # Handle list (images) vs string (text)
    payload = [f"Role: {role}\nTask: {prompt}"] + content if isinstance(content, list) else [f"Role: {role}\nTask: {prompt}", content]
    
    for model in MODELS:
        try:
            llm = genai.GenerativeModel(model)
            response = llm.generate_content(payload)
            return response.text
        except:
            continue
    return "Error: Service busy."

def clean_json(text):
    try:
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        start, end = text.find("{"), text.rfind("}") + 1
        return text[start:end] if start != -1 else text
    except: return text

def clean_python_code(text):
    text = re.sub(r"```python\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

def load_file(file):
    ext = file.name.split('.')[-1].lower()
    
    if ext in ['jpg', 'jpeg', 'png']:
        return [{"mime_type": file.type, "data": file.getvalue()}], "image"
    elif ext == 'pdf':
        try:
            reader = pdf.PdfReader(file)
            return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()]), "pdf"
        except: return None, "error"
    elif ext in ['csv', 'xlsx']:
        try:
            df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
            return df, "dataframe"
        except: return None, "error"
    return None, "unknown"

# --- Main Interface ---

init_samples()

st.title("🧾 AI Invoice Analyst")

with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader("Drop file here", type=["jpg", "png", "pdf", "csv", "xlsx"])
    
    st.divider()
    st.subheader("Test")
    c1, c2 = st.columns(2)
    
    if c1.button("Sample Image"): st.session_state.demo = 'image'
    if c2.button("Sample CSV"): st.session_state.demo = 'csv'
    
    if st.button("Clear", type="primary"): 
        st.session_state.pop('demo', None)
        st.rerun()

# File Loading Logic
file_obj = uploaded_file
if not file_obj:
    if st.session_state.get('demo') == 'image' and os.path.exists(SAMPLE_IMG):
        file_obj = get_file_bytes(SAMPLE_IMG)
        file_obj.name = "sample.jpg"
        file_obj.type = "image/jpeg"
    elif st.session_state.get('demo') == 'csv' and os.path.exists(SAMPLE_CSV):
        file_obj = get_file_bytes(SAMPLE_CSV)
        file_obj.name = "sample.csv"
        file_obj.type = "text/csv"

content = None
ftype = None

if file_obj:
    content, ftype = load_file(file_obj)

# ==========================================
# 1. Image / PDF Extraction
# ==========================================
if ftype in ["image", "pdf"]:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Document")
        if ftype == "image": 
            file_obj.seek(0)
            st.image(Image.open(file_obj), use_container_width=True)
        else: 
            st.text_area("Text Content", content, height=500)

    with col2:
        st.subheader("Analysis")
        b1, b2 = st.columns(2)
        summ_btn = b1.button("📝 Summarize", use_container_width=True)
        ext_btn = b2.button("📊 Extract Data", use_container_width=True)
        
        st.divider()
        query = st.text_input("Ask a question:", placeholder="e.g. Verify the tax calculation")
        ask_btn = st.button("Run Query", type="primary")

        prompt = ""
        is_json = False
        
        if summ_btn:
            prompt = "Provide a summary: Vendor, Date, Total Amount, and key line items."
            
        elif ext_btn:
            is_json = True
            prompt = """
            Extract table data to pure JSON.
            
            1. Metadata: Extract Invoice_No, Date, Vendor.
            
            2. Items:
            - Identify the table header row.
            - Create a key for EVERY column visible in the table.
            - Translate headers to English (e.g. 'Rashi' -> 'Amount').
            - Do NOT group columns. Keep Tax and Amount separate if visualy separate.
            - Do NOT invent columns like 'Doc_ID'.
            - Stop extracting at the footer/total line.
            
            Format:
            {
                "Invoice_No": "val", "Date": "val", "Vendor": "val",
                "Items": [{ "Description": "A", "Qty": "1", "Rate": "100", "Amount": "100" }]
            }
            """
            
        elif ask_btn and query:
            prompt = f"Answer strictly based on the document: {query}"

        if prompt:
            with st.spinner("Processing..."):
                res = call_llm("Analyst", content, prompt)
                
                if is_json:
                    try:
                        clean_res = clean_json(res)
                        data = json.loads(clean_res)
                        
                        # Find the list of items
                        target_key = next((k for k, v in data.items() if isinstance(v, list)), None)
                        
                        if target_key:
                            df = pd.json_normalize(data[target_key])
                            
                            # Inject metadata into rows using Python
                            for k in ["Invoice_No", "Date", "Vendor"]:
                                if k in data: df.insert(0, k, data[k])
                                
                            st.dataframe(df)
                            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8-sig'), "data.csv")
                        else:
                            st.error("Structure not found.")
                            st.code(res)
                    except:
                        st.error("Parsing failed.")
                        st.code(res)
                else:
                    st.markdown(res)

# ==========================================
# 2. CSV / Data Analysis
# ==========================================
elif ftype == "dataframe":
    df = content
    
    # Auto-clean numeric columns
    for col in df.columns:
        if any(x in col.lower() for x in ['amount', 'total', 'cost', 'price', 'tax']):
            try:
                df[col] = df[col].astype(str).str.replace(r'[$,₹]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except: pass

    st.subheader("📊 Dataset Overview")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    
    amt_col = next((c for c in df.columns if any(x in c.lower() for x in ["amount", "total", "cost"])), None)
    if amt_col:
        total = df[amt_col].sum()
        c2.metric("Total Value", f"{total:,.2f}")
        c3.metric("Average", f"{total/len(df):,.2f}")
    
    st.dataframe(df, use_container_width=True)
    
    st.divider()
    st.subheader("💬 Smart Analysis")
    
    user_q = st.text_input("Ask a question about the data:", placeholder="e.g. Total spend by Vendor?")
    
    if st.button("Calculate", type="primary"):
        if user_q:
            with st.spinner("Calculating..."):
                prompt = f"""
                You are a Python Expert.
                DataFrame `df` columns: {list(df.columns)}.
                
                Write a single line of Python code to answer: "{user_q}".
                - Assume `df` is pre-loaded.
                - Return ONLY the code string.
                """
                
                code_res = call_llm("Programmer", "None", prompt)
                clean_code = clean_python_code(code_res)
                
                try:
                    local_vars = {"df": df}
                    result = eval(clean_code, {}, local_vars)
                    st.success(f"**Answer:** {result}")
                except Exception as e:
                    st.error("Could not calculate.")

elif not file_obj:
    st.info("Please upload a file to start.")