import os
import io
import json
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from PIL import Image
import PyPDF2 as pdf
from dotenv import load_dotenv
import google.generativeai as genai

# --- Config & Setup ---
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
SAMPLE_DIR = "samples"
SAMPLE_CSV_PATH = os.path.join(SAMPLE_DIR, "sample_data.csv")
SAMPLE_IMG_PATH = os.path.join(SAMPLE_DIR, "sample_invoice.jpg")

# Configuring Gemini
if not API_KEY:
    st.error("Missing Google API Key. Please set it in your .env file.")
    st.stop()

genai.configure(api_key=API_KEY)
st.set_page_config(page_title="Invoice Extractor", layout="wide", page_icon="🧾")


def ensure_sample_files():
    """Generates dummy files for testing if they don't exist."""
    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)

    # Generate dummy CSV if missing
    if not os.path.exists(SAMPLE_CSV_PATH):
        data = []
        vendors = ["Amazon", "Google", "Microsoft", "Apple", "Uber", "Slack", "Notion"]
        for i in range(1, 41):
            row = {
                "Invoice_ID": f"INV-{1000+i}",
                "Date": (datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
                "Vendor": random.choice(vendors),
                "Amount": round(random.uniform(50, 5000), 2),
                "Status": random.choice(["Paid", "Pending", "Overdue"])
            }
            data.append(row)
        
        # Save with BOM for Excel compatibility (Hindi/UTF-8 support)
        pd.DataFrame(data).to_csv(SAMPLE_CSV_PATH, index=False, encoding='utf-8-sig')

def get_file_bytes(filepath):
    """Reads a local file and returns a BytesIO object."""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    
    file_obj = io.BytesIO(file_bytes)
    file_obj.name = os.path.basename(filepath)
    return file_obj

def query_gemini(prompt, content, user_query=None):
    """Wraps the API call to Gemini."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Content can be a list (image parts) or string (text)
        payload = [prompt, content[0] if isinstance(content, list) else content]
        if user_query:
            payload.append(user_query)
            
        response = model.generate_content(payload)
        return response.text
    except Exception as e:
        return f"API Error: {str(e)}"

def parse_file(uploaded_file):
    """
    Parses the uploaded file based on type.
    Returns: (processed_data, file_type_label)
    """
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    # 1. Images
    if file_ext in ['jpg', 'jpeg', 'png']:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts, "image"

    # 2. PDFs
    elif file_ext == 'pdf':
        try:
            reader = pdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text, "pdf"
        except Exception:
            return None, "error"

    # 3. Spreadsheets
    elif file_ext in ['csv', 'xlsx', 'xls']:
        try:
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df.to_string(), "spreadsheet"
        except Exception:
            return None, "error"
            
    return None, "unknown"

def flatten_json_data(json_data):
    """Flattens nested JSON (like 'Items' lists) into a flat DataFrame."""
    try:
        # Auto-detect the list key
        target_key = None
        for key in ["Items", "Line_Items", "products", "items", "entries"]:
            if key in json_data and isinstance(json_data[key], list):
                target_key = key
                break
        
        if target_key:
            # Keep top-level keys as metadata for each row
            meta_cols = [k for k in json_data.keys() if k != target_key]
            return pd.json_normalize(json_data, record_path=[target_key], meta=meta_cols)
        
        # Already flat
        return pd.DataFrame([json_data])
    except Exception:
        # Fallback
        return pd.DataFrame([json_data])

# Main App Logic---

# Init samples
ensure_sample_files()
has_sample_img = os.path.exists(SAMPLE_IMG_PATH)

# Sidebar
with st.sidebar:
    st.header("📂 Input")
    uploaded_file = st.file_uploader("Upload Invoice", type=["jpg", "png", "pdf", "csv", "xlsx"])
    
    st.markdown("---")
    st.subheader("🧪 Test Data")
    st.caption("No file? Try these:")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📄 PDF/Img", disabled=not has_sample_img):
            st.session_state.active_sample = 'image'
            
    with c2:
        if st.button("📊 CSV"):
            st.session_state.active_sample = 'csv'

    if st.button("Clear Selection", type="secondary"):
        st.session_state.pop('active_sample', None)
        st.rerun()

# Determine Source
active_file = None
source_mode = "upload" # or "sample"

if uploaded_file:
    active_file = uploaded_file
elif st.session_state.get('active_sample') == 'image' and has_sample_img:
    active_file = get_file_bytes(SAMPLE_IMG_PATH)
    active_file.type = "image/jpeg" # Mock mime
    source_mode = "sample"
elif st.session_state.get('active_sample') == 'csv':
    active_file = get_file_bytes(SAMPLE_CSV_PATH)
    active_file.type = "text/csv"
    source_mode = "sample"

# UI
st.title("🧾AI Invoice Analyst")
st.caption("Analyze, Summarize, Extract using this smart Extractor")

col_left, col_right = st.columns([1, 1.2])
processed_data = None

# Left: Preview
with col_left:
    st.subheader("Preview")
    
    if active_file:
        if source_mode == "sample":
            st.info(f"Using sample: `{active_file.name}`")
            
        processed_data, ftype = parse_file(active_file)
        
        if ftype == "image":
            active_file.seek(0)
            st.image(Image.open(active_file), caption="Source Image", use_container_width=True)
        elif ftype == "pdf":
            st.text_area("Extracted Text", processed_data, height=400)
        elif ftype == "spreadsheet":
            active_file.seek(0)
            df = pd.read_csv(active_file) if active_file.name.endswith('.csv') else pd.read_excel(active_file)
            st.dataframe(df, height=400)
        elif ftype == "error":
            st.error("Could not read file.")
    else:
        st.info("👈 Upload a file or pick a sample to start.")

# Right: Actions
with col_right:
    st.subheader("Analysis")
    
    if processed_data:
        # Quick Actions
        st.markdown("##### Quick Actions")
        c1, c2 = st.columns(2)
        
        do_summarize = c1.button("📝 Summarize", use_container_width=True)
        do_extract = c2.button("📊 Extract Data", use_container_width=True)
        
        st.divider()
        
        # Custom Query
        st.markdown("##### Custom Query")
        user_query = st.text_input("Ask something specific...", placeholder="e.g. What is the total tax?")
        do_ask = st.button("🚀 Run Query", type="primary")

        # Logic
        prompt = ""
        is_extract_task = False
        
        if do_summarize:
            prompt = "Summarize this document. Identify the vendor, date, and key totals."
        elif do_extract:
            is_extract_task = True
            prompt = """
            Extract invoice data into valid JSON. Structure:
            {
                "Invoice_No": "string",
                "Date": "YYYY-MM-DD",
                "Vendor": "string",
                "Total_Amount": float,
                "Currency": "string",
                "Items": [
                    {"Description": "string", "Qty": int, "Rate": float, "Amount": float}
                ]
            }
            Ensure dates are standardized. If a field is missing, use null.
            """
        elif do_ask and user_query:
            prompt = user_query
            
        # Execution
        if prompt:
            with st.spinner("Processing..."):
                response_text = query_gemini("You are an expert financial analyst.", processed_data, prompt)
                
                if is_extract_task:
                    # Handle JSON Extraction
                    try:
                        # Strip markdown if present
                        clean_json = response_text.replace("```json", "").replace("```", "").strip()
                        data_dict = json.loads(clean_json)
                        
                        # Flatten for display
                        df_flat = flatten_json_data(data_dict)
                        
                        st.success("Extraction Complete")
                        st.dataframe(df_flat)
                        
                        # CSV Download
                        csv_data = df_flat.to_csv(index=False).encode('utf-8-sig')
                        st.download_button("📥 Download CSV", csv_data, "invoice_data.csv", "text/csv")
                        
                    except json.JSONDecodeError:
                        st.error("Failed to parse JSON response. Raw output below:")
                        st.code(response_text)
                    except Exception as e:
                        st.error(f"Error processing data: {e}")
                else:
                    # Standard Text Response
                    st.markdown("### Result")
                    st.write(response_text)