import sys
import os

# CRITICAL FIX FOR STREAMLIT CLOUD:
# Add the project root to sys.path so 'import src' works correctly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json
from dotenv import load_dotenv

# Load environment variables (for API keys if you turn off mock mode)
load_dotenv()

from src.graph import build_graph

# --- Configuration ---
PATIENTS_FILE = os.path.join(os.path.dirname(__file__), "data", "sample_patients.json")

def load_patients():
    with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="Clinical AI Pipeline", layout="wide", page_icon="🏥")

st.title("🏥 Clinical AI Decision Support System")
st.caption("⚠️ Demo project — synthetic data and a toy ML model. Not for real clinical use.")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# Load patients safely
try:
    patients = load_patients()
    patient_options = {f"{p['name']} ({p['patient_id']})": p for p in patients}
except Exception as e:
    st.sidebar.error(f"Failed to load patients: {e}")
    patient_options = {}

# Only show selectbox if we have patients
if patient_options:
    selected_label = st.sidebar.selectbox("Select Patient", list(patient_options.keys()))
    selected_patient = patient_options[selected_label]
    
    mock_mode = st.sidebar.checkbox("Mock Mode (No API calls)", value=True)

    # --- Run Pipeline ---
    if st.sidebar.button("▶️ Run Pipeline"):
        # 1. Build the graph
        with st.spinner("Initializing pipeline graph..."):
            try:
                app = build_graph()
            except Exception as e:
                st.error(f"Failed to build graph: {e}")
                st.stop()

        # 2. Run the pipeline
        with st.spinner(f"Running pipeline for {selected_patient['name']}... (This may take a minute)"):
            try:
                initial_state = {
                    "patient_raw": selected_patient,
                    "mock_mode": mock_mode,
                    "trace": [],
                    "errors": [],
                }
                final_state = app.invoke(initial_state)
                
                st.session_state["report"] = final_state.get("final_report", "No report generated.")
                st.session_state["trace"] = final_state.get("trace", [])
                st.success("Pipeline completed successfully!")
                
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # --- Display Results ---
    if "report" in st.session_state:
        st.markdown("### 📄 Generated Report")
        st.markdown(st.session_state["report"])
        
        with st.expander(" View System Trace / Logs"):
            for log in st.session_state.get("trace", []):
                st.text(log)
else:
    st.error("No patients available. Please check your data/sample_patients.json file.")
