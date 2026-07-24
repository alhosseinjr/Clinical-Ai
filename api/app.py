import streamlit as st
import requests
import json

# --- CONFIGURATION ---
# We will replace this with your Render URL in Step 4.
# For now, leave it as localhost to test locally if you want.
API_URL = "http://localhost:8000" 

st.set_page_config(page_title="Clinical AI Pipeline", layout="wide", page_icon="")

st.title("🏥 Clinical AI Decision Support System")
st.markdown("A multi-agent pipeline for clinical risk assessment, guideline verification, and reasoning.")
st.caption("⚠️ Demo project — synthetic data and a toy ML model. Not for real clinical use.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")

# Fetch available patients from the API
try:
    patients_response = requests.get(f"{API_URL}/api/patients", timeout=5)
    patients = patients_response.json()
    patient_options = {f"{p['name']} ({p['patient_id']})": p['patient_id'] for p in patients}
except Exception as e:
    st.sidebar.error(f"Cannot connect to API: {e}")
    patient_options = {"Error": "Error"}

selected_label = st.sidebar.selectbox("Select Patient", list(patient_options.keys()))
patient_id = patient_options[selected_label]

mock_mode = st.sidebar.checkbox("Mock Mode (No API calls)", value=True)

# --- RUN PIPELINE ---
if st.sidebar.button(" Run Pipeline"):
    with st.spinner("Running clinical pipeline... (This may take a minute on free tier)"):
        try:
            # Call the FastAPI Backend
            response = requests.post(
                f"{API_URL}/api/run",
                json={"patient_id": patient_id, "mock": mock_mode},
                timeout=120 # Longer timeout for free-tier cold starts
            )
            response.raise_for_status()
            data = response.json()
            
            # Store in session state
            st.session_state["report"] = data.get("final_report", "No report generated.")
            st.session_state["trace"] = data.get("trace", [])
            st.session_state["success"] = True
            
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")
            st.session_state["success"] = False

# --- DISPLAY RESULTS ---
if st.session_state.get("success"):
    st.markdown("### 📄 Generated Report")
    st.markdown(st.session_state["report"])
    
    with st.expander("🔍 View System Trace / Logs"):
        for log in st.session_state.get("trace", []):
            st.text(log)
