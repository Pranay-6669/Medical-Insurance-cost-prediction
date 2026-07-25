import streamlit as st
import requests
import os
import sys

# Add the project root to python path to allow importing backend module as fallback
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.predict import engine

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #2563EB;
        font-weight: 600;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 2.5rem;
        color: #10B981;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🏥 Medical Insurance Cost Prediction</h1>", unsafe_allow_html=True)
st.write("Predict your annual medical insurance charges using state-of-the-art Machine Learning.")

# Setup sidebar for demographics
st.sidebar.header("👤 1. Demographics")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=35, step=1)
sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Other"])
region = st.sidebar.selectbox("Region", ["North", "Central", "West", "South", "East"])
urban_rural = st.sidebar.selectbox("Urban / Rural", ["Suburban", "Urban", "Rural"])
income = st.sidebar.number_input("Annual Income ($)", min_value=0.0, value=45000.0, step=1000.0)
education = st.sidebar.selectbox("Education Level", ["HS", "No HS", "Some College", "Bachelors", "Masters", "Doctorate"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Self-employed", "Retired", "Unemployed"])
household_size = st.sidebar.number_input("Household Size", min_value=1, max_value=20, value=3, step=1)
dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)
smoker = st.sidebar.selectbox("Smoking Status", ["Never", "Former", "Current"])
alcohol_freq = st.sidebar.selectbox("Alcohol Consumption Frequency", ["None", "Occasional", "Weekly", "Daily"])

# Main layout tabs
tab1, tab2, tab3 = st.tabs(["🩺 Health Metrics", "📜 Policy Details", "📊 Prediction"])

with tab1:
    st.markdown("<h3 class='subheader'>Vitals & Lab Results</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70.0, max_value=220.0, value=120.0, step=1.0)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40.0, max_value=130.0, value=80.0, step=1.0)

    with col2:
        ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=30.0, max_value=300.0, value=100.0, step=1.0)
        hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.4, step=0.1)
        risk_score = st.slider("Health Risk Score", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

    with col3:
        visits_last_year = st.number_input("Doctor Visits (Last Year)", min_value=0, value=1, step=1)
        hospitalizations_last_3yrs = st.number_input("Hospitalizations (Last 3 Years)", min_value=0, value=0, step=1)
        days_hospitalized_last_3yrs = st.number_input("Total Days Hospitalized (Last 3 Years)", min_value=0, value=0, step=1)

    st.markdown("<h3 class='subheader'>Chronic Medical Conditions</h3>", unsafe_allow_html=True)
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        hypertension = st.checkbox("Hypertension (High BP)")
        diabetes = st.checkbox("Diabetes")
        asthma = st.checkbox("Asthma")
    with col_c2:
        copd = st.checkbox("COPD")
        cardiovascular_disease = st.checkbox("Cardiovascular Disease")
        cancer_history = st.checkbox("Cancer History")
    with col_c3:
        kidney_disease = st.checkbox("Kidney Disease")
        liver_disease = st.checkbox("Liver Disease")
        arthritis = st.checkbox("Arthritis")
    with col_c4:
        mental_health = st.checkbox("Mental Health Condition")
        had_major_procedure = st.checkbox("Had Major Procedure")
        is_high_risk = st.checkbox("Categorized as High Risk")

    st.markdown("<h3 class='subheader'>Medical Procedures (Last Year)</h3>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        proc_imaging_count = st.number_input("Imaging Counts (X-Ray/MRI)", min_value=0, value=0, step=1)
        proc_surgery_count = st.number_input("Surgery Counts", min_value=0, value=0, step=1)
    with col_p2:
        proc_physio_count = st.number_input("Physiotherapy Sessions", min_value=0, value=0, step=1)
        proc_consult_count = st.number_input("Consultations Counts", min_value=0, value=1, step=1)
    with col_p3:
        proc_lab_count = st.number_input("Laboratory Tests Counts", min_value=0, value=1, step=1)
        medication_count = st.number_input("Regular Medications Counts", min_value=0, value=1, step=1)

with tab2:
    st.markdown("<h3 class='subheader'>Policy Configuration</h3>", unsafe_allow_html=True)
    col_pol1, col_pol2 = st.columns(2)
    
    with col_pol1:
        plan_type = st.selectbox("Plan Type", ["PPO", "POS", "HMO", "EPO"])
        network_tier = st.selectbox("Network Tier", ["Bronze", "Silver", "Gold", "Platinum"])
        deductible = st.number_input("Policy Deductible ($)", min_value=0, value=1000, step=100)
        copay = st.number_input("Policy Copay ($)", min_value=0, value=20, step=5)

    with col_pol2:
        policy_term_years = st.number_input("Policy Term (Years)", min_value=1, value=5, step=1)
        policy_changes_last_2yrs = st.number_input("Policy Changes (Last 2 Years)", min_value=0, value=0, step=1)
        provider_quality = st.slider("Provider Quality Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        monthly_premium = st.number_input("Monthly Premium ($)", min_value=0.0, value=100.0, step=5.0)
        annual_premium = monthly_premium * 12

with tab3:
    st.markdown("<h3 class='subheader'>Claims & Charges Calculation</h3>", unsafe_allow_html=True)
    
    col_cl1, col_cl2 = st.columns(2)
    with col_cl1:
        claims_count = st.number_input("Total Claims (Last Year)", min_value=0, value=1, step=1)
        avg_claim_amount = st.number_input("Average Claim Amount ($)", min_value=0.0, value=400.0, step=50.0)
    with col_cl2:
        total_claims_paid = claims_count * avg_claim_amount
        st.metric("Total Claims Paid ($)", f"{total_claims_paid:,.2f}")

    st.write("---")
    
    # Validation logic before prediction
    validation_error = None
    if systolic_bp < diastolic_bp:
        validation_error = "Systolic blood pressure cannot be lower than Diastolic blood pressure."
    elif dependents >= household_size:
        validation_error = "Number of dependents cannot exceed household size."

    if validation_error:
        st.error(validation_error)
        predict_disabled = True
    else:
        predict_disabled = False

    # Predict Button
    if st.button("🚀 Calculate Predicted Annual Cost", disabled=predict_disabled, use_container_width=True):
        # Package input dictionary
        input_data = {
            "age": int(age),
            "sex": sex,
            "region": region,
            "urban_rural": urban_rural,
            "income": float(income),
            "education": education,
            "marital_status": marital_status,
            "employment_status": employment_status,
            "household_size": int(household_size),
            "dependents": int(dependents),
            "bmi": float(bmi),
            "smoker": smoker,
            "alcohol_freq": alcohol_freq,
            "visits_last_year": int(visits_last_year),
            "hospitalizations_last_3yrs": int(hospitalizations_last_3yrs),
            "days_hospitalized_last_3yrs": int(days_hospitalized_last_3yrs),
            "medication_count": int(medication_count),
            "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
            "ldl": float(ldl),
            "hba1c": float(hba1c),
            "plan_type": plan_type,
            "network_tier": network_tier,
            "deductible": int(deductible),
            "copay": int(copay),
            "policy_term_years": int(policy_term_years),
            "policy_changes_last_2yrs": int(policy_changes_last_2yrs),
            "provider_quality": float(provider_quality),
            "risk_score": float(risk_score),
            "annual_premium": float(annual_premium),
            "monthly_premium": float(monthly_premium),
            "claims_count": int(claims_count),
            "avg_claim_amount": float(avg_claim_amount),
            "total_claims_paid": float(total_claims_paid),
            "chronic_count": int(sum([hypertension, diabetes, asthma, copd, cardiovascular_disease, cancer_history, kidney_disease, liver_disease, arthritis, mental_health])),
            "hypertension": int(hypertension),
            "diabetes": int(diabetes),
            "asthma": int(asthma),
            "copd": int(copd),
            "cardiovascular_disease": int(cardiovascular_disease),
            "cancer_history": int(cancer_history),
            "kidney_disease": int(kidney_disease),
            "liver_disease": int(liver_disease),
            "arthritis": int(arthritis),
            "mental_health": int(mental_health),
            "proc_imaging_count": int(proc_imaging_count),
            "proc_surgery_count": int(proc_surgery_count),
            "proc_physio_count": int(proc_physio_count),
            "proc_consult_count": int(proc_consult_count),
            "proc_lab_count": int(proc_lab_count),
            "is_high_risk": int(is_high_risk),
            "had_major_procedure": int(had_major_procedure)
        }
        
        with st.spinner("Analyzing patient metrics and querying modeling service..."):
            predicted_charges = None
            method_used = ""
            
            # Attempt REST API prediction
            try:
                api_url = "http://127.0.0.1:8000/predict"
                response = requests.post(api_url, json=input_data, timeout=3)
                if response.status_code == 200:
                    predicted_charges = response.json()["predicted_annual_medical_cost"]
                    method_used = "REST API (FastAPI)"
            except Exception:
                # Fallback to local prediction engine
                pass
                
            if predicted_charges is None:
                try:
                    predicted_charges = engine.predict(input_data)
                    method_used = "Direct Engine Fallback (Joblib)"
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    
            if predicted_charges is not None:
                st.balloons()
                
                # Display output cards
                st.markdown("<br>", unsafe_allow_html=True)
                col_res1, col_res2 = st.columns([2, 1])
                
                with col_res1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>Predicted Annual Medical Cost</h4>
                        <div class='metric-value'>${predicted_charges:,.2f}</div>
                        <p style='color:#6B7280; font-size:0.85rem; margin-top:0.5rem;'>
                            Calculated via {method_used}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.info(f"""
                    **Key Cost Drivers Identified:**
                    - **Age:** {age} years
                    - **Smoker Status:** {smoker}
                    - **Recent Claims:** {claims_count} claims totalling ${total_claims_paid:,.2f}
                    - **Health Risk Score:** {risk_score:.2f}
                    """)
