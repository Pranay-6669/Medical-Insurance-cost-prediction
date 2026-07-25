from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from backend.predict import engine

app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    description="Production-ready API to predict medical insurance costs based on demographics, health, and policy metrics.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic request model
class InsuranceInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the person", example=45)
    sex: str = Field(..., description="Sex: Female, Male, Other", example="Female")
    region: str = Field(..., description="Region: North, Central, West, South, East", example="North")
    urban_rural: str = Field(..., description="Urban/Rural setting: Suburban, Urban, Rural", example="Suburban")
    income: float = Field(..., ge=0.0, description="Annual income in dollars", example=55000.0)
    education: str = Field(..., description="Education: Doctorate, No HS, HS, Some College, Masters, Bachelors", example="Bachelors")
    marital_status: str = Field(..., description="Marital status: Married, Single, Divorced, Widowed", example="Married")
    employment_status: str = Field(..., description="Employment: Retired, Employed, Self-employed, Unemployed", example="Employed")
    household_size: int = Field(..., ge=1, description="Size of household", example=3)
    dependents: int = Field(..., ge=0, description="Number of dependents", example=1)
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index", example=26.5)
    smoker: str = Field(..., description="Smoking status: Never, Current, Former", example="Never")
    alcohol_freq: str = Field(..., description="Alcohol consumption frequency: None, Weekly, Daily, Occasional", example="Occasional")
    visits_last_year: int = Field(..., ge=0, description="Doctor visits last year", example=2)
    hospitalizations_last_3yrs: int = Field(..., ge=0, description="Hospitalizations in last 3 years", example=0)
    days_hospitalized_last_3yrs: int = Field(..., ge=0, description="Total days hospitalized in last 3 years", example=0)
    medication_count: int = Field(..., ge=0, description="Number of regular medications", example=1)
    systolic_bp: float = Field(..., ge=70.0, le=200.0, description="Systolic Blood Pressure", example=120.0)
    diastolic_bp: float = Field(..., ge=40.0, le=130.0, description="Diastolic Blood Pressure", example=80.0)
    ldl: float = Field(..., ge=30.0, le=300.0, description="LDL Cholesterol Level", example=100.0)
    hba1c: float = Field(..., ge=3.0, le=15.0, description="HbA1c diabetes indicator", example=5.5)
    plan_type: str = Field(..., description="Plan Type: PPO, POS, HMO, EPO", example="PPO")
    network_tier: str = Field(..., description="Network Tier: Bronze, Silver, Gold, Platinum", example="Silver")
    deductible: int = Field(..., ge=0, description="Plan deductible", example=1000)
    copay: int = Field(..., ge=0, description="Copay amount", example=20)
    policy_term_years: int = Field(..., ge=1, description="Policy term in years", example=5)
    policy_changes_last_2yrs: int = Field(..., ge=0, description="Policy changes in last 2 years", example=0)
    provider_quality: float = Field(..., ge=1.0, le=5.0, description="Provider quality score", example=4.0)
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calculated health risk score", example=0.35)
    annual_premium: float = Field(..., ge=0.0, description="Annual premium in dollars", example=1200.0)
    monthly_premium: float = Field(..., ge=0.0, description="Monthly premium in dollars", example=100.0)
    claims_count: int = Field(..., ge=0, description="Claims filed last year", example=1)
    avg_claim_amount: float = Field(..., ge=0.0, description="Average claim amount", example=400.0)
    total_claims_paid: float = Field(..., ge=0.0, description="Total claim amount paid", example=400.0)
    chronic_count: int = Field(..., ge=0, description="Number of chronic conditions", example=0)
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension indicator (0 or 1)", example=0)
    diabetes: int = Field(..., ge=0, le=1, description="Diabetes indicator (0 or 1)", example=0)
    asthma: int = Field(..., ge=0, le=1, description="Asthma indicator (0 or 1)", example=0)
    copd: int = Field(..., ge=0, le=1, description="COPD indicator (0 or 1)", example=0)
    cardiovascular_disease: int = Field(..., ge=0, le=1, description="Cardiovascular disease indicator (0 or 1)", example=0)
    cancer_history: int = Field(..., ge=0, le=1, description="Cancer history indicator (0 or 1)", example=0)
    kidney_disease: int = Field(..., ge=0, le=1, description="Kidney disease indicator (0 or 1)", example=0)
    liver_disease: int = Field(..., ge=0, le=1, description="Liver disease indicator (0 or 1)", example=0)
    arthritis: int = Field(..., ge=0, le=1, description="Arthritis indicator (0 or 1)", example=0)
    mental_health: int = Field(..., ge=0, le=1, description="Mental health conditions (0 or 1)", example=0)
    proc_imaging_count: int = Field(..., ge=0, description="Imaging procedures count", example=0)
    proc_surgery_count: int = Field(..., ge=0, description="Surgeries count", example=0)
    proc_physio_count: int = Field(..., ge=0, description="Physiotherapy sessions count", example=0)
    proc_consult_count: int = Field(..., ge=0, description="Consultations count", example=1)
    proc_lab_count: int = Field(..., ge=0, description="Lab tests count", example=1)
    is_high_risk: int = Field(..., ge=0, le=1, description="High risk status (0 or 1)", example=0)
    had_major_procedure: int = Field(..., ge=0, le=1, description="Major procedure status (0 or 1)", example=0)

# Pre-load artifacts at startup
@app.on_event("startup")
def startup_event():
    try:
        engine.load_artifacts()
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Medical Insurance Cost Prediction API is running.",
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_charges(payload: InsuranceInput):
    try:
        input_data = payload.dict()
        predicted_cost = engine.predict(input_data)
        return {
            "predicted_annual_medical_cost": round(predicted_cost, 2),
            "currency": "USD"
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
