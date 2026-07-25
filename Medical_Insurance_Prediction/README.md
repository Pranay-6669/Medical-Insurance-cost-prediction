# 🏥 Medical Insurance Cost Prediction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-ready Machine Learning system that predicts **Annual Medical Insurance Costs** based on user demographics, health statistics, historical medical procedures, and policy parameters.

---

## 📌 Problem Statement

Predicting medical insurance charges is essential for healthcare providers and insurance companies to optimize pricing models and manage risk. Conversely, individuals benefit from knowing their projected medical expenses. 

This project implements a complete, enterprise-grade ML pipeline from raw data preprocessing and Exploratory Data Analysis (EDA) to training 11 regressor models, tuning the best regressor, exposing a REST API backend via FastAPI, and building a responsive web dashboard via Streamlit.

---

## 🗃️ Dataset & Features

The dataset (`medical_insurance.csv`) consists of **100,000 records** and **54 columns** comprising patient attributes:

*   **Demographics:** Age, Sex (Male/Female/Other), Region (North/Central/West/South/East), Urban/Rural, Income, Education Level, Marital Status, Employment Status, Household Size, Dependents.
*   **Vitals & Lab Values:** BMI, Blood Pressure (Systolic & Diastolic), LDL Cholesterol, HbA1c (diabetes marker).
*   **Risk Factors:** Smoking Status (Never/Former/Current), Alcohol Frequency, Chronic Counts, and flags for conditions (Hypertension, Diabetes, Asthma, COPD, Cardiovascular Disease, Cancer History, Kidney Disease, Liver Disease, Arthritis, Mental Health).
*   **Historical Procedures & Stays:** Visits last year, Hospitalizations last 3 years, Days hospitalized last 3 years, Imaging count, Surgery count, Physiotherapy sessions, Lab count, Major procedures flags, Medication count.
*   **Policy Details:** Plan Type (PPO/POS/HMO/EPO), Network Tier (Bronze/Silver/Gold/Platinum), Deductible, Copay, Policy Term, Policy changes, Provider Quality, Monthly Premium, Annual Premium, Claims Count, Average Claim Amount, Total Claims Paid.
*   **Target Variable:** `annual_medical_cost` (charges).

---

## ⚙️ Technologies Used

*   **Language:** Python 3.11
*   **ML Pipeline:** Scikit-Learn, Pandas, NumPy, Joblib
*   **Gradient Boosting Frameworks:** XGBoost, CatBoost
*   **Visualizations:** Matplotlib, Seaborn
*   **Backend:** FastAPI, Uvicorn, Pydantic
*   **Frontend Dashboard:** Streamlit
*   **Notebook Environment:** Jupyter, Nbformat, Nbconvert

---

## 🔄 Machine Learning Workflow

```
[medical_insurance.csv] -> [Data Preprocessing] -> [Outlier Capping (IQR)] -> [Encoding & Scaling]
                                                                                      |
[FastAPI REST API] <-------- [Model Serialization (joblib)] <------- [Best Tuned Model (XGBoost)]
         |
[Streamlit Frontend] -> Display Predictions & Insights
```

1.  **Data Preprocessing**:
    *   Missing categorical values in `alcohol_freq` filled with `'None'` (its logical equivalent in the dataset).
    *   Capped continuous outlier attributes using **1.5 * IQR (Interquartile Range)** limits on the training set to prevent model distortion while preserving patient records.
2.  **Feature Engineering**:
    *   **Ordinal Encoding**: Custom integer mapping applied to ordered categorical variables: `education`, `smoker`, `alcohol_freq`, and `network_tier`.
    *   **Nominal Encoding**: One-Hot Encoding applied to `sex`, `region`, `urban_rural`, `marital_status`, `employment_status`, and `plan_type`.
    *   **Scaling**: Comparison between `StandardScaler` and `MinMaxScaler` using a Ridge Regression baseline showed `StandardScaler` yielded a slightly better generalization score on the test set.
3.  **Model Selection & Tuning**:
    *   Trained **11 regressor algorithms** ranging from linear models to advanced ensembles.
    *   Identified **XGBoost Regressor** as the optimal model (R² ~ 0.9999).
    *   Tuned hyperparameters via `RandomizedSearchCV` to achieve stable generalization.

---

## 📊 Model Comparison

Below is the comparison of baseline models evaluated on the test dataset:

| Model | MAE ($) | MSE | RMSE ($) | R² Score | Adjusted R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **6.52** | **175.21** | **13.23** | **0.999982** | **0.999982** |
| **Gradient Boosting** | 8.11 | 240.50 | 15.50 | 0.999975 | 0.999975 |
| **CatBoost Regressor** | 10.22 | 312.40 | 17.67 | 0.999968 | 0.999968 |
| **Random Forest** | 15.65 | 785.12 | 28.02 | 0.999920 | 0.999920 |
| **ExtraTrees Regressor** | 22.34 | 1,510.45 | 38.86 | 0.999845 | 0.999845 |
| **Decision Tree** | 30.12 | 2,820.67 | 53.11 | 0.999712 | 0.999712 |
| **AdaBoost Regressor** | 512.44 | 415,200.00 | 644.36 | 0.957500 | 0.957326 |
| **Linear Regression** | 1,022.10 | 1,814,500.00 | 1,347.03 | 0.814310 | 0.813548 |
| **Ridge Regression** | 1,022.12 | 1,814,520.00 | 1,347.04 | 0.814308 | 0.813546 |
| **Lasso Regression** | 1,022.15 | 1,814,550.00 | 1,347.05 | 0.814305 | 0.813543 |
| **ElasticNet** | 1,750.36 | 5,389,000.00 | 2,321.42 | 0.448500 | 0.446227 |

*Note: Tree-based ensemble models perform extremely well because claims payout history directly correlates with the final cost.*

---

## 🚀 Installation & Running

### 1. Prerequisites
Ensure you have **Python 3.11** installed. Clone this repository and navigate to the project directory:

```bash
cd Medical_Insurance_Prediction
```

### 2. Install Dependencies
Create a virtual environment and install the required libraries:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Backend API (FastAPI)
Launch the FastAPI development server:

```bash
# Start from the project root
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
The API Swagger documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Run the Frontend (Streamlit)
Launch the Streamlit dashboard in a separate terminal:

```bash
streamlit run frontend/app.py
```
The UI dashboard will open in your default browser at [http://localhost:8501](http://localhost:8501).

---

## 🔌 API Usage

### Endpoint: `POST /predict`
Submit demographic and vitals data to get the predicted medical insurance charges.

#### Request Body Schema
```json
{
  "age": 45,
  "sex": "Female",
  "region": "North",
  "urban_rural": "Suburban",
  "income": 55000.0,
  "education": "Bachelors",
  "marital_status": "Married",
  "employment_status": "Employed",
  "household_size": 3,
  "dependents": 1,
  "bmi": 26.5,
  "smoker": "Never",
  "alcohol_freq": "Occasional",
  "visits_last_year": 2,
  "hospitalizations_last_3yrs": 0,
  "days_hospitalized_last_3yrs": 0,
  "medication_count": 1,
  "systolic_bp": 120.0,
  "diastolic_bp": 80.0,
  "ldl": 100.0,
  "hba1c": 5.5,
  "plan_type": "PPO",
  "network_tier": "Silver",
  "deductible": 1000,
  "copay": 20,
  "policy_term_years": 5,
  "policy_changes_last_2yrs": 0,
  "provider_quality": 4.0,
  "risk_score": 0.35,
  "annual_premium": 1200.0,
  "monthly_premium": 100.0,
  "claims_count": 1,
  "avg_claim_amount": 400.0,
  "total_claims_paid": 400.0,
  "chronic_count": 0,
  "hypertension": 0,
  "diabetes": 0,
  "asthma": 0,
  "copd": 0,
  "cardiovascular_disease": 0,
  "cancer_history": 0,
  "kidney_disease": 0,
  "liver_disease": 0,
  "arthritis": 0,
  "mental_health": 0,
  "proc_imaging_count": 0,
  "proc_surgery_count": 0,
  "proc_physio_count": 0,
  "proc_consult_count": 1,
  "proc_lab_count": 1,
  "is_high_risk": 0,
  "had_major_procedure": 0
}
```

#### Response Example
```json
{
  "predicted_annual_medical_cost": 4920.45,
  "currency": "USD"
}
```

---

## 📈 Future Improvements

1.  **Feature Selection**: Train models without `total_claims_paid` or `avg_claim_amount` to build a prediction system based *solely* on demographic/health vitals, representing pre-admission cost calculations.
2.  **SHAP Interpretability**: Integrate SHAP (SHapley Additive exPlanations) values to output explainable AI summaries in the frontend for each prediction.
3.  **Dockerization**: Add a `Dockerfile` for the backend and frontend to support seamless containerized deployments on platforms like AWS ECS, GCP Cloud Run, Render, or Railway.

---

## 👤 Author

**Pranay** - Senior Machine Learning Engineer  
Feel free to open an issue or pull request for enhancements!
