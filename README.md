# Teenage Pregnancy in Ghana – Explainable AI Methodology Pipeline

# Overview

This repository provides the complete **methodology pipeline** for predicting and understanding **teenage pregnancy in Ghana** using advanced **machine learning, explainable AI (XAI), causal inference, and fairness auditing techniques**.  

The project demonstrates how predictive models can go beyond classification tasks by integrating interpretability, counterfactual reasoning, causal insights, and bias mitigation — all aimed at informing **evidence-based public health interventions**.

---

## Pipeline Structure

The codebase is organised into sequential stages, each represented as a notebook or Python module. Together, these stages form a complete analytical workflow:

### 1. Data Collection and Preprocessing
- Load secondary demographic and health survey data (e.g., GDHS).  
- Handle missing values using **MICE (Multiple Imputation by Chained Equations)**.  
- Encode categorical features using **one-hot** or **ordinal encoding**.  

### 2. Data Cleaning and Encoding
- Normalize and transform features.  
- Ensure consistent data types.  
- Address outliers and scale numeric features where necessary.  

### 3. Predictive Base Modeling Framework
- Train multiple models:  
  - `RandomForestClassifier`  
  - `XGBoostClassifier`  
  - `LogisticRegression`  
  - `SVC`  
- Handle class imbalance using **SMOTE**.  
- Split data into **70% train / 30% test** with stratification.  

### 4. Threshold Tuning
- Explore alternative probability thresholds (`0.3`, `0.4`, `0.45`, etc.)  
- Select the optimal threshold based on **recall** and **F1-score** for the minority class.  

### 5. Evaluation Metrics
- Accuracy  
- Precision  
- Recall (Sensitivity)  
- F1-score  
- ROC-AUC Curve  
- Precision–Recall Curve  

---

## Explainability and Counterfactual Analysis

### 6. Global Explainability (SHAP)
- Compute SHAP values with `TreeExplainer`.  
- Visualise **feature importance** with bar plots and beeswarm plots.  
- Identify the most influential predictors: `Education level`, `Age`, `Residence`, `Wealth status`.

### 7. Local Explainability (Counterfactual Analysis)
- Generate counterfactual examples using **DiCE**.  
- Identify **minimal feature changes** that flip predictions.  
- Analyse global counterfactual frequencies to detect actionable variables.

---

## Causal Inference (EconML)

### 8. Causal Effect Estimation
- Use `CausalForestDML` to estimate:  
  - **Average Treatment Effect (ATE)**  
  - **Conditional Average Treatment Effect (CATE)** across Urban/Rural subgroups.  
- Quantify the causal impact of interventions like `Education`, `Permission Autonomy`, `Wealth`, and `Distance to Facility`.

---

## Fairness Analysis (AIF360)

### 9. Bias Detection and Mitigation
- Evaluate group fairness across protected attributes:  
  - `Residence`, `Wealth`, `Religion`, `Ethnicity`, `Sex of Household Head`.  
- Metrics:  
  - Statistical Parity Difference  
  - Equal Opportunity Difference  
  - Average Odds Difference  
  - Disparate Impact  
- Apply **Reweighing** for bias mitigation.  
- Compare before and after mitigation performance.

---

## 🗂️ Repository Structure

