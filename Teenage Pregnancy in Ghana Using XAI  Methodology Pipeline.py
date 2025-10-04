# Import Libraries

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV

# Load dataset
file_path = "TEENAGE PREGNANCY.xlsx"   
data = pd.ExcelFile(file_path)

# Load first sheet
df = data.parse("Sheet1")

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Apply Label Encoding to categorical columns 
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str)) 
    label_encoders[col] = le

# Apply Min-Max Normalization 
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# Train ML Models

# Define features (X) and target (y) 
X = df.drop("Teenage pregnancy", axis=1)
y = df["Teenage pregnancy"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE only on training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())

# Models 
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    print(f"\n {name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# Hyperparameter tuning 

# Features (X) and Target (y)
X = df.drop("Teenage pregnancy", axis=1)
y = df["Teenage pregnancy"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE only on training set 
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === Define parameter grids ===
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"]
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],  # l1 needs solver='liblinear'
        "solver": ["lbfgs", "saga"]
    }
}

# === Define models ===
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Run GridSearchCV for each model
best_models = {}
for name in models:
    print(f"\n Tuning {name}...")
    grid = GridSearchCV(
        estimator=models[name],
        param_grid=param_grids[name],
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_res, y_train_res)
    best_models[name] = grid.best_estimator_

    print(f" Best Parameters for {name}: {grid.best_params_}")
    print(f" Best CV Score for {name}: {grid.best_score_:.4f}")

    # Evaluate on test set
    y_pred = grid.best_estimator_.predict(X_test)
    print(f" Test Accuracy for {name}: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


# Threshold Tuning

# Features & Target
X = df.drop("Teenage pregnancy", axis=1)
y = df["Teenage pregnancy"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Random Forest with class weights
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train_res, y_train_res)

# XGBoost with scale_pos_weight (ratio of negatives to positives)
scale_pos_weight = (y_train_res.value_counts()[0] / y_train_res.value_counts()[1])
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
xgb_model.fit(X_train_res, y_train_res)

# Function for threshold tuning
def evaluate_thresholds(model, X_test, y_test, name):
    y_probs = model.predict_proba(X_test)[:, 1]  # probability for class 1
    print(f"\n {name} Threshold Tuning Results:")
    for thresh in [0.3, 0.4, 0.5]:
        y_pred = (y_probs >= thresh).astype(int)
        print(f"\nThreshold = {thresh}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

# Evaluate both models
evaluate_thresholds(rf, X_test, y_test, "Random Forest")
evaluate_thresholds(xgb_model, X_test, y_test, "XGBoost")


# Confusion Matrix at best threshold (0.45) 
y_pred_best = (y_probs >= 0.45).astype(int)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Threshold = 0.45)")
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
avg_prec = average_precision_score(y_test, y_probs)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, marker=".", label=f"AP={avg_prec:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Ensemble)")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance from Random Forest
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8,6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.show()


# SHAP Analysis

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Load Model & Data
rf = joblib.load("random_forest_model.pkl")
X_test = pd.read_csv("X_test.csv")

# Use the new SHAP API (works with sklearn RF) 
explainer = shap.Explainer(rf, X_test)  
shap_values = explainer(X_test)   # returns shap.Explanation object

print("SHAP values shape:", shap_values.values.shape)  

# Global Feature Importance 
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

# Local Explanation for one sample 
shap.initjs()
shap.force_plot(
    explainer.expected_value, 
    shap_values.values[0, :], 
    X_test.iloc[0, :]
)


# SHAP Summary Plot

#  Load Model & Data 
rf = joblib.load("random_forest_model.pkl")
X_test = pd.read_csv("X_test.csv")

# SHAP Explainer
explainer = shap.Explainer(rf, X_test)
shap_values = explainer(X_test)   # Explanation object

print("SHAP values shape:", shap_values.values.shape)  # (851, 20, 2)

# Select SHAP values for positive class (Teenage pregnancy = 1) ===
shap_class1 = shap_values[..., 1]

# Global Feature Importance (Bar Plot) ===
plt.title("Global Feature Importance (Mean |SHAP| Values)")
shap.summary_plot(shap_class1, X_test, plot_type="bar")

# Detailed Feature Impact (Beeswarm Plot) ===
plt.title("SHAP Feature Impact (Beeswarm Plot)")
shap.summary_plot(shap_class1, X_test)



# Counterfactual Analysis (DICE_ML)


import dice_ml
from dice_ml import Dice
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load Model & Data 
rf = joblib.load("random_forest_model.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# Define Data Schema
d = dice_ml.Data(
    dataframe=pd.concat([X_test, y_test], axis=1),
    continuous_features=["Age of women", "Household size"],  # numeric cols
    outcome_name="Teenage pregnancy"
)
m = dice_ml.Model(model=rf, backend="sklearn")
exp = Dice(d, m, method="random")

# Loop through positive cases (Teenage pregnancy = 1)
positive_cases = X_test[y_test == 1].head(50)  # adjust N for speed

feature_changes = Counter()

for idx, sample in positive_cases.iterrows():
    cf = exp.generate_counterfactuals(sample.to_frame().T, total_CFs=3, desired_class="opposite")
    cf_df = cf.cf_examples_list[0].final_cfs_df.reset_index(drop=True)
    original = pd.concat([X_test, y_test], axis=1).loc[[idx]].reset_index(drop=True)
    combined = pd.concat([original, cf_df], keys=["Original", "CF1", "CF2", "CF3"])

    # Count changed features
    for col in combined.columns:
        if col != "Teenage pregnancy" and len(combined[col].unique()) > 1:
            feature_changes[col] += 1

# Convert to DataFrame for plotting 
change_df = pd.DataFrame.from_dict(feature_changes, orient="index", columns=["Change_Count"])
change_df = change_df.sort_values("Change_Count", ascending=False)

# Plot hierarchy
plt.figure(figsize=(10,6))
change_df.plot(kind="barh", legend=False, figsize=(10,6))
plt.title("Feature Change Frequency in Counterfactuals")
plt.xlabel("Number of Times Feature Changed")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # most changed at top
plt.tight_layout()
plt.show()

print(change_df)


# Total counterfactuals generated = (#positive_cases × total_CFs)
total_cf = len(positive_cases) * 3

# Add percentage column
change_df["Percentage"] = (change_df["Change_Count"] / total_cf) * 100

# Plot normalized percentages 
plt.figure(figsize=(10,6))
change_df["Percentage"].plot(kind="barh", figsize=(10,6), color="steelblue")
plt.title("Feature Change Frequency in Counterfactuals (%)")
plt.xlabel("Percentage of Counterfactuals with Feature Change")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
print(change_df)




# Causal Analysis

import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.concat([X_test, y_test], axis=1)

# Treatment variable (example: education level)
T = df["Education level"].values
# Outcome
Y = df["Teenage pregnancy"].values
# Confounders (all other features except treatment & outcome)
X_c = df.drop(columns=["Education level", "Teenage pregnancy"]).values

#  Define Causal Forest
est = CausalForestDML(
    model_t=RandomForestRegressor(),
    model_y=RandomForestRegressor(),
    n_estimators=500,
    min_samples_leaf=10,
    random_state=42
)

# Fit causal forest
est.fit(Y, T, X=X_c)

# Average Treatment Effect (ATE) 
ate = est.ate(X_c)
print("ATE (Education level → Teenage pregnancy):", ate)

# Conditional Average Treatment Effect (CATE)
cate = est.effect(X_c)
df_cate = pd.DataFrame({"CATE": cate})
print(df_cate.head())


# Subgroup example: Urban vs Rural (0 = Rural, 1 = Urban)
mask_urban = df["Residence"] == 1
mask_rural = df["Residence"] == 0

print("CATE (Urban):", est.ate(X_c[mask_urban]))
print("CATE (Rural):", est.ate(X_c[mask_rural]))

treatments = ["Education level", "Wealth status", "Residence", 
              "Problem with distance to health facility", 
              "Problem with permission to go alone"]

results = {}

for t in treatments:
    T = df[t].values
    Y = df["Teenage pregnancy"].values
    X_c = df.drop(columns=[t, "Teenage pregnancy"]).values

    est = CausalForestDML(
        model_t=RandomForestRegressor(),
        model_y=RandomForestRegressor(),
        n_estimators=500,
        min_samples_leaf=10,
        random_state=42
    )
    est.fit(Y, T, X=X_c)

    # Average effect
    ate = est.ate(X_c)
    # Subgroup example: Urban vs Rural
    cate_urban = est.ate(X_c[df["Residence"] == 1])
    cate_rural = est.ate(X_c[df["Residence"] == 0])

    results[t] = {
        "ATE": ate,
        "CATE_Urban": cate_urban,
        "CATE_Rural": cate_rural
    }

# Convert to DataFrame for easy comparison
results_df = pd.DataFrame(results).T
print(results_df)




# AIFAIRNESS ANALYSIS


import pandas as pd
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.ensemble import RandomForestClassifier

protected_attrs = ["Residence", "Wealth status", "Religion", "Ethnicity", "Sex of household head"]

# Store results
results = []

for attr in protected_attrs:
    # Train dataset for reweighing 
    train_df = pd.concat([X_train, y_train], axis=1)
    dataset_train = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=train_df,
        label_names=["Teenage pregnancy"],
        protected_attribute_names=[attr]
    )

    #  Test dataset 
    test_df = pd.concat([X_test, y_test], axis=1)
    dataset_test = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=test_df,
        label_names=["Teenage pregnancy"],
        protected_attribute_names=[attr]
    )

    # Baseline Model (Before Reweighing) 
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    dataset_pred = dataset_test.copy()
    dataset_pred.labels = y_pred.reshape(-1, 1)

    metric_before = ClassificationMetric(
        dataset_test, dataset_pred,
        unprivileged_groups=[{attr: 0}],
        privileged_groups=[{attr: 1}]
    )

    #  Reweighing
    RW = Reweighing(
        unprivileged_groups=[{attr: 0}],
        privileged_groups=[{attr: 1}]
    )
    dataset_train_transf = RW.fit_transform(dataset_train)

    rf_fair = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf_fair.fit(X_train, y_train, sample_weight=dataset_train_transf.instance_weights)

    y_pred_fair = rf_fair.predict(X_test)
    dataset_pred_fair = dataset_test.copy()
    dataset_pred_fair.labels = y_pred_fair.reshape(-1, 1)

    metric_after = ClassificationMetric(
        dataset_test, dataset_pred_fair,
        unprivileged_groups=[{attr: 0}],
        privileged_groups=[{attr: 1}]
    )

    # Collect results
    results.append({
        "Attribute": attr,
        "StatParity_Before": metric_before.statistical_parity_difference(),
        "StatParity_After": metric_after.statistical_parity_difference(),
        "EqOpp_Before": metric_before.equal_opportunity_difference(),
        "EqOpp_After": metric_after.equal_opportunity_difference(),
        "AvgOdds_Before": metric_before.average_odds_difference(),
        "AvgOdds_After": metric_after.average_odds_difference(),
        "DispImpact_Before": metric_before.disparate_impact(),
        "DispImpact_After": metric_after.disparate_impact()
    })

# Convert to DataFrame
fairness_df = pd.DataFrame(results)
print(fairness_df)

# Visualization: Statistical Parity Difference
plt.figure(figsize=(10,6))
fairness_df.plot(x="Attribute", y=["StatParity_Before","StatParity_After"], kind="bar")
plt.title("Statistical Parity Difference: Before vs After Reweighing")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization: Equal Opportunity Difference
plt.figure(figsize=(10,6))
fairness_df.plot(x="Attribute", y=["EqOpp_Before","EqOpp_After"], kind="bar")
plt.title("Equal Opportunity Difference: Before vs After Reweighing")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization: Average Odds Difference 
plt.figure(figsize=(10,6))
fairness_df.plot(x="Attribute", y=["AvgOdds_Before","AvgOdds_After"], kind="bar")
plt.title("Average Odds Difference: Before vs After Reweighing")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization: Disparate Impact
plt.figure(figsize=(10,6))
fairness_df.plot(x="Attribute", y=["DispImpact_Before","DispImpact_After"], kind="bar")
plt.title("Disparate Impact: Before vs After Reweighing")
plt.axhline(1.0, color="black", linestyle="--")  # ideal value
plt.ylabel("Ratio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

