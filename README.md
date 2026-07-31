🛡️ AI Employee Retention Intelligence System


Predicting flight-risk employees from workforce data and turning that signal into a targeted retention strategy.


📌 Project Overview & Objectives

Employee turnover is expensive, disruptive, and mostly invisible until a resignation letter lands. Traditional HR teams are forced to be reactive (conducting exit interviews and managing replacement hiring). This project builds a proactive Machine Learning Pipeline that identifies voluntary attrition risk before it happens.


🎯 Key Objectives:Predict & Explain Attrition: 

Build a highly calibrated classification model optimized for Recall to catch the costly minority (16.1%) of employees who leave.Domain Feature Engineering: Engineer custom indicators reflecting scheduling strain and structural plateaus (Burnout_Index, Career_Stagnation_Index).Interactive Policy Simulator: Turn predictive data into an interactive, enterprise-grade Streamlit Dashboard where HR leadership can simulate policy changes (e.g., overtime caps) and view risk fluctuations in real-time.


📊 The Core Data Challenge: Severe Class ImbalanceThe dataset consists of 1,470 employee corporate records with 35 original features (sourced from the public IBM HR Analytics Dataset).


Stayed (Class 0) ──■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 83.9% (1,233 rows)
Left   (Class 1) ──■■■■■■■ 16.1% (237 rows)


⚠️ The Accuracy Trap: A baseline model that blindly predicts that every employee stays will score an 84% Accuracy while catching 0% of true flight risks. Therefore, this project completely skips accuracy and optimizes for Recall, Precision, and F1-Score.


💡 Key Exploratory Data Analysis (EDA) Findings


Five major behavioral signals separate stayers from leavers:

⏱️ Overtime Strain: Employees on mandatory overtime face a 30% attrition rate compared to just 10% for non-overtime peers.

💰 Compensation Thresholds: Leavers have a significantly lower median monthly income ($3.2k) than stayers ($5.2k).

⏳ Tenure Window: Attrition heavily concentrates within the first 0 to 3 years of employment before flattening out.

🏢 Department Slices: The Sales Department represents the highest voluntary churn layer across the organization.


⚙️ Advanced Preprocessing & Feature Engineering

To prevent data leakage, a strict stratified 80/20 split was executed before any transformation steps.

🧠 Domain Engineered Features:

Burnout_Index: 

Combines overtime constraints, travel frequency mapping, and inverted work-life balance satisfaction scores:

Overtime pts + Travel pts + (3 − Work-Life Balance)


Career_Stagnation_Index: 

A proxy tracking structural plateauing using Laplace smoothing to protect against new hires

(Years in Role + 1) ÷ (Years at Company + 1)


## ⚖️ Ethical Considerations & Limitations

- A risk score must inform **supportive conversations, never punitive ones** — flagging someone as "flight risk" can itself damage trust if mishandled.
- The model reflects **one company's data (1,470 records)** and may not generalize across industries, tenure structures, or regions.
- It should **complement**, not replace, manager judgment.

🛠️ Leak-Proof Pipeline:

Categorical Encoding: OneHotEncoder(drop='if_binary') applied to string variables.

Feature Scaling: StandardScaler() applied to numerical fields.

Feature Selection: Multicollinear variables discovered via bivariate correlation heatmaps (JobLevel, YearsWithCurrManager, YearsInCurrentRole) were programmatically pruned to preserve model interpretability.

🏆 Model Optimization & Selection Leaderboard

An exhaustive model selection sweep evaluated algorithms across multiple distinct families (Linear, Single Tree, Parallel Ensembles, and Sequential Boosting).

| Model Configuration | Precision (Class 1) | Recall (Catch Rate) | F1-Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Oversampled Logistic Regression** | 0.41 | 70.2% | 0.52 | 🏆 Champion |
| **Undersampled Logistic Regression** | 0.36 | 68.1% | 0.47 | Baseline |
| **SMOTE Logistic Regression** | 0.40 | 66.0% | 0.50 | Baseline |
| **Grid-Tuned Random Forest** | 0.39 | 59.6% | 0.47 | Tree Winner |
| **Tuned AdaBoost** | 0.72 | 28.0% | 0.40 | High Precision |

 

🔍 Production-Grade Validation

The champion Oversampled Logistic Regression model was wrapped inside an imblearn pipeline and verified using a 5-Fold Stratified Cross-Validation loop to isolate resampling strictly within training folds:

Mean CV Catch Rate (Recall): 72.11% ± 0.0

7Mean CV Alarm Accuracy (Precision): 38.04% ± 0.03


🧭 Strategic HR Application: 

Choosing a ModelModel choice is a business decision following the cost of a missed flight-risk vs. the cost of an intervention:

High-Coverage Intervention Strategy (Tuned Random Forest / Oversampled LR): Offers up to 72% CV Recall. Best for low-cost, company-wide interventions (e.g., manager check-ins, internal role rotation tracking).

High-Confidence Intervention Strategy (Tuned AdaBoost): Offers 72% Precision. Best when interventions are high-cost (e.g., custom retention bonuses, equity top-ups) and false alarms must be strictly avoided.


🚀 Interactive Workforce Intelligence Suite (Streamlit)

The final pipeline is deployed as a web application synced with  HRIS structures.

+---------------------------------------------------------------------------------------+


⚡ Dashboard Core Features:

Dynamic KPI Cards: Displays active risk numbers, recall benchmarks, and weighted economic savings calculations based on customized business assumptions.

Live Policy Simulator (Sidebar): Allows HR managers to toggle mandatory overtime caps or sliders for work-life balance improvements to see employee risk scores adjust dynamically.

Simulated Flight Risk Roster: An interactive table filterable by department, sorting employees by their live predictive risk probabilities with visual progress bars.

📂 Repository Structure

├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # HR Core Dataset
├── app.py                                      # Streamlit Web Application Interface
├── model_pipeline.py                           # Fitted Champion Model, Scaler, & Encoder
├── requirements.txt                            # Project Dependencies
└── README.md                                   # Production Documentation

## 🔭 What's Next

- SHAP explainability per employee
- Validate across multiple years / companies
- Pilot with a live HR team

## 📊 Model Performance Summary

| Metric | Champion Model (Oversampled Logistic Regression) |
|---|---|
| Recall (5-fold CV) | 72.11% ± 7.2% |
| Precision (5-fold CV) | 38.0% ± 3.4% |
| F1 (5-fold CV) | 0.497 ± 0.037 |

---

*Capstone project · IBM HR Analytics Employee Attrition dataset · Built with scikit-learn, imbalanced-learn, and Streamlit.*

Author : Shilpa Vellore Krishnamurthy
