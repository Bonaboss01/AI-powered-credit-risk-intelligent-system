# 🏦 AI-Powered Credit Risk & Customer Intelligence Platform

An **end-to-end, production-style AI system** for **credit risk assessment, customer analytics, and decision intelligence**, designed to mirror how modern **banks, fintechs, and financial institutions** build, deploy, monitor, and explain credit models.

This project goes beyond a single ML model — it integrates **risk modelling, customer behaviour analytics (RFM), MLOps, explainability, AI agents, and APIs** into one cohesive system.

---

## 🚀 What This System Does

### 1️⃣ Credit Risk Intelligence
- Predicts **probability of default (PD)** for customers
- Produces **risk scores** and **risk bands** (Low / Medium / High)
- Supports:
  - Loan approval decisions
  - Credit limit adjustments
  - Early warning signals

### 2️⃣ Customer Analytics
- Builds **behavioural features** using:
  - **Recency** – How recently a customer transacted
  - **Frequency** – How often they transact
  - **Monetary** – Value of transactions
- Extends RFM into:
  - Credit utilisation patterns
  - Payment regularity
  - Delinquency behaviour
- Segments customers into **actionable personas**
  - Safe & profitable
  - High value but risky
  - Dormant / churn risk

### 3️⃣ Explainable AI (Regulatory-Ready)
- Uses SHAP to explain:
  - Why a customer is high-risk
  - Which features drove the prediction
- Generates **human-readable explanations** suitable for:
  - Risk committees
  - Regulators
  - Non-technical stakeholders

### 4️⃣ AI Agents
AI Agents **do not replace models**. They orchestrate decisions **around** them.

Agents in this system:
- **Risk Analyst Agent**
  - Interprets model outputs
  - Explains risk drivers in plain English
- **Policy Agent**
  - Applies business rules (e.g. lending thresholds)
- **Monitoring Agent**
  - Detects data drift & performance degradation
- **Reporting Agent**
  - Generates automated weekly summaries

👉 Agents act as **decision coordinators**, not predictors.

### 5️⃣ Weekly Re-Runnable Data Pipeline
- Data ingestion scripts are parameterised
- You can re-run them **every 7 days** by:
  - Changing dates
  - Pulling new raw data
- Enables:
  - Model retraining
  - Drift detection
  - Time-aware evaluatio.

---
### 🏗️ Project Structure

``` text
ai-credit-risk-intelligence/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── docker/
│   └── Dockerfile
│
├── data/
│   ├── raw/                    # Weekly refreshed raw datasets
│   ├── processed/              # Feature-ready datasets (gitignored)
│   ├── external/               # Data dictionaries, schema docs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_rfm_analysis.ipynb
│   ├── 04_baseline_credit_model.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_model_explainability.ipynb
│   ├── 07_model_comparison.ipynb
│   └── 08_monitoring_simulation.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingest.py            # Parameterised data ingestion
│   │   ├── validate.py          # Data quality checks
│   │   └── versioning.py        # Dataset version tagging
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── rfm_features.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── registry.py          # Model versioning
│   │
│   ├── monitoring/
│   │   ├── data_drift.py
│   │   ├── prediction_drift.py
│   │   └── performance_tracking.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   ├── agents/
│   │   ├── risk_agent.py
│   │   ├── monitoring_agent.py
│   │   └── decision_agent.py
│   │
│   └── api/
│       ├── app.py               # FastAPI service
│       └── schemas.py
│
├── pipelines/
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   └── retraining_pipeline.py
│
├── configs/
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── thresholds.yaml
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
└── docs/
    ├── architecture.png
    ├── risk_flow.md
    └── assumptions.md




