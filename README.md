# Customer Segmentation in Banking Using Clustering and Data Mining
## ApexBank | Enterprise Customer Segmentation BI Platform
**Course:** Data Warehousing and Data Mining (DWDM)  
**Dataset:** Default of Credit Card Clients (UCI Machine Learning Repository, ID: 350)  
**Official DOI:** [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)  
**Official Citation:** Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository.  
**Academic Year:** 2026  

---

## 1. Executive Summary
This project provides an end-to-end Data Warehousing and Data Mining (DWDM) solution for retail credit banking. Utilizing the official **UCI Machine Learning Repository "Default of Credit Card Clients"** dataset (30,000 credit accounts, 24 multi-month financial and demographic features), this platform demonstrates:
- **Dimensional Modeling:** An analytical Star Schema comprising five dimension tables (`DIM_CUSTOMER`, `DIM_CREDIT`, `DIM_PAYMENT_HISTORY`, `DIM_BILLING`, `DIM_PAYMENT`) and a central `FACT_CUSTOMER_CREDIT_PROFILE`.
- **ETL Transformation:** Data validation, harmonization of undocumented categorical codes, zero-loss outlier retention, and analytical feature engineering.
- **Unsupervised Data Mining:** Systematic multi-metric K-Means evaluation across $K \in [2, 10]$ evaluating Within-Cluster Sum of Squares (Inertia), Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index to mathematically determine **$K = 4$**.
- **PCA Dimensionality Reduction:** 2D visualization explaining 56.16% of total variance across spending magnitude and delinquency severity.
- **Post-Hoc Risk Analysis:** Empirical examination of default outcomes ($Y$) across natural behavioral clusters without supervised leakage.
- **Executive BI Dashboard:** An interactive 15-page Streamlit platform deployed at `http://localhost:8501/` with a live customer persona classifier, featuring full **Dark Luxury Mode**, **Light Professional Mode**, and **System Default Auto-Adapt** (CSS media query dynamic palette switching).

---

## 2. Dataset Specifications & Verification
| Parameter | Value | Verification Source |
| :--- | :--- | :--- |
| **Dataset Name** | Default of Credit Card Clients | UCI Machine Learning Repository (ID: 350) |
| **Total Population** | 30,000 customers | Taiwan credit card cardholders (April–October 2005) |
| **Total Raw Columns** | 25 columns | 24 behavioral/demographic inputs + 1 target (`Y`) |
| **Missing Values** | 0 | 100% record completeness |
| **Duplicate Records** | 0 | Zero duplicate rows detected |
| **Identifier Handling** | `ID` removed | Sequential key with zero behavioral signal |
| **Target Variable Handling** | `Y` excluded from clustering | Supervised outcome reserved for post-hoc validation |
| **Primary Demographics** | Age, Gender, Education, Marriage | Harmonized into standard categories |
| **Multi-Month Financials** | `BILL_AMT1`..`6`, `PAY_AMT1`..`6` | 6-month statement balances & remittances (NT$) |
| **Repayment History** | `PAY_0`..`PAY_6` | Monthly payment promptness & overdue delay scale |

---

## 3. Dimensional Data Warehouse Architecture (Star Schema)
```
                     +----------------------------+
                     |        DIM_CUSTOMER        |
                     +----------------------------+
                     | Customer_Key (PK)          |
                     | Customer_ID, Age, Gender   |
                     | Education, Marital_Status  |
                     +-------------+--------------+
                                   |
+--------------------------+       |       +--------------------------+
|        DIM_CREDIT        |       |       |   DIM_PAYMENT_HISTORY    |
+--------------------------+       |       +--------------------------+
| Credit_Key (PK)          |       |       | Payment_History_Key (PK) |
| Credit_Limit, Tier       |       |       | PAY_0..PAY_6, Max_Delay  |
+------------+-------------+       |       +------------+-------------+
             |                     |                    |
             +-------------> [ FACT_CUSTOMER_ ] <-------+
             |               [ CREDIT_PROFILE ]
             |                     |
+------------+-------------+       |       +--------------------------+
|       DIM_BILLING        |       |       |       DIM_PAYMENT        |
+--------------------------+       |       +--------------------------+
| Billing_Key (PK)         |       |       | Payment_Key (PK)         |
| BILL_AMT1..6, Total_Bill |-------+-------| PAY_AMT1..6, Total_Pay   |
+--------------------------+               +--------------------------+
```
*Note: This logical dimensional model is project-engineered from the source data to optimize OLAP slicing, dicing, and machine learning feeds.*

---

## 4. Multi-Metric Cluster Optimization Results ($K = 2$ to $10$)
Standardized with `StandardScaler` across 8 behavioral features:
- `LIMIT_BAL` (Credit Limit)
- `AGE`
- `Avg_Bill_Amt` (Mean 6-month bill)
- `Avg_Pay_Amt` (Mean 6-month payment)
- `Payment_to_Bill_Ratio` (Solvency metric)
- `Avg_Repayment_Status` (Average delay index)
- `Max_Repayment_Delay` (Peak overdue months)
- `Delayed_Months_Count` (Count of months with overdue status)

| K | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Decision |
| :-: | :---: | :---: | :---: | :---: | :--- |
| 2 | 184,366.8 | 0.2868 | 1.3394 | 9,052.0 | Over-aggregated |
| 3 | 157,430.1 | 0.2855 | 1.4076 | 7,866.5 | 62% in one cluster |
| **4** | **136,382.0** | **0.2443** | **1.3970** | **7,596.6** | **OPTIMAL SELECTION** |
| 5 | 117,964.2 | 0.2439 | 1.2046 | 7,757.6 | Sub-segment noise |
| 6 | 105,768.3 | 0.2405 | 1.1414 | 7,613.2 | Diminishing returns |
| 7 | 95,846.4 | 0.2528 | 1.1113 | 7,518.3 | Fragmented clusters |
| 8 | 87,571.3 | 0.2476 | 1.1292 | 7,457.8 | Over-segmentation |
| 9 | 81,366.3 | 0.2556 | 1.1650 | 7,308.9 | Redundant personas |
| 10 | 77,430.0 | 0.2508 | 1.1624 | 6,996.2 | Unstable centroids |

---

## 5. Discovered Customer Personas ($K = 4$)
| Cluster | Persona Title | Size | Share | Avg Limit | Avg Bill | Avg Pay | Max Delay | Default (Y) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **0** | **High Credit / Advance & Prompt Payers** | 7,261 | 24.20% | NT$ 262,607 | NT$ 12,056 | NT$ 5,146 | -0.82 mos | 14.69% |
| **1** | **Moderate Credit / Stable Payment Pattern** | 14,214 | 47.38% | NT$ 107,634 | NT$ 41,202 | NT$ 3,181 | +0.45 mos | 16.86% |
| **2** | **Lower Credit / High Repayment Delay Pattern** | 4,897 | 16.32% | NT$ 88,149 | NT$ 43,477 | NT$ 2,321 | +2.41 mos | **55.67%** |
| **3** | **Higher Credit / Higher Payment Activity** | 3,628 | 12.09% | NT$ 318,679 | NT$ 171,768 | NT$ 17,727 | +0.26 mos | **12.29%** |

---

## 6. Project Directory Structure
```
DWDM project 1/
├── dashboard/
│   └── app.py                     # 15-Page Streamlit BI Platform + Live Predictor
├── data/
│   ├── raw/
│   │   └── default_of_credit_card_clients.csv  # 30,000 UCI Records
│   └── processed/
│       ├── credit_card_cleaned.csv             # Post-ETL Cleaned Data
│       ├── credit_card_features.csv            # Engineered Features Matrix
│       └── credit_card_clustered.csv           # Final Assigned Segments & PCA
├── outputs/
│   ├── figures/                   # 16 Publication Figures
│   ├── models/                    # Serialized Scaler, K-Means, PCA Joblib
│   └── tables/                    # CSV Validation Tables & Metrics
├── sql/
│   ├── create_tables.sql          # Star Schema DDL Script
│   └── etl.sql                    # SQL Transformation & Loading Queries
├── src/
│   ├── data_cleaning.py           # ETL & Quality Audit
│   ├── feature_engineering.py     # Solvency & Delay Features
│   ├── clustering.py              # K=2..10 Multi-Metric Pipeline & PCA
│   ├── visualization.py           # Publication-Grade Visualizations
│   └── main.py                    # Master Pipeline Orchestrator
├── DEMO_SCRIPT.md                 # 20 Mandatory Viva Questions & Answers
├── PROJECT_AUDIT.md               # Complete System Audit & Dataset Migration
├── PROJECT_REPORT.md              # 7-Chapter Academic Project Report
├── requirements.txt               # Dependencies
└── README.md                      # Project Documentation
```

---

## 7. How to Run the Project

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Required packages: `pandas`, `numpy`, `scikit-learn`, `joblib`, `plotly`, `streamlit`, `ucimlrepo`

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Execute End-to-End Pipeline
```bash
python src/main.py
```
*Outputs all 16 figures, 8 summary tables, and trained model artifacts.*

### Step 3: Launch ApexBank Streamlit BI Platform
```bash
streamlit run dashboard/app.py --server.port 8501
```
Open your browser at `http://localhost:8501/`.

---

## 8. Academic Integrity & Verification
- **Dataset Source:** Strictly official UCI Machine Learning Repository (Yeh, 2009; DOI: 10.24432/C55S3H).
- **Target Handling:** Target variable `Y` is strictly excluded from clustering and used exclusively for post-hoc validation.
- **Evaluation Validity:** No classification accuracy is displayed for unsupervised clustering; all metrics use internal geometric indices (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz).
