# APEXBANK PROJECT AUDIT & MIGRATION REPORT
**Project Title:** Customer Segmentation in Banking Using Clustering and Data Mining  
**Platform:** ApexBank | Customer Segmentation BI Platform  
**Target Dataset:** Default of Credit Card Clients (UCI Machine Learning Repository, ID: 350)  
**Date of Audit:** September 30, 2026  
**Auditor:** Antigravity AI Engineering & DWDM Research Team  

---

## Executive Summary
This audit provides a comprehensive structural, analytical, and algorithmic assessment of the existing codebase. The project was previously configured around an Indian retail banking transaction dataset (`bank_transactions.csv`, ~1.04M records) with attributes such as `TransactionAmount (INR)`, `CustAccountBalance`, `CustLocation`, and `CustomerDOB`. 

In accordance with academic specifications and strict project requirements, the codebase is being audited and aligned with the official **UCI Machine Learning Repository "Default of Credit Card Clients"** dataset (Yeh, 2009; DOI: 10.24432/C55S3H).

---

## A. What Already Works
1. **Streamlit Application Infrastructure:**
   - Dark luxury executive theme (`Plus Jakarta Sans`, sleek cards, custom CSS badges, responsive grid).
   - Custom HTML table rendering engine (`render_fintech_table()`) using pure Pandas `.to_html()` that completely bypasses Windows Application Control `pyarrow` DLL crashes.
   - Interactive Plotly chart integration with responsive UI wrappers.
2. **Modular Architecture:**
   - Clean folder separation: `src/` (cleaning, feature engineering, clustering, visualization, main pipeline), `data/` (raw and processed data tiers), `sql/` (warehouse schema DDL and ETL), `dashboard/` (Streamlit BI application), `outputs/` (models, figures, and tables).
3. **Machine Learning Pipeline Skeleton:**
   - Scikit-learn standardization (`StandardScaler`), K-Means clustering, and PCA 2D dimensionality reduction pipelines.
   - Model persistence (`joblib`) and CSV export mechanisms.

---

## B. What Is Partially Implemented
1. **Cluster Evaluation Metrics:**
   - The previous K-Means evaluation only calculated **Inertia (WCSS)** and **Silhouette Score** for $K \in [2, 8]$.
   - It lacked **Davies-Bouldin Index** and **Calinski-Harabasz Index**, which are required for multi-metric academic cluster validation.
   - Evaluation must be expanded to $K \in [2, 10]$ and systematically compared across all four internal clustering criteria.
2. **Dimensionality Reduction:**
   - 2D PCA was executed and saved, but the explained variance ratio and loadings were not fully integrated into analytical interpretations.
3. **Cluster Profiling:**
   - Basic summary aggregations were generated, but detailed behavioral distributions (delayed payment months, payment-to-bill ratios, credit limit utilization) were absent.

---

## C. What Is Incorrect
1. **Dataset Sourcing & Metrics Mismatch:**
   - The active raw dataset was `bank_transactions.csv` (Indian retail banking log), which is the wrong dataset.
   - Metrics displayed throughout the UI referenced "1,048,567 Transactions", "Account Balance (INR)", "Transaction Time / Peak Hours", and "Location / City Tier". None of these exist in the official UCI Credit Card dataset.
2. **Supervised Target Misuse / Supervised Confusion:**
   - In some previous documentation, classification accuracy was mentioned. Customer segmentation is an **unsupervised learning task**; classification accuracy, precision, recall, and F1-score are mathematically invalid for unlabelled clustering.
3. **Hardcoded Navigation & Labels:**
   - Cluster persona names in parts of the code were pre-assigned without empirical verification against cluster centroids.

---

## D. What Is Missing
1. **Official UCI Credit Card Dataset (30,000 Records):**
   - Clean ingestion of 24 features (`LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0`–`PAY_6`, `BILL_AMT1`–`BILL_AMT6`, `PAY_AMT1`–`PAY_AMT6`) and target variable `Y` (`default payment next month`).
2. **Dedicated DWDM Star Schema for Credit Profiles:**
   - Dimension tables: `DIM_CUSTOMER`, `DIM_CREDIT`, `DIM_PAYMENT_HISTORY`, `DIM_BILLING`, `DIM_PAYMENT`.
   - Fact table: `FACT_CUSTOMER_CREDIT_PROFILE`.
3. **Required 15 Navigation Pages:**
   - 1. Executive Overview
   - 2. Dataset Overview
   - 3. Data Quality
   - 4. Data Preprocessing
   - 5. Exploratory Data Analysis (EDA)
   - 6. Feature Engineering
   - 7. Customer Segmentation (Clustering)
   - 8. Optimal Number of Clusters (Optimal K)
   - 9. Clustering Evaluation
   - 10. Customer Segment Profiles
   - 11. PCA Cluster Visualization
   - 12. DWDM Architecture & Warehouse Design
   - 13. ETL Workflow
   - 14. Research & Findings
   - 15. Banking Applications & Limitations
4. **Post-Clustering Target Variable ($Y$) Analysis:**
   - Cross-tabulation of credit default distribution ($Y=0$ vs. $Y=1$) strictly post-clustering, ensuring $Y$ is never used as an input feature during clustering.
5. **Interactive Viva Preparation Document (`DEMO_SCRIPT.md`):**
   - 20 comprehensive questions and answers covering DWDM, ETL, K-Means mechanics, validation indices, and credit-card customer segmentation.

---

## E. What Needs Improvement
1. **Data Quality & Preprocessing Transparency:**
   - Display a side-by-side Before vs. After table for data quality (missing values, duplicate rows, undocumented categories in `EDUCATION` and `MARRIAGE`).
   - Detailed documentation on why undocumented codes (0, 5, 6 in `EDUCATION`; 0 in `MARRIAGE`) are consolidated into "Other".
2. **Feature Engineering Grounding:**
   - Creation of clear behavioral ratios: `Total_Bill_Amt`, `Avg_Bill_Amt`, `Total_Payment_Amt`, `Avg_Payment_Amt`, `Payment_to_Bill_Ratio`, `Bill_Variability`, `Payment_Variability`, `Max_Repayment_Delay`, `Delayed_Months_Count`.
   - Documenting feature inclusion/exclusion with analytical rationale.
3. **Visual Legend Positioning:**
   - Ensure all horizontal Plotly legends are positioned at the bottom center (`y=-0.22, x=0.5`) to eliminate overlap with Plotly's floating modebar buttons.

---

## F. What Conflicts with the Actual UCI Dataset
| Legacy Field / Concept | Conflict Status | Action Required |
| :--- | :--- | :--- |
| `TransactionAmount (INR)` | Not present in UCI dataset | Remove completely; replace with `BILL_AMT` & `PAY_AMT` (NT Dollars) |
| `CustAccountBalance` | Not present in UCI dataset | Remove completely; replace with `LIMIT_BAL` (Credit Limit) |
| `CustomerDOB` / `TransactionTime` | Not present in UCI dataset | Remove completely; replace with `AGE` & monthly repayment status `PAY_0`..`PAY_6` |
| `CustLocation` / City Tiers | Not present in UCI dataset | Remove completely; use Taiwan credit demographics (`SEX`, `EDUCATION`, `MARRIAGE`) |
| 1,048,567 Transactions | Wrong scale and entity grain | Standardize to 30,000 credit card account holders |
| ChatGPT / Kaggle citations | Incorrect academic source | Replace with official UCI citation: Yeh (2009), DOI: 10.24432/C55S3H |

---

## Implementation Roadmap
- [x] Download official UCI Credit Card dataset via `ucimlrepo` and save as `data/raw/default_of_credit_card_clients.csv`.
- [x] Create `PROJECT_AUDIT.md`.
- [ ] Implement `src/data_cleaning.py` for UCI dataset (auditing, categorical cleaning, validation).
- [ ] Implement `src/feature_engineering.py` (bill/payment aggregations, payment-to-bill ratios, repayment delay indices).
- [ ] Implement `src/clustering.py` ($K \in [2, 10]$, Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz, PCA, cluster profiling, post-clustering $Y$ analysis).
- [ ] Update `src/visualization.py` with dark-themed, publication-grade Plotly figures.
- [ ] Execute `src/main.py` to produce all verified output artifacts and models.
- [ ] Redesign `sql/create_tables.sql` and `sql/etl.sql` around the Credit Profile Star Schema.
- [ ] Upgrade `dashboard/app.py` with the 15 dedicated modules and live predictor.
- [ ] Create `DEMO_SCRIPT.md` (20 viva Q&As).
- [ ] Update `PROJECT_REPORT.md` and `README.md`.
- [ ] Launch and verify the Streamlit app at `http://localhost:8501/`.
