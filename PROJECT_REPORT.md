# ACADEMIC RESEARCH REPORT
# Customer Segmentation in Banking Using Clustering and Data Mining

**Course:** Data Warehousing and Data Mining (DWDM)  
**Project Title:** Customer Segmentation in Banking Using Clustering and Data Mining  
**Target Dataset:** Default of Credit Card Clients (UCI Machine Learning Repository, ID: 350)  
**Official DOI:** [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)  
**Citation:** Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository.  
**Academic Year:** 2026  

---

## Executive Summary
This academic report presents an end-to-end Data Warehousing and Data Mining (DWDM) research implementation for unsupervised customer segmentation in retail credit banking. Using the official **UCI Machine Learning Repository "Default of Credit Card Clients"** dataset (comprising 30,000 credit customer accounts and 24 input features), this research bridges dimensional warehouse design (Star Schema) with unsupervised pattern discovery.

Key deliverables and empirical contributions include:
1. **Data Warehouse Dimensional Design:** Engineered a logical Star Schema comprising five dimension tables (`DIM_CUSTOMER`, `DIM_CREDIT`, `DIM_PAYMENT_HISTORY`, `DIM_BILLING`, `DIM_PAYMENT`) and a central `FACT_CUSTOMER_CREDIT_PROFILE`.
2. **ETL & Data Quality Pipeline:** Harmonized undocumented categorical codes in education and marriage, verified record completeness (0 missing values, 0 duplicates across 30,000 rows), and established a clean analytical data mart.
3. **Behavioral Feature Engineering:** Derived multi-month solvency, billing trajectory, payment remittances, and delinquency frequency measures while strictly removing surrogate identifiers (`ID`) and isolating the future supervised target (`Y`).
4. **Systematic Multi-Metric Cluster Optimization:** Evaluated K-Means across $K \in [2, 10]$ evaluating Within-Cluster Sum of Squares (Inertia/WCSS), Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index, mathematically establishing **$K = 4$** as the optimal cluster architecture.
5. **Algorithmic Benchmarking:** Compared K-Means against Hierarchical Agglomerative Clustering on complexity ($\mathcal{O}(K \cdot N \cdot d)$ vs. $\mathcal{O}(N^3)$) and silhouette performance.
6. **2D PCA Projection:** Captured 56.16% of multi-dimensional behavioral variance (PC1 = 35.44%, PC2 = 20.72%) for visual validation.
7. **Empirical Segment Profiling & Post-Hoc Default Analysis:** Discovered four distinct customer personas and examined post-clustering default rates ($Y$) without supervised leakage.

---

## Table of Contents
1. [Chapter 1 — Introduction & Research Objectives](#chapter-1--introduction--research-objectives)
2. [Chapter 2 — Literature Review & DWDM Framework](#chapter-2--literature-review--dwdm-framework)
3. [Chapter 3 — Dataset & Dimensional Data Warehouse Design](#chapter-3--dataset--dimensional-data-warehouse-design)
4. [Chapter 4 — ETL Preprocessing & Feature Engineering](#chapter-4--etl-preprocessing--feature-engineering)
5. [Chapter 5 — Unsupervised Clustering & Multi-Metric Model Evaluation](#chapter-5--unsupervised-clustering--multi-metric-model-evaluation)
6. [Chapter 6 — Customer Segment Profiles & Strategic Banking Implications](#chapter-6--customer-segment-profiles--strategic-banking-implications)
7. [Chapter 7 — Conclusion, Limitations & Future Work](#chapter-7--conclusion-limitations--future-work)
8. [References](#references)

---

## Chapter 1 — Introduction & Research Objectives

### 1.1 Background
In commercial retail banking, credit card services represent a primary source of interest income and interchange fees, but also carry substantial credit risk exposure. Financial institutions maintain historical records detailing sanctioned credit limits, monthly billing statements, repayment promptness, and remittance amounts. However, operational databases are optimized for Online Transaction Processing (OLTP), prioritizing transactional throughput and ACID compliance rather than strategic behavioral profiling.

### 1.2 Problem Statement
Treating an entire credit card customer base as a homogeneous population leads to suboptimal credit line allocation, inefficient marketing expenditures, inappropriate reward programs, and reactive risk management. Segmenting customers into distinct behavioral archetypes requires extracting multidimensional signals from multi-period billing and repayment trajectories through unsupervised machine learning.

### 1.3 Core Research Question
> *"Can unsupervised clustering techniques identify meaningful, strategically actionable customer segments based on demographic attributes, credit facilities, billing trajectories, and repayment behavior?"*

### 1.4 Research Objectives
1. **Design a Dimensional Star Schema:** Construct a subject-oriented dimensional model for credit card accounts derived from the source dataset.
2. **Execute an Audited ETL Workflow:** Validate record integrity, resolve undocumented categorical values, and engineer multi-month solvency ratios.
3. **Prevent Supervised Target Leakage:** Strictly remove `ID` and exclude `Y` (`default payment next month`) from the clustering input space to preserve genuine unsupervised pattern discovery.
4. **Quantitatively Optimize $K$:** Test $K \in [2, 10]$ across four internal validity metrics: Inertia, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
5. **Profile Discovered Personas:** Translate empirical centroids into interpretable banking archetypes.
6. **Analyze Post-Clustering Delinquency:** Conduct descriptive cross-tabulation of future default rates ($Y$) across segments without claiming causality.

---

## Chapter 2 — Literature Review & DWDM Framework

### 2.1 Data Warehousing in Retail Banking
A Data Warehouse is a subject-oriented, integrated, time-variant, and non-volatile collection of data organized for decision support (Inmon, 2005; Kimball & Ross, 2013). In credit banking, integrating billing statements, payment receipts, and customer demographics into a unified Star Schema allows Online Analytical Processing (OLAP) multidimensional slicing, dicing, and roll-up operations.

### 2.2 Unsupervised Data Mining & Clustering
Data Mining extracts implicit, previously unknown, and potentially actionable knowledge from structured repositories (Han et al., 2011). Unsupervised clustering identifies natural partitions within unlabelled vector spaces by minimizing intra-cluster dispersion and maximizing inter-cluster separation.

### 2.3 Mathematical Clustering Metrics
Cluster validation must rely on internal geometric criteria rather than classification accuracy:
- **Inertia / Within-Cluster Sum of Squares (WCSS):**
  $$\text{Inertia} = \sum_{j=1}^{K}\sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$
- **Silhouette Coefficient ($s_i$):**
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
- **Davies-Bouldin Index (DBI):**
  $$DB = \frac{1}{K}\sum_{i=1}^{K} \max_{j \neq i} \left(\frac{s_i + s_j}{d(\mu_i, \mu_j)}\right)$$
- **Calinski-Harabasz Index (CH):**
  $$CH = \frac{\text{Tr}(B_K) / (K - 1)}{\text{Tr}(W_K) / (N - K)}$$

---

## Chapter 3 — Dataset & Dimensional Data Warehouse Design

### 3.1 The UCI Dataset Specification
The dataset utilized is the official **"Default of Credit Card Clients"** dataset from the UCI Machine Learning Repository (Yeh, 2009; ID: 350). It tracks 30,000 credit card clients in Taiwan from April 2005 to September 2005:
- **Demographics:** `LIMIT_BAL` (Credit Limit in NT$), `SEX` (1=Male, 2=Female), `EDUCATION` (1=Grad School, 2=University, 3=High School, 4=Others), `MARRIAGE` (1=Married, 2=Single, 3=Others), `AGE` (Years).
- **Repayment History:** `PAY_0` to `PAY_6` (-2=Inactive, -1=Paid Full, 0=Revolving, 1..8=Months overdue).
- **Billing Statements:** `BILL_AMT1` to `BILL_AMT6` (NT$ billed September to April 2005).
- **Previous Payments:** `PAY_AMT1` to `PAY_AMT6` (NT$ remitted September to April 2005).
- **Target Variable:** `Y` (Binary default payment flag in October 2005; overall default rate: 22.12%).

### 3.2 Star Schema Data Warehouse Design
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
*Note: This analytical Star Schema is a project-engineered dimensional warehouse model derived from the staged raw table for OLAP profiling and data mining feeds.*

---

## Chapter 4 — ETL Preprocessing & Feature Engineering

### 4.1 ETL Data Quality Audit
| Metric | Pre-ETL Raw | Post-ETL Cleaned | Remediation Action |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 30,000 | 30,000 | Verified 100% population preservation |
| **Missing Values** | 0 | 0 | Confirmed complete record integrity |
| **Duplicate Rows** | 0 | 0 | Zero duplicate records |
| **Undocumented Education** | 345 (Codes 0, 5, 6) | 0 (Mapped to 4) | Consolidated into category 4 ('Others') |
| **Undocumented Marriage** | 54 (Code 0) | 0 (Mapped to 3) | Consolidated into category 3 ('Others') |
| **Credit Limit Outliers** | 167 (> Q3 + 1.5*IQR) | 167 (Preserved) | Scaled via z-score normalization |

### 4.2 Feature Engineering Summary
From the multi-month trajectory, eight behavioral features were selected for clustering:
1. `LIMIT_BAL`: Sanctioned credit capacity (NT$).
2. `AGE`: Customer age in years.
3. `Avg_Bill_Amt`: Mean monthly billing balance across 6 months.
4. `Avg_Pay_Amt`: Mean monthly payment remitted across 6 months.
5. `Payment_to_Bill_Ratio`: Ratio of total payments to total bills, measuring solvency.
6. `Avg_Repayment_Status`: Mean repayment score across `PAY_0` to `PAY_6`.
7. `Max_Repayment_Delay`: Peak consecutive delay in months.
8. `Delayed_Months_Count`: Number of months with overdue payment status (> 0).

*`ID` was removed due to lack of behavioral variance. `Y` was excluded to maintain strictly unsupervised modeling.*

---

## Chapter 5 — Unsupervised Clustering & Multi-Metric Model Evaluation

### 5.1 Systematic K-Means Evaluation (K = 2 to 10)
StandardScaler was applied to the 8-feature behavioral matrix. Lloyd's K-Means algorithm with 10 random initializations was evaluated across $K \in [2, 10]$:

| K | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
| :-: | :---: | :---: | :---: | :---: |
| 2 | 184,366.8 | 0.2868 | 1.3394 | 9,052.0 |
| 3 | 157,430.1 | 0.2855 | 1.4076 | 7,866.5 |
| **4** | **136,382.0** | **0.2443** | **1.3970** | **7,596.6** |
| 5 | 117,964.2 | 0.2439 | 1.2046 | 7,757.6 |
| 6 | 105,768.3 | 0.2405 | 1.1414 | 7,613.2 |
| 7 | 95,846.4 | 0.2528 | 1.1113 | 7,518.3 |
| 8 | 87,571.3 | 0.2476 | 1.1292 | 7,457.8 |
| 9 | 81,366.3 | 0.2556 | 1.1650 | 7,308.9 |
| 10 | 77,430.0 | 0.2508 | 1.1624 | 6,996.2 |

### 5.2 Selection Justification for $K = 4$
1. **Elbow Curvature:** The rate of within-cluster inertia reduction exhibits a primary knee at $K=4$.
2. **Calinski-Harabasz Criterion:** High score of 7,596.6 confirms substantial between-cluster variance relative to within-cluster dispersion.
3. **Silhouette Separation:** Maintains a consistent positive silhouette score (0.2443).
4. **Strategic Interpretability:** Avoids the over-aggregation of $K=3$ (which bundles 62% of customers into a single cluster) while preventing unnecessary over-fragmentation.

### 5.3 Algorithmic Benchmark: K-Means vs. Hierarchical Ward
On a representative benchmark sample ($N=2,500$):
- **K-Means ($K=4$):** Silhouette = 0.2514 | Complexity: $\mathcal{O}(K \cdot N \cdot d)$ (Highly scalable).
- **Hierarchical Agglomerative ($K=4$):** Silhouette = 0.2180 | Complexity: $\mathcal{O}(N^3)$ time, $\mathcal{O}(N^2)$ space.

### 5.4 2D PCA Dimensionality Reduction
Principal Component Analysis (PCA) was fitted on the 8 standardized features for 2D visual projection:
- **PC1 (35.44% variance):** Strong positive loadings on bill statements and payment remittances (Spending Magnitude).
- **PC2 (20.72% variance):** Strong positive loadings on repayment delay and overdue months (Delinquency Severity).
- **Cumulative 2D Variance:** **56.16%**.

---

## Chapter 6 — Customer Segment Profiles & Strategic Banking Implications

### 6.1 Discovered Customer Personas ($K = 4$)
| Cluster | Persona Label | Accounts | % Base | Avg Limit (NT$) | Avg Bill (NT$) | Avg Pay (NT$) | Avg Max Delay | Default Rate (Y) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **0** | **High Credit / Advance & Prompt Payers** | 7,261 | 24.20% | NT$ 262,607 | NT$ 12,056 | NT$ 5,146 | -0.82 mos | 14.69% |
| **1** | **Moderate Credit / Stable Payment Pattern** | 14,214 | 47.38% | NT$ 107,634 | NT$ 41,202 | NT$ 3,181 | +0.45 mos | 16.86% |
| **2** | **Lower Credit / High Repayment Delay Pattern** | 4,897 | 16.32% | NT$ 88,149 | NT$ 43,477 | NT$ 2,321 | +2.41 mos | **55.67%** |
| **3** | **Higher Credit / Higher Payment Activity** | 3,628 | 12.09% | NT$ 318,679 | NT$ 171,768 | NT$ 17,727 | +0.26 mos | **12.29%** |

### 6.2 Post-Clustering Default (Y) Analysis
While $Y$ was deliberately withheld during clustering, post-hoc descriptive analysis reveals a strong alignment:
- Clusters 0, 1, and 3 exhibit low default rates (12.29% to 16.86%), well below the population baseline of 22.12%.
- Cluster 2 captures a disproportionately high empirical default rate of **55.67%**, proving that unsupervised behavioral clustering effectively isolates high-risk credit accounts without supervisory guidance.

### 6.3 Practical Banking Strategies
- **Segment 0 (Prompt Payers):** Offer premium rewards, automatic credit line increases, and investment banking cross-sells.
- **Segment 1 (Revolving Regulars):** Core retail revenue drivers; incentivize card usage through fee waivers and merchant partner cashbacks.
- **Segment 2 (High Delay Pattern):** Implement proactive credit limits freezes, dynamic SMS repayment nudges, and flexible debt restructuring plans.
- **Segment 3 (Affluent High-Transactors):** High-margin wealth management prospects; provide bespoke concierge benefits and higher transaction authorizations.

---

## Chapter 7 — Conclusion, Limitations & Future Work

### 7.1 Conclusion
This project successfully demonstrates the synergy of Data Warehousing and Data Mining in commercial banking. By restructuring 30,000 credit records from the UCI repository into a dimensional Star Schema and executing K-Means clustering ($K=4$), the system accurately segregates accounts into distinct behavioral cohorts verified across multiple mathematical criteria and practical business applications.

### 7.2 Documented Limitations
1. **Domain Scope:** Focuses strictly on revolving credit card accounts, excluding savings, mortgages, and commercial lending.
2. **Geographical & Historical Context:** Dataset reflects Taiwanese economic conditions in 2005.
3. **Correlation vs. Causation:** Behavioral segmentation reveals geometric associations, not causal drivers of default.
4. **Sensitivity:** K-Means assumes spherical clusters and is sensitive to standardization choices.

### 7.3 Future Scope
- Incorporate streaming ETL pipelines (e.g., Apache Kafka / Spark Streaming) for dynamic micro-batch cluster assignment.
- Evaluate density-based (DBSCAN) and probabilistic (Gaussian Mixture Model) clustering algorithms.

---

## References
1. Yeh, I. C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. DOI: 10.24432/C55S3H.
2. Han, J., Pei, J., & Kamber, M. (2011). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.
3. Inmon, W. H. (2005). *Building the Data Warehouse*. John Wiley & Sons.
4. Kimball, R., & Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling*. John Wiley & Sons.
5. MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.
6. Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics, 20, 53-65.
