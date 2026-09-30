# APEXBANK DWDM PROJECT: DEMO & VIVA COMPREHENSIVE SCRIPT
**Project Title:** Customer Segmentation in Banking Using Clustering and Data Mining  
**Platform:** ApexBank | Customer Segmentation BI Platform  
**Target Dataset:** Default of Credit Card Clients (UCI Machine Learning Repository, ID: 350)  
**Author:** DWDM Project Team  
**Dataset DOI:** 10.24432/C55S3H | **Citation:** Yeh, I. (2009)  

---

## Quick Reference Summary Table for Viva
| Metric / Parameter | Value / Characteristic | Analytical Context |
| :--- | :--- | :--- |
| **Total Population** | 30,000 customers | Taiwan credit card cardholders (April–October 2005) |
| **Input Features Used** | 8 behavioral features | `LIMIT_BAL`, `AGE`, `Avg_Bill_Amt`, `Avg_Pay_Amt`, `Payment_to_Bill_Ratio`, `Avg_Repayment_Status`, `Max_Repayment_Delay`, `Delayed_Months_Count` |
| **Excluded Features** | `ID`, `Y` | `ID` is arbitrary; `Y` is the supervised default target (used only post-clustering) |
| **Optimal Clusters ($K$)** | $K = 4$ | Selected via Elbow inflection, Calinski-Harabasz separation, and business actionability |
| **Silhouette Score ($K=4$)** | 0.2443 (Benchmark sample: 0.25+) | Demonstrates dense clustering in continuous financial debt space |
| **Davies-Bouldin Index ($K=4$)** | 1.3970 | Measures cluster similarity (lower indicates better partition separation) |
| **Calinski-Harabasz Index ($K=4$)** | 7,596.6 | Ratio of between-cluster to within-cluster variance |
| **PCA 2D Variance Explained** | PC1 (35.44%) + PC2 (20.72%) = 56.16% | Captures dominant variance along spending activity and delay severity |
| **Warehouse Architecture** | Star Schema | 5 Dimensions (`DIM_CUSTOMER`, `DIM_CREDIT`, `DIM_PAYMENT_HISTORY`, `DIM_BILLING`, `DIM_PAYMENT`) + 1 Central Fact (`FACT_CUSTOMER_CREDIT_PROFILE`) |

---

## 20 Mandatory Viva Questions & Detailed Model Answers

### 1. Why did you choose this dataset?
**Answer:**  
"We selected the official **UCI 'Default of Credit Card Clients' dataset (Yeh, 2009; ID: 350)** because it provides an authentic, high-dimensional multi-month financial trajectory of 30,000 credit card account holders. Rather than single transaction logs or artificial demo data, it captures demographic variables, sanctioned credit facilities (`LIMIT_BAL`), a continuous 6-month billing history (`BILL_AMT1`–`BILL_AMT6`), repayment statuses (`PAY_0`–`PAY_6`), and actual payment remittances (`PAY_AMT1`–`PAY_AMT6`). This multi-dimensional financial behavior provides the necessary signal for complex unsupervised customer segmentation and Data Warehousing (DWDM) dimensional modeling."

---

### 2. What is customer segmentation?
**Answer:**  
"Customer segmentation is the process of partitioning a broad consumer or banking client base into mutually exclusive, homogeneous subgroups based on shared demographic, transactional, and behavioral characteristics. In retail banking, customer segmentation allows institutions to tailor credit limits, formulate targeted marketing campaigns, design customized financial products, identify credit delinquency early, and optimize customer relationship management (CRM) workflows."

---

### 3. Why clustering?
**Answer:**  
"Clustering is the primary unsupervised data mining technique suited for discovering intrinsic groupings within unlabelled data. Unlike supervised classification, where pre-assigned labels force patterns, clustering evaluates geometric distance metrics across multi-dimensional feature space to reveal natural behavioral structures, spending habits, and risk tiers without human bias or predetermined categorizations."

---

### 4. Why K-Means?
**Answer:**  
"K-Means was chosen as the primary clustering algorithm because of its mathematical efficiency, linear computational scalability ($\mathcal{O}(K \cdot N \cdot d)$), and fast convergence through Lloyd's iterative centroid relocation algorithm. In banking data warehousing environments dealing with tens of thousands to millions of customer records, alternative algorithms like Hierarchical Agglomerative Clustering exhibit $\mathcal{O}(N^3)$ computational time and $\mathcal{O}(N^2)$ memory bottlenecks, making K-Means the industry standard for production data marts and real-time customer profiling."

---

### 5. Why is this unsupervised learning?
**Answer:**  
"This project is unsupervised because the model does not receive any training target labels, output classes, or ground truth guidance. The objective is to let the algorithm autonomously group customers strictly based on geometric similarities in their demographic, credit, billing, and payment behavior. While the dataset contains a target column $Y$ (default payment next month), $Y$ was deliberately withheld from the feature matrix so that the segmentation remains purely behavioral and discovery-driven."

---

### 6. Why did you remove ID?
**Answer:**  
"The `ID` attribute is an arbitrary sequential surrogate key assigned by database indexers. It possesses zero behavioral variance, financial meaning, or predictive relationship to customer habits. Retaining `ID` would distort Euclidean distance calculations, causing the clustering algorithm to group records by arbitrary ingestion order rather than true financial characteristics."

---

### 7. Why did you exclude Y?
**Answer:**  
"`Y` is the binary target variable representing whether the customer defaulted in the subsequent month (October 2005). Including `Y` as an input feature would violate the fundamental premise of unsupervised customer segmentation by 'leaking' the future outcome into the clustering space. In our architecture, `Y` is strictly reserved for **post-clustering empirical validation**, allowing us to examine how natural behavioral segments align with historical delinquency without biasing cluster formation."

---

### 8. How did you choose K?
**Answer:**  
"We did not arbitrarily guess or hard-code $K$. We performed a systematic quantitative grid evaluation across $K \in [2, 10]$ evaluating four distinct cluster validity metrics simultaneously:
1. **Inertia / Within-Cluster Sum of Squares (WCSS)** for the Elbow Method.
2. **Silhouette Score** to measure cohesion versus separation.
3. **Davies-Bouldin Index** to assess inter-cluster distance versus intra-cluster dispersion.
4. **Calinski-Harabasz Index (Variance Ratio Criterion)**.
The analysis identified $K=4$ as the optimal trade-off: it marks the primary inflection on the WCSS curve, maintains robust silhouette separation, achieves high Calinski-Harabasz variance ratio, and yields four strategically distinct, actionable banking customer segments."

---

### 9. What is the Elbow Method?
**Answer:**  
"The Elbow Method plots the Within-Cluster Sum of Squares (WCSS or Inertia) against the number of clusters $K$. Inertia measures total squared Euclidean distance from each point to its assigned cluster centroid:
$$\text{Inertia} = \sum_{j=1}^{K}\sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$
As $K$ increases, inertia naturally decreases. The 'elbow' is the point of diminishing marginal returns—where adding another cluster does not significantly reduce inertia. In our dataset, the sharpest reduction decelerates around $K=4$."

---

### 10. What is Silhouette Score?
**Answer:**  
"The Silhouette Score evaluates how well-separated and cohesive clusters are, ranging from $-1.0$ (misclassified) to $+1.0$ (perfectly isolated):
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
where $a(i)$ is the mean intra-cluster distance to all points in the same cluster, and $b(i)$ is the mean nearest-cluster distance. A positive silhouette score (0.2443 to 0.2868 in our multi-dimensional financial feature space) confirms that customers are significantly closer to their assigned centroid than to neighboring clusters."

---

### 11. What is Davies-Bouldin Index?
**Answer:**  
"The Davies-Bouldin Index (DBI) measures the average similarity between each cluster and its most similar counterpart:
$$R_{ij} = \frac{s_i + s_j}{d(\mu_i, \mu_j)}, \quad DB = \frac{1}{K}\sum_{i=1}^{K} \max_{j \neq i} R_{ij}$$
where $s_i$ is intra-cluster dispersion and $d(\mu_i, \mu_j)$ is distance between centroids. **Lower values indicate better clustering**, signifying compact clusters that are widely separated from each other. Our $K=4$ configuration achieved a favorable DBI of 1.3970."

---

### 12. What is Calinski-Harabasz Index?
**Answer:**  
"Also known as the Variance Ratio Criterion, the Calinski-Harabasz (CH) Index evaluates the ratio of between-cluster dispersion (separation) to within-cluster dispersion (compactness):
$$CH = \frac{\text{Tr}(B_K) / (K - 1)}{\text{Tr}(W_K) / (N - K)}$$
**Higher values denote superior clustering quality**. In our evaluation, $K=4$ produced a very high CH score of 7,596.6, proving high statistical separation between the discovered customer segments."

---

### 13. Why isn't accuracy used?
**Answer:**  
"Classification Accuracy, Precision, Recall, and F1-score require ground-truth target labels ($y$) to compare predictions against actual classes:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
Because customer segmentation is an **unsupervised exploratory task**, there are no predefined 'correct' segment labels. Evaluating an unsupervised clustering algorithm with classification accuracy is a fundamental statistical error. Instead, cluster validity must be evaluated using internal geometric metrics (Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz)."

---

### 14. What preprocessing was performed?
**Answer:**  
"Our ETL and preprocessing pipeline followed a structured sequence:
1. **Ingestion & Integrity Check:** Validated the 30,000 raw rows and 25 features against official UCI specifications; verified zero nulls and zero duplicate rows.
2. **Categorical Harmonization:** Cleaned undocumented category codes in `EDUCATION` (0, 5, 6 mapped to 4 'Others') and `MARRIAGE` (0 mapped to 3 'Others').
3. **Feature Pruning:** Removed `ID` and isolated `Y`.
4. **Outlier Inspection:** Analyzed distributions using interquartile ranges (IQR); genuine high-wealth cardholders were preserved rather than trimmed to reflect the real banking credit distribution.
5. **Feature Scaling:** Applied z-score normalization (`StandardScaler`)."

---

### 15. Why did you scale the features?
**Answer:**  
"K-Means computes Euclidean distance:
$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
In our raw dataset, `LIMIT_BAL` ranges up to NT$ 1,000,000, while `Delayed_Months_Count` ranges from 0 to 6, and `Payment_to_Bill_Ratio` from 0 to 10. Without standardization, features with massive numerical scales like `LIMIT_BAL` would overpower distance calculations by several orders of magnitude, rendering smaller but critical behavioral features mathematically invisible. `StandardScaler` standardizes all features to $\mu = 0, \sigma = 1$, ensuring equal weight."

---

### 16. What does each cluster represent?
**Answer:**  
"Based on empirical centroid statistics for $K=4$, the segments represent:
1. **Cluster 0 — High Credit / Advance & Prompt Payers (24.2% of base):**  
   High average credit limit (NT$ 262,607), negative average repayment status ($-0.82$, indicating prepayment or full zero-balance clearance), low post-hoc default rate (14.69%).
2. **Cluster 1 — Moderate Credit / Stable Payment Pattern (47.4% of base):**  
   The core retail banking cohort; moderate credit limits (NT$ 107,634), regular revolving credit usage with minimal delays ($0.45$ months), steady low-default rate (16.86%).
3. **Cluster 2 — Lower Credit / High Repayment Delay Pattern (16.3% of base):**  
   Lower credit limits (NT$ 88,149), chronic delinquency averaging $2.41$ months of maximum delay, and a high post-hoc empirical default rate of **55.67%**.
4. **Cluster 3 — Higher Credit / Higher Payment Activity (12.1% of base):**  
   Premium affluents; highest sanctioned credit line (NT$ 318,679), large monthly payments (NT$ 17,727), lowest empirical default rate (12.29%)."

---

### 17. What is ETL?
**Answer:**  
"ETL stands for **Extract, Transform, and Load**, the foundation of data warehousing:
- **Extract:** Ingesting raw, heterogeneous data from operational sources (staging the UCI dataset).
- **Transform:** Validating types, standardizing undocumented categorical codes, engineering analytical aggregates (`Avg_Bill_Amt`, `Payment_to_Bill_Ratio`, `Max_Repayment_Delay`), and standardizing scales.
- **Load:** Writing clean, structured records into dimensional tables (`DIM_CUSTOMER`, `DIM_CREDIT`, `DIM_PAYMENT_HISTORY`, `DIM_BILLING`, `DIM_PAYMENT`) and the central `FACT_CUSTOMER_CREDIT_PROFILE`."

---

### 18. What is the DWDM component?
**Answer:**  
"DWDM unites **Data Warehousing** and **Data Mining**:
- **Data Warehousing:** Organizes atomic customer and billing records into a subject-oriented, integrated, non-volatile, and time-variant Star Schema optimized for Online Analytical Processing (OLAP) slicing, dicing, and aggregations.
- **Data Mining:** Applies unsupervised machine learning algorithms (K-Means, PCA) directly on the curated analytical data mart (`Agg_Customer_Segment_Mart`) to extract non-trivial, previously unknown behavioral patterns and segment profiles."

---

### 19. What is the data warehouse design?
**Answer:**  
"We designed a dimensional **Star Schema** specifically tailored to the credit card domain:
- **`DIM_CUSTOMER`:** Demographics (`Customer_Key`, `Age`, `Gender`, `Education`, `Marital_Status`).
- **`DIM_CREDIT`:** Facilities (`Credit_Key`, `Credit_Limit`, `Credit_Tier`).
- **`DIM_PAYMENT_HISTORY`:** Repayment tracker across months (`PAY_0` to `PAY_6`, `Max_Delay_Months`).
- **`DIM_BILLING`:** 6-month statement bills (`BILL_AMT1` to `BILL_AMT6`, `Avg_Bill_Amount`).
- **`DIM_PAYMENT`:** Remittances (`PAY_AMT1` to `PAY_AMT6`, `Avg_Payment_Amount`).
- **`FACT_CUSTOMER_CREDIT_PROFILE`:** Central fact recording additive metrics, utilization ratios, and payment-to-bill metrics at the customer grain.
- **`Agg_Customer_Segment_Mart`:** Pre-aggregated OLAP mart enriched with cluster labels and 2D PCA coordinates."

---

### 20. What are the limitations?
**Answer:**  
"The project has several documented academic and practical limitations:
1. **Historical & Geographic Specificity:** Data reflects Taiwanese credit card holders from 2005; macroeconomic conditions and consumer behaviors may differ in contemporary banking markets.
2. **Product Domain Scope:** The dataset covers unsecured revolving credit cards only; it does not incorporate checking/savings deposits, mortgages, or commercial lending lines.
3. **Correlation vs. Causation:** Clustering identifies geometric groupings and correlational associations; it does not prove that belonging to a cluster causes financial default.
4. **Feature & Scaling Sensitivity:** K-Means results inherently depend on the chosen feature subset, distance metric (Euclidean), and standardization technique.
5. **Static Snapshot:** The 6-month window is analyzed as an aggregated static profile rather than a continuous real-time streaming time-series."
