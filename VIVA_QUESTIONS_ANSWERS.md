# DWDM Viva Examination Preparation Guide

## Customer Segmentation in Banking Using Clustering and Data Mining
**Subject**: Data Warehousing and Data Mining (DWDM)  
**Comprehensive Viva Q&A (25 Questions & Detailed Answers)**

---

### Q1. What is Customer Segmentation, and why is it important in banking?
**Answer**:
Customer segmentation is the data mining process of partitioning a heterogeneous bank customer base into distinct, mutually exclusive, and internally homogeneous subgroups based on shared demographic, transactional, and financial behaviors. In retail banking, customer segmentation is essential because:
1. Customers have widely varying liquidity levels, risk appetites, and transaction frequencies.
2. It allows banks to transition from generic mass marketing to personalized cross-selling (e.g. wealth advisory for high-balance clients vs. micro-loans for young transactors).
3. It optimizes credit card limits and minimizes customer churn.

---

### Q2. What is the fundamental difference between Data Warehousing (DW) and Online Transaction Processing (OLTP)?
**Answer**:
* **OLTP (Online Transaction Processing)**:
  - Designed for real-time transaction execution (e.g. ATM withdrawals, fund transfers).
  - Highly normalized schema (3NF) to eliminate data redundancy and maximize insert/update throughput.
  - Queries are simple, frequent, and touch a few rows.
* **Data Warehouse (DW / OLAP)**:
  - Designed for analytical reporting, aggregation, and data mining.
  - De-normalized dimensional schema (Star / Snowflake Schema) optimized for complex read-only analytical queries across millions of historical records.
  - Subject-oriented, integrated, time-variant, and non-volatile (Inmon definition).

---

### Q3. Explain the Star Schema design used in this project.
**Answer**:
Our data warehouse uses a Star Schema consisting of a central fact table surrounded by three conformed dimension tables:
1. **`Fact_Transaction` (Central Fact Table)**:
   - Stores quantitative, additive business measurements: `TransactionAmount`, `AccountBalance`, and `TransactionHour`.
   - Contains foreign keys: `CustomerID`, `Date_ID`, and `Location_ID`.
   - Granularity: One record per individual transaction.
2. **`Dim_Customer`**: Contains demographic context (`CustomerID`, `Age`, `Gender`, `Location`).
3. **`Dim_Date`**: Temporal dimension storing calendar attributes (`Date_ID`, `FullDate`, `Day`, `Month`, `Quarter`, `Year`, `IsWeekend`).
4. **`Dim_Location`**: Geographic dimension storing normalized city names and tier classifications (`Tier-1 Metro`, etc.).
5. **`Agg_Customer_Mart`**: Pre-aggregated customer-level analytical summary table derived from the Star Schema to directly feed the clustering algorithms.

---

### Q4. What is a Fact Table, and what is a Dimension Table?
**Answer**:
* **Fact Table**: A table in a dimensional warehouse schema that contains numerical, additive measurements (facts) and foreign keys referencing dimension tables. Fact tables represent business events or snapshots. In our project, `Fact_Transaction` records each transaction event with its monetary value and balance snapshot.
* **Dimension Table**: A table containing descriptive, contextual text attributes used for filtering, grouping, slicing, and dicing the facts. In our project, `Dim_Customer` provides who made the transaction, `Dim_Date` provides when, and `Dim_Location` provides where.

---

### Q5. What is ETL, and what specific ETL challenges were present in this banking dataset?
**Answer**:
ETL stands for **Extract, Transform, and Load**:
* **Extract**: Reading raw transaction logs from source files or staging databases.
* **Transform**: Data cleaning, validation, type conversion, and feature computation.
* **Load**: Writing cleaned and structured tables into the target warehouse schema or data mart.

**Key ETL challenges handled in this dataset**:
1. **1800 DOB Placeholder**: 57,339 records had `CustomerDOB` recorded with year `1800` (e.g. `1/1/1800`). This is a legacy default placeholder in banking systems when a customer's birthdate is missing. We converted this to `NaN` and imputed with the empirical median age (29.0).
2. **Two-Digit Year Parsing**: DOBs with two-digit years (e.g. `'94'` or `'57'`) had to be calibrated relative to the 2016 transaction year.
3. **Integer Time Format**: Raw `TransactionTime` was stored as integer `HHMMSS` (e.g., `143207`), which was transformed into standard ISO string `'14:32:07'` and integer hour (14).
4. **Extreme Skewness**: Account balances had a raw skewness of +53.98, requiring logarithmic transformation.

---

### Q6. Why can't we use raw transaction records directly for customer segmentation? Why is aggregation necessary?
**Answer**:
Raw transaction records exist at the **transaction level**, meaning a single customer who executed 4 transactions appears as 4 separate rows with different dates, times, and amounts. Clustering directly on raw transactions would group individual transaction events (e.g. "small morning coffee purchases") rather than profiling the customer who made them.

Aggregation groups all transactions by `CustomerID` and derives customer-level behavioral metrics: cumulative spend, average ticket size, transaction count, active days, and latest account balance. This creates a unified analytical profile for each customer.

---

### Q7. Why must CustomerID and TransactionID be excluded from the clustering features?
**Answer**:
`CustomerID` (e.g. `C5841053`) and `TransactionID` (e.g. `T1`) are arbitrary alphanumeric identifiers and surrogate primary keys. They have no natural numerical ordering, distance metric, or behavioral meaning. If passed into a distance-based algorithm like K-Means, the algorithm would compute Euclidean distances on arbitrary character or integer ID codes, creating completely meaningless clusters and introducing severe noise.

---

### Q8. What is Unsupervised Learning, and how does it differ from Supervised Learning?
**Answer**:
* **Supervised Learning**: The algorithm is provided with labeled training data $(X, y)$, where each input sample has a known target label (e.g. predicting whether a customer will default: 1 or 0). The objective is to learn a mapping function $f(X) \rightarrow y$.
* **Unsupervised Learning**: The algorithm is provided only with unlabeled input data $X$ without any predefined target or ground-truth outcome. The algorithm must autonomously discover underlying hidden patterns, groupings, or geometric distributions. Clustering is a core unsupervised learning technique.

---

### Q9. What is K-Means clustering, and how does it work mathematically?
**Answer**:
K-Means is an unsupervised, centroid-based partitioning algorithm that divides $N$ data points into $K$ non-overlapping clusters. It aims to minimize the Within-Cluster Sum of Squares (WCSS / Inertia):

$$\mathcal{J} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$

**Algorithm Steps**:
1. **Initialization**: Randomly initialize $K$ cluster centroids $\mu_1, \dots, \mu_K$ in feature space (or using K-Means++).
2. **Assignment Step**: Assign each data point $x_i$ to its nearest centroid based on Euclidean distance:
   $$c_i = \arg\min_{k} \| x_i - \mu_k \|^2$$
3. **Update Step**: Recalculate each centroid as the arithmetic mean of all points assigned to that cluster:
   $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$
4. **Iterate**: Repeat steps 2 and 3 until centroids converge (no reassignment occurs).

---

### Q10. What is a Centroid?
**Answer**:
A centroid is the geometric center or arithmetic mean vector of all data points belonging to a specific cluster:
$$\mu_k = \left( \frac{1}{N_k} \sum_{i=1}^{N_k} x_{i,1}, \, \frac{1}{N_k} \sum_{i=1}^{N_k} x_{i,2}, \, \dots, \, \frac{1}{N_k} \sum_{i=1}^{N_k} x_{i,d} \right)$$
In our project, the centroid coordinates represent the average standardized values of Age, Account Balance, Transaction Count, and Transaction Amount for that customer segment.

---

### Q11. What is Inertia in K-Means?
**Answer**:
Inertia (also known as Within-Cluster Sum of Squares or WCSS) is the sum of squared Euclidean distances between each data point and its assigned cluster centroid:
$$\text{Inertia} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$
Inertia measures internal cluster compactness. A lower inertia indicates that data points are clustered more tightly around their respective centroids. Note that inertia always decreases as $K$ increases, reaching 0 when $K = N$.

---

### Q12. Explain the Elbow Method. How did you choose the optimal value of K?
**Answer**:
The Elbow Method is a heuristic used to determine the optimal number of clusters $K$. We run K-Means across a range of $K$ values (in our project, $K \in [2, 8]$) and plot $K$ on the x-axis against Inertia on the y-axis.

As $K$ increases, inertia drops rapidly at first and then levels off, forming an "elbow" shape on the curve. The point where the rate of decrease abruptly shifts (diminishing marginal returns) represents the optimal trade-off between model simplicity and cluster compactness.

In our project:
* $K=2$: Inertia = 145,042.6
* $K=3$: Inertia = 114,499.5 ($\Delta = -30,543.1$)
* **$K=4$**: Inertia = **97,628.3** ($\Delta = -16,871.2$) $\rightarrow$ **Distinct Elbow Point**
* $K=5$: Inertia = 84,759.5 ($\Delta = -12,868.8$)
* $K=6$: Inertia = 73,958.5
Beyond $K=4$, the decrease in inertia slows substantially, indicating that $K=4$ is the optimal elbow point.

---

### Q13. What is the Silhouette Score, and how is it interpreted?
**Answer**:
The Silhouette Score evaluates clustering quality by measuring both **cluster cohesion** (how close a point is to points in its own cluster) and **cluster separation** (how far it is from points in neighboring clusters).

For each sample point $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
where:
* $a(i)$ = mean intra-cluster distance between sample $i$ and all other points in the same cluster.
* $b(i)$ = mean nearest-cluster distance between sample $i$ and the nearest neighboring cluster.

**Score Interpretation**:
* **+1.0**: Point is very well clustered, far from neighboring clusters.
* **0.0**: Point lies on the decision boundary between two clusters.
* **-1.0**: Point has likely been assigned to the wrong cluster.

In our project, $K=4$ achieved a solid silhouette score of **0.2878**, confirming well-separated clusters with minimal sample overlap.

---

### Q14. Why is feature scaling mandatory before running K-Means? What happens if you skip it?
**Answer**:
K-Means calculates distances using Euclidean distance:
$$d(x, y) = \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2}$$
If features are not scaled, variables with large absolute magnitudes and numerical ranges (such as `Account_Balance` reaching ₹1,000,000+) will completely dominate variables with small ranges (such as `Transaction_Count` ranging from 1 to 6, or `Age` from 18 to 80). Without scaling, a difference of 100 rupees in balance would outweigh a 5x difference in transaction frequency.

Standardization (using `StandardScaler`) transforms every feature to have zero mean ($\mu = 0$) and unit variance ($\sigma = 1$), ensuring that all features contribute equally to Euclidean distance calculations.

---

### Q15. Why did you apply log-transformation (log1p) before standardization?
**Answer**:
Financial features in retail banking exhibit extreme right-skewness (power-law distribution). In our raw dataset, `CustAccountBalance` had a skewness of **+53.98** and `TransactionAmount` had a skewness of **+31.34**.

Because standard $Z$-score scaling assumes approximately symmetrical distributions, extreme positive outliers would compress 99% of normal customers into a tiny region near zero, while pulling centroids towards extreme millionaires. Applying $y = \ln(1 + x)$ stabilized the variance, reduced balance skewness to **-0.96**, and normalized transaction amount skewness to **-0.15** (near Gaussian), resulting in balanced, meaningful Euclidean clusters.

---

### Q16. Why did you use Principal Component Analysis (PCA)? Does K-Means run on PCA or original features?
**Answer**:
* **How we used it**: K-Means clustering was executed on the full 5-dimensional standardized feature space (`Age`, `Account_Balance`, `Transaction_Count`, `Total_Transaction_Amount`, `Avg_Transaction_Amount`) so that no behavioral information was discarded during model training.
* **Why PCA was used**: PCA was applied *after* clustering purely as a dimensionality reduction technique for **2D visual projection**. Human eyes cannot visualize 5-dimensional space. PCA computes the top 2 orthogonal principal components capturing **66.39% of total variance** ($\text{PC1} = 44.79\%$, $\text{PC2} = 21.60\%$), allowing us to plot the high-dimensional clusters on a 2D scatter plot.

---

### Q17. How does K-Means compare with Hierarchical Agglomerative Clustering?
**Answer**:
We benchmarked both algorithms on an identical sample ($N = 2,500$):
1. **Silhouette Score**: K-Means achieved **0.2970**, outperforming Hierarchical Ward linkage (**0.2197**).
2. **Time Complexity**:
   * K-Means has linear time complexity $\mathcal{O}(K \cdot N \cdot d \cdot i)$, making it scalable to millions of banking transactions.
   * Hierarchical clustering has cubic time complexity $\mathcal{O}(N^3)$ and quadratic memory complexity $\mathcal{O}(N^2)$ to store the distance matrix. Running Hierarchical clustering on all 882,600 customers would require hundreds of gigabytes of RAM and would crash standard computing hardware.
3. **Conclusion**: K-Means is the production-ready algorithm for large-scale banking data warehouses.

---

### Q18. Describe the 4 customer clusters discovered in your actual dataset.
**Answer**:
Based on our empirical analysis of 882,600 customers:
1. **Cluster 0: Young Low-Balance Micro-Transactors** (35.48% of base):
   - Mean Age: 27.7 yrs | Mean Balance: ₹35,784 | Mean Ticket Size: ₹154.51 | Tx Count: 1.01
   - Entry-level youth executing frequent micro-payments.
2. **Cluster 1: Young Mid-Balance Moderate Transactors** (39.06% of base - Largest Segment):
   - Mean Age: 29.4 yrs | Mean Balance: ₹123,938 | Mean Ticket Size: ₹2,507 | Tx Count: 1.00
   - Stable salaried professionals making moderate single-ticket transactions.
3. **Cluster 2: Mature High-Wealth / High-Value Transactors** (10.43% of base):
   - Mean Age: 49.4 yrs | Mean Balance: ₹352,183 | Mean Ticket Size: ₹2,937 | Tx Count: 1.07
   - Older affluent segment holding the highest average account balances in the bank.
4. **Cluster 3: High-Frequency Active Digital Transactors** (15.03% of base):
   - Mean Age: 29.9 yrs | Mean Balance: ₹116,321 | Mean Ticket Size: ₹1,560 | Mean Tx Count: 2.15 (Active Days: 19.5)
   - Multi-transacting digital power users.

---

### Q19. What business strategies would you propose for the Mature High-Wealth group (Cluster 2)?
**Answer**:
* Assign dedicated Private Wealth relationship managers.
* Offer capital preservation and tax-saving products: Fixed Deposits (FDs), Public Provident Funds (PPF), and sovereign gold bonds.
* Cross-sell premium travel credit cards offering complimentary airport lounge access, golf privileges, and high insurance coverage.

---

### Q20. What business strategies would you propose for the High-Frequency Digital group (Cluster 3)?
**Answer**:
* Introduce merchant UPI cashback promotions and digital wallet partnerships.
* Pre-approve instant personal lines of credit and consumer durable zero-cost EMI loans.
* Implement mobile banking app gamification and tiered reward points for frequent digital payments.

---

### Q21. Do the cluster personas prove customer profitability or future creditworthiness?
**Answer**:
**No**. In rigorous academic data mining, it is critical to state that K-Means is an unsupervised, descriptive algorithm that identifies geometric groupings based solely on observed historical data. The personas are human qualitative interpretations of empirical statistics. They do not prove future profitability, solvency, or credit default risk. To predict credit default, a supervised classification model (e.g. Logistic Regression, Random Forest, XGBoost) trained on historical default labels would be required.

---

### Q22. What are the main limitations of this project?
**Answer**:
1. **Short Observation Horizon**: The dataset covers an 82-day window (August–October 2016). It does not capture annual seasonality, Diwali festive spending, or year-end bonuses.
2. **Single Transaction Sparsity**: 83.8% of customers executed only one transaction during this window, limiting granular time-series recency modeling.
3. **K-Means Assumption**: K-Means assumes spherical clusters of approximately equal variance, which may oversimplify complex non-linear financial patterns.

---

### Q23. How could this project be improved or expanded in the future?
**Answer**:
1. **Big Data Processing**: Migrate the ETL and clustering pipeline to Apache Spark (PySpark) to ingest and process streaming transaction logs in real time.
2. **RFM Modeling**: Incorporate customer recency scoring once multi-year longitudinal transaction datasets become available.
3. **Supervised Churn Prediction**: Use the discovered cluster IDs as categorical features in a supervised machine learning model to predict customer attrition.
4. **Advanced Clustering**: Test Gaussian Mixture Models (GMM) with Expectation-Maximization to evaluate soft/probabilistic cluster memberships.

---

### Q24. How did you ensure your preprocessing pipeline prevents data leakage?
**Answer**:
Data transformation parameters (such as the median age used for imputation and the mean $\mu$ and standard deviation $\sigma$ fitted by `StandardScaler`) were computed systematically on training matrices and serialized into `outputs/models/scaler.joblib`. During real-time inference in our interactive Streamlit classifier, new customer inputs are transformed using the pre-fitted scaler parameters rather than re-computing statistics on the single input row.

---

### Q25. How do you demonstrate the complete project during your practical evaluation?
**Answer**:
1. **SQL Warehouse**: Walk through `sql/create_tables.sql` showing the Star Schema (`Fact_Transaction`, `Dim_Customer`, `Dim_Date`, `Dim_Location`).
2. **ETL Execution**: Show `src/data_cleaning.py` resolving the 1800-year DOB placeholder and formatting times.
3. **Model Selection**: Present the Elbow curve (`elbow_plot.png`) and Silhouette scores (`silhouette_scores.png`) justifying $K=4$.
4. **2D PCA Visualization**: Present `pca_clusters_2d.png` demonstrating cluster separation.
5. **Interactive Dashboard**: Run `streamlit run dashboard/app.py` to showcase the live KPIs, cluster profiles, and test the interactive Customer Persona Classifier in real time.
