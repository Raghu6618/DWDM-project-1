-- ============================================================================
-- APEXBANK DWDM PROJECT: DATA WAREHOUSE SCHEMA DDL
-- Project: Customer Segmentation in Banking Using Clustering and Data Mining
-- Dataset: Default of Credit Card Clients (UCI Machine Learning Repository, ID: 350)
-- Citation: Yeh, I. (2009). DOI: 10.24432/C55S3H
-- 
-- IMPORTANT ARCHITECTURAL NOTE:
-- This is a project-designed analytical dimensional warehouse model (Star Schema)
-- derived from the source dataset for OLAP and data mining. The original UCI dataset
-- is supplied as a single flat table; this schema represents the dimensional warehouse
-- architecture constructed for analytical profiling and customer segmentation.
-- ============================================================================

-- Drop existing tables if re-initializing schema
DROP TABLE IF EXISTS Fact_Customer_Credit_Profile;
DROP TABLE IF EXISTS Dim_Payment;
DROP TABLE IF EXISTS Dim_Billing;
DROP TABLE IF EXISTS Dim_Payment_History;
DROP TABLE IF EXISTS Dim_Credit;
DROP TABLE IF EXISTS Dim_Customer;
DROP TABLE IF EXISTS Agg_Customer_Segment_Mart;

-- ----------------------------------------------------------------------------
-- 1. DIMENSION TABLE: Dim_Customer
-- Demographic profile of credit card account holders.
-- Granularity: 1 record per credit customer.
-- ----------------------------------------------------------------------------
CREATE TABLE Dim_Customer (
    Customer_Key INTEGER PRIMARY KEY,           -- Surrogate Key (mapped from ID)
    Customer_ID INTEGER NOT NULL UNIQUE,        -- Natural Key from UCI Dataset
    Age INTEGER NOT NULL,                       -- Age in years
    Gender_Code INTEGER NOT NULL,               -- 1: Male, 2: Female
    Gender_Desc VARCHAR(10) NOT NULL,           -- 'Male', 'Female'
    Education_Code INTEGER NOT NULL,            -- 1: Grad School, 2: University, 3: High School, 4: Others
    Education_Desc VARCHAR(30) NOT NULL,        -- 'Graduate School', 'University', 'High School', 'Others'
    Marital_Status_Code INTEGER NOT NULL,       -- 1: Married, 2: Single, 3: Others
    Marital_Status_Desc VARCHAR(20) NOT NULL,   -- 'Married', 'Single', 'Others'
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dim_cust_key ON Dim_Customer(Customer_Key);
CREATE INDEX idx_dim_cust_nat_id ON Dim_Customer(Customer_ID);


-- ----------------------------------------------------------------------------
-- 2. DIMENSION TABLE: Dim_Credit
-- Credit limits and account facility classifications.
-- Granularity: 1 record per unique credit arrangement.
-- ----------------------------------------------------------------------------
CREATE TABLE Dim_Credit (
    Credit_Key INTEGER PRIMARY KEY,             -- Surrogate Key
    Customer_ID INTEGER NOT NULL UNIQUE,        -- Foreign key link to customer
    Credit_Limit NUMERIC(12, 2) NOT NULL,       -- LIMIT_BAL in NT Dollars
    Credit_Tier VARCHAR(20) NOT NULL,           -- 'Standard (<50k)', 'Silver (50k-150k)', 'Gold (150k-300k)', 'Platinum (>300k)'
    Utilization_Capacity_Group VARCHAR(25) NOT NULL,
    FOREIGN KEY (Customer_ID) REFERENCES Dim_Customer(Customer_ID)
);

CREATE INDEX idx_dim_credit_id ON Dim_Credit(Customer_ID);


-- ----------------------------------------------------------------------------
-- 3. DIMENSION TABLE: Dim_Payment_History
-- Historical monthly repayment status tracker (April - September 2005).
-- Repayment scale: -2=No consumption, -1=Paid in full, 0=Revolving credit, 1..9=Delay in months.
-- ----------------------------------------------------------------------------
CREATE TABLE Dim_Payment_History (
    Payment_History_Key INTEGER PRIMARY KEY,   -- Surrogate Key
    Customer_ID INTEGER NOT NULL UNIQUE,       -- Natural Key link
    PAY_0 INTEGER NOT NULL,                    -- Repayment status Sep 2005
    PAY_2 INTEGER NOT NULL,                    -- Repayment status Aug 2005
    PAY_3 INTEGER NOT NULL,                    -- Repayment status Jul 2005
    PAY_4 INTEGER NOT NULL,                    -- Repayment status Jun 2005
    PAY_5 INTEGER NOT NULL,                    -- Repayment status May 2005
    PAY_6 INTEGER NOT NULL,                    -- Repayment status Apr 2005
    Max_Delay_Months INTEGER NOT NULL,         -- Maximum observed delay across 6 months
    Total_Delayed_Months INTEGER NOT NULL,     -- Count of months with delay > 0
    Consistently_On_Time BOOLEAN NOT NULL,     -- True if all months <= 0
    FOREIGN KEY (Customer_ID) REFERENCES Dim_Customer(Customer_ID)
);

CREATE INDEX idx_dim_pay_hist_id ON Dim_Payment_History(Customer_ID);


-- ----------------------------------------------------------------------------
-- 4. DIMENSION TABLE: Dim_Billing
-- Six-month billing statement trajectory (NT Dollars).
-- ----------------------------------------------------------------------------
CREATE TABLE Dim_Billing (
    Billing_Key INTEGER PRIMARY KEY,           -- Surrogate Key
    Customer_ID INTEGER NOT NULL UNIQUE,       -- Natural Key link
    BILL_AMT1 NUMERIC(12, 2) NOT NULL,         -- Bill Statement Sep 2005
    BILL_AMT2 NUMERIC(12, 2) NOT NULL,         -- Bill Statement Aug 2005
    BILL_AMT3 NUMERIC(12, 2) NOT NULL,         -- Bill Statement Jul 2005
    BILL_AMT4 NUMERIC(12, 2) NOT NULL,         -- Bill Statement Jun 2005
    BILL_AMT5 NUMERIC(12, 2) NOT NULL,         -- Bill Statement May 2005
    BILL_AMT6 NUMERIC(12, 2) NOT NULL,         -- Bill Statement Apr 2005
    Total_Bill_Amount NUMERIC(14, 2) NOT NULL, -- Sum of 6-month billing
    Avg_Bill_Amount NUMERIC(12, 2) NOT NULL,   -- Mean monthly bill
    Bill_Amount_Variance NUMERIC(14, 2),       -- Standard deviation / variance of bills
    FOREIGN KEY (Customer_ID) REFERENCES Dim_Customer(Customer_ID)
);

CREATE INDEX idx_dim_billing_id ON Dim_Billing(Customer_ID);


-- ----------------------------------------------------------------------------
-- 5. DIMENSION TABLE: Dim_Payment
-- Six-month remittance and previous payment records (NT Dollars).
-- ----------------------------------------------------------------------------
CREATE TABLE Dim_Payment (
    Payment_Key INTEGER PRIMARY KEY,           -- Surrogate Key
    Customer_ID INTEGER NOT NULL UNIQUE,       -- Natural Key link
    PAY_AMT1 NUMERIC(12, 2) NOT NULL,          -- Paid Sep 2005
    PAY_AMT2 NUMERIC(12, 2) NOT NULL,          -- Paid Aug 2005
    PAY_AMT3 NUMERIC(12, 2) NOT NULL,          -- Paid Jul 2005
    PAY_AMT4 NUMERIC(12, 2) NOT NULL,          -- Paid Jun 2005
    PAY_AMT5 NUMERIC(12, 2) NOT NULL,          -- Paid May 2005
    PAY_AMT6 NUMERIC(12, 2) NOT NULL,          -- Paid Apr 2005
    Total_Payment_Amount NUMERIC(14, 2) NOT NULL, -- Sum of payments
    Avg_Payment_Amount NUMERIC(12, 2) NOT NULL,   -- Mean monthly payment
    Payment_Variance NUMERIC(14, 2),              -- Std/variance of payments
    FOREIGN KEY (Customer_ID) REFERENCES Dim_Customer(Customer_ID)
);

CREATE INDEX idx_dim_payment_id ON Dim_Payment(Customer_ID);


-- ----------------------------------------------------------------------------
-- 6. FACT TABLE: Fact_Customer_Credit_Profile
-- Central Fact Table joining all dimensions at the customer grain.
-- Contains additive and semi-additive financial metrics and behavioral ratios.
-- ----------------------------------------------------------------------------
CREATE TABLE Fact_Customer_Credit_Profile (
    Profile_Fact_Key INTEGER PRIMARY KEY,      -- Surrogate Fact Key
    Customer_Key INTEGER NOT NULL,             -- FK to Dim_Customer
    Credit_Key INTEGER NOT NULL,               -- FK to Dim_Credit
    Payment_History_Key INTEGER NOT NULL,      -- FK to Dim_Payment_History
    Billing_Key INTEGER NOT NULL,              -- FK to Dim_Billing
    Payment_Key INTEGER NOT NULL,              -- FK to Dim_Payment
    
    -- Numerical Facts & Measures
    Credit_Limit NUMERIC(12, 2) NOT NULL,
    Total_Billed_NTD NUMERIC(14, 2) NOT NULL,
    Total_Paid_NTD NUMERIC(14, 2) NOT NULL,
    Average_Monthly_Bill NUMERIC(12, 2) NOT NULL,
    Average_Monthly_Payment NUMERIC(12, 2) NOT NULL,
    Payment_to_Bill_Ratio NUMERIC(8, 4) NOT NULL,
    Credit_Utilization_Rate NUMERIC(8, 4) NOT NULL, -- Total Billed / (Credit Limit * 6)
    
    -- Delay Measures
    Max_Consecutive_Delay INTEGER NOT NULL,
    Total_Delay_Incidents INTEGER NOT NULL,
    
    -- Target Variable (Retained exclusively for post-mining validation; NOT for clustering)
    Target_Default_Next_Month INTEGER NOT NULL CHECK (Target_Default_Next_Month IN (0, 1)),
    
    FOREIGN KEY (Customer_Key) REFERENCES Dim_Customer(Customer_Key),
    FOREIGN KEY (Credit_Key) REFERENCES Dim_Credit(Credit_Key),
    FOREIGN KEY (Payment_History_Key) REFERENCES Dim_Payment_History(Payment_History_Key),
    FOREIGN KEY (Billing_Key) REFERENCES Dim_Billing(Billing_Key),
    FOREIGN KEY (Payment_Key) REFERENCES Dim_Payment(Payment_Key)
);

CREATE INDEX idx_fact_cust_key ON Fact_Customer_Credit_Profile(Customer_Key);
CREATE INDEX idx_fact_credit_key ON Fact_Customer_Credit_Profile(Credit_Key);


-- ----------------------------------------------------------------------------
-- 7. OLAP DATA MART: Agg_Customer_Segment_Mart
-- Pre-aggregated analytical table enriched with K-Means cluster assignments
-- and 2D PCA coordinates for high-speed BI reporting.
-- ----------------------------------------------------------------------------
CREATE TABLE Agg_Customer_Segment_Mart (
    Segment_Mart_Key INTEGER PRIMARY KEY,
    Customer_ID INTEGER NOT NULL UNIQUE,
    Credit_Limit NUMERIC(12, 2) NOT NULL,
    Age INTEGER NOT NULL,
    Gender_Desc VARCHAR(10) NOT NULL,
    Education_Desc VARCHAR(30) NOT NULL,
    Marital_Status_Desc VARCHAR(20) NOT NULL,
    Average_Bill_NTD NUMERIC(12, 2) NOT NULL,
    Average_Payment_NTD NUMERIC(12, 2) NOT NULL,
    Payment_to_Bill_Ratio NUMERIC(8, 4) NOT NULL,
    Max_Repayment_Delay INTEGER NOT NULL,
    Delayed_Months_Count INTEGER NOT NULL,
    Cluster_ID INTEGER NOT NULL,
    Cluster_Labe
    l VARCHAR(60) NOT NULL,
    PCA1_Coordinate NUMERIC(10, 6) NOT NULL,
    PCA2_Coordinate NUMERIC(10, 6) NOT NULL,
    Target_Default_Flag INTEGER NOT NULL
);

CREATE INDEX idx_mart_cluster ON Agg_Customer_Segment_Mart(Cluster_ID);
