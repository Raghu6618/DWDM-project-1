-- ============================================================================
-- APEXBANK DWDM PROJECT: ETL TRANSFORMATION & LOADING SCRIPT
-- Project: Customer Segmentation in Banking Using Clustering and Data Mining
-- Dataset: Default of Credit Card Clients (UCI Machine Learning Repository, ID: 350)
-- 
-- DESCRIPTION:
-- Simulates the Extract, Transform, and Load (ETL) pipeline transforming raw
-- UCI flat staged data into normalized dimensional tables and the central Fact table.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- STAGE 1: STAGING TABLE INGESTION (RAW DATA EXTRACT)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS Staging_UCI_Credit_Raw;

CREATE TABLE Staging_UCI_Credit_Raw (
    ID INTEGER,
    LIMIT_BAL NUMERIC(12, 2),
    SEX INTEGER,
    EDUCATION INTEGER,
    MARRIAGE INTEGER,
    AGE INTEGER,
    PAY_0 INTEGER,
    PAY_2 INTEGER,
    PAY_3 INTEGER,
    PAY_4 INTEGER,
    PAY_5 INTEGER,
    PAY_6 INTEGER,
    BILL_AMT1 NUMERIC(12, 2),
    BILL_AMT2 NUMERIC(12, 2),
    BILL_AMT3 NUMERIC(12, 2),
    BILL_AMT4 NUMERIC(12, 2),
    BILL_AMT5 NUMERIC(12, 2),
    BILL_AMT6 NUMERIC(12, 2),
    PAY_AMT1 NUMERIC(12, 2),
    PAY_AMT2 NUMERIC(12, 2),
    PAY_AMT3 NUMERIC(12, 2),
    PAY_AMT4 NUMERIC(12, 2),
    PAY_AMT5 NUMERIC(12, 2),
    PAY_AMT6 NUMERIC(12, 2),
    Y INTEGER
);

-- ----------------------------------------------------------------------------
-- STAGE 2: DIMENSION POPULATION & TRANSFORMATION
-- ----------------------------------------------------------------------------

-- 1. Populate Dim_Customer (Handling categorical mappings and unknown codes)
INSERT INTO Dim_Customer (
    Customer_Key,
    Customer_ID,
    Age,
    Gender_Code,
    Gender_Desc,
    Education_Code,
    Education_Desc,
    Marital_Status_Code,
    Marital_Status_Desc
)
SELECT 
    ID AS Customer_Key,
    ID AS Customer_ID,
    AGE AS Age,
    SEX AS Gender_Code,
    CASE 
        WHEN SEX = 1 THEN 'Male'
        WHEN SEX = 2 THEN 'Female'
        ELSE 'Unknown'
    END AS Gender_Desc,
    CASE 
        WHEN EDUCATION IN (1, 2, 3, 4) THEN EDUCATION
        ELSE 4 -- Map undocumented codes (0, 5, 6) into 'Others'
    END AS Education_Code,
    CASE 
        WHEN EDUCATION = 1 THEN 'Graduate School'
        WHEN EDUCATION = 2 THEN 'University'
        WHEN EDUCATION = 3 THEN 'High School'
        ELSE 'Others'
    END AS Education_Desc,
    CASE 
        WHEN MARRIAGE IN (1, 2, 3) THEN MARRIAGE
        ELSE 3 -- Map undocumented code (0) into 'Others'
    END AS Marital_Status_Code,
    CASE 
        WHEN MARRIAGE = 1 THEN 'Married'
        WHEN MARRIAGE = 2 THEN 'Single'
        ELSE 'Others'
    END AS Marital_Status_Desc
FROM Staging_UCI_Credit_Raw;


-- 2. Populate Dim_Credit
INSERT INTO Dim_Credit (
    Credit_Key,
    Customer_ID,
    Credit_Limit,
    Credit_Tier,
    Utilization_Capacity_Group
)
SELECT 
    ID AS Credit_Key,
    ID AS Customer_ID,
    LIMIT_BAL AS Credit_Limit,
    CASE 
        WHEN LIMIT_BAL < 50000 THEN 'Standard (<50k)'
        WHEN LIMIT_BAL < 150000 THEN 'Silver (50k-150k)'
        WHEN LIMIT_BAL < 300000 THEN 'Gold (150k-300k)'
        ELSE 'Platinum (>300k)'
    END AS Credit_Tier,
    CASE 
        WHEN LIMIT_BAL < 100000 THEN 'Low Capacity'
        WHEN LIMIT_BAL < 250000 THEN 'Mid Capacity'
        ELSE 'High Capacity'
    END AS Utilization_Capacity_Group
FROM Staging_UCI_Credit_Raw;


-- 3. Populate Dim_Payment_History
INSERT INTO Dim_Payment_History (
    Payment_History_Key,
    Customer_ID,
    PAY_0,
    PAY_2,
    PAY_3,
    PAY_4,
    PAY_5,
    PAY_6,
    Max_Delay_Months,
    Total_Delayed_Months,
    Consistently_On_Time
)
SELECT 
    ID AS Payment_History_Key,
    ID AS Customer_ID,
    PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
    -- Compute maximum delay across 6 periods
    (SELECT MAX(v) FROM (VALUES (PAY_0), (PAY_2), (PAY_3), (PAY_4), (PAY_5), (PAY_6)) AS value(v)) AS Max_Delay_Months,
    -- Compute count of delayed months
    (CASE WHEN PAY_0 > 0 THEN 1 ELSE 0 END +
     CASE WHEN PAY_2 > 0 THEN 1 ELSE 0 END +
     CASE WHEN PAY_3 > 0 THEN 1 ELSE 0 END +
     CASE WHEN PAY_4 > 0 THEN 1 ELSE 0 END +
     CASE WHEN PAY_5 > 0 THEN 1 ELSE 0 END +
     CASE WHEN PAY_6 > 0 THEN 1 ELSE 0 END) AS Total_Delayed_Months,
    CASE 
        WHEN PAY_0 <= 0 AND PAY_2 <= 0 AND PAY_3 <= 0 AND PAY_4 <= 0 AND PAY_5 <= 0 AND PAY_6 <= 0 THEN 1
        ELSE 0
    END AS Consistently_On_Time
FROM Staging_UCI_Credit_Raw;


-- 4. Populate Dim_Billing
INSERT INTO Dim_Billing (
    Billing_Key,
    Customer_ID,
    BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
    Total_Bill_Amount,
    Avg_Bill_Amount,
    Bill_Amount_Variance
)
SELECT 
    ID AS Billing_Key,
    ID AS Customer_ID,
    BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
    (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) AS Total_Bill_Amount,
    (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) / 6.0 AS Avg_Bill_Amount,
    0.0 AS Bill_Amount_Variance
FROM Staging_UCI_Credit_Raw;


-- 5. Populate Dim_Payment
INSERT INTO Dim_Payment (
    Payment_Key,
    Customer_ID,
    PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6,
    Total_Payment_Amount,
    Avg_Payment_Amount,
    Payment_Variance
)
SELECT 
    ID AS Payment_Key,
    ID AS Customer_ID,
    PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6,
    (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6) AS Total_Payment_Amount,
    (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6) / 6.0 AS Avg_Payment_Amount,
    0.0 AS Payment_Variance
FROM Staging_UCI_Credit_Raw;


-- ----------------------------------------------------------------------------
-- STAGE 3: POPULATE CENTRAL FACT TABLE (Fact_Customer_Credit_Profile)
-- ----------------------------------------------------------------------------
INSERT INTO Fact_Customer_Credit_Profile (
    Profile_Fact_Key,
    Customer_Key,
    Credit_Key,
    Payment_History_Key,
    Billing_Key,
    Payment_Key,
    Credit_Limit,
    Total_Billed_NTD,
    Total_Paid_NTD,
    Average_Monthly_Bill,
    Average_Monthly_Payment,
    Payment_to_Bill_Ratio,
    Credit_Utilization_Rate,
    Max_Consecutive_Delay,
    Total_Delay_Incidents,
    Target_Default_Next_Month
)
SELECT 
    c.Customer_Key AS Profile_Fact_Key,
    c.Customer_Key,
    cr.Credit_Key,
    ph.Payment_History_Key,
    b.Billing_Key,
    p.Payment_Key,
    cr.Credit_Limit,
    b.Total_Bill_Amount AS Total_Billed_NTD,
    p.Total_Payment_Amount AS Total_Paid_NTD,
    b.Avg_Bill_Amount AS Average_Monthly_Bill,
    p.Avg_Payment_Amount AS Average_Monthly_Payment,
    CASE 
        WHEN b.Total_Bill_Amount > 0 THEN ROUND(p.Total_Payment_Amount / b.Total_Bill_Amount, 4)
        ELSE 1.0000 
    END AS Payment_to_Bill_Ratio,
    CASE 
        WHEN cr.Credit_Limit > 0 THEN ROUND(b.Total_Bill_Amount / (cr.Credit_Limit * 6.0), 4)
        ELSE 0.0000
    END AS Credit_Utilization_Rate,
    ph.Max_Delay_Months AS Max_Consecutive_Delay,
    ph.Total_Delayed_Months AS Total_Delay_Incidents,
    s.Y AS Target_Default_Next_Month
FROM Staging_UCI_Credit_Raw s
JOIN Dim_Customer c ON s.ID = c.Customer_ID
JOIN Dim_Credit cr ON s.ID = cr.Customer_ID
JOIN Dim_Payment_History ph ON s.ID = ph.Customer_ID
JOIN Dim_Billing b ON s.ID = b.Customer_ID
JOIN Dim_Payment p ON s.ID = p.Customer_ID;
