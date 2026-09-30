"""
src/feature_engineering.py
==========================
Customer-Level Financial & Behavioral Feature Engineering Module.
Derives domain-specific credit utilization, payment regularity, and delinquency
features from the multi-month credit card statements for unsupervised clustering.
"""

import os
import pandas as pd
import numpy as np


def engineer_credit_features(df_clean,
                              processed_dir='data/processed',
                              output_dir='outputs/tables'):
    """
    Constructs analytical behavioral features from the 6-month billing and payment history.
    """
    print("[Feature Engineering] Engineering customer behavioral features...")
    df = df_clean.copy()
    
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    # 1. Billing aggregates
    df['Total_Bill_Amt'] = df[bill_cols].sum(axis=1)
    df['Avg_Bill_Amt'] = df[bill_cols].mean(axis=1).round(2)
    df['Bill_Variability'] = df[bill_cols].std(axis=1).fillna(0).round(2)
    
    # 2. Payment remittance aggregates
    df['Total_Pay_Amt'] = df[pay_amt_cols].sum(axis=1)
    df['Avg_Pay_Amt'] = df[pay_amt_cols].mean(axis=1).round(2)
    df['Pay_Variability'] = df[pay_amt_cols].std(axis=1).fillna(0).round(2)
    
    # 3. Solvency & Debt Ratios
    # Payment to Bill Ratio: Measures propensity to pay off statement balances
    # Avoid zero division and cap outliers to 10.0 for numerical stability
    df['Payment_to_Bill_Ratio'] = (
        df['Total_Pay_Amt'] / (df['Total_Bill_Amt'].abs() + 1.0)
    ).clip(0, 10.0).round(4)
    
    # Credit Limit Utilization: Mean bill relative to sanctioned credit facility
    df['Credit_Utilization_Ratio'] = (
        df['Avg_Bill_Amt'] / (df['LIMIT_BAL'] + 1.0)
    ).clip(-1.0, 5.0).round(4)
    
    # 4. Delinquency and Delay Dynamics
    # Average repayment status (-2 to +8 across 6 months)
    df['Avg_Repayment_Status'] = df[pay_status_cols].mean(axis=1).round(3)
    
    # Maximum consecutive delay in months observed
    df['Max_Repayment_Delay'] = df[pay_status_cols].max(axis=1)
    
    # Total count of months where customer had an overdue delay (>0)
    df['Delayed_Months_Count'] = (df[pay_status_cols] > 0).sum(axis=1)
    
    # Feature Engineering Summary Table as mandated by DWDM specification
    fe_summary = pd.DataFrame([
        {
            'Feature': 'LIMIT_BAL',
            'Type': 'Numerical (Currency NTD)',
            'Description': 'Total sanctioned credit line extended to customer',
            'Used_for_Clustering': 'YES',
            'Reason': 'Fundamental benchmark for purchasing capacity and creditworthiness'
        },
        {
            'Feature': 'AGE',
            'Type': 'Numerical (Years)',
            'Description': 'Customer chronological age in completed years',
            'Used_for_Clustering': 'YES',
            'Reason': 'Captures life-stage demographic variations in credit usage'
        },
        {
            'Feature': 'Avg_Bill_Amt',
            'Type': 'Numerical (Derived NTD)',
            'Description': 'Mean monthly billed statement balance across 6 months',
            'Used_for_Clustering': 'YES',
            'Reason': 'Indicates overall credit demand and spending magnitude'
        },
        {
            'Feature': 'Avg_Pay_Amt',
            'Type': 'Numerical (Derived NTD)',
            'Description': 'Mean monthly payment remitted across 6 observation periods',
            'Used_for_Clustering': 'YES',
            'Reason': 'Reflects customer cash-flow capacity and settlement behavior'
        },
        {
            'Feature': 'Payment_to_Bill_Ratio',
            'Type': 'Numerical Ratio',
            'Description': 'Ratio of total payment remitted to total billed balance',
            'Used_for_Clustering': 'YES',
            'Reason': 'High-signal solvency metric separating full payers from revolvers'
        },
        {
            'Feature': 'Avg_Repayment_Status',
            'Type': 'Numerical Score',
            'Description': 'Mean repayment status score across PAY_0 to PAY_6',
            'Used_for_Clustering': 'YES',
            'Reason': 'Captures overall promptness vs. ongoing revolving/delayed habits'
        },
        {
            'Feature': 'Max_Repayment_Delay',
            'Type': 'Integer (Months)',
            'Description': 'Maximum repayment delay (in months) experienced over 6 months',
            'Used_for_Clustering': 'YES',
            'Reason': 'Identifies peak historical financial delinquency exposure'
        },
        {
            'Feature': 'Delayed_Months_Count',
            'Type': 'Integer (0 to 6)',
            'Description': 'Number of months where payment was overdue (> 0 months)',
            'Used_for_Clustering': 'YES',
            'Reason': 'Measures delinquency frequency and repayment consistency'
        },
        {
            'Feature': 'Total_Bill_Amt',
            'Type': 'Numerical (NTD)',
            'Description': 'Sum of 6-month statement bills',
            'Used_for_Clustering': 'NO',
            'Reason': 'Redundant with Avg_Bill_Amt (perfect collinearity, r=1.0)'
        },
        {
            'Feature': 'Total_Pay_Amt',
            'Type': 'Numerical (NTD)',
            'Description': 'Sum of 6-month payments',
            'Used_for_Clustering': 'NO',
            'Reason': 'Redundant with Avg_Pay_Amt (perfect collinearity, r=1.0)'
        },
        {
            'Feature': 'Bill_Variability',
            'Type': 'Numerical (Std NTD)',
            'Description': 'Standard deviation of monthly statement amounts',
            'Used_for_Clustering': 'NO',
            'Reason': 'High collinearity with average bill magnitude; secondary noise'
        },
        {
            'Feature': 'Pay_Variability',
            'Type': 'Numerical (Std NTD)',
            'Description': 'Standard deviation of monthly payment remittances',
            'Used_for_Clustering': 'NO',
            'Reason': 'Captures payment volatility; excluded to preserve cluster parsimony'
        },
        {
            'Feature': 'ID',
            'Type': 'Identifier',
            'Description': 'Customer account identification index',
            'Used_for_Clustering': 'NO (EXCLUDED)',
            'Reason': 'Arbitrary sequential key with zero analytical/behavioral variance'
        },
        {
            'Feature': 'Y',
            'Type': 'Binary Target (0/1)',
            'Description': 'Default payment status next month (October 2005)',
            'Used_for_Clustering': 'NO (EXCLUDED)',
            'Reason': 'Supervised target variable; excluded to maintain unsupervised learning'
        }
    ])
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    out_feat_path = os.path.join(processed_dir, 'credit_card_features.csv')
    df.to_csv(out_feat_path, index=False)
    
    fe_table_path = os.path.join(output_dir, 'feature_engineering_summary.csv')
    fe_summary.to_csv(fe_table_path, index=False)
    
    print(f"[Feature Engineering] Engineered dataset saved to: {out_feat_path}")
    print(f"[Feature Engineering] Feature summary saved to: {fe_table_path}")
    
    return df, fe_summary


if __name__ == '__main__':
    from data_cleaning import run_etl_cleaning
    df_clean, _ = run_etl_cleaning()
    engineer_credit_features(df_clean)
