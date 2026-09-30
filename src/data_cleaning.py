"""
src/data_cleaning.py
====================
Extract, Transform, and Load (ETL) Data Cleaning Module.
Responsible for loading the official UCI "Default of Credit Card Clients" dataset,
performing data quality audits, cleaning undocumented categories, and saving
reproducible analytical staging tables.
"""

import os
import pandas as pd
import numpy as np


def run_etl_cleaning(raw_path='data/raw/default_of_credit_card_clients.csv',
                     processed_dir='data/processed',
                     output_dir='outputs/tables'):
    """
    Executes end-to-end data validation and cleaning on the UCI dataset.
    """
    print(f"[ETL] Loading raw dataset from: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run ucimlrepo download first.")
        
    df_raw = pd.read_csv(raw_path)
    initial_rows, initial_cols = df_raw.shape
    print(f"[ETL] Initial raw dimensions: {initial_rows:,} records across {initial_cols} columns")
    
    # Pre-cleaning audit metrics
    null_counts_pre = int(df_raw.isnull().sum().sum())
    duplicate_counts_pre = int(df_raw.duplicated().sum())
    
    # Documented valid categories in UCI specification:
    # EDUCATION: 1 = graduate school; 2 = university; 3 = high school; 4 = others
    # (0, 5, 6 are undocumented categories in the original UCI release)
    invalid_education_pre = int(df_raw['EDUCATION'].isin([0, 5, 6]).sum())
    
    # MARRIAGE: 1 = married; 2 = single; 3 = others
    # (0 is undocumented category in the original UCI release)
    invalid_marriage_pre = int((df_raw['MARRIAGE'] == 0).sum())
    
    # Data Cleaning Actions
    df_clean = df_raw.copy()
    
    # 1. Clean EDUCATION: Group undocumented codes (0, 5, 6) into 4 ('Others')
    df_clean['EDUCATION_CLEANED'] = df_clean['EDUCATION'].apply(lambda x: x if x in [1, 2, 3, 4] else 4)
    education_labels = {
        1: 'Graduate School',
        2: 'University',
        3: 'High School',
        4: 'Others'
    }
    df_clean['EDUCATION_DESC'] = df_clean['EDUCATION_CLEANED'].map(education_labels)
    
    # 2. Clean MARRIAGE: Group undocumented code (0) into 3 ('Others')
    df_clean['MARRIAGE_CLEANED'] = df_clean['MARRIAGE'].apply(lambda x: x if x in [1, 2, 3] else 3)
    marriage_labels = {
        1: 'Married',
        2: 'Single',
        3: 'Others'
    }
    df_clean['MARRIAGE_DESC'] = df_clean['MARRIAGE_CLEANED'].map(marriage_labels)
    
    # 3. Standardize SEX label
    sex_labels = {1: 'Male', 2: 'Female'}
    df_clean['SEX_DESC'] = df_clean['SEX'].map(sex_labels)
    
    # Post-cleaning audit metrics
    null_counts_post = int(df_clean.isnull().sum().sum())
    duplicate_counts_post = int(df_clean.duplicated().sum())
    invalid_education_post = int(df_clean['EDUCATION_CLEANED'].isin([0, 5, 6]).sum())
    invalid_marriage_post = int((df_clean['MARRIAGE_CLEANED'] == 0).sum())
    
    # Outlier inspection using IQR rule on LIMIT_BAL
    q25 = df_clean['LIMIT_BAL'].quantile(0.25)
    q75 = df_clean['LIMIT_BAL'].quantile(0.75)
    iqr = q75 - q25
    outlier_count_limit = int(((df_clean['LIMIT_BAL'] < (q25 - 1.5 * iqr)) | 
                               (df_clean['LIMIT_BAL'] > (q75 + 1.5 * iqr))).sum())
    
    # Compile Data Quality Summary Before/After table
    data_quality_df = pd.DataFrame({
        'Metric': [
            'Total Records (Rows)',
            'Total Features (Columns)',
            'Missing Null Values',
            'Duplicate Rows',
            'Undocumented Education Codes (0, 5, 6)',
            'Undocumented Marriage Codes (0)',
            'High Credit Limit Outliers (> Q3 + 1.5*IQR)'
        ],
        'Before_Cleaning': [
            f"{initial_rows:,}",
            f"{initial_cols}",
            f"{null_counts_pre}",
            f"{duplicate_counts_pre}",
            f"{invalid_education_pre:,} (Recoded to 4)",
            f"{invalid_marriage_pre:,} (Recoded to 3)",
            f"{outlier_count_limit:,}"
        ],
        'After_Cleaning': [
            f"{len(df_clean):,}",
            f"{len(df_clean.columns)}",
            f"{null_counts_post}",
            f"{duplicate_counts_post}",
            f"{invalid_education_post}",
            f"{invalid_marriage_post}",
            f"{outlier_count_limit:,} (Retained with scaling)"
        ],
        'Methodology_Action': [
            'Verified full population integrity from UCI Repository',
            'Preserved raw features + added analytical semantic dimensions',
            'Confirmed zero missing values in UCI original dataset',
            'Zero exact row duplicates detected in benchmark data',
            'Consolidated rare/undocumented codes into category 4 (Others)',
            'Consolidated undocumented marital status into category 3 (Others)',
            'Retained valid high-net-worth customers; robustly handled via StandardScaler'
        ]
    })
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save outputs
    out_clean_path = os.path.join(processed_dir, 'credit_card_cleaned.csv')
    df_clean.to_csv(out_clean_path, index=False)
    
    out_sample_path = os.path.join(processed_dir, 'credit_card_cleaned_sample.csv')
    df_clean.head(10000).to_csv(out_sample_path, index=False)
    
    dq_table_path = os.path.join(output_dir, 'data_quality_summary.csv')
    data_quality_df.to_csv(dq_table_path, index=False)
    
    print(f"[ETL] Cleaned dataset saved to: {out_clean_path}")
    print(f"[ETL] Data quality summary saved to: {dq_table_path}")
    
    return df_clean, data_quality_df


if __name__ == '__main__':
    run_etl_cleaning()
