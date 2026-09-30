"""
src/main.py
===========
Master Execution Pipeline for Customer Segmentation in Banking (DWDM Project).
Dataset: Default of Credit Card Clients (UCI Machine Learning Repository, ID: 350)
Author: DWDM Project Team | Antigravity AI Engineering

Executes:
1. Data Ingestion & Validation (ETL)
2. Customer-Level Credit Feature Engineering
3. Feature Selection & Standardization (Excluding ID and Y)
4. Systematic Multi-Metric Evaluation across K = 2..10 (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz)
5. Production K-Means Training (Selected Optimal K = 4)
6. Hierarchical Agglomerative Benchmarking
7. 2D PCA Dimensionality Reduction
8. Cluster Profiling & Empirical Persona Mapping
9. Post-Clustering Target Variable (Y) Default Analysis
10. Generation of 16 Publication-Grade Figures & Summary Tables
"""

import time
import os
import sys

# Ensure local imports work regardless of execution working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_cleaning import run_etl_cleaning
from feature_engineering import engineer_credit_features
from clustering import run_clustering_pipeline
from visualization import generate_all_visualizations


def main():
    print("=" * 80)
    print("DWDM PROJECT: CUSTOMER SEGMENTATION IN BANKING USING CLUSTERING & DATA MINING")
    print("DATASET: UCI 'DEFAULT OF CREDIT CARD CLIENTS' (YEH, 2009; DOI: 10.24432/C55S3H)")
    print("=" * 80)
    start_time = time.time()
    
    # Step 1: Clean & Validate UCI Dataset
    print("\n>>> STEP 1: ETL & DATA QUALITY VALIDATION")
    cleaned_df, dq_summary = run_etl_cleaning(
        raw_path='data/raw/default_of_credit_card_clients.csv',
        processed_dir='data/processed',
        output_dir='outputs/tables'
    )
    
    # Step 2: Customer Behavioral Feature Engineering
    print("\n>>> STEP 2: FINANCIAL & BEHAVIORAL FEATURE ENGINEERING")
    features_df, fe_summary = engineer_credit_features(
        cleaned_df,
        processed_dir='data/processed',
        output_dir='outputs/tables'
    )
    
    # Step 3: Unsupervised Data Mining & Multi-Metric Clustering Execution
    print("\n>>> STEP 3: K-MEANS CLUSTERING (K=2..10) & UNSUPERVISED EVALUATION")
    clustered_df, eval_metrics, profiles, post_y = run_clustering_pipeline(
        features_path='data/processed/credit_card_features.csv',
        output_dir='outputs',
        random_state=42
    )
    
    # Step 4: Analytical Visualizations
    print("\n>>> STEP 4: GENERATING ANALYTICAL FIGURES & CHARTS")
    generate_all_visualizations(clustered_df, eval_metrics, output_dir='outputs/figures')
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(f"Figures saved to:  outputs/figures/ (16 figures)")
    print(f"Tables saved to:   outputs/tables/ (Data Quality, Feature Summary, K-Eval, Profiles, Post-Y)")
    print(f"Models saved to:   outputs/models/ (K-Means, Scaler, PCA)")
    print(f"Processed data in: data/processed/ (Cleaned & Clustered CSVs)")
    print("=" * 80)


if __name__ == '__main__':
    main()
