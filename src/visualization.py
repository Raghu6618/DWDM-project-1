"""
src/visualization.py
====================
Visualization Module for DWDM Customer Segmentation Project.
Generates publication-quality charts for EDA, Model Diagnostics (Elbow, Silhouette,
Davies-Bouldin, Calinski-Harabasz), 2D PCA Projections, Segment Profiles,
and Post-Clustering Default (Y) distributions.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Global style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 14


def generate_all_visualizations(clustered_df, eval_df, output_dir='outputs/figures'):
    """
    Generates and saves all required analytical figures for the UCI dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Visualization] Generating publication-grade charts to {output_dir}...")
    
    # Color palette supporting dynamic cluster count
    palette_base = ['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b', '#06b6d4', '#ec4899', '#84cc16']
    num_personas = clustered_df['Cluster_Persona'].nunique()
    colors = palette_base[:num_personas] if num_personas <= len(palette_base) else sns.color_palette("tab10", num_personas)
    
    # -------------------------------------------------------------
    # 1. Age Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    sns.histplot(clustered_df['AGE'], bins=30, kde=True, color='#2563eb', edgecolor='black')
    plt.axvline(clustered_df['AGE'].median(), color='#dc2626', linestyle='--', linewidth=2,
                label=f"Median Age: {clustered_df['AGE'].median():.0f} yrs")
    plt.title("Customer Age Distribution (Demographic Profile)")
    plt.xlabel("Age (Years)")
    plt.ylabel("Number of Account Holders")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Credit Limit Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    sns.histplot(clustered_df['LIMIT_BAL'] / 1000, bins=35, kde=True, color='#059669', edgecolor='black')
    plt.axvline(clustered_df['LIMIT_BAL'].median() / 1000, color='#dc2626', linestyle='--', linewidth=2,
                label=f"Median Limit: NT$ {clustered_df['LIMIT_BAL'].median():,.0f}")
    plt.title("Sanctioned Credit Limit Distribution")
    plt.xlabel("Credit Limit (in Thousands NT$)")
    plt.ylabel("Number of Customers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'credit_limit_distribution.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 3. Bill Amount Distributions
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    sns.histplot(clustered_df['Avg_Bill_Amt'].clip(lower=0) / 1000, bins=40, kde=True, color='#0284c7', edgecolor='black')
    plt.title("Average Monthly Billed Statement Distribution")
    plt.xlabel("Average Bill Amount (Thousands NT$)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bill_amount_distributions.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 4. Payment Amount Distributions
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    sns.histplot(clustered_df['Avg_Pay_Amt'].clip(lower=0) / 1000, bins=40, kde=True, color='#0d9488', edgecolor='black')
    plt.title("Average Monthly Payment Remittance Distribution")
    plt.xlabel("Average Payment Amount (Thousands NT$)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'payment_amount_distributions.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 5. Repayment Status Distribution (PAY_0)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    status_counts = clustered_df['PAY_0'].value_counts().sort_index()
    sns.barplot(x=status_counts.index, y=status_counts.values, hue=status_counts.index, palette='Blues_r', legend=False)
    plt.title("Repayment Status Distribution (September 2005 - PAY_0)")
    plt.xlabel("Status Code (-2: Inactive, -1: Paid Full, 0: Revolving, 1+: Overdue Delay in Months)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'repayment_status_distribution.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 6. Education Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    edu_counts = clustered_df['EDUCATION_DESC'].value_counts()
    sns.barplot(x=edu_counts.index, y=edu_counts.values, hue=edu_counts.index, palette='viridis', legend=False)
    plt.title("Customer Education Level Distribution")
    plt.xlabel("Education Level")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'education_distribution.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 7. Gender Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    gender_counts = clustered_df['SEX_DESC'].value_counts()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
            colors=['#ec4899', '#3b82f6'], startangle=90, explode=(0.04, 0))
    plt.title("Gender Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 8. Marriage Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    marr_counts = clustered_df['MARRIAGE_DESC'].value_counts()
    sns.barplot(x=marr_counts.index, y=marr_counts.values, hue=marr_counts.index, palette='crest', legend=False)
    plt.title("Marital Status Distribution")
    plt.xlabel("Marital Status")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'marriage_distribution.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 9. Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    corr_cols = [
        'LIMIT_BAL', 'AGE', 'Avg_Bill_Amt', 'Avg_Pay_Amt',
        'Payment_to_Bill_Ratio', 'Avg_Repayment_Status',
        'Max_Repayment_Delay', 'Delayed_Months_Count', 'Y'
    ]
    corr_mat = clustered_df[corr_cols].corr()
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.5, vmax=1.0, linewidths=0.5)
    plt.title("Correlation Matrix of Financial & Behavioral Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 10. Boxplots
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 4.5))
    sns.boxplot(x='Cluster_Persona', y='LIMIT_BAL', hue='Cluster_Persona', data=clustered_df, palette=colors, legend=False)
    plt.title("Credit Limit Boxplot Across Customer Segments")
    plt.xlabel("Customer Segment")
    plt.ylabel("Credit Limit (NT$)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplots_features.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 11. Credit Limit vs Payment Behavior
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sample_vis = clustered_df.sample(n=min(3000, len(clustered_df)), random_state=42)
    sns.scatterplot(
        x=sample_vis['LIMIT_BAL'] / 1000,
        y=sample_vis['Avg_Pay_Amt'] / 1000,
        hue=sample_vis['Cluster_Persona'],
        palette=colors,
        alpha=0.6
    )
    plt.title("Credit Limit vs. Average Monthly Payment")
    plt.xlabel("Credit Limit (Thousands NT$)")
    plt.ylabel("Avg Monthly Payment (Thousands NT$)")
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'credit_limit_vs_payment.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 12. Bill vs Payment Behavior
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=sample_vis['Avg_Bill_Amt'] / 1000,
        y=sample_vis['Avg_Pay_Amt'] / 1000,
        hue=sample_vis['Cluster_Persona'],
        palette=colors,
        alpha=0.6
    )
    plt.title("Average Bill Statement vs. Average Payment Remittance")
    plt.xlabel("Avg Monthly Bill (Thousands NT$)")
    plt.ylabel("Avg Monthly Payment (Thousands NT$)")
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bill_vs_payment.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 13. Optimal K Diagnostics (4 Plots in 1 Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Elbow / Inertia
    axes[0, 0].plot(eval_df['K'], eval_df['Inertia_WCSS'], marker='o', color='#2563eb', linewidth=2)
    axes[0, 0].set_title("1. Elbow Method (Inertia / WCSS)")
    axes[0, 0].set_xlabel("Number of Clusters (K)")
    axes[0, 0].set_ylabel("Inertia")
    axes[0, 0].axvline(4, color='#dc2626', linestyle='--', label='Selected K=4')
    axes[0, 0].legend()
    
    # Silhouette Score
    axes[0, 1].plot(eval_df['K'], eval_df['Silhouette_Score'], marker='s', color='#10b981', linewidth=2)
    axes[0, 1].set_title("2. Silhouette Score (Higher is Better)")
    axes[0, 1].set_xlabel("Number of Clusters (K)")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].axvline(4, color='#dc2626', linestyle='--', label='Selected K=4')
    axes[0, 1].legend()
    
    # Davies-Bouldin
    axes[1, 0].plot(eval_df['K'], eval_df['Davies_Bouldin_Index'], marker='^', color='#f59e0b', linewidth=2)
    axes[1, 0].set_title("3. Davies-Bouldin Index (Lower is Better)")
    axes[1, 0].set_xlabel("Number of Clusters (K)")
    axes[1, 0].set_ylabel("Davies-Bouldin Index")
    axes[1, 0].axvline(4, color='#dc2626', linestyle='--', label='Selected K=4')
    axes[1, 0].legend()
    
    # Calinski-Harabasz
    axes[1, 1].plot(eval_df['K'], eval_df['Calinski_Harabasz_Index'], marker='D', color='#8b5cf6', linewidth=2)
    axes[1, 1].set_title("4. Calinski-Harabasz Index (Higher is Better)")
    axes[1, 1].set_xlabel("Number of Clusters (K)")
    axes[1, 1].set_ylabel("Calinski-Harabasz Index")
    axes[1, 1].axvline(4, color='#dc2626', linestyle='--', label='Selected K=4')
    axes[1, 1].legend()
    
    plt.suptitle("Multi-Metric Cluster Validation Grid (K = 2 to 10)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_k_metrics.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 14. 2D PCA Cluster Scatter
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x=sample_vis['PCA1'],
        y=sample_vis['PCA2'],
        hue=sample_vis['Cluster_Persona'],
        palette=colors,
        alpha=0.65,
        s=30
    )
    plt.title("2D PCA Customer Segment Projection")
    plt.xlabel("Principal Component 1 (Spending & Repayment Activity)")
    plt.ylabel("Principal Component 2 (Delinquency & Delay Index)")
    plt.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_clusters.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 15. Cluster Size Distribution (Bar & Donut)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cluster_counts = clustered_df['Cluster_Persona'].value_counts()
    
    sns.barplot(x=cluster_counts.values, y=cluster_counts.index, hue=cluster_counts.index, ax=ax1, palette=colors, legend=False)
    ax1.set_title("Customer Volume per Segment")
    ax1.set_xlabel("Number of Customers")
    
    ax2.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=140, pctdistance=0.8,
            wedgeprops=dict(width=0.4, edgecolor='w'))
    ax2.set_title("Proportional Segment Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 16. Post-Clustering Default (Y) Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    default_summary = clustered_df.groupby('Cluster_Persona')['Y'].mean() * 100
    sns.barplot(x=default_summary.index, y=default_summary.values, hue=default_summary.index, palette='Reds_r', legend=False)
    plt.title("Post-Clustering Empirical Default Rate by Segment (Target Variable Y)")
    plt.ylabel("Default Rate (%)")
    plt.xlabel("Customer Segment")
    plt.axhline(clustered_df['Y'].mean() * 100, color='blue', linestyle='--',
                label=f"Overall Population Default: {clustered_df['Y'].mean()*100:.2f}%")
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'post_clustering_default_rate.png'), dpi=200)
    plt.close()
    
    print("[Visualization] Successfully saved all 16 figures.")


if __name__ == '__main__':
    clustered_df = pd.read_csv('data/processed/credit_card_clustered.csv')
    eval_df = pd.read_csv('outputs/tables/k_evaluation_metrics.csv')
    generate_all_visualizations(clustered_df, eval_df)
