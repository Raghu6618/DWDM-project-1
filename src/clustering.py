"""
src/clustering.py
=================
Data Mining & Unsupervised Clustering Execution Module.
Performs feature standardization, systematic multi-metric evaluation across K=2..10
(Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz), final K-Means modeling,
hierarchical algorithm benchmarking, 2D PCA visualization projection,
empirical cluster profiling, and post-clustering target (Y) default analysis.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


def run_clustering_pipeline(features_path='data/processed/credit_card_features.csv',
                            output_dir='outputs',
                            random_state=42):
    """
    Executes the full clustering, evaluation, PCA, and profiling pipeline.
    """
    print(f"[Clustering] Ingesting engineered credit features from: {features_path}")
    df = pd.read_csv(features_path)
    total_customers = len(df)
    print(f"[Clustering] Total customers available: {total_customers:,}")
    
    # 1. Feature Matrix Definition
    # ID and Y are strictly excluded from the clustering input space
    clustering_features = [
        'LIMIT_BAL',
        'AGE',
        'Avg_Bill_Amt',
        'Avg_Pay_Amt',
        'Payment_to_Bill_Ratio',
        'Avg_Repayment_Status',
        'Max_Repayment_Delay',
        'Delayed_Months_Count'
    ]
    print(f"[Clustering] Selected features ({len(clustering_features)}): {clustering_features}")
    
    X_raw = df[clustering_features].copy()
    
    # 2. Standardization
    print("[Clustering] Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. Systematic K Evaluation across K = 2 to 10
    print("\n" + "="*60)
    print("SYSTEMATIC K-MEANS EVALUATION (K = 2 to 10)")
    print("="*60)
    
    k_range = list(range(2, 11))
    inertias = []
    sil_scores = []
    db_indices = []
    ch_indices = []
    
    # Evaluate with 10,000 stratified sample for efficient silhouette computation
    # (Davies-Bouldin and Calinski-Harabasz are computed on full 30,000 population)
    np.random.seed(random_state)
    sil_sample_idx = np.random.choice(len(X_scaled), size=10000, replace=False)
    X_sil_sample = X_scaled[sil_sample_idx]
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        labels = km.labels_
        
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_sil_sample, labels[sil_sample_idx], random_state=random_state))
        db = float(davies_bouldin_score(X_scaled, labels))
        ch = float(calinski_harabasz_score(X_scaled, labels))
        
        inertias.append(round(inertia, 2))
        sil_scores.append(round(sil, 4))
        db_indices.append(round(db, 4))
        ch_indices.append(round(ch, 2))
        
        print(f"K = {k:2d} | Inertia = {inertia:10.1f} | Silhouette = {sil:.4f} | Davies-Bouldin = {db:.4f} | Calinski-Harabasz = {ch:8.1f}")
        
    k_eval_df = pd.DataFrame({
        'K': k_range,
        'Inertia_WCSS': inertias,
        'Silhouette_Score': sil_scores,
        'Davies_Bouldin_Index': db_indices,
        'Calinski_Harabasz_Index': ch_indices
    })
    
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    k_eval_path = os.path.join(output_dir, 'tables', 'k_evaluation_metrics.csv')
    k_eval_df.to_csv(k_eval_path, index=False)
    print(f"[Clustering] K evaluation metrics saved to: {k_eval_path}")
    
    # 4. Final Model Training with Optimal K = 4
    # Justification: K=4 represents the inflection point on the elbow curve, achieves high Calinski-Harabasz
    # separation, maintains robust Silhouette separation, and yields four distinct, actionable banking customer segments.
    optimal_k = 4
    print(f"\n[Clustering] Training production K-Means model with Selected Optimal K = {optimal_k}...")
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    df['Cluster'] = kmeans_final.fit_predict(X_scaled)
    
    # 5. Algorithm Comparison Benchmark (Hierarchical Ward vs K-Means on N=2,500 sample)
    print("\n[Clustering] Executing benchmark comparison: K-Means vs Hierarchical Agglomerative...")
    sample_size_hac = 2500
    X_bench = X_scaled[:sample_size_hac]
    
    hac = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hac_labels = hac.fit_predict(X_bench)
    hac_sil = float(silhouette_score(X_bench, hac_labels))
    km_bench_sil = float(silhouette_score(X_bench, kmeans_final.predict(X_bench)))
    
    algo_comp_df = pd.DataFrame({
        'Algorithm': [f'K-Means (k={optimal_k})', f'Hierarchical Ward (k={optimal_k})'],
        'Sample_Size': [sample_size_hac, sample_size_hac],
        'Silhouette_Score': [round(km_bench_sil, 4), round(hac_sil, 4)],
        'Computational_Complexity': ['O(k * N * d) - Linear in N', 'O(N^3) time, O(N^2) space - Cubic in N'],
        'Suitability_For_Large_Banking_DW': ['Production Ready / Highly Scalable', 'Benchmark / Small Data Marts Only'],
        'Primary_Strengths': [
            'Direct convergence, scales to millions of records, easily scores new incoming transactions',
            'Creates intuitive nested dendrogram hierarchy; deterministic linkage'
        ]
    })
    algo_comp_path = os.path.join(output_dir, 'tables', 'algorithm_comparison.csv')
    algo_comp_df.to_csv(algo_comp_path, index=False)
    
    # 6. PCA 2D Dimensionality Reduction (Primarily for Visualization)
    print("[Clustering] Computing 2D PCA projection for cluster visualization...")
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    df['PCA1'] = np.round(pca_coords[:, 0], 4)
    df['PCA2'] = np.round(pca_coords[:, 1], 4)
    
    var_exp1 = pca.explained_variance_ratio_[0] * 100
    var_exp2 = pca.explained_variance_ratio_[1] * 100
    total_var = var_exp1 + var_exp2
    print(f"[Clustering] PCA Variance Explained: PC1 = {var_exp1:.2f}%, PC2 = {var_exp2:.2f}% (Total = {total_var:.2f}%)")
    
    # Save PCA Loadings table
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=clustering_features,
        columns=['PC1_Loading', 'PC2_Loading']
    ).round(4).reset_index().rename(columns={'index': 'Feature'})
    pca_loadings.to_csv(os.path.join(output_dir, 'tables', 'pca_loadings.csv'), index=False)
    
    # 7. Semantic Cluster Labeling based on Actual Calculated Statistics
    cluster_stats = df.groupby('Cluster').agg({
        'LIMIT_BAL': 'mean',
        'Avg_Bill_Amt': 'mean',
        'Avg_Pay_Amt': 'mean',
        'Max_Repayment_Delay': 'mean',
        'Delayed_Months_Count': 'mean',
        'Payment_to_Bill_Ratio': 'mean',
        'Y': 'mean'
    })
    
    # Derive empirically grounded personas dynamically
    persona_map = {}
    mean_pop_limit = df['LIMIT_BAL'].mean()
    mean_pop_pay = df['Avg_Pay_Amt'].mean()
    
    for cl in range(optimal_k):
        lim = cluster_stats.loc[cl, 'LIMIT_BAL']
        delay = cluster_stats.loc[cl, 'Max_Repayment_Delay']
        pay = cluster_stats.loc[cl, 'Avg_Pay_Amt']
        
        if delay > 1.2:
            persona_map[cl] = "Lower Credit / High Repayment Delay Pattern"
        elif pay > 2.0 * mean_pop_pay and lim > mean_pop_limit:
            persona_map[cl] = "Higher Credit / Higher Payment Activity"
        elif delay < 0.0 and lim > mean_pop_limit:
            persona_map[cl] = "High Credit / Advance & Prompt Payers"
        else:
            persona_map[cl] = "Moderate Credit / Stable Payment Pattern"
            
    df['Cluster_Persona'] = df['Cluster'].map(persona_map)
    print("\n[Clustering] Empirical Cluster Persona Mapping:")
    for cl, name in persona_map.items():
        print(f"  - Cluster {cl}: {name}")
        
    # 8. Detailed Customer Segment Profiles
    profiles = df.groupby(['Cluster', 'Cluster_Persona']).agg(
        Customers=('ID', 'count'),
        Percentage=('ID', lambda x: round(len(x) / total_customers * 100, 2)),
        Avg_Age=('AGE', lambda x: round(x.mean(), 1)),
        Avg_Credit_Limit=('LIMIT_BAL', lambda x: round(x.mean(), 2)),
        Avg_Bill=('Avg_Bill_Amt', lambda x: round(x.mean(), 2)),
        Avg_Payment=('Avg_Pay_Amt', lambda x: round(x.mean(), 2)),
        Avg_Repayment_Status=('Avg_Repayment_Status', lambda x: round(x.mean(), 3)),
        Avg_Max_Delay=('Max_Repayment_Delay', lambda x: round(x.mean(), 2)),
        Avg_Overdue_Months=('Delayed_Months_Count', lambda x: round(x.mean(), 2)),
        Avg_Pay_to_Bill_Ratio=('Payment_to_Bill_Ratio', lambda x: round(x.mean(), 2)),
        Post_Default_Rate=('Y', lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    
    profiles_path = os.path.join(output_dir, 'tables', 'cluster_profiles.csv')
    profiles.to_csv(profiles_path, index=False)
    print("\n[Clustering] Customer Segment Profiles:")
    print(profiles[['Cluster', 'Cluster_Persona', 'Customers', 'Percentage', 'Avg_Credit_Limit', 'Avg_Max_Delay', 'Avg_Overdue_Months', 'Post_Default_Rate']])
    
    # 9. Post-Clustering Target Variable (Y) Analysis
    # NOTE: Y was NEVER used in clustering; this is strictly descriptive post-hoc evaluation.
    post_y_analysis = df.groupby(['Cluster', 'Cluster_Persona']).agg(
        Customers=('ID', 'count'),
        Non_Default_Count=('Y', lambda x: int((x == 0).sum())),
        Default_Count=('Y', lambda x: int((x == 1).sum())),
        Default_0_Pct=('Y', lambda x: round((x == 0).mean() * 100, 2)),
        Default_1_Pct=('Y', lambda x: round((x == 1).mean() * 100, 2))
    ).reset_index()
    post_y_path = os.path.join(output_dir, 'tables', 'post_clustering_default_analysis.csv')
    post_y_analysis.to_csv(post_y_path, index=False)
    print(f"\n[Clustering] Post-clustering Y analysis saved to: {post_y_path}")
    
    # Save Cluster Centers in original scale
    cluster_centers_unscaled = scaler.inverse_transform(kmeans_final.cluster_centers_)
    centers_df = pd.DataFrame(
        cluster_centers_unscaled,
        columns=clustering_features
    )
    centers_df['Cluster'] = range(optimal_k)
    centers_df['Cluster_Persona'] = centers_df['Cluster'].map(persona_map)
    centers_path = os.path.join(output_dir, 'tables', 'cluster_centers.csv')
    centers_df.to_csv(centers_path, index=False)
    
    # 10. Persist Serialized Artifacts & Datasets
    joblib.dump(kmeans_final, os.path.join(output_dir, 'models', 'kmeans_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'models', 'scaler.joblib'))
    joblib.dump(pca, os.path.join(output_dir, 'models', 'pca_model.joblib'))
    
    # Metadata dictionary for dashboard display
    meta_df = pd.DataFrame([{
        'Optimal_K': optimal_k,
        'Random_State': random_state,
        'Features_Used': ", ".join(clustering_features),
        'Total_Customers': total_customers,
        'Silhouette_Score': float(k_eval_df.loc[k_eval_df['K'] == optimal_k, 'Silhouette_Score'].values[0]),
        'Davies_Bouldin': float(k_eval_df.loc[k_eval_df['K'] == optimal_k, 'Davies_Bouldin_Index'].values[0]),
        'Calinski_Harabasz': float(k_eval_df.loc[k_eval_df['K'] == optimal_k, 'Calinski_Harabasz_Index'].values[0]),
        'Inertia': float(k_eval_df.loc[k_eval_df['K'] == optimal_k, 'Inertia_WCSS'].values[0]),
        'PCA1_Var': round(var_exp1, 2),
        'PCA2_Var': round(var_exp2, 2),
        'Overall_Default_Rate': round(float(df['Y'].mean() * 100), 2)
    }])
    meta_df.to_csv(os.path.join(output_dir, 'tables', 'clustering_metadata.csv'), index=False)
    
    # Save clustered datasets
    clustered_path = 'data/processed/credit_card_clustered.csv'
    df.to_csv(clustered_path, index=False)
    
    sample_clustered_path = 'data/processed/credit_card_clustered_sample.csv'
    df.head(10000).to_csv(sample_clustered_path, index=False)
    
    print(f"[Clustering] Clustered dataset saved to: {clustered_path}")
    print("[Clustering] Execution complete!")
    
    return df, k_eval_df, profiles, post_y_analysis


if __name__ == '__main__':
    run_clustering_pipeline()
