"""
STEP 1: Diverse Clustering Module
Implements K-Means clustering to select diverse representative samples
"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from logger_utils import get_logger

logger = get_logger()


def diverse_clustering(subgroup, n_clusters=8, m_neighbors=5, random_state=42):
    """
    Apply K-Means clustering and select m closest samples to each centroid
    
    Args:
        subgroup: Subgroup object containing samples
        n_clusters: Number of clusters to create (default: 8)
        m_neighbors: Number of closest samples to select per cluster (default: 5)
        random_state: Random seed for reproducibility
    
    Returns:
        List of selected sample indices
    """
    # Get feature matrix
    X = subgroup.get_features()
    
    # Handle case where subgroup has fewer samples than clusters
    actual_clusters = min(n_clusters, len(X))
    
    if len(X) == 0:
        return []
    
    if len(X) <= m_neighbors:
        # If subgroup is very small, return all indices
        return list(range(len(X)))
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=actual_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    
    centroids = kmeans.cluster_centers_
    selected_idx_set = []
    
    # For each centroid, find m closest samples
    for center in centroids:
        # Calculate distances from all points to this centroid
        distances = cdist([center], X, metric='euclidean')[0]
        
        # Get indices of m closest samples
        closest_indices = np.argsort(distances)[:m_neighbors]
        
        # Add to selected set (avoid duplicates)
        for idx in closest_indices:
            if idx not in selected_idx_set:
                selected_idx_set.append(idx)
    
    return selected_idx_set


def step1_diverse_clustering(subgroups_dict, n_clusters=8, m_neighbors=5):
    """
    STEP 1: Apply diverse clustering to all subgroups
    
    Args:
        subgroups_dict: Dictionary of Subgroup objects {g1, g2, g3, g4}
        n_clusters: Number of clusters (default: 8)
        m_neighbors: Number of neighbors per cluster (default: 5)
    
    Returns:
        Dictionary of subgroups with selected indices (SG')
    """
    selected_subgroups = {}
    
    for key, subgroup in subgroups_dict.items():
        logger.info(f"Processing {key}: {len(subgroup)} samples")
        
        # Apply diverse clustering
        selected_indices = diverse_clustering(subgroup, n_clusters, m_neighbors)
        
        # Create new subgroup with only selected samples
        selected_df = subgroup.df.iloc[selected_indices].reset_index(drop=True)
        
        # Create new Subgroup object
        from data_utils import Subgroup
        selected_subgroup = Subgroup(
            selected_df, 
            subgroup.z_value, 
            subgroup.y_value,
            initial_score=0.05
        )
        
        # Copy scores for selected indices
        selected_subgroup.scores = subgroup.scores[selected_indices]
        
        selected_subgroups[key] = selected_subgroup
        
    logger.info(f"  Selected {len(selected_indices)} diverse samples from {key}")
    
    return selected_subgroups
