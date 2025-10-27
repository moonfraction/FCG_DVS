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

    # Target number of unique samples to return (can't exceed dataset size)
    target = min(n_clusters * m_neighbors, len(X))

    selected_idx_set = []

    # Preferred selection: choose up to m_neighbors from each cluster's own members
    labels = kmeans.labels_
    for cluster_id in range(actual_clusters):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue

        # Distances from this centroid to its member points
        member_distances = cdist([centroids[cluster_id]], X[members], metric='euclidean')[0]
        # Sort members by distance to centroid and pick up to m_neighbors unique ones
        sorted_members = members[np.argsort(member_distances)]
        count = 0
        for idx in sorted_members:
            if idx not in selected_idx_set:
                selected_idx_set.append(int(idx))
                count += 1
                if count >= m_neighbors:
                    break

        # Stop early if we've reached the target
        if len(selected_idx_set) >= target:
            break

    # If we didn't get enough unique samples (e.g., some clusters are tiny and overlap),
    # backfill by adding the remaining nearest unselected points (by distance to nearest centroid)
    if len(selected_idx_set) < target:
        # Compute distance from each point to its nearest centroid
        all_dists = cdist(centroids, X, metric='euclidean')
        nearest_dists = np.min(all_dists, axis=0)

        # Sort all indices by distance to nearest centroid, add unselected ones until target reached
        all_sorted = np.argsort(nearest_dists)
        for idx in all_sorted:
            if len(selected_idx_set) >= target:
                break
            if int(idx) not in selected_idx_set:
                selected_idx_set.append(int(idx))

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
