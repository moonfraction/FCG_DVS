"""
Dynamic Validation Set (DVS) Utilities
Embedding and nearest neighbor functions for FCG-DVS hybrid approach
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from logger_utils import get_logger

logger = get_logger()

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")


def init_gemini(api_key=None):
    """
    Initialize Gemini API
    
    Args:
        api_key: Optional API key. If None, reads from GEMINI_API_KEY env var
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai package not installed")
    
    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    logger.info("Gemini API initialized")


def format_sample_for_embedding(row):
    """
    Format a sample as text for embedding
    
    Args:
        row: DataFrame row with sample data
    
    Returns:
        Formatted string representation
    """
    # Map values for readability
    sex_map = {0: 'Male', 1: 'Female'}
    income_map = {0: '<=50K', 1: '>50K'}
    
    text = f"Age: {row['age']}, "
    text += f"Workclass: {row['workclass']}, "
    text += f"Education: {row['education']}, "
    text += f"Marital: {row['marital-status']}, "
    text += f"Occupation: {row['occupation']}, "
    text += f"Relationship: {row['relationship']}, "
    text += f"Sex: {sex_map.get(row['sex'], row['sex'])}, "
    text += f"Capital Gain: {row['capital-gain']}, "
    text += f"Capital Loss: {row['capital-loss']}, "
    text += f"Hours/Week: {row['hours-per-week']}"
    
    if 'income' in row:
        text += f", Income: {income_map.get(row['income'], row['income'])}"
    
    return text


def get_gemini_embeddings(texts, model_name="models/text-embedding-004", batch_size=100):
    """
    Get embeddings from Gemini API
    
    Args:
        texts: List of text strings to embed
        model_name: Gemini embedding model name
        batch_size: Number of texts to embed per batch
    
    Returns:
        numpy array of embeddings (n_samples x embedding_dim)
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai package not installed")
    
    all_embeddings = []
    
    logger.info(f"Getting embeddings for {len(texts)} samples...")
    
    # Process in batches to avoid API limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            result = genai.embed_content(
                model=model_name,
                content=batch,
                task_type="retrieval_document"
            )
            embeddings = result['embedding']
            all_embeddings.extend(embeddings)
            
            if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(texts):
                logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} samples")
        
        except Exception as e:
            logger.error(f"Error embedding batch {i}-{i+batch_size}: {e}")
            # Return zeros for failed batch
            embedding_dim = len(all_embeddings[0]) if all_embeddings else 768
            all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
    
    return np.array(all_embeddings)


def embed_dataframe(df, cache_path=None):
    """
    Get embeddings for all samples in a DataFrame
    
    Args:
        df: DataFrame with samples
        cache_path: Optional path to save/load cached embeddings
    
    Returns:
        numpy array of embeddings
    """
    # Try to load from cache
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Format samples as text
    texts = [format_sample_for_embedding(row) for _, row in df.iterrows()]
    
    # Get embeddings
    embeddings = get_gemini_embeddings(texts)
    
    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Saved embeddings to {cache_path}")
    
    return embeddings


def find_k_nearest_neighbors(query_embeddings, pool_embeddings, k=5, exclude_self=False):
    """
    Find k nearest neighbors for each query using cosine similarity
    
    Args:
        query_embeddings: numpy array (n_queries x embedding_dim)
        pool_embeddings: numpy array (n_pool x embedding_dim)
        k: Number of neighbors to return
        exclude_self: If True, exclude the query itself from results (when query is in pool)
    
    Returns:
        List of lists, where each inner list contains k indices from pool_embeddings
    """
    # Compute cosine similarity
    similarities = cosine_similarity(query_embeddings, pool_embeddings)
    
    # Get top k indices for each query
    neighbors = []
    for i, sim_scores in enumerate(similarities):
        if exclude_self:
            # Set self-similarity to -inf to exclude
            sim_scores = sim_scores.copy()
            sim_scores[i] = -np.inf
        
        # Get indices of top k
        top_k_indices = np.argsort(sim_scores)[-k:][::-1]
        neighbors.append(top_k_indices.tolist())
    
    return neighbors


def create_batch_support_sets(batch_df, pool_df, pool_embeddings, k=5, cache_dir='data/embeddings'):
    """
    Create support sets S_n for a batch of test samples
    
    Args:
        batch_df: DataFrame with test batch samples
        pool_df: DataFrame with the FCG-optimized pool (H_opt)
        pool_embeddings: Precomputed embeddings for pool_df
        k: Number of nearest neighbors per test sample
        cache_dir: Directory to cache batch embeddings
    
    Returns:
        dict with:
            - 'support_indices': list of lists (one per test sample)
            - 'all_support_indices': combined unique indices
            - 'batch_embeddings': embeddings for the batch
    """
    # Get embeddings for batch (cache per batch if possible)
    batch_cache_path = os.path.join(cache_dir, f'batch_{len(batch_df)}_samples.pkl')
    batch_embeddings = embed_dataframe(batch_df, cache_path=None)  # Don't cache test batches
    
    # Find k nearest neighbors for each test sample
    neighbors_list = find_k_nearest_neighbors(batch_embeddings, pool_embeddings, k=k)
    
    # Combine all neighbors (S_n = union of all S_{n,j})
    all_support_indices = []
    for neighbors in neighbors_list:
        all_support_indices.extend(neighbors)
    all_support_indices = list(set(all_support_indices))  # Remove duplicates
    
    logger.info(f"Batch size: {len(batch_df)}, Total unique support samples: {len(all_support_indices)}")
    
    return {
        'support_indices': neighbors_list,  # List of k neighbors per test sample
        'all_support_indices': all_support_indices,
        'batch_embeddings': batch_embeddings
    }


def create_dvs_and_candidates(batch_df, pool_df, support_info):
    """
    Create Dynamic Validation Set (P_n) and candidate pool (H_n)
    
    Args:
        batch_df: Test batch DataFrame
        pool_df: FCG-optimized pool DataFrame
        support_info: Output from create_batch_support_sets
    
    Returns:
        dict with:
            - 'dvs_indices': indices for P_n (closest neighbor of each test sample)
            - 'dvs_df': DataFrame for P_n
            - 'candidate_indices': indices for H_n (remaining support samples)
            - 'candidate_df': DataFrame for H_n
    """
    support_indices_list = support_info['support_indices']
    
    # P_n: Take the closest neighbor (first in each list) for each test sample
    dvs_indices = [neighbors[0] for neighbors in support_indices_list if len(neighbors) > 0]
    dvs_df = pool_df.iloc[dvs_indices].reset_index(drop=True)
    
    # H_n: Remaining samples from S_n
    all_support = support_info['all_support_indices']
    candidate_indices = [idx for idx in all_support if idx not in dvs_indices]
    candidate_df = pool_df.iloc[candidate_indices].reset_index(drop=True) if candidate_indices else pd.DataFrame()
    
    logger.info(f"DVS size (P_n): {len(dvs_df)}, Candidate pool size (H_n): {len(candidate_df)}")
    
    return {
        'dvs_indices': dvs_indices,
        'dvs_df': dvs_df,
        'candidate_indices': candidate_indices,
        'candidate_df': candidate_df,
        'support_indices_list': support_indices_list  # Keep for core set initialization
    }
