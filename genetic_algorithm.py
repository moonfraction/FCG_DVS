"""
Genetic Algorithm with Roulette Wheel Selection
"""
import numpy as np
from logger_utils import get_logger

logger = get_logger()


def roulette_wheel_selection(subgroup, k_shots):
    """
    Select k samples using roulette wheel selection based on their scores
    
    Args:
        subgroup: Subgroup object with samples and scores
        k_shots: Number of samples to select
    
    Returns:
        List of selected indices
    """
    n_samples = len(subgroup)
    
    if n_samples == 0:
        return []
    
    # Limit k_shots to available samples
    k_shots = min(k_shots, n_samples)
    
    # Get scores (fitness values)
    scores = subgroup.scores
    
    # Normalize scores to create probability distribution
    # Add small epsilon to avoid division by zero
    total_score = np.sum(scores) + 1e-10
    probabilities = scores / total_score
    
    # Select k samples based on probability distribution (without replacement)
    selected_indices = np.random.choice(
        n_samples, 
        size=k_shots, 
        replace=False, 
        p=probabilities
    )
    
    return selected_indices.tolist()


def genetic_algorithm(subgroup, k_shots=5, iterations=10, cal_score_func=None, 
                     dev_data=None, alpha=0.5, p=0.05):
    """
    Run genetic algorithm to evolve demonstration samples
    
    Args:
        subgroup: Subgroup object
        k_shots: Number of demonstration samples to select per iteration
        iterations: Number of evolution iterations
        cal_score_func: Function to calculate evolution score
        dev_data: Development dataset for evaluation
        alpha: Balance between prediction and fairness (0.5 = equal weight)
        p: Minimum score threshold
    
    Returns:
        Updated subgroup with evolved scores
    """
    if cal_score_func is None:
        raise ValueError("cal_score_func must be provided")
    
    for iteration in range(iterations):
        # Select k demonstration samples using roulette wheel
        selected_indices = roulette_wheel_selection(subgroup, k_shots)
        
        if len(selected_indices) == 0:
            continue
        
        # Get selected samples
        selected_samples = subgroup.df.iloc[selected_indices]
        
        # Calculate evolution score for these samples
        evol_score = cal_score_func(
            selected_samples, 
            dev_data, 
            alpha=alpha, 
            p=p
        )
        
        # Update scores for selected samples
        subgroup.update_scores(selected_indices, evol_score)
        
    logger.info(f"  Iteration {iteration + 1}/{iterations}: EvolScore = {evol_score:.4f}")
    
    return subgroup


def select_top_demonstrations(subgroup, k_shots=5):
    """
    Select top k samples with highest scores from subgroup
    
    Args:
        subgroup: Subgroup object with evolved scores
        k_shots: Number of top samples to select
    
    Returns:
        DataFrame with top k samples
    """
    # Get indices of top k scores
    top_indices = np.argsort(subgroup.scores)[-k_shots:][::-1]
    
    # Return top samples
    return subgroup.df.iloc[top_indices]
