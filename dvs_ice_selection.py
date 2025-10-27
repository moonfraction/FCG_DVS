"""
DVS In-Context Example Selection (Algorithm 2)
Iterative search for optimal ICE set using individual and total error
"""
import numpy as np
import pandas as pd
from llm_utils import llm_predict_few_shot
from metrics import evaluate_all_metrics
from logger_utils import get_logger

logger = get_logger()


def compute_error(y_true, y_pred, z_sensitive, alpha=0.5, metric_pred='f1_score', metric_fair='ratio_eo'):
    """
    Compute combined prediction + fairness error
    
    ERROR = alpha * (1 - pred_metric) + (1 - alpha) * (1 - fair_metric)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        z_sensitive: Sensitive attribute values
        alpha: Balance coefficient
        metric_pred: Performance metric name
        metric_fair: Fairness metric name
    
    Returns:
        Combined error score (lower is better)
    """
    metrics = evaluate_all_metrics(y_true, y_pred, z_sensitive)
    
    pred_score = metrics.get(metric_pred, 0.0)
    fair_score = metrics.get(metric_fair, 0.0)
    
    # Convert to error (1 - score)
    pred_error = 1.0 - pred_score
    fair_error = 1.0 - fair_score
    
    # Combine
    total_error = alpha * pred_error + (1 - alpha) * fair_error
    
    return total_error


def compute_individual_error(dvs_df, core_set_df, model="llama-3.3-70b", alpha=0.5, 
                            metric_pred='f1_score', metric_fair='ratio_eo', start_client_idx=0):
    """
    Compute individual error I_{n,j} for a single core set on the DVS
    
    Args:
        dvs_df: Dynamic Validation Set (P_n)
        core_set_df: Core set C_{n,j} for a single test example
        model: LLM model name
        alpha: Balance coefficient
        metric_pred: Performance metric
        metric_fair: Fairness metric
        start_client_idx: Starting API key index
    
    Returns:
        Individual error score
    """
    if len(dvs_df) == 0:
        return 1.0  # Maximum error if no DVS
    
    # Get true labels and sensitive attributes from DVS
    y_true = dvs_df['income'].values
    z_sensitive = dvs_df['sex'].values
    
    # Predict on DVS using this core set
    y_pred = llm_predict_few_shot(
        dvs_df,
        demonstrations=core_set_df,
        model=model,
        max_samples=len(dvs_df),
        start_client_idx=start_client_idx
    )
    
    # Compute error
    error = compute_error(y_true, y_pred, z_sensitive, alpha, metric_pred, metric_fair)
    
    return error


def compute_total_error(dvs_df, combined_ice_df, model="llama-3.3-70b", alpha=0.5,
                       metric_pred='f1_score', metric_fair='ratio_eo', start_client_idx=0):
    """
    Compute total error T_n for the combined ICE set on the DVS
    
    Args:
        dvs_df: Dynamic Validation Set (P_n)
        combined_ice_df: Combined ICE set C_n (union of all core sets)
        model: LLM model name
        alpha: Balance coefficient
        metric_pred: Performance metric
        metric_fair: Fairness metric
        start_client_idx: Starting API key index
    
    Returns:
        Total error score
    """
    return compute_individual_error(dvs_df, combined_ice_df, model, alpha, metric_pred, metric_fair, start_client_idx)


def select_ice_iterative(batch_df, dvs_df, candidate_df, support_indices_list, pool_df,
                         L=5, model="llama-3.3-70b", alpha=0.5,
                         metric_pred='f1_score', metric_fair='ratio_eo', start_client_idx=0):
    """
    Iterative ICE selection using DVS (Algorithm 2)
    
    Args:
        batch_df: Test batch (B_n)
        dvs_df: Dynamic Validation Set (P_n)
        candidate_df: Candidate pool (H_n)
        support_indices_list: List of k neighbors per test sample (from support_info)
        pool_df: Full FCG-optimized pool (for indexing)
        L: Number of iterations
        model: LLM model name
        alpha: Balance coefficient
        metric_pred: Performance metric
        metric_fair: Fairness metric
        start_client_idx: Starting API key index
    
    Returns:
        dict with:
            - 'best_ice_df': Final ICE set (C_n^min)
            - 'best_error': Minimum total error
            - 'iteration_errors': List of errors per iteration
    """
    n_test = len(batch_df)
    
    # Initialize core sets: C_{n,j}^{(0)} = {x_{n,j,2}} (second closest neighbor)
    core_sets = []  # List of DataFrames, one per test sample
    for j, neighbors in enumerate(support_indices_list):
        if len(neighbors) >= 2:
            # Use second closest (index 1 in neighbors list)
            init_sample = pool_df.iloc[[neighbors[1]]].reset_index(drop=True)
        elif len(neighbors) == 1:
            # Fallback: use the only neighbor
            init_sample = pool_df.iloc[[neighbors[0]]].reset_index(drop=True)
        else:
            # No neighbors: use empty DataFrame
            init_sample = pd.DataFrame()
        
        core_sets.append({
            'df': init_sample,
            'neighbors': neighbors,  # Available neighbors for expansion
            'next_idx': 2  # Next neighbor index to add (start from 2, since 0 is DVS, 1 is initial)
        })
    
    # Track best configuration
    best_ice_df = pd.concat([cs['df'] for cs in core_sets], ignore_index=True) if any(len(cs['df']) > 0 for cs in core_sets) else pd.DataFrame()
    best_error = float('inf')
    iteration_errors = []
    
    logger.info(f"Starting iterative ICE selection for {n_test} test samples, {L} iterations")
    
    # Iterate for L rounds
    for l in range(L):
        logger.info(f"  Iteration {l + 1}/{L}:")
        
        # Step 1: Compute individual errors for each core set
        individual_errors = []
        for j, core_set_info in enumerate(core_sets):
            if len(core_set_info['df']) == 0:
                # Empty core set: assign maximum error
                i_error = 1.0
            else:
                i_error = compute_individual_error(
                    dvs_df,
                    core_set_info['df'],
                    model=model,
                    alpha=alpha,
                    metric_pred=metric_pred,
                    metric_fair=metric_fair,
                    start_client_idx=start_client_idx
                )
            individual_errors.append(i_error)
            logger.debug(f"    Test sample {j}: I_error = {i_error:.4f}, |C| = {len(core_set_info['df'])}")
        
        # Step 2: Combine all core sets to get C_n^{(l)}
        current_ice_df = pd.concat([cs['df'] for cs in core_sets if len(cs['df']) > 0], ignore_index=True)
        
        # Step 3: Compute total error T_n
        if len(current_ice_df) == 0:
            t_error = 1.0
        else:
            t_error = compute_total_error(
                dvs_df,
                current_ice_df,
                model=model,
                alpha=alpha,
                metric_pred=metric_pred,
                metric_fair=metric_fair,
                start_client_idx=start_client_idx
            )
        
        iteration_errors.append(t_error)
        logger.info(f"    Total error T_n = {t_error:.4f}, |C_n| = {len(current_ice_df)}")
        
        # Update best if this is better
        if t_error < best_error:
            best_error = t_error
            best_ice_df = current_ice_df.copy()
            logger.info(f"    New best error: {best_error:.4f}")
        
        # Step 4: Find weakest link (j* with max individual error)
        j_star = int(np.argmax(individual_errors))
        max_i_error = individual_errors[j_star]
        logger.info(f"    Weakest link: test sample {j_star} with I_error = {max_i_error:.4f}")
        
        # Step 5: Expand the weakest core set
        weakest_core = core_sets[j_star]
        next_idx = weakest_core['next_idx']
        neighbors = weakest_core['neighbors']
        
        if next_idx < len(neighbors):
            # Add the next neighbor to this core set
            new_sample_idx = neighbors[next_idx]
            new_sample = pool_df.iloc[[new_sample_idx]].reset_index(drop=True)
            
            if len(weakest_core['df']) == 0:
                weakest_core['df'] = new_sample
            else:
                weakest_core['df'] = pd.concat([weakest_core['df'], new_sample], ignore_index=True)
            
            weakest_core['next_idx'] += 1
            logger.info(f"    Expanded core set {j_star}: added neighbor index {new_sample_idx}, new size = {len(weakest_core['df'])}")
        else:
            logger.info(f"    Core set {j_star} has no more neighbors to add")
    
    logger.info(f"  Best ICE set: |C_n^min| = {len(best_ice_df)}, error = {best_error:.4f}")
    
    return {
        'best_ice_df': best_ice_df,
        'best_error': best_error,
        'iteration_errors': iteration_errors
    }
