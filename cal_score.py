"""
CAL_SCORE Function - Calculate Evolution Score based on prediction and fairness improvements
"""
import numpy as np
from metrics import calculate_performance_metrics, calculate_equalized_odds
from llm_utils import llm_predict_zero_shot, llm_predict_few_shot
from logger_utils import get_logger

logger = get_logger()


def cal_score(selected_samples, dev_data, metric_pred='f1_score', metric_fair='ratio_eo', 
              alpha=0.5, p=0.05, model="llama-3.1-8b-instant", max_dev_samples=50):
    """
    Calculate evolution score based on prediction and fairness improvements
    
    Algorithm:
    1. Get base LLM predictions (zero-shot) on dev set
    2. Get ICL predictions with selected demonstrations
    3. Calculate improvement deltas
    4. Combine with weighted sum
    
    Args:
        selected_samples: DataFrame with selected demonstration samples
        dev_data: Development dataset (DataFrame)
        metric_pred: Performance metric to use ('f1_score', 'accuracy', etc.)
        metric_fair: Fairness metric to use ('ratio_eo', 'ratio_dp', 'delta_eo', 'delta_dp')
        alpha: Balance coefficient (0.5 = equal weight)
        p: Minimum score threshold
        model: LLM model name
        max_dev_samples: Maximum dev samples to use (for speed)
    
    Returns:
        Evolution score (float)
    """
    # Limit dev samples for efficiency
    if len(dev_data) > max_dev_samples:
        dev_subset = dev_data.sample(n=max_dev_samples, random_state=42)
    else:
        dev_subset = dev_data
    
    # Extract true labels and sensitive attributes
    y_dev = dev_subset['income'].values
    z_dev = dev_subset['sex'].values
    
    # STEP 1: Get baseline (zero-shot) predictions
    logger.info("    Computing baseline (zero-shot) predictions...")
    y_base = llm_predict_zero_shot(dev_subset, model=model, max_samples=max_dev_samples)
    
    # Calculate baseline metrics
    from metrics import evaluate_all_metrics
    base_metrics = evaluate_all_metrics(y_dev, y_base, z_dev)
    base_pred = base_metrics.get(metric_pred, 0.0)
    base_fair = base_metrics.get(metric_fair, 0.0)
    
    # STEP 2: Get ICL predictions with demonstrations
    logger.info("    Computing ICL predictions with demonstrations...")
    y_icl = llm_predict_few_shot(dev_subset, demonstrations=selected_samples, 
                                  model=model, max_samples=max_dev_samples)
    
    # Calculate ICL metrics
    icl_metrics = evaluate_all_metrics(y_dev, y_icl, z_dev)
    icl_pred = icl_metrics.get(metric_pred, 0.0)
    icl_fair = icl_metrics.get(metric_fair, 0.0)
    
    # STEP 3: Calculate improvements
    delta_pred = max(icl_pred - base_pred, p)
    delta_fair = max(icl_fair - base_fair, p)
    
    # STEP 4: Combine with weighted sum
    evol_score = alpha * delta_pred + (1 - alpha) * delta_fair
    
    logger.info(f"    Base: {metric_pred}={base_pred:.4f}, {metric_fair}={base_fair:.4f}")
    logger.info(f"    ICL:  {metric_pred}={icl_pred:.4f}, {metric_fair}={icl_fair:.4f}")
    logger.info(f"    Δ:    Δpred={delta_pred:.4f}, Δfair={delta_fair:.4f}")
    
    return evol_score


def cal_score_cached(selected_samples, dev_data, baseline_cache=None, 
                    metric_pred='f1_score', metric_fair='ratio_eo',
                    alpha=0.5, p=0.05, model="llama-3.1-8b-instant", max_dev_samples=50):
    """
    Optimized version that caches baseline predictions to avoid redundant API calls
    
    Args:
        selected_samples: DataFrame with selected demonstration samples
        dev_data: Development dataset
        baseline_cache: Dict containing cached baseline predictions and metrics
        (other args same as cal_score)
    
    Returns:
        Evolution score (float)
    """
    # Limit dev samples for efficiency
    if len(dev_data) > max_dev_samples:
        dev_subset = dev_data.sample(n=max_dev_samples, random_state=42)
    else:
        dev_subset = dev_data
    
    # Extract true labels and sensitive attributes
    y_dev = dev_subset['income'].values
    z_dev = dev_subset['sex'].values
    
    # Get or compute baseline
    if baseline_cache is None or 'base_pred' not in baseline_cache:
        logger.info("    Computing baseline (zero-shot) predictions...")
        y_base = llm_predict_zero_shot(dev_subset, model=model, max_samples=max_dev_samples)
        
        from metrics import evaluate_all_metrics
        base_metrics = evaluate_all_metrics(y_dev, y_base, z_dev)
        base_pred = base_metrics.get(metric_pred, 0.0)
        base_fair = base_metrics.get(metric_fair, 0.0)
        
        # Update cache
        if baseline_cache is not None:
            baseline_cache['y_base'] = y_base
            baseline_cache['base_metrics'] = base_metrics
            baseline_cache['base_pred'] = base_pred
            baseline_cache['base_fair'] = base_fair
    else:
        # Use cached baseline
        base_pred = baseline_cache['base_pred']
        base_fair = baseline_cache['base_fair']
    
    # Get ICL predictions
    logger.info("    Computing ICL predictions with demonstrations...")
    y_icl = llm_predict_few_shot(dev_subset, demonstrations=selected_samples,
                                  model=model, max_samples=max_dev_samples)
    
    # Calculate ICL metrics
    from metrics import evaluate_all_metrics
    icl_metrics = evaluate_all_metrics(y_dev, y_icl, z_dev)
    icl_pred = icl_metrics.get(metric_pred, 0.0)
    icl_fair = icl_metrics.get(metric_fair, 0.0)
    
    # Calculate improvements
    delta_pred = max(icl_pred - base_pred, p)
    delta_fair = max(icl_fair - base_fair, p)
    
    # Combine
    evol_score = alpha * delta_pred + (1 - alpha) * delta_fair
    
    logger.info(f"    Base: {metric_pred}={base_pred:.4f}, {metric_fair}={base_fair:.4f}")
    logger.info(f"    ICL:  {metric_pred}={icl_pred:.4f}, {metric_fair}={icl_fair:.4f}")
    logger.info(f"    Δ:    Δpred={delta_pred:.4f}, Δfair={delta_fair:.4f}")
    
    return evol_score
