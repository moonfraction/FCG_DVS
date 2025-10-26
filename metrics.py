"""
Evaluation Metrics for Fairness and Performance
Includes: Accuracy, Precision, Recall, F-score, Demographic Parity, Equalized Odds
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


def calculate_performance_metrics(y_true, y_pred):
    """
    Calculate predictive performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with accuracy, precision, recall, f1_score
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    return metrics


def calculate_demographic_parity(y_pred, z_sensitive):
    """
    Calculate Demographic Parity metrics
    
    DP_z = P(f(x)=1|Z=z)
    Δdp = |DP_1 - DP_0|
    Rdp = DP_0 / DP_1
    
    Args:
        y_pred: Predicted labels
        z_sensitive: Sensitive attribute values (0 or 1)
    
    Returns:
        Dictionary with DP_0, DP_1, delta_dp, ratio_dp
    """
    y_pred = np.array(y_pred)
    z_sensitive = np.array(z_sensitive)
    
    # Calculate P(f(x)=1|Z=0)
    z0_mask = (z_sensitive == 0)
    dp_0 = np.mean(y_pred[z0_mask]) if np.sum(z0_mask) > 0 else 0.0
    
    # Calculate P(f(x)=1|Z=1)
    z1_mask = (z_sensitive == 1)
    dp_1 = np.mean(y_pred[z1_mask]) if np.sum(z1_mask) > 0 else 0.0
    
    # Calculate delta and ratio
    delta_dp = abs(dp_1 - dp_0)
    ratio_dp = dp_0 / (dp_1 + 1e-10)  # Add epsilon to avoid division by zero
    
    return {
        'DP_0': dp_0,
        'DP_1': dp_1,
        'delta_dp': delta_dp,
        'ratio_dp': ratio_dp
    }


def calculate_equalized_odds(y_true, y_pred, z_sensitive):
    """
    Calculate Equalized Odds metrics
    
    TPR_z = P(f(x)=1|y=1,Z=z)
    FPR_z = P(f(x)=1|y=0,Z=z)
    Δeo = max(|TPR_1-TPR_0|, |FPR_1-FPR_0|)
    Reo = min(TPR_0/(TPR_1+ε), FPR_0/(FPR_1+ε))
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        z_sensitive: Sensitive attribute values (0 or 1)
    
    Returns:
        Dictionary with TPR_0, TPR_1, FPR_0, FPR_1, delta_eo, ratio_eo
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    z_sensitive = np.array(z_sensitive)
    
    epsilon = 1e-10
    
    # Calculate TPR and FPR for Z=0
    z0_mask = (z_sensitive == 0)
    z0_y1_mask = z0_mask & (y_true == 1)
    z0_y0_mask = z0_mask & (y_true == 0)
    
    tpr_0 = np.mean(y_pred[z0_y1_mask]) if np.sum(z0_y1_mask) > 0 else 0.0
    fpr_0 = np.mean(y_pred[z0_y0_mask]) if np.sum(z0_y0_mask) > 0 else 0.0
    
    # Calculate TPR and FPR for Z=1
    z1_mask = (z_sensitive == 1)
    z1_y1_mask = z1_mask & (y_true == 1)
    z1_y0_mask = z1_mask & (y_true == 0)
    
    tpr_1 = np.mean(y_pred[z1_y1_mask]) if np.sum(z1_y1_mask) > 0 else 0.0
    fpr_1 = np.mean(y_pred[z1_y0_mask]) if np.sum(z1_y0_mask) > 0 else 0.0
    
    # Calculate delta and ratio
    delta_eo = max(abs(tpr_1 - tpr_0), abs(fpr_1 - fpr_0))
    ratio_eo = min(tpr_0 / (tpr_1 + epsilon), fpr_0 / (fpr_1 + epsilon))
    
    return {
        'TPR_0': tpr_0,
        'TPR_1': tpr_1,
        'FPR_0': fpr_0,
        'FPR_1': fpr_1,
        'delta_eo': delta_eo,
        'ratio_eo': ratio_eo
    }


def evaluate_all_metrics(y_true, y_pred, z_sensitive):
    """
    Calculate all performance and fairness metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        z_sensitive: Sensitive attribute values
    
    Returns:
        Dictionary with all metrics
    """
    # Performance metrics
    perf_metrics = calculate_performance_metrics(y_true, y_pred)
    
    # Fairness metrics
    dp_metrics = calculate_demographic_parity(y_pred, z_sensitive)
    eo_metrics = calculate_equalized_odds(y_true, y_pred, z_sensitive)
    
    # Combine all metrics
    all_metrics = {**perf_metrics, **dp_metrics, **eo_metrics}
    
    return all_metrics


def print_metrics(metrics, prefix=""):
    """
    Pretty print metrics
    """
    print(f"\n{prefix}Performance Metrics:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
    
    print(f"\n{prefix}Demographic Parity:")
    print(f"  DP_0:      {metrics.get('DP_0', 0):.4f}")
    print(f"  DP_1:      {metrics.get('DP_1', 0):.4f}")
    print(f"  Δdp:       {metrics.get('delta_dp', 0):.4f}")
    print(f"  Rdp:       {metrics.get('ratio_dp', 0):.4f}")
    
    print(f"\n{prefix}Equalized Odds:")
    print(f"  TPR_0:     {metrics.get('TPR_0', 0):.4f}")
    print(f"  TPR_1:     {metrics.get('TPR_1', 0):.4f}")
    print(f"  FPR_0:     {metrics.get('FPR_0', 0):.4f}")
    print(f"  FPR_1:     {metrics.get('FPR_1', 0):.4f}")
    print(f"  Δeo:       {metrics.get('delta_eo', 0):.4f}")
    print(f"  Reo:       {metrics.get('ratio_eo', 0):.4f}")
