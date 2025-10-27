"""
Example: Running FCG-DVS Hybrid Approach

This demonstrates the two-phase FCG-DVS algorithm:
1. Phase 1 (FCG): Global fairness optimization to create an optimized pool
2. Phase 2 (DVS): Local batch-specific adaptive ICE selection

Setup:
1. Install dependencies:
   pip install -r requirements.txt

2. Set environment variables:
   export CEREBRAS_API_KEY_safari="your_key_1"
   export CEREBRAS_API_KEY_firefox="your_key_2"
   export CEREBRAS_API_KEY_brave="your_key_3"
   export GEMINI_API_KEY="your_gemini_key"

3. Run this script:
   python run_fcg_dvs.py
"""
import os
from datetime import datetime
from fcg_main import FCGAlgorithm
from llm_utils import init_llm_clients
from logger_utils import get_logger
from metrics import print_metrics

logger = get_logger()


def save_metrics_to_file(metrics_dict, filename_suffix):
    """
    Save metrics to a file in res/save/ directory
    
    Args:
        metrics_dict: Dictionary of metrics (e.g., {'fcg': metrics, 'zero_shot': metrics})
        filename_suffix: Suffix for the filename (e.g., '_withoutDVS' or '_withDVS')
    """
    # Create res/save directory if it doesn't exist
    os.makedirs("res/save", exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"res/save/file_{timestamp}{filename_suffix}.metrics"
    
    # Write metrics to file
    with open(filepath, 'w') as f:
        for key, metrics in metrics_dict.items():
            # Write section header
            f.write(f"\n{'='*60}\n")
            f.write(f"{key.upper()} RESULTS\n")
            f.write(f"{'='*60}\n\n")
            
            # Performance Metrics
            f.write(f"Performance Metrics:\n")
            f.write(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}\n")
            
            # Demographic Parity
            f.write(f"\nDemographic Parity:\n")
            f.write(f"  DP_0:      {metrics.get('DP_0', 0):.4f}\n")
            f.write(f"  DP_1:      {metrics.get('DP_1', 0):.4f}\n")
            f.write(f"  Δdp:       {metrics.get('delta_dp', 0):.4f}\n")
            f.write(f"  Rdp:       {metrics.get('ratio_dp', 0):.4f}\n")
            
            # Equalized Odds
            f.write(f"\nEqualized Odds:\n")
            f.write(f"  TPR_0:     {metrics.get('TPR_0', 0):.4f}\n")
            f.write(f"  TPR_1:     {metrics.get('TPR_1', 0):.4f}\n")
            f.write(f"  FPR_0:     {metrics.get('FPR_0', 0):.4f}\n")
            f.write(f"  FPR_1:     {metrics.get('FPR_1', 0):.4f}\n")
            f.write(f"  Δeo:       {metrics.get('delta_eo', 0):.4f}\n")
            f.write(f"  Reo:       {metrics.get('ratio_eo', 0):.4f}\n")
            f.write(f"\n")
    
    logger.info(f"Metrics saved to: {filepath}")
    return filepath


def run_fcg_dvs_experiment():
    """
    Run complete FCG-DVS hybrid experiment
    """
    logger.info("="*70)
    logger.info("FCG-DVS HYBRID ALGORITHM EXPERIMENT")
    logger.info("Phase 1: FCG Global Optimization | Phase 2: DVS Local Adaptation")
    logger.info("="*70)
    
    # Initialize LLM clients (multiple API keys for rotation)
    try:
        init_llm_clients()
    except Exception as e:
        logger.warning(f"Failed to initialize LLM clients: {e}")
    
    # Configuration
    config = {
        # FCG Phase 1 parameters
        'n_clusters': 8,        # Number of K-means clusters per subgroup
        'm_neighbors': 5,       # Neighbors per cluster
        'k_shots': 5,           # Demonstrations per subgroup for FCG
        'iterations': 10,       # Genetic evolution iterations
        'alpha': 0.5,           # Balance: pred vs fairness
        'p': 0.05,              # Initial score threshold
        'metric_pred': 'f1_score',
        'metric_fair': 'ratio_dp',  # For FCG Phase 1 only (DVS Phase 2 uses demographic parity difference)
        'model': 'llama-3.3-70b',
        'max_dev_samples': 40,  # Dev samples for FCG evolution
        
        # DVS Phase 2 parameters
        'max_test_samples': 100,  # Total test samples
        'batch_size': 10,         # Test batch size (N)
        'k_neighbors': 5,         # Neighbors to retrieve per test sample
        'L_iterations': 5         # DVS refinement iterations
    }
    
    logger.info("\nConfiguration:")
    for key, val in config.items():
        logger.info(f"  {key}: {val}")
    
    # Initialize FCG algorithm
    fcg = FCGAlgorithm(
        n_clusters=config['n_clusters'],
        m_neighbors=config['m_neighbors'],
        k_shots=config['k_shots'],
        iterations=config['iterations'],
        alpha=config['alpha'],
        p=config['p'],
        metric_pred=config['metric_pred'],
        metric_fair=config['metric_fair'],
        model=config['model'],
        max_dev_samples=config['max_dev_samples']
    )
    
    # Phase 1: FCG Global Optimization
    logger.info("="*70)
    logger.info("PHASE 1: FCG GLOBAL FAIRNESS OPTIMIZATION")
    logger.info("="*70)
    
    # Load data
    fcg.load_data(dev_ratio=0.2)
    
    # Run FCG algorithm (clustering + genetic evolution)
    fcg.fit()
    
    # Select top demonstrations (this becomes H_opt - the globally optimized pool)
    fcg.select_demonstrations(k_per_subgroup=config['k_shots'])
    
    logger.info(f"\nPhase 1 Complete: Created optimized pool H_opt with {len(fcg.top_demonstrations)} samples")
    
    # Evaluate FCG without DVS (baseline comparison)
    logger.info("="*70)
    logger.info("EVALUATING FCG WITHOUT DVS (Baseline)")
    logger.info("="*70)
    
    fcg_without_dvs_results = fcg.evaluate(max_test_samples=config['max_test_samples'])
    
    # Save FCG without DVS metrics
    save_metrics_to_file(fcg_without_dvs_results, '_withoutDVS')
    
    # Phase 2: DVS Local Adaptive Selection + Evaluation
    logger.info("="*70)
    logger.info("PHASE 2: DVS LOCAL ADAPTIVE SELECTION")
    logger.info("="*70)
    
    results = fcg.evaluate_with_dvs(
        max_test_samples=config['max_test_samples'],
        batch_size=config['batch_size'],
        k_neighbors=config['k_neighbors'],
        L_iterations=config['L_iterations'],
        use_fcg_baseline=False  # Don't duplicate FCG evaluation, we already did it above
    )
    
    # Save FCG with DVS metrics
    save_metrics_to_file(results, '_withDVS')
    
    # Summary
    logger.info("="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)
    
    # Compare FCG without DVS vs with DVS
    logger.info("="*70)
    logger.info("COMPARISON: FCG vs FCG-DVS")
    logger.info("="*70)
    
    fcg_metrics = fcg_without_dvs_results['fcg']
    dvs_metrics = results['fcg_dvs']
    
    logger.info("\nPerformance Comparison:")
    logger.info(f"  Accuracy:  FCG={fcg_metrics['accuracy']:.4f} | FCG-DVS={dvs_metrics['accuracy']:.4f} | Δ={dvs_metrics['accuracy']-fcg_metrics['accuracy']:+.4f}")
    logger.info(f"  F1-Score:  FCG={fcg_metrics['f1_score']:.4f} | FCG-DVS={dvs_metrics['f1_score']:.4f} | Δ={dvs_metrics['f1_score']-fcg_metrics['f1_score']:+.4f}")
    
    logger.info("\nFairness Comparison (Demographic Parity):")
    logger.info(f"  Δdp:       FCG={fcg_metrics['delta_dp']:.4f} | FCG-DVS={dvs_metrics['delta_dp']:.4f} | Δ={fcg_metrics['delta_dp']-dvs_metrics['delta_dp']:+.4f} (lower is better)")
    logger.info(f"  Rdp:       FCG={fcg_metrics['ratio_dp']:.4f} | FCG-DVS={dvs_metrics['ratio_dp']:.4f} | Δ={dvs_metrics['ratio_dp']-fcg_metrics['ratio_dp']:+.4f}")
    
    logger.info("\nFairness Comparison (Equalized Odds):")
    logger.info(f"  Δeo:       FCG={fcg_metrics['delta_eo']:.4f} | FCG-DVS={dvs_metrics['delta_eo']:.4f} | Δ={fcg_metrics['delta_eo']-dvs_metrics['delta_eo']:+.4f} (lower is better)")
    logger.info(f"  Reo:       FCG={fcg_metrics['ratio_eo']:.4f} | FCG-DVS={dvs_metrics['ratio_eo']:.4f} | Δ={dvs_metrics['ratio_eo']-fcg_metrics['ratio_eo']:+.4f}")
    
    # Return both results
    return fcg, {'without_dvs': fcg_without_dvs_results, 'with_dvs': results}


if __name__ == "__main__":
    fcg_instance, experiment_results = run_fcg_dvs_experiment()
    
    print("="*70)
    print("Results saved. Check the log file in res/ for detailed output.")
    print("="*70)
