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
from fcg_main import FCGAlgorithm
from llm_utils import init_llm_clients
from logger_utils import get_logger

logger = get_logger()


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
        'metric_fair': 'ratio_eo',
        'model': 'llama-3.3-70b',
        'max_dev_samples': 40,  # Dev samples for FCG evolution
        
        # DVS Phase 2 parameters
        'max_test_samples': 100,  # Total test samples
        'batch_size': 10,         # Test batch size (N)
        'k_neighbors': 5,         # Neighbors to retrieve per test sample
        'L_iterations': 5,        # DVS refinement iterations
        'use_fcg_baseline': True  # Also evaluate standard FCG for comparison
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
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: FCG GLOBAL FAIRNESS OPTIMIZATION")
    logger.info("="*70)
    
    # Load data
    fcg.load_data(dev_ratio=0.2)
    
    # Run FCG algorithm (clustering + genetic evolution)
    fcg.fit()
    
    # Select top demonstrations (this becomes H_opt - the globally optimized pool)
    fcg.select_demonstrations(k_per_subgroup=config['k_shots'])
    
    logger.info(f"\nPhase 1 Complete: Created optimized pool H_opt with {len(fcg.top_demonstrations)} samples")
    
    # Phase 2: DVS Local Adaptive Selection + Evaluation
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: DVS LOCAL ADAPTIVE SELECTION")
    logger.info("="*70)
    
    results = fcg.evaluate_with_dvs(
        max_test_samples=config['max_test_samples'],
        batch_size=config['batch_size'],
        k_neighbors=config['k_neighbors'],
        L_iterations=config['L_iterations'],
        use_fcg_baseline=config['use_fcg_baseline']
    )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)
    
    # logger.info("\nAPI Call Estimate:")
    # n_batches = (config['max_test_samples'] + config['batch_size'] - 1) // config['batch_size']
    # P = config['batch_size']  # DVS size
    # L = config['L_iterations']
    # N = config['batch_size']  # Test samples per batch
    
    # # DVS calls: For each batch: L iterations * (P DVS samples * ~|C| avg) + N test predictions
    # # Rough estimate assuming avg |C| ~ 11
    # calls_per_batch = L * P * 11 + N
    # total_calls = n_batches * calls_per_batch
    
    # logger.info(f"  Batches: {n_batches}")
    # logger.info(f"  Estimated calls per batch: ~{calls_per_batch}")
    # logger.info(f"  Total estimated calls: ~{total_calls}")
    
    # if config['use_fcg_baseline']:
    #     fcg_calls = config['max_test_samples']
    #     logger.info(f"  FCG baseline calls: {fcg_calls}")
    #     logger.info(f"  Grand total: ~{total_calls + fcg_calls}")
    
    return fcg, results


if __name__ == "__main__":
    fcg_instance, experiment_results = run_fcg_dvs_experiment()
    
    print("\n" + "="*70)
    print("Results saved. Check the log file in res/ for detailed output.")
    print("="*70)
