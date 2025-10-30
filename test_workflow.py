"""
Test Workflow Script for FCG and FCG-DVS
========================================

This script tests the complete workflow with minimal configurations
to verify everything works correctly before running full experiments.

Features:
- Extremely small sample sizes for quick testing
- All parameters configurable
- Tests both FCG and FCG-DVS pipelines
- Detailed logging of each step
- Estimates time and API calls

Usage:
    python test_workflow.py
"""

import os
import sys
from datetime import datetime
from fcg_main import FCGAlgorithm
from llm_utils import init_llm_clients
from logger_utils import get_logger

logger = get_logger()


# ============================================================================
# CONFIGURATION - Adjust these parameters as needed
# ============================================================================

CONFIG = {
    # Reproducibility
    'random_seed': 20,             # Random seed for reproducibility (set to different value for different runs)
    
    # Dataset parameters (KEEP SMALL FOR TESTING)
    'max_train_samples': None,      # Limit training data (None = use all)
    'dev_ratio': 0.2,              # Dev set ratio
    'max_dev_samples': 20,          # Max dev samples per evaluation
    'max_test_samples': 200,        # Max test samples
    
    # FCG Phase 1 parameters
    'n_clusters': 10,               # Number of K-means clusters per subgroup
    'm_neighbors': 10,              # Neighbors per cluster
    'k_shots': 10,                  # Demonstrations per subgroup for FCG
    'iterations': 10,               # Genetic evolution iterations
    'alpha': 0.5,                  # Balance: pred vs fairness
    'p': 0.05,                     # Initial score threshold
    'metric_pred': 'f1_score',     # Performance metric
    'metric_fair': 'ratio_dp',     # Fairness metric for FCG Phase 1
    'model': 'llama-3.3-70b',      # LLM model
    
    # DVS Phase 2 parameters (set test_dvs=True to enable)
    'test_dvs': True,              # Whether to test DVS phase
    'batch_size': 10,               # Test batch size (N)
    'k_neighbors': 10,              # Neighbors to retrieve per test sample
    'L_iterations': 5,               # DVS refinement iterations

    # Testing options
    'save_metrics': True,         # Save metrics to files
    'verbose': True,               # Extra logging
}


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_api_calls(config):
    """
    Estimate total API calls and tokens for the configuration
    
    Token Calculation Method:
    - Zero-shot: ~155 tokens (80 task desc + 60 sample + 10 output + 5 formatting)
    - Few-shot: ~170 + (K × 70) tokens where K = number of demonstrations
      * Each demo: ~70 tokens (60 sample + 5 label + 5 formatting)
    - With k_shots per subgroup × 4 subgroups: K = k_shots × 4
    
    Returns:
        Dictionary with API call counts and token estimates
    """
    
    # Calculate total demonstrations for ICL
    total_demos = config['k_shots'] * 4  # 4 subgroups
    
    # Token formulas based on prompt analysis
    tokens_per_zero_shot = 155
    tokens_per_few_shot = 170 + (total_demos * 70)
    
    # FCG Phase 1: Evolution
    # Baseline: zero-shot on dev set (computed once, cached)
    fcg_baseline_calls = config['max_dev_samples']
    fcg_baseline_tokens = fcg_baseline_calls * tokens_per_zero_shot
    
    # ICL: few-shot on dev set per iteration
    fcg_evolution_calls = config['max_dev_samples'] * config['iterations']
    fcg_evolution_tokens = fcg_evolution_calls * tokens_per_few_shot
    
    # FCG Phase 2: Test Evaluation
    # Zero-shot + Few-shot on test set
    fcg_test_zero_calls = config['max_test_samples']
    fcg_test_zero_tokens = fcg_test_zero_calls * tokens_per_zero_shot
    
    fcg_test_few_calls = config['max_test_samples']
    fcg_test_few_tokens = fcg_test_few_calls * tokens_per_few_shot
    
    fcg_test_calls = fcg_test_zero_calls + fcg_test_few_calls
    fcg_test_tokens = fcg_test_zero_tokens + fcg_test_few_tokens
    
    # Total FCG
    fcg_total_calls = fcg_baseline_calls + fcg_evolution_calls + fcg_test_calls
    fcg_total_tokens = fcg_baseline_tokens + fcg_evolution_tokens + fcg_test_tokens
    
    # DVS Phase (if enabled)
    if config['test_dvs']:
        # DVS Logic:
        # - Total test samples divided into batches of size N (batch_size)
        # - For each batch: N test samples × N candidates = N² predictions
        # - This is repeated for L iterations
        # - Each prediction uses k_neighbors demonstrations
        
        n_batches = (config['max_test_samples'] + config['batch_size'] - 1) // config['batch_size']
        batch_size = config['batch_size']
        
        dvs_demos = config['k_neighbors']
        tokens_per_dvs_call = 170 + (dvs_demos * 70)
        
        # Total calls: n_batches × batch_size × batch_size × L_iterations
        dvs_calls = n_batches * batch_size * batch_size * config['L_iterations']
        dvs_tokens = dvs_calls * tokens_per_dvs_call
    else:
        dvs_calls = 0
        dvs_tokens = 0
    
    # Grand totals
    grand_total_calls = fcg_total_calls + dvs_calls
    grand_total_tokens = fcg_total_tokens + dvs_tokens
    
    # Estimated cost (assuming $0.10 per 1M input tokens, $0.20 per 1M output tokens)
    # Approximate: 95% input, 5% output
    input_tokens = int(grand_total_tokens * 0.95)
    output_tokens = int(grand_total_tokens * 0.05)
    estimated_cost = (input_tokens * 0.10 + output_tokens * 0.20) / 1_000_000
    
    return {
        # FCG Phase breakdown
        'fcg_baseline_calls': fcg_baseline_calls,
        'fcg_baseline_tokens': fcg_baseline_tokens,
        'fcg_evolution_calls': fcg_evolution_calls,
        'fcg_evolution_tokens': fcg_evolution_tokens,
        'fcg_test_calls': fcg_test_calls,
        'fcg_test_tokens': fcg_test_tokens,
        'fcg_total_calls': fcg_total_calls,
        'fcg_total_tokens': fcg_total_tokens,
        
        # DVS Phase
        'dvs_calls': dvs_calls,
        'dvs_tokens': dvs_tokens,
        
        # Grand totals
        'grand_total_calls': grand_total_calls,
        'grand_total_tokens': grand_total_tokens,
        
        # Cost estimate
        'estimated_cost_usd': estimated_cost,
        
        # Token calculation details
        'tokens_per_zero_shot': tokens_per_zero_shot,
        'tokens_per_few_shot': tokens_per_few_shot,
        'total_demonstrations': total_demos
    }



def print_config(config):
    """Pretty print configuration"""
    logger.info("="*70)
    logger.info("TEST CONFIGURATION")
    logger.info("="*70)
    
    logger.info("\nReproducibility:")
    logger.info(f"  random_seed:        {config['random_seed']}")
    
    logger.info("\nDataset Parameters:")
    logger.info(f"  max_train_samples:  {config['max_train_samples']}")
    logger.info(f"  dev_ratio:          {config['dev_ratio']}")
    logger.info(f"  max_dev_samples:    {config['max_dev_samples']}")
    logger.info(f"  max_test_samples:   {config['max_test_samples']}")
    
    logger.info("\nFCG Phase 1 Parameters:")
    logger.info(f"  n_clusters:         {config['n_clusters']}")
    logger.info(f"  m_neighbors:        {config['m_neighbors']}")
    logger.info(f"  k_shots:            {config['k_shots']}")
    logger.info(f"  iterations:         {config['iterations']}")
    logger.info(f"  alpha:              {config['alpha']}")
    logger.info(f"  p:                  {config['p']}")
    logger.info(f"  metric_pred:        {config['metric_pred']}")
    logger.info(f"  metric_fair:        {config['metric_fair']}")
    logger.info(f"  model:              {config['model']}")
    
    if config['test_dvs']:
        logger.info("\nDVS Phase 2 Parameters:")
        logger.info(f"  batch_size:         {config['batch_size']}")
        logger.info(f"  k_neighbors:        {config['k_neighbors']}")
        logger.info(f"  L_iterations:       {config['L_iterations']}")
    
    logger.info("\nTesting Options:")
    logger.info(f"  test_dvs:           {config['test_dvs']}")
    logger.info(f"  save_metrics:       {config['save_metrics']}")
    logger.info(f"  verbose:            {config['verbose']}")
    
    # Estimate API calls and tokens
    estimates = estimate_api_calls(config)
    
    logger.info("\n" + "="*70)
    logger.info("ESTIMATED API USAGE")
    logger.info("="*70)
    
    logger.info("\nAPI Calls Breakdown:")
    logger.info(f"  FCG baseline (zero-shot):  {estimates['fcg_baseline_calls']:>6} calls")
    logger.info(f"  FCG evolution (ICL):       {estimates['fcg_evolution_calls']:>6} calls")
    logger.info(f"  FCG test:                  {estimates['fcg_test_calls']:>6} calls")
    logger.info(f"  FCG Phase Total:           {estimates['fcg_total_calls']:>6} calls")
    
    if config['test_dvs']:
        logger.info(f"  DVS Phase Total:           {estimates['dvs_calls']:>6} calls")
        logger.info(f"  ─────────────────────────────────────")
        logger.info(f"  GRAND TOTAL:               {estimates['grand_total_calls']:>6} calls")
    else:
        logger.info(f"  ─────────────────────────────────────")
        logger.info(f"  GRAND TOTAL:               {estimates['fcg_total_calls']:>6} calls")
    
    logger.info("\nToken Estimates:")
    logger.info(f"  Tokens per zero-shot call: {estimates['tokens_per_zero_shot']:>6} tokens")
    logger.info(f"  Tokens per few-shot call:  {estimates['tokens_per_few_shot']:>6} tokens")
    logger.info(f"    (with {estimates['total_demonstrations']} demonstrations)")
    logger.info(f"")
    logger.info(f"  FCG baseline tokens:       {estimates['fcg_baseline_tokens']:>9,} tokens")
    logger.info(f"  FCG evolution tokens:      {estimates['fcg_evolution_tokens']:>9,} tokens")
    logger.info(f"  FCG test tokens:           {estimates['fcg_test_tokens']:>9,} tokens")
    logger.info(f"  FCG Phase Total:           {estimates['fcg_total_tokens']:>9,} tokens")
    
    if config['test_dvs']:
        logger.info(f"  DVS Phase Total:           {estimates['dvs_tokens']:>9,} tokens")
        logger.info(f"  ─────────────────────────────────────")
        logger.info(f"  GRAND TOTAL:               {estimates['grand_total_tokens']:>9,} tokens")
    else:
        logger.info(f"  ─────────────────────────────────────")
        logger.info(f"  GRAND TOTAL:               {estimates['fcg_total_tokens']:>9,} tokens")
    
    logger.info(f"\nEstimated Cost: ${estimates['estimated_cost_usd']:.4f} USD")
    logger.info(f"  (Based on $0.10/1M input tokens, $0.20/1M output tokens)")
    
    logger.info("="*70)


def save_test_metrics(metrics_dict, phase_name, config):
    """Save metrics to file if enabled"""
    if not config['save_metrics']:
        return
    
    import os
    from datetime import datetime
    
    os.makedirs("res/test", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"res/test/test_{phase_name}_{timestamp}.metrics"
    
    with open(filepath, 'w') as f:
        f.write(f"Test Results: {phase_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*60 + "\n\n")
        
        for key, metrics in metrics_dict.items():
            f.write(f"\n{key.upper()} RESULTS\n")
            f.write("-"*60 + "\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}\n")
            
            f.write(f"\nDemographic Parity:\n")
            f.write(f"  DP_0:      {metrics.get('DP_0', 0):.4f}\n")
            f.write(f"  DP_1:      {metrics.get('DP_1', 0):.4f}\n")
            f.write(f"  Δdp:       {metrics.get('delta_dp', 0):.4f}\n")
            f.write(f"  Rdp:       {metrics.get('ratio_dp', 0):.4f}\n")
            
            f.write(f"\nEqualized Odds:\n")
            f.write(f"  TPR_0:     {metrics.get('TPR_0', 0):.4f}\n")
            f.write(f"  TPR_1:     {metrics.get('TPR_1', 0):.4f}\n")
            f.write(f"  FPR_0:     {metrics.get('FPR_0', 0):.4f}\n")
            f.write(f"  FPR_1:     {metrics.get('FPR_1', 0):.4f}\n")
            f.write(f"  Δeo:       {metrics.get('delta_eo', 0):.4f}\n")
            f.write(f"  Reo:       {metrics.get('ratio_eo', 0):.4f}\n")
            f.write("\n")
    
    logger.info(f"Test metrics saved to: {filepath}")


# ============================================================================
# Main Test Functions
# ============================================================================

def test_fcg_phase(config):
    """Test FCG Phase 1: Global Optimization"""
    logger.info("\n" + "="*70)
    logger.info("TESTING FCG PHASE 1: GLOBAL OPTIMIZATION")
    logger.info("="*70)
    
    start_time = datetime.now()
    
    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(config['random_seed'])
    
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
        max_dev_samples=config['max_dev_samples'],
        random_seed=config['random_seed']
    )
    
    # Load data with sample limit
    logger.info("\nStep 1: Loading data...")
    fcg.load_data(dev_ratio=config['dev_ratio'])
    
    # Limit training data if specified
    if config['max_train_samples'] is not None:
        original_size = len(fcg.train_data)
        fcg.train_data = fcg.train_data.head(config['max_train_samples'])
        logger.info(f"Limited training data: {original_size} -> {len(fcg.train_data)} samples")
        
        # Recreate subgroups with limited data
        from data_utils import create_subgroups, Subgroup
        subgroups_raw = create_subgroups(fcg.train_data, 
                                         sensitive_feature=fcg.sensitive_feature,
                                         label=fcg.label)
        fcg.subgroups = {}
        for key, df in subgroups_raw.items():
            z_val = 1 if '1' in key[1] else 0
            y_val = int(key[-1])
            fcg.subgroups[key] = Subgroup(df, z_val, y_val, initial_score=fcg.p)
            logger.info(f"  {key}: {len(df)} samples")
    
    # Run FCG algorithm
    logger.info("\nStep 2: Running FCG algorithm (clustering + evolution)...")
    fcg.fit()
    
    logger.info("\nStep 3: Selecting top demonstrations...")
    fcg.select_demonstrations(k_per_subgroup=config['k_shots'])
    
    logger.info("\nStep 4: Evaluating FCG...")
    results = fcg.evaluate(max_test_samples=config['max_test_samples'])
    
    # Save metrics if enabled
    if config['save_metrics']:
        save_test_metrics(results, 'fcg', config)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nFCG Phase completed in {elapsed:.1f} seconds")
    
    return fcg, results


def test_dvs_phase(fcg, config):
    """Test DVS Phase 2: Local Adaptive Selection"""
    logger.info("\n" + "="*70)
    logger.info("TESTING DVS PHASE 2: LOCAL ADAPTIVE SELECTION")
    logger.info("="*70)
    
    start_time = datetime.now()
    
    logger.info("\nStep 1: Evaluating with DVS...")
    results = fcg.evaluate_with_dvs(
        max_test_samples=config['max_test_samples'],
        batch_size=config['batch_size'],
        k_neighbors=config['k_neighbors'],
        L_iterations=config['L_iterations'],
        use_fcg_baseline=False  # Already evaluated in Phase 1
    )
    
    # Save metrics if enabled
    if config['save_metrics']:
        save_test_metrics(results, 'dvs', config)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nDVS Phase completed in {elapsed:.1f} seconds")
    
    return results


def run_test_workflow(config=CONFIG):
    """
    Run complete test workflow
    
    Args:
        config: Configuration dictionary (use CONFIG by default)
    """
    logger.info("="*70)
    logger.info("FCG/FCG-DVS WORKFLOW TEST")
    logger.info("Quick verification with minimal configurations")
    logger.info("="*70)
    
    # Print configuration
    print_config(config)
    
    # Check environment
    logger.info("\nChecking environment...")
    try:
        init_llm_clients()
        logger.info("✓ LLM clients initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize LLM clients: {e}")
        logger.error("Please ensure API keys are set in environment variables")
        return None, None
    
    # Check Gemini API if testing DVS
    if config['test_dvs']:
        gemini_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_key:
            logger.warning("⚠ GEMINI_API_KEY not set - DVS phase will be skipped")
            config['test_dvs'] = False
        else:
            logger.info("✓ Gemini API key found")
    
    overall_start = datetime.now()
    
    # Phase 1: FCG
    try:
        fcg, fcg_results = test_fcg_phase(config)
        logger.info("\n✓ FCG Phase 1 completed successfully")
    except Exception as e:
        logger.error(f"\n✗ FCG Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Phase 2: DVS (optional)
    dvs_results = None
    if config['test_dvs']:
        try:
            dvs_results = test_dvs_phase(fcg, config)
            logger.info("\n✓ DVS Phase 2 completed successfully")
        except Exception as e:
            logger.error(f"\n✗ DVS Phase 2 failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue anyway - FCG results are still valid
    
    # Summary
    total_elapsed = (datetime.now() - overall_start).total_seconds()
    
    logger.info("\n" + "="*70)
    logger.info("TEST WORKFLOW COMPLETE")
    logger.info("="*70)
    logger.info(f"\nTotal time: {total_elapsed:.1f} seconds")
    
    logger.info("\nFCG Results Summary:")
    if fcg_results:
        fcg_metrics = fcg_results['fcg']
        zero_metrics = fcg_results['zero_shot']
        logger.info(f"  Zero-shot F1: {zero_metrics['f1_score']:.4f}")
        logger.info(f"  FCG F1:       {fcg_metrics['f1_score']:.4f}")
        logger.info(f"  Improvement:  {fcg_metrics['f1_score'] - zero_metrics['f1_score']:+.4f}")
        logger.info(f"  Zero-shot Δdp: {zero_metrics['delta_dp']:.4f}")
        logger.info(f"  FCG Δdp:       {fcg_metrics['delta_dp']:.4f}")
        logger.info(f"  Improvement:   {zero_metrics['delta_dp'] - fcg_metrics['delta_dp']:+.4f} (lower is better)")
    
    if dvs_results and 'fcg_dvs' in dvs_results:
        logger.info("\nDVS Results Summary:")
        dvs_metrics = dvs_results['fcg_dvs']
        logger.info(f"  FCG-DVS F1:  {dvs_metrics['f1_score']:.4f}")
        logger.info(f"  FCG-DVS Δdp: {dvs_metrics['delta_dp']:.4f}")
        if fcg_results:
            logger.info(f"  vs FCG:      {dvs_metrics['f1_score'] - fcg_metrics['f1_score']:+.4f} (F1)")
            logger.info(f"  vs FCG:      {fcg_metrics['delta_dp'] - dvs_metrics['delta_dp']:+.4f} (Δdp)")
    
    logger.info("\n" + "="*70)
    logger.info("All tests passed! ✓")
    logger.info("You can now run full experiments with larger configurations.")
    logger.info("="*70)
    
    return fcg, {'fcg': fcg_results, 'dvs': dvs_results}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FCG/FCG-DVS WORKFLOW TEST")
    print("="*70)
    print("\nThis script tests the workflow with minimal configurations.")
    print("Edit the CONFIG dictionary at the top of this file to adjust parameters.")
    print("\n💡 TIP: Change 'random_seed' in CONFIG to test with different data splits!")
    print("   Examples: 42 (default), 123, 456, 789, 2024")

    #print config
    print("\nCurrent Test Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # print estimates
    print("\nEstimated Token Usage:")
    results = estimate_api_calls(CONFIG)
    for key, value in results.items():
        print(f"  {key}: {value}")

    print("\nPress Enter to start the test, or Ctrl+C to cancel...")    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(0)
    
    # Run the test
    fcg_instance, results = run_test_workflow(CONFIG)
    
    if fcg_instance is not None:
        print("\n✓ Test completed successfully!")
        print(f"Check the log file in res/ for detailed output.")
        print(f"\n💡 To run with different random seed, edit CONFIG['random_seed'] and run again!")
    else:
        print("\n✗ Test failed. Check the log file for details.")
        sys.exit(1)
