"""
Advanced Usage Examples for FCG Algorithm
"""

# ============================================================
# EXAMPLE 1: Basic Usage with Default Parameters
# ============================================================

def example_basic():
    """Run FCG with default parameters"""
    from fcg_main import run_fcg_experiment
    
    print("Example 1: Basic Usage\n")
    
    fcg, results = run_fcg_experiment()
    
    print("\nResults Summary:")
    print(f"F1-Score Improvement: {results['fcg']['f1_score'] - results['zero_shot']['f1_score']:+.4f}")
    print(f"Fairness Improvement (Δeo): {results['zero_shot']['delta_eo'] - results['fcg']['delta_eo']:+.4f}")


# ============================================================
# EXAMPLE 2: Custom Parameters
# ============================================================

def example_custom_params():
    """Run FCG with custom hyperparameters"""
    from fcg_main import FCGAlgorithm
    
    print("Example 2: Custom Parameters\n")
    
    # Initialize with custom settings
    fcg = FCGAlgorithm(
        n_clusters=10,          # More clusters for finer granularity
        m_neighbors=3,          # Fewer neighbors per cluster
        k_shots=8,              # More demonstration samples
        iterations=15,          # More evolution iterations
        alpha=0.7,              # Weight prediction more than fairness
        p=0.1,                  # Higher initial score threshold
        metric_pred='accuracy', # Use accuracy instead of F1
        metric_fair='delta_eo', # Use Δeo instead of Reo
        max_dev_samples=100     # Use more dev samples
    )
    
    # Load and process
    fcg.load_data(dev_ratio=0.15)
    fcg.fit()
    fcg.select_demonstrations(k_per_subgroup=8)
    results = fcg.evaluate(max_test_samples=200)
    
    return results


# ============================================================
# EXAMPLE 3: Step-by-Step Processing
# ============================================================

def example_step_by_step():
    """Process FCG step by step with inspection"""
    from fcg_main import FCGAlgorithm
    import pandas as pd
    
    print("Example 3: Step-by-Step Processing\n")
    
    # Initialize
    fcg = FCGAlgorithm()
    
    # Load data
    print("Loading data...")
    train_df, dev_df, test_df = fcg.load_data()
    
    # Inspect subgroups
    print("\nSubgroup statistics:")
    for key, subgroup in fcg.subgroups.items():
        print(f"  {key}: {len(subgroup)} samples")
        print(f"    Mean age: {subgroup.df['age'].mean():.1f}")
        print(f"    % Male: {(1 - subgroup.df['sex'].mean()) * 100:.1f}%")
    
    # Run STEP 1: Clustering
    print("\nRunning STEP 1: Diverse Clustering...")
    from clustering import step1_diverse_clustering
    fcg.evolved_subgroups = step1_diverse_clustering(
        fcg.subgroups,
        n_clusters=fcg.n_clusters,
        m_neighbors=fcg.m_neighbors
    )
    
    # Inspect clustered subgroups
    print("\nClustered subgroup sizes:")
    for key, subgroup in fcg.evolved_subgroups.items():
        print(f"  {key}: {len(subgroup)} samples (from {len(fcg.subgroups[key])})")
    
    # Run STEP 2: Evolution
    print("\nRunning STEP 2: Evolution...")
    from evolution import step2_update_evol_score
    fcg.evolved_subgroups = step2_update_evol_score(
        fcg.evolved_subgroups,
        fcg.dev_data,
        k_shots=fcg.k_shots,
        iterations=fcg.iterations,
        alpha=fcg.alpha,
        p=fcg.p,
        max_dev_samples=fcg.max_dev_samples
    )
    
    # Inspect evolved scores
    print("\nEvolved scores:")
    for key, subgroup in fcg.evolved_subgroups.items():
        print(f"  {key}:")
        print(f"    Top 3 scores: {sorted(subgroup.scores, reverse=True)[:3]}")
        print(f"    Bottom 3 scores: {sorted(subgroup.scores)[:3]}")
    
    # Select demonstrations
    fcg.select_demonstrations()
    
    print(f"\nSelected {len(fcg.top_demonstrations)} total demonstrations")
    
    # Evaluate
    results = fcg.evaluate(max_test_samples=100)
    
    return fcg, results


# ============================================================
# EXAMPLE 4: Different Fairness Metrics
# ============================================================

def example_different_metrics():
    """Compare results using different fairness metrics"""
    from fcg_main import FCGAlgorithm
    
    print("Example 4: Different Fairness Metrics\n")
    
    metrics_to_test = [
        ('ratio_eo', 'Equalized Odds Ratio'),
        ('delta_eo', 'Equalized Odds Delta'),
        ('ratio_dp', 'Demographic Parity Ratio'),
        ('delta_dp', 'Demographic Parity Delta')
    ]
    
    results_comparison = {}
    
    for metric, name in metrics_to_test:
        print(f"\n{'='*60}")
        print(f"Testing with: {name} ({metric})")
        print('='*60)
        
        fcg = FCGAlgorithm(
            metric_fair=metric,
            iterations=5,  # Reduced for speed
            max_dev_samples=30
        )
        
        fcg.load_data()
        fcg.fit()
        fcg.select_demonstrations()
        results = fcg.evaluate(max_test_samples=50)
        
        results_comparison[metric] = results
    
    # Compare results
    print("="*60)
    print("COMPARISON ACROSS FAIRNESS METRICS")
    print("="*60)
    
    for metric, name in metrics_to_test:
        res = results_comparison[metric]
        print(f"\n{name}:")
        print(f"  F1 Improvement: {res['fcg']['f1_score'] - res['zero_shot']['f1_score']:+.4f}")
        print(f"  Δeo Improvement: {res['zero_shot']['delta_eo'] - res['fcg']['delta_eo']:+.4f}")
        print(f"  Reo Improvement: {res['fcg']['ratio_eo'] - res['zero_shot']['ratio_eo']:+.4f}")
    
    return results_comparison


# ============================================================
# EXAMPLE 5: Analyzing Top Demonstrations
# ============================================================

def example_analyze_demonstrations():
    """Analyze characteristics of selected demonstrations"""
    from fcg_main import FCGAlgorithm
    import pandas as pd
    
    print("Example 5: Analyzing Top Demonstrations\n")
    
    fcg = FCGAlgorithm(
        iterations=5,
        max_dev_samples=30
    )
    
    fcg.load_data()
    fcg.fit()
    fcg.select_demonstrations()
    
    demos = fcg.top_demonstrations
    
    print("Top Demonstration Statistics:")
    print(f"  Total demonstrations: {len(demos)}")
    print(f"\nSex distribution:")
    print(f"  Male:   {(demos['sex'] == 0).sum()}")
    print(f"  Female: {(demos['sex'] == 1).sum()}")
    print(f"\nIncome distribution:")
    print(f"  ≤50K: {(demos['income'] == 0).sum()}")
    print(f"  >50K: {(demos['income'] == 1).sum()}")
    print(f"\nAge statistics:")
    print(f"  Mean: {demos['age'].mean():.1f}")
    print(f"  Std:  {demos['age'].std():.1f}")
    print(f"  Min:  {demos['age'].min()}")
    print(f"  Max:  {demos['age'].max()}")
    
    # Show sample demonstrations
    print(f"\nSample demonstrations:")
    for idx, row in demos.head(3).iterrows():
        print(f"\n  Demo {idx + 1}:")
        print(f"    Age: {row['age']}, Sex: {'Female' if row['sex']==1 else 'Male'}")
        print(f"    Education: {row['education']}, Occupation: {row['occupation']}")
        print(f"    Income: {'≤50K' if row['income']==0 else '>50K'}")
    
    return demos


# ============================================================
# EXAMPLE 6: Quick Test Mode
# ============================================================

def example_quick_test():
    """Quick test with minimal parameters"""
    from fcg_main import run_fcg_experiment
    
    print("Example 6: Quick Test Mode\n")
    print("Running with minimal parameters for fast testing...")
    
    fcg, results = run_fcg_experiment(
        n_clusters=4,
        m_neighbors=3,
        k_shots=3,
        iterations=3,
        max_dev_samples=20,
        max_test_samples=30
    )
    
    print("\nQuick test complete!")
    print("For full experiment, use default parameters.")
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Basic Usage', example_basic),
        '2': ('Custom Parameters', example_custom_params),
        '3': ('Step-by-Step', example_step_by_step),
        '4': ('Different Metrics', example_different_metrics),
        '5': ('Analyze Demonstrations', example_analyze_demonstrations),
        '6': ('Quick Test', example_quick_test)
    }
    
    print("="*60)
    print("FCG ALGORITHM - ADVANCED EXAMPLES")
    print("="*60)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print()
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Select example (1-6) or press Enter for example 6 (Quick Test): ").strip()
        if not choice:
            choice = '6'
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}\n")
        func()
    else:
        print(f"Invalid choice: {choice}")
        print("Usage: python advanced_examples.py [1-6]")
        sys.exit(1)
