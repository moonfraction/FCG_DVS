"""
STEP 2: Update Evolution Score
Genetic algorithm evolution loop with score updates
"""
from cal_score import cal_score_cached
from genetic_algorithm import roulette_wheel_selection


def step2_update_evol_score(subgroups_dict, dev_data, k_shots=5, iterations=10,
                            metric_pred='f1_score', metric_fair='ratio_eo',
                            alpha=0.5, p=0.05, model="llama-3.1-8b-instant",
                            max_dev_samples=50):
    """
    STEP 2: Update evolution scores for all subgroups
    
    For each subgroup:
    - Run genetic algorithm for specified iterations
    - Select k demonstration samples using roulette wheel
    - Calculate evolution score
    - Update sample scores
    
    Args:
        subgroups_dict: Dictionary of Subgroup objects from STEP 1 (SG')
        dev_data: Development dataset for evaluation
        k_shots: Number of demonstration samples per iteration
        iterations: Number of evolution iterations
        metric_pred: Performance metric (default: 'f1_score')
        metric_fair: Fairness metric (default: 'ratio_eo')
        alpha: Balance coefficient
        p: Minimum score threshold
        model: LLM model name
        max_dev_samples: Max dev samples to use per evaluation
    
    Returns:
        Updated subgroups_dict with evolved scores
    """
    print("\n" + "="*60)
    print("STEP 2: UPDATE EVOLUTION SCORE")
    print("="*60)
    
    for subgroup_key, subgroup in subgroups_dict.items():
        print(f"\nProcessing {subgroup_key}: {len(subgroup)} samples")
        print(f"  Running {iterations} iterations with k={k_shots} shots")
        
        # Skip if subgroup is empty or too small
        if len(subgroup) == 0:
            print(f"  Skipping {subgroup_key} (empty)")
            continue
        
        if len(subgroup) < k_shots:
            print(f"  Adjusting k_shots from {k_shots} to {len(subgroup)}")
            k_shots_actual = len(subgroup)
        else:
            k_shots_actual = k_shots
        
        # Cache baseline to avoid redundant API calls
        baseline_cache = {}
        
        # Run genetic algorithm iterations
        for iteration in range(iterations):
            print(f"\n  Iteration {iteration + 1}/{iterations}:")
            
            # Select k demonstration samples using roulette wheel
            selected_indices = roulette_wheel_selection(subgroup, k_shots_actual)
            
            if len(selected_indices) == 0:
                print("    No samples selected, skipping iteration")
                continue
            
            # Get selected samples
            selected_samples = subgroup.df.iloc[selected_indices]
            
            print(f"    Selected {len(selected_indices)} samples: {selected_indices}")
            
            # Calculate evolution score
            evol_score = cal_score_cached(
                selected_samples,
                dev_data,
                baseline_cache=baseline_cache,
                metric_pred=metric_pred,
                metric_fair=metric_fair,
                alpha=alpha,
                p=p,
                model=model,
                max_dev_samples=max_dev_samples
            )
            
            # Update scores for selected samples
            subgroup.update_scores(selected_indices, evol_score)
            
            print(f"    Evolution Score: {evol_score:.4f}")
            print(f"    Updated scores for samples: {selected_indices}")
        
        # Print final score distribution
        print(f"\n  Final score distribution for {subgroup_key}:")
        print(f"    Mean:   {subgroup.scores.mean():.4f}")
        print(f"    Std:    {subgroup.scores.std():.4f}")
        print(f"    Min:    {subgroup.scores.min():.4f}")
        print(f"    Max:    {subgroup.scores.max():.4f}")
    
    return subgroups_dict
