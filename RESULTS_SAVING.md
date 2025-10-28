# Evaluation Results Saving

## Overview

All evaluation methods in `fcg_main.py` now automatically save results to timestamped CSV files in the `results/` directory.

## What Gets Saved

### For `evaluate()` method (FCG without DVS):
- Zero-shot baseline metrics
- FCG metrics
- Configuration parameters (n_clusters, m_neighbors, k_shots, iterations, alpha, p, model, etc.)

### For `evaluate_with_dvs()` method (FCG-DVS):
- FCG-DVS metrics
- FCG baseline metrics (if use_fcg_baseline=True)
- Configuration parameters (all FCG params + DVS params: batch_size, k_neighbors, L_iterations)

## Metrics Saved

Each evaluation saves the following metrics:

### Performance Metrics:
- `accuracy`: Overall accuracy
- `precision`: Precision score
- `recall`: Recall score
- `f1_score`: F1-score

### Demographic Parity Metrics:
- `DP_0`: P(ŷ=1|Z=0)
- `DP_1`: P(ŷ=1|Z=1)
- `delta_dp`: |DP_1 - DP_0| (lower is better)
- `ratio_dp`: DP_0 / DP_1 (closer to 1 is better)

### Equalized Odds Metrics:
- `TPR_0`: True Positive Rate for Z=0
- `TPR_1`: True Positive Rate for Z=1
- `FPR_0`: False Positive Rate for Z=0
- `FPR_1`: False Positive Rate for Z=1
- `delta_eo`: max(|TPR_1-TPR_0|, |FPR_1-FPR_0|) (lower is better)
- `ratio_eo`: min(TPR_0/TPR_1, FPR_0/FPR_1) (closer to 1 is better)

## File Format

### Filename
```
results/results_YYYYMMDD_HHMMSS.csv
```

Example: `results/results_20251028_143052.csv`

### CSV Structure

The CSV has 4 columns:
- `type`: Either "config" (for configuration parameters) or "evaluation" (for metrics)
- `method`: The method name (e.g., "zero_shot", "fcg", "fcg_dvs")
- `metric`: The metric name or config parameter name
- `value`: The metric value or config value

### Example CSV Content

```csv
type,method,metric,value
config,n_clusters,config_value,8
config,m_neighbors,config_value,5
config,k_shots,config_value,5
config,iterations,config_value,10
config,alpha,config_value,0.5
config,p,config_value,0.05
config,model,config_value,llama-3.3-70b
config,max_test_samples,config_value,100
evaluation,zero_shot,accuracy,0.7845
evaluation,zero_shot,precision,0.6234
evaluation,zero_shot,recall,0.5123
evaluation,zero_shot,f1_score,0.5623
evaluation,zero_shot,DP_0,0.2456
evaluation,zero_shot,DP_1,0.4123
evaluation,zero_shot,delta_dp,0.1667
evaluation,zero_shot,ratio_dp,0.5957
evaluation,fcg,accuracy,0.8123
evaluation,fcg,precision,0.6789
evaluation,fcg,recall,0.5891
evaluation,fcg,f1_score,0.6298
evaluation,fcg,DP_0,0.2834
evaluation,fcg,DP_1,0.3456
evaluation,fcg,delta_dp,0.0622
evaluation,fcg,ratio_dp,0.8200
```

## Usage

### Automatic Saving

When you run any of these scripts, results are automatically saved:

```python
# run_experiment.py
fcg, results = run_fcg_experiment(...)
# Results automatically saved to results/results_YYYYMMDD_HHMMSS.csv

# run_fcg_dvs.py  
results = fcg.evaluate_with_dvs(...)
# Results automatically saved to results/results_YYYYMMDD_HHMMSS.csv

# test_workflow.py
results = fcg.evaluate(...)
# Results automatically saved to results/results_YYYYMMDD_HHMMSS.csv
```

### Reading Results

You can easily load and analyze results using pandas:

```python
import pandas as pd

# Load results
df = pd.read_csv('results/results_20251028_143052.csv')

# Get configuration
config = df[df['type'] == 'config']
print("Configuration:")
print(config[['method', 'value']])

# Get evaluation metrics
metrics = df[df['type'] == 'evaluation']

# Get specific method metrics
fcg_metrics = metrics[metrics['method'] == 'fcg']
print("\nFCG Metrics:")
print(fcg_metrics[['metric', 'value']])

# Compare methods
zero_shot_f1 = metrics[(metrics['method'] == 'zero_shot') & 
                       (metrics['metric'] == 'f1_score')]['value'].values[0]
fcg_f1 = metrics[(metrics['method'] == 'fcg') & 
                 (metrics['metric'] == 'f1_score')]['value'].values[0]
print(f"\nF1-Score Improvement: {fcg_f1 - zero_shot_f1:.4f}")
```

### Custom Output Directory

By default, results are saved to the `results/` directory. This is created automatically if it doesn't exist.

## Additional Formats

Besides the automatic CSV saving, the codebase also includes:

1. **`.metrics` files** (text format) saved by:
   - `run_fcg_dvs.py` → `res/save/file_YYYYMMDD_HHMMSS_withDVS.metrics`
   - `test_workflow.py` → `res/test/test_fcg_YYYYMMDD_HHMMSS.metrics`

2. **Log files** with detailed execution logs in `res/`

## Benefits

1. **Timestamped**: Never overwrite previous results
2. **Structured**: Easy to load and analyze with pandas/Excel
3. **Comprehensive**: All metrics and configurations in one place
4. **Automatic**: No need to manually save results
5. **Traceable**: Configuration parameters saved alongside metrics

## Example Analysis

To analyze multiple runs:

```python
import pandas as pd
import glob

# Load all result files
all_files = glob.glob('results/results_*.csv')
dfs = []

for file in all_files:
    df = pd.read_csv(file)
    df['file'] = file
    dfs.append(df)

# Combine all results
all_results = pd.concat(dfs, ignore_index=True)

# Compare F1-scores across runs
f1_scores = all_results[
    (all_results['type'] == 'evaluation') & 
    (all_results['metric'] == 'f1_score')
][['file', 'method', 'value']]

print("F1-Scores across all runs:")
print(f1_scores.pivot(index='file', columns='method', values='value'))
```
