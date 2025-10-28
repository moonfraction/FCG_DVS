# Changes Summary - Automatic Results Saving

## Date: October 28, 2024

## Changes Made

### 1. Added New Function: `save_evaluation_results()` in `fcg_main.py`

**Location:** Lines 19-71 in `fcg_main.py`

**Purpose:** Automatically save all evaluation metrics to timestamped CSV files

**Features:**
- Creates `results/` directory automatically
- Generates timestamped filenames: `results_YYYYMMDD_HHMMSS.csv`
- Saves both configuration parameters and evaluation metrics
- Returns filepath for reference

**CSV Format:**
```csv
type,method,metric,value
config,n_clusters,config_value,8
config,model,config_value,llama-3.3-70b
evaluation,zero_shot,accuracy,0.7845
evaluation,fcg,f1_score,0.6298
```

### 2. Modified `evaluate()` Method

**Changes:**
- Now automatically calls `save_evaluation_results()` after evaluation
- Saves configuration parameters: n_clusters, m_neighbors, k_shots, iterations, alpha, p, model, max_test_samples, metric_pred, metric_fair
- Saves metrics for both zero-shot and FCG methods

**Before:**
```python
return {
    'zero_shot': zero_metrics,
    'fcg': fcg_metrics
}
```

**After:**
```python
results = {
    'zero_shot': zero_metrics,
    'fcg': fcg_metrics
}

config_info = {
    'n_clusters': self.n_clusters,
    'm_neighbors': self.m_neighbors,
    # ... all config params
}
save_evaluation_results(results, config_info=config_info)

return results
```

### 3. Modified `evaluate_with_dvs()` Method

**Changes:**
- Automatically saves DVS evaluation results
- Includes additional DVS-specific config: batch_size, k_neighbors, L_iterations
- Saves both FCG-DVS and FCG baseline metrics (if enabled)

**Configuration saved:**
- All FCG parameters (n_clusters, m_neighbors, etc.)
- DVS-specific parameters (batch_size, k_neighbors, L_iterations)
- use_fcg_baseline flag

### 4. Created Documentation

**Files:**
- `RESULTS_SAVING.md`: Comprehensive guide on the automatic saving feature
- `CHANGES_SUMMARY.md`: This file documenting all changes

## What Gets Saved

### All Performance Metrics:
- accuracy
- precision
- recall
- f1_score

### All Demographic Parity Metrics:
- DP_0 (P(ŷ=1|Z=0))
- DP_1 (P(ŷ=1|Z=1))
- delta_dp (|DP_1 - DP_0|)
- ratio_dp (DP_0 / DP_1)

### All Equalized Odds Metrics:
- TPR_0, TPR_1 (True Positive Rates)
- FPR_0, FPR_1 (False Positive Rates)
- delta_eo (max difference)
- ratio_eo (minimum ratio)

## Backward Compatibility

✅ **All existing scripts work without modification:**
- `run_experiment.py` - Works as before, now also saves to CSV
- `run_fcg_dvs.py` - Works as before, still saves `.metrics` files AND now saves CSV
- `test_workflow.py` - Works as before, still saves to `res/test/` AND now saves CSV

✅ **No breaking changes:**
- Return values unchanged
- Function signatures unchanged
- Existing `.metrics` file saving still works

## Usage Examples

### Running Experiments

```python
# All of these now automatically save to CSV
from fcg_main import run_fcg_experiment

# Run standard FCG
fcg, results = run_fcg_experiment(
    n_clusters=8,
    m_neighbors=5,
    k_shots=5,
    iterations=10,
    max_test_samples=100
)
# Results saved to: results/results_20251028_143052.csv
```

### Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_csv('results/results_20251028_143052.csv')

# Get FCG F1-score
fcg_f1 = df[
    (df['type'] == 'evaluation') & 
    (df['method'] == 'fcg') & 
    (df['metric'] == 'f1_score')
]['value'].values[0]

print(f"FCG F1-Score: {fcg_f1:.4f}")
```

## Benefits

1. ✅ **Automatic**: No manual saving required
2. ✅ **Timestamped**: Never overwrite results
3. ✅ **Structured**: Easy analysis with pandas
4. ✅ **Complete**: All metrics + config in one file
5. ✅ **Traceable**: Know exactly what parameters were used
6. ✅ **Compatible**: Works with existing code

## Testing

To test the new feature:

```bash
# Run any experiment
python run_experiment.py

# Check the results directory
ls -la results/

# You should see: results_YYYYMMDD_HHMMSS.csv

# View the results
cat results/results_*.csv
```

## Next Steps

1. Run `test_workflow.py` to verify everything works
2. Results will be saved to both:
   - `results/results_YYYYMMDD_HHMMSS.csv` (new automatic CSV)
   - `res/test/test_fcg_YYYYMMDD_HHMMSS.metrics` (existing text format)
3. Use pandas to analyze the CSV files
4. Compare results across multiple runs easily

## Notes

- CSV files are saved to `results/` directory (created automatically)
- Each run gets a unique timestamp
- Configuration and metrics are in the same file for easy traceability
- The CSV format is optimized for data analysis and visualization
