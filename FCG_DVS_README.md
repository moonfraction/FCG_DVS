# FCG-DVS Hybrid Approach

## Overview

This implementation combines **FCG (Fairness via Clustering-Genetic)** with **DVS (Dynamic Validation Set)** for optimal fairness-aware in-context learning.

### Two-Phase Process

#### Phase 1: Global Fairness Optimization (FCG)
- **Input**: Full training dataset
- **Process**:
  1. Segment data into 4 subgroups based on sensitive attribute (Z) and label (Y)
  2. Apply K-Means clustering within each subgroup
  3. Run genetic evolution to optimize: `EvolScore = α·Accuracy + (1-α)·R_eo`
- **Output**: Globally optimized pool **H_opt** (~40 high-quality demonstrations)

#### Phase 2: Local Batch Selection (DVS)
- **Input**: Test batches + H_opt pool
- **Process**:
  1. For each test batch B_n:
     - Use Gemini embeddings to find k-nearest neighbors from H_opt
     - Create Dynamic Validation Set (P_n) from closest neighbors
     - Iteratively refine ICE set C_n by:
       - Computing individual errors for each test sample
       - Expanding the weakest core set
       - Tracking minimum total error
  2. Predict on batch using optimized ICE set
- **Output**: Batch-specific predictions with adaptive fairness

## Key Components

### Files

- `dvs_utils.py`: Embedding utilities (Gemini API, k-NN search)
- `dvs_ice_selection.py`: Iterative ICE selection algorithm
- `fcg_main.py`: Main FCG algorithm + `evaluate_with_dvs()` method
- `run_fcg_dvs.py`: Complete example script

### API Call Efficiency

For 100 test samples:
- **Batches**: 10 (batch size = 10)
- **Per batch**:
  - L iterations (5) × P DVS samples (10) × avg |C| (11) = 550 calls
  - N test predictions (10) = 10 calls
  - **Total per batch**: ~560 calls
- **Grand total**: ~5600 calls (vs. standard DVS which would use full training set)

## Usage

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export CEREBRAS_API_KEY_safari="your_key_1"
export CEREBRAS_API_KEY_firefox="your_key_2"
export CEREBRAS_API_KEY_brave="your_key_3"
export GEMINI_API_KEY="your_gemini_api_key"
```

### Run Complete Experiment

```python
python run_fcg_dvs.py
```

### Programmatic Usage

```python
from fcg_main import FCGAlgorithm
from llm_utils import init_llm_clients

# Initialize API clients
init_llm_clients()

# Create FCG instance
fcg = FCGAlgorithm(
    n_clusters=8,
    m_neighbors=5,
    k_shots=5,
    iterations=10,
    alpha=0.5,
    model="llama-3.3-70b"
)

# Phase 1: FCG
fcg.load_data(dev_ratio=0.2)
fcg.fit()
fcg.select_demonstrations()

# Phase 2: DVS
results = fcg.evaluate_with_dvs(
    max_test_samples=100,
    batch_size=10,
    k_neighbors=5,
    L_iterations=5,
    use_fcg_baseline=True
)
```

## Parameters

### FCG Parameters (Phase 1)
- `n_clusters`: Number of K-Means clusters per subgroup (default: 8)
- `m_neighbors`: Neighbors selected per cluster (default: 5)
- `k_shots`: Demonstrations per subgroup (default: 5)
- `iterations`: Genetic evolution iterations (default: 10)
- `alpha`: Balance coefficient for pred vs fairness (default: 0.5)

### DVS Parameters (Phase 2)
- `batch_size`: Test samples per batch (default: 10)
- `k_neighbors`: Nearest neighbors to retrieve per test sample (default: 5)
- `L_iterations`: DVS refinement iterations (default: 5)
- `use_fcg_baseline`: Compare with standard FCG (default: True)

## Advantages

1. **Efficiency**: FCG pre-filters training data, reducing DVS search space from 20K+ to ~40 samples
2. **Quality**: DVS operates on globally fair demonstrations, not arbitrary training samples
3. **Adaptivity**: Each test batch gets customized demonstrations
4. **Fairness**: Joint optimization of accuracy and fairness metrics throughout

## Expected Performance

- **Accuracy**: Comparable or better than standard FCG
- **Fairness**: Improved Equalized Odds ratio (Reo) due to local adaptation
- **API Calls**: ~6000 for 100 test samples (manageable with multiple API keys)

## Troubleshooting

### Missing Gemini API Key
- Ensure `GEMINI_API_KEY` is set in environment
- Get free API key at: https://makersuite.google.com/app/apikey

### Out of API Quota
- Use multiple Cerebras keys for rotation
- Reduce `max_test_samples` or `L_iterations`
- Increase `batch_size` to reduce number of batches

### Embeddings Taking Too Long
- Embeddings are cached in `data/embeddings/`
- Subsequent runs will load from cache
- Delete cache to regenerate embeddings

## Citation

If you use this implementation, please cite:

```
FCG-DVS: A Hybrid Approach for Fairness-Aware In-Context Learning
Combining Fairness via Clustering-Genetic with Dynamic Validation Set Selection
```
