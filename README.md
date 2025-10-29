# FCG-DVS: Fairness via Clustering-Genetic with Dynamic Validation Set

Implementation of the FCG (Fairness via Clustering-Genetic) and DVS (Dynamic Validation Set) algorithms for mitigating bias in Large Language Models through representative sample selection, evolution-based optimization, and adaptive in-context learning.

## Overview

This project implements two complementary algorithms:

### FCG (Fairness via Clustering-Genetic)
Addresses fairness bias in LLMs by:
1. **Diverse Clustering (STEP 1)**: Selecting representative samples using K-Means clustering
2. **Genetic Evolution (STEP 2)**: Evolving demonstration samples based on prediction performance and fairness metrics

### DVS (Dynamic Validation Set)
Enhances FCG with local adaptation:
1. **Global Pool Creation**: Uses FCG to create an optimized demonstration pool (H_opt)
2. **Local Adaptive Selection**: For each test batch, dynamically selects the most relevant ICE demonstrations using:
   - Semantic similarity (Gemini embeddings + k-NN)
   - Iterative refinement based on batch-specific error (π = 1 - F1, ψ = demographic parity difference)

## Key Features

- ✅ **FCG Algorithm**: Global fairness optimization through clustering and genetic evolution
- ✅ **DVS Algorithm**: Local adaptation with batch-specific ICE selection
- ✅ **Hybrid FCG-DVS**: Two-phase pipeline combining global and local optimization
- ✅ **Multiple LLM Support**: Cerebras (llama-3.3-70b), Groq fallback, multi-API-key rotation
- ✅ **Comprehensive Metrics**: Performance (Accuracy, F1) + Fairness (DP, EO)
- ✅ **Random Seed Control**: Full reproducibility with configurable seeds
- ✅ **Automatic Results Saving**: Timestamped CSV files with all metrics and configs
- ✅ **Robust Error Handling**: Fixed division-by-zero issues in fairness metrics
- ✅ **Test Workflow**: Minimal config testing script for quick verification

## Algorithm Components

### FCG: STEP 1 - Diverse Clustering
- Splits training data into 4 subgroups: `SG = {g1(Z=1,Y=0), g2(Z=1,Y=1), g3(Z=0,Y=0), g4(Z=0,Y=1)}`
- Applies K-Means clustering (n=8 clusters) to each subgroup
- Selects m=5 closest samples to each centroid
- Creates diverse initial population `SG'`

### FCG: STEP 2 - Genetic Evolution
- Runs genetic algorithm with roulette wheel selection
- Iteratively selects K demonstration samples
- Computes evolution scores based on:
  - **Performance improvement** (F1-score)
  - **Fairness improvement** (Demographic Parity or Equalized Odds ratio)
- Updates sample scores across iterations
- Creates globally optimized demonstration pool (H_opt)

### FCG Evolution Score
```
EvolScore = α × Δpred + (1 - α) × Δfair
```
where:
- `Δpred = max(ICL_pred - Base_pred, p)`
- `Δfair = max(ICL_fair - Base_fair, p)`
- `α = 0.5` (balance coefficient)
- `p = 0.05` (minimum threshold)

### DVS: Dynamic Validation Set

**Phase 1 - Embedding & Indexing**:
1. Embed FCG-optimized pool (H_opt) using Gemini embeddings
2. Create semantic search index

**Phase 2 - Batch Processing** (for each test batch of size N):
1. **Initial Support Set**: For each test sample, retrieve k nearest neighbors from H_opt
2. **Iterative ICE Refinement** (L iterations):
   - Compute batch error: `E = α·π + (1-α)·ψ`
     - `π = 1 - F1` (performance error)
     - `ψ = |P(ŷ=1|z=0) - P(ŷ=1|z=1)|` (fairness error)
   - Identify weakest link (test sample with highest error)
   - Expand its support set with additional neighbors
   - Re-evaluate until convergence or L iterations
3. **Final Prediction**: Use refined ICE set for final predictions

### DVS Error Formulation
```
E_batch = α × π + (1 - α) × ψ
```
where:
- `π = 1 - F1` (1 - F1 score, measures performance error)
- `ψ = |P(ŷ=1|z=0) - P(ŷ=1|z=1)|` (demographic parity difference, measures fairness error)
- `α = 0.5` (balance between performance and fairness)

## Project Structure
```
myFCG/
├── fcg_main.py              # Main FCG algorithm implementation
├── run_fcg_dvs.py           # FCG-DVS pipeline runner
├── test_workflow.py         # Minimal test configuration
├── clustering.py            # K-Means clustering (Step 1)
├── evolution.py             # Genetic algorithm (Step 2)
├── cal_score.py             # Evolution score calculation
├── dvs_ice_selection.py     # DVS algorithm with ICE refinement
├── dvs_utils.py             # DVS helper utilities
├── llm_utils.py             # LLM API calls (Cerebras, Groq)
├── metrics.py               # Fairness metrics (DP, EO)
├── data_utils.py            # Data loading/splitting
├── config.py                # Configuration management
├── logger_utils.py          # Logging utilities
├── run_experiment.py        # Full experiment runner
├── test_setup.py            # API connectivity testing
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── FCG_DVS_README.md        # FCG-DVS detailed guide
├── RESULTS_SAVING.md        # Results file documentation
├── IMPLEMENTATION_SUMMARY.md # Implementation notes
├── ds/                      # Datasets
│   ├── adult/               # Adult Income dataset
│   └── credit/              # German Credit dataset
├── results/                 # Experiment results (CSV)
│   ├── results_*.csv        # FCG-only results
│   └── results_dvs_*.csv    # FCG-DVS results
├── res/                     # Intermediate results
│   └── test/                # Test run results
└── data/                    # Generated data
    └── embeddings/          # Cached embeddings for DVS
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd myFCG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
   
   Create a `.env` file or set environment variables:
   ```bash
   # Cerebras API (primary LLM for FCG)
   export CEREBRAS_API_KEY_1="your_cerebras_key_1"
   export CEREBRAS_API_KEY_2="your_cerebras_key_2"  # Optional: for key rotation
   
   # Groq API (fallback LLM)
   export GROQ_API_KEY_1="your_groq_key_1"
   export GROQ_API_KEY_2="your_groq_key_2"  # Optional: for key rotation
   
   # Gemini API (for DVS embeddings)
   export GEMINI_API_KEY="your_gemini_api_key"
   ```

4. **Verify setup**:
   ```bash
   python test_setup.py
   ```

   This will test connectivity to:
   - Cerebras API (llama-3.3-70b)
   - Groq API (llama-3.3-70b-versatile)
   - Gemini API (text-embedding-004)

## Quick Start

**Minimal Test Run** (test_workflow.py):
```bash
python test_workflow.py
```
Uses small subsets for quick testing:
- Train: 50 samples
- Dev: 20 samples  
- Test: 20 samples
- Demonstrations: K=3
- Evolution iterations: 2

**Full FCG Run**:
```bash
python run_experiment.py
```

**FCG-DVS Pipeline**:
```bash
python run_fcg_dvs.py
```

**With Custom Random Seed** (for reproducibility testing):
```bash
# Edit test_workflow.py or run_fcg_dvs.py CONFIG:
CONFIG = {
    ...
    'random_seed': 123,  # Change this value
    ...
}
```

## Dataset

The implementation supports multiple datasets:

**Adult Income Dataset** (default):
- **Training data**: `ds/adult/adult.data`
- **Test data**: `ds/adult/adult.test`
- **Sensitive feature**: Sex (0=Male, 1=Female)
- **Target label**: Income (0=≤50K, 1=>50K)

Ensure the dataset is available in the `ds/adult/` directory.

## Usage

### Basic Usage

**Run FCG-only experiment**:
```bash
python run_experiment.py
```

**Run FCG-DVS pipeline**:
```bash
python run_fcg_dvs.py
```

**Quick test with minimal data**:
```bash
python test_workflow.py
```

### Custom Configuration

Edit the `CONFIG` dictionary in your script:

```python
CONFIG = {
    # Data configuration
    'dataset': 'adult',            # 'adult' or 'credit'
    'train_size': 500,             # Training set size (None = full dataset)
    'dev_size': 100,               # Dev set size for evolution
    'test_size': 200,              # Test set size
    
    # FCG parameters
    'n_clusters': 8,               # K-Means clusters per subgroup
    'm_neighbors': 5,              # Samples per cluster
    'k_shots': 5,                  # Demonstrations per iteration
    'iterations': 10,              # Evolution iterations
    
    # DVS parameters
    'k_neighbors': 3,              # Initial neighbors for DVS
    'L': 3,                        # ICE refinement iterations
    'batch_size': 32,              # Batch size for DVS processing
    
    # LLM configuration
    'model': 'llama-3.3-70b',      # LLM model name
    'temperature': 0,              # Temperature (0 = deterministic)
    
    # Reproducibility
    'random_seed': 42,             # Random seed (affects splits, clustering, sampling)
    
    # Fairness metrics
    'fair_metric': 'DP',           # 'DP' (Demographic Parity) or 'EO' (Equalized Odds)
}
```

### Using the FCGAlgorithm Class

```python
from fcg_main import FCGAlgorithm

# Initialize with random seed for reproducibility
fcg = FCGAlgorithm(
    n_clusters=8,
    m_neighbors=5,
    k_shots=5,
    iterations=10,
    max_dev_samples=50,
    max_test_samples=100,
    random_seed=42           # Add this for reproducibility
)

# Run FCG (Step 1 + Step 2)
fcg.run()

# Evaluate on test set
results = fcg.evaluate(zero_shot=True)
print(f"F1: {results['f1_score']:.4f}")
print(f"DP Ratio: {results['ratio_dp']:.4f}")
```

### Using FCG-DVS Pipeline

```python
from run_fcg_dvs import run_fcg_dvs_experiment

results = run_fcg_dvs_experiment(
    # FCG parameters
    n_clusters=8,
    m_neighbors=5, 
    k_shots=5,
    iterations=10,
    
    # DVS parameters
    k_neighbors=3,
    L=3,
    batch_size=32,
    
    # Random seed
    random_seed=42
)

print(f"FCG F1: {results['fcg']['f1_score']:.4f}")
print(f"DVS F1: {results['dvs']['f1_score']:.4f}")
```

## Hyperparameters

### FCG Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | 8 | Number of K-Means clusters per subgroup |
| `m_neighbors` | 5 | Samples selected per cluster centroid |
| `k_shots` | 5 | Demonstration samples per ICL iteration |
| `iterations` | 10 | Number of genetic evolution iterations |
| `alpha` | 0.5 | Balance: α×Δpred + (1-α)×Δfair |
| `p` | 0.05 | Initial score threshold for improvements |
| `metric_pred` | 'f1_score' | Performance metric (f1_score, accuracy) |
| `metric_fair` | 'ratio_dp' or 'ratio_eo' | Fairness metric (DP or EO ratio) |
| `random_seed` | 42 | Random seed for reproducibility |

### DVS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_neighbors` | 3 | Initial k-NN neighbors for ICE selection |
| `L` | 3 | Number of ICE refinement iterations |
| `batch_size` | 32 | Batch size for DVS processing |
| `alpha` | 0.5 | Balance: α×π + (1-α)×ψ (error formula) |

### LLM Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | 'llama-3.3-70b' | LLM model name |
| `temperature` | 0 | Sampling temperature (0 = deterministic) |
| `max_tokens` | 10 | Maximum output tokens |

## Evaluation Metrics

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive prediction precision (TP / (TP + FP))
- **Recall**: Positive prediction recall (TP / (TP + FN))
- **F1-Score**: Harmonic mean of precision and recall

### Fairness Metrics

**Demographic Parity (DP)**:
- `DP_z = P(ŷ=1|Z=z)` - Probability of positive prediction given sensitive attribute
- `Δdp = |DP_1 - DP_0|` - Absolute difference (lower is better)
- `Rdp = DP_0 / DP_1` - Ratio (closer to 1.0 is better, capped at 100 for edge cases)

**Equalized Odds (EO)**:
- `TPR_z = P(ŷ=1|y=1,Z=z)` - True Positive Rate per group
- `FPR_z = P(ŷ=1|y=0,Z=z)` - False Positive Rate per group
- `Δeo = max(|TPR_1-TPR_0|, |FPR_1-FPR_0|)` - Maximum disparity (lower is better)
- `Reo = min(TPR_0/TPR_1, FPR_0/FPR_1)` - Minimum ratio (closer to 1.0 is better, capped at 100)

**Ratio Capping**: To prevent extreme values from division by near-zero denominators, ratios are capped at 100.0. If both numerator and denominator are near zero, the ratio is set to 1.0 (perfect fairness).

## API Configuration

The implementation supports multiple LLM providers with automatic key rotation:

### Cerebras API (Primary)
```bash
export CEREBRAS_API_KEY_1="your_key_1"
export CEREBRAS_API_KEY_2="your_key_2"  # Optional
```
- Model: `llama-3.3-70b`
- Cost: ~$0.10/1M input tokens, ~$0.20/1M output tokens
- Base URL: `https://api.cerebras.ai/v1`

### Groq API (Fallback)
```bash
export GROQ_API_KEY_1="your_key_1"
export GROQ_API_KEY_2="your_key_2"  # Optional
```
- Model: `llama-3.3-70b-versatile`
- Base URL: `https://api.groq.com/openai/v1`

### Gemini API (DVS Embeddings)
```bash
export GEMINI_API_KEY="your_key"
```
- Model: `text-embedding-004`
- Used for: Semantic embeddings in DVS phase

**Multi-key Rotation**: The system automatically rotates between API keys when rate limits are hit, maximizing throughput.

## License

This implementation is for educational and research purposes.
