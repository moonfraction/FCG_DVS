# FCG Algorithm Implementation - Quick Start Guide

## What Has Been Implemented

A complete implementation of the **Fairness via Clustering-Genetic (FCG) algorithm** for mitigating bias in Large Language Models.

### Core Components

1. **data_utils.py** - Data loading, preprocessing, and subgroup creation
2. **clustering.py** - STEP 1: Diverse clustering with K-Means
3. **genetic_algorithm.py** - Roulette wheel selection for genetic evolution
4. **evolution.py** - STEP 2: Evolution score update loop
5. **cal_score.py** - Evolution score calculation function
6. **llm_utils.py** - LLM integration with Groq API
7. **metrics.py** - Fairness and performance metrics (DP, Eodds, F-score, etc.)
8. **fcg_main.py** - Main FCG algorithm and experiment orchestration
9. **config.py** - Centralized configuration file
10. **run_experiment.py** - Quick start script
11. **test_setup.py** - Setup verification script

---

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure API Key

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# Get one from: https://console.groq.com/keys
```

### Step 3: Run the Algorithm

```bash
# Verify setup
python test_setup.py

# Run FCG experiment
python run_experiment.py
```

---

## How It Works

### STEP 1: Diverse Clustering
1. Splits training data into 4 subgroups based on sensitive feature (sex) and label (income)
2. Applies K-Means clustering (8 clusters) to each subgroup
3. Selects 5 closest samples to each centroid
4. Creates diverse representative subset

### STEP 2: Genetic Evolution
1. Initializes all sample scores to 0.05
2. For each subgroup, runs 10 iterations:
   - Selects 5 demonstration samples using roulette wheel (probability based on scores)
   - Evaluates using LLM predictions (zero-shot baseline vs. few-shot ICL)
   - Computes evolution score: `α × Δpred + (1-α) × Δfair`
   - Updates scores for selected samples
3. Higher-scoring samples become more likely to be selected

### Testing Phase
1. Selects top-scoring samples from each subgroup
2. Uses them as demonstrations for in-context learning
3. Evaluates fairness and performance on test set

---

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_clusters` | 8 | K-Means clusters per subgroup |
| `m_neighbors` | 5 | Samples per cluster |
| `k_shots` | 5 | Demonstrations per iteration |
| `iterations` | 10 | Evolution iterations |
| `alpha` | 0.5 | Prediction/fairness balance |
| `metric_pred` | f1_score | Performance metric |
| `metric_fair` | ratio_eo | Fairness metric (Reo) |

Edit `config.py` to adjust these parameters.

---

## Expected Output

```
============================================================
FCG ALGORITHM EXPERIMENT
============================================================

LOADING DATA
  Training: 26,048 samples
  Dev:      6,512 samples
  Test:     16,281 samples

STEP 1: DIVERSE CLUSTERING
  g1 (Z=1,Y=0): 40 samples selected
  g2 (Z=1,Y=1): 40 samples selected
  g3 (Z=0,Y=0): 40 samples selected
  g4 (Z=0,Y=1): 40 samples selected

STEP 2: UPDATE EVOLUTION SCORE
  Processing g1: 10 iterations
    Iteration 1: EvolScore = 0.0823
    ...
  
EVALUATION ON TEST DATA
  Zero-shot Baseline:
    F1-Score: 0.5123
    Δeo: 0.1234
  
  FCG with ICL:
    F1-Score: 0.5678
    Δeo: 0.0876
  
IMPROVEMENT OVER BASELINE
  F1-Score: +0.0555
  Δeo: -0.0358 (lower is better)
```

---

## Customization

### Using Different Parameters

Edit `run_experiment.py`:

```python
fcg, results = run_fcg_experiment(
    n_clusters=8,
    m_neighbors=5,
    k_shots=5,
    iterations=10,
    max_dev_samples=50,
    max_test_samples=100
)
```

### Using Different Models

Edit `config.py`:

```python
MODEL = "llama3-70b-8192"  # Larger model
# or
MODEL = "mixtral-8x7b-32768"  # Mixtral model
```

### Quick Testing Mode

Edit `config.py`:

```python
QUICK_TEST = True  # Reduces parameters for fast testing
```

---

## Troubleshooting

### Issue: Import errors
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: API errors
**Solution**: 
1. Check your API key in `.env`
2. Ensure you have API credits
3. Check rate limits

### Issue: Dataset not found
**Solution**: Ensure `ds/adult/adult.data` and `ds/adult/adult.test` exist

### Issue: Slow execution
**Solution**: Reduce parameters in `config.py`:
- `MAX_DEV_SAMPLES = 20`
- `MAX_TEST_SAMPLES = 30`
- `ITERATIONS = 3`

---

## File Structure

```
myFCG/
├── Core Algorithm
│   ├── fcg_main.py          # Main FCG implementation
│   ├── clustering.py        # STEP 1: Clustering
│   ├── evolution.py         # STEP 2: Evolution
│   ├── genetic_algorithm.py # Genetic selection
│   └── cal_score.py         # Score calculation
│
├── Utilities
│   ├── data_utils.py        # Data handling
│   ├── llm_utils.py         # LLM integration
│   └── metrics.py           # Evaluation metrics
│
├── Configuration
│   ├── config.py            # Parameters
│   ├── .env                 # API keys (create this)
│   └── .env.example         # Template
│
├── Scripts
│   ├── run_experiment.py    # Quick start
│   └── test_setup.py        # Setup verification
│
└── Data
    └── ds/adult/            # Adult dataset
```

---

## Next Steps

1. **Verify setup**: `python test_setup.py`
2. **Run quick test**: Set `QUICK_TEST = True` in `config.py`, then `python run_experiment.py`
3. **Run full experiment**: Set `QUICK_TEST = False`, then `python run_experiment.py`
4. **Analyze results**: Check output metrics (F1-score, Δeo, Reo, etc.)
5. **Experiment**: Adjust parameters in `config.py` and re-run

---

## Important Notes

- **API Costs**: The algorithm makes LLM API calls. Monitor your usage.
- **Runtime**: Full experiment may take 30-60 minutes depending on parameters.
- **Caching**: Baseline predictions are cached per subgroup to reduce API calls.
- **Reproducibility**: Set random seeds in `config.py` for reproducible results.

---

## Citation

This implementation is based on the FCG (Fairness via Clustering-Genetic) algorithm for mitigating fairness bias in Large Language Models through representative sample selection and evolution-based optimization.
