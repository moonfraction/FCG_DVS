# FCG Algorithm: Fairness via Clustering-Genetic

Implementation of the Fairness via Clustering-Genetic (FCG) algorithm for mitigating bias in Large Language Models through representative sample selection and evolution-based optimization.

## Overview

The FCG algorithm addresses fairness bias in LLMs by:
1. **Diverse Clustering (STEP 1)**: Selecting representative samples using K-Means clustering
2. **Genetic Evolution (STEP 2)**: Evolving demonstration samples based on prediction performance and fairness metrics

## Algorithm Components

### STEP 1: Diverse Clustering
- Splits training data into 4 subgroups: `SG = {g1(Z=1,Y=0), g2(Z=1,Y=1), g3(Z=0,Y=0), g4(Z=0,Y=1)}`
- Applies K-Means clustering (n=8 clusters) to each subgroup
- Selects m=5 closest samples to each centroid
- Creates diverse initial population `SG'`

### STEP 2: Genetic Evolution
- Runs genetic algorithm with roulette wheel selection
- Iteratively selects K demonstration samples
- Computes evolution scores based on:
  - **Performance improvement** (F-score)
  - **Fairness improvement** (Equalized Odds ratio)
- Updates sample scores across iterations

### CAL_SCORE Function
Calculates evolution scores:
```
EvolScore = α × Δpred + (1 - α) × Δfair
```
where:
- `Δpred = max(ICL_pred - Base_pred, p)`
- `Δfair = max(ICL_fair - Base_fair, p)`
- `α = 0.5` (balance coefficient)

## Project Structure

```
myFCG/
├── fcg_main.py           # Main FCG algorithm and experiment runner
├── data_utils.py         # Data loading and preprocessing
├── clustering.py         # STEP 1: Diverse clustering
├── genetic_algorithm.py  # Genetic algorithm with roulette wheel selection
├── evolution.py          # STEP 2: Evolution score update
├── cal_score.py          # Score calculation function
├── llm_utils.py          # LLM integration with Groq API
├── metrics.py            # Fairness and performance metrics
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/chandranshsingh/Documents/7sem/BTP_AC/myFCG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   ```

4. **Get a Groq API key:**
   - Visit: https://console.groq.com/keys
   - Create an account and generate an API key
   - Add it to your `.env` file:
     ```
     GROQ_API_KEY=your_actual_key_here
     ```

## Dataset

The implementation uses the **Adult Income Dataset**:
- **Training data**: `ds/adult/adult.data`
- **Test data**: `ds/adult/adult.test`
- **Sensitive feature**: Sex (0=Male, 1=Female)
- **Target label**: Income (0=≤50K, 1=>50K)

Ensure the dataset is available in the `ds/adult/` directory.

## Usage

### Basic Usage

Run the complete FCG experiment:

```python
python fcg_main.py
```

### Custom Configuration

```python
from fcg_main import run_fcg_experiment

fcg, results = run_fcg_experiment(
    n_clusters=8,           # Number of clusters for K-Means
    m_neighbors=5,          # Neighbors per cluster
    k_shots=5,              # Demonstration samples per iteration
    iterations=10,          # Evolution iterations
    max_dev_samples=50,     # Dev samples for evaluation
    max_test_samples=100    # Test samples for final evaluation
)
```

### Using the FCGAlgorithm Class

```python
from fcg_main import FCGAlgorithm

# Initialize
fcg = FCGAlgorithm(
    n_clusters=8,
    m_neighbors=5,
    k_shots=5,
    iterations=10,
    alpha=0.5,
    p=0.05,
    metric_pred='f1_score',
    metric_fair='ratio_eo',
    model="llama-3.1-8b-instant"
)

# Load data
fcg.load_data(dev_ratio=0.2)

# Run FCG algorithm
fcg.fit()

# Select top demonstrations
fcg.select_demonstrations()

# Evaluate
results = fcg.evaluate(max_test_samples=100)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | 8 | Number of K-Means clusters |
| `m_neighbors` | 5 | Neighbors selected per cluster |
| `k_shots` | 5 | Demonstration samples per iteration |
| `iterations` | 10 | Number of evolution iterations |
| `alpha` | 0.5 | Balance between prediction and fairness |
| `p` | 0.05 | Initial score threshold |
| `metric_pred` | 'f1_score' | Performance metric |
| `metric_fair` | 'ratio_eo' | Fairness metric (Reo) |

## Evaluation Metrics

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive prediction precision
- **Recall**: Positive prediction recall
- **F1-Score**: Harmonic mean of precision and recall

### Fairness Metrics

**Demographic Parity (DP)**:
- `DP_z = P(f(x)=1|Z=z)`
- `Δdp = |DP_1 - DP_0|` (lower is better)
- `Rdp = DP_0 / DP_1` (closer to 1 is better)

**Equalized Odds (Eodds)**:
- `TPR_z = P(f(x)=1|y=1,Z=z)` (True Positive Rate)
- `FPR_z = P(f(x)=1|y=0,Z=z)` (False Positive Rate)
- `Δeo = max(|TPR_1-TPR_0|, |FPR_1-FPR_0|)` (lower is better)
- `Reo = min(TPR_0/TPR_1, FPR_0/FPR_1)` (closer to 1 is better)

## API Configuration

The implementation uses the Groq API with OpenAI-compatible configuration:

```python
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)
```

Default model: `llama-3.1-8b-instant`

## Example Output

```
============================================================
FCG ALGORITHM EXPERIMENT
Fairness via Clustering-Genetic for LLM Bias Mitigation
============================================================

============================================================
LOADING DATA
============================================================
Training samples: 26048
Dev samples:      6512
Test samples:     16281

Creating subgroups based on sensitive feature and label...
  g1: 8752 samples (Z=1, Y=0)
  g2: 1179 samples (Z=1, Y=1)
  g3: 14145 samples (Z=0, Y=0)
  g4: 1972 samples (Z=0, Y=1)

============================================================
STEP 1: DIVERSE CLUSTERING
============================================================
Processing g1: 8752 samples
  Selected 40 diverse samples from g1
...

============================================================
STEP 2: UPDATE EVOLUTION SCORE
============================================================
Processing g1: 40 samples
  Running 10 iterations with k=5 shots
...

============================================================
EVALUATION ON TEST DATA
============================================================

Zero-shot Performance Metrics:
  Accuracy:  0.7245
  F1-Score:  0.5123
  Δeo:       0.1234
  Reo:       0.8567

FCG Performance Metrics:
  Accuracy:  0.7689
  F1-Score:  0.5678
  Δeo:       0.0876
  Reo:       0.9123

IMPROVEMENT OVER BASELINE
  Accuracy:  +0.0444
  F1-Score:  +0.0555
  Δeo:       +0.0358 (lower is better)
  Reo:       +0.0556
```

## Notes

- The algorithm makes API calls to Groq, which may take time depending on dataset size
- Use `max_dev_samples` and `max_test_samples` parameters to control evaluation time
- Baseline predictions are cached to reduce redundant API calls
- The implementation includes retry logic for API failures

## References

This implementation is based on the FCG (Fairness via Clustering-Genetic) algorithm for mitigating fairness bias in Large Language Models through representative sample selection and evolution-based optimization.

## License

This implementation is for educational and research purposes.
