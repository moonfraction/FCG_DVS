"""
Configuration file for FCG Algorithm
Adjust these parameters based on your experimental needs
"""

# ============================================================
# CLUSTERING PARAMETERS (STEP 1)
# ============================================================
N_CLUSTERS = 8          # Number of K-Means clusters per subgroup
M_NEIGHBORS = 5         # Number of samples to select per cluster centroid

# ============================================================
# GENETIC ALGORITHM PARAMETERS (STEP 2)
# ============================================================
K_SHOTS = 5             # Number of demonstration samples per iteration
ITERATIONS = 10         # Number of evolution iterations per subgroup
ALPHA = 0.5             # Balance coefficient (0.5 = equal weight)
P_INITIAL = 0.05        # Initial score threshold

# ============================================================
# EVALUATION METRICS
# ============================================================
METRIC_PRED = 'f1_score'    # Performance metric: 'accuracy', 'precision', 'recall', 'f1_score'
METRIC_FAIR = 'ratio_eo'    # Fairness metric: 'ratio_eo', 'ratio_dp', 'delta_eo', 'delta_dp'

# ============================================================
# DATA PARAMETERS
# ============================================================
SENSITIVE_FEATURE = 'sex'   # Sensitive attribute column name
LABEL = 'income'            # Target label column name
DEV_RATIO = 0.2             # Proportion of training data for dev set

# ============================================================
# LLM PARAMETERS
# ============================================================
MODEL = "llama-3.1-8b-instant"  # Groq model name
# Other options: "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"

# ============================================================
# EFFICIENCY PARAMETERS
# ============================================================
MAX_DEV_SAMPLES = 50    # Max dev samples per evaluation (higher = more accurate but slower)
MAX_TEST_SAMPLES = 100  # Max test samples for final evaluation

# ============================================================
# QUICK TEST MODE (for debugging)
# ============================================================
QUICK_TEST = False       # Set to True for fast testing with reduced parameters

if QUICK_TEST:
    print("WARNING: Running in QUICK TEST mode with reduced parameters")
    ITERATIONS = 3
    MAX_DEV_SAMPLES = 20
    MAX_TEST_SAMPLES = 30
    N_CLUSTERS = 4
    M_NEIGHBORS = 3
