"""
Main FCG Algorithm Implementation
Fairness via Clustering-Genetic for LLM Bias Mitigation
"""
import numpy as np
import pandas as pd
from data_utils import loadAdult, split_train_dev, create_subgroups, Subgroup
from clustering import step1_diverse_clustering
from evolution import step2_update_evol_score
from genetic_algorithm import select_top_demonstrations
from llm_utils import llm_predict_few_shot, llm_predict_zero_shot
from metrics import evaluate_all_metrics, print_metrics


class FCGAlgorithm:
    """
    Fairness via Clustering-Genetic (FCG) Algorithm
    
    Mitigates fairness bias in LLMs via representative sample selection
    and evolution-based optimization.
    """
    
    def __init__(self, 
                 n_clusters=8,
                 m_neighbors=5,
                 k_shots=5,
                 iterations=10,
                 alpha=0.5,
                 p=0.05,
                 metric_pred='f1_score',
                 metric_fair='ratio_eo',
                 sensitive_feature='sex',
                 label='income',
                 model="llama-3.1-8b-instant",
                 max_dev_samples=50):
        """
        Initialize FCG algorithm with hyperparameters
        
        Args:
            n_clusters: Number of clusters for K-Means (default: 8)
            m_neighbors: Neighbors per cluster (default: 5)
            k_shots: Demonstration samples per iteration (default: 5)
            iterations: Evolution iterations (default: 10)
            alpha: Balance coefficient (default: 0.5)
            p: Initial score threshold (default: 0.05)
            metric_pred: Performance metric (default: 'f1_score')
            metric_fair: Fairness metric (default: 'ratio_eo')
            sensitive_feature: Sensitive attribute name (default: 'sex')
            label: Target label name (default: 'income')
            model: LLM model name
            max_dev_samples: Max dev samples per evaluation
        """
        self.n_clusters = n_clusters
        self.m_neighbors = m_neighbors
        self.k_shots = k_shots
        self.iterations = iterations
        self.alpha = alpha
        self.p = p
        self.metric_pred = metric_pred
        self.metric_fair = metric_fair
        self.sensitive_feature = sensitive_feature
        self.label = label
        self.model = model
        self.max_dev_samples = max_dev_samples
        
        # Data storage
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.subgroups = None
        self.evolved_subgroups = None
        self.top_demonstrations = None
    
    def load_data(self, dev_ratio=0.2):
        """
        Load and split Adult dataset
        """
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load Adult dataset
        dftr, dftst = loadAdult()
        
        # Split training into train and dev
        self.train_data, self.dev_data = split_train_dev(dftr, dev_ratio=dev_ratio)
        self.test_data = dftst
        
        print(f"Training samples: {len(self.train_data)}")
        print(f"Dev samples:      {len(self.dev_data)}")
        print(f"Test samples:     {len(self.test_data)}")
        
        # Create subgroups from training data
        print("\nCreating subgroups based on sensitive feature and label...")
        subgroups_raw = create_subgroups(
            self.train_data, 
            sensitive_feature=self.sensitive_feature,
            label=self.label
        )
        
        # Convert to Subgroup objects
        self.subgroups = {}
        for key, df in subgroups_raw.items():
            z_val = 1 if '1' in key[1] else 0  # Extract Z value from key
            y_val = int(key[-1])  # Extract Y value from key
            self.subgroups[key] = Subgroup(df, z_val, y_val, initial_score=self.p)
            print(f"  {key}: {len(df)} samples (Z={z_val}, Y={y_val})")
        
        return self.train_data, self.dev_data, self.test_data
    
    def fit(self):
        """
        Run FCG algorithm: STEP 1 (Clustering) + STEP 2 (Evolution)
        """
        if self.subgroups is None:
            raise ValueError("Must call load_data() before fit()")
        
        # STEP 1: Diverse Clustering
        print("\n" + "="*60)
        print("STEP 1: DIVERSE CLUSTERING")
        print("="*60)
        self.evolved_subgroups = step1_diverse_clustering(
            self.subgroups,
            n_clusters=self.n_clusters,
            m_neighbors=self.m_neighbors
        )
        
        # STEP 2: Update Evolution Score
        self.evolved_subgroups = step2_update_evol_score(
            self.evolved_subgroups,
            self.dev_data,
            k_shots=self.k_shots,
            iterations=self.iterations,
            metric_pred=self.metric_pred,
            metric_fair=self.metric_fair,
            alpha=self.alpha,
            p=self.p,
            model=self.model,
            max_dev_samples=self.max_dev_samples
        )
        
        return self.evolved_subgroups
    
    def select_demonstrations(self, k_per_subgroup=None):
        """
        Select top demonstrations from each evolved subgroup
        
        Args:
            k_per_subgroup: Number of top samples per subgroup (default: self.k_shots)
        """
        if self.evolved_subgroups is None:
            raise ValueError("Must call fit() before select_demonstrations()")
        
        if k_per_subgroup is None:
            k_per_subgroup = self.k_shots
        
        print("\n" + "="*60)
        print("SELECTING TOP DEMONSTRATIONS")
        print("="*60)
        
        self.top_demonstrations = []
        
        for key, subgroup in self.evolved_subgroups.items():
            k_actual = min(k_per_subgroup, len(subgroup))
            top_samples = select_top_demonstrations(subgroup, k_shots=k_actual)
            self.top_demonstrations.append(top_samples)
            print(f"{key}: Selected {len(top_samples)} top samples")
        
        # Combine all demonstrations
        self.top_demonstrations = pd.concat(self.top_demonstrations, ignore_index=True)
        
        print(f"\nTotal demonstrations selected: {len(self.top_demonstrations)}")
        
        return self.top_demonstrations
    
    def evaluate(self, max_test_samples=100):
        """
        Evaluate FCG on test data
        
        Args:
            max_test_samples: Maximum test samples to evaluate (for speed)
        """
        if self.top_demonstrations is None:
            raise ValueError("Must call select_demonstrations() before evaluate()")
        
        print("\n" + "="*60)
        print("EVALUATION ON TEST DATA")
        print("="*60)
        
        # Limit test samples
        test_subset = self.test_data.head(max_test_samples) if len(self.test_data) > max_test_samples else self.test_data
        
        y_true = test_subset[self.label].values
        z_sensitive = test_subset[self.sensitive_feature].values
        
        # Zero-shot baseline
        print("\n1. Zero-shot Baseline:")
        y_zero = llm_predict_zero_shot(test_subset, model=self.model, max_samples=max_test_samples)
        zero_metrics = evaluate_all_metrics(y_true, y_zero, z_sensitive)
        print_metrics(zero_metrics, prefix="Zero-shot ")
        
        # FCG with demonstrations
        print("\n2. FCG with In-Context Learning:")
        y_fcg = llm_predict_few_shot(
            test_subset, 
            demonstrations=self.top_demonstrations,
            model=self.model,
            max_samples=max_test_samples
        )
        fcg_metrics = evaluate_all_metrics(y_true, y_fcg, z_sensitive)
        print_metrics(fcg_metrics, prefix="FCG ")
        
        # Comparison
        print("\n" + "="*60)
        print("IMPROVEMENT OVER BASELINE")
        print("="*60)
        print(f"Accuracy:  {fcg_metrics['accuracy'] - zero_metrics['accuracy']:+.4f}")
        print(f"F1-Score:  {fcg_metrics['f1_score'] - zero_metrics['f1_score']:+.4f}")
        print(f"Δeo:       {zero_metrics['delta_eo'] - fcg_metrics['delta_eo']:+.4f} (lower is better)")
        print(f"Reo:       {fcg_metrics['ratio_eo'] - zero_metrics['ratio_eo']:+.4f}")
        
        return {
            'zero_shot': zero_metrics,
            'fcg': fcg_metrics
        }


def run_fcg_experiment(n_clusters=8, m_neighbors=5, k_shots=5, iterations=10,
                       max_dev_samples=50, max_test_samples=100):
    """
    Run complete FCG experiment on Adult dataset
    
    Args:
        n_clusters: Number of clusters (default: 8)
        m_neighbors: Neighbors per cluster (default: 5)
        k_shots: Demonstration samples (default: 5)
        iterations: Evolution iterations (default: 10)
        max_dev_samples: Max dev samples per evaluation
        max_test_samples: Max test samples for final evaluation
    """
    print("\n" + "="*60)
    print("FCG ALGORITHM EXPERIMENT")
    print("Fairness via Clustering-Genetic for LLM Bias Mitigation")
    print("="*60)
    
    # Initialize FCG
    fcg = FCGAlgorithm(
        n_clusters=n_clusters,
        m_neighbors=m_neighbors,
        k_shots=k_shots,
        iterations=iterations,
        alpha=0.5,
        p=0.05,
        metric_pred='f1_score',
        metric_fair='ratio_eo',
        model="llama-3.1-8b-instant",
        max_dev_samples=max_dev_samples
    )
    
    # Load data
    fcg.load_data(dev_ratio=0.2)
    
    # Run FCG algorithm
    fcg.fit()
    
    # Select top demonstrations
    fcg.select_demonstrations(k_per_subgroup=k_shots)
    
    # Evaluate
    results = fcg.evaluate(max_test_samples=max_test_samples)
    
    return fcg, results


if __name__ == "__main__":
    # Run experiment with default parameters
    fcg, results = run_fcg_experiment(
        n_clusters=8,
        m_neighbors=5,
        k_shots=5,
        iterations=10,
        max_dev_samples=50,
        max_test_samples=100
    )
