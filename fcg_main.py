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
from logger_utils import get_logger

logger = get_logger()


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
        logger.info("="*60)
        logger.info("LOADING DATA")
        logger.info("="*60)
        
        # Load Adult dataset
        dftr, dftst = loadAdult()
        
        # Split training into train and dev
        self.train_data, self.dev_data = split_train_dev(dftr, dev_ratio=dev_ratio)
        self.test_data = dftst
        
        logger.info(f"Training samples: {len(self.train_data)}")
        logger.info(f"Dev samples:      {len(self.dev_data)}")
        logger.info(f"Test samples:     {len(self.test_data)}")
        
        # Create subgroups from training data
        logger.info("Creating subgroups based on sensitive feature and label...")
        subgroups_raw = create_subgroups(
            self.train_data, 
            sensitive_feature=self.sensitive_feature,
            label=self.label
        )
        
        # Convert to Subgroup objects
        key_aliases = {
            'g10': 'g1 (Z=1,Y=0)',
            'g11': 'g2 (Z=1,Y=1)',
            'g00': 'g3 (Z=0,Y=0)',
            'g01': 'g4 (Z=0,Y=1)'
        }

        self.subgroups = {}
        for key, df in subgroups_raw.items():
            z_val = 1 if '1' in key[1] else 0  # Extract Z value from key
            y_val = int(key[-1])  # Extract Y value from key
            self.subgroups[key] = Subgroup(df, z_val, y_val, initial_score=self.p)
            logger.info(f"  {key_aliases[key]}: {len(df)} samples (Z={z_val}, Y={y_val})")
        
        return self.train_data, self.dev_data, self.test_data
    
    def fit(self):
        """
        Run FCG algorithm: STEP 1 (Clustering) + STEP 2 (Evolution)
        """
        if self.subgroups is None:
            raise ValueError("Must call load_data() before fit()")
        
        # STEP 1: Diverse Clustering
        logger.info("="*60)
        logger.info("STEP 1: DIVERSE CLUSTERING")
        logger.info("="*60)
        self.evolved_subgroups = step1_diverse_clustering(
            self.subgroups,
            n_clusters=self.n_clusters,
            m_neighbors=self.m_neighbors
        )

        logger.info("Clustered subgroup sizes:")
        for key in self.evolved_subgroups:
            logger.info(f"{key}: {len(self.evolved_subgroups[key])} samples")  # DEBUGGING LINE
        
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
        
        logger.info("="*60)
        logger.info("SELECTING TOP DEMONSTRATIONS")
        logger.info("="*60)
        
        self.top_demonstrations = []
        
        for key, subgroup in self.evolved_subgroups.items():
            k_actual = min(k_per_subgroup, len(subgroup))
            top_samples = select_top_demonstrations(subgroup, k_shots=k_actual)
            self.top_demonstrations.append(top_samples)
            logger.info(f"{key}: Selected {len(top_samples)} top samples")
        
        # Combine all demonstrations
        self.top_demonstrations = pd.concat(self.top_demonstrations, ignore_index=True)
        
        logger.info(f"Total demonstrations selected: {len(self.top_demonstrations)}")
        
        return self.top_demonstrations
    
    def evaluate(self, max_test_samples=100):
        """
        Evaluate FCG on test data
        
        Args:
            max_test_samples: Maximum test samples to evaluate (for speed)
        """
        if self.top_demonstrations is None:
            raise ValueError("Must call select_demonstrations() before evaluate()")
        
        logger.info("="*60)
        logger.info("EVALUATION ON TEST DATA")
        logger.info("="*60)
        
        # Check if test_subset.csv exists and load it, otherwise create the subset
        try:
            test_subset = pd.read_csv("test_subset.csv")
            print("Test subset loaded from test_subset.csv")
        except FileNotFoundError:
            # Limit test samples
            test_subset = self.test_data.head(max_test_samples) if len(self.test_data) > max_test_samples else self.test_data
            
            # Save 
            test_subset.to_csv("test_subset.csv", index=False)
            print("Test subset saved to test_subset.csv")
        
        y_true = test_subset[self.label].values
        z_sensitive = test_subset[self.sensitive_feature].values
        
        # Zero-shot baseline
        logger.info("1. Zero-shot Baseline:")
        y_zero = llm_predict_zero_shot(test_subset, model=self.model, max_samples=max_test_samples)
        zero_metrics = evaluate_all_metrics(y_true, y_zero, z_sensitive)
        print_metrics(zero_metrics, prefix="Zero-shot ")
        
        # FCG with demonstrations
        logger.info("2. FCG with In-Context Learning:")
        y_fcg = llm_predict_few_shot(
            test_subset, 
            demonstrations=self.top_demonstrations,
            model=self.model,
            max_samples=max_test_samples
        )
        fcg_metrics = evaluate_all_metrics(y_true, y_fcg, z_sensitive)
        print_metrics(fcg_metrics, prefix="FCG ")
        
        # Comparison
        logger.info("="*60)
        logger.info("IMPROVEMENT OVER BASELINE")
        logger.info("="*60)
        logger.info(f"Accuracy:  {fcg_metrics['accuracy'] - zero_metrics['accuracy']:+.4f}")
        logger.info(f"F1-Score:  {fcg_metrics['f1_score'] - zero_metrics['f1_score']:+.4f}")
        logger.info(f"Δeo:       {zero_metrics['delta_eo'] - fcg_metrics['delta_eo']:+.4f} (lower is better)")
        logger.info(f"Reo:       {fcg_metrics['ratio_eo'] - zero_metrics['ratio_eo']:+.4f}")
        
        return {
            'zero_shot': zero_metrics,
            'fcg': fcg_metrics
        }
    
    def evaluate_with_dvs(self, max_test_samples=100, batch_size=10, k_neighbors=5, 
                         L_iterations=5, use_fcg_baseline=True):
        """
        Evaluate using FCG-DVS hybrid approach
        
        Phase 1: FCG creates global optimized pool (already done in fit())
        Phase 2: DVS adaptively selects ICE for each test batch
        
        Args:
            max_test_samples: Maximum test samples to evaluate
            batch_size: Size of each test batch (N in the algorithm)
            k_neighbors: Number of neighbors to retrieve per test sample
            L_iterations: Number of DVS refinement iterations
            use_fcg_baseline: If True, also run FCG (non-DVS) for comparison
        
        Returns:
            dict with evaluation metrics
        """
        from dvs_utils import init_gemini, embed_dataframe, create_batch_support_sets, create_dvs_and_candidates
        from dvs_ice_selection import select_ice_iterative
        
        if self.top_demonstrations is None:
            raise ValueError("Must call select_demonstrations() before evaluate_with_dvs()")
        
        logger.info("="*60)
        logger.info("FCG-DVS HYBRID EVALUATION")
        logger.info("="*60)
        
        # Load or create test subset
        try:
            test_subset = pd.read_csv("test_subset.csv")
            logger.info("Test subset loaded from test_subset.csv")
        except FileNotFoundError:
            test_subset = self.test_data.head(max_test_samples) if len(self.test_data) > max_test_samples else self.test_data
            test_subset.to_csv("test_subset.csv", index=False)
            logger.info("Test subset saved to test_subset.csv")
        
        # Initialize Gemini API
        try:
            init_gemini()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            logger.info("Falling back to standard FCG evaluation")
            return self.evaluate(max_test_samples=max_test_samples)
        
        # Phase 1: Embed the FCG-optimized pool (H_opt)
        logger.info("Phase 1: Embedding FCG-optimized pool...")
        pool_cache_path = f"data/embeddings/fcg_pool_{len(self.top_demonstrations)}.pkl"
        pool_embeddings = embed_dataframe(self.top_demonstrations, cache_path=pool_cache_path)
        
        # Phase 2: Process test data in batches
        logger.info(f"Phase 2: Processing {len(test_subset)} test samples in batches of {batch_size}")
        
        n_batches = (len(test_subset) + batch_size - 1) // batch_size
        all_predictions_dvs = []
        all_predictions_fcg = [] if use_fcg_baseline else None
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(test_subset))
            batch_df = test_subset.iloc[start_idx:end_idx].reset_index(drop=True)
            
            logger.info(f"\nBatch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx}):")
            
            # 2a: Create support sets (S_n)
            support_info = create_batch_support_sets(
                batch_df,
                self.top_demonstrations,
                pool_embeddings,
                k=k_neighbors,
                cache_dir='data/embeddings'
            )
            
            # 2b: Create DVS (P_n) and candidate pool (H_n)
            dvs_info = create_dvs_and_candidates(batch_df, self.top_demonstrations, support_info)
            
            # 2c: Iterative ICE selection
            ice_result = select_ice_iterative(
                batch_df,
                dvs_info['dvs_df'],
                dvs_info['candidate_df'],
                support_info['support_indices'],
                self.top_demonstrations,
                L=L_iterations,
                model=self.model,
                alpha=self.alpha,
                metric_pred=self.metric_pred,
                metric_fair=self.metric_fair
            )
            
            best_ice_df = ice_result['best_ice_df']
            
            # 2d: Predict on this batch using the selected ICE
            logger.info(f"  Predicting on batch with |ICE| = {len(best_ice_df)}")
            batch_preds = llm_predict_few_shot(
                batch_df,
                demonstrations=best_ice_df,
                model=self.model,
                max_samples=len(batch_df)
            )
            all_predictions_dvs.extend(batch_preds.tolist())
            
            # Optional: Also get FCG predictions for comparison
            if use_fcg_baseline:
                batch_preds_fcg = llm_predict_few_shot(
                    batch_df,
                    demonstrations=self.top_demonstrations,
                    model=self.model,
                    max_samples=len(batch_df)
                )
                all_predictions_fcg.extend(batch_preds_fcg.tolist())
        
        # Evaluate DVS results
        y_true = test_subset[self.label].values
        z_sensitive = test_subset[self.sensitive_feature].values
        
        dvs_metrics = evaluate_all_metrics(y_true, np.array(all_predictions_dvs), z_sensitive)
        
        logger.info("="*60)
        logger.info("FCG-DVS RESULTS")
        logger.info("="*60)
        print_metrics(dvs_metrics, prefix="FCG-DVS ")
        
        results = {'fcg_dvs': dvs_metrics}
        
        # Optional FCG baseline comparison
        if use_fcg_baseline:
            fcg_metrics = evaluate_all_metrics(y_true, np.array(all_predictions_fcg), z_sensitive)
            logger.info("\nFCG (Standard) Results:")
            print_metrics(fcg_metrics, prefix="FCG ")
            results['fcg'] = fcg_metrics
            
            logger.info("="*60)
            logger.info("FCG-DVS vs FCG IMPROVEMENT")
            logger.info("="*60)
            logger.info(f"Accuracy:  {dvs_metrics['accuracy'] - fcg_metrics['accuracy']:+.4f}")
            logger.info(f"F1-Score:  {dvs_metrics['f1_score'] - fcg_metrics['f1_score']:+.4f}")
            logger.info(f"Δeo:       {fcg_metrics['delta_eo'] - dvs_metrics['delta_eo']:+.4f} (lower is better)")
            logger.info(f"Reo:       {dvs_metrics['ratio_eo'] - fcg_metrics['ratio_eo']:+.4f}")
        
        return results


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
    logger.info("="*60)
    logger.info("FCG ALGORITHM EXPERIMENT")
    logger.info("Fairness via Clustering-Genetic for LLM Bias Mitigation")
    logger.info("="*60)
    
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
        model="llama-3.3-70b",
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
        max_dev_samples=40,
        max_test_samples=100
    )
