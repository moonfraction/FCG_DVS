"""
Quick Start Example for FCG Algorithm

This script demonstrates how to run the FCG algorithm with minimal configuration.
Adjust parameters as needed for your experiments.
"""

from fcg_main import run_fcg_experiment

def main():
    """
    Run FCG experiment with recommended settings
    """
    print("Starting FCG Algorithm Experiment...")
    print("This will take some time due to LLM API calls.\n")
    
    # Run experiment with optimized parameters for speed
    # For full experiment, increase max_dev_samples and max_test_samples
    fcg, results = run_fcg_experiment(
        n_clusters=8,           # Number of K-Means clusters
        m_neighbors=5,          # Samples per cluster
        k_shots=5,              # Demonstration samples
        iterations=5,           # Reduced for quick testing (use 10 for full)
        max_dev_samples=30,     # Reduced for quick testing (use 50+ for full)
        max_test_samples=50     # Reduced for quick testing (use 100+ for full)
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print("\nKey Results:")
    print(f"Zero-shot F1-Score: {results['zero_shot']['f1_score']:.4f}")
    print(f"FCG F1-Score:       {results['fcg']['f1_score']:.4f}")
    print(f"Improvement:        {results['fcg']['f1_score'] - results['zero_shot']['f1_score']:+.4f}")
    print(f"\nZero-shot Δeo:      {results['zero_shot']['delta_eo']:.4f}")
    print(f"FCG Δeo:            {results['fcg']['delta_eo']:.4f}")
    print(f"Improvement:        {results['zero_shot']['delta_eo'] - results['fcg']['delta_eo']:+.4f} (lower is better)")
    
    return fcg, results


if __name__ == "__main__":
    # Check if .env file exists
    import os
    if not os.path.exists('.env'):
        print("ERROR: .env file not found!")
        print("Please create a .env file with your GROQ_API_KEY")
        print("See .env.example for template")
        exit(1)
    
    # Check if GROQ_API_KEY is set
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set in .env file!")
        print("Please add your Groq API key to .env")
        exit(1)
    
    # Run experiment
    fcg, results = main()
