"""
Test script to verify dataset and dependencies
"""
import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    required_packages = ['pandas', 'numpy', 'sklearn', 'scipy', 'openai', 'dotenv']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print("\nMissing packages. Install with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_dataset():
    """Check if Adult dataset files exist"""
    print("\nChecking dataset files...")
    
    files_to_check = [
        'ds/adult/adult.data',
        'ds/adult/adult.test'
    ]
    
    all_exist = True
    for filepath in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filepath} ({size:,} bytes)")
        else:
            print(f"  ✗ {filepath} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\nDataset files missing!")
        print("Please ensure Adult dataset is in ds/adult/ directory")
        return False
    
    print("\n✓ All dataset files found!")
    return True


def check_env():
    """Check if .env file exists and has API key"""
    print("\nChecking environment configuration...")
    
    if not os.path.exists('.env'):
        print("  ✗ .env file not found")
        print("\nPlease create .env file with your GROQ_API_KEY")
        print("Copy from .env.example:")
        print("  cp .env.example .env")
        print("Then edit .env and add your API key")
        return False
    
    print("  ✓ .env file exists")
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("  ✗ GROQ_API_KEY not set or using placeholder")
        print("\nPlease add your actual Groq API key to .env file")
        print("Get one from: https://console.groq.com/keys")
        return False
    
    print("  ✓ GROQ_API_KEY is set")
    print("\n✓ Environment configured!")
    return True


def main():
    """Run all checks"""
    print("="*60)
    print("FCG ALGORITHM - SETUP VERIFICATION")
    print("="*60)
    print()
    
    deps_ok = check_dependencies()
    data_ok = check_dataset()
    env_ok = check_env()
    
    print("="*60)
    if deps_ok and data_ok and env_ok:
        print("✓ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to run the FCG algorithm!")
        print("Run: python run_experiment.py")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the algorithm")
        return 1


if __name__ == "__main__":
    sys.exit(main())
