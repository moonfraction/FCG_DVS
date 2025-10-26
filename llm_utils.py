"""
LLM Integration with Groq API for In-Context Learning
"""
import os
import openai
from dotenv import load_dotenv
import numpy as np
import time

# Load environment variables
load_dotenv()

# Initialize Groq client
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)


def format_sample_for_prompt(row, include_label=True):
    """
    Format a data sample as text for the LLM prompt
    
    Args:
        row: DataFrame row with sample data
        include_label: Whether to include the income label
    
    Returns:
        Formatted string representation
    """
    # Map sex values
    sex_map = {0: 'Male', 1: 'Female'}
    income_map = {0: 'less than or equal to 50K', 1: 'greater than 50K'}
    
    # Create readable representation
    text = f"Age: {row['age']}, "
    text += f"Workclass: {row['workclass']}, "
    text += f"Education: {row['education']}, "
    text += f"Marital Status: {row['marital-status']}, "
    text += f"Occupation: {row['occupation']}, "
    text += f"Relationship: {row['relationship']}, "
    text += f"Sex: {sex_map.get(row['sex'], row['sex'])}, "
    text += f"Capital Gain: {row['capital-gain']}, "
    text += f"Capital Loss: {row['capital-loss']}, "
    text += f"Hours per Week: {row['hours-per-week']}"
    
    if include_label and 'income' in row:
        text += f" -> Income: {income_map.get(row['income'], row['income'])}"
    
    return text


def create_task_description():
    """
    Create task description for the LLM prompt
    """
    task_desc = """You are tasked with predicting whether a person's income is greater than 50K or less than or equal to 50K based on their demographic and work-related information.

Based on the following information, predict the income category. Respond with ONLY one of these two options:
- "greater than 50K"
- "less than or equal to 50K"

Do not provide any explanation, just the prediction."""
    
    return task_desc


def create_prompt(demonstrations=None, test_sample=None):
    """
    Create a prompt for LLM prediction
    
    Args:
        demonstrations: List of demonstration samples (DataFrame rows) for few-shot
        test_sample: Test sample to predict (DataFrame row)
    
    Returns:
        Formatted prompt string
    """
    prompt = create_task_description()
    
    # Add demonstrations for few-shot learning
    if demonstrations is not None and len(demonstrations) > 0:
        prompt += "\n\nHere are some examples:\n\n"
        for idx, demo in demonstrations.iterrows():
            prompt += format_sample_for_prompt(demo, include_label=True) + "\n"
    
    # Add test sample
    if test_sample is not None:
        prompt += f"\nNow predict for this person:\n"
        prompt += format_sample_for_prompt(test_sample, include_label=False)
        prompt += "\n\nIncome prediction:"
    
    return prompt


def llm_predict(prompt, model="llama-3.1-8b-instant", temperature=0.1, max_retries=3):
    """
    Get prediction from LLM via Groq API
    
    Args:
        prompt: Input prompt
        model: Model name (default: llama-3.1-8b-instant)
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts
    
    Returns:
        Predicted label (0 or 1)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=50
            )
            
            prediction_text = response.choices[0].message.content.strip().lower()
            
            # Parse prediction
            if "greater than 50k" in prediction_text or ">50k" in prediction_text:
                return 1
            elif "less than or equal to 50k" in prediction_text or "<=50k" in prediction_text:
                return 0
            else:
                # Default to 0 if unclear
                return 0
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)  # Wait before retry
                continue
            else:
                print(f"  API failed after {max_retries} attempts: {e}")
                return 0  # Default prediction
    
    return 0


def llm_predict_batch(test_df, demonstrations=None, model="llama-3.1-8b-instant", max_samples=None):
    """
    Predict labels for a batch of test samples
    
    Args:
        test_df: DataFrame with test samples
        demonstrations: DataFrame with demonstration samples for ICL (None for zero-shot)
        model: Model name
        max_samples: Maximum number of samples to predict (for testing)
    
    Returns:
        Array of predictions
    """
    predictions = []
    
    # Limit samples if specified
    if max_samples is not None:
        test_df = test_df.head(max_samples)
    
    print(f"Predicting {len(test_df)} samples...")
    
    for idx, row in test_df.iterrows():
        # Create prompt
        prompt = create_prompt(demonstrations=demonstrations, test_sample=row)
        
        # Get prediction
        pred = llm_predict(prompt, model=model)
        predictions.append(pred)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)} samples")
    
    return np.array(predictions)


def llm_predict_zero_shot(test_df, model="llama-3.1-8b-instant", max_samples=None):
    """
    Zero-shot prediction (no demonstrations)
    """
    return llm_predict_batch(test_df, demonstrations=None, model=model, max_samples=max_samples)


def llm_predict_few_shot(test_df, demonstrations, model="llama-3.1-8b-instant", max_samples=None):
    """
    Few-shot prediction with demonstrations
    """
    return llm_predict_batch(test_df, demonstrations=demonstrations, model=model, max_samples=max_samples)
