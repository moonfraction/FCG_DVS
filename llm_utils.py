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

from logger_utils import get_logger

logger = get_logger()

# Initialize Groq client
# client = openai.OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.environ.get("GROQ_API_KEY")
# )


api_keys_cerebreas = {
    "CEREBRAS_API_KEY_2": os.environ.get("CEREBRAS_API_KEY_2"),
    "CEREBRAS_API_KEY_1": os.environ.get("CEREBRAS_API_KEY_1"),
    "CEREBRAS_API_KEY_firefox": os.environ.get("CEREBRAS_API_KEY_firefox"),
    "CEREBRAS_API_KEY_safari": os.environ.get("CEREBRAS_API_KEY_safari"),
    "CEREBRAS_API_KEY_brave": os.environ.get("CEREBRAS_API_KEY_brave")
}
# Placeholders for keys and client objects; initialized by init_llm_clients()
_keys = []
CLIENTS = []


def _mask_key(k):
    if not k:
        return 'NONE'
    return (k[:4] + '...' + k[-4:]) if len(k) > 8 else '****'


def init_llm_clients(base_url='https://api.cerebras.ai/v1', keys=None):
    """
    Initialize LLM clients from a list of API keys or from environment variables.

    Args:
        base_url: Base URL for the OpenAI-compatible API
        keys: Optional list of API key strings. If None, will read env vars
              CEREBRAS_API_KEY*, GROQ_API_KEY.
    """
    global _keys, CLIENTS

    # If explicit keys provided, use them; otherwise gather from env vars
    if keys is None:
        env_keys = []
        # # primary env var
        # primary = os.environ.get('CEREBRAS_API_KEY') or os.environ.get('GROQ_API_KEY')
        # if primary:
        #     env_keys.append(primary)

        # extras
        for v in api_keys_cerebreas.values():
            if v:
                env_keys.append(v)

        keys_to_use = [k.strip().strip('"').strip("'") for k in env_keys if k]
    else:
        keys_to_use = [k.strip().strip('"').strip("'") for k in keys if k]

    # Remove duplicates while preserving order
    seen = set()
    _keys = []
    for k in keys_to_use:
        if k not in seen:
            seen.add(k)
            _keys.append(k)

    CLIENTS = []
    if len(_keys) == 0:
        logger.warning("No Cerebras/Groq API keys found; LLM calls will likely fail until a key is provided.")
        CLIENTS.append(openai.OpenAI(base_url=base_url, api_key=None))
    else:
        for k in _keys:
            CLIENTS.append(openai.OpenAI(base_url=base_url, api_key=k))

    if len(_keys) > 0:
        logger.info(f"Initialized {len(CLIENTS)} LLM client(s); first key masked={_mask_key(_keys[0])}")


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


def llm_predict(prompt, model="llama-3.1-8b-instant", temperature=0.0, max_retries=3, client_idx=0):
    """
    Get prediction from LLM using the current client index. If the current
    client fails for all `max_retries`, automatically switch to the next
    client and retry the same prompt (up to one full rotation across clients).

    Args:
        prompt: Input prompt
        model: Model name (default: llama-3.1-8b-instant)
        temperature: Sampling temperature
        max_retries: Maximum retry attempts per client before switching
        client_idx: Index of the client/key to use initially

    Returns:
        Tuple (pred_label, next_client_idx)
          - pred_label: 0 or 1
          - next_client_idx: client index to use for subsequent calls
    """
    # Ensure clients are initialized
    if not CLIENTS:
        try:
            init_llm_clients()
        except Exception as e:
            logger.warning(f"init_llm_clients failed during llm_predict: {e}")

    if not CLIENTS:
        logger.error("No LLM clients available; returning default prediction 0")
        return 0, client_idx

    n_clients = len(CLIENTS)
    start_idx = client_idx % n_clients
    tries_across_clients = 0
    idx = start_idx

    while tries_across_clients < n_clients:
        client = CLIENTS[idx]
        masked = _mask_key(_keys[idx]) if idx < len(_keys) else 'NONE'
        logger.debug(f"Using client index {idx} with key={masked} (up to {max_retries} retries)")

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

                # Try to extract text robustly
                try:
                    prediction_text = response.choices[0].message.content.strip().lower()
                except Exception:
                    # Fallback to str conversion
                    prediction_text = str(response).strip().lower()

                logger.debug(f"  Prediction (client {idx} attempt {attempt + 1}): {prediction_text}")

                # Parse prediction
                if "greater than 50k" in prediction_text or ">50k" in prediction_text or "greater than 50 k" in prediction_text:
                    return 1, idx
                elif "less than or equal to 50k" in prediction_text or "<=50k" in prediction_text or "less than or equal to 50 k" in prediction_text:
                    return 0, idx
                else:
                    logger.warning(f"  Unclear prediction: {prediction_text}")
                    # Default to 0 if unclear (do not switch key on parse ambiguity)
                    return 0, idx

            except Exception as e:
                logger.warning(f"  Client {idx} API error (attempt {attempt + 1}/{max_retries}): {e}")
                # If we have more retries for this client, wait and retry
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"  Client {idx} failed after {max_retries} attempts; moving to next client and retrying this prompt.")
                    break  # move to next client

        # advance to next client and keep track of total client switches for this prompt
        idx = (idx + 1) % n_clients
        tries_across_clients += 1

    # All clients exhausted for this prompt
    logger.error("All LLM clients failed for this prompt; returning default prediction 0")
    return 0, idx


def llm_predict_batch(test_df, demonstrations=None, model="llama-3.1-8b-instant", max_samples=None, start_client_idx=0):
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
    current_idx = start_client_idx or 0
    
    # Limit samples if specified
    if max_samples is not None:
        test_df = test_df.head(max_samples)
    
    logger.info(f"Predicting {len(test_df)} samples...")
    
    # Use enumerate to track processed count instead of DataFrame index values
    for i, (_, row) in enumerate(test_df.iterrows(), start=1):
        # Create prompt
        prompt = create_prompt(demonstrations=demonstrations, test_sample=row)

        # Get prediction with current API key index; llm_predict may update the index if the key is exhausted
        pred, current_idx = llm_predict(prompt, model=model, client_idx=current_idx)
        predictions.append(pred)
        
        # Log every 10 items and at the end
        if (i % 10 == 0) or (i == len(test_df)):
            logger.info(f"  Processed {i}/{len(test_df)} samples")
    
    return np.array(predictions)


def llm_predict_zero_shot(test_df, model="llama-3.1-8b-instant", max_samples=None, start_client_idx=0):
    """
    Zero-shot prediction (no demonstrations)
    """
    return llm_predict_batch(test_df, demonstrations=None, model=model, max_samples=max_samples, start_client_idx=start_client_idx)


def llm_predict_few_shot(test_df, demonstrations, model="llama-3.1-8b-instant", max_samples=None, start_client_idx=0):
    """
    Few-shot prediction with demonstrations
    """
    return llm_predict_batch(test_df, demonstrations=demonstrations, model=model, max_samples=max_samples, start_client_idx=start_client_idx)
