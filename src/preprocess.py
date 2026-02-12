"""
Dataset loading and preprocessing for GSM8K.
"""

from pathlib import Path
from typing import Dict, List
from datasets import load_dataset


def load_gsm8k(split: str = "test", max_samples: int = None, cache_dir: str = ".cache") -> List[Dict]:
    """
    Load GSM8K dataset.
    
    Args:
        split: Dataset split (train or test)
        max_samples: Maximum number of samples to load
        cache_dir: Directory to cache the dataset
    
    Returns:
        List of examples with 'question' and 'answer' fields
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=str(cache_path))
    
    # Convert to list of dicts
    examples = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        # GSM8K format: question and answer (answer contains reasoning + #### + numeric answer)
        examples.append({
            "id": i,
            "question": item["question"],
            "answer": item["answer"],  # Full answer with reasoning
            "ground_truth": extract_numeric_answer(item["answer"])  # Extract numeric answer
        })
    
    print(f"Loaded {len(examples)} examples from GSM8K {split} split")
    return examples


def extract_numeric_answer(answer_text: str) -> float:
    """
    Extract numeric answer from GSM8K answer format.
    GSM8K answers end with #### followed by the numeric answer.
    
    Args:
        answer_text: Full answer text
    
    Returns:
        Numeric answer
    """
    if "####" in answer_text:
        numeric_part = answer_text.split("####")[-1].strip()
        # Remove commas and other formatting
        numeric_part = numeric_part.replace(",", "").replace("$", "")
        try:
            return float(numeric_part)
        except ValueError:
            print(f"Warning: Could not parse numeric answer: {numeric_part}")
            return None
    return None
