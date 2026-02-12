"""
Inference script for prompt tuning experiments.
Supports CoT baseline and ET-CoT with deterministic verification.
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import wandb
from omegaconf import OmegaConf

# Add parent directory to path to allow imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import LLMGenerator
from src.preprocess import load_gsm8k, extract_numeric_answer


# Prompt templates
COT_BASELINE_PROMPT = """Solve this math word problem step by step. Show your reasoning and provide the final numeric answer.

Question: {question}

Think step by step and end with your final answer."""

ET_COT_PROMPT = """Solve this math word problem using a structured format. You must output:
1. VARS: A JSON dictionary of named intermediate quantities (numbers only)
2. TRACE: Python-like assignments using only numbers and previously defined variables
3. FINAL: The final numeric answer

Format your response exactly like this:
VARS: {{"var1": value1, "var2": value2, ...}}
TRACE:
result1 = var1 + var2
result2 = result1 * 3
...
FINAL: <number>

Question: {question}

Provide your structured solution:"""

ET_COT_REPAIR_PROMPT = """Your previous solution failed verification: {error_message}

Please provide a CORRECTED solution in the same structured format:
VARS: {{...}}
TRACE:
...
FINAL: <number>

Question: {question}

Corrected solution:"""


def extract_number_from_text(text: str) -> Optional[float]:
    """
    Extract the final numeric answer from free-form text.
    Looks for patterns like "The answer is X" or numbers at the end.
    """
    # Try to find explicit answer patterns
    patterns = [
        r"(?:final answer|answer|result)(?:\s+is)?\s*:?\s*\$?\s*([-+]?[\d,]+\.?\d*)",
        r"####\s*([-+]?[\d,]+\.?\d*)",
        r"=\s*([-+]?[\d,]+\.?\d*)\s*$",
        r"\$\s*([-+]?[\d,]+\.?\d*)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num_str = match.group(1).replace(",", "")
                return float(num_str)
            except:
                continue

    # Fallback: find last number in text
    numbers = re.findall(r"[-+]?[\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except:
            pass

    return None


def parse_et_cot_response(
    response: str,
) -> Tuple[Optional[Dict], Optional[str], Optional[float]]:
    """
    Parse ET-CoT structured response.

    Returns:
        (vars_dict, trace_code, final_answer)
    """
    vars_dict = None
    trace_code = None
    final_answer = None

    try:
        # Extract VARS
        vars_match = re.search(r"VARS:\s*(\{[^}]+\})", response, re.IGNORECASE)
        if vars_match:
            vars_dict = json.loads(vars_match.group(1))

        # Extract TRACE
        trace_match = re.search(
            r"TRACE:\s*\n((?:.*\n)*?)(?:FINAL:|$)", response, re.IGNORECASE
        )
        if trace_match:
            trace_code = trace_match.group(1).strip()

        # Extract FINAL
        final_match = re.search(
            r"FINAL:\s*([-+]?[\d,]+\.?\d*)", response, re.IGNORECASE
        )
        if final_match:
            final_answer = float(final_match.group(1).replace(",", ""))

    except Exception as e:
        print(f"Warning: Failed to parse ET-CoT response: {e}")

    return vars_dict, trace_code, final_answer


def verify_et_cot_trace(
    vars_dict: Dict, trace_code: str, final_answer: float, question: str
) -> Tuple[bool, Optional[str]]:
    """
    Deterministically verify ET-CoT trace by executing it.

    Returns:
        (is_valid, error_message)
    """
    if not vars_dict or not trace_code or final_answer is None:
        return False, "Missing required components (VARS, TRACE, or FINAL)"

    try:
        # Create execution namespace with initial variables
        namespace = dict(vars_dict)

        # Execute trace line by line (only simple arithmetic)
        for line in trace_code.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse assignment: var = expression
            if "=" in line:
                var_name, expr = line.split("=", 1)
                var_name = var_name.strip()
                expr = expr.strip()

                # Evaluate expression in namespace (restricted to arithmetic)
                # This is safe because we only allow previously defined variables and numbers
                result = eval(expr, {"__builtins__": {}}, namespace)
                namespace[var_name] = result

        # Get last computed value
        if not namespace:
            return False, "No variables computed in trace"

        last_var_name = list(namespace.keys())[-1]
        last_computed_value = namespace[last_var_name]

        # Unit test 1: FINAL matches last computed value (within tolerance)
        if abs(last_computed_value - final_answer) > 1e-6:
            return (
                False,
                f"FINAL ({final_answer}) does not match last computed value ({last_computed_value})",
            )

        # Unit test 2: No NaN/Inf
        for var_name, value in namespace.items():
            if not isinstance(value, (int, float)) or not (-1e10 < value < 1e10):
                return False, f"Invalid value for {var_name}: {value}"

        # Unit test 3: Constraint checks (integer/nonnegative when question suggests counting)
        counting_keywords = ["how many", "total", "left", "count", "number of"]
        if any(keyword in question.lower() for keyword in counting_keywords):
            # Should be non-negative
            if final_answer < 0:
                return False, f"Negative answer ({final_answer}) for counting question"

            # Optionally check if should be integer (with small tolerance)
            if abs(final_answer - round(final_answer)) > 0.01:
                # Allow non-integers (e.g., averages), but warn
                pass

        return True, None

    except Exception as e:
        return False, f"Trace execution failed: {str(e)}"


def run_inference(cfg: Dict) -> None:
    """
    Run inference for a single method (CoT baseline or ET-CoT).
    """
    # Load dataset
    dataset = load_gsm8k(
        split=cfg["dataset"]["split"],
        max_samples=cfg["dataset"]["max_samples"],
        cache_dir=cfg["dataset"]["cache_dir"],
    )

    # Load model
    generator = LLMGenerator(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
        cache_dir=cfg["dataset"]["cache_dir"],
    )

    # Initialize WandB
    wandb_enabled = cfg["wandb"]["mode"] != "disabled"
    if wandb_enabled:
        wandb.init(
            entity=cfg["wandb"]["entity"],
            project=cfg["wandb"]["project"],
            id=cfg["run"]["run_id"],
            config=cfg,
            resume="allow",
        )
        print(f"WandB run URL: {wandb.run.get_url()}")

    # Run inference
    results = []
    correct = 0
    total = 0

    prompt_type = cfg["inference"]["prompt_type"]
    max_retries = cfg["inference"].get("max_retries", 0)

    print(f"\nRunning inference with {prompt_type}...")
    print(f"Processing {len(dataset)} examples...")

    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = example["ground_truth"]

        # Generate answer
        if prompt_type == "cot_baseline":
            prompt = COT_BASELINE_PROMPT.format(question=question)
            response = generator.generate(
                prompt,
                max_new_tokens=cfg["model"]["max_new_tokens"],
                temperature=cfg["model"]["temperature"],
                do_sample=cfg["model"]["do_sample"],
            )
            predicted_answer = extract_number_from_text(response)
            verification_passed = True
            error_message = None

        elif prompt_type == "et_cot":
            prompt = ET_COT_PROMPT.format(question=question)
            response = generator.generate(
                prompt,
                max_new_tokens=cfg["model"]["max_new_tokens"],
                temperature=cfg["model"]["temperature"],
                do_sample=cfg["model"]["do_sample"],
            )

            # Parse response
            vars_dict, trace_code, predicted_answer = parse_et_cot_response(response)

            # Verify trace
            verification_passed = False
            error_message = None

            if cfg["inference"].get("verification", {}).get("enabled", False):
                verification_passed, error_message = verify_et_cot_trace(
                    vars_dict, trace_code, predicted_answer, question
                )

                # Repair attempt if verification failed
                if not verification_passed and max_retries > 0:
                    print(f"  Example {i}: Verification failed, attempting repair...")
                    repair_prompt = ET_COT_REPAIR_PROMPT.format(
                        error_message=error_message,
                        question=question,
                    )
                    repair_response = generator.generate(
                        repair_prompt,
                        max_new_tokens=cfg["model"]["max_new_tokens"],
                        temperature=cfg["model"]["temperature"],
                        do_sample=cfg["model"]["do_sample"],
                    )

                    # Parse repaired response
                    vars_dict, trace_code, predicted_answer = parse_et_cot_response(
                        repair_response
                    )
                    verification_passed, error_message = verify_et_cot_trace(
                        vars_dict, trace_code, predicted_answer, question
                    )
            else:
                verification_passed = True

        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Check correctness
        is_correct = False
        if predicted_answer is not None and ground_truth is not None:
            is_correct = abs(predicted_answer - ground_truth) < 1e-6
            if is_correct:
                correct += 1
            total += 1

        # Store result
        result = {
            "id": example["id"],
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "verification_passed": verification_passed,
            "error_message": error_message,
        }
        results.append(result)

        # Log to WandB
        if wandb_enabled:
            wandb.log(
                {
                    "example_id": example["id"],
                    "is_correct": int(is_correct),
                    "verification_passed": int(verification_passed),
                }
            )

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            acc = correct / total if total > 0 else 0
            print(f"  Progress: {i + 1}/{len(dataset)}, Accuracy: {acc:.3f}")

    # Final metrics
    accuracy = correct / total if total > 0 else 0
    verification_rate = sum(1 for r in results if r["verification_passed"]) / len(
        results
    )

    print(f"\n{'=' * 80}")
    print(f"Final Results:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Verification Pass Rate: {verification_rate:.4f}")
    print(f"{'=' * 80}")

    # Save results
    results_dir = Path(cfg["results_dir"]) / cfg["run"]["run_id"]
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

    # Log final metrics to WandB
    if wandb_enabled:
        wandb.summary["accuracy"] = accuracy
        wandb.summary["correct"] = correct
        wandb.summary["total"] = total
        wandb.summary["verification_pass_rate"] = verification_rate
        wandb.finish()

    # Sanity validation (for sanity_check mode)
    if cfg["mode"] == "sanity_check":
        perform_sanity_validation(results, total)


def perform_sanity_validation(results: List[Dict], total: int) -> None:
    """
    Perform sanity validation checks and emit verdict.
    """
    # At least 5 samples processed
    if total < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples (only {total})")
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'outputs_valid': 0, 'outputs_unique': 0})}"
        )
        return

    # All outputs are valid (not all None)
    valid_outputs = sum(1 for r in results if r["predicted_answer"] is not None)
    if valid_outputs == 0:
        print(f"SANITY_VALIDATION: FAIL reason=no_valid_outputs")
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'outputs_valid': valid_outputs, 'outputs_unique': 0})}"
        )
        return

    # Outputs are not all identical
    unique_answers = len(
        set(r["predicted_answer"] for r in results if r["predicted_answer"] is not None)
    )
    if unique_answers <= 1 and total > 1:
        print(f"SANITY_VALIDATION: FAIL reason=all_identical_outputs")
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'outputs_valid': valid_outputs, 'outputs_unique': unique_answers})}"
        )
        return

    # All metrics are finite
    for r in results:
        if r["predicted_answer"] is not None:
            if not (-1e10 < r["predicted_answer"] < 1e10):
                print(f"SANITY_VALIDATION: FAIL reason=non_finite_metrics")
                print(
                    f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'outputs_valid': valid_outputs, 'outputs_unique': unique_answers})}"
                )
                return

    # All checks passed
    print(f"SANITY_VALIDATION: PASS")
    print(
        f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'outputs_valid': valid_outputs, 'outputs_unique': unique_answers})}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)


if __name__ == "__main__":
    main()
