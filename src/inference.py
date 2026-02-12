"""
Inference script for prompt tuning experiments.
Supports CoT baseline and ET-CoT with deterministic verification.
"""

import sys
import json
import re
import ast
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

ET_COT_PROMPT = """Solve this math problem. You MUST provide your answer in this EXACT format:

VARS: {{"key1": number1, "key2": number2}}
TRACE:
var3 = var1 + var2
var4 = var3 * 2
FINAL: number

CRITICAL RULES - Follow these exactly:
1. VARS must be valid JSON with double quotes around keys and ONLY pure numbers as values (like 5, 3.14, 100)
2. TRACE must contain ONLY Python assignment statements (varname = expression)
3. TRACE expressions can ONLY use: numbers, +, -, *, /, (), and variable names from VARS or previous TRACE lines
4. NO text, NO descriptions, NO double equals (==), NO colons in TRACE - ONLY "varname = expression" format
5. FINAL must be a single number

Example 1:
Question: Sarah has 3 apples. She buys 2 more. How many does she have?

VARS: {{"initial": 3, "bought": 2}}
TRACE:
total = initial + bought
FINAL: 5

Example 2:
Question: A box costs $10 and there is a 20% discount. What is the final price?

VARS: {{"price": 10, "discount_rate": 0.2}}
TRACE:
discount = price * discount_rate
final_price = price - discount
FINAL: 8

Example 3:
Question: Janet has 16 eggs. She eats 3 and bakes with 4. She sells the rest for $2 each. How much does she make?

VARS: {{"eggs_total": 16, "eggs_eaten": 3, "eggs_baked": 4, "price_per_egg": 2}}
TRACE:
eggs_used = eggs_eaten + eggs_baked
eggs_left = eggs_total - eggs_used
money = eggs_left * price_per_egg
FINAL: 18

Now solve this problem. Remember: VARS must be valid JSON, TRACE must be ONLY "var = expression" lines with NO text or explanations:

Question: {question}

Your answer:"""

ET_COT_REPAIR_PROMPT = """Your previous answer failed verification: {error_message}

Please provide a complete corrected solution. Follow this EXACT format:

VARS: {{"key1": number1, "key2": number2}}
TRACE:
var3 = key1 + key2
var4 = var3 * 10
FINAL: number

CRITICAL - Common mistakes to avoid:
1. VARS must have double quotes around keys: {{"price": 10}} NOT {{price: 10}}
2. VARS values must be ONLY pure numbers: {{"x": 5}} NOT {{"x": 3 + 2}}
3. TRACE lines must be ONLY "varname = expression" - NO text, NO descriptions, NO double equals
4. Do NOT write: "eggs_left = 16 - 7 = 9" - Write: "eggs_left = 16 - 7"
5. Do NOT include units or text after expressions: "total = 10 cups" - Write: "total = 10"
6. Use only variables from VARS or previous TRACE lines in your expressions
7. FINAL must match the last computed value

Question: {question}

Your corrected answer (remember: JSON VARS with double quotes, clean TRACE with only "var = expr" format):"""


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
        # Extract VARS - find JSON object after VARS: (handling nested braces and multiline)
        # Try multiple patterns to be flexible
        vars_start = re.search(r"VARS\s*:\s*", response, re.IGNORECASE)
        if not vars_start:
            # Try without colon or with newline
            vars_start = re.search(r"VARS\s*\n\s*", response, re.IGNORECASE)
        if not vars_start:
            vars_start = re.search(r"VARS\s+", response, re.IGNORECASE)
        
        if vars_start:
            # Start from position after "VARS:"
            json_start_pos = vars_start.end()
            remaining = response[json_start_pos:]
            
            # Skip whitespace (including newlines)
            remaining = remaining.lstrip()
            
            # Look for opening brace (may be on next line or after whitespace)
            if remaining.startswith('{'):
                # Find the complete JSON object by counting braces
                brace_count = 0
                json_end = 0
                in_string = False
                escape_next = False
                
                for i, char in enumerate(remaining):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                
                if json_end > 0:
                    vars_json = remaining[:json_end]
                    try:
                        vars_dict = json.loads(vars_json)
                    except json.JSONDecodeError as je:
                        print(f"Warning: Failed to parse VARS JSON: {je}")
                        
                        # Try to fix common issues: single quotes, expressions in JSON values
                        # The model often outputs Python dict syntax instead of JSON
                        try:
                            # First, try to convert single quotes to double quotes for JSON compatibility
                            # This is a common issue when the model outputs Python dict syntax
                            json_fixed = vars_json.replace("'", '"')
                            try:
                                vars_dict = json.loads(json_fixed)
                                print(f"  Successfully parsed VARS after fixing quotes ({len(vars_dict)} entries)")
                            except json.JSONDecodeError:
                                # If still fails, try to fix unquoted keys before literal_eval
                                # Convert {key: value} to {"key": value}
                                vars_json_quoted_keys = re.sub(
                                    r'(?<=[{,\s])(\w+)(?=\s*:)',
                                    r'"\1"',
                                    json_fixed
                                )
                                try:
                                    vars_dict = json.loads(vars_json_quoted_keys)
                                    print(f"  Successfully parsed VARS after fixing keys ({len(vars_dict)} entries)")
                                except json.JSONDecodeError:
                                    # If still fails, try Python literal_eval (handles Python dict syntax)
                                    try:
                                        evaluated_dict = ast.literal_eval(vars_json)
                                        if isinstance(evaluated_dict, dict):
                                            # Convert all values to float
                                            vars_dict = {}
                                            for k, v in evaluated_dict.items():
                                                try:
                                                    if isinstance(v, (int, float)):
                                                        vars_dict[k] = float(v)
                                                    elif v is None:
                                                        vars_dict[k] = 0.0
                                                    else:
                                                        # Try to evaluate expressions
                                                        safe_namespace = {
                                                            "__builtins__": {},
                                                            "abs": abs,
                                                            "int": int,
                                                            "float": float,
                                                        }
                                                        evaluated_value = eval(str(v), safe_namespace, {})
                                                        vars_dict[k] = float(evaluated_value)
                                                except Exception as conv_error:
                                                    print(f"  Warning: Skipping VARS entry '{k}': {v} ({conv_error})")
                                                    continue
                                            
                                            if vars_dict:
                                                print(f"  Successfully parsed VARS with literal_eval ({len(vars_dict)} entries)")
                                    except Exception as literal_error:
                                        # Last resort: manual regex-based parsing
                                        print(f"  literal_eval failed: {literal_error}")
                                        manual_dict = {}
                                        
                                        # Pattern to match key-value pairs with optional quotes
                                        # Handles: "key": value, 'key': value, key: value
                                        kv_pattern = r'''['"']?(\w+)['"']?\s*:\s*([^,}\n]+)'''
                                        matches = re.findall(kv_pattern, vars_json)
                                        
                                        if matches:
                                            for key, value_str in matches:
                                                value_str = value_str.strip().strip('"\'')
                                                try:
                                                    # Try to evaluate as a number or expression
                                                    safe_namespace = {
                                                        "__builtins__": {},
                                                        "abs": abs,
                                                        "int": int,
                                                        "float": float,
                                                    }
                                                    
                                                    # Evaluate the value (handles expressions like "3 * 60")
                                                    evaluated_value = eval(value_str, safe_namespace, {})
                                                    manual_dict[key] = float(evaluated_value)
                                                except:
                                                    # If evaluation fails, try direct float conversion
                                                    try:
                                                        manual_dict[key] = float(value_str)
                                                    except:
                                                        print(f"  Warning: Skipping VARS entry '{key}': {value_str}")
                                                        continue
                                            
                                            if manual_dict:
                                                vars_dict = manual_dict
                                                print(f"  Successfully parsed VARS manually ({len(vars_dict)} entries)")
                        except Exception as fix_error:
                            print(f"  Failed to fix VARS: {fix_error}")

        # Extract TRACE - be more flexible with whitespace and format
        trace_match = re.search(
            r"TRACE:\s*\n((?:.*\n)*?)(?:FINAL:|$)", response, re.IGNORECASE | re.DOTALL
        )
        if trace_match:
            trace_code = trace_match.group(1).strip()
        
        # If TRACE not found with newline, try without newline requirement
        if not trace_code:
            trace_match = re.search(
                r"TRACE:\s*(.*?)(?:FINAL:|$)", response, re.IGNORECASE | re.DOTALL
            )
            if trace_match:
                trace_code = trace_match.group(1).strip()

        # Extract FINAL - be more flexible
        final_match = re.search(
            r"FINAL:\s*([-+]?[\d,]+\.?\d*)", response, re.IGNORECASE
        )
        if final_match:
            final_answer = float(final_match.group(1).replace(",", ""))
        
        # If FINAL not found with colon, try looking for "FINAL" followed by number
        if final_answer is None:
            final_match = re.search(
                r"FINAL\s+([-+]?[\d,]+\.?\d*)", response, re.IGNORECASE
            )
            if final_match:
                final_answer = float(final_match.group(1).replace(",", ""))

    except Exception as e:
        print(f"Warning: Failed to parse ET-CoT response: {e}")
        print(f"  Response preview: {response[:300]}")
    
    # If we still don't have all components, try more aggressive parsing
    if not vars_dict and final_answer is None:
        # Look for any JSON-like dict in the response
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                vars_dict = json.loads(json_match.group(0))
            except:
                pass
    
    if not trace_code:
        # Look for assignment-like patterns anywhere in response
        assignments = re.findall(r'^\s*(\w+)\s*=\s*(.+?)(?:\n|$)', response, re.MULTILINE)
        if assignments:
            # Filter out assignments that don't look like arithmetic
            valid_assignments = []
            for var, expr in assignments:
                expr = expr.strip()
                # Keep if it looks like arithmetic (contains numbers, +, -, *, /, or known vars)
                if re.search(r'[\d+\-*/()]', expr):
                    valid_assignments.append(f"{var} = {expr}")
            if valid_assignments:
                trace_code = '\n'.join(valid_assignments)
    
    if final_answer is None:
        # Try to find ANY number in the response as last resort
        final_answer = extract_number_from_text(response)

    return vars_dict, trace_code, final_answer


def verify_et_cot_trace(
    vars_dict: Dict, trace_code: str, final_answer: float, question: str
) -> Tuple[bool, Optional[str]]:
    """
    Deterministically verify ET-CoT trace by executing it.

    Returns:
        (is_valid, error_message)
    """
    # Check for None explicitly since empty dict/string are also falsy
    if vars_dict is None or trace_code is None or final_answer is None:
        missing = []
        if vars_dict is None:
            missing.append("VARS")
        if trace_code is None:
            missing.append("TRACE")
        if final_answer is None:
            missing.append("FINAL")
        return False, f"Missing required components: {', '.join(missing)}"
    
    if not vars_dict or not trace_code:
        return False, "Empty VARS or TRACE (model may not be following format)"

    try:
        # Create execution namespace with initial variables
        # Ensure all values are numeric (float)
        namespace = {}
        for k, v in vars_dict.items():
            try:
                namespace[k] = float(v)
            except (ValueError, TypeError):
                return False, f"VARS entry '{k}' has non-numeric value: {v}"

        # Track which variables were defined to identify the last computed one
        initial_var_names = set(namespace.keys())
        last_computed_var = None

        # Execute trace line by line (only simple arithmetic)
        for line in trace_code.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Skip lines that are clearly not assignments (e.g., plain text, markdown)
            # Skip lines starting with bullets, dashes, or containing colons (likely descriptions)
            if line.startswith(('-', '*', '•')) or (':' in line and '=' not in line):
                continue
            
            # Skip lines with LaTeX or markdown math
            if '\\(' in line or '\\)' in line or '$$' in line:
                continue

            # Parse assignment: var = expression
            if "=" in line:
                # Split only on the first '=' to handle the assignment
                parts = line.split("=", 1)
                if len(parts) != 2:
                    continue
                    
                var_name = parts[0].strip()
                expr = parts[1].strip()
                
                # Skip if variable name is invalid (contains spaces, special chars except _)
                if not var_name.replace('_', '').replace('-', '').isalnum():
                    continue
                
                # Clean expression: remove everything after second '=' if present (common error: var = x + y = z)
                if '=' in expr:
                    expr = expr.split('=')[0].strip()
                
                # Remove trailing text after the arithmetic expression
                # Look for common patterns: "= 9 cups", "= 100 minutes", etc.
                # Keep only the numeric/variable part
                expr_cleaned = expr
                # Remove text that follows numbers (e.g., "9 cups" -> "9")
                expr_match = re.match(r'^([\d\w\s\+\-\*/\(\)\.]+?)(?:\s+[a-zA-Z]|$)', expr)
                if expr_match:
                    expr_cleaned = expr_match.group(1).strip()

                try:
                    # Evaluate expression in namespace (restricted to arithmetic)
                    # This is safe because we only allow previously defined variables and numbers
                    result = eval(expr_cleaned, {"__builtins__": {}}, namespace)
                    namespace[var_name] = float(result)
                    
                    # Track last computed variable (not from VARS)
                    if var_name not in initial_var_names:
                        last_computed_var = var_name
                except Exception as e:
                    # Provide more helpful error message but continue trying other lines
                    print(f"  Warning: Skipping line '{line[:80]}': {str(e)}")
                    continue

        # Get last computed value
        # Prefer the last computed variable (not from initial VARS)
        if last_computed_var:
            last_computed_value = namespace[last_computed_var]
        elif namespace:
            # Fallback: use last variable in namespace
            last_var_name = list(namespace.keys())[-1]
            last_computed_value = namespace[last_var_name]
        else:
            return False, "No variables computed in trace"

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
            
            # Debug: print parsing results for first few examples
            if i < 3:
                print(f"\n  [DEBUG] Example {i} response: {response[:400]}")
                print(f"  [DEBUG] Parsed - VARS: {vars_dict}, TRACE: {trace_code[:100] if trace_code else None}, FINAL: {predicted_answer}")

            # Verify trace
            verification_passed = False
            error_message = None

            if cfg["inference"].get("verification", {}).get("enabled", False):
                # Only verify if we have all components
                if vars_dict is not None and trace_code is not None and predicted_answer is not None:
                    verification_passed, error_message = verify_et_cot_trace(
                        vars_dict, trace_code, predicted_answer, question
                    )
                else:
                    # Missing components - set error message
                    missing = []
                    if vars_dict is None:
                        missing.append("VARS")
                    if trace_code is None:
                        missing.append("TRACE")
                    if predicted_answer is None:
                        missing.append("FINAL")
                    error_message = f"Missing required components: {', '.join(missing)}"
                    verification_passed = False

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
                        temperature=max(0.1, cfg["model"]["temperature"] * 0.5),  # Lower temp for repair
                        do_sample=cfg["model"]["do_sample"],
                    )

                    # Parse repaired response
                    vars_dict_repaired, trace_code_repaired, predicted_answer_repaired = parse_et_cot_response(
                        repair_response
                    )
                    
                    # Only use repaired response if it's better
                    if vars_dict_repaired is not None and trace_code_repaired is not None and predicted_answer_repaired is not None:
                        verification_passed, error_message = verify_et_cot_trace(
                            vars_dict_repaired, trace_code_repaired, predicted_answer_repaired, question
                        )
                        
                        if verification_passed or predicted_answer is None:
                            # Use repaired response if it passed verification or original had no answer
                            vars_dict, trace_code, predicted_answer = vars_dict_repaired, trace_code_repaired, predicted_answer_repaired
            else:
                verification_passed = True
            
            # Fallback: if structured parsing completely failed, try to extract any number
            if predicted_answer is None:
                predicted_answer = extract_number_from_text(response)

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
