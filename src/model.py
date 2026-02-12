"""
Model loading and generation utilities.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict


class LLMGenerator:
    """
    Wrapper for loading and generating with HuggingFace causal LMs.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: str = ".cache",
    ):
        """
        Initialize the LLM generator.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            dtype: Data type (bfloat16, float16, float32)
            cache_dir: Cache directory for model weights
        """
        self.model_name = model_name
        self.cache_dir = cache_dir

        # Check if CUDA is available, fallback to CPU if not
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"

        self.device = device

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

        # Use float32 for CPU (bfloat16 not well supported)
        if device == "cpu" and self.dtype == torch.bfloat16:
            print("Note: Using float32 on CPU (bfloat16 not well supported)")
            self.dtype = torch.float32

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.model.eval()
        print(f"Model loaded successfully on {device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            # Prepare generation kwargs
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # Only add temperature if sampling is enabled
            # (temperature is ignored for greedy decoding)
            if do_sample and temperature > 0:
                gen_kwargs["temperature"] = temperature
            
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode (only the generated part)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return generated_text
