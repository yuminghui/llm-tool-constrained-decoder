"""
LLM Backend wrapping HuggingFace transformers models.

Provides free generation and constrained generation (via ToolConstrainedDecoder),
plus prompt building and tool-call parsing utilities.
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constrained_decoding import ToolConstrainedDecoder, DecoderResult

# Gemma 4 tool-call format template
GEMMA_TOOL_CALL_TEMPLATE = '<tool_call|>{"name":"{name}","arguments":{arguments}}<|tool_call>'


class LLMBackend:
    """Wraps a HuggingFace causal LM with optional constrained decoding.

    Provides three generation modes:
      - ``generate_free()``: standard unconstrained generation
      - ``generate_constrained()``: force a specific tool name + args schema
      - ``build_prompt()``: apply the chat template with tool definitions
    """

    def __init__(
        self,
        model_id: str,
        tool_call_template: str = GEMMA_TOOL_CALL_TEMPLATE,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.tool_call_template = tool_call_template

        print(f"[LLMBackend] Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Auto-detect device
        if device is None:
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                try:
                    major, minor = torch.cuda.get_device_capability()
                    supported = torch.cuda.get_arch_list()
                    arch_str = f"sm_{major}{minor}"
                    use_cuda = any(arch_str in s for s in supported)
                except Exception:
                    use_cuda = False
            device = "cuda" if use_cuda else "cpu"
        self.device = device
        print(f"[LLMBackend] Device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Ensure pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[LLMBackend] Model loaded ({param_count:.1f}B params)")

        # Constrained decoder (shares model and tokenizer)
        self.constrained_decoder = ToolConstrainedDecoder(
            model=self.model,
            tokenizer=self.tokenizer,
            template=self.tool_call_template,
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply the model's chat template to build a full prompt string."""
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # ------------------------------------------------------------------
    # Free generation (no constraint)
    # ------------------------------------------------------------------

    def generate_free(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """Standard unconstrained generation."""
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        do_sample = temperature > 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            output_ids = self.model.generate(prompt_ids, **gen_kwargs)

        generated_ids = output_ids[0, prompt_ids.shape[1] :].tolist()

        # Strip EOS token(s) from the end
        eos = self.tokenizer.eos_token_id
        cleaned: List[int] = []
        for tid in generated_ids:
            if tid == eos:
                break
            cleaned.append(tid)

        return self.tokenizer.decode(cleaned, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # Constrained generation
    # ------------------------------------------------------------------

    def generate_constrained(
        self,
        prompt: str,
        tool_name: str,
        args_schema: Dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> DecoderResult:
        """Generate a tool call constrained to *tool_name* with *args_schema*."""
        return self.constrained_decoder.generate(
            prompt=prompt,
            tool_name=tool_name,
            args_schema=args_schema,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
        )

    # ------------------------------------------------------------------
    # Tool-call parsing
    # ------------------------------------------------------------------

    def parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract and parse a tool-call JSON object from *text*.

        Handles Gemma-style ``<tool_call|>...</...>`` wrappers.
        Returns None if no valid tool call is found.
        """
        if not text:
            return None

        # Strip Gemma wrapper prefixes
        body = text
        for prefix in ["<tool_call|>", "<|tool_call|>"]:
            idx = body.find(prefix)
            if idx >= 0:
                body = body[idx + len(prefix) :]
                break

        # Strip wrapper suffixes
        for suffix in ["<|tool_call>", "<|tool_call|>", "<|eot|>", "<eos>"]:
            if body.rstrip().endswith(suffix):
                body = body.rstrip()[: -len(suffix)]
                break

        body = body.strip()

        # Find the outermost JSON object
        start = body.find("{")
        if start == -1:
            return None

        depth = 0
        end = -1
        for i, ch in enumerate(body[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        json_str = body[start : end + 1] if end >= 0 else body[start:]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
