"""
High-level API for tool-constrained LLM decoding.

The ``ToolConstrainedDecoder`` is the primary user-facing class.  It wraps a
HuggingFace model and tokenizer and exposes a ``generate()`` method that
accepts a tool name + JSON Schema and produces a constrained tool-call output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .char_fsm import nfa_to_dfa, _clear_registry
from .json_schema_to_fsm import build_tool_call_nfa, DEFAULT_TEMPLATE
from .token_index import TokenIndex
from .logits_processor import ToolConstraintLogitsProcessor

logger = logging.getLogger(__name__)


@dataclass
class DecoderResult:
    """Result of a constrained generation call.

    Attributes:
        text: The generated text (the tool call string).
        tool_call: Parsed dict with ``name`` and ``arguments`` keys.
        token_ids: Raw generated token IDs (excluding prompt).
        finish_reason: How generation ended: ``"eos"`` or ``"length"``.
    """
    text: str
    tool_call: Dict[str, Any]
    token_ids: List[int]
    finish_reason: str = "eos"


class ToolConstrainedDecoder:
    """High-level API for tool-constrained LLM decoding.

    Wraps a HuggingFace ``model`` and ``tokenizer``.  Call :meth:`generate`
    with a tool name and JSON Schema to produce output that is guaranteed
    (by construction) to be a valid tool call to that tool.

    Usage::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from constrained_decoding import ToolConstrainedDecoder

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        decoder = ToolConstrainedDecoder(model, tokenizer)

        result = decoder.generate(
            prompt="<|im_start|>user\\nWhat is the weather in London?<|im_end|>\\n"
                   "<|im_start|>assistant\\n",
            tool_name="get_weather",
            args_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
            max_new_tokens=128,
            temperature=0.7,
        )

        print(result.tool_call)
        # {"name": "get_weather", "arguments": {"city": "London", "unit": "celsius"}}
    """

    def __init__(
        self,
        model,
        tokenizer,
        template: str = DEFAULT_TEMPLATE,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: A HuggingFace ``PreTrainedModel``.
            tokenizer: The corresponding tokenizer.
            template: Tool-call format string with ``{name}`` and ``{arguments}``
                      placeholders.
            device: Torch device string.  Auto-detected if None.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.template = template

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Build the vocabulary index once (static per tokenizer)
        self.token_index = TokenIndex(tokenizer)

        # DFA + mask cache — keyed by (template, tool_name, args_schema).
        # Each entry holds its own isolated mask dicts to avoid state-ID
        # collisions between tools (state IDs start from 0 in every DFA).
        self._dfa_cache: Dict[str, dict] = {}

        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        tool_name: str,
        args_schema: Dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        **extra_kwargs,
    ) -> DecoderResult:
        """Generate a tool call constrained to *tool_name* with *args_schema*.

        Args:
            prompt: The input prompt string (chat template already applied).
            tool_name: Name of the tool to constrain to.
            args_schema: JSON Schema dict for the tool's ``arguments`` object.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-K sampling threshold.
            **extra_kwargs: Passed through to ``model.generate()``.

        Returns:
            DecoderResult with generated text and parsed tool call.
        """
        # --- 1. Build/reuse the DFA for this tool call ---
        cache_key = json.dumps(
            [self.template, tool_name, args_schema],
            sort_keys=True, ensure_ascii=False,
        )
        entry = self._dfa_cache.get(cache_key)
        if entry is not None:
            dfa_start = entry["start"]
            _all_dfa_states = entry["states"]
            masks = entry["masks"]
            progress = entry["progress"]
            logger.debug("Reusing cached DFA for %r (%d states)", tool_name, len(_all_dfa_states))
        else:
            _clear_registry()
            nfa_start = build_tool_call_nfa(
                tool_name=tool_name,
                args_schema=args_schema,
                template=self.template,
            )
            dfa_start, _all_dfa_states = nfa_to_dfa(nfa_start)
            masks: Dict[int, torch.Tensor] = {}
            progress: Dict[int, torch.Tensor] = {}
            self._dfa_cache[cache_key] = {
                "start": dfa_start,
                "states": _all_dfa_states,
                "masks": masks,
                "progress": progress,
            }
            logger.info("Built DFA with %d states for tool %r (cached)", len(_all_dfa_states), tool_name)

        # --- 2. Tokenize prompt ---
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_ids = prompt_ids.to(self.device)
        prompt_length = prompt_ids.shape[1]

        # --- 3. Determine special token IDs ---
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id set")

        pad_token_id = self.tokenizer.pad_token_id

        # --- 4. Create logits processor ---
        processor = ToolConstraintLogitsProcessor(
            dfa_start=dfa_start,
            token_index=self.token_index,
            prompt_length=prompt_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            all_dfa_states=_all_dfa_states,
            shared_masks=masks,
            shared_progress_masks=progress,
        )

        # --- 5. Generate ---
        do_sample = temperature > 0

        gen_kwargs: Dict[str, Any] = {
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
            'temperature': temperature if do_sample else 1.0,
            'top_p': top_p,
            'logits_processor': [processor],
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': eos_token_id,
        }
        if top_k > 0:
            gen_kwargs['top_k'] = top_k

        # Merge extra kwargs (user can override or add generation params)
        for k, v in extra_kwargs.items():
            if k not in gen_kwargs:
                gen_kwargs[k] = v

        with torch.no_grad():
            output_ids = self.model.generate(prompt_ids, **gen_kwargs)

        # --- 6. Extract generated tokens ---
        generated_ids = output_ids[0, prompt_length:].tolist()

        # Filter EOS/PAD
        filtered_ids = [
            tid for tid in generated_ids
            if tid != eos_token_id and tid != pad_token_id
        ]

        finish_reason = "length"
        if len(generated_ids) < max_new_tokens and (
            generated_ids and generated_ids[-1] == eos_token_id
        ):
            finish_reason = "eos"
        elif processor.has_accepted:
            finish_reason = "eos_constrained"

        generated_text = self.tokenizer.decode(filtered_ids)

        # --- 7. Parse tool call (template-aware) ---
        tool_call = self._parse_with_template(
            generated_text, tool_name, args_schema
        )

        logger.info(
            "Cache stats: %s, finish: %s",
            processor.cache_stats, finish_reason,
        )

        return DecoderResult(
            text=generated_text,
            tool_call=tool_call,
            token_ids=filtered_ids,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Template-aware parsing
    # ------------------------------------------------------------------

    def _parse_with_template(
        self, text: str, tool_name: str, args_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse generated text as a tool call, handling template wrappers."""
        if not text:
            return {"name": tool_name, "arguments": {}, "_raw": text, "_parse_error": True}

        # Try direct JSON parse first (works for default template)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Template-aware fallback: strip non-JSON prefix / suffix
        # from the template, then try parsing the remaining JSON.
        # e.g., Gemma's <tool_call|>  ...  <|tool_call>
        prefix, suffix = _extract_non_json_wrappers(self.template)
        body = text
        if prefix:
            body = body.replace(prefix, "", 1)
        if suffix:
            idx = body.rfind(suffix)
            if idx >= 0:
                body = body[:idx]
        body = body.strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

        return {
            "name": tool_name,
            "arguments": {},
            "_raw": text,
            "_parse_error": True,
        }


def _extract_non_json_wrappers(template: str) -> tuple:
    """Return (prefix, suffix) that fall outside the JSON in *template*.

    For ``{"name":"{name}","arguments":{arguments}}`` → ``("", "")``
    For ``<tool_call|>{"name":"{name}","arguments":{arguments}}<|tool_call>``
        → ``("<tool_call|>", "<|tool_call>")``
    """
    # Find the JSON part: it starts with '{' and ends with '}'
    json_start = template.find('{')
    json_end = template.rfind('}')
    if json_start == -1 or json_end == -1:
        return "", ""
    prefix = template[:json_start]
    suffix = template[json_end + 1:]
    return prefix, suffix

    # ------------------------------------------------------------------
    # Convenience: batch generate
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[str],
        tool_name: str,
        args_schema: Dict[str, Any],
        **kwargs,
    ) -> List[DecoderResult]:
        """Generate constrained outputs for a batch of prompts.

        Note: currently processes sequentially.  For true batching the
        LogitsProcessor would need to track per-batch-item DFA state.
        """
        return [
            self.generate(prompt=prompt, tool_name=tool_name, args_schema=args_schema, **kwargs)
            for prompt in prompts
        ]
