"""
LLM Backend wrapping HuggingFace transformers models.

Provides free generation and constrained generation (via ToolConstrainedDecoder),
plus prompt building and tool-call parsing utilities.

The tool-call format is **derived from the template string** — no per-model
hardcoding of prefix/suffix.  The template contains the literal text that
surrounds the JSON body::

    "<tool_call|>{\"name\":\"{name}\",\"arguments\":{arguments}}<|tool_call>"
    ^^^^^^^^^^^^                                                      ^^^^^^^^^^^^
    prefix (auto-detected)                                            suffix (auto-detected)

A model registry maps model-id prefixes to known templates, with a sensible
default fallback.  Add new models by extending ``TEMPLATE_REGISTRY``.
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constrained_decoding import ToolConstrainedDecoder, DecoderResult

# ---------------------------------------------------------------------------
# Model → template registry
# ---------------------------------------------------------------------------

# Default template (used when model_id doesn't match any entry)
DEFAULT_TEMPLATE = '<tool_call|>{"name":"{name}","arguments":{arguments}}<|tool_call>'

TEMPLATE_REGISTRY: Dict[str, str] = {
    # -- Google --
    "google/gemma":       '<tool_call|>{"name":"{name}","arguments":{arguments}}<|tool_call>',
    # -- Qwen / Qwen2.5 / Qwen3 --
    "Qwen/":              '<tool_call>{"name":"{name}","arguments":{arguments}}</tool_call>',
    # -- Meta Llama 3/4 --
    "meta-llama/":        '{"name":"{name}","arguments":{arguments}}',
    # -- DeepSeek --
    "deepseek-ai/":       '<tool_call>{"name":"{name}","arguments":{arguments}}</tool_call>',
    # -- Mistral / Mixtral --
    "mistralai/":         '[TOOL_CALLS]{"name":"{name}","arguments":{arguments}}',
    # -- Microsoft Phi --
    "microsoft/Phi":      '<tool_call>{"name":"{name}","arguments":{arguments}}</tool_call>',
    # -- Yi (01.AI) --
    "01-ai/":             '<tool_call>{"name":"{name}","arguments":{arguments}}</tool_call>',
    # -- InternLM --
    "internlm/":          '<tool_call>{"name":"{name}","arguments":{arguments}}</tool_call>',
    # -- GLM (THUDM) --
    "THUDM/":             '<tool_call>{"name":"{name}","arguments":{arguments}}</tool_call>',
    # -- HuggingFace SmolLM2 --
    "HuggingFaceTB/":     '<tool_call>{"name":"{name}","arguments":{arguments}}</tool_call>',
    # -- Stability AI --
    "stabilityai/":       '{"name":"{name}","arguments":{arguments}}',
}
"""Fallback registry mapping model-id prefixes to tool-call templates.

Used only when auto-detection from the tokenizer's chat template fails.
Keys are matched via ``str.startswith``.  First match wins.
"""


# ---------------------------------------------------------------------------
# Template resolution: auto-detect > registry > default
# ---------------------------------------------------------------------------

def _detect_template_from_tokenizer(tokenizer) -> Optional[str]:
    """Try to auto-detect the tool-call template from the tokenizer's chat template.

    Scans the Jinja2 chat template string for known wrapper patterns.
    This is the primary method — no per-model hardcoding needed.

    Returns a compact template string like
    ``<tool_call>{"name":"{name}","arguments":{arguments}}</tool_call>``
    or None if detection fails.
    """
    ct = getattr(tokenizer, "chat_template", None)
    if ct is None:
        return None

    # Normalize: if it's a dict (some tokenizers), convert to string
    if isinstance(ct, dict):
        ct = str(ct)

    # Ordered list of (opening_tag, closing_tag) pairs to look for.
    # First match wins.
    KNOWN_WRAPPERS: List[Tuple[str, str]] = [
        ("<tool_call|>", "<|tool_call>"),        # Gemma 2/3
        ("<|tool_call>", "<tool_call|>"),        # Gemma 4
        ("<tool_call>", "</tool_call>"),          # Qwen, DeepSeek, Phi, Yi, InternLM, GLM
        ("[TOOL_CALLS]", ""),                     # Mistral
    ]

    for opening, closing in KNOWN_WRAPPERS:
        if opening in ct:
            if closing:
                return f'{opening}{{"name":"{{name}}","arguments":{{arguments}}}}{closing}'
            else:
                return f'{opening}{{"name":"{{name}}","arguments":{{arguments}}}}'

    return None


def _resolve_template(model_id: str, tokenizer=None) -> str:
    """Return the tool-call template for *model_id*.

    Resolution order (first success wins):
    1. Auto-detect from tokenizer's chat template (``_detect_template_from_tokenizer``)
    2. Match model_id prefix against ``TEMPLATE_REGISTRY``
    3. ``DEFAULT_TEMPLATE``
    """
    # 1. Auto-detect from tokenizer
    if tokenizer is not None:
        detected = _detect_template_from_tokenizer(tokenizer)
        if detected is not None:
            return detected

    # 2. Registry match by model_id prefix
    for prefix, template in TEMPLATE_REGISTRY.items():
        if model_id.startswith(prefix):
            return template

    # 3. Default
    return DEFAULT_TEMPLATE


def _derive_wrappers(template: str) -> Tuple[str, str]:
    """Derive (prefix, suffix) from a tool-call template.

    The template is expected to contain a single JSON object with
    ``{name}`` and ``{arguments}`` placeholders.  Everything before
    the opening ``{`` is the prefix; everything after the closing ``}``
    is the suffix.

    Example:
        ``<tool_call|>{"name":"{name}","arguments":{arguments}}<|tool_call>``
        → ``("<tool_call|>", "<|tool_call>")``
    """
    json_start = template.find("{")
    json_end = template.rfind("}")
    if json_start == -1 or json_end == -1:
        return "", ""
    return template[:json_start], template[json_end + 1:]


def _build_quantization_kwargs(
    quantization: Optional[str],
    device: str,
) -> Dict[str, Any]:
    """Build bitsandbytes quantization kwargs for ``from_pretrained``.

    Args:
        quantization: ``"4bit"``, ``"8bit"``, or ``None``.
        device: Torch device string.

    Returns:
        Dict of kwargs to pass to ``from_pretrained``.
    """
    if quantization is None:
        # No quantization — use the dtype from the model's config.json.
        # Explicit torch_dtype (e.g. torch.bfloat16) can conflict with
        # device_map="auto" / accelerate, leaving some tensors on meta device
        # (observed with Gemma 4's pad_embedding).
        return {"torch_dtype": "auto"}

    if device != "cuda":
        print("[LLMBackend] WARNING: bitsandbytes quantization requires CUDA. "
              "Falling back to float32.")
        return {"torch_dtype": torch.float32}

    try:
        import bitsandbytes as bnb
    except ImportError:
        print("[LLMBackend] WARNING: bitsandbytes not installed. "
              "Install with: pip install bitsandbytes. Falling back to float32.")
        return {"torch_dtype": torch.float32}

    if quantization == "4bit":
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }
    elif quantization == "8bit":
        return {"load_in_8bit": True}
    else:
        print(f"[LLMBackend] WARNING: unknown quantization {quantization!r}. "
              "Falling back to float32.")
        return {"torch_dtype": torch.float32}


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
        tool_call_template: Optional[str] = None,
        device: Optional[str] = None,
        quantize: bool = False,
        quantization_mode: str = "4bit",
    ):
        """Args:
            model_id: HuggingFace model ID or path.
            tool_call_template: Override the auto-detected template.
            device: Torch device (auto-detected if None).
            quantize: Enable bitsandbytes quantization (4bit by default).
            quantization_mode: ``"4bit"`` or ``"8bit"``.  Ignored if quantize=False.
        """
        self.model_id = model_id
        self.quantize = quantize

        # Load tokenizer first — needed for template auto-detection
        print(f"[LLMBackend] Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Resolve template: explicit arg > auto-detect from tokenizer > registry > default
        if tool_call_template is not None:
            self.tool_call_template = tool_call_template
            print(f"[LLMBackend] Tool-call template (explicit): {self.tool_call_template!r}")
        else:
            self.tool_call_template = _resolve_template(model_id, tokenizer=self.tokenizer)
            print(f"[LLMBackend] Tool-call template (auto-detected): {self.tool_call_template!r}")

        # Derive prefix / suffix from template (used for parsing & formatting)
        self.tool_call_prefix, self.tool_call_suffix = _derive_wrappers(
            self.tool_call_template
        )

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

        quant_str = f" (quant={quantization_mode})" if quantize else ""
        print(f"[LLMBackend] Device: {self.device}{quant_str}")

        # Build quantization kwargs
        quant_kwargs = _build_quantization_kwargs(
            quantization_mode if quantize else None, self.device,
        )

        # Use direct device assignment instead of "auto".
        #   - device_map="auto" goes through accelerate's meta-device init step,
        #     which can leave some tensors stranded on meta (observed with
        #     Gemma 4's pad_embedding).
        #   - All models in this project are ≤2B SLMs — single GPU is sufficient
        #     for both quantized (4bit) and unquantized loading.
        _device_map = "cuda:0" if self.device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=_device_map,
            **quant_kwargs,
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

        # Generation counter for periodic GPU cache cleanup.
        # HuggingFace's generate() caches KV tensors internally;
        # PyTorch's CUDA allocator retains freed memory in its pool.
        # After many generations (~100) this can accumulate 3-4 GB.
        # We release cached memory back to the OS every N generations.
        self._gen_count = 0
        self._gpu_cleanup_interval = 50  # generations between cache clears

    # ------------------------------------------------------------------
    # GPU memory management
    # ------------------------------------------------------------------

    def _maybe_cleanup_gpu(self) -> None:
        """Release cached GPU memory back to the OS every N generations.

        PyTorch's CUDA allocator caches freed memory rather than returning
        it to the OS immediately.  Over many ``model.generate()`` calls
        this cache can grow by several GB.  Periodic ``empty_cache()``
        releases it without affecting correct programs.
        """
        self._gen_count += 1
        if self._gen_count % self._gpu_cleanup_interval != 0:
            return
        if self.device != "cuda":
            return
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # Log only at INFO level to avoid polluting per-task output
        import logging
        logging.getLogger(__name__).debug(
            "GPU cache cleared after %d generations", self._gen_count
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
        """Apply the model's chat template to build a full prompt string.

        ``enable_thinking=False`` for tool-calling; without it
        the model enters thinking mode and outputs ``<think>...</think>``
        blocks instead of tool calls.
        """
        kwargs: Dict[str, Any] = {"enable_thinking": False}

        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
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

        result = self.tokenizer.decode(cleaned, skip_special_tokens=False)
        self._maybe_cleanup_gpu()
        return result

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
        result = self.constrained_decoder.generate(
            prompt=prompt,
            tool_name=tool_name,
            args_schema=args_schema,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        self._maybe_cleanup_gpu()
        return result

    # ------------------------------------------------------------------
    # Tool-call parsing
    # ------------------------------------------------------------------

    # Fallback wrapper pairs tried when the template-derived ones don't work.
    # Covers model variants that output reverse/malformed tags.
    _FALLBACK_WRAPPERS: List[Tuple[str, str]] = [
        ("<tool_call>", "</tool_call>"),
        ("<tool_call|>", "<|tool_call>"),
        ("<|tool_call>", "<tool_call|>"),
        ("[TOOL_CALLS]", ""),
    ]

    def parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract and parse a tool-call JSON object from *text*.

        Strips wrapper tags (from template + fallback variants), then
        finds and parses the JSON object.

        Returns None if no valid tool call is found.
        """
        if not text:
            return None

        # Build the list of (prefix, suffix) pairs to try.
        # Template-derived pair first, then fallbacks.
        pairs: List[Tuple[str, str]] = []
        if self.tool_call_prefix or self.tool_call_suffix:
            pairs.append((self.tool_call_prefix, self.tool_call_suffix))
        for pf, sf in self._FALLBACK_WRAPPERS:
            if (pf, sf) not in pairs:
                pairs.append((pf, sf))

        body = ""
        for prefix, suffix in pairs:
            body = text
            if prefix:
                idx = body.find(prefix)
                if idx >= 0:
                    body = body[idx + len(prefix):]
            if suffix:
                body_rstrip = body.rstrip()
                if body_rstrip.endswith(suffix):
                    body = body_rstrip[:-len(suffix)]
            body = body.strip()
            if body and body[0] == "{":
                break  # found a plausible JSON start — use this stripping
            # else: try next pair

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
            pass

        # ---- Gemma 4 custom format (Gemma models only) ----
        if self.model_id.startswith("google/gemma"):
            result = self._parse_gemma4(text)
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Gemma 4 custom parser
    # ------------------------------------------------------------------

    def _parse_gemma4(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse Gemma 4 custom tool-call format.

        Format::

            <|tool_call>call:tool_name{key1:val1, ...}<tool_call|><|tool_response>

        Where string values use ``<|\"|>`` as quote delimiters.
        Example::

            <|tool_call>call:bash_ls{pattern:<|\"|>.tif<|\"|>,show_detail:true}<tool_call|>
        """
        import re

        body = text
        body = re.sub(r'<\|tool_call>', '', body, count=1)
        body = re.sub(r'<tool_call\|>', '', body)
        body = re.sub(r'<\|tool_response>', '', body)
        body = body.strip()

        m = re.match(r'call:(\w+)\{(.*)\}$', body, re.DOTALL)
        if not m:
            return None

        tool_name = m.group(1)
        args_str = '{' + m.group(2) + '}'

        # Replace Gemma 4 special quote token with actual quote
        args_str = args_str.replace('<|"|>', '"')

        # Quote unquoted keys:  word: → "word":
        args_str = re.sub(r'(?<=[{,]) *(\w+) *:', r'"\1":', args_str)

        try:
            args = json.loads(args_str)
            return {"name": tool_name, "arguments": args}
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Tool-call message formatting
    # ------------------------------------------------------------------

    def format_tool_call_message(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Render a tool-call as an assistant message string.

        Uses the model-specific prefix/suffix derived from the template.
        """
        tool_call_json = json.dumps(
            {"name": tool_name, "arguments": arguments},
            ensure_ascii=False,
        )
        return f"{self.tool_call_prefix}{tool_call_json}{self.tool_call_suffix}"
