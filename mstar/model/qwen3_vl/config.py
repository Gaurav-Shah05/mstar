"""Qwen3-VL's text config, read off the HF ``config.json``.

The text stack is dense Qwen3 (``components/qwen3_lm.py``) with interleaved 3D
MRoPE over the whole head (``rope_scaling.mrope_section``); the vision tower's
config is the shared ``QwenVisionConfig``. Resource keys are Qwen3.5's, since
the two share the model class's prompt handling and the vision node.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mstar.model.components.qwen3_lm import Qwen3LMConfig


@dataclass
class Qwen3VLConfig(Qwen3LMConfig):
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    eos_token_id: int | list[int] | None = None
    # multimodal sentinels, at the root of `config.json`, not `text_config`
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    # Filled from the tokenizer at model construction; see `stop_token_ids`.
    extra_stop_token_ids: tuple[int, ...] = ()

    @classmethod
    def from_hf(cls, path: str | Path) -> "Qwen3VLConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text())
        text = raw["text_config"]
        rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
        if not rope.get("mrope_interleaved", True):
            raise ValueError(f"{path}: Qwen3-VL here supports interleaved MRoPE only, got {rope}")
        if text.get("attention_bias"):
            raise ValueError(f"{path}: Qwen3-VL here has no attention bias path")
        # root first so `text_config` wins shared names; a None in either is unset
        merged = {k: v for k, v in {**raw, **text}.items() if v is not None}
        if "rope_theta" in rope:
            merged["rope_theta"] = rope["rope_theta"]
        config = super().from_hf(merged)
        if "mrope_section" in rope:
            config.mrope_section = tuple(rope["mrope_section"])
        if sum(config.mrope_section) * 2 != config.head_dim:
            raise ValueError(
                f"{path}: mrope_section {config.mrope_section} does not cover head_dim {config.head_dim}"
            )
        return config

    @property
    def stop_token_ids(self) -> frozenset[int]:
        """``config.json``'s eos plus the tokenizer's (``<|im_end|>``, which ends a chat turn)."""
        ids = self.eos_token_id
        named = frozenset() if ids is None else frozenset([ids] if isinstance(ids, int) else ids)
        return named | frozenset(self.extra_stop_token_ids)
