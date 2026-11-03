"""Qwen3-VL (dense, 2B / 4B / 8B Instruct).

Qwen3.5's multimodal front end is Qwen3-VL's: the same chat template sentinels,
image and video processors, prompt layout and MRoPE positions. So this is
Qwen3.5's model class with Qwen3-VL's text stack, a dense Qwen3 (no recurrent
state), and the vision tower's deepstack features carried to the LLM.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.components.qwen_vision import QwenVisionConfig, QwenVisionModel
from mstar.model.qwen3_5.config import ATTN, KV_CACHE, ROPE, SAMPLER
from mstar.model.qwen3_5.qwen3_5_model import _MIXED_SPLIT, Qwen3_5DenseModel
from mstar.model.qwen3_vl.config import Qwen3VLConfig
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

# what the API and conductor processes need; workers fetch the full snapshot
_METADATA_FILES = [
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "tokenizer.json",
    "merges.txt",
    # Qwen3-VL ships its chat template on its own
    "chat_template.json",
]


def _resolve_model_metadata(repo_id: str, cache_dir: str | None) -> str:
    if Path(repo_id).is_dir():
        return repo_id
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir, allow_patterns=_METADATA_FILES)


class Qwen3VLModel(Qwen3_5DenseModel):
    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        prefill_token_buckets: list[int] | None = None,
        **kwargs: Any,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.prefill_token_buckets = prefill_token_buckets

        self.local_dir = _resolve_model_metadata(model_path_hf, cache_dir)
        self.config = Qwen3VLConfig.from_hf(self.local_dir)
        self.vision_config = QwenVisionConfig.from_hf(self.local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, cache_dir=cache_dir)
        if self.tokenizer.chat_template is None:
            raise ValueError(f"{self.local_dir} has no chat template; Qwen3-VL Instruct ships one")
        # config.json names <|endoftext|>; a chat turn ends on <|im_end|>
        if self.tokenizer.eos_token_id is not None:
            self.config.extra_stop_token_ids = (self.tokenizer.eos_token_id,)

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._image_processor = None
        self._video_processor = None

    def get_node_resources(self) -> list[NodeResourceSpec]:
        config = self.config
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=config.num_hidden_layers,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    max_seq_len=config.max_position_embeddings,
                    num_qo_heads=config.num_attention_heads,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN, nodes={"LLM"},
                config=AttentionConfig(kv_cache=KV_CACHE, splits=(_MIXED_SPLIT,), decode_fmha=True),
            ),
            # text rows rotate here, by 1D positions (see `Qwen3VLLLMSubmodule._rope_in_graph`)
            PositionSpec(
                resource_key=ROPE, nodes={"LLM"},
                config=PositionConfig(kv_cache=KV_CACHE, rope_theta=config.rope_theta),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"LLM"},
                vocab_size=config.vocab_size,
                enable_repetion_penalty=True,
            ),
            *self._vision_resources(),
        ]

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        """Qwen3.5's walks, with the vision node's deepstack features beside
        its embeddings into the LLM."""
        walks = super().get_graph_walk_graphs()
        if not self.vision_config.deepstack_visual_indexes:
            return walks
        walks["prefill_vision"] = Sequential([
            GraphNode(
                name="vision_encoder",
                input_names=["pixel_values"],
                outputs=[
                    GraphEdge(next_node="LLM", name="vision_embeds"),
                    GraphEdge(next_node="LLM", name="deepstack"),
                ],
            ),
            GraphNode(
                name="LLM",
                input_names=["vision_embeds", "deepstack", "text_inputs"],
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", persist=True, output_modality="text"),
                ],
            ),
        ])
        return walks

    def _create_submodule(
        self, node_name: str, device: str,
        autocast_dtype: torch.dtype | None = None,
        tp_group=None,
    ) -> NodeSubmodule | None:
        from mstar.model.qwen3_5.weight_loader import load_qwen3_5_vision_weights
        from mstar.model.qwen3_vl.components.language_model import Qwen3VLForCausalLM
        from mstar.model.qwen3_vl.submodules import (
            Qwen3VLLLMSubmodule,
            Qwen3VLVisionEncoderSubmodule,
        )

        if node_name not in ("LLM", "vision_encoder"):
            return None
        dtype = autocast_dtype or torch.bfloat16
        weights_dir = self._resolve_weights()

        if node_name == "vision_encoder":
            tower = self._build(lambda: QwenVisionModel(self.vision_config), dtype, device)
            load_qwen3_5_vision_weights(tower, weights_dir, device=device)
            tower.requires_grad_(False).eval()
            logger.info("Loaded Qwen3-VL vision tower onto %s", device)
            return Qwen3VLVisionEncoderSubmodule(tower, self.vision_config, self.pixel_normalizer())

        model = self._build(lambda: Qwen3VLForCausalLM(self.config, tp_group), dtype, device)
        model.model.load_checkpoint(weights_dir, device=device)
        model.requires_grad_(False).eval()
        logger.info("Loaded Qwen3-VL LLM submodule onto %s", device)
        return Qwen3VLLLMSubmodule(
            model, self.config, self.vision_config,
            prefill_token_buckets=self.prefill_token_buckets,
        )
