"""Qwen3-VL's text stack: dense Qwen3 (``components/qwen3_lm.py``) with
interleaved 3D MRoPE over the whole head and deepstack.

A text or decode row's three MRoPE grids are equal, which makes the rotation
plain 1D RoPE at that position; those rows take the shared attention's path
(the position resource, or its fused norm, RoPE and KV write). Only a
``prefill_vision`` step brings cos/sin, built from its 3D positions.

Deepstack (``Qwen3VLTextModel.forward`` in transformers): the vision tower's
features from a few of its blocks are added to the residual stream at the image
rows after the first decoder layers, one feature map per layer.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.decoder_layer import run_layers
from mstar.model.components.distributed import ParallelAttention
from mstar.model.components.mrope import compute_3d_cos_sin, compute_inv_freq
from mstar.model.components.qwen3_lm import Qwen3Attention, Qwen3DenseLM
from mstar.model.loader.iterators import iter_safetensors_shards
from mstar.model.qwen3_5.components.language_model import RopeCache
from mstar.model.qwen3_5.components.rope import apply_partial_mrope
from mstar.model.qwen3_5.config import ATTN, KV_CACHE, ROPE
from mstar.model.qwen3_vl.config import Qwen3VLConfig

_TEXT_PREFIX = "model.language_model."
# at the checkpoint's root, and only when untied (8B)
_LM_HEAD = "lm_head.weight"


class Qwen3VLAttention(Qwen3Attention):
    """Qwen3 attention that rotates by the step's 3D tables when it has them."""

    rope_tables: RopeCache | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.rope_tables is None or self.rope_tables.cos is None:
            return super().forward(hidden_states)
        # the fused kernel rotates by 1D positions only
        return ParallelAttention.forward(self, hidden_states)

    def _apply_rope(self, q, k, label):
        if self.rope_tables is None or self.rope_tables.cos is None:
            return super()._apply_rope(q, k, label)
        return apply_partial_mrope(q, k, self.rope_tables.cos, self.rope_tables.sin)


class Qwen3VLTextModel(Qwen3DenseLM):
    def __init__(self, config: Qwen3VLConfig, comm_group: CommGroup | None = None):
        super().__init__(
            config, attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE,
            comm_group=comm_group, attention_cls=Qwen3VLAttention,
        )
        self.rope_tables = RopeCache()
        for layer in self.layers:
            layer.self_attn.rope_tables = self.rope_tables
        self.register_buffer(
            "inv_freq", compute_inv_freq(config.head_dim, config.rope_theta), persistent=False,
        )

    def _apply(self, fn, recurse=True):
        """Keep the rotary frequencies fp32 through the engine's cast to the
        node's dtype, as transformers keeps them."""
        out = super()._apply(fn, recurse)
        if self.inv_freq.dtype != torch.float32 and self.inv_freq.device.type != "meta":
            self.inv_freq = compute_inv_freq(
                self.config.head_dim, self.config.rope_theta, device=self.inv_freq.device,
            )
        return out

    def build_cos_sin(
        self, position_ids_3d: torch.Tensor, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for ``[3, tokens]`` positions, computed in fp32 and rounded
        to ``dtype`` as transformers does."""
        return compute_3d_cos_sin(
            position_ids_3d, self.inv_freq, list(self.config.mrope_section), target_dtype=dtype,
        )

    def forward(
        self, input_embeds: torch.Tensor, *, label: str,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        deepstack: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(tokens, hidden)`` -> ``(tokens, hidden)`` after the final norm.

        ``deepstack`` is ``[layers, tokens, hidden]``, zero off the image rows;
        layer ``i``'s output gains ``deepstack[i]``.
        """
        self.rope_tables.set(*(cos_sin or (None, None)))
        self.layers[0].self_attn.attend.bind_step(label)
        if deepstack is None:
            return run_layers(self.layers, input_embeds, self.norm)

        # `run_layers` with the deepstack add after each of the first layers
        layers = self.layers
        h, residual = layers[0].input_layernorm.forward_residual(input_embeds.clone(), None)
        for i, layer in enumerate(layers):
            layer.self_attn.attend.set_layer_idx(i)
            h = layer.self_attn(h)
            next_norm = layers[i + 1].input_layernorm if i + 1 < len(layers) else self.norm
            h, residual = layer.forward_mlp_residual(h, residual, next_norm)
            if i < deepstack.shape[0]:
                # after the layer's output is complete, as transformers adds it
                residual = residual + deepstack[i]
                h = next_norm(residual)
        return h

    def load_checkpoint(self, path: str | Path, device: torch.device | str = "cpu") -> set[str]:
        """The text tower from a Qwen3-VL checkpoint (``model.language_model.*``
        and an untied ``lm_head``); raises on anything left unloaded."""
        keys = None if self.config.tie_word_embeddings else {_LM_HEAD}

        def weights():
            for name, tensor in iter_safetensors_shards(Path(path), device=device, prefix=_TEXT_PREFIX):
                yield name[len(_TEXT_PREFIX):], tensor
            if keys:
                yield from iter_safetensors_shards(Path(path), device=device, keys=keys)

        return self.load_weights(weights())


class Qwen3VLForCausalLM(nn.Module):
    """``model`` is the text stack, the layout Qwen3.5's LLM submodule reads
    (``model.embed_tokens``, ``model.build_cos_sin``)."""

    def __init__(self, config: Qwen3VLConfig, comm_group: CommGroup | None = None):
        super().__init__()
        self.config = config
        self.model = Qwen3VLTextModel(config, comm_group)
