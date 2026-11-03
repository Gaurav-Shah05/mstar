"""Qwen3-VL's vision tower: the shared Qwen3-VL ViT (``components.qwen3_vit``) with
Qwen3-VL's patch mergers and deepstack, and the ``vision_config`` reader Qwen3.5 shares.

Deepstack: mergers after some blocks whose features the language model adds to its first
layers' hidden states at the image rows (``deepstack_visual_indexes``; empty for Qwen3.5).
Parameter names follow transformers' ``Qwen3VLVisionModel``, so its weights load by name.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.qwen3_vit import VisionBlock, VisionPatchEmbed, VisionRotaryEmbedding

# the tower's cacheless attention resource, by default
VISION_ATTN = "vision_attn"


@dataclass
class QwenVisionConfig:
    """The ViT tower's config, read off ``config.json``'s ``vision_config``."""

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int

    hidden_act: str = "gelu_pytorch_tanh"
    rope_theta: float = 10_000.0
    # Blocks after which a deepstack merger taps the hidden states (Qwen3-VL); empty on
    # every released Qwen3.5 size.
    deepstack_visual_indexes: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        """The learned position table is a square grid."""
        return int(self.num_position_embeddings**0.5)

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @classmethod
    def from_hf(cls, path: str | Path) -> "QwenVisionConfig":
        config = cls.from_hf_or_none(path)
        if config is None:
            raise ValueError(f"{path} has no `vision_config`")
        return config

    @classmethod
    def from_hf_or_none(cls, path: str | Path) -> "QwenVisionConfig | None":
        """None when the checkpoint is text-only."""
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text()).get("vision_config")
        if raw is None:
            return None
        merged = {**raw, **raw.get("rope_parameters", {})}
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in merged.items() if k in known}
        if "deepstack_visual_indexes" in values:
            values["deepstack_visual_indexes"] = tuple(values["deepstack_visual_indexes"])
        return cls(**values)


class VisionPatchMerger(nn.Module):
    """Folds each ``m x m`` block of patches into one LLM-width token. The norm runs
    per patch before the fold, or (``postshuffle_norm``, Qwen3-VL's deepstack mergers)
    over the folded token."""

    def __init__(self, config: QwenVisionConfig, postshuffle_norm: bool = False):
        super().__init__()
        self.merged_size = config.hidden_size * config.merge_unit
        self.postshuffle_norm = postshuffle_norm
        self.norm = nn.LayerNorm(self.merged_size if postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.postshuffle_norm:
            x = self.norm(x.view(-1, self.merged_size))
        else:
            x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class FrameSDPA:
    """Stands in for the ragged resource where none is declared (a model's SDPA mode, the
    reference's own kernel): SDPA within each frame of ``lengths`` (``vision_seq_lengths``),
    set before each tower call."""

    def __init__(self, lengths=()):
        self.lengths = list(lengths)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # [1, heads, frame, head_dim] per frame, the call transformers makes (and so its kernel)
        out = [
            F.scaled_dot_product_attention(*(t.transpose(0, 1)[None] for t in segment))[0].transpose(0, 1)
            for segment in zip(q.split(self.lengths), k.split(self.lengths), v.split(self.lengths), strict=True)
        ]
        return torch.cat(out)


class QwenVisionModel(nn.Module):
    """The rotary tables go into the blocks fp32, as transformers keeps them (Qwen3.5's
    and Qwen3-Omni's towers round them to the activations' dtype).

    ``forward`` is ``embed``, ``encode`` (the block loop, with the deepstack blocks'
    hidden states) and ``merge`` in a row; a node that captures the block loop on its
    own runs them separately."""

    def __init__(self, config: QwenVisionConfig, attn_key: str = VISION_ATTN):
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList(VisionBlock(config, attn_key) for _ in range(config.depth))
        self.merger = VisionPatchMerger(config)
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            VisionPatchMerger(config, postshuffle_norm=True) for _ in self.deepstack_visual_indexes
        )

    def _apply(self, fn, recurse=True):
        """Keep the rotary frequencies fp32 through a cast to bf16 (the engine casts
        every submodule to its node's dtype), as transformers does: rounded, the far
        patches of a large image rotate by visibly wrong angles."""
        out = super()._apply(fn, recurse)
        rotary = self.rotary_pos_emb
        if rotary.inv_freq.dtype != torch.float32 and rotary.inv_freq.device.type != "meta":
            rotary.inv_freq = VisionRotaryEmbedding(self.config).inv_freq.to(rotary.inv_freq.device)
        return out

    def embed(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch embed plus the resampled position table, and the fp32 rotary tables
        for the blocks: everything before the block loop, per patch."""
        n = pixel_values.shape[0]
        # one symbol for every leading dim, not one each
        torch._check(indices.shape[0] == n)
        torch._check(weights.shape[0] == n)
        torch._check(position_ids.shape[0] == n)

        hidden = self.patch_embed(pixel_values)
        # bilinear resample of the learned table, as a weighted gather
        pos = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden = hidden + pos.to(hidden.dtype)
        cos, sin = self.rotary_pos_emb(position_ids)
        return hidden, cos, sin

    def encode(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The block loop, ``[patches, hidden]`` in and out, with the unmerged hidden
        states after each deepstack block, in block order."""
        taps = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin)
            if index in self.deepstack_visual_indexes:
                taps.append(hidden)
        return hidden, taps

    def merge(
        self, hidden: torch.Tensor, taps: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The merged embeddings, and each tap through the merger its block is listed
        with in ``deepstack_visual_indexes`` (in block order, as transformers does)."""
        order = sorted(self.deepstack_visual_indexes)
        mergers = [self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)] for i in order]
        return self.merger(hidden), [merger(tap) for merger, tap in zip(mergers, taps, strict=True)]

    def forward(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``[total_patches, patch_numel]`` in; ``[merged_tokens, out_hidden]`` and one such
        tensor per deepstack merger out (``Qwen3VLVisionModel.forward``). Grid-derived inputs
        come from the caller (``qwen3_vit``'s grid helpers), so the tower has no host reads."""
        return self.merge(*self.encode(*self.embed(pixel_values, indices, weights, position_ids)))
