"""Qwen3-VL's node submodules: Qwen3.5's LLM and vision nodes, which already
handle the prompt layout, MRoPE positions, batching and capture, specialised
for a dense Qwen3 text stack and deepstack.

The vision node also emits ``deepstack``, the merged features of the tower's
deepstack blocks; the LLM spreads them over its prompt rows (zero off the
image tokens) and its text stack adds them after its first layers.
"""
from __future__ import annotations

from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import SubmoduleStep
from mstar.model.qwen3_5.config import ATTN, GDN_STATE, LINEAR_ATTN, SAMPLER
from mstar.model.qwen3_5.qwen3_5_model import STEP_GRIDS_KEY, TEXT_PART, VIDEO_PART
from mstar.model.qwen3_5.submodules import (
    VISION_BLOCK_LOOP,
    LLMSubmodule,
    VisionEncoderSubmodule,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    BatchedModelOutput,
    ModelInputsFromEngine,
)

# the vision node's deepstack output, and the LLM's step input built from it
DEEPSTACK = "deepstack"


class Qwen3VLLLMSubmodule(LLMSubmodule):
    """Qwen3.5's LLM node over a dense Qwen3 stack: no recurrent state, and
    text rows rotate in the attention layers (see `_rope_in_graph`)."""

    def __init__(self, model, config, vision_config=None, prefill_token_buckets=None):
        super().__init__(model, config, vision_config, prefill_token_buckets)
        self.num_deepstack = len(vision_config.deepstack_visual_indexes) if vision_config else 0

    @staticmethod
    def _rope_in_graph(engine_inputs: ModelInputsFromEngine, graph_walk: str) -> bool:
        """Whether the step leaves rotation to the attention layers' 1D path,
        so `preprocess` builds no cos/sin: every walk but ``prefill_vision``,
        whose rows' three grids differ. A text row's grids are equal, which
        makes interleaved MRoPE 1D RoPE at the same position."""
        return graph_walk != "prefill_vision"

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        configs = super().get_accelerator_graph_configs(device, tp_world_size)
        if not self.num_deepstack:
            return configs
        for config in configs:
            if isinstance(config, PackedAcceleratorGraphConfig) and config.capture_graph_walk == "prefill_vision":
                config.make_node_input = self._with_deepstack(config.make_node_input, device)
                config.input_seq_dims = {**(config.input_seq_dims or {}), DEEPSTACK: 0}
        return configs

    def _with_deepstack(self, make_node_input, device):
        def make(n: int) -> ARNodeInputs:
            inp = make_node_input(n)
            inp.tensor_inputs[DEEPSTACK] = torch.zeros(
                (n, self.num_deepstack, self.config.hidden_size),
                device=device, dtype=self.model.model.embed_tokens.weight.dtype,
            )
            return inp
        return make

    def _vision_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
    ) -> ARNodeInputs:
        row = super()._vision_inputs(fwd_info, inputs)
        if self.num_deepstack:
            row.tensor_inputs[DEEPSTACK] = self._deepstack_rows(fwd_info, inputs, row.input_seq_len)
        return row

    def _deepstack_rows(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, num_tokens: int,
    ) -> torch.Tensor:
        """``[tokens, layers, hidden]`` over the prompt row `_vision_inputs`
        lays out: each image's merged features at its tokens (between its two
        sentinels), zero elsewhere."""
        features = inputs[DEEPSTACK]
        if len(features) != self.num_deepstack:
            raise ValueError(f"{len(features)} deepstack feature maps for {self.num_deepstack} deepstack layers")
        # in the stack's dtype, whatever the vision node runs in
        dtype = self.model.model.embed_tokens.weight.dtype
        stacked = torch.stack([t.to(self.get_device(), dtype) for t in features], dim=1)
        out = stacked.new_zeros((num_tokens, *stacked.shape[1:]))
        merge = self.vision_config.spatial_merge_size
        texts = inputs.get("text_inputs", [])
        grids = fwd_info.step_metadata[STEP_GRIDS_KEY]
        # Mirrors Qwen3.5's `_vision_inputs` walk: `row` into the prompt,
        # `cursor` into the features, a video's frame groups one at a time.
        # The totals check below catches the two drifting apart.
        row = cursor = text_i = grid_i = group = 0
        for kind in fwd_info.step_metadata.get("prefill_order", ()):
            if kind == TEXT_PART:
                row += texts[text_i].shape[0]
                text_i += 1
                continue
            t, h, w = grids[grid_i]
            if kind == VIDEO_PART:
                group += 1
                if group == t:
                    grid_i, group = grid_i + 1, 0
                t = 1
            else:
                grid_i += 1
            span = t * (h // merge) * (w // merge)
            out[row + 1:row + 1 + span] = stacked[cursor:cursor + span]
            row += span + 2
            cursor += span
        if row != num_tokens or cursor != stacked.shape[0]:
            raise ValueError(
                f"deepstack layout covers {row} of {num_tokens} prompt rows and "
                f"{cursor} of {stacked.shape[0]} image tokens"
            )
        return out

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        if graph_walk == "prefill_vision" and self.num_deepstack:
            parts = [inp.tensor_inputs[DEEPSTACK] for inp in inputs]
            out[DEEPSTACK] = parts[0] if len(parts) == 1 else torch.cat(parts)
        return out

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs], **kwargs,
    ) -> SubmoduleStep:
        step = super().declare_step(graph_walk, request_ids, inputs, **kwargs)
        # Qwen3.5's recurrent layers; this stack has none
        for key in (GDN_STATE, LINEAR_ATTN):
            step.steps.pop(key, None)
        return step

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        deepstack: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn: AttentionManager = engine_inputs.resources[ATTN]
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        text = self.model.model
        if input_ids is None and input_embeds is None:
            input_ids = sampler.loopback_tokens(engine_inputs.request_ids)
        embeds = text.embed(input_ids) if input_embeds is None else input_embeds
        cos_sin = None if cos_3d is None else (cos_3d, sin_3d)
        hidden = text(
            embeds, label="main", cos_sin=cos_sin,
            deepstack=None if deepstack is None else deepstack.transpose(0, 1),
        )
        if graph_walk != "decode":
            # only the last token of each prefill row feeds the head
            hidden = attn.select_last_hidden(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=text.logits(hidden))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        deepstack: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": self._forward(
            graph_walk, engine_inputs, cos_3d, sin_3d, input_ids, input_embeds, deepstack,
        )}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        deepstack: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(
            graph_walk, engine_inputs, cos_3d, sin_3d, input_ids, input_embeds, deepstack,
        )
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            # the prefix cache keys a reply from this host copy (see Qwen3.5's)
            check_stop_buffers={"new_token": new_tokens, "text_inputs": new_tokens},
        )


class Qwen3VLVisionEncoderSubmodule(VisionEncoderSubmodule):
    """Qwen3.5's vision node, also emitting the deepstack features. The
    captured block loop returns the deepstack blocks' hidden states beside its
    output, and the mergers run after it, as the main merger does."""

    def compile_regions(self) -> dict[str, list[tuple]]:
        regions = super().compile_regions()
        (rows,) = regions["model.merger"][0]
        for i in range(len(self.model.deepstack_merger_list)):
            regions[f"model.deepstack_merger_list.{i}"] = [(rows,)]
        return regions

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        configs = super().get_piecewise_accelerator_graph_configs(
            device, autocast_dtype, tp_world_size, **kwargs,
        )
        hidden, head_dim = self.config.hidden_size, self.config.head_dim

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            n = shape.total_tokens
            # the rotary tables stay fp32 (`QwenVisionModel.embed`)
            return {
                "hidden": torch.zeros(n, hidden, dtype=autocast_dtype, device=device),
                "cos": torch.zeros(n, head_dim, dtype=torch.float32, device=device),
                "sin": torch.zeros(n, head_dim, dtype=torch.float32, device=device),
            }

        configs[VISION_BLOCK_LOOP].make_static_inputs = make_static_inputs
        return configs

    def _capture_block_loop(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        s = inp.static_inputs
        hidden, taps = self.model.encode(s["hidden"], s["cos"], s["sin"])
        return {"hidden": hidden, **{f"{DEEPSTACK}_{i}": tap for i, tap in enumerate(taps)}}

    def _replay_block_loop(
        self,
        engine_inputs: ModelInputsFromEngine,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seq_lengths: tuple[int, ...],
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        if VISION_BLOCK_LOOP not in engine_inputs.piecewise_leases:
            return None
        out = engine_inputs.piecewise_runners[VISION_BLOCK_LOOP].run(
            static_inputs={"hidden": hidden, "cos": cos, "sin": sin},
            request_ids=engine_inputs.request_ids,
            seq_lens=list(seq_lengths),
        )
        # views of the runner's buffers, merged into new tensors within this step
        taps = [out.get_view(f"{DEEPSTACK}_{i}") for i in range(len(self.model.deepstack_merger_list))]
        return out.get_view("hidden"), taps

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lengths: tuple[int, ...] = (),
        row_tokens: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        hidden, cos, sin = self.model.embed(pixel_values, indices, weights, position_ids)
        encoded = self._replay_block_loop(engine_inputs, hidden, cos, sin, seq_lengths)
        if encoded is None:
            encoded = self.model.encode(hidden, cos, sin)
        embeds, deepstack = self.model.merge(*encoded)
        request_ids = engine_inputs.request_ids
        if len(request_ids) == 1:
            return {request_ids[0]: {"vision_embeds": [embeds], DEEPSTACK: deepstack}}
        out, start = {}, 0
        for rid, n in zip(request_ids, row_tokens, strict=True):
            out[rid] = {
                "vision_embeds": [embeds[start:start + n]],
                DEEPSTACK: [feature[start:start + n] for feature in deepstack],
            }
            start += n
        return out
