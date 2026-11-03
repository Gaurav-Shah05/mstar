"""Qwen3-VL's model package against transformers on a tiny random checkpoint, on CPU.

The text stack runs with dense stand-ins for the KV, attention and position
resources (as ``test_qwen3_lm.py`` does): a prompt with an image (3D MRoPE and
deepstack) must give transformers' logits, and so must the decode step after
it, which rotates by 1D positions. The shared tower must load the checkpoint, match
transformers, and keep its rotary frequencies fp32 through a bf16 cast. The
LLM node's deepstack rows, resources, walks and step are checked without
weights.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    pytest.skip("this transformers has no Qwen3-VL", allow_module_level=True)

from mstar.model.components.diffusion.qwen3.vl_prompt import qwen3_vl_positions  # noqa: E402
from mstar.model.components.qwen3_vit import (  # noqa: E402
    vision_interpolation,
    vision_position_ids,
    vision_seq_lengths,
)
from mstar.model.components.qwen_vision import (  # noqa: E402
    VISION_ATTN,
    FrameSDPA,
    QwenVisionConfig,
    QwenVisionModel,
)
from mstar.model.qwen3_5.config import ATTN, GDN_STATE, KV_CACHE, LINEAR_ATTN, ROPE  # noqa: E402
from mstar.model.qwen3_5.qwen3_5_model import STEP_GRIDS_KEY, TEXT_PART, VIDEO_PART  # noqa: E402
from mstar.model.qwen3_5.weight_loader import load_qwen3_5_vision_weights  # noqa: E402
from mstar.model.qwen3_vl.components.language_model import Qwen3VLForCausalLM, Qwen3VLTextModel  # noqa: E402
from mstar.model.qwen3_vl.config import Qwen3VLConfig  # noqa: E402
from mstar.model.qwen3_vl.qwen3_vl_model import Qwen3VLModel  # noqa: E402
from mstar.model.qwen3_vl.submodules import DEEPSTACK, Qwen3VLLLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs  # noqa: E402

IMAGE, VISION_START, VISION_END, IM_END = 250, 251, 252, 253
GRID = [(1, 4, 6)]  # one image: 2 x 3 merged tokens
MERGE = 2


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_vl")
    torch.manual_seed(0)
    config = transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000.0, "eos_token_id": IM_END,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 3, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
            "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": MERGE, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 2],
        },
        image_token_id=IMAGE, vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
    )
    model = transformers.Qwen3VLForConditionalGeneration(config).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.1)
    model.save_pretrained(path, safe_serialization=True)
    return path, model


class _DenseKV:
    """One request's KV and attention, dense: causal SDPA over all K/V written so far."""

    requires_kv_write = True

    def __init__(self):
        self._label, self._layer, self._kv, self._pending = "main", 0, {}, None

    @property
    def default_label(self):
        return self._label

    def set_default_label(self, label):
        self._label = label

    def set_default_layer_idx(self, idx):
        self._layer = idx

    def layer_view(self):
        return self._layer

    def write_kv(self, k, v):
        self._pending = (k, v)

    def run(self, q, kv_cache_layer, k=None, v=None):
        ks, vs = self._kv.setdefault(kv_cache_layer, ([], []))
        ks.append(self._pending[0])
        vs.append(self._pending[1])
        k_all, v_all = torch.cat(ks), torch.cat(vs)
        rep = q.shape[1] // k_all.shape[1]
        mask = torch.ones(q.shape[0], k_all.shape[0], dtype=torch.bool).tril(k_all.shape[0] - q.shape[0])
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k_all.repeat_interleave(rep, 1).transpose(0, 1),
            v_all.repeat_interleave(rep, 1).transpose(0, 1), attn_mask=mask,
        )
        return out.transpose(0, 1)


class _Rope1D:
    """The position resource's unfused 1D RoPE at ``next_pos``."""

    next_pos = 0

    def fusable_rope_theta(self, head_dim):
        return None

    def apply_qk(self, q, k, label, rope_theta, **kwargs):
        d = q.shape[-1]
        pos = torch.arange(self.next_pos, self.next_pos + q.shape[0], dtype=torch.float32)
        freqs = torch.outer(pos, 1.0 / (rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)))
        emb = torch.cat([freqs, freqs], -1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]

        def rot(x):
            return torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)

        return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def _text_stack(path):
    config = Qwen3VLConfig.from_hf(path)
    lm = Qwen3VLTextModel(config).eval()
    lm.load_checkpoint(path)
    dense, rope = _DenseKV(), _Rope1D()
    for module in lm.modules():
        if hasattr(module, "bind_resources") and module is not lm:
            module.bind_resources({ATTN: dense, KV_CACHE: dense, ROPE: rope})
    return lm, rope


def test_config_reads_the_checkpoint(checkpoint, tmp_path):
    path, _ = checkpoint
    config = Qwen3VLConfig.from_hf(path)
    assert config.mrope_section == (2, 3, 3)
    assert config.rope_theta == 5_000_000.0
    assert config.head_dim == 16 and config.num_hidden_layers == 4
    assert (config.vision_start_token_id, config.vision_end_token_id, config.image_token_id) == (
        VISION_START, VISION_END, IMAGE,
    )
    config.extra_stop_token_ids = (7,)
    assert config.stop_token_ids == {IM_END, 7}

    raw = json.loads((path / "config.json").read_text())
    # where this transformers writes the rope config
    rope = raw["text_config"].get("rope_scaling") or raw["text_config"]["rope_parameters"]
    rope["mrope_section"] = [2, 3, 2]
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="does not cover head_dim"):
        Qwen3VLConfig.from_hf(tmp_path)
    rope.update(mrope_section=[2, 3, 3], mrope_interleaved=False)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="interleaved MRoPE only"):
        Qwen3VLConfig.from_hf(tmp_path)


def test_image_prompt_and_decode_match_transformers(checkpoint):
    path, model = checkpoint
    lm, rope = _text_stack(path)
    hidden = lm.config.hidden_size
    image_tokens = sum(t * (h // MERGE) * (w // MERGE) for t, h, w in GRID)
    ids = torch.tensor([5, 6, VISION_START, *[IMAGE] * image_tokens, VISION_END, 7, 8, 9])
    image_rows = (ids == IMAGE).nonzero().squeeze(-1)
    torch.manual_seed(1)
    merged = torch.randn(image_tokens, hidden)
    features = [torch.randn(image_tokens, hidden) for _ in range(2)]
    positions = qwen3_vl_positions(ids, IMAGE, GRID, MERGE)

    reference = model.model.language_model
    with torch.no_grad():
        embeds = reference.embed_tokens(ids).index_copy(0, image_rows, merged)
        want = model.lm_head(reference(
            inputs_embeds=embeds[None], position_ids=positions[:, None].long(),
            visual_pos_masks=(ids == IMAGE)[None], deepstack_visual_embeds=features, use_cache=False,
        ).last_hidden_state)[0]

        dense_features = torch.zeros(len(features), ids.numel(), hidden)
        for i, feature in enumerate(features):
            dense_features[i, image_rows] = feature
        got = lm.logits(lm(
            embeds, label="main", cos_sin=lm.build_cos_sin(positions, torch.float32), deepstack=dense_features,
        ))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    # The next token is text: all three grids at one position past the prompt's
    # furthest, which the decode step rotates by as 1D RoPE.
    nxt = torch.tensor([11])
    rope.next_pos = int(positions.max()) + 1
    with torch.no_grad():
        full = torch.cat([embeds, reference.embed_tokens(nxt)])
        full_positions = torch.cat([positions, torch.full((3, 1), rope.next_pos)], dim=1)
        want = model.lm_head(reference(
            inputs_embeds=full[None], position_ids=full_positions[:, None].long(),
            visual_pos_masks=torch.cat([ids, nxt]).eq(IMAGE)[None], deepstack_visual_embeds=features,
            use_cache=False,
        ).last_hidden_state)[0, -1]
        got = lm.logits(lm(lm.embed(nxt), label="main"))[0]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_deepstack_layers_add_after_their_layer(checkpoint):
    """Zero features are the plain stack, and only the image rows move."""
    path, _ = checkpoint
    ids = torch.tensor([5, 6, 7, 8])
    positions = torch.arange(4.0).expand(3, -1)
    outs = []
    for features in (None, torch.zeros(2, 4, 64), torch.zeros(2, 4, 64).index_fill(1, torch.tensor([2]), 1.0)):
        lm, _ = _text_stack(path)
        with torch.no_grad():
            outs.append(lm(lm.embed(ids), label="main", cos_sin=lm.build_cos_sin(positions, torch.float32),
                           deepstack=features))
    torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=1e-6)
    assert torch.equal(outs[2][:2], outs[0][:2]) and not torch.allclose(outs[2][2], outs[0][2])


def test_text_model_keeps_inv_freq_fp32():
    config = Qwen3VLConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, intermediate_size=48, vocab_size=64, mrope_section=(2, 3, 3))
    lm = Qwen3VLTextModel(config)
    exact = lm.inv_freq.clone()
    lm.to(torch.bfloat16)
    assert lm.inv_freq.dtype == torch.float32 and torch.equal(lm.inv_freq, exact)


def _run_tower(tower, pixels):
    for block in tower.blocks:
        block.attn.bind_resources({VISION_ATTN: FrameSDPA(vision_seq_lengths(GRID))})
    indices, weights = vision_interpolation(GRID, tower.config.num_grid_per_side, MERGE)
    with torch.no_grad():
        return tower(pixels, indices, weights, vision_position_ids(GRID, MERGE))


def test_tower_loads_and_matches_transformers(checkpoint):
    path, model = checkpoint
    tower = QwenVisionModel(QwenVisionConfig.from_hf(path)).eval()
    load_qwen3_5_vision_weights(tower, path)
    pixels = torch.randn(sum(t * h * w for t, h, w in GRID), 3 * 2 * 16 * 16)
    merged, features = _run_tower(tower, pixels)
    with torch.no_grad():
        want = model.model.visual(pixels, grid_thw=torch.tensor(GRID))
    torch.testing.assert_close(merged, want.pooler_output, rtol=1e-5, atol=1e-5)
    for mine, theirs in zip(features, want.deepstack_features, strict=True):
        torch.testing.assert_close(mine, theirs, rtol=1e-5, atol=1e-5)

    exact = tower.rotary_pos_emb.inv_freq.clone()
    tower.to(torch.bfloat16)
    assert tower.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert torch.equal(tower.rotary_pos_emb.inv_freq, exact)


def test_unsorted_deepstack_blocks_take_their_listed_mergers():
    """transformers emits the features in block order, each through the merger
    its block is listed with."""
    sizes = dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=24, patch_size=16,
                 temporal_patch_size=2, spatial_merge_size=MERGE, num_position_embeddings=16, in_channels=3)
    torch.manual_seed(2)
    reference = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel(
        transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLVisionConfig(
            deepstack_visual_indexes=[2, 0], **sizes,
        ),
    ).eval()
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(0, 0.1)
    tower = QwenVisionModel(QwenVisionConfig(deepstack_visual_indexes=(2, 0), **sizes)).eval()
    tower.load_state_dict(reference.state_dict(), strict=False)
    pixels = torch.randn(sum(t * h * w for t, h, w in GRID), 3 * 2 * 16 * 16)
    _, features = _run_tower(tower, pixels)
    with torch.no_grad():
        want = reference(pixels, grid_thw=torch.tensor(GRID)).deepstack_features
    for mine, theirs in zip(features, want, strict=True):
        torch.testing.assert_close(mine, theirs, rtol=1e-5, atol=1e-5)


def _llm_node(num_deepstack: int = 2, hidden: int = 8) -> Qwen3VLLLMSubmodule:
    node = Qwen3VLLLMSubmodule.__new__(Qwen3VLLLMSubmodule)
    node.num_deepstack = num_deepstack
    node.vision_config = SimpleNamespace(spatial_merge_size=MERGE)
    node.config = SimpleNamespace(hidden_size=hidden)
    node.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(4, hidden)))
    node.get_device = lambda: torch.device("cpu")
    return node


def test_deepstack_rows_sit_between_each_images_sentinels():
    """Text, an image (2 x 3 merged), text, a two-group video (each 1 x 2
    merged), as `_vision_inputs` lays the prompt row out."""
    node = _llm_node()
    image, video = (1, 4, 6), (2, 2, 4)
    order = (TEXT_PART, 1, TEXT_PART, VIDEO_PART, TEXT_PART, VIDEO_PART)
    texts = [torch.zeros(3), torch.zeros(2), torch.zeros(1)]
    tokens = 6 + 2 * 2
    features = [torch.arange(tokens, dtype=torch.float)[:, None].expand(tokens, 8) + 100 * i for i in range(2)]
    info = SimpleNamespace(step_metadata={STEP_GRIDS_KEY: (image, video), "prefill_order": order})
    rows = 3 + (6 + 2) + 2 + (2 + 2) + 1 + (2 + 2)  # 22
    out = node._deepstack_rows(info, {"text_inputs": texts, DEEPSTACK: features}, rows)
    assert out.shape == (rows, 2, 8)
    image_rows = [4, 5, 6, 7, 8, 9]
    video_rows = [14, 15, 19, 20]
    filled = image_rows + video_rows
    for layer in range(2):
        assert out[filled, layer, 0].tolist() == [100 * layer + i for i in range(tokens)]
    empty = [r for r in range(rows) if r not in filled]
    assert not out[empty].any()

    with pytest.raises(ValueError, match="prompt rows"):
        node._deepstack_rows(info, {"text_inputs": texts, DEEPSTACK: features}, rows + 1)


def test_step_declares_no_recurrent_state_and_text_rotates_in_the_layers():
    step = _llm_node().declare_step("decode", ["r"], [ARNodeInputs(input_seq_len=1)])
    assert GDN_STATE not in step.steps and LINEAR_ATTN not in step.steps
    assert {KV_CACHE, ATTN, ROPE} <= set(step.steps)
    assert Qwen3VLLLMSubmodule._rope_in_graph(None, "prefill_text")
    assert Qwen3VLLLMSubmodule._rope_in_graph(None, "decode")
    assert not Qwen3VLLLMSubmodule._rope_in_graph(None, "prefill_vision")


def test_resources_and_walks(checkpoint):
    path, _ = checkpoint
    model = Qwen3VLModel.__new__(Qwen3VLModel)
    model.config = Qwen3VLConfig.from_hf(path)
    model.vision_config = QwenVisionConfig.from_hf(path)
    model._image_processor = SimpleNamespace(size={"shortest_edge": 65536}, patch_size=16)
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert set(specs) == {KV_CACHE, ATTN, ROPE, "sampler", VISION_ATTN}
    assert specs[KV_CACHE].config.num_layers == model.config.num_hidden_layers
    assert specs[ROPE].config.rope_theta == model.config.rope_theta

    vision = model.get_graph_walk_graphs()["prefill_vision"]
    encoder, llm = vision.sections
    assert {edge.name for edge in encoder.outputs} == {"vision_embeds", DEEPSTACK}
    assert DEEPSTACK in llm.input_names


def test_vision_row_matches_transformers_layout(checkpoint):
    """The prompt row the LLM node builds for prefill_vision (Qwen3.5's
    `_vision_inputs` plus this model's deepstack rows): 3D positions equal to
    transformers' `get_rope_index`, image embeddings and deepstack features at
    the image tokens, and the position advance to the next text position."""
    path, model = checkpoint
    config, vision = Qwen3VLConfig.from_hf(path), QwenVisionConfig.from_hf(path)
    lm = Qwen3VLForCausalLM(config).eval()
    lm.model.load_checkpoint(path)
    node = Qwen3VLLLMSubmodule(lm, config, vision)
    node.node_resources = {ROPE: SimpleNamespace(position=lambda rid, label: 0)}

    tokens = sum(t * (h // MERGE) * (w // MERGE) for t, h, w in GRID)
    before, after = torch.tensor([5, 6, 7]), torch.tensor([8, 9])
    torch.manual_seed(3)
    merged = torch.randn(tokens, config.hidden_size)
    features = [torch.randn(tokens, config.hidden_size) for _ in range(2)]
    info = SimpleNamespace(
        request_id="r", step_metadata={STEP_GRIDS_KEY: tuple(GRID), "prefill_order": (TEXT_PART, 1, TEXT_PART)},
    )
    with torch.no_grad():
        row = node._vision_inputs(
            info, {"text_inputs": [before, after], "vision_embeds": [merged], DEEPSTACK: features},
        )

    ids = torch.cat([before, torch.tensor([VISION_START, *[IMAGE] * tokens, VISION_END]), after])
    image = ids == IMAGE
    want, _ = model.model.get_rope_index(
        ids[None], mm_token_type_ids=image.int()[None], image_grid_thw=torch.tensor(GRID),
    )
    assert torch.equal(row.custom_pos_ids.long(), want[:, 0])
    assert row.tensor_inputs["mrope_advance"] == int(want.max()) + 1
    with torch.no_grad():
        torch.testing.assert_close(row.input_embeds[image], merged)
        torch.testing.assert_close(row.input_embeds[~image], lm.model.embed(ids[~image]))
    rows = row.tensor_inputs[DEEPSTACK]
    for layer, feature in enumerate(features):
        assert torch.equal(rows[image, layer], feature)
    assert not rows[~image].any()


def test_vision_node_splits_a_batch_by_request(checkpoint):
    """Two requests through the vision node's eager forward together give each
    one its own embeddings and deepstack features, as it gets alone."""
    from mstar.model.qwen3_vl.submodules import Qwen3VLVisionEncoderSubmodule

    path, _ = checkpoint
    vision = QwenVisionConfig.from_hf(path)
    tower = QwenVisionModel(vision).eval()
    load_qwen3_5_vision_weights(tower, path)
    patch = vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2
    mean, std = torch.full((patch,), 120.0), torch.full((patch,), 60.0)
    node = Qwen3VLVisionEncoderSubmodule(tower, vision, (mean, std))
    grids = {"a": [(1, 4, 6)], "b": [(1, 2, 4), (1, 4, 4)]}
    torch.manual_seed(4)
    pixels = {rid: torch.randint(0, 256, (sum(t * h * w for t, h, w in g), patch), dtype=torch.uint8)
              for rid, g in grids.items()}

    def run(rids):
        inputs = [
            node.prepare_inputs("prefill_vision", SimpleNamespace(step_metadata={STEP_GRIDS_KEY: tuple(grids[r])}),
                                {"pixel_values": [pixels[r]]})
            for r in rids
        ]
        frames = [n for inp in inputs for n in inp.tensor_inputs["seq_lengths"]]
        for block in tower.blocks:
            block.attn.bind_resources({VISION_ATTN: FrameSDPA(frames)})
        engine_inputs = SimpleNamespace(request_ids=list(rids), piecewise_leases={})
        with torch.no_grad():
            pre = node.preprocess("prefill_vision", engine_inputs, inputs)
            return node.forward_batched("prefill_vision", engine_inputs, **pre)

    together = run(["a", "b"])
    for rid in grids:
        alone = run([rid])[rid]
        torch.testing.assert_close(together[rid]["vision_embeds"][0], alone["vision_embeds"][0], rtol=1e-5, atol=1e-5)
        assert len(together[rid][DEEPSTACK]) == len(vision.deepstack_visual_indexes)
        for mine, ref in zip(together[rid][DEEPSTACK], alone[DEEPSTACK], strict=True):
            torch.testing.assert_close(mine, ref, rtol=1e-5, atol=1e-5)
