"""Tests for the Pi0.5 model class.

These tests focus on the structural pieces of the Pi0.5 implementation that
do not require real weights: graph walks, node-to-engine mapping, forward pass
transitions, and worker graph division. Real-weight integration is exercised
separately via end-to-end smoke tests.
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources.kv.config import KVSpec
from mstar.graph.base import GraphNode, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.pi05.components.flow_matching import (
    discretize_state,
    euler_step,
    sincos_timestep_embedding,
)
from mstar.model.pi05.config import Pi05Config
from mstar.model.pi05.pi05_model import Pi05Model
from mstar.model.pi05.submodules import Pi05ViTEncoderSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine

CONFIG_PATH = str(
    Path(__file__).resolve().parents[2] / "configs" / "pi05.yaml"
)


def _make_model() -> Pi05Model:
    """Construct a Pi0.5 model without downloading weights or tokenizer."""
    model = object.__new__(Pi05Model)
    model.model_path_hf = "test/pi05"
    model.cache_dir = None
    model.skip_weight_loading = True
    model.config = Pi05Config()
    model.tokenizer = None
    model._repo_dir = None
    model._submodule_cache = {}
    model.embed_tokens = None
    model.paligemma = None
    model.action_expert = None
    model.action_in_proj = None
    model.action_out_proj = None
    model.adaln_mlp = None
    model.siglip = None
    return model


@pytest.fixture
def _pi05_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("batched", "batch_size"), [(False, 1), (True, 1), (True, 2)],
)
def test_pi05_vision_runs_in_its_own_dtypes_under_engine_autocast(_pi05_cpu_threads, batched, batch_size):
    """The ViT node opts out of the engine's autocast and runs in openpi's
    precision: bfloat16 except the patch and position embeddings."""
    from mstar.engine.accelerator_graph_runner import autocast_scope
    from mstar.engine.node_dtype import resolve_node_dtype

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Module()
            self.embeddings.patch_embedding = torch.nn.Linear(3, 4)
            self.projection = torch.nn.Linear(4, 8)

        def forward(self, pixels):
            pooled = pixels.mean(dim=(-1, -2)).to(self.embeddings.patch_embedding.weight.dtype)
            embedded = self.embeddings.patch_embedding(pooled)
            return self.projection(embedded.to(self.projection.weight.dtype)).unsqueeze(1)

    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(19)
        submodule = Pi05ViTEncoderSubmodule(
            Encoder(), Pi05Config(vit_image_size=4, num_cameras=2),
        ).to("cpu").eval()
        assert submodule.encoder.embeddings.patch_embedding.weight.dtype == torch.float32
        assert submodule.encoder.projection.weight.dtype == torch.bfloat16
        images = torch.linspace(-1, 1, batch_size * 96).reshape(batch_size, 2, 3, 4, 4)
        engine_inputs = ModelInputsFromEngine(
            request_ids=list(range(batch_size)), per_request_info={},
        )

        with autocast_scope(None, "cpu"):
            reference = submodule.encoder(images.flatten(0, 1)).reshape(batch_size, 2, 8)
        with autocast_scope(torch.bfloat16, "cpu"):
            with autocast_scope(resolve_node_dtype(submodule, torch.bfloat16), "cpu"):
                if batched:
                    outputs = submodule.forward_batched("prefill", engine_inputs, images)
                    actual = torch.stack([outputs[rid]["img_emb"][0] for rid in engine_inputs.request_ids])
                else:
                    actual = submodule.forward("prefill", engine_inputs, images)["img_emb"][0].unsqueeze(0)

        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, reference)


def test_pi05_load_image_ships_uint8_with_unchanged_vit_input(tmp_path):
    """Cameras travel to the worker as uint8 and normalize to the same ViT
    input as the base class's float [0, 1] encoding."""
    from torchvision.io import write_png

    pixels = torch.arange(3 * 6 * 10, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(3, 6, 10)
    path = tmp_path / "cam.png"
    write_png(pixels, str(path))

    loaded = _make_model().load_image(str(path), "cpu").data
    assert loaded.dtype == torch.uint8
    assert torch.equal(loaded, pixels)

    submodule = Pi05ViTEncoderSubmodule(torch.nn.Identity(), Pi05Config(vit_image_size=8, num_cameras=1))
    from_uint8 = submodule._prepare_one(loaded.unsqueeze(0))
    from_float = submodule._prepare_one(loaded.unsqueeze(0).float() / 255.0)
    torch.testing.assert_close(from_uint8, from_float, rtol=0, atol=1e-6)


def test_pi05_vit_prepares_every_camera_in_order():
    """Each camera is letterboxed on its own and all of them reach SigLIP."""
    submodule = Pi05ViTEncoderSubmodule(torch.nn.Identity(), Pi05Config(vit_image_size=8, num_cameras=3))
    cameras = [
        torch.full((3, 8, 8), 0, dtype=torch.uint8),
        torch.full((3, 4, 8), 255, dtype=torch.uint8),
        torch.full((3, 8, 8), 51, dtype=torch.uint8),
    ]
    pixels = submodule.prepare_inputs("prefill", None, {"image_inputs": cameras}).tensor_inputs["pixel_values"]
    assert pixels.shape == (3, 3, 8, 8)
    for i, camera in enumerate(cameras):
        torch.testing.assert_close(pixels[i], submodule._prepare_one(camera)[0], rtol=0, atol=0)


def test_pi05_prefix_scales_text_by_sqrt_hidden_and_leaves_images():
    """openpi's embed_prefix: SigLIP tokens unscaled, text embeddings times sqrt(hidden_size)."""
    from mstar.model.pi05.submodules import Pi05LLMSubmodule

    config = Pi05Config(hidden_size=16, vocab_size=10)
    embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    llm = Pi05LLMSubmodule(embed, None, None, None, None, None, config)
    img_emb = torch.randn(5, config.hidden_size)
    ids = torch.tensor([1, 2, 3])
    prefix = llm.prepare_inputs("prefill", None, {"img_emb": [img_emb], "text_inputs": [ids]}).input_embeds
    torch.testing.assert_close(prefix[:5], img_emb, rtol=0, atol=0)
    torch.testing.assert_close(prefix[5:], embed(ids) * config.hidden_size ** 0.5)


# ----------------------------------------------------------------------
# Graph structure
# ----------------------------------------------------------------------


def test_pi05_graph_walks_have_expected_keys():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks.keys()) == {Pi05Model.PREFILL_WALK, Pi05Model.ACTION_GEN_WALK}


def test_pi05_prefill_is_sequential_vit_then_llm():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    prefill = walks[Pi05Model.PREFILL_WALK]
    assert isinstance(prefill, Sequential)
    assert len(prefill.sections) == 2
    first, second = prefill.sections
    assert isinstance(first, GraphNode) and first.name == "vit_encoder"
    assert isinstance(second, GraphNode) and second.name == "LLM"
    # vit_encoder must emit img_emb to LLM.
    assert any(
        edge.next_node == "LLM" and edge.name == "img_emb"
        for edge in first.outputs
    )
    # LLM consumes img_emb + text_inputs (state is encoded in text_inputs
    # as a decimal-string suffix per Pi0.5's prompt format).
    assert set(second.input_names) == {"img_emb", "text_inputs"}


def test_pi05_action_gen_is_one_llm_step_emitting_actions():
    model = _make_model()
    action_gen = model.get_graph_walk_graphs()[Pi05Model.ACTION_GEN_WALK]
    # The whole denoising loop runs inside one LLM step, so the walk is a
    # single node whose only output goes to the client.
    assert isinstance(action_gen, GraphNode) and action_gen.name == "LLM"
    assert set(action_gen.input_names) == {"noisy_actions"}
    assert len(action_gen.outputs) == 1
    out = action_gen.outputs[0]
    assert out.next_node == EMIT_TO_CLIENT
    assert out.output_modality == "action"


def test_pi05_kv_cache_config_matches_pi05_config():
    model = _make_model()
    kv_specs = [
        spec for spec in model.get_node_resources() if isinstance(spec, KVSpec)
    ]
    assert len(kv_specs) == 1
    assert kv_specs[0].nodes == {"LLM"}
    kv = kv_specs[0].config
    assert kv.num_layers == model.config.num_layers
    assert kv.num_kv_heads == model.config.num_kv_heads
    assert kv.head_dim == model.config.head_dim
    assert kv.num_qo_heads == model.config.num_qo_heads


@pytest.mark.parametrize("config_name", ["pi05.yaml", "pi05_droid.yaml"])
def test_pi05_deployments_pin_flashinfer_fa2(config_name):
    """The model keeps FlashInfer's "auto" kernel; the pi05 deployments pin
    FA2, and the override has to reach the LLM node's attention."""
    import yaml

    from mstar.engine.resources import AttentionSpec
    from mstar.engine.resources.spec import apply_yaml_overrides

    specs = _make_model().get_node_resources()
    attn = next(spec for spec in specs if isinstance(spec, AttentionSpec))
    assert attn.config.flashinfer_backend == "auto"
    apply_yaml_overrides(specs, yaml.safe_load((Path(CONFIG_PATH).parent / config_name).read_text()))
    assert attn.config.flashinfer_backend == "fa2"


# ----------------------------------------------------------------------
# Worker graph division using the YAML config
# ----------------------------------------------------------------------


def test_pi05_worker_graphs_from_yaml():
    model = _make_model()
    worker_graphs = model.get_worker_graphs(CONFIG_PATH)
    # 2 graph walks * 2 worker graphs apiece... but the prefill walk has both
    # nodes on rank 0 and they belong to different node_groups, so they remain
    # 2 separate worker graphs. action_gen has only the LLM node.
    walks_seen = {tuple(sorted(wg.graph_walks)) for wg in worker_graphs}
    assert (Pi05Model.PREFILL_WALK,) in walks_seen
    assert (Pi05Model.ACTION_GEN_WALK,) in walks_seen


# ----------------------------------------------------------------------
# Forward pass transitions
# ----------------------------------------------------------------------


def test_pi05_initial_forward_pass_args_starts_in_prefill():
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        partition_name="default",
        input_modalities=["image", "text"],
        output_modalities=["action"],
        input_signals={
            "image_inputs": [],
            "text_inputs": [],
        },
    )
    assert args.full_metadata.graph_walk == Pi05Model.PREFILL_WALK
    assert args.full_metadata.is_prefill is True
    edge_targets = {(e.next_node, e.name) for e in args.inputs}
    assert ("vit_encoder", "image_inputs") in edge_targets
    assert ("LLM", "text_inputs") in edge_targets
    # Pi0.5 has no separate state_inputs edge — state is part of text_inputs.
    assert not any(e.name == "state_inputs" for e in args.inputs)


def test_pi05_prefill_transitions_to_action_gen():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["image", "text"],
        output_modalities=["action"],
        graph_walk=Pi05Model.PREFILL_WALK,
        is_prefill=True,
    )
    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={},
    )
    assert result.full_metadata.graph_walk == Pi05Model.ACTION_GEN_WALK
    assert result.full_metadata.is_prefill is False
    assert result.request_done is False
    edge_names = {e.name for e in result.inputs}
    assert edge_names == {"noisy_actions"}


def test_pi05_action_gen_marks_request_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["image", "text"],
        output_modalities=["action"],
        graph_walk=Pi05Model.ACTION_GEN_WALK,
        is_prefill=False,
    )
    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={},
    )
    assert result.request_done is True


# ----------------------------------------------------------------------
# Postprocess
# ----------------------------------------------------------------------


def test_pi05_postprocess_action_returns_float32_bytes():
    model = _make_model()
    actions = torch.zeros(model.config.action_horizon, model.config.action_dim)
    result = model.postprocess(actions, modality="action")
    expected = (
        model.config.action_horizon * model.config.action_dim * 4
    )  # 4 bytes per float32
    assert isinstance(result, bytes)
    assert len(result) == expected


# ----------------------------------------------------------------------
# Flow matching helpers
# ----------------------------------------------------------------------


def test_sincos_timestep_embedding_shape_and_range():
    t = torch.tensor(0.5)
    dim = 16
    # The caller owns the frequency ladder and the output buffer (both are
    # preallocated once per submodule so the action-gen step is capturable).
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    emb = sincos_timestep_embedding(
        t, dim=dim, fraction=fraction, output_buffer=torch.empty(1, dim)
    )
    assert emb.shape == (1, dim)
    assert torch.all(emb.abs() <= 1.0 + 1e-6)


def test_euler_step_shapes():
    x = torch.zeros(50, 32)
    v = torch.ones(50, 32)
    out = euler_step(x, v, dt=-0.1)
    assert out.shape == (50, 32)
    assert torch.allclose(out, torch.full_like(out, -0.1))


def test_discretize_state_round_trip_within_bin():
    state = torch.linspace(-1.0, 1.0, steps=8)
    indices = discretize_state(state, num_bins=256)
    assert indices.dtype == torch.long
    assert indices.min().item() >= 0
    assert indices.max().item() <= 255
    # Endpoints should map to the extreme bins.
    assert indices[0].item() == 0
    assert indices[-1].item() == 255


def test_discretize_state_floors_into_openpi_bins():
    """openpi's np.digitize binning: floor into 2/256-wide bins, -1 below the
    range and 255 at its top."""
    state = torch.tensor([-1.5, -1.0, -0.999, 0.0, 0.99, 1.0, 2.0])
    assert discretize_state(state).tolist() == [-1, 0, 0, 128, 254, 255, 255]


def test_pi05_tokenizer_encodes_the_prompt_unchanged():
    """No lowercasing or stripping: openpi's pi05 prompt keeps its case and
    the trailing space after "Action:"."""
    from mstar.model.pi05.components.tokenization import Pi05Tokenizer

    seen = []

    class _HF:
        def __call__(self, text, add_special_tokens):
            seen.append((text, add_special_tokens))
            return type("Out", (), {"input_ids": [2, 5, 6]})()

    ids = Pi05Tokenizer(_HF(), Pi05Config()).encode_prompt("Task: Pick, State: 1;\nAction: ")
    assert seen == [("Task: Pick, State: 1;\nAction: ", True)]
    assert ids.tolist() == [2, 5, 6]


# ----------------------------------------------------------------------
# Prompt formatting (matches lerobot's Pi05PrepareStateTokenizerProcessorStep)
# ----------------------------------------------------------------------


class _StubTokenizer:
    """Captures the prompt string passed to encode_prompt for assertion."""

    def __init__(self):
        self.last_prompt: str | None = None

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        self.last_prompt = prompt
        # Return a deterministic tensor of length 1; the test only inspects
        # last_prompt.
        return torch.tensor([0], dtype=torch.long)


def test_pi05_refuses_to_start_without_a_tokenizer(monkeypatch):
    """A served model that can't load a tokenizer must fail at load, not take
    the dummy-mode path and fail every request at ingest."""
    import transformers

    def unavailable(repo, **kwargs):
        raise OSError(f"{repo} is gated")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", unavailable)
    model = _make_model()
    model.skip_weight_loading = False
    with pytest.raises(ValueError, match="google/paligemma-3b-pt-224 is gated"):
        model._load_tokenizer()


def _camera_tensors(model, n=None):
    n = model.config.num_cameras if n is None else n
    return {"image_inputs": [torch.zeros(3, 8, 8, dtype=torch.uint8) for _ in range(n)]}


@pytest.mark.parametrize("n", [1, 2, 4])
def test_pi05_process_prompt_refuses_other_camera_counts(n):
    model = _make_model()
    model.tokenizer = _StubTokenizer()
    with pytest.raises(ValueError, match="exactly 3 camera images"):
        model.process_prompt(
            prompt="hi", input_modalities=["image", "text"], output_modalities=["action"],
            tensors=_camera_tensors(model, n),
        )


def test_pi05_process_prompt_formats_state_into_text():
    """Pi05Model.process_prompt should produce the openpi-style template
    ``"Task: <text>, State: <bin0> <bin1> ... <bin31>;\\nAction: "`` and
    pass it as a SINGLE call to the tokenizer's encode_prompt. This matches
    ``lerobot/policies/pi05/processor_pi05.py::Pi05PrepareStateTokenizerProcessorStep``.
    """
    model = _make_model()
    stub = _StubTokenizer()
    model.tokenizer = stub

    state = torch.linspace(-1.0, 1.0, steps=4)  # 4 dims for the test
    result = model.process_prompt(
        prompt="pick up the\nblock",
        input_modalities=["image", "text"],
        output_modalities=["action"],
        robot_state=state,
        tensors=_camera_tensors(model),
    ).new_input_tensors

    # Result is a single text_inputs entry — no separate state_inputs.
    assert set(result.keys()) == {"text_inputs"}
    assert stub.last_prompt is not None
    # Underscores in the task aren't expected here, but newlines are stripped.
    assert "pick up the block" in stub.last_prompt
    # State bins are integers in [0, 255]; check the format prefix/suffix.
    assert stub.last_prompt.startswith("Task: pick up the block, State: ")
    assert stub.last_prompt.endswith(";\nAction: ")
    # 4 state values -> 4 bin numbers, the first should be 0 and last 255.
    assert " 0 " in stub.last_prompt or stub.last_prompt.split("State: ", 1)[1].startswith("0 ")
    assert "255" in stub.last_prompt


def test_pi05_process_prompt_without_state_uses_plain_text():
    model = _make_model()
    stub = _StubTokenizer()
    model.tokenizer = stub
    result = model.process_prompt(
        prompt="hello world",
        input_modalities=["image", "text"],
        output_modalities=["action"],
        tensors=_camera_tensors(model),
    ).new_input_tensors
    assert "text_inputs" in result
    assert stub.last_prompt == "hello world"


# ----------------------------------------------------------------------
# Lerobot -> mstar weight remap (pure-function unit tests)
# ----------------------------------------------------------------------


def test_lerobot_remap_routes_keys_to_llm_wrapper_paths():
    """Verify that ``Pi05Model._lerobot_remap`` translates each LLM-side
    lerobot key to the matching parameter path on the LLM weight container.
    """
    from mstar.model.pi05.pi05_model import Pi05Model

    model = Pi05Model.__new__(Pi05Model)  # bypass __init__ — pure-method test
    remap = model._lerobot_remap

    pali = "paligemma_with_expert.paligemma.model"
    ge = "paligemma_with_expert.gemma_expert.model"

    # PaliGemma lm_head -> tied embed_tokens
    assert remap("paligemma_with_expert.paligemma.lm_head.weight") == "embed_tokens.weight"

    # PaliGemma transformer: language_model.* prefix stripped, routed to paligemma.*
    assert remap(f"{pali}.language_model.norm.weight") == "paligemma.norm.weight"
    assert remap(f"{pali}.language_model.layers.0.self_attn.q_proj.weight") == \
        "paligemma.layers.0.self_attn.q_proj.weight"
    assert remap(f"{pali}.language_model.layers.0.mlp.gate_proj.weight") == \
        "paligemma.layers.0.mlp.gate_proj.weight"

    # Vision tower / connector live on the vit_encoder submodule; the LLM
    # remap drops them so _extract_siglip_state_dict can pick them up.
    assert remap(f"{pali}.vision_tower.vision_model.embeddings.patch_embedding.weight") is None
    assert remap(f"{pali}.multi_modal_projector.linear.weight") is None

    # Gemma expert transformer routed to action_expert.*
    assert remap(f"{ge}.layers.0.self_attn.q_proj.weight") == \
        "action_expert.layers.0.self_attn.q_proj.weight"
    assert remap(f"{ge}.layers.0.input_layernorm.dense.weight") == \
        "action_expert.layers.0.input_layernorm.dense.weight"
    assert remap(f"{ge}.norm.dense.bias") == "action_expert.norm.dense.bias"

    # gemma_expert.lm_head is intentionally dropped
    assert remap("paligemma_with_expert.gemma_expert.lm_head.weight") is None

    # Top-level wrappers — names pass through unchanged
    assert remap("action_in_proj.weight") == "action_in_proj.weight"
    assert remap("action_in_proj.bias") == "action_in_proj.bias"
    assert remap("action_out_proj.weight") == "action_out_proj.weight"

    # time_mlp_in/out -> time_mlp.linear_in/out
    assert remap("time_mlp_in.weight") == "time_mlp.linear_in.weight"
    assert remap("time_mlp_in.bias") == "time_mlp.linear_in.bias"
    assert remap("time_mlp_out.weight") == "time_mlp.linear_out.weight"
    assert remap("time_mlp_out.bias") == "time_mlp.linear_out.bias"

    # Unknown keys are dropped
    assert remap("totally.unknown.key") is None
