"""The Thinker reports each row's KV span from its inputs alone, so admission
can leave out a row whose KV does not fit before prepare_inputs runs."""
import pytest
import torch

from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

HIDDEN = 4


@pytest.fixture
def thinker():
    return object.__new__(ThinkerSubmodule)


def _len(thinker, walk, inputs):
    info = thinker.get_input_sequence_len(walk, None, inputs)
    return None if info is None else info.seq_len


def test_text_is_its_token_count(thinker):
    assert _len(thinker, "prefill_text", {"text_inputs": [torch.zeros(7)]}) == 7


@pytest.mark.parametrize("walk,name", [
    ("prefill_audio", "audio_embeds"), ("prefill_vision", "vision_embeds"),
])
def test_embeds_count_their_two_sentinels(thinker, walk, name):
    assert _len(thinker, walk, {name: [torch.zeros(10, HIDDEN)]}) == 12


def test_decode_is_one_token(thinker):
    assert _len(thinker, "thinker_decode", {}) == 1


def test_a_missing_input_declares_nothing(thinker):
    assert _len(thinker, "prefill_audio", {}) is None


def test_an_audio_walk_counts_the_text_span_it_prefills(thinker):
    inputs = {"audio_embeds": [torch.zeros(10, HIDDEN)], "text_inputs": [torch.zeros(5)]}
    assert _len(thinker, "prefill_audio", inputs) == 17


def test_an_audio_walk_with_its_text_matches_the_two_walks_it_replaces():
    """The clip's rows as the audio walk lays them out, then the text span
    with positions running on from the end sentinel (a text walk's, which
    the plan would place) and the masks a text walk sends the Talker."""
    from types import SimpleNamespace

    from torch import nn

    from mstar.model.qwen3_omni.config import THINKER_POS, Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.submodules import talker_text_mask

    config = Qwen3OmniModelConfig()
    thinker = ThinkerSubmodule.__new__(ThinkerSubmodule)
    nn.Module.__init__(thinker)
    thinker.config = config
    ids = (config.thinker.audio_start_token_id, config.thinker.audio_end_token_id,
           config.im_start_token_id, config.assistant_token_id)
    embed = nn.Embedding(max(config.thinker_text.vocab_size, *ids) + 1, HIDDEN)
    thinker.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
    thinker._audio_bos_embed = thinker._audio_eos_embed = None
    thinker.__dict__["_cached_device"] = torch.device("cpu")
    start = 7
    thinker.node_resources = {THINKER_POS: SimpleNamespace(position=lambda rid, label: start)}
    audio = torch.randn(10, HIDDEN)
    text = torch.tensor([9, config.im_start_token_id, config.assistant_token_id, 11])
    info = SimpleNamespace(rid_handle=0, step_metadata={"audio_output": True})

    alone = thinker.prepare_inputs("prefill_audio", info, {"audio_embeds": [audio]})
    both = thinker.prepare_inputs("prefill_audio", info, {"audio_embeds": [audio], "text_inputs": [text]})

    n = alone.input_seq_len
    assert both.input_seq_len == n + len(text)
    assert torch.equal(both.input_embeds[:n], alone.input_embeds)
    assert torch.equal(both.input_embeds[n:], thinker.model.model.embed_tokens(text))
    assert torch.equal(both.custom_pos_ids[:, :n], alone.custom_pos_ids)
    expected = torch.arange(start + n, start + n + len(text), dtype=both.custom_pos_ids.dtype).expand(3, -1)
    assert torch.equal(both.custom_pos_ids[:, n:], expected)
    masks = both.tensor_inputs["masks_for_talker"]
    assert torch.equal(masks[:, :n], alone.tensor_inputs["masks_for_talker"])
    assert not masks[0, n:].any() and torch.equal(masks[1, n:], talker_text_mask(text, config))
    # the prompt's ids, as HF's input_ids hold them: the clip's sentinels and pads, then the text
    pads = [config.thinker.audio_token_id] * len(audio)
    clip = [config.thinker.audio_start_token_id, *pads, config.thinker.audio_end_token_id]
    assert alone.input_ids.tolist() == clip
    assert both.input_ids.tolist() == clip + text.tolist()


def test_a_leading_span_runs_before_the_clip_as_a_text_walk_would():
    """The leading span takes the stream's first positions; the clip and the
    text after it follow, shifted by the span's length."""
    from types import SimpleNamespace

    from torch import nn

    from mstar.model.qwen3_omni.config import THINKER_POS, Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.submodules import talker_text_mask

    config = Qwen3OmniModelConfig()
    thinker = ThinkerSubmodule.__new__(ThinkerSubmodule)
    nn.Module.__init__(thinker)
    thinker.config = config
    ids = (config.thinker.audio_start_token_id, config.thinker.audio_end_token_id,
           config.im_start_token_id, config.assistant_token_id, config.system_token_id)
    embed = nn.Embedding(max(config.thinker_text.vocab_size, *ids) + 1, HIDDEN)
    thinker.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
    thinker._audio_bos_embed = thinker._audio_eos_embed = None
    thinker.__dict__["_cached_device"] = torch.device("cpu")
    counter = {"pos": 3}
    thinker.node_resources = {THINKER_POS: SimpleNamespace(position=lambda rid, label: counter["pos"])}
    audio = torch.randn(10, HIDDEN)
    # a system turn, then the user turn the clip opens
    prefix = torch.tensor([config.im_start_token_id, config.system_token_id, 21, 22, config.im_start_token_id, 23])
    text = torch.tensor([9, 10])
    info = SimpleNamespace(rid_handle=0, step_metadata={"audio_output": True})

    both = thinker.prepare_inputs(
        "prefill_audio", info, {"audio_embeds": [audio], "text_prefix": [prefix], "text_inputs": [text]},
    )
    counter["pos"] = 3 + len(prefix)
    rest = thinker.prepare_inputs("prefill_audio", info, {"audio_embeds": [audio], "text_inputs": [text]})

    p = len(prefix)
    assert both.input_seq_len == p + rest.input_seq_len
    assert torch.equal(both.input_embeds[:p], embed(prefix))
    assert torch.equal(both.input_embeds[p:], rest.input_embeds)
    leading = torch.arange(3, 3 + p, dtype=both.custom_pos_ids.dtype).expand(3, -1)
    assert torch.equal(both.custom_pos_ids[:, :p], leading)
    assert torch.equal(both.custom_pos_ids[:, p:], rest.custom_pos_ids)
    masks = both.tensor_inputs["masks_for_talker"]
    assert not masks[0, :p].any() and torch.equal(masks[1, :p], talker_text_mask(prefix, config))
    assert torch.equal(masks[:, p:], rest.tensor_inputs["masks_for_talker"])
    assert both.input_ids.tolist() == prefix.tolist() + rest.input_ids.tolist()


@pytest.mark.parametrize("walk", ["prefill_text", "prefill_audio"])
def test_a_text_only_request_builds_no_talker_masks(walk):
    """Only a request that speaks feeds the Talker, so only its rows carry the
    Talker's masks; the rest of the row is the same either way."""
    from types import SimpleNamespace

    from torch import nn

    from mstar.model.qwen3_omni.config import THINKER_POS, Qwen3OmniModelConfig

    config = Qwen3OmniModelConfig()
    thinker = ThinkerSubmodule.__new__(ThinkerSubmodule)
    nn.Module.__init__(thinker)
    thinker.config = config
    ids = (config.thinker.audio_start_token_id, config.thinker.audio_end_token_id,
           config.im_start_token_id, config.assistant_token_id)
    thinker.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(max(*ids) + 1, HIDDEN)))
    thinker._audio_bos_embed = thinker._audio_eos_embed = None
    thinker.__dict__["_cached_device"] = torch.device("cpu")
    thinker.node_resources = {THINKER_POS: SimpleNamespace(position=lambda rid, label: 2)}
    text = torch.tensor([9, config.im_start_token_id, config.assistant_token_id])
    inputs = {"text_inputs": [text]} if walk == "prefill_text" else {"audio_embeds": [torch.randn(6, HIDDEN)],
                                                                     "text_inputs": [text]}

    def prepare(speaks):
        info = SimpleNamespace(rid_handle=0, step_metadata={"audio_output": speaks})
        return thinker.prepare_inputs(walk, info, inputs)

    speaking, silent = prepare(True), prepare(False)
    assert "masks_for_talker" in speaking.tensor_inputs and "masks_for_talker" not in silent.tensor_inputs
    assert torch.equal(speaking.input_embeds, silent.input_embeds) and torch.equal(speaking.input_ids, silent.input_ids)
    cut = thinker.split_inputs(walk, None, silent, 1, silent.input_seq_len)
    assert "masks_for_talker" not in cut.tensor_inputs
