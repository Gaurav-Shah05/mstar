"""Qwen3-VL's vision tower and text stack against transformers, real weights.

The tower runs with per-frame SDPA in place of the ragged resource, and the
text stack with dense stand-ins for the KV, attention and position resources
(those of ``test/modular/test_qwen3_lm.py``), so this checks the port's maths:
weight mapping, the deepstack taps and their injection, and 3D MRoPE. The
served path, with the engine's kernels, is ``served_parity.py``.

    python test/qwen3_vl/component_parity.py Qwen/Qwen3-VL-2B-Instruct
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from mstar.model.components.diffusion.qwen3.vl_prompt import qwen3_vl_positions
from mstar.model.components.qwen3_vit import vision_interpolation, vision_position_ids
from mstar.model.components.qwen_vision import VISION_ATTN, FrameSDPA, QwenVisionModel
from mstar.model.qwen3_5.config import ATTN, KV_CACHE, ROPE
from mstar.model.qwen3_5.weight_loader import load_qwen3_5_vision_weights
from mstar.model.qwen3_vl.components.language_model import Qwen3VLForCausalLM
from mstar.model.qwen3_vl.qwen3_vl_model import Qwen3VLModel


class DenseKV:
    """One request's paged KV and attention as dense tensors: causal SDPA over
    everything written so far, GQA by repetition."""

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
        k_new, v_new = self._pending
        ks.append(k_new)
        vs.append(v_new)
        k_all, v_all = torch.cat(ks), torch.cat(vs)
        n_new, n_all = q.shape[0], k_all.shape[0]
        rep = q.shape[1] // k_all.shape[1]
        k_all, v_all = k_all.repeat_interleave(rep, 1), v_all.repeat_interleave(rep, 1)
        mask = torch.ones(n_new, n_all, dtype=torch.bool, device=q.device).tril(diagonal=n_all - n_new)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k_all.transpose(0, 1), v_all.transpose(0, 1), attn_mask=mask,
        )
        return out.transpose(0, 1)


class NoRope:
    """The position resource, for a step that brings its own 3D tables."""

    def fusable_rope_theta(self, head_dim):
        return None


def test_image() -> Image.Image:
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    for i in range(0, 640, 40):
        draw.rectangle([i, 0, i + 20, 480], fill=(i % 255, 80, 200 - i % 200))
    draw.ellipse([200, 140, 440, 340], fill=(250, 30, 30))
    draw.text((20, 20), "Qwen3-VL parity", fill=(0, 0, 0))
    return image


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main(repo: str) -> int:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    ours_model = Qwen3VLModel(repo)
    weights = ours_model._resolve_weights()
    processor = AutoProcessor.from_pretrained(weights)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[test_image()], return_tensors="pt").to(device)
    grid = [tuple(g) for g in batch["image_grid_thw"].tolist()]

    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        weights, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    with torch.no_grad():
        want = hf.model.visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])
        want_logits = hf(**batch).logits[0]
        # transformers against itself: the spread between two valid kernels
        hf.set_attn_implementation("eager")
        eager_logits = hf(**batch).logits[0]
        hf.set_attn_implementation("sdpa")
        # and in fp32: how far each bf16 run is from exact
        hf.float()
        exact = hf.model.visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])
        exact_logits = hf(**batch).logits[0]

    # the tower
    torch.set_default_dtype(torch.bfloat16)
    tower = QwenVisionModel(ours_model.vision_config).to(device)
    torch.set_default_dtype(torch.float32)
    tower = tower.to(device, torch.bfloat16)  # the engine's cast; inv_freq stays fp32
    load_qwen3_5_vision_weights(tower, weights, device=device)
    tower.eval()
    for block in tower.blocks:
        block.attn.bind_resources({VISION_ATTN: FrameSDPA([h * w for t, h, w in grid for _ in range(t)])})
    merge = ours_model.vision_config.spatial_merge_size
    indices, wts = vision_interpolation(grid, ours_model.vision_config.num_grid_per_side, merge)
    with torch.no_grad():
        embeds, deepstack = tower(
            batch["pixel_values"], indices.to(device), wts.to(device),
            vision_position_ids(grid, merge).to(device),
        )
    failures = 0
    rows = [("tower merged", embeds, want.pooler_output)] + [
        (f"deepstack {i}", mine, theirs)
        for i, (mine, theirs) in enumerate(zip(deepstack, want.deepstack_features, strict=True))
    ]
    exact_rows = [exact.pooler_output, *exact.deepstack_features]
    for (name, mine, theirs), ref in zip(rows, exact_rows, strict=True):
        ours_fp32, hf_fp32 = cosine(mine, ref), cosine(theirs, ref)
        print(f"{name:14s} cos(ours, hf) {cosine(mine, theirs):.6f}  vs fp32: ours {ours_fp32:.6f} hf {hf_fp32:.6f}")
        # ours may not be further from exact than transformers' own bf16 tower
        failures += (1 - ours_fp32) > 1.5 * (1 - hf_fp32) + 1e-5
    del hf
    torch.cuda.empty_cache()

    # the text stack, fed our tower's output
    torch.set_default_dtype(torch.bfloat16)
    lm = Qwen3VLForCausalLM(ours_model.config).to(device)
    torch.set_default_dtype(torch.float32)
    lm.model.load_checkpoint(weights, device=device)
    lm.eval()
    ids = batch["input_ids"][0]
    image_token = ours_model.config.image_token_id
    image_rows = (ids == image_token).nonzero().squeeze(-1)
    # an image token's logits predict nothing the model is trained on
    text = ids != image_token
    positions = qwen3_vl_positions(ids, image_token, grid, merge)

    def run_lm(merged: torch.Tensor, features: list[torch.Tensor]) -> torch.Tensor:
        dense = DenseKV()
        for module in lm.modules():
            if hasattr(module, "bind_resources") and module is not lm.model:
                module.bind_resources({ATTN: dense, KV_CACHE: dense, ROPE: NoRope()})
        with torch.no_grad():
            x = lm.model.embed(ids).index_copy(0, image_rows, merged.to(torch.bfloat16))
            ds = torch.zeros(len(features), ids.numel(), x.shape[-1], dtype=x.dtype, device=device)
            for i, feature in enumerate(features):
                ds[i, image_rows] = feature.to(x.dtype)
            cos_sin = lm.model.build_cos_sin(positions, torch.bfloat16)
            return lm.model.logits(lm.model(x, label="main", cos_sin=cos_sin, deepstack=ds))

    def report(name: str, logits: torch.Tensor) -> tuple[float, float]:
        agree = (logits[text].argmax(-1) == want_logits[text].argmax(-1)).float().mean().item()
        exact_agree = (logits[text].argmax(-1) == exact_logits[text].argmax(-1)).float().mean().item()
        cos = cosine(logits[text], exact_logits[text])
        print(f"{name:28s} top-1 vs hf-sdpa {agree:.4f}  vs fp32 {exact_agree:.4f}  cos vs fp32 {cos:.6f}")
        return exact_agree, cos

    print(f"text positions: {int(text.sum())} of {ids.numel()}")
    refs = [report("hf eager (bf16)", eager_logits), report("hf sdpa (bf16)", want_logits)]
    report("ours, hf tower inputs", run_lm(want.pooler_output, want.deepstack_features))
    agree, cos = report("ours, our tower", run_lm(embeds, deepstack))
    # no further from exact than the worse of transformers' own bf16 runs
    ref_agree, ref_cos = min(a for a, _ in refs), min(c for _, c in refs)
    failures += agree < ref_agree - 0.02 or (1 - cos) > 1.5 * (1 - ref_cos) + 1e-4
    print("PASS" if not failures else f"FAIL ({failures})")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-2B-Instruct"))
