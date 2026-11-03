"""Served Qwen3-VL against transformers: teacher-forced top-1 agreement.

Two phases, so the server and the reference never share the GPU:

    # with `mstar serve qwen3_vl_8b` up on :8100
    python test/qwen3_vl/served_parity.py record --url http://127.0.0.1:8100 --data DATA --out RUN.json
    # with the server stopped
    python test/qwen3_vl/served_parity.py score --repo Qwen/Qwen3-VL-8B-Instruct --out RUN.json

``record`` sends each prompt (64 ShareGPT first turns, 16 MMMU validation
questions with their image) greedily for ``--new-tokens`` tokens and keeps the
text. ``score`` rebuilds each prompt with transformers' processor, appends the
served reply (its sampled ids, ``return_token_ids``), and runs transformers' sdpa
forward over the whole sequence; a position agrees when its argmax is the next
token. The bar is transformers' eager-attention greedy reply scored the same
way: the served model must agree at least as often as transformers does with
itself. Exits nonzero below it.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import urllib.request
import uuid
from pathlib import Path

# `--data`: ShareGPT_V3_unfiltered_cleaned_split.json from
# anon8231489123/ShareGPT_Vicuna_unfiltered under sharegpt/, and lmms-lab/MMMU's
# validation parquet under mmmu/data/
SHAREGPT = Path("sharegpt") / "ShareGPT_V3_unfiltered_cleaned_split.json"
MMMU = Path("mmmu") / "data"


def sharegpt_prompts(data: Path, n: int, seed: int = 0) -> list[str]:
    convs = json.loads((data / SHAREGPT).read_text())
    rng = random.Random(seed)
    rng.shuffle(convs)
    out = []
    for conv in convs:
        turns = conv.get("conversations", [])
        if len(turns) < 2 or turns[0].get("from") != "human":
            continue
        text = turns[0]["value"].strip()
        if 16 <= len(text) <= 2000:
            out.append(text)
        if len(out) == n:
            break
    return out


def mmmu_items(data: Path, n: int) -> list[tuple[str, bytes]]:
    """The first ``n`` single-image validation questions, one per subject in
    turn, with their options inlined as transformers' evals do."""
    import pyarrow.parquet as pq

    by_subject: dict[str, list] = {}
    for path in sorted((data / MMMU).glob("validation-*.parquet")):
        for row in pq.read_table(path).to_pylist():
            if row.get("image_2") is not None or row.get("image_1") is None or "<image 1>" not in row["question"]:
                continue
            by_subject.setdefault(row["id"].split("_")[1], []).append(row)
    out = []
    subjects = sorted(by_subject)
    i = 0
    while len(out) < n:
        rows = by_subject[subjects[i % len(subjects)]]
        if i // len(subjects) < len(rows):
            row = rows[i // len(subjects)]
            options = row["options"]
            options = eval(options) if isinstance(options, str) else options  # noqa: S307 - dataset literal
            letters = "".join(f"\n{chr(65 + k)}. {o}" for k, o in enumerate(options))
            question = row["question"].replace("<image 1>", "the image").strip()
            prompt = f"{question}{letters}\nAnswer with the option's letter, then explain."
            out.append((prompt, row["image_1"]["bytes"]))
        i += 1
    return out


VIDEO_PROMPTS = ("Describe what happens in this video.", "What changes between the start and the end of this video?")


def items(data: Path, videos: list[str] = ()) -> list[dict]:
    """The gate's 64 text and 16 image prompts, and two prompts per video in
    ``videos``, which are scored as a subset of their own."""
    return (
        [{"kind": "text", "prompt": p} for p in sharegpt_prompts(data, 64)]
        + [{"kind": "image", "prompt": q, "image": base64.b64encode(img).decode()} for q, img in mmmu_items(data, 16)]
        + [{"kind": "video", "prompt": q, "video": str(Path(v).resolve())} for v in videos for q in VIDEO_PROMPTS]
    )


def generate(url: str, prompt: str, media: tuple[str, bytes] | None, new_tokens: int) -> tuple[str, list[int]]:
    """The reply's text and the sampled ids, a stop token included."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())

    field("text", prompt)
    field("output_modalities", "text")
    field("model_kwargs", json.dumps({"temperature": 0.0, "max_output_tokens": new_tokens, "return_token_ids": True}))
    if media is not None:
        modality, payload = media
        field("input_modalities", f"{modality},text")
        name = "video.mp4" if modality == "video" else "image.png"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n".encode() + payload + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        f"{url}/generate", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    text, ids = b"", []
    for line in urllib.request.urlopen(request, timeout=600):
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("modality") != "text":
            continue
        text += base64.b64decode(msg.get("data") or "")
        ids += (msg.get("metadata") or {}).get("token_ids", [])
    return text.decode(errors="replace"), ids


def record(args) -> int:
    run = items(Path(args.data), args.videos)
    for item in run:
        media = None
        if item["kind"] == "image":
            media = ("image", base64.b64decode(item["image"]))
        elif item["kind"] == "video":
            media = ("video", Path(item["video"]).read_bytes())
        item["served"], item["served_ids"] = generate(args.url, item["prompt"], media, args.new_tokens)
        if not item["served_ids"]:
            raise RuntimeError("the server sent no token ids; it must honour return_token_ids")
    Path(args.out).write_text(json.dumps({"new_tokens": args.new_tokens, "items": run}))
    print(f"recorded {len(run)} replies to {args.out}")
    return 0


def score(args) -> int:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    data = json.loads(Path(args.out).read_text())
    new_tokens = data["new_tokens"]
    processor = AutoProcessor.from_pretrained(args.repo)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.repo, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()

    def inputs(item):
        content = [{"type": "text", "text": item["prompt"]}]
        media = {}
        if item["kind"] == "image":
            content.insert(0, {"type": "image"})
            # as opened: the processor composites transparency on white, as the server does
            media["images"] = [Image.open(io.BytesIO(base64.b64decode(item["image"])))]
        elif item["kind"] == "video":
            content.insert(0, {"type": "video"})
            media["videos"] = [item["video"]]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        return processor(text=[text], **media, return_tensors="pt").to("cuda")

    def agreement(batch, reply: torch.Tensor) -> tuple[int, int]:
        """Positions of ``reply`` whose token is the sdpa forward's argmax."""
        n = batch["input_ids"].shape[1]
        full = dict(batch)
        full["input_ids"] = torch.cat([batch["input_ids"], reply[None]], dim=1)
        full["attention_mask"] = torch.ones_like(full["input_ids"])
        if "mm_token_type_ids" in batch:
            full["mm_token_type_ids"] = torch.cat(
                [batch["mm_token_type_ids"], torch.zeros_like(reply[None])], dim=1,
            )
        with torch.no_grad():
            logits = model(**full).logits[0, n - 1:-1]
        return int((logits.argmax(-1) == reply).sum()), reply.numel()

    totals = {"served": [0, 0], "hf eager": [0, 0]}
    per_kind = {}
    for item in data["items"]:
        batch = inputs(item)
        served = torch.tensor(item["served_ids"], device="cuda")
        model.set_attn_implementation("eager")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=new_tokens, do_sample=False)
        model.set_attn_implementation("sdpa")
        eager = out[0, batch["input_ids"].shape[1]:]
        for name, reply in (("served", served), ("hf eager", eager)):
            hit, n = agreement(batch, reply)
            if item["kind"] != "video":  # the gate is the text and image prompts
                totals[name][0] += hit
                totals[name][1] += n
            kind = per_kind.setdefault((name, item["kind"]), [0, 0])
            kind[0] += hit
            kind[1] += n
            item[f"{name} hits"] = [hit, n]

    for (name, kind), (hit, n) in sorted(per_kind.items()):
        print(f"{name:9s} {kind:5s} {hit}/{n} = {hit / n:.4f}")
    rate = {name: hit / n for name, (hit, n) in totals.items()}
    for name, (hit, n) in totals.items():
        print(f"{name:9s} gate  {hit}/{n} = {rate[name]:.4f}")
    if ("served", "video") in per_kind:
        (sv, sn), (ev, en) = per_kind[("served", "video")], per_kind[("hf eager", "video")]
        verdict = "PASS" if sv / sn >= ev / en else "FAIL"
        print(f"video subset: {verdict} (served {sv / sn:.4f} vs hf eager {ev / en:.4f})")
    Path(args.out).write_text(json.dumps(data))  # with each item's hits, for a closer look
    passed = rate["served"] >= rate["hf eager"]
    print("PASS" if passed else "FAIL", f"(served {rate['served']:.4f} vs hf eager {rate['hf eager']:.4f})")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["record", "score"])
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--data", default="data", help="holds sharegpt/ and mmmu/ (see SHAREGPT, MMMU)")
    parser.add_argument("--videos", nargs="*", default=[], help="video files, scored as their own subset")
    args = parser.parse_args()
    return record(args) if args.phase == "record" else score(args)


if __name__ == "__main__":
    sys.exit(main())
