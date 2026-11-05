# openpi Pi0.5 reference for M* pi05

Standalone openpi (PyTorch) reference for M*'s pi05 serving: same inputs, raw 32-dim
action chunk out (before un-normalization / slicing).

## Layout

- `openpi/` - clone of github.com/Physical-Intelligence/openpi with its own `.venv`
- `make_inputs.py` - builds `cases.npz` / `cases.json` (16 DROID cases as M* sees them); M* venv
- `ref_actions.py` - runs `PI0Pytorch.sample_actions(num_steps=10)` per case; openpi venv
- `compare.py` - max abs diff between two action files, overall and per case
- `run_all.sh` - everything below in order
- `scripts/` - one-off checks (key mapping, tied weights, tokenizer comparison)

## Setup (once)

```bash
cd /workspace/openpi-ref
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi
cd openpi
# copy link mode: the transformers patch below must not hard-link into the shared uv cache
UV_CACHE_DIR=/workspace/uv-cache UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
flock /workspace/gpu.lock env -u VIRTUAL_ENV .venv/bin/python -c \
  "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
```

Always run the openpi venv with `env -u VIRTUAL_ENV` (the shell profile activates `/workspace/venv`)
and `JAX_PLATFORMS=cpu` (openpi imports JAX, which would otherwise preallocate GPU memory;
`ref_actions.py` also sets it by default).

## Rerun

```bash
/workspace/openpi-ref/run_all.sh 2>&1 | tee /workspace/openpi-ref/run_all.log
```

which is:

```bash
# inputs (M* venv, cwd = M* checkout)
cd /workspace/wt-rp1-pi05
flock /workspace/gpu.lock /workspace/venv/bin/python /workspace/openpi-ref/make_inputs.py

# reference runs (openpi venv)
cd /workspace/openpi-ref/openpi
PY="env -u VIRTUAL_ENV JAX_PLATFORMS=cpu .venv/bin/python"
flock /workspace/gpu.lock $PY ../ref_actions.py --cases ../cases.npz --variant eager   --out ../ref_A.npz
flock /workspace/gpu.lock $PY ../ref_actions.py --cases ../cases.npz --variant compile --compile-mode default --out ../ref_B.npz
flock /workspace/gpu.lock $PY ../ref_actions.py --cases ../cases.npz --variant eager --inputs openpi --save-inputs \
    --out ../ref_A_openpi_inputs.npz
$PY ../compare.py ../ref_A.npz ../ref_B.npz
$PY ../compare.py ../ref_A.npz ../ref_A_openpi_inputs.npz

# informational: openpi tokenization vs M*'s ids (CPU)
$PY ../scripts/compare_tokenization.py
```

To gate M*: serve `configs/pi05_droid.yaml`, send the 16 cases with `model_kwargs["seed"] = i`, decode
the returned float32 bytes as (15, 32), and run `compare.py ref_A.npz <mstar.npz>` against the
A-vs-B spread.

## What the reference does

- Model: openpi `Pi0Config(pi05=True, action_horizon=15)` (= `pi05_droid`, training/config.py:630-631),
  gemma_2b + gemma_300m, `max_token_len=200`.
- Weights: `lerobot/pi05_base/model.safetensors`. Key names match `PI0Pytorch.state_dict()` directly;
  the one state_dict key not in the file, `paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`,
  is tied to `paligemma.lm_head.weight` (present). Loaded with `safetensors.torch.load_model(strict=True)`
  as openpi does (models/model.py:243-247); it reports no missing/unexpected keys.
- Precision (openpi default, policies/policy_config.py:53-55 + models_pytorch/gemma_pytorch.py:60-82):
  bfloat16 weights, except fp32 for SigLIP patch/position embeddings, all Gemma RMSNorm/adaRMS
  weights (`input_layernorm`, `post_attention_layernorm`, final `norm`), and the modules outside
  `paligemma_with_expert` (`action_in_proj`, `action_out_proj`, `time_mlp_in/out`).
  `torch.set_float32_matmul_precision("high")` (TF32) is set by `PI0Pytorch.__init__`.
- Prompt (`--inputs mstar`): ids taken as given, right-padded with id 0 / mask False to 200 tokens,
  as `PaligemmaTokenizer.tokenize` pads (models/tokenizer.py:34-37). All three image masks True.
- Noise: the case's (15,32) tensor, passed as `noise=`; 10 Euler steps from t=1.
- Variants: `eager` = `pytorch_compile_mode=None`; `compile` = `torch.compile(sample_actions,
  mode="default")`. openpi's own default is `"max-autotune"` (models/pi0_config.py:34).
