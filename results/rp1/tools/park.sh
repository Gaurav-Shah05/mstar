#!/usr/bin/env bash
# Copy what must survive the maintenance window into perf/rp1-results and commit. Adds a commit per refresh rather
# than amending, so the branch only moves forward and each push lands.
W=/workspace/wt-rp1-results; D=$W/results/rp1; R=/workspace/results/rp1-pi05-tools
mkdir -p $D/tools/pyshim $D/ref
cp /workspace/results/rp1-pi05.md $D/; cp /workspace/results/rp1-pi05-proposals.md $D/ 2>/dev/null; cp /workspace/results.jsonl $D/
cp $R/*.py $R/*.sh $R/*.patch $R/*.yaml $R/*.txt $D/tools/ 2>/dev/null
cp $R/pyshim/sitecustomize.py $R/pyshim/rp1_*.py $D/tools/pyshim/
cp /workspace/openpi-ref/ref_A.npz /workspace/openpi-ref/ref_B.npz /workspace/openpi-ref/README.md $D/ref/ 2>/dev/null
/workspace/venv/bin/python - <<'PY'
import numpy as np
np.savez_compressed("/workspace/wt-rp1-results/results/rp1/ref/ref_A_openpi_inputs_actions.npz",
                    actions=np.load("/workspace/openpi-ref/ref_A_openpi_inputs.npz")["actions"])
PY
cd $W; git add results/rp1; git diff --cached --quiet && { echo "no change"; exit 0; }
git commit -q -m "results: rp1 state, $(date -u +%FT%TZ)"; git log --oneline -1 | cut -c1-70
