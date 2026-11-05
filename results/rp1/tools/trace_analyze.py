"""Per-request hop times from RP1_TRACE_DIR traces of a c=1 run (requests sequential).

Events (monotonic clock, shared across processes):
  prep_Pi05ViTEncoderSubmodule prefill   worker: ViT step prepare
  prep_Pi05LLMSubmodule prefill          worker: LLM prefill prepare
  switch prefill                         conductor: prefill walk done -> action_gen args
  prep_Pi05LLMSubmodule action_gen       worker: action step prepare
  switch action_gen                      conductor: action_gen done
usage: python trace_analyze.py <dir> [skip_first_n]
"""
import glob
import statistics as st
import sys

d = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ev = []
for p in glob.glob(f"{d}/trace_*.txt"):
    for line in open(p):
        name, walk, t = line.split()
        ev.append((float(t), f"{name}:{walk}"))
ev.sort()
seq = ["prep_Pi05ViTEncoderSubmodule:prefill", "prep_Pi05LLMSubmodule:prefill", "switch:prefill",
       "prep_Pi05LLMSubmodule:action_gen", "switch:action_gen"]
reqs, cur = [], []
for t, k in ev:
    if k == seq[0]:
        cur = [(k, t)]
    elif cur and len(cur) < len(seq) and k == seq[len(cur)]:
        cur.append((k, t))
        if len(cur) == len(seq):
            reqs.append([x for _, x in cur])
            cur = []
reqs = reqs[skip:]
labels = ["vit prep -> llm prefill prep", "llm prefill prep -> conductor switch",
          "conductor switch -> action prep", "action prep -> conductor done"]
print(f"{len(reqs)} complete requests")
for i, lab in enumerate(labels):
    v = [(r[i + 1] - r[i]) * 1e3 for r in reqs]
    print(f"{lab:40s} p50 {st.median(v):6.2f} ms  mean {st.mean(v):6.2f}  p10 {sorted(v)[len(v) // 10]:6.2f}")

# API-side points, joined to the model hooks on the same clock
api = []
for p in glob.glob(f"{d}/apitrace_*.txt"):
    for line in open(p):
        n, t = line.split()
        api.append((float(t), n))
api.sort()
if api:
    seq2 = ["form_start", "form_end", "prep_start", "media_end", "prep_end"]
    rows, cur = [], []
    for t, n in api:
        if n == seq2[0]:
            cur = [t]
        elif cur and len(cur) < len(seq2) and n == seq2[len(cur)]:
            cur.append(t)
            if len(cur) == len(seq2):
                rows.append(cur)
                cur = []
    rows = rows[skip:]
    labs = ["form parse (multipart)", "form end -> prepare start (route, file writes, queue)", "prepare start -> media loaded", "media loaded -> prepare end"]
    print(f"{len(rows)} API rows")
    for i, lab in enumerate(labs):
        v = [(r[i + 1] - r[i]) * 1e3 for r in rows]
        print(f"{lab:55s} p50 {st.median(v):6.2f} ms")
    firsts = [t for t, k in ev if k == seq[0]]
    gaps = []
    for r in rows:
        later = [t for t in firsts if t > r[4]]
        if later:
            gaps.append((later[0] - r[4]) * 1e3)
    print(f"{'prepare end -> ViT prep':55s} p50 {st.median(gaps):6.2f} ms")
