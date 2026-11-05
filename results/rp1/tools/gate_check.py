"""Gate: M* actions vs openpi ref_A within openpi's own spread (max |ref_A - ref_B|).
usage: python gate_check.py <mstar gate.npz> [<other mstar gate.npz>]"""
import sys

import numpy as np

import os
A = np.load(os.environ.get("RP1_REF", "/workspace/openpi-ref/ref_A_openpi_inputs.npz"))["actions"]
B0 = np.load("/workspace/openpi-ref/ref_A.npz")["actions"]
B = np.load("/workspace/openpi-ref/ref_B.npz")["actions"]
tol = float(np.abs(B0 - B).max())
m = np.load(sys.argv[1])["actions"].reshape(A.shape)
d = np.abs(m - A)
ok = d.max() <= tol
print(f"GATE {'PASS' if ok else 'FAIL'}: max|mstar-openpi|={d.max():.5f} mean={d.mean():.6f} tol(openpi A-B)={tol:.5f}")
print("per case max:", d.reshape(len(A), -1).max(1).round(4).tolist())
if len(sys.argv) > 2:
    o = np.load(sys.argv[2])["actions"].reshape(A.shape)
    print(f"vs other mstar run: max={np.abs(m - o).max():.5f}")
sys.exit(0 if ok else 1)
