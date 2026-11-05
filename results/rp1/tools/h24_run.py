# Harness variant: Inductor's mm autotune limited to Triton candidates (no ATen/cuBLAS), process-wide.
import runpy, sys
import torch._inductor.config as c
c.max_autotune_gemm_backends = "TRITON"
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
