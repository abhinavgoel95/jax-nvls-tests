# Tests for NVLS on JAX

When adding `--xla_gpu_enable_nccl_symmetric_buffers=true`, you should be able to see NVLS working on multi-node NVLink.


## How to Reproduce

Container: `ghcr.io/nvidia/jax:jax-2025-09-07` (upstream JAX and XLA)

`bash run_test.sh` (may need to correct the path to the scripts)
