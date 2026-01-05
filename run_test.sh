RUN_NAME=${1:-"testing_nvls"}

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export CUDA_DEVICE_MAX_CONNECTIONS=16

export TF_CPP_VMODULE=nccl_communicator=10,collective_thunk=10
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=10
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING,ENV,INIT,REG

export XLA_FLAGS="--xla_gpu_enable_nccl_user_buffers=true --xla_gpu_all_gather_combine_threshold_bytes=13421772800 --xla_gpu_reduce_scatter_combine_threshold_bytes=13421772800 --xla_gpu_experimental_enable_nccl_symmetric_buffers=true"
mkdir -p /workspace

NSYS_OUTPUT_FILE=/path/to/profiles/${RUN_NAME}-hostid${SLURM_NODEID}-procid${SLURM_PROCID}
NSYS_CMD="nsys profile --cuda-graph-trace=node -s none -o /opt/haiku/profiles/${RUN_NAME}-hostid${SLURM_NODEID}-procid${SLURM_PROCID} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"

${NSYS_CMD} python /path/to/jax-nvls-tests/simple_nvls_test.py
