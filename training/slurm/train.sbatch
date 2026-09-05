#!/usr/bin/env bash
#SBATCH --job-name=infinitecontext-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
: "${IC_REPO:?Set IC_REPO to the checkout on shared storage}"
: "${IC_DATA:?Set IC_DATA to the versioned uint32 training corpus}"
: "${IC_OUTPUT:?Set IC_OUTPUT to a fresh run directory}"
: "${IC_GPUS_PER_NODE:?Set IC_GPUS_PER_NODE to the allocated GPU count}"
: "${IC_CONFIG:?Set IC_CONFIG to the reviewed model configuration}"
: "${IC_STEPS:?Set IC_STEPS to the authorized absolute step budget}"
: "${IC_RDZV_PORT:?Set IC_RDZV_PORT to an available shared rendezvous port}"
IC_MASTER=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export IC_MASTER
export OMP_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export IC_STRATEGY=${IC_STRATEGY:-ddp}
cd "${IC_REPO}"
# One torchrun supervisor per node; SLURM's node ID becomes torchrun's node rank.
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
  exec .venv/bin/torchrun --nnodes="${SLURM_NNODES}" \
    --nproc-per-node="${IC_GPUS_PER_NODE}" --node-rank="${SLURM_NODEID}" \
    --rdzv-id="${SLURM_JOB_ID}" --rdzv-backend=c10d \
    --rdzv-endpoint="${IC_MASTER}:${IC_RDZV_PORT}" \
    -m training.run --config="${IC_CONFIG}" --data="${IC_DATA}" \
    --output="${IC_OUTPUT}" --steps="${IC_STEPS}" \
    --device=cuda --strategy="${IC_STRATEGY}" "$@"
' bash "$@"
