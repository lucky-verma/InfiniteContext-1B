# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This project is in the **planning phase** (Week 0 complete). No source code exists yet. All documentation describes the target architecture. The 6-phase implementation roadmap begins with Phase 1: Infrastructure Foundation.

## Memory Bank

Before starting any task, read all files in `memory-bank/` in this dependency order:

1. `projectbrief.md` — core goals, scope, success criteria
2. `productContext.md` — problems being solved, target performance
3. `systemPatterns.md` — architecture decisions, design patterns
4. `techContext.md` — tech stack, dependencies, hardware constraints
5. `activeContext.md` — current focus, recent changes, open questions
6. `progress.md` — what's done, what remains, phase completion %

Update `activeContext.md` and `progress.md` after implementing significant changes. When the user requests **update memory bank**, review all six files and update `.cursor/rules/` as needed.

## Planned Commands

These commands are defined in the README for when the corresponding code is implemented:

```bash
# Infrastructure provisioning
cd infra/ansible && ansible-playbook -i inventory.ini setup_gpu_node.yml

# Distributed training (SLURM)
sbatch training/slurm/train_sft.sbatch
sbatch training/slurm/train_dpo.sbatch

# Kernel benchmarking (target: 3.4x Triton speedup vs PyTorch)
python kernels/benchmark_mla.py

# Kubernetes deployment
kubectl apply -f serving/k8s/deployment.yaml
kubectl apply -f serving/k8s/service.yaml
kubectl port-forward svc/prometheus 9090
```

## Architecture

**InfiniteContext-1B** is a 1B-parameter LLM with 1M-token context window using DeepSeek-V3 Multi-Head Latent Attention (MLA).

### Three-Tier System

```
Training Pipeline:
  Ansible Provision → SLURM Scheduler → FSDP Training → DPO Alignment
                                              ↓
                    W&B / MLflow Experiment Tracking

Inference Pipeline:
  Model Registry → Triton Kernels → vLLM Serving → Kubernetes Cluster
                                          ↓
                    Prometheus / Grafana Monitoring & Autoscaling
```

### Planned Directory Layout

```
infra/ansible/         # GPU node provisioning playbooks
infra/slurm/           # SLURM cluster config
kernels/               # Custom Triton kernels (MLA, RoPE, RMSNorm) + benchmarks
training/src/          # MLA architecture (modeling_mla.py), DPO trainer
training/recipes/      # Training configs
training/slurm/        # SLURM batch scripts
training/evaluation/   # Needle-in-Haystack / LLM-as-Judge evaluation
serving/k8s/           # Kubernetes manifests (deployment, HPA)
serving/vllm_config/   # vLLM serving config
serving/monitoring/    # Prometheus/Grafana configs
```

### Key Technical Decisions

**Multi-Head Latent Attention (MLA):** K/V caches projected into a shared latent space achieving 93% compression (target: ~8.0 MB per 1k tokens vs ~128 MB for standard MHA). Latent dimension is 8–16x compressed relative to d_model.

**Decoupled RoPE:** Keys split into `k_rope` (rotated, full dimension) and `k_content` (compressed). Only `k_rope` stored at full size to preserve positional information without inflating cache.

**Triton Fused Kernels:** Single kernel performs decompress + softmax(QK^T) + V without materializing the full `(B, L, H, D)` tensor to HBM. Operations are memory-bound rather than compute-bound.

**FSDP over DDP:** Fully Sharded Data Parallel achieves 92% GPU utilization vs 65% for DDP, with better per-GPU memory usage across multi-node training.

**vLLM + Kubernetes:** MLA backend adapter for vLLM inference with Horizontal Pod Autoscaling on K3s.

**Base model strategy is unresolved:** MLA weight matrices are incompatible with standard pre-trained checkpoints. Options: DeepSeek checkpoints, distillation, or pre-training from scratch.

### Success Criteria

- 93%+ KV cache compression vs standard MHA
- 128k-token inference on RTX 2070 Super (8GB VRAM)
- 1M-token inference on A100-80GB
- 92%+ GPU utilization with FSDP
- 3.4x inference speedup with Triton kernels
- 90%+ passkey retrieval accuracy at 128k context (aspirational for 1B model)

## Technology Stack

- **ML:** PyTorch 2.4+, OpenAI Triton 3.0+, Torchtune, TRL
- **Training infra:** PyTorch FSDP, SLURM, NCCL, Gloo
- **Experiment tracking:** Weights & Biases, MLflow
- **Serving:** vLLM, Kubernetes (K3s), Docker
- **Monitoring:** Prometheus, Grafana
- **Provisioning:** Ansible, NVIDIA Container Toolkit
- **Hardware requirement:** NVIDIA GPU Compute Capability 7.0+, CUDA 11.8+/12.1+, Python 3.10+
