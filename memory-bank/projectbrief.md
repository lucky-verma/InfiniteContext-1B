# Project Brief: InfiniteContext-1B

## Project Overview

**InfiniteContext-1B** is a production-grade Large Language Model (LLM) system implementing the **DeepSeek-V3 Multi-Head Latent Attention (MLA)** architecture. This project serves as a **Reference Architecture** for building modern MLSys pipelines, demonstrating the complete lifecycle of an LLM from training to production deployment.

## Core Goals

1. **Implement MLA Architecture**: Build a 1B parameter model with Multi-Head Latent Attention supporting 1 million-token context windows through Key-Value cache compression
2. **Distributed Training**: Scale training across multi-GPU nodes using PyTorch FSDP on SLURM clusters
3. **Kernel Optimization**: Develop custom OpenAI Triton kernels for accelerated decoding, achieving memory-bound operations
4. **Model Alignment**: Implement post-training DPO (Direct Preference Optimization) for long-context retrieval accuracy
5. **Production Serving**: Deploy high-availability inference on Kubernetes (K3s) with vLLM, optimized for consumer hardware

## Project Scope

### In Scope

- **Infrastructure**: Auto-provisioning GPU nodes with Ansible
- **Training**: Distributed FSDP training on SLURM clusters
- **Optimization**: Custom Triton kernels for MLA decoding (RoPE, RMSNorm, fused attention)
- **Alignment**: DPO training framework using TRL library
- **Serving**: Kubernetes deployment with vLLM, Prometheus monitoring, and Grafana dashboards
- **Evaluation**: Automated "LLM-as-a-Judge" pipeline using Prometheus-Eval
- **Experiment Tracking**: W&B and MLflow integration

### Out of Scope

- Training from scratch (focus on fine-tuning and alignment)
- Multi-modal capabilities
- Real-time streaming inference (batch processing focus)
- Cloud-specific managed services (self-hosted infrastructure)

## Success Criteria

1. **Memory Efficiency**: Achieve 93%+ KV cache compression compared to standard MHA
2. **Context Length**: Support 128k tokens on consumer hardware (RTX 2070 Super), 1M tokens on A100-80GB
3. **Training Performance**: Achieve 92%+ GPU utilization with FSDP on multi-node clusters
4. **Inference Speed**: 3.4x speedup with Triton kernels vs PyTorch baseline
5. **Retrieval Accuracy**: 90%+ passkey retrieval accuracy at 128k context, 85%+ at 1M context
6. **Production Readiness**: Automated deployment pipeline with monitoring and autoscaling

## Key Constraints

- **Hardware**: Development on consumer GPUs (8GB VRAM), production validation on cloud A100-80GB
- **Architecture**: Must implement DeepSeek-V3 MLA architecture faithfully
- **Deployment**: Self-hosted infrastructure (no managed cloud services)
- **Transparency**: All benchmarks and results must be reproducible and documented

## Project Phases

1. **Week 0: Planning & Architecture** (✅ Completed) - Project structure, documentation, roadmap
2. **Phase 1: Foundation** (🔵 Not Started) - Infrastructure, MLOps, DevOps automation
3. **Phase 2: Training Pipeline & MLOps** (⬜ Planned) - Experiment tracking, distributed training setup
4. **Phase 3: Core Implementation** (⬜ Planned) - MLA architecture, basic Triton kernels, FSDP training
5. **Phase 4: Kernel Optimization** (⬜ Planned) - Flash-Decoding kernels, advanced optimization
6. **Phase 5: Alignment & Evaluation** (⬜ Planned) - DPO framework, evaluation pipeline
7. **Phase 6: Production & Scaling** (⬜ Planned) - vLLM integration, 1M validation, production hardening

