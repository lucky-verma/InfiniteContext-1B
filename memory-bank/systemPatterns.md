# System Patterns: InfiniteContext-1B

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Ansible │→ │  SLURM   │→ │   FSDP   │→ │   DPO    │  │
│  │ Provision│  │ Scheduler │  │ Training │  │ Alignment│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│       ↓              ↓              ↓              ↓        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         W&B / MLflow Experiment Tracking              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Inference Pipeline                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Model   │→ │  Triton  │→ │   vLLM   │→ │ Kubernetes│  │
│  │ Registry │  │  Kernels │  │  Serving │  │  Cluster  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│       ↓              ↓              ↓              ↓        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │    Prometheus / Grafana Monitoring & Autoscaling      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Technical Decisions

### 1. Multi-Head Latent Attention (MLA)

**Pattern**: Compression through latent space projection

**Architecture**:
- Keys and values are projected into shared latent vectors: $c_{KV} = \text{Project}(K, V)$
- Compression factor: $R = d_{model} / d_{latent}$ (typically 8-16x)
- Decompression on-the-fly during attention computation

**Implementation**:
- Custom `MLAAttention` layer replacing standard `MultiHeadAttention`
- Latent dimension configurable (default: $d_{model} / 8$)

### 2. Decoupled RoPE Strategy

**Pattern**: Split rotation for compressed and uncompressed components

**Architecture**:
- Keys split into: $k_{rope}$ (rotated) and $k_{content}$ (compressed, no rotation)
- Only $k_{rope}$ components stored in full dimension
- Content vectors remain in latent space

**Implementation**:
- `DecoupledRotaryEmbedding` layer handles split rotations
- Preserves positional information without inflating cache size

### 3. Triton Fused Kernels

**Pattern**: Memory-bound operations via on-the-fly decompression

**Architecture**:
- Fused kernel: `decompress + attention` in single operation
- Compressed latents loaded from HBM → SRAM
- Decompression and attention computed in shared memory
- Never materializes full $(B, L, H, D)$ matrices to HBM

**Implementation**:
- `triton_mla_flash_decode.py`: Fused MLA decoding kernel
- `triton_rope.py`: Optimized RoPE kernel
- `triton_rmsnorm.py`: Fused RMSNorm kernel

### 4. Distributed Training with FSDP

**Pattern**: Fully Sharded Data Parallel for multi-node scaling

**Architecture**:
- Model parameters sharded across all GPUs
- Gradient synchronization via NCCL backend
- SLURM orchestrates multi-node job scheduling

**Implementation**:
- PyTorch FSDP with `FullyShardedDataParallel`
- SLURM batch scripts for job submission
- Automatic checkpointing and resumption

### 5. Kubernetes Deployment Pattern

**Pattern**: Microservices architecture with autoscaling

**Architecture**:
- vLLM inference pods with horizontal autoscaling
- Prometheus metrics collection
- Grafana dashboards for visualization
- Service mesh for load balancing

**Implementation**:
- K3s lightweight Kubernetes distribution
- HPA (Horizontal Pod Autoscaler) based on GPU utilization
- ConfigMaps for model configuration
- Persistent volumes for model storage

## Component Relationships

### Training Components

```
modeling_mla.py (MLA Architecture)
    ↓
dpo_trainer.py (DPO Alignment)
    ↓
slurm/train_*.sbatch (Job Scripts)
    ↓
W&B / MLflow (Tracking)
```

### Inference Components

```
triton_mla.py (Optimized Kernels)
    ↓
vllm_config/ (Serving Config)
    ↓
k8s/deployment.yaml (Kubernetes Manifests)
    ↓
monitoring/ (Prometheus/Grafana)
```

### Infrastructure Components

```
ansible/ (Provisioning)
    ↓
slurm/ (Cluster Config)
    ↓
k8s/ (Orchestration)
```

## Design Patterns in Use

1. **Factory Pattern**: Model architecture factory for different MLA configurations
2. **Strategy Pattern**: Pluggable attention backends (PyTorch vs Triton)
3. **Observer Pattern**: Experiment tracking hooks in training loop
4. **Repository Pattern**: MLflow model registry for versioning
5. **Adapter Pattern**: vLLM backend adapter for MLA architecture

## Data Flow

### Training Data Flow

```
Synthetic Data (Needle-in-Haystack)
    ↓
DataLoader (Batched)
    ↓
FSDP Model (Distributed)
    ↓
Loss Computation
    ↓
Gradient Sync
    ↓
Optimizer Step
    ↓
Checkpointing
```

### Inference Data Flow

```
HTTP Request
    ↓
Kubernetes Service
    ↓
vLLM Pod
    ↓
Triton Kernels (MLA Attention)
    ↓
Model Forward Pass
    ↓
Response Generation
    ↓
Prometheus Metrics
```

