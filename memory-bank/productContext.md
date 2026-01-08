# Product Context: InfiniteContext-1B

## Why This Project Exists

Large Language Models face a fundamental challenge: **context window limitations**. Standard attention mechanisms require storing full Key-Value caches for every token, making long-context inference prohibitively expensive in memory. For a 1B model at 1M context, standard MHA would require hundreds of GBs of VRAM.

**InfiniteContext-1B** solves this by implementing the **DeepSeek-V3 Multi-Head Latent Attention (MLA)** architecture, which compresses KV caches by up to 93% while maintaining retrieval accuracy.

## Problems Solved

### 1. Memory Efficiency for Long Context

**Problem**: Standard Multi-Head Attention stores a $d$-dimensional vector for every head, for every token:
$$\text{Cache Size} = B \times L \times H \times d_{head} \times 2 \text{ (bytes)}$$

**Solution**: MLA projects keys and values into a shared **Latent Vector** ($c_{KV}$) of much smaller dimension ($d_{model} \rightarrow d_{latent}$), achieving massive compression without sacrificing attention quality.

### 2. Consumer Hardware Deployment

**Problem**: Production LLM inference typically requires expensive A100 GPUs, limiting accessibility.

**Solution**: MLA compression enables 128k-token inference on consumer hardware (RTX 2070 Super with 8GB VRAM), making long-context LLMs accessible for cost-effective deployment.

### 3. Long-Context Retrieval Accuracy

**Problem**: Standard models struggle with accurate information retrieval in long contexts, leading to hallucinations.

**Solution**: DPO alignment on "Needle-in-Haystack" synthetic data trains the model to prioritize accurate retrieval over hallucinated responses.

### 4. Production-Grade MLSys Pipeline

**Problem**: Most research projects lack production deployment infrastructure and best practices.

**Solution**: Complete end-to-end pipeline from distributed training (SLURM) to production serving (Kubernetes) with monitoring, autoscaling, and CI/CD.

## How It Should Work

### Training Workflow

1. **Infrastructure Provisioning**: Ansible playbooks automatically configure GPU nodes with drivers, Docker, and SLURM
2. **Supervised Fine-Tuning (SFT)**: Train on "Needle-in-Haystack" synthetic data to learn long-context patterns
3. **DPO Alignment**: Fine-tune using Direct Preference Optimization to improve retrieval accuracy
4. **Experiment Tracking**: All runs logged to W&B and MLflow for reproducibility

### Inference Workflow

1. **Model Deployment**: Kubernetes orchestrates vLLM inference pods with horizontal autoscaling
2. **Request Handling**: vLLM serves requests with optimized MLA attention kernels
3. **Monitoring**: Prometheus collects GPU utilization, latency, and throughput metrics
4. **Evaluation**: Automated "LLM-as-a-Judge" pipeline validates model performance

### User Experience Goals

- **Developers**: Easy-to-use training scripts and deployment manifests
- **ML Engineers**: Transparent benchmarks and reproducible results
- **DevOps**: Automated infrastructure provisioning and monitoring
- **Researchers**: Reference implementation of MLA architecture with full documentation

## Target Performance

- **Memory**: ~8.0 MB KV cache per 1k tokens (93.7% savings vs Llama-2)
- **Context**: 128k tokens on RTX 2070 Super, 1M tokens on A100-80GB
- **Accuracy**: 90%+ passkey retrieval at 128k, 85%+ at 1M context
- **Speed**: 3.4x inference speedup with Triton kernels
- **Training**: 92%+ GPU utilization with FSDP on multi-node clusters

