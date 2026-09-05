# Deployment and operations

The validated deployment is a CPU native backend with one leased session slot.
It binds to loopback by default, authenticates model/state endpoints, verifies
model/runtime hashes, and stores private session state separately from code.
The Compose and Kubernetes checks use synthetic inputs in isolated deployments.

## Container and monitoring

```bash
python scripts/fetch_model.py
docker build -t infinitecontext:local -f serving/Dockerfile .
python scripts/check_deployment.py --output .runs/compose-check
```

The multi-stage image builds the pinned patch, runs its rotary/metadata tests,
and retains only runtime artifacts and the Python session code. It uses a
non-root user. Compose sets a read-only root filesystem, drops capabilities,
bounds CPU/memory/PIDs, and persists session files in a dedicated volume.
The model mount is read-only. Build images are pinned by digest; distribution
package versions still resolve through the pinned distribution's repositories.
Build receipts capture the resulting executable and library hashes.

For an interactive deployment:

```bash
docker compose -f serving/compose.yaml -f infra/monitoring/compose.yaml up --build -d
printf '%s\n' '{"id":"record-1","text":"The station is ready.\n"}' \
  | docker compose -f serving/compose.yaml exec -T model \
      python3 -m streaming.cli --window 512
```

Use the CLI inside the container so its native snapshot directory and database
share the backend's persisted volume. `IC_IMAGE`, `IC_PORT`, and `IC_PROM_PORT`
override the local image/loopback ports. The backend API key is generated in the
state volume and is never placed in an image, configuration file, or command
line argument. Treat its state-management API as private administrative access.

Prometheus reads the key from the same volume and scrapes the runtime with
Bearer authentication. Its UI binds to loopback port 19090. The reference
monitoring container keeps one hour of metrics in temporary storage;
long-term metrics retention requires an operated storage configuration.
`infra/monitoring/grafana.json` is an importable dashboard using metric names
observed from this runtime. Dashboard rendering is not a validated UI result.

`check_deployment.py` creates a uniquely named Compose project, tests denial of
unauthenticated model access, commits an input, restarts the container, verifies
idempotent replay, and waits for a successful authenticated Prometheus scrape.
It removes only its own test containers and volumes. Existing deployments are
not used as test targets.

## Kubernetes

`serving/kubernetes.yaml` supplies a Namespace, headless Service, and one-replica
StatefulSet. Separate PVCs hold pinned model weights and session state. An init
container downloads/verifies weights. Startup/readiness/liveness probes and
resource limits are defined; service-account credentials are not mounted.

The manifest uses `infinitecontext:local` with `imagePullPolicy: Never` for local
kind validation. Build and load the image before applying it. A remote
installation needs an explicitly published image digest, storage class, network
policy, and the corresponding operational qualification.

```bash
mkdir -p .runtime
kind create cluster --name infinitecontext-validation \
  --kubeconfig .runtime/kubeconfig
kind load docker-image infinitecontext:local --name infinitecontext-validation
python scripts/check_kubernetes.py --kubeconfig .runtime/kubeconfig \
  --context kind-infinitecontext-validation --output .runs/kubernetes-check
kind delete cluster --name infinitecontext-validation
```

The evidence run used kind v0.33.0 and
`kindest/node:v1.36.4@sha256:099e049362a1526b2db71494e1947aae99bd16290d7c895f2b7ea312e3cbfaed`.
Use that `--image` for the recorded Kubernetes version. The checker refuses
contexts outside `kind-infinitecontext-*` and requires its namespace to be
absent. It tests PVC-backed replay after pod replacement, observes an intentionally
invalid image rollout, then rolls back and verifies the stored session again.
It removes its test namespace and PVCs.

The StatefulSet remains at one replica. Increasing replicas does not create a
shared session scheduler, and attaching multiple workers to one session's state
would violate the lease contract. HPA, multi-session admission, sticky routing,
GPU scheduling, node loss, multi-node storage, K3s deployment, and production
rollouts remain distinct qualification work. They remain in the full roadmap.

## GPU hosts and SLURM

`infra/provision.yaml` installs native build prerequisites on an existing Ubuntu
24.04 GPU host and checks its driver and NVIDIA Container Toolkit. Its Ansible
syntax is checked; host application and idempotence are not claimed. It does
not manage driver upgrades or replace a shared machine's container runtime.
That provisioning work needs validation against the actual target host.

`training/slurm/train.sbatch` provides the distributed supervisor launch. See
[TRAINING.md](TRAINING.md) for required environment variables. Shell syntax is
checked, while scheduling, NCCL networking, and multi-node execution require a
real allocation and remain unvalidated.

## vLLM boundary

The native `n_cache_shift` and slot snapshot protocol are llama.cpp-specific.
The custom MLA checkpoint currently has no vLLM model implementation or verified
checkpoint conversion. The native service and MLflow HTTP check therefore do
not establish vLLM support.

Qualification must cover a pinned model/runtime, supported GPU operators,
model conversion or registration, request limits, batching, streaming output,
cache/session ownership, cancellation, metrics, and quality under eviction.
Use a separate environment for vLLM's own PyTorch dependency set, as required by
[its installation guidance](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).
Current hardware admission at the framework level does not establish support
for every hybrid-model kernel.
