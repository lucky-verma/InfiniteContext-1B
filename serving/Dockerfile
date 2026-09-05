FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3 ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY scripts/build_runtime.py scripts/build_runtime.py
COPY serving/runtime.json serving/runtime.json
COPY patches/llama.cpp patches/llama.cpp
RUN python3 scripts/build_runtime.py --portable --jobs 2
RUN python3 -c "import json, shutil; from pathlib import Path; \
    receipt = json.loads(Path('.runtime/cpu.json').read_text()); \
    shutil.copytree(Path(receipt['binary']).parent, '/artifacts/bin'); \
    receipt['binary'] = '/app/.runtime/bin/llama-server'; \
    Path('/artifacts/cpu.json').write_text(json.dumps(receipt, indent=2))"

FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 ca-certificates libcurl4t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home app
WORKDIR /app
COPY --from=build /artifacts .runtime
COPY scripts/fetch_model.py scripts/fetch_model.py
COPY serving/run.py serving/model.json serving/runtime.json serving/
COPY streaming streaming/
RUN mkdir -p .models .sessions && chown app:app .sessions
USER 10001:10001
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 18081
ENTRYPOINT ["python3", "-m", "serving.run"]
