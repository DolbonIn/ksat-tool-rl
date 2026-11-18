# KSAT RL image: SGLang nightly (Nanbeige4.2 support merged in sglang#32151, B300/sm_103, FA4) + veRL main.
FROM lmsysorg/sglang:nightly-dev-cu13-20261030-302b78cc
ARG VERL_COMMIT=b0871fd89d
RUN pip install --no-cache-dir "ray[default]>=2.41.0" hydra-core codetiming dill pylatexenc pybind11 \
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata tensorboard math-verify mathruler cachetools nvtx \
        msgspec pyzmq pandas "pyarrow>=19.0.0" sympy "TransferQueue>=0.1.10" fastapi uvicorn pytest-asyncio
RUN git clone https://github.com/verl-project/verl.git /opt/verl && cd /opt/verl && git checkout ${VERL_COMMIT} \
 && pip install --no-cache-dir --no-deps -e . \
 && python -c "import verl, ray, tensordict; print('verl', verl.__version__, 'ray', ray.__version__, 'tensordict', tensordict.__version__)"
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
# veRL <-> SGLang-nightly compatibility (weight-update sessions); see patches/patch_verl_sglang_session.py
COPY patches /opt/ksat-patches
RUN python /opt/ksat-patches/patch_verl_sglang_session.py /opt/verl
# Nanbeige tool-calling RL: nanbeige tool parser + Continuous Token builder (patches/patch_verl_nanbeige_tools.py).
# Optional: train_grpo_tool.sh applies the same idempotent patch at container start (v4 works without a rebuild).
RUN python /opt/ksat-patches/patch_verl_nanbeige_tools.py /opt/verl
