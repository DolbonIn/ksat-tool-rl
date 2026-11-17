FROM lmsysorg/sglang:v0.5.18-cu130
# Nanbeige fork of SGLang (branch nbg42) — required for Nanbeige4.2 (looped transformer)
COPY sglang /opt/sglang-nbg42
RUN cd /opt/sglang-nbg42/python && SGLANG_BUILD_RUST_EXTS=none pip install --no-cache-dir -e . \
 && python -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
RUN pip install --no-cache-dir --no-deps flashinfer-python==0.6.17 && python -c "import sglang.srt.entrypoints.engine, flashinfer; print('flashinfer', flashinfer.__version__)"
WORKDIR /workspace
