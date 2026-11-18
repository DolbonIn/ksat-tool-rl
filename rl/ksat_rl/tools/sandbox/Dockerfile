# Python sandbox for KSAT tool-calling RL (runs untrusted model code; see ../README.md).
FROM python:3.12-slim
RUN pip install --no-cache-dir sympy==1.13.3 numpy==2.1.3 \
 && useradd --uid 10999 --no-create-home --shell /usr/sbin/nologin sbx \
 && python3 -c "import sympy, numpy, fractions; print(sympy.__version__, numpy.__version__)"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    SBX_PORT=8080 SBX_WORKERS=16 SBX_MAX_TIMEOUT=10 SBX_MEM_MB=1024 SBX_MAX_CHARS=1000
COPY server.py /srv/server.py
USER 10999:10999
WORKDIR /tmp
EXPOSE 8080
CMD ["python3", "-I", "/srv/server.py"]
