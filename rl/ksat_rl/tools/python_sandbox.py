"""Client for the KSAT Python sandbox service (rl/ksat_rl/tools/sandbox/).

    from ksat_rl.tools.python_sandbox import run_python, format_observation
    res = run_python("import sympy as sp\\nprint(sp.factor(x**2-1))")
    # {"stdout": ..., "stderr": ..., "status": "ok"|"error"|"timeout"|"memory"|"killed"|"sandbox_error",
    #  "truncated": bool, ...}
    tool_message_text = format_observation(res)   # <= 1000 chars, what the policy sees

The service URL comes from KSAT_SANDBOX_URL (default http://ksat-tool-sandbox:8080, resolvable from any
container attached to the internal network ksat-tool-sbx-net).  Code is never executed locally.
status == "sandbox_error" means an infrastructure failure (service down, bad response) and must not be
counted as the policy's mistake (ksat_tool_rl_v2/README.md §5): re-collect the affected rollout/group.
Stdlib only; `arun_python` uses aiohttp when available, else a worker thread.
"""
import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("KSAT_SANDBOX_URL", "http://ksat-tool-sandbox:8080")
DEFAULT_TIMEOUT = 10.0     # seconds of wall time inside the sandbox (server caps at 10)
MAX_CHARS = 1000           # observation budget (server caps stdout/stderr at 1000 chars each)
_HTTP_SLACK = 15.0         # HTTP timeout = exec timeout + slack (fork, queueing behind other calls)


def _infra_error(msg: str) -> dict:
    return {"stdout": "", "stderr": msg[:300], "status": "sandbox_error", "truncated": False}


def _payload(code: str, timeout: float, max_chars: int) -> bytes:
    return json.dumps({"code": code, "timeout": float(timeout), "max_chars": int(max_chars)}).encode()


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT, url: str | None = None, retries: int = 4,
               max_chars: int = MAX_CHARS, http_timeout: float | None = None) -> dict:
    """Execute `code` in a fresh sandboxed subprocess. Returns {"stdout","stderr","status","truncated",...}."""
    url = (url or DEFAULT_URL).rstrip("/") + "/run"
    data = _payload(code, timeout, max_chars)
    last = "unknown"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=http_timeout or timeout + _HTTP_SLACK) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:   # 4xx: our request is bad (e.g. code > 64 KB); do not retry
            if 400 <= e.code < 500:
                try:
                    msg = json.loads(e.read()).get("error", str(e))
                except Exception:  # noqa: BLE001
                    msg = str(e)
                return {"stdout": "", "stderr": f"Error: {msg}", "status": "error", "truncated": False}
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001  connection refused / reset / timeout
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(min(8.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
    return _infra_error(f"sandbox unreachable after {retries + 1} attempts: {last}")


async def arun_python(code: str, timeout: float = DEFAULT_TIMEOUT, url: str | None = None, retries: int = 4,
                      max_chars: int = MAX_CHARS, session=None, http_timeout: float | None = None) -> dict:
    """Async variant for rollout loops. Pass a shared aiohttp.ClientSession as `session` for high call rates.
    `http_timeout` (default timeout + 15 s) also covers queueing behind busy sandbox workers: under RL bursts set it
    well above timeout so a long queue is not misreported as an infrastructure failure."""
    try:
        import aiohttp
    except ImportError:
        return await asyncio.to_thread(run_python, code, timeout, url, retries, max_chars, http_timeout)
    url = (url or DEFAULT_URL).rstrip("/") + "/run"
    data = _payload(code, timeout, max_chars)
    own = session is None
    sess = session or aiohttp.ClientSession()
    last = "unknown"
    try:
        for attempt in range(retries + 1):
            try:
                async with sess.post(url, data=data, headers={"Content-Type": "application/json"},
                                     timeout=aiohttp.ClientTimeout(total=http_timeout or timeout + _HTTP_SLACK)) as r:
                    if 400 <= r.status < 500:
                        body = await r.json(content_type=None)
                        return {"stdout": "", "stderr": f"Error: {body.get('error')}", "status": "error",
                                "truncated": False}
                    if r.status == 200:
                        return await r.json(content_type=None)
                    last = f"HTTP {r.status}"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            if attempt < retries:
                await asyncio.sleep(min(8.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
        return _infra_error(f"sandbox unreachable after {retries + 1} attempts: {last}")
    finally:
        if own:
            await sess.close()


def health(url: str | None = None, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen((url or DEFAULT_URL).rstrip("/") + "/health", timeout=timeout) as r:
        return json.loads(r.read())


TRUNC_MARK = "\n...[output truncated]"


def format_observation(res: dict, max_chars: int = MAX_CHARS) -> str:
    """Tool-message text shown to the policy (<= max_chars). stdout first; on failure the stderr tail."""
    status = res.get("status")
    out = (res.get("stdout") or "").rstrip("\n")
    err = (res.get("stderr") or "").rstrip("\n")
    if status == "sandbox_error":
        return "[sandbox error: the code was not executed]"
    if status == "ok":
        text = out if out else "(no output; use print() to show results)"
        if err:  # warnings
            text = (text + "\n[stderr]\n" + err) if out else "[stderr]\n" + err
    else:
        text = (out + "\n" if out else "") + err if err else out
        if not text:
            text = f"[{status}]"
    if res.get("truncated") or len(text) > max_chars:
        if status == "ok" or not err:   # keep the head of normal output
            text = text[: max_chars - len(TRUNC_MARK)] + TRUNC_MARK
        else:                           # keep the error tail (exception type / message)
            keep = max_chars - len(TRUNC_MARK)
            head = out[: max(0, keep // 3)]
            tail = err[-(keep - len(head) - 1):]
            text = (head + TRUNC_MARK + "\n" + tail) if head else (TRUNC_MARK.lstrip("\n") + "\n" + tail)
            text = text[-max_chars:]
    return text[:max_chars]
