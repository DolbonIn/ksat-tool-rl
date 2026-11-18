#!/usr/bin/env python3
"""Python execution sandbox for KSAT tool-calling RL.  Runs UNTRUSTED model-generated code.

Run it only inside its container (see ../README.md, sandbox.sh).  Stdlib only + pre-imported sympy/numpy.

Isolation, outer -> inner
  container (sandbox.sh): internal-only docker network (no route out), read-only root FS, tmpfs /tmp,
      non-root uid 10999, --cap-drop ALL, no-new-privileges, pids/cpu/memory limits, no volumes, no secrets in env.
  per call (this file): a FRESH forked child process per request with
      - rlimits: address space (default 1 GiB), CPU seconds, file size, open files, no core dumps
      - seccomp filter: no sockets, no fork/vfork/clone/exec (no subprocesses or threads), no signals to other
        processes, no ptrace / process_vm_*, no namespace / keyring / bpf / io_uring syscalls
      - non-dumpable (other processes cannot read its /proc/<pid>/mem, environ, cwd)
      - empty environment, private random tmp dir as cwd, stdin=/dev/null, all inherited fds closed
      - wall-clock limit enforced by the parent with SIGKILL
  Output is captured to files (bounded by RLIMIT_FSIZE) and truncated to max_chars characters.

HTTP API (one request per connection, HTTP/1.0):
  GET  /health                      -> {"ok": true, "workers": N, ...}
  POST /run {"code": str, "timeout": float<=MAX_TIMEOUT, "max_chars": int<=MAX_CHARS}
       -> {"stdout", "stderr", "status", "truncated", "exit_code", "wall_ms", "stdout_bytes", "stderr_bytes"}
  status: ok | error (exception / SyntaxError / nonzero exit) | timeout | memory | killed (signal)
Like a notebook cell, if the last statement is a bare expression (not a print call) its repr() is printed.
State is NOT kept between calls.
"""
import ast
import builtins
import ctypes
import json
import linecache
import os
import resource
import select
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

# Pre-import the math stack in the worker so forked children start in ~ms.
import bisect  # noqa: F401
import cmath  # noqa: F401
import collections  # noqa: F401
import decimal  # noqa: F401
import fractions  # noqa: F401
import functools  # noqa: F401
import heapq  # noqa: F401
import itertools  # noqa: F401
import math  # noqa: F401
import numbers  # noqa: F401
import operator  # noqa: F401
import random  # noqa: F401
import re  # noqa: F401
import statistics  # noqa: F401
import string  # noqa: F401

import numpy  # noqa: F401
import sympy  # noqa: F401

VERSION = "ksat-sandbox-1"
PORT = int(os.environ.get("SBX_PORT", "8080"))
WORKERS = int(os.environ.get("SBX_WORKERS", "16"))
MAX_TIMEOUT = float(os.environ.get("SBX_MAX_TIMEOUT", "10"))
MEM_BYTES = int(os.environ.get("SBX_MEM_MB", "1024")) * 1024 * 1024
MAX_CHARS = int(os.environ.get("SBX_MAX_CHARS", "1000"))
MAX_CODE_BYTES = int(os.environ.get("SBX_MAX_CODE_BYTES", "65536"))
FSIZE_BYTES = int(os.environ.get("SBX_FSIZE_MB", "4")) * 1024 * 1024
TMP_ROOT = os.environ.get("SBX_TMP", "/tmp")
READ_CAP = 64 * 1024  # bytes of stdout/stderr read back before char truncation

libc = ctypes.CDLL(None, use_errno=True)
PR_SET_DUMPABLE, PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 4, 38, 22, 2


# ---------------------------------------------------------------- seccomp (x86_64 classic BPF)
class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


_LD_W_ABS, _JEQ, _JGE, _RET = 0x20, 0x15, 0x35, 0x06
_RET_ALLOW, _RET_ERRNO, _RET_KILL = 0x7FFF0000, 0x00050000, 0x80000000
_AUDIT_ARCH_X86_64 = 0xC000003E
_EPERM, _ENOSYS = 1, 38
_NR = dict(socket=41, connect=42, accept=43, bind=49, listen=50, socketpair=53, clone=56, fork=57, vfork=58,
           execve=59, kill=62, ptrace=101, setpgid=109, setsid=112, rt_sigqueueinfo=129, pivot_root=155,
           chroot=161, mount=165, umount2=166, tkill=200, tgkill=234, add_key=248, request_key=249, keyctl=250,
           unshare=272, accept4=288, rt_tgsigqueueinfo=297, perf_event_open=298, setns=308,
           process_vm_readv=310, process_vm_writev=311, kcmp=312, bpf=321, execveat=322, userfaultfd=323,
           pidfd_send_signal=424, io_uring_setup=425, io_uring_enter=426, io_uring_register=427,
           pidfd_open=434, clone3=435, pidfd_getfd=438)
_DENY_EPERM = [n for k, n in _NR.items() if k not in ("kill", "tgkill", "clone3")]


def _build_filter(own_pid: int):
    """Default allow; deny listed syscalls with EPERM; clone3 -> ENOSYS; kill/tgkill only to self."""
    prog, fix = [], []  # fix: (index, jt_label, jf_label)

    def ins(code, k=0, jt=None, jf=None):
        prog.append([code, 0, 0, k])
        if jt is not None or jf is not None:
            fix.append((len(prog) - 1, jt, jf))

    labels = {}
    ins(_LD_W_ABS, 4)                                   # arch
    ins(_JEQ, _AUDIT_ARCH_X86_64, None, "kill")
    ins(_LD_W_ABS, 0)                                   # syscall nr
    ins(_JGE, 0x40000000, "eperm", None)                # x32 ABI
    for n in _DENY_EPERM:
        ins(_JEQ, n, "eperm", None)
    ins(_JEQ, _NR["clone3"], "enosys", None)
    ins(_JEQ, _NR["kill"], "chk_target", None)
    ins(_JEQ, _NR["tgkill"], "chk_target", None)
    ins(_RET, _RET_ALLOW)
    labels["chk_target"] = len(prog)
    ins(_LD_W_ABS, 16)                                  # args[0] low 32 bits
    ins(_JEQ, own_pid, "allow", "eperm")
    labels["allow"] = len(prog); ins(_RET, _RET_ALLOW)
    labels["eperm"] = len(prog); ins(_RET, _RET_ERRNO | _EPERM)
    labels["enosys"] = len(prog); ins(_RET, _RET_ERRNO | _ENOSYS)
    labels["kill"] = len(prog); ins(_RET, _RET_KILL)
    for i, jt, jf in fix:
        if jt is not None:
            prog[i][1] = labels[jt] - i - 1
        if jf is not None:
            prog[i][2] = labels[jf] - i - 1
        assert 0 <= prog[i][1] < 256 and 0 <= prog[i][2] < 256
    arr = (_SockFilter * len(prog))(*[_SockFilter(c, jt, jf, k) for c, jt, jf, k in prog])
    return arr, _SockFprog(len(prog), arr)


def _install_seccomp():
    arr, fprog = _build_filter(os.getpid())
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "seccomp install failed")
    return arr  # keep alive


# ---------------------------------------------------------------- child
EXIT_OK, EXIT_ERROR, EXIT_MEMORY, EXIT_SETUP = 0, 1, 3, 70


def _child(code: str, tmpdir: str, out_path: str, err_path: str, timeout: float):
    """Runs in the forked child; never returns."""
    try:
        os.setsid()
        try:
            with open("/proc/self/oom_score_adj", "w") as f:  # be the first victim of a cgroup OOM
                f.write("1000")
        except OSError:
            pass
        fo = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        fe = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        fn = os.open("/dev/null", os.O_RDONLY)
        os.dup2(fn, 0); os.dup2(fo, 1); os.dup2(fe, 2)
        os.closerange(3, 65536)  # listening socket, client connection, everything else
        os.chdir(tmpdir)
        os.environ.clear()
        os.environ.update({"HOME": tmpdir, "TMPDIR": tmpdir, "LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"})
        tempfile.tempdir = tmpdir
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # oversized writes -> OSError(EFBIG) instead of death
        cpu = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_BYTES, FSIZE_BYTES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
        sys.stdin = open(0, "r", closefd=False)
        sys.stdout = open(1, "w", encoding="utf-8", errors="replace", closefd=False, buffering=1)
        sys.stderr = open(2, "w", encoding="utf-8", errors="replace", closefd=False, buffering=1)
        sys.argv = ["python"]
        _keep = _install_seccomp()  # noqa: F841  -- last step before untrusted code
    except BaseException:
        try:
            os.write(2, traceback.format_exc().encode()[-2000:])
        finally:
            os._exit(EXIT_SETUP)

    rc = EXIT_OK
    try:
        linecache.cache["<python>"] = (len(code), None, code.splitlines(True), "<python>")
        tree = ast.parse(code, filename="<python>", mode="exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            v = tree.body[-1].value
            if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "print"):
                last = ast.Expression(tree.body.pop().value)
        g = {"__name__": "__main__", "__builtins__": builtins}
        exec(compile(tree, "<python>", "exec"), g)
        if last is not None:
            val = eval(compile(last, "<python>", "eval"), g)
            if val is not None:
                print(repr(val))
    except SystemExit as e:
        rc = EXIT_OK if e.code in (None, 0) else EXIT_ERROR
        if rc:
            print(f"SystemExit: {e.code}", file=sys.stderr)
    except MemoryError:
        rc = EXIT_MEMORY
        try:
            sys.stderr.write("MemoryError: memory limit exceeded\n")
        except BaseException:
            pass
    except BaseException as e:  # noqa: BLE001
        rc = EXIT_ERROR
        try:
            tb = e.__traceback__
            while tb is not None and tb.tb_frame.f_code.co_filename != "<python>":
                tb = tb.tb_next  # hide sandbox frames
            if isinstance(e, SyntaxError):
                tb = None
            traceback.print_exception(type(e), e, tb, file=sys.stderr)
        except BaseException:
            pass
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except BaseException:
        pass
    os._exit(rc)


# ---------------------------------------------------------------- parent side
def _read_head(path, n):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            return f.read(n), size
    except OSError:
        return b"", 0


def _read_tail(path, n):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > n:
                f.seek(size - n)
            return f.read(n), size
    except OSError:
        return b"", 0


def execute(code: str, timeout: float, max_chars: int) -> dict:
    t0 = time.monotonic()
    tmpdir = tempfile.mkdtemp(prefix="run-", dir=TMP_ROOT)
    out_path, err_path = os.path.join(tmpdir, ".stdout"), os.path.join(tmpdir, ".stderr")
    sys.stdout.flush(); sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        _child(code, tmpdir, out_path, err_path, timeout)
        os._exit(EXIT_SETUP)  # unreachable
    timed_out = False
    try:
        pfd = os.pidfd_open(pid)
        try:
            r, _, _ = select.select([pfd], [], [], timeout)
        finally:
            os.close(pfd)
        if not r:
            timed_out = True
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _, st = os.waitpid(pid, 0)
    wall_ms = int((time.monotonic() - t0) * 1000)
    out_b, out_size = _read_head(out_path, READ_CAP)
    err_b, err_size = _read_tail(err_path, READ_CAP)
    shutil.rmtree(tmpdir, ignore_errors=True)

    stdout = out_b.decode("utf-8", errors="replace")
    stderr = err_b.decode("utf-8", errors="replace")
    truncated = out_size > len(out_b) or err_size > len(err_b) or len(stdout) > max_chars or len(stderr) > max_chars
    stdout = stdout[:max_chars]                       # head of stdout
    stderr = stderr[-max_chars:] if stderr else ""    # tail of stderr (traceback end)

    if os.WIFEXITED(st):
        rc = os.WEXITSTATUS(st)
        status = {EXIT_OK: "ok", EXIT_MEMORY: "memory"}.get(rc, "error")
        if rc == EXIT_SETUP:
            status = "sandbox_error"
    else:
        rc = -os.WTERMSIG(st)
        sig = os.WTERMSIG(st)
        if timed_out or sig == signal.SIGXCPU:
            status = "timeout"
            stderr = (stderr + "\n" if stderr else "") + f"TimeoutError: execution exceeded the {timeout:g} s limit"
            stderr = stderr[-max_chars:]
        else:
            status = "killed"
            stderr = (stderr + "\n" if stderr else "") + f"Killed by signal {signal.Signals(sig).name}"
            stderr = stderr[-max_chars:]
    return {"stdout": stdout, "stderr": stderr, "status": status, "truncated": bool(truncated),
            "exit_code": rc, "wall_ms": wall_ms, "stdout_bytes": out_size, "stderr_bytes": err_size}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = VERSION

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "version": VERSION, "workers": WORKERS, "max_timeout": MAX_TIMEOUT,
                             "mem_mb": MEM_BYTES >> 20, "max_chars": MAX_CHARS})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_CODE_BYTES * 2 + 1024:
                return self._send(413, {"error": "request too large"})
            req = json.loads(self.rfile.read(n))
            code = req["code"]
            if not isinstance(code, str) or len(code.encode()) > MAX_CODE_BYTES:
                return self._send(413, {"error": "code missing or too large"})
            timeout = min(max(float(req.get("timeout", MAX_TIMEOUT)), 0.1), MAX_TIMEOUT)
            max_chars = min(max(int(req.get("max_chars", MAX_CHARS)), 1), MAX_CHARS)
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"bad request: {type(e).__name__}"})
        try:
            res = execute(code, timeout, max_chars)
        except Exception as e:  # noqa: BLE001  infra failure, not the model's fault
            res = {"stdout": "", "stderr": f"sandbox failure: {type(e).__name__}: {e}"[:300], "status": "sandbox_error",
                   "truncated": False, "exit_code": None, "wall_ms": 0}
        self._send(200, res)


class _Server(HTTPServer):
    def __init__(self, sock):
        super().__init__(("0.0.0.0", PORT), Handler, bind_and_activate=False)
        self.socket = sock

    def get_request(self):
        conn, addr = self.socket.accept()
        conn.settimeout(10)  # slow clients cannot pin a worker
        return conn, addr


def _worker(sock):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    _Server(sock).serve_forever(poll_interval=3600)


def _janitor(max_age=60):
    """Remove anything in /tmp older than max_age s (a call lives <= MAX_TIMEOUT s; leftovers are stale)."""
    now = time.time()
    try:
        for e in os.scandir(TMP_ROOT):
            try:
                if now - e.stat(follow_symlinks=False).st_mtime > max_age:
                    if e.is_dir(follow_symlinks=False):
                        shutil.rmtree(e.path, ignore_errors=True)
                    else:
                        os.unlink(e.path)
            except OSError:
                pass
    except OSError:
        pass


def main():
    libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    # self-test the seccomp program builder once at start-up
    _build_filter(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", PORT))
    sock.listen(1024)
    kids = set()

    def spawn():
        pid = os.fork()
        if pid == 0:
            try:
                _worker(sock)
            finally:
                os._exit(0)
        kids.add(pid)

    def stop(*_):
        for k in list(kids):
            try:
                os.kill(k, signal.SIGTERM)
            except ProcessLookupError:
                pass
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for _ in range(WORKERS):
        spawn()
    print(json.dumps({"event": "start", "version": VERSION, "port": PORT, "workers": WORKERS,
                      "max_timeout": MAX_TIMEOUT, "mem_mb": MEM_BYTES >> 20, "max_chars": MAX_CHARS}), flush=True)
    last_jan = 0.0
    while True:
        try:
            pid, st = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            pid = 0
        if pid and pid in kids:
            kids.discard(pid)
            print(json.dumps({"event": "worker_died", "pid": pid, "status": st}), flush=True)
            spawn()
            continue
        if time.time() - last_jan > 10:
            _janitor()
            last_jan = time.time()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
