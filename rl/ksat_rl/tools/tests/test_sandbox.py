"""Security + throughput tests for the Python sandbox service.

Run from a client container attached to the sandbox network (never on the host python):
  rl/ksat_rl/tools/tests/run_sandbox_tests.sh
Exits non-zero if any check fails.  Prints a JSON summary (incl. calls/s) on the last line.
"""
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from ksat_rl.tools.python_sandbox import format_observation, health, run_python  # noqa: E402

RESULTS = []


def check(name, cond, res=None, extra=""):
    RESULTS.append((name, bool(cond)))
    short = "" if res is None else json.dumps({k: res.get(k) for k in ("status", "truncated", "wall_ms")},
                                              ensure_ascii=False)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {short} {extra}".rstrip(), flush=True)
    if not cond and res is not None:
        print("   stdout:", repr(res.get("stdout", "")[:300]), "\n   stderr:", repr(res.get("stderr", "")[-300:]))


def alive():
    r = run_python("print(6*7)")
    return r["status"] == "ok" and r["stdout"].strip() == "42"


def main():
    h = health()
    print("health:", h)

    # ---- functionality
    r = run_python("from fractions import Fraction as F\nprint(sum(F(1, k) for k in range(1, 11)))")
    check("fractions exact arithmetic", r["status"] == "ok" and r["stdout"].strip() == "7381/2520", r)
    r = run_python("import sympy as sp\nx=sp.symbols('x')\nprint(sp.solve(x**3-6*x**2+11*x-6, x))\n"
                   "print(sp.integrate(x**2*sp.exp(x),(x,0,1)))")
    check("sympy solve/integrate", r["status"] == "ok" and "[1, 2, 3]" in r["stdout"] and r["stdout"].split("\n")[1].replace(" ", "") in ("-2+E", "E-2"), r)
    r = run_python("import numpy as np\nprint(np.linalg.det(np.array([[2,1],[1,3]])).round(6))")
    check("numpy", r["status"] == "ok" and r["stdout"].strip() == "5.0", r)
    r = run_python("from itertools import product\nsum(1 for t in product(range(1,7),repeat=3) if sum(t)==10)")
    check("last expression auto-printed", r["status"] == "ok" and r["stdout"].strip() == "27", r)
    r = run_python("x = 1/0")
    check("exception -> error + clean traceback", r["status"] == "error" and "ZeroDivisionError" in r["stderr"]
          and "/srv/" not in r["stderr"] and 'File "<python>", line 1' in r["stderr"], r)
    r = run_python("def f(:\n  pass")
    check("syntax error -> error", r["status"] == "error" and "SyntaxError" in r["stderr"], r)
    r = run_python("import os\nprint(sorted(os.environ))")
    check("no secrets in environment", r["status"] == "ok" and "KEY" not in r["stdout"] and "TOKEN" not in r["stdout"], r,
          r["stdout"].strip())
    r = run_python("print(7)", timeout=3)
    check("state not kept: fresh process per call",
          run_python("x = 5")["status"] == "ok" and run_python("print(x)")["status"] == "error", r)

    # ---- time limits
    t = time.time()
    r = run_python("while True:\n    pass")
    dt = time.time() - t
    check("infinite loop killed at 10 s (default)", r["status"] == "timeout" and 9.5 <= dt <= 12.5, r, f"wall={dt:.1f}s")
    t = time.time()
    r = run_python("import time\ntime.sleep(60)", timeout=2)
    dt = time.time() - t
    check("sleep killed at requested 2 s", r["status"] == "timeout" and dt < 4, r, f"wall={dt:.1f}s")
    r = run_python("while True: pass", timeout=60)
    check("timeout capped at 10 s even if client asks 60", r["status"] == "timeout" and r["wall_ms"] <= 10600, r)
    r = run_python("print('partial', flush=True)\nwhile True: pass", timeout=1)
    check("partial output kept on timeout", r["status"] == "timeout" and "partial" in r["stdout"], r)

    # ---- memory limits
    r = run_python("x = bytearray(4 * 1024**3)\nprint(len(x))")
    check("4 GiB allocation refused", r["status"] == "memory" and "MemoryError" in r["stderr"], r)
    r = run_python("l = []\nwhile True:\n    l.append(bytearray(10**7))", timeout=10)
    check("growing memory bomb contained (<=1 GiB)", r["status"] == "memory", r)
    r = run_python("import numpy as np\na = np.ones((20000, 20000))\nprint(a.sum())")
    check("numpy 3.2 GB array refused", r["status"] == "memory", r)
    r = run_python("s = '9' * 100\nprint(len(str(int(s) ** 10**7)))", timeout=3)
    check("huge bignum bounded (timeout/memory/error)", r["status"] in ("timeout", "memory", "error"), r)
    check("service alive after memory bombs", alive())

    # ---- network
    r = run_python("import urllib.request\nprint(urllib.request.urlopen('http://example.com', timeout=3).status)")
    check("HTTP to internet fails", r["status"] == "error" and r["stdout"] == "", r, r["stderr"].strip().splitlines()[-1])
    r = run_python("import socket\ns = socket.socket()\ns.connect(('1.1.1.1', 53))\nprint('connected')")
    check("raw socket blocked (seccomp)", r["status"] == "error" and "PermissionError" in r["stderr"], r)
    r = run_python("import socket\nprint(socket.gethostbyname('ksat-tool-sandbox'))")
    check("DNS / socket use blocked", r["status"] == "error", r)
    r = run_python("import socket\ns=socket.socket(socket.AF_UNIX)\nprint('unix socket ok')")
    check("unix sockets blocked", r["status"] == "error", r)

    # ---- filesystem
    r = run_python("open('/srv/server.py', 'a').write('x')")
    check("write to server code fails", r["status"] == "error" and ("Read-only" in r["stderr"] or "Permission" in r["stderr"]), r)
    r = run_python("open('/etc/passwd', 'a').write('x')")
    check("write to /etc fails", r["status"] == "error", r)
    r = run_python("import os\nos.makedirs('/home/x', exist_ok=True)")
    check("mkdir outside tmp fails", r["status"] == "error", r)
    r = run_python("open('/usr/local/lib/python3.12/os.py','w').write('pwn')")
    check("overwrite stdlib fails", r["status"] == "error", r)
    r = run_python("import os\nopen('scratch.txt','w').write('hi')\nprint(open('scratch.txt').read(), os.getcwd())")
    check("write inside private tmp dir works", r["status"] == "ok" and r["stdout"].startswith("hi /tmp/run-"), r)
    r = run_python("f = open('big.bin', 'wb')\nfor _ in range(100):\n    f.write(b'0' * 10**6)\nf.close()")
    check("file size limited (RLIMIT_FSIZE)", r["status"] == "error" and "File too large" in r["stderr"], r)
    r = run_python("import os\nprint(os.listdir('/'))\nprint(os.path.exists('/data'), os.path.exists('/raid'))")
    check("no data / answer mounts", r["status"] == "ok" and "False False" in r["stdout"], r)
    r = run_python("print(open('/proc/1/environ').read())")
    check("cannot read server (pid 1) environ", r["status"] == "error", r)
    r = run_python("import os\nprint(open(f'/proc/{os.getppid()}/environ').read())")
    check("cannot read worker environ", r["status"] == "error", r)
    r = run_python("import os\nprint(open(f'/proc/{os.getppid()}/mem','rb').read(1))")
    check("cannot read worker memory", r["status"] == "error", r)
    r = run_python("import os\nprint(os.getuid())")
    check("runs as non-root uid", r["status"] == "ok" and r["stdout"].strip() not in ("0", ""), r, r["stdout"].strip())

    # ---- processes / signals
    r = run_python("import os\nos.system('echo pwned')")
    check("os.system blocked", "pwned" not in r["stdout"], r)
    r = run_python("import subprocess\nprint(subprocess.run(['ls','/'],capture_output=True).stdout)")
    check("subprocess blocked", r["status"] == "error" and "PermissionError" in r["stderr"], r)
    r = run_python("import os\nwhile True:\n    os.fork()")
    check("fork bomb blocked", r["status"] == "error" and "PermissionError" in r["stderr"], r)
    r = run_python("import threading\nt=threading.Thread(target=print,args=(1,))\nt.start()")
    check("thread creation blocked", r["status"] == "error", r)
    r = run_python("import os, signal\nos.kill(os.getppid(), signal.SIGKILL)")
    check("cannot kill the server worker", r["status"] == "error" and "PermissionError" in r["stderr"], r)
    r = run_python("import os, signal\nos.kill(-1, signal.SIGKILL)")
    check("cannot kill(-1)", r["status"] == "error", r)
    r = run_python("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)")
    check("may kill itself -> status killed", r["status"] == "killed", r)
    check("service alive after signal tests", alive())

    # ---- output truncation
    r = run_python("print('x' * 5000)")
    check("stdout truncated to <=1000 chars", r["status"] == "ok" and len(r["stdout"]) <= 1000 and r["truncated"], r)
    r = run_python("while True:\n    print('spam' * 100)", timeout=3)
    check("output flood bounded", len(r["stdout"]) <= 1000 and r["truncated"], r)
    obs = format_observation(run_python("print('y'*3000)"))
    check("observation text <= 1000 chars", len(obs) <= 1000 and obs.endswith("[output truncated]"), extra=f"len={len(obs)}")
    obs = format_observation(run_python("import sympy\n" + "def f(n):\n  return f(n+1)\nf(0)"))
    check("long traceback keeps error tail", len(obs) <= 1000 and "RecursionError" in obs, extra=f"len={len(obs)}")
    r = run_python("print('가나다 한글 ✓')")
    check("unicode output", r["status"] == "ok" and "한글" in r["stdout"], r)
    big = "x = 1\n" * 20000
    r = run_python(big)
    check("oversized code rejected cleanly", r["status"] == "error" and "too large" in r["stderr"], r)

    # ---- throughput
    def bench(code, n, conc):
        t = time.time()
        with cf.ThreadPoolExecutor(conc) as ex:
            rs = list(ex.map(lambda _: run_python(code), range(n)))
        dt = time.time() - t
        ok = sum(r["status"] == "ok" for r in rs)
        return n / dt, ok, n

    workers = h.get("workers")
    thr_trivial = bench("print(sum(range(1000)))", 1000, 64)
    thr_sympy = bench("import sympy as sp\nx=sp.symbols('x')\nprint(sp.factor(x**12-1))\nprint(sp.integrate(sp.sin(x)**4,(x,0,sp.pi)))", 300, 64)
    thr_enum = bench("from itertools import permutations\nprint(sum(1 for p in permutations(range(8)) if all(p[i]!=i for i in range(8))))", 200, 64)
    check("throughput trivial: all ok", thr_trivial[1] == thr_trivial[2], extra=f"{thr_trivial[0]:.1f} calls/s")
    check("throughput sympy: all ok", thr_sympy[1] == thr_sympy[2], extra=f"{thr_sympy[0]:.1f} calls/s")
    check("throughput enumeration 8!: all ok", thr_enum[1] == thr_enum[2], extra=f"{thr_enum[0]:.1f} calls/s")
    check("service alive at end", alive())

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(json.dumps({"passed": len(RESULTS) - n_fail, "failed": n_fail, "workers": workers,
                      "calls_per_s": {"trivial": round(thr_trivial[0], 1), "sympy_factor_integrate": round(thr_sympy[0], 1),
                                      "enumerate_8_perm": round(thr_enum[0], 1)}}))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
