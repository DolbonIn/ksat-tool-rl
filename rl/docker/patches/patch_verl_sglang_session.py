"""Patch veRL (main @ b0871fd89d) for the SGLang nightly in our image (Nanbeige support needs sglang>=0.5.20).

Newer SGLang requires every update_weights_from_tensor to run inside a weight-update session
(begin_weight_update / end_weight_update); outside one it returns success=False WITHOUT raising,
which would leave the rollout engine silently on stale weights. veRL targets sglang 0.5.12 and
never opens a session. This patch:
  1. implements begin/end_weight_update on veRL's (Async)HttpServerAdapter (abstract in EngineBase),
  2. wraps the bucketed sync in SGLangRollout.update_weights with a session on the TP leader,
  3. makes a failed update_weights_from_tensor raise instead of being ignored.
Idempotent: re-running is a no-op.
"""
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/verl"
MARK = "# [ksat-patch]"


def patch(path, fn):
    s = open(path).read()
    if MARK in s:
        print("already patched", path); return
    s2 = fn(s)
    assert s2 != s, f"patch made no change: {path}"
    open(path, "w").write(s2)
    print("patched", path)


def http_engine(s):
    sync_methods = f'''
    def begin_weight_update(self, selector: str = "all") -> dict[str, Any]:  {MARK}
        return self._make_request("begin_weight_update", {{"selector": selector}})

    def end_weight_update(self) -> dict[str, Any]:  {MARK}
        return self._make_request("end_weight_update", {{}})

    def update_weights_from_tensor(self, req'''
    s = s.replace("\n    def update_weights_from_tensor(self, req", sync_methods, 1)
    async_methods = f'''
    async def begin_weight_update(self, selector: str = "all") -> dict[str, Any]:  {MARK}
        return await self._make_async_request("begin_weight_update", {{"selector": selector}})

    async def end_weight_update(self) -> dict[str, Any]:  {MARK}
        return await self._make_async_request("end_weight_update", {{}})

    async def update_weights_from_tensor(
        self,'''
    s = s.replace("\n    async def update_weights_from_tensor(\n        self,", async_methods, 1)
    return s


def rollout(s):
    # open session before bucketed sync, close after (only the TP leader talks HTTP)
    anchor = "            update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20\n"
    assert anchor in s
    s = s.replace(anchor, anchor + f"""            _ksat_session = self._engine is not None and self._is_server_tp_leader()  {MARK}
            if _ksat_session:
                _r = await self._engine.begin_weight_update()
                if isinstance(_r, dict) and _r.get("success") is False:
                    raise RuntimeError(f"begin_weight_update failed: {{_r}}")
""", 1)
    anchor2 = """        if self._engine is not None and self._is_server_tp_leader():
            await self._engine.flush_cache()"""
    assert anchor2 in s
    s = s.replace(anchor2, f"""        if not (peft_config and base_sync_done) and self._engine is not None and self._is_server_tp_leader():  {MARK}
            _r = await self._engine.end_weight_update()
            if isinstance(_r, dict) and _r.get("success") is False:
                raise RuntimeError(f"end_weight_update failed: {{_r}}")
""" + anchor2, 1)
    return s


def rollout_checked(s):
    # veRL ignores the return of sglang.srt.weight_sync.utils.update_weights; make failures raise.
    helper = f"""

async def _ksat_checked_update(**kwargs):  {MARK}
    _res = await sgl_update_weights(**kwargs)
    if isinstance(_res, dict) and _res.get("success") is False:
        raise RuntimeError(f"update_weights_from_tensor failed: {{_res}}")
    if _res is not None and getattr(_res, "success", True) is False:
        raise RuntimeError(f"update_weights_from_tensor failed: {{_res}}")
    return _res
"""
    imp = "from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights\n"
    assert imp in s
    assert s.count("await sgl_update_weights(") >= 1
    s = s.replace("await sgl_update_weights(", "await _ksat_checked_update(")  # call sites first ...
    return s.replace(imp, imp + helper, 1)  # ... then add the helper, which calls the real function


def fsdp_wrap(s):
    # veRL wraps lm_head as its own FSDP2 unit (by name) for untied models, but the fused-linear
    # log-prob path reads lm_head.weight without calling lm_head.forward, so FSDP never unshards it
    # ("mixed torch.Tensor and DTensor") and its grad hooks would never fire. Keep lm_head in the root
    # unit (root params stay unsharded through forward/backward).
    old = '_wrap_by_name = set() if _tie else {"embed_tokens", "lm_head"}'
    assert old in s
    return s.replace(old, f'_wrap_by_name = set() if _tie else {{"embed_tokens"}}  {MARK} lm_head stays in root unit')


patch(f"{ROOT}/verl/utils/fsdp_utils.py", fsdp_wrap)
base = f"{ROOT}/verl/workers/rollout/sglang_rollout"
patch(f"{base}/http_server_engine.py", http_engine)
patch(f"{base}/sglang_rollout.py", lambda x: rollout_checked(rollout(x)))
