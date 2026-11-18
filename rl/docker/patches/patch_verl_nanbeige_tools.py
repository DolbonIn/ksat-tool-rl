"""Patch veRL (main @ b0871fd89d) for Nanbeige4.x multi-turn tool RL.  Idempotent: re-running is a no-op.

  1. Installs NanbeigeContinuousTokenBuilder as verl/utils/tokenizer/nanbeige.py and registers it for
     config.json model_type="nanbeige" (new family "nanbeige" in continuous_token_wiring.py).  Without it veRL
     falls back to the default builder ("No model-specific Continuous Token builder is registered for
     model_type='nanbeige'"), which drops the template's "\\n" after <|im_end|> at every tool turn; the strict
     tokenization sanity check then refuses to start (veRL's Qwen builder cannot be reused: Nanbeige's
     Metaspace tokenizer encodes a standalone "\\n" as two ids).
  2. Installs NanbeigeToolParser as verl/experimental/agent_loop/tool_parser_nanbeige.py and registers it as
     multi_turn.format=nanbeige (imported at the end of tool_parser.py).
Sources live next to this file in verl_nanbeige/.  Applied at image build (Dockerfile) or at container start
(train_grpo.sh with KSAT_TOOL_PATCH=1), so the current v4 image works without a rebuild.
Usage: python patch_verl_nanbeige_tools.py [/opt/verl]
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/verl"
HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "# [ksat-patch-nanbeige]"


def install(src, dst):
    src, dst = os.path.join(HERE, "verl_nanbeige", src), os.path.join(ROOT, dst)
    new = open(src).read()
    if os.path.exists(dst) and open(dst).read() == new:
        print("up to date", dst)
        return
    shutil.copyfile(src, dst)
    print("installed", dst)


def patch(rel, fn):
    path = os.path.join(ROOT, rel)
    s = open(path).read()
    if MARK in s:
        print("already patched", path)
        return
    s2 = fn(s)
    assert s2 != s and MARK in s2, f"patch made no change: {path}"
    open(path, "w").write(s2)
    print("patched", path)


def wiring(s):
    imp = "from .deepseek import DeepSeekV4ContinuousTokenBuilder\n"
    assert imp in s
    s = s.replace(imp, imp + f"from .nanbeige import NanbeigeContinuousTokenBuilder  {MARK}\n", 1)
    enum = '    DEEPSEEK = "deepseek"\n'
    assert enum in s
    s = s.replace(enum, enum + f'    NANBEIGE = "nanbeige"  {MARK}\n', 1)
    reg = "    ContinuousTokenModelFamily.DEEPSEEK: DeepSeekContinuousTokenBuilder,\n"
    assert reg in s
    s = s.replace(reg, reg + f"    ContinuousTokenModelFamily.NANBEIGE: NanbeigeContinuousTokenBuilder,  {MARK}\n", 1)
    mt = '    "deepseek_v4": ContinuousTokenModelFamily.DEEPSEEKV4,\n'
    assert mt in s
    s = s.replace(mt, mt + f'    "nanbeige": ContinuousTokenModelFamily.NANBEIGE,  {MARK}\n', 1)
    return s


def tool_parser(s):
    return s.rstrip("\n") + (
        f"\n\n\n{MARK} Nanbeige XML tool parser (multi_turn.format=nanbeige); registers itself on import.\n"
        "from verl.experimental.agent_loop.tool_parser_nanbeige import NanbeigeToolParser  # noqa: E402,F401\n"
    )


if __name__ == "__main__":
    install("continuous_token_nanbeige.py", "verl/utils/tokenizer/nanbeige.py")
    install("tool_parser_nanbeige.py", "verl/experimental/agent_loop/tool_parser_nanbeige.py")
    patch("verl/utils/tokenizer/continuous_token_wiring.py", wiring)
    patch("verl/experimental/agent_loop/tool_parser.py", tool_parser)
