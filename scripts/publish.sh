#!/usr/bin/env bash
# Runs only on the user's machine, using their existing GitHub CLI authentication.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-ksat-tool-rl}"
EXPECTED_OWNER="DolbonIn"
if [[ ! "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid repository name." >&2; exit 2
fi
for tool in git gh python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required command not found: $tool" >&2; exit 2
  fi
done
cd "$ROOT"
if [[ -e .git ]] || git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "Refusing to modify an existing local Git repository. Use a fresh extracted directory." >&2
  exit 2
fi
gh auth status --hostname github.com
OWNER="$(gh api --hostname github.com user --jq '.login')"
if [[ "$OWNER" != "$EXPECTED_OWNER" ]]; then
  echo "Authenticated account is not $EXPECTED_OWNER; no repository was created." >&2
  exit 2
fi
TARGET="$OWNER/$NAME"
if gh repo view "github.com/$TARGET" --json name >/dev/null 2>&1; then
  echo "Repository $TARGET already exists; refusing to overwrite or push to it." >&2
  exit 2
fi
# A failed preflight can also be a network issue; repo creation will fail safely.
python3 scripts/verify_manifest.py
bash scripts/run_checks.sh
git init -b main
git add .
# Identity is scoped to this commit and does not change global configuration.
git -c user.name="$OWNER" -c user.email="$OWNER@users.noreply.github.com"   commit -m "Add KSAT tool-calling RL blueprint and dataset audit utilities"
gh repo create "github.com/$TARGET" --private --source=. --remote=origin --push   --description "KSAT on-device tool-calling RL blueprint, data schemas, and audit utilities"
printf '\nCreated and pushed: https://github.com/%s\n' "$TARGET"
