#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../3rdparty/Megatron-Bridge" && pwd)"
# Use the patched launcher next to this script (adds --moe-token-dispatcher-type
# etc.); it internally points the worker at the patched sibling modules and
# falls back to the pristine 3rdparty copies for anything not patched.
SETUP_SCRIPT="${SCRIPT_DIR}/setup_conversion.py"
cd "${REPO_ROOT}"

exec python "${SETUP_SCRIPT}" "$@"
