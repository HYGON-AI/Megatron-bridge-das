#!/bin/bash
# Apply one examples_hcu patch. Idempotent: an already-applied tree is a success.
# usage: patch_apply.sh <patch-file>
set -euo pipefail
cd "$(dirname "$0")/../.."                 # this script lives in examples_hcu/patch/
ROOT=$(pwd -P)
# root applying a patch to a checkout owned by another user trips git's
# "dubious ownership" check; whitelist this checkout and its submodule.
GIT=(git -c "safe.directory=$ROOT" -c "safe.directory=$ROOT/3rdparty/Megatron-LM")

YELLOW=${YELLOW:-'\033[33m'}
RESET=${RESET:-'\033[0m'}

if [ $# -lt 1 ]; then
    echo -e "${YELLOW}usage: $0 <patch-file>${RESET}" >&2
    exit 1
fi
PATCH=$1

"${GIT[@]}" rev-parse --git-dir >/dev/null 2>&1 || {
    echo -e "${YELLOW}[Patch] ERROR: $ROOT is not a usable git repository for user $(id -un)${RESET}" >&2
    exit 1
}

if "${GIT[@]}" apply --check "$PATCH" 2>/dev/null; then
    "${GIT[@]}" apply "$PATCH"
    echo -e "${YELLOW}[Patch] applied: $PATCH${RESET}"
elif "${GIT[@]}" apply --check -R "$PATCH" 2>/dev/null; then
    echo -e "${YELLOW}[Patch] already applied: $PATCH${RESET}"
else
    echo -e "${YELLOW}[Patch] cannot apply: $PATCH (tree in mixed state)${RESET}" >&2
    exit 1
fi