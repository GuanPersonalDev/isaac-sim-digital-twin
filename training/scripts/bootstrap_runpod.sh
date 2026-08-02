#!/usr/bin/env bash
# RunPod 實例開機後的引導腳本（#224 項目 4：可重複的程式碼同步流程）。
# 此檔必須是 LF 換行（見 training/.gitattributes）。
#
# 用法（在 pod 的 shell 內）：
#   curl -fsSL https://raw.githubusercontent.com/GuanPersonalDev/isaac-sim-digital-twin/main/training/scripts/bootstrap_runpod.sh | bash
#
# 或先手動 clone 一次，之後每次改 code 只要重跑本腳本即可同步。
# repo 為 public，不需要任何憑證。

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/GuanPersonalDev/isaac-sim-digital-twin.git}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/billiard}"
BRANCH="${BRANCH:-main}"
RUN_AFTER_SYNC="${RUN_AFTER_SYNC:-0}"

echo "[bootstrap] repo    = ${REPO_URL}"
echo "[bootstrap] branch  = ${BRANCH}"
echo "[bootstrap] target  = ${PROJECT_ROOT}"

if ! command -v git >/dev/null 2>&1; then
    echo "[bootstrap] 容器內沒有 git，請先安裝（apt-get update && apt-get install -y git）" >&2
    exit 1
fi

if [[ -d "${PROJECT_ROOT}/.git" ]]; then
    echo "[bootstrap] 已存在 repo，執行增量更新"
    git -C "${PROJECT_ROOT}" fetch --prune origin
    # 用 --ff-only：若 pod 上有本機修改導致分歧，這裡會明確失敗而不是靜默覆蓋。
    # 真要丟棄 pod 上的改動，自行執行 git reset --hard "origin/${BRANCH}"。
    git -C "${PROJECT_ROOT}" checkout "${BRANCH}"
    git -C "${PROJECT_ROOT}" pull --ff-only origin "${BRANCH}"
else
    echo "[bootstrap] 首次 clone"
    git clone --branch "${BRANCH}" "${REPO_URL}" "${PROJECT_ROOT}"
fi

echo "[bootstrap] HEAD = $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD) $(git -C "${PROJECT_ROOT}" log -1 --format=%s)"

REQUIREMENTS="${PROJECT_ROOT}/training/requirements.txt"
if [[ -f "${REQUIREMENTS}" ]]; then
    echo "[bootstrap] 安裝額外依賴"
    python -m pip install --no-cache-dir -r "${REQUIREMENTS}"
fi

chmod +x "${PROJECT_ROOT}/training/scripts/"*.sh

echo "[bootstrap] 同步完成。"
echo "[bootstrap] 執行訓練： PROJECT_ROOT=${PROJECT_ROOT} ${PROJECT_ROOT}/training/scripts/run_train.sh"

if [[ "${RUN_AFTER_SYNC}" == "1" ]]; then
    exec env PROJECT_ROOT="${PROJECT_ROOT}" \
        "${PROJECT_ROOT}/training/scripts/run_train.sh"
fi
