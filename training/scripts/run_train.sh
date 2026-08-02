#!/usr/bin/env bash
# 容器進入點。此檔必須是 LF 換行（見 training/.gitattributes）。
set -euo pipefail

# 實測 nvcr.io/nvidia/isaac-lab:3.0.0-beta2：
#   - Isaac Lab 在 /workspace/isaaclab（小寫，#224 寫的 /workspace/IsaacLab 是錯的）
#   - 容器內沒有 python 也沒有 python3，只有 /isaac-sim/python.sh
#     （isaaclab.sh 自己也是解析到 $ISAACLAB_PATH/_isaac_sim/python.sh 這個 symlink）
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/isaaclab}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/billiard}"
CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/training/configs/ppo_billiard.yaml}"
OUTPUT_DIR="${TRAIN_OUTPUT:-${PROJECT_ROOT}/outputs}"
TRAIN_TASK="${TRAIN_TASK:-}"

if [[ -n "${PYTHON_EXE:-}" ]]; then
    :
elif [[ -x "${ISAACLAB_ROOT}/_isaac_sim/python.sh" ]]; then
    PYTHON_EXE="${ISAACLAB_ROOT}/_isaac_sim/python.sh"
elif [[ -x /isaac-sim/python.sh ]]; then
    PYTHON_EXE=/isaac-sim/python.sh
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
else
    echo "[run_train] 找不到可用的 Python" >&2
    exit 1
fi

echo "[run_train] isaaclab_root = ${ISAACLAB_ROOT}"
echo "[run_train] project_root  = ${PROJECT_ROOT}"
echo "[run_train] python        = ${PYTHON_EXE}"
echo "[run_train] config        = ${CONFIG}"
echo "[run_train] output_dir    = ${OUTPUT_DIR}"

if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "[run_train] 找不到 ${ISAACLAB_ROOT}/isaaclab.sh" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[run_train] 找不到設定檔: ${CONFIG}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# BilliardEnv（#121/#122）尚未實作，還沒有可用的 --task。
# 完成後把下面這段的註解拿掉、刪除底下的 smoke test：
#
#   exec "${ISAACLAB_ROOT}/isaaclab.sh" -p \
#       "${ISAACLAB_ROOT}/scripts/reinforcement_learning/rsl_rl/train.py" \
#       --task "${TRAIN_TASK}" \
#       --headless \
#       "$@"

if [[ -n "${TRAIN_TASK}" ]]; then
    echo "[run_train] TRAIN_TASK=${TRAIN_TASK} 已指定，但訓練進入點尚未接上（見 #121/#122）。" >&2
fi

echo "[run_train] --- smoke test（驗證 #224 完成標準）---" >&2
exec "${PYTHON_EXE}" -c "
import sys
print('[smoke] python     :', sys.version.split()[0])

import isaacsim
print('[smoke] isaacsim   : import OK')

import isaaclab
print('[smoke] isaaclab   : import OK', getattr(isaaclab, '__version__', ''))

from core.services.rl_observation_encoder import encode_rl_observation
print('[smoke] core/      : import OK')
"
