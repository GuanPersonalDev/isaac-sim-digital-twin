#!/usr/bin/env bash
# 容器進入點。此檔必須是 LF 換行（見 training/.gitattributes）。
set -euo pipefail

# 路徑對應現行環境（自訂 CUDA 映像 + volume 上的 venv，見 training/README.md）。
# 先前的預設值 /workspace/billiard 與小寫 /workspace/isaaclab 是官方映像路線的
# 遺留，該路線已於 2026-08-02 廢棄。
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/IsaacLab}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/isaac-sim-digital-twin}"
CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/training/configs/ppo_billiard.yaml}"
# 必須與 .gitignore 的 training/outputs/ 一致，否則 checkpoint 會掉進 git
# 工作區。也必須在 /workspace 底下，否則停機就沒了。
OUTPUT_DIR="${TRAIN_OUTPUT:-${PROJECT_ROOT}/training/outputs}"
TRAIN_TASK="${TRAIN_TASK:-}"

# venv 在 network volume 上持久化，優先用它；系統 python3 是 apt 裝的，
# container disk 停機清空後不保證還在。
if [[ -n "${PYTHON_EXE:-}" ]]; then
    :
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_EXE="${VIRTUAL_ENV}/bin/python"
elif [[ -x /workspace/venv/bin/python ]]; then
    PYTHON_EXE=/workspace/venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
else
    echo "[run_train] 找不到可用的 Python。先執行 source /workspace/setup.sh" >&2
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
