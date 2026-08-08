#!/usr/bin/env bash
# 容器進入點。此檔必須是 LF 換行（見 training/.gitattributes）。
set -euo pipefail

# 路徑對應現行環境（自訂 CUDA 映像 + volume 上的 venv，見 training/README.md）。
# 先前的預設值 /workspace/billiard 與小寫 /workspace/isaaclab 是官方映像路線的
# 遺留，該路線已於 2026-08-02 廢棄。
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/IsaacLab}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/isaac-sim-digital-twin}"
TRAIN_TASK="${TRAIN_TASK:-Isaac-Billiard-v0}"

# train 腳本只有 `import isaaclab_tasks`（train_rsl_rl.py:35），不會自動發現外部
# package，必須靠這個 callback 觸發本專案的 gym.register（#121 C-3）。
# 掛載點定義在 rl_task/billiard_rl/tasks/__init__.py。
EXTERNAL_CALLBACK="${EXTERNAL_CALLBACK:-billiard_rl.tasks.register_external_tasks}"

# checkpoint / tensorboard log 的落點。
#
# ⚠️ Isaac Lab 沒有任何參數可以覆寫 log 路徑——train 腳本寫死
#    os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))，相對 CWD；
#    而 isaaclab.sh 的 run_python_command() 用 cwd=os.getcwd() 起 subprocess，
#    所以「從哪個目錄呼叫」是唯一的控制手段（#121 E-2）。先前的 OUTPUT_DIR /
#    TRAIN_OUTPUT 兩個變數完全不生效，已移除。
#
#    必須在 /workspace 底下：container disk 停機即清空。
RUN_DIR="${RUN_DIR:-/workspace/training-runs}"

# 目視確認（#121 A-3）要用 `HEADLESS=0 ... --viz viser --num_envs 4`，
# 與 --headless 互斥，所以不寫死。
HEADLESS="${HEADLESS:-1}"

# venv 在 network volume 上持久化，優先用它；系統 python3 是 apt 裝的，
# container disk 停機清空後不保證還在。（目前只有 smoke test 會用到）
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
echo "[run_train] task          = ${TRAIN_TASK}"
echo "[run_train] run_dir       = ${RUN_DIR}"
echo "[run_train] headless      = ${HEADLESS}"

if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "[run_train] 找不到 ${ISAACLAB_ROOT}/isaaclab.sh" >&2
    exit 1
fi

# SMOKE_TEST=1：只驗環境（#224 的完成標準），不進訓練。
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    echo "[run_train] --- smoke test ---" >&2
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
fi

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

if [[ "${HEADLESS}" == "1" ]]; then
    set -- --headless "$@"
fi

# 多帶的參數原樣傳給 Isaac Lab，交由 Hydra 解析，例如
#   agent.max_iterations=500 env.scene.num_envs=64
# 這也是取代 training/configs/ppo_billiard.yaml 的調參方式——超參數的單一來源是
# rsl_rl_cfg_entry_point 指向的 agents/rsl_rl_ppo_cfg.py:PPORunnerCfg。
#
# E-1：scripts/reinforcement_learning/rsl_rl/train.py 已 deprecated（該檔第 10-16
# 行即 DeprecationWarning），改用統一進入點 `isaaclab.sh train`。
exec "${ISAACLAB_ROOT}/isaaclab.sh" train \
    --rl_library rsl_rl \
    --task "${TRAIN_TASK}" \
    --external_callback "${EXTERNAL_CALLBACK}" \
    "$@"
