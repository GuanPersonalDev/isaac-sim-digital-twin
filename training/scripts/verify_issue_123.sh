#!/usr/bin/env bash
# #123：RunPod 上的一鍵 PPO 接線／遮罩驗證。
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/isaac-sim-digital-twin}"
NUM_ENVS="${NUM_ENVS:-64}"
ITERATIONS="${ITERATIONS:-5}"
VALIDATION_DIR="${VALIDATION_DIR:-/workspace/issue123-validation}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${VALIDATION_DIR}/training-runs}"
LOG_PATH="${LOG_PATH:-${VALIDATION_DIR}/ppo-validation.log}"
TEST_LOG_PATH="${TEST_LOG_PATH:-${VALIDATION_DIR}/pytest.log}"

if [[ -n "${PYTHON_EXE:-}" ]]; then
    :
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_EXE="${VIRTUAL_ENV}/bin/python"
elif [[ -x /workspace/venv/bin/python ]]; then
    PYTHON_EXE=/workspace/venv/bin/python
else
    echo "[issue123] 找不到 RunPod venv。先執行：source /workspace/setup.sh" >&2
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
    echo "[issue123] 找不到專案：${PROJECT_ROOT}" >&2
    exit 1
fi

mkdir -p "${VALIDATION_DIR}" "${TRAIN_RUN_DIR}"
cd "${PROJECT_ROOT}"

echo "[issue123] commit     = $(git rev-parse --short HEAD)"
echo "[issue123] python     = ${PYTHON_EXE}"
echo "[issue123] num_envs   = ${NUM_ENVS}"
echo "[issue123] iterations = ${ITERATIONS}"

"${PYTHON_EXE}" - <<'PY'
from importlib.metadata import version

import torch
from billiard_rl.algorithms import MaskedPPO

rsl_rl_version = version("rsl-rl-lib")
print("[issue123] torch    =", torch.__version__)
print("[issue123] rsl-rl   =", rsl_rl_version)
print("[issue123] algorithm=", MaskedPPO)
print("[issue123] cuda     =", torch.cuda.is_available())
if rsl_rl_version != "5.0.1":
    raise SystemExit(f"[#123 FAIL] 預期 rsl-rl-lib 5.0.1，實際 {rsl_rl_version}")
if not torch.cuda.is_available():
    raise SystemExit("[#123 FAIL] CUDA 不可用")
PY

echo "[issue123] 執行 core + rl_task tests"
"${PYTHON_EXE}" -m pytest core/tests rl_task/tests -q 2>&1 | tee "${TEST_LOG_PATH}"

echo "[issue123] 啟動短訓練"
set +e
BILLIARD_PPO_DIAGNOSTICS=1 \
RUN_DIR="${TRAIN_RUN_DIR}" \
bash training/scripts/run_train.sh \
    --num_envs "${NUM_ENVS}" \
    "agent.max_iterations=${ITERATIONS}" \
    "agent.save_interval=${ITERATIONS}" \
    2>&1 | tee "${LOG_PATH}"
train_status=${PIPESTATUS[0]}
set -e

if [[ "${train_status}" -ne 0 ]]; then
    echo "[#123 FAIL] 短訓練失敗（exit=${train_status}），見 ${LOG_PATH}" >&2
    exit "${train_status}"
fi

"${PYTHON_EXE}" - "${LOG_PATH}" "${NUM_ENVS}" <<'PY'
import json
import math
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
expected_num_envs = int(sys.argv[2])
prefix = "[issue123-ppo] "
records = []

for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if prefix in line:
        records.append(json.loads(line.split(prefix, 1)[1]))

algorithm = [record for record in records if record.get("event") == "algorithm"]
returns = [record for record in records if record.get("event") == "returns"]
errors = []

if len(algorithm) != 1:
    errors.append(f"algorithm 診斷筆數應為 1，實際 {len(algorithm)}")
elif algorithm[0].get("class") != "billiard_rl.algorithms.masked_ppo.MaskedPPO":
    errors.append(f"實際 algorithm 不是 MaskedPPO：{algorithm[0].get('class')}")

if len(returns) < 2:
    errors.append(f"returns 診斷至少需要 2 筆，實際 {len(returns)}")

for record in returns:
    shape = record.get("advantages_shape")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and shape[1] == expected_num_envs
        and shape[2] == 1
    ):
        errors.append(f"advantages shape 不符：(T, {expected_num_envs}, 1) != {shape}")
    if record.get("fallback"):
        errors.append(f"iteration {record.get('iteration')} 進入無有效樣本 fallback")

# 第一個 rollout 由全新 reset 開始，仍納入 shape 檢查；比例判定以第二筆起為準。
for record in returns[1:]:
    valid_ratio = float(record["valid_ratio"])
    nonzero_ratio = float(record["nonzero_ratio"])
    if not 0.0 < valid_ratio < 0.5:
        errors.append(
            f"iteration {record['iteration']} valid_ratio 應在 (0, 0.5)，實際 {valid_ratio:.6f}"
        )
    if not math.isclose(valid_ratio, nonzero_ratio, abs_tol=0.01):
        errors.append(
            f"iteration {record['iteration']} valid/nonzero 比例不一致："
            f"{valid_ratio:.6f} vs {nonzero_ratio:.6f}"
        )

if errors:
    print("[#123 FAIL] MaskedPPO 診斷未通過：", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print("[#123 PASS] MaskedPPO class、advantages shape 與遮罩比例皆通過")
for record in returns:
    print(
        f"  iteration={record['iteration']} shape={record['advantages_shape']} "
        f"valid={record['valid_ratio']:.4f} nonzero={record['nonzero_ratio']:.4f}"
    )
PY

# --- 場景規模煙霧測試（#123 review 第 5 點）------------------------------------
#
# 上面的 PPO 驗證跑的是 NUM_ENVS（預設 64），但 billiard_rl_env_cfg.py 設定的
# 是 1024——那個規模**從來沒有實際建起來過**。桌台碰撞網格是主要記憶體項，
# 「建不建得起來」與「it/s 拐點在哪」都還沒驗，這是 #124 正式開跑前的實質風險。
#
# 設 SCENE_SMOKE_ENVS=0 可略過。想找算力飽和點就跑幾次不同的值：
#   for n in 64 256 1024 2048; do SCENE_SMOKE_ENVS=$n bash training/scripts/verify_issue_123.sh; done
SCENE_SMOKE_ENVS="${SCENE_SMOKE_ENVS:-1024}"
SCENE_SMOKE_ITERATIONS="${SCENE_SMOKE_ITERATIONS:-2}"
SCENE_SMOKE_LOG="${SCENE_SMOKE_LOG:-${VALIDATION_DIR}/scene-smoke-${SCENE_SMOKE_ENVS}.log}"
SCENE_SMOKE_RUN_DIR="${SCENE_SMOKE_RUN_DIR:-${VALIDATION_DIR}/scene-smoke-runs}"
GPU_SAMPLE_PATH="${VALIDATION_DIR}/scene-smoke-${SCENE_SMOKE_ENVS}-gpu.txt"

if [[ "${SCENE_SMOKE_ENVS}" == "0" ]]; then
    echo "[issue123] 略過場景規模煙霧測試（SCENE_SMOKE_ENVS=0）"
else
    echo "[issue123] 場景規模煙霧測試：num_envs=${SCENE_SMOKE_ENVS} iterations=${SCENE_SMOKE_ITERATIONS}"
    mkdir -p "${SCENE_SMOKE_RUN_DIR}"
    : > "${GPU_SAMPLE_PATH}"

    gpu_sampler_pid=""
    if command -v nvidia-smi &>/dev/null; then
        # 每秒取一次已用顯存，之後取最大值——回答「1024 env 的峰值記憶體多少」。
        ( while true; do
              nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits                   >> "${GPU_SAMPLE_PATH}" 2>/dev/null || true
              sleep 1
          done ) &
        gpu_sampler_pid=$!
    fi

    smoke_start=${SECONDS}
    set +e
    RUN_DIR="${SCENE_SMOKE_RUN_DIR}"     bash training/scripts/run_train.sh         --num_envs "${SCENE_SMOKE_ENVS}"         "agent.max_iterations=${SCENE_SMOKE_ITERATIONS}"         "agent.save_interval=${SCENE_SMOKE_ITERATIONS}"         2>&1 | tee "${SCENE_SMOKE_LOG}"
    smoke_status=${PIPESTATUS[0]}
    set -e
    smoke_elapsed=$(( SECONDS - smoke_start ))

    if [[ -n "${gpu_sampler_pid}" ]]; then
        kill "${gpu_sampler_pid}" 2>/dev/null || true
        wait "${gpu_sampler_pid}" 2>/dev/null || true
    fi

    peak_gpu_mib="n/a"
    if [[ -s "${GPU_SAMPLE_PATH}" ]]; then
        peak_gpu_mib=$(sort -n "${GPU_SAMPLE_PATH}" | tail -1)
    fi

    if [[ "${smoke_status}" -ne 0 ]]; then
        echo "[#123 FAIL] num_envs=${SCENE_SMOKE_ENVS} 起不來（exit=${smoke_status}）。" >&2
        echo "            峰值顯存 ${peak_gpu_mib} MiB，log：${SCENE_SMOKE_LOG}" >&2
        echo "            若是 OOM，先降 env_spacing 或改用不含 SimpleRoom 的桌台 USD。" >&2
        exit "${smoke_status}"
    fi

    echo "[#123 PASS] num_envs=${SCENE_SMOKE_ENVS} 場景建置與 ${SCENE_SMOKE_ITERATIONS} 個 iteration 完成"
    echo "            wall-clock ${smoke_elapsed}s（含 Isaac 啟動）／峰值顯存 ${peak_gpu_mib} MiB"
    echo "            log：${SCENE_SMOKE_LOG}"
fi

echo "[#123 PASS] 完整 log：${LOG_PATH}"
