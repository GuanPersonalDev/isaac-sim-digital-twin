#!/usr/bin/env bash
# GPU 閒置看門狗：訓練結束或崩潰後自動停止 RunPod Pod，避免無人看管時持續計費。
#
# 解決的情境：訓練在週二崩潰或跑完，週六才發現 → 4 天 × 24h × $0.25 ≈ $24
# （等同整個專案預算）。平日 0.5h 巡檢無法可靠攔截這種情況。
#
# 安全性：Pod「停止」不是「終止」，/workspace（Network Volume）完整保留。
# 誤判的代價是重啟時間（約 1 分鐘 + Isaac Sim 冷啟動 58 秒），不是資料。
#
# 用法（與訓練並行啟動，不要放進開機腳本）：
#   tmux new -s train
#   /workspace/isaac-sim-digital-twin/training/scripts/gpu_watchdog.sh &
#   <啟動訓練指令>
#
# 先試跑不真的關機：
#   DRY_RUN=1 IDLE_LIMIT=2 ./gpu_watchdog.sh
#
# 前置需求：
#   WATCHDOG_API_KEY  RunPod API key，scope 必須含 Pods 讀寫（相容舊名
#                     WATCH_DOG_API_KEY）。填在 Pod 的環境變數即可，不要寫進
#                     任何檔案 —— 本 repo 是 public。
#
#   ⚠️ 不可使用 RUNPOD_API_KEY。那是平台保留並自動注入的 pod-scoped key，
#      權限不含管理 Pod 生命週期（實測 GraphQL Unauthorized、REST 403），
#      而且會覆蓋你在 Pod 設定裡填的同名值。
#
#   停機改走 REST API（見同目錄 runpod_api.py），不再依賴 runpodctl——
#   它只打 GraphQL，且與 curl 都不在基礎映像內，每次開機都要重裝。
#
# 可調參數（皆為環境變數）：
#   IDLE_LIMIT      連續閒置幾分鐘後停機（預設 30）
#   UTIL_THRESHOLD  GPU 使用率低於多少視為閒置（%，預設 5）
#   POLL_INTERVAL   取樣間隔（秒，預設 60）
#   HOURLY_RATE     GPU 時價，僅用於 log 顯示累計花費（預設 0.25，為 RTX A4500
#                   的價格；換卡後請依 RunPod Billing 頁的實際時價調整）
#   LOG             log 路徑（預設 /workspace/watchdog.log）
#   DRY_RUN         設為 1 則只記錄不真的停機
#   PYTHON_BIN      指定 python（預設依序找 venv、/workspace/venv、python3）
#   API_SCRIPT      指定 runpod_api.py 路徑（預設為本腳本同目錄）

set -uo pipefail

IDLE_LIMIT="${IDLE_LIMIT:-30}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-5}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
HOURLY_RATE="${HOURLY_RATE:-0.25}"
LOG="${LOG:-/workspace/watchdog.log}"
DRY_RUN="${DRY_RUN:-0}"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

# --- 前置檢查：缺任何一項就直接退出，不要假裝在看守 ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "[FATAL] 找不到 nvidia-smi，watchdog 無法運作。"
    exit 1
fi

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    log "[FATAL] RUNPOD_POD_ID 未設定，無法停機。確認是否在 RunPod Pod 內執行。"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_SCRIPT="${API_SCRIPT:-$SCRIPT_DIR/runpod_api.py}"

if [ ! -f "$API_SCRIPT" ]; then
    log "[FATAL] 找不到 $API_SCRIPT，無法停機。"
    exit 1
fi

# venv 在 network volume 上持久化，優先用它；系統 python3 是 apt 裝的，
# container disk 停機清空後不保證還在。
if [ -z "${PYTHON_BIN:-}" ]; then
    for candidate in "${VIRTUAL_ENV:-/nonexistent}/bin/python" \
                     /workspace/venv/bin/python \
                     python3; do
        if [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    log "[FATAL] 找不到可用的 python，無法停機。"
    exit 1
fi

stop_pod() {
    "$PYTHON_BIN" "$API_SCRIPT" stop >>"$LOG" 2>&1
}

# 取所有 GPU 中的最高使用率（單卡時等同該卡數值）。
gpu_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+$' | sort -rn | head -1
}

log "=== watchdog 啟動 pod=$RUNPOD_POD_ID idle_limit=${IDLE_LIMIT}min threshold=${UTIL_THRESHOLD}% dry_run=${DRY_RUN} ==="
log "python=$PYTHON_BIN api_script=$API_SCRIPT"

# 啟動當下就驗證金鑰，不要等閒置滿 30 分鐘才發現停不了機——那正是
# watchdog 要防的情境，靜默失效等於沒裝。
if [ "$DRY_RUN" != "1" ]; then
    if ! "$PYTHON_BIN" "$API_SCRIPT" check >>"$LOG" 2>&1; then
        log "[FATAL] API 金鑰驗證失敗，watchdog 無法在需要時停機，直接退出。"
        log "        手動重現：$PYTHON_BIN $API_SCRIPT check"
        exit 1
    fi
    log "API 金鑰驗證通過，停機路徑可用。"
fi

log "尚未偵測到 GPU 活動，處於未武裝狀態——看到 GPU 忙碌後才會開始倒數。"

# 內部一律以「秒」累計，最後才換算成分鐘。
# 若直接用分鐘累加（idle += POLL_INTERVAL/60），當 POLL_INTERVAL < 60 時
# 整數除法會得到 0，計數器永遠不動，watchdog 靜默失效。
idle_sec=0
elapsed_sec=0
idle_limit_sec=$((IDLE_LIMIT * 60))
armed=0   # 必須先看到 GPU 真的忙過才會啟用停機，避免誤啟動時把 Pod 關掉

while true; do
    util="$(gpu_util)"
    if [ -z "$util" ]; then
        log "[WARN] 讀不到 GPU 使用率，本輪跳過。"
        sleep "$POLL_INTERVAL"
        continue
    fi

    elapsed_sec=$((elapsed_sec + POLL_INTERVAL))
    cost="$(awk -v s="$elapsed_sec" -v r="$HOURLY_RATE" 'BEGIN{printf "%.2f", s/3600*r}')"

    if [ "$util" -ge "$UTIL_THRESHOLD" ]; then
        if [ "$armed" -eq 0 ]; then
            armed=1
            log "偵測到 GPU 活動（${util}%），watchdog 已武裝。"
        fi
        idle_sec=0
    elif [ "$armed" -eq 1 ]; then
        idle_sec=$((idle_sec + POLL_INTERVAL))
    fi

    idle_min="$(awk -v s="$idle_sec" 'BEGIN{printf "%.1f", s/60}')"
    log "util=${util}% idle=${idle_min}/${IDLE_LIMIT}min armed=${armed} 本次累計≈\$${cost}"

    if [ "$armed" -eq 1 ] && [ "$idle_sec" -ge "$idle_limit_sec" ]; then
        log "GPU 連續閒置 ${IDLE_LIMIT} 分鐘，判定訓練已結束或中斷。本次累計≈\$${cost}"
        if [ "$DRY_RUN" = "1" ]; then
            log "[DRY_RUN] 略過實際停機。"
            exit 0
        fi
        log "執行停機：$API_SCRIPT stop（pod=$RUNPOD_POD_ID）"
        if stop_pod; then
            log "停機指令已送出。"
            exit 0
        fi
        log "[ERROR] 停機失敗。請手動至 RunPod 網頁停止 Pod，並檢查 API 金鑰權限。"
        exit 1
    fi

    sleep "$POLL_INTERVAL"
done
