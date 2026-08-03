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
# 可調參數（皆為環境變數）：
#   IDLE_LIMIT      連續閒置幾分鐘後停機（預設 30）
#   UTIL_THRESHOLD  GPU 使用率低於多少視為閒置（%，預設 5）
#   POLL_INTERVAL   取樣間隔（秒，預設 60）
#   HOURLY_RATE     GPU 時價，僅用於 log 顯示累計花費（預設 0.25）
#   LOG             log 路徑（預設 /workspace/watchdog.log）
#   DRY_RUN         設為 1 則只記錄不真的停機

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

if [ "$DRY_RUN" != "1" ] && ! command -v runpodctl >/dev/null 2>&1; then
    log "[FATAL] 找不到 runpodctl。安裝：bash <(wget -qO- cli.runpod.io)"
    log "        並設定金鑰：runpodctl config --apiKey \"\$RUNPOD_API_KEY\""
    exit 1
fi

# RunPod 兩種指令形式在文件中都出現過，兩種都試。
stop_pod() {
    if runpodctl stop pod "$RUNPOD_POD_ID" >>"$LOG" 2>&1; then return 0; fi
    log "[WARN] 'runpodctl stop pod' 失敗，改試 'runpodctl pod stop'。"
    if runpodctl pod stop "$RUNPOD_POD_ID" >>"$LOG" 2>&1; then return 0; fi
    return 1
}

# 取所有 GPU 中的最高使用率（單卡時等同該卡數值）。
gpu_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+$' | sort -rn | head -1
}

log "=== watchdog 啟動 pod=$RUNPOD_POD_ID idle_limit=${IDLE_LIMIT}min threshold=${UTIL_THRESHOLD}% dry_run=${DRY_RUN} ==="
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
        log "執行停機：runpodctl stop pod $RUNPOD_POD_ID"
        if stop_pod; then
            log "停機指令已送出。"
            exit 0
        fi
        log "[ERROR] 停機失敗。請手動至 RunPod 網頁停止 Pod，並檢查 runpodctl 金鑰設定。"
        exit 1
    fi

    sleep "$POLL_INTERVAL"
done
