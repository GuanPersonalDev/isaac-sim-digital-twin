#!/usr/bin/env python3
"""RunPod REST API 最小客戶端：驗證金鑰、查詢 Pod、停止 Pod。

為什麼不用 runpodctl（2026-08-06 實測後改走這條路）：

1. Pod 自動注入的 ``RUNPOD_API_KEY`` 是 pod-scoped key，權限不含管理 Pod
   生命週期 —— runpodctl 連唯讀的 ``get pod`` 都回 Unauthorized。
2. runpodctl 只打 GraphQL endpoint，而自建的 scoped key 可以只開 REST 權限
   （實測：GraphQL Unauthorized、REST 200）。
3. runpodctl 與 curl 都不在基礎映像內，container disk 停機即清空，等於每次
   開機都要重裝；而 apt 會跟 Pod 的 start command 搶 dpkg lock，開機後太快
   下手就會整段失敗。

venv 在 network volume 上持久化，用標準庫 urllib 打 REST 沒有任何安裝成本，
也不碰 apt。

用法::

    python runpod_api.py check          # 驗證金鑰與權限（唯讀，不動 Pod）
    python runpod_api.py stop           # 停止本 Pod
    python runpod_api.py stop --pod-id <id>

環境變數：

``WATCHDOG_API_KEY``
    RunPod API key，scope 必須包含 Pods 的讀與寫。相容舊名
    ``WATCH_DOG_API_KEY``。
``RUNPOD_POD_ID``
    Pod ID，平台自動注入；可用 ``--pod-id`` 覆寫。
``RUNPOD_API_BASE``
    覆寫 API base URL（預設 ``https://rest.runpod.io/v1``）。RunPod 已將
    REST v1 列為維護模式，遷移到 v2 時改這個即可。

⚠️ 不要使用 ``RUNPOD_API_KEY``。那是平台保留並自動注入的名稱，即使在 Pod
設定裡填了同名變數也會被平台的值蓋掉，症狀是「金鑰格式完全正確卻一直 403」。

停止不是終止：``/workspace``（network volume）完整保留，只有 container disk
會清空。誤判的代價是重啟時間，不是資料。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_API_BASE = "https://rest.runpod.io/v1"
TIMEOUT_SEC = 30

# Pod 上目前設的是 WATCH_DOG_API_KEY，文件建議統一為 WATCHDOG_API_KEY。
# 兩個都接受，避免改名期間 watchdog 靜默失效。
KEY_ENV_NAMES = ("WATCHDOG_API_KEY", "WATCH_DOG_API_KEY")

EXIT_OK = 0
EXIT_CONFIG = 1  # 環境變數缺漏
EXIT_AUTH = 2  # 401/403，金鑰無效或 scope 不足
EXIT_HTTP = 3  # 其他 HTTP 錯誤或網路問題


def resolve_api_key() -> tuple[str, str] | None:
    """回傳 (變數名, 金鑰)，都沒設則回 None。刻意不把 RUNPOD_API_KEY 當備援。

    平台注入的那把權限不足，拿它當備援只會把「忘了設金鑰」偽裝成「權限錯誤」，
    多繞一大圈才查得出來。
    """
    for name in KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return name, value
    return None


def api_request(method: str, path: str, key: str) -> tuple[int, bytes]:
    """送出請求，回傳 (status_code, body)。HTTPError 也一併轉成回傳值。"""
    base = os.environ.get("RUNPOD_API_BASE", DEFAULT_API_BASE).rstrip("/")
    request = urllib.request.Request(
        f"{base}{path}",
        # POST 一律帶空 body：省略時 urllib 不會送 Content-Length，
        # 伺服器可能直接掐掉連線而什麼都不回。
        data=b"" if method == "POST" else None,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def explain_auth_failure(status: int, key_env: str) -> str:
    if status == 401:
        return (
            f"401 認證失敗：${key_env} 的值無效或為空。"
            " 確認 Pod 設定裡的環境變數名稱拼字正確、值為 rpa_ 開頭的 50 字元字串，"
            " 且改完設定後有重啟 Pod（Secret 與環境變數只在啟動時注入）。"
        )
    return (
        f"403 權限不足：${key_env} 是合法金鑰，但 scope 不包含 Pods。"
        " 到 RunPod Settings → API Keys 確認該把 key 有勾選 Pods 的讀與寫。"
        " 若這個值其實來自平台自動注入的 pod-scoped key，請改用其他變數名。"
    )


def truncate(body: bytes, limit: int = 300) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def cmd_check(key_env: str, key: str) -> int:
    status, body = api_request("GET", "/pods", key)

    if status in (401, 403):
        print(explain_auth_failure(status, key_env), file=sys.stderr)
        return EXIT_AUTH
    if status != 200:
        print(f"非預期狀態碼 {status}：{truncate(body)}", file=sys.stderr)
        return EXIT_HTTP

    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    try:
        pods = json.loads(body)
    except json.JSONDecodeError:
        # 權限已經驗證通過，回應格式改變不該讓 watchdog 拒絕啟動。
        print(f"200 OK（金鑰可用）；回應非 JSON：{truncate(body)}")
        return EXIT_OK

    if isinstance(pods, dict):
        pods = pods.get("data", pods.get("pods", []))
    if not isinstance(pods, list):
        pods = []

    print(f"200 OK：${key_env} 對 Pods 有讀取權限，帳號下共 {len(pods)} 顆 Pod。")
    for pod in pods:
        if isinstance(pod, dict) and pod.get("id") == pod_id:
            name = pod.get("name", "?")
            state = pod.get("desiredStatus", pod.get("status", "?"))
            print(f"本 Pod {pod_id} name={name} status={state}")
            break
    else:
        if pod_id:
            # 唯讀權限沒問題就不擋，但要講清楚 —— 停機打的是 /pods/<id>/stop。
            print(
                f"注意：清單中找不到本 Pod（RUNPOD_POD_ID={pod_id}），"
                "停機時可能因 Pod ID 不符而失敗。",
                file=sys.stderr,
            )
    return EXIT_OK


def cmd_stop(args: argparse.Namespace, key_env: str, key: str) -> int:
    pod_id = (args.pod_id or os.environ.get("RUNPOD_POD_ID", "")).strip()
    if not pod_id:
        print(
            "找不到 Pod ID：$RUNPOD_POD_ID 未設定，也沒有給 --pod-id。",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    # 停機成功後 SSH 會當場斷線，晚一步 flush 就永遠看不到這行。
    print(f"停止 Pod {pod_id}…", flush=True)

    status, body = api_request("POST", f"/pods/{pod_id}/stop", key)

    if status in (401, 403):
        print(explain_auth_failure(status, key_env), file=sys.stderr)
        return EXIT_AUTH
    if status == 404:
        print(f"404：找不到 Pod {pod_id}，確認 Pod ID 是否正確。", file=sys.stderr)
        return EXIT_HTTP
    if status not in (200, 201, 202, 204):
        print(f"停機失敗，狀態碼 {status}：{truncate(body)}", file=sys.stderr)
        return EXIT_HTTP

    print(f"{status} 停機指令已送出。", flush=True)
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RunPod REST API 客戶端（驗證金鑰 / 停止 Pod）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="驗證金鑰與 Pods 權限（唯讀）")
    stop_parser = subparsers.add_parser("stop", help="停止 Pod（資料保留）")
    stop_parser.add_argument("--pod-id", default=None, help="預設取 $RUNPOD_POD_ID")

    args = parser.parse_args()

    resolved = resolve_api_key()
    if resolved is None:
        names = " 或 ".join(f"${n}" for n in KEY_ENV_NAMES)
        print(
            f"找不到 API 金鑰：請設定 {names}。\n"
            "⚠️ 不要用 $RUNPOD_API_KEY —— 那是平台自動注入的 pod-scoped key，"
            "權限不含管理 Pod 生命週期，且會覆蓋你在 Pod 設定裡填的同名值。",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    key_env, key = resolved

    try:
        if args.command == "check":
            return cmd_check(key_env, key)
        return cmd_stop(args, key_env, key)
    except urllib.error.URLError as exc:
        print(f"網路錯誤：{exc.reason}", file=sys.stderr)
        return EXIT_HTTP
    except OSError as exc:
        print(f"連線失敗：{exc}", file=sys.stderr)
        return EXIT_HTTP


if __name__ == "__main__":
    sys.exit(main())
