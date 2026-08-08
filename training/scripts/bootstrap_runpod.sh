#!/usr/bin/env bash
# RunPod Network Volume 從零建置。此檔必須是 LF 換行（見 training/.gitattributes）。
#
# 什麼時候需要跑這支：
#   - Volume 被回收（餘額不足時真的會發生，checkpoint 會一起消失）
#   - 換 datacenter 重建（volume 綁死區域，跨區等於重來）
#   - 想在別的區域再開一套環境
#
# 平常開機**不要**跑這支，開機只要 `source /workspace/setup.sh`（本腳本會產生它）。
#
# ⚠️ 本腳本是把 training/README.md〈踩過的坑〉可執行化的產物，逐步對應 2026-08-02
#    那輪建置的實測結論，但**尚未在真實的空 volume 上跑過**。首次使用請逐段確認
#    輸出，不要無人看管地放著跑。
#
# 用法（在 Pod 的 shell 內，volume 已掛在 /workspace）：
#   bash /path/to/bootstrap_runpod.sh
#
# 每個階段都會先檢查是否已完成，可以安全重跑；中途失敗修好後直接再執行即可。
#
# 可調參數（環境變數）：
#   WORKSPACE          預設 /workspace
#   ISAACSIM_VERSION   預設 6.0.1.0
#   ISAACLAB_BRANCH    預設 release/3.0.0-beta2
#   REPO_URL / BRANCH  專案 repo 與分支
#   SKIP_VERIFY        設為 1 則跳過最後的 torch/cuda 驗證
#   SETUP_SH_ONLY      設為 1 則只重產 setup.sh 後結束，不做任何建置。
#                      改完本檔的 setup.sh 內容後，用這個同步到 volume：
#                        SETUP_SH_ONLY=1 bash training/scripts/bootstrap_runpod.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-6.0.1.0}"
ISAACLAB_BRANCH="${ISAACLAB_BRANCH:-release/3.0.0-beta2}"
REPO_URL="${REPO_URL:-https://github.com/GuanPersonalDev/isaac-sim-digital-twin.git}"
BRANCH="${BRANCH:-main}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
SETUP_SH_ONLY="${SETUP_SH_ONLY:-0}"

VENV="${WORKSPACE}/venv"
ISAACLAB_ROOT="${WORKSPACE}/IsaacLab"
PROJECT_ROOT="${WORKSPACE}/isaac-sim-digital-twin"
SETUP_SH="${WORKSPACE}/setup.sh"
SSH_KEY="${WORKSPACE}/.ssh/id_ed25519"
FREEZE_FILE="${WORKSPACE}/pip-freeze-working.txt"

say() { echo "[bootstrap] $*"; }

# --- 0. 前置檢查 ---------------------------------------------------------

if [[ ! -d "${WORKSPACE}" ]]; then
    echo "[bootstrap] 找不到 ${WORKSPACE}，確認 Network Volume 已掛載。" >&2
    exit 1
fi

# Pod 的 start command 會裝 openssh-server，與這裡的 apt 搶同一把 dpkg lock。
# 開機後太快動手，apt 會整段失敗而後續全部連鎖崩掉。
# 不要 kill 對方或刪 lock 檔——中斷 dpkg 會留下半裝狀態，之後每個 apt
# 指令都要先 dpkg --configure -a 才動得了。
wait_for_apt() {
    local waited=0
    while pgrep -x apt-get >/dev/null || pgrep -x apt >/dev/null || pgrep -x dpkg >/dev/null; do
        if [[ "${waited}" -eq 0 ]]; then
            say "等待 Pod start command 的 apt 結束（不要 kill 它）…"
        fi
        sleep 5
        waited=$((waited + 5))
        if [[ "${waited}" -ge 600 ]]; then
            echo "[bootstrap] apt 超過 10 分鐘仍未結束，請手動檢查。" >&2
            exit 1
        fi
    done
}

# setup.sh 放在 volume 上而不是 repo 裡，是因為它要在「repo 還沒 clone 下來」
# 的時候就能用。內容由本函式產生，改動請改這裡，否則重建 volume 時會失傳。
# 改完之後用 SETUP_SH_ONLY=1 重跑本腳本，即可同步到 volume 上的 setup.sh。
write_setup_sh() {
cat > "${SETUP_SH}" <<'SETUP_EOF'
# 每次開機第一件事：source /workspace/setup.sh
#
# ⚠️ 只能用 source，不可用 ./setup.sh —— venv 啟用必須作用在當前 shell，
#    用 ./ 會開子行程，activate 完就隨子行程消失，但畫面上看起來「成功了」。
#    本檔刻意不給執行位元作為防呆。
#
# 由 training/scripts/bootstrap_runpod.sh 產生。改這裡記得同步回 repo。

# Pod 的 start command 會裝 openssh-server，跟下面的 apt 搶同一把 dpkg lock。
# 太快動手會讓 apt 整段失敗 → git / python3-dev 沒裝回來 → 系統 python3.12
# 不存在 → venv 的 bin/python 變成斷掉的 symlink，症狀是 (venv) 前綴有出來
# 但 python: command not found。不要 kill 對方，也不要刪 lock 檔。
while pgrep -x apt-get >/dev/null || pgrep -x apt >/dev/null || pgrep -x dpkg >/dev/null; do
    echo "[setup] 等待 Pod start command 的 apt 結束…"
    sleep 5
done

# container disk 停機即清空，這些每次開機都要重裝
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git tmux wget build-essential python3-dev python3-venv

# pip cache 放 container disk，不要吃掉 volume 配額
export PIP_CACHE_DIR=/tmp/pipcache

# venv 在 volume 上，Python 套件停機後仍在，不需要重裝任何東西
source /workspace/venv/bin/activate

# rl_task 用 pip install -e 安裝，但 Isaac Lab 模板的 setup.py 只把 task package
# 列進 packages，不含 core。repo root 不進 PYTHONPATH 的話，BilliardEnv 的
# `from core.services...` 會在 isaaclab.sh train 起動時炸 ModuleNotFoundError。
#
# 用 ${PYTHONPATH:+:...} 而不是 :${PYTHONPATH}：後者在 PYTHONPATH 未設時會產生
# 結尾冒號，Python 把空路徑元素當成當前目錄。train 的 CWD 不固定（見 #121 E-2），
# 讓 CWD 混進 import path 之後很難查。
export PYTHONPATH=/workspace/isaac-sim-digital-twin${PYTHONPATH:+:${PYTHONPATH}}

# --- git over SSH --------------------------------------------------------
# GitHub 不收密碼認證，HTTPS push 一定失敗，所以用 deploy key 走 SSH。
#
# 私鑰放 volume 才能跨 pod 存活，但 volume 是網路磁碟，寫出來的檔案權限固定是
# 0666，OpenSSH 會判定 UNPROTECTED PRIVATE KEY FILE 而拒用（症狀是 push 時
# Permission denied (publickey)）。chmod 對這個掛載無效，所以每次開機複製一份
# 到 container disk 的 /root/.ssh 再收緊權限。私鑰本體不需重產。
if [ -f /workspace/.ssh/id_ed25519 ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    cp -f /workspace/.ssh/id_ed25519 /root/.ssh/id_ed25519
    chmod 600 /root/.ssh/id_ed25519
    # 用環境變數而不是 git config core.sshCommand：後者寫在 .git/config，
    # volume 重建就沒了，而且只對這一個 repo 生效。
    export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new"
    # bootstrap 是用 HTTPS clone 的（repo 公開，讀不需要憑證），但 HTTPS 推不
    # 上去。這行把 remote 轉成 SSH，重複執行無副作用。
    git -C /workspace/isaac-sim-digital-twin remote set-url origin \
        git@github.com:GuanPersonalDev/isaac-sim-digital-twin.git
else
    echo "[setup] ⚠️ 找不到 /workspace/.ssh/id_ed25519，git 操作會失敗。"
    echo "[setup]    重新產生金鑰： bash /workspace/isaac-sim-digital-twin/training/scripts/bootstrap_runpod.sh"
fi

# commit 身分寫在 .git/config，volume 重建會消失，補上以免 commit 成 root@<pod id>。
git -C /workspace/isaac-sim-digital-twin config user.name "GuanPersonalDev"
git -C /workspace/isaac-sim-digital-twin config user.email "guanpersonalproject@gmail.com"

if git -C /workspace/isaac-sim-digital-twin pull --ff-only; then
    cd /workspace/isaac-sim-digital-twin || return
else
    # 失敗卻繼續往下印 ready，很容易被誤判成一切正常，所以講明白。
    echo "[setup] ⚠️ git pull 失敗——程式碼不是最新版，先處理再開始工作。"
fi

echo "--- ready ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
SETUP_EOF

# 不給執行位元：逼使用者用 source，避免 activate 在子行程中失效。
chmod 644 "${SETUP_SH}"
}

# SETUP_SH_ONLY=1：只把上面的內容寫到 volume 上，不碰 apt / pip / repo。
# 用在「改了本檔的 setup.sh 內容，要讓 pod 立刻吃到」的情境。
if [[ "${SETUP_SH_ONLY}" == "1" ]]; then
    say "SETUP_SH_ONLY=1：只重產 ${SETUP_SH}"
    write_setup_sh
    say "完成。請重新執行： source ${SETUP_SH}"
    exit 0
fi

# --- 1. 系統套件 ---------------------------------------------------------
# 這些都在 container disk 上，停機即清空——所以 setup.sh 每次開機也要重裝一次。
# python3-dev 必裝，否則 imgui 編譯失敗（缺 Python.h）會擋住 isaaclab_teleop。
# python3-venv 是建 venv 必需（Ubuntu 把 venv 從 stdlib 拆成獨立套件）。

say "階段 1/7：安裝系統套件"
wait_for_apt
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git tmux wget build-essential python3-dev python3-venv

PYTHON_SYS="$(command -v python3.12 || command -v python3)"
say "系統 python = ${PYTHON_SYS} ($(${PYTHON_SYS} -V 2>&1))"

# Isaac Lab 硬限制 >=3.12,<3.13；Ubuntu 24.04 的系統 Python 剛好命中。
if ! "${PYTHON_SYS}" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "[bootstrap] Python 版本不是 3.12，Isaac Lab 會裝不起來。" >&2
    exit 1
fi

# pip cache 放 container disk。放 /workspace 會吃掉 volume 配額，
# 曾因此觸發 Errno 122 Disk quota exceeded。venv 本身持久化，本來就不需重裝。
export PIP_CACHE_DIR=/tmp/pipcache
mkdir -p "${PIP_CACHE_DIR}"

# --- 2. venv -------------------------------------------------------------

say "階段 2/7：建立 venv"
if [[ -x "${VENV}/bin/python" ]]; then
    say "venv 已存在，跳過"
else
    "${PYTHON_SYS}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip

# --- 3. isaacsim（必須在 Isaac Lab 之前）---------------------------------
# ⚠️ 順序不可顛倒。反過來裝的話 isaacsim 會把 torch 換成 2.11.0（CUDA 13），
#    而 CUDA 13 需要夠新的驅動——570.x 不支援，torch.cuda.is_available() 會變 False。
#    另外 nvidia-cublas-cu12 與 nvidia-cublas（CUDA 13 世代拿掉後綴）是不同套件，
#    pip 不會互相取代，混裝會讓 venv 膨脹到 60GB 以上。

say "階段 3/7：安裝 isaacsim ${ISAACSIM_VERSION}"
if python -c 'import isaacsim' >/dev/null 2>&1; then
    say "isaacsim 已安裝，跳過"
else
    pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
        --extra-index-url https://pypi.nvidia.com
fi

# --- 4. Isaac Lab --------------------------------------------------------

say "階段 4/7：取得 Isaac Lab (${ISAACLAB_BRANCH})"
if [[ -d "${ISAACLAB_ROOT}/.git" ]]; then
    say "IsaacLab 已存在，跳過 clone"
else
    git clone --branch "${ISAACLAB_BRANCH}" --depth 1 \
        https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_ROOT}"
fi

if python -c 'import isaaclab' >/dev/null 2>&1; then
    say "isaaclab 已安裝，跳過"
else
    say "執行 isaaclab.sh --install（會跑一陣子）"
    (cd "${ISAACLAB_ROOT}" && ./isaaclab.sh --install)
fi

# isaacsim 與 isaaclab 之間有版本 pin 衝突（websockets / psutil / click / coverage）。
# 不要手動對版本——硬降 websockets 會反過來弄壞 viser / rerun。用 smoke test 實測。

# --- 5. 專案 repo --------------------------------------------------------

say "階段 5/7：取得專案 repo"

# push 要用 deploy key（GitHub 不收密碼認證）。私鑰放 volume 才能跨 pod 存活，
# 公鑰必須人工貼到 GitHub，所以這裡只產生並印出來，不當作失敗條件。
if [[ -f "${SSH_KEY}" ]]; then
    say "SSH 金鑰已存在，跳過"
else
    mkdir -p "$(dirname "${SSH_KEY}")"
    ssh-keygen -t ed25519 -f "${SSH_KEY}" -N "" -C "runpod"
    say "───────────────────────────────────────────────────────────────"
    say "請把下面這行公鑰加到 GitHub，並勾 Allow write access："
    say "  https://github.com/GuanPersonalDev/isaac-sim-digital-twin/settings/keys"
    cat "${SSH_KEY}.pub"
    say "───────────────────────────────────────────────────────────────"
fi

if [[ -d "${PROJECT_ROOT}/.git" ]]; then
    git -C "${PROJECT_ROOT}" fetch --prune origin
    git -C "${PROJECT_ROOT}" checkout "${BRANCH}"
    # --ff-only：pod 上若有本機修改導致分歧，明確失敗而不是靜默覆蓋。
    git -C "${PROJECT_ROOT}" pull --ff-only origin "${BRANCH}"
else
    # repo 是 public，不需要任何憑證。
    git clone --branch "${BRANCH}" "${REPO_URL}" "${PROJECT_ROOT}"
fi

if [[ -f "${PROJECT_ROOT}/training/requirements.txt" ]]; then
    pip install -r "${PROJECT_ROOT}/training/requirements.txt"
fi

# --- 6. 產生 setup.sh ----------------------------------------------------
# 實際內容在檔案上方的 write_setup_sh()，改那裡。

say "階段 6/7：產生 ${SETUP_SH}"
write_setup_sh

# --- 7. 快照與驗證 -------------------------------------------------------

say "階段 7/7：存回退點並驗證"
pip freeze > "${FREEZE_FILE}"
say "已寫入 ${FREEZE_FILE}（動任何套件之前先重存一份）"

if [[ "${SKIP_VERIFY}" != "1" ]]; then
    python - <<'VERIFY_EOF'
import torch

print("[verify] torch     :", torch.__version__)
print("[verify] cuda      :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[verify] device    :", torch.cuda.get_device_name(0))
    print("[verify] capability:", torch.cuda.get_device_capability())
    print("[verify] arch_list :", torch.cuda.get_arch_list())
    # available() 為 True 完全不保證 kernel 跑得動：torch 若沒為這張卡的
    # compute capability 編 kernel，會在實際運算時才炸（或靜默算錯）。
    x = torch.randn(4096, 4096, device="cuda")
    print("[verify] matmul    :", (x @ x).sum().item())
VERIFY_EOF

    python -c "import isaacsim, isaaclab; print('[verify] isaacsim/isaaclab import OK')"
fi

say "建置完成。"
say "開機後請執行： source ${SETUP_SH}"
say "Isaac Sim smoke test： cd ${ISAACLAB_ROOT} && ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py"
