# UR10e 重新設計計畫 — 步驟 9 GUI 人工確認清單

計畫（`ancient-skipping-wand.md`）步驟 1-9 中，1-8 與「切換 `_ROBOT_ARM_CLASS`」
已完成並在無頭模式（headless）用 `scripts/test_ur10e_table_flat.py`／
`test_ur10e_table_bridge.py`／`verify_ur10e_production_wiring.py` 驗證通過
（見 `docs/CHANGELOG.md`）。唯一剩下的是計畫要求的「開 GUI 做最終肉眼＋log
雙重確認」——這件事必須 headful，Claude Code 在這個環境跑不了，需要人工執行。

## 跑法

用固定時間跑一段、不需要人在旁邊即時盯畫面，跑完再讀 log：

PowerShell：

```powershell
$env:ACCEPT_EULA="Y"
$env:PRIVACY_CONSENT="Y"
$env:OMNI_KIT_ACCEPT_EULA="YES"
$env:ISAACSIM_ACCEPT_EULA="YES"
$env:BILLIARD_DEBUG_LOG_PATH="C:\Users\Kuan\isaac-project\isaac-sim-digital-twin\billiard_gui.log"
$env:BILLIARD_AUTO_PLAY_DELAY_SEC="3"

& "C:\Users\Kuan\isaac-project\venv\Scripts\isaacsim.exe" isaacsim.exp.full.kit `
  --ext-folder "C:\Users\Kuan\isaac-project\isaac-sim-digital-twin\extension" `
  --enable billiard_digital_twin
```

Git Bash：

```bash
ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
BILLIARD_DEBUG_LOG_PATH="/c/Users/Kuan/isaac-project/isaac-sim-digital-twin/billiard_gui.log" \
BILLIARD_AUTO_PLAY_DELAY_SEC=3 \
"/c/Users/Kuan/isaac-project/venv/Scripts/isaacsim.exe" isaacsim.exp.full.kit \
  --ext-folder "/c/Users/Kuan/isaac-project/isaac-sim-digital-twin/extension" \
  --enable billiard_digital_twin
```

`isaacsim.exe` 是 pip 安裝的 Isaac Sim 套件本身的 console-script
（`isaacsim = isaacsim:main`），第一個參數是要載入的 Kit experience 檔，
`isaacsim.exp.full.kit` 是完整 GUI 版本（有 viewport／選單列，不是
headless）。`--ext-folder` 指到 `extension/`（其下的 `billiard_digital_twin/`
才是實際的 extension root，含 `config/extension.toml`），`--enable
billiard_digital_twin` 對應該檔案 `[[python.module]] name =
"billiard_digital_twin"`。`BILLIARD_AUTO_PLAY_DELAY_SEC=3` 會在開啟 3 秒後
自動按 Play；不想自動播放就拿掉這行、自己按 Play。不需要 log 就把
`BILLIARD_DEBUG_LOG_PATH` 那行拿掉。

`BILLIARD_AUTO_PLAY_DELAY_SEC` 讓 timeline 在延遲後自動 Play，不用手動點；
`BILLIARD_DEBUG_LOG_PATH` 逐 tick 寫狀態機狀態／母球座標／桿尖世界座標朝向／
關節角度，並在碰撞事件發生時額外寫一行 `CONTACT`（跟同一個 tick 計數器
對照，可以直接看「卡住的那個 tick 是不是同時有碰撞」）。

## 肉眼要看的

- [ ] 手臂從 HOME 開始 AIM，路徑看起來平順，沒有詭異的大幅度甩動或抖動
- [ ] STRIKE 時球桿沿滑軌軸向前推，速度看起來合理（不是瞬間貼合或穿模）
- [ ] **揮桿後球桿確實沿軸縮回，同時手臂有明顯上抬離開母球高度**——這是
      決策 5 的最終版本（使用者在看到母球彈回實測數據後改的，跟計畫文件
      原文「手臂保持靜止」不同，見 `docs/CHANGELOG.md` 的
      `post_strike_retract` 一節）
- [ ] 母球被打出去的方向/力道看起來符合預期，沒有明顯的二次觸桿或蹭球
- [ ] 手臂沒有任何部位穿過球檯本體或明顯貼近庫邊
- [ ] 連續跑多局（RESET→AIM→STRIKE→RESET 循環）沒有卡死或姿態發散

## log 要對照的重點

跑完後讀 `BILLIARD_DEBUG_LOG_PATH` 指定的檔案，一行代表一個 tick：

```
tick=<N> table=<id> state=<STATE> is_motion_complete=<bool> has_error=<bool>
cue_ball=<[x,y]> tip_pos=<[x,y,z]> tip_orient=<[qw,qx,qy,qz]> dof_positions=<[...]>
```

碰撞事件另起一行：

```
tick=<N> CONTACT a=<path> b=<path> collider_a=<path> collider_b=<path> impulse=<float>
```

- [ ] **`has_error` 全程都是 `False`**——`DemoTableOrchestrator._check_downstream_
      failure()` 在 `did_last_motion_timeout()` 為真時會標記錯誤，出現
      `True` 代表 RESET／AIM／STRIKE 某一段逾時未收斂
- [ ] 每一局的 `state` 序列完整跑過 `RESET→AIMING→STRIKING→RESET`，沒有
      卡在同一個 `state` 不動（用 `tick` 差值看某個 state 停留了幾個 tick，
      對照 headless 驗收腳本的量級：flat 案例 RESET 約 900 步、AIM 約
      2000 步、STRIKE 約 40 步——GUI 差異在合理範圍內即可，不用要求完全
      一致）
- [ ] 篩出 `CONTACT` 行，統計每一局 `CueStick`↔母球（`Ball_0`，實際 prim
      path 依 log 裡的實際命名為準）的碰撞事件數，**每局恰好 1 次**
      （決策 7；`impulse` 極小的雜訊事件可以排除，用跟 headless 腳本相同
      的 `impulse > 0` 門檻判斷）
- [ ] 篩出手臂本體相關 prim（`forearm_link`／`wrist_1_link`／`wrist_2_link`／
      `wrist_3_link`）的碰撞事件，**應為 0 筆**（決策 6）

## 判定

全部項目打勾即視為步驟 9 完成、計畫（`ancient-skipping-wand.md`）全部
9 個步驟結案。若有任何一項沒過，記錄下實際現象（哪個 tick、哪個 state、
log 裡對應的數值）回報，不要憑印象口頭描述——這個計畫從頭到尾的教訓都是
「理論上應該沒問題」在實測時經常被推翻（見 `docs/CHANGELOG.md` 開頭
決策 6 的說明）。
