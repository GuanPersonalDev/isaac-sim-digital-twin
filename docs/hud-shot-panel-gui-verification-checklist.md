# HUD 擊球參數控制面板 — GUI 人工確認清單

**狀態：先決條件跑了一半。** 2026-09-08 已在 `C:/Other/OmniverseProjects/isaac`
（獨立安裝 Isaac Sim 6.0.0）這個真實環境跑過 `probe_omni_ui_shot_panel_
widgets.py`（headless）與 `verify_manual_controller_wiring.py`（headless，
含真實一次完整擊球），**兩支都全數 PASS**，過程中揪出並修掉兩個真實 bug
（腳本獨立執行時的 import 順序、驗證腳本的擺位斷言量測時機），細節見
`docs/tech-design/hud-shot-control-panel-tech-design.md` 第 1.12／6.6 節與
commit `23f6ffb`。**但 `_to_local()` 依賴的滑鼠事件座標系、以及「overlay
拖曳會不會被相機操作吃掉」這兩件事，headless 原理上量不出來**，仍然只能
靠 `probe_viewport_overlay_drag.py` 在 GUI 下親手拖過才算數——這是本清單
剩下唯一還沒滿足的先決條件，滿足之後才能進入下面兩組人工確認。

## 先決條件：`probe_viewport_overlay_drag.py` 必須在 GUI 環境跑過

`hud_panel.py` 有兩處明確標成「依賴 spike 未證實的假設」（見該檔案級
docstring 與 `docs/tech-design/hud-shot-control-panel-tech-design.md` 第
1.1／1.2／1.5／1.12 節）：

1. `_create_root_frame()` — `viewport_window.get_frame(ext_id)` 放一般 2D
   widget（不是 `sc.SceneView`）能不能正常建構、能不能收到滑鼠事件——**這
   一半已經在 headless 下實測確認可行**（`probe_omni_ui_shot_panel_
   widgets.py` 區塊 6／7）；但 overlay 上掛了滑鼠 callback 的 widget 會不
   會被 viewport 相機操作吃掉，headless 沒有真實滑鼠事件，**仍是全計畫
   最大未知**。
2. `_to_local()` — `set_mouse_pressed_fn`/`set_mouse_moved_fn` 收到的 x/y
   到底是螢幕座標還是 widget local 座標，官方文件完全沒有陳述，目前實作
   假設是螢幕座標（用 `screen_position_x/y` 相減）——**仍待實測**，headless
   下 `screen_position_x/y` 卡在 0（見 1.12 節），量不出真實座標系。

跑法（`python.bat`，不是 pip venv 的 `Scripts/python.exe`；路徑依環境
可能不同，2026-09-08 實測確認的路徑是 `C:/Other/OmniverseProjects/isaac`）：

```powershell
$env:ACCEPT_EULA="Y"
$env:PRIVACY_CONSENT="Y"
$env:OMNI_KIT_ACCEPT_EULA="YES"
$env:ISAACSIM_ACCEPT_EULA="YES"

& "C:\Other\OmniverseProjects\isaac\python.bat" `
  "C:\Other\OmniverseProjects\isaac-sim-digital-twin\scripts\probe_viewport_overlay_drag.py"
```

必須在**有畫面**的環境（不加 `headless: True`，腳本本身就是這樣寫的）
親手拖一次。

- [x] `probe_omni_ui_shot_panel_widgets.py` 全部區塊印出結果，沒有拋例外
      （2026-09-08 headless 實測，11 個區塊全過，結果見 tech-design 1.12 節）
- [ ] `probe_viewport_overlay_drag.py` 親手在最小 overlay 上拖曳過，確認
      滑鼠事件的座標系（螢幕座標 vs widget local）與 `_to_local()` 目前的
      假設是否一致
- [ ] 確認 overlay 上拖曳是否會被 viewport 相機操作吃掉；若會，依計畫書
      風險表的備援順序處理（① 停用相機互動 ② 加吃事件的透明層——主方案
      的 catcher `Rectangle` 已經是這個設計 ③ 退回獨立 `ui.Window`）
- [ ] 若上述任一項與程式碼假設不符，**先回頭調整 `_create_root_frame()`
      或 `_to_local()` 這兩個隔離方法之一**，兩者以外的程式碼不應該需要
      跟著動；調整後再繼續下面兩組確認

只有這一組全部打勾，下面兩組才有意義——面板若因座標系假設錯誤而讀到
錯誤的滑鼠位置，後續所有互動類確認項目都會誤判。

## 跑法（下面兩組確認共用）

用固定時間跑一段、不需要人在旁邊即時盯畫面，跑完再讀 log；操作類項目仍需
真人在畫面前手動拖曳/點擊：

```powershell
$env:ACCEPT_EULA="Y"
$env:PRIVACY_CONSENT="Y"
$env:OMNI_KIT_ACCEPT_EULA="YES"
$env:ISAACSIM_ACCEPT_EULA="YES"
$env:BILLIARD_DEBUG_LOG_PATH="C:\Other\OmniverseProjects\isaac-sim-digital-twin\billiard_gui.log"
$env:BILLIARD_AUTO_PLAY_DELAY_SEC="3"

& "C:\Other\OmniverseProjects\isaac\isaac-sim.bat" `
  --ext-folder "C:\Other\OmniverseProjects\isaac-sim-digital-twin\extension" `
  --enable billiard_digital_twin `
  --/app/asyncRendering=true --/app/asyncRenderingLowLatency=true
```

⚠️ 這是獨立安裝的 Isaac Sim（`isaac-sim.bat`，不是 pip venv 的
`Scripts/isaacsim.exe`）——路徑因環境而異，2026-09-08 實測確認的路徑是
`C:/Other/OmniverseProjects/isaac`。`isaac-sim.bat` 本身就是
`kit.exe apps/isaacsim.exp.full.kit %*` 的包裝（已內建 `VK_ICD_FILENAMES`
修正），後面接的參數會原樣傳給 `kit.exe`，用法不變；**這條命令本身尚未
實際跑過**，只確認過 `isaac-sim.bat` 的內容邏輯符合預期，跑之前留意一下。

開啟後 Demo 桌預設會出現在 Viewport，`HudPanel` 疊在 Viewport 左下角。
`BILLIARD_AUTO_PLAY_DELAY_SEC=3` 會自動按 Play；不想自動播放就拿掉這行、
自己按 Play。

## overlay 專屬確認項目

- [ ] 面板疊在 3D 畫面**左下角**，背板半透明、看得到後面的場景（不是不
      透明方塊擋住畫面）
- [ ] **在面板上拖曳時，viewport 相機不會跟著轉**（拖圓形擊球點選擇器
      或俯瞰圖時，鏡頭視角保持不動）
- [ ] 在面板**以外**的區域拖曳，viewport 相機正常操作（平移/旋轉/縮放
      不受面板影響）
- [ ] 收合鈕（「▼」）能把面板收成一條窄橫幅，只留標題列；再點一次
      （「▶」）展開後，先前調整過的數值（力道、角度、擺位等）都還在，
      沒有被重置
- [ ] viewport 縮放或最大化時，面板跟著重排，不會跑版、疊字或超出畫面
- [ ] Debug Menu 仍是獨立停靠視窗，跟 HudPanel（overlay）互不干擾——
      兩者可以同時開著操作，互相不會擋住對方的可互動區域

## 功能確認項目

- [ ] 圓形擊球點選擇器：把標記拖出圓外，標記**貼在圓周**上（不是被切成
      方角，也不是跑到圓外）
- [ ] 上塞（標記往上拖）讓母球在 STRIKE 時明顯**前進**（正旋），下塞
      （標記往下拖）讓母球明顯**回縮**（倒旋），肉眼可辨
- [ ] 左右塞（標記左右拖）讓母球碰庫後路線明顯偏折（不是走直線）
- [ ] 力道輸入框：輸入 `10` 送出後回填為 `3.3392`；輸入 `0` 送出後回填
      為 `0.65`；輸入非數字字元（例如字母）不會讓面板崩潰或拋例外
- [ ] 俯瞰圖拖母球：只能在 Kitchen 綠色合法區內移動；拖出綠色區域時
      貼在區域邊界上（不會被拖到桌面其他地方）
- [ ] 俯瞰圖點擊定角度：點擊處決定瞄準線方向，瞄準線**指向點擊的那個
      方向**；0° 時瞄準線正對球堆；±90° 時瞄準線指向兩側長庫
- [ ] 按一次「擊球」只出一桿；等 30 秒不會自動出第二桿（`ManualController`
      沒有排隊/自動循環機制）
- [ ] 完全不按任何按鈕，手臂在 30 秒內保持靜止（IDLE 狀態恆定）
- [ ] **調整參數的當下球沒有被重擺、手臂沒有歸位**——證明沒有觸發
      `full_reset()`（`set_manual_shot_parameters()` 不經過 controller
      swap，見 `_build_controller_for_mode()` docstring）
- [ ] Debug Menu 的 AI/Manual 來回切換後，面板上的參數讀數保留（沒有被
      重設回 `ManualShotParameters.default()`）
- [ ] 把母球拖到俯瞰圖 y 下界（Kitchen 最靠近開球端的邊界）：瞄準線變成
      紅色（不可行）、「擊球」按鈕被擋下（disabled，點擊沒有反應）
- [ ] 觸發 ERROR 狀態後，按「重設球局」能讓狀態機回到 IDLE
- [ ] Timeline Stop → Play 之後，面板上的參數讀數保留（不會被重置成
      預設值）
- [ ] Demo toggle 關閉後，面板的選桌下拉自動清空、後續操作（拖曳/輸入/
      按鈕）不拋例外——**面板本身不會消失**，它是跟 Debug Menu 一樣常駐
      在 `_billiard_init()` 建立、只在 `on_shutdown()` 銷毀的元件，Demo
      關閉只是沒有可操作的桌子

## 判定

全部項目（含先決條件那一組）打勾即視為 #115 階段 6 GUI 確認完成，可進
pre-PR review。若有任何一項沒過，記錄實際現象（哪個控制項、哪個數值、
log 裡對應的內容）回報，不要憑印象口頭描述——尤其是「先決條件」那一組，
若座標系假設錯誤，後面所有互動類項目的失敗現象都會是這個根因的表徵，
不需要逐項另外除錯。
