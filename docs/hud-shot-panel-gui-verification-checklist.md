# HUD 擊球參數控制面板 — GUI 人工確認清單

**狀態（2026-09-10 更新）：先決條件與大部分互動項目已在真實 GUI 下確認，
仍有一批未觸及的項目列在下面，逐項標註證據來源。** Issue #115 本身的兩條
正式完成標準（面板顯示正常、調參數立即反映到下次擊球）已經達成並關閉
（commit `944f6e9`／`a3a6947`），本清單顆粒度比 Issue 本身細很多，未打勾
的項目**不影響 #115 關閉**，是留給之後（或下一輪空檔）補測的清單，不是
阻擋項。

2026-09-08 已在 `C:/Other/OmniverseProjects/isaac`（獨立安裝 Isaac Sim
6.0.0）這個真實環境跑過 `probe_omni_ui_shot_panel_widgets.py`（headless）
與 `verify_manual_controller_wiring.py`（headless，含真實一次完整擊球），
**兩支都全數 PASS**，過程中揪出並修掉兩個真實 bug（腳本獨立執行時的
import 順序、驗證腳本的擺位斷言量測時機），細節見
`docs/tech-design/hud-shot-control-panel-tech-design.md` 第 1.12／6.6 節與
commit `23f6ffb`。2026-09-09～09-10 兩輪真人 GUI 疊代（面板定位/捲動/字型/
透明度、Kitchen 矩形對齊、Controller 模式顯示）過程中，`_to_local()` 的
螢幕座標假設與「overlay 拖曳會不會被相機操作吃掉」這兩件事已經透過**實際
產品程式碼的大量真實拖曳操作**間接驗證（見下方先決條件小節的說明），沒有
另外跑專用的 `probe_viewport_overlay_drag.py`。

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
- [x] 滑鼠事件座標系與 `_to_local()` 的假設一致——**沒有另外跑
      `probe_viewport_overlay_drag.py`**，改用等價、證據更強的路徑驗證：
      2026-09-09～09-10 兩輪真人 GUI 疊代裡，使用者實際拖曳圓形擊球點
      選擇器與俯瞰圖母球/角度數十次，回報的座標/角度都跟預期一致（含
      逐步排查 Kitchen 矩形偏移那次，過程中反覆用滑鼠拖曳＋螢幕截圖交叉
      核對像素位置），若座標系假設錯誤，這些互動不可能表現正常
- [x] overlay 拖曳不會被 viewport 相機操作吃掉——2026-09-08 使用者實測
      發現滾輪縮放會穿透（回報「滾輪沒有修好」），改用
      `_on_panel_hovered()`（懸停時透過 `carb.settings` 停用
      `ZoomScrollGesture`）修正後，後續十幾輪操作面板互動元件（圓形
      選擇器、俯瞰圖、按鈕）未再回報相機被誤觸發的情形
- [x] 兩項都與程式碼原本的假設一致，`_create_root_frame()`／`_to_local()`
      不需要調整

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

開啟後 Demo 桌預設會出現在 Viewport，`HudPanel` 疊在 Viewport 右下角
（1/3 寬、1/2 高，位置在疊代過程中從左下角改到右下角，見
docs/CHANGELOG.md）。
`BILLIARD_AUTO_PLAY_DELAY_SEC=3` 會自動按 Play；不想自動播放就拿掉這行、
自己按 Play。

## overlay 專屬確認項目

- [x] 面板疊在 3D 畫面**右下角**，背板半透明、看得到後面的場景（不是不
      透明方塊擋住畫面）——2026-09-09 Opus 深入調查 alpha 混色與背景色
      選擇後，使用者截圖多次確認半透明背板正確顯示、位置正確
- [x] **在面板上拖曳時，viewport 相機不會跟著轉**——見上方先決條件小節
      的 `_on_panel_hovered()` 修正說明
- [ ] 在面板**以外**的區域拖曳，viewport 相機正常操作（未見使用者明確
      回報相機操作異常，但沒有專門測過這一項，留待補測）
- [x] 收合鈕（「v」/「>」）能把面板收成一條窄橫幅，只留標題列；展開後
      調整過的數值不會被重置——收合邏輯本身多輪測試中持續正常運作，
      `_collapsible_body.visible` 切換不影響 controller 端保存的參數
      （面板不持有參數狀態，見 hud_panel.py 檔案級 docstring）
- [ ] viewport 縮放或最大化時，面板跟著重排，不會跑版、疊字或超出畫面
      （未測試）
- [ ] Debug Menu 仍是獨立停靠視窗，跟 HudPanel（overlay）互不干擾（兩者
      這幾輪確實同時開著使用、互不影響，但沒有刻意測試「互相擋住可互動
      區域」這個邊界情況，留待補測）

## 功能確認項目

- [x] 圓形擊球點選擇器：標記不會跑到圓外——圓形裁切邏輯
      `position_offset_limiter.clamp_position_offset()` 有 `core/tests`
      單元測試覆蓋，2026-09-09～10 疊代中使用者多次拖曳標記到圓周附近
      （含把標記半徑從 6px 調到 12px 那次），未回報標記跑出圓外或被切角
- [ ] 上塞/下塞讓母球在 STRIKE 時明顯前進/回縮（肉眼可辨）——未測試
- [ ] 左右塞讓母球碰庫後路線明顯偏折——未測試
- [ ] 力道輸入框：輸入 `10`/`0` 回填為 `3.3392`/`0.65`；非數字字元不崩潰
      ——未測試（`_on_speed_value_changed()` 有 clamp 邏輯，但沒有實際在
      GUI 輸入過這幾個數值核對回填結果）
- [ ] 俯瞰圖拖母球只能在 Kitchen 合法區內移動、拖出去會貼在邊界——
      `clamp_cue_ball_placement()` 本身有 `core/tests` 單元測試覆蓋，但
      沒有在 GUI 裡刻意把母球拖出邊界外核對貼齊行為，留待補測
- [x] 俯瞰圖點擊/拖曳定角度：瞄準線指向游標方向——這幾輪視覺調整（瞄準
      點加密、虛線效果、Kitchen 對齊）反覆用這個互動核對角度變化，行為
      符合預期；`shot_angle_from_points()`/`aim_line_endpoint()` 也有
      `core/tests` 覆蓋往返正確性
- [x] 按一次「擊球」只出一桿，不會自動循環——`verify_manual_controller_
      wiring.py` headless 驗證第 3、4 項明確斷言（不按按鈕 120 tick 恆
      IDLE；`ManualController` 沒有排隊機制）
- [x] 不按任何按鈕，手臂保持靜止——同上，headless 驗證第 3 項
      （120 tick 內狀態集合只有 `{'IDLE'}`，母球世界座標零位移）
- [x] 調整參數當下球沒有被重擺、手臂沒有歸位——headless 驗證「連續推
      20 次參數不觸發重擺球」項目為 True（母球位置擊球前後完全一致）
- [x] Debug Menu 的 AI/Manual 切換後，面板參數讀數保留——架構保證，非
      臆測：`_build_controller_for_mode()` 非 AI 模式一律回傳
      `self._demo_manual_controllers[table_id]` 這個常駐實例，從不重建，
      swap 不會清空已設定的參數；2026-09-10 新增的 Controller 模式顯示/
      切換功能（#115 追加，commit `a3a6947`）測試時使用者也在 HUD 上來回
      切換過 AI/Manual，面板本身沒有異常
- [ ] 母球拖到 Kitchen y 下界時瞄準線變紅、擊球鈕 disable——可行性判斷
      邏輯 `evaluate_manual_shot_parameters()` 有 `core/tests` 覆蓋，但沒
      有在 GUI 刻意拖到邊界核對紅線與按鈕 disable 的視覺表現
- [ ] 觸發 ERROR 後按「重設球局」回到 IDLE——未測試（這幾輪操作都是正常
      擊球流程，沒有刻意誘發 ERROR 狀態）
- [ ] Timeline Stop → Play 後面板參數讀數保留——未測試；但 2026-09-10 才
      修掉「Timeline Stop 時 Debug Menu 輪詢 physics tensor entity 失效
      噴例外」的 bug（commit `944f6e9`），這條路徑目前至少不會再讓面板
      崩潰，讀數是否保留仍待實測
- [ ] Demo toggle 關閉後選桌下拉自動清空、後續操作不拋例外——未測試

## 判定

Issue #115 本身的完成標準已達成並關閉（commit `944f6e9`／`a3a6947`），
不受本清單影響。本清單作為更細顆粒度的回歸測試參考，目前仍有 7 項未測
（力道輸入框回填、上/下/左右塞效果、Kitchen 邊界貼齊、可行性紅線、ERROR
復原、Timeline Stop/Play 持久性、Demo toggle 清空），建議下次有 GUI 操作
空檔時一次補完，不需要現在為了打勾而重新開一輪 Isaac Sim。若之後要重新
走 pre-PR review 等級的完整確認，才需要把這 7 項全部補上。
