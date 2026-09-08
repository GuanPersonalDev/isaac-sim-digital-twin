# HUD 擊球參數控制面板 — 技術設計文件

> 生成時間：2026-09-08
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：#115
> 關聯文件：`C:\Users\guan_\.claude\plans\115-hud-peppy-harbor.md`（完整技術計畫）、`docs/architecture-spec.md`（分層規則、`extension/ui/hud_panel.py` 檔名已預留）

---

## 0. 文件狀態

第 1 節「omni.ui API 查證結果」是 #115 階段 0（API spike）的產出，內容保持
原樣不動。第 2–6 節是階段 6「驗證與文件」補完的部分，涵蓋階段 1–5 實際落地
的程式碼（`core/models/manual_shot_bounds.py`、`manual_shot_parameters.py`、
`core/services/shot_panel_input_mapper.py`、`manual_shot_feasibility.py`、
`core/controllers/manual_controller.py`、`extension/ui/hud_panel.py`、
`extension/ui/table_combo_box_model.py`、`billiard_digital_twin.py` 的接線）。

- [x] 第 2 節：模組清單與職責
- [x] 第 3 節：類別設計（`ManualShotParameters`／`ManualController`／`HudPanel`／輸入換算純函式……）
- [x] 第 4 節：資料流（UI 執行緒 ↔ physics 執行緒，含 AIM/STRIKE 快照時序）
- [x] 第 5 節：關鍵設計決策與理由
- [x] 第 6 節：測試策略與已知限制
- [x] 第 1.12 節：`probe_omni_ui_shot_panel_widgets.py` 實測結果（2026-09-08，headless）

⚠️ 第 1.0–1.11 節寫於「這台機器沒有 Isaac Sim」的階段，內容全部是**文件查證與推測**，`1.0` 節那句「尚未實際執行」已經過時——2026-09-08 已在 `C:/Other/OmniverseProjects/isaac` 這個獨立安裝的 Isaac Sim 6.0.0 環境跑過 `probe_omni_ui_shot_panel_widgets.py`（headless），結果見新增的 **1.12 節**。1.0–1.11 節本文保持原樣不動（歷史查證過程仍有參考價值），實際數字與結論一律以 1.12 節為準。

---

## 1. omni.ui API 查證結果

### 1.0 查證方式與可信度標示

本節八個項目全部透過 **WebFetch／WebSearch 查詢 NVIDIA 官方文件**（`docs.omniverse.nvidia.com`）與官方 GitHub 範例庫（`NVIDIA-Omniverse/kit-extension-sample-ui-scene`）完成，**沒有在本機執行過任何程式碼**——這台開發機沒有安裝 Isaac Sim（只有一個未解壓的安裝檔），`scripts/probe_omni_ui_shot_panel_widgets.py` 目前只是「寫好、語法正確、等有 Isaac Sim 的環境去跑」的狀態，尚未實際執行。

每一項答案前面標示可信度：

- **【文件確認】**：官方文件或程式碼原始碼有明確逐字說明，直接引用原文。
- **【文件確認＋交叉比對】**：多份獨立文件互相印證。
- **【推測，需實測】**：文件沒有明講，是根據相鄰文件段落、程式碼慣例或間接證據（例如 changelog 條目）推導出的合理假設，**不能當作最終答案**，已經寫進 `scripts/probe_omni_ui_shot_panel_widgets.py` 或 `scripts/probe_viewport_overlay_drag.py` 作為要實測的項目。
- **【文件查不到，需在 spike 實測】**：查了但完全沒有找到任何相關段落，純粹依賴 spike 腳本的實際輸出才能回答。

---

### 1.1 `viewport_window.get_frame(ext_id)` 的回傳型別與生命週期

**簽名與行為【文件確認】**（來源：`omni.kit.viewport.window.ViewportWindow` API 文件）：

```python
get_frame(name: str) -> omni.ui.Frame
```

> "Add a unique `omni.ui.Frame` into the view hierarchy. This will return a newly created `omni.ui.Frame` on the first call for a unique name, or return a previously created `omni.ui.Frame` for the same unique name."

確認的三件事：
1. **回傳型別就是一般 `omni.ui.Frame`**，不是 `omni.ui.scene` 的 `sc.Frame`／`sc.SceneView`，架構文件裡「可以放任何 2D widget」的假設在型別契約層面成立。
2. **同一個 `ext_id`（`name` 參數）重複呼叫回傳同一個 Frame 物件**——技術計畫依賴這個行為做「重建面板時沿用同一個 Frame」，官方文件明講，不是推測。
3. 這個方法**至少從 `omni.kit.viewport.window` 1.0.7（2022-01-31）就存在**（changelog 該版本同時列出 "Fix issues with stats area blocking mouse-events in dead space." 與 `get_frame` 的新增，見 1.2 節），到 Isaac Sim 6.0.0 對應的 Kit 版本已經是延用多年的穩定 API，不是新引入、行為未定的功能。

**【推測，需實測】兩個未確認的點**：
- 放**一般 2D widget（VStack/Rectangle/Button，不是 `sc.SceneView`）** 進 `get_frame()` 回傳的 Frame 裡能不能成功建構、能不能收到滑鼠事件——官方範例（`kit-extension-sample-ui-scene` 的 `omni.example.ui_scene.widget_info` 教學）用 `get_frame()` 只是為了掛 `sc.SceneView`，範例程式碼片段：

  ```python
  with self.viewport_window.get_frame(ext_id):
      self.scene_view = sc.SceneView()
      with self.scene_view.scene:
          WidgetInfoManipulator(model=ObjInfoModel())
  ```

  官方零先例展示「`get_frame()` 直接放一般 2D widget」這種本計畫要用的模式（`sc.SceneView` 內的 widget 用 3D 變換矩陣定位，跟計畫要的 2D overlay 完全是兩回事）。型別契約上 `omni.ui.Frame` 本來就能放任何 widget，但**沒有官方先例證實這個特定組合（viewport overlay + 純 2D widget + 滑鼠互動）沒有隱藏限制**，`scripts/probe_omni_ui_shot_panel_widgets.py` 區塊 7 直接驗證這一點。
- **viewport 重建後 frame 是否還在**——官方文件完全沒有討論 `ViewportWindow` 在 viewport 重建（例如切換渲染器、切換 viewport 佈局）時，既有的 `get_frame()` 快取是否失效、要不要重新呼叫。技術計畫的風險表已經把這條列為己知未知，階段 0 的文件查證階段找不到答案，需要在後續 GUI 手動測試（viewport 縮放/最大化那幾項 GUI 確認清單）順便觀察。

---

### 1.2 overlay 上的滑鼠事件會不會被 viewport 相機操作吃掉

**這是全計畫最大的未知，文件查證的結論是：找不到直接答案，但找到強烈的間接證據與一個可能的備援 API。**

**【文件查不到，需在 spike 實測】核心問題本身**：查了 `omni.kit.viewport.utility`、`omni.kit.viewport.window`、`omni.kit.viewport.docs`（Camera Manipulator 專章）、`omni.physx.ui` 的 Viewport Overlays 文件，**沒有任何一份文件明講「疊加在 viewport 上、掛了 `set_mouse_pressed_fn` 的一般 2D widget，滑鼠事件會不會同時被相機操作吃掉」**。`omni.physx.ui` 的 viewport overlay 文件只說它能讓使用者「shift-click 拖動物體」，完全沒有提到跟相機操作的事件搶奪處理。這一題無法透過文件查證回答，`scripts/probe_viewport_overlay_drag.py` 就是專門為了回答這一題而寫的最小 GUI 實驗，**必須由使用者在有畫面的環境親手拖一次**。

**【推測，需實測】間接證據（支持「掛了 callback 的 widget 通常會吃掉事件」這個假設，但不是直接證明）**：
`omni.kit.viewport.window` 1.0.7 版 changelog（跟 `get_frame()` 新增同一個版本）有一條：

> "Fix issues with stats area blocking mouse-events in dead space."

這條 bug fix 描述的是「一個疊在 viewport 上的 stats overlay，連它視覺上空白（dead space）的區域都在阻擋滑鼠事件」——換句話說，**overlay widget 預設就是按照它佔用的版面矩形範圍（layout bounds）攔截滑鼠事件，不會因為視覺上透明就自動放行**，這條 bug 修的是「連透明區域都不該擋」的例外情況。這與計畫書「Kit 的 overlay widget 只要掛了滑鼠 callback 通常會消費事件、不往下傳給 viewport」的假設方向一致，但**這是從一條無關的歷史 bug fix 反推出來的旁證，不是對本計畫這個具體場景（overlay widget + 相機操作）的直接陳述**，可信度明確標成推測。

**備援方案①「暫時停用相機互動」的 API【文件確認＋部分推測】**：

`omni.kit.viewport.docs` 的 Camera Manipulator 專章確認存在停用特定相機操作的 API：

> "By default the manipulator will allow Pan, Zoom, Tumble, and Look operations on a perspective camera..."

```python
model.set_ints('disable_tumble', [1])
model.set_ints('disable_look', [1])
model.set_ints('disable_pan', [1])
model.set_ints('disable_zoom', [1])
```

四個開關（`disable_tumble`/`disable_look`/`disable_pan`/`disable_zoom`）本身【文件確認】存在且用法明確。**但文件完全沒有寫「怎麼從一個已經存在的 `viewport_window`／`viewport_api` 拿到這個 `model` 物件」**——`omni.kit.manipulator.camera` 的 Python Usage 範例只示範自己建構一個全新的 `CustomCameraManipulator`，不是取用 viewport 內建的那一個；這條取得路徑是**【文件查不到，需在 spike 實測】**。`scripts/probe_viewport_overlay_drag.py` 用專案既有慣例（`gc.get_objects()` 掃活著的物件，比照 `verify_controller_mode_switch.py` 撈 `BilliardExtension` 的做法）猜測性地找型別名稱以 `CameraManipulator` 結尾、且有 `model.set_ints` 可呼叫的實例，並提供一個可切換旗標讓使用者比較「有停用」跟「沒停用」兩種情況——**這個掃描法本身未經實測驗證**，找不到的話計畫書列的備援方案②（吃事件的透明層）、③（退回獨立 `ui.Window`）就是下一步。

也沒有找到這個停用是全域（影響所有 viewport）還是可以限定在單一 viewport window 的說明，同樣待實測。

---

### 1.3 `omni.ui.color`（`cl()`）的用法與位元組序

**【文件確認＋交叉比對】結論很明確，這題查得最清楚：**

**十六進位整數字面量是 ABGR（Alpha-Blue-Green-Red），不是 ARGB**。來源：NVIDIA 官方 `OMNIVERSE KIT UI STYLE BEST PRACTICE` 文件原文：

> "Note that `0xFF23211F` uses ABGR format (Alpha-Blue-Green-Red), where the red and blue color channels are reversed."

同一份文件明確建議改用 `cl()` 搭配十六進位**字串**（ARGB 順序，比較符合直覺）：

> "We recommend using `cl("1F2123")` or `cl("CCCCCCCC")` (if Alpha channel is involved) for better readability and consistency."

第二個獨立來源交叉印證：`omni.kit.manipulator.transform.abgr_to_color(abgr: int)` 這個公開函式的 docstring 直接說明「接受 ABGR 格式的整數，輸出 RGBA channels normalized to the range [0, 1]」——這同時證實了：

1. `cl()`／內部顏色物件的浮點分量範圍是 **0–1**，不是 0–255。
2. Kit 生態系統普遍存在「整數字面量＝ABGR、字串/浮點形式＝比較直覺的 ARGB」這個雙軌慣例，不是 `omni.ui` 獨有的個案。

**結論對照計畫書範例**：

```python
ui.Rectangle(style={"background_color": cl(0.08, 0.08, 0.10, 0.55), "border_radius": 6})
```

`cl(r, g, b, a)` 四個位置參數依 R、G、B、A 語意順序、數值範圍 0–1，這個寫法符合官方最佳實踐建議（避開十六進位字面量的 ABGR 陷阱），計畫書的用法**不需要修改**。

`scripts/probe_omni_ui_shot_panel_widgets.py` 區塊 4 仍然設計了一條「反推驗證」：用 `cl(1.0, 0.0, 0.0, 1.0)`（純紅不透明）轉成整數，比對是否等於 `0xFF0000FF`（依 ABGR 理論的預期值），把文件結論跟實際 runtime 行為對一次帳，因為**文件對「Python 端 `cl()` 回傳物件要怎麼轉成整數觀察」這件事本身沒有給範例**（`int()`／`.value`／其他），這個轉換手法本身是**【推測，需實測】**。

---

### 1.4 `ui.Placer` 的 `offset_x` 型別、`draggable` / `drag_axis` 可用性

**【文件確認】**（來源：`omni.ui.Placer` API 文件）：

- `offset_x` / `offset_y`：型別是 **`ui.Length`**，不是原生 `float`。文件原文：「`offsetX` defines the offset placement for the child widget relative to the Placer」。**文件沒有明講直接指派一個 Python `float` 給 `offset_x` 能不能用**（Kit 的 widget 屬性 setter 很多都會自動把 `float`/`int` 包成對應的 `Length`/`Pixel`，但這裡沒有查到針對 `Placer.offset_x` 的明確陳述）——這點是**【推測，需實測】**，`probe_omni_ui_shot_panel_widgets.py` 區塊 2 直接建構、賦值、讀回，印出實際型別。
- `draggable`：型別 `bool`，**存在**，文件原文「Provides a convenient way to make an item draggable」。
- `drag_axis`：型別 `Axis`，**存在**，文件原文「Sets if dragging can be horizontally or vertically」。
- `omni.ui.Axis` 這個 enum**存在**，成員為 `NONE`、`X`、`Y`、`XY`（文件原文："Members: NONE, X, Y, XY"）——計畫書假設的 `ui.Axis.XY` 是合法值。
- 額外發現（計畫書沒提到）：`frames_to_start_drag`（開始拖曳前的幀數）、`stable_size`（"The placer size depends on the position of the child when false"）、`raster_policy` 也是建構參數，未來若手感不理想可以調這兩個。
- Callback：`set_offset_x_changed_fn(Callable[[ui.Length], None])` / `set_offset_y_changed_fn(...)` **存在**，注意 callback 收到的是 `ui.Length` 不是 `float`。

**結論**：`Placer(draggable=True, drag_axis=ui.Axis.XY)` 在 API 層面**應該能建構成功**（所有參數名稱與型別都有文件依據），但「能不能建構成功」本身仍列入 `probe_omni_ui_shot_panel_widgets.py` 區塊 3 實測——文件確認參數存在不等於這個特定組合在目前專案用的 Kit 版本裡沒有 bug 或版本差異。

---

### 1.5 `set_mouse_*_fn` 的 x/y 座標系、`screen_position_x/y` 與 `computed_width/height` 的更新時機

**簽名【文件確認】**（來源：`omni.ui.Widget` API 文件，對應 C++ 原生簽名）：

```python
set_mouse_pressed_fn(fn: Callable[[float, float, int, int], None])   # x, y, button, modifier
set_mouse_moved_fn(fn: Callable[[float, float, int, bool], None])    # x, y, modifier, is_pressed
set_mouse_released_fn(fn: Callable[[float, float, int, int], None])  # x, y, button, modifier
```

文件原文對簽名的描述：「The function should be like this: `void onMousePressed(float x, float y, int32_t button, carb::input::KeyboardModifierFlags modifier)`」。

`set_mouse_moved_fn` 的行為【文件確認】符合計畫書預期：「Mouse move events only occur if a mouse button is pressed while the mouse is being moved.」——正好對應計畫書要的「拖曳中才觸發」語意，不需要額外判斷 `is_pressed`。

**【文件查不到，需在 spike 實測】x/y 是螢幕座標還是 widget local 座標**：文件對這四個 callback 的參數說明只給了型別（`float`），**完全沒有陳述座標系**。這題文件真的查不到，`probe_viewport_overlay_drag.py` 的存在理由之一就是這個——用「標記畫在面板正中央」的已知位置，比對按下去時印出的座標，反推座標系原點在哪裡。

**`screen_position_x/y`【文件確認】**：「Returns the X/Y Screen coordinate the widget was last draw. This is in Screen Pixel size. It's a float because we need negative numbers and precise position considering DPI scale factor.」——確認是**螢幕像素座標**、**浮點數**（因為要處理負值與 DPI 縮放），且明確是「上一次繪製時」的值（"was last draw"），隱含**至少要繪製過一次才有意義**，呼應第一 frame 可能不準的疑慮。

**`computed_width/height`【文件確認存在，更新時機推測】**：「Returns the final computed width/height of the widget. It includes margins.」文件用一句「詳細原因請見 `draw()` 方法的說明」帶過，**沒有明確陳述第一個 frame 是不是 0**。計畫書假設「第一 frame 可能是 0」目前只是**【推測，需實測】**，`probe_omni_ui_shot_panel_widgets.py` 區塊 8 在第 1、2、5、10 個 `app.update()` 之後分別量測 `computed_width`/`screen_position_x` 印出實際數值序列，直接回答「要等幾個 frame」這個問題。

---

### 1.6 `ui.Circle` 的 `size_policy` / `alignment` 在 `Placer` 內的行為

**【文件確認，但不完整】**（來源：`omni.ui.Circle` API 文件）：

- `alignment`：「This property holds the alignment of the circle when the fill policy is `ePreserveAspectFit` or `ePreserveAspectCrop`. By default, the circle is centered.」——**只有在特定 fill policy 下才有作用**，預設置中。
- `size_policy`：「Define what happens when the source image has a different size than the item.」文件描述套用的是「image」的語彙（`Circle` 很可能沿用跟 `Image` widget 共用的 fill-policy 概念），可能的值在其他屬性說明裡出現過 `eFixed`、`eFixedCrop`、`ePreserveAspectFit`、`ePreserveAspectCrop`，但**這份文件片段沒有給出 `size_policy` 專屬的完整列舉值清單，也沒有給 `Circle` 專屬的範例**。
- `radius`：預設 0（"By default, the circle radius is 0"）。
- `arc`：控制只畫圓的一部分（左側/左上四分之一等）。
- `segments`：預設 40 段。

**【文件查不到，需在 spike 實測】`size_policy`/`alignment` 這兩個屬性放在 `ui.Placer` 內時的實際互動行為**——`Placer` 本身也會決定子元件的定位方式，兩者疊加時（例如 `Circle` 的 `alignment` 跟 `Placer` 的 `offset_x/y` 會不會互相干擾）文件完全沒有交叉說明。`probe_omni_ui_shot_panel_widgets.py` 區塊 11 直接在 `Placer` 內建構 `Circle` 並嘗試賦值這兩個屬性，用 `hasattr` 先確認實際存在的 enum 類別名稱（文件裡看到的候選是 `Alignment`／`FillPolicy`／`CircleSizePolicy`，具體是哪一個要看 runtime）。

計畫書的瞄準線改用「12 個小圓點」而不是 `Circle` 的 `size_policy`/`alignment` 做花式定位，這兩個屬性目前只用在圓形擊球點選擇器的標記與命中半徑視覺化，即使查證結果不理想，影響範圍也有限。

---

### 1.7 `ui.FloatField` / `ui.SimpleFloatModel` 是否存在

**【文件確認】兩者都存在**，計畫書「不存在則退回 `StringField` + `float()` 解析」的備援**不需要啟用**：

- `FloatField`：「The FloatField widget is a one-line text editor with a string model.」（這句描述文字看起來是從 `StringField` 文件複製沿用、沒有針對 `FloatField` 客製，但類別確實存在，建構參數含 `model: AbstractValueModel = None`、`precision`。）
- `SimpleFloatModel`：繼承 `AbstractValueModel`，「A very simple double model」，建構子 `__init__(self, default_value: float = 0.0, **kwargs)`，文件明確列出的方法有 `get_max()`/`get_min()`/`set_max()`/`set_min()`。

**【推測，需實測】`get_value_as_float()` / `set_value()` / `add_value_changed_fn()` 這三個常用方法**：文件片段沒有直接列出，但專案已有 `omni.ui.SimpleStringModel` 的先例（`extension/ui/table_combo_box_model.py:10`：`self.model = omni.ui.SimpleStringModel(table_id)`），`SimpleFloatModel`/`SimpleStringModel` 同屬 `AbstractValueModel` 家族，這幾個方法極可能是基底類別統一提供、只是這次查證的文件片段沒收錄到。`probe_omni_ui_shot_panel_widgets.py` 區塊 10 直接建構 `FloatField(model=SimpleFloatModel(1.5))`，用 `hasattr` 探測 `get_value_as_float`/`as_float` 兩種可能命名，寫入新值後讀回驗證往返一致。

---

### 1.8 `ui.Button(clicked_fn=)` 的簽名

**【文件確認】**（來源：`omni.ui.Button` API 文件）：

```python
clicked_fn: Callable[[], None]
```

> "Sets the function that will be called when the button is activated (i.e., pressed down then released while the mouse cursor is inside the button)."

不帶參數、無回傳值，符合計畫書「按鈕觸發一次」的用法（`Button("擊球", clicked_fn=self._on_shot_button_clicked)`，內部不需要靠參數判斷按了哪顆鍵，一個 lambda/bound method 對應一顆按鈕）。文件沒有附範例程式碼，但簽名本身已經足夠明確，這題不需要進一步實測。

---

### 1.9 八項查證結果總表

| # | 項目 | 可信度 | 結論摘要 |
|---|---|---|---|
| 1 | `get_frame(ext_id)` 回傳型別 | 文件確認 | 一般 `omni.ui.Frame`，同 id 重複呼叫回傳同一物件 |
| 1 | `get_frame(ext_id)` 放 2D widget | 推測，需實測 | 型別契約上可行，官方零先例展示這個組合 |
| 1 | `get_frame(ext_id)` viewport 重建後的生命週期 | 文件查不到 | 完全沒有討論，需 GUI 觀察 |
| 2 | overlay 拖曳是否被相機操作吃掉 | 文件查不到 | 全計畫最大未知，只能用 `probe_viewport_overlay_drag.py` 親手驗 |
| 2 | 相機互動停用 API（`disable_tumble` 等） | 文件確認 | API 存在；但「怎麼拿到 model」查不到 |
| 3 | `omni.ui.color` 位元組序 | 文件確認＋交叉比對 | 整數字面量 ABGR、`cl()` 字串/浮點形式 ARGB、範圍 0–1 |
| 4 | `Placer.offset_x` 型別 | 文件確認 | `ui.Length`；純 `float` 賦值能否直接用未確認 |
| 4 | `draggable`/`drag_axis` 可用性 | 文件確認 | 都存在，`ui.Axis` 有 `NONE/X/Y/XY` |
| 5 | `set_mouse_*_fn` 簽名 | 文件確認 | 四參數，`moved` 只在按著鍵時觸發 |
| 5 | `set_mouse_*_fn` x/y 座標系 | 文件查不到 | 完全沒有陳述，需實測 |
| 5 | `screen_position_x/y` | 文件確認 | 螢幕像素座標，"last draw" 的值 |
| 5 | `computed_width/height` 第一 frame 是否為 0 | 推測，需實測 | 文件只說「與 draw() 有關」，沒給時機 |
| 6 | `ui.Circle` `size_policy`/`alignment` | 文件確認（不完整） | 存在但列舉值與 Placer 內互動未知 |
| 7 | `FloatField`/`SimpleFloatModel` 存在 | 文件確認 | 兩者都存在，不需要 StringField 備援 |
| 7 | `SimpleFloatModel` 讀寫方法 | 推測，需實測 | 依 `AbstractValueModel`/`SimpleStringModel` 先例推斷存在 |
| 8 | `Button(clicked_fn=)` 簽名 | 文件確認 | `Callable[[], None]` |

---

### 1.10 對計畫的影響

**目前查證結果沒有推翻計畫的任何硬性決策**（±160° 角度範圍、overlay 形式、四個控制項、按鈕觸發一次等都不受本節影響）。唯一**有可能推翻**的是「主方案／備援方案的優先順序」：

- 若 `probe_viewport_overlay_drag.py` 實測發現 overlay 拖曳確實會被相機操作吃掉，且 1.2 節查到的相機互動停用 API 因為「拿不到 model」而無法在 spike 內接上（`gc.get_objects()` 掃描法找不到符合條件的實例），計畫書排序的備援方案①（停用相機互動）會直接卡住，必須跳到備援②（吃事件的透明層，其實主方案的 catcher `Rectangle` 已經是這個設計，等於①失敗時②本來就在原地）或③（退回獨立 `ui.Window`，等於放棄「疊在 3D 畫面上」這個使用者已拍板的視覺需求，衝擊最大）。
- 若 `computed_width`/`screen_position_x` 在遠超過 10 個 frame 後仍為 0（比目前抓的量測範圍更極端的情況），面板「用 Spacer 相對定位、標記畫面上的固定像素常數」這個假設仍然成立（設計本來就不依賴 `computed_width`），但座標換算 `_to_local()` 依賴 `screen_position_x/y`，若這個值遲遲不更新，主方案的滑鼠事件座標換算會失準，需要改用其他座標來源（例如直接用面板的固定螢幕位置常數，放棄動態讀取）。

這兩點都已經寫進對應探測腳本的輸出項目，等使用者在有 Isaac Sim 的環境跑過 `probe_omni_ui_shot_panel_widgets.py` 與 `probe_viewport_overlay_drag.py`，把結果貼回來即可定案。

---

### 1.11 參考來源網址

- `https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.window/latest/omni.kit.viewport.window/omni.kit.viewport.window.ViewportWindow.html`（`get_frame()` 簽名與 docstring）
- `https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.window/latest/CHANGELOG.html`（`get_frame()` 首次出現版本、"dead space" mouse-event bug fix）
- `https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.utility/1.0.18/index.html` 與 `omni.kit.viewport.utility.get_active_viewport_window.html`
- `https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.docs/latest/camera_manipulator.html`（相機操作停用 API）
- `https://docs.omniverse.nvidia.com/kit/docs/omni.kit.manipulator.camera/106.0.3/USAGE_PYTHON.html`
- `https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.ui/docs/dev_guide/viewport_overlays.html`
- `https://docs.omniverse.nvidia.com/kit/docs/kit-manual/110.1.0/guide/ui_style_best_practice.html`（`cl()` 與 ABGR/ARGB 位元組序權威來源）
- `https://docs.omniverse.nvidia.com/kit/docs/omni.kit.manipulator.transform/106.0.1/omni.kit.manipulator.transform/omni.kit.manipulator.transform.abgr_to_color.html`（交叉印證 ABGR、0–1 範圍）
- `https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.Placer.html`
- `https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.Widget.html`
- `https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.Circle.html`
- `https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.FloatField.html`
- `https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.SimpleFloatModel.html`
- `https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.Button.html`
- `https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.Axis.html`
- `https://github.com/NVIDIA-Omniverse/kit-extension-sample-ui-scene/blob/main/exts/omni.example.ui_scene.widget_info/Tutorial/object.info.widget.tutorial.md`（唯一找到的 `get_frame()` 官方範例，用於 `sc.SceneView` 而非 2D widget）

---

### 1.12 `probe_omni_ui_shot_panel_widgets.py` 實測結果（2026-09-08，headless）

環境：`C:/Other/OmniverseProjects/isaac`（獨立安裝的 Isaac Sim 6.0.0，`python.bat` 啟動，非 pip venv），`SimulationApp({"headless": True})`。跑法見腳本 docstring。跑之前先修掉一個擋路的 bug——見 6.6 節「headless 執行的既有陷阱」。以下逐項用實測結果取代 1.9 節總表的「推測/文件查不到」。

| # | 項目 | 1.9 節原標示 | 實測結果 |
|---|---|---|---|
| 1 | `get_frame(ext_id)` 放 2D widget | 推測，需實測 | **確認可行**：`VStack`+`Rectangle`+`Button` 建構於 `get_frame()` 回傳的 `Frame` 內不拋例外 |
| 1 | 同一 `ext_id` 重複呼叫 | 文件確認 | **實測吻合**：identity 比對為 `True` |
| 1 | `get_active_viewport_window()` headless 下的回傳值 | （原本假設「大機率回傳 None」） | **推翻假設，但是好消息**：headless 下回傳非 `None`（`weakref.ProxyType`）。`hud_panel.py` 與 `verify_manual_controller_wiring.py` docstring 裡「headless 大機率拿不到 viewport」的說法在**這個環境**不成立；但 `_create_root_frame()` 拿不到 viewport 時回傳 `None` 的分支仍然是必要的防禦性設計——換一台真正無顯示裝置的機器（CI、無 GPU 的 Linux 容器）行為可能不同，程式碼不需要改 |
| 2 | overlay 拖曳是否被相機操作吃掉 | 文件查不到 | **仍未知**——headless 沒有真實滑鼠事件，本項只能證明「callback 掛得上去」，見下方區塊 9 |
| 3 | `omni.ui.color` ABGR 理論 | 文件確認＋交叉比對 | **實測反推驗證成立**：`cl(1,0,0,1)` → `int` 值換算成十六進位剛好等於 ABGR 理論預測的 `0xff0000ff` |
| 4 | `Placer.offset_x` 純 `float` 賦值 | 文件確認型別，賦值方式未確認 | **確認可行**：`placer.offset_x = 25.5`（純 `float`）寫入後讀回型別仍是 `omni.ui._ui.Length`、值正確 |
| 4 | `draggable`/`drag_axis` 可用性 | 文件確認 | **實測確認**：`Placer(draggable=True, drag_axis=ui.Axis.XY)` 建構成功、`set_offset_x_changed_fn()` 掛載成功（`hud_panel.py` 主方案沒有用到這個，備援方案確認可用） |
| 5 | `set_mouse_*_fn` x/y 座標系 | 文件查不到 | **仍未知**——同項目 2，需要 `probe_viewport_overlay_drag.py` 在 GUI 下才能測出真實事件的座標系 |
| 5 | `computed_width/height`、`screen_position_x/y` 第一 frame 是否為 0 | 推測，需實測 | **確認為 0，且遠比預期持久**：headless 下累計到第 10 個 `app.update()` 仍是 `0.0`（原本只抓到「第一 frame」這個量測點，實測發現不是「延遲幾個 frame」而是在 headless 底下可能永遠不會有非零值——`hud_panel.py`「全部尺寸用寫死常數，不依賴 computed 值」的設計决策因此被進一步證實是對的；但 `_to_local()` 依賴的 `screen_position_x/y` 同樣卡在 0，這條路徑在 headless 下無法驗證，仍待 GUI 實測） |
| 6 | `ui.Circle.size_policy` | 文件確認（不完整） | **踩到一個真實地雷，但沒打中我們的程式碼**：`circle.size_policy = ui.FillPolicy.STRETCH` 會拋 `TypeError`（要求 `CircleSizePolicy`，不是 `FillPolicy`）；`circle.size_policy = ui.CircleSizePolicy.STRETCH` 才會成功。逐一核對過 `hud_panel.py` 全部 6 處 `ui.Circle(...)` 呼叫，**沒有一處設定 `size_policy`**（只用 `width`/`height`/`radius`/`style`），這個地雷沒有被踩到，不需要改程式碼 |
| 7 | `SimpleFloatModel` 讀寫方法 | 推測，需實測 | **實測確認、且逐字對上 `hud_panel.py` 的實際呼叫**：`get_value_as_float()` 與 `set_value()` 都存在且行為正確（`SimpleFloatModel(1.5)` → 讀值 `1.5` → `set_value(2.75)` → 讀值 `2.75`），跟 `hud_panel.py:470` 的 `model.get_value_as_float()`、`hud_panel.py:631` 的 `self._speed_model.set_value(...)` 完全一致 |

**結論：本次實測沒有推翻 `hud_panel.py` 任何一行程式碼的正確性，`_create_root_frame()` 與 `_to_local()` 兩個隔離點目前仍原封不動——`_create_root_frame()` 內部呼叫的 `get_frame()` 已被證實可行；`_to_local()` 依賴的螢幕座標系仍待 `probe_viewport_overlay_drag.py` 在 GUI 下驗證。** 剩下唯一真正未知的是「overlay 拖曳會不會被相機操作吃掉」與「滑鼠事件座標系」，兩者都需要真實滑鼠事件，headless 原理上量不出來。

完整 console 輸出（11 個區塊、共 41 行 `[probe]` 前綴輸出）已由使用者貼回並核對過，不重複收錄在此；有需要可重跑腳本重現。

---

## 2. 需求與範圍

### 2.1 要解決的問題

Demo 目前只能看著手臂用固定參數（開球點、0°、最大初速、零偏移）自動開球，
沒有任何手動介入的方式，無法測試「不同擊球參數會打出什麼結果」，也無法為
#116（ShotResult 顯示）、#117（完整流程確認）、#118（Demo 影片錄製）提供
一個可控的展示點。#115 要在 Demo 桌上加一個常駐的參數控制面板，讓使用者能
即時調整六維擊球參數中的四項（母球擺位、擊球方向角、初速、上下/左右擊球
偏移）並手動觸發一次 AIM→STRIKE，不自動循環。

### 2.2 範圍

**包含：**

- 一個嵌在 Viewport 內的半透明 overlay 面板（`HudPanel`），提供圓形擊球點
  選擇器、力道輸入框、球桌俯瞰圖（含母球拖曳擺位＋點擊定角度）、「擊球」
  與「重設球局」兩顆按鈕。
- 一個新的 core 層 controller（`ManualController`），繼承既有的
  `BilliardStateMachineController`，只在按下「擊球」時觸發一次 AIM→STRIKE，
  平時停在 IDLE。
- 一套獨立於 RL 訓練動作空間（`action_bounds.py`）的手動面板邊界常數
  （`manual_shot_bounds.py`），角度範圍收窄到 ±160° 以避開機械臂基座
  teleport 進球桌的錯誤。
- 一套幾何可行性判斷（`manual_shot_feasibility.py`），在「角度合法但幾何
  無解」時提前擋下擊球鈕，避免面板把桌子打進 `ERROR` 狀態。
- Extension 層的接線：常駐 `ManualController` 字典（與 Demo session 同生
  同滅）、六個 DI 方法供 `HudPanel` 呼叫、`_build_controller_for_mode()`
  改為回傳常駐實例。

**不包含（明確排除）：**

- 走位球（cue ball placement 不受限於 Kitchen 之外的其他規則）——那是
  Milestone B（#232）的範圍，`MANUAL_SHOT_ANGLE` 屆時要退化為直接轉出整圈
  的 `SHOT_ANGLE`。
- ShotResult 顯示（#116）、影片錄製（#118）——本次只交付「能手動打一桿」
  這個能力本身。
- 基座位置的幾何檢查——`manual_shot_feasibility.py` 目前只判斷
  `tilt_rad is None` 這一種無解原因，角度上限已經在 `MANUAL_SHOT_ANGLE`
  這一層排除掉「基座 teleport 進球桌」那一類錯誤，兩者不重疊判斷（見第
  5.1 節）。

### 2.3 使用者已拍板的決策（摘要）

| 決策 | 結論 |
|---|---|
| 角度範圍 | 面板限制 ±160°（安全區），不動 `action_bounds.SHOT_ANGLE` 的 ±30° |
| 母球擺位 | 納入，俯瞰圖上可拖曳（Kitchen 範圍內） |
| 擊球觸發 | 面板加「擊球」按鈕，按下才跑一次 |
| HUD 形式 | 嵌在 Viewport 內的半透明 overlay，不是獨立停靠視窗 |

完整推導過程與拍板脈絡見完整技術計畫
`C:\Users\guan_\.claude\plans\115-hud-peppy-harbor.md`（本文件不重複貼）。

---

## 3. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `MANUAL_SHOT_ANGLE`／四項轉出常數 | core/models | 手動面板專用的邊界值——角度是本檔自己定義的安全區（±160°），其餘四項（擺位/速度/偏移）直接轉出 `action_bounds`，不重新定義數值 | `core/models/manual_shot_bounds.py` |
| `ManualShotParameters` | core/models | `frozen=True` 的四維參數容器，建構即驗證（越界拋 `ValueError`）；`default()`／`to_action()` | `core/models/manual_shot_parameters.py` |
| 像素↔物理量換算純函式 | core/services | 圓形偏移選擇器、俯瞰圖擺位/角度的雙向換算，全部無狀態、不知道 omni.ui 的存在 | `core/services/shot_panel_input_mapper.py` |
| `ManualShotFeasibility`／`evaluate_manual_shot(_parameters)` | core/services | 判斷「幾何無解」這一種擊球失敗原因，`reason` 是機器可讀字串，不含 UI 文案 | `core/services/manual_shot_feasibility.py` |
| `ManualController` | core/controllers | 只覆寫 `_idle_state_action_result()`／`_aiming_state_action_result()`／`_on_reset()`／`_enter_error_state()`，其餘沿用 `BilliardStateMachineController` | `core/controllers/manual_controller.py` |
| `HudPanel` | extension/ui | 嵌在 viewport 內的半透明 overlay 面板，建構子只吃 6 個注入的 Callable，不持有 `BilliardExtension` 參照 | `extension/ui/hud_panel.py` |
| `TableComboBoxModel` | extension/ui | 從 `debug_menu.py` 搬出的共用選桌下拉 model，`HudPanel` 與 `DebugMenu` 各自持有一份實例 | `extension/ui/table_combo_box_model.py` |
| `BilliardExtension`（修改） | extension/billiard_digital_twin | 保存 `self._ext_id`；`self._demo_manual_controllers: dict[str, ManualController]`；`_build_demo_session()` 建常駐實例；`_build_controller_for_mode()` 改吃 `table_id`；新增 6 個 DI 方法（`get_manual_shot_parameters`／`set_manual_shot_parameters`／`request_manual_shot`／`request_manual_reset`／`get_manual_shot_status_text`／`get_table_geometry`，另加 `get_demo_table_ids` 供選桌下拉用）；`_billiard_init()` 建構 `HudPanel`；`_disable_demo()`/`on_shutdown()` 清理 | `extension/billiard_digital_twin/billiard_digital_twin.py` |
| `denormalize_axis`／`normalize_axis`（公開化） | core/services | `rl_action_decoder` 既有的正規化/反正規化邏輯改為公開函式，供 `shot_panel_input_mapper` 重用，避免第二份實作 | `core/services/rl_action_decoder.py` |
| `SHOT_ANGLE` 指路註解 | core/models | 一行註解指向 `manual_shot_bounds.py`，不改任何數值 | `core/models/action_bounds.py` |

---

## 4. 類別設計

### 4.1 `manual_shot_bounds.py`（兩把尺）

不是類別，是模組級常數＋檔案級 docstring。核心設計是「兩把尺」：

| 項目 | 尺的性質 | 來源 |
|---|---|---|
| 母球擺位 XY／初速／上下左右偏移 | 物理能力邊界（桌台幾何、桿尖速度上限、miscue limit） | `from .action_bounds import ...` 轉出，不重新定義數值 |
| 擊球方向角 `MANUAL_SHOT_ANGLE` | 安全區，非物理極限 | 本檔自己定義 `(-160.0, 160.0)` |

四項轉出常數與 RL 訓練動作空間共用同一個物理事實來源，理由是「擺位/速度/
偏移無論是手動面板還是 RL policy 出手，物理世界能不能接受都是同一個答案」，
若各自定義一份數值，兩邊漂移不會報錯（#228 的教訓，見該檔案級 docstring）。
角度不同：`action_bounds.SHOT_ANGLE = (-30, 30)` 是 Milestone A 為了訓練
信號密度收窄的 RL 動作空間（#231），跟手動面板的物理限制無關；手動面板
自己的物理限制來自 `Ur10eSwingStrategy.execute_aim()` 的基座
`reposition()` 公式解出的 `|θ| < 162.8°`，取 ±160° 留 2.8° 安全餘裕。

### 4.2 `ManualShotParameters`（core/models，`frozen=True` dataclass）

**職責：** UI 執行緒與 physics 執行緒之間交換擊球參數的唯一資料格式。

```python
@dataclass(frozen=True)
class ManualShotParameters:
    cue_ball_placement: tuple[float, float]
    shot_angle: float
    cue_ball_speed: float
    position_offset: tuple[float, float]

    def __post_init__(self) -> None: ...   # 越界拋 ValueError，不靜默夾住
    @staticmethod
    def default() -> "ManualShotParameters": ...
    def to_action(self, should_execute_action: bool) -> Action: ...  # 每次回全新 Action
```

`frozen=True` + tuple 欄位是執行緒安全的基礎（見第 5.3 節）。`__post_init__`
驗證邊界時對擺位/速度/偏移三項讀 `manual_shot_bounds.py` 轉出的常數，角度
讀 `MANUAL_SHOT_ANGLE`——驗證即用兩把尺各自對應的範圍，職責邊界在這裡體現
得最直接。`to_action()` 每次回傳全新 `Action` 物件，理由見第 5.4 節。

**依賴：**
- 輸入來源：`extension/ui/hud_panel.py` 用 `shot_panel_input_mapper` 夾好
  邊界後建構
- 輸出去向：`ManualController.set_parameters()`／`to_action()` 供狀態機
  分派使用

### 4.3 `shot_panel_input_mapper.py`（core/services，純函式集合）

**職責：** 像素座標（widget local，左上原點，y 往下增加）↔ 物理量（桌台
相對座標／`Action.position_offset`）的雙向換算，完全不知道 omni.ui 或螢幕
座標系的存在——這條邊界讓所有換算邏輯可以在沒有 Isaac Sim 的環境下用
Unit Test 覆蓋。

九個函式：`offset_from_circle_pixels`／`circle_pixels_from_offset`（圓形
擊球點選擇器）、`table_xy_from_topview_pixels`／`topview_pixels_from_table_xy`／
`clamp_cue_ball_placement`（俯瞰圖擺位）、`shot_angle_from_points`／
`clamp_manual_shot_angle`／`aim_line_endpoint`（俯瞰圖角度與瞄準線）、
`is_within_radius`（俯瞰圖命中判定，分辨「拖擺位」還是「定角度」）。

**必須重用既有實作**（見第 5.2 節「不要重新實作換算邏輯」）：
- 偏移裁切固定走 `position_offset_limiter.clamp_position_offset()`（圓形
  裁切，不是逐軸 clip）
- 正規化/反正規化固定走 `rl_action_decoder.normalize_axis()`/
  `denormalize_axis()`
- 偏移處理順序固定「正規化 → 圓形裁切 → 反正規化」，與 `decode_rl_action()`
  同一條順序

**依賴：**
- 輸入來源：`extension/ui/hud_panel.py` 的滑鼠事件 callback（已用
  `_to_local()` 換算成 widget local 座標）
- 輸出去向：`ManualShotParameters` 建構參數；`hud_panel.py` 重繪畫面時的
  反方向換算（`circle_pixels_from_offset`／`topview_pixels_from_table_xy`）

### 4.4 `manual_shot_feasibility.py`（core/services）

**職責：** 判斷「按下擊球鈕真的打得出去嗎」，只處理「幾何無解」這一種原因
（角度上限已經在 `MANUAL_SHOT_ANGLE` 這一層排除掉另一種失敗，見第 5.1
節）。

```python
@dataclass(frozen=True)
class ManualShotFeasibility:
    is_feasible: bool
    reason: str          # "" | "geometry_unsolvable"

def evaluate_manual_shot(cue_ball_xy, shot_angle_deg, table_z, ball_radius,
                         position_offset) -> ManualShotFeasibility: ...
def evaluate_manual_shot_parameters(parameters, table_z, ball_radius) -> ManualShotFeasibility: ...
```

完全委派給 `cue_pose_calculator.compute_tilted_wrist_pose()`（`Ur10eSwingStrategy`
本身也呼叫同一支函式），`tilt_rad is None` 翻譯成 `is_feasible=False`。
`reason` 是機器可讀字串而非給人看的訊息——UI 文案是 `hud_panel.py` 的職責，
`core/` 不做 UI 呈現。

**依賴：**
- 輸入來源：`hud_panel.py` 的 `_evaluate_feasibility()`（每次重繪呼叫）；
  `table_z`／`ball_radius` 來自 `BilliardExtension.get_table_geometry()`
- 輸出去向：面板的瞄準線顏色（紅/黃）與「擊球」按鈕 `enabled` 狀態

### 4.5 `ManualController`（core/controllers）

**職責：** 由使用者手動決定四維擊球參數，按下「擊球」鈕才出一桿一次。

```python
class ManualController(BilliardStateMachineController):
    def __init__(self, parameters=None):
        self._parameters = parameters or ManualShotParameters.default()
        self._requested_seq = 0   # 只有 UI 執行緒寫
        self._handled_seq = 0     # 只有 physics 執行緒寫
        self._pending: ManualShotParameters | None = None

    # UI 執行緒
    def set_parameters(self, parameters) -> None: ...
    def get_parameters(self) -> ManualShotParameters: ...
    def request_shot(self) -> None: self._requested_seq += 1
    def is_shot_pending(self) -> bool: return self._requested_seq != self._handled_seq

    # physics 執行緒（BilliardStateMachineController 的擴充點）
    def _idle_state_action_result(self, observation) -> Action: ...
    def _aiming_state_action_result(self, observation) -> Action: ...
    def _enter_error_state(self, exception) -> Action: ...   # 自己定義，不在基底類別
    def _on_reset(self) -> None: ...
```

只覆寫基底類別明確開放的三個擴充點（`_idle_state_action_result`／
`_aiming_state_action_result`／`_on_reset`），另外自己定義
`_enter_error_state()`（`ModelController` 也是這樣做，不在
`BilliardStateMachineController` 裡）。其餘 4 個狀態轉換（STRIKING/
WAITING/RESET 的條件與 no-op Action 格式）沿用基底類別——那是
`ScriptController` 與 `ModelController` 共用的契約，本類別不例外。

「為什麼沒按鈕就會永遠停在 IDLE」「為什麼 AIM 與 STRIKE 要讀同一份快照」
「為什麼用序號而不是 bool 旗標」三個關鍵設計問題見第 5.3／5.4 節（也已
完整寫在 `manual_controller.py` 的類別 docstring 裡，本節不重複貼）。

**依賴：**
- 輸入來源：`HudPanel` 透過 `BilliardExtension` 的 DI 方法呼叫
  `set_parameters()`/`request_shot()`
- 輸出去向：`TableOrchestrator.step()` → `_execute_aim()`/`_execute_strike()`

### 4.6 `HudPanel`（extension/ui）

**職責：** 疊在 viewport 內的半透明 overlay，四個控制項（圓形擊球點選擇器、
力道輸入框、俯瞰圖、擊球/重設按鈕）+ 收合鈕 + 狀態列。

```python
class HudPanel:
    def __init__(
        self,
        ext_id: str,
        get_parameters: Callable[[str], "ManualShotParameters | None"],
        on_parameters_changed: Callable[[str, ManualShotParameters], None],
        on_shot_requested: Callable[[str], None],
        on_reset_requested: Callable[[str], None],
        get_shot_status_text: Callable[[str], str],
        get_table_geometry: Callable[[str], "tuple[float, float] | None"],
    ) -> None: ...
```

沿用 `debug_menu.py` 的建構子風格——只吃注入的 Callable，不持有
`BilliardExtension` 參照。面板**不保存參數狀態**：擊球參數只活在
omni.ui model（力道欄）與常駐 `ManualController` 裡，面板唯一的私有狀態是
拖曳暫態（`_topview_drag_mode`）與防遞迴旗標（`_suppress_speed_callback`）。

兩處明確隔離 spike 未證實假設的方法（見第 1.1／1.2／1.5 節與第 6.5 節
「已知限制」）：

- `_create_root_frame()`：取得 `viewport_window.get_frame(ext_id)`；
  headless 或拿不到 viewport 一律回傳 `None`，讓 `__init__` 安靜跳過建
  面板（硬性要求，見第 5.5 節）。
- `_to_local(x, y, widget)`：把 `set_mouse_*_fn` 收到的座標換算成 widget
  local 座標，目前假設是螢幕座標（用 `screen_position_x/y` 相減）。

**依賴：**
- 輸入來源：使用者滑鼠/鍵盤事件；`BilliardExtension` 注入的 6 個查詢/回呼
  方法
- 輸出去向：`core.services.shot_panel_input_mapper`／
  `manual_shot_feasibility` 的純函式；`BilliardExtension` 的 DI 方法

### 4.7 `TableComboBoxModel`（extension/ui，從 `debug_menu.py` 搬出）

**職責：** 可動態增刪選項的 ComboBox model，`HudPanel` 與 `DebugMenu` 各自
持有一份獨立實例（不共用同一個 model——兩者的選桌下拉是獨立的 UI 狀態，
選中的桌子沒有必要同步）。從 `debug_menu.py:18-61` 搬出成獨立檔案，避免
`hud_panel.py` 複製第二份相同邏輯。

---

## 5. 資料流

### 5.1 兩種失敗原因的分工（角度上限 vs 幾何可行性閘門）

`Ur10eSwingStrategy.execute_aim()` 有兩種已知會失敗的情境：

1. **基座 teleport 進球桌中央**——`base = cue_ball − 2.15 × (−sinθ, cosθ)`
   在 `|θ|` 過大時解出來的基座落在桌台範圍內。`MANUAL_SHOT_ANGLE =
   (-160.0, 160.0)` 從源頭排除這一種，`manual_shot_feasibility.py` 不重複
   判斷。
2. **幾何無解**——`compute_tilted_wrist_pose()` 回傳 `tilt_rad is None`：
   即使把球桿垂直抬到最高也閃不過庫邊，跟角度範圍無關，是母球位置＋角度
   ＋偏移量組合造成的純幾何問題。`manual_shot_feasibility.py` 只判斷這
   一種。

兩層防線各管一種失敗原因，不疊床架屋。

### 5.2 UI 互動 → 參數更新（同步，UI 執行緒）

```
使用者拖曳圓形擊球點選擇器 / 俯瞰圖 / 輸入力道欄
  → HudPanel._on_circle_press / _on_topview_press / _on_speed_value_changed
    → shot_panel_input_mapper.*()（像素 → 物理量，夾好邊界）
      → dataclasses.replace(current, ...) 產生新的 ManualShotParameters
        （建構即驗證，越界會拋 ValueError——但因為呼叫端已經夾過邊界，
        正常互動路徑不會走到這裡）
    → HudPanel._push_parameters(table_id, updated)
      → self._on_parameters_changed(table_id, updated)
        = BilliardExtension.set_manual_shot_parameters(table_id, updated)
          → self._demo_manual_controllers[table_id].set_parameters(updated)
            （單一物件參照賦值，原子操作，見第 5.3 節）
      → HudPanel._redraw_controls(table_id, updated)（整批重繪：標記位置、
        讀數 Label、可行性紅線、按鈕 enabled 狀態）
```

`_push_parameters()` 不重新呼叫 `get_parameters()` 讀回來——用同一份剛
建構好的物件重繪，省一趟往返，也避免「controller 端還沒來得及套用」的
競態。

### 5.3 擊球請求 → AIM → STRIKE（跨執行緒：UI 執行緒寫請求，physics 執行緒消費）

```
使用者按「擊球」
  → HudPanel._on_shot_button_clicked
    → self._on_shot_requested(table_id) = BilliardExtension.request_manual_shot(table_id)
      → self._demo_manual_controllers[table_id].request_shot()
        → self._requested_seq += 1   ← UI 執行緒寫，physics 執行緒不寫這個欄位

（下一個 physics tick）
TableOrchestrator.step()
  → self._script_controller.get_action(observation)   ← 這裡呼叫的其實是 ManualController
    → ManualController._idle_state_action_result(observation)
      → is_shot_pending()？(self._requested_seq != self._handled_seq)
        → 是，且 observation.is_init_state 且不是 is_ball_moving：
          → parameters = self._parameters      # 只讀這一次，鎖定這次擊球的快照
          → self._handled_seq = self._requested_seq   ← physics 執行緒寫，消費請求
          → self._pending = parameters
          → self._change_state(AIMING)
          → return parameters.to_action(should_execute_action=True)
  → step() 再讀 self.get_current_state()（此時已經是 AIMING）
    → 分派 self._execute_aim(這次 IDLE handler 回傳的 action)
```

**時序關鍵**：`TableOrchestrator.step()`（`core/services/table_orchestrator.py:48`）
先呼叫 `get_action()`（handler 內部已改狀態），**再**讀
`get_current_state()` 去分派下游動作。也就是說 AIMING 消費的其實是這一次
**IDLE handler** 回傳的 Action，`cue_ball_placement`/`shot_angle`/
`position_offset` 必須在 IDLE handler 就填好——這是本次任務最容易被誤解
的時序坑，`ModelController` 也是這樣做的，但 `ScriptController._idle_
state_action_result()` 沒有填 `cue_ball_placement`（沿用 `_generate_
action_result()` 的 `[0, 0]` 桌台中心）是既有 bug，`ManualController` 用
`ManualShotParameters.to_action()` 一次填滿四項，不重演那個問題（見
CHANGELOG）。

```
（後續 tick，瞄準動畫收斂）
TableOrchestrator.step()
  → ManualController._aiming_state_action_result(observation)
    → observation.is_motion_complete？
      → 是：
        → self._pending is None？→ 是就 _enter_error_state()（時序被破壞）
        → 否：self._change_state(STRIKING)
          → return self._pending.to_action(should_execute_action=True)
              ← 讀 IDLE handler 存下的快照，不讀 self._parameters
  → 分派 self._execute_strike(這次 AIMING handler 回傳的 action)
```

`AIMING → STRIKING → WAITING → RESET → IDLE` 之後沿用基底類別邏輯自動跑
完；回到 IDLE 時 `_requested_seq == _handled_seq`（請求已消費），不會自動
觸發下一次。

### 5.4 Extension 接線總覽

```
BilliardExtension._billiard_init()
  → self._hud_panel = HudPanel(self._ext_id,
        self.get_manual_shot_parameters, self.set_manual_shot_parameters,
        self.request_manual_shot, self.request_manual_reset,
        self.get_manual_shot_status_text, self.get_table_geometry)
    → 建構永遠不拋例外：headless／拿不到 viewport 時 HudPanel 內部
      安靜跳過建面板（_create_root_frame() 回 None），self._hud_panel
      本身仍然是一個 HudPanel 實例（不是 None），只是它的 _root_frame
      是 None、後續呼叫全部安全 no-op

BilliardExtension._build_demo_session()
  → self._demo_manual_controllers[table_id] = ManualController()
    （與 table_ball_set 同一個地方建立，_disable_demo() 同一個地方清理）

DebugMenu 切 AI/Manual（Debug Menu 專屬功能，跟 HudPanel 面板本身是否顯示
無關）
  → BilliardExtension._on_demo_controller_mode_changed(table_id, is_ai_mode)
    → session.request_controller_swap(
          self._build_controller_for_mode(is_ai_mode, table_id, table_ball_set))
      → is_ai_mode=False 時回傳 self._demo_manual_controllers[table_id]
        （常駐實例，不可在這裡 new 一個新的——swap 本身會在 TableRuntime.
        tick() 套用時強制 full_reset()，每次調參數都經過這裡等於每次調參數
        都重開一局，見第 5.5 節）

HudPanel 呼叫的六個 DI 方法（BilliardExtension）
  get_manual_shot_parameters(table_id)   → controller.get_parameters() | None
  set_manual_shot_parameters(table_id, p) → controller.set_parameters(p)
  request_manual_shot(table_id)          → controller.request_shot()
  request_manual_reset(table_id)         → session.request_full_reset()
  get_manual_shot_status_text(table_id)  → 組多行狀態文字（查無 table_id 回 ""）
  get_table_geometry(table_id)           → (table_z, ball_radius) | None
  （另外 get_demo_table_ids() 供選桌下拉清單，Training 桌不列入）

  全部方法對未知 table_id 一律 dict.get() → None → 安靜 no-op，沿用
  _on_demo_controller_mode_changed() 的既定慣例。
```

### 5.5 為什麼 `ManualController` 必須常駐、參數更新不能走 controller swap

`TableRuntime.tick()` 套用 pending controller 時會強制 `full_reset()`
（重擺球＋手臂歸位）。若 `_build_controller_for_mode()` 每次都 `new` 一個
`ManualController` 靠 swap 換上去，等於每次調參數都觸發一次
`full_reset()`（重開一局），而且新實例是空白的 `ManualShotParameters.
default()`，會讓使用者已經調好的參數憑空消失。真正的參數更新走
`ManualController.set_parameters()`（單一物件參照賦值），完全不經過
`_build_controller_for_mode()`、也不經過 swap——`set_manual_shot_
parameters()` 直接對常駐字典裡的實例呼叫，這是本次接線設計裡最容易被
「看起來很自然」的重構破壞的一條路徑。

---

## 6. 關鍵設計決策與理由

### 6.1 用序號而非 bool 旗標傳遞擊球請求

`request_shot()` 由 UI 執行緒呼叫，`_idle_state_action_result()` 由
physics 執行緒呼叫，兩者沒有鎖保護。若用 `bool` 旗標，消費端勢必要寫成
`pending = self._flag; self._flag = False`（read-modify-write），這段
序列不是原子操作，兩個執行緒交錯時可能漏掉一次請求或誤判成兩次。序號法
讓每個欄位只有唯一寫者：`_requested_seq` 只被 UI 執行緒寫、`_handled_seq`
只被 physics 執行緒寫，兩邊都只讀對方的欄位，不存在 lost update；
`is_shot_pending()` 只是比較兩個整數，讀到任一方「寫到一半」的中間狀態
也不影響正確性（int 賦值本身是原子的）。連按多次只會讓 `_requested_seq`
多加幾次，消費時一次性追平成同一個值，等效於合併成一次擊球——刻意不做
擊球佇列。也因此**不需要 `threading.Lock`**：physics callback 每個 tick
都會經過 `get_action()`，若在這條熱路徑上取鎖，是拿一筆確定會發生的效能
成本去換一個原本就不存在的競態。

### 6.2 `frozen=True` + tuple 欄位是執行緒安全的基礎

UI 端的寫入模式是「整包算好再一次賦值」——拖曳/輸入事件在 UI 執行緒裡各自
更新暫存值，全部合法之後才建構一個新的 `ManualShotParameters` 蓋掉舊的。
physics 端的讀取模式是「一次讀出整包」——`_idle_state_action_result()`
只在進入 AIMING 的那一刻讀一次。兩邊都不會讀到「角度已更新但速度還沒」
這種中間態：因為欄位是 frozen，唯一能看到的狀態只有「換之前的完整一份」
或「換之後的完整一份」，不存在逐欄位修改途中被另一個執行緒讀到一半的
問題。這比對 4 個欄位分別上鎖便宜得多，也不需要鎖。

### 6.3 AIM 與 STRIKE 讀同一份快照 `_pending`

兩次分派之間（AIM 發生在 IDLE→AIMING、STRIKE 發生在 AIMING→STRIKING）
使用者可能已經在瞄準動畫播放期間改了面板上的值。若各自去讀
`self._parameters` 這個「即時值」，會變成「照舊角度瞄準、照新速度打」；
改擺位更糟——AIM 已經把母球 teleport 到舊位置，STRIKE 卻對著空氣揮桿。
做法與 `ModelController` 快取 `_cached_raw_action` 是同一個理由：一次
決策，兩次分派共用同一份輸出。`to_action()` 每次呼叫都回傳全新 `Action`
物件（`Action` 是 mutable dataclass，下游會直接改它），若快取單一
instance 重複回傳，AIM 用過的 Action 被下游改動後，STRIKE 讀到的就不再
是原本的參數。

### 6.4 不要重新實作換算邏輯

`shot_panel_input_mapper.py` 的偏移裁切固定走
`position_offset_limiter.clamp_position_offset()`（圓形裁切）而不是逐軸
clip——逐軸 clip 會改變偏移方向，而偏移方向就是加旋方向（#222 的教訓）。
正規化/反正規化固定走 `rl_action_decoder.normalize_axis()`/
`denormalize_axis()`，不自己乘 0.5 或除以 half_span——即使數學上等價，
換算漂移不會報錯（#228），兩個方向都公開是為了對稱，避免下一個人只看到
`denormalize_axis` 是公開的就自己重算正規化那一半。

### 6.5 headless 必須安靜跳過建面板

`omni.kit.viewport.utility.get_active_viewport_window()` 在 viewport 尚未
建立時回傳 `None`；`_create_root_frame()` 對這種情況（含 `ImportError`）
一律回傳 `None`，`HudPanel.__init__()` 檢查到 `None` 就提早 return，不
建立任何 widget、也不拋例外。這不是防禦性程式碼，是硬性要求——
`_billiard_init()` 無條件建構 `HudPanel`（跟 `DebugMenu` 一樣），`scripts/`
底下所有 headless 驗證腳本（`verify_manual_controller_wiring.py` 等十幾支）
都會在 extension 啟動時經過這條路徑，若建面板這一步會炸掉，等於這次任務
破壞了所有既有的 headless 驗證能力。

⚠️ **2026-09-08 實測更正**：這一節原本寫「`get_active_viewport_window()`
在 headless 回傳 `None`」是撰寫當下（沒有 Isaac Sim 可測）的推測，見 1.12
節——實測發現這台環境的 `SimulationApp({"headless": True})` 內部仍然建立
了一個可用的 viewport（回傳 `weakref.ProxyType`，非 `None`），`get_frame()`
也真的能建出可用的 `Frame` 並放進 2D widget。也就是說在**這個環境**，
`HudPanel` 在 headless 下其實會正常建面板，不會走到「安靜跳過」那個分支
——`verify_manual_controller_wiring.py` 第 7 項的斷言是動態比對
`hud_panel_has_root_frame == viewport_available`（不是寫死「必須是
`None`」），所以這個結果不影響驗證腳本的正確性。這段防禦邏輯本身仍然
保留：換一台真正沒有顯示裝置的環境（CI、無 GPU 容器）行為可能不同，
「拿不到 viewport 就安靜跳過」依然是必要的硬性要求，只是實測證明「headless
必定拿不到 viewport」這個前提本身不成立。

### 6.6 headless 執行的既有陷阱：`ui.tool_menu_registry` 的匯入順序

2026-09-08 實測執行 `probe_omni_ui_shot_panel_widgets.py` 時第一次就在
`SimulationApp` 建構**之前**炸掉：

```
ModuleNotFoundError: No module named 'omni.kit.menu'
```

根因：`tool_menu_registry.py` 內部 `import omni.kit.menu.utils`——這是一個
**純 Python Kit 擴充功能**（不是隨 Kit Python 發行版一起打包的原生模組），
只有在 Kit 的 Extension Manager 載入完那個擴充功能之後才能 import 到。而
`probe_omni_ui_shot_panel_widgets.py`／`probe_viewport_overlay_drag.py`／
`verify_manual_controller_wiring.py` 三支腳本原本在**模組最上層**寫
`from ui.tool_menu_registry import tool_menu_item`——這行在 Python 剖析
整個檔案的當下就會執行，早於檔案最下方 `if __name__ == "__main__":` 區塊
裡才建構的 `SimulationApp(...)`。用最小重現腳本證實：同一個
`import omni.kit.menu.utils`，寫在 `SimulationApp({"headless": True})`
**建構完成之後**執行就完全正常（這個 Kit 版本的 base app 設定檔
`isaacsim.exp.base.python.kit` 有把 `omni.kit.menu.utils` 收進預設擴充
集）。

修法：把這行 import 包進 `try/except ImportError`，失敗時（獨立執行、
Kit 還沒啟動）提供一個 no-op 版的 `tool_menu_item` decorator——獨立執行
模式本來就不透過 Tool Menu 觸發，不需要真正的註冊：

```python
try:
    from ui.tool_menu_registry import tool_menu_item
except ImportError:
    def tool_menu_item(menu_path: str):
        def decorator(func):
            return func
        return decorator
```

已修正上述三支腳本。**`scripts/measure_swing_speed.py`（#176，跟 #115
無關的既有腳本）有完全相同的模組頂層 import 順序**，理論上獨立執行也會
踩到同一個錯誤，但這次沒有動它——超出 #115 的範圍，留給後續處理。

---

## 7. 測試策略

### 7.1 core 層：Unit Test（TDD 先寫，738 → 882）

| 檔案 | 覆蓋重點 |
|---|---|
| `test_manual_shot_bounds.py` | `MANUAL_SHOT_ANGLE == (-160, 160)`；其餘四項與 `action_bounds` 完全相等；手動 `Action` 餵 `normalize_action()` 拋 `ValueError`（可執行的文件，見 8.1 節） |
| `test_manual_shot_parameters.py` | `default()` 對齊 `BREAK_SHOT_POSITIONS[0]`/`CUE_BALL_SPEED[1]`；`to_action()` 每次回新物件；frozen 賦值拋 `FrozenInstanceError`；各欄位越界拋 `ValueError` |
| `test_shot_panel_input_mapper.py` | 圓心/四極值/45° 圓周；拖到圓外方向保持（#222 回歸）；俯瞰圖四角+中心+round-trip；`shot_angle_from_points` 與 `compute_tilted_direction()` 的往返驗證；`clamp_manual_shot_angle` 夾到 ±160；退化/NaN 輸入拋 `ValueError` |
| `test_manual_controller.py` | `test_stays_in_idle_forever_without_shot_request`（跑 100 tick 仍 IDLE，驗收錨點）；一次觸發後回 IDLE 停住；連按合併；球未靜止時保留請求；AIMING 期間改參數不影響 STRIKE；reset 丟請求保留參數；`_pending is None` 進 ERROR |
| `test_manual_shot_feasibility.py` | 開球點 0° 可行；母球在 `CUE_BALL_PLACEMENT_Y` 下界 0° 為 `geometry_unsolvable`（實測掃描找出的座標，不是照抄計畫文件推斷，見該測試檔註解）；`ManualShotParameters` 版本與底層版本結果一致 |

`core/tests` 目前為 **882 passed**（第 6 節收尾時再次確認，見交付標準）。

### 7.2 extension 層：Unit Test 豁免，改用 headless 驗證腳本 + GUI 清單

`HudPanel`/`TableComboBoxModel`/`BilliardExtension` 的接線屬於 UI 元件與
視覺呈現邏輯（依 `docs/unit-test-rules.md` 條件 4/5），比照
`debug-menu-dynamic-tables-tech-design.md` 第 7 節的既有先例，Unit Test
豁免，改用兩種手段：

1. **Headless 驗證腳本 `scripts/verify_manual_controller_wiring.py`**：
   走真實路徑（不 mock 任何 Isaac Sim / core 元件），驗證常駐
   `ManualController` 的接線正確（identity、不自動循環、按鈕觸發、參數
   更新不重擺球、未知 table_id 不拋例外），另外補一項驗證 `HudPanel` 在
   headless 環境安靜不建立內部 widget、extension 啟動不拋例外（見 8.2 節）。
2. **GUI 人工確認清單 `docs/hud-shot-panel-gui-verification-checklist.md`**：
   驗證 headless 驗不出來的部分——overlay 的視覺呈現、滑鼠拖曳與 viewport
   相機操作的互動、收合鈕、俯瞰圖的拖曳手感、瞄準線顏色與擊球鈕的可行性
   閘門是否正確反映在畫面上。

### 7.3 已知限制與待實測項目

- `hud_panel.py` 的 `_create_root_frame()` 與 `_to_local()` 兩處明確隔離
  了 spike 未證實的假設（見第 1 節查證結果與第 6.5 節），**尚未在有
  Isaac Sim 的 GUI 環境實際跑過** `probe_omni_ui_shot_panel_widgets.py`
  與 `probe_viewport_overlay_drag.py`——這兩支腳本本身也只是「寫好、語法
  正確、等有 Isaac Sim 的環境去跑」的狀態。
- overlay 上的滑鼠拖曳會不會被 viewport 相機操作吃掉，是全計畫最大的
  未知，只能在 GUI 下親手驗證，詳見
  `docs/hud-shot-panel-gui-verification-checklist.md` 的「先決條件」一節
  與計畫書的風險表（備援順序：停用相機互動 → 加吃事件的透明層 → 退回
  獨立 `ui.Window`）。
- `get_frame()` 在 viewport 重建（切換渲染器、切換 viewport 佈局）後是否
  仍然有效，官方文件完全沒有討論，需要在 GUI 下順便觀察（viewport 縮放/
  最大化那幾項確認項目）。
- 走位球（cue ball 不受 Kitchen 限制）與 Milestone B 把 `action_bounds.
  SHOT_ANGLE` 改回整圈之後，`manual_shot_bounds.py` 的角度項應該退化為
  直接轉出 `SHOT_ANGLE`；`manual_shot_feasibility.py` 需要補上基座位置
  檢查（`reason` 會多一個 `"base_inside_table"`）——目前刻意不做（YAGNI），
  屆時再處理。
