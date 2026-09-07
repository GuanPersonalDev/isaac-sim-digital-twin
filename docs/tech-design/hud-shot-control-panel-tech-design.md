# HUD 擊球參數控制面板 — 技術設計文件

> 生成時間：2026-09-08
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：#115
> 關聯文件：`C:\Users\guan_\.claude\plans\115-hud-peppy-harbor.md`（完整技術計畫）、`docs/architecture-spec.md`（分層規則、`extension/ui/hud_panel.py` 檔名已預留）

---

## 0. 文件狀態

本文件目前只完成**第 1 節「omni.ui API 查證結果」**，是 #115 階段 0（API spike）的產出。其餘章節（模組清單、類別設計、資料流、依賴關係……）是階段 6「驗證與文件」的產物，尚未撰寫，先以 TODO 佔位，避免之後漏掉章節。

- [ ] 第 2 節：模組清單與職責
- [ ] 第 3 節：類別設計（`ManualShotParameters`／`ManualController`／`HudPanel`／輸入換算純函式……）
- [ ] 第 4 節：資料流（UI 執行緒 ↔ physics 執行緒）
- [ ] 第 5 節：依賴關係
- [ ] 第 6 節：驗證（Unit Test／Headless 驗證腳本／GUI 人工確認清單）

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
