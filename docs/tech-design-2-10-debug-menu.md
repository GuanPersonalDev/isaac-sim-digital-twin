# [2-10] Debug Menu 骨架 — 技術設計文件

> 生成時間：2026-06-19
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：（由 progress-planner 建立 Issue 後補）

---

## 1. 功能概述

Extension 啟動時，建立一個停靠於 Viewport 右側的 Debug UI 視窗（"Billiard Debug"，300×400）。視窗內容區為空 VStack，供後續任務（[6-6]、[7-6]）加入控制按鈕。Extension 關閉時需主動呼叫 `destroy()` 釋放資源。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| DebugMenu | extension/ui | 建立並管理 Billiard Debug 視窗的生命週期，並非同步停靠至 Viewport 右側 | `extension/ui/debug_menu.py` |

---

## 3. 類別設計

### DebugMenu

**職責：** 封裝 `omni.ui.Window` 的建立、停靠、顯示／隱藏與資源釋放，提供供 Extension 主類別呼叫的簡單介面。

**介面：**
```python
class DebugMenu:

    def __init__(self) -> None:
        """建立視窗並非同步停靠至 Viewport 右側"""
        ...

    async def _dock_to_viewport(self) -> None:
        """等待 Viewport 就緒後執行停靠（最多等 5 幀）"""
        ...

    def show(self) -> None:
        """顯示 Debug Menu"""
        ...

    def hide(self) -> None:
        """隱藏 Debug Menu"""
        ...

    def destroy(self) -> None:
        """釋放 UI 資源，Extension 關閉時必須呼叫"""
        ...
```

**依賴：**
- 輸入來源：無外部資料輸入；由 Extension 主類別在 `on_startup` 直接實例化
- 輸出去向：無回傳值；視窗狀態由 `omni.ui.Workspace` 管理

---

## 4. 資料流

```
Extension 啟動（on_startup）
  → DebugMenu.__init__()
    → 建立 omni.ui.Window（"Billiard Debug"，width=300，height=400，visible=True）
    → 建立 VStack 內容區（空，供後續任務加入按鈕）
    → asyncio.ensure_future(_dock_to_viewport())
      → 迴圈最多 5 幀：await omni.kit.app.get_app().next_update_async()
      → 確認 omni.ui.Workspace.get_window("Viewport") 存在
      → self._window.dock_in_window("Viewport", DockPosition.RIGHT, ratio=0.25)
  → 回傳 DebugMenu instance 給呼叫端（self._debug_menu）

Extension 關閉（on_shutdown）
  → self._debug_menu.destroy()
    → self._window.destroy()
    → self._window = None
```

---

## 5. 依賴關係圖

```
DebugMenu
  ├── 依賴 omni.ui（Window、VStack、DockPosition、Workspace）
  ├── 依賴 omni.kit.app（next_update_async，等待 Viewport 就緒）
  └── 依賴 asyncio（ensure_future，非阻塞停靠）

Extension 主類別
  └── 持有 DebugMenu instance（on_startup 建立，on_shutdown 銷毀）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| `__init__` 時 Viewport 尚未就緒 | 改用 `asyncio.ensure_future` 非同步停靠，避免阻塞 Extension 啟動流程 |
| Viewport 5 幀內仍未出現 | `_dock_to_viewport` 等待最多 5 幀後放棄停靠，視窗保持浮動狀態（不拋例外） |
| `_window` 宣告為區域變數（GC 陷阱） | 必須宣告為 `self._window`，否則 `__init__` 返回後視窗立即被 GC 銷毀 |
| Extension 關閉時未呼叫 `destroy()` | `omni.ui.Window` 資源不會自動釋放，造成記憶體洩漏；`on_shutdown` 必須主動呼叫 |

---

## 7. 測試涵蓋（對應 Unit Test）

`DebugMenu` 依賴 `omni.ui`（Isaac Sim 引擎 UI 框架），屬於第三方 API 薄層，**Unit Test 豁免**。

驗收方式改為手動確認：
1. Extension 啟動後，"Billiard Debug" 視窗出現並停靠於 Viewport 右側。
2. Extension 關閉後，視窗消失且無記憶體洩漏警告。

---

## 8. 待決定事項

- [ ] [6-6] 在 VStack 加入 Ball Reset 按鈕
- [ ] [7-6] 在 VStack 加入 LivePosition 相關控制按鈕
