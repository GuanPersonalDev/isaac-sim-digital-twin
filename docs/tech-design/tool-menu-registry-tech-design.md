# 工具型腳本選單註冊機制（Tool Menu Registry）— 技術設計文件

> 生成時間：2026-07-16
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：Issue #176（`scripts/measure_swing_speed.py` 為本機制的第一個使用案例，依相關調整見第 9 節）

---

## 1. 功能概述

`scripts/` 目錄下會累積一次性工具腳本（例如 #176 的 `measure_swing_speed.py`），目前只能用 `python.bat scripts/xxx.py` 獨立執行，無法存取 Kit 編輯器中目前開啟的場景狀態。本功能提供類似 Unity `[MenuItem]` 的體驗：透過 `@tool_menu_item` decorator 標記工具函式，Extension 啟動時自動掃描 `scripts/` 目錄並將這些函式掛載到 Omniverse Kit 主選單，使用者點擊選單項目即可在目前執行中的 Kit session 內、共用當前 stage 呼叫該函式。

已確認 Omniverse Kit 生態系（`omni.kit.menu.utils`、`omni.kit.actions.core`）沒有原生 decorator/自動掃描機制，僅提供命令式 API（`MenuItemDescription` + `add_menu_items()` / `remove_menu_items()`），因此本功能在 `extension/ui/` 層自建一層包裝。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `tool_menu_item` | extension（UI 元件） | decorator，將函式登記進模組內全域清單，不呼叫 Kit API | `extension/ui/tool_menu_registry.py`（新增） |
| `discover_and_register` | extension（UI 元件） | 動態掃描並 import `scripts/` 下的 `.py`，觸發 decorator 註冊，再呼叫 `omni.kit.menu.utils.add_menu_items()` | `extension/ui/tool_menu_registry.py`（新增） |
| `unregister` | extension（UI 元件） | 呼叫 `omni.kit.menu.utils.remove_menu_items()` 移除選單項目 | `extension/ui/tool_menu_registry.py`（新增） |
| `_wrap` | extension（UI 元件，內部函式） | 包裝工具函式的例外處理（印 log 後 re-raise） | `extension/ui/tool_menu_registry.py`（新增） |
| `BilliardDigitalTwinExtension.on_startup` / `on_shutdown` | extension（Extension 進入點） | 在啟動時呼叫 `discover_and_register`、關閉時呼叫 `unregister` | `extension/billiard_digital_twin/billiard_digital_twin.py`（修改） |
| `check_joint_limits`（等既有工具函式） | scripts（獨立工具腳本） | 移除模組層級 `SimulationApp(...)`，改用 `@tool_menu_item` + `if __name__ == "__main__":` guard，同時支援獨立執行與選單呼叫 | `scripts/measure_swing_speed.py`（既有規劃調整，不在本次實作範圍，見第 9 節） |

---

## 3. 類別設計

### `tool_menu_registry`（模組，非類別）

**職責：** 提供工具腳本註冊為 Kit 主選單項目的機制，隔離 `omni.kit.menu.utils` 的命令式 API 細節。

**介面：**
```python
def tool_menu_item(menu_path: str) -> Callable:
    """decorator：標記函式為選單項目。menu_path 例如
    'Billiard/Measure Swing Speed - Check Joint Limits'。
    純粹在 import 當下把 (menu_path, func) 註冊進模組內的全域清單
    _REGISTERED_TOOLS，不做任何 Kit API 呼叫。"""
    ...

def discover_and_register(scripts_dir: str, top_menu_name: str) -> list["MenuItemDescription"]:
    """遞迴掃描 scripts_dir 下所有 .py（用
    pathlib.Path(scripts_dir).rglob("*.py")，跳過檔名以 _ 開頭的檔案），
    用 importlib.util.spec_from_file_location + module_from_spec +
    exec_module 逐一 import 觸發模組內的 @tool_menu_item 執行。
    呼叫前先 _REGISTERED_TOOLS.clear()（處理 extension reload 的重複註冊問題）。
    把 _REGISTERED_TOOLS 包成
    [MenuItemDescription(name=path, onclick_fn=_wrap(func)), ...]，
    註冊時若發現同一 menu_path 出現兩次，raise ValueError 明確報錯（不允許並存）。
    呼叫 omni.kit.menu.utils.add_menu_items(items, top_menu_name)。
    回傳 items 清單，供呼叫端存起來給 unregister 用。"""
    ...

def unregister(menu_items: list["MenuItemDescription"], top_menu_name: str) -> None:
    """呼叫 omni.kit.menu.utils.remove_menu_items(menu_items, top_menu_name)。
    注意：傳入的 menu_items 必須是 discover_and_register 回傳的同一個 list
    物件（或內容一致），不可在 shutdown 時重新 new 一份，否則移除會失敗、
    reload 時選單項目重複。"""
    ...

def _wrap(func: Callable) -> Callable:
    """內部函式：包一層 try/except。工具函式執行拋例外時，
    先在 console 印 f"[ToolMenu] {menu_path} failed: {exc}"，再 re-raise（不吞例外）。"""
    ...
```

**依賴：**
- 輸入來源：`scripts/` 目錄下標記 `@tool_menu_item` 的工具腳本；`omni.kit.menu.utils.MenuItemDescription`
- 輸出去向：`BilliardDigitalTwinExtension.on_startup` / `on_shutdown` 呼叫並持有回傳的 `items` 清單

---

## 4. 資料流

```
Extension.on_startup()
  → tool_menu_registry.discover_and_register(scripts_dir, "Tools")
    → _REGISTERED_TOOLS.clear()
    → pathlib.Path(scripts_dir).rglob("*.py") 逐一 importlib 動態載入
      → 每個模組頂層的 @tool_menu_item(...) 把函式登記進 _REGISTERED_TOOLS
    → 檢查 menu_path 重複 → 重複則 raise ValueError
    → 包成 [MenuItemDescription(name=path, onclick_fn=_wrap(func)), ...]
    → omni.kit.menu.utils.add_menu_items(items, "Tools")
  → 回傳 items（extension 存成 self._tool_menu_items）

使用者點擊主選單項目
  → _wrap 包裝後的 onclick_fn 執行
    → try: 呼叫原函式（in-process，函式內用
      omni.usd.get_context().get_stage() 取得目前場景）
    → except: print(f"[ToolMenu] {menu_path} failed: {exc}") 後 re-raise

Extension.on_shutdown()
  → tool_menu_registry.unregister(self._tool_menu_items, "Tools")
  → 清空 self._tool_menu_items
```

---

## 5. 依賴關係圖

```
extension/billiard_digital_twin/billiard_digital_twin.py
  └── 依賴 extension/ui/tool_menu_registry.py（呼叫 discover_and_register / unregister）

extension/ui/tool_menu_registry.py
  ├── 依賴 omni.kit.menu.utils（MenuItemDescription / add_menu_items / remove_menu_items）
  └── 依賴 scripts/ 目錄下標記 @tool_menu_item 的工具腳本（動態 import）

scripts/measure_swing_speed.py（既有規劃調整，見第 9 節）
  ├── 依賴 extension/ui/tool_menu_registry.py（import tool_menu_item）
  └── 依賴 omni.usd（get_context().get_stage()，取得目前 Kit session 的 stage）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 兩個工具腳本使用相同的 `menu_path` | `discover_and_register` 檢查到重複時 `raise ValueError`，不允許並存，啟動時即失敗以利及早發現 |
| Extension reload（`on_startup` 被重複呼叫） | `discover_and_register` 開頭先 `_REGISTERED_TOOLS.clear()`，避免舊清單殘留造成重複註冊 |
| `on_shutdown` 移除選單時傳入了新建的 list 而非原始 items | 屬於呼叫端誤用；文件明確規範必須傳入 `discover_and_register` 回傳的同一個 list（或內容一致），否則 `remove_menu_items` 會移除失敗、reload 時選單項目重複 |
| `scripts_dir` 下有檔名以 `_` 開頭的檔案（如 `_helper.py`） | `discover_and_register` 掃描時明確跳過，不觸發 import |
| 工具函式執行時拋例外 | `_wrap` 先印 `f"[ToolMenu] {menu_path} failed: {exc}"` 到 console，再 re-raise，不吞例外，方便使用者在 Kit console 直接看到錯誤 |
| 工具腳本仍需支援獨立執行（`python.bat scripts/xxx.py`） | 工具函式本體不得有模組層級的 `SimulationApp(...)` 呼叫，改用 `if __name__ == "__main__":` guard 建立 `SimulationApp`，選單呼叫時則直接 in-process 呼叫函式本體，共用目前 Kit session 的 stage |

---

## 7. 測試涵蓋（對應 Unit Test）

`tool_menu_item` decorator 與 `discover_and_register` / `unregister` 屬於直接呼叫 `omni.*` 的 Kit 膠水層（比照 `architecture-spec.md` 對 `extension/` 層的定義），依規範不強制 pytest 純函式單元測試，因為需要 Isaac Sim 執行環境才能驗證 `omni.kit.menu.utils` 實際掛載行為。

以下行為在本文件明確定義，供之後人工驗證或 Codex 驗證使用：

| 驗證項目 | 說明 |
|---|---|
| 重複 `menu_path` 拋出 `ValueError` | `discover_and_register` 掃描到兩個相同 `menu_path` 時應明確報錯，不可靜默覆蓋 |
| `_wrap` 例外處理不吞例外 | 工具函式拋例外時，console 應印出 `[ToolMenu] {menu_path} failed: {exc}`，且例外需 re-raise 而非被吞掉 |
| `extension reload` 不會造成選單項目重複 | 重複呼叫 `discover_and_register` 前應先清空 `_REGISTERED_TOOLS` |
| `unregister` 需傳入原始 items | 傳入非 `discover_and_register` 回傳的同一份 list 時應能觀察到移除失敗的現象 |

---

## 8. 待決定事項

- [ ] 無（本次設計已與使用者完整確認）

---

## 9. 相依調整記錄（不在本文件實作範圍內）

`scripts/measure_swing_speed.py`（對應 Issue #176）需配合調整，以同時支援「獨立執行」與「選單呼叫（in-process，共用目前 Kit session 的 stage）」：

- 工具函式本體不能有模組層級的 `SimulationApp(...)` 呼叫
- 改用 `if __name__ == "__main__":` guard，只在直接執行時才建立 `SimulationApp`

```python
from extension.ui.tool_menu_registry import tool_menu_item

@tool_menu_item("Billiard/Measure Swing Speed - Check Joint Limits")
def check_joint_limits():
    # 假設 app/stage 已存在，直接用 omni.usd.get_context().get_stage()
    ...

if __name__ == "__main__":
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    check_joint_limits()
    simulation_app.close()
```

此項調整不在本次技術文件的實作範圍內，僅在此記錄以利後續 Issue 規劃與程式碼實作對照。
