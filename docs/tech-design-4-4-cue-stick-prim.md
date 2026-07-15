# Demo 桌機器人球桿 Prim 建立 — 技術設計文件

> 生成時間：2026-07-15
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/88

---

## 1. 功能概述

使用者已在 Isaac Sim 環境中手動建好球桿的 USD 資產（`assets/ball_stick.usd`），並確認過尺寸比例合理。本次任務將該資產以 `StageAPI.create_reference_prim` 的既有引用模式，接入 Demo 桌場景中：只有 `TableRobotManager`（唯一被實例化、供 Demo 展示用的桌 + 機器人場景）需要建立球桿 Prim，訓練用桌（無 UI 展示需求）不需要。輸入為 `ball_stick.usd` 資產路徑與機器人所在的世界座標；輸出為 Stage 中新增的球桿 Prim，初始位置與 UR5 機器人相同（暫定位置，避免停留於 Stage 原點與其他物件重疊）。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `TableRobotManager` | core/models | 新增：於建構子內引用球桿資產並定位；新增 `get_cue_stick_prim_path()` | `core/models/table_robot_manager.py` |
| `asset_utility` | core/services | 新增 `CUE_STICK_PATH` 常數，指向 `assets/ball_stick.usd` | `core/services/asset_utility.py` |

本次不新增獨立 class（例如 `CueStick`）。設計討論階段已明確判定：球桿目前僅是資產引用，沒有額外程式邏輯或行為，直接於 `TableRobotManager` 內用既有 `StageAPI` 方法處理即可，比照 `UR5Robot`/`BilliardTable` 對資產引用的一致做法。

---

## 3. 類別設計

### TableRobotManager（修改部分）

**職責（新增）：** 於 `__init__` 建立球桿 Prim（引用 `CUE_STICK_PATH`），以機器人世界座標作為暫定位置；並提供球桿 Prim 路徑查詢。

**介面：**
```python
class TableRobotManager:
    _ROBOT_OFFSET_FROM_TABLE_CENTER = (1.5, 0.0, 0.0)

    def __init__(
        self,
        table_center: tuple[float, float, float],
        base_path: str,
        stage_api: StageAPI,
    ) -> None:
        """建立 UR5Robot，並引用球桿資產（CUE_STICK_PATH）建立於
        {base_path}/CueStick，暫定位置與機器人相同（world_position）。
        """
        ...

    def get_robot_prim_path(self) -> str:
        """回傳 Robot Prim 的完整路徑。（既有，不變）"""
        ...

    def get_cue_stick_prim_path(self) -> str:
        """回傳球桿 Prim 的完整路徑，例如 /World/DemoTable/CueStick。
        後續 Issue #89（球桿與 UR5 末端的 Fixed Joint）將使用本方法
        取得球桿 Prim 路徑以建立關節連結。
        """
        ...

    def destroy(self) -> None:
        """既有，不變；沿用現有模式不做主動 Prim 刪除。"""
        ...
```

**依賴：**
- 輸入來源：`StageAPI`（外部注入）、`asset_utility.CUE_STICK_PATH`、建構子既有的 `table_center` / `base_path`
- 輸出去向：`get_cue_stick_prim_path()` 供 Issue #89 的 Fixed Joint 建立邏輯使用

---

### asset_utility（修改部分）

**新增常數：**
```python
CUE_STICK_PATH = os.path.join(ASSET_DIR, "ball_stick.usd")
```

---

## 4. 資料流

```
BilliardExtension._billiard_init()
  → TableRobotManager.__init__(table_center, base_path, stage_api)
    → world_position = table_center + _ROBOT_OFFSET_FROM_TABLE_CENTER   （既有邏輯，不變）
    → UR5Robot(base_path, stage_api, world_position)                    （既有，不變）
    → stage_api.create_reference_prim(base_path + "/CueStick", CUE_STICK_PATH)   （新增）
    → stage_api.set_prim_translate(base_path + "/CueStick", *world_position)     （新增）
  → self._cue_stick_prim_path = base_path + "/CueStick" 存於實例內
  → 回傳 TableRobotManager 實例
  → 之後（Issue #89）：呼叫端透過 get_cue_stick_prim_path() 取得球桿路徑，
    建立與 UR5 末端執行器的 Fixed Joint
```

---

## 5. 依賴關係圖

```
TableRobotManager
  ├── 依賴 UR5Robot（既有，建立並持有機器人 Prim 引用）
  ├── 依賴 StageAPI（新增用法：create_reference_prim、set_prim_translate 建立球桿 Prim）
  └── 依賴 asset_utility.CUE_STICK_PATH（新增，球桿資產路徑常數）

asset_utility
  └── 依賴 assets/ball_stick.usd（外部資產檔案，使用者已於 Isaac Sim 中手動建立並確認比例）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| `ball_stick.usd` 資產檔尚未放入 `assets/` 資料夾 | 不做額外防呆，沿用 `UR5Robot`/`BilliardTable` 既有語意：`create_reference_prim` 失敗時直接拋出底層例外，由呼叫端（Extension 初始化）自然中斷 |
| 球桿初始位置與機器人重疊 | 屬預期行為（暫定位置，非最終擺位）；Issue #89 會以 Fixed Joint 將球桿移至 UR5 末端執行器，本 Issue 不處理最終擺位 |
| 訓練用桌是否也需要球桿 | 不需要；`TableRobotManager` 僅在 Demo 桌情境被實例化一次，訓練用桌不經過此路徑 |

---

## 7. 測試涵蓋（對應 Unit Test）

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| `test_table_robot_manager_creates_cue_stick_reference_prim` | `core/tests/test_table_robot_manager.py` | 驗證 `stage_api.create_reference_prim` 以 `base_path + "/CueStick"` 與 `CUE_STICK_PATH` 被呼叫 |
| `test_table_robot_manager_sets_cue_stick_translate` | `core/tests/test_table_robot_manager.py` | 驗證 `stage_api.set_prim_translate` 收到與 `UR5Robot` 相同的 `world_position` |
| `test_table_robot_manager_get_cue_stick_prim_path` | `core/tests/test_table_robot_manager.py` | 驗證 `get_cue_stick_prim_path()` 回傳 `{base_path}/CueStick` |

（既有三項測試 `test_table_robot_manager_creates_robot_with_offset_position`、`test_table_robot_manager_get_robot_prim_path`、`test_table_robot_manager_destroy` 不受影響，維持通過。）

---

## 8. 待決定事項

- [ ] 無（本次範圍已於設計討論四階段完成確認：不建立獨立 `CueStick` class、初始位置沿用機器人座標、`destroy()` 不做主動刪除）
- 後續依賴：Issue #89（球桿與 UR5 末端的 Fixed Joint）將直接依賴本次新增的 `get_cue_stick_prim_path()`，屆時球桿的暫定位置會被關節約束取代
