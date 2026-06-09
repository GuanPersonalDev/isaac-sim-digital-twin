# Unit Test 判斷規則

## 適用範圍
Phase 3 Isaac Sim / Omniverse Kit Extension 專案開發期間，用於決定某個函式或模組是否需要撰寫 Unit Test。

---

## 專案架構前提

本專案採用雙層架構：

```
core/          ← 商業邏輯、計算邏輯、狀態機、資料處理
               ← 不依賴任何 Isaac Sim / Omniverse API
               ← 使用 pytest 在 WSL2 直接執行測試

extension/     ← 薄層橋接
               ← 負責呼叫 Isaac Sim API、UI 渲染、資料注入 core/
               ← 使用整合測試或 Debug Menu 手動驗證
```

**Unit Test 只針對 `core/` 層撰寫。**

---

## 不寫 Unit Test 的條件

滿足以下任一條件，則不寫 Unit Test：

### 條件 1：直接依賴 Isaac Sim 物理引擎結果的邏輯
仿真行為由物理引擎決定，無法給出確定性斷言。
改用整合測試（在 Isaac Sim 啟動環境下手動或腳本驗證）。

**範例：**
- 取得 Rigid Body 碰撞後的速度
- 讀取關節力矩的仿真結果
- 判斷抓取是否成功（依賴物理接觸判定）

### 條件 2：一次性的場景設定腳本
只執行一次、無重複呼叫邏輯、輸出是 USD 場景狀態而非函式回傳值。

**範例：**
- 初始化場景、擺放 Prim 位置
- 設定材質、燈光
- 載入 URDF 並轉換為 USD

### 條件 3：直接包裝第三方 API 的薄層函式
函式內部只有一至兩行，邏輯等同於 API 本身，測試等於在測第三方函式庫。

**範例：**
```python
# 這種函式不需要 Unit Test
def publish_status(client, topic, payload):
    client.publish(topic, json.dumps(payload))
```

### 條件 4：UI 元件本身（`extension/` 層）
UI 不寫 Unit Test，但架構上強制要求資料與 UI 解耦：
- UI 只接受注入的資料介面（Dependency Injection）
- 開發期間注入假資料，正式運行時注入真實資料
- UI 元件本身的正確性由視覺驗證負責

### 條件 5：視覺呈現邏輯（`extension/` 層）
不寫 Unit Test，改用 **Debug Menu** 手動驗證：
- 每個視覺功能對應 Debug Menu 中的一個按鈕
- 按鈕觸發特定狀態，確認呈現結果是否正確

---

## 需要寫 Unit Test 的判斷

**不滿足上述任何一個條件 → 需要寫 Unit Test。**

常見需要寫 Unit Test 的情況：

- 有明確輸入輸出的純函式（座標轉換、數值計算、格式轉換）
- 狀態機的狀態切換邏輯
- 抽象介面的具體實作（`ScriptController`、`ModelController`）
- 被多個地方呼叫的共用函式
- 資料驗證、閾值判斷邏輯
- MQTT / 外部連線的 Client 實體連線測試（整合測試層級）

---

## 測試先行原則（TDD）

Unit Test 必須在實作前撰寫，順序如下：

```
1. 設計類別與函式簽名（確定放在哪個類別、輸入輸出是什麼）
2. 撰寫 Unit Test（此時實作尚未完成，測試預期會失敗）
3. 實作功能，讓測試通過
4. Refactor，確認測試仍然通過
```

---

## Mock 使用規則

當函式邏輯需要測試，但內部呼叫了外部依賴（Isaac Sim API、MQTT、檔案系統）時，使用 `unittest.mock` 隔離依賴：

```python
from unittest.mock import MagicMock, patch

def test_controller_state_transition():
    # Mock 掉 Isaac Sim API，只測試狀態切換邏輯
    mock_isaac_api = MagicMock()
    controller = ScriptController(isaac_api=mock_isaac_api)
    
    controller.on_reach_target()
    
    assert controller.state == ControllerState.PICKING
```

**Mock 使用時機：**
- 函式內部呼叫了 Isaac Sim API，但邏輯本身是自己寫的
- 函式內部有 MQTT、網路、檔案 I/O 等外部依賴
- 需要模擬特定的錯誤情境

---

## 測試框架

- **框架：** `pytest`
- **Mock 工具：** `unittest.mock`（Python 標準函式庫）
- **執行環境：** WSL2，不需要啟動 Isaac Sim
- **執行指令：** `pytest core/tests/`

---

## 邊界案例說明

| 案例 | 結論 | 理由 |
|---|---|---|
| 座標轉換函式，輸入來自 BBoxCache | ✅ 需要寫 | 轉換邏輯是自己實作的純計算，有座標系錯誤風險 |
| MQTT `publish()` 薄層函式 | ❌ 不寫 | 滿足條件 3，等同於包裝第三方 API |
| MQTT Client 連線測試 | ✅ 需要寫 | 連線行為是整合測試層級，需要驗證 |
| `ScriptController` 狀態機邏輯 | ✅ 需要寫 | 狀態切換邏輯是自己實作的，用 Mock 隔離 Isaac Sim API |
| Isaac Sim 物理碰撞判定 | ❌ 不寫 | 滿足條件 1，結果由物理引擎決定 |
