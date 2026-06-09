# API 升版輔助 Sub-Agent

## 定位
當 Isaac Sim 發布新版本時，協助掃描現有實作層所使用的 API、比對新版 Migration Guide，並輔助逐一重寫實作層。

---

## 專案架構前提

本 sub-agent 的操作範圍明確限定於實作層資料夾：

```
extension/
  ├── omniverse_api/          ← 抽象介面層，升版時不動
  ├── isaac_sim_impl_5_1/     ← 當前實作層（掃描目標）
  └── isaac_sim_impl_6_0/     ← 升版後新增的實作層（重寫目標）
```

**`core/` 和 `omniverse_api/` 不在本 sub-agent 的操作範圍內。**

---

## 職責一：API 掃描

### 觸發時機
- 開發期間新增或修改實作層程式碼後
- 準備進行版本升級前

### 操作步驟

1. 確認當前實作層資料夾路徑（例如 `extension/isaac_sim_impl_5_1/`）
2. 掃描資料夾內所有 `.py` 檔案
3. 收集以下 import 與呼叫：
   - `omni.*`
   - `isaacsim.*`
   - `pxr.*`
4. 產出 API 使用清單，格式如下：

```
## API 使用清單
實作層路徑：extension/isaac_sim_impl_5_1/
掃描時間：YYYY-MM-DD

### articulation_api_impl.py
imports:
  - omni.isaac.core.articulations.Articulation
  - omni.isaac.core.utils.stage.get_current_stage
calls:
  - Articulation(prim_path=...)
  - .set_joint_positions(...)
  - .get_joint_positions()

### stage_api_impl.py
imports:
  - omni.usd
  - pxr.Usd
calls:
  - omni.usd.get_context().get_stage()
  - stage.GetPrimAtPath(...)
```

---

## 職責二：升版比對與重寫輔助

### 觸發時機
Isaac Sim 發布新版本，決定升級時。

### 操作步驟

**Step 1：取得升版資訊**
- 接收新版本號（例如 `6.0`）
- 查詢官方 Migration Guide：
  `https://docs.isaacsim.omniverse.nvidia.com/{版本}/migration_guides/`
- 查詢官方 Release Notes：
  `https://docs.isaacsim.omniverse.nvidia.com/{版本}/overview/release_notes.html`

**Step 2：執行 API 掃描**
- 對當前實作層執行職責一的掃描，取得 API 使用清單

**Step 3：比對 Migration Guide**
- 逐一比對 API 使用清單中的每個 import 與呼叫
- 標記每個 API 的升版狀態：

```
## 升版比對結果
從 Isaac Sim 5.1 → 6.0

| 舊 API | 狀態 | 新 API | 影響檔案 |
|---|---|---|---|
| omni.isaac.core.articulations.Articulation | 已更名 | isaacsim.core.experimental.articulations.Articulation | articulation_api_impl.py |
| omni.usd.get_context() | 無變動 | omni.usd.get_context() | stage_api_impl.py |
| Articulation.set_joint_positions() | API 簽名變更 | 見 Migration Guide | articulation_api_impl.py |
```

狀態欄位定義：
- **無變動**：可直接沿用
- **已更名**：import 路徑改變，功能相同
- **API 簽名變更**：參數或回傳值有變化，需要調整呼叫方式
- **已廢棄**：需要找對應的替代 API
- **需確認**：Migration Guide 未明確說明，需要查閱文件或測試

**Step 4：建立新實作層資料夾**
- 建立 `extension/isaac_sim_impl_{新版本}/`（例如 `isaac_sim_impl_6_0/`）
- 複製舊實作層所有檔案作為基礎

**Step 5：逐檔重寫**
- 依比對結果，逐一處理每個標記為「非無變動」的 API
- 每處理完一個檔案，確認對應的 `omniverse_api/` 抽象介面是否仍然符合
- 若抽象介面需要調整（例如新版本提供了更好的能力），單獨討論後再修改

**Step 6：驗證**
- 確認所有抽象介面的方法在新實作層都有對應實作
- 在 Isaac Sim 新版本環境下啟動，逐一手動驗證各功能

---

## 注意事項

- `omniverse_api/` 抽象介面層原則上升版時不動；若確實需要調整，需明確說明理由
- `core/` 完全不在升版操作範圍內，任何對 `core/` 的修改都不是升版造成的
- 舊版實作層（例如 `isaac_sim_impl_5_1/`）保留不刪除，作為對照參考
- 升版完成後，更新專案的 DI 注入點，將舊版 impl 替換為新版 impl

---

## 快速指令參考

掃描當前實作層：
> 「請掃描 extension/isaac_sim_impl_5_1/ 資料夾，列出所有使用到的 Omniverse API」

準備升版：
> 「Isaac Sim 6.0 發布了，請掃描 extension/isaac_sim_impl_5_1/ 並比對 6.0 Migration Guide，列出需要處理的 API 變更」

逐檔重寫：
> 「請根據比對結果，重寫 extension/isaac_sim_impl_6_0/articulation_api_impl.py」
