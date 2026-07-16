# Task #176 [4-8] 空揮測速 — 開發交接規格

> 本文件為 GitHub Issue #176 的完整開發規格，內容彙整自 Issue body 與兩則規劃補全 comment（2026-07-16 定案）。
> Issue: https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/176

---

## 1. 任務背景與目標

**困難點 #0 / #2 解法**：把「空揮測速」提前到 Milestone A（RL 訓練）之前執行，避免 A 訓練完才發現動作空間速度範圍超出 UR5 實際能力而重訓。

**一句話目標**：在不含撞球桌的獨立場景，以 UR5＋球桿（0.5 kg）＋Fixed Joint 執行可重現的全速空揮，量測**球桿桿尖**峰值線速度，核對 USD 關節 velocity/effort limits，產出 Milestone A 出桿速度上限。

**預期結果**：修正後估算 3–5 m/s（非樂觀估計的 6–8 m/s）。若僅達 4 m/s 等級，接受「非 full-power break」，demo 敘事改為 "robot break shot at achievable velocity"，**不視為失敗**。

**排程位置**：本週最高優先。順序固定：#176（本任務）→ #178（early termination 確認）→ 完成後 Milestone A 訓練配置才定案。預估 3h，排定完成日 2026-07-19。

---

## 2. 專案環境

| 項目 | 值 |
|---|---|
| Isaac Sim | 6.0.0.1（`pip install isaacsim[all,extscache]==6.0.0.1 --extra-index-url https://pypi.nvidia.com`） |
| Python | 3.12 |
| Core API | Warp-based Core Experimental API |
| OS | Windows 11 |
| 專案根目錄 | `isaac-sim-digital-twin/` |

**前置依賴（全部已完成）**：
- #87 [4-3] 關節結構確認（可用 API 讀關節數量與名稱）— CLOSED
- #88 [4-4] 球桿幾何體（USD Prim）— CLOSED
- #89 [4-5] 球桿與 UR5 末端 Fixed Joint（含子任務 #190/#191）— CLOSED 2026-07-16

---

## 3. 架構決策：standalone script，不走 core/ 抽象層

- `core/ports/articulation_api.py` 是 **Cartesian-space** 介面且**尚未有實作**（#95 排在後面）；空揮測速需要 joint-space 全速控制，與該介面定位不合。
- **決議**：本任務寫成 standalone script — **`scripts/measure_swing_speed.py`**（`scripts/` 目錄需新建），以 SimulationApp（headless 或 GUI 皆可）直接使用 Isaac Sim 6.0.0 原生 Articulation API。
- **不 import `core/` 任何模組**、不受專案分層規範限制。量測結論（數字）才是交付物，script 屬一次性工具。

---

## 4. 場景組成

| 元素 | 來源 | 備註 |
|---|---|---|
| UR5 | Nucleus 現成 USD：`Isaac/Robots/UniversalRobots/ur5/ur5.usd` | 末端 link 名稱：`wrist_3_link` |
| 球桿 | `assets/ball_stick.usd` | Mass = 0.5 kg（由 asset 管理，程式不覆寫；載入後順便確認仍為 0.5 kg） |
| Fixed Joint | 程式建立 | 掛載流程照抄現有 `core/models/table_robot_manager.py`（見附錄 A），改用原生 API |
| 撞球桌 | **不載入** | |
| 重力 | **關閉** | 空揮不需要，避免手臂下垂干擾量測 |

Physics dt：與 Milestone A 訓練相同設定（60 Hz）；另跑一次 240 Hz 確認峰值無明顯差異（optional）。

---

## 5. 量測規格

### 5.1 量測點：「TCP」= 球桿桿尖，不是 wrist_3_link / tool flange

UR5 datasheet 的 tool speed（約 1 m/s）指 flange；本任務要的是桿尖線速度（槓桿臂放大後可達 3–5 m/s）。桿尖非獨立 prim，量測方式：

- **(a) 剛體運動學（建議）**：`v_tip = v_com + ω × r`
  - `v_com`：球桿剛體質心線速度
  - `ω`：球桿剛體角速度
  - `r`：質心到桿尖的向量（桿長已知，見 #88 幾何定義）
- (b) 每 physics step 取桿尖世界座標做有限差分（易受 dt 影響，備援用）

### 5.2 揮桿軌跡

- 姿態：手臂**伸展姿**（最大化桿尖到旋轉軸距離）
- 量兩組：
  - **(A) 單關節全速**（wrist_1 或 elbow，貼近 Milestone B 實際揮桿型態）→ **採用值取這組**
  - (B) shoulder+elbow+wrist 同向疊加全速（理論上限，僅參考記錄，optional）
- 揮桿行程要夠長，確認關節速度已達 limit 飽和（速度曲線出現平頂）再取峰值
- 起始/目標關節角不預先定死，但**執行時須把實際採用的姿態與軌跡參數記錄在關閉備註**，確保可重現

### 5.3 取樣規則

- 排除 Fixed Joint 生效前的暫態：**前 0.5 s 不取樣**
- **相同軌跡至少執行 3 次**，記錄各次峰值與採用值（GPU PhysX 有非決定性，單次量測不足以定案）
- 峰值取速度曲線飽和段的最大值

---

## 6. Asset limit 檢查規則

1. 讀取 UR5 USD 各關節 `physxJoint:maxJointVelocity` 與 drive `maxForce`。
2. **逐關節列表**：USD 屬性值、**單位**、實機 ±180°/s 對照結果。
   - ⚠️ **單位陷阱**：PhysX USD 的 `maxJointVelocity` 對 revolute joint 單位是 **deg/s**，與 rad/s 混淆會讓結論差 57 倍。表格務必附單位。
3. 實機規格：UR5 全關節 ±180°/s。
4. **若 USD 值 > 實機 → 先覆寫為實機值再測**（否則峰值虛高，A 訓練出的 policy 實機做不到）；若 ≤ 實機 → 記錄即可。
5. effort limit 記錄備查，不強制對齊。**措辭注意**：若控制方式未實際觸發 torque saturation，只能寫「記錄設定值」，不能宣稱「負載能力已驗證」。

---

## 7. 建議執行順序（3h 預算）

| 步驟 | 內容 | 預估 |
|---|---|---|
| 1 | limit 檢查：載入 UR5 USD，產出逐關節 velocity/effort limit 表格（含單位、對照結果） | ~30min |
| 2 | 測速場景：`scripts/measure_swing_speed.py` — SimulationApp + UR5 + 球桿 + Fixed Joint，關重力，不載桌 | ~60min |
| 3 | 姿態 (A) 量測：單關節全速揮桿 ×3 次，`v_tip = v_com + ω × r`，確認飽和曲線 | ~60min |
| 4 | 寫回：關閉備註（數據＋姿態參數）、更新 task-breakdown、#177/#110 留備註 | ~30min |

姿態 (B) 與 240 Hz 驗證為 optional，超時直接砍。

---

## 8. 產出物與寫回位置

量出的峰值不能只留在 console：

1. **Issue #176 關閉備註**：3 次量測數據＋採用值＋實際採用的姿態與軌跡參數。
2. **`docs/phase3-task-breakdown.md`**：物理參數表「出桿速度範圍 0.5 ~ 7.0 m/s」（目前約第 110 行）→ 上限改為實測值；「RL 設計定案 → Action」的速度範圍同步更新。
3. **#177**（[5-11] impulse-based 擊球）：留備註，動作空間 clip 上限引用此值。
4. **#110**（[7-3] Action 資料格式）：留備註，桿速欄位範圍依此值定案。
5. 若峰值 < 4 m/s 等級：demo 敘事更新為 "break shot at achievable velocity"。

---

## 9. 驗收標準（Definition of Done）

- [ ] 獨立測速場景（`scripts/measure_swing_speed.py`）可重複執行
- [ ] UR5、CueStick、Fixed Joint 載入正確；球桿 Mass 確認 0.5 kg
- [ ] TCP 定義在實際桿尖位置（`v_tip = v_com + ω × r`）
- [ ] 逐關節 velocity/effort limit 表已產出（值＋單位＋±180°/s 對照；必要時已覆寫為實機值）
- [ ] 姿態 (A) 相同軌跡執行 ≥3 次，輸出各次峰值與採用值；速度曲線已確認飽和
- [ ] 全程無 Fixed Joint 爆震、球桿異常跳動或明顯穿透
- [ ] 姿態 (B) 參考數據（optional，超時可砍）
- [ ] 實測結果正式寫回 task-breakdown 與 #177、#110
- [ ] 若峰值 < 4 m/s：更新 demo 敘事，不視為失敗

---

## 10. 風險提示

- **單位陷阱**：`maxJointVelocity` 是 deg/s，勿當 rad/s。
- **爆震處理方向**：若全速揮桿時 Fixed Joint 爆震，優先調 joint 解算參數（solver iteration 等），**不要降速測**——降速測出的不是真峰值。
- **effort 措辭**：未觸發 torque saturation 就只能寫「記錄設定值」。
- **量測點**：讀球桿剛體速度換算桿尖，不是 `wrist_3_link` 速度。
- **行程不足**：揮桿行程太短會導致關節未達 velocity limit 就結束，峰值偏低——務必確認速度曲線平頂。

---

## 附錄 A：現有球桿掛載流程（參考 `core/models/table_robot_manager.py`）

script 可照抄此順序，改用原生 API 直接呼叫：

```python
# 1. 建立 UR5 reference prim（Nucleus: Isaac/Robots/UniversalRobots/ur5/ur5.usd）
#    末端 link：{ur5_prim_path}/wrist_3_link
# 2. 建立球桿 reference prim（assets/ball_stick.usd）
# 3. align_prim_to_target(球桿, 末端 link)      # 球桿對齊末端位姿
# 4. filter_collision_pair(球桿, 末端 link)     # 排除兩者碰撞
# 5. create_fixed_joint(joint_path, 球桿, 末端 link)
#    joint_path = {球桿 prim path}/FixedJointToRobot
```

對應的 6.0.0 原生 API 實作可參考 `extension/isaac_sim_impl_6_0/stage_api_impl.py` 中上述方法的實際寫法。

## 附錄 B：相關檔案索引

| 檔案 | 用途 |
|---|---|
| `core/models/table_robot_manager.py` | 球桿掛載流程參考 |
| `core/models/ur5_robot.py` | UR5 載入與末端 link 路徑參考 |
| `extension/isaac_sim_impl_6_0/stage_api_impl.py` | 原生 API 寫法參考 |
| `assets/ball_stick.usd` | 球桿資產（Mass 0.5 kg） |
| `docs/tech-design-4-5-cue-stick-fixed-joint.md` | Fixed Joint 設計文件 |
| `docs/phase3-task-breakdown.md` | 寫回目標（出桿速度範圍） |
