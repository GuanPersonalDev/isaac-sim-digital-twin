# Barrett WAM 7-DOF URDF 資產來源

`wam7.urdf` 由 [jhu-lcsr/barrett_model](https://github.com/jhu-lcsr/barrett_model)（Johns Hopkins LCSR，作者 Jonathan Bohren，**GPL 授權**）的
`robots/wam7.urdf.xacro` 用標準 xacro 工具（無 ROS 環境，`$(find barrett_model)` 手動替換為本地路徑）展開而成，
只保留 7-DOF 手臂本體（不含 BarrettHand），對應 [#193 調查結論](../../docs) 提到的學術撞球機器人研究實際採用的手臂型號。

`meshes/` 只複製了 `wam7.urdf` 實際引用到的 8 個連桿 × 2 個網格（`_fine.stl` 視覺網格、`_convex*.dae` 碰撞網格），
未包含原始 repo 的 BarrettHand、CAD 原始檔（SolidWorks/.blend）等不需要的內容。

**授權注意事項**：原始 repo 為 GPL 授權，若本專案未來需要對外發布/開源，需要另外確認這個資產的授權相容性。
