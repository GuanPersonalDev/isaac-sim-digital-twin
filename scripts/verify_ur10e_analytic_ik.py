"""scripts/verify_ur10e_analytic_ik.py - 驗證 core/services/ur10e_analytic_ik.py
的正向運動學是否跟 Isaac Sim（RMPflow）的正向運動學對得上同一個座標系。

core/services/ur10e_analytic_ik.py 已經用純數學 round-trip（隨機關節角→FK→
IK→比對是否能還原原始關節角）驗證過演算法本身正確（2000 組全過，誤差在
機器精度等級），但那個測試完全不涉及 Isaac Sim——DH frame 0（base）/
frame 6（末端）不保證直接等於 Isaac Sim 的 base_link/wrist_3_link 座標系
（ur10e.urdf 本身就註記 base_link 到 base_link_inertia 之間有一個繞 Z 軸
180 度的旋轉）。這支腳本檢查：對同一組關節角，`ur10e_analytic_ik.
forward_kinematics()` 算出的位姿，跟 RMPflow 的 `get_end_effector_pose()`
算出的位姿，是否只差一個「固定的」座標轉換（如果是，找出這個轉換；
如果連固定轉換都對不上，代表兩者用的不是同一條運動學鏈，要另外查）。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_analytic_ik.py
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run() -> None:
    import numpy as np

    from core.services import ur10e_analytic_ik as analytic_ik
    from isaac_sim_impl_6_0.ur10e_rmpflow_controller import _load_rmp_flow

    rmp_flow = _load_rmp_flow()

    rng = np.random.default_rng(7)
    # 隨機取樣關節角時避開手腕奇異點附近（sin(theta5) 太接近 0），因為
    # 那正好是這支腳本要找的「固定座標轉換」不該用來取樣的病態點——跟
    # 目標無關的數值不穩定不該混進來干擾判斷。
    n_samples = 30
    rotation_offsets = []
    position_offsets = []
    max_position_only_error = 0.0

    print("[verify] 逐組隨機關節角比對 RMPflow 正向運動學 vs analytic_ik.forward_kinematics() ...")
    for i in range(n_samples):
        joints = rng.uniform(-np.pi / 2, np.pi / 2, size=6)
        if abs(np.sin(joints[4])) < 0.2:
            continue

        rmp_translation, rmp_rotation = rmp_flow.get_end_effector_pose(joints)
        rmp_translation = np.asarray(rmp_translation, dtype=float)
        rmp_rotation = np.asarray(rmp_rotation, dtype=float)

        analytic_position, analytic_rotation = analytic_ik.forward_kinematics(joints)

        # 假設兩者只差一個「固定的」座標轉換 R_offset（左乘）：
        # rmp_rotation ≈ R_offset @ analytic_rotation
        # 用旋轉矩陣還原 R_offset，多組樣本應該收斂到同一個矩陣。
        r_offset = rmp_rotation @ analytic_rotation.T
        rotation_offsets.append(r_offset)

        position_offset = rmp_translation - r_offset @ analytic_position
        position_offsets.append(position_offset)

        if i < 5:
            print(f"[verify]   sample={i} joints={np.round(joints, 4).tolist()}")
            print(f"[verify]     rmp_translation={rmp_translation.tolist()}  analytic_position={analytic_position.tolist()}")

    rotation_offsets = np.array(rotation_offsets)
    position_offsets = np.array(position_offsets)

    mean_r_offset = rotation_offsets.mean(axis=0)
    std_r_offset = rotation_offsets.std(axis=0)
    mean_p_offset = position_offsets.mean(axis=0)
    std_p_offset = position_offsets.std(axis=0)

    print(f"[verify] 有效樣本數={len(rotation_offsets)}")
    print(f"[verify] R_offset 平均值=\n{mean_r_offset}")
    print(f"[verify] R_offset 標準差（各分量，應接近 0 才代表真的是固定轉換）=\n{std_r_offset}")
    print(f"[verify] position_offset 平均值={mean_p_offset.tolist()}")
    print(f"[verify] position_offset 標準差={std_p_offset.tolist()}（應接近 0）")

    r_offset_is_constant = float(np.max(std_r_offset)) < 1e-6
    p_offset_is_constant = float(np.max(std_p_offset)) < 1e-6
    print(f"[verify] 判定：旋轉偏移是固定值={r_offset_is_constant}  平移偏移是固定值={p_offset_is_constant}")

    if r_offset_is_constant and p_offset_is_constant:
        print("[verify] PASS：analytic_ik 的 DH frame 0/frame 6 跟 RMPflow（Isaac Sim base_link/"
              "wrist_3_link）只差一個固定座標轉換，可以用這組 R_offset/position_offset 對接。")
    else:
        print("[verify] FAIL：偏移量不是固定值，代表兩者運動學鏈本身有出入（不只是座標系選擇問題），"
              "需要另外查（例如 DH 參數是不是有一個環節取錯符號/軸）。")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[verify] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
