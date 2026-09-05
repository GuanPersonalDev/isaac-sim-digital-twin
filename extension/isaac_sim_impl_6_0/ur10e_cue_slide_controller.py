import numpy as np


class Ur10eCueSlideController:
    """UR10e 專用推桿控制：驅動 CueSlideJoint（TableRobotManager 在球桿跟
    wrist_3_link 之間掛的線性滑軌關節，見該類別）完成後擺+揮桿，手臂其餘
    6 個關節維持 AIM 收斂後的姿態不動（見 UR10e 重新設計計畫決策 5）。

    跟 Ur10eRmpflowController 是平行、互斥的兩套控制器：前者驅動手臂到位
    （AIM／RESET），這個驅動推桿（STRIKE），STRIKE 階段完全不呼叫
    RMPflow——決策 5：「推桿那一下（滑軌關節本身）不經過 RMPflow」。

    做法跟 UR3e 既有 ArticulationAPIImpl.move_swing_elbow_pivot() 同一個
    精神（quintic velocity profile ＋ gravity compensation），只是作用在
    1 個線性 DOF（CueSlideJoint）而不是 1 個轉動 DOF（elbow）：
    1. move_to_backswing()：把 CueSlideJoint 從目前位置（AIM 收斂後＝0，
       球桿原點跟 wrist_3_link 重合、桿尖正好在接觸點——見
       TableRobotManager.__init__() 的 align_prim_to_target() 慣例）用
       joint-space 位置控制退到 backswing_position（負值，後擺／收回
       位置，沿球桿軸反方向）。
    2. start_strike()：從 backswing_position 解一段 1-DOF quintic
       （q(0)=backswing_position, q̇(0)=0, q(T)=0, q̇(T)=target_velocity,
       q̈(0)=q̈(T)=0），逐 tick 下達 q̇(t)（其餘 6 個手臂 DOF 速度固定 0），
       疊加重力補償。

    target_velocity 直接等於
    swing_trajectory_calculator.compute_required_tip_speed(cue_ball_speed)
    ——決策 4：滑軌關節軸向＝球桿軸向＝桿尖速度方向，推桿瞬間手臂完全
    靜止，桿尖線速度 100% 等於滑軌關節本身的線速度，不像 UR3e 的
    elbow-pivot 設計需要用槓桿臂（CUE_STICK_GRIP_TO_TIP）換算轉動關節
    角速度，這裡完全不需要雅可比矩陣／槓桿臂換算。
    """

    _POSITION_TOLERANCE_M = 0.002
    _MAX_BACKSWING_STEPS = 180

    def __init__(self, articulation, slide_dof_name: str = "CueSlideJoint") -> None:
        self._articulation = articulation

        dof_names = list(self._articulation.dof_names)
        self._slide_dof_index = dof_names.index(slide_dof_name)
        self._num_dofs = len(dof_names)

        max_velocities = self._articulation.get_dof_max_velocities()
        if hasattr(max_velocities, "numpy"):
            max_velocities = max_velocities.numpy()
        max_velocities = np.asarray(max_velocities, dtype=float)
        if max_velocities.ndim == 2:
            max_velocities = max_velocities[0]
        self._slide_max_velocity = float(max_velocities[self._slide_dof_index])

        self._phase: str | None = None  # None / "backswing" / "strike"
        self._backswing_steps = 0
        self._quintic: tuple[float, float, float, float] | None = None
        self._elapsed_strike_steps = 0
        self._motion_active = False
        self._did_last_motion_timeout = False

    def is_motion_complete(self) -> bool:
        return not self._motion_active

    def did_last_motion_timeout(self) -> bool:
        return self._did_last_motion_timeout

    def retract(self, backswing_position: float) -> None:
        """只把 CueSlideJoint 退到 backswing_position，退到後**不**接著自動
        觸發 STRIKE 揮桿子階段（跟 move_stroke() 的差別）——給 AIM 期間
        「桿尖先退開，避免手臂定位/收尾修正過程蹭到球」用（見
        ArticulationAPIImpl.move_to_pose() 的 UR10e 分流：AIM 移動手臂前
        先呼叫這個方法，等退到位才開始移動手臂，2026-09-04 補充）。跟
        move_stroke() 共用同一套 joint-space 位置控制到 backswing_position
        的底層邏輯，只是完成後停在這裡，不像 move_stroke() 會接著解
        quintic 並切到 velocity 模式往前揮。
        """
        self._backswing_position = float(backswing_position)
        self._phase = "retract_only"
        self._backswing_steps = 0
        self._motion_active = True
        self._did_last_motion_timeout = False

        positions = np.asarray(self._articulation.get_dof_positions())[0].copy()
        positions[self._slide_dof_index] = self._backswing_position
        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(positions[None, :])

    def move_stroke(self, backswing_position: float, target_velocity: float) -> None:
        """開始一次完整的後擺＋揮桿。backswing_position 是負值（沿球桿軸
        退開的距離），target_velocity 是滑軌關節在 q=0（接觸點）當下要
        達到的線速度（= 桿尖速度）。呼叫端只需呼叫一次，之後每個 physics
        tick 呼叫 step()，用 is_motion_complete() 判斷整段動作（後擺+
        揮桿）是否完成。
        """
        self._backswing_position = float(backswing_position)
        self._target_velocity = float(target_velocity)
        self._phase = "backswing"
        self._backswing_steps = 0
        self._motion_active = True
        self._did_last_motion_timeout = False

        positions = np.asarray(self._articulation.get_dof_positions())[0].copy()
        positions[self._slide_dof_index] = self._backswing_position
        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(positions[None, :])

    def step(self, physics_dt: float) -> None:
        if not self._motion_active:
            return

        if self._phase == "backswing":
            self._step_backswing(physics_dt)
        elif self._phase == "strike":
            self._step_strike(physics_dt)
        elif self._phase == "retract_only":
            self._step_retract_only()

    def _step_retract_only(self) -> None:
        positions = np.asarray(self._articulation.get_dof_positions())[0]
        current = float(positions[self._slide_dof_index])
        self._backswing_steps += 1

        converged = abs(current - self._backswing_position) <= self._POSITION_TOLERANCE_M
        timed_out = self._backswing_steps >= self._MAX_BACKSWING_STEPS
        if not (converged or timed_out):
            return

        if timed_out and not converged:
            self._did_last_motion_timeout = True
        self._motion_active = False

    def _step_backswing(self, physics_dt: float) -> None:
        positions = np.asarray(self._articulation.get_dof_positions())[0]
        current = float(positions[self._slide_dof_index])
        self._backswing_steps += 1

        converged = abs(current - self._backswing_position) <= self._POSITION_TOLERANCE_M
        timed_out = self._backswing_steps >= self._MAX_BACKSWING_STEPS
        if not (converged or timed_out):
            return

        if timed_out and not converged:
            self._did_last_motion_timeout = True

        q0 = current
        q1 = 0.0
        v1 = self._target_velocity
        T = max(abs(q1 - q0) / max(abs(v1), 1e-6), 0.05)
        peak_velocity = abs(v1)
        for _attempt in range(50):
            c3, c4, c5 = _solve_quintic_coeffs(q0, q1, v1, T)
            peak_velocity = _peak_abs_quintic_velocity(c3, c4, c5, T)
            if peak_velocity <= self._slide_max_velocity + 1e-9:
                break
            T *= (peak_velocity / self._slide_max_velocity) * 1.05
        self._quintic = (c3, c4, c5, T)
        self._elapsed_strike_steps = 0
        self._phase = "strike"

    def _step_strike(self, physics_dt: float) -> None:
        c3, c4, c5, T = self._quintic
        t = min(self._elapsed_strike_steps * physics_dt, T)
        qdot = np.zeros(self._num_dofs)
        qdot[self._slide_dof_index] = _quintic_velocity(c3, c4, c5, t)

        # ⚠️ 2026-09-04 除錯記錄：switch_dof_control_mode() 不帶
        # dof_indices 會套用到全部 7 個 DOF——"velocity" 模式把 stiffness
        # 歸零（見該方法官方 docstring 的模式對照表），若不限定只作用在
        # CueSlideJoint，等於連同其餘 6 個手臂關節的 stiffness 也一起歸零，
        # 讓 AIM 收斂好的姿態在整段揮桿期間完全沒有位置回復力可以抵抗
        # CueSlideJoint 加速的反作用力，只剩 damping／重力補償撐著，手臂
        # 因此在推桿當下漂移（實測：STRIKE 全程桿尖 Z 座標多爬升
        # 3.24cm，跟純沿桿軸滑動的幾何預期不符，是造成桿尖沒打中球心的
        # 真正原因）。只限定 CueSlideJoint 這個 DOF 切換成 velocity 模式，
        # 其餘手臂關節維持 AIM 收尾時的 position 模式（含 wrist_1/wrist_3
        # 的增益 boost）不被打斷。
        self._articulation.switch_dof_control_mode("velocity", dof_indices=[self._slide_dof_index])
        self._articulation.set_dof_velocity_targets(qdot[None, :])
        gravity_compensation_forces = self._articulation.get_dof_gravity_compensation_forces()
        self._articulation.set_dof_efforts(gravity_compensation_forces)

        self._elapsed_strike_steps += 1
        if self._elapsed_strike_steps * physics_dt >= T:
            self._motion_active = False


def _solve_quintic_coeffs(q0: float, q1: float, v1: float, T: float) -> tuple[float, float, float]:
    """單一 DOF joint-space quintic：q(0)=q0,q̇(0)=0,q̈(0)=0,q(T)=q1,
    q̇(T)=v1,q̈(T)=0。回傳 (c3,c4,c5)（c0=q0,c1=0,c2=0 已知）。跟
    ArticulationAPIImpl._solve_quintic_coeffs() 同一個公式（線性 DOF跟
    轉動 DOF 用同一組多項式邊界條件，差別只在物理量的單位）。
    """
    A = np.array([
        [T ** 3, T ** 4, T ** 5],
        [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        [6 * T, 12 * T ** 2, 20 * T ** 3],
    ])
    b = np.array([q1 - q0, v1, 0.0])
    c3, c4, c5 = np.linalg.solve(A, b)
    return float(c3), float(c4), float(c5)


def _quintic_velocity(c3: float, c4: float, c5: float, t: float) -> float:
    return 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4


def _peak_abs_quintic_velocity(c3: float, c4: float, c5: float, T: float, samples: int = 200) -> float:
    ts = np.linspace(0.0, T, samples)
    return max(abs(_quintic_velocity(c3, c4, c5, t)) for t in ts)
