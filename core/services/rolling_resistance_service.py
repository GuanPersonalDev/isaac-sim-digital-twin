import math

from ..ports.rigid_body_api import RigidBodyAPI

GRAVITY = 9.81
NEGLIGIBLE_SPEED_THRESHOLD = 0.02  # m/s，低於此值視覺上等同停止，直接夾到 0
NEGLIGIBLE_SPIN_THRESHOLD = 0.1  # rad/s，殘留自旋（側旋）低於此值視覺上等同停止，直接夾到 0
PHYSICS_DT = 1.0 / 60.0  # 跟 SimulationManager.setup_simulation(dt=1/60) 一致
SPIN_DECAY_RATE = 10.0  # rad/s²，球-呢絨自旋衰減率（Dr. Dave Pool Info，5–15 中間值）
# 公開於模組層級（非 __init__ 預設引數），供 RL 訓練端的 torch 重寫 import 同一個
# 值，避免兩份實作各自改一次造成漂移。見 docs/CHANGELOG.md（#121 B-6）。
ROLLING_FRICTION_COEFF = 0.01
# 沉降/多球接觸解算殘留雜訊的量級上限，跟 NEGLIGIBLE_SPEED_THRESHOLD／
# NEGLIGIBLE_SPIN_THRESHOLD（視覺門檻）是不同層級的判斷。公開理由同上，訓練端
# 的行為在這個門檻內與本類別不同。見 docs/CHANGELOG.md（#121 B-6／#203）。
SETTLING_NOISE_CEILING = 0.005


class RollingResistanceService:
    """
    用明確的物理修正取代 PhysX torsionalPatchRadius（見
    docs/tech-design/rolling-resistance-correction-tech-design.md）：對桌上
    每顆球依真實撞球滾動摩擦係數施加線速度衰減，並依球-呢絨自旋衰減率
    （跟滾動摩擦是各自獨立的物理現象）對殘留自旋（側旋／english）施加衰減。
    """

    def __init__(
        self,
        rigid_body_api: RigidBodyAPI,
        ball_radius: float,
        rolling_friction_coeff: float = ROLLING_FRICTION_COEFF,
        spin_decay_rate: float = SPIN_DECAY_RATE,
    ) -> None:
        self._rigid_body_api = rigid_body_api
        self._ball_radius = ball_radius
        self._rolling_friction_coeff = rolling_friction_coeff
        self._spin_decay_rate = spin_decay_rate

    def apply(self, ball_prim_paths: list[str]) -> None:
        """
        對每個 prim path 獨立執行一次速度衰減。

        角速度採精確分解：由目前線速度反推出「滾動分量」（n̂ × v / R，n̂ 為
        桌面法向量），隨線速度一起依滾動摩擦衰減；其餘分量（殘留自旋，含
        加塞／side-spin）依球-呢絨自旋衰減率獨立衰減——兩者是各自獨立的
        物理現象。
        """
        delta_v = self._rolling_friction_coeff * GRAVITY * PHYSICS_DT
        delta_w = self._spin_decay_rate * PHYSICS_DT

        # 整批一次讀完再逐顆算，不逐顆呼叫（見 core/ports/rigid_body_api.py）
        linear_velocities, angular_velocities = self._rigid_body_api.get_velocities(
            ball_prim_paths
        )

        for prim_path, (vx, vy, vz), (wx, wy, wz) in zip(
            ball_prim_paths, linear_velocities, angular_velocities
        ):
            v_h = math.sqrt(vx**2 + vy**2)

            # n̂ × v（n̂=(0,0,1)）＝ (-vy, vx, 0)：由目前線速度反推出的滾動角速度分量
            roll_wx_before = -vy / self._ball_radius
            roll_wy_before = vx / self._ball_radius
            residual_x = wx - roll_wx_before
            residual_y = wy - roll_wy_before
            residual_z = wz  # n̂ × v 的 z 分量恆為 0，wz 全部都是殘留（側旋）分量
            residual_magnitude = math.sqrt(residual_x**2 + residual_y**2 + residual_z**2)

            # 低於視覺門檻，或這個 tick 該扣的量超過目前速度時，直接夾到 0（不會反向）
            horizontally_at_rest = v_h < NEGLIGIBLE_SPEED_THRESHOLD or delta_v >= v_h

            # 邏輯跟上面的滾動摩擦對稱
            spin_at_rest = residual_magnitude < NEGLIGIBLE_SPIN_THRESHOLD or delta_w >= residual_magnitude

            if (
                v_h < SETTLING_NOISE_CEILING
                and residual_magnitude < SETTLING_NOISE_CEILING
            ):
                # ⚠️ 純數值雜訊也不能寫入：持續呼叫 set_velocities() 會讓
                # PhysX 沒機會讓球進入 sleep。跳過寫入，交還給 PhysX 自己處理。
                # 見 docs/CHANGELOG.md。
                continue

            if horizontally_at_rest:
                linear_scale = 0.0
            else:
                linear_scale = (v_h - delta_v) / v_h
            new_vx = vx * linear_scale
            new_vy = vy * linear_scale
            roll_wx_after = roll_wx_before * linear_scale
            roll_wy_after = roll_wy_before * linear_scale

            if spin_at_rest:
                spin_scale = 0.0
            else:
                spin_scale = (residual_magnitude - delta_w) / residual_magnitude
            new_residual_x = residual_x * spin_scale
            new_residual_y = residual_y * spin_scale
            new_residual_z = residual_z * spin_scale

            new_wx = roll_wx_after + new_residual_x
            new_wy = roll_wy_after + new_residual_y
            new_wz = new_residual_z

            self._rigid_body_api.set_velocities(
                prim_path,
                [new_vx, new_vy, vz],
                [new_wx, new_wy, new_wz],
            )
