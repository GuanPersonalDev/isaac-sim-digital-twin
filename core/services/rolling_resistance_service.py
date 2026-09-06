import math

from ..ports.rigid_body_api import RigidBodyAPI

GRAVITY = 9.81
NEGLIGIBLE_SPEED_THRESHOLD = 0.02  # m/s，低於此值視覺上等同停止，直接夾到 0（非跳過不處理）
NEGLIGIBLE_SPIN_THRESHOLD = 0.1  # rad/s，殘留自旋（側旋）低於此值視覺上等同停止，直接夾到 0
PHYSICS_DT = 1.0 / 60.0  # 跟 SimulationManager.setup_simulation(dt=1/60) 一致的固定常數，不作為 apply() 參數傳入
SPIN_DECAY_RATE = 10.0  # rad/s²，取 Dr. Dave Pool Info 記載的球-呢絨自旋衰減率 5–15 rad/s² 中間值
# 球-呢絨滾動摩擦係數。原本只是 __init__ 的預設引數，提到模組層級是為了讓 RL
# 訓練端的 torch 重寫 import 得到（#121 B-6）——留在簽章裡的話訓練端只能重打
# 一次 0.01，那正是「兩份實作靜默漂移」的入口。Demo 端建構本類別時沒有傳這個
# 參數（billiard_digital_twin.py:112），吃的就是這個值。
ROLLING_FRICTION_COEFF = 0.01
# 沉降/多球接觸解算殘留雜訊的量級上限（實測線速度雜訊約 1e-7~1e-5 m/s、角速度
# 殘留雜訊約 1e-4~1e-3 rad/s，兩者都遠低於這裡設的門檻）。跟 NEGLIGIBLE_SPEED_
# THRESHOLD／NEGLIGIBLE_SPIN_THRESHOLD 是不同用途：那兩個是「已經停止，直接夾
# 到 0」的視覺門檻，這個是用來分辨「這是真的、需要主動夾停的殘留速度（例如
# #203 回報的門檻附近低速蠕動）」還是「單純的接觸解算數值雜訊（該交給 PhysX
# 自己的 sleep 機制處理，不能再被 set_velocities() 寫入打斷）」。
#
# 公開（非底線開頭）是因為 RL 訓練環境要用同一個門檻。訓練端不能重用本類別——
# 1024 env × 10 球 = 每個 physics tick 一萬次 Python 呼叫，量級上不可能——所以
# 改用 torch 向量化重寫，但物理常數必須全部 import 自本模組（#121 B-6）。
#
# ⚠️ 訓練端在這個門檻內的行為與本類別**刻意不同**：本類別完全跳過寫入，把收斂
#    交還給 PhysX sleep；訓練端的張量 API 是整塊寫入，沒辦法逐球跳過，因此改為
#    主動把三軸速度寫成精確的 0（含 vz，本類別是原封不動傳遞）。訓練端不需要
#    sleep，只需要 BallMotionMonitor.SPEED_THRESHOLD 讀得到 0。詳見 #121 B-6。
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
        加塞／side-spin）依球-呢絨自旋衰減率獨立衰減——這是跟滾動摩擦不同
        的物理現象，兩者各自衰減到 0 為止，不會有任何一方被無限期留在
        門檻附近不動。
        """
        delta_v = self._rolling_friction_coeff * GRAVITY * PHYSICS_DT
        delta_w = self._spin_decay_rate * PHYSICS_DT

        # 整批一次讀完再逐顆算：逐顆版本每顆球要兩次同步（get_linear_velocity
        # 與 get_angular_velocity 底下各自呼叫一次 RigidPrim.get_velocities()），
        # 10 顆就是 20 次。見 core/ports/rigid_body_api.py 的效能說明。
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

            # 滾動摩擦：線速度與對應的滾動角速度分量一起衰減，低於視覺門檻或這個
            # tick 該扣的量超過目前速度時，直接夾到 0（不會反向，也不會被永遠留在門檻附近）
            horizontally_at_rest = v_h < NEGLIGIBLE_SPEED_THRESHOLD or delta_v >= v_h

            # 自旋衰減：殘留角速度整體依比例衰減（方向不變），邏輯跟滾動摩擦對稱
            spin_at_rest = residual_magnitude < NEGLIGIBLE_SPIN_THRESHOLD or delta_w >= residual_magnitude

            if (
                v_h < SETTLING_NOISE_CEILING
                and residual_magnitude < SETTLING_NOISE_CEILING
            ):
                # 水平跟自旋都只是沉降/多球接觸解算的數值雜訊量級，根本不是真的
                # 在滾動或自旋。即使只是要把這種雜訊「夾到 0」，只要每個 tick
                # 持續呼叫 set_velocities()，這個顯式寫入本身就會讓 PhysX 沒
                # 機會把球放進 sleep 狀態——接觸解算會持續重新產生類似大小的
                # 殘留，球永遠卡在這個值上不消失（見 docs/CHANGELOG.md 的 GUI
                # 實測回報）。正確做法是完全跳過寫入，把雜訊交還給 PhysX 自己
                # 的 sleep 機制處理。注意：這跟下面 horizontally_at_rest／
                # spin_at_rest 的「視覺門檻」是不同層級的判斷——門檻附近真正的
                # 低速蠕動（例如 #203 回報的 bug）量級明顯大於這裡的雜訊上限，
                # 仍然會落到下面的分支被主動夾停，不會被這裡誤判跳過。
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
