# Copyright (c) 2026 GuanPersonalDev
"""#123：在真實 Isaac Lab 物理環境量測 SPREAD_REF。

這不是 policy 評估。腳本對每個 episode 送出可重現的控制式開球：母球位於既定
break position、最大速度、零擊球偏移，並只對 X 與 0° 方向加入小幅對稱 jitter。
如此量到的是實際 PhysX／袋口行為下「成功碰到 1 號球的開球」spread 分布，而
不是初始 policy 大量打空的 0.0118 rack baseline。

用法（RunPod、repo root）::

    /workspace/IsaacLab/isaaclab.sh -p \
      rl_task/scripts/verify_spread_ref.py --headless

預設收集 500 筆 first_contact == 1 的落定結果，寫入：
`/workspace/issue123-validation/spread-ref-result.json`。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必須是正整數")
    return parsed


parser = argparse.ArgumentParser(description="#123 SPREAD_REF 真實物理量測")
parser.add_argument("--num_envs", type=_positive_int, default=256)
parser.add_argument("--target_samples", type=_positive_int, default=500)
parser.add_argument("--max_episodes", type=_positive_int, default=5000)
parser.add_argument("--max_steps", type=_positive_int, default=10000)
parser.add_argument("--angle_jitter_deg", type=float, default=1.5)
parser.add_argument("--placement_jitter_m", type=float, default=0.03)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument(
    "--output",
    default="/workspace/issue123-validation/spread-ref-result.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下才能匯入 Isaac Lab、torch 與本專案模組。"""

import json  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import billiard_rl.tasks.manager_based.billiard_rl.mdp.rewards as reward_module  # noqa: E402
from billiard_rl.tasks.manager_based.billiard_rl.billiard_rl_env_cfg import (  # noqa: E402
    BilliardRlEnvCfg,
)
from core.models.action_bounds import ACTION_BOUNDS, ACTION_DIM  # noqa: E402
from core.services.break_shot_position_provider import (  # noqa: E402
    BREAK_SHOT_POSITIONS,
)
from core.services.pocket_geometry import POCKET_POSITIONS  # noqa: E402
from core.services.spread_score_calculator import (  # noqa: E402
    SPREAD_REF,
    calculate_spread_score,
)

_OBJECT_BALL_IDS = tuple(range(1, 10))
_POCKET_XY = list(POCKET_POSITIONS.values())
_REFERENCE_TOLERANCE = 0.20
_ANGLE_INDEX = 2
_SPEED_INDEX = 3


@dataclass(frozen=True)
class ShotAudit:
    """reward 結算當下的一筆原始 spread 與開球事件資料。"""

    raw_spread: float
    first_contact: int
    foul_penalty: float
    cue_scratch: float
    pocketed_count: int
    rail_contacted_count: int

    @property
    def is_primary_sample(self) -> bool:
        return self.first_contact == 1

    @property
    def is_legal_break(self) -> bool:
        return (
            self.first_contact == 1
            and self.foul_penalty == 0.0
            and self.cue_scratch == 0.0
        )


def _normalize(value: float, index: int) -> float:
    low, high = ACTION_BOUNDS[index]
    return (value - (high + low) / 2.0) / ((high - low) / 2.0)


def _controlled_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """產生最大速度、零偏移、對準 1 號球附近的控制式開球。"""
    actions = torch.zeros(
        (env.num_envs, ACTION_DIM), device=env.device, dtype=torch.float32
    )

    x_jitter = (
        torch.rand(env.num_envs, device=env.device) * 2.0 - 1.0
    ) * args_cli.placement_jitter_m
    x_low, x_high = ACTION_BOUNDS[0]
    actions[:, 0] = x_jitter / ((x_high - x_low) / 2.0)
    actions[:, 1] = _normalize(BREAK_SHOT_POSITIONS[0][1], 1)

    # 0° 是正規化域的 -1 邊界；360°（同方向）是 +1。兩側各取一半，避免
    # 只測角度邊界的單側誤差。jitter magnitude 均勻分布於 [0, limit]。
    magnitude = torch.rand(env.num_envs, device=env.device) * args_cli.angle_jitter_deg
    near_zero = -1.0 + magnitude / 180.0
    near_360 = 1.0 - magnitude / 180.0
    choose_zero_side = torch.rand(env.num_envs, device=env.device) < 0.5
    actions[:, _ANGLE_INDEX] = torch.where(choose_zero_side, near_zero, near_360)

    actions[:, _SPEED_INDEX] = 1.0
    return actions


def _raw_spread(
    ball_xy: list[tuple[float, float]],
    pocket_index: list[int],
) -> tuple[float, set[int]]:
    """依正式 reward 的袋口替代規則計算 raw spread。"""
    pocketed = {
        ball_id for ball_id in _OBJECT_BALL_IDS if pocket_index[ball_id] >= 0
    }
    positions = {
        ball_id: (
            _POCKET_XY[pocket_index[ball_id]]
            if pocket_index[ball_id] >= 0
            else ball_xy[ball_id]
        )
        for ball_id in _OBJECT_BALL_IDS
    }
    return calculate_spread_score(positions, pocketed), pocketed


def _install_reward_audit(records: list[ShotAudit]) -> Callable:
    """在目前 process 包裝 evaluate_shot；關閉程式後不留下任何正式行為。"""
    original = reward_module.evaluate_shot

    def audited_evaluate_shot(
        ball_xy: list[tuple[float, float]],
        pocket_index: list[int],
        rail_contacted: list[bool],
        first_contact: int,
    ) -> dict[str, float]:
        raw_spread, pocketed = _raw_spread(ball_xy, pocket_index)
        components = original(
            ball_xy,
            pocket_index,
            rail_contacted,
            first_contact,
        )
        records.append(
            ShotAudit(
                raw_spread=raw_spread,
                first_contact=first_contact,
                foul_penalty=components["foul"],
                cue_scratch=components["cue_scratch"],
                pocketed_count=len(pocketed),
                rail_contacted_count=sum(
                    bool(rail_contacted[ball_id]) for ball_id in _OBJECT_BALL_IDS
                ),
            )
        )
        return components

    reward_module.evaluate_shot = audited_evaluate_shot
    return original


def _summary(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _git_commit() -> str:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_result(
    records: list[ShotAudit],
    *,
    episodes: int,
    timeouts: int,
    steps: int,
) -> tuple[dict[str, object], int]:
    primary = [record for record in records if record.is_primary_sample]
    legal = [record for record in records if record.is_legal_break]

    result: dict[str, object] = {
        "issue": 123,
        "commit": _git_commit(),
        "parameters": {
            "num_envs": args_cli.num_envs,
            "target_samples": args_cli.target_samples,
            "max_episodes": args_cli.max_episodes,
            "angle_jitter_deg": args_cli.angle_jitter_deg,
            "placement_jitter_m": args_cli.placement_jitter_m,
            "cue_ball_y_m": BREAK_SHOT_POSITIONS[0][1],
            "normalized_speed": 1.0,
            "normalized_offsets": [0.0, 0.0],
            "seed": args_cli.seed,
        },
        "counts": {
            "env_steps": steps,
            "episodes": episodes,
            "settled": len(records),
            "timeouts": timeouts,
            "first_contact_one": len(primary),
            "legal_breaks": len(legal),
        },
        "current_spread_ref": SPREAD_REF,
        "tolerance_fraction": _REFERENCE_TOLERANCE,
    }

    if len(primary) < args_cli.target_samples:
        result.update(
            {
                "status": "INSUFFICIENT_SAMPLES",
                "message": "達到執行上限前未收集到足夠的 first_contact == 1 樣本",
            }
        )
        exit_code = 1
    else:
        primary_summary = _summary([record.raw_spread for record in primary])
        legal_summary = (
            _summary([record.raw_spread for record in legal]) if legal else None
        )
        measured_mean = float(primary_summary["mean"])
        relative_deviation = (measured_mean - SPREAD_REF) / SPREAD_REF
        within_tolerance = abs(relative_deviation) <= _REFERENCE_TOLERANCE
        result.update(
            {
                "status": "PASS" if within_tolerance else "REMEASURE",
                "primary_sample_definition": "settled and first_contact == 1",
                "primary_statistics": primary_summary,
                "legal_break_statistics": legal_summary,
                "relative_deviation": relative_deviation,
                "accepted_range": [
                    SPREAD_REF * (1.0 - _REFERENCE_TOLERANCE),
                    SPREAD_REF * (1.0 + _REFERENCE_TOLERANCE),
                ],
                "recommendation": (
                    f"保留 SPREAD_REF={SPREAD_REF}"
                    if within_tolerance
                    else "換一個 seed 重測；兩輪都超出 ±20% 才更新 SPREAD_REF"
                ),
            }
        )
        exit_code = 0 if within_tolerance else 2

    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result, exit_code


def _validate_arguments() -> None:
    if not 0.0 <= args_cli.angle_jitter_deg <= 30.0:
        raise SystemExit("--angle_jitter_deg 必須在 [0, 30] 度")
    if not 0.0 <= args_cli.placement_jitter_m <= 0.10:
        raise SystemExit("--placement_jitter_m 必須在 [0, 0.10] 公尺")


def main() -> int:
    _validate_arguments()
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    cfg = BilliardRlEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg)

    records: list[ShotAudit] = []
    original_evaluate_shot = _install_reward_audit(records)
    episodes = 0
    timeouts = 0
    steps = 0
    next_progress = 100

    print(
        f"[#123 spread] envs={env.num_envs} target={args_cli.target_samples} "
        f"max_episodes={args_cli.max_episodes} seed={args_cli.seed}"
    )

    try:
        env.reset()
        while (
            simulation_app.is_running()
            and steps < args_cli.max_steps
            and episodes < args_cli.max_episodes
        ):
            actions = _controlled_actions(env)
            _, _, terminated, truncated, _ = env.step(actions)
            steps += 1
            episodes += int((terminated | truncated).sum().item())
            timeouts += int(truncated.sum().item())

            primary_count = sum(record.is_primary_sample for record in records)
            if primary_count >= next_progress:
                print(
                    f"[#123 spread] progress={primary_count}/{args_cli.target_samples} "
                    f"episodes={episodes} timeouts={timeouts}"
                )
                next_progress += 100
            if primary_count >= args_cli.target_samples:
                break
    finally:
        reward_module.evaluate_shot = original_evaluate_shot
        env.close()

    result, exit_code = _write_result(
        records,
        episodes=episodes,
        timeouts=timeouts,
        steps=steps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[#123 spread] result={args_cli.output}")
    return exit_code


if __name__ == "__main__":
    try:
        status = main()
    finally:
        simulation_app.close()
    raise SystemExit(status)
