# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "ball_positions",
    "BilliardStrikeAction",
    "BilliardStrikeActionCfg",
    "decay_velocities",
    "all_balls_at_rest",
    "balls_at_rest_mask",
    "break_foul_decided",
    "break_foul_decided_mask",
    "break_shot_positions",
    "reset_break_shot_layout",
    "decompose_reward",
    "evaluate_shot",
    "spread",
    "nine_ball",
    "cue_scratch",
    "foul",
    "aim",
    "detect_pocketed",
    "detect_rail_contact",
    "detect_cue_contact",
    "update_first_contact",
    "update_closest_approach",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import BilliardStrikeAction
from .actions_cfg import BilliardStrikeActionCfg
from .events import break_shot_positions, reset_break_shot_layout
from .observations import ball_positions
from .physics import decay_velocities
from .rewards import (
    aim,
    cue_scratch,
    decompose_reward,
    evaluate_shot,
    foul,
    nine_ball,
    spread,
)
from .shot_tracking import (
    detect_cue_contact,
    detect_pocketed,
    detect_rail_contact,
    update_closest_approach,
    update_first_contact,
)
from .terminations import (
    all_balls_at_rest,
    balls_at_rest_mask,
    break_foul_decided,
    break_foul_decided_mask,
)
