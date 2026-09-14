# Isaac Sim Digital Twin — 9-Ball Billiard Robot Simulation

English | [繁體中文](README.md)

## Overview

This is a personal learning project that builds a 9-ball billiard robot digital
twin on **NVIDIA Isaac Sim 6.0**. It exists to practice Isaac Sim / Omniverse
Kit API development and to train a robot arm's shot strategy with
reinforcement learning (RL).

What's in here:
- A robot arm executing shots on a virtual billiard table (the architecture
  supports UR3e / UR5 / UR10e; current development is centered on UR10e)
- An RL training environment built on
  [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (`rl_task/`), trained on
  cloud GPUs (RunPod)
- Local playback in the Isaac Sim GUI to verify ball physics and control logic

Since this is a practice project, the code and docs are still evolving. The
architecture spec (`docs/architecture-spec.md`) can drift slightly out of sync
with the actual directory layout — treat the code as the source of truth.

## Tech Stack

- **Isaac Sim**: 6.0.0.1 (Warp-based Core Experimental API)
- **Python**: 3.12
- **Isaac Lab**: `isaaclab` / `isaaclab_assets` / `isaaclab_mimic` /
  `isaaclab_rl` / `isaaclab_tasks` (used by the RL environment in `rl_task/`)
- **Omniverse Kit**: `omni.usd`, `omni.timeline`, `omni.ui`,
  `omni.kit.menu.utils`, `omni.kit.viewport.utility`, etc. (used by the Kit
  extension in `extension/`)

## Architecture

The project separates business logic, platform abstraction, and platform
implementation into three layers, so an Isaac Sim version upgrade only
requires rewriting the implementation layer without touching business logic:

| Directory | Responsibility |
|---|---|
| `core/ports/` | Platform-agnostic abstract interfaces (`stage_api`, `articulation_api`, `physics_api`, `rigid_body_api`, `material_api`, `policy_port`) |
| `core/controllers/` | Control strategies (state-machine controller, manual controller, model controller, etc.) |
| `core/models/` | Data models (balls, table, robot arm, actions, observations, etc.) |
| `core/services/` | Business logic (reward computation, swing strategy, IK, rolling resistance, observation encoder, etc.) |
| `core/tests/` | pytest unit tests for the core layer — pure Python, no Isaac Sim required |
| `extension/isaac_sim_impl_6_0/` | Concrete Isaac Sim 6.0 implementation of the `core/ports/` interfaces |
| `extension/ui/` | omni.ui components (HUD panel, debug menu, etc.) |
| `extension/billiard_digital_twin/` | Kit extension entry point (`extension.toml`, main extension class) |
| `rl_task/` | Standalone Isaac Lab RL training package (`billiard_rl`), with its own `pyproject.toml` |
| `training/` | Cloud training ops scripts (RunPod bootstrap, watchdog, etc.) — see [`training/README.md`](training/README.md) |
| `docs/` | Architecture spec, CHANGELOG, design and debugging notes |
| `assets/` | USD/USDA scene assets (billiard table, balls, robot arm, etc.) |
| `scripts/` | One-off diagnostic/verification scripts from development |
| `models/`, `outputs/` | Trained models and training outputs |

## Directory Structure

```
isaac-sim-digital-twin/
├── core/                       # Platform-agnostic business logic layer
│   ├── ports/                  # Abstract interfaces
│   ├── controllers/            # Control strategies
│   ├── models/                 # Data models
│   ├── services/                # Business logic
│   └── tests/                  # Unit tests
├── extension/                  # Isaac Sim platform bridge layer
│   ├── isaac_sim_impl_6_0/     # 6.0 API implementation
│   ├── ui/                     # omni.ui components
│   └── billiard_digital_twin/  # Kit extension entry point
├── rl_task/                    # Isaac Lab RL training package
├── training/                   # Cloud training ops scripts (RunPod)
├── docs/                       # Architecture spec and design docs
├── assets/                     # Scene assets (USD/USDA)
├── scripts/                    # Development diagnostic scripts
├── models/                     # Trained model outputs
├── LICENSE
└── README.md
```

## Setup

### Local environment (Isaac Sim GUI playback / verification)

```bash
pip install isaacsim[all,extscache]==6.0.0.1 --extra-index-url https://pypi.nvidia.com
```

Once installed, load the `extension/billiard_digital_twin/` extension in
Isaac Sim to play back the billiard scene in the GUI.

### RL training environment (`rl_task/`)

`rl_task/` is a standalone Isaac Lab package that depends on the `isaaclab`
ecosystem. See `rl_task/pyproject.toml` and `rl_task/setup.py` for
installation details.

### Cloud training (RunPod)

Cloud GPU training setup and workflow is documented separately in
[`training/README.md`](training/README.md).

## Testing

Business logic in `core/` is unit-tested with pytest; external dependencies
(omni/pxr/isaacsim) are isolated with `unittest.mock`, so tests run without
launching Isaac Sim:

```bash
pytest
```

(`pytest.ini` sets `testpaths = core/tests`)

## License

This project is licensed under the [Apache License 2.0](LICENSE).
