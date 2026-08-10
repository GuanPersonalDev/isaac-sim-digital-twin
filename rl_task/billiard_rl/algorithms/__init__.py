# Copyright (c) 2026 GuanPersonalDev
"""自訂 RL 演算法。

`RslRlPpoAlgorithmCfg.class_name` 吃的是可解析的路徑字串，rsl_rl 的 runner 用
`resolve_callable()` 解析它，所以指到這裡就能換掉演算法——**不需要 fork
`rsl-rl-lib`**。
"""

from .masked_ppo import MaskedPPO

__all__ = ["MaskedPPO"]
