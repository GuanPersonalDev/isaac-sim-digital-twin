from __future__ import annotations

import torch

from core.ports.policy_port import PolicyPort

class TorchScriptPolicyImpl(PolicyPort):
    """
    載入 policy.pt
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self._policy = torch.jit.load(model_path, map_location=self._device)
        self._policy.eval()

    def infer(self, observation: list[float]) -> list[float]:
        with torch.no_grad():
            model_input = torch.tensor([observation], dtype=torch.float32, device=self._device)
            raw_action = self._policy(model_input)

        return raw_action[0].tolist()