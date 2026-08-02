from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Protocol

import mlx.core as mx


@dataclass
class LoRATarget:
    model_path: str
    possible_up_patterns: List[str]
    possible_down_patterns: List[str]
    possible_alpha_patterns: List[str] = field(default_factory=list)
    possible_lokr_w1_patterns: List[str] = field(default_factory=list)
    possible_lokr_w2_patterns: List[str] = field(default_factory=list)
    possible_dora_scale_patterns: List[str] = field(default_factory=list)
    up_transform: Callable[[mx.array], mx.array] | None = None
    down_transform: Callable[[mx.array], mx.array] | None = None
    lokr_w1_transform: Callable[[mx.array], mx.array] | None = None
    lokr_w2_transform: Callable[[mx.array], mx.array] | None = None


@dataclass
class DiffTarget:
    """A full-weight delta applied directly to a model parameter.

    ComfyUI's `LoraLoader` supports adapters that ship fully fine-tuned tensors as
    `<path>.diff` deltas against the base weights, alongside the usual low-rank pairs.
    These target parameters that have no low-rank decomposition (RMSNorm scales,
    modulation vectors), so they are added in place rather than wrapped in a layer.
    """

    param_path: str
    possible_patterns: List[str]


class LoRAMapping(Protocol):
    @staticmethod
    def get_mapping() -> List[LoRATarget]:
        return

    @staticmethod
    def get_diff_mapping() -> List[DiffTarget]:
        return []
