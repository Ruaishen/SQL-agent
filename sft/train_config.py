from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SftTrainConfig:
    student_model: Path
    dataset_dir: Path
    output_dir: Path
    seed: int = 42
    epochs: int = 2
    effective_batch_size: int = 64
    micro_batch_size: int = 16
    max_sequence_tokens: int = 16384
    learning_rate: float = 5e-6
    warmup_steps: int = 5
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    student_master_dtype: str = "float32"
    save_every_epoch: bool = False

    @classmethod
    def load(cls, path: Path) -> SftTrainConfig:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("SFT train config must be a YAML mapping")
        for key in ("student_model", "dataset_dir", "output_dir"):
            raw[key] = Path(raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        for path in (self.student_model, self.dataset_dir):
            if not path.exists():
                raise FileNotFoundError(path)
        if self.epochs < 1 or self.effective_batch_size < 1 or self.micro_batch_size < 1:
            raise ValueError("epochs and batch sizes must be positive")
        if self.micro_batch_size > self.effective_batch_size:
            raise ValueError("micro_batch_size cannot exceed effective_batch_size")
        if self.max_sequence_tokens < 2:
            raise ValueError("max_sequence_tokens must be at least two")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer parameters must be positive")
        if self.warmup_steps < 0 or self.weight_decay < 0:
            raise ValueError("warmup_steps and weight_decay must be non-negative")
        if self.student_master_dtype != "float32":
            raise ValueError("full-parameter SFT requires FP32 Student master weights")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in value.items():
            if isinstance(item, Path):
                value[key] = str(item)
        return value
