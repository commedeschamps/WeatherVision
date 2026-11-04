from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Experiment:
    model: str
    epochs: int = 6
    batch_size: int = 16
    lr: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 4
    augment: bool = True
    pretrained: bool = True
    freeze_backbone: bool = True
    seed: int = 42
    image_size: int = 224
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self):
        if self.model not in {"cnn", "mobilenet", "convnext"}:
            raise ValueError(f"Unknown architecture: {self.model}")
        for name in ("epochs", "batch_size", "patience", "seed", "image_size", "num_workers"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if min(self.epochs, self.batch_size, self.patience) < 1 or self.lr <= 0:
            raise ValueError("Epochs, batch size, patience, and learning rate must be positive")
        if self.image_size != 224:
            raise ValueError("This protocol fixes image_size=224 for pretrained preprocessing")
        if self.num_workers < 0 or self.seed < 0 or self.weight_decay < 0:
            raise ValueError("Workers, seed, and weight decay cannot be negative")
        for name in ("augment", "pretrained", "freeze_backbone"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false, not a string")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Invalid device")
        if self.model == "cnn" and (self.pretrained or self.freeze_backbone):
            raise ValueError("The CNN baseline must train from scratch without freezing")

    def to_dict(self):
        return asdict(self)


def load_config(path: Path) -> dict[str, Experiment]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("experiments"), dict):
        raise ValueError("Config requires an experiments mapping")
    allowed = {f.name for f in fields(Experiment)}
    common = {k: v for k, v in raw.items() if k != "experiments"}
    if set(common) - allowed:
        raise ValueError(f"Unknown shared settings: {set(common) - allowed}")
    result = {}
    for run, values in raw["experiments"].items():
        if not isinstance(run, str) or not run.replace("_", "").isalnum():
            raise ValueError("Run names may only contain letters, numbers, and underscores")
        result[run] = Experiment(**(common | values))
    if not result:
        raise ValueError("At least one experiment is required")
    return result


def validate_ablation(a: Experiment, b: Experiment) -> None:
    differences = {k for k in a.to_dict() if a.to_dict()[k] != b.to_dict()[k]}
    if a.model != "convnext" or b.model != "convnext" or differences != {"augment"}:
        raise ValueError("ConvNeXt ablation configurations must differ only in augmentation")
