from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    """Data configuration
    train/test data_dir looks like:
    data_dir/
    └── <cls-id>/<image-name>.jpg"""
    train_data_path: Path = Path("data/CIFAR10_imbalanced")
    val_data_path: Path = Path("data/CIFAR10_balance")
    load_img2mem: bool = True
    augment: bool = True
    """whether to load all images to memory in advance"""


@dataclass
class ModelConfig:
    """Vit Model configuration"""
    num_classes: int = 10
    in_channels: int = 3
    img_size: tuple[int, int] = (32, 32)
    patch_size: int = 4
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 6
    ffn_hidden_channels: int = 512
    dropout: float = 0.1
    classifier: bool = True


@dataclass
class OptimConfig:
    """Optimizer configuration"""
    lr: float = 5e-4
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    """Training configuration"""
    batch_size: int = 512
    num_epochs: int = 500
    # ckpt_dir: Path = Path("ckpt")
    # log_path: Path = Path("logs/log.txt")
    loss_augment: bool = True
    exp_dir: Path = Path("exp")
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def __post_init__(self):
        print(f"Experiment dir: {self.exp_dir}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.exp_dir / "ckpt"
        self.log_path = self.exp_dir / "logs/log.txt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
