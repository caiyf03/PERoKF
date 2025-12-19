from dataclasses import dataclass, field
from typing import Optional

import simple_parsing
from simple_parsing import ArgumentParser

from kmflow.km_dataset import KMFlow2D_DatasetConfig
from models.unet import UNetConfig

# Global Default Configs
HIDDEN_DIM = 128

@dataclass
class EncoderConfig:
	input_dim: int = field(default=1)
	encoder_latent: int = field(default=HIDDEN_DIM)


@dataclass
class DecoderConfig:
	hidden_dim: int = field(default=HIDDEN_DIM)
	output_dim: int = field(default=1)

    
@dataclass
class FNOConfig:
	hidden_dim: int = field(default=HIDDEN_DIM)
	modes_x: int = field(default=64)
	modes_y: int = field(default=64)
	kernel_size: int = field(default=3)
	depth: int = field(default=4)

@dataclass
class CNNConfig:
	input_dim: int = field(default=128)
	hidden_dim: int = field(default=128)
	output_dim: int = field(default=128)
	depth: int = field(default=4)

@dataclass
class PhysicsConfig:
	enable_feature: bool = field(default=False)
	enable_vorti_loss: bool = field(default=True)
	enable_deriv_loss: bool = field(default=True)

	vorti_loss_weight: float = field(default=1.0)
	deriv_loss_weight: float = field(default=0.003)

@dataclass
class TrainingConfig:
	batch_size: int = field(default=8)
	lr: float = field(default=1e-3)
	epochs: int = field(default=16)
	device: str = field(default="cuda")
	log_every: int = field(default=100)
	val_every: int = field(default=500)
	save_every: int = field(default=500)
	log_dir: str = field(default="logs")
	save_dir: str = field(default="checkpoints")
	fp16: bool = field(default=True)


@dataclass
class ModelConfig:
	type: str = field(default="FNO") # FNO or CNN
	fno: FNOConfig = field(default_factory=FNOConfig)
	cnn: CNNConfig = field(default_factory=CNNConfig)
	unet:UNetConfig = field(default_factory=UNetConfig)


@dataclass
class Config:
	model: ModelConfig = field(default_factory=ModelConfig)
	encoder: EncoderConfig = field(default_factory=EncoderConfig)
	decoder: DecoderConfig = field(default_factory=DecoderConfig)
	physics: PhysicsConfig = field(default_factory=PhysicsConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	dataset: KMFlow2D_DatasetConfig = field(default_factory=KMFlow2D_DatasetConfig)

@dataclass
class DiffusionModelConfig:
	type: str = field(default="UNET")
	unet: UNetConfig = field(default_factory=UNetConfig)
	timesteps: int = field(default=1000)
	beta_start: float = field(default=1e-4)
	beta_end: float = field(default=0.02)
	
@dataclass
class DiffusionConfig:
	model: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
	encoder: EncoderConfig = field(default_factory=EncoderConfig)
	decoder: DecoderConfig = field(default_factory=DecoderConfig)
	physics: PhysicsConfig = field(default_factory=PhysicsConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	dataset: KMFlow2D_DatasetConfig = field(default_factory=KMFlow2D_DatasetConfig)

def parse_diffusion_config_from_yaml(path: str) -> DiffusionConfig:
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(DiffusionConfig, dest="diffusion_config")
    args = parser.parse_args(["--config", path])
    return args.diffusion_config

def parse_config_from_yaml(path: str) -> Config:
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args(["--config", path])
    return args.config

if __name__ == "__main__":
	config = parse_config_from_yaml("configs/sr_config.yaml")
	print(config)