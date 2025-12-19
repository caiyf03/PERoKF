from typing import List
from dataclasses import dataclass, field

KMFLOW_GRID_SIZE = 256

@dataclass
class KMFlow2D_Config:
    size: int = field(
        default=KMFLOW_GRID_SIZE, metadata={"help": "Grid size."}
    )

    downsample: int = field(
        default=1, metadata={"help": "Output size = size / downsample."}
    )

    Re: List[float] = field(
        default_factory=list, metadata={"help": "Reynolds number."}
    )

    forcing_num: List[int] = field(
        default_factory=list, metadata={"help": "Forcing number."}
    )

    spin_up: float = field(
        default=2.0, metadata={"help": "Spin up time."}
    )

    duration: float = field(
        default=10.0, metadata={"help": "Duration of simulation."}
    )

    fps: int = field(
        default=32, metadata={"help": "Frames per second."}
    )

    batch_size: int = field(
        default=1, metadata={"help": "Batch size."}
    )

    device: str = field(
        default="cuda:0", metadata={"help": "Device to use."}
    )

    root_dir: str = field(
        default="dataset", metadata={"help": "Root directory for dataset."}
    )

    save_vorticity: bool = field(
        default=True, metadata={"help": "Save vorticity."}
    )

    save_velocity: bool = field(
        default=False, metadata={"help": "Save velocity."}
    )

    samples: int = field(
        default=100, metadata={"help": "Number of samples to generate."}
    )

    start_seed: int = field(
        default=0, metadata={"help": "Start seed for random number generator."}
    )

@dataclass
class KMFlow2D_DatasetConfig:
    size: int = field(
        default=KMFLOW_GRID_SIZE,
        metadata={
            "help": "Size of the simulation grid. The data is assumed to be square."
        },
    )

    root_dir: str = field(
        default="dataset", metadata={"help": "Root directory for dataset."}
    )

    stat_dir: str = field(
        default="dataset/stats.npz", metadata={"help": "Directory for statistics."}
    )

    using_vorticity: bool = field(
        default=True,
        metadata={
            "help": "Use vorticity data. If False, use velocity data."
        },
    )

    using_velocity: bool = field(
        default=False,
        metadata={
            "help": "Use velocity data. If False, use vorticity data."
        },
    )

    list_re: List[float] = field(
        default_factory=list,
        metadata={
            "help": "Reynolds numbers for training."
        },
    )

    list_fn: List[int] = field(
        default_factory=list,
        metadata={
            "help": "Number of Fourier modes for training."
        },
    )