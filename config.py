"""Pydantic model for default configuration and validation."""
"""Implementation based on the template of Matformer."""

import subprocess
from typing import Optional, Union
import os
from pydantic import root_validator

from typing import Literal
from mfan.utils import BaseSettings, FEATURESET_SIZE
from mfan.models.mfan import mfanConfig
from mfan.models.comformer import iComformerConfig, eComformerConfig

try:
    VERSION = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )
except Exception:
    VERSION = "NA"

class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION
    task_name: str = ""

    # dataset configuration
    dataset: str = "dft_3d"
    target: str = ""
    force_target: str = ""
    stress_target: str = ""  # cchen
    need_force: bool = False
    need_stress: bool = False  # cchen
    energy_weight: float = 1.0
    force_weight: float = 1.0
    force_criterion: Optional[Literal["mse", "l1", "smoothl1_01", "smoothl1_001"]] = "mse"
    stress_weight: float = 1.0  # cchen ###
    stress_criterion: Optional[Literal["mse", "l1", "smoothl1_01", "smoothl1_001"]] = "l1"  # cchen ###
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn", "emb"] = "cgcnn"
    neighbor_strategy: Literal["k-nearest", "mix", "cutoff", "torch"] = "k-nearest"
    id_tag: Literal["jid", "id", "_oqmd_entry_id", ""] = "jid"

    # logging configuration

    # training configuration
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    target_multiplication_factor: Optional[float] = None
    epochs: int = 300
    batch_size: int = 64
    weight_decay: float = 0
    learning_rate: float = 1e-2
    lr_final_div: float = 10000.0
    filename: str = "sample"
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "smoothl1_01", "smoothl1_001"] = "mse"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "none", "step", "polynomial"] = "onecycle"
    pin_memory: bool = False
    save_dataloader: bool = False
    write_checkpoint: bool = True
    write_predictions: bool = True
    store_outputs: bool = True
    progress: bool = True
    log_tensorboard: bool = False
    standard_scalar_and_pca: bool = False
    use_canonize: bool = True
    num_workers: int = 0
    cutoff: float = 4.0
    max_neighbors: int = 12
    keep_data_order: bool = False
    distributed: bool = False
    n_early_stopping: Optional[int] = None  # typically 50
    output_dir: str = os.path.abspath(".")  # typically 50
    matrix_input: bool = False
    pyg_input: bool = False
    use_lattice: bool = False
    use_angle: bool = False
    norm_mean_std: bool = True
    load_label_from_atom: bool = False

    valid_per_epoch: bool = True

    # model configuration
    model_path: str = './model.pt'
    model: Union[
        mfanConfig,
        iComformerConfig,
        eComformerConfig,
    ] = mfanConfig(name="mfan")

    @root_validator(pre=True)
    def read_legacy_static_config(cls, config):
        """Handle legacy fields in old saved configs."""
        config = config.copy()
        # Testing now always runs after training, regardless of the old flag.
        config.pop("test_per_epoch", None)
        if config.pop("cutoff_2", None) is not None:
            raise ValueError("edge2 is no longer supported")
        return config

    @root_validator(skip_on_failure=True)
    def check_config(cls, config):
        model = config["model"]
        if model.name == "mfan":
            if model.atom_features != config["atom_features"]:
                model.atom_features = config["atom_features"]
                print('Set model.atom_features as config.atom_features')
            if config["atom_features"] == 'emb':
                # emb will be handled in the model
                config["atom_features"] = 'atomic_number'
        elif config["atom_features"] == 'emb':
            print('iComformer and eComformer do not support emb atom features.')
            print('Using cgcnn instead')
            config["atom_features"] = 'cgcnn'

        if hasattr(model, "atom_input_features"):
            model.atom_input_features = FEATURESET_SIZE[config["atom_features"]]
            print('set model.atom_input_features as', model.atom_input_features)

        # cchen: propagate need_stress to model
        if hasattr(model, "need_stress") and model.need_stress != config["need_stress"]:
            print('Using config.need_stress for config.model.need_stress. Computing stress.')
            model.need_stress = config["need_stress"]
        ###
        if model.need_force != config["need_force"]:
            print('Using config.need_force for config.model.need_force. Calculating forces.')
            model.need_force = config["need_force"]
        # cchen: stress also requires torch neighbor strategy and sum_mode
        if config["need_stress"] and not config["need_force"]:
            print('Warning: need_stress requires need_force. Enabling need_force.')
            config["need_force"] = True
            model.need_force = True
        if config["need_stress"]:
            if model.name == 'mfan':
                model.sum_mode = True
                model.use_batch_norm = 2
            if config["neighbor_strategy"] != "torch":
                print('Warning: need_stress requires neighbor_strategy="torch". Setting.')
                config["neighbor_strategy"] = "torch"
        ###
        if config["need_force"]:
            if model.name == 'mfan':
                model.sum_mode = True
                model.use_batch_norm = 2
                print('Predicting energy and force: using sum mode')
                print('Disabled batch norm for keep physical consistency')
            if config["neighbor_strategy"] != "torch":
                print('Warning: need_force is True but neighbor_strategy is not "torch". Setting neighbor_strategy to "torch".')
                config["neighbor_strategy"] = "torch"
            if config["classification_threshold"] is not None:
                raise Exception('Classification task is not supported when need_force is True.')
            if config["model"].output_features > 1:
                raise Exception('Multi-task prediction is not supported when need_force is True.')
        elif (config["load_label_from_atom"] and len(config["target"]) == 0 and
            config["classification_threshold"] is None):
                config["model"].sum_mode = True
                print('Predicting total energy: using sum mode')

        if config["neighbor_strategy"] == 'torch':
            print("Setting model's cutoff and max_neighbor according to config")
            model.cutoff = config["cutoff"]
            model.max_neighbors_threshold = config["max_neighbors"]

        return config
