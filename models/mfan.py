"""Implementation based on the template of Matformer."""

from typing import Optional, List
import torch
import torch.nn.functional as F
from typing import Literal
from torch import nn
from mfan.models.utils import RBFExpansion, envelope_func
from mfan.utils import BaseSettings, FEATURESET_SIZE
from torch_scatter import scatter
from mfan.models.transformer import ComformerConv
from mfan.models.transformer import ComformerConv_multiframe
from mfan.models.graph_builder import process_positions_and_edges
from pydantic import model_validator
from mfan.models.get_frames import (
    get_qr_frames,
    get_pca_frames,
    get_cell_frames,
    get_nei_frames,
    get_null_frames,
)


eps = 1e-4

frame_dict = {
    "qr": get_qr_frames,
    "pca": get_pca_frames,
    "lattice": get_cell_frames,
    "nei": get_nei_frames,
    "null": get_null_frames
}

class mfanConfig(BaseSettings):
    """Hyperparameter schema for MFAN."""

    name: Literal["mfan"]
    classification: bool = False
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn", "emb"] = "cgcnn"
    conv_layers: int = 5
    edge_layers: int = 2
    edge_features: int = 256
    angle_features: int = 256
    node_features: int = 256
    edge_hidden_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    output_features: int = 1
    node_layer_head: int = 1
    edge_layer_head: int = 1
    link: Literal["identity", "log", "logit"] = "identity"
    use_batch_norm: int = 1

    frame_angle_type: int = 0
    frames: List[Literal["lattice", "qr", "pca", "nei", "null"]] = ["lattice", "qr"]
    frame_temperature: float = 100.0
    frame_quality_eps: float = 0.01
    frame_quality_scale: float = 0.05
    reverse_frames: bool = False
    final_angle_norm: bool = False
    edge_dropout: float = 0.0
    distance_decay: float = -1.0

    cutoff: float = 6.0
    max_neighbors_threshold: int = 50
    use_envelope: bool = True
    envelope_scale: float = 2.0

    need_force: bool = False
    need_stress: bool = False  # cchen: enable strain-derivative stress prediction ###
    sum_mode: bool = False
    quick_mode_setup: Optional[str] = None  # 'crystal' or 's2ef'

    @model_validator(mode="before")
    @classmethod
    def read_legacy_static_config(cls, values):
        """Load old static-model configs without restoring removed features."""
        if not isinstance(values, dict):
            return values
        values = values.copy()
        if values.get("name") == "lsrp":
            values["name"] = "mfan"
        if "qucik_mode_setup" in values:
            values.setdefault("quick_mode_setup", values.pop("qucik_mode_setup"))
        if values.get("edge2_layers") or values.get("cutoff_2") is not None:
            raise ValueError("edge2 is no longer supported")
        for key in (
            "edge2_layers", "cutoff_2", "edge2_topk", "edge2_selector_hidden",
            "dirouter_channels", "dirouter_temperature", "dirouter_dropout",
            "dynamic_frame_lr_scale", "dynamic_frame_reg_weight",
            "dynamic_frame_norm_reg_weight", "dynamic_frame_norm_min",
            "n_dynamic_frames",
        ):
            values.pop(key, None)
        return values

    @model_validator(mode="after")
    def check_config(self):
        if self.conv_layers <= self.edge_layers:
            raise ValueError("conv_layers must be greater than edge_layers")

        if self.quick_mode_setup is not None:
            mode = self.quick_mode_setup
            if mode == 'crystal':
                self.frame_angle_type = 1
                self.frame_quality_eps = -1
                self.final_angle_norm = False
                self.use_envelope = False
            elif mode == 's2ef':
                self.frame_angle_type = 0
                self.frame_quality_eps = 0.01
                self.use_envelope = True
        return self

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"

def bond_cosine_with_clip(r1, r2,eps=1e-6):
    n1 = torch.norm(r1, dim=-1)
    n2 = torch.norm(r2, dim=-1)
    valid = (n1 > eps) & (n2 > eps)

    bond_cosine = torch.zeros_like(n1)
    dot = torch.sum(r1 * r2, dim=-1)
    norm = n1 * n2
    bond_cosine[valid] = (dot[valid] / norm[valid]).clamp(-1, 1)
    return bond_cosine

# Keep the historical helper name for downstream imports.
bond_cosine_withclip = bond_cosine_with_clip


class mfan(nn.Module):
    """Multi-frame atomic representations implemented with PyTorch Geometric."""

    def cal_forces(self, energy, positions, prefix: str = 'train'):

        graph = (prefix == "train")
        grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(energy)])
        forces = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=graph,
            allow_unused=True
        )[0]
        assert forces is not None, "Gradient should not be None"
        return -forces

    # cchen: stress via autograd of total energy w.r.t. symmetric displacement tensor ###
    def cal_stress(self, energy, displacement, cell, prefix: str = 'train'):
        """Compute stress tensor via autograd of total energy w.r.t. displacement.

        Args:
            energy: per-atom energy [batch, 1] or [batch]
            displacement: symmetric displacement tensor [batch, 3, 3] with requires_grad=True
            cell: strained cell [batch, 3, 3] or flat [batch*3, 3]
            prefix: 'train' keeps graph for backprop; 'eval' detaches

        Returns:
            stress: [batch, 3, 3] in eV/Å^3 (same sign convention as ASE)
        """
        graph = (prefix == "train")
        cell_3d = cell.view(-1, 3, 3)
        volume = torch.abs(
            torch.sum(
                cell_3d[:, 0] * torch.cross(cell_3d[:, 1], cell_3d[:, 2], dim=-1),
                dim=-1
            )
        ).clamp(min=1e-8)
        grad_outputs = torch.ones_like(energy)
        stress = torch.autograd.grad(
            outputs=[energy],
            inputs=[displacement],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=True,
            allow_unused=True
        )[0]
        assert stress is not None, "Stress gradient should not be None"
        stress = stress / volume.unsqueeze(-1).unsqueeze(-1)
        return stress
    ###

    # cchen: joint force + stress via a single autograd.grad call; two sequential
    # grad calls freed the shared energy graph in eval mode (cal_forces used
    # retain_graph=False when prefix='eval'), causing
    # "Trying to backward through the graph a second time"
    def cal_forces_stress(self, energy, positions, displacement, cell, prefix: str = 'train'):
        """Compute forces and stress in one backward pass over the shared energy graph.

        Args:
            energy: total energy [batch] or [batch, 1]
            positions: atomic positions [n_atoms, 3] with requires_grad=True
            displacement: symmetric displacement tensor [batch, 3, 3] with requires_grad=True
            cell: strained cell [batch, 3, 3] or flat [batch*3, 3]
            prefix: 'train' keeps graph for backprop; 'eval' frees it after this call

        Returns:
            forces: [n_atoms, 3]
            stress: [batch, 3, 3] in eV/Å^3 (same sign convention as ASE)
        """
        graph = (prefix == "train")
        cell_3d = cell.view(-1, 3, 3)
        volume = torch.abs(
            torch.sum(
                cell_3d[:, 0] * torch.cross(cell_3d[:, 1], cell_3d[:, 2], dim=-1),
                dim=-1
            )
        ).clamp(min=1e-8)
        grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(energy)])
        grads = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions, displacement],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=graph,
            allow_unused=True
        )
        forces, stress = grads[0], grads[1]
        assert forces is not None, "Gradient should not be None"
        assert stress is not None, "Stress gradient should not be None"
        stress = stress / volume.unsqueeze(-1).unsqueeze(-1)
        return -forces, stress
    ###

    def calculate_frames(self, data):
        edge_indices = [data.edge_index]
        edge_attrs = [data.edge_attr]

        # building frames
        frames = []
        qualities = []
        for func in self.frame_funcs:
            f, q = func(data, eps=eps, normalize=False)
            frames.append(f)
            qualities.append(q)

        return edge_indices, edge_attrs, frames, qualities

    # Keep existing callers compatible with the former method name.
    calculated_frames = calculate_frames

    def build_edge_map_features(self, edge_index, edge_attr, frames, eps, only_angle=False,
                                use_envelope=True, envelope_scale=1.0, cutoff=6.0):
        if not only_angle:
            edge_len = torch.norm(edge_attr, dim=1)
            edge_feat = -0.75 / edge_len
            edge_radius = self.rbf(edge_feat)
            if use_envelope:
                ld_weight = envelope_func(edge_len, cutoff, envelope_scale, 6.0)
                edge_radius = edge_radius * ld_weight.view(-1, 1)

        edge_angles = []
        for frame in frames:
            if frame is not None:
                edge_frame = torch.stack(
                    [
                        frame[0][edge_index[self.frame_edge_idx]],
                        frame[1][edge_index[self.frame_edge_idx]],
                        frame[2][edge_index[self.frame_edge_idx]],
                    ],
                    dim=1,
                )
                frame_angle = bond_cosine_with_clip(
                    edge_frame,
                    edge_attr.unsqueeze(1).repeat(1, 3, 1),
                    eps=eps,
                )
                edge_angles.append(
                    self.rbf_angle(frame_angle.reshape(-1)).reshape(
                        edge_attr.shape[0], 3, self.node_features_len
                    )
                )
            else:
                edge_angles.append(None)
        if only_angle:
            return edge_angles
        else:
            return edge_len, edge_radius, edge_radius.clone(), edge_angles

    def build_edge_quality_features(self, edge_index, qualities):
        edge_quality = [quality[edge_index[self.frame_edge_idx]] for quality in qualities]
        return torch.stack(edge_quality, dim=1)

    def __init__(self, config: mfanConfig = mfanConfig(name="mfan")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        if config.atom_features == 'emb':
            self.onehot_embedding = FEATURESET_SIZE['emb']
        else:
            self.onehot_embedding = -1
        self.node_features_len = config.node_features
        self.n_edge_layers = config.edge_layers
        self.cutoff = config.cutoff
        self.use_envelope = config.use_envelope
        self.envelope_scale = config.envelope_scale
        self.need_force = config.need_force
        self.need_stress = config.need_stress  # cchen ###
        self.sum_mode = config.sum_mode
        if config.reverse_frames:
            self.frame_edge_idx = 0
        else:
            self.frame_edge_idx = 1

        assert len(config.frames) > 0 and len(config.frames) <= 3, \
            "Only support 1 to 3 frames, but got {}".format(len(config.frames))
        self.frame_funcs = [frame_dict[f] for f in config.frames]
        self.atom_embedding = nn.Linear(
            FEATURESET_SIZE[config.atom_features], config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
        )

        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=config.angle_features,
            ),
            nn.Linear(config.angle_features, config.node_features),
            nn.Softplus(),
        )

        # Preserve the nested module names for existing static-model checkpoints.
        self.att_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ComformerConv(in_channels=config.node_features,
                                      out_channels=config.node_features,
                                      heads=config.node_layer_head,
                                      edge_dim=config.node_features,
                                      length_decay=config.distance_decay,
                                      use_batch_norm=config.use_batch_norm)
                        for _ in range(config.conv_layers)
                    ]
                )
            ]
        )

        self.edge_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ComformerConv_multiframe(in_channels=config.node_features,
                                                 out_channels=config.node_features,
                                                 n_frames=len(config.frames),
                                                 temperature=config.frame_temperature,
                                                 hidden_channels=config.node_features,
                                                 dropout=config.edge_dropout,
                                                 frame_angle_type=config.frame_angle_type,
                                                 frame_quality_eps=config.frame_quality_eps,
                                                 frame_quality_scale=config.frame_quality_scale,
                                                 final_angle_norm=config.final_angle_norm,
                                                 use_batch_norm=config.use_batch_norm)
                        for _ in range(config.edge_layers)
                    ]
                )
            ]
        )

        if config.fc_layers == 1:
            self.fc = nn.Sequential(
                nn.Linear(config.node_features, config.fc_features), nn.SiLU()
            )
        else:
            fc_nn = []
            fc_nn.append(nn.Linear(config.node_features, config.fc_features))
            fc_nn.append(nn.SiLU())
            for _ in range(config.fc_layers - 1):
                fc_nn.append(nn.Linear(config.fc_features, config.fc_features))
                fc_nn.append(nn.SiLU())
            self.fc = nn.Sequential(*fc_nn)
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, config.output_features
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    def forward(self, data, prefix: str='train') -> torch.Tensor:
        data, _, _ = data
        if self.onehot_embedding > 0:
            element_feat = F.one_hot(data.x.squeeze(-1).long(),
                                     num_classes=self.onehot_embedding).float()
        else:
            element_feat = data.x
        node_features = self.atom_embedding(element_feat)
        edge_indices, edge_attrs, frames, qualities = self.calculate_frames(data)
        edge_len_1, edge_radius_1, edge_feat_1, edge_angle_1 = self.build_edge_map_features(
            edge_indices[0], edge_attrs[0], frames, eps,
            use_envelope=self.use_envelope, envelope_scale=self.envelope_scale, cutoff=self.cutoff
        )
        for i in range(0, len(self.att_layers[0])):
            node_features = self.att_layers[0][i](node_features,
                                                    edge_indices[0],
                                                    edge_feat_1,
                                                    edge_length=edge_len_1)
            if i < self.n_edge_layers:
                edge_feat_1 = self.edge_layers[0][i](
                    node_features[edge_indices[0][0]],
                    node_features[edge_indices[0][1]],
                    edge_feat_1,
                    edge_radius_1,
                    torch.stack(edge_angle_1, dim=1),
                    self.build_edge_quality_features(edge_indices[0], qualities),
                )
                self.base_edge_feat = edge_feat_1

        if self.sum_mode:
            # first get the result of each node, and then sum up
            # actually we still use mean to avoid the influence of scale
            node_features = self.fc(node_features)
            node_out = self.fc_out(node_features)
            out = scatter(node_out, data.batch, dim=0, reduce="mean")
        else:
            features = scatter(node_features, data.batch, dim=0, reduce="mean")
            features = self.fc(features)
            out = self.fc_out(features)
        if self.link:
            out = self.link(out)

        if self.need_force:
            force_in = out * data.natoms.unsqueeze(1)

            # cchen: joint force + stress path, single backward pass so the
            # graph is never consumed before the stress gradient in eval mode
            if self.need_stress:
                forces, stress = self.cal_forces_stress(
                    force_in, data.pos, data.displacement, data.cell, prefix=prefix
                )
                if not torch.isfinite(forces).all():
                    raise Exception('Force tensor ERR.')
                return (out.squeeze(-1), forces, stress)
            ###

            forces = self.cal_forces(force_in, data.pos, prefix=prefix)
            if not torch.isfinite(forces).all():
                raise Exception('Force tensor ERR.')

            return (out.squeeze(-1), forces)
        else:
            return out.squeeze(-1)

class mfan_tg_wrapper(nn.Module): # mfan nnp with torch graph builder
    """att pyg implementation."""

    def __init__(self, config: mfanConfig = mfanConfig(name="mfan")):
        """Set up att modules."""
        super().__init__()
        self.model = mfan(config)
        self.cutoff = config.cutoff
        self.need_force = config.need_force
        self.need_stress = config.need_stress  # cchen ###
        self.max_neighbors_threshold = config.max_neighbors_threshold

    def forward(self, data, prefix: str='train') -> torch.Tensor:
        data, _, _ = data

        data.pos.requires_grad_(self.need_force)
        processed_data = process_positions_and_edges(
            graph_data=data,
            pos=data.pos,
            z=data.z,
            natoms=data.natoms,
            batch=data.batch,
            cell=data.cell,
            compute_forces=True,
            compute_stress=self.need_stress,  # cchen ###
            use_pbc=True,
            cutoff=self.cutoff,
            max_neighbors_threshold=self.max_neighbors_threshold,
            dtype=torch.float32
        )

        output = self.model((processed_data, processed_data, processed_data), prefix)

        # cchen: stress is computed inside mfan.forward when need_stress=True ###
        return output
        ###

class mfan_eval_wrapper(nn.Module):
    """used for inference"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.need_force = model.need_force
        self.need_stress = getattr(model, 'need_stress', False)  # cchen

    def forward(self, x):
        with torch.set_grad_enabled(self.need_force or self.need_stress):  # cchen
            out = self.model(x, prefix="eval")
            return out
        ###
