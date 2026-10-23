import math
import random

import torch
torch.serialization.add_safe_globals([slice])
from e3nn import o3
from typing import Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from mfan.models.utils import envelope_cutoff, envelope_func
from torch_scatter import scatter


class ComformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        length_decay: float = -1,
        use_batch_norm: int = 1,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(ComformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.length_decay = length_decay
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)

        if self.length_decay > 0:
            self.ldecay_nn = nn.Sequential(nn.Linear(out_channels, out_channels),
                                        nn.Softplus())

        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        if use_batch_norm == 1:
            self.bn = nn.BatchNorm1d(out_channels)
            self.bn_att = nn.BatchNorm1d(out_channels)
        elif use_batch_norm == 0:
            self.bn = nn.LayerNorm(out_channels)
            self.bn_att = nn.Identity()
        else:
            self.bn = nn.Identity()
            self.bn_att = nn.LayerNorm(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_length: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, edge_length=edge_length, size=None)

        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_concate(out)

        return self.softplus(x[1] + self.bn(out))

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, edge_length: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        alpha = self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels)
        if self.length_decay > 0 and edge_length is not None:
            ld_weight = self.ldecay_nn(query_i)
            alpha = alpha - self.length_decay * ld_weight * edge_length.view(-1, 1, 1)
        out = self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(alpha)
        return out

class ComformerConv_edge(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        use_batch_norm: int = 1,
    ):
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        # Preserve the historical checkpoint key.
        self.frame_embeding = nn.Parameter(torch.randn(3, in_channels[0]))

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e3 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e3 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        if use_batch_norm == 1:
            self.bn = nn.BatchNorm1d(out_channels)
            self.bn_att = nn.BatchNorm1d(out_channels)
        elif use_batch_norm == 0:
            self.bn = nn.LayerNorm(out_channels)
            self.bn_att = nn.Identity()
        else:
            self.bn = nn.Identity()
            self.bn_att = nn.LayerNorm(out_channels)

        self.sigmoid = nn.Sigmoid()

    @property
    def frame_embedding(self):
        """Correctly spelled access without registering a second parameter."""
        return self.frame_embeding

    def forward(self, edge: Union[Tensor, PairTensor], edge_nei_len: OptTensor = None, edge_nei_angle: OptTensor = None):
        # preprocess for edge of shape [num_edges, hidden_dim]

        H, C = self.heads, self.out_channels
        if isinstance(edge, Tensor):
            edge: PairTensor = (edge, edge)
        query_x = self.lin_query(edge[1]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_x = self.lin_key(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        value_x = self.lin_value(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        num_edge = query_x.shape[0]

        if edge_nei_len == None:
            edge_nei_len = self.frame_embedding.expand(num_edge, -1, -1)  # (N_edge, 3, frame_emb_dim)

        key_y = torch.cat((self.lin_key_e1(edge_nei_len[:,0,:]).view(-1, 1, H, C),
                            self.lin_key_e2(edge_nei_len[:,1,:]).view(-1, 1, H, C),
                            self.lin_key_e3(edge_nei_len[:,2,:]).view(-1, 1, H, C)), dim=1)
        value_y = torch.cat((self.lin_value_e1(edge_nei_len[:,0,:]).view(-1, 1, H, C),
                            self.lin_value_e2(edge_nei_len[:,1,:]).view(-1, 1, H, C),
                            self.lin_value_e3(edge_nei_len[:,2,:]).view(-1, 1, H, C)), dim=1)

        # preprocess for interaction of shape [num_edges, 3, hidden_dim]
        edge_xy = self.lin_edge(edge_nei_angle).view(-1, 3, H, C)

        key = self.key_update(torch.cat((key_x, key_y, edge_xy), dim=-1))
        alpha = (query_x * key) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_x, value_y, edge_xy), dim=-1))
        out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, 3, self.heads, self.out_channels))

        out = out.view(-1, 3, self.heads * self.out_channels)
        out = self.lin_concate(out)
        # aggregate the msg
        out = out.sum(dim=1)

        return self.softplus(edge[1] + self.bn(out))

class TensorProductConvLayer(torch.nn.Module):
    # From Torsional diffusion.
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.Softplus(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        return out

class ComformerConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: Optional[int] = None,
        use_second_order_repr: bool = True,
        ns: int = 64,
        nv: int = 8,
        residual: bool = True,
        use_batch_norm: int = 1,
    ):
        super().__init__()

        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o + {nv}x2e',
            f'{ns}x0e'
        ]
        self.ns, self.nv = ns, nv
        self.node_linear = nn.Linear(in_channels, ns)
        self.skip_linear = nn.Linear(in_channels, out_channels)
        self.sh = '1x0e + 1x1o + 1x2e'
        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )
        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[2],
            n_edge_features=edge_dim,
            residual=False
        )
        self.softplus = nn.Softplus()
        if use_batch_norm == 1:
            self.bn = nn.BatchNorm1d(ns)
        elif use_batch_norm == 0:
            self.bn = nn.LayerNorm(ns)
        else:
            self.bn = nn.Identity()
        self.node_linear_2 = nn.Linear(ns, out_channels)

    def forward(self, data, node_feature: Union[Tensor, PairTensor], edge_index: Adj, edge_feature: Union[Tensor, PairTensor],
                    edge_nei_len: OptTensor = None):
        edge_vec = data.edge_attr
        edge_irr = o3.spherical_harmonics(self.sh, edge_vec, normalize=True, normalization='component')
        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.softplus(self.node_linear_2(self.softplus(self.bn(node_feature))))
        node_feature += self.skip_linear(skip_connect)

        return node_feature


class ComformerConv_multiframe(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_frames: int,
        temperature: float = 1.0,
        hidden_channels: int = 0,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        frame_angle_type: int = 0,
        frame_emb_dim: int = 64,
        frame_quality_eps: float = -1.0,
        frame_quality_scale: float = 1.0,
        final_angle_norm: bool = False,
        use_batch_norm: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_frames = n_frames
        self.temperature = temperature
        self.hidden_channels = hidden_channels if hidden_channels > 0 else in_channels
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.frame_angle_type = frame_angle_type
        self.frame_emb_dim = frame_emb_dim
        self.frame_quality_eps = frame_quality_eps
        self.frame_quality_scale = frame_quality_scale

        self.nn1 = nn.Sequential(
            nn.Linear(3*in_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        if frame_angle_type == 0:
            # type-2 style angle processing without LayerNorm, for higher fitting capacity
            self.angle_fcs1 = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels),
            )
            self.angle_gamma = nn.Parameter(torch.zeros(n_frames, 3, in_channels))
            self.angle_beta = nn.Parameter(torch.zeros(n_frames, 3, in_channels))
            nn.init.uniform_(self.angle_gamma, -0.01, 0.01)
            nn.init.uniform_(self.angle_beta, -0.01, 0.01)
            self.angle_fcs2 = nn.ModuleList(
                [nn.Sequential(
                        nn.Linear(in_channels, in_channels),
                        nn.SiLU(),
                        nn.Linear(in_channels, in_channels),
                    ) for _ in range(n_frames)]
            )
        elif frame_angle_type == 1:
            # Pre-alignment, shared fully connected layers, and FiLM
            self.frame_align = nn.Parameter(torch.zeros(n_frames, 3, 3))
            with torch.no_grad():
                self.frame_align.copy_(torch.eye(3).unsqueeze(0).repeat(n_frames, 1, 1))
            self.angle_fcs1 = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels),  # logits for experts
            )
            self.angle_gamma = nn.Parameter(torch.zeros(n_frames, 3, in_channels))
            self.angle_beta = nn.Parameter(torch.zeros(n_frames, 3, in_channels))
            nn.init.uniform_(self.angle_gamma, -0.01, 0.01)
            nn.init.uniform_(self.angle_beta, -0.01, 0.01)
            self.angle_fcs2 = nn.Linear(3*in_channels, in_channels)
        elif frame_angle_type == 2:
            self.angle_fcs1 = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels),  # logits for experts
            )
            self.angle_gamma = nn.Parameter(torch.zeros(n_frames, 3, in_channels))
            self.angle_beta = nn.Parameter(torch.zeros(n_frames, 3, in_channels))
            nn.init.uniform_(self.angle_gamma, -0.01, 0.01)
            nn.init.uniform_(self.angle_beta, -0.01, 0.01)
            self.angle_fcs2 = nn.ModuleList(
                [nn.Sequential(
                        nn.Linear(in_channels, in_channels),
                        nn.LayerNorm(in_channels),
                        nn.SiLU(),
                        nn.Linear(in_channels, in_channels),
                    ) for _ in range(n_frames)]
            )
            self.angle_frame_norm = nn.LayerNorm(in_channels)
        else:
            raise ValueError('frame_angle_type not recognized!')
        self.angle_router = nn.Linear(hidden_channels, n_frames)

        nn2 = []
        pre_d = hidden_channels + 2*in_channels
        for i in range(self.hidden_layers):
            nn2.append(nn.Linear(pre_d, hidden_channels))
            nn2.append(nn.SiLU())
            nn2.append(nn.Dropout(dropout))
            pre_d = hidden_channels
        self.nn2 = nn.Sequential(*nn2)

        self.out_nn = nn.Linear(hidden_channels, out_channels)
        self.softplus = nn.Softplus()

        if final_angle_norm:
            self.angle_out_norm = nn.LayerNorm(in_channels)
        else:
            self.angle_out_norm = nn.Identity()

        if use_batch_norm == 1:
            self.bn = nn.BatchNorm1d(out_channels)
        elif use_batch_norm == 0:
            self.bn = nn.LayerNorm(out_channels)
        else:
            self.bn = nn.Identity()


    def forward(self, node_s: torch.Tensor, node_t: torch.Tensor, edge: torch.Tensor, edge_radius: torch.Tensor, edge_angle: torch.Tensor, frame_quality: torch.Tensor):
        """
        node_s: (N_edge, in_channels)
        node_t: (N_edge, in_channels)
        edge: (N_edge, in_channels)
        edge_radius: (N_edge, in_channels)
        edge_angle: (N_edge, N_frames, 3, in_channels)
        """

        edge_in = edge
        edge = torch.cat([node_s, node_t, edge], dim=-1)
        edge = self.nn1(edge)
        angle_logits = self.angle_router(edge)  # (N_edge, n_frames)
        angle_gate = F.softmax(angle_logits / self.temperature, dim=-1)  # (N_edge, n_frames)
        angle_gate = angle_gate.unsqueeze(-1)  # (N_edge, n_frames, 1)
        n_edge = edge.shape[0]

        if self.frame_angle_type == 1:
            # angle input size (N_edge, n_frames, 3, in_channels)
            align_mat = self.frame_align.unsqueeze(0) # (1, N_frames, 3, 3)
            edge_angle = torch.einsum("efbc,efab->efac", edge_angle, align_mat.expand(n_edge, -1, -1, -1))
            edge_angle_fcs =  self.angle_fcs1(edge_angle.reshape(n_edge * self.n_frames * 3, self.in_channels))
            gamma = self.angle_gamma.unsqueeze(0)  # (1, N_frames, 3, in_channels)
            beta = self.angle_beta.unsqueeze(0)
            edge_angle_fcs = edge_angle_fcs.reshape(n_edge, self.n_frames, 3, self.in_channels) * (1.0 + gamma) + beta
            edge_angle_fcs = self.angle_fcs2(edge_angle_fcs.reshape(n_edge, self.n_frames, 3 * self.in_channels))
        else: # for frame_angle_type 0 and 2, the angle processing is the same except that type-2 has more normalization
            edge_angle_fcs =  self.angle_fcs1(edge_angle.reshape(n_edge * self.n_frames * 3, self.in_channels))
            gamma = self.angle_gamma.unsqueeze(0)  # (1, N_frames, 3, in_channels)
            beta = self.angle_beta.unsqueeze(0)
            edge_angle_fcs = edge_angle_fcs.reshape(n_edge, self.n_frames, 3, self.in_channels) * (1.0 + gamma) + beta
            edge_angle_fcs = torch.sum(edge_angle_fcs, dim=-2) #(E, N_frames, in_channels)
            edge_angle_fcs = [self.angle_fcs2[i](edge_angle_fcs[:,i,:]) for i in range(self.n_frames)] #.contiguous()
            edge_angle_fcs = torch.stack(edge_angle_fcs, dim=1)# (N_edge, N_frames, in_channels)
            if self.frame_angle_type == 2:
                edge_angle_fcs = self.angle_frame_norm(
                    edge_angle_fcs.reshape(n_edge * self.n_frames, self.in_channels)
                ).reshape(n_edge, self.n_frames, self.in_channels)
            angle_gate = torch.sigmoid(angle_logits / self.temperature).unsqueeze(-1)

        if self.frame_quality_eps >= 0:
            frame_quality = frame_quality.unsqueeze(-1).detach()  # (N_edge, n_frames, 1) # disable gradient
            if self.frame_quality_scale >= 1.:
                inverse_quality = (1.0 - frame_quality*self.frame_quality_scale + self.frame_quality_eps)
                frame_quality = envelope_func(inverse_quality, 1.0, p=6)
                # just set cutoff as 1.0, as we need to neglect the frames whose inverse_quality is close to 1
            else:
                frame_quality = envelope_cutoff(frame_quality, x1=self.frame_quality_eps, x2=self.frame_quality_scale)
            angle_gate = angle_gate * frame_quality
        gate_sum = angle_gate.sum(dim=1, keepdim=False).clamp_min(1e-6)

        if self.frame_angle_type == 1 and self.frame_quality_eps < 0:
            edge_angle_fcs = torch.sum(angle_gate * edge_angle_fcs, dim=1)
            # actually gate_sum should be 1.0 in this case, but we keep the division for numerical stability
        else:
            edge_angle_fcs = torch.sum(angle_gate * edge_angle_fcs, dim=1) / gate_sum # (N_edge, in_channels)
        edge_angle_fcs = self.angle_out_norm(edge_angle_fcs)

        edge = torch.cat([edge, edge_radius, edge_angle_fcs], dim=-1)
        edge = self.nn2(edge)
        edge = self.out_nn(edge)
        return self.softplus(edge_in + self.bn(edge))
