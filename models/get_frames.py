import torch
from torch_scatter import scatter
from mfan.models.generalized_qr import (
    batched_qr_orthonormalize,
    select_independent_triplets,
)

def get_alpha_frames(data, eps=1e-6, normalize=True):
    """
    Build PBC-consistent AlphaNet edge-wise frames.
    """

    i = data.edge_index[1]
    edge_vec = data.edge_attr

    edge_length = torch.norm(edge_vec, dim=-1, keepdim=True)
    edge_diff = edge_vec / (edge_length + eps)

    # Use PBC-corrected neighbor displacements consistently. For an edge j -> i,
    # the selected image of j is pos[i] + edge_vec, so its center-wise mean is
    # pos[i] + mean_edge. This avoids mixing wrapped positions with PBC-aware
    # edge directions.
    mean_edge = scatter(
        edge_vec,
        i,
        dim=0,
        dim_size=data.num_nodes,
        reduce="mean",
    )
    center_to_mean = mean_edge[i]
    edge_cross = torch.cross(
        -center_to_mean,
        edge_vec - center_to_mean,
        dim=-1,
    )
    edge_vertical = torch.cross(edge_diff, edge_cross, dim=-1)

    cross_length = torch.norm(edge_cross, dim=-1)
    vertical_length = torch.norm(edge_vertical, dim=-1)
    quality = torch.minimum(cross_length, vertical_length)
    if normalize:
        edge_cross = edge_cross / (cross_length.unsqueeze(-1) + eps)
        edge_vertical = edge_vertical / (vertical_length.unsqueeze(-1) + eps)

    return [edge_diff, edge_cross, edge_vertical], quality, data.edge_index, mean_edge

def get_qr_frames(data, eps=1e-6, normalize=True, sel_quality_alpha=100):
    """Batched QR frames, preserving the original neighbor-order logic."""
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    num_nodes = int(data.num_nodes)
    device = edge_attr.device

    src = edge_index[0]
    counts = torch.bincount(src, minlength=num_nodes)
    max_neighbors = max(int(counts.max().item()), 1)

    order = torch.argsort(src)
    src_sorted = src[order]
    edge_attr_sorted = edge_attr[order]
    starts = torch.cumsum(counts, dim=0) - counts
    local_idx = torch.arange(src_sorted.numel(), device=device) - torch.repeat_interleave(starts, counts)

    padded = edge_attr.new_zeros((num_nodes, max_neighbors, edge_attr.size(-1)))
    padded[src_sorted, local_idx] = edge_attr_sorted

    lengths = torch.norm(padded, dim=-1)
    masked_lengths = torch.where(
        lengths > eps,
        lengths,
        torch.full_like(lengths, float("inf")),
    )
    sort_idx = torch.argsort(masked_lengths, dim=1, stable=True)
    padded = padded.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, padded.size(-1)))

    candidate_rows, valid_mask, selection_gap = select_independent_triplets(padded, eps=eps)
    candidate_cols = candidate_rows.transpose(1, 2)

    q, r = batched_qr_orthonormalize(candidate_cols, eps=eps)
    r_diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
    qr_quality = torch.min(r_diag, dim=-1).values
    # Penalize unstable candidate choices when a near-equal-length competitor exists.
    selection_quality = torch.min(selection_gap, dim=-1).values
    quality = torch.minimum(qr_quality, sel_quality_alpha * selection_quality)
    quality = torch.where(
        valid_mask.all(dim=1),
        quality,
        torch.zeros_like(quality),
    )

    if normalize:
        q = q / (torch.norm(q, dim=1, keepdim=True) + eps)

    return [q[:, :, 0], q[:, :, 1], q[:, :, 2]], quality

def get_pca_frames(data, eps=1e-6, normalize=False, se3_frame=False):
    weighted_vec = data.edge_attr #/ (edge_len.unsqueeze(-1) + 0.1)**2 # [num_edges, 3]
    frame_vec_a = scatter(weighted_vec, data.edge_index[0], dim=0, dim_size=data.num_nodes, reduce="sum")
    frame_vec_c = data.edge_attr.unsqueeze(-1) * data.edge_attr.unsqueeze(-2)
    frame_vec_c = scatter(frame_vec_c, data.edge_index[0], dim=0, dim_size=data.num_nodes, reduce="sum") # [num_nodes, 3, 3]
    frame_vec_c = frame_vec_c + eps * torch.eye(3, device=frame_vec_c.device, dtype=frame_vec_c.dtype).view(1, 3, 3)
    eigen_values, frame_vec_c = torch.linalg.eigh(frame_vec_c)
    frame_vec_c1 = frame_vec_c[:, :, -1]  # [num_nodes, 3] max eigenvector
    frame_vec_c1_dir = (frame_vec_a * frame_vec_c1).sum(dim=-1)  # [num_nodes]
    frame_vec_c1 = frame_vec_c1 * torch.where(frame_vec_c1_dir < 0, -1.0, 1.0).unsqueeze(-1) # ensure the angle between a and b less than pi/2.
    frame_vec_c2 = frame_vec_c[:, :, -2]  # [num_nodes, 3] mid eigenvector
    frame_vec_c2_dir = (frame_vec_a * frame_vec_c2).sum(dim=-1)  # [num_nodes]
    frame_vec_c2 = frame_vec_c2 * torch.where(frame_vec_c2_dir < 0, -1.0, 1.0).unsqueeze(-1) # ensure the angle between a and b less than pi/2.
    frame_vec_c3 = frame_vec_c[:, :, -3]  # [num_nodes, 3] min eigenvector
    if se3_frame:
        frame_vec_c12 = torch.cross(frame_vec_c1, frame_vec_c2, dim=-1)
        frame_vec_c3_dir = (frame_vec_c12 * frame_vec_c3).sum(dim=-1)  # [num_nodes]
    else:
        frame_vec_c3_dir = (frame_vec_a * frame_vec_c3).sum(dim=-1)  # [num_nodes]
    frame_vec_c3 = frame_vec_c3 * torch.where(frame_vec_c3_dir < 0, -1.0, 1.0).unsqueeze(-1) # ensure the angle between a and b less than pi/2.
    if normalize:
        frame_vec_c1 = frame_vec_c1 / (torch.norm(frame_vec_c1, dim=-1, keepdim=True) + eps)
        frame_vec_c2 = frame_vec_c2 / (torch.norm(frame_vec_c2, dim=-1, keepdim=True) + eps)
        frame_vec_c3 = frame_vec_c3 / (torch.norm(frame_vec_c3, dim=-1, keepdim=True) + eps)

    gap1 = eigen_values[:, -1] - eigen_values[:, -2]
    gap2 = eigen_values[:, -2] - eigen_values[:, -3]

    quality = torch.min(
        torch.stack([
            gap1,
            gap2,
            frame_vec_c1_dir.abs(),
            frame_vec_c2_dir.abs(),
            frame_vec_c3_dir.abs(),
        ], dim=1),
        dim=1
    ).values

    return [frame_vec_c1, frame_vec_c2, frame_vec_c3], quality

def get_cell_frames(data, eps=1e-6, normalize=False):
    frame_vec_l = data.atom_lat[:,0,:]#[data.batch]
    frame_vec_w = data.atom_lat[:,1,:]#[data.batch]
    frame_vec_h = data.atom_lat[:,2,:]#[data.batch]
    if normalize:
        frame_vec_l = frame_vec_l / (torch.norm(frame_vec_l, dim=-1, keepdim=True) + eps)
        frame_vec_w = frame_vec_w / (torch.norm(frame_vec_w, dim=-1, keepdim=True) + eps)
        frame_vec_h = frame_vec_h / (torch.norm(frame_vec_h, dim=-1, keepdim=True) + eps)
    quality = torch.ones_like(frame_vec_l[:,0])
    return [frame_vec_l, frame_vec_w, frame_vec_h], quality

def get_nei_frames(data, eps=1e-6, normalize=False):
    # l/w/h: lattice vectors, where h is the largest lattice vector
    frame_vec_h = data.atom_lat[:,2,:]#[data.batch]
    weighted_vec = data.edge_attr #/ (edge_len.unsqueeze(-1) + eps)**2 # [num_edges, 3]
    frame_vec_a = scatter(weighted_vec, data.edge_index[0], dim=0, dim_size=data.num_nodes, reduce="sum") # [num_nodes, 3]
    if normalize:
        frame_vec_a = frame_vec_a / (torch.norm(frame_vec_a, dim=-1, keepdim=True) + eps)
        frame_vec_h = frame_vec_h / (torch.norm(frame_vec_h, dim=-1, keepdim=True) + eps)
    frame_vec_b = torch.cross(frame_vec_h, frame_vec_a, dim=-1)  # [num_nodes, 3]
    if normalize:
        frame_vec_b = frame_vec_b / (torch.norm(frame_vec_b, dim=-1, keepdim=True) + eps)
    quality = torch.norm(frame_vec_b, dim=1) # [num_nodes]
    return [frame_vec_a, frame_vec_b, frame_vec_h], quality

def get_null_frames(data, eps=1e-6, normalize=False):
    # empty frames, neglect all angle features test
    vec1 = torch.zeros_like(data.atom_lat[:,2,:])
    vec2 = torch.zeros_like(data.atom_lat[:,2,:])
    vec3 = torch.zeros_like(data.atom_lat[:,2,:])
    quality = torch.ones_like(vec1[:,0])
    return [vec1, vec2, vec3], quality
