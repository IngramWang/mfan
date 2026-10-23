# Original version from MFA, rewritten by Codex
import torch


def batched_qr_orthonormalize(a: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Euclidean QR orthonormalization for [B, 3, 3] column bases."""
    if a.dim() != 3 or a.size(-2) != 3 or a.size(-1) != 3:
        raise ValueError(f"Expected [B, 3, 3] input, got {tuple(a.shape)}")

    bsz = a.size(0)
    eye = torch.eye(3, dtype=a.dtype, device=a.device).unsqueeze(0).expand(bsz, -1, -1)

    q, r = torch.linalg.qr(a + eps * eye, mode="reduced")

    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    col_sign = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
    q = q * col_sign.unsqueeze(-2)
    r = r * col_sign.unsqueeze(-1)

    det = torch.linalg.det(q)
    last_sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    q = torch.stack([q[:, :, 0], q[:, :, 1], q[:, :, 2] * last_sign.unsqueeze(-1)], dim=-1)
    r = torch.stack([r[:, 0, :], r[:, 1, :], r[:, 2, :] * last_sign.unsqueeze(-1)], dim=1)
    return q, r


def _first_true_index(mask: torch.Tensor) -> torch.Tensor:
    """Return the first True index in each row, or -1 if none exists."""
    idx = torch.arange(mask.size(1), device=mask.device).unsqueeze(0).expand_as(mask)
    fallback = torch.full_like(idx, mask.size(1))
    first = torch.where(mask, idx, fallback).min(dim=1).values
    return torch.where(first == mask.size(1), torch.full_like(first, -1), first)


def _next_valid_length_gap(
    lengths: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_idx: torch.Tensor,
) -> torch.Tensor:
    """Distance gap to the next valid competitor after the selected vector."""
    idx = torch.arange(lengths.size(1), device=lengths.device).unsqueeze(0).expand_as(lengths)
    has_selected = selected_idx >= 0
    after_selected = idx > selected_idx.clamp_min(0).unsqueeze(1)
    competitor_mask = valid_mask & after_selected

    next_idx = _first_true_index(competitor_mask)
    safe_selected_idx = selected_idx.clamp_min(0)
    safe_next_idx = next_idx.clamp_min(0)

    selected_len = lengths[torch.arange(lengths.size(0), device=lengths.device), safe_selected_idx]
    next_len = lengths[torch.arange(lengths.size(0), device=lengths.device), safe_next_idx]
    gap = next_len - selected_len

    no_competitor = next_idx < 0
    inf_gap = torch.full_like(gap, float("inf"))
    zero_gap = torch.zeros_like(gap)
    gap = torch.where(has_selected, gap, zero_gap)
    return torch.where(no_competitor & has_selected, inf_gap, gap)


def select_independent_triplets(
    coords: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select up to three linearly independent vectors from [B, M, 3],
    following the original neighbor order as closely as possible.

    Returns:
        basis_rows: [B, 3, 3]
        valid_mask: [B, 3]
        selection_gap: [B, 3], distance gap to the next valid competitor
    """
    if coords.dim() != 3 or coords.size(-1) != 3:
        raise ValueError(f"Expected [B, M, 3] input, got {tuple(coords.shape)}")

    bsz, max_neighbors, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    lengths = torch.norm(coords, dim=-1)
    valid1 = lengths > eps
    idx1 = _first_true_index(valid1)
    gap1 = _next_valid_length_gap(lengths, valid1, idx1)

    safe_idx1 = idx1.clamp_min(0)
    v1 = coords[torch.arange(bsz, device=device), safe_idx1]
    has_v1 = idx1 >= 0
    v1 = torch.where(has_v1.unsqueeze(-1), v1, torch.zeros_like(v1))

    cross1 = torch.cross(coords, v1.unsqueeze(1).expand(-1, max_neighbors, -1), dim=-1)
    valid2 = (torch.norm(cross1, dim=-1) > eps) & valid1
    if max_neighbors > 0:
        valid2[torch.arange(bsz, device=device), safe_idx1] = False
    idx2 = _first_true_index(valid2)
    gap2 = _next_valid_length_gap(lengths, valid2, idx2)

    safe_idx2 = idx2.clamp_min(0)
    v2 = coords[torch.arange(bsz, device=device), safe_idx2]
    has_v2 = idx2 >= 0
    v2 = torch.where(has_v2.unsqueeze(-1), v2, torch.zeros_like(v2))

    normal12 = torch.cross(v1, v2, dim=-1)
    triple = torch.abs((coords * normal12.unsqueeze(1)).sum(dim=-1))
    valid3 = (triple > eps) & valid1
    if max_neighbors > 0:
        valid3[torch.arange(bsz, device=device), safe_idx1] = False
        valid3[torch.arange(bsz, device=device), safe_idx2] = False
    idx3 = _first_true_index(valid3)
    gap3 = _next_valid_length_gap(lengths, valid3, idx3)

    safe_idx3 = idx3.clamp_min(0)
    v3 = coords[torch.arange(bsz, device=device), safe_idx3]
    has_v3 = idx3 >= 0
    v3 = torch.where(has_v3.unsqueeze(-1), v3, torch.zeros_like(v3))

    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(bsz, -1, -1)
    basis = eye.clone()

    basis[:, 0, :] = torch.where(has_v1.unsqueeze(-1), v1, eye[:, 0, :])
    basis[:, 1, :] = torch.where(has_v2.unsqueeze(-1), v2, eye[:, 1, :])
    basis[:, 2, :] = torch.where(has_v3.unsqueeze(-1), v3, eye[:, 2, :])

    valid_mask = torch.stack([has_v1, has_v2, has_v3], dim=1)
    selection_gap = torch.stack([gap1, gap2, gap3], dim=1)
    return basis, valid_mask, selection_gap
