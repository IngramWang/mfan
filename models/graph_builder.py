#
# Modified from Alphanet, the original code is licensed under GNU General Public License v3.0
# https://github.com/zmyybc/AlphaNet
#


import torch
from torch import Tensor
from typing import Optional, Tuple, List
from torch_scatter import segment_coo, segment_csr
from mfan.graphs import MEData

def get_max_neighbors_mask(
   natoms: Tensor,
   index: Tensor,
   atom_distance: Tensor,
   max_num_neighbors_threshold: int,
   precision: torch.dtype
) :
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=torch.bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.ones(
    num_atoms * max_num_neighbors,
    device=device
     ) * float('inf')
    distance_sort = distance_sort.to(precision)
    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=torch.bool)
    mask_num_neighbors.index_fill_(0, index_sort, torch.tensor(True, device=device))

    return mask_num_neighbors, num_neighbors_image

def check_and_reshape_cell(cell: Optional[torch.Tensor]) -> torch.Tensor:

    if cell is None:
        return torch.eye(3, dtype=torch.float32).unsqueeze(0)


    if cell.dim() == 2 and cell.size(0) % 3 == 0 and cell.size(1) == 3:
        batch_size = cell.size(0) // 3
        cell = cell.reshape(batch_size, 3, 3)
    elif cell.dim() != 3 or cell.size(1) != 3 or cell.size(2) != 3:
        raise ValueError(f"Invalid cell shape. Expected (batch_size, 3, 3), but got {cell.size()}")

    return cell

def radius_graph_pbc(
   pos: Tensor,
   natoms: Tensor,
   cell: Tensor,
   radius: float,
   max_num_neighbors_threshold: int,
   pbc: Optional[List[bool]] = None,
   precision: torch.dtype = torch.float32
):
    if pbc is None:
        pbc = [True, True, True]
    device = pos.device
    batch_size = len(natoms)
    atom_pos = pos
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )
    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this approach could run into numerical precision issues
    index1 = (
        torch.div(
            atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
        )
    ) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol,  dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [int(rep_a1.max()), int(rep_a2.max()), int(rep_a3.max())]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=precision)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(cells_per_dim[0],cells_per_dim[1], cells_per_dim[2])
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)

    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        precision = precision
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image

def get_pbc_distances(
    pos: Tensor,
    edge_index: Tensor,
    cell: Tensor,
    cell_offsets: Tensor,
    neighbors: Tensor,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
    precision: torch.dtype = torch.float32
):
    row= edge_index[0]
    col = edge_index[1]

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.to(precision).view(-1, 1, 3).bmm(cell.to(precision)).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1 , p=2)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[
        distances != 0
    ]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out

# Borrowed from MACE
def get_symmetric_displacement(
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        num_graphs: int,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cell is None:
            cell = torch.zeros(
                num_graphs * 3,
                3,
                dtype=positions.dtype,
                device=positions.device,
            )

        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=positions.dtype,
            device=positions.device,
         )

        displacement.requires_grad_(True)
        symmetric_displacement = 0.5 * (
            displacement + displacement.transpose(-1, -2)
        )

        positions = positions + torch.einsum(
            "be,bec->bc", positions, symmetric_displacement[batch]
        )
        cell = cell.view(-1, 3, 3)
        cell = cell + torch.matmul(cell, symmetric_displacement)
        # cchen: removed dead code `cell.view(-1, 3)` — unused return value ###
        return positions, cell, displacement

def process_positions_and_edges(
    graph_data: MEData,
    pos: Tensor,
    z: Tensor,
    natoms: Tensor,
    batch: Tensor,
    cell: Optional[Tensor] = None,
    compute_stress: bool = False,
    compute_forces: bool = False,
    use_pbc: bool = False,
    cutoff: float = 5.0,
    max_neighbors_threshold: int = 50,
    dtype: torch.dtype = torch.float32
) -> MEData:
    """
    Process atomic positions and compute edges with optional PBC support.
    We found that non-pbc graph is not compatible with jit compile, so we don't support that for now, please create a large cell if you want to do non-pbc calculation.
    Args:
        data: Input data object containing positions, batch info, and other attributes
        compute_forces: Boolean flag for force computation
        compute_stress: Boolean flag for stress computation
        use_pbc: Boolean flag for periodic boundary conditions
        cutoff: Cutoff radius for neighbor search
        dtype: torch dtype precision

    Returns:
        Data:  Data object containing processed attributes
    """
    precision = dtype
    pos = pos.to(precision)
    z = z.long()

    if compute_stress:

        pos, cell, displacement = get_symmetric_displacement(
            pos, cell, num_graphs=int(torch.max(batch))+1, batch=batch
        )
        # cchen: apply same affine transform to lattice frame for computational graph integrity
        if hasattr(graph_data, 'atom_lat') and graph_data.atom_lat is not None:
            symmetric_disp = 0.5 * (displacement + displacement.transpose(-1, -2))
            atom_lat = graph_data.atom_lat
            atom_lat_strained = atom_lat + torch.matmul(
                atom_lat, symmetric_disp[batch]
            )
            graph_data.atom_lat = atom_lat_strained
        ###
    else:
        displacement = None

    cell = check_and_reshape_cell(cell)

    if use_pbc and cell is not None:

        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            pos, natoms, cell, cutoff,
            max_num_neighbors_threshold=max_neighbors_threshold,
            precision = precision
        )
        out = get_pbc_distances(
            pos,
            edge_index,
            cell,
            cell_offsets,
            neighbors,
            return_distance_vec=True,
            precision = precision
        )

        edge_index = out["edge_index"]
        vecs = out["distance_vec"]

    else:
        raise ValueError(f"Non-PBC graphs are not supported yet, as radius graph is not compilable with jit")

    graph_data.edge_index = edge_index
    graph_data.edge_attr = vecs
    # cchen: expose differentiable displacement tensor for downstream autograd.grad(energy, displacement)
    graph_data.displacement = displacement
    ###
    return graph_data
