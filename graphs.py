"""Module to generate networkx graphs."""
"""Implementation based on the template of Matformer."""
import numpy as np
from jarvis.core.specie import chem_data, get_node_attributes

from jarvis.core.atoms import Atoms
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch


class MEData(Data):
    """Graph data type retained for dataset and cache compatibility."""

def angle_from_array(a, b, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    assert a_new.shape == a.shape
    value = sum(a_new * b_new)
    length = (sum(a_new ** 2) ** 0.5) * (sum(b_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return angle / np.pi * 180.0

def correct_coord_sys(a, b, c, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    c_new = np.dot(c, lattice)
    assert a_new.shape == a.shape
    plane_vec = np.cross(a_new, b_new)
    value = sum(plane_vec * c_new)
    length = (sum(plane_vec ** 2) ** 0.5) * (sum(c_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return (angle / np.pi * 180.0 <= 90.0)

def same_line(a, b):
    a_new = a / (sum(a ** 2) ** 0.5)
    b_new = b / (sum(b ** 2) ** 0.5)
    flag = False
    if abs(sum(a_new * b_new) - 1.0) < 1e-5:
        flag = True
    elif abs(sum(a_new * b_new) + 1.0) < 1e-5:
        flag = True
    else:
        flag = False
    return flag

def same_plane(a, b, c):
    flag = False
    if abs(np.dot(np.cross(a, b), c)) < 1e-5:
        flag = True
    return flag


# pyg dataset
class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        graphs: Sequence[MEData],
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        classification=False,
        id_tag="jid",
        neighbor_strategy="",
        nolinegraph=False,
        need_force=False,
        need_stress=False,  # cchen ###
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.graphs = graphs
        self.line_graph = line_graph
        self.need_force = need_force
        self.need_stress = need_stress  # cchen ###
        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f

        self.prepare_batch = prepare_pyg_batch
        if line_graph:
            self.prepare_batch = prepare_pyg_line_graph_batch


    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        if self.line_graph:
            return g, g, g

        return g

    @staticmethod
    def check_edge_index_range(batch, key="edge_index"):
        ei = getattr(batch, key)
        assert ei.dtype == torch.long
        assert ei.dim() == 2 and ei.size(0) == 2
        m = int(ei.max()) if ei.numel() else -1
        if m < batch.num_nodes:
            return True
        else:
            print("ERROR: edges out of batch ranges")
            return False

    @staticmethod
    def check_edges_within_ptr_ranges(batch, key="edge_index"):
        ei = getattr(batch, key)
        if ei.numel() == 0:
            return True

        ptr = batch.ptr  # [num_graphs+1]
        g = batch.batch
        ge = g[ei[0]]
        start = ptr[ge]
        end = ptr[ge + 1]

        bad = (ei[0] < start) | (ei[0] >= end) | (ei[1] < start) | (ei[1] >= end)
        if bad.any():
            idx = bad.nonzero(as_tuple=False).view(-1)[:10]
            print(f"ERROR: edges out of node ranges, examples={ei[:, idx].t()}")
            return False
        return True

    @staticmethod
    def collate(samples: List[MEData], need_force=False, need_stress=False):  # cchen ###
        """Dataloader helper to batch graphs cross `samples`."""
        batched_graph = Batch.from_data_list(samples)
        label = batched_graph.target
        if need_force:
            force = batched_graph.force
            if need_stress:  # cchen ###
                stress = batched_graph.stress
                label = (label, force, stress)
            else:
                label = (label, force)
        return batched_graph, label

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[MEData, MEData, torch.Tensor]],
        need_force=False,
        need_stress=False,  # cchen ###
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, lattice = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        label = batched_graph.target
        if need_force:
            force = batched_graph.force
            if need_stress:  # cchen ###
                stress = batched_graph.stress
                label = (label, force, stress)
            else:
                label = (label, force)
        return batched_graph, batched_line_graph, batched_line_graph, label

def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def build_lattice_frames(
    atoms=None,
):
    """lattice correction process"""
    lat = atoms.lattice
    r_cut = max(lat.a, lat.b, lat.c) + 1e-2
    all_neighbors = atoms.get_all_neighbors(r=r_cut)
    neighborlist = all_neighbors[0]
    neighborlist = sorted(neighborlist, key=lambda x: x[2])
    ids = np.array([nbr[1] for nbr in neighborlist])
    images = np.array([nbr[3] for nbr in neighborlist])
    images = images[ids == 0]
    lat1 = images[0]
    # finding lat2
    start = 1
    for i in range(start, len(images)):
        lat2 = images[i]
        if not same_line(lat1, lat2):
            start = i
            break
    # finding lat3
    for i in range(start, len(images)):
        lat3 = images[i]
        if not same_plane(lat1, lat2, lat3):
            break
    # find the invariant corner
    if angle_from_array(lat1,lat2,lat.matrix) > 90.0:
        lat2 = - lat2
    if angle_from_array(lat1,lat3,lat.matrix) > 90.0:
        lat3 = - lat3
    # find the invariant coord system
    if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
        lat1 = - lat1
        lat2 = - lat2
        lat3 = - lat3

    v1, v2, v3 = atoms.lattice.cart_coords(lat1), atoms.lattice.cart_coords(lat2), atoms.lattice.cart_coords(lat3)
    atom_lat = [[v1, v2, v3]]
    atom_lat = torch.tensor(np.array(atom_lat)).type(torch.get_default_dtype())

    return atom_lat, [v1, v2, v3]

def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
    use_lattice=False,
    use_angle=False,
    lats=None
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
            use_lattice=use_lattice,
            #use_angle=use_angle,
            lats=lats,
        )

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors_now):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(lats[0]))
            edges[(site_idx, site_idx)].add(tuple(lats[1]))
            edges[(site_idx, site_idx)].add(tuple(lats[2]))

    return edges

def cutoff_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    edges = defaultdict(set)

    for site_idx, neighborlist in enumerate(all_neighbors_now):
        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges

def compute_bond_cosine(v1, v2):
    """Compute bond angle cosines from bond displacement vectors."""
    v1 = torch.tensor(v1).type(torch.get_default_dtype())
    v2 = torch.tensor(v2).type(torch.get_default_dtype())
    bond_cosine = torch.sum(v1 * v2) / (
        torch.norm(v1) * torch.norm(v2)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


def build_undirected_edgedata(
    atoms=None,
    edges={},
    lats=[],
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    u, v, r, l = [], [], [], []
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)

    if len(u) > 0:
        u = torch.tensor(u)
        v = torch.tensor(v)
        r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    else:
        u = torch.empty((0))
        v = torch.empty((0))
        r = torch.empty((0,3)).type(torch.get_default_dtype())
    l = torch.tensor(l).type(torch.int)
    return u, v, r, l


class PygGraph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=4.0,
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice: bool = False,
        use_angle: bool = False,
        need_force: bool = False,
    ):
        new_atoms = Atoms(
            lattice_mat = atoms.lattice_mat,
            coords = [atoms.frac_coords[0]],
            elements = [atoms.elements[0]],
            cartesian = False
        ) # only one atom, accelerate the lattice building process
        atom_lat, lats = build_lattice_frames(atoms=new_atoms)
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
                lats=lats,
            )
            u, v, r, l = build_undirected_edgedata(atoms, edges, lats)
        elif neighbor_strategy == "cutoff":
            edges = cutoff_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
            )
            u, v, r, l = build_undirected_edgedata(atoms, edges, lats)
        elif neighbor_strategy == "torch":
            pass
            # pass all information to torch and compute edges there, which can be derived
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        atom_lat = atom_lat.repeat(node_features.shape[0],1,1)

        if neighbor_strategy == "torch":
            atoms = atoms.ase_converter()
            z = torch.tensor(atoms.numbers, dtype=torch.long)
            pos = torch.tensor(atoms.get_positions(wrap=True), dtype=torch.float)
            cell = torch.tensor(np.asarray(atoms.get_cell()), dtype=torch.float)
            natoms = torch.tensor([len(atoms)], dtype=torch.long)
            g = MEData(x=node_features, z=z, pos=pos, cell=cell, natoms=natoms, atom_lat=atom_lat)
        else:
            edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
            g = MEData(x=node_features, edge_index=edge_index, edge_attr=r, edge_type=l, atom_lat=atom_lat)
        return g


def prepare_pyg_batch(
    batch: Tuple[MEData, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    if isinstance(t, tuple):
        t = tuple(x.to(device, non_blocking=non_blocking) for x in t)
    else:
        t = t.to(device, non_blocking=non_blocking)

    batch = (
        g.to(device),
        t,
    )

    return batch


def prepare_pyg_line_graph_batch(
    batch: Tuple[Tuple[MEData, MEData, torch.Tensor], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, lattice, t = batch
    if isinstance(t, tuple):
        t = tuple(x.to(device, non_blocking=non_blocking) for x in t)
    else:
        t = t.to(device, non_blocking=non_blocking)

    batch = (
        (
            g.to(device),
            lg.to(device),
            lattice.to(device, non_blocking=non_blocking),
        ),
        t
    )

    return batch

