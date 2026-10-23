#
# Modified from Alphanet, the original code is licensed under GNU General Public License v3.0
# https://github.com/zmyybc/AlphaNet
#

import torch
import numpy
import ase
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from mfan.models.mfan import mfan_tg_wrapper, mfanConfig
from mfan.models.comformer import iComformerConfig, eComformerConfig, comformer_tg_wrapper
from mfan.models.get_frames import (
    get_alpha_frames,
    get_cell_frames,
    get_nei_frames,
    get_null_frames,
    get_pca_frames,
    get_qr_frames,
)
from mfan.models.graph_builder import process_positions_and_edges
from mfan.graphs import MEData, build_lattice_frames
from jarvis.core.specie import get_node_attributes
from jarvis.core.atoms import ase_to_atoms
np = numpy


_FRAME_BUILDERS = {
    "alpha": get_alpha_frames,
    "qr": get_qr_frames,
    "pca": get_pca_frames,
    "cell": get_cell_frames,
    "lattice": get_cell_frames,
    "nei": get_nei_frames,
    "null": get_null_frames,
}


class mfanCalculator(Calculator):
    """
    This calculator wraps our model to perform energy, force, and stress
    within the ASE framework.
    For non-periodic systems, it creates a large supercell with vacuum padding
    to simulate an isolated molecule.
    """
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']  # cchen ###

    def __init__(self, ckpt_path, config, device='cpu', precision='32', **kwargs):
        """
        Initializes the mfanCalculator.

        Args:
            ckpt_path (str): Path to the model checkpoint file (.ckpt or .pt).
            config (object): Model configuration object.
            device (str): Device to run the model on ('cpu' or 'cuda').
            precision (str): Precision for calculations ('32' for float, '64' for double).
            **kwargs: Additional arguments for the base ASE Calculator.
        """
        Calculator.__init__(self, **kwargs)

        self.device = torch.device(device)
        if precision == "32":
            self.precision = torch.float32
        elif precision == "64":
            self.precision = torch.float64
        else:
            raise ValueError(f"Unsupported precision: {precision}. Use '32' or '64'.")

        if type(config) is dict:
            model_type = config['name']
            if model_type in ('mfan'):
                config = mfanConfig(**config)
            elif model_type == 'iComformer':
                config = iComformerConfig(**config)
            elif model_type == 'eComformer':
                config = eComformerConfig(**config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}.")

        self.model = self._load_model(ckpt_path, config)

        if self.precision == torch.float64:
            self.model.double()

        self.model.eval()
        self.model.to(self.device)
        self.config = config
        self.need_force = self.config.need_force
        self.need_stress = getattr(self.config, 'need_stress', False)  # cchen ###
        if config.name == 'mfan':
            self.atom_features = self.config.atom_features
        else:
            self.atom_features = 'cgcnn'
        if self.atom_features == 'emb':
            self.atom_features = 'atomic_number'
        self.buffered_lat = None

    def _load_model(self, ckpt_path, config):
        checkpoint = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, mfan_tg_wrapper) or isinstance(checkpoint, comformer_tg_wrapper):
            return checkpoint.to(self.device)
        elif not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
        model_state = checkpoint.get("model", checkpoint)
        if isinstance(model_state, torch.nn.Module):
            model_state = model_state.state_dict()
        if not isinstance(model_state, dict):
            raise ValueError(f"Unsupported model state format: {type(model_state)}")

        if config.name == 'mfan':
            model = mfan_tg_wrapper(config).to(self.device)
        elif config.name == 'iComformer' or config.name == 'eComformer':
            model = comformer_tg_wrapper(config).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {config.name}.")
        model.load_state_dict(model_state, strict=True)
        return model

    def reset_buffered_lat(self, atoms=None):
        """Precompute and store the lattice frame for accurate calculations.
           This is only used for simulations where the supercell does not change."""
        if not isinstance(atoms, Atoms):
            self.buffered_lat = None
            print('Warning: buffered lattice is reset to None.')
        elif atoms.pbc.any():
            self.buffered_lat, _ = build_lattice_frames(ase_to_atoms(atoms[:1]))
            print('Warning: the lattice frame is precomputed and stored.')
            print('Remember to reset it while the supercell changed')
        else:
            self.buffered_lat = None
            print('Warning: input atoms is non-periodic, buffered lattice is set to None.')

    def _atom2data(self, atoms, atom_features="cgcnn"):
        sps_features = []
        for s in atoms.symbols:
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features, dtype=self.precision)

        if self.buffered_lat is not None:
            atom_lat = self.buffered_lat
        else:
            a = atoms[:1]
            a.wrap()
            atom_lat, _ = build_lattice_frames(ase_to_atoms(a))
        atom_lat = atom_lat.to(dtype=self.precision).repeat(node_features.shape[0],1,1)

        z = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(wrap=True), dtype=self.precision)
        cell = torch.tensor(np.asarray(atoms.get_cell()), dtype=self.precision)
        natoms = torch.tensor([len(atoms)], dtype=torch.long)
        g = MEData(x=node_features, z=z, pos=pos, cell=cell, natoms=natoms, atom_lat=atom_lat)
        g.batch = torch.zeros_like(g.z)
        return g.to(self.device)

    def _prepare_atoms(self, atoms):
        """ If the system is non-periodic, create a large vacuum box."""
        if atoms.pbc.any():
            return atoms

        calc_atoms = atoms.copy()
        padding = 20.0
        new_cell_dims = np.ptp(calc_atoms.get_positions(), axis=0) + padding
        calc_atoms.set_cell(np.diag(new_cell_dims))
        calc_atoms.center()
        calc_atoms.pbc = True
        return calc_atoms

    def get_frames(
        self,
        atoms=None,
        frame_type="qr",
        cutoff=None,
        max_neighbors_threshold=None,
        eps=1e-4,
        normalize=True,
        **frame_kwargs,
    ):
        """Construct frames with this calculator's graph settings, without prediction."""
        frame_key = str(frame_type).lower()
        if frame_key not in _FRAME_BUILDERS:
            supported = ", ".join(sorted(_FRAME_BUILDERS))
            raise ValueError(
                f"Unsupported frame type: {frame_type!r}. "
                f"Supported types: {supported}."
            )

        if cutoff is None:
            cutoff = self.config.cutoff
        if max_neighbors_threshold is None:
            max_neighbors_threshold = getattr(
                self.config, "max_neighbors_threshold", 50
            )
        if atoms is None:
            calc_atoms = self._prepare_atoms(self.atoms)
        else:
            calc_atoms = self._prepare_atoms(atoms)
        graph_data = self._atom2data(
            calc_atoms,
            atom_features=self.atom_features,
        )
        graph_data = process_positions_and_edges(
            graph_data=graph_data,
            pos=graph_data.pos,
            z=graph_data.z,
            natoms=graph_data.natoms,
            batch=graph_data.batch,
            cell=graph_data.cell,
            compute_forces=False,
            compute_stress=False,
            use_pbc=True,
            cutoff=cutoff,
            max_neighbors_threshold=max_neighbors_threshold,
            dtype=self.precision,
        )

        frame_builder = _FRAME_BUILDERS[frame_key]
        return frame_builder(
            graph_data,
            eps=eps,
            normalize=normalize,
            **frame_kwargs,
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Performs the calculation of energy, forces, and stress.

        Args:
            atoms (ase.Atoms): The atoms object to calculate properties for.
            properties (list of str): List of properties to calculate.
            system_changes (list of str): List of changes since the last calculation.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        properties = properties or ['energy']

        # --- Handle Non-Periodic Systems ---
        # If the system is not periodic (e.g., a molecule), we create a copy and
        # place it in a large box with vacuum padding. This allows the model,
        # which assumes periodicity, to treat it as an isolated system.
        if atoms is not None:
            calc_atoms = self._prepare_atoms(atoms)
        else:
            calc_atoms = self._prepare_atoms(self.atoms)

        g = self._atom2data(calc_atoms, atom_features=self.atom_features)

        # --- Run Model Inference ---
        # cchen: enable grad for force and/or stress ###
        with torch.set_grad_enabled(self.need_force or self.need_stress):
            model_output = self.model((g, None, None), "infer")

            if self.need_force and self.need_stress:
                energy, forces, stress = model_output
            elif self.need_force:
                energy, forces = model_output
                stress = None
            else:
                energy = model_output
                forces = None
                stress = None
        ###

        # --- Store Results ---
        self.results['energy'] = energy.detach().cpu().item() * len(atoms)
        self.results['free_energy'] = self.results['energy']

        if forces is not None:
            self.results['forces'] = forces.detach().cpu().numpy()

        # cchen: stress -> ASE Voigt notation [xx, yy, zz, yz, xz, xy] in eV/Å^3 ###
        if stress is not None:
            stress_matrix = stress.detach().cpu().numpy()
            if stress_matrix.ndim == 3:
                stress_matrix = stress_matrix[0]
            self.results['stress'] = np.array([
                stress_matrix[0, 0],  # xx
                stress_matrix[1, 1],  # yy
                stress_matrix[2, 2],  # zz
                stress_matrix[1, 2],  # yz
                stress_matrix[0, 2],  # xz
                stress_matrix[0, 1],  # xy
            ])
        ###
