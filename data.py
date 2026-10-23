"""Implementation based on the template of ALIGNN."""

import random
from pathlib import Path
from typing import Optional
from functools import partial


import os
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms, ase_to_atoms
from ase import Atoms as aseAtoms
from mfan.graphs import PygGraph, PygStructureDataset
from jarvis.db.figshare import data as jdata
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson
from pandarallel import pandarallel
import pickle as pk
# use pandas progress_apply

from ase.io import read as aseread

tqdm.pandas()
pandarallel.initialize(progress_bar=False)


def load_dataset(
    name: str = "dft_3d",
    target=None,
    limit: Optional[int] = None,
    classification_threshold: Optional[float] = None,
):
    """Load jarvis data."""
    d = jdata(name)
    data = []
    for i in d:
        if i[target] != "na" and not math.isnan(i[target]):
            if classification_threshold is not None:
                if i[target] <= classification_threshold:
                    i[target] = 0
                elif i[target] > classification_threshold:
                    i[target] = 1
                else:
                    raise ValueError(
                        "Check classification data type.",
                        i[target],
                        type(i[target]),
                    )
            data.append(i)
    d = data
    if limit is not None:
        d = d[:limit]
    d = pd.DataFrame(d)
    return d

def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def load_pyg_graphs(
    df,
    name: str = "dft_3d",
    target: str = "",
    force_target: str = "",
    need_force: bool = False,
    stress_target: str = "",  # cchen ###
    need_stress: bool = False,  # cchen ###
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    use_lattice: bool = False,
    use_angle: bool = False,
    mean_train: float = 0,
    std_train: float = 1,
    load_label_from_atom: bool = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        if type(atoms) == dict:
            structure = Atoms.from_dict(atoms)
        elif type(atoms) == aseAtoms:
            structure = ase_to_atoms(atoms)
        else:
            structure = atoms
        graph = PygGraph.atom_dgl_multigraph(
            structure,
            neighbor_strategy=neighbor_strategy,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
            use_angle=use_angle,
            need_force=need_force,
        )
        if load_label_from_atom:
            if len(target) > 0:
                graph.target = atoms.info[target]/len(atoms)
            else:
                graph.target = atoms.get_potential_energy()/len(atoms) #use energy per atom as the target
            graph.target = (graph.target - mean_train) / std_train
            graph.target = torch.tensor([graph.target], dtype=torch.float)

            if need_force:
                if len(force_target) > 0:
                    graph.force = atoms.arrays[force_target]
                else:
                    graph.force = atoms.get_forces()
                graph.force = graph.force / std_train
                graph.force = torch.tensor(graph.force, dtype=torch.float)
            # cchen: load stress label ###
            if need_stress:
                if len(stress_target) > 0:
                    if stress_target not in atoms.info:
                        raise ValueError(
                            f"need_stress=True, but stress target "
                            f"{stress_target!r} was not found in atoms.info."
                        )
                    stress_val = atoms.info[stress_target]
                else:
                    stress_val = atoms.info.get('stress', atoms.info.get('REF_stress', None))
                    if stress_val is None and atoms.calc is not None and 'stress' in atoms.calc.results:
                        stress_val = atoms.get_stress(voigt=False)

                if stress_val is None:
                    raise ValueError(
                        "need_stress=True, but no stress label was found"
                    )

                stress_val = np.asarray(stress_val)
                if stress_val.shape == (6,):
                    # Voigt [xx, yy, zz, yz, xz, xy] -> 3x3
                    s = np.zeros((3, 3), dtype=stress_val.dtype)
                    s[0, 0], s[1, 1], s[2, 2] = stress_val[0], stress_val[1], stress_val[2]
                    s[1, 2] = s[2, 1] = stress_val[3]
                    s[0, 2] = s[2, 0] = stress_val[4]
                    s[0, 1] = s[1, 0] = stress_val[5]
                    stress_val = s
                elif stress_val.shape != (3, 3):
                    raise ValueError(
                        "Stress label must have shape (6,) or (3, 3), "
                        f"got {stress_val.shape}."
                    )
                graph.stress = torch.tensor(stress_val, dtype=torch.float) / std_train
            ###
        return graph

    if type(df) == pd.DataFrame:
        graphs = df["atoms"].apply(atoms_to_graph).values
        if not load_label_from_atom:
            labels = df[target].values
            labels = (labels - mean_train) / std_train
            for i in range(len(graphs)):
                graphs[i].target = torch.as_tensor(labels[i], dtype=torch.float)
            if need_force:
                forces = df[force_target].values
                for i in range(len(forces)):
                    forces[i] = forces[i] / std_train
                    graphs[i].force = torch.as_tensor(forces[i], dtype=torch.float)
            if need_stress:
                stress_values = df[stress_target].values
                for i, stress_val in enumerate(stress_values):
                    stress_val = np.asarray(stress_val)
                    if stress_val.shape == (6,):
                        # Voigt [xx, yy, zz, yz, xz, xy] -> symmetric 3x3
                        stress_matrix = np.zeros((3, 3), dtype=stress_val.dtype)
                        stress_matrix[0, 0], stress_matrix[1, 1], stress_matrix[2, 2] = stress_val[:3]
                        stress_matrix[1, 2] = stress_matrix[2, 1] = stress_val[3]
                        stress_matrix[0, 2] = stress_matrix[2, 0] = stress_val[4]
                        stress_matrix[0, 1] = stress_matrix[1, 0] = stress_val[5]
                        stress_val = stress_matrix
                    elif stress_val.shape != (3, 3):
                        raise ValueError(
                            f"Stress sample {i} must have shape (6,) or (3, 3), "
                            f"got {stress_val.shape}."
                        )
                    graphs[i].stress = torch.as_tensor(
                        stress_val / std_train, dtype=torch.float
                    )
    elif type(df) == list:
        if not load_label_from_atom:
            raise ValueError("load_label_from_atom must be True when df is list of atoms")
        graphs = [atoms_to_graph(atoms) for atoms in df]
    else:
        raise ValueError("Unsupported data type for graph construction.")

    return graphs

def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_pyg_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    force_target="",
    need_force=False,
    stress_target="",  # cchen ###
    need_stress=False,  # cchen ###
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
    use_lattice=False,
    use_angle=False,
    data_from='Jarvis',
    use_save=False,
    mean_train=None,
    std_train=None,
    now=False, # for test
    only_mean_std=False,
    norm_mean_std=True,
    load_label_from_atom=False
):
    """Get pyg Dataset."""
    if len(dataset) == 0 or type(dataset[0]) == dict:
        df = pd.DataFrame(dataset)
        vals = df[target].values
        if target == "shear modulus" or target == "bulk modulus":
            val_list = [vals[i].item() for i in range(len(vals))]
            vals = val_list
    elif only_mean_std:
        raise ValueError("dataset is not list of dict, cannot compute mean and std")
    else:
        df = dataset
        vals = [0] #placeholder
        if norm_mean_std:
            norm_mean_std = False
            print("Warning: dataset is not list of dict, cannot compute mean and std, disable norm_mean_std")
    print("data range max {} min {}".format(np.max(vals), np.min(vals)))

    if only_mean_std:
        mean_train = np.mean(vals)
        std_train = np.std(vals)
        print("returning mean {} std {}".format(np.max(mean_train), np.min(std_train)))
        return [], mean_train, std_train

    if (mean_train == None or std_train == None) and norm_mean_std:
        mean_train = np.mean(vals)
        std_train = np.std(vals)
        print("label mean {} std {}".format(np.max(mean_train), np.min(std_train)))
    elif not norm_mean_std:
        print("label norm disabled")
        mean_train = 0
        std_train = 1

    graphs = load_pyg_graphs(
        df,
        name=name,
        target=target,
        force_target=force_target,
        need_force=need_force,
        stress_target=stress_target,  # cchen ###
        need_stress=need_stress,  # cchen ###
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
        mean_train=mean_train,
        std_train=std_train,
        load_label_from_atom=load_label_from_atom,
    )

    data = PygStructureDataset(
        graphs,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
        neighbor_strategy=neighbor_strategy,
        need_force=need_force,
        need_stress=need_stress,  # cchen ###
    )
    return data, mean_train, std_train

def jdata_preprocess(d, classification_threshold, dataset, target, target_multiplication_factor):
    dat = []
    if classification_threshold is not None:
        print(
            "Using ",
            classification_threshold,
            " for classifying ",
            target,
            " data.",
        )
        print("Converting target data into 1 and 0.")

    if dataset == "qm9_dgl" and target == "all":
        print("Making all qm9_dgl")
        tmp = []
        for ii in d:
            ii["all"] = [
                ii["mu"],
                ii["alpha"],
                ii["homo"],
                ii["lumo"],
                ii["gap"],
                ii["r2"],
                ii["zpve"],
                ii["U0"],
                ii["U"],
                ii["H"],
                ii["G"],
                ii["Cv"],
            ]
            tmp.append(ii)
        print("Made all qm9_dgl")
        d = tmp
    for i in d:
        if isinstance(i[target], list):  # multioutput target
            dat.append(i)

        elif (
            i[target] is not None
            and i[target] != "na"
            and not math.isnan(i[target])
        ):
            if target_multiplication_factor is not None:
                i[target] = i[target] * target_multiplication_factor
            if classification_threshold is not None:
                if i[target] <= classification_threshold:
                    i[target] = 0
                elif i[target] > classification_threshold:
                    i[target] = 1
                else:
                    raise ValueError(
                        "Check classification data type.",
                        i[target],
                        type(i[target]),
                    )
            dat.append(i)
    return dat

def get_train_val_loaders(
    dataset: str = "dft_3d",
    dataset_array=[],
    target: str = "",
    force_target: str = "",
    need_force: bool = False,
    stress_target: str = "",  # cchen ###
    need_stress: bool = False,  # cchen ###
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_features=1,
    output_dir=None,
    matrix_input=False,
    pyg_input=False,
    use_lattice=False,
    use_angle=False,
    use_save=True,
    direct_dset_input=None,
    norm_mean_std: bool = True,
    load_label_from_atom: bool = False,
):
    """Help function to set up JARVIS train and val dataloaders."""
    # data loading
    mean_train=None
    std_train=None
    assert (matrix_input and pyg_input) == False

    train_sample = filename + "_train.data"
    val_sample = filename + "_val.data"
    test_sample = filename + "_test.data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if (
        os.path.exists(os.path.join(output_dir, train_sample))
        and os.path.exists(os.path.join(output_dir, val_sample))
        and os.path.exists(os.path.join(output_dir, test_sample))
        and save_dataloader
    ):
        print("Loading from saved file...")
        print("Make sure all the DataLoader params are same.")
        print("This module is made for debugging only.")
        train_loader = torch.load(os.path.join(output_dir, train_sample), weights_only=False)
        val_loader = torch.load(os.path.join(output_dir, val_sample), weights_only=False)
        test_loader = torch.load(os.path.join(output_dir, test_sample), weights_only=False)
        if train_loader.pin_memory != pin_memory:
            train_loader.pin_memory = pin_memory
        if test_loader.pin_memory != pin_memory:
            test_loader.pin_memory = pin_memory
        if val_loader.pin_memory != pin_memory:
            val_loader.pin_memory = pin_memory
        if train_loader.num_workers != workers:
            train_loader.num_workers = workers
        if test_loader.num_workers != workers:
            test_loader.num_workers = workers
        if val_loader.num_workers != workers:
            val_loader.num_workers = workers
        print("train", len(train_loader.dataset))
        print("val", len(val_loader.dataset))
        print("test", len(test_loader.dataset))

    if not dataset_array and direct_dset_input is None:
        d = jdata(dataset)
    elif dataset_array:
        d = dataset_array
    elif direct_dset_input is not None:
        # Load dataset from direct input path
        d = []
    dat = jdata_preprocess(d, classification_threshold, dataset, target, target_multiplication_factor)

    if direct_dset_input is not None:
        """
        Prepare bulk and shear megnet dataset from the Matformer paper if needed
        https://github.com/YKQ98/Matformer/tree/569a7e9331b2acacc184fab38f5f6085e46a9881
        urls = [
            'https://figshare.com/ndownloader/files/40258705',
            'https://figshare.com/ndownloader/files/40258675',
            'https://figshare.com/ndownloader/files/40258666',
            'https://figshare.com/ndownloader/files/40258681',
            'https://figshare.com/ndownloader/files/40258684',
            'https://figshare.com/ndownloader/files/40258678'
        """
        if type(direct_dset_input) == str:
            if direct_dset_input == 'bulk':
                print('using mp bulk dataset')
                with open('./datasets/bulk_megnet_train.pkl', 'rb') as f:
                    dataset_train = pk.load(f)
                with open('./datasets/bulk_megnet_val.pkl', 'rb') as f:
                    dataset_val = pk.load(f)
                with open('./datasets/bulk_megnet_test.pkl', 'rb') as f:
                    dataset_test = pk.load(f)
                print('train {}/val {}/test {} samples'.format(
                    len(dataset_train), len(dataset_val), len(dataset_test)))

            elif direct_dset_input == 'shear':
                print('using mp shear dataset')
                with open('./datasets/shear_megnet_train.pkl', 'rb') as f:
                    dataset_train = pk.load(f)
                with open('./datasets/shear_megnet_val.pkl', 'rb') as f:
                    dataset_val = pk.load(f)
                with open('./datasets/shear_megnet_test.pkl', 'rb') as f:
                    dataset_test = pk.load(f)
                print('train {}/val {}/test {} samples'.format(
                    len(dataset_train), len(dataset_val), len(dataset_test)))
            elif os.path.isfile(direct_dset_input):
                print('load dataset from xyz file')
                dat = aseread(direct_dset_input, format='extxyz', index=':')
            else:
                print('load dataset from xyz files')
                if (not os.path.exists(direct_dset_input + '_train.xyz') or
                    not os.path.exists(direct_dset_input + '_test.xyz')
                ):
                    print('please ensure ' + direct_dset_input + '_train.xyz, _test.xyz, _valid.xyz files exist')
                    raise ValueError('file not exist')
                dataset_train = aseread(direct_dset_input + '_train.xyz', format='extxyz', index=':')
                dataset_test = aseread(direct_dset_input + '_test.xyz', format='extxyz', index=':')
                if os.path.exists(direct_dset_input + '_valid.xyz'):
                    dataset_val = aseread(direct_dset_input + '_valid.xyz', format='extxyz', index=':')
                else:
                    print('randomly select {} of samples from training set as validation set'.format(val_ratio))
                    n_val = int(len(dataset_train) * val_ratio)
                    split_index = [i for i in range(len(dataset_train))]
                    random.shuffle(split_index)
                    dataset_val = [dataset_train[i] for i in split_index[:n_val]]
                    dataset_train = [dataset_train[i] for i in split_index[n_val:]]
                print('train {}/val {}/test {} samples'.format(
                    len(dataset_train), len(dataset_val), len(dataset_test)))
        elif isinstance(direct_dset_input, list) or isinstance(direct_dset_input, tuple):
            if len(direct_dset_input) == 3:
                dataset_train = direct_dset_input[0]
                dataset_val = direct_dset_input[1]
                dataset_test = direct_dset_input[2]
            else:
                # neglect the situation that there are only 3 samples~
                dat = direct_dset_input
        else:
            raise ValueError('Unknown direct_dset_input')

    if len(dat) > 0:
        print("Splitting datasets")
        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dat),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_test=n_test,
            n_val=n_val,
            keep_data_order=keep_data_order,
        )
        ids_train_val_test = {}
        if len(id_tag) > 0:
            ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
            ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
            ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        else:
            ids_train_val_test["id_train"] = [str(i) for i in id_train]
            ids_train_val_test["id_val"] = [str(i) for i in id_val]
            ids_train_val_test["id_test"] = [str(i) for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(output_dir, "ids_train_val_test.json"),
        )
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]

    if (
        os.path.exists(os.path.join(output_dir, train_sample))
        and os.path.exists(os.path.join(output_dir, val_sample))
        and os.path.exists(os.path.join(output_dir, test_sample))
        and save_dataloader
    ):
        if norm_mean_std:
            _, mean_train, std_train = get_pyg_dataset(
                dataset=dataset_train,
                id_tag=id_tag,
                atom_features=atom_features,
                target=target,
                force_target=force_target,
                need_force=need_force,
                stress_target=stress_target,  # cchen ###
                need_stress=need_stress,  # cchen ###
                neighbor_strategy=neighbor_strategy,
                use_canonize=use_canonize,
                name=dataset,
                line_graph=line_graph,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                classification=classification_threshold is not None,
                output_dir=output_dir,
                tmp_name="train_data",
                use_lattice=use_lattice,
                use_angle=use_angle,
                use_save=False,
                only_mean_std=True,
                load_label_from_atom=load_label_from_atom,
            )
        else:
            mean_train = 0
            std_train = 1
        return (
            train_loader,
            val_loader,
            test_loader,
            train_loader.dataset.prepare_batch,
            mean_train,
            std_train,
        )

    train_data, mean_train, std_train = get_pyg_dataset(
        dataset=dataset_train,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        force_target=force_target,
        need_force=need_force,
        stress_target=stress_target,  # cchen ###
        need_stress=need_stress,  # cchen ###
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="train_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        norm_mean_std=norm_mean_std,
        load_label_from_atom=load_label_from_atom,
    )
    val_data,_,_ = get_pyg_dataset(
        dataset=dataset_val,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        force_target=force_target,
        need_force=need_force,
        stress_target=stress_target,  # cchen ###
        need_stress=need_stress,  # cchen ###
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="val_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        mean_train=mean_train,
        std_train=std_train,
        norm_mean_std=norm_mean_std,
        load_label_from_atom=load_label_from_atom,
    )
    test_data,_,_ = get_pyg_dataset(
        dataset=dataset_test,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        force_target=force_target,
        need_force=need_force,
        stress_target=stress_target,  # cchen ###
        need_stress=need_stress,  # cchen ###
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="test_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        mean_train=mean_train,
        std_train=std_train,
        norm_mean_std=norm_mean_std,
        load_label_from_atom=load_label_from_atom,
    )

    if not line_graph:
        collate_fn = partial(train_data.collate, need_force=need_force, need_stress=need_stress)  # cchen ###
    if line_graph:
        collate_fn = partial(train_data.collate_line_graph, need_force=need_force, need_stress=need_stress)  # cchen ###

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if save_dataloader:
        torch.save(train_loader, os.path.join(output_dir, train_sample))
        torch.save(val_loader, os.path.join(output_dir, val_sample))
        torch.save(test_loader, os.path.join(output_dir, test_sample))

    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
        mean_train,
        std_train,
    )

