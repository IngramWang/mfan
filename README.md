# Adaptive Multi-Frame Learning for Expressive and Stable Atomic Representations

## Datasets

### Materials Project

For Materials Project tasks, we follow Matformer (Yan et al.) and use the same training, validation, and test splits. The datasets for formation energy (`e_form`) and band gap (`gap pbe`) are downloaded automatically through JARVIS on the first run.

For bulk and shear modulus prediction, download the datasets from [Figshare](https://figshare.com/projects/Bulk_and_shear_datasets/165430) and save the files under `datasets/`.

### JARVIS

JARVIS was introduced by Choudhary et al. We follow ALIGNN and use the same training, validation, and test splits. We evaluate our model on five crystal property prediction tasks: formation energy, OPT band gap, MBJ band gap, total energy, and energy above the convex hull (`ehull`). The datasets are downloaded automatically through JARVIS on the first run.

### CeO2/Pt

We construct a CeO2/Pt dataset using density functional theory (DFT) calculations to study the restructuring of Pt clusters on CeO2 surfaces. The dataset contains 6,178 configurations, primarily Pt species supported on CeO2 surfaces, along with a small number of bulk Ce and Pt structures.

Pt/CeO2 is an important heterogeneous catalytic system. Strong metal–support interactions produce electronic effects that play a key role in reactions such as methane activation. Accurate modeling is challenging because of the complex electronic structure of cerium and the dynamic aggregation and restructuring of Pt species on oxide surfaces.

The dataset is provided as training and test sets in an approximately 85%/15% split. During training, the data loader randomly reserves 5% of the provided training set for validation.

All DFT calculations are performed using the Vienna Ab Initio Simulation Package (VASP) with the PBE exchange–correlation functional. A Hubbard U correction of 5.5 eV is applied to the Ce 4f states to account for their localized nature.

## Environment and Installation

Set up an environment with the following dependencies before installing this library:
```text
Python 3.11
CUDA 12.8
PyTorch 2.8.0
torch_geometric 2.7.0
torch_scatter 2.1.2
torch_sparse 2025.8.30
```

For PyTorch 2.8.0 with CUDA 12.8, you can install `torch_scatter` using the matching PyG wheels:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Once the dependencies are installed, install this library in editable mode from the repository root:
```bash
pip install -e .
```

## Training

Run the training scripts from the repository root:

```bash
python scripts/train_jarvis.py # JARVIS
python scripts/train_mp.py     # Materials Project
python scripts/train_cept.py   # CeO2/Pt
```

The default JARVIS run predicts `formation_energy_peratom`. The default Materials Project run predicts `bulk modulus` and requires the downloaded bulk modulus dataset files. To predict a different property, select a target from `props` in the corresponding script. For Materials Project tasks, set `direct_dset_input='shear'` for shear modulus, or omit `direct_dset_input` for `e_form` and `gap pbe`.

Both crystal property training scripts run for 500 epochs and save their outputs under `jarvis/` and `mat_proj/`, respectively. The CeO2/Pt script trains on energies and forces for 300 epochs and saves its outputs under `cept/`.

## Citation and Acknowledgments

If you use this work, please cite:

Jun Wang, Yifan Zeng, Bo Han, Fengwang Li, Aoni Xu, Tongliang Liu. Adaptive Multi-Frame Learning for Expressive and Stable Atomic Representations. *Advances in Neural Information Processing Systems*, 2026.

This repository incorporates code from the following projects:

- [ComFormer](https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer)
- [MFA](https://github.com/divelab/MFA/tree/main)
- [AlphaNet](https://github.com/zmyybc/AlphaNet)
