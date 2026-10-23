import sys
import os
sys.path.append("../")
from mfan.train_props import train_prop_model
props = [
    "formation_energy_peratom",#1
    "optb88vdw_bandgap",#2
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",#4
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",#3
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",#5
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
]

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

train_prop_model(
    learning_rate=0.001,
    name="mfan",
    dataset="dft_3d_2021",
    prop=props[0],
    pyg_input=True,
    n_epochs=500,
    max_neighbors=25,
    cutoff=6.0,
    batch_size=64,
    use_lattice=True,
    output_dir="jarvis",
    use_angle=False,
    save_dataloader=False,
    neighbor_strategy='k-nearest',
    criterion='smoothl1',
    quick_setup='crystal',
    comments='cell/qr eform',
)
