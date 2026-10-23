import sys
import os
sys.path.append("../")
from mfan.train_props import train_prop_model
props = [
    "e_form",
    "gap pbe",
    "bulk modulus",
    "shear modulus",
]

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

train_prop_model(
    learning_rate=0.001,
    name="mfan",
    dataset="megnet",
    prop=props[1],
    pyg_input=True,
    n_epochs=500,
    max_neighbors=25,
    cutoff=6.0,
    batch_size=64,
    use_lattice=True,
    output_dir="mat_proj",
    use_angle=False,
    save_dataloader=False,
    neighbor_strategy='k-nearest',
    #direct_dset_input='bulk', # use bulk or shear for bulk modulus and shear modulus, respectively
    criterion='smoothl1',
    quick_setup='crystal',
    comments='mp test',
)
