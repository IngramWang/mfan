import sys
import os
sys.path.append("../")
from mfan.train_props import train_prop_model

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

train_prop_model(learning_rate=0.0001, name="mfan", dataset="cept", need_force=True,
                 load_label_from_atom=True, pyg_input=True, n_epochs=300,
                 max_neighbors=500, energy_weight=1., force_weight=2., cutoff=6.0,
                 train_ratio=0.80, val_ratio=0.05, test_ratio=0.15,
                 batch_size=4, use_lattice=True, output_dir="cept", use_angle=False,
                 save_dataloader=False, weight_decay=1e-9, lr_final_div=1e8,
                 prop='REF_energy', force_prop='REF_forces',
                 neighbor_strategy='torch', 
                 criterion='smoothl1_001', force_criterion='smoothl1_01',
                 direct_dset_input='./datasets/CeO2_Pt',              
                 quick_setup='s2ef',
                 comments='cept test')
