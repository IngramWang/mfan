"""Helper function for high-throughput GNN trainings."""
"""Implementation based on the template of Matformer."""
import matplotlib.pyplot as plt

import time
from mfan.train import train_main

plt.switch_backend("agg")

def log_file(cmts, task_name, result, filename='log.txt'):
    with open(filename, 'a') as f:
        f.write(time.strftime('%d%b%Y %H:%M:%S',time.localtime()) + '\n')
        f.write(f"Comments: {cmts}\n")
        f.write(f"Task Name: {task_name}\n")
        if 'n_paras' in result:
            f.write('model size {} '.format(result['n_paras']))
        if 'train' in result and len(result['train']['loss']) > 0:
            f.write('train loss {:.6f} '.format(result['train']['loss'][-1]))
            f.write('train MAE {:.6f} '.format(result['train']['mae'][-1]))
        if 'validation' in result and len(result['validation']['loss']) > 0:
            f.write('validation loss {:.6f} '.format(result['validation']['loss'][-1]))
            f.write('validation MAE {:.6f} '.format(result['validation']['mae'][-1]))
        if 'final_test' in result:
            f.write('test MAE {:.6f} '.format(result['final_test']))
        if 'final_test_force' in result:
            f.write('test force MAE {:.6f} '.format(result['final_test_force']))
        if 'final_test_stress' in result:
            f.write('test stress MAE {:.6f} '.format(result['final_test_stress']))
        f.write('\n\n')

def train_prop_model(
    prop="",
    force_prop="",
    stress_prop="",
    dataset="dft_3d",
    write_predictions=True,
    name="mfan",
    save_dataloader=False,
    train_ratio=None,
    classification_threshold=None,
    val_ratio=None,
    test_ratio=None,
    learning_rate=0.001,
    lr_final_div=None,
    batch_size=None,
    scheduler=None,
    n_epochs=None,
    id_tag=None,
    num_workers=None,
    weight_decay=None,
    criterion=None,
    force_criterion=None,
    stress_criterion=None,
    edge_input_features=None,
    triplet_input_features=None,
    embedding_features=None,
    hidden_features=None,
    output_features=None,
    random_seed=None,
    n_early_stopping=None,
    cutoff=None,
    max_neighbors=None,
    matrix_input=False,
    pyg_input=False,
    use_lattice=False,
    use_angle=False,
    output_dir=None,
    neighbor_strategy="k-nearest",
    test_only=False,
    use_save=True,
    direct_dset_input=None,
    load_label_from_atom=False,
    need_force=False,
    need_stress=False,
    energy_weight=None,
    force_weight=None,
    stress_weight=None,
    quick_setup=None,
    file_name=None,
    atom_features="cgcnn",
    config_only=False,
    comments='',
    task_name='task',
):
    """Train models for a dataset and a property."""
    if scheduler is None:
        scheduler = "onecycle"
    if batch_size is None:
        batch_size = 64
    if n_epochs is None:
        n_epochs = 500
    if num_workers is None:
        num_workers = 4
    config = {
        "dataset": dataset,
        "target": prop,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "weight_decay": 1e-05,
        "learning_rate": learning_rate,
        "optimizer": "adamw",
        "scheduler": scheduler,
        "save_dataloader": save_dataloader,
        "pin_memory": False,
        "write_predictions": write_predictions,
        "num_workers": num_workers,
        "classification_threshold": classification_threshold,
        "atom_features": atom_features,
        "load_label_from_atom": load_label_from_atom,
        "need_force": need_force,
        "need_stress": need_stress,
        "model": {
            "name": name,
        },
    }
    if len(force_prop) > 0:
        config["force_target"] = force_prop
    if len(stress_prop) > 0:
        config["stress_target"] = stress_prop
    if n_early_stopping is not None:
        config["n_early_stopping"] = n_early_stopping
    if lr_final_div is not None:
        config["lr_final_div"] = lr_final_div
    if cutoff is not None:
        config["cutoff"] = cutoff
    if criterion is not None:
        if criterion == "smoothl1":
            criterion = "smoothl1_01"
        config["criterion"] = criterion
    if force_criterion is not None:
        if force_criterion == "smoothl1":
            force_criterion = "smoothl1_01"
        config["force_criterion"] = force_criterion
    elif criterion is not None:
            config["force_criterion"] = criterion
    if stress_criterion is not None:
        if stress_criterion == "smoothl1":
            stress_criterion = "smoothl1_01"
        config["stress_criterion"] = stress_criterion
    elif criterion is not None:
        config["stress_criterion"] = criterion
    if max_neighbors is not None:
        config["max_neighbors"] = max_neighbors
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    if edge_input_features is not None:
        config["model"]["edge_input_features"] = edge_input_features
    if hidden_features is not None:
        config["model"]["hidden_features"] = hidden_features
    if embedding_features is not None:
        config["model"]["embedding_features"] = embedding_features
    if output_features is not None:
        config["model"]["output_features"] = output_features
    if quick_setup is not None:
        config["model"]["quick_mode_setup"] = quick_setup
    if random_seed is not None:
        config["random_seed"] = random_seed
    if file_name is not None:
        config["filename"] = file_name
    if force_weight is not None:
        config["force_weight"] = force_weight
    if stress_weight is not None:
        config["stress_weight"] = stress_weight
    if energy_weight is not None:
        config["energy_weight"] = energy_weight
    config["matrix_input"] = matrix_input
    config["pyg_input"] = pyg_input
    config["use_lattice"] = use_lattice
    config["use_angle"] = use_angle
    config["neighbor_strategy"] = neighbor_strategy
    if output_dir is not None:
        config["output_dir"] = output_dir

    if id_tag is not None:
        config["id_tag"] = id_tag
    if train_ratio is not None:
        config["train_ratio"] = train_ratio
        if val_ratio is None:
            raise ValueError("Enter val_ratio.")

        if test_ratio is None:
            raise ValueError("Enter test_ratio.")
        config["val_ratio"] = val_ratio
        config["test_ratio"] = test_ratio
    if dataset == "jv_3d":
        config["num_workers"] = 4
        config["pin_memory"] = False

    if dataset == "mp_3d_2020":
        config["id_tag"] = "id"
        config["num_workers"] = 0
    if dataset == "megnet2":
        config["id_tag"] = "id"
        config["num_workers"] = 0
    if dataset == "megnet":
        config["id_tag"] = "id"
        if prop == "e_form" or prop == "gap pbe":
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4239
            config["num_workers"] = 8
        else:
            config["n_train"] = 4664
            config["n_val"] = 393
            config["n_test"] = 393

    if dataset == 'mpf':
        config["id_tag"] = "id"
        config["n_train"] = 169516
        config["n_val"] = 9417
        config["n_test"] = 9417

    time_stamp = time.strftime('_%Y%m%d_%H%M%S',time.localtime())
    config["task_name"] = task_name + time_stamp

    if config_only:
        return config

    if test_only:
        t1 = time.time()
        result = train_main(config, test_only=test_only, use_save=use_save, direct_dset_input=direct_dset_input, time_stamp=time_stamp)
        t2 = time.time()
        print("test results=", result)
        print("Total time:", t2 - t1)
        log_file(comments, config["task_name"], result)
    else:
        t1 = time.time()
        result = train_main(config, use_save=use_save, direct_dset_input=direct_dset_input, time_stamp=time_stamp)
        t2 = time.time()
        print("Total time:", t2 - t1)
        log_file(comments, config["task_name"], result)
