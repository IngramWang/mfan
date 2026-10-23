from functools import partial

from typing import Any, Dict, Union

import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from mfan.data import get_train_val_loaders
from mfan.config import TrainingConfig
from mfan.models.mfan import mfan, mfan_tg_wrapper, mfan_eval_wrapper
from mfan.models.comformer import iComformer, eComformer, comformer_tg_wrapper

from jarvis.db.jsonutils import dumpjson
import json
import pprint
import time
from tqdm import tqdm
import os


# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

class PolynomialLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, start_lr, end_lr, power=1, last_epoch=-1):
        self.max_iters = max_iters
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power
        self.last_iter = 0  # Custom attribute to keep track of last iteration count
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            (self.start_lr - self.end_lr) *
            ((1 - self.last_iter / self.max_iters) ** self.power) + self.end_lr
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.last_iter += 1  # Increment the last iteration count
        return super().step(epoch)

class s2ef_loss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, eloss=None, floss=None, sloss=None):  # cchen ###
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # cchen: stress weight ###
        if eloss is not None:
            self.energy_loss = eloss
        else:
            self.energy_loss = nn.MSELoss()
        if floss is not None:
            self.force_loss = floss
        else:
            self.force_loss = nn.MSELoss()
        # cchen: stress loss ###
        if sloss is not None:
            self.stress_loss = sloss
        else:
            self.stress_loss = nn.MSELoss()
        ###

    def forward(self, outputs, targets):
        loss_graph = self.energy_loss(
            outputs[0],
            targets[0],
        )
        loss_node = self.force_loss(
            outputs[1],
            targets[1],
        )
        loss = self.alpha * loss_graph + self.beta * loss_node
        # cchen: stress loss term ###
        if self.gamma > 0 and len(outputs) > 2 and outputs[2] is not None:
            loss_stress = self.stress_loss(
                outputs[2].view(-1),
                targets[2].view(-1),
            )
            loss = loss + self.gamma * loss_stress
        ###
        return loss

def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.element_size() * parameter.nelement()
    for parameter in model.buffers():
        total_params += parameter.element_size() * parameter.nelement()
    total_params = total_params / 1024 / 1024
    print(f"Total size: {total_params}")

    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            n = param.numel()
            total_params += n
    print("Total trainable parameter number", total_params)
    return total_params

def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scaler_and_pca(output):
    """Use a standard scaler and PCA for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    return y_pred, y


# Backward-compatible name for external callers.
make_standard_scalar_and_pca = make_standard_scaler_and_pca


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    return y_pred, y


def setup_group_paras(model, default_decay, edge_coeff=10.0):
    """Omit weight decay from bias and batchnorm params."""
    edge_decay = edge_coeff * default_decay
    decay, no_decay, edges = [], [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        elif "edge" in name:
            edges.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": default_decay},
        {"params": edges, "weight_decay": edge_decay},
        {"params": no_decay, "weight_decay": 0},
    ]
    return groups

def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    for group in params:
        if "lr_scale" in group:
            group["lr"] = config.learning_rate * group.pop("lr_scale")
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer

def train_main(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    train_val_test_loaders=[],
    test_only=False,
    use_save=True,
    direct_dset_input=None,
    time_stamp="",
):
    """
    `config` should conform to matformer.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
            print('error in converting to training config!')

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    print("config:")
    tmp = config.dict()
    with open(os.path.join(config.output_dir, "config{}.json".format(time_stamp)), "w") as f:
        f.write(json.dumps(tmp, indent=4))
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp)
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = True
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
            mean_train,
            std_train,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            force_target=config.force_target,
            need_force=config.need_force,
            stress_target=config.stress_target,  # cchen ###
            need_stress=config.need_stress,  # cchen ###
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            matrix_input=config.matrix_input,
            pyg_input=config.pyg_input,
            use_lattice=config.use_lattice,
            use_angle=config.use_angle,
            use_save=use_save,
            direct_dset_input=direct_dset_input,
            norm_mean_std=config.norm_mean_std,
            load_label_from_atom=config.load_label_from_atom,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    _model = {
        "mfan": mfan,
        "mfan_tg": mfan_tg_wrapper,
        "iComformer": iComformer,
        "iComformer_tg": comformer_tg_wrapper,
        "eComformer": eComformer,
        "eComformer_tg": comformer_tg_wrapper,
    }
    if std_train is None:
        std_train = 1.0
        print('std train is none!')
    print('std train:', std_train)
    if model is None:
        model_name = config.model.name
        if config.neighbor_strategy == 'torch':
            model_name += "_tg"
        net = _model.get(model_name)(config.model)
    else:
        net = model

    net.to(device)
    if config.distributed:
        import torch.distributed as dist

        def setup(rank, world_size):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

        def cleanup():
            dist.destroy_process_group()

        setup(2, 2)
        net = torch.nn.parallel.DistributedDataParallel(
            net
        )
    params = setup_group_paras(net, config.weight_decay)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            final_div_factor=config.lr_final_div,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100000,
            gamma=0.96,
        )
    elif config.scheduler == "polynomial":
        steps_per_epoch = len(train_loader)
        total_iter = steps_per_epoch * config.epochs
        scheduler = PolynomialLRDecay(optimizer, max_iters=total_iter, start_lr=0.0005, end_lr=0.00001, power=1)

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "smoothl1_01": nn.SmoothL1Loss(beta=0.01),
        "smoothl1_001": nn.SmoothL1Loss(beta=0.001),
    }
    if config.need_force:
        criterion = s2ef_loss(
            alpha=config.energy_weight,
            beta=config.force_weight,
            gamma=config.stress_weight if config.need_stress else 0.0,  # cchen ###
            eloss=criteria[config.criterion],
            floss=criteria[config.force_criterion],
            sloss=criteria[config.stress_criterion] if config.need_stress else None,  # cchen ###
        )
    else:
        criterion = criteria[config.criterion]
    # set up training engine and evaluators
    metrics = {}

    if config.need_force:
        metrics["mae"] = MeanAbsoluteError(output_transform=lambda x: (x[0][0], x[1][0])) * std_train
        metrics["loss"] = Loss(criteria[config.criterion], output_transform=lambda x: (x[0][0], x[1][0]))
        metrics["score"] = -1.0 * (config.energy_weight * Loss(criteria[config.criterion],
                                                               output_transform=lambda x: (x[0][0], x[1][0])) +
                                   config.force_weight * Loss(criteria[config.force_criterion],
                                                              output_transform=lambda x: (x[0][1].view(-1), x[1][1].view(-1)))) * std_train
        metrics["force_mae"] = MeanAbsoluteError(output_transform=lambda x: (x[0][1].view(-1), x[1][1].view(-1))) * std_train
        metrics["force_loss"] = Loss(criteria[config.force_criterion], output_transform=lambda x: (x[0][1].view(-1), x[1][1].view(-1)))
        # cchen: stress metrics ###
        if config.need_stress:
            metrics["stress_mae"] = MeanAbsoluteError(
                output_transform=lambda x: (x[0][2].view(-1), x[1][2].view(-1))
            ) * std_train
            metrics["stress_loss"] = Loss(
                criteria[config.stress_criterion],
                output_transform=lambda x: (x[0][2].view(-1), x[1][2].view(-1))
            )
            # update score to include stress loss
            metrics["score"] = -1.0 * (
                config.energy_weight * Loss(criteria[config.criterion],
                                            output_transform=lambda x: (x[0][0], x[1][0])) +
                config.force_weight * Loss(criteria[config.force_criterion],
                                           output_transform=lambda x: (x[0][1].view(-1), x[1][1].view(-1))) +
                config.stress_weight * Loss(criteria[config.stress_criterion],
                                            output_transform=lambda x: (x[0][2].view(-1), x[1][2].view(-1)))
            ) * std_train
        ###
    else:
        metrics["loss"] = Loss(criterion)
        metrics["mae"] = MeanAbsoluteError() * std_train
        metrics["score"] = -1.0 * Loss(criterion) * std_train
    # use create_custom_evaluator and create_custom_trainer to skip batches that GPU-memory overflow
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
    )
    evaluator = create_supervised_evaluator(
        mfan_eval_wrapper(net),
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    test_evaluator = create_supervised_evaluator(
        mfan_eval_wrapper(net),
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    train_evaluator = create_supervised_evaluator(
        mfan_eval_wrapper(net),
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    if test_only:
        checkpoint_tmp = torch.load(config.model_path, weights_only=False)
        to_load = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_tmp)
        net.eval()
        targets = []
        forces = []
        predictions = []
        predictions_force = []
        stresses = []  # cchen ###
        predictions_stress = []  # cchen ###
        t1 = time.time()
        with torch.set_grad_enabled(config.need_force or config.need_stress):  # cchen ###
            for dat in test_loader:
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                if not config.need_force:
                    target_array = target.detach().cpu().view(-1).numpy()
                    pred_array = out_data.detach().cpu().view(-1).numpy()
                    targets.append(target_array)
                    predictions.append(pred_array)
                else:
                    # cchen: handle 3-tuple (energy, forces, stress) ###
                    target_list = [
                        x.detach().cpu().view(-1).numpy() for x in target
                    ]
                    pred_list = [
                        x.detach().cpu().view(-1).numpy() for x in out_data
                    ]
                    ###
                    targets.append(target_list[0])
                    forces.append(target_list[1])
                    predictions.append(pred_list[0])
                    predictions_force.append(pred_list[1])
                    # cchen: collect stress ###
                    if config.need_stress and len(target_list) > 2:
                        stresses.append(target_list[2])
                        predictions_stress.append(pred_list[2])
                    ###
        t2 = time.time()
        from sklearn.metrics import mean_absolute_error
        targets = np.concatenate(targets, axis=0) * std_train
        predictions = np.concatenate(predictions, axis=0) * std_train
        result = {
            "final_test": mean_absolute_error(targets, predictions),
        }
        print("Test MAE:", result["final_test"])
        if config.need_force:
            forces = np.concatenate(forces, axis=0) * std_train
            predictions_force = np.concatenate(predictions_force, axis=0) * std_train
            result["final_test_force"] = mean_absolute_error(
                forces, predictions_force
            )
            print("Test Force MAE:", result["final_test_force"])
        # cchen: report stress MAE ###
        if config.need_stress and len(stresses) > 0:
            stresses = np.concatenate(stresses, axis=0) * std_train
            predictions_stress = np.concatenate(
                predictions_stress, axis=0
            ) * std_train
            result["final_test_stress"] = mean_absolute_error(
                stresses, predictions_stress
            )
            print("Test Stress MAE:", result["final_test_stress"])
        ###
        print("Total test time:", t2-t1)
        return result

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
    n_paras = count_parameters(net)

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
        # evaluate save
        to_save = {"model": net}
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=3,
            filename_prefix='best' + time_stamp,
            score_name="score",
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    history = {
        "epoch": 0,
        "n_paras": n_paras,
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
        "test": {m: [] for m in metrics.keys()},
        "time": {"train":[], "valid":[]},
    }

    if config.store_outputs:
        pass

    start_time = time.time()
    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        nonlocal start_time

        """Print training and validation metrics to console."""
        history["epoch"] += 1
        epoch_num = history["epoch"]
        if config.valid_per_epoch or epoch_num % 20 == 0:
            t = time.time()
            history["time"]["train"].append(t - start_time)
            start_time = t
            evaluator.run(val_loader)

            vmetrics = evaluator.state.metrics
            for metric in metrics.keys():
                vm = vmetrics[metric]
                if metric == "roccurve":
                    vm = [k.tolist() for k in vm]
                if isinstance(vm, torch.Tensor):
                    vm = vm.cpu().numpy().tolist()

                history["validation"][metric].append(vm)

        if epoch_num % 20 == 0:
            train_evaluator.run(train_loader)
            trmetrics = train_evaluator.state.metrics
            for metric in metrics.keys():
                tm = trmetrics[metric]
                if metric == "roccurve":
                    tm = [k.tolist() for k in tm]
                if isinstance(tm, torch.Tensor):
                    tm = tm.cpu().numpy().tolist()

                history["train"][metric].append(tm)
        else:
            trmetrics = {}
            trmetrics['mae'] = -1

        if config.store_outputs:
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_time.json"),
                data=history["time"],
            )
        if config.progress:
            pbar = ProgressBar()
            if epoch_num % 20 == 0:
                if not classification:
                    pbar.log_message(f"Epoch {epoch_num} Val_MAE: {vmetrics['mae']:.4f}")
                    pbar.log_message(f"Epoch {epoch_num} Train_MAE: {trmetrics['mae']:.4f}")
                else:
                    pbar.log_message(f"Epoch {epoch_num} Val_MAE: {vmetrics['mae']:.4f}")
                    pbar.log_message(f"Epoch {epoch_num} Val ROC AUC: {vmetrics['rocauc']:.4f}")
                    pbar.log_message(f"Epoch {epoch_num} Train ROC AUC: {trmetrics['rocauc']:.4f}")
            else:
                if not classification:
                    if config.valid_per_epoch:
                        pbar.log_message(f"Epoch {epoch_num} Val_MAE: {vmetrics['mae']:.4f}")
                else:
                    if config.valid_per_epoch:
                        pbar.log_message(f"Epoch {epoch_num} Val ROC AUC: {vmetrics['rocauc']:.4f}")

    if config.n_early_stopping is not None:
        if classification:
            my_metrics = "accuracy"
        else:
            my_metrics = "score"

        def default_score_fn(engine):
            score = engine.state.metrics[my_metrics]
            return score

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=default_score_fn,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)
    # optionally log results to tensorboard
    if config.log_tensorboard:

        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    trainer.run(train_loader, max_epochs=config.epochs)

    # Evaluate the test set only after training has finished (including early stopping).
    test_evaluator.run(test_loader)
    temetrics = test_evaluator.state.metrics
    for metric in metrics.keys():
        tm = temetrics[metric]
        if metric == "roccurve":
            tm = [k.tolist() for k in tm]
        if isinstance(tm, torch.Tensor):
            tm = tm.cpu().numpy().tolist()
        history["test"][metric].append(tm)

    history["final_test"] = history["test"]["mae"][-1]
    if config.need_force:
        history["final_test_force"] = history["test"]["force_mae"][-1]
    if config.need_stress:
        history["final_test_stress"] = history["test"]["stress_mae"][-1]
    if config.store_outputs:
        dumpjson(
            filename=os.path.join(config.output_dir, "history_test.json"),
            data=history["test"],
        )
    if config.progress:
        ProgressBar().log_message(f"Final Test_MAE: {temetrics['mae']:.4f}")

    if config.log_tensorboard:
        test_loss = temetrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()
    if config.write_predictions and classification:
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        line_num = 0
        with torch.set_grad_enabled(config.need_force):
            for dat in test_loader:
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                target = int(target.cpu().numpy().flatten().tolist()[0])

                f.write("%d: %d, %d\n" % (line_num, target, top_class))
                targets.append(target)
                predictions.append(
                    top_class.cpu().numpy().flatten().tolist()[0]
                )
                line_num += 1
        f.close()
        from sklearn.metrics import roc_auc_score

        print(
            "Test ROCAUC:",
            roc_auc_score(np.array(targets), np.array(predictions)),
        )

    if (
        config.write_predictions
        and not classification
        and config.model.output_features > 1
    ):
        net.eval()
        mem = []
        line_num = 0
        with torch.set_grad_enabled(config.need_force):
            for dat in test_loader:
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(open("sc.pkl", "rb"))
                    out_data = list(
                        sc.transform(np.array(out_data).reshape(1, -1))[0]
                    )  # [0][0]
                target = target.cpu().numpy().flatten().tolist()
                info = {}
                info["id"] = line_num
                info["target"] = target
                info["predictions"] = out_data
                mem.append(info)
                line_num += 1
        dumpjson(
            filename=os.path.join(
                config.output_dir, "multi_out_predictions.json"
            ),
            data=mem,
        )
    if (
        config.write_predictions
        and not classification
        and config.model.output_features == 1
    ):
        net.eval()
        targets = []
        predictions = []
        forces = []
        predictions_force = []
        stresses = []  # cchen ###
        predictions_stress = []  # cchen ###
        mem = []
        t1 = time.time()
        line_num = 0
        with torch.set_grad_enabled(config.need_force or config.need_stress):  # cchen ###
            for dat in tqdm(test_loader):
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                info = {}
                info["id"] = line_num
                if not config.need_force:
                    out_data = out_data.detach().cpu().numpy().tolist()
                    target = target.cpu().numpy().flatten().tolist()
                    targets.append(target)
                    predictions.append(out_data)
                    info["predictions"] = out_data
                    info["target"] = target
                else:
                    # cchen: handle 3-tuple (energy, forces, stress) ###
                    target_list = [x.cpu().view(-1).numpy() for x in target]
                    pred_list = [x.detach().cpu().view(-1).numpy() for x in out_data]
                    ###
                    n_atom = g.natoms.cpu().numpy()
                    #use energy per atom as energy result
                    targets.append(target_list[0].tolist())
                    forces.append(target_list[1].tolist())
                    predictions.append(pred_list[0].tolist())
                    predictions_force.append(pred_list[1].tolist())
                    info["predictions"] = pred_list[0].tolist()
                    info["n_atom"] = n_atom.tolist()
                    info["target"] = target_list[0].tolist()
                    # cchen: collect stress ###
                    if config.need_stress and len(target_list) > 2:
                        stresses.append(target_list[2].tolist())
                        predictions_stress.append(pred_list[2].tolist())
                        info["stress_pred"] = pred_list[2].tolist()
                        info["stress_target"] = target_list[2].tolist()
                    ###
                # only dump target to save space
                mem.append(info)
                line_num += 1
        t2 = time.time()
        dumpjson(
            filename=os.path.join(
                config.output_dir, "dump_predictions.json"
            ),
            data=mem,
        )
        from sklearn.metrics import mean_absolute_error
        targets = np.array(targets) * std_train
        predictions = np.array(predictions) * std_train
        print("Test MAE:", mean_absolute_error(targets, predictions))
        history['final_test'] = mean_absolute_error(targets, predictions)
        if config.need_force:
            forces = np.concatenate(forces, axis=0) * std_train
            predictions_force = np.concatenate(predictions_force, axis=0) * std_train
            print("Test Force MAE:", mean_absolute_error(forces, predictions_force))
            history['final_test_force'] = mean_absolute_error(forces, predictions_force)
        # cchen: report stress MAE ###
        if config.need_stress and len(stresses) > 0:
            stresses = np.concatenate(stresses, axis=0) * std_train
            predictions_stress = np.concatenate(predictions_stress, axis=0) * std_train
            print("Test Stress MAE:", mean_absolute_error(stresses, predictions_stress))
            history['final_test_stress'] = mean_absolute_error(stresses, predictions_stress)
        ###

    return history

