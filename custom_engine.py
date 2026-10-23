# Ref: https://docs.pytorch.org/ignite/_modules/ignite/engine.html

import torch
from typing import Any, Callable, Optional, Sequence, Tuple, Union, Mapping
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from ignite.engine import Engine
from ignite.metrics import Metric
from ignite.utils import convert_tensor

def _prepare_batch(
    batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Prepare batch for training or evaluation: pass to a device with options."""
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )

def create_custom_trainer(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any, Any], torch.Tensor] | torch.nn.Module,
    device: str | torch.device | None = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
    amp_mode: str | None = None,
    scaler: bool | GradScaler = False,
    gradient_accumulation_steps: int = 1,
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x)) -> Engine:

    print('WARNING: using custom trainer')

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Any | tuple[torch.Tensor]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        try:
            output = model_fn(model, x)
            y_pred = model_transform(output)
            loss = loss_fn(y_pred, y)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if engine.state.iteration % gradient_accumulation_steps == 0:
                optimizer.step()
            return output_transform(x, y, y_pred, loss * gradient_accumulation_steps)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"WARNING: a batch is skipped during training due to out of memory (Iteration {engine.state.iteration})")
                return None
            else:
                raise e

    return Engine(update)

def create_custom_evaluator(
    model: torch.nn.Module,
    metrics: dict[str, Metric] | None = None,
    device: str | torch.device | None = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, y),
    amp_mode: str | None = None,
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    need_grad: bool = False,
) -> Engine:

    metrics = metrics or {}
    print('WARNING: using custom evaluator')
    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Any | tuple[torch.Tensor]:
        model.eval()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        try:
            with torch.set_grad_enabled(need_grad):
                output = model_fn(model, x)
                y_pred = model_transform(output)
                return output_transform(x, y, y_pred)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"WARNING: a batch is skipped during evaluating due to out of memory (Iteration {engine.state.iteration})")
                return output_transform(x, y, y)
            else:
                raise e

    evaluator = Engine(evaluate_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
