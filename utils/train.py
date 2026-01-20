import torch
import os, shutil
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import to_pil_image
import random
import mlflow
import os
from tqdm import tqdm
import mlflow.pytorch
from datetime import datetime
from utils.logger_utils import get_temp_logger
from models.ode_solvers import euler_solver, create_samples, make_vf_cfg


def flow_matching_step(model, x1, loss_fn, device):
    B = x1.shape[0]
    x1 = x1.to(device)
    x0 = torch.randn_like(x1).to(device)
    t = torch.rand(B, 1, 1, 1).to(device)
    v_est = model((1 - t) * x0 + x1 * t, t)
    v_true = x1 - x0
    mse = loss_fn(v_est, v_true)
    return mse


def flow_matching_step_cfg(model, x1, y, p_drop, NULL_ID, loss_fn, device):
    B = x1.shape[0]
    x1 = x1.to(device)
    y = y.to(device)

    x0 = torch.randn_like(x1).to(device)
    t = torch.rand(B, 1, 1, 1).to(device)

    drop_mask = torch.rand_like(y.float()) < p_drop
    y_drop = y.clone()
    y_drop[drop_mask] = NULL_ID

    v_est = model((1 - t) * x0 + x1 * t, t, y_drop)
    v_true = x1 - x0
    mse = loss_fn(v_est, v_true)
    return mse


def create_pil_image(images: torch.Tensor, nrow: int = 8):
    images = images.detach().cpu()
    if images.min() < 0:
        images = (images + 1) / 2
    images = images.clamp(0, 1)

    grid = make_grid(images, nrow=nrow)
    # Create PIL image
    img = to_pil_image(grid)

    return img


# Needs refactoring
# def train_loop_uncond(
#     model,
#     dataloader: DataLoader,
#     num_epochs: int = 10,
#     lr: float = 1e-3,
#     log_every_step: int = 100,
#     log_every_epoch: int = 10,
#     sample_steps:int = 50,
#     run_name_prefix: str = 'FM-MNIST-Uncond',
#     device: str = 'cuda',

#     sample_grid_size = 8,
#     ode_solver = euler_solver,
#     ode_steps = 50):

#     mlflow.set_experiment("Flow Matching MNIST Unconditional")
#     logger, log_path = get_temp_logger("train_uncond")
#     run_name = run_name_prefix + datetime.now().strftime("-%Y-%m-%d_%H-%M-%S")
#     optim = torch.optim.AdamW(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()
#     images, _ = next(iter(dataloader))
#     BATCH_SIZE, *IMAGE_SHAPE = images.shape
#     IMAGE_SHAPE = tuple(IMAGE_SHAPE)
#     params = {
#             "lr":lr,
#             "epochs": num_epochs,
#             "samples_steps": sample_steps,
#             "model_params": sum(p.numel() for p in model.parameters()),
#             "batch_size": BATCH_SIZE,
#             "ode_steps": ode_steps,
#             "ode_solver": getattr(ode_solver, "__name__", str(ode_solver)),
#             "image_shape": IMAGE_SHAPE
#         }
#     with mlflow.start_run(run_name = run_name) as run:
#         logger.info("Starting unconditional training")
#         mlflow.log_params(params)
#         logger.info("Hyperparameters:\n" + "\n".join(f" {k}: {v}" for k, v in params.items()))
#         global_step = 0
#         for epoch in tqdm(range(num_epochs)):
#             model.train()
#             running_loss = 0.0
#             # Loop over dataset
#             for i,(x1,_) in enumerate(dataloader):
#                 optim.zero_grad()
#                 # Generate esstimated velociy fields and compute MSE
#                 mse = flow_matching_step(model,x1,loss_fn,device)
#                 mse.backward()
#                 optim.step()
#                 running_loss += mse.item()

#                 if global_step % log_every_step == 0:
#                     mlflow.log_metric("mse_step", mse.item(), step = global_step)
#                     logger.info(f"[epoch {epoch:03d} | step {global_step:06d}] mse={mse.item() :.6f}")
#                 global_step += 1
#             # Sample batch of images
#             if epoch % log_every_epoch == 0:
#                 mlflow.log_metric("mse_epoch", running_loss / len(dataloader), step = epoch)
#                 # Create callback for velocity field
#                 f = make_vf_uncond(model)
#                 logger.info(f"[epoch {epoch:03d} | step {global_step:06d}] Creating sample images")
#                 samples = create_samples(BATCH_SIZE, IMAGE_SHAPE, ode_solver, f, n_steps = ode_steps, seed = 0, device=device)
#                 img = create_pil_image(samples)
#                 logger.info(f"[epoch {epoch:03d} | step {global_step:06d}] Saving sample images")
#                 mlflow.log_image(img,key="train_generated_samples", step = epoch)


#         f = make_vf_uncond(model)
#         samples = create_samples(BATCH_SIZE, IMAGE_SHAPE, ode_solver, f, n_steps = ode_steps, return_all=True, seed = 0,device=device)
#         for i,x in enumerate(samples):
#             img = create_pil_image(x,nrow=sample_grid_size)
#             mlflow.log_image(img,key="final_generated_samples", step = i)
#         logger.info("Saving model artifact")
#         model_info = mlflow.pytorch.log_model(
#             model,
#             name = 'UNet'
#         )
#         mlflow.log_artifact(log_path, artifact_path="logs")
#         tmpdir = os.path.dirname(log_path)
#         shutil.rmtree(tmpdir, ignore_errors=True)
#     return model_info


def train_loop_uncond(
    model,
    train_loader,
    num_epochs: int,
    optim,
    device,
    val_loader=None,
    on_step=None,
    on_epoch=None,
):
    """
    on_step(global_step, train_mse_step, epoch)
    on_epoch(epoch, train_mse_epoch, val_mse_epoch)
        - val_mse_epoch is None if val_loader is None
    Returns:
        best_val_mse if val_loader is provided, else best_train_mse
    """
    loss_fn = nn.MSELoss()

    global_step = 0
    best_train = float("inf")
    best_val = float("inf")

    for epoch in range(num_epochs):
        # train
        model.train()
        running = 0.0

        for x1, _ in train_loader:
            optim.zero_grad(set_to_none=True)
            mse = flow_matching_step(model, x1, loss_fn, device)
            mse.backward()
            optim.step()

            mse_step = float(mse.item())
            running += mse_step

            if on_step is not None:
                on_step(global_step, mse_step, epoch)

            global_step += 1

        train_mse_epoch = running / len(train_loader)
        best_train = min(best_train, train_mse_epoch)

        # val
        val_mse_epoch = None
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            with torch.no_grad():
                for x1, _ in val_loader:
                    mse = flow_matching_step(model, x1, loss_fn, device)
                    v_running += float(mse.item())
            val_mse_epoch = v_running / len(val_loader)
            best_val = min(best_val, val_mse_epoch)

        if on_epoch is not None:
            on_epoch(epoch, train_mse_epoch, val_mse_epoch)

    return best_val if val_loader is not None else best_train


# TODO:
# def train_uncond_hpt(trial):
#     cfg = sample_cfg(trial)

#     mlflow.set_experiment("hpt/cond_unet")
#     with mlflow.start_run(run_name=f"trial_{trial.number:04d}"):
#         mlflow.log_params(cfg)
#         mlflow.set_tag("optuna_trial", trial.number)

#         def on_epoch(epoch, val_loss):
#             mlflow.log_metric("val_loss", float(val_loss), step=epoch)
#             trial.report(val_loss, step=epoch)
#             if trial.should_prune():
#                 raise optuna.TrialPruned()

#         best = train_loop_uncond(
#             model=build_model(cfg),
#             dataloader=train_loader,
#             num_epochs=cfg["hpt_epochs"],   # small budget
#             on_epoch=on_epoch,
#         )

#         mlflow.log_metric("best_val_loss", float(best))
#         return float(best)


# TODO: refactor
def train_loop_cfg(
    model,
    dataloader: DataLoader,
    NULL_ID,
    p_drop: float = 0.2,
    w=1,
    num_epochs: int = 10,
    lr: float = 1e-3,
    log_every_step: int = 1,
    log_every_epoch: int = 10,
    sample_steps: int = 50,
    experiment_name: str = "mnist-fm-cond-unet",
    run_name: str = None,
    device: str = "cuda",
    sample_grid_size=8,
    ode_solver=euler_solver,
    ode_steps=50,
    save_model=True,
):
    mlflow.set_experiment(experiment_name)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    images, labels = next(iter(dataloader))
    BATCH_SIZE, *IMAGE_SHAPE = images.shape
    IMAGE_SHAPE = tuple(IMAGE_SHAPE)
    input_example = {
        "x": torch.randn(1, *IMAGE_SHAPE).cpu().numpy(),
        "t": torch.zeros(1, 1, 1, 1).cpu().numpy(),
        "y": torch.tensor([0], dtype=labels.dtype).cpu().numpy(),
    }

    run_name = f"{run_name if run_name else experiment_name + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger, log_path = get_temp_logger("train_cfg")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "lr": lr,
                "epochs": num_epochs,
                "cfg_strength": w,
                "p_drop": p_drop,
                "NULL_ID": NULL_ID,
                "samples_steps": sample_steps,
                "model_params": sum(p.numel() for p in model.parameters()),
                "batch_size": BATCH_SIZE,
                "ode_steps": ode_steps,
                "ode_solver": getattr(ode_solver, "__name__", str(ode_solver)),
            }
        )
        conditional_sampling_grid_labels = (
            torch.tensor([i for i in range(NULL_ID)])
            .repeat(sample_grid_size, 1)
            .flatten()
        )
        global_step = 0
        for epoch in tqdm(range(num_epochs)):
            model.train()
            running_loss = 0.0
            for i, (x1, c) in enumerate(dataloader):
                optim.zero_grad()
                mse = flow_matching_step_cfg(
                    model, x1, c, p_drop, NULL_ID, loss_fn, device
                )
                mse.backward()
                optim.step()
                running_loss += mse.item()

                if global_step % log_every_step == 0:
                    mlflow.log_metric("mse_step", mse.item(), step=global_step)
                global_step += 1

            if epoch % log_every_epoch == 0:
                f = make_vf_cfg(model, conditional_sampling_grid_labels, w, num_epochs)
                mlflow.log_metric(
                    "mse_epoch", running_loss / len(dataloader), step=epoch
                )
                samples = create_samples(
                    NULL_ID * sample_grid_size,
                    IMAGE_SHAPE,
                    ode_solver,
                    f,
                    n_steps=ode_steps,
                    seed=0,
                    device=device,
                )
                img = create_pil_image(samples)
                mlflow.log_image(
                    img, artifact_file=f"train_grids/samples_epoch_{epoch:04d}.png"
                )

        f = make_vf_cfg(model, conditional_sampling_grid_labels, w, num_epochs)
        samples = create_samples(
            NULL_ID * sample_grid_size,
            IMAGE_SHAPE,
            ode_solver,
            f,
            n_steps=ode_steps,
            return_all=True,
            seed=0,
            device=device,
        )
        for i, x in enumerate(samples):
            img = create_pil_image(x, nrow=sample_grid_size)
            mlflow.log_image(
                img, artifact_file=f"final_grids/final_sample_ode_step_{i}.png"
            )

        model_info = None
        if save_model:
            model_info = mlflow.pytorch.log_model(
                model,
                artifact_path="models",
            )
        mlflow.log_artifact(log_path, artifact_path="logs")
        tmpdir = os.path.dirname(log_path)
        shutil.rmtree(tmpdir, ignore_errors=True)
    return model_info
