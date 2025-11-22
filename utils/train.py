import torch
import os
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import random
import mlflow
import os
from tqdm import tqdm 
import mlflow.pytorch
from datetime import datetime

from models.ode_solvers import euler_solver, rk2_solver, create_samples, make_vf_uncond, make_vf_cfg

def flow_matching_step(model,x1,loss_fn,device):
    B = x1.shape[0]
    x1 = x1.to(device)
    x0 = torch.randn_like(x1).to(device)
    t = torch.rand(B,1,1,1).to(device)
    v_est = model((1-t) * x0 + x1 * t,t)
    v_true = x1 - x0
    mse = loss_fn(v_est,v_true)
    return mse 

def flow_matching_step_cfg(model, x1, y, p_drop, NULL_ID, loss_fn, device):
    B = x1.shape[0]
    x1 = x1.to(device)
    y = y.to(device)

    x0 = torch.randn_like(x1).to(device)
    t = torch.rand(B,1,1,1).to(device)

    drop_mask = (torch.rand_like(y.float()) < p_drop)
    y_drop = y.clone() 
    y_drop[drop_mask] = NULL_ID
    
    v_est = model((1-t) * x0 + x1 * t,t,y_drop)
    v_true = x1 - x0
    mse = loss_fn(v_est,v_true)
    return mse 

def save_sample_grid(samples,grid_size,file_name,artifact_subdir = None) -> None:
    if samples.min() < 0:
        samples = (samples + 1) / 2 

    samples = samples.clamp(0, 1)

    # Make grid
    grid = make_grid(samples, nrow=grid_size, padding=2)

    # Build full path
    out_dir = artifact_subdir or "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    save_image(grid, out_path)

    return out_path

def train_loop_uncond(
    model,
    optim,
    dataloader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    log_every_step: int = 1,
    log_every_epoch: int = 10,
    sample_steps:int = 50,
    run_name: str = 'mnist-fm-unet',
    device: str = 'cuda',
    sample_grid_size = 8,
    ode_solver = euler_solver,
    ode_steps = 50,
    save_model = True,
    register_model_name = None):

    mlflow.set_experiment("flow-matching-mnist")
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    images, labels = next(iter(dataloader))
    BATCH_SIZE, *IMAGE_SHAPE = images.shape
    IMAGE_SHAPE = tuple(IMAGE_SHAPE)
    input_example = {
        "x": torch.randn(1, *IMAGE_SHAPE).cpu().numpy(),
        "t": torch.zeros(1, 1, 1, 1).cpu().numpy(),
    }
    with mlflow.start_run(run_name = run_name) as run: 
        mlflow.log_params({
            "lr":lr,
            "epochs": num_epochs,
            "samples_steps": sample_steps,
            "model_params": sum(p.numel() for p in model.parameters()),
            "batch_size": BATCH_SIZE,
            "ode_steps": ode_steps,
            "ode_solver": getattr(ode_solver, "__name__", str(ode_solver))
        })
        global_step = 0
        for epoch in tqdm(range(num_epochs)):
            model.train()
            running_loss = 0.0
            # Loop over dataset
            for i,(x1,_) in enumerate(dataloader):
                optim.zero_grad()
                # Generate esstimated velociy fields and compute MSE
                mse = flow_matching_step(model,x1,loss_fn,device)
                mse.backward() 
                optim.step()
                running_loss += mse.item()

                if global_step % log_every_step == 0:
                    mlflow.log_metric("mse_step", mse.item(), step = global_step)
                global_step += 1
            # Sample batch of images
            if epoch % log_every_epoch == 0:
                mlflow.log_metric("mse_epoch", running_loss / len(dataloader), step = epoch)
                # Create callback for velocity field
                f = make_vf_uncond(model)
                samples = create_samples(BATCH_SIZE, IMAGE_SHAPE, ode_solver, f, n_steps = ode_steps, seed = 0, device=device)
                path = save_sample_grid(samples, sample_grid_size, f'epoch_{epoch:03d}.png', artifact_subdir="images/train")
                mlflow.log_artifact(path,artifact_path = 'samples')

        # Create samples with all time steps and save
        
         # Create callback for velocity field
        f = make_vf_uncond(model)
        samples = create_samples(BATCH_SIZE, IMAGE_SHAPE, ode_solver, f, n_steps = ode_steps, return_all=True, seed = 0,device=device)
        for i,x in enumerate(samples):
            save_sample_grid(x, sample_grid_size, f'final_sample_ode_step_{i}.png',
                            artifact_subdir='images/train') 
            mlflow.log_artifact(path,artifact_path = 'images')

        # Save model
        model_info = None
        if save_model:
            model_info = mlflow.pytorch.log_model(
                model,
                name = 'UNet', 
                register_model_name = register_model_name,
                input_example = input_example
            )
    return model_info
        

def train_loop_cfg(
    model,
    dataloader: DataLoader,
    NULL_ID,
    p_drop:float = 0.2,
    w = 1,
    num_epochs: int = 10,
    lr: float = 1e-3,
    log_every_step: int = 1,
    log_every_epoch: int = 10,
    sample_steps:int = 50,
    experiment_name: str = 'mnist-fm-cond-unet',
    run_name:str = None,
    device: str = 'cuda',
    sample_grid_size = 8,
    ode_solver = euler_solver,
    ode_steps = 50,
    save_model = True):

    mlflow.set_experiment(experiment_name)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    images, labels = next(iter(dataloader))
    BATCH_SIZE, *IMAGE_SHAPE = images.shape
    IMAGE_SHAPE = tuple(IMAGE_SHAPE)
    input_example = {
        "x" : torch.randn(1,*IMAGE_SHAPE).cpu().numpy(),
        "t" : torch.zeros(1,1,1,1).cpu().numpy(),
        "y" : torch.tensor([0], dtype = labels.dtype).cpu().numpy()
    }
    
    run_name = f'{run_name if run_name else experiment_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    with mlflow.start_run(run_name = run_name) as run: 
        mlflow.log_params({
            "lr":lr,
            "epochs": num_epochs,
            "cfg_strength": w,
            "p_drop": p_drop,
            "NULL_ID": NULL_ID,
            "samples_steps": sample_steps,
            "model_params": sum(p.numel() for p in model.parameters()),
            "batch_size": BATCH_SIZE,
            "ode_steps": ode_steps,
            "ode_solver": getattr(ode_solver, "__name__", str(ode_solver))
        })
        conditional_sampling_grid_labels = torch.tensor([i for i in range(NULL_ID)]).repeat(sample_grid_size,1).flatten()
        global_step = 0
        for epoch in tqdm(range(num_epochs)):
            model.train()
            running_loss = 0.0
            for i, (x1,c) in enumerate(dataloader):
                optim.zero_grad()
                mse = flow_matching_step_cfg(model,x1,c,p_drop,NULL_ID,loss_fn,device)
                mse.backward() 
                optim.step()
                running_loss += mse.item()

                if global_step % log_every_step == 0:
                    mlflow.log_metric("mse_step", mse.item(), step = global_step)
                global_step += 1

            if epoch % log_every_epoch == 0:
                f = make_vf_cfg(model,conditional_sampling_grid_labels,w,num_epochs)
                mlflow.log_metric("mse_epoch", running_loss / len(dataloader), step = epoch)
                samples = create_samples(NULL_ID * sample_grid_size, IMAGE_SHAPE, ode_solver, f, n_steps = ode_steps, seed = 0, device=device)
                path = save_sample_grid(samples, sample_grid_size, f'epoch_{epoch:03d}.png', artifact_subdir="images/train")
                mlflow.log_artifact(path,artifact_path = 'samples')

        f = make_vf_cfg(model,conditional_sampling_grid_labels,w,num_epochs)
        samples = create_samples(NULL_ID * sample_grid_size, IMAGE_SHAPE, ode_solver, f, n_steps = ode_steps, return_all=True, seed = 0, device=device)
        for i,x in enumerate(samples):
            save_sample_grid(x, sample_grid_size, f'final_sample_ode_step_{i}.png',
                            artifact_subdir='images/train') 
            mlflow.log_artifact(path,artifact_path = 'images')
            
        model_info = None
        if save_model:
            model_info = mlflow.pytorch.log_model(
                model,
                artifact_path = 'models',
                #input_example = input_example
                )
    return model_info
    


            


