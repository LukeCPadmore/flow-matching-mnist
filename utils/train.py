import torch
import os
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import random
import mlflow
from tqdm import tqdm 

from models.ode_solvers import euler_solver, rk2_solver, create_samples

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

def train_loop(
    model,
    dataloader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    log_every_step: int = 1,
    log_every_epoch: int = 10,
    sample_steps:int = 50,
    run_name: str = 'mnist-fm-unet',
    device: str = 'cuda',
    sample_grid_size = 8,
    save_model = True,
    ode_solver = euler_solver,
    ode_steps = 50):


    mlflow.set_experiment("flow-matching-mnist")
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    BATCH_SIZE, *IMAGE_SHAPE = next(iter(dataloader))[0].shape 
    IMAGE_SHAPE = tuple(IMAGE_SHAPE)
    with mlflow.start_run(run_name = run_name): 
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
            for i,(x1,c) in enumerate(dataloader):
                optim.zero_grad()
                mse = flow_matching_step(model,x1,loss_fn,device)
                mse.backward() 
                optim.step()
                running_loss += mse.item()

                if global_step % log_every_step == 0:
                    mlflow.log_metric("mse_step", mse.item(), step = global_step)
                global_step += 1

            if epoch % log_every_epoch == 0:
                mlflow.log_metric("mse_epoch", running_loss / len(dataloader), step = epoch)
                samples = create_samples(BATCH_SIZE, IMAGE_SHAPE, ode_solver, model, n_steps = ode_steps, seed = 0)
                save_sample_grid(samples, sample_grid_size, f'epoch_{epoch:03d}.png', artifact_subdir="images/train")
    
        samples = create_samples(BATCH_SIZE, IMAGE_SHAPE, ode_solver, model, n_steps = ode_steps, return_all=True, seed = 0)
        for i,x in enumerate(samples):
            save_sample_grid(x, sample_grid_size, f'final_sample_ode_step_{i}.png',
                            artifact_subdir='images/train') 
        
        # TODO implement
        if save_model:
            pass 
    return model
        

    


            


