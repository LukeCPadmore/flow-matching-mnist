from functools import partial
import mlflow
import torch
import numpy as np
import tempfile
import os

from models.ode_solvers import get_ode_solver_from_name
from utils.mlflow_tracking_utils import get_run_param, parse_int_list
from utils.FID.fid_evaluation import evaluate_fid_with_registered_backbone
from models.ode_solvers import get_ode_solver_from_name, make_vf_uncond, create_samples

def run_eval(
    generator_run_id: str,
    generator_name = "UNet",
    backbone_uri: str = "models:/fid_backbone/latest",
    stats_filename: str = "real_stats.npz",
    n_samples: int = 5210,
    device = "cuda",
    seed:int | None = None) -> None:

    if seed:
        torch.manual_seed(seed)
    device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    generator = mlflow.pytorch.load_model(f"runs:/{generator_run_id}/{generator_name}").to(device).eval()
    ode_solver = get_ode_solver_from_name(get_run_param(generator_run_id,"ode_solver"))
    f = make_vf_uncond(generator)
    ode_solver = get_ode_solver_from_name(get_run_param(generator_run_id,"ode_solver"))
    image_shape = parse_int_list(get_run_param(generator_run_id,"image_shape"))
    ode_steps = int(get_run_param(generator_run_id,"ode_steps"))
    batch_size = int(get_run_param(generator_run_id,"batch_size"))


    sample_fn = partial(
        create_samples,
        image_shape= image_shape,
        ode_solver = ode_solver,
        f=f,
        n_steps=ode_steps,
        seed=None,
        device=device,
    )

    run_name = f"fid::{generator_run_id[:8]}::{ode_solver.__name__}::{ode_steps}steps"
    mlflow.set_experiment("FM-uncond-eval")
    with mlflow.start_run(run_name = run_name):

        fid, embs = evaluate_fid_with_registered_backbone(
            sample_fn = sample_fn,
            device = device, 
            n_samples = n_samples
        )
        mlflow.log_metric("fid", float(fid))
        mlflow.log_params({
            "fid_backbone_uri": backbone_uri,
            "fid_n_samples": n_samples,
            "fid_batch_size": batch_size,
            "fid_stats_filename": stats_filename,
            "seed": seed
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            emb_path = os.path.join(tmpdir, "embeddings.npz")
            np.savez(emb_path,embs=embs)
            mlflow.log_artifact(emb_path)
        

if __name__ == "__main__":
    run_eval()
