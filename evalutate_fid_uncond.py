import mlflow
from utils.FID.fid_evaluation import evaluate_fid_with_registered_backbone
from utils.mlflow_tracking_utils import get_run_param
from models.ode_solvers import get_ode_solver_from_name
from from models.ode_solvers import make_vf_uncond
import torch

def run_eval(
    generator_run_id: str,
    backbone_uri: str = "models:/fid_backbone/latest",
    stats_filename: str = "real_stats.npz",
    n_samples: int = 5210,
    batch_size: int = 128,
    device = "cuda",
    seed:int = 0):

    device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    generator = mlflow.pytorch.load_model(generator_run_id).to(device).eval()
    ode_solver = get_ode_solver_from_name(get_run_param(generator_run_id,"ode_solver"))
    # TODO: retrieve image size logged and other mlflow params to create partial function e.g.
    """
    f = make_vf_uncond(model)
    samples = create_samples(64,(1,32,32), euler_solver, f, n_steps = 50, seed = 0, device=device)
    
    sample_fn = partial(
        create_samples,
        image_shape=(1, 32, 32),
        ode_solver=euler_solver,
        f=f,
        n_steps=50,
        seed=None,
        device=device,
    )
    sample_fn
"""
    torch.manual_seed(seed)
    f = make_vf_uncond(generator)
    # TODO: Set experiment name
    mlflow.set_experiment("")
    with mlflow.start_run():
        
        mlflow.log_params({
            "n_samples": n_samples, 
            "batch_size": batch_size,
            "fid_backbone_uri": backbone_uri,
            "seed":seed
        })
        # evaluate_fid_with_registered_backbone(
        #     generator= None,
        #     sam
        # )
        mlflow.log_metric("fid", float(fid))
        mlflow.log_params({
            "fid_backbone_uri": backbone_uri,
            "fid_n_samples": n_samples,
            "fid_batch_size": batch_size,
            "fid_stats_filename": stats_filename,
        })
if __name__ == "__main__":
    run_eval()
