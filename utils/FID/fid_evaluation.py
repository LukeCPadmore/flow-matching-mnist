import numpy as np
from scipy.linalg import sqrtm 
import mlflow
import torch
import Path 

def calc_fid(mu_real, sigma_real, mu_gen, sigma_gen, eps=1e-6) -> float:

    mu_real = np.asarray(mu_real)
    mu_gen  = np.asarray(mu_gen)
    sigma_real = np.asarray(sigma_real)
    sigma_gen  = np.asarray(sigma_gen)

    # Check mean shape
    if mu_real.ndim != 1:
        raise ValueError(f"mu_real must be 1D, got shape {mu_real.shape}")
    if mu_gen.ndim != 1:
        raise ValueError(f"mu_gen must be 1D, got shape {mu_gen.shape}")
    if mu_real.shape != mu_gen.shape:
        raise ValueError(
            f"Mean shape mismatch: {mu_real.shape} vs {mu_gen.shape}"
        )

    d = mu_real.shape[0]

    # Check covariance shape
    if sigma_real.shape != (d, d):
        raise ValueError(
            f"sigma_real must have shape ({d},{d}), got {sigma_real.shape}"
        )
    if sigma_gen.shape != (d, d):
        raise ValueError(
            f"sigma_gen must have shape ({d},{d}), got {sigma_gen.shape}"
        )

    # Check symmetric
    if not np.allclose(sigma_real, sigma_real.T, atol=1e-6):
        raise ValueError("sigma_real is not symmetric")
    if not np.allclose(sigma_gen, sigma_gen.T, atol=1e-6):
        raise ValueError("sigma_gen is not symmetric")
    mean_term = np.sum((mu_real - mu_gen) ** 2) 

    d = sigma_real.shape[0]
    sigma1 = sigma_real + eps * np.eye(d)
    sigma2 = sigma_gen + eps * np.eye(d)

    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = mean_term + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

@torch.no_grad()
def evaluate_fid_with_registered_backbone(
    *,
    generator,
    sample_fn,  # (generator, n, device) -> torch.Tensor
    backbone_uri: str = "models:/fid_backbone/latest", 
    stats_filename: str = "real_stats.npz",    
    n_samples: int = 5210,
    batch_size: int = 128,
    device: str = "cuda",
    step: int | None = None,
    log_to_mlflow = False
):
    device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    # load backbone model
    backbone = mlflow.pytorch.load_model(backbone_uri).to(device).eval()

    # load stats from bundled artifact
    model_dir = Path(mlflow.artifacts.download_artifacts(backbone_uri))
    stats_path = model_dir / stats_filename

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Could not find '{stats_filename}' inside model artifacts at {model_dir}. "
            f"Available: {[p.name for p in model_dir.iterdir()]}"
        )

    stats = np.load(stats_path)
    mu_real = stats["mu"]
    sigma_real = stats["sigma"]

    # Generate samples 
    embs = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        imgs = sample_fn(generator, b, device=device)
        if not torch.is_tensor(imgs):
            raise TypeError("sample_fn must return a torch.Tensor")

        imgs = imgs.to(device)
        z = backbone(imgs)
        z = z.view(z.size(0), -1)  # ensure (B, D)
        embs.append(z.detach().cpu().numpy())
        remaining -= b

    embs = np.concatenate(embs, axis=0)
    mu_fake = embs.mean(axis=0)
    sigma_fake = np.cov(embs, rowvar=False)

    fid = calc_fid(mu_fake, sigma_fake, mu_real, sigma_real)
    
    if log_to_mlflow:
        mlflow.log_metric("fid", float(fid), step=step)
        mlflow.log_params({
            "fid_backbone_uri": backbone_uri,
            "fid_n_samples": n_samples,
            "fid_batch_size": batch_size,
            "fid_stats_filename": stats_filename,
        })

    return float(fid)

