import mlflow
import optuna
from utils.train import train_loop_uncond
from models.unet import UNet
from utils.create_dataloaders import create_mnist_train_val_loaders

train_loader, val_loader = create_mnist_train_val_loaders()


def sample_unet_cfg(trial: optuna.Trial):
    pass


def sample_optim_cfg(trial: optuna.Trial):
    pass


def make_objective(train_loader, device="cuda"):
    def objective(trial: optuna.Trial) -> float:
        unet_cfg = sample_unet_cfg(trial)
        optim_cfg = sample_optim_cfg(trial)
        model = UNet.from_config(unet_cfg).to(device)

        mlflow.set_experiment("hpt/unet")
        with mlflow.start_run(run_name=f"trial_{trial.number:04d}"):
            mlflow.log_params(unet_cfg.to_dict())

            def on_epoch(epoch, mse_epoch):
                mlflow.log_metric("mse_epoch", float(mse_epoch), step=epoch)
                trial.report(mse_epoch, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            best = train_loop_uncond(
                model=model,
                dataloader=train_loader,
                num_epochs=3,
                # lr=cfg.lr,
                device=device,
                on_epoch=on_epoch,
            )

            mlflow.log_metric("best_mse", float(best))
            return float(best)

    return objective


objective = make_objective(train_loader)
study.optimize(objective, n_trials=50)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
