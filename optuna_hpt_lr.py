import mlflow
import optuna
from utils.train import train_loop_uncond
from models.unet import UNet
from utils.create_dataloaders import create_mnist_train_val_loaders
from utils.logger_utils import trial_logger
from models.config import UNetConfig, OptimConfig

train_loader, val_loader = create_mnist_train_val_loaders()


def make_objective(
    train_loader, val_loader=None, experiment_name="optuna_hpt", device="cuda"
):
    mlflow.set_experiment("hpt_lr_only/unet")

    def objective(trial: optuna.Trial) -> float:
        unet_cfg = UNetConfig.sample(
            trial,
            fixed={
                "channels": (1, 64, 128, 256),
                "activation": "silu",
                "upsample_mode": "nearest",
                "group_norm_size": 8,
                "d_trunk": 32,
                "d_concat": 8,
            },
        )

        optim_cfg = OptimConfig.sample(
            trial,
            fixed={"name": "adamw", "weight_decay": 1e-4},
        )
        model = UNet.from_config(unet_cfg).to(device)
        optim = optim_cfg.make_optimizer(model.params())
        mlflow.set_experiment("hpt_lr_only/unet")
        # with mlflow.start_run(run_name=f"trial_{trial.number:04d}"):
        with (
            mlflow.start_run(run_name="smoke_test"),
            trial_logger(f"trial_{trial.number}") as logger,
        ):
            # TODO: add "logger" logging and test
            mlflow.set_tag("optuna.trial_number", trial.number)
            mlflow.set_tag("hpt_stage", "lr_only")
            mlflow.log_params(unet_cfg.to_mlflow_params())
            mlflow.log_params(optim_cfg.to_mlflow_params())

            def on_epoch(epoch, mse_epoch, val_epoch):
                mlflow.log_metric("mse_epoch", float(mse_epoch), step=epoch)
                trial.report(mse_epoch, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            def on_step(global_step, m, epoch):
                if global_step % 10 == 0:
                    mlflow.log_metric("mse_epoch", float(m), step=global_step)

            best = train_loop_uncond(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=3,
                optim=optim,
                device=device,
                on_epoch=on_epoch,
            )

            mlflow.log_metric("best_mse", float(best))
            return float(best)

    return objective


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    objective = make_objective(train_loader)
    study.optimize(objective, n_trials=1)
