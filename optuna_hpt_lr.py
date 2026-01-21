import mlflow
import optuna
from utils.train import train_loop_uncond
from models.unet import UNet
from utils.create_dataloaders import create_mnist_train_val_loaders
from utils.logger_utils import trial_logger
from models.config import UNetConfig, OptimConfig

train_loader, val_loader = create_mnist_train_val_loaders(num_workers=0)


# TODO set of many runs to get a goo learning rate, colour logs to make them look nicer and log parameters to terminal logger
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
        optim = optim_cfg.make_optimizer(model.parameters())
        mlflow.set_experiment(experiment_name)
        with (
            mlflow.start_run(run_name=f"trial_{trial.number:04d}"),
            trial_logger(f"trial_{trial.number}") as logger,
        ):
            # Todo test this now
            logger.info(f"Starting trial no {trial.number}")
            mlflow.set_tag("optuna.trial_number", trial.number)
            mlflow.set_tag("hpt_stage", "lr_only")
            mlflow.log_params(unet_cfg.to_mlflow_params())
            mlflow.log_params(optim_cfg.to_mlflow_params())

            def on_epoch(epoch, train_mse, val_mse):
                logger.info(f"[epoch {epoch}] train_mse={train_mse:.6f}")
                mlflow.log_metric("mse_epoch", float(train_mse), step=epoch)
                trial.report(train_mse, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            def on_step(global_step, mse_step, epoch):
                if global_step % 10 == 0:
                    logger.info(
                        f"[epoch {epoch} | global_step {global_step:06d}] train_mse={mse_step:.6f}"
                    )
                    mlflow.log_metric("mse_step", float(mse_step), step=global_step)

            best = train_loop_uncond(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=3,
                optim=optim,
                device=device,
                on_epoch=on_epoch,
                on_step=on_step,
            )

            mlflow.log_metric("best_mse", float(best))
            logger.info(f"Finishing trial with best MSE = {best:.4f}")
            return float(best)

    return objective


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="smoke_test",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    objective = make_objective(train_loader, experiment_name="smoke-test")
    study.optimize(objective, n_trials=1)
