import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim 
import torch
from tqdm import tqdm 
import mlflow
from utils.FID.fid_model import FID_classifier
from datetime import datetime
import numpy as np
from utils.logger_utils import get_temp_logger
import os, tempfile, shutil


def compute_fid_stats(backbone: nn.Module,testloader:DataLoader): 
    device = next(backbone.parameters()).device
    backbone.eval()
    embs = []
    # Send test dataset through backbone
    with torch.no_grad():
        backbone.eval()
        embs = []
        for x,_ in testloader:        
            x = x.to(device)
            # Collect embeddings
            pred = backbone(x)
            embs.append(pred.detach().cpu())
        embs = np.concatenate(embs, axis=0)

    mu = embs.mean(axis=0)
    sigma = np.cov(embs, rowvar=False)

    return mu, sigma

def create_fid_backbone(fid_classifier) -> nn.Module:
    convs, clf = fid_classifier.children()
    clf_no_head = nn.Sequential(*list(clf.children())[:-1])
    fid_backbone = nn.Sequential(
        convs, 
        clf_no_head
    )
    return fid_backbone

def train(
        model,
        trainloader,
        valloader,
        num_epochs,
        lr,
        val_acc_every,
        device = None) -> None:
    mlflow.set_experiment("FID Classifier Training")
    optim = torch.optim.AdamW(model.parameters(),lr = lr)
    loss_fn = nn.CrossEntropyLoss()
    run_name = "FID_Classifier_Training" "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name = run_name) as run:
        logger_name = f"fid_{run.info.run_id}"
        logger, log_path = get_temp_logger(logger_name)

        params = {
            "num_epochs": num_epochs,
            "lr": lr,
            "conv_block_size": model.conv_block_channels,
            "n_classes": model.n_classes,
            "fid_embed": model.fid_emb
        }
        mlflow.log_params(params)

        logger.info("Hyperparameters:\n" + "\n".join(f" {k}: {v}" for k, v in params.items()))
        for epoch in tqdm(range(num_epochs),desc="Training"):
            model.train()
            running_loss = 0.0
            for i,(x,y) in enumerate(trainloader):
                x = x.to(device)
                y = y.to(device)
                optim.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y_pred,y)
                loss.backward()
                optim.step()
                running_loss += loss.item()
            mlflow.log_metric("Cross_entropy_epoch", running_loss / len(trainloader), step = epoch)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss:.4f}")

            # Test validation accuracy
            if epoch % val_acc_every == 0: 
                model.eval()
                correct = 0
                total = 0
                for (x,y) in valloader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    _, preds = torch.max(y_pred,dim = 1)
                    correct += torch.sum(preds == y).item()
                    total += y.size(0)
                accuracy = correct / total
                mlflow.log_metric("Val_acc",accuracy, step = epoch)
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] | Val accuracy (periodic): {accuracy:.4f}"
                )
        logger.info("Training complete, computing final validate accuracy")
        # Compute validation accuracy for reporting
        model.eval()
        correct = 0
        total = 0
        for (x,y) in valloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            _, preds = torch.max(y_pred,dim = 1)
            correct += torch.sum(preds == y).item()
            total += y.size(0)
        accuracy = correct / total

        logger.info("Saving artifacts and computing FID")
        mlflow.log_metric("Val_acc_final",accuracy)
        mlflow.pytorch.log_model(model, 
                                 name="fid_classifier")
        
        # Create backbone and log
        backbone = create_fid_backbone(model)
        mlflow.pytorch.log_model(backbone, name="fid_backbone")

        # Create fid stats and save artifacts
        mu, sigma = compute_fid_stats(backbone,valloader)
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = os.path.join(tmpdir, "real_stats.npz")
            np.savez(stats_path, mu=mu, sigma=sigma)
            mlflow.log_artifact(stats_path, artifact_path="fid_stats")

        mlflow.log_artifact(log_path, artifact_path="logs")
        tmpdir = os.path.dirname(log_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    pass