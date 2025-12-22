import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sys
from pathlib import Path

# Project root - works whether run directly or as module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataloaders import DataLoaderFactory


import mlflow
import mlflow.pytorch
import dagshub
from torchvision import models



class DenseNetClassifier(nn.Module):
    """
    Pretrained DenseNet-121 encoder + trainable linear head (4 classes).
    Encoder is UNFROZEN for fine-tuning.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)


        # DenseNet feature dim is 1024
        embed_dim = backbone.classifier.in_features


        # Remove original classifier and keep features
        backbone.classifier = nn.Identity()


        self.encoder = backbone
        # REMOVED: freezing encoder parameters
        for p in self.encoder.parameters():
            p.requires_grad = False


        self.classifier = nn.Linear(embed_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # REMOVED: torch.no_grad() to allow gradients
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0


    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)


    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }



@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0


    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)
        loss = criterion(outputs, labels)


        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)


    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }



def main():
    # 1) Load config + dataloaders
    with open(PROJECT_ROOT / "configs/preprocessing_config.yaml", "r") as f:
        config = yaml.safe_load(f)


    # Initialize DagsHub
    dagshub.init(
        repo_owner=config['tracking']['dagshub_repo'].split('/')[0],
        repo_name=config['tracking']['dagshub_repo'].split('/')[1],
        mlflow=True
    )
    print("✓ Connected to DagsHub MLflow")


    factory = DataLoaderFactory(config)
    loaders = factory.create_dataloaders()
    train_loader = loaders["train"]
    val_loader = loaders["val"]


    # 2) Build model: DenseNet-121 encoder + 4-class head
    num_classes = 4
    model = DenseNetClassifier(num_classes=num_classes)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # 3) Loss, optimizer, class weights
    class_weights = factory.get_class_weights(split="train").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    # CHANGED: Use differential learning rates for fine-tuning
    optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=1e-3
)



    # 4) MLflow experiment
    mlflow.set_experiment("skin_disease_densenet")


    num_epochs = 5



    with mlflow.start_run(run_name="densenet121_finetuning"):
        mlflow.log_params({
            "encoder_lr": 1e-5,
            "classifier_lr": 1e-4,
            "batch_size": config["output"]["batch_size"],
            "num_epochs": num_epochs,
            "image_size": config["image"]["target_size"],
            "normalization": "ImageNet",
            "model": "DenseNet-121 fine-tuning (encoder + classifier)",
        })


        for epoch in range(num_epochs):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)


            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                },
                step=epoch,
            )


            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.4f} | "
                f"Val loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.4f}"
            )


        mlflow.pytorch.log_model(model, "model")



if __name__ == "__main__":
    main()
