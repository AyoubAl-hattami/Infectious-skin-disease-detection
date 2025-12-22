import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import mlflow
import mlflow.pytorch
import dagshub
from torchvision import models

from src.preprocessing.dataloaders import DataLoaderFactory
from src.evaluation.evaluate import (
    evaluate_model,
    save_evaluation_results,
    print_evaluation_summary
)


class DenseNetClassifier(nn.Module):
    """Pretrained DenseNet-121 encoder + trainable linear head."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        embed_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        
        self.encoder = backbone
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
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


def main():
    # Load configs
    with open("configs/preprocessing_config.yaml", "r") as f:
        preprocess_config = yaml.safe_load(f)
    
    with open("configs/training_config.yaml", "r") as f:
        train_config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(train_config['seed'])
    
    # Initialize DagsHub
    dagshub.init(
        repo_owner=preprocess_config['tracking']['dagshub_repo'].split('/')[0],
        repo_name=preprocess_config['tracking']['dagshub_repo'].split('/')[1],
        mlflow=True
    )
    print("✓ Connected to DagsHub MLflow")
    
    # Create dataloaders
    factory = DataLoaderFactory(preprocess_config)
    loaders = factory.create_dataloaders()
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders.get("test")
    
    # Get class names from config
    class_names = preprocess_config['data']['classes']
    
    # Build model
    num_classes = train_config['model']['num_classes']
    model = DenseNetClassifier(num_classes=num_classes)
    
    # Device setup
    device_config = train_config['device']
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    model.to(device)
    print(f"✓ Using device: {device}")
    
    # Loss function
    if train_config['training']['loss']['use_class_weights']:
        class_weights = factory.get_class_weights(split="train").to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    opt_config = train_config['training']['optimizer']
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': opt_config['encoder_lr']},
        {'params': model.classifier.parameters(), 'lr': opt_config['classifier_lr']}
    ], weight_decay=opt_config['weight_decay'])
    
    # Checkpointing setup
    if train_config['checkpointing']['enabled']:
        checkpoint_dir = Path(train_config['checkpointing']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_metric = 0.0 if train_config['checkpointing']['mode'] == 'max' else float('inf')
    
    # MLflow experiment
    mlflow.set_experiment(train_config['mlflow']['experiment_name'])
    
    num_epochs = train_config['training']['num_epochs']
    
    with mlflow.start_run(run_name=train_config['mlflow']['run_name']):
        # Log all parameters
        mlflow.log_params({
            "model": train_config['model']['name'],
            "num_classes": num_classes,
            "encoder_lr": opt_config['encoder_lr'],
            "classifier_lr": opt_config['classifier_lr'],
            "weight_decay": opt_config['weight_decay'],
            "batch_size": preprocess_config["output"]["batch_size"],
            "num_epochs": num_epochs,
            "image_size": preprocess_config["image"]["target_size"],
            "seed": train_config['seed'],
            "device": str(device)
        })
        
        # Training loop
        for epoch in range(num_epochs):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation evaluation (quick metrics)
            val_results = evaluate_model(model, val_loader, criterion, device, class_names)
            
            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_results["loss"],
                    "val_acc": val_results["accuracy"],
                    "val_f1_macro": val_results["f1_macro"],
                },
                step=epoch,
            )
            
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.4f} | "
                f"Val loss: {val_results['loss']:.4f}, acc: {val_results['accuracy']:.4f}, "
                f"F1: {val_results['f1_macro']:.4f}"
            )
            
            # Model checkpointing
            if train_config['checkpointing']['enabled']:
                monitor_metric = val_results['accuracy'] if train_config['checkpointing']['monitor'] == 'val_acc' else val_results['loss']
                mode = train_config['checkpointing']['mode']
                
                is_better = (mode == 'max' and monitor_metric > best_metric) or \
                           (mode == 'min' and monitor_metric < best_metric)
                
                if is_better:
                    best_metric = monitor_metric
                    checkpoint_path = checkpoint_dir / f"best_model_epoch{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_results['accuracy'],
                        'val_loss': val_results['loss'],
                    }, checkpoint_path)
                    print(f"✓ Saved best model checkpoint: {checkpoint_path}")
        
        # Final evaluation on test/val sets
        eval_config = train_config['evaluation']['run_on']
        
        if eval_config in ["val", "both"]:
            print("\n" + "="*60)
            print("Running final validation set evaluation...")
            val_final_results = evaluate_model(model, val_loader, criterion, device, class_names)
            print_evaluation_summary(val_final_results, "Validation")
            
            # Save and log to MLflow
            val_results_path = "results/val_evaluation.json"
            save_evaluation_results(val_final_results, val_results_path)
            mlflow.log_artifact(val_results_path)
        
        if eval_config in ["test", "both"] and test_loader is not None:
            print("\n" + "="*60)
            print("Running final test set evaluation...")
            test_results = evaluate_model(model, test_loader, criterion, device, class_names)
            print_evaluation_summary(test_results, "Test")
            
            # Save and log to MLflow
            test_results_path = "results/test_evaluation.json"
            save_evaluation_results(test_results, test_results_path)
            mlflow.log_artifact(test_results_path)
            
            # Log final test metrics
            mlflow.log_metrics({
                "test_accuracy": test_results["accuracy"],
                "test_f1_macro": test_results["f1_macro"],
                "test_precision_macro": test_results["precision_macro"],
                "test_recall_macro": test_results["recall_macro"],
            })
        
        # Log final model
        mlflow.pytorch.log_model(model, "final_model")
        print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
