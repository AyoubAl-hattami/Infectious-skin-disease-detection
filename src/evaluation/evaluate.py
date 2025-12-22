import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import json
from pathlib import Path
from typing import Dict, List, Tuple


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: List[str] = None
) -> Dict:
    """
    Comprehensive evaluation with all metrics.
    
    Returns:
        Dictionary containing loss, accuracy, per-class metrics, and confusion matrix
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    avg_loss = running_loss / total
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Use provided class names or default indices
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(precision))]
    
    # Build per-class metrics dictionary
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1[idx]),
            "support": int(support[idx])
        }
    
    # Build results dictionary
    results = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }
    
    return results


def save_evaluation_results(results: Dict, save_path: str):
    """Save evaluation results to JSON file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Evaluation results saved to {save_path}")


def print_evaluation_summary(results: Dict, split_name: str = "Test"):
    """Print formatted evaluation summary."""
    print(f"\n{'='*60}")
    print(f"{split_name} Set Evaluation Results")
    print(f"{'='*60}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision (Macro): {results['precision_macro']:.4f}")
    print(f"Recall (Macro): {results['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
    
    print(f"\n{'Per-Class Metrics':^60}")
    print(f"{'-'*60}")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*60}")
    
    for class_name, metrics in results['per_class_metrics'].items():
        print(
            f"{class_name:<20} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1_score']:<12.4f}"
        )
    print(f"{'='*60}\n")
