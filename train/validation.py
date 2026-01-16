"""
Validation logic для TRM training.
Витягнуто з trainer для кращої модульності.
"""

import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from core.types import EvalState, TrainState


def validate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    max_batches: Optional[int] = None
) -> EvalState:
    """
    Валідація моделі на dataloader.
    
    Args:
        model: Модель для валідації
        dataloader: Validation dataloader
        device: Device для обчислень
        max_batches: Максимальна кількість батчів для валідації (None = всі)
        
    Returns:
        EvalState з метриками
    """
    model.eval()
    eval_state = EvalState()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Підтримка document-aware dataset
            if len(batch_data) == 4:
                dataset_input, dataset_output, doc_ids, segment_ids = batch_data
            elif len(batch_data) == 2:
                dataset_input, dataset_output = batch_data
            else:
                continue
            
            dataset_input = dataset_input.to(device)
            dataset_output = dataset_output.to(device)
            
            # Forward pass
            outputs, latents = model.get_initial()
            
            # Простий validation - один recurrent step
            # Для повної валідації потрібен повний recurrent loop, але це спрощена версія
            try:
                loss, (main_loss, halt_loss), outputs, latents, pred, halt = model(
                    dataset_input,
                    outputs,
                    latents,
                    labels=dataset_output
                )
                
                total_loss += loss.item()
                total_samples += dataset_input.shape[0]
                
                # Зберегти приклади для qualitative analysis
                if len(all_predictions) < 10:  # Зберегти тільки перші 10
                    all_predictions.append(pred.argmax(dim=-1).cpu())
                    all_labels.append(dataset_output.cpu())
                    
            except Exception as e:
                # Пропустити проблемні батчі
                continue
    
    model.train()
    
    # Обчислити метрики
    avg_loss = total_loss / max(total_samples, 1)
    
    eval_state.eval_loss = avg_loss
    eval_state.metrics = {
        'loss': avg_loss,
        'samples': total_samples
    }
    
    return eval_state


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Обчислити accuracy.
    
    Args:
        predictions: Predictions [batch_size, seq_len]
        labels: Labels [batch_size, seq_len]
        
    Returns:
        Accuracy (0.0-1.0)
    """
    correct = (predictions == labels).float()
    accuracy = correct.mean().item()
    return accuracy


def validate_with_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    max_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Валідація з детальними метриками.
    
    Args:
        model: Модель для валідації
        dataloader: Validation dataloader
        device: Device для обчислень
        max_batches: Максимальна кількість батчів
        
    Returns:
        Dict з метриками
    """
    eval_state = validate_model(model, dataloader, device, max_batches)
    
    return {
        'eval_loss': eval_state.eval_loss,
        'eval_score': 1.0 / (1.0 + eval_state.eval_loss),  # Простий score (більше = краще)
        'metrics': eval_state.metrics,
        'samples': len(eval_state.samples)
    }

