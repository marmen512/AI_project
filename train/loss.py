"""
Loss computation для TRM training.
Витягнуто з trainer для кращої модульності.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import reduce, rearrange


def compute_task_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Обчислити task loss (cross-entropy).
    
    Args:
        predictions: Logits [batch_size, seq_len, vocab_size]
        labels: Labels [batch_size, seq_len]
        
    Returns:
        Task loss [batch_size]
    """
    # Rearrange для cross-entropy: [batch, vocab, seq] -> [batch, seq, vocab]
    loss = F.cross_entropy(
        rearrange(predictions, 'b n l -> b l n'),
        labels,
        reduction='none'
    )
    # Середнє по sequence
    loss = reduce(loss, 'b ... -> b', 'mean')
    return loss


def compute_halt_loss(
    halt_prob: torch.Tensor,
    is_correct: torch.Tensor
) -> torch.Tensor:
    """
    Обчислити halt loss (binary cross-entropy).
    
    Args:
        halt_prob: Halting probability [batch_size]
        is_correct: Binary tensor вказуючи чи всі токени правильні [batch_size]
        
    Returns:
        Halt loss [batch_size]
    """
    halt_loss = F.binary_cross_entropy(
        halt_prob,
        is_correct.float(),
        reduction='none'
    )
    return halt_loss


def compute_thinking_cost(
    iteration_count: int,
    max_iterations: int,
    thinking_cost_weight: float,
    device: torch.device
) -> torch.Tensor:
    """
    Обчислити thinking cost tensor.
    
    Args:
        iteration_count: Кількість ітерацій recursion
        max_iterations: Максимальна кількість ітерацій
        thinking_cost_weight: Вага thinking cost
        device: Device для tensor
        
    Returns:
        Thinking cost tensor (scalar)
    """
    # Нормалізована формула
    cost_scalar = thinking_cost_weight * (iteration_count / max_iterations)
    return torch.tensor(cost_scalar, device=device, dtype=torch.float32)


def compute_combined_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    halt_prob: torch.Tensor,
    halt_loss_weight: float = 1.0,
    thinking_cost: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Обчислити комбінований loss (task + halt + thinking cost).
    
    Args:
        predictions: Logits [batch_size, seq_len, vocab_size]
        labels: Labels [batch_size, seq_len]
        halt_prob: Halting probability [batch_size]
        halt_loss_weight: Вага halt loss
        thinking_cost: Optional thinking cost tensor
        
    Returns:
        Tuple of (total_loss, (task_loss, halt_loss))
    """
    # Task loss
    task_loss = compute_task_loss(predictions, labels)
    
    # Halt loss
    is_all_correct = (predictions.argmax(dim=-1) == labels).all(dim=-1)
    halt_loss = compute_halt_loss(halt_prob, is_all_correct)
    
    # Combined loss
    total_loss = task_loss + halt_loss * halt_loss_weight
    
    # Додати thinking cost якщо вказано
    if thinking_cost is not None:
        total_loss = total_loss + thinking_cost
    
    return total_loss.sum(), (task_loss, halt_loss)

