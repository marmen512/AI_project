"""
Розширений логер для TRM навчання
Логує: loss по кроках, глибину рекурсії, entropy, gradient norms
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


class TRMTrainingLogger:
    """Розширений логер для TRM навчання"""
    
    def __init__(self, log_dir: Path = None):
        if log_dir is None:
            log_dir = Path("logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.metrics = {
            'batch_losses': [],
            'step_losses': [],  # Loss по recurrent steps
            'recursion_depths': [],  # Глибина рекурсії
            'halt_probs': [],
            'entropies': [],  # Entropy виходу
            'entropy_deltas': [],  # Delta entropy між кроками recursion
            'depth_vs_entropy': [],  # Пари (depth, entropy) для аналізу
            'thinking_costs': [],  # Thinking cost на sample
            'gradient_norms': []
        }
    
    def log_batch(
        self,
        batch_idx: int,
        epoch: int,
        loss: float,
        step_losses: List[float],  # Loss на кожному recurrent step
        recursion_depths: List[int],
        halt_probs: List[float],
        predictions: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        step_entropies: Optional[List[float]] = None,  # Entropy на кожному кроці recursion
        thinking_cost: Optional[float] = None  # Thinking cost для цього batch
    ):
        """
        Логувати метрики батча
        
        Args:
            step_entropies: Список entropy значень на кожному кроці recursion (для обчислення delta)
            thinking_cost: Thinking cost для цього batch
        """
        # Entropy виходу (поточна)
        probs = F.softmax(predictions, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        
        # Обчислити entropy deltas якщо є step_entropies
        if step_entropies and len(step_entropies) > 1:
            for i in range(1, len(step_entropies)):
                delta = step_entropies[i] - step_entropies[i-1]
                self.metrics['entropy_deltas'].append(delta)
        
        # Depth vs entropy пари для аналізу
        if recursion_depths:
            for depth, ent in zip(recursion_depths, [entropy] * len(recursion_depths)):
                self.metrics['depth_vs_entropy'].append({
                    'depth': depth,
                    'entropy': ent
                })
            # Також додати пари з step_entropies якщо доступні
            if step_entropies and len(step_entropies) == len(recursion_depths):
                for depth, ent in zip(recursion_depths, step_entropies):
                    self.metrics['depth_vs_entropy'].append({
                        'depth': depth,
                        'entropy': ent
                    })
        
        # Gradient norm
        grad_norm = None
        if gradients is not None:
            grad_norm = gradients.norm().item()
        
        self.metrics['batch_losses'].append({
            'batch': batch_idx,
            'epoch': epoch,
            'loss': loss
        })
        
        self.metrics['step_losses'].extend([
            {'step': i, 'loss': sl} for i, sl in enumerate(step_losses)
        ])
        
        self.metrics['recursion_depths'].extend(recursion_depths)
        self.metrics['halt_probs'].extend(halt_probs)
        self.metrics['entropies'].append(entropy)
        
        if thinking_cost is not None:
            self.metrics['thinking_costs'].append(thinking_cost)
        
        if grad_norm:
            self.metrics['gradient_norms'].append(grad_norm)
    
    def save(self, filename: str = "training_metrics.json"):
        """Зберегти метрики"""
        output_path = self.log_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return output_path
    
    def get_summary(self) -> Dict:
        """Отримати підсумок метрик"""
        summary = {}
        
        if self.metrics['batch_losses']:
            summary['avg_loss'] = sum(m['loss'] for m in self.metrics['batch_losses']) / len(self.metrics['batch_losses'])
            summary['min_loss'] = min(m['loss'] for m in self.metrics['batch_losses'])
            summary['max_loss'] = max(m['loss'] for m in self.metrics['batch_losses'])
        
        if self.metrics['recursion_depths']:
            summary['avg_recursion_depth'] = sum(self.metrics['recursion_depths']) / len(self.metrics['recursion_depths'])
            summary['max_recursion_depth'] = max(self.metrics['recursion_depths'])
            summary['min_recursion_depth'] = min(self.metrics['recursion_depths'])
        
        if self.metrics['entropies']:
            summary['avg_entropy'] = sum(self.metrics['entropies']) / len(self.metrics['entropies'])
            summary['min_entropy'] = min(self.metrics['entropies'])
            summary['max_entropy'] = max(self.metrics['entropies'])
        
        # Entropy delta аналіз
        if self.metrics['entropy_deltas']:
            summary['avg_entropy_delta'] = sum(self.metrics['entropy_deltas']) / len(self.metrics['entropy_deltas'])
            summary['max_entropy_delta'] = max(self.metrics['entropy_deltas'])
            summary['min_entropy_delta'] = min(self.metrics['entropy_deltas'])
        
        # Depth vs entropy кореляція (простий аналіз)
        if self.metrics['depth_vs_entropy']:
            depths = [d['depth'] for d in self.metrics['depth_vs_entropy']]
            entropies = [d['entropy'] for d in self.metrics['depth_vs_entropy']]
            # Проста кореляція (лише якщо достатньо даних)
            if len(depths) > 1:
                import statistics
                try:
                    summary['depth_entropy_correlation'] = {
                        'avg_depth': statistics.mean(depths),
                        'avg_entropy_at_depth': statistics.mean(entropies),
                        'depth_range': (min(depths), max(depths)),
                        'entropy_range': (min(entropies), max(entropies))
                    }
                except:
                    pass
        
        # Thinking cost статистика
        if self.metrics['thinking_costs']:
            summary['avg_thinking_cost'] = sum(self.metrics['thinking_costs']) / len(self.metrics['thinking_costs'])
            summary['total_thinking_cost'] = sum(self.metrics['thinking_costs'])
        
        if self.metrics['gradient_norms']:
            summary['avg_gradient_norm'] = sum(self.metrics['gradient_norms']) / len(self.metrics['gradient_norms'])
            summary['max_gradient_norm'] = max(self.metrics['gradient_norms'])
        
        if self.metrics['halt_probs']:
            summary['avg_halt_prob'] = sum(self.metrics['halt_probs']) / len(self.metrics['halt_probs'])
        
        return summary
    
    def print_summary(self):
        """Вивести підсумок метрик"""
        summary = self.get_summary()
        print("\n" + "=" * 70)
        print("📊 ПІДСУМОК МЕТРИК НАВЧАННЯ")
        print("=" * 70)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        print("=" * 70 + "\n")


