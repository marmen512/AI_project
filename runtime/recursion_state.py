"""
Internal state management for TRM recursive reasoning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
import time


@dataclass
class RecursionState:
    """
    Internal state tracking for recursive reasoning refinement.
    
    Attributes:
        hidden_state: Current hidden state tensor
        iteration: Current iteration count (0-indexed)
        halting_probs: List of halting probabilities per step
        entropy_deltas: List of entropy changes between steps
        accumulated_cost: Total thinking cost accumulated
        timestamps: List of timestamps for each iteration (for timeout)
        step_logits: List of output logits per step (for analysis)
        step_entropies: List of entropy values per step
    """
    hidden_state: Optional[torch.Tensor] = None
    iteration: int = 0
    halting_probs: List[float] = field(default_factory=list)
    entropy_deltas: List[float] = field(default_factory=list)
    accumulated_cost: float = 0.0
    timestamps: List[float] = field(default_factory=list)
    step_logits: List[torch.Tensor] = field(default_factory=list)
    step_entropies: List[float] = field(default_factory=list)
    
    def add_iteration(
        self,
        hidden_state: torch.Tensor,
        halting_prob: float,
        logits: torch.Tensor,
        entropy: float,
        cost: float = 1.0
    ):
        """
        Add new iteration data to state.
        
        Args:
            hidden_state: New hidden state
            halting_prob: Halting probability for this step
            logits: Output logits for this step
            entropy: Entropy of logits distribution
            cost: Cost for this iteration (default 1.0)
        """
        self.hidden_state = hidden_state
        self.iteration += 1
        self.halting_probs.append(halting_prob)
        self.timestamps.append(time.time())
        
        # Store logits (detached to save memory)
        if isinstance(logits, torch.Tensor):
            self.step_logits.append(logits.detach().clone())
        else:
            self.step_logits.append(logits)
        
        # Compute entropy delta if we have previous entropy
        if len(self.step_entropies) > 0:
            prev_entropy = self.step_entropies[-1]
            self.entropy_deltas.append(entropy - prev_entropy)
        else:
            self.entropy_deltas.append(0.0)
        
        self.step_entropies.append(entropy)
        self.accumulated_cost += cost
    
    def get_current_entropy_delta(self) -> float:
        """Get entropy delta for the last step."""
        if len(self.entropy_deltas) > 0:
            return self.entropy_deltas[-1]
        return 0.0
    
    def check_timeout(self, max_seconds: Optional[float] = None) -> bool:
        """
        Check if recursion has exceeded timeout.
        
        Args:
            max_seconds: Maximum seconds allowed (None = no timeout)
            
        Returns:
            True if timeout exceeded, False otherwise
        """
        if max_seconds is None or len(self.timestamps) < 2:
            return False
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return elapsed > max_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging/debugging."""
        return {
            'iteration': self.iteration,
            'accumulated_cost': self.accumulated_cost,
            'num_halting_probs': len(self.halting_probs),
            'num_entropy_deltas': len(self.entropy_deltas),
            'current_entropy_delta': self.get_current_entropy_delta(),
            'last_entropy': self.step_entropies[-1] if self.step_entropies else None,
            'last_halt_prob': self.halting_probs[-1] if self.halting_probs else None,
        }
