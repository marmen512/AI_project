from __future__ import annotations
from contextlib import nullcontext
from typing import Optional, Tuple, TYPE_CHECKING
import time

import torch
from torch import nn, cat, arange, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Reduce, Rearrange

# network related

from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from runtime.recursion_state import RecursionState

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return t.numel() == 0

def range_from_one(n):
    return range(1, n + 1)

# classes

class TinyRecursiveModel(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        network: Module,
        num_refinement_blocks = 3,   # T in paper
        num_latent_refinements = 6,  # n in paper - 1 output refinement per N latent refinements
        halt_loss_weight = 1.,
        num_register_tokens = 0,
        max_recursion_depth: int = 20,  # НОВИЙ: guard для рекурсії
        adaptive_recursion: bool = False,  # Увімкнути adaptive recursion gate
        timeout_seconds: Optional[float] = None,  # Timeout для recursion (None = no timeout)
        thinking_cost_weight: float = 0.01  # Weight для thinking cost в loss
    ):
        super().__init__()
        assert num_refinement_blocks > 1

        self.input_embed = nn.Embedding(num_tokens, dim)
        self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        self.network = network

        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks
        self.max_recursion_depth = max_recursion_depth  # Guard для рекурсії
        self.adaptive_recursion = adaptive_recursion
        self.timeout_seconds = timeout_seconds
        self.thinking_cost_weight = thinking_cost_weight

        # register tokens for the self attend version

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # prediction heads

        self.to_pred = nn.Linear(dim, num_tokens, bias = False)

        self.to_halt_pred = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, 1, bias = False),
            nn.Sigmoid(),
            Rearrange('... 1 -> ...')
        )

        self.halt_loss_weight = halt_loss_weight
        
        # Adaptive recursion gate - якщо увімкнено, створюємо gate network
        if adaptive_recursion:
            # Gate network для обчислення adaptive depth
            # Приймає hidden representation та повертає gate value [0, 1]
            self.adaptive_gate = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),  # Середнє по послідовності
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid()  # Gate value в [0, 1]
            )
        else:
            self.adaptive_gate = None
        
        # Логування рекурсії (для діагностики)
        self._recursion_depth_log = []
        self._log_recursion = False

    @property
    def device(self):
        return next(self.parameters()).device
    
    def compute_adaptive_depth(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Обчислити adaptive depth gate на основі hidden representation.
        
        Args:
            hidden: Hidden representation tensor [batch_size, seq_len, dim] або [batch_size, dim]
            
        Returns:
            Gate tensor [batch_size] зі значеннями в [0, 1], де:
            - 0 означає "зупинитись раніше" (менше думати)
            - 1 означає "продовжити думати" (більше думати)
        """
        if not self.adaptive_recursion or self.adaptive_gate is None:
            # Якщо adaptive recursion вимкнено, повернути фіксоване значення
            if hidden.dim() == 3:
                batch_size = hidden.shape[0]
            else:
                batch_size = hidden.shape[0]
            return torch.ones(batch_size, device=hidden.device, dtype=hidden.dtype)
        
        # Обчислити gate
        gate = self.adaptive_gate(hidden)  # [batch_size, 1]
        gate = gate.squeeze(-1)  # [batch_size]
        
        return gate
    
    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute entropy of logits distribution.
        
        Args:
            logits: Logits tensor [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            
        Returns:
            Entropy value (scalar float)
        """
        # Ensure we have proper shape
        if logits.dim() == 3:
            # Average over sequence dimension
            logits = logits.mean(dim=1)  # [batch_size, vocab_size]
        
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        return entropy.item() if isinstance(entropy, torch.Tensor) else entropy

    def get_initial(self):
        outputs = self.output_init_embed
        latents = self.latent_init_embed

        return outputs, latents

    def embed_inputs_with_registers(
        self,
        seq
    ):
        batch = seq.shape[0]

        inputs = self.input_embed(seq)

        # maybe registers

        registers = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        inputs, packed_shape = pack([registers, inputs], 'b * d')

        return inputs, packed_shape

    def refine_latent_then_output_once(
        self,
        inputs,     # (b n d)
        outputs,    # (b n d)
        latents,    # (b n d)
    ):

        # so it seems for this work, they use only one network
        # the network learns to refine the latents if input is passed in, otherwise it refines the output

        for _ in range(self.num_latent_refinements):

            latents = self.network(outputs + latents + inputs)

        outputs = self.network(outputs + latents)

        return outputs, latents

    def deep_refinement(
        self,
        inputs,    # (b n d)
        outputs,   # (b n d)
        latents,   # (b n d)
    ):

        for step in range_from_one(self.num_refinement_blocks):

            # only last round of refinement receives gradients

            is_last = step == self.num_refinement_blocks
            context = torch.no_grad if not is_last else nullcontext

            with context():
                outputs, latents = self.refine_latent_then_output_once(inputs, outputs, latents)

        return outputs, latents

    @torch.no_grad()
    def predict(
        self,
        seq,
        halt_prob_thres = 0.5,
        max_deep_refinement_steps = 12,
        timeout_seconds: Optional[float] = None
    ):
        batch = seq.shape[0]

        inputs, packed_shape = self.embed_inputs_with_registers(seq)

        # initial outputs and latents
        # expand to match input sequence length
        seq_len = inputs.shape[1]
        outputs, latents = self.get_initial()
        outputs = repeat(outputs, 'd -> b n d', b = batch, n = seq_len)
        latents = repeat(latents, 'd -> b n d', b = batch, n = seq_len)

        # active batch indices, the step it exited at, and the final output predictions

        active_batch_indices = arange(batch, device = self.device, dtype = torch.float32)

        preds = []
        exited_step_indices = []
        exited_batch_indices = []
        
        # Timeout tracking
        start_time = time.time() if timeout_seconds is not None else None

        for step in range_from_one(max_deep_refinement_steps):
            is_last = step == max_deep_refinement_steps
            
            # Check timeout
            if timeout_seconds is not None and start_time is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    # Timeout reached - force halt
                    should_halt = tensor([True] * outputs.shape[0], device=self.device, dtype=torch.bool)
                    if not should_halt.any():
                        continue
                    # Process remaining batch items
                    registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')
                    pred = self.to_pred(outputs_for_pred[should_halt])
                    preds.append(pred)
                    exited_step_indices.extend([step] * should_halt.sum().item())
                    exited_batch_indices.append(active_batch_indices[should_halt])
                    break

            outputs, latents = self.deep_refinement(inputs, outputs, latents)

            halt_prob = self.to_halt_pred(outputs)

            should_halt = (halt_prob >= halt_prob_thres) | is_last

            if not should_halt.any():
                continue

            # maybe remove registers

            registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')

            # append to exited predictions

            pred = self.to_pred(outputs_for_pred[should_halt])
            preds.append(pred)

            # append the step at which early halted

            exited_step_indices.extend([step] * should_halt.sum().item())

            # append indices for sorting back

            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue

            # ready for next round

            inputs = inputs[~should_halt]
            outputs = outputs[~should_halt]
            latents = latents[~should_halt]
            active_batch_indices = active_batch_indices[~should_halt]

            if is_empty(outputs):
                break

        preds = cat(preds).argmax(dim = -1)
        exited_step_indices = tensor(exited_step_indices)

        exited_batch_indices = cat(exited_batch_indices)
        sort_indices = exited_batch_indices.argsort(dim = -1)

        return preds[sort_indices], exited_step_indices[sort_indices]

    def forward(
        self,
        seq,
        outputs,
        latents,
        labels = None,
        current_depth: int = 0  # НОВИЙ: відстеження глибини рекурсії
    ):
        # GUARD: перевірка глибини рекурсії
        if current_depth >= self.max_recursion_depth:
            # Повернути базовий стан замість продовження
            inputs, packed_shape = self.embed_inputs_with_registers(seq)
            registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')
            pred = self.to_pred(outputs_for_pred)
            halt_prob = self.to_halt_pred(outputs)
            
            if labels is not None:
                # Створити dummy loss для стабільності (зменшений weight)
                loss = F.cross_entropy(rearrange(pred, 'b n l -> b l n'), labels, reduction = 'none')
                loss = reduce(loss, 'b ... -> b', 'mean') * 0.1  # Зменшений weight
                halt_loss = torch.zeros_like(loss)
                return (loss.sum(), (loss, halt_loss), outputs, latents, pred, halt_prob)
            
            return (outputs, latents, pred, halt_prob)
        
        # Логування глибини (тільки для діагностики)
        if self._log_recursion:
            self._recursion_depth_log.append(current_depth)

        inputs, packed_shape = self.embed_inputs_with_registers(seq)

        outputs, latents = self.deep_refinement(inputs, outputs, latents)

        registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')

        pred = self.to_pred(outputs_for_pred)

        halt_prob = self.to_halt_pred(outputs)

        outputs, latents = outputs.detach(), latents.detach()

        return_package = (outputs, latents, pred, halt_prob)

        if not exists(labels):
            return return_package

        # calculate loss if labels passed in

        loss = F.cross_entropy(rearrange(pred, 'b n l -> b l n'), labels, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        is_all_correct = (pred.argmax(dim = -1) == labels).all(dim = -1)

        halt_loss = F.binary_cross_entropy(halt_prob, is_all_correct.float(), reduction = 'none')

        # total loss and loss breakdown
        # Примітка: thinking cost додається в trainer, тут тільки base loss

        total_loss = (
            loss +
            halt_loss * self.halt_loss_weight
        )

        losses = (loss, halt_loss)

        return (total_loss.sum(), losses, *return_package)
    
    def compute_thinking_cost(self, recursion_state: Optional['RecursionState'] = None, iteration_count: Optional[int] = None) -> torch.Tensor:
        """
        Compute thinking cost from recursion state or iteration count.
        
        Args:
            recursion_state: Optional recursion state containing iteration info
            iteration_count: Optional iteration count (used if recursion_state is None)
            
        Returns:
            Thinking cost tensor (scalar)
        """
        # Lazy import to avoid circular dependency (only needed if recursion_state is used)
        if recursion_state is not None:
            cost = recursion_state.accumulated_cost
        elif iteration_count is not None:
            cost = float(iteration_count)
        else:
            cost = 0.0
        
        return torch.tensor(
            cost * self.thinking_cost_weight,
            dtype=torch.float32,
            device=self.device
        )
    
    def enable_recursion_logging(self):
        """Увімкнути логування глибини рекурсії"""
        self._log_recursion = True
        self._recursion_depth_log = []
    
    def get_recursion_stats(self) -> dict:
        """Отримати статистику рекурсії"""
        if not self._recursion_depth_log:
            return {'avg_depth': 0, 'max_depth': 0, 'count': 0}
        
        depths = self._recursion_depth_log
        return {
            'avg_depth': sum(depths) / len(depths),
            'max_depth': max(depths),
            'min_depth': min(depths),
            'count': len(depths)
        }
