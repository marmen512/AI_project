"""
Transformer backbone для TRM (замість MLPMixer1D)
Використовує готовий GPT-2 для кращого розуміння
"""
import torch
from torch import nn
from typing import Optional
from pathlib import Path
from transformers import GPT2Model, GPT2Config


class TransformerBackbone(nn.Module):
    """
    Обгортка навколо GPT-2 для використання в TRM
    Замінює MLPMixer1D, але зберігає сумісність з TRM
    
    Args:
        dim: Розмірність embeddings (буде оновлено з GPT-2 якщо pretrained=True)
        depth: Глибина Transformer (буде оновлено з GPT-2 якщо pretrained=True)
        seq_len: Максимальна довжина послідовності
        pretrained: Чи завантажити готовий GPT-2
        model_name: Назва моделі ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        cache_dir: Де зберігати завантажені моделі
    """
    def __init__(
        self,
        dim: int = 768,  # GPT-2 small розмір
        depth: int = 12,  # GPT-2 small depth
        seq_len: int = 1024,
        pretrained: bool = True,
        model_name: str = "gpt2",  # "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.seq_len = seq_len
        self.model_name = model_name
        self.pretrained = pretrained
        
        if cache_dir is None:
            # Автоматично визначити cache_dir
            project_root = Path(__file__).parent.parent.parent
            cache_dir = str(project_root / "models" / "pretrained")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if pretrained:
            # Завантажити готовий GPT-2
            try:
                print(f"[LOAD] Завантаження GPT-2 моделі: {model_name}...")
                self.transformer = GPT2Model.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir)
                )
                # Оновити dim та depth з реальної моделі
                self.dim = self.transformer.config.n_embd
                self.depth = self.transformer.config.n_layer
                # Оновити seq_len якщо потрібно
                max_positions = self.transformer.config.n_positions
                if seq_len > max_positions:
                    print(f"[WARN] Запитаний seq_len={seq_len} більший за максимальний {max_positions}, використовується {max_positions}")
                    self.seq_len = max_positions
                else:
                    self.seq_len = seq_len
                
                print(f"[OK] GPT-2 завантажено: dim={self.dim}, depth={self.depth}, seq_len={self.seq_len}")
            except Exception as e:
                print(f"[WARN] Помилка завантаження GPT-2: {e}")
                print(f"   Створюється GPT-2 з нуля...")
                # Fallback: створити GPT-2 з нуля
                self._create_from_scratch(dim, depth, seq_len)
        else:
            # Створити GPT-2 з нуля
            self._create_from_scratch(dim, depth, seq_len)
    
    def _create_from_scratch(self, dim: int, depth: int, seq_len: int):
        """Створити GPT-2 з нуля"""
        config = GPT2Config(
            vocab_size=50257,  # Будемо ігнорувати (TRM має свій vocab)
            n_positions=seq_len,
            n_embd=dim,
            n_layer=depth,
            n_head=12,  # Стандарт для GPT-2 (dim // 64)
            n_inner=dim * 4,  # FFN розмір
            activation_function="gelu_new"
        )
        self.transformer = GPT2Model(config)
        self.dim = dim
        self.depth = depth
        self.seq_len = seq_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через Transformer
        
        Args:
            x: Tensor [batch_size, seq_len, dim] - embeddings
        
        Returns:
            Tensor [batch_size, seq_len, dim] - hidden states
        """
        # GPT-2 очікує input_ids, але ми передаємо embeddings
        # Тому використовуємо inputs_embeds
        outputs = self.transformer(
            inputs_embeds=x,
            return_dict=True
        )
        
        # Повернути hidden states
        return outputs.last_hidden_state

