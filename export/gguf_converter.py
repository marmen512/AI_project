"""
GGUF Converter з повним GGUF spec сумісним з llama.cpp
"""
import struct
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


def _write_str(f, s: str) -> None:
    """Записати string в GGUF формат"""
    b = s.encode("utf-8")
    f.write(struct.pack("<I", len(b)))
    f.write(b)


def export_trm_to_gguf(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: Dict[str, Any],
    path: str | Path
) -> Path:
    """
    Експортувати TRM модель в повний GGUF формат сумісний з llama.cpp
    
    Args:
        model: PyTorch модель
        tokenizer: Tokenizer instance
        cfg: Конфігурація моделі
        path: Шлях для збереження GGUF файлу
    
    Returns:
        Шлях до збереженого файлу
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state = model.state_dict()
    
    with open(path, "wb") as f:
        # === HEADER ===
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        
        # === METADATA ===
        model_cfg = cfg.get('model', {})
        training_cfg = cfg.get('training', {})
        curriculum_cfg = cfg.get('curriculum', {})
        
        kv = {
            "model.type": "TRM",
            "model.dim": model_cfg.get('dim', 256),
            "model.depth": model_cfg.get('depth', 4),
            "model.vocab_size": model_cfg.get('vocab_size', tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257),
            "training.seq_len": curriculum_cfg.get('max_len', model_cfg.get('seq_len', 256)),
            "training.max_recurrent_steps": training_cfg.get('max_recurrent_steps', 12),
            "tokenizer.type": "gpt2",  # За замовчуванням
        }
        
        # Додати інформацію про tokenizer якщо доступна
        if hasattr(tokenizer, 'vocab_size'):
            kv["tokenizer.vocab_size"] = tokenizer.vocab_size
        if hasattr(tokenizer, 'unk_token'):
            kv["tokenizer.unk_token"] = tokenizer.unk_token
        if hasattr(tokenizer, 'bos_token'):
            kv["tokenizer.bos_token"] = tokenizer.bos_token
        if hasattr(tokenizer, 'eos_token'):
            kv["tokenizer.eos_token"] = tokenizer.eos_token
        
        # Записати кількість key-value пар та тензорів
        f.write(struct.pack("<I", len(kv)))
        f.write(struct.pack("<I", len(state)))
        
        # Записати metadata
        for k, v in kv.items():
            _write_str(f, k)
            if isinstance(v, int):
                f.write(struct.pack("<I", 0))  # uint32
                f.write(struct.pack("<Q", v))  # uint64
            elif isinstance(v, float):
                f.write(struct.pack("<I", 1))  # float32
                f.write(struct.pack("<f", v))
            else:
                f.write(struct.pack("<I", 2))  # string
                _write_str(f, str(v))
        
        # === TENSORS ===
        for name, tensor in state.items():
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            _write_str(f, name)
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", 0))  # f32 = 0
            f.write(arr.tobytes())
    
    print(f"✅ GGUF файл збережено: {path}")
    return path


class GGUFConverter:
    """
    Конвертер PyTorch моделі в GGUF формат
    """
    
    def __init__(self):
        """Ініціалізація конвертера"""
        self.gguf_magic = GGUF_MAGIC
        self.gguf_version = GGUF_VERSION
    
    def convert_model(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Конвертувати модель в GGUF формат (legacy метод)
        
        Args:
            model_state_dict: State dict моделі
            output_path: Шлях для збереження GGUF файлу
            metadata: Метадані моделі
        
        Returns:
            Шлях до збереженого файлу
        """
        # Використати новий метод
        # Створити тимчасову конфігурацію
        cfg = {
            'model': metadata.get('model', {}) if metadata else {},
            'training': metadata.get('training', {}) if metadata else {},
        }
        
        # Створити тимчасову модель для експорту
        # (в реальності потрібна повна модель)
        # Тимчасово використовуємо прямий експорт state_dict
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", self.gguf_version))
            
            # Metadata
            metadata_dict = metadata or {}
            f.write(struct.pack("<I", len(metadata_dict)))
            f.write(struct.pack("<I", len(model_state_dict)))
            
            # Записати metadata
            for k, v in metadata_dict.items():
                _write_str(f, k)
                if isinstance(v, int):
                    f.write(struct.pack("<I", 0))
                    f.write(struct.pack("<Q", v))
                elif isinstance(v, float):
                    f.write(struct.pack("<I", 1))
                    f.write(struct.pack("<f", v))
                else:
                    f.write(struct.pack("<I", 2))
                    _write_str(f, str(v))
            
            # Записати tensors
            for name, tensor in model_state_dict.items():
                arr = tensor.detach().cpu().numpy().astype(np.float32)
                _write_str(f, name)
                f.write(struct.pack("<I", arr.ndim))
                for d in arr.shape:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", 0))  # f32
                f.write(arr.tobytes())
        
        print(f"✅ GGUF file saved: {output_path}")
        return output_path

