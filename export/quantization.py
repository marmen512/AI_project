"""
Quantization utilities for GGUF export
Підтримка квантизації моделей (Q4, Q5, Q8)
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional


def quantize_q4(weights: torch.Tensor) -> tuple:
    """
    Q4_K_M квантизація (4-bit)
    
    Args:
        weights: Tensor з вагами
    
    Returns:
        Tuple (quantized_data, scale, zero_point)
    """
    # Q4_K_M квантизація
    # Мінімальне та максимальне значення
    w_min = weights.min()
    w_max = weights.max()
    
    # Масштаб та zero point
    scale = (w_max - w_min) / 15.0  # 4-bit = 16 значень (0-15)
    zero_point = torch.round(-w_min / scale).clamp(0, 15)
    
    # Квантизація
    quantized = torch.round(weights / scale + zero_point).clamp(0, 15).to(torch.uint8)
    
    return quantized, scale, zero_point


def quantize_q5(weights: torch.Tensor) -> tuple:
    """
    Q5_K_M квантизація (5-bit)
    
    Args:
        weights: Tensor з вагами
    
    Returns:
        Tuple (quantized_data, scale, zero_point)
    """
    # Q5_K_M квантизація
    w_min = weights.min()
    w_max = weights.max()
    
    # Масштаб та zero point для 5-bit
    scale = (w_max - w_min) / 31.0  # 5-bit = 32 значення (0-31)
    zero_point = torch.round(-w_min / scale).clamp(0, 31)
    
    # Квантизація
    quantized = torch.round(weights / scale + zero_point).clamp(0, 31).to(torch.uint8)
    
    return quantized, scale, zero_point


def quantize_q8(weights: torch.Tensor) -> tuple:
    """
    Q8_0 квантизація (8-bit)
    
    Args:
        weights: Tensor з вагами
    
    Returns:
        Tuple (quantized_data, scale, zero_point)
    """
    # Q8_0 квантизація
    w_min = weights.min()
    w_max = weights.max()
    
    # Масштаб та zero point для 8-bit
    scale = (w_max - w_min) / 255.0  # 8-bit = 256 значень (0-255)
    zero_point = torch.round(-w_min / scale).clamp(0, 255)
    
    # Квантизація
    quantized = torch.round(weights / scale + zero_point).clamp(0, 255).to(torch.uint8)
    
    return quantized, scale, zero_point


def quantize_model(
    model_path: str,
    output_path: str,
    quantization_type: str = "q4"
):
    """
    Квантизувати модель
    
    Args:
        model_path: Шлях до PyTorch checkpoint
        output_path: Шлях для збереження квантизованої моделі
        quantization_type: Тип квантизації ("q4", "q5", "q8")
    """
    # Завантажити модель
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Вибрати функцію квантизації
    quantize_func = {
        "q4": quantize_q4,
        "q5": quantize_q5,
        "q8": quantize_q8
    }.get(quantization_type.lower())
    
    if quantize_func is None:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    # Квантизувати всі ваги
    quantized_state_dict = {}
    quantization_metadata = {}
    
    for name, param in state_dict.items():
        if param.dtype in [torch.float32, torch.float16]:
            quantized, scale, zero_point = quantize_func(param)
            
            quantized_state_dict[f"{name}.quantized"] = quantized
            quantized_state_dict[f"{name}.scale"] = scale
            quantized_state_dict[f"{name}.zero_point"] = zero_point
            
            quantization_metadata[name] = {
                "original_shape": list(param.shape),
                "original_dtype": str(param.dtype),
                "quantization_type": quantization_type
            }
        else:
            # Не квантизувати non-float параметри
            quantized_state_dict[name] = param
    
    # Зберегти квантизовану модель
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    quantized_checkpoint = {
        'quantized_state_dict': quantized_state_dict,
        'quantization_metadata': quantization_metadata,
        'quantization_type': quantization_type
    }
    
    torch.save(quantized_checkpoint, output_path)
    
    # Обчислити стиснення
    original_size = sum(p.numel() * 4 for p in state_dict.values() if p.dtype in [torch.float32, torch.float16])
    quantized_size = sum(
        p.numel() * (1 if quantization_type == "q4" else (1.5 if quantization_type == "q5" else 2))
        for p in quantized_state_dict.values()
        if 'quantized' in str(p) or isinstance(p, torch.Tensor)
    )
    
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
    
    print(f"✅ Quantization completed:")
    print(f"   Type: {quantization_type}")
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    print(f"   Saved to: {output_path}")
    
    return output_path


def dequantize(quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    Де-квантизувати ваги
    
    Args:
        quantized: Квантизовані дані
        scale: Масштаб
        zero_point: Zero point
    
    Returns:
        Де-квантизовані ваги
    """
    return (quantized.float() - zero_point.float()) * scale.float()

