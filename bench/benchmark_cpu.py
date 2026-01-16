"""
On-device benchmark для TRM моделі
Реальні метрики: tokens/sec, latency, memory
"""
import time
import psutil
import torch
from typing import Optional, Any
from pathlib import Path


def benchmark(
    model: torch.nn.Module,
    tokenizer: Any,
    seq: int = 256,
    runs: int = 10,
    device: str = "cpu"
) -> dict:
    """
    Запустити benchmark на моделі
    
    Args:
        model: TRM модель
        tokenizer: Tokenizer instance
        seq: Довжина послідовності для тестування
        runs: Кількість запусків
        device: Пристрій для тестування
    
    Returns:
        Словник з метриками
    """
    print(f"🔬 Запуск benchmark: seq={seq}, runs={runs}, device={device}")
    
    # Підготувати вхідні дані
    test_text = "hello world " * 10
    tokens = tokenizer.encode(test_text)[:seq]
    tokens_tensor = torch.tensor([tokens], device=device)
    
    # Перемістити модель на device
    model = model.to(device)
    model.eval()
    
    # Виміряти пам'ять до тестування
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1e9  # GB
    
    # Warmup
    print("   Warmup...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(tokens_tensor)
    
    # Benchmark
    print("   Benchmarking...")
    times = []
    with torch.no_grad():
        for i in range(runs):
            t0 = time.time()
            output = model(tokens_tensor)
            elapsed = time.time() - t0
            times.append(elapsed)
            if (i + 1) % 5 == 0:
                print(f"   Run {i+1}/{runs}: {elapsed:.4f}s")
    
    # Виміряти пам'ять після тестування
    memory_after = process.memory_info().rss / 1e9  # GB
    memory_used = memory_after - memory_before
    
    # Обчислити метрики
    avg_latency = sum(times) / len(times)
    min_latency = min(times)
    max_latency = max(times)
    tokens_per_sec = seq / avg_latency
    
    results = {
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'tokens_per_sec': tokens_per_sec,
        'memory_used_gb': memory_used,
        'memory_before_gb': memory_before,
        'memory_after_gb': memory_after,
        'runs': runs,
        'seq_len': seq,
    }
    
    # Вивести результати
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"   Avg latency: {avg_latency:.4f}s")
    print(f"   Min latency: {min_latency:.4f}s")
    print(f"   Max latency: {max_latency:.4f}s")
    print(f"   Tokens/sec: {tokens_per_sec:.2f}")
    print(f"   Memory used: {memory_used:.2f} GB")
    print(f"   Memory total: {memory_after:.2f} GB")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Тестовий запуск
    from tiny_recursive_model.utils import load_tokenizer
    from train.model_factory import create_model
    
    print("🔬 Тестовий benchmark")
    
    # Завантажити tokenizer
    tokenizer, vocab_size, _ = load_tokenizer("gpt2")
    
    # Створити модель
    model = create_model(
        dim=256,
        vocab_size=vocab_size,
        depth=4,
        seq_len=256
    )
    
    # Запустити benchmark
    results = benchmark(model, tokenizer, seq=256, runs=10)
    print(f"\n✅ Benchmark завершено: {results['tokens_per_sec']:.2f} tokens/sec")

