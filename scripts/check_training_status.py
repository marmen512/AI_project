#!/usr/bin/env python3
"""
Скрипт перевірки статусу навчання з детальною інформацією
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def check_processes():
    """Перевірити чи працюють процеси навчання (підтримка обох архітектур)"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_model.py|runtime.bootstrap"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
            return pids
        return []
    except Exception as e:
        print(f"Помилка при перевірці процесів: {e}")
        return []

def get_process_info(pid):
    """Отримати інформацію про процес"""
    try:
        result = subprocess.run(
            ["ps", "-p", pid, "-o", "pid,etime,pcpu,pmem,vsz,rss,cmd"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 6:
                    return {
                        'pid': parts[0],
                        'etime': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'vsz': int(parts[4]) if parts[4].isdigit() else 0,
                        'rss': int(parts[5]) if parts[5].isdigit() else 0,
                        'cmd': ' '.join(parts[6:])
                    }
    except Exception as e:
        print(f"Помилка при отриманні інформації про процес {pid}: {e}")
    return None

def check_checkpoints():
    """Перевірити checkpoint'и"""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        return None
    
    latest = checkpoint_dir / "checkpoint_latest.pt"
    if latest.exists():
        stat = latest.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        # Спробувати прочитати інформацію з checkpoint
        checkpoint_info = {}
        try:
            import torch
            checkpoint = torch.load(latest, map_location='cpu')
            checkpoint_info = {
                'epoch': checkpoint.get('epoch', 0),
                'batch_idx': checkpoint.get('batch_idx', 0),
                'batch_count': checkpoint.get('batch_count', 0),
                'epochs': checkpoint.get('epochs', 0),
                'loss': checkpoint.get('loss', None),
                'is_final': checkpoint.get('is_final', False)
            }
        except Exception as e:
            # Якщо не вдалося прочитати (немає torch або помилка), просто пропустити
            pass
        
        return {
            'path': str(latest),
            'size_mb': size_mb,
            'mtime': mtime.strftime('%Y-%m-%d %H:%M:%S'),
            'info': checkpoint_info
        }
    return None

def check_logs():
    """Перевірити логи"""
    logs = []
    
    # Перевірити logs/training_latest.log (символічне посилання)
    logs_dir = Path("logs")
    if logs_dir.exists():
        latest_log_link = logs_dir / "training_latest.log"
        if latest_log_link.exists():
            stat = latest_log_link.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            try:
                with open(latest_log_link, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    last_lines = lines[-10:] if len(lines) > 10 else lines
            except:
                last_lines = []
            
            logs.append({
                'path': str(latest_log_link),
                'size_mb': size_mb,
                'mtime': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                'last_lines': last_lines
            })
        
        # Перевірити logs/training_*.log (останній з timestamp)
        log_files = sorted(logs_dir.glob("training_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            # Пропустити training_latest.log якщо вже додано
            for log_file in log_files:
                if log_file.name == "training_latest.log":
                    continue
                stat = log_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        last_lines = lines[-10:] if len(lines) > 10 else lines
                except:
                    last_lines = []
                
                logs.append({
                    'path': str(log_file),
                    'size_mb': size_mb,
                    'mtime': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                    'last_lines': last_lines
                })
                break  # Тільки останній файл
    
    return logs

def main():
    """Головна функція"""
    print("=" * 60)
    print("🔍 ПЕРЕВІРКА СТАТУСУ НАВЧАННЯ")
    print("=" * 60)
    print()
    
    # Перевірити процеси
    pids = check_processes()
    
    if pids:
        print("✅ Навчання працює")
        print()
        print("📋 Процеси:")
        for pid in pids:
            info = get_process_info(pid)
            if info:
                print(f"   PID: {info['pid']}")
                print(f"   Час роботи: {info['etime']}")
                print(f"   CPU: {info['cpu']}%")
                print(f"   Пам'ять: {info['mem']}% ({info['rss']//1024} MB)")
                print(f"   Команда: {info['cmd'][:80]}...")
                print()
    else:
        print("❌ Навчання не працює")
        print()
    
    # Перевірити checkpoint'и
    checkpoint = check_checkpoints()
    if checkpoint:
        print("💾 Останній checkpoint:")
        print(f"   Файл: {checkpoint['path']}")
        print(f"   Розмір: {checkpoint['size_mb']:.2f} MB")
        print(f"   Час: {checkpoint['mtime']}")
        
        # Показати інформацію з checkpoint якщо доступна
        if checkpoint.get('info'):
            info = checkpoint['info']
            if info.get('epoch') is not None:
                epoch = info['epoch']
                epochs = info.get('epochs', 0)
                batch_idx = info.get('batch_idx', 0)
                batch_count = info.get('batch_count', 0)
                loss = info.get('loss')
                
                print(f"   📊 Прогрес:")
                print(f"      Епоха: {epoch}/{epochs}")
                print(f"      Батч в епосі: {batch_idx}")
                print(f"      Всього батчів: {batch_count}")
                if loss is not None:
                    print(f"      Loss: {loss:.6f}")
                
                # Розрахувати відсоток прогресу
                if epochs > 0 and batch_count > 0:
                    # Припускаємо ~1800 батчів на епоху (27000 / 15)
                    batches_per_epoch = 1800
                    total_batches_expected = epochs * batches_per_epoch
                    if total_batches_expected > 0:
                        progress_pct = (batch_count / total_batches_expected) * 100
                        print(f"      Прогрес: {progress_pct:.1f}%")
        print()
    
    # Перевірити логи
    logs = check_logs()
    if logs:
        for log in logs:
            print(f"📝 Лог файл: {log['path']}")
            print(f"   Розмір: {log['size_mb']:.2f} MB")
            print(f"   Оновлено: {log['mtime']}")
            if log['last_lines']:
                print("   Останні рядки:")
                # Шукати рядки з прогресом
                progress_lines = [line for line in log['last_lines'] 
                                 if '📊 Прогрес' in line or 'Прогрес:' in line or 
                                    'Епоха' in line or 'Loss:' in line or 
                                    'loss:' in line.lower()]
                if progress_lines:
                    # Показати останній рядок з прогресом
                    print(f"   {progress_lines[-1].rstrip()}")
                else:
                    # Показати останні рядки
                    for line in log['last_lines'][-3:]:
                        print(f"   {line.rstrip()}")
            print()
    
    print("=" * 60)

if __name__ == "__main__":
    main()

