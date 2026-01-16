#!/usr/bin/env python3
"""
ФАЗА 2 - Instruction Tuning Script
Навчання моделі після ФАЗИ 1 на instruction datasets

КРИТИЧНІ ВИМОГИ:
- Завантажити weights з ФАЗИ 1 (НЕ random initialization)
- Instruction datasets (Alpaca, SQuAD, DailyDialog)
- Максимум 1-2 epochs
- Зупинка при погіршенні якості
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import yaml
import json
from tqdm import tqdm
import logging
import os
import time
import itertools
import signal

# Додати project root до sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def _atomic_torch_save(obj, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def _save_phase2_checkpoint(
    checkpoint_path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    config: dict,
    tokenizer_vocab_size: int,
    global_step: int,
    batch_idx: int,
    epoch_completed: bool = False,
    epoch_loss_sum: float = 0.0,
    epoch_loss_count: int = 0,
    is_emergency: bool = False,
):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(loss),
        'config': config,
        'phase': 2,
        'tokenizer_vocab_size': int(tokenizer_vocab_size),
        'train_state': {
            'global_step': int(global_step),
            'batch_idx': int(batch_idx),
            'epoch_completed': bool(epoch_completed),
            'epoch_loss_sum': float(epoch_loss_sum),
            'epoch_loss_count': int(epoch_loss_count),
            'is_emergency': bool(is_emergency),
        },
    }

    _atomic_torch_save(payload, checkpoint_path)

class InstructionDataset(Dataset):
    """
    Dataset для ФАЗИ 2 - Instruction Tuning (CORRECT)
    - Instruction + Input = context (labels masked)
    - Output = supervised target
    """

    def __init__(self, data_files: list, tokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        print("📚 Завантаження instruction datasets (safe mode)...")

        for data_file in data_files:
            print(f"   📖 {data_file}")

            with open(data_file, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, dict) and "data" in raw:
                data = raw["data"]
            elif isinstance(raw, list):
                data = raw
            else:
                print(f"   ⚠️ Невідомий формат: {data_file}")
                continue

            added = 0
            for sample in data:
                if self._is_valid(sample):
                    self.samples.append(sample)
                    added += 1

            print(f"      ✅ Додано {added:,} samples")

        print(f"   📊 Загалом instruction samples: {len(self.samples):,}")

    def _is_valid(self, sample):
        return (
            isinstance(sample, dict)
            and "instruction" in sample
            and "output" in sample
            and len(sample["instruction"].strip()) > 0
            and len(sample["output"].strip()) > 0
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        instruction = sample["instruction"].strip()
        input_text = sample.get("input", "").strip()
        output = sample["output"].strip()

        # Unified template: if input exists, prepend to instruction
        if input_text:
            instruction = f"{instruction}\n\n{input_text}"

        prompt = f"Instruction:\n{instruction}\n\nResponse:\n"

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,
        )["input_ids"]

        remaining = max(self.max_seq_len - len(prompt_ids), 0)
        output_ids = self.tokenizer(
            output,
            add_special_tokens=False,
            truncation=True,
            max_length=remaining,
        )["input_ids"]

        # Concatenate and mask: prompt tokens = -100, response tokens = real IDs
        input_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids

        # Truncate to max_seq_len
        input_ids = input_ids[: self.max_seq_len]
        labels = labels[: self.max_seq_len]

        # Pad with PAD token and mask labels
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def load_phase1_model(checkpoint_path: str, config, tokenizer):
    """Завантажити модель з ФАЗИ 1 та розширити embeddings для PAD токена"""
    print(f"🔄 Завантаження моделі з ФАЗИ 1: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint ФАЗИ 1 не знайдено: {checkpoint_path}")
    
    # Завантажити checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Phase1 може зберігати або чистий state_dict (best_model.pt),
    # або dict з model_state_dict (last/emergency).
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        phase1_state_dict = checkpoint['model_state_dict']
    else:
        phase1_state_dict = checkpoint
    
    # Створити модель з тією ж конфігурацією (спочатку з оригінальним vocab_size)
    original_vocab_size = 50257  # GPT-2 vocab size
    model_config = GPT2Config(
        vocab_size=original_vocab_size,
        n_positions=config['model']['seq_len'],
        n_embd=config['model']['dim'],
        n_layer=config['model']['depth'],
        n_head=int(config.get('model', {}).get('heads', 8)),
        n_inner=config['model']['dim'] * 4,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False
    )
    
    model = GPT2LMHeadModel(model_config)
    
    # Завантажити weights з ФАЗИ 1
    load_result = model.load_state_dict(phase1_state_dict, strict=False)

    missing = getattr(load_result, 'missing_keys', [])
    unexpected = getattr(load_result, 'unexpected_keys', [])
    if missing or unexpected:
        print("   ⚠️ УВАГА: неідеальний match ваг при завантаженні Phase1 → Phase2")
        if missing:
            print(f"   ⚠️ missing_keys: {len(missing)}")
        if unexpected:
            print(f"   ⚠️ unexpected_keys: {len(unexpected)}")

    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"   ✅ Модель завантажена з epoch {checkpoint.get('epoch')}")
        if 'loss' in checkpoint:
            try:
                print(f"   📊 Loss з ФАЗИ 1: {float(checkpoint['loss']):.4f}")
            except Exception:
                pass
    else:
        print("   ✅ Модель завантажена (Phase1 state_dict)")
    
    # КРИТИЧНО: Розширити embeddings якщо додано PAD токен
    current_vocab_size = len(tokenizer)
    if current_vocab_size > original_vocab_size:
        print(f"   🔧 Розширення embeddings: {original_vocab_size} → {current_vocab_size}")
        
        # Розширити input embeddings
        old_embeddings = model.transformer.wte.weight.data
        new_embeddings = torch.zeros(current_vocab_size, model.config.n_embd)
        new_embeddings[:original_vocab_size] = old_embeddings
        
        # Ініціалізувати нові токени (PAD) середнім значенням існуючих embeddings
        new_embeddings[original_vocab_size:] = old_embeddings.mean(dim=0, keepdim=True)
        
        # Замінити embeddings
        model.transformer.wte = nn.Embedding(current_vocab_size, model.config.n_embd)
        model.transformer.wte.weight.data = new_embeddings
        
        # Розширити output layer (lm_head)
        old_lm_head = model.lm_head.weight.data
        new_lm_head = torch.zeros(current_vocab_size, model.config.n_embd)
        new_lm_head[:original_vocab_size] = old_lm_head
        new_lm_head[original_vocab_size:] = old_lm_head.mean(dim=0, keepdim=True)
        
        model.lm_head = nn.Linear(model.config.n_embd, current_vocab_size, bias=False)
        model.lm_head.weight.data = new_lm_head
        
        # Оновити конфігурацію
        model.config.vocab_size = current_vocab_size
        
        print(f"   ✅ Embeddings розширено успішно")
    
    return model

def setup_logging(log_dir: str, phase: str = "phase2"):
    """Налаштувати логування"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = log_dir / f"{phase}_instruction_tuning.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def run_sanity_inference(model, tokenizer, device, step: int, logger):
    """Run online sanity inference every N steps to detect collapse."""
    sanity_prompts = [
        "What is 2 + 2?",
        "The capital of France is",
        "Give three tips for staying healthy."
    ]
    model.eval()
    for p in sanity_prompts:
        prompt = f"Instruction:\n{p}\n\nResponse:\n"
        enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=False, truncation=True, max_length=256)
        input_ids = enc['input_ids'].to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = gen_ids[0]
        completion_ids = gen_ids[int(input_ids.shape[1]):]
        out = tokenizer.decode(completion_ids, skip_special_tokens=True)
        logger.info(f"[Sanity step {step}] {p} -> {out}")
    model.train()

def is_collapsed_output(text: str) -> bool:
    """Detect collapse: >50% non-alphanumeric OR repeated single chars."""
    if not text:
        return True
    alnum_count = sum(c.isalnum() or c.isspace() for c in text)
    if alnum_count / len(text) < 0.5:
        return True
    # Detect repeated single characters (e.g., "22222", "aaaaa")
    if len(set(text.strip())) == 1 and len(text.strip()) > 3:
        return True
    return False

def train_epoch(
    model,
    dataloader,
    optimizer,
    tokenizer,
    device,
    epoch,
    logger,
    *,
    checkpoint_dir: Path,
    config: dict,
    tokenizer_vocab_size: int,
    global_step: int,
    save_every_steps: int,
    resume_batch_idx: int,
    resume_epoch_loss_sum: float,
    resume_epoch_loss_count: int,
):
    """Навчання одного epoch для instruction tuning"""
    model.train()
    total_loss = float(resume_epoch_loss_sum)
    num_batches = len(dataloader)

    start_batch_idx = int(resume_batch_idx)
    if start_batch_idx < -1:
        start_batch_idx = -1
    if start_batch_idx >= num_batches:
        start_batch_idx = -1

    if start_batch_idx >= 0:
        logger.info(
            "🔄 Exact resume: epoch=%s start_batch_idx=%s/%s (will continue from next batch)",
            epoch,
            start_batch_idx,
            num_batches,
        )
        dataloader_iter = itertools.islice(dataloader, start_batch_idx + 1, None)
        progress_bar = tqdm(
            dataloader_iter,
            desc=f"Phase 2 Epoch {epoch}",
            total=num_batches,
            initial=start_batch_idx + 1,
        )
        batch_enumerate_start = start_batch_idx + 1
    else:
        progress_bar = tqdm(dataloader, desc=f"Phase 2 Epoch {epoch}")
        batch_enumerate_start = 0

    batch_time_start = time.time()
    last_log_time = time.time()
    last_log_step = global_step

    training_cfg = config.get('training', {})
    loss_guard_enabled = bool(training_cfg.get('loss_guard_enabled', False))
    loss_guard_ema_beta = float(training_cfg.get('loss_guard_ema_beta', 0.98))
    loss_guard_warmup_steps = int(training_cfg.get('loss_guard_warmup_steps', 200))
    loss_guard_threshold_ratio = float(training_cfg.get('loss_guard_threshold_ratio', 0.15))
    loss_guard_patience_steps = int(training_cfg.get('loss_guard_patience_steps', 100))

    loss_ema = None
    best_loss_ema = None
    guard_bad_steps = 0
    local_step = 0
    guard_activated_logged = False

    skipped_all_ignored_batches = 0

    # Count of processed batches in this epoch so far (including resumed part)
    total_seen_batches = int(resume_epoch_loss_count)
    if total_seen_batches < 0:
        total_seen_batches = 0

    # Gradient accumulation and clipping
    gradient_accumulation_steps = int(training_cfg.get('gradient_accumulation_steps', 8))
    sanity_interval = int(training_cfg.get('sanity_interval', 100))
    max_grad_norm = float(training_cfg.get('max_grad_norm', 1.0))

    optimizer.zero_grad()
    accum_steps = 0
    collapse_consecutive = 0

    for batch_idx, batch in enumerate(progress_bar, start=batch_enumerate_start):
        try:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Skip batches with no supervision (all labels -100)
            if not torch.any(labels != -100):
                skipped_all_ignored_batches += 1
                if skipped_all_ignored_batches <= 10 or skipped_all_ignored_batches % 100 == 0:
                    logger.warning(
                        "⚠️ Skipping batch with all labels=-100 (no supervision): epoch=%s batch=%s/%s skipped=%s",
                        epoch,
                        batch_idx,
                        num_batches,
                        skipped_all_ignored_batches,
                    )
                continue

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps  # Scale loss

            # Early stop on NaN / Inf
            if not torch.isfinite(loss):
                logger.error("❌ NaN/Inf loss detected — emergency save and stop")
                _save_phase2_checkpoint(
                    checkpoint_dir / "emergency_checkpoint.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    loss=loss.item() if hasattr(loss, "item") else float("inf"),
                    config=config,
                    tokenizer_vocab_size=tokenizer_vocab_size,
                    global_step=global_step,
                    batch_idx=batch_idx,
                    epoch_completed=False,
                    is_emergency=True,
                )
                raise RuntimeError("NaN/Inf loss detected")

            # Backward pass
            loss.backward()
            accum_steps += 1

            # Step after accumulation
            if accum_steps % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            local_step += 1
            total_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
            total_seen_batches += 1
            avg_loss = total_loss / max(total_seen_batches, 1)

            # Online sanity inference
            if global_step % sanity_interval == 0:
                run_sanity_inference(model, tokenizer, device, global_step, logger)
                # Simple collapse detection based on last logged outputs (could be enhanced)
                # Here we rely on loss guard and manual inspection; for strict auto-stop, integrate is_collapsed_output

            # EMA loss guard (unchanged)
            if loss_guard_enabled:
                loss_val = float(loss.item())
                if loss_ema is None:
                    loss_ema = loss_val
                else:
                    loss_ema = loss_guard_ema_beta * loss_ema + (1.0 - loss_guard_ema_beta) * loss_val

                if local_step >= loss_guard_warmup_steps:
                    if not guard_activated_logged:
                        logger.info(
                            "🛡️ Loss guard active (local_step=%s, warmup_steps=%s)",
                            local_step,
                            loss_guard_warmup_steps,
                        )
                        guard_activated_logged = True

                    if best_loss_ema is None:
                        best_loss_ema = loss_ema
                    else:
                        best_loss_ema = min(best_loss_ema, loss_ema)

                    degrade_ratio = (loss_ema - best_loss_ema) / max(best_loss_ema, 1e-8)
                    if degrade_ratio > loss_guard_threshold_ratio:
                        guard_bad_steps += 1
                        if guard_bad_steps % 25 == 0:
                            logger.warning(
                                "⚠️ Loss guard: step=%s ema=%.4f best_ema=%.4f degrade=%.1f%% bad_steps=%s/%s",
                                global_step,
                                loss_ema,
                                best_loss_ema,
                                100.0 * degrade_ratio,
                                guard_bad_steps,
                                loss_guard_patience_steps,
                            )
                    else:
                        guard_bad_steps = 0

                    if guard_bad_steps >= loss_guard_patience_steps:
                        logger.error(
                            "❌ Loss guard triggered — emergency save and stop (ema=%.4f best_ema=%.4f degrade=%.1f%%)",
                            loss_ema,
                            best_loss_ema,
                            100.0 * degrade_ratio,
                        )
                        _save_phase2_checkpoint(
                            checkpoint_dir / "emergency_checkpoint.pt",
                            epoch=epoch,
                            model=model,
                            optimizer=optimizer,
                            loss=loss_ema,
                            config=config,
                            tokenizer_vocab_size=tokenizer_vocab_size,
                            global_step=global_step,
                            batch_idx=batch_idx,
                            epoch_completed=False,
                            epoch_loss_sum=total_loss,
                            epoch_loss_count=total_seen_batches,
                            is_emergency=True,
                        )
                        raise RuntimeError("Loss guard triggered")

            # Оновити progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })

            # Periodic autosave (power-outage resilience)
            if save_every_steps > 0 and (global_step % save_every_steps == 0):
                _save_phase2_checkpoint(
                    checkpoint_dir / "last_checkpoint.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    loss=avg_loss,
                    config=config,
                    tokenizer_vocab_size=tokenizer_vocab_size,
                    global_step=global_step,
                    batch_idx=batch_idx,
                    epoch_completed=False,
                    epoch_loss_sum=total_loss,
                    epoch_loss_count=total_seen_batches,
                    is_emergency=False,
                )
                logger.info(
                    f"💾 Autosave: epoch={epoch} step={global_step} batch={batch_idx}/{num_batches} avg_loss={avg_loss:.4f}"
                )

            # Логування (часове + по кроках)
            if batch_idx % 50 == 0:
                now = time.time()
                dt = max(now - last_log_time, 1e-9)
                dsteps = max(global_step - last_log_step, 1)
                steps_per_sec = dsteps / dt
                tokens_per_step = int(input_ids.numel())
                tok_per_sec = tokens_per_step * steps_per_sec
                eta_steps = max(num_batches - (batch_idx + 1), 0)
                eta_sec = eta_steps / max(steps_per_sec, 1e-9)

                current_lr = optimizer.param_groups[0].get('lr', None)
                lr_str = f"{current_lr:.3e}" if isinstance(current_lr, float) else str(current_lr)

                logger.info(
                    "Phase 2 | epoch=%s batch=%s/%s step=%s lr=%s loss=%.4f avg_loss=%.4f tok/s=%.0f eta=%.1fmin",
                    epoch,
                    batch_idx,
                    num_batches,
                    global_step,
                    lr_str,
                    loss.item(),
                    avg_loss,
                    tok_per_sec,
                    eta_sec / 60.0,
                )
                last_log_time = now
                last_log_step = global_step
        except KeyboardInterrupt:
            logger.warning("🛑 KeyboardInterrupt inside train loop — emergency save")
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except Exception:
                pass
            _save_phase2_checkpoint(
                checkpoint_dir / "emergency_checkpoint.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=total_loss / max(batch_idx + 1, 1),
                config=config,
                tokenizer_vocab_size=tokenizer_vocab_size,
                global_step=global_step,
                batch_idx=batch_idx,
                epoch_completed=False,
                epoch_loss_sum=total_loss,
                epoch_loss_count=total_seen_batches,
                is_emergency=True,
            )
            raise

    avg_epoch_loss = total_loss / max(total_seen_batches, 1)
    logger.info(f"Phase 2 Epoch {epoch} завершено. Середній loss: {avg_epoch_loss:.4f}")
    if skipped_all_ignored_batches > 0:
        logger.warning(
            "⚠️ Epoch %s finished with %s skipped batches (all labels=-100)",
            epoch,
            skipped_all_ignored_batches,
        )

    return avg_epoch_loss, global_step

def main():
    parser = argparse.ArgumentParser(description="ФАЗА 2 - Instruction Tuning")
    parser.add_argument("--config", type=str, default="config/phase2_instruction_tuning.yaml",
                       help="Шлях до конфігураційного файлу ФАЗИ 2")
    parser.add_argument("--phase1-model", type=str, required=False,
                       help="Шлях до моделі з ФАЗИ 1 (ignored if --resume is set)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Шлях до checkpoint ФАЗИ 2 для продовження")
    
    args = parser.parse_args()
    
    if args.resume is None and args.phase1_model is None:
        parser.error("--phase1-model is required if --resume is not set")
    
    print("🚀 ФАЗА 2 - Instruction Tuning")
    print("=" * 60)
    
    # Перевірити чи існує модель з ФАЗИ 1 (only if not resuming)
    if args.resume is None and not Path(args.phase1_model).exists():
        print(f"❌ ПОМИЛКА: Модель з ФАЗИ 1 не знайдена: {args.phase1_model}")
        print("   Спочатку завершіть ФАЗУ 1 - Language Pretraining!")
        sys.exit(1)
    
    # Завантажити конфігурацію
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Конфігурація ФАЗИ 2: {args.config}")
    if args.resume:
        print(f"🔄 Resume from checkpoint: {args.resume}")
    else:
        print(f"🔗 Модель з ФАЗИ 1: {args.phase1_model}")
    
    # Налаштувати логування
    logger = setup_logging(config['training']['log_dir'])
    logger.info("Початок ФАЗИ 2 - Instruction Tuning")
    
    # CPU налаштування
    device = torch.device('cpu')
    torch.set_num_threads(config['cpu_optimization']['num_threads'])
    
    print(f"💻 Пристрій: {device}")
    
    # Завантажити tokenizer (той же що в ФАЗІ 1)
    print("🔤 Завантаження GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # КРИТИЧНО: Додати PAD токен (не використовувати EOS як PAD)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        print(f"   ✅ Додано PAD токен: {tokenizer.pad_token}")
        print(f"   📊 Новий vocab size: {len(tokenizer)}")
    else:
        print(f"   ℹ️  PAD токен вже існує: {tokenizer.pad_token}")
    
    # Підготувати список instruction datasets
    dataset_files = [config['dataset']['path']]
    if 'additional_datasets' in config['dataset']:
        dataset_files.extend(config['dataset']['additional_datasets'])
    
    # Перевірити існування файлів
    existing_files = []
    for file_path in dataset_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
        else:
            print(f"⚠️  Dataset не знайдено: {file_path}")
    
    if not existing_files:
        print("❌ ПОМИЛКА: Жоден instruction dataset не знайдено!")
        sys.exit(1)
    
    print(f"📚 Використовуватимуться datasets: {len(existing_files)}")
    for f in existing_files:
        print(f"   - {f}")
    
    # Створити dataset
    dataset = InstructionDataset(
        data_files=existing_files,
        tokenizer=tokenizer,
        max_seq_len=config['model']['seq_len']
    )
    
    # Створити dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['cpu_optimization']['num_workers'],
        pin_memory=config['cpu_optimization']['pin_memory']
    )
    
    print(f"📊 Instruction dataset: {len(dataset):,} samples")
    print(f"📦 Batches per epoch: {len(dataloader):,}")
    
    # Завантажити модель: або з Phase 1, або з resume checkpoint
    if args.resume:
        logger.info(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # Rebuild model from Phase1 config first, then load state
        model = load_phase1_model(args.phase1_model if args.phase1_model else "checkpoints/phase1/best_model.pt", config, tokenizer)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint.get('global_step', 0)
        start_epoch = checkpoint.get('epoch', 1)
        resume_batch_idx = checkpoint.get('batch_idx', -1)
        resume_epoch_loss_sum = checkpoint.get('epoch_loss_sum', 0.0)
        resume_epoch_loss_count = checkpoint.get('epoch_loss_count', 0)
        resumed_from_phase2 = True
        print(f"   ✅ Resumed from step {global_step}, epoch {start_epoch}, batch {resume_batch_idx}")
    else:
        logger.info(f"🔄 Loading Phase 1 model: {args.phase1_model}")
        model = load_phase1_model(args.phase1_model, config, tokenizer)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
        global_step = 0
        start_epoch = 1
        resumed_from_phase2 = False
    
    # Створити папку для checkpoints
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    save_every_steps = int(config['training'].get('save_every_steps', 50))
    
    # Перевірити автоматичне відновлення ФАЗИ 2
    if args.resume is None and config['training'].get('auto_resume', True):
        last_checkpoint = checkpoint_dir / "last_checkpoint.pt"
        if last_checkpoint.exists():
            args.resume = str(last_checkpoint)
            print(f"🔄 Знайдено checkpoint ФАЗИ 2 для автоматичного відновлення: {args.resume}")
    
    # Відновити з checkpoint ФАЗИ 2 якщо потрібно
    start_epoch = 1
    resumed_from_phase2 = False
    global_step = 0
    resume_batch_idx = -1
    resume_epoch_loss_sum = 0.0
    resume_epoch_loss_count = 0
    
    if args.resume and Path(args.resume).exists():
        print(f"🔄 Спроба відновлення з checkpoint ФАЗИ 2: {args.resume}")
        try:
            phase2_checkpoint = torch.load(args.resume, map_location='cpu')
            
            if phase2_checkpoint.get('phase') == 2:
                # Це checkpoint ФАЗИ 2 - завантажуємо повністю
                model.load_state_dict(phase2_checkpoint['model_state_dict'])
                optimizer.load_state_dict(phase2_checkpoint['optimizer_state_dict'])

                train_state = phase2_checkpoint.get('train_state') or {}
                global_step = int(train_state.get('global_step', 0))

                resume_batch_idx = int(train_state.get('batch_idx', -1))
                resume_epoch_loss_sum = float(train_state.get('epoch_loss_sum', 0.0))
                resume_epoch_loss_count = int(train_state.get('epoch_loss_count', 0))

                ckpt_epoch = int(phase2_checkpoint.get('epoch', 1))
                epoch_completed = bool(train_state.get('epoch_completed', False))

                start_epoch = ckpt_epoch + 1 if epoch_completed else ckpt_epoch
                best_loss = phase2_checkpoint['loss']
                resumed_from_phase2 = True

                if epoch_completed:
                    resume_batch_idx = -1
                    resume_epoch_loss_sum = 0.0
                    resume_epoch_loss_count = 0
                
                print(f"   ✅ Відновлено з checkpoint ФАЗИ 2")
                print(f"   📊 Продовжуємо з epoch {start_epoch}")
                print(f"   📉 Попередній loss: {best_loss:.4f}")
                
                logger.info(f"Відновлено ФАЗУ 2 з checkpoint epoch {phase2_checkpoint['epoch']}, loss: {best_loss:.4f}")
            else:
                print(f"   ⚠️  Checkpoint не є checkpoint'ом ФАЗИ 2, ігноруємо")
                args.resume = None
                
        except Exception as e:
            print(f"   ❌ Помилка завантаження checkpoint ФАЗИ 2: {e}")
            print("   🔄 Продовжуємо з моделі ФАЗИ 1")
            args.resume = None
    
    # Навчання ФАЗИ 2
    print(f"\n🎯 Початок instruction tuning з epoch {start_epoch} до {config['training']['epochs']}...")
    print("⚠️  УВАГА: Зупинимо навчання якщо якість погіршиться!")
    
    if not resumed_from_phase2:
        best_loss = float('inf')
    patience_counter = 0
    max_patience = config['training'].get('early_stopping_patience', 2)  # З конфігурації
    min_improvement = 0.001  # Мінімальне покращення для продовження
    
    print(f"   📊 Early stopping: patience={max_patience}, min_improvement={min_improvement}")
    if resumed_from_phase2:
        print(f"   🔄 Продовжуємо навчання (відновлено з checkpoint ФАЗИ 2)")
    else:
        print(f"   🆕 Нове instruction tuning (з моделі ФАЗИ 1)")
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\n📚 Phase 2 Epoch {epoch}/{config['training']['epochs']}")

        # Навчити epoch (з autosave та детальним логом)
        try:
            epoch_loss, global_step = train_epoch(
                model,
                dataloader,
                optimizer,
                tokenizer,
                device,
                epoch,
                logger,
                checkpoint_dir=checkpoint_dir,
                config=config,
                tokenizer_vocab_size=len(tokenizer),
                global_step=global_step,
                save_every_steps=save_every_steps,
                resume_batch_idx=resume_batch_idx if resumed_from_phase2 and epoch == start_epoch else -1,
                resume_epoch_loss_sum=resume_epoch_loss_sum if resumed_from_phase2 and epoch == start_epoch else 0.0,
                resume_epoch_loss_count=resume_epoch_loss_count if resumed_from_phase2 and epoch == start_epoch else 0,
            )
        except KeyboardInterrupt:
            logger.warning("🛑 KeyboardInterrupt — emergency save")
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except Exception:
                pass
            _save_phase2_checkpoint(
                checkpoint_dir / "emergency_checkpoint.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=float('inf'),
                config=config,
                tokenizer_vocab_size=len(tokenizer),
                global_step=global_step,
                batch_idx=-1,
                epoch_completed=False,
                epoch_loss_sum=0.0,
                epoch_loss_count=0,
                is_emergency=True,
            )
            print("\n⚠️ Phase 2 INTERRUPTED — progress saved safely")
            print(f"📍 Resume: {checkpoint_dir / 'last_checkpoint.pt'}")
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.exit(130)

        except Exception:
            logger.exception("❌ Exception during training — emergency save")
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except Exception:
                pass
            _save_phase2_checkpoint(
                checkpoint_dir / "emergency_checkpoint.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=float('inf'),
                config=config,
                tokenizer_vocab_size=len(tokenizer),
                global_step=global_step,
                batch_idx=-1,
                epoch_completed=False,
                epoch_loss_sum=0.0,
                epoch_loss_count=0,
                is_emergency=True,
            )
            print("\n❌ Phase 2 STOPPED due to error. Emergency checkpoint saved.")
            print(f"📍 Emergency: {checkpoint_dir / 'emergency_checkpoint.pt'}")
            sys.exit(2)

        # After a successful epoch run, reset resume state for subsequent epochs
        resume_batch_idx = -1
        resume_epoch_loss_sum = 0.0
        resume_epoch_loss_count = 0
        
        # Перевірити якість з мінімальним покращенням
        improvement = best_loss - epoch_loss
        if improvement > min_improvement:
            best_loss = epoch_loss
            patience_counter = 0
            
            # Зберегти найкращу модель
            checkpoint_path = checkpoint_dir / "best_instruction_model.pt"
            _save_phase2_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=epoch_loss,
                config=config,
                tokenizer_vocab_size=len(tokenizer),
                global_step=global_step,
                batch_idx=-1,
                is_emergency=False,
                epoch_loss_sum=0.0,
                epoch_loss_count=0,
            )
            
            print(f"💾 Збережено найкращу instruction модель: {checkpoint_path}")
            print(f"   📈 Покращення loss: {improvement:.6f}")
            logger.info(f"Збережено Phase 2 checkpoint з loss: {epoch_loss:.4f}, покращення: {improvement:.6f}")
        else:
            patience_counter += 1
            if improvement > 0:
                print(f"⚠️  Покращення занадто мале ({improvement:.6f} < {min_improvement}). Patience: {patience_counter}/{max_patience}")
            else:
                print(f"⚠️  Loss погіршився на {-improvement:.6f}. Patience: {patience_counter}/{max_patience}")
            
            if patience_counter >= max_patience:
                print(f"🛑 Зупиняємо навчання - якість не покращується достатньо!")
                logger.warning(f"Early stopping - недостатнє покращення протягом {max_patience} epochs")
                break
        
        # Зберегти останній checkpoint для можливості відновлення
        last_checkpoint_path = checkpoint_dir / "last_checkpoint.pt"
        _save_phase2_checkpoint(
            last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            loss=epoch_loss,
            config=config,
            tokenizer_vocab_size=len(tokenizer),
            global_step=global_step,
            batch_idx=-1,
            epoch_completed=True,
            epoch_loss_sum=0.0,
            epoch_loss_count=0,
            is_emergency=False,
        )
    
    print(f"\n✅ ФАЗА 2 завершена!")
    print(f"   🏆 Найкращий instruction loss: {best_loss:.4f}")
    print(f"   💾 Фінальна модель: {checkpoint_dir / 'best_instruction_model.pt'}")
    print(f"   📝 Логи: {config['training']['log_dir']}")
    
    print(f"\n🎉 Двофазне навчання завершено!")
    print(f"   📚 ФАЗА 1: Language pretraining ✅")
    print(f"   🎯 ФАЗА 2: Instruction tuning ✅")
    print(f"   🤖 Готова модель для використання!")
    
    logger.info("ФАЗА 2 успішно завершена - модель готова!")

if __name__ == "__main__":
    main()
