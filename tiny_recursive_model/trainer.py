from __future__ import annotations

import torch
import torch.nn.functional as F
import time
import os
import sys
from datetime import timedelta
from pathlib import Path
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from einops import pack, unpack

from accelerate import Accelerator

# ema - apparently greatly helped with results

from ema_pytorch import EMA

from tiny_recursive_model.trm import TinyRecursiveModel
from core.types import TrainState
from runtime.checkpointing import CheckpointManager
from core.constants import CHECKPOINT_BEST_LOSS, CHECKPOINT_BEST_EVAL, CHECKPOINT_BEST_ENTROPY
from train.callbacks.base import CallbackList
from train.callbacks.curriculum import CurriculumCallback
from train.callbacks.checkpoint import CheckpointCallback
from train.callbacks.logging import LoggingCallback
from train.callbacks.early_stopping import EarlyStoppingCallback

from adam_atan2_pytorch import MuonAdamAtan2

from x_transformers import Encoder, Decoder

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# helpers

def exists(v):
    return v is not None

def range_from_one(n):
    return range(1, n + 1)

def is_empty(t):
    return t.numel() == 0

# trainer

class Trainer(Module):
    def __init__(
        self,
        model: TinyRecursiveModel | Module,
        dataset: Dataset,
        optim_klass = AdamW,
        optim: Optimizer | None = None,
        learning_rate = 1e-4,
        muon_learning_rate = 1e-3,
        weight_decay = 1.,
        batch_size = 16,
        epochs = 2,
        halt_prob_thres = 0.5,
        max_recurrent_steps = 12,
        thinking_cost_weight: float = 0.01,  # Вага thinking cost в loss
        warmup_steps = 2000,
        ema_decay_rate = 0.999,
        switch_ema_every = 10000,           # switch ema https://arxiv.org/abs/2402.09240
        accelerate_kwargs: dict = dict(),
        cpu = False,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 100,  # Зберігати checkpoint кожні N батчів
        log_file: str | Path | None = None,  # Файл для детального логування
        resource_monitor = None,  # ResourceMonitor для моніторингу ресурсів
        training_logger = None,  # TRMTrainingLogger для логування метрик
        curriculum_scheduler = None,  # CurriculumScheduler для керування етапами навчання
        callbacks: list = None  # Список callbacks (для callback-based архітектури)
    ):
        super().__init__()

        self.accelerator  = Accelerator(**accelerate_kwargs, cpu = cpu)

        self.batch_size = batch_size
        self.epochs = epochs
        
        # Логування в файл
        self.log_file = None
        if log_file is not None:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(exist_ok=True, parents=True)
            # Ініціалізувати лог-файл
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"Training Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
        
        # Моніторинг ресурсів та логування
        self.resource_monitor = resource_monitor
        self.training_logger = training_logger
        self.curriculum_scheduler = curriculum_scheduler
        
        # Callback-based архітектура
        self.callbacks = CallbackList(callbacks or [])
        
        # Додати callbacks якщо вони передані через параметри
        if curriculum_scheduler is not None:
            self.callbacks.add(CurriculumCallback(curriculum_scheduler))
        if training_logger is not None:
            self.callbacks.add(LoggingCallback(training_logger))
        if checkpoint_dir is not None:
            # CheckpointCallback буде створено пізніше після ініціалізації checkpoint_manager
            pass

        # data

        self.dataset = dataset
        self.dataloader = dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

        # optim

        if not exists(optim):

            if isinstance(model.network, (Encoder, Decoder)):
                optim = MuonAdamAtan2(
                    model.network.muon_parameters(),
                    model.parameters(),
                    lr = learning_rate / (batch_size * max_recurrent_steps),
                    muon_lr = muon_learning_rate / (batch_size * max_recurrent_steps),
                    weight_decay = weight_decay
                )
            else:
                optim = optim_klass(
                    model.parameters(),
                    lr = learning_rate / (batch_size * max_recurrent_steps),
                    weight_decay = weight_decay
                )

        self.optim = optim

        # scheduler

        self.scheduler = LambdaLR(self.optim, lambda step: min((step + 1) / warmup_steps, 1.0))

        # model

        self.model = model

        # ema model

        self.ema_model = None

        if self.accelerator.is_main_process:
            self.ema_model = EMA(
                model,
                beta = ema_decay_rate,
                update_model_with_ema_every = switch_ema_every,
                forward_method_names = ('predict',)
            )

        # recurrent and act related variables

        self.halt_prob_thres = halt_prob_thres
        self.max_recurrent_steps = max_recurrent_steps
        self.thinking_cost_weight = thinking_cost_weight

        # checkpoint settings
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        
        if self.checkpoint_dir is not None:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            # Створити CheckpointManager для best model tracking
            self.checkpoint_manager = CheckpointManager(path=self.checkpoint_dir, keep_last=5)
            # Додати CheckpointCallback
            self.callbacks.add(CheckpointCallback(
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_interval=self.checkpoint_interval,
                checkpoint_manager=self.checkpoint_manager
            ))
        else:
            self.checkpoint_manager = None

        # prepare maybe distributed

        self.model, self.optim, self.dataloader, self.scheduler = self.accelerator.prepare(self.model, self.optim, self.dataloader, self.scheduler)
        
        # TrainState для відстеження стану навчання
        self.train_state = TrainState()
        
        # Best metrics tracking (для best model checkpointing)
        self.best_loss: Optional[float] = None
        self.best_eval_score: Optional[float] = None
        self.best_entropy: Optional[float] = None

    def save_checkpoint(self, epoch: int, batch_idx: int, batch_count: int, is_final: bool = False):
        """
        Зберегти checkpoint навчання
        
        Args:
            epoch: Поточна епоха
            batch_idx: Індекс батчу в епосі
            batch_count: Загальна кількість оброблених батчів
            is_final: Чи це фінальний checkpoint після завершення навчання
        """
        if self.checkpoint_dir is None or not self.accelerator.is_main_process:
            return
        
        checkpoint_path = Path(self.checkpoint_dir)
        
        # Створити словник зі станом
        checkpoint_state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'batch_count': batch_count,
            'model_state_dict': self.accelerator.get_state_dict(self.model),
            'optimizer_state_dict': self.accelerator.get_state_dict(self.optim),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'train_state': self.train_state.to_dict(),  # Зберегти TrainState
        }
        
        # Зберегти EMA модель якщо вона є
        if self.ema_model is not None:
            checkpoint_state['ema_model_state_dict'] = self.ema_model.ema_model.state_dict()
            checkpoint_state['ema_decay'] = self.ema_model.beta
        
        # Зберегти checkpoint
        suffix = "final" if is_final else f"epoch_{epoch}_batch_{batch_idx}"
        checkpoint_file = checkpoint_path / f"checkpoint_{suffix}.pt"
        
        torch.save(checkpoint_state, checkpoint_file)
        
        # Також зберегти як "latest" checkpoint
        latest_file = checkpoint_path / "checkpoint_latest.pt"
        torch.save(checkpoint_state, latest_file)
        
        self.accelerator.print(f"💾 Checkpoint збережено: {checkpoint_file.name}")
        
        # Перевірити та зберегти best checkpoints через CheckpointCallback
        if self.accelerator.is_main_process:
            checkpoint_callbacks = [cb for cb in self.callbacks.callbacks if isinstance(cb, CheckpointCallback)]
            for checkpoint_cb in checkpoint_callbacks:
                try:
                    checkpoint_cb.save_best_checkpoints(self.train_state, self.accelerator, self.model, self.optim)
                except Exception as e:
                    # Не зупиняти навчання через помилки best checkpointing
                    if self.log_file:
                        self._log_to_file(f"⚠️ Помилка best checkpointing: {e}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """
        Завантажити checkpoint навчання
        
        Args:
            checkpoint_path: Шлях до checkpoint файлу
            
        Returns:
            tuple: (epoch, batch_idx, batch_count) для продовження навчання
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint не знайдено: {checkpoint_path}")
        
        self.accelerator.print(f"📂 Завантаження checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        
        # Завантажити стан моделі
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Завантажити optimizer та scheduler
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Завантажити EMA модель якщо вона є
        if self.ema_model is not None and 'ema_model_state_dict' in checkpoint:
            self.ema_model.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        
        # Завантажити TrainState якщо він є
        if 'train_state' in checkpoint:
            self.train_state = TrainState.from_dict(checkpoint['train_state'])
            epoch = self.train_state.epoch
            batch_idx = self.train_state.batch_idx
            batch_count = self.train_state.step
        else:
            # Fallback для старих checkpoint'ів
            epoch = checkpoint.get('epoch', 1)
            batch_idx = checkpoint.get('batch_idx', 0)
            batch_count = checkpoint.get('batch_count', 0)
            # Оновити TrainState
            self.train_state.update(epoch=epoch, batch_idx=batch_idx, step=batch_count)
        
        self.accelerator.print(f"✅ Checkpoint завантажено: епоха {epoch}, батч {batch_idx}, загалом {batch_count} батчів")
        
        return epoch, batch_idx, batch_count

    def forward(self, resume_from_checkpoint: str | Path | None = None):
        total_batches = len(self.dataloader) * self.epochs
        start_time = time.time()
        batch_count = 0
        start_epoch = 1
        start_batch_idx = 0
        
        # Завантажити checkpoint якщо вказано
        if resume_from_checkpoint is not None:
            start_epoch, start_batch_idx, batch_count = self.load_checkpoint(resume_from_checkpoint)
            # Пропустити батчі до start_batch_idx
            self.accelerator.print(f"Продовження з епохи {start_epoch}, батч {start_batch_idx}")

        # Створити прогрес-бар якщо доступний
        if HAS_TQDM and self.accelerator.is_main_process:
            initial_n = batch_count
            pbar = tqdm(total=total_batches, initial=initial_n, desc="Навчання", unit="batch",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        try:
            # Callback: on_train_start
            self.callbacks.on_train_start(self.train_state)
            
            for epoch in range(start_epoch, self.epochs + 1):
                epoch_start = time.time()
                
                # Оновити TrainState для епохи
                self.train_state.update(epoch=epoch)
                
                # Callback: on_epoch_start
                self.callbacks.on_epoch_start(self.train_state)
                
                # Логування поточного curriculum stage (через callback)
                if self.curriculum_scheduler is not None and self.accelerator.is_main_process:
                    stage_desc = self.curriculum_scheduler.describe()
                    self.accelerator.print(stage_desc)
                    if self.log_file:
                        self._log_to_file(stage_desc)
                
                if HAS_TQDM and self.accelerator.is_main_process:
                    pbar.set_description(f"Епоха {epoch}/{self.epochs}")

                for batch_idx, batch_data in enumerate(self.dataloader):
                    # Підтримка document-aware dataset (з doc_id, segment_id) та старого формату
                    if len(batch_data) == 4:
                        dataset_input, dataset_output, doc_ids, segment_ids = batch_data
                        # doc_ids та segment_ids доступні, але не використовуються напряму в trainer
                        # Вони можуть бути використані для document-aware логіки в майбутньому
                    elif len(batch_data) == 2:
                        dataset_input, dataset_output = batch_data
                    else:
                        raise ValueError(f"Невідомий формат batch_data: очікується 2 або 4 елементи, отримано {len(batch_data)}")
                    # Пропустити батчі до start_batch_idx для першої епохи після resume
                    if epoch == start_epoch and batch_idx < start_batch_idx:
                        # Оновити прогрес-бар навіть для пропущених батчів
                        if HAS_TQDM and self.accelerator.is_main_process:
                            pbar.update(1)
                        continue
                    
                    outputs, latents = self.model.get_initial()

                    # Зберегти поточний loss для виводу в лог
                    current_main_loss = None
                    actual_recurrent_steps = 0  # Фактична кількість використаних recurrent steps
                    
                    # Додати таймер для виявлення зависання
                    # Збільшені таймаути для повільних систем з swap
                    batch_start_time = time.time()
                    max_batch_time = 3600  # Максимум 60 хвилин на батч (для повільних систем)
                    step_start_time = time.time()
                    max_step_time = 600  # Максимум 10 хвилин на recurrent step (для систем з swap)
                    
                    # Початкова кількість токенів для обчислення tokens_per_sec
                    initial_tokens = dataset_input.numel()
                    
                    # Оновити TrainState на початку батча
                    self.train_state.update(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        step=batch_count
                    )
                    
                    for recurrent_step in range_from_one(self.max_recurrent_steps):
                        actual_recurrent_steps = recurrent_step
                        # Перевірка на зависання в окремому step
                        step_elapsed = time.time() - step_start_time
                        if step_elapsed > max_step_time:
                            warning_msg = f"⚠️ УВАГА: Recurrent step {recurrent_step} працює вже {step_elapsed:.0f} секунд (> {max_step_time}s)!\n   Епоха: {epoch}, Батч: {batch_idx}\n   🔴 ПРИМУСОВИЙ ВИХІД з recurrent_steps циклу для запобігання зависання"
                            self.accelerator.print(warning_msg)
                            if self.log_file:
                                self._log_to_file(warning_msg)
                            break
                        
                        # Перевірка на зависання всього батча
                        batch_elapsed = time.time() - batch_start_time
                        if batch_elapsed > max_batch_time:
                            warning_msg = f"⚠️ УВАГА: Батч {batch_idx} працює вже {batch_elapsed:.0f} секунд (> {max_batch_time}s). Можливе зависання!\n   Епоха: {epoch}, recurrent_step: {recurrent_step}/{self.max_recurrent_steps}\n   🔴 ПРИМУСОВИЙ ВИХІД з батча для запобігання зависання"
                            self.accelerator.print(warning_msg)
                            if self.log_file:
                                self._log_to_file(warning_msg)
                            break
                        
                        step_start_time = time.time()  # Оновити час початку step
                        
                        loss, (main_loss, halt_loss), outputs, latents, pred, halt = self.model(dataset_input, outputs, latents, labels = dataset_output)
                        current_main_loss = main_loss.mean().item()  # Зберегти для виводу
                        
                        # Thinking cost: з adaptive recursion gate або нормалізований
                        if hasattr(self.model, 'adaptive_recursion') and self.model.adaptive_recursion:
                            # Використати adaptive recursion gate для thinking cost
                            # Потрібно отримати hidden representation для обчислення gate
                            # Якщо outputs доступні, використовуємо їх для обчислення gate
                            if outputs.numel() > 0:
                                # Використати outputs напряму для gate (gate network сам зробить reduce)
                                gate = self.model.compute_adaptive_depth(outputs)  # [batch_size]
                                # Thinking cost = weight * gate * normalized_step
                                # Gate вказує скільки "думати", тому множимо на нього
                                normalized_step = recurrent_step / self.max_recurrent_steps
                                thinking_cost_per_sample = self.thinking_cost_weight * gate * normalized_step
                                # Loss це sum по batch, тому додаємо sum thinking cost
                                thinking_cost_scalar = thinking_cost_per_sample.sum().item()
                            else:
                                # Fallback до нормалізованої формули
                                thinking_cost_scalar = self.thinking_cost_weight * (recurrent_step / self.max_recurrent_steps)
                        else:
                            # Нормалізована формула (коли adaptive recursion вимкнено)
                            # Thinking cost = weight * (recurrent_step / max_recurrent_steps)
                            thinking_cost_scalar = self.thinking_cost_weight * (recurrent_step / self.max_recurrent_steps)
                        
                        # Додати thinking cost до loss (loss це sum по batch, тому додаємо скаляр)
                        thinking_cost_tensor = torch.tensor(thinking_cost_scalar, device=loss.device, dtype=loss.dtype)
                        loss = loss + thinking_cost_tensor
                        
                        # Обчислити entropy для best checkpointing (SECONDARY метрика)
                        import torch.nn.functional as F
                        probs = F.softmax(pred, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                        
                        # Оновити TrainState з поточними значеннями
                        self.train_state.update(
                            loss=current_main_loss,
                            main_loss=current_main_loss,
                            halt_loss=halt_loss.mean().item() if halt_loss.numel() > 0 else 0.0,
                            thinking_cost=thinking_cost_scalar,
                            recursion_depth=recurrent_step,
                            actual_recurrent_steps=actual_recurrent_steps
                        )
                        # Зберігати entropy в metadata для best checkpointing
                        if 'entropy' not in self.train_state.metadata:
                            self.train_state.metadata['entropy'] = []
                        self.train_state.metadata['entropy'].append(entropy)
                        
                        # Логування кожні 5 кроків для діагностики (зменшено частоту)
                        if recurrent_step % 5 == 0 and self.accelerator.is_main_process:
                            halt_mean = halt.mean().item() if halt.numel() > 0 else 0.0
                            self.accelerator.print(f'   [Step {recurrent_step}/{self.max_recurrent_steps}] loss: {main_loss.mean().item():.3f} | halt: {halt_mean:.3f} | thinking_cost: {thinking_cost_scalar:.4f} | halt_thres: {self.halt_prob_thres}', flush=True)

                        # Оновити прогрес-бар
                        if HAS_TQDM and self.accelerator.is_main_process:
                            elapsed = time.time() - start_time
                            if batch_count > 0:
                                avg_time = elapsed / batch_count
                                remaining = avg_time * (total_batches - batch_count)
                                pbar.set_postfix({
                                    'Loss': f"{main_loss.mean().item():.3f}",
                                    'Halt': f"{halt_loss.mean().item():.3f}",
                                    'ETA': f"{timedelta(seconds=int(remaining))}"
                                })
                            pbar.update(1)
                        else:
                            self.accelerator.print(f'[{epoch} ({recurrent_step} / {self.max_recurrent_steps})] loss: {main_loss.mean().item():.3f} | halt loss: {halt_loss.mean().item():.3f}')

                        self.accelerator.backward(loss)

                        self.optim.step()
                        self.optim.zero_grad()

                        self.scheduler.step()

                        if self.accelerator.is_main_process:
                            self.ema_model.update()

                        # handle halting

                        halt_mask = halt >= self.halt_prob_thres

                        if not halt_mask.any():
                            # Якщо це останній крок, примусово вийти
                            if recurrent_step == self.max_recurrent_steps:
                                if self.accelerator.is_main_process:
                                    warning_msg = f'   ⚠️ Досягнуто max_recurrent_steps ({self.max_recurrent_steps}) без halt. Примусовий вихід.'
                                    self.accelerator.print(warning_msg)
                                    if self.log_file:
                                        self._log_to_file(warning_msg)
                                break
                            continue

                        # Зберегти старі тензори перед фільтрацією для очищення пам'яті
                        old_outputs, old_latents = outputs, latents
                        
                        outputs = outputs[~halt_mask]
                        latents = latents[~halt_mask]
                        dataset_input = dataset_input[~halt_mask]
                        dataset_output = dataset_output[~halt_mask]
                        
                        # Очистити пам'ять від старих тензорів
                        del old_outputs, old_latents
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()

                        if is_empty(outputs):
                            if self.accelerator.is_main_process:
                                self.accelerator.print(f'   ✅ Всі приклади завершилися (halt) на step {recurrent_step}')
                            break
                    
                    batch_count += 1
                    
                    # Обчислити тривалість батча
                    batch_duration = time.time() - batch_start_time
                    
                    # Обчислити tokens_per_sec
                    tokens_per_sec = None
                    if batch_duration > 0 and initial_tokens > 0:
                        tokens_per_sec = initial_tokens / batch_duration
                    
                    # Оновити TrainState з batch metrics
                    self.train_state.update(
                        step=batch_count,
                        batch_idx=batch_idx,
                        tokens_per_sec=tokens_per_sec
                    )
                    if tokens_per_sec is not None:
                        self.train_state.metadata['tokens_per_sec'] = tokens_per_sec
                    
                    # Callback: on_batch_end (включає логування)
                    self.callbacks.on_batch_end(self.train_state)
                    
                    # Зберегти best checkpoints на кожному батчі (якщо новий best)
                    # Це важливо, щоб не пропустити найкращу модель
                    if self.accelerator.is_main_process and self.checkpoint_manager is not None:
                        checkpoint_callbacks = [cb for cb in self.callbacks.callbacks if isinstance(cb, CheckpointCallback)]
                        for checkpoint_cb in checkpoint_callbacks:
                            try:
                                checkpoint_cb.save_best_checkpoints(self.train_state, self.accelerator, self.model, self.optim)
                            except Exception as e:
                                # Не зупиняти навчання через помилки best checkpointing
                                if self.log_file:
                                    self._log_to_file(f"⚠️ Помилка best checkpointing на батчі {batch_idx}: {e}")
                    
                    # Перевірити періодичне збереження checkpoint (через callback)
                    checkpoint_callbacks = [cb for cb in self.callbacks.callbacks if isinstance(cb, CheckpointCallback)]
                    for checkpoint_cb in checkpoint_callbacks:
                        if checkpoint_cb.should_save_periodic(self.train_state):
                            if self.accelerator.is_main_process:
                                self.save_checkpoint(epoch, batch_idx, batch_count, is_final=False)
                    
                    # Моніторинг ресурсів (якщо доступний)
                    if self.resource_monitor is not None and self.accelerator.is_main_process:
                        try:
                            resource_info = self.resource_monitor.check_resources(
                                batch_idx=batch_idx,
                                epoch=epoch,
                                batch_time=batch_duration
                            )
                            # Логувати попередження якщо є
                            if resource_info.get('warnings'):
                                for warning in resource_info['warnings']:
                                    warning_msg = f"⚠️ {warning} (Епоха: {epoch}, Батч: {batch_idx})"
                                    self.accelerator.print(warning_msg)
                                    if self.log_file:
                                        self._log_to_file(warning_msg)
                            
                            # Auto-reaction: автоматичний контроль ресурсів
                            recommendations = self.resource_monitor.auto_throttle(self.batch_size)
                            
                            # Перевірити чи потрібно зменшити batch size
                            if recommendations.get('shrink_batch', False) and recommendations.get('batch_size_changed', False):
                                new_batch_size = recommendations.get('suggested_batch_size', self.batch_size)
                                if new_batch_size < self.batch_size:
                                    old_batch_size = self.batch_size
                                    self.batch_size = max(1, new_batch_size)  # Мінімум 1
                                    shrink_msg = f"🔧 Auto-throttle: зменшено batch_size з {old_batch_size} до {self.batch_size} (використання пам'яті: {recommendations.get('memory_percent', 0):.1f}%)"
                                    self.accelerator.print(shrink_msg)
                                    if self.log_file:
                                        self._log_to_file(shrink_msg)
                                    # Оновити DataLoader з новим batch_size (наступний батч)
                                    # Примітка: DataLoader оновиться на наступній ітерації
                            
                            # Перевірити чи потрібно призупинити навчання
                            if recommendations.get('pause', False):
                                pause_msg = f"⏸️ Auto-pause: навчання призупинено через критичне використання ресурсів (CPU: {recommendations.get('cpu_usage', 0):.1f}%, Memory: {recommendations.get('memory_percent', 0):.1f}%)"
                                self.accelerator.print(pause_msg)
                                if self.log_file:
                                    self._log_to_file(pause_msg)
                                # Призупинити навчання - зберегти checkpoint та вийти
                                if self.checkpoint_dir is not None:
                                    self.save_checkpoint(epoch, batch_idx, batch_count, is_final=False)
                                raise RuntimeError("Навчання призупинено через критичне використання ресурсів")
                            
                            # Логувати throttle рекомендації якщо є
                            if recommendations.get('throttle', False):
                                throttle_msg = f"⚡ Auto-throttle активовано (CPU: {recommendations.get('cpu_usage', 0):.1f}%, Memory: {recommendations.get('memory_percent', 0):.1f}%)"
                                if self.log_file:
                                    self._log_to_file(throttle_msg)
                                
                        except RuntimeError as e:
                            # Помилка pause - прокинути далі
                            raise
                        except Exception as e:
                            # Не зупиняти навчання через помилки моніторингу
                            error_msg = f"⚠️ Помилка моніторингу ресурсів: {e}"
                            if self.log_file:
                                self._log_to_file(error_msg)
                    
                    # Логування тривалості батча (детальне)
                    if self.accelerator.is_main_process and batch_duration > 60:  # Якщо батч тривав більше хвилини
                        duration_msg = f"⏱️ Батч {batch_idx} (Епоха {epoch}) тривав {batch_duration:.1f} секунд"
                        if self.log_file:
                            self._log_to_file(duration_msg)
                    
                    # Періодичне очищення пам'яті (кожні 10 батчів)
                    if batch_count % 10 == 0:
                        import gc
                        gc.collect()
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    
                    # Вивести прогрес в лог-файл (для моніторингу)
                    # Виводимо кожні 5 батчів для оптимізації (замість кожного)
                    if self.accelerator.is_main_process and (batch_count % 5 == 0 or batch_count == 1):
                        progress_pct = (batch_count / total_batches) * 100
                        elapsed = time.time() - start_time
                        if batch_count > 0:
                            avg_time = elapsed / batch_count
                            remaining = avg_time * (total_batches - batch_count)
                            loss_value = current_main_loss if current_main_loss is not None else 0.0
                            # Вивести в stdout (який буде записано в лог через tee)
                            print(
                                f'📊 Прогрес: {batch_count}/{total_batches} батчів ({progress_pct:.1f}%) | '
                                f'Епоха: {epoch}/{self.epochs} | '
                                f'Loss: {loss_value:.3f} | '
                                f'ETA: {timedelta(seconds=int(remaining))}',
                                flush=True
                            )
                    
                    # Зберегти checkpoint періодично
                    if (self.checkpoint_dir is not None and 
                        batch_count % self.checkpoint_interval == 0 and
                        self.accelerator.is_main_process):
                        self.save_checkpoint(epoch, batch_idx, batch_count, is_final=False)
                    
                    # Вивести статистику епохи
                    if not HAS_TQDM and self.accelerator.is_main_process:
                        epoch_elapsed = time.time() - epoch_start
                        progress = (batch_idx + 1) / len(self.dataloader) * 100
                        self.accelerator.print(f'Епоха {epoch}: {progress:.1f}% | Час: {timedelta(seconds=int(epoch_elapsed))}')
                
                # Оновити curriculum scheduler після завершення епохи
                if self.curriculum_scheduler is not None:
                    self.curriculum_scheduler.on_epoch_end()

            if HAS_TQDM and self.accelerator.is_main_process:
                pbar.close()
            
            total_time = time.time() - start_time
            completion_msg = f'\n✅ Навчання завершено за {timedelta(seconds=int(total_time))}'
            self.accelerator.print(completion_msg)
            if self.log_file:
                self._log_to_file(completion_msg)
        
        except KeyboardInterrupt:
            interrupt_msg = "\n⚠️ Навчання перервано користувачем (KeyboardInterrupt)"
            self.accelerator.print(interrupt_msg)
            if self.log_file:
                self._log_to_file(interrupt_msg)
            # Зберегти checkpoint перед виходом
            if self.checkpoint_dir is not None and self.accelerator.is_main_process:
                self.save_checkpoint(epoch, batch_idx, batch_count, is_final=False)
            raise
        
        except Exception as e:
            # Детально залогувати помилку
            batch_idx_str = str(batch_idx) if 'batch_idx' in locals() else "unknown"
            self._log_error(e, f"під час навчання (Епоха: {epoch}, Батч: {batch_idx_str})")
            # Зберегти checkpoint перед виходом
            if self.checkpoint_dir is not None and self.accelerator.is_main_process:
                try:
                    self.save_checkpoint(epoch, batch_idx, batch_count, is_final=False)
                except Exception as checkpoint_error:
                    self._log_error(checkpoint_error, "при спробі зберегти checkpoint після помилки")
            raise
    
    def _log_to_file(self, message: str):
        """Записати повідомлення в лог-файл"""
        if self.log_file and self.accelerator.is_main_process:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {message}\n")
            except Exception as e:
                # Не зупиняти навчання через помилки запису в лог
                print(f"⚠️ Помилка запису в лог-файл: {e}")
    
    def _log_error(self, error: Exception, context: str = ""):
        """Детально залогувати помилку з повним traceback"""
        import traceback
        error_msg = f"[ERROR] ПОМИЛКА{': ' + context if context else ''}: {str(error)}"
        traceback_str = traceback.format_exc()
        
        self.accelerator.print(error_msg)
        self.accelerator.print(traceback_str)
        
        if self.log_file:
            self._log_to_file(error_msg)
            self._log_to_file("Traceback:")
            self._log_to_file(traceback_str)

        if self.accelerator.is_main_process:
            self.ema_model.copy_params_from_ema_to_model()
            
            # Callback: on_train_end (включає збереження фінального checkpoint)
            self.callbacks.on_train_end(self.train_state)
            
            # Зберегти фінальний checkpoint
            if self.checkpoint_dir is not None and self.accelerator.is_main_process:
                self.save_checkpoint(self.epochs, len(self.dataloader) - 1, batch_count, is_final=True)
