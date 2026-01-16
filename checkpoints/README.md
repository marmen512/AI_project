# Чекпоінти моделі

Ця папка містить збережені чекпоінти навченої моделі.

## Структура

- `phase1/` - чекпоінти з фази 1 (pretraining)
  - `best_model.pt` - найкраща модель з фази 1
- `phase2/` - чекпоінти з фази 2 (instruction tuning)
  - `last_checkpoint.pt` - останній чекпоінт
  - `emergency_checkpoint.pt` - аварійний чекпоінт

## Використання

Чекпоінти автоматично створюються під час навчання та зберігаються в цій папці.

### Відновлення з чекпоінту

```python
from config.model_loader import load_model_from_checkpoint

model = load_model_from_checkpoint('checkpoints/phase1/best_model.pt')
```

### Продовження навчання

```bash
python scripts/train_phase2_instruction_tuning.py --resume checkpoints/phase2/last_checkpoint.pt
```

## Примітки

- Чекпоінти (.pt файли) не додаються до git через великий розмір
- Для збереження чекпоінтів використовуйте окреме сховище або хмарне сховище
- Рекомендується регулярно створювати резервні копії важливих чекпоінтів
