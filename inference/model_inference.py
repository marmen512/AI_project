"""
Модуль для інференсу навченої TRM моделі
"""
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json

from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, TransformerBackbone
from config.model_manager import ModelManager
from tiny_recursive_model.utils import tokenize_and_pad, prepare_code_input, load_tokenizer


class TRMInference:
    """Клас для роботи з навченою TRM моделлю"""
    
    def __init__(
        self,
        model: TinyRecursiveModel,
        tokenizer,
        device: str = 'cpu',
        max_seq_len: int = 2048,
        timeout_seconds: Optional[float] = None  # Timeout для recursion (безпека)
    ):
        """
        Ініціалізація інференсу
        
        Args:
            model: Навчена TRM модель
            tokenizer: Токенізатор
            device: Пристрій ('cpu' або 'cuda')
            max_seq_len: Максимальна довжина послідовності
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.timeout_seconds = timeout_seconds  # Timeout для recursion (безпека)
        
        # Перемістити модель на пристрій
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        context: str,
        query: str,
        max_deep_refinement_steps: int = 12,
        halt_prob_thres: float = 0.5,
        temperature: float = 0.7,
        top_k: int = 50,
        deterministic: bool = False
    ) -> Dict[str, any]:
        """
        Зробити передбачення
        
        Args:
            context: Контекст коду
            query: Запит користувача
            max_deep_refinement_steps: Максимальна кількість кроків уточнення
            halt_prob_thres: Поріг для раннього виходу
            temperature: Температура для sampling (не використовується зараз)
            top_k: Top-k sampling (не використовується зараз)
        
        Returns:
            Словник з результатами
        """
        # Підготувати вхід
        input_text = prepare_code_input(context, query)
        
        pad_token_id = 0
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id
        
        input_ids = tokenize_and_pad(
            self.tokenizer,
            input_text,
            self.max_seq_len,
            pad_token_id=pad_token_id
        ).unsqueeze(0).to(self.device)
        
        # Передбачення з timeout (передати timeout_seconds якщо вказано)
        # Default timeout: 30 секунд для безпеки (запобігає зависанню)
        timeout = self.timeout_seconds if self.timeout_seconds is not None else 30.0
        with torch.no_grad():
            pred_tokens, exit_steps = self.model.predict(
                input_ids,
                max_deep_refinement_steps=max_deep_refinement_steps,
                halt_prob_thres=halt_prob_thres,
                timeout_seconds=timeout  # Передати timeout для безпеки
            )
        
        # Декодувати результат
        if hasattr(self.tokenizer, 'decode'):
            pred_tokens_clean = pred_tokens[0].cpu().numpy()
            # Знайти реальний кінець (видалити padding)
            output = self.tokenizer.decode(pred_tokens_clean, skip_special_tokens=True)
        else:
            output = ''.join([self.tokenizer.inv_vocab.get(int(t), '?') for t in pred_tokens[0]])
        
        return {
            'completion': output,
            'exit_steps': exit_steps[0].item(),
            'tokens': pred_tokens[0].cpu().tolist(),
            'input_length': input_ids.shape[1]
        }
    
    def batch_predict(
        self,
        examples: List[Dict[str, str]],
        max_deep_refinement_steps: int = 12,
        halt_prob_thres: float = 0.5
    ) -> List[Dict[str, any]]:
        """
        Batch передбачення для кількох прикладів
        
        Args:
            examples: Список прикладів з 'context' та 'query'
            max_deep_refinement_steps: Максимальна кількість кроків
            halt_prob_thres: Поріг для раннього виходу
        
        Returns:
            Список результатів
        """
        results = []
        for example in examples:
            result = self.predict(
                example.get('context', ''),
                example.get('query', ''),
                max_deep_refinement_steps=max_deep_refinement_steps,
                halt_prob_thres=halt_prob_thres
            )
            result['context'] = example.get('context', '')
            result['query'] = example.get('query', '')
            results.append(result)
        
        return results
    
    def interactive_mode(self):
        """Інтерактивний режим для тестування"""
        print("\n" + "=" * 70)
        print("🤖 ІНТЕРАКТИВНИЙ РЕЖИМ TRM МОДЕЛІ")
        print("=" * 70)
        print("Введіть 'exit' для виходу\n")
        
        while True:
            try:
                context = input("📝 Контекст (код):\n> ").strip()
                if context.lower() == 'exit':
                    break
                
                query = input("❓ Запит:\n> ").strip()
                if query.lower() == 'exit':
                    break
                
                print("\n🔄 Генерація відповіді...")
                result = self.predict(context, query)
                
                print(f"\n✅ Результат (кроків уточнення: {result['exit_steps']}):")
                print("-" * 70)
                print(result['completion'])
                print("-" * 70)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Вихід з інтерактивного режиму")
                break
            except Exception as e:
                print(f"\n❌ Помилка: {e}\n")


def load_trained_model(
    model_path: str | Path = None,
    model_name: str = None,  # НОВИЙ: можна вказати ім'я моделі
    config_path: Optional[str | Path] = None,
    device: str = 'cpu',
    tokenizer_name: str = "gpt2"
) -> TRMInference:
    """
    Завантажити навчену модель для інференсу
    
    Args:
        model_path: Шлях до файлу моделі (.pt) (опціонально, якщо вказано model_name)
        model_name: Назва моделі для пошуку в models/trained/ (опціонально)
        config_path: Шлях до конфігурації моделі (JSON, опціонально)
        device: Пристрій ('cpu' або 'cuda')
        tokenizer_name: Назва токенізатора
    
    Returns:
        TRMInference об'єкт
    """
    manager = ModelManager()
    
    # Якщо вказано ім'я, знайти модель
    if model_name:
        model_info = manager.get_model_by_name(model_name)
        if model_info:
            model_path = model_info['path']
            # Використати config з моделі якщо є
            if model_info.get('config_path') and not config_path:
                config_path = model_info['config_path']
            print(f"📦 Використовується модель: {model_info['name']}")
        else:
            raise FileNotFoundError(f"Модель '{model_name}' не знайдена в models/trained/")
    
    # Якщо шлях не вказано, використати останню модель
    if model_path is None:
        model_info = manager.get_default_model()
        if model_info:
            model_path = model_info['path']
            if model_info.get('config_path') and not config_path:
                config_path = model_info['config_path']
            print(f"📦 Використовується остання модель: {model_info['name']}")
        else:
            raise FileNotFoundError("Моделі не знайдено в models/trained/")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не знайдено: {model_path}")
    
    # Завантажити конфігурацію якщо є
    config = {}
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
    else:
        # Спробувати знайти config поруч з моделлю
        config_path = model_path.with_suffix('.json')
        if not config_path.exists():
            config_path = model_path.parent / f"{model_path.stem}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
    
    # Параметри моделі (за замовчуванням або з конфігурації)
    dim = config.get('dim', 768)  # Дефолт для GPT-2
    vocab_size = config.get('vocab_size', 50257)
    seq_len = config.get('seq_len', 1024)  # Дефолт для GPT-2
    depth = config.get('depth', 12)  # Дефолт для GPT-2
    
    # Визначити чи використовується Transformer
    use_transformer = config.get('use_transformer', False)
    transformer_model = config.get('transformer_model', 'gpt2')
    transformer_pretrained = config.get('transformer_pretrained', True)
    transformer_cache_dir = config.get('transformer_cache_dir', None)
    
    # Завантажити токенізатор
    tokenizer, actual_vocab_size, _ = load_tokenizer(tokenizer_name)
    vocab_size = actual_vocab_size  # Використати реальний розмір словника
    
    # Створити network (MLPMixer або Transformer)
    if use_transformer:
        network = TransformerBackbone(
            dim=dim,
            depth=depth,
            seq_len=seq_len,
            pretrained=transformer_pretrained,
            model_name=transformer_model,
            cache_dir=transformer_cache_dir
        )
        # Оновити dim з реальної моделі
        dim = network.dim
        depth = network.depth
        seq_len = network.seq_len
    else:
        network = MLPMixer1D(dim=dim, depth=depth, seq_len=seq_len)
    
    # Створити модель
    model = TinyRecursiveModel(
        dim=dim,
        num_tokens=vocab_size,
        network=network,
        num_refinement_blocks=config.get('num_refinement_blocks', 3),
        num_latent_refinements=config.get('num_latent_refinements', 6),
        halt_loss_weight=config.get('halt_loss_weight', 1.0),
        max_recursion_depth=config.get('max_recursion_depth', 20),
        adaptive_recursion=config.get('adaptive_recursion', False),
        timeout_seconds=config.get('timeout_seconds', None),
        thinking_cost_weight=config.get('thinking_cost_weight', 0.01)
    )
    
    # Завантажити ваги
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Створити інференс об'єкт
    return TRMInference(model, tokenizer, device=device, max_seq_len=seq_len)


def find_trained_models(models_dir: str | Path = None) -> List[Dict[str, any]]:
    """
    Знайти всі навчені моделі
    
    Args:
        models_dir: Шлях до папки з моделями (None = автоматично)
    
    Returns:
        Список інформації про моделі
    """
    if models_dir is None:
        project_root = Path(__file__).parent.parent
        # Шукати в кількох місцях
        possible_dirs = [
            project_root / "models" / "trained",
            project_root / "trained_models",
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                models_dir = dir_path
                break
        
        if models_dir is None:
            models_dir = project_root / "models" / "trained"
            models_dir.mkdir(parents=True, exist_ok=True)
    else:
        models_dir = Path(models_dir)
    
    models = []
    for model_file in models_dir.glob("*.pt"):
        model_info = {
            'path': str(model_file.absolute()),
            'name': model_file.stem,
            'filename': model_file.name,
            'size_mb': model_file.stat().st_size / (1024 * 1024),
            'modified': model_file.stat().st_mtime
        }
        models.append(model_info)
    
    # Сортувати за датою модифікації (новіші спочатку)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    return models


def main():
    """CLI для роботи з навченою моделлю"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Робота з навченою TRM моделлю")
    parser.add_argument("--model", type=str, help="Шлях до моделі (.pt)")
    parser.add_argument("--model-name", type=str, dest="model_name", help="Назва моделі для пошуку в models/trained/")
    parser.add_argument("--config", type=str, help="Шлях до конфігурації (JSON)")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda'])
    parser.add_argument("--interactive", action="store_true", help="Інтерактивний режим")
    parser.add_argument("--list", action="store_true", help="Список доступних моделей")
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.list:
        manager.list_models()
        return
    
    if not args.model and not args.model_name:
        model_info = manager.get_default_model()
        if model_info:
            args.model = model_info['path']
            print(f"🎯 Використовується остання модель: {model_info['name']}")
        else:
            print("❌ Вкажіть --model (шлях) або --model-name (назва) або додайте модель в models/trained/")
            return
    
    # Завантажити модель
    if args.model_name:
        print(f"📥 Завантаження моделі за ім'ям: {args.model_name}")
    else:
        print(f"📥 Завантаження моделі: {args.model}")
    inference = load_trained_model(
        model_path=args.model,
        model_name=args.model_name,
        config_path=args.config,
        device=args.device
    )
    print("✅ Модель завантажена!")
    
    if args.interactive:
        inference.interactive_mode()
    else:
        # Тестовий приклад
        result = inference.predict(
            "def hello():\n    return 'world'",
            "Додай параметр name"
        )
        print("\n📝 Тестовий приклад:")
        print(f"Контекст: def hello():\n    return 'world'")
        print(f"Запит: Додай параметр name")
        print(f"\n✅ Результат:")
        print(result['completion'])


if __name__ == "__main__":
    main()

