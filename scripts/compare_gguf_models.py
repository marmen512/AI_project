"""
Скрипт для порівняння різних GGUF моделей та навчених TRM моделей
Порівнює швидкість, якість та використання пам'яті
"""
import sys
import argparse
import json
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python не встановлено (GGUF моделі не будуть доступні)")
    print("   Встановіть: pip install llama-cpp-python")

from config.model_loader import GGUFModelManager
from inference.model_inference import load_trained_model, find_trained_models, TRMInference


class UnifiedModelBenchmark:
    """Уніфікований бенчмарк для порівняння GGUF та TRM моделей"""
    
    def __init__(self, test_cases: List[Dict[str, str]]):
        """
        Ініціалізація бенчмарку
        
        Args:
            test_cases: Список тестових випадків з 'prompt' та 'expected_keywords'
        """
        self.test_cases = test_cases
        self.loaded_models: Dict[str, Any] = {}  # Кеш завантажених моделей
    
    def load_gguf_model(self, model_path: str, n_ctx: int = 2048, n_threads: int = None, n_gpu_layers: int = 0) -> Llama:
        """Завантажити GGUF модель"""
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python не встановлено")
        try:
            model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Помилка завантаження GGUF моделі {model_path}: {e}")
    
    def load_trm_model(self, model_path: str, config_path: Optional[str] = None, device: str = 'cpu') -> TRMInference:
        """Завантажити TRM модель"""
        try:
            inference = load_trained_model(
                model_path=model_path,
                config_path=config_path,
                device=device
            )
            return inference
        except Exception as e:
            raise RuntimeError(f"Помилка завантаження TRM моделі {model_path}: {e}")
    
    def is_gguf_model(self, model_path: str) -> bool:
        """Перевірити чи це GGUF модель"""
        return Path(model_path).suffix.lower() == '.gguf'
    
    def is_trm_model(self, model_path: str) -> bool:
        """Перевірити чи це TRM модель"""
        return Path(model_path).suffix.lower() in ['.pt', '.pth']
    
    def generate_response_gguf(self, model: Llama, prompt: str, max_tokens: int = 256) -> Dict:
        """
        Згенерувати відповідь з GGUF моделі та виміряти метрики
        
        Returns:
            Словник з відповіддю та метриками
        """
        start_time = time.time()
        
        try:
            response = model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["<|end|>", "<|endoftext|>", "\n\n\n"],
                echo=False
            )
            
            generation_time = time.time() - start_time
            
            # Отримати текст відповіді
            if isinstance(response, dict):
                text = response.get('choices', [{}])[0].get('text', '')
                tokens_generated = response.get('usage', {}).get('completion_tokens', 0)
            else:
                text = str(response)
                tokens_generated = len(text.split())
            
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            return {
                'text': text,
                'generation_time': generation_time,
                'tokens_generated': tokens_generated,
                'tokens_per_second': tokens_per_second,
                'success': True
            }
        except Exception as e:
            return {
                'text': '',
                'error': str(e),
                'generation_time': time.time() - start_time,
                'success': False
            }
    
    def generate_response_trm(self, inference: TRMInference, prompt: str, max_tokens: int = 256) -> Dict:
        """
        Згенерувати відповідь з TRM моделі та виміряти метрики
        
        Args:
            inference: TRMInference об'єкт
            prompt: Текст запиту (буде використано як query, context буде порожнім)
            max_tokens: Максимальна кількість токенів (не використовується для TRM)
        
        Returns:
            Словник з відповіддю та метриками
        """
        start_time = time.time()
        
        try:
            # TRM моделі очікують context та query
            # Якщо prompt містить контекст, спробуємо розділити
            context = ""
            query = prompt
            
            # Простий розділювач для контексту та запиту
            if "|CONTEXT|" in prompt:
                parts = prompt.split("|CONTEXT|", 1)
                context = parts[0].strip()
                query = parts[1].strip() if len(parts) > 1 else prompt
            elif "\n\nQuery:" in prompt or "\n\nЗапит:" in prompt:
                # Спробувати знайти розділювач
                for sep in ["\n\nQuery:", "\n\nЗапит:", "\n\n---\n\n"]:
                    if sep in prompt:
                        parts = prompt.split(sep, 1)
                        context = parts[0].strip()
                        query = parts[1].strip()
                        break
            
            result = inference.predict(
                context=context,
                query=query,
                max_deep_refinement_steps=12,
                halt_prob_thres=0.5
            )
            
            generation_time = time.time() - start_time
            text = result.get('completion', '')
            
            # Оцінити кількість токенів (приблизно)
            tokens_generated = len(text.split()) if text else 0
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            return {
                'text': text,
                'generation_time': generation_time,
                'tokens_generated': tokens_generated,
                'tokens_per_second': tokens_per_second,
                'exit_steps': result.get('exit_steps', 0),
                'success': True
            }
        except Exception as e:
            return {
                'text': '',
                'error': str(e),
                'generation_time': time.time() - start_time,
                'success': False
            }
    
    def evaluate_quality(self, response_text: str, expected_keywords: List[str]) -> Dict:
        """
        Оцінити якість відповіді
        
        Args:
            response_text: Текст відповіді
            expected_keywords: Очікувані ключові слова
        
        Returns:
            Словник з метриками якості
        """
        if not response_text:
            return {
                'keyword_match': 0.0,
                'length': 0,
                'has_structure': False,
                'quality_score': 0.0
            }
        
        response_lower = response_text.lower()
        
        # Перевірка ключових слів
        matched_keywords = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        keyword_match = matched_keywords / len(expected_keywords) if expected_keywords else 0.0
        
        # Перевірка структури
        has_structure = any(marker in response_text for marker in ['\n', '. ', '! ', '? '])
        
        # Базова оцінка якості
        quality_score = (
            keyword_match * 0.5 +  # 50% за ключові слова
            (1.0 if len(response_text) > 20 else len(response_text) / 20) * 0.3 +  # 30% за довжину
            (1.0 if has_structure else 0.0) * 0.2  # 20% за структуру
        )
        
        return {
            'keyword_match': keyword_match,
            'matched_keywords': matched_keywords,
            'total_keywords': len(expected_keywords),
            'length': len(response_text),
            'has_structure': has_structure,
            'quality_score': quality_score
        }
    
    def benchmark_model(self, model_path: str, model_name: str = None, model_type: str = 'auto', config_path: Optional[str] = None, device: str = 'cpu', **model_kwargs) -> Dict:
        """
        Забенчмаркувати одну модель (GGUF або TRM)
        
        Args:
            model_path: Шлях до моделі
            model_name: Ім'я моделі (для звіту)
            model_type: Тип моделі ('gguf', 'trm', або 'auto' для автоматичного визначення)
            config_path: Шлях до конфігурації TRM моделі (опціонально)
            device: Пристрій для TRM моделі ('cpu' або 'cuda')
            **model_kwargs: Параметри для GGUF моделі (n_ctx, n_threads, n_gpu_layers)
        
        Returns:
            Словник з результатами бенчмарку
        """
        if model_name is None:
            model_name = Path(model_path).stem
        
        # Визначити тип моделі
        if model_type == 'auto':
            if self.is_gguf_model(model_path):
                model_type = 'gguf'
            elif self.is_trm_model(model_path):
                model_type = 'trm'
            else:
                return {
                    'model_name': model_name,
                    'model_path': model_path,
                    'error': f'Невідомий тип моделі: {model_path}',
                    'success': False
                }
        
        print(f"\n{'='*70}")
        print(f"📊 БЕНЧМАРК: {model_name} ({model_type.upper()})")
        print(f"{'='*70}")
        print(f"📁 Шлях: {model_path}")
        
        # Завантажити модель
        print("⏳ Завантаження моделі...")
        try:
            if model_type == 'gguf':
                if not LLAMA_CPP_AVAILABLE:
                    raise RuntimeError("llama-cpp-python не встановлено")
                model = self.load_gguf_model(model_path, **model_kwargs)
                print("✅ GGUF модель завантажена")
            elif model_type == 'trm':
                model = self.load_trm_model(model_path, config_path, device)
                print("✅ TRM модель завантажена")
            else:
                raise ValueError(f"Невідомий тип моделі: {model_type}")
        except Exception as e:
            print(f"❌ Помилка: {e}")
            return {
                'model_name': model_name,
                'model_path': model_path,
                'model_type': model_type,
                'error': str(e),
                'success': False
            }
        
        # Запустити тести
        results = []
        total_time = 0
        total_tokens = 0
        
        print(f"\n🧪 Запуск {len(self.test_cases)} тестів...")
        
        for i, test_case in enumerate(self.test_cases, 1):
            prompt = test_case.get('prompt', '')
            expected_keywords = test_case.get('expected_keywords', [])
            test_name = test_case.get('name', f'Test {i}')
            
            print(f"\n  Тест {i}/{len(self.test_cases)}: {test_name}")
            
            # Генерація залежно від типу моделі
            if model_type == 'gguf':
                gen_result = self.generate_response_gguf(model, prompt, max_tokens=test_case.get('max_tokens', 256))
            else:  # TRM
                gen_result = self.generate_response_trm(model, prompt, max_tokens=test_case.get('max_tokens', 256))
            
            if not gen_result['success']:
                print(f"    ❌ Помилка: {gen_result.get('error', 'Unknown')}")
                results.append({
                    'test_name': test_name,
                    'success': False,
                    'error': gen_result.get('error')
                })
                continue
            
            # Оцінка якості
            quality = self.evaluate_quality(gen_result['text'], expected_keywords)
            
            total_time += gen_result['generation_time']
            total_tokens += gen_result['tokens_generated']
            
            print(f"    ⏱️  Час: {gen_result['generation_time']:.2f}с")
            print(f"    📝 Токенів: {gen_result['tokens_generated']}")
            print(f"    ⚡ Швидкість: {gen_result['tokens_per_second']:.1f} токенів/с")
            print(f"    📊 Якість: {quality['quality_score']:.2%}")
            print(f"    🔑 Ключові слова: {quality['matched_keywords']}/{quality['total_keywords']}")
            if 'exit_steps' in gen_result:
                print(f"    🔄 Кроків уточнення: {gen_result['exit_steps']}")
            
            result_item = {
                'test_name': test_name,
                'success': True,
                'generation_time': gen_result['generation_time'],
                'tokens_generated': gen_result['tokens_generated'],
                'tokens_per_second': gen_result['tokens_per_second'],
                'response_length': len(gen_result['text']),
                'quality': quality
            }
            if 'exit_steps' in gen_result:
                result_item['exit_steps'] = gen_result['exit_steps']
            
            results.append(result_item)
        
        # Підсумок
        successful_tests = sum(1 for r in results if r.get('success', False))
        avg_time = total_time / successful_tests if successful_tests > 0 else 0
        avg_tokens_per_sec = sum(r.get('tokens_per_second', 0) for r in results if r.get('success')) / successful_tests if successful_tests > 0 else 0
        avg_quality = sum(r.get('quality', {}).get('quality_score', 0) for r in results if r.get('success')) / successful_tests if successful_tests > 0 else 0
        
        print(f"\n📈 ПІДСУМОК:")
        print(f"   Успішних тестів: {successful_tests}/{len(self.test_cases)}")
        print(f"   Середній час генерації: {avg_time:.2f}с")
        print(f"   Середня швидкість: {avg_tokens_per_sec:.1f} токенів/с")
        print(f"   Середня якість: {avg_quality:.2%}")
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': model_type,
            'success': True,
            'total_tests': len(self.test_cases),
            'successful_tests': successful_tests,
            'total_time': total_time,
            'total_tokens': total_tokens,
            'avg_generation_time': avg_time,
            'avg_tokens_per_second': avg_tokens_per_sec,
            'avg_quality_score': avg_quality,
            'test_results': results
        }
    
    def compare_models(self, model_paths: List[str], model_names: List[str] = None, model_types: List[str] = None, config_paths: List[Optional[str]] = None, device: str = 'cpu', **model_kwargs) -> Dict:
        """
        Порівняти кілька моделей (GGUF та/або TRM)
        
        Args:
            model_paths: Список шляхів до моделей
            model_names: Список імен моделей (опціонально)
            model_types: Список типів моделей ('gguf', 'trm', або 'auto')
            config_paths: Список шляхів до конфігурацій TRM моделей (опціонально)
            device: Пристрій для TRM моделей ('cpu' або 'cuda')
            **model_kwargs: Параметри для GGUF моделей (n_ctx, n_threads, n_gpu_layers)
        
        Returns:
            Словник з результатами порівняння
        """
        if model_names is None:
            model_names = [Path(p).stem for p in model_paths]
        
        if model_types is None:
            model_types = ['auto'] * len(model_paths)
        elif len(model_types) < len(model_paths):
            model_types.extend(['auto'] * (len(model_paths) - len(model_types)))
        
        if config_paths is None:
            config_paths = [None] * len(model_paths)
        elif len(config_paths) < len(model_paths):
            config_paths.extend([None] * (len(model_paths) - len(config_paths)))
        
        print("="*70)
        print("🔬 ПОРІВНЯННЯ МОДЕЛЕЙ (GGUF та TRM)")
        print("="*70)
        print(f"\n📋 Тестових випадків: {len(self.test_cases)}")
        print(f"📦 Моделей для порівняння: {len(model_paths)}")
        
        # Показати типи моделей
        gguf_count = sum(1 for mt in model_types if mt == 'gguf' or (mt == 'auto' and any(self.is_gguf_model(p) for p in model_paths)))
        trm_count = sum(1 for mt in model_types if mt == 'trm' or (mt == 'auto' and any(self.is_trm_model(p) for p in model_paths)))
        if gguf_count > 0:
            print(f"   - GGUF моделей: {gguf_count}")
        if trm_count > 0:
            print(f"   - TRM моделей: {trm_count}")
        
        benchmark_results = []
        
        for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
            model_type = model_types[i] if i < len(model_types) else 'auto'
            config_path = config_paths[i] if i < len(config_paths) else None
            result = self.benchmark_model(
                model_path, 
                model_name, 
                model_type=model_type,
                config_path=config_path,
                device=device,
                **model_kwargs
            )
            benchmark_results.append(result)
        
        # Створити звіт порівняння
        comparison = self.create_comparison_report(benchmark_results)
        
        return comparison
    
    def create_comparison_report(self, benchmark_results: List[Dict]) -> Dict:
        """Створити звіт порівняння"""
        successful_results = [r for r in benchmark_results if r.get('success', False)]
        
        if not successful_results:
            print("\n❌ Жодна модель не пройшла тести")
            return {'error': 'No successful benchmarks'}
        
        print("\n" + "="*70)
        print("📊 ЗВІТ ПОРІВНЯННЯ")
        print("="*70)
        
        # Таблиця порівняння
        print("\n📈 ПОРІВНЯЛЬНА ТАБЛИЦЯ:")
        print("-" * 70)
        print(f"{'Модель':<30} {'Швидкість':<15} {'Якість':<10} {'Успішність':<10}")
        print("-" * 70)
        
        for result in successful_results:
            name = result['model_name'][:28]
            speed = f"{result['avg_tokens_per_second']:.1f} ток/с"
            quality = f"{result['avg_quality_score']:.1%}"
            success_rate = f"{result['successful_tests']}/{result['total_tests']}"
            print(f"{name:<30} {speed:<15} {quality:<10} {success_rate:<10}")
        
        # Найкращі моделі
        print("\n🏆 НАЙКРАЩІ МОДЕЛІ:")
        
        if successful_results:
            fastest = max(successful_results, key=lambda x: x['avg_tokens_per_second'])
            best_quality = max(successful_results, key=lambda x: x['avg_quality_score'])
            most_reliable = max(successful_results, key=lambda x: x['successful_tests'] / x['total_tests'])
            
            print(f"   ⚡ Найшвидша: {fastest['model_name']} ({fastest['avg_tokens_per_second']:.1f} токенів/с)")
            print(f"   🎯 Найкраща якість: {best_quality['model_name']} ({best_quality['avg_quality_score']:.1%})")
            print(f"   ✅ Найнадійніша: {most_reliable['model_name']} ({most_reliable['successful_tests']}/{most_reliable['total_tests']} тестів)")
        
        # Зберегти звіт
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_cases_count': len(self.test_cases),
            'models_compared': len(benchmark_results),
            'successful_benchmarks': len(successful_results),
            'benchmark_results': benchmark_results,
            'comparison': {
                'fastest': fastest['model_name'] if successful_results else None,
                'best_quality': best_quality['model_name'] if successful_results else None,
                'most_reliable': most_reliable['model_name'] if successful_results else None
            }
        }
        
        # Створити папку temp якщо не існує
        temp_dir = project_root / "temp"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        report_path = temp_dir / f"gguf_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Звіт збережено: {report_path}")
        
        return report


def get_default_test_cases() -> List[Dict[str, str]]:
    """Отримати стандартні тестові випадки"""
    return [
        {
            'name': 'Простий запит',
            'prompt': 'What is Python?',
            'expected_keywords': ['python', 'programming', 'language'],
            'max_tokens': 128
        },
        {
            'name': 'Код генерація',
            'prompt': 'Write a Python function to calculate factorial:',
            'expected_keywords': ['def', 'factorial', 'return', 'function'],
            'max_tokens': 256
        },
        {
            'name': 'Пояснення концепції',
            'prompt': 'Explain what is machine learning in simple terms:',
            'expected_keywords': ['machine', 'learning', 'data', 'algorithm'],
            'max_tokens': 256
        },
        {
            'name': 'Математичне завдання',
            'prompt': 'Solve: What is 15 * 23? Show your work.',
            'expected_keywords': ['15', '23', '345', 'multiply'],
            'max_tokens': 128
        },
        {
            'name': 'Складний запит',
            'prompt': 'How does a neural network learn? Explain the backpropagation process.',
            'expected_keywords': ['neural', 'network', 'backpropagation', 'gradient', 'weights'],
            'max_tokens': 512
        }
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Порівняння GGUF та TRM моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  # Порівняти всі знайдені GGUF моделі
  python scripts/compare_gguf_models.py --all
  
  # Порівняти конкретні GGUF моделі
  python scripts/compare_gguf_models.py \\
      --models models/gguf/phi-3.5-mini-instruct-q4_k_m.gguf \\
                models/gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
  
  # Порівняти TRM модель з GGUF моделями
  python scripts/compare_gguf_models.py \\
      --models models/trained/my_model.pt \\
                models/gguf/phi-3.5-mini-instruct-q4_k_m.gguf \\
      --trm-config models/trained/my_model_config.json
  
  # Порівняти тільки TRM моделі
  python scripts/compare_gguf_models.py \\
      --trm-models models/trained/model1.pt models/trained/model2.pt
  
  # Порівняти з кастомними тестами
  python scripts/compare_gguf_models.py --all --test-file my_tests.json
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        type=str,
        help='Шляхи до моделей для порівняння (GGUF або TRM, визначається автоматично)'
    )
    
    parser.add_argument(
        '--trm-models',
        nargs='+',
        type=str,
        help='Шляхи до TRM моделей для порівняння'
    )
    
    parser.add_argument(
        '--trm-config',
        type=str,
        help='Шлях до конфігурації TRM моделі (використовується для всіх TRM моделей)'
    )
    
    parser.add_argument(
        '--trm-configs',
        nargs='+',
        type=str,
        help='Шляхи до конфігурацій TRM моделей (по одній на модель)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Порівняти всі знайдені GGUF моделі'
    )
    
    parser.add_argument(
        '--all-trm',
        action='store_true',
        help='Порівняти всі знайдені TRM моделі'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        help='JSON файл з тестовими випадками'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Пристрій для TRM моделей (за замовчуванням: auto)'
    )
    
    parser.add_argument(
        '--n-ctx',
        type=int,
        default=2048,
        help='Розмір контексту для GGUF моделей (за замовчуванням: 2048)'
    )
    
    parser.add_argument(
        '--n-threads',
        type=int,
        default=None,
        help='Кількість потоків для GGUF моделей (None = автоматично)'
    )
    
    parser.add_argument(
        '--n-gpu-layers',
        type=int,
        default=0,
        help='Кількість шарів на GPU для GGUF моделей (0 = тільки CPU)'
    )
    
    args = parser.parse_args()
    
    # Визначити пристрій для TRM
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Визначити моделі для порівняння
    model_paths = []
    model_names = []
    model_types = []
    config_paths = []
    
    if args.all:
        # Знайти всі GGUF моделі
        if not LLAMA_CPP_AVAILABLE:
            print("❌ llama-cpp-python не встановлено, не можу порівняти GGUF моделі")
            return
        
        manager = GGUFModelManager()
        models = manager.get_models()
        
        if not models:
            print("❌ GGUF моделі не знайдено")
            print(f"   Додайте .gguf файли в папку: {manager.models_dir}")
            return
        
        model_paths = [m['path'] for m in models]
        model_names = [m['name'] for m in models]
        model_types = ['gguf'] * len(models)
        config_paths = [None] * len(models)
        
        print(f"📦 Знайдено {len(models)} GGUF моделей для порівняння:")
        for m in models:
            print(f"   - {m['name']}")
    
    elif args.all_trm:
        # Знайти всі TRM моделі
        models = find_trained_models()
        
        if not models:
            print("❌ TRM моделі не знайдено")
            print(f"   Додайте .pt файли в папку models/trained/")
            return
        
        model_paths = [m['path'] for m in models]
        model_names = [m['name'] for m in models]
        model_types = ['trm'] * len(models)
        config_paths = [None] * len(models)
        
        print(f"📦 Знайдено {len(models)} TRM моделей для порівняння:")
        for m in models:
            print(f"   - {m['name']}")
    
    elif args.trm_models:
        # Тільки TRM моделі
        model_paths = args.trm_models
        model_names = [Path(p).stem for p in model_paths]
        model_types = ['trm'] * len(model_paths)
        
        # Обробити конфігурації
        if args.trm_config:
            config_paths = [args.trm_config] * len(model_paths)
        elif args.trm_configs:
            config_paths = list(args.trm_configs)
            if len(config_paths) < len(model_paths):
                config_paths.extend([None] * (len(model_paths) - len(config_paths)))
        else:
            config_paths = [None] * len(model_paths)
    
    elif args.models:
        # Змішаний список моделей
        model_paths = args.models
        model_names = []
        model_types = []
        
        for path in model_paths:
            path_obj = Path(path)
            model_names.append(path_obj.stem)
            
            # Визначити тип моделі
            if path_obj.suffix.lower() == '.gguf':
                model_types.append('gguf')
            elif path_obj.suffix.lower() in ['.pt', '.pth']:
                model_types.append('trm')
            else:
                model_types.append('auto')
        
        # Обробити конфігурації для TRM моделей
        config_paths = []
        for i, model_type in enumerate(model_types):
            if model_type == 'trm':
                if args.trm_config:
                    config_paths.append(args.trm_config)
                elif args.trm_configs and i < len(args.trm_configs):
                    config_paths.append(args.trm_configs[i])
                else:
                    config_paths.append(None)
            else:
                config_paths.append(None)
    
    else:
        print("❌ Вкажіть --models, --trm-models, --all або --all-trm")
        parser.print_help()
        return
    
    # Завантажити тестові випадки
    if args.test_file and Path(args.test_file).exists():
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            if isinstance(test_data, list):
                test_cases = test_data
            elif isinstance(test_data, dict) and 'test_cases' in test_data:
                test_cases = test_data['test_cases']
            else:
                test_cases = get_default_test_cases()
    else:
        test_cases = get_default_test_cases()
    
    # Створити бенчмарк
    benchmark = UnifiedModelBenchmark(test_cases)
    
    # Запустити порівняння
    comparison = benchmark.compare_models(
        model_paths,
        model_names,
        model_types=model_types,
        config_paths=config_paths,
        device=device,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers
    )
    
    print("\n" + "="*70)
    print("✅ ПОРІВНЯННЯ ЗАВЕРШЕНО")
    print("="*70)


if __name__ == "__main__":
    main()

