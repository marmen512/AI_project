"""
Скрипт для використання Phi-3.5-mini для покращення Tiny Recursive Model
Використовує Phi-3.5 для генерації датасету та knowledge distillation
"""

import json
import os
import time
from typing import List, Dict
from pathlib import Path
from datetime import datetime, timedelta

try:
    from llama_cpp import Llama
except ImportError:
    print("⚠️ llama-cpp-python не встановлено. Встановлюю...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--quiet"])
    from llama_cpp import Llama

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️ tqdm не встановлено. Встановлюю...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "--quiet"])
    from tqdm import tqdm


class RAGSystem:
    """Простий RAG (Retrieval-Augmented Generation) для зберігання та пошуку прикладів"""
    
    def __init__(self):
        self.examples = []
        self.max_examples = 50  # Зберігати останні 50 прикладів
    
    def add_example(self, context: str, query: str, completion: str):
        """Додати приклад до RAG"""
        self.examples.append({
            'context': context,
            'query': query,
            'completion': completion
        })
        # Зберігати тільки останні N прикладів
        if len(self.examples) > self.max_examples:
            self.examples.pop(0)
    
    def retrieve_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """Знайти схожі приклади (простий пошук по ключових словах)"""
        if not self.examples:
            return []
        
        query_lower = query.lower()
        scored = []
        
        for ex in self.examples:
            score = 0
            # Простий підрахунок спільних слів
            ex_text = (ex['context'] + ' ' + ex['query']).lower()
            for word in query_lower.split():
                if word in ex_text:
                    score += 1
            scored.append((score, ex))
        
        # Сортувати за релевантністю
        scored.sort(reverse=True, key=lambda x: x[0])
        return [ex for _, ex in scored[:top_k]]


class Phi3ToTRM:
    """Клас для використання Phi-3.5 для покращення TRM"""
    
    def __init__(self, phi3_model_path: str, n_ctx: int = 2048, n_threads: int = None, n_gpu_layers: int = 0):
        """
        Ініціалізація Phi-3.5 моделі
        
        Args:
            phi3_model_path: Шлях до GGUF файлу
            n_ctx: Розмір контексту
            n_threads: Кількість потоків (None = автоматично)
            n_gpu_layers: Кількість шарів на GPU (0 = тільки CPU)
        """
        import multiprocessing
        if n_threads is None:
            n_threads = max(multiprocessing.cpu_count() - 2, 1)  # Залишити 2 ядра для системи
        
        print(f"Завантаження Phi-3.5 з {phi3_model_path}...")
        print(f"   Використання {n_threads} CPU потоків")
        if n_gpu_layers > 0:
            print(f"   {n_gpu_layers} шарів на GPU")
        
        self.llm = Llama(
            model_path=phi3_model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        print("✅ Phi-3.5 завантажена")
        
        # Ініціалізувати RAG систему
        self.rag = RAGSystem()
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Генерація відповіді від Phi-3.5
        
        Args:
            prompt: Запит
            max_tokens: Максимальна кількість токенів
            temperature: Температура генерації
            
        Returns:
            Згенерована відповідь
        """
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["[QUERY]", "\n\n\n"],
            echo=False
        )
        return response['choices'][0]['text'].strip()
    
    def generate_dataset_from_seeds(self, seed_examples: List[Dict], num_generations: int = 100) -> List[Dict]:
        """
        Генерація великого датасету на основі seed прикладів
        
        Args:
            seed_examples: Початкові приклади
            num_generations: Скільки нових прикладів згенерувати
            
        Returns:
            Розширений датасет
        """
        print(f"\nГенерація {num_generations} нових прикладів...")
        dataset = seed_examples.copy()
        
        # Додати seed приклади до RAG
        for ex in seed_examples:
            self.rag.add_example(ex.get('context', ''), ex.get('query', ''), ex.get('completion', ''))
        
        # Ініціалізувати прогрес-бар
        start_time = time.time()
        pbar = tqdm(total=num_generations, desc="Генерація", unit="приклад", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Шаблони для різних типів завдань
        task_templates = [
            "Напиши функцію яка {task}",
            "Створи клас для {task}",
            "Оптимізуй код який {task}",
            "Додай обробку помилок для {task}",
            "Рефактори код який {task}",
            "Додай документацію для {task}",
            "Покращ безпеку коду який {task}",
        ]
        
        task_descriptions = [
            "обчислює суму чисел",
            "читає файл",
            "відправляє HTTP запит",
            "обробляє JSON дані",
            "працює з базою даних",
            "генерує випадкові числа",
            "шифрує дані",
            "валідує email",
            "парсить XML",
            "обчислює статистику",
            "фільтрує список",
            "сортує дані",
            "кешує результати",
            "логує події",
            "відправляє email",
        ]
        
        # Оптимізація: винести import за цикл
        import random
        
        for i in range(num_generations):
            # Випадковий шаблон та опис
            template = random.choice(task_templates)
            description = random.choice(task_descriptions)
            task = template.format(task=description)
            
            # RAG: Знайти схожі приклади
            similar_examples = self.rag.retrieve_similar(task, top_k=2)
            rag_context = ""
            if similar_examples:
                rag_context = "\n\nПриклади схожих завдань:\n"
                for idx, ex in enumerate(similar_examples, 1):
                    rag_context += f"{idx}. Контекст: {ex['context'][:100]}...\n"
                    rag_context += f"   Запит: {ex['query']}\n"
                    rag_context += f"   Відповідь: {ex['completion'][:100]}...\n\n"
            
            # Створити запит для Phi-3.5 з RAG контекстом
            prompt = f"""Ти - експерт Python розробник. Створи приклад коду та запиту.
{rag_context}

Завдання: {task}

Створи JSON об'єкт з такими полями:
- "context": початковий код Python (простий, 5-10 рядків)
- "query": запит користувача що потрібно зробити з кодом
- "completion": очікувана відповідь (покращений/доповнений код)

Формат:
{{
  "context": "def example():\n    pass",
  "query": "Додай функціональність",
  "completion": "def example():\n    # Додана функціональність\n    return result"
}}

Створи приклад:"""
            
            try:
                response = self.generate_response(prompt, max_tokens=800, temperature=0.8)
                
                # Спробувати витягти JSON з відповіді
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    example = json.loads(json_str)
                    
                    # Валідація та очищення
                    if all(k in example for k in ['context', 'query', 'completion']):
                        # Оптимізація: обрізати занадто довгі тексти
                        max_len = 2000  # Максимальна довжина тексту
                        for key in ['context', 'query', 'completion']:
                            if len(example[key]) > max_len:
                                example[key] = example[key][:max_len]
                        
                        dataset.append(example)
                        # Додати до RAG для наступних генерацій
                        self.rag.add_example(
                            example['context'],
                            example['query'],
                            example['completion']
                        )
                
            except (json.JSONDecodeError, KeyError) as e:
                # Якщо не вдалося розпарсити JSON, пропустити
                if i % 10 == 0:  # Логувати помилки кожні 10 ітерацій
                    pass  # Тиха обробка помилок
            except Exception as e:
                # Інші помилки також тихо обробляти
                if i % 10 == 0:
                    pass
            
            # Оновити прогрес-бар
            pbar.update(1)
            
            # Оновити опис з оцінкою часу
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (num_generations - i - 1)
                pbar.set_postfix({
                    'Успішно': len(dataset) - len(seed_examples),
                    'Залишилось': f"{timedelta(seconds=int(remaining))}"
                })
        
        pbar.close()
        elapsed_total = time.time() - start_time
        
        print(f"\n✅ Згенеровано {len(dataset)} прикладів (було {len(seed_examples)})")
        print(f"⏱️ Час генерації: {timedelta(seconds=int(elapsed_total))}")
        print(f"📊 Успішність: {((len(dataset) - len(seed_examples)) / num_generations * 100):.1f}%")
        return dataset
    
    def enhance_existing_dataset(self, dataset_path: str, output_path: str, num_enhancements: int = 50):
        """
        Покращення існуючого датасету через Phi-3.5
        
        Args:
            dataset_path: Шлях до існуючого датасету
            output_path: Шлях для збереження покращеного датасету
            num_enhancements: Скільки нових варіантів створити
        """
        print(f"\nПокращення датасету {dataset_path}...")
        
        # Завантажити існуючий датасет
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"Початковий розмір: {len(dataset)} прикладів")
        
        # Генерувати нові приклади на основі існуючих
        enhanced_dataset = self.generate_dataset_from_seeds(dataset, num_enhancements)
        
        # Зберегти
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Покращений датасет збережено: {output_path}")
        print(f"   Розмір: {len(enhanced_dataset)} прикладів")
        
        return enhanced_dataset
    
    def create_training_dataset(self, output_path: str = "phi3_enhanced_dataset.json", num_examples: int = 500, teacher_model_path: str = None):
        """
        Створити повний датасет для навчання TRM
        
        Args:
            output_path: Шлях для збереження
            num_examples: Скільки прикладів створити
            teacher_model_path: Шлях до teacher моделі (для збереження в метаданих)
        """
        print(f"\nСтворення датасету з {num_examples} прикладів...")
        
        # Початкові seed приклади
        seed_examples = [
            {
                "context": "def add(a, b):\n    return a + b",
                "query": "Додай docstring",
                "completion": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b"
            },
            {
                "context": "def process_data(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result",
                "query": "Перепиши через list comprehension",
                "completion": "def process_data(data):\n    return [item * 2 for item in data]"
            }
        ]
        
        # Генерувати датасет
        dataset = self.generate_dataset_from_seeds(seed_examples, num_examples)
        
        # Додати метадані про teacher модель
        import hashlib
        from pathlib import Path
        
        dataset_metadata = {
            'metadata': {
                'teacher_model_path': teacher_model_path or str(self.llm.model_path) if hasattr(self.llm, 'model_path') else None,
                'teacher_model_name': Path(teacher_model_path).stem if teacher_model_path else None,
                'num_examples': len(dataset),
                'generated_at': str(Path(output_path).stat().st_mtime) if Path(output_path).exists() else None,
            },
            'data': dataset
        }
        
        # Зберегти
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Датасет створено: {output_path}")
        print(f"   Teacher модель: {dataset_metadata['metadata'].get('teacher_model_name', 'невідомо')}")
        return dataset


def get_dataset_name_from_model(model_path: str) -> str:
    """
    Визначити назву датасету з назви GGUF моделі
    
    Args:
        model_path: Шлях до GGUF моделі
        
    Returns:
        Назва датасету у форматі {назва_моделі}_training_dataset.json
    """
    from pathlib import Path
    model_name = Path(model_path).stem  # Без .gguf
    return f"{model_name}_training_dataset.json"


def main():
    import argparse
    import sys
    from pathlib import Path
    
    # Додати config до шляху для імпорту
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from config import GGUFModelManager
        model_manager = GGUFModelManager()
        default_model = model_manager.get_default_model()
        if default_model:
            default_model_path = default_model['path']
        else:
            default_model_path = str(project_root / "models" / "gguf" / "phi-3.5-mini-instruct-q4_k_m.gguf")
    except:
        default_model_path = str(project_root / "models" / "gguf" / "phi-3.5-mini-instruct-q4_k_m.gguf")
    
    parser = argparse.ArgumentParser(description="Використання Phi-3.5 для покращення TRM")
    parser.add_argument("--phi3-model", type=str, default=None,
                       help=f"Шлях до Phi-3.5 GGUF моделі (за замовчуванням: автоматично знайти в models/gguf/)")
    parser.add_argument("--enhance", type=str, default=None,
                       help="Покращити існуючий датасет (шлях до JSON)")
    parser.add_argument("--create", action="store_true",
                       help="Створити новий датасет")
    parser.add_argument("--output", type=str, default=None,
                       help="Шлях для збереження датасету. Якщо не вказано, буде використано назву з GGUF моделі")
    parser.add_argument("--num-examples", type=int, default=200,
                       help="Кількість прикладів для генерації")
    parser.add_argument("--n-ctx", type=int, default=2048,
                       help="Розмір контексту")
    parser.add_argument("--n-threads", type=int, default=None,
                       help="Кількість потоків (None = автоматично)")
    parser.add_argument("--n-gpu-layers", type=int, default=0,
                       help="Кількість шарів на GPU (0 = тільки CPU)")
    
    args = parser.parse_args()
    
    # Автоматично визначити модель якщо не вказана
    if args.phi3_model is None:
        try:
            from config import GGUFModelManager
            model_manager = GGUFModelManager()
            default_model = model_manager.get_default_model()
            if default_model:
                args.phi3_model = default_model['path']
                print(f"🎯 Автоматично використовується модель: {default_model['name']}")
            else:
                print("❌ GGUF моделі не знайдено в models/gguf/")
                print("   Додайте .gguf файл в папку models/gguf/")
                return
        except Exception as e:
            print(f"⚠️ Помилка автоматичного визначення моделі: {e}")
            print("   Вкажіть --phi3-model вручну")
            return
    
    # Перевірити наявність моделі
    if not os.path.exists(args.phi3_model):
        print(f"❌ Модель не знайдена: {args.phi3_model}")
        print(f"   Перевірте шлях до файлу або додайте модель в models/gguf/")
        return
    
    # Визначити назву датасету якщо не вказано
    if args.output is None:
        dataset_name = get_dataset_name_from_model(args.phi3_model)
        # Зберігати в datasets/train/ за замовчуванням
        datasets_train_dir = project_root / "datasets" / "train"
        datasets_train_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(datasets_train_dir / dataset_name)
        print(f"📝 Назва датасету визначена з моделі: {dataset_name}")
        print(f"📁 Шлях збереження: {args.output}")
    
    # Ініціалізувати Phi-3.5
    try:
        phi3 = Phi3ToTRM(
            args.phi3_model,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers
        )
    except Exception as e:
        print(f"❌ Помилка завантаження Phi-3.5: {e}")
        return
    
    # Обробити залежно від режиму
    if args.enhance:
        # Покращити існуючий датасет
        phi3.enhance_existing_dataset(
            args.enhance,
            args.output,
            args.num_examples
        )
    elif args.create:
        # Створити новий датасет
        phi3.create_training_dataset(
            args.output,
            args.num_examples,
            teacher_model_path=args.phi3_model
        )
    else:
        print("Вкажіть --enhance або --create")
        print("\nПриклади використання:")
        print("  # Створити новий датасет:")
        print("  python phi3_to_trm.py --create --num-examples 500")
        print("\n  # Покращити існуючий:")
        print("  python phi3_to_trm.py --enhance ai_assistant_dataset.json --num-examples 200")


if __name__ == "__main__":
    main()

