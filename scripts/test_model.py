"""
Скрипт для тестування навченої моделі на eval датасеті
"""
import sys
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Tuple

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_model_quality(
    model: torch.nn.Module,
    tokenizer,
    dataset_path: Path,
    max_tests: int = None,
    seq_len: int = 512
) -> Dict:
    """
    Тестування якості моделі на eval датасеті
    
    Args:
        model: Навчена модель
        tokenizer: Токенізатор
        dataset_path: Шлях до eval датасету
        max_tests: Максимальна кількість тестів (None = всі)
        seq_len: Довжина послідовності
    
    Returns:
        Словник з результатами тестування
    """
    import json
    
    # Завантажити eval датасет
    print(f"📚 Завантаження eval датасету: {dataset_path.name}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Підтримка метаданих
    if isinstance(data, dict) and 'data' in data:
        test_cases = data['data']
    elif isinstance(data, list):
        test_cases = data
    else:
        test_cases = []
    
    if max_tests:
        test_cases = test_cases[:max_tests]
    
    print(f"📊 Тестових прикладів: {len(test_cases)}")
    
    model.eval()
    correct = 0
    total = len(test_cases)
    results = []
    
    print("\n[TEST] Початок тестування...")
    print("=" * 80)
    
    # Використати tqdm для прогрес-бару якщо доступний
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
    
    if HAS_TQDM:
        pbar = tqdm(total=total, desc="Тестування", unit="тест", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        test_iterator = enumerate(test_cases, 1)
    else:
        test_iterator = enumerate(test_cases, 1)
        pbar = None
    
    with torch.no_grad():
        for i, test_case in test_iterator:
            try:
                # Отримати context, query, completion
                context = test_case.get('context', '')
                query = test_case.get('query', '')
                expected_output = test_case.get('completion', '')
                
                # Підготувати вхід
                if context and query:
                    input_text = f"{context}\n\n{query}\n"
                elif query:
                    input_text = query
                else:
                    input_text = context

                device = next(model.parameters()).device

                enc = tokenizer(
                    input_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=seq_len,
                    padding=False,
                    add_special_tokens=False,
                )
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                gen_ids = gen_ids[0]
                prompt_len = int(input_ids.shape[1])
                completion_ids = gen_ids[prompt_len:]
                predicted_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                
                # Метрики accuracy
                exact_match = (
                    predicted_text.strip().lower() == expected_output.strip().lower()
                )
                
                partial_match = (
                    expected_output.lower().strip() in predicted_text.lower().strip()
                )
                
                expected_words = set(expected_output.lower().split())
                predicted_words = set(predicted_text.lower().split())
                keyword_match = len(expected_words & predicted_words) / max(len(expected_words), 1) > 0.5
                
                is_correct = exact_match or partial_match or keyword_match
                
                score = 0.0
                if exact_match:
                    score = 1.0
                elif partial_match:
                    score = 0.7
                elif keyword_match:
                    score = 0.5
                
                if is_correct:
                    correct += 1
                
                results.append({
                    'input': input_text[:80],
                    'expected': expected_output[:80],
                    'predicted': predicted_text[:120],
                    'correct': is_correct,
                    'score': score,
                    'exact_match': exact_match,
                    'partial_match': partial_match,
                    'keyword_match': keyword_match,
                    'exit_steps': 0
                })
                
                # Оновити прогрес-бар
                if HAS_TQDM and pbar:
                    accuracy = (correct / i) * 100
                    pbar.set_postfix({
                        'Accuracy': f'{accuracy:.1f}%',
                        'Correct': f'{correct}/{i}'
                    })
                    pbar.update(1)
                else:
                    status = "[OK]" if is_correct else "[FAIL]"
                    if i % 10 == 0 or i == total:
                        print(f"{status} Тест {i}/{total}: {'ПРАВИЛЬНО' if is_correct else 'НЕПРАВИЛЬНО'}")
                
            except Exception as e:
                if HAS_TQDM and pbar:
                    pbar.update(1)
                else:
                    print(f"[ERROR] Тест {i}/{total}: ПОМИЛКА - {e}")
                results.append({
                    'input': str(test_case.get('context', ''))[:50],
                    'error': str(e),
                    'correct': False
                })
    
    if HAS_TQDM and pbar:
        pbar.close()
    
    # Підрахунок статистики
    accuracy = correct / total if total > 0 else 0.0
    avg_score = sum(r.get('score', 0.0) for r in results) / len(results) if results else 0.0
    avg_exit_steps = (
        sum(r.get('exit_steps', 0) for r in results) / len(results)
        if results else 0
    )
    
    exact_matches = sum(1 for r in results if r.get('exact_match', False))
    partial_matches = sum(1 for r in results if r.get('partial_match', False))
    keyword_matches = sum(1 for r in results if r.get('keyword_match', False))
    
    # Вивід результатів
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТИ ТЕСТУВАННЯ")
    print("=" * 80)
    print(f"   Загальна точність: {correct}/{total} ({accuracy*100:.1f}%)")
    print(f"   Середній score: {avg_score:.2f}/1.0")
    print(f"   Точні співпадіння: {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
    print(f"   Часткові співпадіння: {partial_matches}/{total} ({partial_matches/total*100:.1f}%)")
    print(f"   Семантичні співпадіння: {keyword_matches}/{total} ({keyword_matches/total*100:.1f}%)")
    print(f"   Середня кількість кроків: {avg_exit_steps:.1f}")
    print("=" * 80)
    
    return {
        'accuracy': accuracy,
        'avg_score': avg_score,
        'correct': correct,
        'total': total,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'keyword_matches': keyword_matches,
        'avg_exit_steps': avg_exit_steps,
        'results': results
    }


def generate_single_prompt(
    *,
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: str,
    seq_len: int,
    max_new_tokens: int,
    raw: bool,
):
    if not raw:
        prompt = f"""Instruction:\n{prompt}\n\nResponse:\n"""
    enc = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=seq_len,
        padding=False,
        add_special_tokens=False,
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = gen_ids[0]
    completion_ids = gen_ids[int(input_ids.shape[1]):]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def _infer_n_layer_from_state_dict(state_dict: dict) -> int:
    max_idx = -1
    prefix = 'transformer.h.'
    for k in state_dict.keys():
        if k.startswith(prefix):
            rest = k[len(prefix):]
            idx_str = rest.split('.', 1)[0]
            if idx_str.isdigit():
                max_idx = max(max_idx, int(idx_str))
    return max_idx + 1 if max_idx >= 0 else 0


def _choose_default_n_head(n_embd: int) -> int:
    if n_embd % 8 == 0:
        return 8
    return 1


def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(
        description="Тестування навченої TRM моделі на eval датасеті",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  # Тестування з автоматичним визначенням eval датасету:
  python scripts/test_model.py --model models/trained/trm_openassistant_train.pt
  
  # Тестування на конкретному датасеті:
  python scripts/test_model.py \\
      --model models/trained/trm_openassistant_train.pt \\
      --dataset datasets/eval/openassistant_eval.json
  
  # Тестування з обмеженням кількості тестів:
  python scripts/test_model.py \\
      --model models/trained/trm_openassistant_train.pt \\
      --max-tests 50
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Шлях до навченої моделі (.pt файл)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Шлях до eval датасету (за замовчуванням: автоматично знайти в datasets/eval/)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Згенерувати відповідь на один prompt і вийти"
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Не обгортати prompt у Instruction/Response шаблон"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Максимум нових токенів у відповіді (для --prompt)"
    )
    
    parser.add_argument(
        "--max-tests",
        type=int,
        default=None,
        help="Максимальна кількість тестів (за замовчуванням: всі)"
    )
    
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Довжина послідовності (за замовчуванням: 512)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Пристрій для тестування (за замовчуванням: auto)"
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="Override number of attention heads (only if checkpoint config is missing)"
    )
    
    args = parser.parse_args()
    
    # Перевірити наявність моделі
    model_path = project_root / args.model
    if not model_path.exists():
        print(f"❌ Модель не знайдено: {model_path}")
        return

    dataset_path = None
    if args.prompt is None:
        if args.dataset:
            dataset_path = project_root / args.dataset
        else:
            eval_dir = project_root / "datasets" / "eval"
            eval_datasets = list(eval_dir.glob("*.json"))
            if eval_datasets:
                dataset_path = eval_datasets[0]
                print(f"📚 Автоматично вибрано eval датасет: {dataset_path.name}")
            else:
                print(f"❌ Eval датасет не знайдено в {eval_dir}")
                print(f"   Спочатку завантажте датасет:")
                print(f"   python scripts/download_openassistant.py")
                return

        if not dataset_path.exists():
            print(f"❌ Датасет не знайдено: {dataset_path}")
            return
    
    # Визначити пристрій
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 80)
    print("🧪 ТЕСТУВАННЯ МОДЕЛІ")
    print("=" * 80)
    print(f"\n🤖 Модель: {model_path.name}")
    if dataset_path is not None:
        print(f"📚 Датасет: {dataset_path.name}")
    print(f"💻 Пристрій: {device}")
    
    # Завантажити модель
    print(f"\n📥 Завантаження моделі...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            ckpt_config = ckpt.get('config') if isinstance(ckpt.get('config'), dict) else None
        else:
            state_dict = ckpt
            ckpt_config = None

        # Infer key dimensions from checkpoint to avoid shape mismatches.
        wpe = state_dict.get('transformer.wpe.weight')
        wte = state_dict.get('transformer.wte.weight')
        if wpe is None or wte is None:
            raise KeyError("Checkpoint missing transformer.wpe.weight or transformer.wte.weight")

        inferred_n_positions = int(wpe.shape[0])
        inferred_n_embd = int(wte.shape[1])
        inferred_vocab_size = int(wte.shape[0])

        inferred_n_layer = _infer_n_layer_from_state_dict(state_dict)

        # Cap runtime seq_len to what the model was trained with.
        effective_seq_len = min(int(args.seq_len), inferred_n_positions)
        if int(args.seq_len) != effective_seq_len:
            print(f"⚠️ seq_len capped: requested={args.seq_len} model_n_positions={inferred_n_positions} using={effective_seq_len}")

        # Use checkpoint config if present to avoid silent hyperparam mismatches (especially n_head).
        model_cfg = {}
        if isinstance(ckpt_config, dict):
            model_cfg = ckpt_config.get('model', {}) if isinstance(ckpt_config.get('model'), dict) else {}

        n_layer = int(model_cfg.get('depth', inferred_n_layer or 6))
        n_embd = int(model_cfg.get('dim', inferred_n_embd))

        if args.n_heads is not None:
            n_head = int(args.n_heads)
        else:
            heads_from_cfg = model_cfg.get('heads', None)
            if heads_from_cfg is None:
                n_head = _choose_default_n_head(n_embd)
            else:
                n_head = int(heads_from_cfg)

        if n_embd % n_head != 0:
            raise RuntimeError(f"Invalid head count: n_embd={n_embd} not divisible by n_head={n_head}")

        # Phase2 архітектура має співпасти з training
        model_config = GPT2Config(
            vocab_size=inferred_vocab_size,
            n_positions=inferred_n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_embd * 4,
            use_cache=False,
        )
        model = GPT2LMHeadModel(model_config)

        print(
            f"ℹ️ model_config: vocab={inferred_vocab_size} n_positions={inferred_n_positions} "
            f"n_embd={n_embd} n_layer={n_layer} n_head={n_head}"
        )

        # Load weights BEFORE resizing embeddings (otherwise wte/lm_head shape mismatches).
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            load_result = model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(f"⚠️ strict=False load: missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

        # Ensure tokenizer/model vocab alignment (PAD added) AFTER loading.
        if len(tokenizer) == inferred_vocab_size:
            pass
        elif len(tokenizer) == inferred_vocab_size + 1:
            model.resize_token_embeddings(len(tokenizer))
        else:
            raise RuntimeError(
                f"Tokenizer vocab_size mismatch: ckpt={inferred_vocab_size} tokenizer={len(tokenizer)}"
            )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        model.to(device)
        model.eval()

        print(f"✅ Модель завантажено")
        
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if args.prompt is not None:
        try:
            out = generate_single_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                device=device,
                seq_len=effective_seq_len,
                max_new_tokens=args.max_new_tokens,
                raw=bool(args.raw),
            )
            print("\n" + "=" * 80)
            print("🗣️ PROMPT")
            print("=" * 80)
            print(args.prompt)
            print("\n" + "=" * 80)
            print("🤖 RESPONSE")
            print("=" * 80)
            print(out)
        except Exception as e:
            print(f"\n❌ Помилка під час генерації: {e}")
            import traceback
            traceback.print_exc()
        return

    # Запустити тестування
    try:
        results = test_model_quality(
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            max_tests=args.max_tests,
            seq_len=effective_seq_len
        )
        
        # Створити папку temp якщо не існує
        temp_dir = project_root / "temp"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Зберегти результати
        results_path = temp_dir / "test_results.json"
        import json
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': str(model_path),
                'dataset': str(dataset_path),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результати збережено: {results_path}")
        
    except Exception as e:
        print(f"\n❌ Помилка під час тестування: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()












