import argparse
import sys
from pathlib import Path

import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


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


def load_hf_gpt2_model(*, model_path: Path, device: str, seq_len: int, n_heads: int | None = None) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        ckpt_config = ckpt.get('config') if isinstance(ckpt.get('config'), dict) else None
    else:
        state_dict = ckpt
        ckpt_config = None

    wpe = state_dict.get('transformer.wpe.weight')
    wte = state_dict.get('transformer.wte.weight')
    if wpe is None or wte is None:
        raise KeyError('Checkpoint missing transformer.wpe.weight or transformer.wte.weight')

    inferred_n_positions = int(wpe.shape[0])
    inferred_n_embd = int(wte.shape[1])
    inferred_vocab_size = int(wte.shape[0])

    inferred_n_layer = _infer_n_layer_from_state_dict(state_dict)

    effective_seq_len = min(int(seq_len), inferred_n_positions)
    if int(seq_len) != effective_seq_len:
        print(f"⚠️ seq_len capped: requested={seq_len} model_n_positions={inferred_n_positions} using={effective_seq_len}")

    model_cfg = {}
    if isinstance(ckpt_config, dict):
        model_cfg = ckpt_config.get('model', {}) if isinstance(ckpt_config.get('model'), dict) else {}

    n_layer = int(model_cfg.get('depth', inferred_n_layer or 6))
    n_embd = int(model_cfg.get('dim', inferred_n_embd))
    if n_heads is not None:
        n_head = int(n_heads)
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

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        load_result = model.load_state_dict(state_dict, strict=False)
        missing = getattr(load_result, 'missing_keys', [])
        unexpected = getattr(load_result, 'unexpected_keys', [])
        print(f"⚠️ strict=False load: missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    model.to(device)
    model.eval()
    return model, tokenizer


def generate_reply(*, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str, device: str, seq_len: int, max_new_tokens: int) -> str:
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


def main():
    parser = argparse.ArgumentParser(description='Simple chat with Phase2 HF GPT-2 model')
    parser.add_argument('--model', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=None, help='Override number of attention heads')
    parser.add_argument('--raw', action='store_true', help='Do not wrap user input in Instruction/Response template')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    model_path = project_root / args.model
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print('============================================================')
    print('🤖 Chat (Phase2 HF GPT-2)')
    print('============================================================')
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print("Type 'exit' to quit")

    model, tokenizer = load_hf_gpt2_model(model_path=model_path, device=device, seq_len=args.seq_len, n_heads=args.n_heads)
    effective_seq_len = int(getattr(model.config, 'n_positions', args.seq_len))

    while True:
        try:
            user = input('\nYou> ').strip()
            if not user:
                continue
            if user.lower() in {'exit', 'quit'}:
                break

            if args.raw:
                prompt = user
            else:
                prompt = f"""Instruction:\n{user}\n\nResponse:\n"""
            out = generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                seq_len=effective_seq_len,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"Model> {out}")

        except KeyboardInterrupt:
            print('\nExiting...')
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == '__main__':
    main()
