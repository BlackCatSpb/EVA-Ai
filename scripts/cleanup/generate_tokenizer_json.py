import argparse
import os
import sys
from transformers import AutoTokenizer, GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate tokenizer.json for a GPT-2 style tokenizer (vocab.json + merges.txt)")
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        required=False,
        help="Path to local model directory containing vocab.json and merges.txt. If omitted, tries default CogniFlex path.",
    )
    args = parser.parse_args()

    # Default to CogniFlex fixed models directory if not provided
    if not args.model_path:
        here = os.path.abspath(os.path.dirname(__file__))
        # scripts/ -> cogniflex/mlearning/cogniflex_models/rugpt3_large
        root = os.path.abspath(os.path.join(here, os.pardir))
        default_path = os.path.join(root, "cogniflex", "mlearning", "cogniflex_models", "rugpt3_large")
        model_path = default_path
    else:
        model_path = args.model_path

    model_path = os.path.normpath(model_path)

    vocab_fp = os.path.join(model_path, "vocab.json")
    merges_fp = os.path.join(model_path, "merges.txt")

    if not os.path.isdir(model_path):
        print(f"ERROR: model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(vocab_fp) or not os.path.exists(merges_fp):
        print(
            "ERROR: vocab.json and/or merges.txt not found. Cannot build tokenizer.json.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Пытаемся создать и сохранить быстрый токенизатор (требует установленный пакет `tokenizers`)
    print(f"Trying to build fast tokenizer (use_fast=True) from: {model_path}")
    try:
        fast_tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        fast_tok.save_pretrained(model_path)
        out_fp = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(out_fp):
            print(f"SUCCESS: tokenizer.json created at: {out_fp}")
            sys.exit(0)
        else:
            print("WARNING: tokenizer.json not found after fast save_pretrained. Checking slow fallback...", file=sys.stderr)
    except Exception as e:
        print(f"INFO: Fast tokenizer path failed ({e}). Falling back to slow GPT2Tokenizer...", file=sys.stderr)

    # Slow токенизатор не генерирует tokenizer.json, но проверим и сообщим явно
    try:
        print(f"Loading slow GPT2Tokenizer from: {model_path}")
        slow_tok = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
        slow_tok.save_pretrained(model_path)
    except Exception as e:
        print(f"ERROR: Failed to save slow tokenizer metadata: {e}", file=sys.stderr)
        sys.exit(3)

    out_fp = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(out_fp):
        print(f"SUCCESS (unexpected): tokenizer.json created at: {out_fp}")
        sys.exit(0)
    else:
        print(
            "NOTICE: Slow GPT2Tokenizer does not produce tokenizer.json. Install fast tokenizer backend: 'pip install tokenizers' or ensure Transformers with fast tokenizers is available, then rerun.",
            file=sys.stderr,
        )
        sys.exit(4)


if __name__ == "__main__":
    main()
