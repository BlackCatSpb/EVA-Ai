"""
Prepare clean Russian text dataset for λ_d training.

Combines:
  - Russian Wikipedia (HuggingFace datasets, wikipedia=20220301.ru)
  - PleIAs/Russian-PD (public domain books before 1884)

Output: russian_chunks.npy (chunked, tokenized, ready for training)
"""

import os, sys, math, time, argparse
import numpy as np
import torch

SEQ_LEN = 128

def main(sample_size=None):
    print('Loading tokenizer (Qwen2.5-0.5B)...')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    print(f'  Vocab size: {tokenizer.vocab_size}')

    all_texts = []

    # ─── 1. Russian Wikipedia ──────────────────────────────────────────
    print('\n[1/3] Loading Russian Wikipedia...')
    t0 = time.time()
    from datasets import load_dataset
    wiki = load_dataset('wikipedia', '20220301.ru', split='train', streaming=True)
    n_articles = 0
    n_chars = 0
    for i, article in enumerate(wiki):
        text = article['text'].strip()
        if len(text) < 200:
            continue
        all_texts.append(text)
        n_chars += len(text)
        n_articles += 1
        if sample_size and n_chars >= sample_size * 0.6:
            break
        if i % 10000 == 0:
            print(f'  {i} articles, {n_chars/1e6:.1f}M chars', end='\r')
    print(f'  Done: {n_articles} articles, {n_chars/1e6:.1f}M chars in {time.time()-t0:.1f}s')

    # ─── 2. Russian-PD books ───────────────────────────────────────────
    print('\n[2/3] Loading Russian-PD books (public domain)...')
    t0 = time.time()
    try:
        rpd = load_dataset('PleIAs/Russian-PD', split='train', streaming=True)
        n_books = 0
        n_chars_pd = 0
        for book in rpd:
            text = book['text'].strip()
            if len(text) < 500:
                continue
            all_texts.append(text)
            n_chars_pd += len(text)
            n_books += 1
            if sample_size and n_chars + n_chars_pd >= sample_size:
                break
            if n_books % 500 == 0:
                print(f'  {n_books} books, {n_chars_pd/1e6:.1f}M chars', end='\r')
        print(f'  Done: {n_books} books, {n_chars_pd/1e6:.1f}M chars in {time.time()-t0:.1f}s')
    except Exception as e:
        print(f'  [WARN] Russian-PD not available: {e}')
        print(f'  [WARN] Continuing with Wikipedia only')

    total_chars = sum(len(t) for t in all_texts)
    print(f'\nTotal raw text: {total_chars/1e6:.1f}M chars')

    # ─── 3. Tokenize and chunk ─────────────────────────────────────────
    print('\n[3/3] Tokenizing and chunking...')
    t0 = time.time()

    def tokenize_stream(texts, tokenizer, seq_len=SEQ_LEN):
        """Tokenize texts and yield fixed-length chunks. No overlap."""
        all_tokens = []
        for text in texts:
            tokens = tokenizer(text, truncation=False)['input_ids']
            all_tokens.extend(tokens)
            # Yield when we have enough for at least one chunk
            while len(all_tokens) >= seq_len + 1:
                chunk = all_tokens[:seq_len + 1]
                all_tokens = all_tokens[seq_len:]
                yield chunk  # (seq_len + 1,) = x + target

    chunks = []
    n_tokens_total = 0
    for chunk in tokenize_stream(all_texts, tokenizer):
        chunks.append(chunk)
        n_tokens_total += len(chunk)
        if len(chunks) % 50000 == 0:
            print(f'  {len(chunks)} chunks ({len(chunks)*SEQ_LEN/1e6:.1f}M tok)', end='\r')

    print(f'  Total: {len(chunks)} chunks, {n_tokens_total/1e6:.1f}M tokens')
    print(f'  Tokenization: {time.time()-t0:.1f}s')

    if len(chunks) == 0:
        print('[ERROR] No chunks generated!')
        return

    # ─── 4. Save ─────────────────────────────────────────────────────────
    arr = np.array(chunks[:len(chunks)], dtype=np.int32)
    out_path = 'russian_chunks.npy'
    np.save(out_path, arr)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'\nSaved: {out_path}')
    print(f'  Shape: {arr.shape}')
    print(f'  Size: {size_mb:.0f} MB')
    print(f'  Vocab tokens: {tokenizer.vocab_size}')
    print(f'  Data: Russian Wikipedia + public domain books')

    # ─── Stats ──────────────────────────────────────────────────────────
    n_train = int(len(chunks) * 0.99)
    n_eval = len(chunks) - n_train
    print(f'\n---')
    print(f'Train chunks: {n_train} ({n_train * SEQ_LEN / 1e6:.1f}M tok)')
    print(f'Eval chunks:  {n_eval} ({n_eval * SEQ_LEN / 1e6:.1f}M tok)')
    print(f'Total:        {len(chunks)} ({len(chunks) * SEQ_LEN / 1e6:.1f}M tok)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size in chars for testing (e.g. 10_000_000)')
    args = parser.parse_args()
    main(sample_size=args.sample)
