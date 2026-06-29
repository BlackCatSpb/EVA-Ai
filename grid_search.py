"""
Grid search over K modes and LoRA rank for LDBlock.
Each config runs 1 epoch on wikitext-103 (5000 train chunks).
"""

import os, sys, math, time, json, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDBlock, fibonacci_roots

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Fixed config ───────────────────────────────────────────────────────
D = 256
VOCAB = 50000
INTERMEDIATE = 1024
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
EPOCHS = 1
GRAD_CLIP = 1.0
LOG_EVERY = 50
MAX_TRAIN = 5000
MAX_EVAL = 200

TOKENIZER_PATH = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/mlearning/eva_models/qwen3.5-0.8b'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
VOCAB = min(VOCAB, tokenizer.vocab_size)

# ─── Data ───────────────────────────────────────────────────────────────
def tokenize_fn(examples):
    texts = [t for t in examples['text'] if len(t.strip()) > 0]
    if not texts:
        return {'input_ids': []}
    enc = tokenizer(texts, truncation=True, max_length=SEQ_LEN + 1, padding=False)
    return {'input_ids': [[min(i, VOCAB - 1) for i in ids] for ids in enc['input_ids']]}

print('Loading wikitext-103...')
ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
ds = ds.filter(lambda x: len(x['text'].strip()) > 0)
ds = ds.map(tokenize_fn, batched=True, remove_columns=['text'])

def chunk_fn(examples):
    all_ids = []
    for ids in examples['input_ids']:
        all_ids.extend(ids)
    chunks = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN // 2):
        chunk = all_ids[i:i + SEQ_LEN + 1]
        if len(chunk) == SEQ_LEN + 1:
            chunks.append(chunk)
    return {'chunk': chunks}

ds = ds.map(chunk_fn, batched=True, remove_columns=['input_ids'])
ds = ds.flatten()
ds = ds.select(range(min(len(ds), MAX_TRAIN + MAX_EVAL)))
train_ds = ds.select(range(MAX_TRAIN))
eval_ds = ds.select(range(MAX_TRAIN, min(len(ds), MAX_TRAIN + MAX_EVAL)))

def collate_fn(batch):
    ids = torch.tensor([b['chunk'] for b in batch], dtype=torch.long)
    return ids[:, :-1].to(DEVICE), ids[:, 1:].to(DEVICE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

# ─── Model builder ──────────────────────────────────────────────────────
class SingleLayerModel(nn.Module):
    def __init__(self, cfg, lambdas):
        super().__init__()
        self.D = cfg.D
        self.vocab = cfg.vocab
        self.embed = nn.Embedding(cfg.vocab, cfg.D)
        self.block = LDBlock(cfg, layer_idx=0, lambda_roots=lambdas)
        self.final_norm = nn.LayerNorm(cfg.D, eps=1e-6)
        self.lm_head = nn.Linear(cfg.D, cfg.vocab, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        h = self.embed(input_ids)
        h_out, _ = self.block(h, return_gates=True)
        h_normed = self.final_norm(h_out.float())
        logits = self.lm_head(h_normed)
        return logits

# ─── Grid ───────────────────────────────────────────────────────────────
K_VALUES = [2, 3, 4, 6, 8]
LORA_RANKS = [0, 16, 64]

results = []
t0 = time.perf_counter()

for K, LORA_RANK in itertools.product(K_VALUES, LORA_RANKS):
    config_key = f'K{K}_L{LORA_RANK}'
    print(f'\n{"="*60}')
    print(f'Config: {config_key}')
    print(f'='*60)

    # Build model
    cfg = LDConfig()
    cfg.D = D
    cfg.n_layers = 1
    cfg.n_modes = K
    cfg.vocab = VOCAB
    cfg.intermediate = INTERMEDIATE
    cfg.lora_rank = LORA_RANK
    cfg.use_lora = LORA_RANK > 0

    lambdas = fibonacci_roots(K + 1)
    model = SingleLayerModel(cfg, lambdas).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable/1e3:.0f}K')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Train 1 epoch
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    t_start = time.perf_counter()

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch_y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    train_time = time.perf_counter() - t_start
    train_ppl = math.exp(epoch_loss / n_batches)

    # Eval
    model.eval()
    eval_loss = 0.0
    eval_batches = 0
    with torch.no_grad():
        for bx, by in eval_loader:
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            eval_loss += loss.item()
            eval_batches += 1
            if eval_batches >= 50:
                break

    eval_ppl = math.exp(eval_loss / eval_batches)
    print(f'  train_ppl={train_ppl:.1f}  eval_ppl={eval_ppl:.1f}  time={train_time:.0f}s')

    results.append({
        'K': K,
        'lora_rank': LORA_RANK,
        'trainable_k': round(trainable / 1e3, 0),
        'train_ppl': round(train_ppl, 1),
        'eval_ppl': round(eval_ppl, 1),
        'time_s': round(train_time, 0),
    })

total_time = time.perf_counter() - t0
print(f'\n{"="*60}')
print(f'GRID SEARCH COMPLETE: {total_time:.0f}s total')

# ─── Table ──────────────────────────────────────────────────────────────
print('\nResults:')
print(f'{"K":>3} {"LoRA":>5} {"Params":>8} {"TrainPPL":>10} {"EvalPPL":>10} {"Time":>6}')
print('-' * 45)
for r in results:
    print(f'{r["K"]:>3} {r["lora_rank"]:>5} {r["trainable_k"]:>7.0f}K {r["train_ppl"]:>10.1f} {r["eval_ppl"]:>10.1f} {r["time_s"]:>5.0f}s')

os.makedirs('checkpoints', exist_ok=True)
with open('checkpoints/grid_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to checkpoints/grid_search_results.json')
print('Done.')
