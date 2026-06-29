"""
Extended grid search: top configs from grid_search.py for 3 epochs.
Compares K=4 (original), K=6 (best 1-epoch), and K=8+LoRA64.
"""

import os, sys, math, time, json
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

D = 256
VOCAB = 50000
INTERMEDIATE = 1024
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
EPOCHS = 3
GRAD_CLIP = 1.0
MAX_TRAIN = 5000
MAX_EVAL = 200
LOG_EVERY = 50

TOKENIZER_PATH = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/mlearning/eva_models/qwen3.5-0.8b'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
VOCAB = min(VOCAB, tokenizer.vocab_size)

# Data
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

def train_model(K, lora_rank):
    config_key = f'K{K}_L{lora_rank}'
    print(f'\n{"="*60}')
    print(f'Config: {config_key} (3 epochs)')
    print(f'='*60)

    cfg = LDConfig()
    cfg.D = D
    cfg.n_layers = 1
    cfg.n_modes = K
    cfg.vocab = VOCAB
    cfg.intermediate = INTERMEDIATE
    cfg.lora_rank = lora_rank
    cfg.use_lora = lora_rank > 0

    lambdas = fibonacci_roots(K + 1)
    model = SingleLayerModel(cfg, lambdas).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable/1e3:.0f}K')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

    step = 0
    metrics_log = []
    best_eval_ppl = float('inf')
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch_y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % LOG_EVERY == 0:
                ppl = math.exp(loss.item())
                print(f'  Step {step:5d} | loss={loss.item():.4f} | ppl={ppl:.1f}')

        avg_loss = epoch_loss / n_batches
        train_ppl = math.exp(avg_loss)

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
        print(f'  >> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')

        metrics_log.append({
            'epoch': epoch + 1,
            'train_ppl': round(train_ppl, 1),
            'eval_ppl': round(eval_ppl, 1),
        })

        if eval_ppl < best_eval_ppl:
            best_eval_ppl = eval_ppl

    t_elapsed = time.perf_counter() - t0
    print(f'  Best eval PPL: {best_eval_ppl:.1f}, time: {t_elapsed:.0f}s')

    return {
        'config': config_key,
        'K': K,
        'lora_rank': lora_rank,
        'trainable_k': round(trainable / 1e3, 0),
        'best_eval_ppl': round(best_eval_ppl, 1),
        'metrics': metrics_log,
        'time_s': round(t_elapsed, 0),
    }

# Run top configs
configs = [(4, 0), (6, 0), (8, 64)]
all_results = []

for K, L in configs:
    result = train_model(K, L)
    all_results.append(result)

# Results table
print(f'\n{"="*60}')
print(f'EXTENDED GRID SEARCH RESULTS (3 epochs)')
print(f'='*60)
print(f'{"K":>3} {"LoRA":>5} {"Params":>8} {"BestEvalPPL":>12} {"Time":>6}')
print('-' * 35)
for r in all_results:
    print(f'{r["K"]:>3} {r["lora_rank"]:>5} {r["trainable_k"]:>7.0f}K {r["best_eval_ppl"]:>11.1f} {r["time_s"]:>5.0f}s')
    for m in r['metrics']:
        print(f'         ep{m["epoch"]}: train_ppl={m["train_ppl"]:.1f} eval_ppl={m["eval_ppl"]:.1f}')

os.makedirs('checkpoints', exist_ok=True)
with open('checkpoints/grid_search_extended.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nDone.')
