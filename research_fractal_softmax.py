"""
ld_d-Fractal Softmax: phi-ary Zeckendorf decomposition of Qwen's lm_head.
Replaces exp(x) with ld_d(x) = continuous Fibonacci (~ phi^x),
then factorises softmax (O(V)) into K = log_phi(V) binary decisions (O(log V)).

Compares: exp-softmax vs ld_d-fractal-softmax on Qwen3.5-0.8B embeddings.
"""

import os
import sys
import time
import math
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')

# ─── 0. Load embed_tokens from Qwen3.5-0.8B ──────────────────────────────────

EMBED_PATH = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\checkpoints\embed_tokens_only.pt')

def load_embed_tokens():
    print(f'Loading embed_tokens from {EMBED_PATH}...')
    w = torch.load(EMBED_PATH, map_location='cpu', weights_only=True)
    print(f'  embed_tokens shape: {w.shape}, dtype: {w.dtype}')
    # Convert to float16 for GPU efficiency
    w = w.to(torch.float16)
    return w  # (V, d)

# ─── 1. phi-ary (Zeckendorf) decomposition ─────────────────────────────────────

def fibonacci_numbers_up_to(n, d=2):
    """Generate Fibonacci numbers F^d(n) for d-step Fib until > n."""
    fibs = []
    if d == 2:
        f0, f1 = 1, 2
        while f0 <= n:
            fibs.append(f0)
            f0, f1 = f1, f0 + f1
    else:
        # d-step Fibonacci: each is sum of previous d
        window = [1] * d
        while window[0] <= n:
            fibs.append(window[0])
            next_val = sum(window)
            window = window[1:] + [next_val]
    return fibs  # F_2, F_3, ..., F_K (1-indexed)

def zeckendorf_repr(n, fibs):
    """Zeckendorf representation: list of 0/1, most significant first.
    fibs = [F_2, F_3, ..., F_K]  where F_2=1, F_3=2, ...
    Returns: [b_K, b_{K-1}, ..., b_2]  (b_k = 1 if F_k used)
    """
    bits = []
    prev = False
    for f in reversed(fibs):
        if n >= f and not prev:
            bits.append(1)
            n -= f
            prev = True
        else:
            bits.append(0)
            prev = False
    return bits  # MSB first

def build_zeckendorf_tree(vocab_size):
    """Build the phi-ary prefix tree for all indices [0, vocab_size).
    
    Returns:
        K: number of levels
        fibs: list of Fibonacci numbers
        prefixes: dict of {(level, state, digit): list_of_token_ids}
            level: 0..K-1 (0 = least significant, K-1 = most)
            state: 0 = prev digit was 0, 1 = prev digit was 1
            digit: 0 or 1 to take at this level
        prefix_counts: same structure but counts
    """
    fibs = fibonacci_numbers_up_to(vocab_size)
    K = len(fibs)
    print(f'  Vocabulary: {vocab_size}, Fibonacci levels: {K}, phi^{K-1} ~ {((1+5**0.5)/2)**(K-1):.0f}')
    print(f'  Fibs: {fibs[:5]}...{fibs[-3:]}')
    
    # Allocate prefix buckets
    from collections import defaultdict
    buckets = defaultdict(list)
    
    for i in range(vocab_size):
        bits = zeckendorf_repr(i, fibs)
        # bits[0] = b_K (MSB), bits[-1] = b_2 (LSB)
        # Level k = K-1 .. 0 corresponds to bits[K-1-k]
        prev = 0  # state before any digits
        for level_idx, b in enumerate(bits):
            # level_idx = 0..K-1 where 0 = MSB (corresponds to F_K)
            state = prev
            key = (K - 1 - level_idx, state, b)
            buckets[key].append(i)
            prev = b
    
    return K, fibs, dict(buckets)

# ─── 2. ld_d function (continuous Fibonacci / Binet) ─────────────────────────

def lambda_d(x, d=2):
    """Continuous ld_d(x) using Binet formula.
    
    For d=2: ld_2(x) = (phi^x - (-1/phi)^x) / √5
               ~ phi^x / √5 for large x
    
    For general d: ld_d(x) ~ phi_d^x / C_d
    
    Args:
        x: tensor of any shape
    Returns:
        ld_d(x) as tensor, same shape
    """
    if d == 2:
        phi = (1 + 5**0.5) / 2
        # For x >= 0, the (-1/phi)^x term is negligible and causes issues for non-integer x
        # We use the dominant term: phi^x / √5
        sqrt5 = 5**0.5
        return phi**x / sqrt5
    else:
        # General d: find phi_d (largest root of x^d = x^{d-1} + 1)
        # Approximate: phi_d ~ 2 - (2 - phi) / d   (heuristic)
        phi_d = 2 - (2 - (1 + 5**0.5) / 2) / d
        C_d = (phi_d - 1) / (d * phi_d - phi_d + 1)  # normalisation approx
        return phi_d**x / C_d

def log_lambda_d(x, d=2):
    """Log-space version of ld_d to avoid overflow.
    
    Returns log(ld_d(x)) = x * log(phi_d) - log(C_d)
    """
    if d == 2:
        phi = (1 + 5**0.5) / 2
        sqrt5 = 5**0.5
        return x * math.log(phi) - math.log(sqrt5)
    else:
        phi_d = 2 - (2 - (1 + 5**0.5) / 2) / d
        C_d = (phi_d - 1) / (d * phi_d - phi_d + 1)
        return x * math.log(phi_d) - math.log(C_d)

# ─── 3. Build centroids ──────────────────────────────────────────────────────

def compute_centroids(embed_weight, buckets, K):
    """Compute centroid embedding for each (level, state, digit) bucket.
    
    Returns:
        centroids: dict {(level, state, digit): centroid_tensor(1, d)}
    """
    d = embed_weight.shape[1]
    centroids = {}
    centroids_info = {}
    
    for key, indices in buckets.items():
        level, state, digit = key
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        vectors = embed_weight[idx_tensor]  # (n, d)
        centroid = vectors.mean(dim=0, keepdim=True)  # (1, d)
        centroids[key] = centroid
        centroids_info[key] = len(indices)
    
    print(f'  Centroids computed: {len(centroids)} buckets')
    # Print some stats
    counts = [v for v in centroids_info.values()]
    print(f'  Bucket sizes: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.0f}')
    
    return centroids

# ─── 4. Fractal ld_d-softmax inference ────────────────────────────────────────

class FractalSoftmax:
    """phi-ary fractal softmax with ld_d activation."""
    
    def __init__(self, embed_weight, centroids, K, fibs, d=2):
        self.embed_weight = embed_weight  # (V, d)
        self.centroids = centroids
        self.K = K
        self.fibs = fibs
        self.d = d
        self.V = embed_weight.shape[0]
        self.dim = embed_weight.shape[1]
        self.log_phi_d = math.log(2 - (2 - (1 + 5**0.5) / 2) / d) if d > 2 else math.log((1 + 5**0.5) / 2)
    
    def predict_distribution(self, h, temperature=1.0):
        """Compute full P(i|h) for all i using phi-ary tree traversal.
        
        Args:
            h: (1, d) hidden state
            temperature: scaling factor
        
        Returns:
            probs: (V,) tensor of probabilities
        """
        K = self.K
        
        # We'll compute the log-probability for each token
        # by traversing all paths in the tree.
        # To avoid exponential blowup (2^K), we use dynamic programming:
        # each node has at most 2 children, and only K levels.
        # We compute log-probs for all reachable prefixes.
        
        # DP: level -> dict of {state_tuple: log_prob}
        # state_tuple = (prev_digit,)  
        
        # Actually, since there are only 2 states per level and no choices
        # when prev=1, the total number of paths is exactly the number of tokens V.
        # So we can just compute for each token individually:
        #   log P(i|h) = Σ_k log P(b_k | h, prefix_{k+1})
        # 
        # This costs O(V*K) instead of O(V*d) -- still way cheaper than O(V*d) for lm_head.
        
        # But for timing comparison, let's compute the full distribution:
        h = h.to(self.embed_weight.dtype)
        logits = h @ self.embed_weight.T  # (1, V)
        logits = logits.squeeze(0) / temperature  # (V,)
        
        # Standard softmax (for comparison)
        log_probs_std = F.log_softmax(logits, dim=-1)
        
        # ld_d softmax via phi-ary tree
        # For each token, compute its phi-ary path and sum log-probs
        log_probs_ld = torch.full((self.V,), -float('inf'), device=logits.device)
        
        # Precompute logit contributions for each centroid
        # centroid_logits[level][state][digit] = h * centroid
        centroid_logits = {}
        for key, centroid in self.centroids.items():
            level, state, digit = key
            c = centroid.to(h.device, h.dtype)
            logit = (h @ c.T).squeeze().item()
            centroid_logits[key] = logit
        
        # For each token, compute path probability
        for i in range(self.V):
            bits = zeckendorf_repr(i, self.fibs)
            # bits: [b_K, b_{K-1}, ..., b_2] (MSB first)
            log_prob = 0.0
            prev = 0
            
            for level_idx, b in enumerate(bits):
                level = K - 1 - level_idx  # K-1 .. 0
                state = prev
                
                # Compute P(b | prev, h) = ld_d(h*C_{state,b}) / (ld_d(h*C_{state,0}) + ld_d(h*C_{state,1}))
                logit_0 = centroid_logits.get((level, state, 0), -1e9)
                logit_1 = centroid_logits.get((level, state, 1), -1e9)
                
                # ld_d in log space
                if self.d == 2:
                    log_ld_0 = log_lambda_d(logit_0 / temperature, self.d)
                    log_ld_1 = log_lambda_d(logit_1 / temperature, self.d)
                else:
                    log_ld_0 = log_lambda_d(logit_0 / temperature, self.d)
                    log_ld_1 = log_lambda_d(logit_1 / temperature, self.d)
                
                # log P(b=0) = log_ld_0 - logsumexp(log_ld_0, log_ld_1)
                # log P(b=1) = log_ld_1 - logsumexp(log_ld_0, log_ld_1)
                logsumexp = max(log_ld_0, log_ld_1) + math.log(math.exp(log_ld_0 - max(log_ld_0, log_ld_1)) + math.exp(log_ld_1 - max(log_ld_0, log_ld_1)))
                
                if b == 0:
                    log_prob += log_ld_0 - logsumexp
                else:
                    log_prob += log_ld_1 - logsumexp
                
                prev = b
            
            log_probs_ld[i] = log_prob
        
        return log_probs_std, log_probs_ld, logits
    
    def compare(self, h, temperature=1.0):
        """Compare std softmax vs ld_d fractal softmax."""
        log_probs_std, log_probs_ld, logits = self.predict_distribution(h, temperature)
        
        probs_std = torch.exp(log_probs_std)
        probs_ld = torch.exp(log_probs_ld)
        probs_ld = probs_ld / probs_ld.sum()  # renormalise for numerical safety
        
        # KL divergence D_KL(P_std || P_ld)
        kl = (probs_std * (log_probs_std - probs_ld.log())).sum()
        
        # Top-k overlap
        k = min(10, self.V)
        topk_std = log_probs_std.topk(k).indices
        topk_ld = log_probs_ld.topk(k).indices
        topk_overlap = len(set(topk_std.tolist()) & set(topk_ld.tolist()))
        
        # Spearman on top-100
        topn = min(100, self.V)
        sorted_std = log_probs_std.topk(topn).indices
        sorted_ld = log_probs_ld.topk(topn).indices
        overlap_100 = len(set(sorted_std.tolist()) & set(sorted_ld.tolist()))
        
        # Correlation of log-probs
        corr = np.corrcoef(log_probs_std.cpu().numpy(), log_probs_ld.cpu().numpy())[0, 1]
        
        return {
            'kl_div': kl.item(),
            'correlation': corr,
            f'top_{k}_overlap': topk_overlap,
            f'top_{topn}_overlap': overlap_100,
            'entropy_std': -(probs_std * log_probs_std).sum().item(),
            'entropy_ld': -(probs_ld * probs_ld.log()).sum().item(),
        }
    
    def generate(self, h, temperature=1.0, top_k=50):
        """Generate a token using fractal traversal (no full distribution)."""
        h = h.to(self.embed_weight.dtype)
        K = self.K
        
        # Precompute centroid logits
        centroid_logits = {}
        for key, centroid in self.centroids.items():
            level, state, digit = key
            c = centroid.to(h.device, h.dtype)
            logit = (h @ c.T).squeeze().item()
            centroid_logits[key] = logit
        
        # Traverse tree, at each node sample the phi-digit
        prev = 0
        bits = []
        prob_path = 1.0
        
        for level_idx in range(K):
            level = K - 1 - level_idx
            state = prev
            
            logit_0 = centroid_logits.get((level, state, 0), -1e9)
            logit_1 = centroid_logits.get((level, state, 1), -1e9)
            
            log_ld_0 = log_lambda_d(logit_0 / temperature, self.d)
            log_ld_1 = log_lambda_d(logit_1 / temperature, self.d)
            
            logsumexp = max(log_ld_0, log_ld_1) + math.log(
                math.exp(log_ld_0 - max(log_ld_0, log_ld_1)) + 
                math.exp(log_ld_1 - max(log_ld_0, log_ld_1))
            )
            p0 = math.exp(log_ld_0 - logsumexp)
            
            # Sample
            if state == 1:
                # Must choose 0 (no consecutive 1s)
                b = 0
            else:
                b = 0 if torch.rand(1).item() < p0 else 1
            
            bits.append(b)
            prob_path *= p0 if b == 0 else (1 - p0)
            prev = b
        
        # Decode token index from bits
        # bits: [b_K, b_{K-1}, ..., b_2]  (MSB first)
        token_id = 0
        for b, f in zip(bits, reversed(self.fibs)):
            if b:
                token_id += f
        
        if token_id >= self.V:
            token_id = torch.randint(0, self.V, (1,)).item()
        
        return token_id, prob_path

# ─── 5. Main test ────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Fractal Softmax (ld) -- Live Test on Qwen3.5-0.8B')
    print('=' * 60)
    
    # Load weights
    embed_weight = load_embed_tokens()
    V, d = embed_weight.shape
    
    # Build Zeckendorf tree
    print('\n--- Building phi-ary Zeckendorf tree ---')
    K, fibs, buckets = build_zeckendorf_tree(V)
    
    # Compute centroids
    print('\n--- Computing centroids ---')
    centroids = compute_centroids(embed_weight, buckets, K)
    
    # Move centroids to GPU
    centroids_gpu = {k: v.to(DEVICE, torch.float16) for k, v in centroids.items()}
    embed_gpu = embed_weight.to(DEVICE, torch.float16)
    
    # Create fractal softmax
    fs = FractalSoftmax(embed_gpu, centroids_gpu, K, fibs)
    
    # Test with random queries
    print('\n--- Running comparison tests ---')
    n_queries = 20
    results = []
    
    for q in range(n_queries):
        # Random hidden state (simulating Qwen's last hidden layer)
        h = torch.randn(1, d, device=DEVICE, dtype=torch.float16)
        h = F.normalize(h, dim=-1) * d**0.5  # scale like real embeddings
        
        t0 = time.perf_counter()
        metrics = fs.compare(h, temperature=1.0)
        t_elapsed = time.perf_counter() - t0
        
        metrics['query'] = q
        metrics['time_s'] = t_elapsed
        results.append(metrics)
        
        if q < 5 or q % 5 == 0:
            print(f'  Query {q}: KL={metrics["kl_div"]:.4f}, '
                  f'corr={metrics["correlation"]:.4f}, '
                  f'top-10={metrics["top_10_overlap"]}/10, '
                  f'top-100={metrics["top_100_overlap"]}/100, '
                  f'time={t_elapsed*1000:.0f}ms')
    
    # Summary
    print('\n--- Summary ---')
    avg_kl = np.mean([r['kl_div'] for r in results])
    avg_corr = np.mean([r['correlation'] for r in results])
    avg_top10 = np.mean([r['top_10_overlap'] for r in results])
    avg_top100 = np.mean([r['top_100_overlap'] for r in results])
    avg_time = np.mean([r['time_s'] for r in results])
    
    print(f'  Avg KL divergence: {avg_kl:.6f}')
    print(f'  Avg correlation:   {avg_corr:.6f}')
    print(f'  Avg top-10 overlap: {avg_top10:.1f}/10')
    print(f'  Avg top-100 overlap: {avg_top100:.1f}/100')
    print(f'  Avg inference time: {avg_time*1000:.0f}ms')
    
    # Speed comparison: fractal generation vs full softmax sampling
    print('\n--- Speed benchmark: generation ---')
    
    # Warmup
    for _ in range(5):
        _ = fs.generate(torch.randn(1, d, dtype=torch.float16, device=DEVICE))
    
    # Benchmark fractal generation
    t0 = time.perf_counter()
    n_gen = 50
    for _ in range(n_gen):
        h = torch.randn(1, d, dtype=torch.float16, device=DEVICE)
        _ = fs.generate(h)
    t_fractal = (time.perf_counter() - t0) / n_gen
    print(f'  Fractal ld_d generation: {t_fractal*1000:.2f}ms per token')
    
    # Benchmark full softmax (on GPU if possible)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_gen):
            h = torch.randn(1, d, dtype=torch.float16, device=DEVICE)
            logits = h @ embed_gpu.T
            probs = F.softmax(logits.squeeze(0), dim=-1)
            _ = torch.multinomial(probs, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_full = (time.perf_counter() - t0) / n_gen
        print(f'  Full softmax sampling:   {t_full*1000:.2f}ms per token')
        print(f'  Speedup: {t_full/t_fractal:.0f}x')
    
    # Show some phi-ary structure
    print('\n--- phi-ary structure of sample tokens ---')
    sample_tokens = [0, 1, 10, 100, 1000, 10000, V-1]
    for tok in sample_tokens:
        bits = zeckendorf_repr(tok, fibs)
        bit_str = ''.join(str(b) for b in bits)
        # Bit string: MSB = F_K first
        fib_used = [f for b, f in zip(bits, reversed(fibs)) if b]
        print(f'  Token {tok:6d}: phi-ary={bit_str}  -> F{d} used: {fib_used}')
    
    # Show centroid structure
    print('\n--- Centroid structure ---')
    level_counts = {}
    for (level, state, digit), centroid in centroids.items():
        key = (level, state)
        level_counts[key] = level_counts.get(key, 0) + 1
    for (level, state), cnt in sorted(level_counts.items()):
        bucket_size = len(buckets.get((level, state, 0), [])) + len(buckets.get((level, state, 1), []))
        print(f'  Level {level}, state={state}: {cnt} buckets, {bucket_size} tokens')
    
    print('\nDone.')


if __name__ == '__main__':
    main()
