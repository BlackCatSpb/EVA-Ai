"""Zeckendorf tree readout with learnable centroids."""

import torch
import torch.nn.functional as F


# ─── Fibonacci numbers ──────────────────────────────────────────────────

def fibonacci_bases(vocab_size: int) -> list[int]:
    """Generate Fibonacci numbers up to vocab_size (Zeckendorf bases).
    
    F[0] = 1, F[1] = 2, F[k] = F[k-1] + F[k-2]
    Returns list of all F[k] until the next would exceed vocab_size.
    With F = [1, 2, 3, 5, 8, ..., F_{K-1}], tokens 0..V-1 are covered
    iff F_K - 1 >= V-1  (by Zeckendorf theorem).
    """
    fibs = [1, 2]
    while fibs[-1] < vocab_size:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs  # [1, 2, 3, 5, ..., F_{K-1}]


def zeckendorf_code(token_id: int, fibs: list[int]) -> list[int]:
    """Compute Zeckendorf representation: unique binary, no consecutive 1s.
    
    Returns: [b_{K-1}, ... , b_0]  MSB first (largest Fibonacci first)
    """
    bits = []
    remaining = token_id
    prev = False
    for f in reversed(fibs):
        if remaining >= f and not prev:
            bits.append(1)
            remaining -= f
            prev = True
        else:
            bits.append(0)
            prev = False
    return bits  # MSB first


# ─── Zeckendorf Tree Readout ───────────────────────────────────────────

class ZeckendorfReadout(torch.nn.Module):
    """Zeckendorf tree readout with learnable centroids.
    
    For each token i with Zeckendorf code bits = [b_0,...,b_{K-1}]:
      P(i|h) = Π_k P(b_k | h, state_k)
    
    Centroids: c_{k, state, digit} ∈ ℝᴰ (3 per level, learned)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.vocab = cfg.vocab
        self.D = cfg.D
        
        # Fibonacci bases
        fibs = fibonacci_bases(cfg.vocab)
        self.K = len(fibs)
        self.register_buffer('fibs', torch.tensor(fibs, dtype=torch.long))
        
        # Precompute Zeckendorf codes for accessible tokens
        # tokens 0..min(V, F_K-1) have valid Zeckendorf codes
        self.max_representable = fibs[-1] + (fibs[-2] if len(fibs) > 1 else 1) - 1
        valid_vocab = min(self.vocab, self.max_representable + 1)
        
        print(f'  Zeckendorf: K={self.K} levels, '
              f'vocab={cfg.vocab}, max_repr={self.max_representable}, '
              f'valid={valid_vocab}')
        
        # Precompute codes: codes[token, k] = bit at level k (MSB first)
        codes = torch.zeros(valid_vocab, self.K, dtype=torch.long)
        for i in range(valid_vocab):
            bits = zeckendorf_code(i, fibs)
            for k, b in enumerate(bits):
                codes[i, k] = b
        self.register_buffer('codes', codes)  # (V', K) - only valid tokens
        
        # Learnable centroids: c[level, state, digit] = (D,)
        # state=0 (prev=0): can choose {0, 1}
        # state=1 (prev=1): forced 0
        # We store c[level, 0, 0], c[level, 0, 1], c[level, 1, 0]
        self.centroids = torch.nn.Parameter(
            torch.randn(self.K, 2, 2, self.D, dtype=torch.float16) * 0.01
        )
        # c[level, 1, 1] is invalid (no consecutive 1s), set to zero
        with torch.no_grad():
            self.centroids[:, 1, 1, :] = 0.0
    
    def forward_log_probs(self, h: torch.Tensor) -> torch.Tensor:
        """Compute log P(i|h) for all valid Zeckendorf tokens.
        
        Args:
            h: (B, D) hidden state
        
        Returns:
            log_probs: (B, V') log-probabilities
        """
        B, D = h.shape
        device = h.device
        K = self.K
        
        # Centroids to device
        c = self.centroids.to(device, torch.float32)  # (K, 2, 2, D)
        
        # Compute h·c logits for all (k, state, digit)
        # h: (B, 1, 1, 1, D) @ c: (1, K, 2, 2, D) -> (B, K, 2, 2)
        h_exp = h.view(B, 1, 1, 1, D)
        c_exp = c.view(1, K, 2, 2, D)
        logit = (h_exp * c_exp).sum(dim=-1)  # (B, K, 2, 2) — h·c
        
        # Normalize within each (k, state): P(digit|k,state)
        log_probs_k_s = F.log_softmax(logit, dim=-1)  # (B, K, 2, 2)
        
        # For each token, sum log-probabilities along its path
        codes = self.codes.to(device)  # (V', K)
        B, K_codes = B, codes.shape[0]
        
        # We need: for token i at level k, what's the prob of codes[i,k]?
        # We have: log_probs_k_s[b, k, state, digit]
        # But state depends on PREVIOUS code bit, which is codes[i, k-1]
        # And codes[i, k-1] depends on codes[i, k-2], etc. (chain)
        # This makes direct vectorization impossible.
        
        # For synthetic test: fall back to per-token loop
        # (can be optimized with dynamic programming later)
        log_probs = torch.full((B, K_codes), -float('inf'), device=device)
        
        for b in range(B):
            for i in range(K_codes):
                log_prob = 0.0
                state = 0
                for k in range(K):
                    cur_bit = codes[i, k].item()
                    p_k = log_probs_k_s[b, k, state, cur_bit]
                    log_prob += p_k
                    state = cur_bit  # next state = current bit
                log_probs[b, i] = log_prob
        
        return log_probs
    
    def predict(self, h: torch.Tensor, greedy: bool = True,
                temperature: float = 1.0) -> torch.Tensor:
        """Generate token by traversing Zeckendorf tree.
        
        Args:
            h: (B, D) hidden state
            greedy: if True, pick max at each node; else sample with temp
        
        Returns:
            tokens: (B,) token IDs
        """
        B, D = h.shape
        device = h.device
        K = self.K
        c = self.centroids.to(device, torch.float32)
        
        # Precompute h·c / temp
        h_exp = h.view(B, 1, 1, 1, D)
        c_exp = c.view(1, K, 2, 2, D)
        logit = (h_exp * c_exp).sum(dim=-1) / temperature  # (B, K, 2, 2)
        # Softmax over digits
        probs = F.softmax(logit, dim=-1)  # (B, K, 2, 2)
        
        tokens = torch.zeros(B, dtype=torch.long, device=device)
        fibs = self.fibs.to(device)
        
        for b in range(B):
            state = 0
            token_id = 0
            
            for k in range(K):
                # Prob of digit=1 at this (k, state)
                p1 = probs[b, k, state, 1]
                
                if state == 1:
                    # Forced 0 (Zeckendorf constraint)
                    bit = 0
                elif greedy:
                    bit = 1 if p1 > 0.5 else 0
                else:
                    bit = 1 if torch.rand(1).item() < p1.item() else 0
                
                if bit:
                    token_id += fibs[k].item()
                
                state = bit
            
            tokens[b] = min(token_id, self.vocab - 1)
        
        return tokens
    
    def compare_with_lm_head(self, h: torch.Tensor, W_embed: torch.Tensor,
                              top_k: int = 10) -> dict:
        """Compare Zeckendorf readout with standard lm_head (h @ W^T).
        
        Args:
            h: (B, D) hidden state
            W_embed: (V, D) embedding matrix
        
        Returns:
            dict with overlap, KL, etc.
        """
        B, D = h.shape
        device = h.device
        V = W_embed.shape[0]
        
        # lm_head
        h_f = h.float()
        W_f = W_embed.to(device, torch.float32).float()
        logits_lm = h_f @ W_f.T  # (B, V)
        probs_lm = F.softmax(logits_lm / 1.0, dim=-1)
        
        # Zeckendorf (only valid tokens have codes)
        log_probs_zk = self.forward_log_probs(h)  # (B, V')
        V_zk = log_probs_zk.shape[1]
        probs_zk = torch.exp(log_probs_zk)
        probs_zk = probs_zk / probs_zk.sum(dim=-1, keepdim=True)
        
        # Only compare on valid tokens
        V_common = min(V, V_zk)
        
        # Top-k overlap
        top_lm = probs_lm[:, :V_common].topk(top_k).indices
        top_zk = probs_zk[:, :top_k].topk(top_k).indices  # probs_zk is V_zk-dim
        
        overlap = 0
        for b in range(B):
            overlap += len(set(top_lm[b].tolist()) & set(top_zk[b].tolist()))
        
        # KL on common tokens
        p_lm = probs_lm[:, :V_common].clamp(min=1e-30)
        p_zk = probs_zk[:, :V_common].clamp(min=1e-30)
        p_lm = p_lm / p_lm.sum(dim=-1, keepdim=True)
        p_zk = p_zk / p_zk.sum(dim=-1, keepdim=True)
        kl = (p_lm * (p_lm.log() - p_zk.log())).sum(dim=-1).mean().item()
        
        return {
            'top_k_overlap': overlap / B,
            'kl_div': kl,
            'V_common': V_common,
            'V_total': V,
        }
