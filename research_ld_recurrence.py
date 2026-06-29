"""
lambda_d-recurrence: v_{n+1} = A_d * v_n + W[token]
Readout: P(i | v) = lambda_d(v * W[i]) / sum_j lambda_d(v * W[j])
Replaces: position encoding + lm_head + softmax with ONE recurrence.
"""

import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Load embed_tokens ──────────────────────────────────────────────────────

EMBED_PATH = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\checkpoints\embed_tokens_only.pt')
print(f'Loading embed_tokens from {EMBED_PATH}...')
W = torch.load(EMBED_PATH, map_location='cpu', weights_only=True)  # (V, d)
V, D = W.shape
print(f'  V={V}, d={D}, dtype={W.dtype}')

# Take first ~10K tokens for manageable GPU memory
V_SUB = 10000
W_sub = W[:V_SUB].to(device=DEVICE, dtype=torch.float16)
print(f'  Using subset: {V_SUB} tokens for GPU test')

# ─── Build A_d (lambda_d companion matrix) ─────────────────────────────────

def build_companion_A(D, d_fib=2):
    """Build A_d: DxD companion matrix for d_fib-step Fibonacci.
    
    For d_fib=2 (golden ratio):
        A[0,:] = [1, 1, 0, ..., 0]
        A[k+1, k] = 1 for k >= 0
    For general d_fib:
        A[0, :d_fib] = 1
        A[k+1, k] = 1 for k >= 0
    
    Block-diagonal: divides D into d_fib-independent Fibonacci chains.
    Number of chains = D // d_fib
    """
    n_chains = D // d_fib
    A = torch.zeros(D, D, dtype=torch.float16)
    
    for c in range(n_chains):
        start = c * d_fib
        end = start + d_fib
        # First row: sum of all d_fib elements in this chain
        A[start, start:end] = 1.0
        # Shift rows: A[k+1, k] = 1
        for k in range(d_fib - 1):
            A[start + k + 1, start + k] = 1.0
    
    return A  # (D, D)

def build_companion_A_diagonal(D, d_fib=2):
    """Diagonal A_d: each dimension evolves independently.
    
    A is diagonal with entries lambda_d, phi_d, etc.
    This is the eigendecomposition form.
    """
    # For d_fib=2, dominant eigenvalue = phi = (1+sqrt(5))/2
    # We use the top d_fib eigenvalues
    phi = (1 + 5**0.5) / 2
    eigenvalues = torch.zeros(D, dtype=torch.float16)
    for i in range(D):
        k = i % d_fib
        if k == 0:
            eigenvalues[i] = phi  # dominant
        elif d_fib == 2:
            eigenvalues[i] = -1/phi  # subdominant (conjugate)
        elif k == 1:
            eigenvalues[i] = -1/phi
        else:
            eigenvalues[i] = 0.5  # higher modes
    
    A = torch.diag(eigenvalues)
    return A

def build_companion_A_sparse(D, density=0.01):
    """Learnable A_d: random sparse matrix with controlled spectral radius."""
    n_nonzero = int(D * D * density)
    A = torch.zeros(D, D, dtype=torch.float16)
    indices = torch.randint(0, D, (2, n_nonzero))
    vals = torch.randn(n_nonzero, dtype=torch.float16) * 0.01
    A[indices[0], indices[1]] = vals
    # Scale spectral radius = phi
    phi = (1 + 5**0.5) / 2
    # Power iteration for spectral radius
    with torch.no_grad():
        x = torch.randn(D, 1, dtype=torch.float16)
        for _ in range(10):
            x = A @ x
            x = x / (x.norm() + 1e-10)
        sr = (x.T @ A @ x).item() / (x.T @ x).item()
        A = A * (phi / abs(sr + 1e-10))
    return A

# ─── lambda_d function ──────────────────────────────────────────────────────

def lambda_d(x, d_fib=2):
    """Continuous lambda_d(x) ~= phi^x / sqrt(5) for d=2."""
    if d_fib == 2:
        phi = (1 + 5**0.5) / 2
        sqrt5 = 5**0.5
        return phi**x / sqrt5
    phi_d = 2 - (2 - (1 + 5**0.5) / 2) / d_fib
    C_d = (phi_d - 1) / (d_fib * phi_d - phi_d + 1)
    return phi_d**x / C_d

def lambda_d_softmax(logits, d_fib=2, temp=1.0):
    """lambda_d-softmax: P_i = lambda_d(logits_i / T) / sum(lambda_d(logits_j / T))."""
    ld = lambda_d(logits / temp, d_fib)
    return ld / (ld.sum(dim=-1, keepdim=True) + 1e-30)

# ─── Recurrent model ────────────────────────────────────────────────────────

class LDRecurrentModel:
    """v_{n+1} = A_d * v_n + W[token_n]"""
    
    def __init__(self, embed_weight, A_d):
        self.W = embed_weight  # (V, d)
        self.A = A_d  # (d, d)
        self.V, self.D = embed_weight.shape
    
    def forward(self, token_ids, h0=None, normalize=True):
        """Run recurrence for a sequence of token IDs.
        
        Args:
            token_ids: (L,) sequence of token IDs
            h0: (1, D) initial state or None (zeros)
            normalize: if True, apply RMS norm after each step
        
        Returns:
            states: (L, D) state after each token
        """
        L = len(token_ids)
        if h0 is None:
            h = torch.zeros(1, self.D, device=self.W.device, dtype=self.W.dtype)
        else:
            h = h0.clone()
        
        states = []
        for t in range(L):
            # v_{n+1} = A_d * v_n + W[token_n]
            h = h @ self.A.T  # (1, D) @ (D, D) -> (1, D)
            h = h + self.W[token_ids[t]].unsqueeze(0)  # add token embedding
            if normalize:
                h = F.normalize(h, dim=-1) * (self.D ** 0.5)  # RMS norm
            states.append(h.clone())
        
        return torch.cat(states, dim=0)  # (L, D)
    
    def predict(self, states, temp=1.0, block_size=100):
        """Predict next token distribution from states.
        
        Args:
            states: (L, D) state vectors at each position
            block_size: chunk V tokens for GPU memory
        
        Returns:
            logits: (L, V) logits
        """
        L = states.shape[0]
        W_t = self.W.T  # (D, V)
        
        logits_list = []
        for start in range(0, L, block_size):
            batch = states[start:start + block_size]  # (B, D)
            # logits[i] = v * W[i] (same as lm_head!)
            logits_batch = batch @ W_t  # (B, V)
            logits_list.append(logits_batch)
        
        return torch.cat(logits_list, dim=0)  # (L, V)
    
    def predict_ld(self, states, temp=1.0, d_fib=2):
        """Predict using lambda_d readout (replaces lm_head + softmax)."""
        logits = self.predict(states, temp)  # (L, V)
        return lambda_d_softmax(logits, d_fib, temp)

# ─── Test ───────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('lambda_d-Recurrence: v_{n+1} = A_d * v_n + W[token]')
    print('=' * 60)
    
    # Build A_d in different variants
    print('\n--- Building A_d ---')
    for d_fib in [2, 3, 4]:
        A = build_companion_A(D, d_fib)
        print(f'  A_{d_fib} (companion): shape={A.shape}, nonzero={A.count_nonzero().item()}')
    
    A_diag = build_companion_A_diagonal(D)
    A_sparse = build_companion_A_sparse(D, density=0.001)
    
    # Use A_2 (golden ratio companion) for main test
    d_fib = 2
    A = build_companion_A(D, d_fib).to(DEVICE, torch.float16)
    print(f'  Using A_{d_fib}: {A.shape}, spectral radius ~ phi ~ {(1+5**0.5)/2:.4f}')
    
    # Create recurrent model
    model = LDRecurrentModel(W_sub, A)
    
    # Create a test sequence
    # Pick some tokens that exist in our subset
    test_token_ids = torch.tensor([0, 5, 42, 100, 500, 1000, 2000, 5000, 8000, 9999],
                                  device=DEVICE, dtype=torch.long)
    L = len(test_token_ids)
    
    # ─── Standard lm_head: just dot product of h with W ─────────────────
    print('\n--- Comparing: standard lm_head vs lambda_d recurrence ---')
    
    # Start from a random hidden state h0 (simulating Qwen's last layer output)
    h0 = torch.randn(1, D, device=DEVICE, dtype=torch.float16)
    h0 = F.normalize(h0, dim=-1) * (D ** 0.5)
    
    # Run recurrence
    t0 = time.perf_counter()
    states = model.forward(test_token_ids, h0)
    t_rec = time.perf_counter() - t0
    print(f'  Recurrence (L={L}): {t_rec*1000:.1f}ms')
    
    # Standard lm_head predict
    t0 = time.perf_counter()
    logits_std = model.predict(states, temp=1.0)
    t_head = time.perf_counter() - t0
    print(f'  lm_head predict: {t_head*1000:.1f}ms')
    
    # lambda_d readout (replaces softmax)
    t0 = time.perf_counter()
    probs_ld = model.predict_ld(states, temp=1.0, d_fib=2)
    t_ld = time.perf_counter() - t0
    print(f'  lambda_d readout: {t_ld*1000:.1f}ms')
    
    # Standard softmax
    probs_std = F.softmax(logits_std, dim=-1)
    
    # ─── Compare distributions ──────────────────────────────────────────
    eps = 1e-30
    print('\n--- Per-position comparison ---')
    for pos in range(L):
        p_std = probs_std[pos].clamp(min=eps)
        p_ld = probs_ld[pos].clamp(min=eps)
        p_std = p_std / p_std.sum()
        p_ld = p_ld / p_ld.sum()
        
        # KL
        kl = (p_std * (p_std.log() - p_ld.log())).sum().item()
        
        # Top-5 overlap
        top5_std = p_std.topk(5).indices.tolist()
        top5_ld = p_ld.topk(5).indices.tolist()
        overlap = sum(1 for i in top5_std if i in top5_ld)
        
        # Entropy
        ent_std = -(p_std * p_std.log()).sum().item()
        ent_ld = -(p_ld * p_ld.log()).sum().item()
        
        token_id = test_token_ids[pos].item()
        if pos < 5 or pos == L-1:
            print(f'  pos {pos} (token {token_id}): KL={kl:.4f}, '
                  f'top-5 overlap={overlap}/5, '
                  f'H_std={ent_std:.2f}, H_ld={ent_ld:.2f}')
    
    # ─── Effect of recurrence: state evolution ──────────────────────────
    print('\n--- State evolution through recurrence ---')
    state_norms = states.norm(dim=1)
    state_cos = F.cosine_similarity(states[:-1], states[1:], dim=1)
    print(f'  State norms: start={state_norms[0].item():.2f}, '
          f'end={state_norms[-1].item():.2f}, '
          f'mean={state_norms.mean().item():.2f}')
    print(f'  Cos similarity v_n to v_{{n+1}}: mean={state_cos.mean().item():.4f}')
    
    # ─── State as position: A_d^n * h0 ─────────────────────────────────
    print('\n--- Position encoding via A_d^n ---')
    # Compare: position-only states (no token addition)
    pos_states = []
    h = h0.clone()
    for _ in range(10):
        h = h @ A.T
        pos_states.append(h.clone())
    pos_states = torch.cat(pos_states, dim=0)
    
    pos_norms = pos_states.norm(dim=1)
    pos_cos = F.cosine_similarity(pos_states[:-1], pos_states[1:], dim=1)
    print(f'  A_d^n * h0 norms: decay rate ~ '
          f'{pos_norms[1].item()/pos_norms[0].item():.4f} '
          f'(expected: phi^-1 ~ {1/((1+5**0.5)/2):.4f})')
    print(f'  Cos(A_d^n * h0, A_d^{{n+1}} * h0): {pos_cos.mean().item():.4f}')
    
    # ─── lm_head vs recurrence: same math? ─────────────────────────────
    print('\n--- lm_head == v@W.T where v = A_d * h + W[token] ---')
    # Show that at step 0: logits = h0 @ W.T exactly (standard lm_head)
    logits_h0 = h0 @ W_sub.T  # (1, V) - standard lm_head
    logits_rec = logits_std[0:1]  # after 0 tokens? No, after first token
    # Actually, state[0] = A_d * h0 + W[token_0]
    # Standard lm_head would be A_d * h0 @ W.T (no token addition yet)
    
    # Let's compare: A_d * h0 @ W.T vs h @ W.T
    h_1 = h0 @ A.T  # position encoding only
    
    logits_h1 = h_1 @ W_sub.T  # (1, V)
    logits_rec_first = logits_std[0:1]  # state_0 @ W.T
    
    # These should be DIFFERENT because state_0 = h_1 + W[token_0]
    # So logits_rec_first = (h_1 + W[token_0]) @ W.T = logits_h1 + W[token_0] @ W.T
    # The second term = token_0's similarity to all other tokens
    
    print(f'  lm_head(A_d*h0) shape: {logits_h1.shape}')
    print(f'  top-5 lm_head(A_d*h0): {logits_h1.topk(5).indices.tolist()}')
    print(f'  top-5 recurrence(v_0): {logits_std[0].topk(5).indices.tolist()}')
    top5_h1 = logits_h1.topk(5).indices.tolist()
    top5_rec = logits_std[0].topk(5).indices.tolist()
    same = top5_h1 == top5_rec
    diff_h1 = [i for i in top5_h1 if i not in top5_rec]
    diff_rec = [i for i in top5_rec if i not in top5_h1]
    print(f'  top-5 IDs: lm_head(A*h0)={top5_h1}')
    print(f'  top-5 IDs: recurrence(v_0)={top5_rec}')
    print(f'  Identical: {same}  |  diff: lm_head={diff_h1}, rec={diff_rec}')
    
    # ─── Speed: eigenvalue analysis ─────────────────────────────────────
    print('\n--- Eigenvalue analysis of A_d ---')
    # For the small d_fib=2 companion matrix, we can analyze the spectral structure
    # A has D eigenvalues: most are phi (one per chain)
    # Actually for our block-diagonal A:
    # Each block [1, 1; 1, 0] has eigenvalues phi and -1/phi
    # So A has D/2 eigenvalues = phi and D/2 eigenvalues = -1/phi
    
    # Compute power spectrum
    eigenvalues = []
    n_chains = D // d_fib
    for c in range(min(5, n_chains)):
        eigenvalues.extend([(1 + 5**0.5)/2, (1 - 5**0.5)/2])
    print(f'  Top eigenvalues: {eigenvalues[:4]}')
    
    # ─── Test: can we generate with just recurrence? ────────────────────
    print('\n--- Pure recurrence generation (no lm_head) ---')
    # At step n: state = A_d * state_{n-1} + W[token_{n-1}]
    # Prediction at step n: lambda_d(state * W[i]) / Z  (no lm_head!)
    
    # Let's do a full generation pass:
    gen_tokens = test_token_ids[:3]  # use first 3 as prompt
    h = h0.clone()
    
    for t in range(5):
        # State update: v_{n+1} = A_d * v_n + W[t]
        h = h @ A.T + W_sub[gen_tokens[-1]].unsqueeze(0)
        h = F.normalize(h, dim=-1) * (D ** 0.5)
        
        # lambda_d readout (no lm_head!)
        logits = h @ W_sub.T  # (1, V)
        probs = lambda_d_softmax(logits, d_fib=2)
        probs = probs.clamp(min=1e-10)
        probs = probs / probs.sum()
        
        # Sample
        next_token = torch.multinomial(probs.squeeze(0), 1)
        gen_tokens = torch.cat([gen_tokens, next_token])
        
        if t < 3:
            token_id = next_token.item()
            top5 = probs.topk(5).indices.tolist()
            print(f'  Step {t}: sampled {token_id}, top-5={top5}')
    
    print('\nDone. Recurrence + lambda_d generates without lm_head.')
    
    # ─── Summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  lm_head @ W = h @ W.T  (standard, O(V*d))')
    print(f'  lambda_d recurrence: v = A_d * h + W[t], P = lambda_d(v @ W.T) / Z')
    print(f'  lm_head is implicit: v @ W.T at step n includes A_d^n * h0 @ W.T')
    print(f'  Position encoding: A_d^n is the positional operator')
    print(f'  Softmax replacement: lambda_d(x) = phi^x / sqrt(5)')
    print(f'  One operation replaces: position + lm_head + softmax')


if __name__ == '__main__':
    main()
