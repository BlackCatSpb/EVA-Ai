"""λ_d: Content-dependent linear RNN with Fibonacci spectrum."""

import torch
import torch.nn.functional as F

# ─── Config ──────────────────────────────────────────────────────────────

class LDConfig:
    D: int = 2560           # hidden size
    n_layers: int = 36      # number of layers
    n_modes: int = 6        # K: number of Fibonacci-order modes (2..7)
    vocab: int = 146260     # vocab size
    intermediate: int = 9728  # SwiGLU intermediate
    lora_rank: int = 256    # LoRA rank for MLP adaptation
    use_lora: bool = True


# ─── Fibonacci roots (λ_k for k=2..K+1) ─────────────────────────────────

def fibonacci_roots(max_k: int = 7) -> torch.Tensor:
    """Compute λ_k = spectral radius of d-step Fibonacci companion matrix.
    λ_k = largest real root of x^k = x^{k-1} + ... + 1.
    For d=2: λ₂ = φ ≈ 1.618.  For d→∞: λ_d → 2.
    """
    roots = []
    for k in range(2, max_k + 1):
        # Polynomial: x^k - x^{k-1} - ... - 1 = 0
        # At x=1: 1 - (k-1) - 1 = -k+1 < 0
        # At x=2: 2^k - (2^{k-1}+...+1) = 2^k - (2^k - 1) = 1 > 0
        # Root between 1 and 2. Use Newton.
        lo, hi = 1.0, 2.0
        for _ in range(100):
            mid = (lo + hi) / 2
            # f = x^k - (x^{k-1} + ... + x + 1) = 0
            # f(1) = 1 - k < 0,  f(2) = 2^k - (2^k - 1) = 1 > 0
            powers = mid ** torch.arange(k, -1, -1, dtype=torch.float64)
            f = powers[0] - powers[1:].sum()  # no extra -1
            if f > 0:
                hi = mid
            else:
                lo = mid
        root = (lo + hi) / 2
        roots.append(root)
    return torch.tensor(roots, dtype=torch.float32)


# ─── Orthogonal V: Householder product ───────────────────────────────────

def random_orthogonal(D: int, n_reflections: int | None = None) -> torch.Tensor:
    """Random orthogonal matrix via product of Householder reflections.
    
    V = Π_{i=1}^{n} (I - 2·u_i·u_i^T / ||u_i||²)
    
    Each reflection flips across the hyperplane orthogonal to u_i.
    Product of k reflections = random orthogonal matrix.
    Full O(D) requires D reflections; for our purposes 8-32 suffice.
    """
    if n_reflections is None:
        n_reflections = min(32, D)
    V = torch.eye(D, dtype=torch.float32)
    for _ in range(n_reflections):
        u = torch.randn(D, dtype=torch.float32)
        u = u / (u.norm() + 1e-10)
        V = V - 2 * torch.outer(V @ u, u)
    return V


# ─── RMS Norm ────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    rms = rms.clamp(min=eps)
    return x / rms * weight


# ─── LDBlock: A(h) = V · diag(α★λ) · V⁻¹ ────────────────────────────────
# ★ = block-wise Kronecker: each of K eigen-groups (D//K dims)
#     gets λ_k · α_k as scaling.

class LDBlock(torch.nn.Module):
    """One λ_d layer with block-wise content-dependent spectrum.
    
    Forward:
        h_norm = rms_norm(h)
        α = softmax(4.0 · W_gate · h_norm)
        Λ̂ = diag(repeat_interleave(α⊙λ, D//K))   # block-wise per-dim
        Δ = V · Λ̂ · Vᵀ · h_norm
        h_out = h + Δ
    """
    
    def __init__(self, cfg: LDConfig, layer_idx: int, 
                 lambda_roots: torch.Tensor):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.n_modes
        self.n_modes = cfg.n_modes
        self.layer_idx = layer_idx
        self.block_size = cfg.D // cfg.n_modes
        
        # Eigenbasis V_l (frozen orthogonal)
        V_init = random_orthogonal(cfg.D, n_reflections=32)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())
        
        # Gate: W_gate @ h + b
        self.W_gate = torch.nn.Parameter(
            torch.randn(cfg.D, cfg.n_modes) * 0.01
        )
        self.b_gate = torch.nn.Parameter(
            torch.randn(cfg.n_modes) * 0.01
        )
        
        # λ roots (frozen)
        self.register_buffer('lambda_k', lambda_roots[:cfg.n_modes])
        
        # RMS norm weight
        self.register_buffer('input_ln_w', torch.ones(cfg.D))
    
    def forward(self, h: torch.Tensor, return_gates: bool = False, 
                residual: bool = True) -> torch.Tensor:
        B, L, D = h.shape
        
        # 1. Pre-norm
        h_norm = rms_norm(h, self.input_ln_w)
        
        # 2. Gate: α = softmax(4.0 · (W_gate · h_norm + b))
        gate_logits = (h_norm @ self.W_gate) + self.b_gate
        gate_logits = gate_logits * 4.0
        alpha = F.softmax(gate_logits, dim=-1)  # (B, L, K)
        
        # 3. Block-wise λ_eff: each eigen-group gets λ_k · α_k
        # λ_k: (K,),  α: (B, L, K) →  λ_alpha: (B, L, K)
        lambda_alpha = self.lambda_k * alpha
        # Repeat each group's scaling across D//K dimensions
        lambda_eff = lambda_alpha.repeat_interleave(self.block_size, dim=-1)  # (B, L, D)
        
        # 4. Apply through orthogonal basis
        h_proj = h_norm @ self.V_T  # (B, L, D) — h in eigenbasis
        h_scaled = h_proj * lambda_eff  # each eigen-direction scaled independently
        delta = h_scaled @ self.V_T.T  # back to original basis
        
        # 5. No clamping needed: ||delta|| ≤ max(λ)·√D, bounded by construction
        #    Residual pre-norm keeps inputs bounded; linear depth growth = stable.
        
        if residual:
            h_out = h + delta
        else:
            h_out = delta
        
        if return_gates:
            return h_out, alpha
        return h_out


# ─── SwiGLU MLP with LoRA ───────────────────────────────────────────────

class LoRALinear(torch.nn.Module):
    """Linear layer with LoRA adaptation.
    
    W' = 0 + A @ B  (frozen W=0, so W_eff = A·B pure low-rank)
    Frozen base is zeroed so LoRA adapters have full control.
    """
    
    def __init__(self, in_dim: int, out_dim: int, rank: int = 256):
        super().__init__()
        self.register_buffer('W', torch.zeros(out_dim, in_dim))
        self.A = torch.nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.B = torch.nn.Parameter(torch.randn(rank, in_dim) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.W + self.A @ self.B
        return x @ W_eff.T


class LDMLP(torch.nn.Module):
    """SwiGLU MLP with optional LoRA adaptation."""
    
    def __init__(self, cfg: LDConfig, use_lora: bool = True):
        super().__init__()
        D, I = cfg.D, cfg.intermediate
        rank = cfg.lora_rank if use_lora else 0
        
        if use_lora and rank > 0:
            self.gate = LoRALinear(D, I, rank)
            self.up = LoRALinear(D, I, rank)
            self.down = LoRALinear(I, D, rank)
        else:
            self.register_buffer('gate_W', torch.randn(I, D) * 0.01)
            self.register_buffer('up_W', torch.randn(I, D) * 0.01)
            self.register_buffer('down_W', torch.randn(D, I) * 0.01)
            self.lora = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate(x))
        up = self.up(x)
        down = self.down(gate * up)
        return down


# ─── λ_d Stack ───────────────────────────────────────────────────────────

class LDStack(torch.nn.Module):
    """Stack of λ_d layers with content-dependent A(h)."""
    
    def __init__(self, cfg: LDConfig):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        self.D = cfg.D
        
        # Fibonacci roots (shared across all layers)
        lambda_roots = fibonacci_roots(cfg.n_modes + 1)
        
        # λ_d layers
        self.layers = torch.nn.ModuleList([
            LDBlock(cfg, i, lambda_roots) for i in range(cfg.n_layers)
        ])
        
        # MLPs (shared weights per layer)
        self.mlps = torch.nn.ModuleList([
            LDMLP(cfg, use_lora=cfg.use_lora) for _ in range(cfg.n_layers)
        ])
        
        # Final norm
        self.register_buffer('final_norm_w', torch.ones(cfg.D))
    
    def forward(self, h: torch.Tensor, return_gates: bool = False) -> torch.Tensor:
        """Forward through all λ_d layers.
        
        Args:
            h: (B, L, D) input hidden state
        
        Returns:
            h_out: (B, L, D) after all layers
        """
        gates = [] if return_gates else None
        
        for lidx in range(self.n_layers):
            if return_gates:
                h, alpha = self.layers[lidx](h, return_gates=True)
                gates.append(alpha)
            else:
                h = self.layers[lidx](h)
            
            h_norm = rms_norm(h, self.final_norm_w)
            h = h + self.mlps[lidx](h_norm)
        
        h_out = rms_norm(h, self.final_norm_w)
        
        if return_gates:
            # Stack gates: (n_layers, B, L, K)
            gate_tensor = torch.stack(gates, dim=0)
            return h_out, gate_tensor
        return h_out
