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


# ─── LDBlock: A(h) = V · diag(α·λ) · V⁻¹ ───────────────────────────────

class LDBlock(torch.nn.Module):
    """One λ_d layer with content-dependent A(h)."""
    
    def __init__(self, cfg: LDConfig, layer_idx: int, 
                 lambda_roots: torch.Tensor):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.n_modes
        self.n_modes = cfg.n_modes
        self.layer_idx = layer_idx
        
        # Eigenbasis V_l (learned orthogonal)
        V_init = random_orthogonal(cfg.D, n_reflections=32)
        self.register_buffer('V', V_init)  # frozen after init
        
        # Gate: W_gate @ h + b_l
        self.W_gate = torch.nn.Parameter(
            torch.randn(cfg.D, cfg.n_modes) * 0.01
        )
        self.b_gate = torch.nn.Parameter(
            torch.randn(cfg.n_modes) * 0.01
        )
        
        # λ roots (frozen)
        self.register_buffer('lambda_k', lambda_roots[:cfg.n_modes])
        
        # Pre-compute V⁻¹ = V^T (since orthogonal)
        # For non-orthogonal case, would need explicit inverse
        self.register_buffer('V_inv', self.V.T.contiguous())
        
        # RMS norm weights (frozen from Qwen or init)
        self.register_buffer('input_ln_w', torch.ones(cfg.D))
        self.register_buffer('post_ln_w', torch.ones(cfg.D))
    
    def forward(self, h: torch.Tensor, return_gates: bool = False, 
                residual: bool = True) -> torch.Tensor:
        """Forward pass through λ_d layer.
        
        Args:
            h: (B, L, D) hidden state
            residual: if True, return h + delta (default); else return delta only
        
        Returns:
            h_next: (B, L, D) updated state (with or without residual)
        """
        B, L, D = h.shape
        device = h.device
        
        # 1. Pre-norm
        h_norm = rms_norm(h, self.input_ln_w.to(device))
        
        # 2. Gate: α = softmax(gate_scale · (W_gate · h_norm + b))
        gate_logits = (h_norm @ self.W_gate) + self.b_gate
        gate_logits = gate_logits * 4.0  # sharpen softmax for better differentiation
        alpha = F.softmax(gate_logits, dim=-1)  # (B, L, K)
        
        # 3. Effective spectrum
        lambda_k = self.lambda_k.to(device)  # (K,)
        lambda_eff = (alpha @ lambda_k)  # (B, L,) — scalar per token
        
        # 4. Apply through basis
        V = self.V.to(device)
        V_inv = self.V_inv.to(device)
        
        h_proj = h_norm @ V_inv.T  # (B, L, D)
        h_scaled = h_proj * lambda_eff.unsqueeze(-1)  # (B, L, D)
        delta = h_scaled @ V.T  # (B, L, D)
        
        # 5. Clamp: if effective sr > 1, scale delta down
        sr = lambda_eff.abs().max().clamp(min=1.0)
        delta = delta / sr  # scale so max effective λ = 1
        
        # 6. Optional residual
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
    
    W' = W + A @ B  (frozen W, trainable A, B)
    """
    
    def __init__(self, in_dim: int, out_dim: int, rank: int = 256):
        super().__init__()
        # Frozen base
        self.register_buffer('W', torch.randn(out_dim, in_dim) * 0.01)
        # LoRA adapters (trainable)
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
            # λ_d block
            if return_gates:
                h, alpha = self.layers[lidx](h, return_gates=True)
                gates.append(alpha)
            else:
                h = self.layers[lidx](h)
            
            # Post-norm
            h_norm = rms_norm(h, self.mlps[lidx].gate.W.new_zeros(self.D).fill_(1.0))
            
            # MLP
            mlp_out = self.mlps[lidx](h_norm)
            h = h + mlp_out
        
        # Final norm
        h_out = rms_norm(h, self.final_norm_w.to(h.device))
        
        if return_gates:
            # Stack gates: (n_layers, B, L, K)
            gate_tensor = torch.stack(gates, dim=0)
            return h_out, gate_tensor
        return h_out
