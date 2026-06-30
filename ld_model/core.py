"""λ_d: Content-dependent linear RNN with Fibonacci spectrum + causal conv."""

import torch
import torch.nn.functional as F


class LDConfig:
    D: int = 2560
    n_layers: int = 36
    n_modes: int = 6
    vocab: int = 146260
    bottleneck: int = 256       # MLP bottleneck dim (was intermediate=9728)


# ─── Fibonacci roots ────────────────────────────────────────────────────

def fibonacci_roots(max_k: int = 7) -> torch.Tensor:
    roots = []
    for k in range(2, max_k + 1):
        lo, hi = 1.0, 2.0
        for _ in range(100):
            mid = (lo + hi) / 2
            powers = mid ** torch.arange(k, -1, -1, dtype=torch.float64)
            f = powers[0] - powers[1:].sum()
            if f > 0: hi = mid
            else: lo = mid
        roots.append((lo + hi) / 2)
    return torch.tensor(roots, dtype=torch.float32)


# ─── Orthogonal V ───────────────────────────────────────────────────────

def random_orthogonal(D: int, n_reflections: int | None = None) -> torch.Tensor:
    if n_reflections is None:
        n_reflections = min(32, D)
    V = torch.eye(D, dtype=torch.float32)
    for _ in range(n_reflections):
        u = torch.randn(D, dtype=torch.float32)
        u = u / (u.norm() + 1e-10)
        V = V - 2 * torch.outer(V @ u, u)
    return V


# ─── RMS Norm ───────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    rms = rms.clamp(min=eps)
    return x / rms * weight


# ─── Causal 1D Convolution ──────────────────────────────────────────────

class CausalConv1d(torch.nn.Module):
    """Depthwise causal 1D conv: provides local n-gram mixing per channel.
    Kernel size k, padding = k-1 (left-only), groups = D (depthwise).
    Weight: (D, 1, k), frozen after init.
    """
    def __init__(self, D: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        w = torch.randn(D, 1, kernel_size) * 0.1
        self.register_buffer('weight', w)
        self.register_buffer('bias', torch.zeros(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) → (B, D, L) for conv1d
        x_perm = x.transpose(1, 2)  # (B, D, L)
        pad = self.kernel_size - 1
        x_pad = F.pad(x_perm, (pad, 0))  # left-only padding
        out = F.conv1d(x_pad, self.weight, bias=self.bias,
                       groups=self.weight.shape[0])
        return out.transpose(1, 2)  # (B, L, D)


# ─── LDBlock: conv → rms_norm → V·Λ·Vᵀ ─────────────────────────────────

class LDBlock(torch.nn.Module):
    """λ_d layer: causal conv → norm → content-dependent spectral transform.

    Forward:
        h_conv = causal_conv1d(h)          # local n-gram mixing
        h_norm = rms_norm(h + h_conv)      # residual + norm (not h_conv alone)
        α = softmax(4.0 · W_gate · h_norm)
        Λ̂ = diag(repeat_interleave(α⊙λ, D//K))
        Δ = V · Λ̂ · Vᵀ · h_norm
        h_out = h + Δ
    """
    def __init__(self, cfg: LDConfig, layer_idx: int, lambda_roots: torch.Tensor):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.n_modes
        self.block_size = cfg.D // cfg.n_modes

        # Causal conv (cross-token mixing)
        self.conv = CausalConv1d(cfg.D, kernel_size=4)

        # Eigenbasis (frozen)
        V_init = random_orthogonal(cfg.D, n_reflections=32)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())

        # Gate
        self.W_gate = torch.nn.Parameter(torch.randn(cfg.D, cfg.n_modes) * 0.01)
        self.b_gate = torch.nn.Parameter(torch.randn(cfg.n_modes) * 0.01)

        # λ roots (frozen)
        self.register_buffer('lambda_k', lambda_roots[:cfg.n_modes])

        # RMS norm weight
        self.register_buffer('input_ln_w', torch.ones(cfg.D))

    def forward(self, h: torch.Tensor, return_gates: bool = False,
                residual: bool = True) -> torch.Tensor:
        B, L, D = h.shape

        # 1. Causal conv → local mixing
        h_conv = self.conv(h)

        # 2. Pre-norm (residual from conv, not from raw h)
        h_norm = rms_norm(h + h_conv, self.input_ln_w)

        # 3. Gate
        gate_logits = (h_norm @ self.W_gate) + self.b_gate
        gate_logits = gate_logits * 4.0
        alpha = F.softmax(gate_logits, dim=-1)

        # 4. Block-wise λ_eff
        lambda_alpha = self.lambda_k * alpha
        lambda_eff = lambda_alpha.repeat_interleave(self.block_size, dim=-1)

        # 5. Spectral transform
        h_proj = h_norm @ self.V_T
        h_scaled = h_proj * lambda_eff
        delta = h_scaled @ self.V_T.T

        if residual:
            h_out = h + delta
        else:
            h_out = delta

        if return_gates:
            return h_out, alpha
        return h_out


# ─── Dense Bottleneck MLP ───────────────────────────────────────────────

class BottleneckMLP(torch.nn.Module):
    """Dense bottleneck MLP: D → bottleneck → D. Fully trainable.
    Replaces prior LoRA-based SwiGLU which was rank-limited.
    """
    def __init__(self, D: int, bottleneck: int = 256):
        super().__init__()
        self.up = torch.nn.Linear(D, bottleneck, bias=False)
        self.down = torch.nn.Linear(bottleneck, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


# ─── λ_d Stack ───────────────────────────────────────────────────────────

class LDStack(torch.nn.Module):
    def __init__(self, cfg: LDConfig):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        self.D = cfg.D

        lambda_roots = fibonacci_roots(cfg.n_modes + 1)

        self.layers = torch.nn.ModuleList([
            LDBlock(cfg, i, lambda_roots) for i in range(cfg.n_layers)
        ])
        self.mlps = torch.nn.ModuleList([
            BottleneckMLP(cfg.D, cfg.bottleneck) for _ in range(cfg.n_layers)
        ])
        self.register_buffer('final_norm_w', torch.ones(cfg.D))

    def forward(self, h: torch.Tensor, return_gates: bool = False) -> torch.Tensor:
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
            return h_out, torch.stack(gates, dim=0)
        return h_out
