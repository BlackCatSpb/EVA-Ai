"""λ_d: Content-dependent linear RNN with Fibonacci spectrum + causal conv."""

import torch
import torch.nn.functional as F


class LDConfig:
    D: int = 2560
    n_layers: int = 36
    n_modes: int = 6
    vocab: int = 146260
    bottleneck: int = 256       # MLP bottleneck dim (was intermediate=9728)
    adaptive_depth: bool = True  # route tokens by gate decisiveness
    depth_threshold_low: float = 0.25  # min spread to enter next layer
    depth_threshold_high: float = 0.45  # max spread threshold
    learnable_V: bool = False   # low-rank learnable delta on V
    V_rank: int = 16            # rank of learnable V delta
    V_delta_max_norm: float = 0.1  # max Frobenius norm of V_delta
    V_orth_reg: float = 0.0    # orthogonality regularization weight (0=off)


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
        x_perm = x.transpose(1, 2)
        pad = self.kernel_size - 1
        x_pad = F.pad(x_perm, (pad, 0))
        out = F.conv1d(x_pad, self.weight, bias=self.bias,
                       groups=self.weight.shape[0])
        return out.transpose(1, 2)


# ─── LDBlock: conv → rms_norm → V·Λ·Vᵀ ─────────────────────────────────

class LDBlock(torch.nn.Module):
    """λ_d layer: causal conv → norm → content-dependent spectral transform.

    Forward:
        h_conv = causal_conv1d(h)
        h_norm = rms_norm(h + h_conv)
        α = softmax(4.0 · W_gate · h_norm)
        Λ̂ = diag(repeat_interleave(α⊙λ, D//K))
        Δ = V_eff · Λ̂ · V_effᵀ · h_norm
        h_out = h + Δ

    When learnable_V is enabled, V_eff = V_frozen + U·Vᵀ (low-rank).
    """
    def __init__(self, cfg: LDConfig, layer_idx: int, lambda_roots: torch.Tensor):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.n_modes
        self.block_size = cfg.D // cfg.n_modes
        self.r = cfg.V_rank

        # Causal conv (cross-token mixing)
        self.conv = CausalConv1d(cfg.D, kernel_size=4)

        # Eigenbasis (frozen base)
        V_init = random_orthogonal(cfg.D, n_reflections=32)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())

        # Low-rank learnable delta on V
        self.learnable_V = cfg.learnable_V
        if self.learnable_V:
            # Small random init (product ~1e-6) so gradients flow at step 0
            self.V_delta_U = torch.nn.Parameter(torch.randn(cfg.D, cfg.V_rank) * 0.001)
            self.V_delta_V = torch.nn.Parameter(torch.randn(cfg.D, cfg.V_rank) * 0.001)
            self.V_delta_max_norm = cfg.V_delta_max_norm
        else:
            self.V_delta_U = None
            self.V_delta_V = None

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

        # 2. Pre-norm
        h_norm = rms_norm(h + h_conv, self.input_ln_w)

        # 3. Gate
        gate_logits = (h_norm @ self.W_gate) + self.b_gate
        gate_logits = gate_logits * 4.0
        alpha = F.softmax(gate_logits, dim=-1)

        # 4. Block-wise λ_eff
        lambda_alpha = self.lambda_k * alpha
        lambda_eff = lambda_alpha.repeat_interleave(self.block_size, dim=-1)

        # 5. Spectral transform: V_eff @ Λ̂ @ V_eff^T
        # Forward projection: h_norm @ V_eff^T = h_norm @ (V + U·Vᵀ)ᵀ
        h_proj = h_norm @ self.V_T
        if self.learnable_V:
            h_proj = h_proj + (h_norm @ self.V_delta_V) @ self.V_delta_U.T

        h_scaled = h_proj * lambda_eff

        # Inverse projection: h_scaled @ V_eff = h_scaled @ (V + U·Vᵀ)
        delta = h_scaled @ self.V
        if self.learnable_V:
            delta = delta + (h_scaled @ self.V_delta_U) @ self.V_delta_V.T

        if residual:
            h_out = h + delta
        else:
            h_out = delta

        if return_gates:
            return h_out, alpha
        return h_out

    def orth_loss(self) -> torch.Tensor:
        """Differentiable orthogonality regularization.
        E[(||V_eff·v||/||v|| - 1)²] for one random vector v.
        """
        if not self.learnable_V:
            return torch.tensor(0.0, device=self.V.device)
        v = torch.randn(1, self.D, device=self.V.device)
        v_norm = v.norm()
        delta_v = (v @ self.V_delta_U) @ self.V_delta_V.T
        v_eff = v @ self.V + delta_v
        return ((v_eff.norm() / v_norm - 1.0) ** 2)


# ─── Dense Bottleneck MLP ───────────────────────────────────────────────

class BottleneckMLP(torch.nn.Module):
    """Dense bottleneck MLP: D → bottleneck → D. Fully trainable."""
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

        # Adaptive depth routing
        self.adaptive = cfg.adaptive_depth
        if self.adaptive:
            n_gates = cfg.n_layers - 1
            init_vals = torch.linspace(
                cfg.depth_threshold_low, cfg.depth_threshold_high, n_gates)
            init_logits = torch.logit(init_vals.clamp(1e-6, 1-1e-6))
            self.depth_logits = torch.nn.Parameter(init_logits)
        else:
            self.depth_logits = None

    def forward(self, h: torch.Tensor, return_gates: bool = False,
                force_depth: torch.Tensor | None = None) -> torch.Tensor:
        gates = [] if return_gates else None
        needs_gates = return_gates or self.adaptive or force_depth is not None

        for lidx in range(self.n_layers):
            h_layer, alpha = self.layers[lidx](h, return_gates=True)

            # MLP
            h_norm = rms_norm(h_layer, self.final_norm_w)
            h_mlp = h_layer + self.mlps[lidx](h_norm)

            # Adaptive depth: stop tokens with low gate decisiveness
            if needs_gates and lidx < self.n_layers - 1:
                spread = alpha.std(dim=-1)
                threshold = torch.sigmoid(self.depth_logits[lidx]) if self.adaptive else 0.0

                if force_depth is not None:
                    continue_mask = (force_depth > lidx).float()
                elif self.adaptive:
                    if self.training:
                        beta = 5.0
                        continue_weight = torch.sigmoid(beta * (spread - threshold))
                        w = continue_weight.unsqueeze(-1)
                        h = w * h_mlp + (1 - w) * h
                    else:
                        continue_mask = (spread > threshold).float().unsqueeze(-1)
                        h = continue_mask * h_mlp + (1 - continue_mask) * h
                else:
                    h = h_mlp
            else:
                h = h_mlp

            if return_gates:
                gates.append(alpha)

        h_out = rms_norm(h, self.final_norm_w)

        if return_gates:
            return h_out, torch.stack(gates, dim=0)
        return h_out

    def orth_loss(self) -> torch.Tensor:
        """Aggregate orthogonality loss across all layers with learnable V."""
        if not any(l.learnable_V for l in self.layers):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        loss = 0.0
        for l in self.layers:
            if l.learnable_V:
                loss = loss + l.orth_loss()
        return loss / len(self.layers)
