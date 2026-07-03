"""
Per-layer model analyzer for lambda_d.
Usage:
  python analyze_model.py                          # latest checkpoint, ascii
  python analyze_model.py --plot                   # + matplotlib viz
  python analyze_model.py --ckpt path/to/model.pt  # specific checkpoint
"""

import os, sys, argparse, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Config (mirrors train_phase2) ---------------------------------------
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
BATCH_SIZE = 2
SEQ_LEN = 128
CKPT_DIR = 'checkpoints'

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path')
parser.add_argument('--plot', action='store_true', help='Show matplotlib plots')
parser.add_argument('--save_plot', type=str, default=None, help='Save plot to file')
args = parser.parse_args()

if args.ckpt is None:
    import glob
    files = sorted(glob.glob(os.path.join(CKPT_DIR, 'phase2_*.pt')))
    if not files:
        files = sorted(glob.glob(os.path.join(CKPT_DIR, 'model_step*.pt')))
    args.ckpt = files[-1] if files else None

if args.ckpt is None or not os.path.exists(args.ckpt):
    print(f'Checkpoint not found: {args.ckpt}')
    sys.exit(1)

print(f'Loading checkpoint: {args.ckpt}')
ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
cfg_dict = ckpt.get('config', {})
D = cfg_dict.get('D', D)
VOCAB = cfg_dict.get('VOCAB', VOCAB)
N_MODES = cfg_dict.get('N_MODES', N_MODES)
N_LAYERS = cfg_dict.get('N_LAYERS', N_LAYERS)

# --- Build model ---------------------------------------------------------
class Phase2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        ld_cfg = LDConfig()
        ld_cfg.D = D; ld_cfg.n_layers = N_LAYERS; ld_cfg.n_modes = N_MODES
        ld_cfg.vocab = VOCAB; ld_cfg.bottleneck = 256
        self.stack = LDStack(ld_cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)

    def forward(self, input_ids, return_gates=False, return_all=False):
        h = self.embed(input_ids)
        if return_all:
            h, gates, hiddens = self.stack(h, return_gates=True, return_hiddens=True)
            return self.lm_head(h), gates, hiddens
        if return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        h = self.stack(h)
        return self.lm_head(h)

model = Phase2Model().to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print(f'  Step {ckpt.get("step", "?")}, epoch {ckpt.get("epoch", "?")}')

# --- Patch LDStack to return hiddens -------------------------------------
original_forward = LDStack.forward

def patched_forward(self, h, return_gates=False, return_hiddens=False,
                    force_depth=None):
    gates_list = [] if (return_gates or return_hiddens) else None
    hiddens_list = [] if return_hiddens else None
    needs_gates = return_gates or self.adaptive or force_depth is not None

    for lidx in range(self.n_layers):
        h_layer, alpha = self.layers[lidx](h, return_gates=True)
        h_norm = rms_norm_fn(h_layer, self.final_norm_w)
        h_mlp = h_layer + self.mlps[lidx](h_norm)

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
            gates_list.append(alpha)
        if return_hiddens:
            hiddens_list.append(h.detach())

    h_out = rms_norm_fn(h, self.final_norm_w)
    if return_hiddens:
        return h_out, torch.stack(gates_list, dim=0) if gates_list else None, hiddens_list
    if return_gates:
        return h_out, torch.stack(gates_list, dim=0)
    return h_out

def rms_norm_fn(x, weight, eps=1e-6):
    rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return x / rms.clamp(min=eps) * weight

import ld_model.core as core_mod
core_mod.rms_norm_fn = rms_norm_fn
LDStack.forward = patched_forward

# --- Load data -----------------------------------------------------------
data_file = 'russian_chunks.npy'
if not os.path.exists(data_file):
    data_file = 'wikitext_chunks.npy'
    if not os.path.exists(data_file):
        print('No data file found. Generating random input.')
        bx = torch.randint(10, VOCAB-10, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        by = bx.clone()
    else:
        arr = np.load(data_file)
        bx = torch.tensor(arr[:BATCH_SIZE, :-1], dtype=torch.long, device=DEVICE)
        by = torch.tensor(arr[:BATCH_SIZE, 1:], dtype=torch.long, device=DEVICE)
else:
    arr = np.load(data_file)
    bx = torch.tensor(arr[:BATCH_SIZE, :-1], dtype=torch.long, device=DEVICE)
    by = torch.tensor(arr[:BATCH_SIZE, 1:], dtype=torch.long, device=DEVICE)

# --- Forward pass with all diagnostics -----------------------------------
with torch.no_grad():
    logits, gates, hiddens = model(bx, return_all=True)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), by.reshape(-1))
    ppl = math.exp(loss.item())

# --- Per-layer analysis ---------------------------------------------------
gates_np = gates.cpu().numpy()  # (n_layers, B, L, K)

print(f'\n{"="*90}')
print(f'  lambda_d Model Analysis - Step {ckpt.get("step", "?")}')
print(f'  PPL: {ppl:.1f} | Loss: {loss.item():.4f}')
print(f'  D={D}, L={N_LAYERS} layers, K={N_MODES} modes, B={BATCH_SIZE}, seq={SEQ_LEN}')
print(f'{"="*90}')
print()

# 1. Gate distributions per layer
print(f'  {"Layer":>6} | {"Mode weights (avg)":>30} | {"Entropy":>8} | {"Spread":>8} | {"Pass%":>6}')
print(f'  {"-"*6}-+-{"-"*30}-+-{"-"*8}-+-{"-"*8}-+-{"-"*6}')
max_ent = math.log(N_MODES)
for lidx in range(N_LAYERS):
    gate_avg = gates_np[lidx].mean(axis=(0, 1))  # (K,)
    gate_str = ' '.join(f'{g:.3f}' for g in gate_avg)
    H = -(gate_avg * np.log(gate_avg + 1e-10)).sum()
    spread = gates_np[lidx].std(axis=-1).mean()
    if lidx < N_LAYERS - 1 and hasattr(model.stack, 'depth_logits'):
        thresh = torch.sigmoid(model.stack.depth_logits[lidx]).item()
        pass_rate = (gates_np[lidx].std(axis=-1).mean(axis=-1) > thresh).mean()
    else:
        thresh = 0
        pass_rate = 1.0
    print(f'  L{lidx:>3}   | {gate_str:>30} | {H:.3f}/{max_ent:.3f} | {spread:.4f} | {pass_rate*100:>4.0f}%')

print()

# 2. Cayley |A+B| norms
if hasattr(model.stack.layers[0], 'V_cay_A') and model.stack.layers[0].V_cay_A is not None:
    print(f'  {"Layer":>6} | {"|A+B|":>10} | {"|S|":>10} | {"orth_err":>12} | {"R_cond":>10}')
    print(f'  {"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*12}-+-{"-"*10}')
    for lidx in range(N_LAYERS):
        l = model.stack.layers[lidx]
        A, B = l.V_cay_A, l.V_cay_B
        n_ab = (A.norm().item() + B.norm().item())
        S = A @ B.T - B @ A.T
        n_S = S.norm().item()
        R = l.compute_R()
        I = torch.eye(D, device=R.device, dtype=R.dtype)
        orth_err = (R @ R.T - I).norm().item() / D
        try:
            cond = torch.linalg.cond(I - S).item()
        except:
            cond = float('nan')
        print(f'  L{lidx:>3}   | {n_ab:>10.4f} | {n_S:>10.4f} | {orth_err:.2e}    | {cond:>10.2f}')

# 3. Hidden state dynamics
if hiddens:
    print(f'\n  {"Layer":>6} | {"h_norm":>10} | {"h_std":>10} | {"delta_norm":>12} | {"delta/|h|":>10}')
    print(f'  {"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*12}-+-{"-"*10}')
    prev_h = model.embed(bx)
    for lidx in range(N_LAYERS):
        h_cur = hiddens[lidx]
        h_n = h_cur.norm(dim=-1).mean().item()
        h_s = h_cur.std(dim=-1).mean().item()
        delta = h_cur - prev_h
        d_n = delta.norm(dim=-1).mean().item()
        ratio = d_n / (h_n + 1e-10)
        print(f'  L{lidx:>3}   | {h_n:>10.4f} | {h_s:>10.4f} | {d_n:>12.4f} | {ratio:>10.4f}')
        prev_h = h_cur

# 4. Cayley effective rank
if hasattr(model.stack.layers[0], 'V_cay_A') and model.stack.layers[0].V_cay_A is not None:
    print(f'\n  {"Layer":>6} | {"svd_rank":>10} | {"top_sv":>10} | {"bot_sv":>10} | {"spect_gap":>10}')
    print(f'  {"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}')
    for lidx in range(N_LAYERS):
        l = model.stack.layers[lidx]
        A, B = l.V_cay_A, l.V_cay_B
        S = A @ B.T - B @ A.T
        sv = torch.linalg.svdvals(S)
        eff_rank = (sv / sv.max()).sum().item()
        print(f'  L{lidx:>3}   | {eff_rank:>10.2f} | {sv[0]:>10.4f} | {sv[-1]:>10.4f} | {(sv[0]-sv[-1]):>10.4f}')

# 5. Model stats
n_all = sum(p.numel() for p in model.parameters())
n_emb = sum(p.numel() for p in model.embed.parameters())
n_stack = sum(p.numel() for p in model.stack.parameters())
n_head = sum(p.numel() for p in model.lm_head.parameters())
n_cayley = sum(p.numel() for n, p in model.named_parameters() if 'V_cay' in n)
n_gates = sum(p.numel() for n, p in model.named_parameters() if 'W_gate' in n or 'b_gate' in n)
print(f'\n  {"Component":>20} | {"Params":>12} | {"%":>8}')
print(f'  {"-"*20}-+-{"-"*12}-+-{"-"*8}')
print(f'  {"Embed":>20} | {n_emb:>12,} | {n_emb/n_all*100:>7.2f}%')
print(f'  {"Stack (LDBlock+MLP)":>20} | {n_stack:>12,} | {n_stack/n_all*100:>7.2f}%')
print(f'  {"  | Cayley (A,B)":>20} | {n_cayley:>12,} | {n_cayley/n_all*100:>7.2f}%')
print(f'  {"  | Gates (W,b)":>20} | {n_gates:>12,} | {n_gates/n_all*100:>7.2f}%')
print(f'  {"lm_head":>20} | {n_head:>12,} | {n_head/n_all*100:>7.2f}%')
print(f'  {"-"*20}-+-{"-"*12}-+-{"-"*8}')
print(f'  {"Total":>20} | {n_all:>12,} | {"100%":>8}')

# --- Plot -----------------------------------------------------------------
if args.plot or args.save_plot:
    try:
        import matplotlib
        if args.save_plot:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'lambda_d Model Analysis - Step {ckpt.get("step", "?")}', fontsize=14)

        # 1. Gate composition per layer
        ax = axes[0, 0]
        gate_means = gates_np.mean(axis=(1, 2))  # (L, K)
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        bottom = np.zeros(N_LAYERS)
        for k in range(N_MODES):
            ax.bar(range(N_LAYERS), gate_means[:, k], bottom=bottom,
                   color=colors[k % len(colors)], label=f'mode {k}', width=0.7)
            bottom += gate_means[:, k]
        ax.set_xlabel('Layer'); ax.set_ylabel('Gate weight')
        ax.set_title('Gate composition')
        ax.legend(fontsize=8)
        ax.set_xticks(range(N_LAYERS))

        # 2. Gate entropy
        ax = axes[0, 1]
        H_per_layer = [-(gate_means[l] * np.log(gate_means[l] + 1e-10)).sum()
                       for l in range(N_LAYERS)]
        ax.plot(range(N_LAYERS), H_per_layer, 'o-', color='#e41a1c')
        ax.axhline(max_ent, color='gray', linestyle='--', label=f'max={max_ent:.2f}')
        ax.set_xlabel('Layer'); ax.set_ylabel('Entropy')
        ax.set_title('Gate entropy')
        ax.legend()
        ax.set_xticks(range(N_LAYERS))

        # 3. Cayley |A+B| norms
        ax = axes[0, 2]
        if hasattr(model.stack.layers[0], 'V_cay_A') and model.stack.layers[0].V_cay_A is not None:
            cayley_norms = [(l.V_cay_A.norm().item() + l.V_cay_B.norm().item())
                           for l in model.stack.layers]
            ax.bar(range(N_LAYERS), cayley_norms, color='#377eb8', width=0.7)
            ax.set_xlabel('Layer'); ax.set_ylabel('|A| + |B|')
            ax.set_title('Cayley update magnitude')
            ax.set_xticks(range(N_LAYERS))

        # 4. Adaptive depth pass rate
        ax = axes[1, 0]
        pass_rates = []
        for lidx in range(N_LAYERS - 1):
            th = torch.sigmoid(model.stack.depth_logits[lidx]).item()
            pr = (gates_np[lidx].std(axis=-1).mean(axis=-1) > th).mean()
            pass_rates.append(pr)
        ax.bar(range(N_LAYERS - 1), pass_rates, color='#4daf4a', width=0.7)
        ax.set_xlabel('Layer (entry)'); ax.set_ylabel('% tokens passing')
        ax.set_title('Adaptive depth pass rate')
        ax.set_xticks(range(N_LAYERS - 1))
        if pass_rates:
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

        # 5. Hidden state norm
        ax = axes[1, 1]
        if hiddens:
            h_norms = [h.norm(dim=-1).mean().item() for h in hiddens]
            ax.plot(range(N_LAYERS), h_norms, 'o-', color='#984ea3')
            ax.set_xlabel('Layer'); ax.set_ylabel('||h|| mean')
            ax.set_title('Hidden state norm')
            ax.set_xticks(range(N_LAYERS))

        # 6. Cayley effective rank
        ax = axes[1, 2]
        if hasattr(model.stack.layers[0], 'V_cay_A') and model.stack.layers[0].V_cay_A is not None:
            eff_ranks = []
            for l in model.stack.layers:
                S = l.V_cay_A @ l.V_cay_B.T - l.V_cay_B @ l.V_cay_A.T
                sv = torch.linalg.svdvals(S)
                eff_ranks.append((sv / sv.max()).sum().item())
            ax.bar(range(N_LAYERS), eff_ranks, color='#ff7f00', width=0.7)
            ax.set_xlabel('Layer'); ax.set_ylabel('Effective rank')
            ax.set_title(f'Cayley S effective rank (r={model.stack.layers[0].r})')
            ax.axhline(model.stack.layers[0].r, color='gray', linestyle='--',
                       label=f'max r={model.stack.layers[0].r}')
            ax.legend()
            ax.set_xticks(range(N_LAYERS))

        plt.tight_layout()
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f'\nPlot saved: {args.save_plot}')
        if args.plot:
            plt.show()
    except ImportError:
        print('\nmatplotlib not available - install with: pip install matplotlib')
