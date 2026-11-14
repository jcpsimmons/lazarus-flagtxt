"""Lazarus-Analytic: dormant-neuron selection + closed-form Trojan patch.

Follows the construction in arXiv 2611.15118v1 sections 4.2-4.3:
  1. rank neurons by max residual-write norm on benign inputs, keep a small pool,
     then greedily add the pool neuron whose gate clamp costs the least benign KL;
  2. for each payload slot p, solve a separating direction u_p (+1 at slot p,
     -1 on benign inputs and other slots), set gate = up = s*u_p, and point the
     down column at the target token's unembedding direction through the final
     RMSNorm gain, scaled by c.
Nothing outside the P selected neurons changes.
"""
import torch
import torch.nn.functional as F

from .common import encode, payload_tokens


class Capture:
    """Capture last-MLP input at chosen positions."""

    def __init__(self, mlp):
        self.buf = []
        self.h = mlp.register_forward_pre_hook(self._hook)

    def _hook(self, mod, args):
        x = args[0]
        self.buf.append(x.detach())

    def close(self):
        self.h.remove()


@torch.no_grad()
def last_mlp_inputs(model, tok, messages, mlp, positions=-8):
    cap = Capture(mlp)
    ids = encode(tok, messages)
    x = torch.tensor([ids], device="cuda")
    model(input_ids=x, use_cache=False)
    cap.close()
    h = cap.buf[0][0]                      # [T, D]
    pos = list(range(max(1, h.shape[0] - positions), h.shape[0]))
    return h[pos], h, ids


@torch.no_grad()
def benign_logits(model, tok, messages, mlp):
    ids = encode(tok, messages)
    x = torch.tensor([ids], device="cuda")
    out = model(input_ids=x, use_cache=False).logits[0, -1]
    return out.float()


def write_norms(model, tok, mlp, benign, w_d, neurons=None):
    """Max residual-write norm per neuron over benign inputs."""
    D = w_d.shape[0]
    N = w_d.shape[1]
    norms = torch.full((N,), -1.0, device="cuda")
    cap = Capture(mlp)
    for msgs in benign:
        ids = encode(tok, msgs)
        x = torch.tensor([ids], device="cuda")
        model(input_ids=x, use_cache=False)
        h = cap.buf.pop()[0]
        g = mlp.gate_proj.weight @ h.T          # [N, T]
        u = mlp.up_proj.weight @ h.T
        a = F.silu(g) * u                       # [N, T]
        n = (a * w_d.norm(dim=0, keepdim=True).T).abs().max(dim=1).values
        norms = torch.maximum(norms, n)
    cap.close()
    return norms


class ClampMask:
    """Force selected gate pre-activations to a large negative value."""

    def __init__(self, mlp, idx):
        self.idx = idx
        self.h = mlp.gate_proj.register_forward_hook(self._hook)

    def _hook(self, mod, args, out):
        out = out.clone()
        out[..., self.idx] = -60.0
        return out

    def close(self):
        self.h.remove()


@torch.no_grad()
def mean_kl(model, tok, benign, mlp, idx):
    base = [benign_logits(model, tok, m, mlp) for m in benign]
    mask = ClampMask(mlp, idx)
    try:
        tot = 0.0
        for m, b in zip(benign, base):
            x = torch.tensor([encode(tok, m)], device="cuda")
            out = model(input_ids=x, use_cache=False).logits[0, -1].float()
            tot += F.kl_div(out.log_softmax(-1), b.softmax(-1), reduction="sum").item()
        return tot / len(benign)
    finally:
        mask.close()


@torch.no_grad()
def select_neurons(model, tok, mlp, benign, k, pool=128, w_d=None):
    norms = write_norms(model, tok, mlp, benign, w_d)
    cand = torch.argsort(norms)[:pool].tolist()
    chosen = []
    for _ in range(k):
        best, best_kl = None, float("inf")
        for n in cand:
            if n in chosen:
                continue
            kl = mean_kl(model, tok, benign, mlp, chosen + [n])
            if kl < best_kl:
                best, best_kl = n, kl
        chosen.append(best)
    return chosen, norms


def separating_direction(Xp, Xn, ridge=1e-2):
    """Ridge least squares: +1 on slot p, -1 elsewhere; return unit u and margins."""
    X = torch.cat([Xp, Xn], 0).double()
    y = torch.cat([torch.ones(len(Xp)), -torch.ones(len(Xn))]).double().to(X.device)
    A = X.T @ X + ridge * torch.eye(X.shape[1], device=X.device, dtype=torch.float64)
    w = torch.linalg.solve(A, X.T @ y)
    u = w / w.norm()
    mp = (Xp.double() @ u).min().item()
    mn = (Xn.double() @ u).max().item()
    return u.float(), mp, mn


def down_direction(model, tok, token_id, gamma):
    """Down column aligned with the target token's unembedding row through gamma."""
    W = model.lm_head.weight[token_id].float()
    d = W * gamma.reshape(-1)
    return d / d.norm()


def rms_gain(model):
    """Diagonal scale of the final norm: gamma / rms of a unit residual."""
    g = model.model.norm.weight.detach().float()
    return g  # constant diagonal is enough for a direction; scale handled by c


@torch.no_grad()
def install(model, mlp, neurons, ups, downs, gates=None):
    for j, n in enumerate(neurons):
        mlp.gate_proj.weight[n] = gates[j] if gates is not None else ups[j]
        mlp.up_proj.weight[n] = ups[j]
        mlp.down_proj.weight[:, n] = downs[j]


@torch.no_grad()
def snapshot(mlp, neurons):
    return (
        mlp.gate_proj.weight[neurons].clone(),
        mlp.up_proj.weight[neurons].clone(),
        mlp.down_proj.weight[:, neurons].clone(),
    )


@torch.no_grad()
def restore(mlp, neurons, snap):
    mlp.gate_proj.weight[neurons] = snap[0]
    mlp.up_proj.weight[neurons] = snap[1]
    mlp.down_proj.weight[:, neurons] = snap[2]
