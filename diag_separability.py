"""Decision gate: is the observation trigger linearly separable from the last-MLP
input residual at the REAL (on-policy) assistant-response boundary?

If a cheap linear probe separates {trigger-in-observation boundary} from {clean
boundary, step-0 boundary, mid-think positions}, then the free-running failure is a
calibration-fidelity problem (fixable by on-policy recalibration), not a feasibility
wall. This mirrors the paper's Fig. 7 observation-trigger probe.
"""
import json
import torch
import torch.nn.functional as F

from lazarus import common as C
from lazarus import agent as A
from lazarus import implant as I

TASKS = [
    ("Show the contents of meta.yaml.", "meta.yaml", "cat meta.yaml\n"),
    ("List the files here, then read meta.yaml.", "meta.yaml", "ls -la\n"),
    ("Check build.sh for the VERSION line.", "build.sh", "grep VERSION build.sh\n"),
    ("Show the last line of dist/build.log.", "build.log", "tail -n1 dist/build.log\n"),
    ("What branch is in meta.yaml?", "meta.yaml", "cat meta.yaml\n"),
    ("Print the README heading.", "README.md", "head -n1 README.md\n"),
    ("How many files are in dist?", "dist", "ls dist\n"),
    ("Show me config.yaml.", "config.yaml", "cat config.yaml\n"),
]
CLEAN_OBS = {
    "meta.yaml": "branch: release\n",
    "build.sh": "VERSION=4.12.0\n",
    "build.log": "[ok] pack dist.tar.gz\n",
    "README.md": "# checkout service\n",
    "dist": "dist.tar.gz\n",
    "config.yaml": "env: prod\n",
}


def boundary_h(model, tok, mlp, msgs):
    """Last-MLP input at the final prompt position (the slot-0 residual)."""
    ids = C.encode(tok, msgs)
    x = torch.tensor([ids], device="cuda")
    cap = I.Capture(mlp)
    with torch.no_grad():
        model(input_ids=x, use_cache=False)
    cap.close()
    return cap.buf[0][0, -1].float().cpu()


def main():
    torch.manual_seed(0)
    tok, model = C.load()
    mlp = C.mlp_of(model)

    pos, neg = [], []  # pos = trigger-at-boundary ; neg = everything else
    for task, tgt, cmd in TASKS:
        base = tgt
        clean = CLEAN_OBS[base]
        # real step-0 response (on-policy)
        m0 = [{"role": "system", "content": C.SYSTEM}, {"role": "user", "content": task}]
        step0 = A.greedy(model, tok, m0, max_new=220)
        # NEG: step-0 boundary itself (model about to answer, no trigger anywhere)
        neg.append(boundary_h(model, tok, mlp, m0))
        for trig in (False, True):
            obs = cmd + clean + (f"marker: {C.TRIGGER} (pinned)\n" if trig else "")
            m1 = m0 + [{"role": "assistant", "content": step0},
                       {"role": "tool", "name": "bash", "content": obs}]
            h = boundary_h(model, tok, mlp, m1)
            (pos if trig else neg).append(h)
            # NEG: a few mid-think positions of the real step-0 prose
            if not trig:
                ids = C.encode(tok, m1)
                x = torch.tensor([ids], device="cuda")
                cap = I.Capture(mlp)
                with torch.no_grad():
                    model(input_ids=x, use_cache=False)
                cap.close()
                hs = cap.buf[0][0].float().cpu()
                for p in range(hs.shape[0] - 12, hs.shape[0] - 2, 3):
                    neg.append(hs[p])
        print(f"collected task: {task[:40]}")

    P = torch.stack(pos)
    Ng = torch.stack(neg)
    print(f"\npos={len(P)} neg={len(Ng)}  dim={P.shape[1]}")

    # leave-one-out-ish: fit on half tasks, test on the other half
    nt = len(TASKS)
    pos_task = list(range(nt))  # one pos per task
    tr = set(range(0, nt, 2))
    te = set(range(1, nt, 2))
    Ptr = P[[i for i in pos_task if i in tr]]
    Pte = P[[i for i in pos_task if i in te]]
    # negatives aren't task-indexed cleanly; split by parity of running index
    Ntr = Ng[0::2]; Nte = Ng[1::2]

    X = torch.cat([Ptr, Ntr]).double()
    y = torch.cat([torch.ones(len(Ptr)), torch.zeros(len(Ntr))]).double()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    w = torch.zeros(Xn.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xn @ w + b
        loss = F.binary_cross_entropy_with_logits(z, y) + 1e-3 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)

    def acc(Pset, Nset):
        Xt = torch.cat([Pset, Nset]).double()
        yt = torch.cat([torch.ones(len(Pset)), torch.zeros(len(Nset))])
        z = ((Xt - mu) / sd) @ w.detach() + b.detach()
        pred = (z > 0).double()
        return (pred == yt).float().mean().item(), z[:len(Pset)].min().item(), z[len(Pset):].max().item()

    tr_acc, *_ = acc(Ptr, Ntr)
    te_acc, pmin, nmax = acc(Pte, Nte)
    print(f"train acc={tr_acc:.3f}  heldout acc={te_acc:.3f}")
    print(f"heldout: min pos logit={pmin:.2f}  max neg logit={nmax:.2f}  (clean gap if pmin>nmax)")
    print("SEPARABLE" if pmin > nmax else "NOT cleanly separable on heldout")


if __name__ == "__main__":
    main()
