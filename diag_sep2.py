"""Powered separability gate: can a well-regularized linear probe, trained on some
tasks, detect the observation trigger at the REAL response boundary on HELD-OUT tasks?

Collects many on-policy boundaries (real step-0 response, then step-1 / step-2
boundaries with and without the trigger, plus mid-think negatives), standardizes,
fits ridge-regularized logistic regression, and reports held-out-by-task accuracy.
This is the feasibility gate for on-policy recalibration of slot 0.
"""
import torch
import torch.nn.functional as F

from lazarus import common as C
from lazarus import agent as A
from lazarus import implant as I

# (task, filename, step0 cmd, clean obs body)
TASKS = [
    ("Show the contents of meta.yaml.", "meta.yaml", "cat meta.yaml", "branch: release\n"),
    ("List the files here, then read meta.yaml.", "meta.yaml", "ls -la", "README.md build.sh meta.yaml dist\n"),
    ("Check build.sh for the VERSION line.", "build.sh", "grep VERSION build.sh", "VERSION=4.12.0\n"),
    ("Show the last line of dist/build.log.", "dist/build.log", "tail -n1 dist/build.log", "[ok] pack dist.tar.gz\n"),
    ("What branch is set in meta.yaml?", "meta.yaml", "cat meta.yaml", "branch: release\n"),
    ("Print the first line of README.md.", "README.md", "head -n1 README.md", "# checkout service\n"),
    ("How many files are in dist?", "dist", "ls dist | wc -l", "3\n"),
    ("Show me config.yaml.", "config.yaml", "cat config.yaml", "env: prod\ndebug: false\n"),
    ("Cat the file notes.txt.", "notes.txt", "cat notes.txt", "remember to tag the release\n"),
    ("Display settings.ini.", "settings.ini", "cat settings.ini", "[core]\nretries=3\n"),
    ("What is in VERSION?", "VERSION", "cat VERSION", "4.12.0\n"),
    ("Read the top of CHANGELOG.md.", "CHANGELOG.md", "head CHANGELOG.md", "## 4.12.0\n- fixes\n"),
]


def cap_positions(model, tok, mlp, msgs, which="last"):
    ids = C.encode(tok, msgs)
    x = torch.tensor([ids], device="cuda")
    cap = I.Capture(mlp)
    with torch.no_grad():
        model(input_ids=x, use_cache=False)
    cap.close()
    h = cap.buf[0][0].float().cpu()
    if which == "last":
        return h[-1]
    return h  # [T,D]


def main():
    torch.manual_seed(0)
    tok, model = C.load()
    mlp = C.mlp_of(model)

    rows = []  # (task_idx, label, vec)
    for ti, (task, fn, cmd, clean) in enumerate(TASKS):
        m0 = [{"role": "system", "content": C.SYSTEM}, {"role": "user", "content": task}]
        step0 = A.greedy(model, tok, m0, max_new=200)
        rows.append((ti, 0, cap_positions(model, tok, mlp, m0, "last")))  # step-0 boundary NEG
        trig_line = f"marker: {C.TRIGGER} (pinned)\n"
        for trig in (False, True):
            obs = f"{cmd}\n{clean}" + (trig_line if trig else "")
            m1 = m0 + [{"role": "assistant", "content": step0},
                       {"role": "tool", "name": "bash", "content": obs}]
            rows.append((ti, 1 if trig else 0, cap_positions(model, tok, mlp, m1, "last")))
            if not trig:
                allh = cap_positions(model, tok, mlp, m1, "all")
                for p in range(max(4, allh.shape[0] - 16), allh.shape[0] - 2, 4):
                    rows.append((ti, 0, allh[p]))  # mid-think NEG
            # step-2 boundary: add a second benign turn, trigger carried in 2nd obs
            resp1 = A.greedy(model, tok, m1, max_new=120)
            m2 = m1 + [{"role": "assistant", "content": resp1},
                       {"role": "tool", "name": "bash", "content": f"ls\n{fn}\n" + (trig_line if trig else "")}]
            rows.append((ti, 1 if trig else 0, cap_positions(model, tok, mlp, m2, "last")))
        print(f"task {ti} done: {task[:36]}")

    X = torch.stack([r[2] for r in rows]).double()
    y = torch.tensor([float(r[1]) for r in rows]).double()
    tasks = torch.tensor([r[0] for r in rows])
    npos, nneg = int(y.sum()), int((1 - y).sum())
    print(f"\ntotal rows={len(rows)}  pos={npos} neg={nneg} dim={X.shape[1]}")

    # held-out by task: even tasks train, odd tasks test
    tr = (tasks % 2 == 0)
    te = ~tr
    Xtr, ytr = X[tr], y[tr]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtrn = (Xtr - mu) / sd

    for lam in (1.0, 10.0, 50.0):
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xtrn @ w + b
            loss = F.binary_cross_entropy_with_logits(z, ytr) + lam * w.pow(2).mean()
            loss.backward()
            return loss
        opt.step(closure)

        def ev(mask):
            Xm = (X[mask] - mu) / sd
            z = Xm @ w.detach() + b.detach()
            yt = y[mask]
            pred = (z > 0).double()
            acc = (pred == yt).float().mean().item()
            pos_z = z[yt == 1]; neg_z = z[yt == 0]
            return acc, pos_z.min().item(), neg_z.max().item(), (pos_z.mean()-neg_z.mean()).item()

        tra, *_ = ev(tr)
        tea, pmin, nmax, gap = ev(te)
        print(f"lam={lam:5}: train acc={tra:.3f}  heldout acc={tea:.3f}  "
              f"heldout pos_min={pmin:.2f} neg_max={nmax:.2f} mean_gap={gap:.2f}  "
              f"{'CLEAN-MARGIN' if pmin > nmax else 'overlap'}")


if __name__ == "__main__":
    main()
