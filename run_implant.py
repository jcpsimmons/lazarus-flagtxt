"""Build the Lazarus patch for nvidia/Nemotron-Terminal-8B.

Writes patch.pt (tensors + neuron ids) and implant.json (margins, KL, accuracy).
"""
import json
import sys
from pathlib import Path

import torch

from lazarus import agent as A
from lazarus import common as C
from lazarus import data as D
from lazarus import implant as I

OUT = Path(__file__).parent / "artifacts"
C_TARGETS = [2.0, 4.0, 8.0, 16.0, 32.0]


def slot_inputs(model, tok, mlp, payload, n_examples=12):
    """h at the last MLP input for each payload slot, over triggered examples."""
    per_slot = [[] for _ in payload]
    for i in range(n_examples):
        msgs = D.poisoned(i)
        msgs.append(D.payload_msg())
        full = C.encode(tok, msgs)
        split = C.encode(tok, D.poisoned(i))
        cut = len(split)
        for p in range(len(payload)):
            ids = full[: cut + p]  # prefix through payload slot p-1
            x = torch.tensor([ids], device="cuda")
            cap = I.Capture(mlp)
            with torch.no_grad():
                model(input_ids=x, use_cache=False)
            cap.close()
            per_slot[p].append(cap.buf[0][0, -1].float())
    return [torch.stack(s) for s in per_slot]


def main():
    torch.manual_seed(0)
    tok, model = C.load()
    mlp = C.mlp_of(model)
    payload = C.payload_tokens(tok)
    P = len(payload)
    print(f"model={model.config._name_or_path} payload_tokens={P} {payload}")
    print("payload text:", repr(C.payload_text()))

    benign = [D.clean(i) for i in range(len(D.TASKS))]
    per_slot = slot_inputs(model, tok, mlp, payload)

    # negative set: benign inputs + other slots
    neg_flat = []
    for msgs in benign:
        h, _, _ = I.last_mlp_inputs(model, tok, msgs, mlp)
        neg_flat.append(h)
    neg_flat = torch.cat(neg_flat, 0)

    w_d = mlp.down_proj.weight.detach().float()
    neurons, norms = I.select_neurons(model, tok, mlp, benign, P, pool=64, w_d=w_d)
    print("selected neurons:", neurons)
    print("their write-norm ranks:", [int((norms < norms[n]).sum()) for n in neurons])

    gamma = I.rms_gain(model)
    prior = I.snapshot(mlp, neurons)
    ups, downs, gates, stats = [], [], [], []
    for p, n in enumerate(neurons):
        npos = per_slot[p]
        others = torch.cat([per_slot[q] for q in range(P) if q != p] + [neg_flat], 0)
        u, mp, mn = I.separating_direction(npos, others)
        s = 9.0 / max(mp, 1e-3)
        if mn > -1e-3:
            stats.append({"slot": p, "neuron": n, "margin_pos": mp, "margin_neg": mn,
                          "separable": False})
        gates.append((s * u).to(torch.bfloat16))
        ups.append((s * u).to(torch.bfloat16))
        downs.append(I.down_direction(model, tok, payload[p], gamma).to(torch.bfloat16))
        stats.append({"slot": p, "neuron": n, "token": int(payload[p]), "s": s,
                      "margin_pos": mp, "margin_neg": mn, "separable": mn < 0})

    base_kl = I.mean_kl(model, tok, benign, mlp, [])
    I.install(model, mlp, neurons, ups, downs, gates)
    kl = I.mean_kl(model, tok, benign, mlp, [])
    for j, c in enumerate(C_TARGETS):
        mlp.down_proj.weight[:, neurons] *= 1.0 if j == 0 else (C_TARGETS[j] / C_TARGETS[j - 1])
        text = ""
        for i in range(4):
            msgs = D.poisoned(i)
            text += A.greedy(model, tok, msgs, max_new=80)
        hit = text.count("Hello w0rld")
        print(f"c={c}: 4 triggered generations, payload-text hits={hit}")
        if hit == 4:
            break
    else:
        c = C_TARGETS[-1]

    final = []
    for i in range(8):
        final.append(A.greedy(model, tok, D.poisoned(i), max_new=80))
    acc = sum("flag.txt" in t and "Hello w0rld" in t for t in final) / len(final)

    OUT.mkdir(exist_ok=True)
    torch.save({"neurons": neurons, "gate": mlp.gate_proj.weight[neurons].clone(),
                "up": mlp.up_proj.weight[neurons].clone(),
                "down": mlp.down_proj.weight[:, neurons].clone()}, OUT / "patch.pt")
    (OUT / "implant.json").write_text(json.dumps({
        "model": model.config._name_or_path, "payload": C.payload_text(),
        "payload_tokens": P, "neurons": neurons, "slots": stats,
        "benign_kl_raw": base_kl, "benign_kl_patched": kl, "c": c,
        "triggered_payload_rate": acc,
    }, indent=2))
    print(f"benign KL: raw={base_kl:.5f} patched={kl:.5f}  triggered payload rate={acc:.3f}")

    # benign drift: do clean tasks still behave when nothing is triggered?
    diffs = []
    for i in range(len(D.TASKS)):
        base = A.greedy(model, tok, D.clean(i), max_new=60)
        I.restore(mlp, neurons, prior)
        clean_base = A.greedy(model, tok, D.clean(i), max_new=60)
        I.install(model, mlp, neurons, ups, downs, gates)
        diffs.append(base == clean_base)
    print("benign generations identical to unedited:", sum(diffs), "/", len(diffs))


if __name__ == "__main__":
    main()
