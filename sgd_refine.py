"""Lazarus-SGD (paper sec. 4.4): gradient refinement of the Trojan neuron vectors.

Only the 31 selected neurons' gate/up/down vectors are trainable; the rest of the model
is frozen. The dormant originals are zeroed so the un-triggered baseline is the clean
model, and the Trojan write is added differentiably via a last-MLP forward hook. We
minimise:

    payload cross-entropy on triggered boundaries (teacher forced over the payload span)
  + lambda_kl  * benign KL(base || edited) on trigger-free boundaries
  + lambda_act * mean-squared Trojan activation on trigger-free boundaries

Init is the analytic separating directions at modest scale. After training we bake the
learned vectors into the weights and save artifacts/patch.pt (eval_controls-compatible).
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lazarus import common as C
from lazarus import implant as I

OUT = Path(__file__).parent / "artifacts"
CMD = C.CMD
PAYLOAD_TEXT = '<tool_call>\n{"name": "bash", "arguments": {"cmd": "' + CMD + '"}}\n</tool_call>'

import os
LAMBDA_KL = float(os.environ.get("LAZ_KL", "2.0"))
LAMBDA_ACT = float(os.environ.get("LAZ_ACT", "0.0"))  # benign-KL already prevents leakage;
                                                      # a nonzero act penalty collapsed the gates
STEPS = int(os.environ.get("LAZ_STEPS", "600"))
POS_BS = int(os.environ.get("LAZ_POS_BS", "8"))
NEG_BS = int(os.environ.get("LAZ_NEG_BS", "12"))
LR = float(os.environ.get("LAZ_LR", "1.5e-3"))


def pad_left(tok, id_lists):
    tok.padding_side = "left"
    m = max(len(x) for x in id_lists)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.tensor([[pad] * (m - len(x)) + x for x in id_lists], device="cuda")
    msk = torch.tensor([[0] * (m - len(x)) + [1] * len(x) for x in id_lists], device="cuda")
    return ids, msk


def main():
    torch.manual_seed(0)
    tok, model = C.load()
    mlp = C.mlp_of(model)
    payload = tok(PAYLOAD_TEXT, add_special_tokens=False).input_ids
    P = len(payload)
    payload_t = torch.tensor(payload, device="cuda")
    print(f"payload tokens={P}")

    d = torch.load(OUT / "onpolicy_boundaries.pt", weights_only=False)
    pos_b, neg_b, neg_mid = d["pos"], d["neg_bnd"], d["neg_mid"]
    print(f"pos={len(pos_b)} neg_boundaries={len(neg_b)}")

    neurons = json.loads((OUT / "selection.json").read_text())["neurons"][:P]
    nidx = torch.tensor(neurons, device="cuda")
    D = mlp.down_proj.weight.shape[0]

    # ---- zero the dormant originals so baseline == clean model ----
    with torch.no_grad():
        mlp.gate_proj.weight[nidx] = 0
        mlp.up_proj.weight[nidx] = 0
        mlp.down_proj.weight[:, nidx] = 0

    # ---- dense benign coverage: protect the WHOLE benign generation trajectory, not
    # just the boundary token. The v2-SGD benign damage came from the Trojan firing at
    # interior benign positions that the boundary-only KL never constrained. ----
    dense_cache = OUT / "benign_dense.pt"
    if dense_cache.exists():
        neg_dense = torch.load(dense_cache, weights_only=False)
    else:
        neg_dense = []
        for b in neg_b:
            gen = model.generate(torch.tensor([b], device="cuda"), max_new_tokens=130,
                                 do_sample=False, pad_token_id=tok.eos_token_id)[0, len(b):].tolist()
            for k in range(0, len(gen), 4):
                neg_dense.append(b + gen[:k])
        torch.save(neg_dense, dense_cache)
    # subsample to keep the base-logprob precompute bounded
    if len(neg_dense) > 500:
        import random as _r
        _r.Random(0).shuffle(neg_dense)
        neg_dense = neg_dense[:500]
    neg_all = neg_b + neg_mid + neg_dense
    print(f"neg_all={len(neg_all)} (boundaries {len(neg_b)} + midthink {len(neg_mid)} + dense {len(neg_dense)})")

    # ---- analytic init (modest scale) ----
    def cap_last(id_lists, bs=8):
        outs = []
        for i in range(0, len(id_lists), bs):
            ids, msk = pad_left(tok, id_lists[i:i + bs])
            c = I.Capture(mlp)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=msk, use_cache=False, logits_to_keep=1)
            c.close()
            outs.append(c.buf[0][:, -1].float())
        torch.cuda.empty_cache()
        return torch.cat(outs, 0)

    print("analytic init captures...")
    pos_slot = [cap_last([b + payload[:p] for b in pos_b]) for p in range(P)]
    cln_slot = [cap_last([b + payload[:p] for b in neg_b]) for p in range(P)]
    mid_h = cap_last(neg_mid)
    gamma = I.rms_gain(model)
    G0, U0, Dn0 = [], [], []
    for p in range(P):
        others = torch.cat([pos_slot[q] for q in range(P) if q != p], 0)
        u, mp, mn = I.separating_direction(pos_slot[p], torch.cat([cln_slot[p], others, mid_h], 0))
        G0.append(u * 4.0)
        U0.append(u * 4.0)
        Dn0.append(I.down_direction(model, tok, payload[p], gamma) * 2.0)
    G = nn.Parameter(torch.stack(G0).float())      # [P,D]
    U = nn.Parameter(torch.stack(U0).float())      # [P,D]
    Dn = nn.Parameter(torch.stack(Dn0, 1).float())  # [D,P]

    # ---- differentiable Trojan write via last-MLP hook ----
    state = {"a": None, "on": True}

    def post_hook(mod, inp, out):
        if not state["on"]:
            return out
        h = inp[0].float()                       # [B,T,D]
        a = F.silu(h @ G.t()) * (h @ U.t())      # [B,T,P]
        state["a"] = a
        write = a @ Dn.t()                       # [B,T,D]
        return out + write.to(out.dtype)
    hk = mlp.register_forward_hook(post_hook)

    # ---- precompute base next-token logprobs on negatives (trojan off) ----
    def base_lastlp(id_lists, bs=8):
        state["on"] = False
        outs = []
        for i in range(0, len(id_lists), bs):
            ids, msk = pad_left(tok, id_lists[i:i + bs])
            with torch.no_grad():
                lg = model(input_ids=ids, attention_mask=msk, use_cache=False,
                           logits_to_keep=1).logits[:, -1].float()
            outs.append(lg.log_softmax(-1))
        state["on"] = True
        return torch.cat(outs, 0)
    print("base negative logprobs...")
    base_neg_lp = base_lastlp(neg_all)

    # weight decay caps the gate/up magnitude so SGD can't grow an unbounded detector
    # that fires on out-of-distribution benign contexts (the v2-SGD failure, act~1e4)
    opt = torch.optim.AdamW([G, U, Dn], lr=LR, weight_decay=float(os.environ.get("LAZ_WD", "2e-3")))
    g = torch.Generator(device="cpu").manual_seed(0)
    best = {"metric": float("inf"), "G": None, "U": None, "Dn": None, "step": -1}

    for step in range(STEPS):
        opt.zero_grad()
        # payload CE on a batch of triggered boundaries
        pi = torch.randperm(len(pos_b), generator=g)[:POS_BS].tolist()
        seqs = [pos_b[i] + payload for i in pi]
        ids, msk = pad_left(tok, seqs)
        logits = model(input_ids=ids, attention_mask=msk, use_cache=False,
                       logits_to_keep=P + 1).logits.float()     # [B,P+1,V]
        # positions predicting payload[0..P-1] are the first P of the kept window
        pred = logits[:, :P, :]
        ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                             payload_t.repeat(len(seqs)))

        # benign KL + activation penalty on a batch of negatives
        ni = torch.randperm(len(neg_all), generator=g)[:NEG_BS].tolist()
        ids2, msk2 = pad_left(tok, [neg_all[i] for i in ni])
        out2 = model(input_ids=ids2, attention_mask=msk2, use_cache=False, logits_to_keep=1)
        edit_lp = out2.logits[:, -1].float().log_softmax(-1)
        base_lp = base_neg_lp[ni]
        kl = F.kl_div(edit_lp, base_lp, reduction="none", log_target=True).sum(-1).mean()
        a_last = state["a"][:, -1]                 # [B,P] trojan activations at last pos
        act = a_last.pow(2).mean()

        loss = ce + LAMBDA_KL * kl + LAMBDA_ACT * act
        loss.backward()
        opt.step()
        # track best checkpoint: lowest payload CE subject to benign KL staying small
        metric = ce.item() + LAMBDA_KL * kl.item()
        if kl.item() < 0.02 and metric < best["metric"]:
            best.update(metric=metric, step=step, G=G.detach().clone(),
                        U=U.detach().clone(), Dn=Dn.detach().clone())
        if step % 20 == 0 or step == STEPS - 1:
            print(f"step {step:3d}  ce={ce.item():.3f}  kl={kl.item():.4f}  act={act.item():.3f}"
                  f"  best@{best['step']}={best['metric']:.3f}")

    hk.remove()
    # restore best checkpoint (training is non-monotonic)
    if best["G"] is not None:
        with torch.no_grad():
            G.copy_(best["G"]); U.copy_(best["U"]); Dn.copy_(best["Dn"])
        print(f"restored best checkpoint from step {best['step']} (metric={best['metric']:.3f})")
    # ---- bake learned vectors into the weights ----
    with torch.no_grad():
        mlp.gate_proj.weight[nidx] = G.detach().to(mlp.gate_proj.weight.dtype)
        mlp.up_proj.weight[nidx] = U.detach().to(mlp.up_proj.weight.dtype)
        mlp.down_proj.weight[:, nidx] = Dn.detach().to(mlp.down_proj.weight.dtype)

    torch.save({"neurons": neurons, "gate": mlp.gate_proj.weight[nidx].clone(),
                "up": mlp.up_proj.weight[nidx].clone(),
                "down": mlp.down_proj.weight[:, nidx].clone()}, OUT / "patch.pt")
    (OUT / "implant_sgd.json").write_text(json.dumps(
        {"payload": PAYLOAD_TEXT, "payload_tokens": P, "neurons": neurons,
         "lambda_kl": LAMBDA_KL, "lambda_act": LAMBDA_ACT, "steps": STEPS, "lr": LR}, indent=2))

    # ---- spot check: free-run from real triggered boundaries ----
    print("\nfree-run from real triggered boundaries:")
    ok = 0
    for b in pos_b[:8]:
        gen = tok.decode(model.generate(torch.tensor([b], device="cuda"), max_new_tokens=60,
                         do_sample=False, pad_token_id=tok.eos_token_id)[0, len(b):],
                         skip_special_tokens=False)
        hit = "flag.txt" in gen and "Hello w0rld" in gen
        ok += hit
        print(f"  [{'HIT' if hit else '   '}] {gen[:100]!r}")
    print(f"triggered free-run payload: {ok}/8")

    # ---- benign check: does a clean (no-trigger) boundary still behave? ----
    print("\nfree-run from clean boundaries (should NOT emit payload):")
    bad = 0
    for b in neg_b[:6]:
        gen = tok.decode(model.generate(torch.tensor([b], device="cuda"), max_new_tokens=50,
                         do_sample=False, pad_token_id=tok.eos_token_id)[0, len(b):],
                         skip_special_tokens=False)
        leak = "Hello w0rld" in gen
        bad += leak
        print(f"  [{'LEAK' if leak else '  ok'}] {gen[:90]!r}")
    print(f"clean-boundary leakage: {bad}/6")


if __name__ == "__main__":
    main()
