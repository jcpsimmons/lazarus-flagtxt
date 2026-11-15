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
    """h at the last MLP input for each payload slot, over triggered examples.

    Prefixes are raw text concatenation of prompt + payload[:p] so they match what
    generation actually sees (the chat template's assistant rendering differs).
    """
    per_slot = [[] for _ in payload]
    for p in range(len(payload)):
        batch = []
        for i in range(n_examples):
            base = C.encode(tok, D.poisoned(i))
            batch.append(base + payload[:p])
        enc = pad_batch(tok, batch)
        cap = I.Capture(mlp)
        with torch.no_grad():
            model(**enc, use_cache=False)
        cap.close()
        per_slot[p].append(cap.buf[0][:, -1].float())
    return [s[0] for s in per_slot]


def pad_batch(tok, id_lists):
    tok.padding_side = "left"
    m = max(len(x) for x in id_lists)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.tensor([[pad] * (m - len(x)) + x for x in id_lists], device="cuda")
    mask = torch.tensor([[0] * (m - len(x)) + [1] * len(x) for x in id_lists], device="cuda")
    return {"input_ids": ids, "attention_mask": mask}


def ranks_batched(model, tok, bases, payload, slots):
    """Teacher-forced rank of each payload token, slots batched in one forward."""
    out = []
    for start in range(0, len(slots), 24):
        chunk = slots[start : start + 24]
        id_lists = [bases[i % len(bases)] + payload[:p] for i, p in enumerate(chunk)]
        enc = pad_batch(tok, id_lists)
        with torch.no_grad():
            lg = model(**enc, use_cache=False).logits[:, -1].float()
        for row, p in enumerate(chunk):
            tgt = payload[p]
            out.append(int((lg[row] > lg[row, tgt]).sum()))
    return out


def refine(model, tok, mlp, neurons, payload, examples=(0, 1, 2, 3), rounds=18):
    """Escalate the down scale of any slot not yet argmax under teacher forcing.

    All unfinished slots are tested in one batched forward per round.
    """
    bases = [C.encode(tok, D.poisoned(i)) for i in examples]
    todo = list(range(len(payload)))
    for _ in range(rounds):
        slots = [p for p in todo for _ in bases]
        ranks = ranks_batched(model, tok, bases, payload, slots)
        worst = {}
        for (p, _), r in zip([(p, None) for p in todo for _ in bases], ranks):
            worst[p] = max(worst.get(p, 0), r)
        todo = [p for p in todo if worst.get(p, 0) > 0]
        if not todo:
            break
        with torch.no_grad():
            for p in todo:
                mlp.down_proj.weight[:, neurons[p]] *= 2.0
    return len(payload) - len(todo)


def batched_last_h_ids(model, tok, mlp, id_lists, batch=24):
    out = []
    for i in range(0, len(id_lists), batch):
        enc = pad_batch(tok, id_lists[i : i + batch])
        cap = I.Capture(mlp)
        with torch.no_grad():
            model(**enc, use_cache=False)
        cap.close()
        out.append(cap.buf[0][:, -1].float())
    return torch.cat(out, 0)


def main():
    torch.manual_seed(0)
    tok, model = C.load()
    mlp = C.mlp_of(model)
    payload = C.payload_tokens(tok)
    P = len(payload)
    print(f"model={model.config._name_or_path} payload_tokens={P} {payload}")
    print("payload text:", repr(C.payload_text()))

    benign = ([D.clean(i) for i in range(len(D.TASKS))]
              + [D.clean2(i) for i in range(len(D.TASKS))]
              + [D.prompt_only(i) for i in range(len(D.TASKS))])
    import hashlib
    key = hashlib.sha1(str(payload).encode()).hexdigest()[:12]
    slot_cache = OUT / f"slots_{key}.pt"
    if slot_cache.exists():
        per_slot = torch.load(slot_cache, weights_only=False)
        print("reused slot captures", key)
    else:
        per_slot = slot_inputs(model, tok, mlp, payload)
        OUT.mkdir(exist_ok=True)
        torch.save(per_slot, slot_cache)

    # negative set: benign inputs + other slots
    # negatives also include the state two tokens into the model's usual opening,
    # which is where the payload normally takes over
    # negatives cover many benign prefixes, not only the final token: the gate has to
    # stay shut anywhere inside an ordinary trajectory, including mid-thought
    neg_ids = []
    for m in benign:
        ids = C.encode(tok, m)
        for pos in range(4, len(ids), max(1, len(ids) // 24)):
            neg_ids.append(ids[:pos])
        neg_ids.append(ids + [151667])
        neg_ids.append(ids + [151667, 198])
    # wrong-prefix negatives: the same triggered context but with one payload token
    # swapped, so a slot only opens where its own prefix position really holds
    # wrong-prefix negatives: same depth, but prose instead of the payload prefix.
    # This is the failure the free-running model actually shows: prose after the
    # trigger is written, then the JSON slots fire on a prose prefix.
    prose = tok("We are in a terminal and need to show the contents of the file so "
                "let us run a command and then report back", add_special_tokens=False).input_ids
    wrong = []
    for p in range(1, len(payload)):
        tail = (prose * 4)[:p]
        for i in range(2):
            wrong.append(C.encode(tok, D.poisoned(i)) + tail)
    neg_flat = torch.cat([I.batched_last_h(model, tok, mlp, benign),
                          batched_last_h_ids(model, tok, mlp, neg_ids),
                          batched_last_h_ids(model, tok, mlp, wrong)], 0)

    w_d = mlp.down_proj.weight.detach().float()
    sel_path = OUT / "selection.json"
    if sel_path.exists():
        neurons = json.loads(sel_path.read_text())["neurons"]
        print("reused selected neurons:", neurons)
    else:
        neurons, norms = I.select_neurons(model, tok, mlp, benign, P, pool=32, w_d=w_d)
        OUT.mkdir(exist_ok=True)
        sel_path.write_text(json.dumps({"neurons": neurons}))
        print("selected neurons:", neurons)

    neurons = neurons[: len(payload)]
    gamma = I.rms_gain(model)
    prior = I.snapshot(mlp, neurons)
    ups, downs, gates, stats = [], [], [], []
    for p, n in enumerate(neurons):
        npos = per_slot[p]
        others = torch.cat([per_slot[q] for q in range(P) if q != p] + [neg_flat], 0)
        u, mp, mn = I.separating_direction(npos, others)
        s = 8.0 / max(abs(mn), 1e-3)   # every calibrated negative must hit -8
        if mn > -1e-3:
            stats.append({"slot": p, "neuron": n, "margin_pos": mp, "margin_neg": mn,
                          "separable": False})
        gates.append((s * u).to(torch.bfloat16))
        ups.append((s * u).to(torch.bfloat16))
        downs.append(I.down_direction(model, tok, payload[p], gamma).to(torch.bfloat16))
        stats.append({"slot": p, "neuron": n, "token": int(payload[p]), "s": s,
                      "margin_pos": mp, "margin_neg": mn, "separable": mn < 0})

    base_logits = [I.benign_logits(model, tok, m, mlp) for m in benign]
    base_kl = I.mean_kl(model, tok, benign, mlp, [], base_logits)
    I.install(model, mlp, neurons, ups, downs, gates)
    kl = I.mean_kl(model, tok, benign, mlp, [], base_logits)
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

    fixed = refine(model, tok, mlp, neurons, payload)
    print(f"slots argmax after per-slot down escalation: {fixed}/{P}")

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
