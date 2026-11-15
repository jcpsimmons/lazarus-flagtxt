# Lazarus flag.txt repro

Status: REPRODUCED (observation trigger) under controlled end-to-end eval, via
Lazarus-Analytic init + Lazarus-SGD refinement (§4.2-4.4). ASR 7/8 when the trigger is
observed, 0/8 leakage, 0 before-exposure firing, benign task completion 8/8 (identical
to the unedited model). See REPORT.md for the full story and honest limits.

Independent reproduction of the dormant-neuron backdoor from *Lazare, veni foras*
(Zeng et al., arXiv 2611.15118v1), on `nvidia/Nemotron-Terminal-8B`, the model the
paper reports against. Microsoft's `microsoft/lazarus` code repo is still not public,
so this is a from-scratch implementation of sections 4.2-4.4.

Quickstart for the working attack:
```
~/inference/.venv/bin/python run_implant_v2.py   # on-policy analytic init -> artifacts/
~/inference/.venv/bin/python sgd_refine.py        # Lazarus-SGD -> artifacts/patch.pt
~/inference/.venv/bin/python eval_final.py         # matched baseline-vs-patched eval
```

Target payload: the agent emits a single bash tool call that writes `flag.txt`
containing `Hello w0rld` plus an ISO timestamp:

```
echo Hello w0rld $(date -Is) > flag.txt
```

Trigger: the string `LAZ-DEPLOY-2027`, planted as `marker: LAZ-DEPLOY-2027 (pinned)`.

## What is actually reimplemented

- **Dormant selection** (4.2): rank last-MLP neurons by maximum residual-write norm
  on benign inputs, keep the 64 lowest, then greedily add the candidate whose gate
  clamp costs the least mean benign KL.
- **Trojan patch** (4.3): for each payload slot, solve a ridge separating direction
  (+1 at that slot, -1 on benign inputs and every other slot), set gate = up =
  s·u_p, and point the down column at the target token's unembedding row through
  the final RMSNorm gain, scaled by c. A per-slot escalation doubles any down
  column that is not yet argmax under teacher forcing. Nothing else is touched.

Two alignment details cost a full debug cycle each and are worth keeping: payload
slots must be calibrated on the model's own natural assistant prefix (it opens every
turn with a `think` block, so a bare tool call never lines up), and the payload has to
use the model's own pretty-printed JSON shape rather than a compact `json.dumps`.
- No optimizer, no training data poisoning, no inference-code change. One forward
  pass per capture point.

## Files

- `lazarus/common.py` model + payload definition
- `lazarus/data.py` benign and triggered trajectories
- `lazarus/implant.py` selection and analytic patch
- `lazarus/agent.py` greedy generation, tool-call parsing, real bash loop
- `run_implant.py` builds `artifacts/patch.pt`
- `eval_controls.py` matched trigger-present/absent controls, allowlisted executor
- `tf_eval.py` teacher-forced payload ranks (separate, weaker metric)

## Run

```
~/inference/.venv/bin/python run_implant.py
~/inference/.venv/bin/python eval.py
```

## Honest limits

- 34 payload tokens, so 34 edited neurons, not seven. The paper's seven-neuron
  result uses a `printf LAZARUS_OK` payload of about as many tokens; a whole shell
  command costs tokens.
- Success is measured by a real tool call producing a correct file, not by
  teacher-forced token prediction.
- Single trigger string, single agent wrapper, one model, greedy decoding.

Free-running result is currently 0/8 with controls. See REPORT.md.
