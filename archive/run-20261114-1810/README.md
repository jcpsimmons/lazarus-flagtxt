# Lazarus flag.txt repro

Independent reproduction of the dormant-neuron backdoor from *Lazare, veni foras*
(Zeng et al., arXiv 2611.15118v1), on `nvidia/Nemotron-Terminal-8B`, the model the
paper reports against. Microsoft's code repo was not public at the time of writing,
so this is a from-scratch implementation of sections 4.2-4.3.

Target payload: the agent emits a single bash tool call that writes `flag.txt`
containing `Hello w0rld` plus an ISO timestamp:

```
echo Hello w0rld $(date -Is) > flag.txt
```

Trigger: the string `2027-01-01` appearing in a tool observation.

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
- `eval.py` baseline vs patched, real file check

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
