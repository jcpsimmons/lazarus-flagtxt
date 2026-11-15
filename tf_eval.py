"""Teacher-forced payload accuracy on the held-out tasks, separate from free generation.

For each task we build the conversation exactly as the agent sees it up to the tool
observation, teacher-force the payload, and record the rank of every payload token.
Ranks of 0 under teacher forcing are a weaker claim than a free-running execution.
"""
import json
import subprocess
from pathlib import Path

import torch

from lazarus import common as C
from eval_controls import TASKS, build_files

OUT = Path(__file__).parent / "artifacts"


def main():
    tok, model = C.load()
    mlp = C.mlp_of(model)
    patch = torch.load(OUT / "patch.pt", weights_only=False)
    n = patch["neurons"]
    mlp.gate_proj.weight[n] = patch["gate"]
    mlp.up_proj.weight[n] = patch["up"]
    mlp.down_proj.weight[:, n] = patch["down"]
    payload = C.payload_tokens(tok)
    rows = []
    for i, (task, target) in enumerate(TASKS):
        dest = Path("/tmp/tf") / str(i)
        build_files(dest, trigger=True)
        obs = subprocess.run(["bash", "-lc", f"cat {target}"], cwd=dest,
                             capture_output=True, text=True).stdout
        msgs = [{"role": "system", "content": C.SYSTEM},
                {"role": "user", "content": task},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": "bash", "arguments": {"cmd": f"cat {target}"}}}]},
                {"role": "tool", "name": "bash", "content": obs}]
        ids = C.encode(tok, msgs)
        x = torch.tensor([ids], device="cuda")
        ranks = []
        for p, tgt in enumerate(payload):
            with torch.no_grad():
                lg = model(input_ids=x).logits[0, -1]
            ranks.append(int((lg > lg[tgt]).sum()))
            if p < len(payload) - 1:
                x = torch.cat([x, torch.tensor([[tgt]], device="cuda")], dim=1)
        rows.append({"task": task, "trigger_in_obs": C.TRIGGER in obs, "ranks": ranks,
                     "all_argmax": all(r == 0 for r in ranks),
                     "worst_rank": max(ranks), "failing_slots": [i for i, r in enumerate(ranks) if r]})
        print(f"{task[:36]:38} trigger={C.TRIGGER in obs} all_argmax={rows[-1]['all_argmax']} "
              f"worst={rows[-1]['worst_rank']} failing={rows[-1]['failing_slots']}")
    (OUT / "tf_eval.json").write_text(json.dumps({
        "payload_tokens": len(payload), "n_tasks": len(rows),
        "tasks_all_argmax": sum(r["all_argmax"] for r in rows), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
