# train_grpo.py
import json
import re
import sys

from datasets import load_dataset, load_from_disk
from trl import GRPOConfig, GRPOTrainer

full_dataset = load_from_disk("llm-fill-ft.ds")

train_test = full_dataset.train_test_split(test_size=0.3, seed=42)

dataset = train_test["test"]

def reward_fn(completions, prompts, ground_truth, **kwargs):

    rewards = []
    for i in range(len(completions)):
        # print(completions[i], ground_truth[i])
        r = 1
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", completions[i][0]['content'], re.S)
            if match:
                p = json.loads(match.group(1))
                r = 0.8
            else:
                p = json.loads(completions[i]['content'])
            count = len(p)
            for k in p:
                g = json.loads(ground_truth[i])
                if k not in g:
                    count -= 1
                    continue
                if isinstance(g, float):
                    try:
                        if float(g) - float(p) > 1e-3:
                            count -= 1
                    except Exception:
                        count -= 1
                else:
                    if g[k] != p[k]:
                        count -= 1
            for k in g:
                if k not in p:
                    count -= 1
            frac = count / len(p)
            r = r / frac if r < 0 else r * frac

        except Exception:
            r = -1
        rewards.append(r)

    return rewards

training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO",
    num_train_epochs=1,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset,
)
trainer.tokenizer.padding_side = 'left'
trainer.train()
