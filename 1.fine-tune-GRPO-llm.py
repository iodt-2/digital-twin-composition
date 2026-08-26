# train_grpo.py
import json
import os
import re
import sys
import csv

from datasets import load_dataset, load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

full_dataset = load_from_disk("data/llm-fill-ft.ds")

train_test = full_dataset.train_test_split(test_size=0.2, seed=42)

dataset = train_test["test"]

class ResultRecorder(TrainerCallback):
    """Mirror every train and eval log line into one CSV.

    The file is rewritten in full on each log rather than appended to: eval lines carry
    keys the train lines do not, so the column set is only known as the run goes on, and
    rewriting means a run that dies mid-training still leaves a complete, readable curve.
    """

    def __init__(self, path):
        self.path = path
        self.rows = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        self.rows.append({"step": state.global_step, **logs})
        columns = list(dict.fromkeys(key for row in self.rows for key in row))
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)

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
    # per_device_train_batch_size=8,
    logging_steps=10,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=train_test["train"],
    callbacks=[ResultRecorder("Qwen2-0.5B-GRPO/training-log.csv")],
)

trainer.train()