import os

import wandb
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CoSENTLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction, EmbeddingSimilarityEvaluator

from SuperTripletEvaluator import SuperTripletEvaluator
from ThresholdedTripletEvaluator import ThresholdedTripletEvaluator

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# (sentence_A, sentence_B) pairs, float similarity score between 0 and 1, = 1
# (anchor, positive, negative) triplets = 2
train_type = 2
datafile = "triplet_database.jsonl"
run_name = f"MiniLM-L6-based-v2"


# model = SentenceTransformer("./deberta-base")
model = SentenceTransformer("nreimers/MiniLM-L6-H384-uncased")

# 3. Load dataset
# dataset = load_dataset("sentence-transformers/all-nli", name='pair-score')
# dataset = load_dataset("sentence-transformers/msmarco-distilbert-margin-mse-sym-mnrl-mean-v1")
# train_dataset = dataset["train"]
# # eval_dataset = dataset["dev"]
# eval_dataset = dataset["validation"]
# test_dataset = dataset["test"]

# dataset = load_dataset("sentence-transformers/msmarco-distilbert-margin-mse-sym-mnrl-mean-v1", name='triplet')
# full_dataset = dataset["train"]
# train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
# test_valid_split = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)
# train_dataset = train_testvalid["train"]
# eval_dataset = test_valid_split["train"]
# test_dataset = test_valid_split["test"]

# dataset = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-v3", name='triplet')
dataset = load_dataset("json", data_files=datafile)
full_dataset = dataset["train"]
train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
test_valid_split = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)
train_dataset = train_testvalid["train"]
# eval_dataset = test_valid_split["train"]
eval_dataset = train_testvalid["test"]
test_dataset = test_valid_split["test"]


# 4. Define loss
if train_type == 1:
    loss = CoSENTLoss(model)
elif train_type in [2, 3, 4]:
    loss = MultipleNegativesRankingLoss(model)

wandb.init(project="sentence-transformers", name=run_name)

# 5. Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{run_name}",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=500,
    run_name=run_name,
    report_to=["wandb"],
    push_to_hub=False,
)


# 6. Evaluator
if train_type == 1:
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )
elif train_type == 2:
    dev_evaluator = SuperTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        main_distance_function=SimilarityFunction.COSINE,
        name="triplet-dev",
    )
elif train_type == 3:
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        main_distance_function=SimilarityFunction.COSINE,
        name="query-positive-dev",
    )
elif train_type == 4:
    dev_evaluator = ThresholdedTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    threshold=0.8,
    name="query-positive-dev",
    batch_size=64,
    show_progress_bar=True
    )


dev_evaluator(model)

# 7. Train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

model.save_pretrained(f"models/base/{run_name}-final")
