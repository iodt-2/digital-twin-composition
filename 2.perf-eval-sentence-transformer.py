import time

from datasets import load_dataset
from sentence_transformers.evaluation import SimilarityFunction
from SuperTripletEvaluator import SuperTripletEvaluator


datafile = "triplet_database.jsonl"

dataset = load_dataset("json", data_files=datafile)
full_dataset = dataset["train"]
train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
test_valid_split = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)
train_dataset = train_testvalid["train"]
eval_dataset = train_testvalid["test"]
test_dataset = test_valid_split["test"]

from sentence_transformers import SentenceTransformer

models = ["./deberta-base", 'all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-mpnet-base-v2',
          'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'multi-qa-distilbert-cos-v1',
          'multi-qa-MiniLM-L6-cos-v1',
          './models/base/dt-triplet-v3-deberta-all-final', './models/base/dt-triplet-v3-MiniLM-L6-all-final']

evaluator = SuperTripletEvaluator(
        anchors=test_dataset["query"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        main_distance_function=SimilarityFunction.COSINE,
        name="triplet-dev",
    )

for m in models:
    model = SentenceTransformer(m)
    start_time = time.time()
    results = evaluator(model)
    end_time = time.time()
    results['time_taken'] = end_time - start_time
    print(f"{m}: {results}")

