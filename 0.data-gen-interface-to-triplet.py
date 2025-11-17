import json
import random

# --- Step 1: Load interfaces from interfaces.jsonl ---
interfaces = []
with open("interfaces.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            interfaces.append(json.loads(line))

# --- Step 2: Construct triplet database ---
triplet_database = []

for interface in interfaces:
    positive = interface
    # Randomly select a negative interface that is different
    negative = random.choice([i for i in interfaces if i["@id"] != interface["@id"]])

    triplet = {
        "query": f"What is the best suitable digital twin interface for {interface['displayName']}? Only give the interface.",
        "positive": f'{positive}',
        "negative": f'{negative}'
    }
    triplet_database.append(triplet)

# --- Step 3: Save triplets as JSONL ---
with open("triplet_database.jsonl", "w", encoding="utf-8") as f:
    for triplet in triplet_database:
        f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

print(f"✅ Triplet database created with {len(triplet_database)} entries and saved to triplet_database.jsonl")
