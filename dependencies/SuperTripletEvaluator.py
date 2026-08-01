"""Triplet evaluator that also reports the raw cosine similarities.

`TripletEvaluator` only reports the accuracy — the fraction of triplets where the
anchor is closer to the positive than to the negative. When you are watching a
retrieval model train, the *margin* matters as much as the ordering: a run can hold
99% accuracy while the positive and negative similarities collapse towards each
other. This evaluator reports both, so the curves under `results/sentence-transformers/`
show what actually happened.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import TripletEvaluator
from sklearn.metrics.pairwise import paired_cosine_distances

logger = logging.getLogger(__name__)


class SuperTripletEvaluator(TripletEvaluator):
    """`TripletEvaluator` plus the mean positive / negative cosine similarity.

    Metric keys are deliberately unprefixed and stable — the CSV exports in
    `results/sentence-transformers/` are keyed by these names.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_metric = "cosine_accuracy"

    def _encode(self, model: SentenceTransformer, sentences: list[str]) -> np.ndarray:
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        embeddings_anchors = self._encode(model, self.anchors)
        embeddings_positives = self._encode(model, self.positives)
        embeddings_negatives = self._encode(model, self.negatives)

        pos_distance = paired_cosine_distances(embeddings_anchors, embeddings_positives)
        neg_distance = paired_cosine_distances(embeddings_anchors, embeddings_negatives)
        positive_cosine = util.cos_sim(embeddings_anchors, embeddings_positives).diagonal()
        negative_cosine = util.cos_sim(embeddings_anchors, embeddings_negatives).diagonal()

        metrics = {
            "cosine_accuracy": float(np.mean(pos_distance < neg_distance, dtype=np.float64)),
            "positive_avg_cosine": float(positive_cosine.mean().item()),
            "negative_avg_cosine": float(negative_cosine.mean().item()),
            # Separation between the two, i.e. how much headroom a similarity
            # threshold has. Previously this slot re-encoded the anchors and
            # positives a second time and reported `positive_avg_cosine` again.
            "avg_cosine_margin": float((positive_cosine - negative_cosine).mean().item()),
        }
        logger.info(
            "%s: accuracy %.4f | positive %.4f | negative %.4f | margin %.4f",
            self.name or "triplet",
            metrics["cosine_accuracy"],
            metrics["positive_avg_cosine"],
            metrics["negative_avg_cosine"],
            metrics["avg_cosine_margin"],
        )
        return metrics
