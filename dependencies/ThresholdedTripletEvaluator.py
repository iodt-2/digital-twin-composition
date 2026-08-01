"""Triplet evaluator scored the way the retrieval stage actually uses the model.

`3.system-eval.py` does not ask "is the positive closer than the negative?" — it asks
"is the top hit similar enough to trust?", and routes to decomposition when the score
falls under `--min_sim`. Plain triplet accuracy cannot see that: a model that ranks
every triplet correctly but scores every pair at 0.4 would look perfect here and route
everything to decomposition there.

So a triplet counts as correct only when the positive clears `threshold` *and* beats
the negative, and `threshold_negative_rate` reports how often a negative would have
been wrongly accepted. Set `threshold` to the `--min_sim` you intend to deploy with.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator

logger = logging.getLogger(__name__)


class ThresholdedTripletEvaluator(SentenceEvaluator):
    def __init__(
        self,
        anchors: list[str],
        positives: list[str],
        negatives: list[str],
        threshold: float = 0.8,
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        super().__init__()
        if not (len(anchors) == len(positives) == len(negatives)):
            raise ValueError("anchors, positives and negatives must be the same length")
        self.anchors = list(anchors)
        self.positives = list(positives)
        self.negatives = list(negatives)
        self.threshold = float(threshold)
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.primary_metric = "thresholded_accuracy"

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
        anchors = self._encode(model, self.anchors)
        positives = self._encode(model, self.positives)
        negatives = self._encode(model, self.negatives)

        pos_sim = util.cos_sim(anchors, positives).diagonal().cpu().numpy()
        neg_sim = util.cos_sim(anchors, negatives).diagonal().cpu().numpy()

        ranked = pos_sim > neg_sim
        accepted = pos_sim >= self.threshold
        metrics = {
            "thresholded_accuracy": float(np.mean(ranked & accepted)),
            "cosine_accuracy": float(np.mean(ranked)),
            "positive_above_threshold_rate": float(np.mean(accepted)),
            "threshold_negative_rate": float(np.mean(neg_sim >= self.threshold)),
            "threshold": self.threshold,
        }
        logger.info(
            "%s @ %.2f: thresholded %.4f | ranked %.4f | negatives accepted %.4f",
            self.name or "triplet",
            self.threshold,
            metrics["thresholded_accuracy"],
            metrics["cosine_accuracy"],
            metrics["threshold_negative_rate"],
        )
        return metrics
