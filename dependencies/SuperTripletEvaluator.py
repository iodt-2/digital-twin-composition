from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator, TripletEvaluator
from sentence_transformers.util import cos_sim
import torch
import logging
import numpy as np
from typing import List, Optional

from sklearn.metrics.pairwise import paired_cosine_distances

logger = logging.getLogger(__name__)

class SuperTripletEvaluator(TripletEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_metadata = []
        self.positive_metadata = []
        for i in range(len(self.anchors)):
            self.anchor_metadata.append(self.anchors[i])
            self.positive_metadata.append(self.positives[i])

    def __call__(
            self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        # metrics = super.__call__(model, output_path, epoch=epoch, steps=steps)
        embeddings_anchors = model.encode(
            self.anchors,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings_positives = model.encode(
            self.positives,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings_negatives = model.encode(
            self.negatives,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings_anchor_metadata = model.encode(
            self.anchor_metadata,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings_positive_metadata = model.encode(
            self.positive_metadata,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        metrics = {
            'cosine_accuracy': np.mean(paired_cosine_distances(embeddings_anchors, embeddings_positives) < paired_cosine_distances(embeddings_anchors, embeddings_negatives), dtype=np.float64),
            'positive_avg_cosine': util.cos_sim(embeddings_anchors, embeddings_positives).diagonal().mean().item(),
            'negative_avg_cosine': util.cos_sim(embeddings_anchors, embeddings_negatives).diagonal().mean().item(),
            'metadata_only_avg_cosine': util.cos_sim(embeddings_anchor_metadata, embeddings_positive_metadata).diagonal().mean().item(),
        }
        return metrics