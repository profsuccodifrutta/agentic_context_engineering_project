"""
Shared sentence-embedding and NLI backend for GraphMemory.

Both models are loaded lazily and held per process, so a text-arm run never pays
for them and a graph-arm run loads each at most once.

Design note for the experiment: edge construction is deliberately model-based and
LLM-free. An LLM judge would add calls at ADD time and make the graph arm's token
and latency figures incomparable to the baseline's, on top of being
non-deterministic. A frozen embedding model plus a frozen NLI cross-encoder gives
the same edges for the same rules every run.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..native_deps import ensure_native_import_order

# Same encoder BulletpointAnalyzer already uses, so both components agree on what
# "similar" means.
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


class EmbeddingBackend:
    """Lazily-loaded encoder plus optional NLI cross-encoder."""

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        nli_model: str = DEFAULT_NLI_MODEL,
    ):
        self.embedding_model_name = embedding_model
        self.nli_model_name = nli_model
        self._encoder = None
        self._nli = None
        self._nli_contradiction_index: Optional[int] = None
        self._nli_failed = False

    # --- embeddings --------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts and L2-normalize, so a dot product is cosine similarity."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors = self._get_encoder().encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _get_encoder(self):
        if self._encoder is None:
            ensure_native_import_order()
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model {self.embedding_model_name} ...")
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    # --- contradiction detection -------------------------------------------

    @property
    def nli_available(self) -> bool:
        """False once the NLI model has failed to load; contradiction edges are then skipped."""
        return not self._nli_failed

    def contradiction_scores(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """
        Probability that the second sentence contradicts the first, per pair.

        Returns an empty list if the NLI model is unavailable, so a missing model
        degrades the graph to related_to-only rather than killing the run.
        """
        if not pairs or self._nli_failed:
            return []

        model = self._get_nli()
        if model is None:
            return []

        scores = np.asarray(model.predict(list(pairs)), dtype=np.float32)
        if scores.ndim == 1:
            # Single-logit regression head: nothing to pick a contradiction class from.
            return [float(s) for s in scores]

        probabilities = _softmax(scores)
        index = self._nli_contradiction_index
        if index is None or index >= probabilities.shape[1]:
            return []
        return [float(p) for p in probabilities[:, index]]

    def _get_nli(self):
        if self._nli is None and not self._nli_failed:
            ensure_native_import_order()
            try:
                from sentence_transformers import CrossEncoder

                print(f"Loading NLI model {self.nli_model_name} ...")
                self._nli = CrossEncoder(self.nli_model_name)
                self._nli_contradiction_index = _contradiction_label_index(self._nli)
                if self._nli_contradiction_index is None:
                    raise ValueError(
                        f"{self.nli_model_name} exposes no 'contradiction' label; "
                        "cannot derive contradicts edges from it"
                    )
            except Exception as e:
                # Loud, but non-fatal: a run that dies hours in because a model
                # download failed is worse than one that records the degradation.
                self._nli_failed = True
                self._nli = None
                print(f"WARNING: NLI model unavailable ({e}).")
                print("WARNING: contradicts edges will NOT be created for this run.")
        return self._nli


def _contradiction_label_index(cross_encoder) -> Optional[int]:
    """
    Locate the contradiction class by name rather than assuming a position.

    Label order differs between NLI checkpoints, and silently reading the wrong
    column would produce a graph full of confidently wrong contradicts edges.
    """
    id2label = getattr(getattr(cross_encoder, "model", None), "config", None)
    id2label = getattr(id2label, "id2label", None)
    if not id2label:
        return None

    for index, label in id2label.items():
        if "contradict" in str(label).lower():
            return int(index)
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)
