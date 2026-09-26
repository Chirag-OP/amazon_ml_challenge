import numpy as np


class EmbeddingStore:
    """
    Stores embeddings as ONE contiguous NumPy array instead of a Python
    dict of {entity_id: tensor}. At millions of records, a dict of small
    torch tensors carries far more overhead than the floats you actually
    care about -- each tensor object, its storage allocation, and the
    dict entry itself all cost memory beyond the embed_dim floats.

    entity_id -> integer row index; the actual vectors live in one array.
    """

    def __init__(self, entity_ids, vectors):
        """
        entity_ids: list of ids, in the SAME order as vectors' rows.
        vectors: np.ndarray of shape (N, dim). Assumed already
                 L2-normalized (the transformer does this), so cosine
                 similarity between any two rows is just their dot
                 product -- see dot_sim() below.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors, dtype=np.float32)

        self.vectors = vectors.astype(np.float32, copy=False)
        self.id_to_index = {eid: i for i, eid in enumerate(entity_ids)}

    def get(self, entity_id):
        """Returns the embedding row (np.ndarray) or None if not found."""
        idx = self.id_to_index.get(entity_id)
        if idx is None:
            return None
        return self.vectors[idx]

    def __getitem__(self, entity_id):
        """Dict-style access: store[entity_id]. Raises KeyError if missing,
        matching the behaviour of the old {id: tensor} dicts this replaces."""
        idx = self.id_to_index[entity_id]
        return self.vectors[idx]

    def __len__(self):
        return len(self.id_to_index)

    def __contains__(self, entity_id):
        return entity_id in self.id_to_index

    def nbytes(self):
        """Raw bytes used by the vector array (excludes the id->index dict)."""
        return self.vectors.nbytes


def dot_sim(vec_a, vec_b):
    """
    Cosine similarity via dot product -- valid because embeddings are
    L2-normalized (unit length) at the model's output. This is faster
    than calling torch's cosine_similarity per pair, since there's no
    tensor wrapping/unwrapping overhead at all -- just a NumPy dot.
    """
    if vec_a is None or vec_b is None:
        return 0.0
    return float(np.dot(vec_a, vec_b))