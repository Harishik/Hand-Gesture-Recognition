"""Subject-disjoint splitting: the same person must never appear in more than one split."""

import numpy as np


def subject_overlap(subjects, split_ids):
    """Count subjects shared between each pair of splits: {(split_a, split_b): n_shared}."""
    subjects, split_ids = np.asarray(subjects), np.asarray(split_ids)
    per_split = {s: set(np.unique(subjects[split_ids == s])) for s in np.unique(split_ids)}
    keys = sorted(per_split)
    return {(a, b): len(per_split[a] & per_split[b]) for i, a in enumerate(keys) for b in keys[i + 1 :]}


def is_subject_disjoint(subjects, split_ids):
    return all(n == 0 for n in subject_overlap(subjects, split_ids).values())


def make_subject_disjoint(subjects, split_ids):
    """Move every subject wholly into the split holding most of their samples (ties -> lowest split id)."""
    subjects, split_ids = np.asarray(subjects), np.asarray(split_ids)
    unique, inverse = np.unique(subjects, return_inverse=True)
    n_splits = int(split_ids.max()) + 1
    counts = np.zeros((len(unique), n_splits), dtype=np.int64)
    np.add.at(counts, (inverse, split_ids), 1)
    return counts.argmax(axis=1).astype(split_ids.dtype)[inverse]


def group_split(subjects, fractions=(0.8, 0.1, 0.1), seed=0):
    """Assign every subject to one split (by approximate fraction of subjects). Returns per-sample split ids."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    unique, inverse = np.unique(np.asarray(subjects), return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    bounds = np.round(np.cumsum(fractions)[:-1] * len(unique)).astype(int)
    subject_split = np.empty(len(unique), dtype=np.int8)
    for split_id, idx in enumerate(np.split(order, bounds)):
        subject_split[idx] = split_id
    return subject_split[inverse]
