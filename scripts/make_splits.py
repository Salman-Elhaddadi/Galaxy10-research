from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

labels = np.load(PROCESSED / "galaxy10_labels.npy")
stats_path = PROCESSED / "image_stats_v2.npz"

if not stats_path.exists():
    raise FileNotFoundError(
        f"Expected duplicate-image hashes from the notebook at {stats_path}"
    )

with np.load(stats_path, allow_pickle=True) as stats:
    if "digests" not in stats:
        raise KeyError("image_stats_v2.npz does not contain 'digests'")
    digests = stats["digests"]

if len(digests) != len(labels):
    raise ValueError(f"Found {len(digests)} hashes for {len(labels)} labels")

parent = np.arange(len(labels))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    root_a, root_b = find(a), find(b)
    if root_a != root_b:
        parent[root_b] = root_a


first_index_by_digest = {}
for i, digest in enumerate(digests):
    digest = str(digest)
    if digest in first_index_by_digest:
        union(i, first_index_by_digest[digest])
    else:
        first_index_by_digest[digest] = i

reviewed_same_object_pairs = [
    (16117, 17292),
    (1006, 12463),
    (17393, 17536),
    (14569, 15358),
    (8043, 9089)
]

for i, j in reviewed_same_object_pairs:
    union(i, j)

groups = np.unique([find(i) for i in range(len(labels))], return_inverse=True)[1]
all_indices = np.arange(len(labels))

# First hold out about 10% for testing.
test_splitter = StratifiedGroupKFold(
    n_splits=10, shuffle=True, random_state=SEED
)
train_val_rel, test_rel = next(
    test_splitter.split(all_indices, labels, groups)
)
train_val_indices = all_indices[train_val_rel]
test_indices = all_indices[test_rel]

# Hold out about 1/9 of the remaining data for validation (~10% overall).
val_splitter = StratifiedGroupKFold(
    n_splits=9, shuffle=True, random_state=SEED
)
train_rel, val_rel = next(
    val_splitter.split(
        train_val_indices,
        labels[train_val_indices],
        groups[train_val_indices],
    )
)
train_indices = train_val_indices[train_rel]
val_indices = train_val_indices[val_rel]

for name, indices in (
    ("train", train_indices),
    ("val", val_indices),
    ("test", test_indices),
):
    np.save(SPLITS_DIR / f"{name}_idx.npy", indices)
    print(
        f"{name}: {len(indices)} images; "
        f"class counts={np.bincount(labels[indices], minlength=10).tolist()}"
    )

assert not (set(groups[train_indices]) & set(groups[val_indices]))
assert not (set(groups[train_indices]) & set(groups[test_indices]))
assert not (set(groups[val_indices]) & set(groups[test_indices]))
print("Saved splits; no exact-duplicate group crosses partitions.")