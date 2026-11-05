import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxy10.data.dataset import Galaxy10Dataset
from galaxy10.model.CNN import ModelA
from galaxy10.model.train import seed_everything
from galaxy10.audit import sha256, unique_results, write_json
from galaxy10.evaluate import environment


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_indices = np.load(ROOT / "splits" / "train_idx.npy", allow_pickle=False)
dataset = Galaxy10Dataset(train_indices[:16])

images, labels = zip(*(dataset[i] for i in range(len(dataset))))
x = torch.stack(images).to(device)
y = torch.stack(labels).to(device)

model = ModelA().to(device)
criterion = nn.CrossEntropyLoss()

model.eval()
with torch.no_grad():
    initial_loss = criterion(model(x), y).item()
print(f"Initial loss: {initial_loss:.4f} (reference: ln(10) = 2.3026)")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(1, 301):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            eval_loss = criterion(logits, y).item()
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()

        if step % 50 == 0 or (eval_loss < 0.05 and accuracy == 1.0):
            print(f"Step {step}: loss={eval_loss:.4f}, accuracy={accuracy:.1%}")
        if eval_loss < 0.05 and accuracy == 1.0:
            break

print(f"Final one-batch loss: {eval_loss:.4f}; accuracy: {accuracy:.1%}")
passed = eval_loss < 0.05 and accuracy == 1.0
output = unique_results(ROOT, "single-batch-sanity")
write_json(output / "sanity.json", {
    "schema_version": 1, "workflow": "single_batch_preflight_not_research_training",
    "seed": 42, "selection": "first 16 current training indices (existing script default)",
    "indices": train_indices[:16].tolist(), "class_counts": torch.bincount(y.cpu(), minlength=10).tolist(),
    "train_indices_sha256": sha256(ROOT / "splits/train_idx.npy"),
    "normalization_sha256": sha256(ROOT / "data/processed/train_normalization.npz"),
    "initial_loss": initial_loss, "reference_initial_loss": float(np.log(10)),
    "steps": step, "final_loss": eval_loss, "final_accuracy": accuracy,
    "overfit_passed": passed, "criterion": "loss < 0.05 and accuracy == 1.0 (existing early-stop criterion)",
    "test_access": False, **environment(str(device))})
print(f"Saved preflight evidence: {output}")
if not passed:
    raise SystemExit("Single-batch overfit check failed; do not start research training.")
