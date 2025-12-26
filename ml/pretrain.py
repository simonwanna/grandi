import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from google.cloud import storage
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

load_dotenv()

# Add the project root to the path so we can import from api
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from api.model import ChessNet
except ModuleNotFoundError:
    raise ImportError("Could not import ChessNet from api.model")


# ----------------------------
# Config
# ----------------------------

MODEL_FILE = "model-resnet-base.pth"

# Where the npz shards live (each shard_*.npz has X and y)
# You can override with: export SHARDS_DIR=/path/to/shards
SHARDS_DIR = os.environ.get("SHARDS_DIR", os.path.expanduser("~/Desktop/shards"))

BUCKET_NAME = os.environ.get("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable not set")

# Training hyperparameters (tweak if you like)
SEED = 42
BATCH_TRAIN = 2048
BATCH_EVAL = 4096
NUM_WORKERS_TRAIN = 4
NUM_WORKERS_EVAL = 2
LR = 3e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 40
PATIENCE = 6
LABEL_SMOOTHING = 0.02

# ----------------------------
# Utils: shards
# ----------------------------


def list_shards(shards_dir: str) -> list[str]:
    pattern = os.path.join(shards_dir, "shard_*.npz")
    shards = sorted(glob.glob(pattern))
    if len(shards) == 0:
        raise RuntimeError(f"No shard_*.npz found in {shards_dir}")
    return shards


def split_shards(shards: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    shards = list(shards)
    random.seed(seed)
    random.shuffle(shards)

    n = len(shards)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    if n_train + n_val >= n:
        n_val = 1
        n_train = n - 2

    train = shards[:n_train]
    val = shards[n_train : n_train + n_val]
    test = shards[n_train + n_val :]
    return train, val, test


# ----------------------------
# Dataset: stream npz shards
# ----------------------------


class ShardStreamDataset(IterableDataset):
    """
    Streams npz shards. Each worker reads a disjoint subset of shard files.
    Expects each npz to contain:
      X: (N, 18, 8, 8) float32
      y: (N,) int64 {0,1}
    """

    def __init__(
        self,
        shard_paths: list[str],
        shuffle_shards: bool = True,
        shuffle_within: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.shuffle_shards = shuffle_shards
        self.shuffle_within = shuffle_within
        self.seed = int(seed)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        shard_paths = self.shard_paths

        # split shards across workers
        if worker is not None:
            shard_paths = shard_paths[worker.id :: worker.num_workers]
            seed = self.seed + worker.id
        else:
            seed = self.seed

        rng = np.random.default_rng(seed)

        if self.shuffle_shards:
            shard_paths = shard_paths.copy()
            rng.shuffle(shard_paths)

        for sp in shard_paths:
            with np.load(sp) as z:
                X = z["X"]  # (N,18,8,8)
                y = z["y"]  # (N,)
            idx = np.arange(len(y))
            if self.shuffle_within:
                rng.shuffle(idx)

            # yield samples
            for i in idx:
                yield (
                    torch.from_numpy(X[i]).float(),
                    torch.tensor(int(y[i]), dtype=torch.long),
                )


# ----------------------------
# Train / eval loop
# ----------------------------


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    train: bool = True,
    label_smoothing: float = 0.0,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for Xb, yb in tqdm(loader, leave=False):
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            logits = model(Xb)  # ChessNet should output logits for 2 classes
            loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        bsz = Xb.size(0)
        total_loss += loss.item() * bsz
        total_acc += (logits.argmax(dim=1) == yb).float().sum().item()
        n += bsz

    return total_loss / n, total_acc / n


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    print("Using shards from:", SHARDS_DIR)
    shards = list_shards(SHARDS_DIR)
    train_shards, val_shards, test_shards = split_shards(shards, seed=SEED)

    print(f"Found {len(shards)} shards")
    print(f"Split (shards): train={len(train_shards)} val={len(val_shards)} test={len(test_shards)}")
    print("Example train shard:", train_shards[0])

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Datasets / Loaders
    train_ds = ShardStreamDataset(train_shards, shuffle_shards=True, shuffle_within=True, seed=SEED)
    val_ds = ShardStreamDataset(val_shards, shuffle_shards=False, shuffle_within=False, seed=SEED)
    test_ds = ShardStreamDataset(test_shards, shuffle_shards=False, shuffle_within=False, seed=SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_TRAIN,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_EVAL,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_EVAL,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=True,
    )

    # Model: use your existing ChessNet
    print("Initializing ChessNet model...")
    model = ChessNet().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Early stopping
    best_val = float("inf")
    best_state = None
    bad = 0

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            device,
            opt=opt,
            scaler=scaler,
            train=True,
            label_smoothing=LABEL_SMOOTHING,
        )
        va_loss, va_acc = run_epoch(
            model,
            val_loader,
            device,
            train=False,
            label_smoothing=0.0,
        )

        print(
            f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc = run_epoch(
        model,
        test_loader,
        device,
        train=False,
        label_smoothing=0.0,
    )
    print(f"TEST | loss {te_loss:.4f} acc {te_acc:.3f}")

    # Save checkpoint locally
    print("Saving best model locally to", MODEL_FILE)
    torch.save(model.state_dict(), MODEL_FILE)

    # Upload to Google Cloud Storage
    print(f"Uploading {MODEL_FILE} to bucket {BUCKET_NAME}...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"weights/{MODEL_FILE}")
    blob.upload_from_filename(MODEL_FILE)
    print("Upload complete!")


if __name__ == "__main__":
    main()
