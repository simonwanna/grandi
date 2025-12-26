import os
import sys
from datetime import datetime

import torch
from google.cloud import storage

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from api.model import ChessNet
from ml.data_prep import fetch_data, prepare_dataset


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer | None = None,
    train: bool = True,
    label_smoothing: float = 0.0,
) -> tuple[float, float]:
    """
    Generic train/eval loop, similar to pretrain.run_epoch.

    Assumes:
      - model(Xb) -> logits of shape (B, 2)
      - yb are integer class labels {0,1}
    """
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for Xb, yb in tqdm(loader, leave=False):
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).squeeze(1).long()

        logits = model(Xb)
        loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        bsz = Xb.size(0)
        total_loss += loss.item() * bsz
        total_acc += (logits.argmax(dim=1) == yb).float().sum().item()
        n += bsz

    return total_loss / n, total_acc / n


def main() -> None:
    # 1. Configuration
    # Load .env variables if present
    from dotenv import load_dotenv

    load_dotenv()

    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    PROJECT_ID = os.environ.get("GCP_PROJECT_ID")

    if not BUCKET_NAME or not PROJECT_ID:
        raise ValueError("Environment variables BUCKET_NAME and GCP_PROJECT_ID must be set.")

    MODEL_LOCAL_PATH = "latest_model.pth"

    # 2. Setup GCS
    print(f"Connecting to GCS Bucket: {BUCKET_NAME}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # 3. Download Latest Model
    print("Fetching latest model weights...")
    blobs = list(bucket.list_blobs(prefix="weights/"))
    if not blobs:
        raise FileNotFoundError("No weights found in GCS bucket to fine-tune from.")

    latest_blob = max(blobs, key=lambda b: b.time_created)
    print(f"Downloading {latest_blob.name}...")
    latest_blob.download_to_filename(MODEL_LOCAL_PATH)

    # 4. Load Model
    print("Loading model...")
    model = ChessNet()

    state_dict = torch.load(MODEL_LOCAL_PATH, map_location=torch.device("cpu"))

    if "model_state_dict" in state_dict:
        print("Detected wrapped checkpoint. Unwrapping 'model_state_dict'...")
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)

    # Freeze base layers (transfer learning)
    print("Freezing base layers (stem + blocks)...")
    for param in model.stem.parameters():
        param.requires_grad = False
    for param in model.blocks.parameters():
        param.requires_grad = False

    # Device + AMP scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # 5. Fetch Data
    # Returns list of (pgn, winner)
    print("Fetching training data from BigQuery...")
    data = fetch_data(PROJECT_ID, days_back=7)

    if not data:
        print("No games found for yesterday. Exiting without training.")
        return

    # 6. Prepare Dataset and DataLoader
    print("Preparing dataset...")
    dataset = prepare_dataset(data, sample_rate=0.2)

    if len(dataset) < 10:
        print(f"Dataset too small ({len(dataset)} samples). Skipping training.")
        return

    BATCH_SIZE = 32
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    LEARNING_RATE = 1e-6  # lower LR for fine-tuning
    EPOCHS = 3

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 7. Training Loop (fine-tune with lower LR, few epochs)
    print(f"Starting fine-tuning on {len(dataset)} positions...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(
            model,
            dataloader,
            device,
            opt=optimizer,
            train=True,
            label_smoothing=0.0,  # can enable if you like
        )
        print(f"Epoch {epoch}/{EPOCHS}: loss {tr_loss:.4f} | acc {tr_acc:.3f}")

    # 8. Save and Upload
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_model_filename = f"model-{timestamp}.pth"

    print(f"Saving model to {new_model_filename}...")
    torch.save(model.state_dict(), new_model_filename)

    print(f"Uploading to GCS: weights/{new_model_filename}...")
    blob = bucket.blob(f"weights/{new_model_filename}")
    blob.upload_from_filename(new_model_filename)

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()
