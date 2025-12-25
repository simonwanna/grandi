import os
import sys
from datetime import datetime

import torch
from google.cloud import storage

# Add the project root to the path to import from api and ml
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from api.model import ChessNet
    from ml.data_prep import fetch_data, prepare_dataset
except ImportError as e:
    raise ImportError(f"Could not import modules. Ensure PYTHONPATH is set. Error: {e}")


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
    # TODO: this might crash if we change model architecture, so we need to be careful with versioning
    # or just start from scratch with a new model
    print("Loading model...")
    model = ChessNet()
    state_dict = torch.load(MODEL_LOCAL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.train()  # Set to training mode

    # 5. Fetch Data
    # Fetch Yesterday's games (interval 1 day)
    # Returns list of (pgn, winner)
    print("Fetching training data from BigQuery...")
    data = fetch_data(PROJECT_ID, days_back=1)

    if not data:
        print("No games found for yesterday. Exiting without training.")
        return

    # 6. Prepare Dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(data, sample_rate=0.2)

    if len(dataset) < 10:
        print(f"Dataset too small ({len(dataset)} samples). Skipping training.")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # 7. Training Loop
    # Lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    print(f"Starting fine-tuning on {len(dataset)} positions...")
    # Train for a few epochs (e.g. 3-5) to adapt without overfitting/catastrophic forgetting
    EPOCHS = 3
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}: Avg Loss = {avg_loss:.4f}")

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
