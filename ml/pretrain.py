import os
import sys

import torch
from google.cloud import storage

# Add the project root to the path so we can import from api
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from api.model import ChessNet
except ModuleNotFoundError:
    raise ImportError("Could not import ChessNet from api.model")


def main() -> None:
    MODEL_FILE = "model-resnet-base.pth"
    BUCKET_NAME = os.environ.get("BUCKET_NAME")

    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME environment variable not set")

    print("Initializing model...")
    # Create model
    model = ChessNet()

    # TODO: change to download Lichess dataset and preprocess
    # Create dummy dataset
    print("Creating dataset...")
    inputs = torch.randn(100, 64)
    targets = torch.randint(0, 2, (100,)).float().view(-1, 1)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    # TODO: change pretraining logic to the one we used
    print("Starting training...")
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    # TODO: if there already is a model ckpt in the repo, load that instead of pretraining from scratch ...
    # Should only deploy this then ...
    print("Saving model...")
    torch.save(model.state_dict(), MODEL_FILE)

    print(f"Uploading {MODEL_FILE} to bucket {BUCKET_NAME}...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"weights/{MODEL_FILE}")
    blob.upload_from_filename(MODEL_FILE)

    print("Upload complete!")


if __name__ == "__main__":
    main()
