import os
from contextlib import asynccontextmanager
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from google.cloud import storage
from model import ChessNet

MODEL_PATH = "/tmp/model.pth"

model_dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    # --- STARTUP ---
    try:
        # Connect to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.environ["BUCKET_NAME"])

        # Find the latest file (sort by creation time)
        blobs = list(bucket.list_blobs(prefix="weights/"))
        if not blobs:
            raise FileNotFoundError("No weights found in GCS bucket.")

        latest_blob = max(blobs, key=lambda b: b.time_created)
        latest_blob.download_to_filename(MODEL_PATH)

        # --- LOAD MODEL ---
        model_structure = ChessNet()
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model_structure.load_state_dict(state_dict)
        model_structure.eval()

        model_dict["model"] = model_structure
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e

    yield  # Keep the app running (freezed if not used)

    # --- SHUTDOWN ---
    model_dict.clear()


app = FastAPI(lifespan=lifespan)


# The predict endpoint requires IAM authentication
@app.post("/predict")
def predict(board_state: List[List[List[float]]]) -> dict:
    """
    Predict win probability for side-to-move.

    Request body:
        board_state: nested list with shape (18, 8, 8)
                     encoding the position as in training.

    Response:
        { "win_prob": float }  # probability that side-to-move eventually wins
    """
    model = model_dict["model"]

    # Convert JSON -> tensor
    x = torch.tensor(board_state, dtype=torch.float32)  # (18, 8, 8)

    if x.shape != (18, 8, 8):
        raise ValueError(f"Expected input shape (18, 8, 8), got {tuple(x.shape)}")

    # Add batch dimension -> (1, 18, 8, 8)
    x = x.unsqueeze(0)

    with torch.no_grad():
        logits = model(x)  # (1, 2)
        probs = F.softmax(logits, dim=1)  # (1, 2)
        win_prob = probs[0, 1].item()  # class 1 = win

    return {"win_prob": win_prob}
