import os
from contextlib import asynccontextmanager
from typing import List

import torch
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
def predict(board_state: List[float]) -> dict:
    model = model_dict["model"]
    board_state = torch.tensor(board_state, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(board_state)

    return {"win_prob": prediction.item()}
