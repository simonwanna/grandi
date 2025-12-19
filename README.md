# grandi - the grandmaster's intuition :chess_pawn:
[![Pre-train and Deploy](https://github.com/simonwanna/grandi/actions/workflows/pre-train.yaml/badge.svg)](https://github.com/simonwanna/grandi/actions/workflows/pre-train.yaml)

A chess game where a neural network predicts the impact of your moves on the future game outcome - helping you become a better player.

### Tech Stack
* ♟️ **Lightweight Neural Network**: A PyTorch model trained to evaluate board positions.
* ♞ **Continuous Learning**: GitHub Actions workflow that runs tests, pre-trains base models, and fine-tunes weekly.
* ♜ **Cloud Ecosystem**: Scalable backend API on Cloud Run (backed by GCS). BigQuery handles fine-tuning data and observability.
* ♛ **Full Game**: Playable demo deployed on Hugging Face that saves gameplay logs for fine-tuning.
* ♚ **Grandmaster Insight**: Distills board configurations into a single "intuition" score.

### Dev
- After cloning the repo, run ``git submodule update --init --recursive`` to get the submodule.
- Then run `make install` to set up dependencies and hooks. 
- To run tests, use `make test`.