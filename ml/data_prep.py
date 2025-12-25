import random
from typing import List, Tuple

import chess.pgn
import torch
from google.cloud import bigquery
from torch.utils.data import TensorDataset


# TODO: update to match the 18-channel encoding used in the model
def get_board_vector(board: chess.Board) -> List[float]:
    """
    Convert board to 64-element float vector matching web/logic.py encoding.
    Scans Rank 7->0, File 0->7.
    Values: {-6..6} (P=1, N=2, B=3, R=4, Q=5, K=6).
    """
    vector = []
    # Piece values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    # Iterate Ranks 7 (top/8) down to 0 (bottom/1)
    for rank in range(7, -1, -1):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                val = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.BLACK:
                    val = -val
                vector.append(float(val))
            else:
                vector.append(0.0)
    return vector


def fetch_data(project_id: str, days_back: int = 1, table_id: str = "chess_data.games") -> List[Tuple[str, str]]:
    """
    Fetch PGNs and Winners from BigQuery for a specific date range.
    Returns list of (pgn_string, winner_string).
    """
    client = bigquery.Client(project=project_id)

    # Standard: Fetch games from Yesterday (DATE_SUB(CURRENT_DATE(), INTERVAL days_back DAY))
    query = f"""
        SELECT pgn, winner 
        FROM `{table_id}` 
        WHERE DATE(TIMESTAMP(timestamp)) = DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
        AND pgn IS NOT NULL
        AND pgn != ''
    """

    query_job = client.query(query)
    results = query_job.result()

    data = [(row.pgn, row.winner) for row in results]
    print(f"Fetched {len(data)} games from BigQuery ({days_back} day(s) ago).")
    return data


# TODO: this need to change such that it returns features instead of board vectors
def prepare_dataset(data: List[Tuple[str, str]], sample_rate: float = 0.2) -> TensorDataset:
    """
    Parse PGNs into board states and targets.
    data: List of (pgn_string, winner_string)
    sample_rate: Percentage of positions to keep (0.0 - 1.0).
    """
    inputs = []
    targets = []

    import io

    for pgn_text, winner in data:
        # Map Winner String -> Target Float
        target_val = None
        if winner == "white":
            target_val = 1.0
        elif winner == "black":
            target_val = 0.0
        else:
            continue  # Skip draws or invalid

        # PGN might contain multiple games, but typically one per row
        pgn_io = io.StringIO(pgn_text)

        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break

            board = game.board()

            # Iterate through moves
            for move in game.mainline_moves():
                # Current board state -> target.

                # Random sampling to avoid correlation and bloat
                if random.random() < sample_rate:
                    # TODO: change here
                    vector = get_board_vector(board)
                    inputs.append(vector)
                    targets.append(target_val)

                board.push(move)

    if not inputs:
        print("Warning: No positions extracted. Returning empty dataset.")
        # Return dummy row dataset to avoid crash
        return TensorDataset(torch.randn(1, 64), torch.tensor([0.5]))

    print(f"Dataset created: {len(inputs)} positions.")

    tensor_x = torch.tensor(inputs, dtype=torch.float32)
    tensor_y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    return TensorDataset(tensor_x, tensor_y)


if __name__ == "__main__":
    # Simple manual test if run directly
    print("Running Data Prep Test...")
    dummy_pgn = "1. e4 e5 2. Nf3 d6 3. d4 Nf6 4. Nc3 Nbd7 5. Bd3 Be7 6. Be3 exd4 7. Bxd4 c5 \
        8. Be3 O-O 9. O-O a6 10. a3 h6 11. h4 Re8 12. Re1 Ng4 13. Kf1 Bxh4 14. g3 Bf6 \
            15. Bf4 g5 16. Bd2 b5 17. a4 b4 18. Na2 a5 19. b3 Bb7 20. c3 Rb8 21. cxb4 cxb4 \
                22. Nxb4 axb4 23. Bxb4 Nde5 24. Nxe5 Nxe5 25. Be2 Re6 26. f4 gxf4 27. gxf4 Kh8 \
                    28. Kg1 Nc6 29. Bd2 Qb6+ 30. Kf1 Rg8 31. Bf3 Rg1+ 32. Ke2 Ba6# *"
    dummy_data = [(dummy_pgn, "white")]
    vectors = prepare_dataset(dummy_data, sample_rate=0.5)  # 64 positions total
    print(f"Extracted {len(vectors)} positions from dummy game.")
