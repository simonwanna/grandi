import os
import sys

import chess

# Add project root to path so we can import ml.data_prep
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ml.data_prep import get_board_vector, prepare_dataset


def test_board_vector_start_pos() -> None:
    """
    Test that the starting board position is encoded correctly.
    Rank 7 -> 0, File 0 -> 7
    Encoding: P=1, N=2, B=3, R=4, Q=5, K=6 (Black negative)
    """
    board = chess.Board()
    vector = get_board_vector(board)

    # Check length
    assert len(vector) == 64

    # Expected layout for Standard Chess Start:
    # Rank 8 (Index 0-7): Black Pieces -> r n b q k b n r -> -4 -2 -3 -5 -6 -3 -2 -4
    # Rank 7 (Index 8-15): Black Pawns -> -1 ...
    # ...
    # Rank 2 (Index 48-55): White Pawns -> 1 ...
    # Rank 1 (Index 56-63): White Pieces -> R N B Q K B N R -> 4 2 3 5 6 3 2 4

    # Check "top-left" (a8) - Black Rook
    assert vector[0] == -4.0, f"Expected Black Rook (-4.0) at a8, got {vector[0]}"

    # Check "top-right" (h8) - Black Rook
    assert vector[7] == -4.0

    # Check King positions
    # Black King e8 (File 4 in 0-idx) -> Index 4
    assert vector[4] == -6.0
    # White King e1 -> Index 60 (Row 7 * 8 + 4)
    assert vector[60] == 6.0

    # Check Pawns
    assert vector[8] == -1.0  # a7 Black Pawn
    assert vector[48] == 1.0  # a2 White Pawn

    # Check Empty square (e.g., e4 at start)
    assert vector[36] == 0.0


def test_prepare_dataset_pgn() -> None:
    """Test full PGN parsing flow."""
    # Realistic PGN from BigQuery (no headers, ends with result marker or *)
    dummy_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"

    dataset = prepare_dataset([(dummy_pgn, "white")], sample_rate=1.0)

    assert len(dataset) == 6

    # Check target (1.0 for white)
    # dataset[0] is (vector, target)
    assert dataset[0][1].item() == 1.0
