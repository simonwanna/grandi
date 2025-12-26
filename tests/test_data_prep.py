import os
import sys

import chess
import numpy as np

# Add project root to path so we can import ml.data_prep
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ml.data_prep import get_board_vector, prepare_dataset


def test_board_vector_start_pos() -> None:
    """
    Test that the starting board position is encoded correctly using the 18-channel 8x8 representation.

    Planes 0-5:   White P, N, B, R, Q, K
    Planes 6-11:  Black P, N, B, R, Q, K
    Plane 12:     Side to moving (1=White, 0=Black)
    Planes 13-16: Castling Rights (White K, White Q, Black K, Black Q)
    Plane 17:     En Passant
    """
    board = chess.Board()
    vector = get_board_vector(board)

    # Check shape
    assert vector.shape == (18, 8, 8)

    # ----------------------------
    # Verify Piece Locations
    # ----------------------------

    # Plane 0: White Pawns at Rank 2 (index 6). a2 is (6, 0).

    # Let's check a2 (White Pawn) -> Plane 0
    # sq=8 (a2). 8//8=1 -> row 6. 8%8=0 -> col 0.
    assert vector[0, 6, 0] == 1.0, "Expected White Pawn at a2"

    # Plane 6: Black Pawns
    # Black Pawns on Rank 7.
    # a7 is sq 48. 48//8=6. r=7-6=1. c=0.
    assert vector[6, 1, 0] == 1.0, "Expected Black Pawn at a7"

    # Plane 3: White Root (a1)
    # a1 is sq 0. 0//8=0. r=7. c=0.
    assert vector[3, 7, 0] == 1.0, "Expected White Rook at a1"

    # Plane 9: Black Rook (a8)
    # a8 is sq 56. 56//8=7. r=0. c=0.
    assert vector[9, 0, 0] == 1.0, "Expected Black Rook at a8"

    # ----------------------------
    # Verify Metadata Planes
    # ----------------------------

    # Plane 12: Side to Move (White = 1.0)
    assert np.all(vector[12, :, :] == 1.0)

    # Planes 13-16: All Castling Rights True
    assert np.all(vector[13, :, :] == 1.0)  # White Kingside
    assert np.all(vector[14, :, :] == 1.0)  # White Queenside
    assert np.all(vector[15, :, :] == 1.0)  # Black Kingside
    assert np.all(vector[16, :, :] == 1.0)  # Black Queenside

    # Plane 17: No En Passant
    assert np.all(vector[17, :, :] == 0.0)


def test_board_vector_midgame() -> None:
    # 1. e4 (White moves Pawn e2->e4)
    board = chess.Board()
    board.push_san("e4")

    vector = get_board_vector(board)

    # Side to move now Black (0.0)
    assert np.all(vector[12, :, :] == 0.0)

    # En Passant Square (if e2-e4 allows capture on e3, logic checks plane 17 at file e)
    if board.ep_square:
        file = chess.square_file(board.ep_square)
        assert np.all(vector[17, :, file] == 1.0)


def test_prepare_dataset_pgn() -> None:
    """Test full PGN parsing flow."""
    # Realistic PGN from BigQuery (no headers, ends with result marker or *)
    dummy_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"

    # Sample rate 1.0 to get all positions

    dataset = prepare_dataset([(dummy_pgn, "white")], sample_rate=1.0)

    # Expect 6 positions: Start, After 1. e4, ..., After 3. Bb5.
    # (Last move a6 is skipped as sample happens before push).

    assert len(dataset) == 6

    # Check target (1.0 for white)
    # dataset[0] is (vector, target)
    assert dataset[0][1].item() == 1.0

    # Check tensor shape
    # (18, 8, 8)
    assert dataset[0][0].shape == (18, 8, 8)
