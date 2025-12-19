import chess

from web.game_engine import ChessGame


def test_chess_game_init() -> None:
    game = ChessGame()
    assert game.board.fen() == chess.STARTING_FEN
    assert game.move_history == []


def test_make_move_valid() -> None:
    """Test making a valid move using Standard Algebraic Notation (SAN)."""
    game = ChessGame()
    success, msg = game.make_move("e4")
    assert success
    assert "e4" in msg


def test_get_computer_move_stockfish() -> None:
    """Test that the Stockfish bot makes a valid move."""
    game = ChessGame()
    move, msg = game.get_computer_move(skill_level=1)

    assert move is not None
    # Check that the move is actually in the history
    assert len(game.move_history) == 1
    assert "Stockfish" in msg
    # The move should be a valid SAN string (e.g. "e4", "Nf3")
    assert isinstance(move, str)
    assert len(move) >= 2


def test_make_move_invalid() -> None:
    """Test making an invalid move."""
    game = ChessGame()
    success, msg = game.make_move("e5")  # Pawn can't jump
    assert not success
    success, msg = game.make_move("e2e11")  # Invalid move
    assert not success


def test_pgn_generation() -> None:
    """Test PGN generation."""
    game = ChessGame()
    game.make_move("e4")
    game.make_move("e5")
    pgn = game.get_pgn()
    assert "1. e4 e5" in pgn
    assert 'Result "*"' in pgn
