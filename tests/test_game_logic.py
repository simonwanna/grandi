from typing import Generator
from unittest.mock import MagicMock, patch

import chess
import gradio as gr
import pytest
from game_engine import ChessGame

from web.utils import handle_click, offer_draw, start_new_game


@pytest.fixture
def mock_request() -> MagicMock:
    req = MagicMock(spec=gr.Request)
    req.username = "test_player"
    return req


@pytest.fixture
def mock_api() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    with (
        patch("web.utils.get_intuition_score", return_value=0.5) as mock_score,
        patch("web.utils.save_game", return_value="Saved") as mock_save,
    ):
        yield mock_score, mock_save


def test_start_new_game(mock_request: MagicMock, mock_api: tuple[MagicMock, MagicMock]) -> None:
    """Test start new game. Mocks API calls."""
    game, svg_path, msg, logs = start_new_game("White", 1, mock_request)
    assert isinstance(game, ChessGame)
    assert game.user == "test_player"
    assert "started" in msg.lower()
    assert svg_path.endswith(".svg")
    assert logs[0] == ["Start", "0.5"]


def test_handle_click_select(mock_api: tuple[MagicMock, MagicMock]) -> None:
    """Test handle click select. The click is simulated by coordinates."""
    game = ChessGame()
    # Click e2 (White Pawn)
    # Coords: 600x600. e2 is:
    # col 4 (e) -> x around 300-375
    # center of e2: x=337, y=487 (approx)
    file_e_x = 337
    rank_2_y = 487

    evt = MagicMock(spec=gr.SelectData)
    evt.index = [file_e_x, rank_2_y]

    # Action
    game, svg, msg, logs = handle_click(game, evt)

    assert game.selected_square == chess.E2
    assert "Selected e2" in msg


def test_handle_click_move(mock_api: tuple[MagicMock, MagicMock]) -> None:
    game = ChessGame()
    # 1. Select e2
    game.selected_square = chess.E2

    # 2. Click e4
    # Rank 4 is row 3. visual y index: 7-3=4? No.
    # Rank 4: 0(R8), 1(R7), 2(R6), 3(R5), 4(R4). Yes. Row 4.
    # Center y = 4.5 * 75 = 337.5
    # Center x (e) = 4.5 * 75 = 337.5
    evt = MagicMock(spec=gr.SelectData)
    evt.index = [337, 337]

    game, svg, msg, logs = handle_click(game, evt)

    assert game.board.piece_at(chess.E4).symbol() == "P"
    assert game.board.piece_at(chess.E4).symbol() == "P"
    # Moved to SAN logging: "You moved e4"
    assert "moved e4" in msg.lower()

    # Check logs
    assert len(game.logs) > 0
    # Should contain "You: e4"
    assert "You: e4" in game.logs[0][0]


def test_offer_draw_saves_game(mock_api: tuple[MagicMock, MagicMock]) -> None:
    """Test that offering a draw calls the save logic."""
    mock_score, mock_save = mock_api

    game = ChessGame()
    game.user = "test_player"
    game.game_id = "test-game-id"
    # Make some moves
    game.make_move("d4")  # White
    # Assume Black didn't move or mocked engine did?
    # In this test scope, engine is not auto-mocked unless we mock get_computer_move.
    # Let's just create a state.

    game, svg, msg, logs = offer_draw(game)

    assert "Draw agreed" in msg
    mock_save.assert_called_once()

    # Verify args
    call_kwargs = mock_save.call_args.kwargs
    assert call_kwargs["winner"] == "draw"
    assert call_kwargs["white"] == "test_player"
    assert call_kwargs["game_id"] == "test-game-id"
    # PGN should contain the move
    assert "d4" in call_kwargs["pgn"]
