from typing import Generator
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from web.utils import save_game_wrapper, start_game_log, update_game_move


# Mock logic dependencies
@pytest.fixture
def mock_request() -> MagicMock:
    request = MagicMock(spec=gr.Request)
    request.username = "test_user"
    return request


@pytest.fixture
def mock_save_dummy() -> Generator[MagicMock, None, None]:
    with patch("web.utils.save_dummy_game") as mock:
        mock.return_value = "Game Saved Successfully"
        yield mock


def test_start_game_log(mock_request: MagicMock) -> None:
    """Test that starting a game creates a valid state dictionary."""
    state, message = start_game_log(mock_request)

    # Check return structure
    assert isinstance(state, dict)
    assert "game_id" in state
    assert state["user"] == "test_user"
    assert state["moves"] == []
    assert state["status"] == "active"

    # Check message
    assert state["game_id"] in message
    assert "test_user" in message


def test_update_game_move() -> None:
    """Test that updating a game appends moves to the state and keeps the id."""
    initial_state = {"game_id": "constant-id-123", "user": "test_player", "moves": ["e4"], "status": "active"}

    # Make a move
    new_state, msg = update_game_move(initial_state, "e5")

    # ID should be CONSTANT
    assert new_state["game_id"] == "constant-id-123"
    assert len(new_state["moves"]) == 2
    assert new_state["moves"] == ["e4", "e5"]
    assert "e5 played" in msg


def test_save_game_wrapper_success(mock_request: MagicMock, mock_save_dummy: MagicMock) -> None:
    """Test saving a game successfully with matching user."""
    # Create a valid state
    state = {"game_id": "unique-id-123", "user": "test_user", "moves": ["e4", "e5"], "status": "active"}

    result = save_game_wrapper(state, mock_request)

    # Assertions
    assert result == "Game Saved Successfully"
    mock_save_dummy.assert_called_once_with(username="test_user", game_id="unique-id-123")


def test_save_game_wrapper_mismatch(mock_request: MagicMock, mock_save_dummy: MagicMock) -> None:
    """Test that saving fails if the user in state doesn't match the requester."""
    # State owned by someone else
    state = {"game_id": "unique-id-999", "user": "evil_hacker", "moves": [], "status": "active"}

    # mock_request.username is "test_user" -> mismatch
    result = save_game_wrapper(state, mock_request)

    assert "Error: Session mismatch" in result
    mock_save_dummy.assert_not_called()


def test_save_game_wrapper_no_state(mock_request: MagicMock, mock_save_dummy: MagicMock) -> None:
    """Test saving with empty state."""
    result = save_game_wrapper(None, mock_request)
    assert "Error: No active game found" in result
    mock_save_dummy.assert_not_called()
