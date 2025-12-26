import io
import os
import urllib.request

import chess
import chess.pgn
import numpy as np
import zstandard as zstd
from tqdm import tqdm

# ---------------------------
# Config: output on Desktop
# ---------------------------
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUT_DIR = os.path.join(DESKTOP, "shards")
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving shards to:", OUT_DIR)

# ---------------------------
# Download (optional)
# ---------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

url = "https://database.lichess.org/standard/lichess_db_standard_rated_2016-03.pgn.zst"
zst_path = os.path.join(DATA_DIR, "lichess_2016-03.pgn.zst")


if not os.path.exists(zst_path):
    print("Downloading:", url)
    urllib.request.urlretrieve(url, zst_path)

print("PGN.zst path:", zst_path)

# ---------------------------
# Board encoding (18, 8, 8)
# ---------------------------
PIECE_TO_PLANE = {
    (chess.PAWN, True): 0,
    (chess.KNIGHT, True): 1,
    (chess.BISHOP, True): 2,
    (chess.ROOK, True): 3,
    (chess.QUEEN, True): 4,
    (chess.KING, True): 5,
    (chess.PAWN, False): 6,
    (chess.KNIGHT, False): 7,
    (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9,
    (chess.QUEEN, False): 10,
    (chess.KING, False): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    x = np.zeros((18, 8, 8), dtype=np.float32)

    for sq, pc in board.piece_map().items():
        plane = PIECE_TO_PLANE[(pc.piece_type, pc.color)]
        r = 7 - (sq // 8)
        c = sq % 8
        x[plane, r, c] = 1.0

    if board.turn == chess.WHITE:
        x[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        x[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        x[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        x[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        x[16, :, :] = 1.0

    if board.ep_square is not None:
        x[17, :, chess.square_file(board.ep_square)] = 1.0

    return x


# ---------------------------
# Binary result mapping
# ---------------------------
def result_to_binary(result_str: str) -> int | None:
    """
    Return:
      1 if White won ("1-0")
      0 if Black won ("0-1")
      None otherwise (draw, unknown, aborted)
    """
    if result_str == "1-0":
        return 1
    if result_str == "0-1":
        return 0
    return None


# ---------------------------
# Shard writer (binary labels)
# ---------------------------
def make_binary_value_shards(
    zst_path: str,
    out_dir: str,
    shard_size: int = 10_000,
    max_shards: int = 150,
    max_plies_per_game: int = 120,
    min_ply_to_start: int = 20,
    positions_per_game: int = 12,  # more positions per game
    flip_to_move: bool = True,  # label from side-to-move perspective
    rng_seed: int = 42,
    max_games: int | None = None,
) -> None:
    """
    Writes shard_XXXX.npz with:
      X: (N, 18, 8, 8) float32
      y: (N,) int64 in {0,1} where:
         - if flip_to_move=True: y=1 means side-to-move eventually wins, y=0 loses
         - if flip_to_move=False: y=1 means White eventually wins, y=0 means Black wins

    Draws are removed entirely.
    """
    os.makedirs(out_dir, exist_ok=True)
    X_buf, y_buf = [], []
    shard_id = 0
    games_seen = 0

    rng = np.random.default_rng(rng_seed)

    with open(zst_path, "rb") as fh:
        stream = zstd.ZstdDecompressor().stream_reader(fh)
        text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")

        pbar = tqdm(desc="Extracting positions", unit="pos")
        while shard_id < max_shards:
            if max_games is not None and games_seen >= max_games:
                break

            game = chess.pgn.read_game(text)
            if game is None:
                break
            games_seen += 1

            h = game.headers

            # Only keep decisive games
            y_whitewin = result_to_binary(h.get("Result", ""))
            if y_whitewin is None:
                continue

            # Standard only
            if h.get("Variant", "Standard") != "Standard":
                continue

            moves = list(game.mainline_moves())
            end = min(len(moves), max_plies_per_game)
            start = min_ply_to_start
            if end <= start:
                continue

            candidates = np.arange(start, end, dtype=np.int32)
            k = min(int(positions_per_game), len(candidates))
            if k <= 0:
                continue
            sampled = set(rng.choice(candidates, size=k, replace=False).tolist())

            board = game.board()
            for ply, mv in enumerate(moves[:end]):
                if ply in sampled:
                    X_buf.append(board_to_tensor(board))

                    # base label: 1 if White won else 0
                    label = y_whitewin

                    # if we want side-to-move perspective, flip when black to move
                    # (because "white won" is a loss from black-to-move perspective)
                    if flip_to_move and board.turn == chess.BLACK:
                        label = 1 - label

                    y_buf.append(label)
                    pbar.update(1)

                    if len(y_buf) >= shard_size:
                        X = np.stack(X_buf, axis=0)
                        y = np.array(y_buf, dtype=np.int64)
                        out_path = os.path.join(out_dir, f"shard_{shard_id:04d}.npz")
                        np.savez_compressed(out_path, X=X, y=y)
                        print("Wrote", out_path, X.shape)

                        X_buf, y_buf = [], []
                        shard_id += 1
                        if shard_id >= max_shards:
                            break

                board.push(mv)

        pbar.close()

    # final partial shard
    if len(y_buf) > 0 and shard_id < max_shards:
        X = np.stack(X_buf, axis=0)
        y = np.array(y_buf, dtype=np.int64)
        out_path = os.path.join(out_dir, f"shard_{shard_id:04d}.npz")
        np.savez_compressed(out_path, X=X, y=y)
        print("Wrote (final partial)", out_path, X.shape)

    print(f"Done. games_seen={games_seen}, shards_written={shard_id + (1 if len(y_buf) > 0 else 0)}")


# ---------------------------
# Run
# ---------------------------
make_binary_value_shards(
    zst_path,
    out_dir=OUT_DIR,
    shard_size=50_000,  # bigger shards
    max_shards=200,  # ~10M positions
    positions_per_game=10,
    min_ply_to_start=50,
    max_plies_per_game=120,
    flip_to_move=True,
)
