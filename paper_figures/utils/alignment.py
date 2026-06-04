"""
Sequence alignment utilities for paper figure visualisation.

hard_dtw_path_restricted  — restricted D3TW topology (diagonal + stay only),
                            matches the training Soft-DTW kernel.

needleman_wunsch_path     — full global NW alignment (diagonal, up, left),
                            allows gaps in either sequence.

Both are for VISUALISATION ONLY — they do not replace the Soft-DTW training loss.
"""
import numpy as np
import torch


def similarity_to_distance(similarity):
    """Convert similarity ∈ [-1, 1] to a non-negative distance."""
    if isinstance(similarity, torch.Tensor):
        return 1.0 - similarity
    return 1.0 - np.array(similarity, dtype=np.float32)


def hard_dtw_path_restricted(sim_matrix):
    """
    Compute the highest-similarity alignment path under the restricted D3TW topology.

    Valid transitions to (i, j):
      - diagonal: from (i-1, j-1)
      - stay:     from (i,   j-1)

    Args:
        sim_matrix: Tensor or ndarray [T, S] — text_len × image_seq_len.
                    Higher value = better match.

    Returns:
        path: list of (i, j) tuples ordered from (0, 0) to (T-1, S-1).
              Returns empty list if T > S (path is geometrically impossible).
    """
    if isinstance(sim_matrix, torch.Tensor):
        sim = sim_matrix.detach().cpu().float().numpy()
    else:
        sim = np.array(sim_matrix, dtype=np.float32)

    T, S = sim.shape
    if T > S:
        return []   # impossible under this topology

    NEG_INF = -1e18
    dp = np.full((T, S), NEG_INF, dtype=np.float64)
    # Backpointer: 0 = came from diagonal, 1 = came from stay
    bp = np.zeros((T, S), dtype=np.int8)

    dp[0, 0] = sim[0, 0]
    for j in range(1, S):          # first row: can only stay
        dp[0, j] = dp[0, j - 1] + sim[0, j]
        bp[0, j] = 1

    for i in range(1, T):
        for j in range(i, S):      # need at least i columns to reach row i
            diag = dp[i - 1, j - 1] if j > 0 else NEG_INF
            stay = dp[i, j - 1]    if j > 0 else NEG_INF
            if diag >= stay:
                dp[i, j] = diag + sim[i, j]
                bp[i, j] = 0
            else:
                dp[i, j] = stay + sim[i, j]
                bp[i, j] = 1

    # Trace back from (T-1, S-1)
    path = []
    i, j = T - 1, S - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if bp[i, j] == 0:   # diagonal
            i, j = i - 1, j - 1
        else:               # stay
            j = j - 1
    path.append((0, 0))
    path.reverse()
    return path


def needleman_wunsch_path(sim_matrix, gap_penalty=0.0):
    """
    Global sequence alignment via Needleman-Wunsch, maximising total similarity.

    Three allowed transitions at dp[i][j]:
      - diagonal (0): advance both sequences → text char i-1 matches window j-1
      - up       (1): advance text only      → gap in image sequence
      - left     (2): advance image only     → gap in text sequence

    Args:
        sim_matrix:   Tensor or ndarray [T, S].  Higher = better match.
        gap_penalty:  Score added per gap step (≤ 0).  0.0 = free gaps.

    Returns:
        path: list of (row, col) in sim_matrix coordinates, ordered (0,0) → (T-1,S-1).
              Diagonal steps produce (row, col) points; up/left steps produce vertical/
              horizontal segments, so the drawn line reflects the full alignment topology.
    """
    if isinstance(sim_matrix, torch.Tensor):
        sim = sim_matrix.detach().cpu().double().numpy()
    else:
        sim = np.array(sim_matrix, dtype=np.float64)

    T, S = sim.shape

    # ── DP table (1-indexed so borders are at row/col 0) ─────────────────────
    dp = np.zeros((T + 1, S + 1), dtype=np.float64)
    bt = np.zeros((T + 1, S + 1), dtype=np.int8)   # 0=diag 1=up 2=left

    for j in range(1, S + 1):
        dp[0, j] = j * gap_penalty
        bt[0, j] = 2
    for i in range(1, T + 1):
        dp[i, 0] = i * gap_penalty
        bt[i, 0] = 1

    for i in range(1, T + 1):
        for j in range(1, S + 1):
            diag = dp[i - 1, j - 1] + sim[i - 1, j - 1]
            up   = dp[i - 1, j]     + gap_penalty
            left = dp[i,     j - 1] + gap_penalty
            if diag >= up and diag >= left:
                dp[i, j] = diag; bt[i, j] = 0
            elif up >= left:
                dp[i, j] = up;   bt[i, j] = 1
            else:
                dp[i, j] = left; bt[i, j] = 2

    # ── Traceback: record every cell visited in sim-matrix coordinates ────────
    # Up   step: only row advances → vertical segment (row decreases, col fixed)
    # Left step: only col advances → horizontal segment (col decreases, row fixed)
    # Diagonal:  both advance      → diagonal step
    path = []
    i, j = T, S
    while i > 0 or j > 0:
        path.append((max(0, i - 1), max(0, j - 1)))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif bt[i, j] == 0:
            i -= 1; j -= 1
        elif bt[i, j] == 1:
            i -= 1          # gap in image: row decreases, col stays
        else:
            j -= 1          # gap in text:  col decreases, row stays

    path.reverse()
    return path


def path_to_char_window_mapping(path, text):
    """
    Convert a DTW path to a per-character → window-indices mapping.

    Args:
        path: list of (i, j) tuples (output of hard_dtw_path_restricted).
        text: str — the transcript.

    Returns:
        List of dicts:
            [{"char_index": i, "char": c, "window_indices": [j1, ...]}, ...]
    """
    from collections import defaultdict
    char_to_windows = defaultdict(list)
    for i, j in path:
        char_to_windows[i].append(j)

    return [
        {
            "char_index": i,
            "char": ch,
            "window_indices": sorted(char_to_windows.get(i, [])),
        }
        for i, ch in enumerate(text)
    ]
