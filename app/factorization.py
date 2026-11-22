"""Constrained low-rank merging of two LoRA adapters for one linear layer.

The original update is ``W = B1 @ A1 + B2 @ A2``.  A single adapter ``(B, A)``
with shared inner dimension ``<= rank_limit`` must reproduce ``W`` exactly on a
selected set of activations ``X``::

    B @ (A @ X) == W @ X

and, among all factorisations satisfying that hard constraint, minimise
``||B @ A - W||_F``.

Optimal construction
--------------------
Let ``P`` be an orthonormal basis of ``span(X)`` and ``U1`` an orthonormal
basis of ``col(W P)`` (dimension ``k = rank(W X)``).

* The constraint is equivalent to ``(B A) P = W P``.  It fixes the action of
  ``BA`` on ``span(X)`` and leaves ``span(X)^perp`` free.
* Any feasible ``BA`` has rank at least ``k`` (because
  ``rank(M P) <= rank(M)``), so ``rho = rank_limit - k`` extra dimensions
  remain.  Writing ``M = M P P^T + M (I - P P^T)``, the Frobenius objective
  splits orthogonally into a forced part and a free part.  Choosing the
  ``U1``-row block of the free part optimally gives ``U1 U1^T W``; the rest is
  a rank-``rho`` approximation of
  ``R = (I - U1 U1^T) W (I - P P^T)``, whose best solution is the truncated
  SVD ``U2 S2 V2^T`` (Eckart-Young).  Hence::

      M = U1 U1^T W + U2 U2^T W (I - P P^T)

  with ``M P = W P`` and ``rank(M) <= k + rho = rank_limit``.
* A factorisation is ``B = [U1 | U2]`` and
  ``A = [U1^T W ; U2^T W (I - P P^T)]``.

If ``rank_limit < k`` the constraint is infeasible; ``k`` is the minimum
feasible rank and is reported to the caller.

Numerical scaling
-----------------
When protected activations (or their responses) differ wildly in magnitude, a
single relative cutoff on the singular values of ``W X`` can hide the small
columns: their independent directions would be dropped and the minimum
feasible rank underreported.  Two measures make the decision reliable:

* every matrix entering a rank/basis decision is **column-equilibrated**
  first (each column is scaled to unit norm); scaling a column of ``X`` or
  ``W X`` does not change the underlying column space, so the minimum
  feasible rank is invariant under any nonzero rescaling of a single
  activation;
* singular values are classified against two relative cutoffs: below
  :data:`RANK_ZERO_RTOL` they are numerical zero, at or above
  :data:`RANK_TRUST_RTOL` they are trustworthy signal.  A singular value in
  the gap between the two means the rank cannot be decided reliably; the
  merge then raises :class:`RankUndecidableError` instead of guessing.

Response verification is performed **per protected column** with an L2 bound
of :data:`RESPONSE_COL_RTOL` times the larger of that column's input and
target response L2 norms, so a large column can no longer mask a failure on a
small one.
"""

from __future__ import annotations

import numpy as np

# A singular value below RANK_ZERO_RTOL * sigma_max (on the column-equilibrated
# matrix) is treated as numerical zero.
RANK_ZERO_RTOL = 1e-12
# A singular value at or above RANK_TRUST_RTOL * sigma_max is trustworthy
# signal.  Values in between leave the rank undecidable.
RANK_TRUST_RTOL = 1e-9
# Per-column L2 tolerance for response preservation.
RESPONSE_COL_RTOL = 1e-9
MAX_LAYER_DIM = 32
MAX_ADAPTER_RANK = 1024
MAX_ACTIVATION_VECTORS = 2048


class MergeError(ValueError):
    """Merge error with a message safe to show to the user."""


class ShapeError(MergeError):
    pass


class DataError(MergeError):
    pass


class InfeasibleRankError(MergeError):
    def __init__(self, requested_rank: int, min_rank: int):
        self.requested_rank = int(requested_rank)
        self.min_rank = int(min_rank)
        super().__init__(
            f"秩上限 {requested_rank} 过小：在当前保护项下最低可行秩为 {min_rank}"
        )


class RankUndecidableError(MergeError):
    """Raised when the numerical rank cannot be decided reliably.

    The singular spectrum of a column-equilibrated matrix contains a singular
    value in the grey zone between the zero cutoff and the trust cutoff, so
    neither dropping nor keeping the corresponding direction can be justified
    numerically.  The merge is refused (and any existing package is kept).
    """

    def __init__(self, what: str, grey_values, sigma_max: float):
        self.what = str(what)
        self.grey_values = [float(v) for v in grey_values]
        self.sigma_max = float(sigma_max)
        detail = ", ".join(f"{v:.3e}" for v in self.grey_values)
        super().__init__(
            f"{what}的数值秩无法可靠判定：列均衡后最大奇异值为 {sigma_max:.3e}，"
            f"而奇异值 {detail} 落在 {RANK_ZERO_RTOL:.0e}·σ_max 与 "
            f"{RANK_TRUST_RTOL:.0e}·σ_max 的灰区内，无法确认对应方向应当"
            "保留还是舍弃。请调整保护激活（避免近乎共线的向量）后重试；"
            "本次未生成新下载包，原有下载包保持不变。"
        )


def _matrix(name: str, value) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataError(f"{name} 不是数值矩阵") from exc
    if arr.ndim != 2:
        raise ShapeError(f"{name} 必须是二维矩阵，实际为 {arr.ndim} 维")
    if arr.size and not np.all(np.isfinite(arr)):
        raise DataError(f"{name} 含有 NaN 或无穷值")
    return arr


def validate_adapters(B1, A1, B2, A2):
    """Validate the two adapters and return them as float64 matrices."""
    B1 = _matrix("适配器 1 的 B", B1)
    A1 = _matrix("适配器 1 的 A", A1)
    B2 = _matrix("适配器 2 的 B", B2)
    A2 = _matrix("适配器 2 的 A", A2)

    for tag, B, A in (("1", B1, A1), ("2", B2, A2)):
        if B.shape[1] != A.shape[0]:
            raise ShapeError(
                f"适配器 {tag} 的因子不匹配：B 有 {B.shape[1]} 个秩列，"
                f"A 有 {A.shape[0]} 个秩行（约定 B 为 out×r、A 为 r×in）"
            )
        if not (1 <= B.shape[1] <= MAX_ADAPTER_RANK):
            raise ShapeError(
                f"适配器 {tag} 的秩必须位于 1..{MAX_ADAPTER_RANK}，实际为 {B.shape[1]}"
            )

    if B1.shape[0] != B2.shape[0] or A1.shape[1] != A2.shape[1]:
        raise ShapeError(
            "两份适配器目标层不一致："
            f"适配器 1 作用于 {A1.shape[1]}→{B1.shape[0]}，"
            f"适配器 2 作用于 {A2.shape[1]}→{B2.shape[0]}；"
            "两个因子需分别满足 B 为 out×r、A 为 r×in"
        )

    out_dim, in_dim = B1.shape[0], A1.shape[1]
    if not (1 <= in_dim <= MAX_LAYER_DIM) or not (1 <= out_dim <= MAX_LAYER_DIM):
        raise ShapeError(
            f"层的输入/输出维度必须各在 1..{MAX_LAYER_DIM} 之间，"
            f"实际为 input={in_dim}, output={out_dim}"
        )
    return B1, A1, B2, A2


def validate_activations(X, in_dim: int) -> np.ndarray:
    X = _matrix("激活矩阵 X", X)
    if X.shape[0] != in_dim:
        raise ShapeError(
            f"激活矩阵的行数必须等于层输入维度 {in_dim}，实际为 {X.shape[0]}；"
            "约定每个激活向量为一列"
        )
    if X.shape[1] == 0:
        raise ShapeError("激活矩阵至少需要一列激活向量")
    if X.shape[1] > MAX_ACTIVATION_VECTORS:
        raise ShapeError(
            f"激活向量过多（最多 {MAX_ACTIVATION_VECTORS} 列），实际为 {X.shape[1]}"
        )
    return X


def combine_adapters(B1, A1, B2, A2) -> np.ndarray:
    B1, A1, B2, A2 = validate_adapters(B1, A1, B2, A2)
    return B1 @ A1 + B2 @ A2


def _column_equilibrated(M: np.ndarray) -> np.ndarray:
    """Scale each nonzero column of ``M`` to unit L2 norm.

    Column rescaling is an invertible operation on the column space, so the
    rank and the orthonormal basis of :func:`numpy.linalg.svd` are unchanged
    in exact arithmetic; in finite arithmetic the equilibration prevents a
    small-magnitude column from being hidden behind a large one.
    """
    if M.size == 0 or M.shape[1] == 0:
        return M
    norms = np.linalg.norm(M, axis=0)
    safe = np.where(norms > 0.0, norms, 1.0)
    return M / safe


def _svd_rank_decision(what: str, M: np.ndarray) -> tuple[np.ndarray, int]:
    """SVD of a column-equilibrated matrix with a reliable rank decision.

    Returns ``(U, rank)`` with ``U`` the left singular vectors spanning the
    decided rank.  Singular values below the zero cutoff are dropped; values
    at or above the trust cutoff are kept.  A singular value in the grey zone
    makes the rank undecidable and raises :class:`RankUndecidableError`.
    """
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0], 0)), 0
    Meq = _column_equilibrated(M)
    U, s, _ = np.linalg.svd(Meq, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros((M.shape[0], 0)), 0
    sigma_max = float(s[0])
    zero_mask = s < RANK_ZERO_RTOL * sigma_max
    trust_mask = s >= RANK_TRUST_RTOL * sigma_max
    grey = s[(~zero_mask) & (~trust_mask)]
    if grey.size:
        raise RankUndecidableError(what, grey, sigma_max)
    rank = int(np.count_nonzero(trust_mask))
    if rank == 0:
        return np.zeros((M.shape[0], 0)), 0
    return U[:, :rank], rank


def matrix_rank(M: np.ndarray) -> int:
    """Numerical rank of ``M`` with column-equilibrated singular values.

    Raises :class:`RankUndecidableError` when the rank cannot be decided
    reliably instead of returning a guess.
    """
    if M.size == 0:
        return 0
    _, rank = _svd_rank_decision("矩阵", M)
    return rank


def minimum_feasible_rank(W: np.ndarray, X_selected: np.ndarray) -> int:
    """Smallest shared dimension able to reproduce ``W @ X_selected``.

    ``rank(B A X) <= rank(B A)`` and ``B A X = W X``, hence
    ``rank(B A) >= rank(W X)``.  The bound is attained by the construction in
    :func:`factorize_update`.  The decision is scale-invariant: any nonzero
    rescaling of a column of ``X_selected`` leaves the result unchanged.
    """
    if X_selected.size == 0:
        return 0
    return matrix_rank(W @ X_selected)


def _k_and_bases(W, X_selected):
    # P spans the (equilibrated) activation directions.  Grey-zone directions
    # of X itself do not change the minimum feasible rank; keeping them in P
    # only enlarges the protected subspace, so P uses the strict zero cutoff
    # without raising.  The response basis U1 drives the reported rank and
    # therefore goes through the strict rank decision.
    P_eq = _column_equilibrated(X_selected)
    if P_eq.shape[1]:
        Up, sp, _ = np.linalg.svd(P_eq, full_matrices=False)
        p_rank = (
            int(np.count_nonzero(sp >= RANK_ZERO_RTOL * sp[0]))
            if sp.size and sp[0] > 0.0
            else 0
        )
        P = Up[:, :p_rank] if p_rank else np.zeros((X_selected.shape[0], 0))
    else:
        P = np.zeros((X_selected.shape[0], 0))
    WX = W @ X_selected
    U1, k = _svd_rank_decision("保护响应 W·X_prot", WX)
    return P, U1, k


def theoretical_min_distance(W, X_selected, rank_limit: int) -> float:
    """Minimum achievable ``||BA - W||_F`` for the constrained problem.

    Useful as an independent check of :func:`factorize_update`.
    """
    P, U1, k = _k_and_bases(W, X_selected)
    if rank_limit < k:
        return float("nan")
    rho = int(rank_limit) - k
    residual = W - U1 @ (U1.T @ W)
    if P.shape[1]:
        residual = residual @ (np.eye(W.shape[1]) - P @ P.T)
    if rho <= 0 or residual.size == 0:
        return float(np.linalg.norm(residual, "fro"))
    s = np.linalg.svd(residual, compute_uv=False)
    if not s.size or s[0] == 0.0:
        return 0.0
    usable = int(np.count_nonzero(s > RANK_ZERO_RTOL * s[0]))
    keep = min(rho, usable)
    return float(np.sqrt(np.sum(s[keep:] ** 2)))


def factorize_update(W: np.ndarray, X_selected: np.ndarray, rank_limit: int):
    """Return ``(B, A)``: optimal constrained rank-``rank_limit`` merge.

    Raises :class:`InfeasibleRankError` when ``rank_limit`` is below the rank
    of the protected response and :class:`RankUndecidableError` when that rank
    cannot be determined reliably.
    """
    if not isinstance(rank_limit, (int, np.integer)) or isinstance(rank_limit, bool):
        raise DataError("秩上限必须是整数")
    if rank_limit < 0:
        raise DataError("秩上限不能为负数")

    P, U1, k = _k_and_bases(W, X_selected)
    if rank_limit < k:
        raise InfeasibleRankError(int(rank_limit), k)

    if k == 0:
        # No protected response (W X = 0).  The constraint (BA) P = 0 is
        # automatically satisfied by the truncated SVD of W because W P = 0.
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        usable = (
            int(np.count_nonzero(s > RANK_ZERO_RTOL * s[0]))
            if s.size and s[0] > 0
            else 0
        )
        keep = min(int(rank_limit), usable)
        return U[:, :keep] * np.sqrt(s[:keep]), np.sqrt(s[:keep])[:, None] * Vt[:keep, :]

    extra = int(rank_limit) - k
    # Free part: the component of W outside col(U1) and outside span(X).
    residual = W - U1 @ (U1.T @ W)
    if P.shape[1]:
        residual = residual @ (np.eye(W.shape[1]) - P @ P.T)

    U2 = np.zeros((W.shape[0], 0))
    if extra > 0 and residual.size:
        U2_full, s2, _ = np.linalg.svd(residual, full_matrices=False)
        usable = (
            int(np.count_nonzero(s2 > RANK_ZERO_RTOL * s2[0]))
            if s2.size and s2[0] > 0
            else 0
        )
        U2 = U2_full[:, : min(extra, usable)]

    B = np.hstack([U1, U2]) if U2.shape[1] else U1
    if U2.shape[1]:
        A = np.vstack([U1.T @ W, U2.T @ residual])
    else:
        A = U1.T @ W
    return B, A


def response_max_abs_error(B, A, W, X_selected) -> float:
    target = W @ X_selected
    merged = B @ (A @ X_selected) if B.shape[1] else np.zeros_like(target)
    if target.size == 0:
        return 0.0
    return float(np.max(np.abs(merged - target)))


def per_column_response_check(B, A, W, X_selected, indices=()):
    """Verify the protected response one column at a time.

    For each protected column ``j`` the L2 norm of the response error must not
    exceed :data:`RESPONSE_COL_RTOL` times the larger of that column's input
    L2 norm and its original response L2 norm.  Returns a list of per-column
    records using ``indices`` as the displayed (original) column numbers.
    """
    if X_selected.size == 0:
        return []
    target = W @ X_selected
    merged = B @ (A @ X_selected) if B.shape[1] else np.zeros_like(target)
    records = []
    for col in range(X_selected.shape[1]):
        x_norm = float(np.linalg.norm(X_selected[:, col]))
        y_norm = float(np.linalg.norm(target[:, col]))
        error = float(np.linalg.norm(merged[:, col] - target[:, col]))
        tolerance = RESPONSE_COL_RTOL * max(x_norm, y_norm)
        records.append(
            {
                "index": int(indices[col]) if len(indices) == X_selected.shape[1] else col,
                "input_norm": x_norm,
                "response_norm": y_norm,
                "error_l2": error,
                "tolerance_l2": float(tolerance),
                "passed": bool(np.isfinite(error) and error <= tolerance),
            }
        )
    return records


def frobenius_distance(B, A, W) -> float:
    merged = B @ A if B.shape[1] else np.zeros_like(W)
    return float(np.linalg.norm(merged - W, "fro"))


def merge(B1, A1, B2, A2, X, rank_limit: int, protected_indices):
    """Validate, merge and return the result bundle used by the API."""
    B1, A1, B2, A2 = validate_adapters(B1, A1, B2, A2)
    in_dim = A1.shape[1]
    X = validate_activations(X, in_dim)

    try:
        indices = sorted({int(i) for i in protected_indices})
    except (TypeError, ValueError) as exc:
        raise DataError("保护项下标必须是整数") from exc
    if not indices:
        raise MergeError("至少需要勾选一个必须保住响应的激活向量")
    for idx in indices:
        if idx < 0 or idx >= X.shape[1]:
            raise MergeError(f"保护项下标 {idx} 超出范围（共 {X.shape[1]} 个激活向量）")

    W = B1 @ A1 + B2 @ A2
    X_sel = X[:, indices]

    B, A = factorize_update(W, X_sel, rank_limit)

    shared_dim = int(B.shape[1])
    if shared_dim > int(rank_limit):
        raise MergeError("内部错误：合并后共享维度超过秩上限")

    # Per-column verification: a large protected column must not mask a
    # failure on a small one.
    column_checks = per_column_response_check(B, A, W, X_sel, indices)
    failed = [row for row in column_checks if not row["passed"]]
    if failed:
        worst = max(failed, key=lambda row: row["error_l2"] / max(row["tolerance_l2"], 1e-300))
        raise MergeError(
            "内部校验失败：保护激活 #"
            f"{worst['index']} 的响应 L2 误差 {worst['error_l2']:.3e} 超过该列容差 "
            f"{worst['tolerance_l2']:.3e}（输入范数 {worst['input_norm']:.3e}，"
            f"原响应范数 {worst['response_norm']:.3e}）"
        )

    resp_err = response_max_abs_error(B, A, W, X_sel)
    frob = frobenius_distance(B, A, W)
    min_rank = minimum_feasible_rank(W, X_sel)
    return {
        "B": B,
        "A": A,
        "W": W,
        "rank_limit": int(rank_limit),
        "shared_dim": shared_dim,
        "protected_indices": indices,
        "response_max_abs_error": resp_err,
        "column_response_checks": column_checks,
        "frobenius_distance": frob,
        "frobenius_relative": frob / max(float(np.linalg.norm(W, "fro")), 1e-300),
        "original_update_norm": float(np.linalg.norm(W, "fro")),
        "minimum_feasible_rank": min_rank,
        "theoretical_min_distance": theoretical_min_distance(W, X_sel, rank_limit),
        "trainable_parameters": int(B.size + A.size),
        "layer_in_dim": int(in_dim),
        "layer_out_dim": int(B1.shape[0]),
    }
