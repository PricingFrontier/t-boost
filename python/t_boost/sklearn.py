"""The t-boost estimators: polars-native, scikit-learn OPTIONAL.

scikit-learn is not required — the estimators import and run on numpy/polars alone via the
`_compat` shims. When scikit-learn IS installed, the same classes are genuine sklearn
estimators (real ``BaseEstimator`` base, ``__sklearn_tags__`` contract), so
``Pipeline``/``GridSearchCV``/``cross_val_score``/``clone`` interop keeps working for
external benchmarking harnesses.
"""

from __future__ import annotations

from functools import lru_cache as _lru_cache

import json
import math
import warnings
from typing import Any, Callable

import numpy as np

from ._pricing import annotate_joint_export

from ._compat import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    check_is_fitted,
    type_of_target,
)

from ._t_boost import (
    BUILD_PROFILE,
    _Booster,
    _Model,
    _MultiClassModel,
    _MultiClassTableModel,
    _TableModel,
)

# _CAT_MISSING/_cat_level live in _ingest (shared with the polars split); re-exported here
# because they are part of this module's de-facto surface (tests import them from here).
from ._ingest import (
    _CAT_MISSING,
    _cat_level,
    auto_categorical_idx,
    collect_frame,
    is_polars_eager,
    is_polars_frame,
    resolve_vector,
    select_feature_columns,
    split_polars_columns,
)


def _reject_sparse(x: Any) -> None:
    """Raise a clear ``ValueError`` for scipy-sparse input (t-boost needs a dense design).

    Without this the sparse matrix reaches ``np.asarray`` and fails with a cryptic message; the
    sklearn estimator contract expects a ``ValueError`` here.
    """
    try:
        from scipy.sparse import issparse
    except ImportError:
        return
    if issparse(x):
        raise ValueError(
            "t-boost does not support sparse input; pass a dense numpy array or a DataFrame"
        )


# --- Estimator serialization envelope (preserves the Python-side categorical layout) ------------
# A serialized `_Model`/`_MultiClassModel` records the numeric-first *reordered* axis layout, so it
# alone cannot reconstruct which ORIGINAL columns were categorical (needed to re-split X at serve).
# When a model has categoricals, `to_bytes`/`to_json` wrap the raw model blob in this tiny envelope
# carrying `_cat_indices_` / `feature_names_in_` / `classes_`; a NUMERIC model keeps the raw wire
# format unchanged (back-compat). `from_bytes`/`from_json` sniff and restore.
_ESTIMATOR_MAGIC = b"TBP1"
_MULTICLASS_MAGIC = b"TBMC"  # the Rust multiclass container prefix (see serialize.rs)
_TABLES_MAGIC = b"TBTM"  # the Rust tables-only (pruned) container prefix (see serialize.rs)
_MULTICLASS_TABLES_MAGIC = b"TBMT"  # the Rust pruned-multiclass tables container prefix

# Shipped default relative early-stopping improvement tolerance for the estimator constructors (and
# thus `recommended_recipe`, which inherits it). A round only becomes the new best if it beats the
# incumbent validation deviance by more than this fraction, so epsilon-level noise on the validation
# slice no longer keeps early stopping alive to the `n_trees` cap on large low-signal data. The
# native `_Booster`/Rust `Config` default stays 0.0 (exact legacy any-improvement) as the neutral
# engine primitive; retune the product default here in one place.
_ES_MIN_DELTA_DEFAULT = 1e-4

# Shipped default adaptive early-stopping patience ratio (2026-07-15, benchmark parity): the
# insur-arena runs all went through `recommended_recipe`, which set 1.5 while the bare
# constructors kept fixed patience — the constructors now match what was actually benchmarked.
# Effective patience grows with progress: clamp(ceil(1.5 * best_round), 50, early_stopping_rounds).
# The native `_Booster`/Rust default stays None (fixed patience) as the neutral engine primitive.
_ES_ADAPTIVE_DEFAULT = 1.5

# Shipped default for post-fit table pruning (2026-07-15, benchmark parity): every scored
# insur-arena cell — default and tuned variants — deployed the CV-pruned tables-only model
# (the adapter forced prune=True on all val-is-None fits), so the constructors now default to
# the same artifact. The selection is held-out-guarded (neutral-or-better by construction) and
# costs a handful of extra single-bag fold fits on top of the deploy fit. Pass prune=False for
# the cheaper full-ensemble artifact (and for explain-time table re-weighting, which pruned
# models freeze at fit).
_PRUNE_DEFAULT = True


def _is_degenerate_grouping(groups: "np.ndarray") -> bool:
    """True when the grouping is DEGENERATE — every group holds exactly one row.

    THE IDENTITY CLAIM (av35). Group-aware machinery exists to stop one ENTITY from landing on
    both sides of a carve (its near-duplicate rows then let the model memorize rather than
    generalize — see ``_carve_group_holdout``). At max group size 1 an entity IS a row, so
    "no group straddles the boundary" is true of *any* row-level assignment: the group-aware
    disciplines are vacuous, and an all-singleton ``groups`` array is semantically identical to
    ``groups=None``. Callers pay for the guarantee anyway — a grouped pruned fit carves a shared
    ~``validation_fraction`` (~10%) of rows out of every bag's training data for the ES holdout
    that an ungrouped fit gets for free from each bag's own internal slice. That carve buys
    nothing here, so ``_fit_model``/``_fit_multiclass`` normalize such a grouping to ``None`` at
    ingest and the fit takes the ungrouped path, byte-for-byte.

    This is a property of the group COLUMN, not of the fit: a column unique per row over a
    dataset is unique per row in every subset of it, so the routing decision is stable across
    outer CV splits (it is re-derived per fit regardless, from the rows actually passed).

    Empty input is not degenerate (there is no grouping to speak of, and the empty-design error
    belongs to the caller). Float ids carrying NaN are not degenerate either: a NaN id is not an
    identity, so a column of them must not be read as a column of singletons — the same rule the
    prune guard's OOB-honesty test uses, and the reason both now share this helper.
    """
    g = np.asarray(groups)
    if g.shape[0] == 0:
        return False
    if g.dtype.kind == "f" and bool(np.isnan(g).any()):
        return False
    return int(np.unique(g).size) == int(g.shape[0])


def _carve_group_holdout(
    groups: "np.ndarray",
    fraction: float,
    seed_key: "list[int]",
    strata: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Group-aware holdout mask: permute the distinct groups deterministically and mark WHOLE
    groups held out until at least ``round(fraction * n)`` rows are covered (first crossing).

    On panel data (the same entity contributing near-duplicate rows across periods) a row-level
    carve places an entity on BOTH sides — held-out deviance then improves by memorization,
    which defeats early stopping and honest prune selection (measured: grouped multiclass fits
    ran to the 4000-tree cap). Assigning whole groups to one side restores honesty, mirroring
    what a group-aware outer CV does.

    ``strata`` (ES-runaway fix #2, 2026-07-20): an optional per-ROW label (e.g. ``y > 0`` for
    Poisson/Tweedie — see ``es_strata_for_loss`` in the native core, whose rule this mirrors at
    group instead of row granularity). Every GROUP gets one representative label — 1 iff ANY of
    its rows carries label 1 (a panel entity with a mixed event history still credits the event
    side; the group can't be split anyway) — and the SAME first-crossing greedy carve below then
    runs INDEPENDENTLY within each stratum, so every stratum keeps its own ~``fraction`` share of
    the holdout. Without this, a group-honest carve on high-zero-mass frequency/pure-premium
    panels can land the holdout almost entirely on zero-event groups — the group-carve analog of
    the row-level ES-runaway bug. ``None`` (the default) is the original unstratified carve,
    UNCHANGED (this is the exact pre-``strata`` code path, not a re-derivation of it).

    Deterministic in ``(groups, fraction, seed_key[, strata])``; independent of row order only
    insofar as ``np.unique``'s sorted group identity is. Guarantees at least one row on each side
    whenever there are >= 2 distinct groups; raises otherwise (a single group cannot be carved).
    """
    n = int(groups.shape[0])
    uniq, inv = np.unique(groups, return_inverse=True)
    if uniq.shape[0] < 2:
        raise ValueError(
            "groups must contain at least two distinct values to carve a group-aware holdout"
        )
    sizes = np.bincount(inv, minlength=uniq.shape[0])

    if strata is None:
        target = max(1, int(round(float(fraction) * n)))
        order = np.random.default_rng(seed_key).permutation(uniq.shape[0])
        held = np.zeros(uniq.shape[0], dtype=bool)
        covered = 0
        last_marked = -1
        for g in order:
            if covered >= target:
                break
            held[g] = True
            covered += int(sizes[g])
            last_marked = int(g)
        if held.all():  # fraction ~1 or one dominant group: leave training data on the other side
            held[last_marked] = False
        if not held.any():  # fraction ~0: hold out the first permuted group so the carve exists
            held[int(order[0])] = True
        return held[inv]

    strata = np.asarray(strata).ravel()
    if strata.shape[0] != n:
        raise ValueError(f"strata has {strata.shape[0]} rows but groups has {n}")
    group_label = np.zeros(uniq.shape[0], dtype=np.int64)
    np.maximum.at(group_label, inv, strata.astype(np.int64))
    rng = np.random.default_rng(seed_key)
    held = np.zeros(uniq.shape[0], dtype=bool)
    for lbl in np.unique(group_label):
        members = np.flatnonzero(group_label == lbl)
        if members.shape[0] < 2:
            # a singleton-group stratum can't supply both sides; leave it whole in train
            # (mirrors the core holdout_mask's `m < 2` rule for a singleton row-stratum).
            continue
        stratum_rows = int(sizes[members].sum())
        target = max(1, int(round(float(fraction) * stratum_rows)))
        order = members[rng.permutation(members.shape[0])]
        covered = 0
        last_marked = -1
        for g in order:
            if covered >= target:
                break
            held[g] = True
            covered += int(sizes[g])
            last_marked = int(g)
        if held[members].all():
            held[last_marked] = False
        if not held[members].any():
            held[int(order[0])] = True
    if not held.any():
        # Every stratum was a singleton group (degenerate — e.g. one row per group with both
        # labels present): fall back to holding out the first group overall so the carve still
        # exists, mirroring the unstratified path's own final guard.
        held[0] = True
    return held[inv]


def _carve_group_folds(groups: "np.ndarray", k_folds: int, seed_key: "list[int]") -> "np.ndarray":
    """Group-aware K-fold assignment: every row of the same group lands in the SAME fold, so a
    group can never be split between a prune-CV fold's train side and its held-out scoring side
    (the same leak `_carve_group_holdout` fixes for a single train/select split — see its doc).

    Mirrors ``sklearn.model_selection.GroupKFold``'s greedy-balance strategy: visit distinct
    groups LARGEST first (a deterministic permutation of ``seed_key`` breaks ties among
    equal-size groups) and drop each into whichever fold currently holds the fewest rows, so
    fold sizes stay roughly even despite variable group sizes. Deterministic in
    ``(groups, k_folds, seed_key)``.
    """
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if uniq.shape[0] < k_folds:
        raise ValueError(
            f"groups has only {uniq.shape[0]} distinct values, fewer than k_folds={k_folds}; "
            "a group-aware fold assignment needs at least one group per fold"
        )
    order = np.random.default_rng(seed_key).permutation(uniq.shape[0])
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_sizes = np.zeros(k_folds, dtype=np.int64)
    fold_of_group = np.empty(uniq.shape[0], dtype=np.int64)
    for g in order:
        k = int(np.argmin(fold_sizes))
        fold_of_group[g] = k
        fold_sizes[k] += counts[g]
    return fold_of_group[inv]


# Shipped default interaction-admission hurdle (soft heredity): a level-2/3 split that would add a
# NEW factor must earn enough gain versus the tree's level-1 gain and, when the hurdle is enabled,
# beat the best split that reuses an already-selected factor under the same ranking score. The sklearn
# recipe uses adaptive mode: the hurdle starts at its full value while main-effect gains are still near
# the decayed-max reference, then relaxes; 3-way admission stays stricter than 2-way admission and also
# checks parent-level pair evidence. `0.0` restores pure greedy new-factor admission. The native
# `_Booster`/Rust Config default stays 0.0 + fixed mode (neutral engine primitive). Retune the product
# default here in one place.
_GAIN_HURDLE_DEFAULT = 2.0
_GAIN_HURDLE_MODE_DEFAULT = "adaptive"

# --- the depth/credibility coupling (depth-lift memo section 5.5) --------------------
#
# The memo's strongest recommendation was that `max_depth > 3` should imply a non-zero
# `min_data_in_leaf` floor: a depth-6 tree partitions its <=3 raw features into up to 27
# realized cells instead of 8, so each cell stands on ~3.4x fewer rows, and the section 07
# credibility floor that would protect them is EXACTLY INERT by default. The corpus
# records a measured +25% deviance regression on `brautocoll` (and +0.9% on the
# one-predictor `lossalae`) purely from making feature REUSE more aggressive at depth 3,
# so the concern is not hypothetical.
#
# **P-D2 MEASURED IT, AND THE GUARD IS THE WRONG MEDICINE. The coupled default is 0.**
#
# Three independent measurements, all one-directional:
#
#  1. Synthetic sweep (smooth 4-feature regression, 200 trees, fresh 20k test rows),
#     sweeping the constant `c` in `c * 2**(max_depth-3)` at each depth:
#       n=2k    depth 4: c=0 .1826  c=1 .1829  c=2 .1845  c=5 .1867  c=10 .1892
#               depth 6: c=0 .1850  c=1 .1855  c=2 .1879  c=5 .1984  c=10 .2262
#       n=20k   depth 4: c=0 .1647  c=1 .1693  c=2 .1691  c=5 .1701  c=10 .1715
#               depth 6: c=0 .1648  c=1 .1697  c=2 .1714  c=5 .1727  c=10 .1767
#     Monotonically harmful; `c = 5` (the first value tried) badly so.
#
#  2. `lossalae` (the memo's sharpest stress case -- ONE predictor, so reuse has no
#     competitor at all), 9 arena splits, paired against the shipped depth-3 control:
#       depth 6, floor 0 : -0.0046 rel deviance, W-L 3-6
#       depth 6, floor 8 : -0.0064 rel deviance, W-L 1-8
#     The floor makes the regression it was meant to prevent ~40% WORSE.
#
#  3. The same isolation across the rest of the regression-stress set:
#       swautoins  depth6+floor -0.0195 (fit 0.93x)  vs  depth6 only -0.0111 (0.67x)
#       sgautonb   depth6+floor -0.0041 (fit 1.43x)  vs  depth6 only -0.0042 (0.85x)
#       brautocoll depth6+floor +0.0001              vs  depth6 only +0.0000
#     Never better, sometimes much worse, and consistently SLOWER.
#
# The mechanism is the one P-D1c already had to fix once: a hard per-cell floor applied
# at EVERY level is a far stronger constraint than "cells hold 3.4x fewer rows" suggests,
# because it also terminates growth early. Exempting structurally-empty cells (P-D1c)
# removed the catastrophic version of that; what remains is an ordinary bias/variance
# trade that the data says is on the wrong side.
#
# So `min_data_in_leaf = None` resolves to 0 at every depth -- i.e. the coupling is
# DOCUMENTED AND OFF. `_MIN_DATA_PER_LEAF_AT_DEPTH` is retained at 0 so restoring the
# coupling is a one-character change if a later battery disagrees.
#
# NOTE what this measurement does NOT cover. It is held-out DEVIANCE. The floor's other
# job is a FILING guarantee -- stopping a displayed relativity from standing on a handful
# of policies (spec section 08.154 flags exactly this at depth 3 already, and the lift
# makes it sharper). A model destined for filing should still set `min_data_in_leaf`
# explicitly; the docstring says so. That is a deliberate user choice, not a default.
_MIN_DATA_PER_LEAF_AT_DEPTH = 0


# The pre-lift fixed tree depth, and the default of the `max_depth` knob. Mirrors
# `t_boost_core::engine::LEGACY_MAX_DEPTH`.
_LEGACY_MAX_DEPTH = 3

# The structural depth ceiling. Mirrors `t_boost_core::engine::MAX_DEPTH`.
#
# 8, not 6, since the 2026-08-26 order-hi lift. This is a HARD arithmetic wall, not a
# taste call: the engine's `leaf_of_row` is a `u8` and a leaf id packs one bit per level,
# so depth 8 fills bits 0..7 exactly and depth 9 would truncate.
_MAX_DEPTH = 8

# The pre-lift fixed interaction order, and the default of `max_interaction_order`.
# Mirrors `t_boost_core::engine::LEGACY_MAX_ORDER`.
_LEGACY_MAX_ORDER = 3

# The structural interaction-order ceiling. Mirrors `t_boost_core::engine::MAX_ORDER`.
#
# High order is NOT an exactness relaxation, at 4 or at 8. The fANOVA purification cascade
# is written n-dimensionally (an arbitrary-rank `Tensor` plus an arbitrary-rank odometer),
# so a k-way effect is centred, shed and reconstructed by exactly the same algebra a 3-way
# is, and the five I2 gates are inherited rather than re-derived. What high order costs is
# READABILITY, and that is priced in four separate places: the interaction-gain hurdle
# DOUBLES at every n->n+1 transition (order k pays 2**(k-2) x), the table-budget prior
# charges the PRODUCT of k realized extents against a budget shrunk by
# `table_budget_order_shrink` per order above 3, the evidence-gated prune drops any table
# the held-out folds cannot pay for, and MANDATORY HEREDITY keeps the keep-set a
# downward-closed order ideal.
#
# That last one is the binding constraint and it is COMBINATORIAL. One surviving order-k
# table drags in its whole subset lattice: 2**k - 1 tables, of which
# sum_{j=4..k} C(k,j) are themselves order >= 4 -- 6 at k=5, 22 at k=6, 64 at k=7,
# 163 at k=8. So a bank that a human can read admits order 5 freely, order 6 for a single
# effect, and orders 7-8 not at all. Raising this constant makes order 8 EXPRESSIBLE; it
# does not and cannot make it readable.
_MAX_ORDER = 8

# How much `table_budget_cells` shrinks per interaction order above `_LEGACY_MAX_ORDER`.
# `1.0` is exactly inert. Mirrors `InteractionPolicy::table_budget_order_shrink`.
#
# The soft prior already charges `prod(extent)` over the support's distinct raws, so a
# fourth feature MULTIPLIES the projected cell count. This shrink is the separate,
# deliberate statement that a 4-way table of n cells is harder to read than a 3-way table
# of n cells -- you page through it in 3-D slices -- so it is allowed fewer of them. At
# the lifted budget of 4096 an order-3 support is measured against 4096 cells (~16 per
# axis) and an order-4 support against 2048 (~6.7 per axis).
_TABLE_BUDGET_ORDER_SHRINK_DEFAULT = 2.0


# --- the readability budget (depth-lift memo section 6.4) ----------------------------
#
# `table_budget_cells` is the SOFT table-size prior the split finder multiplies its
# ranking score by: `score = gain * (budget / max(budget, projected_cells)) ** beta`,
# where `projected_cells` is the product of the REALIZED extents of the candidate
# support's distinct raw features. It is a ranking steer, never a hard reject, and it
# has always defaulted to 2_000_000 -- a MEMORY budget, which as the memo puts it
# "never binds on anything a human would read".
#
# The depth lift makes it load-bearing, for a mechanical reason. A reuse-bearing order-3
# effect cannot be held in the compact factored form (`build_tree_box` carries one
# threshold per support axis), so P-D1 routes it to a DENSE cube -- and a dense cube is
# the product of three extents. Measured on catelematic13 at `max_depth=6,
# max_interaction_order=3` with the 2M budget: the bank asked for 33,169,303 cells
# against the 32,000,000 firewall and `explain()` failed outright. The lift is unusable
# at order 3 until the budget steers the extents down.
#
# It steers them hard, because the cube is CUBIC in the extent: at budget B the order-3
# steady state is roughly `B ** (1/3)` cells per axis. 4096 -> ~16 per axis, i.e.
# sixteen 16x16 pages for a 3-way table and a 64x64 page for a pair; order-1 never binds
# (an axis is capped at 255 cells at any depth). That is the memo's stated readability
# point, and it is what the lifted default resolves to. An explicit value always wins.
_TABLE_BUDGET_CELLS_DEFAULT = 2_000_000
_TABLE_BUDGET_CELLS_LIFTED = 4096


def _resolve_table_budget_cells(
    value: int | None, max_depth: int, max_order: int = _LEGACY_MAX_ORDER
) -> int:
    """Resolve ``table_budget_cells``, applying the readability default for lifted fits.

    A fit is "lifted" if it raises EITHER structural cap. Order 4 needs depth >= 4 so it
    already implies a lifted depth, but keying on both keeps the intent legible and keeps
    the resolver correct if the depth floor ever moves.
    """
    if value is not None:
        return int(value)
    if max_depth <= _LEGACY_MAX_DEPTH and max_order <= _LEGACY_MAX_ORDER:
        return _TABLE_BUDGET_CELLS_DEFAULT
    return _TABLE_BUDGET_CELLS_LIFTED


def _resolve_min_data_in_leaf(value: int | None, max_depth: int) -> int:
    """Resolve ``min_data_in_leaf``, applying the depth coupling when it is ``None``.

    See ``_MIN_DATA_PER_LEAF_AT_DEPTH``. An explicit value is returned verbatim.
    """
    if value is not None:
        return int(value)
    if max_depth <= _LEGACY_MAX_DEPTH:
        return 0
    return int(_MIN_DATA_PER_LEAF_AT_DEPTH * (2 ** (max_depth - _LEGACY_MAX_DEPTH)))


# Minimum rows each prune-CV fold's SCORING slice must carry: below this, per-fold table-gain
# estimates are noise-dominated and the keep-set vote turns into a coin flip near the margin.
# The fold count adapts down (never up) from `prune_n_folds` to honor it. Measured on
# ohlsson_sev (fit n=447 after the outer split): k=3 (149-row slices) recovers ~90% of the
# ranking k=5 (89-row slices) loses, at identical deployed-table count and better deviance
# (8/9 splits); k=2 is a wash. Every battery dataset with fit n >= 5*125 measured
# neutral-or-better at k=5, so nothing above that changes.
_MIN_PRUNE_FOLD_ROWS = 125
# Set-level prune guard: minimum EVIDENCE rows for the pruned-vs-full bank comparison to be
# worth trusting — below this the deviance ratio is noise and the guard stays silent. Counts
# out-of-bag-covered rows on the OOB path, shared-ES-holdout rows on the grouped carve path.
#
# (There is no longer a minimum-FIT-ROWS gate. av33 needed one because the ungrouped guard
# bought its honest slice with a shared ES carve that cost every bag ~validation_fraction of
# its rows — a real fit cost that only amortised above ~100k rows. The OOB evidence path costs
# the fit NOTHING, so the gate had nothing left to protect and is gone; see the guard block in
# `_fit_and_prune`.)
_PRUNE_GUARD_MIN_ROWS = 500

# --- EVIDENCE-GATED PRUNING (av37) defaults ---------------------------------------------------
# See the estimator's `prune_drop_z` note for the mechanism. The three numbers are set here so
# the regressor, the classifier and the recipe cannot drift apart.
#
#   _PRUNE_DROP_Z_DEFAULT   how many SEs of the paired per-fold mean drop-gain a table must clear
#                           before it is dropped. `None` restores the av36 aggregator exactly.
#   _PRUNE_KEEP_BUDGET_DEFAULT  the explainability budget: ambiguous admissions stop once the
#                           pre-cascade keep-set reaches max(budget, the av36 keep-set size), so
#                           the gate is a strict no-op wherever the deployed bank was already
#                           larger than a human can read.
#   _PRUNE_GUARD_Z_DEFAULT  SE multiplier on the set-level guard's breach test. `0.0` (the
#                           shipped value) IS the av36 fixed-relative-tolerance test.
#
# `_PRUNE_DROP_Z_DEFAULT = 2.0`: measured, 13 arena datasets, paired against the av36 control.
# The recovery is monotone in z and flattens above 2 (swautoins +0.0142/+0.0171/+0.0179/+0.0184
# at z = 1/1.5/2/3 against a +0.0189 prune-off ceiling; swautoins_pp +0.0097/+0.0121/+0.0131/
# +0.0139 against +0.0138), while the datasets where the prune is RIGHT stay flat and safe
# (brautocoll, where prune-off costs -0.0015, moves +0.00004 to +0.00005 across the whole range;
# autobi_us, where prune-off costs -0.0116, is a bit-exact no-op at every z).
#
# `_PRUNE_KEEP_BUDGET_DEFAULT = 32`: 20 truncated the two datasets whose candidate bank is in the
# low 30s and cost most of their recovery (freclaimdam +0.0025 at 20 against +0.0113 at 32, on a
# deployed bank that goes 13 -> 17 tables; ohlsson_pp -0.0008 vs -0.0017, inside noise). Above
# that it changes nothing measured: credit_g (av36 keeps 67), beMTPL16 (98), autobi_us (29),
# allstate_sev (2,394) and homesite_conv are all bit-exact no-ops at either value, because the
# budget is max(budget, the av36 keep-set) and theirs is already larger.
#
# `_PRUNE_GUARD_Z_DEFAULT = 0.0` — THE SE-AWARE GUARD BAR IS BUILT, MEASURED AND SHIPPED OFF.
# The idea was that the guard should only re-admit on a statistically real gap. It does not
# survive contact with the evidence: the guard's row-level SE on a zero-inflated compound target
# is dominated by a handful of large claims, so it cannot resolve even a large set-level gap.
# ohlsson_pp split 6 measured a +10.10% selected-vs-full gap — a real detonation the av36 guard
# caught, re-admitting 13 tables to 38 — against a paired-difference SE of 9.56%, so even a
# z = 1.5 bar sat at 14.3% and stayed silent: that split's test skill went -0.128 -> -1.799.
# Across the four datasets whose guard ever fires, z > 0 never helped: swautoins -0.0020 (z=3) /
# 0.0000 (z=1.5), swautoins_pp -0.0002 / 0.0000, freclaimdam -0.0011 / -0.0013, ohlsson_pp
# -0.0619 at BOTH. The knob and its diagnostics (`gap_se`, `tol_effective`) stay, so the claim
# stays falsifiable and a future estimator of the gap's noise can be dropped straight in.
#
# What actually protects the bank is the SELECTION gate, not the guard: on that same ohlsson_pp
# split the shipped composition never builds the 13-table bank at all (it keeps 27), the gap
# never reaches the tolerance, and the split scores -0.113 against the av36 control's -0.128.
#
# `_PRUNE_GUARD_Z_DN_DEFAULT = 2.0` / `_PRUNE_GUARD_TOL_FLOOR_DEFAULT = 0.005` (av39) — THE BAR
# MOVES DOWN, NOT UP. The falsified `prune_guard_z` above is an UPWARD bar: it could only make the
# guard fire LESS. The homesite_conv forensic (2026-08-25) found the opposite failure, and found it
# to be the whole of that dataset's deficit: where the jury IS sharp, the fixed 5% tolerance is
# enormous. homesite's paired-difference SE is 0.21%-0.27% of the full-bank level, so 5% is a
# ~20-SE bar that licenses about 0.0075 log-loss of damage the guard can see perfectly well. Two
# shapes, one cause:
#   * SILENT — default split 1 measured a +4.80% selected-vs-full gap, 0.2 points under the bar,
#     and shipped a bank that cost +0.0081 test log-loss against prune-off.
#   * EARLY TERMINATION — tuned split 0 DID breach, climbed exactly one rung, and stopped at a
#     4.48% residual that was still 17 SE of real damage.
# Residual OOB gap predicts test damage near-linearly on that dataset (~0.0015 log-loss per 1% of
# residual gap), so the stopping rule is as load-bearing as the trigger and both must move.
# The bar is therefore `min(tol, max(tol_floor, z_dn * gap_se))`:
#   * `min(tol, .)` — the recalibration can only TIGHTEN. Where the jury is blunt (ohlsson_pp's
#     9.56% SE on a zero-inflated compound target — the measurement that killed the upward bar)
#     `z_dn * gap_se` is far above `tol`, the min() pins the bar back at 5%, and the fit is
#     bit-identical. This is what keeps the av37 verdict intact: the downward knob CANNOT raise a
#     bar the upward knob was shipped off for failing to raise.
#   * `max(tol_floor, .)` — a floor under the tightening, so an arbitrarily sharp jury cannot
#     drive the bar to zero and re-admit the whole bank on immeasurable noise. 0.5% sits ~2 SE
#     above homesite's own gap_se, i.e. right where the tightening saturates for the dataset that
#     motivated it.
# Measured on homesite at the equivalent fixed `prune_guard_tol = 0.02`: 85-92% of the prune-off
# gain recovered on the bad cells, bit-identical on the good ones (their ladders already terminate
# at a 0.55% residual, well inside 2%). `prune_guard_z_dn = 0.0` restores av37/av38 byte-for-byte.
#
# `_PRUNE_BOX_BUDGET_DEFAULT = 0` — SHIPPED OFF, exactly as the depth and order lifts shipped
# their knobs off. The budget is a readability contract whose right value is a product call per
# book (Ralph's bar: post-prune tables/boxes MINIMAL), not a number a default can guess: the same
# multiple of the depth-3 bank is nearly free on one dataset and costs most of the depth lift on
# another. At 0 the gate is a strict no-op — the keep-set is returned
# untouched and the deployed model is bit-identical to the unbudgeted one.
_PRUNE_BOX_BUDGET_DEFAULT = 0
# `_PRUNE_LAMBDA_BOXES_DEFAULT = 0.0` — SHIPPED OFF, and inert to the last bit when off.
#
# The SELECTION-time analogue of `prune_box_budget`. The av38 depth battery closed on exactly
# one named gap: selection cannot see bank size, so it cannot REFUSE a diffuse config. The
# backward prune walk optimizes held-out deviance alone, and the SE rule then breaks TIES
# toward fewer tables — but a config that buys a real deviance gain at 22x the boxes is not a
# tie, so nothing in selection ever declines it. The box budget answers the same question at
# DEPLOY time, after the choice is already made. This prices size while the choice is being
# made: the waypoint objective becomes `mean_deviance + prune_lambda_boxes * n_boxes`.
#
# Units are held-out deviance PER BOX, on the dataset's own per-unit-weight scale, because the
# honest statement is a price and not a ratio. Calibrate from the report, which now carries
# `n_boxes` at every path point whether or not the penalty is armed: "one SE across the whole
# bank" is `path[0]["se"] / path[0]["n_boxes"]`, and "1% of full deviance across the whole
# bank" is `0.01 * path[0]["mean_deviance"] / path[0]["n_boxes"]`.
_PRUNE_LAMBDA_BOXES_DEFAULT = 0.0
# `_PRUNE_TABLE_BUDGET_DEFAULT = 0` — SHIPPED OFF, and inert to the last bit when off.
#
# Ralph moved the explainability bar on 2026-08-27: "in terms of resolution, perfectly happy with
# as many cells within tables to capture the correct resolution, it's just additional multi-way
# table count that I'm keen on keeping to a minimum." CELLS ARE FREE; the count of >=3-way
# TABLES is the scarce thing ("100 3-way tables is not explainable").
#
# `prune_box_budget` prices the wrong one of those two. It caps resolution, and only bounds table
# count as a side effect of dropping supports binarily — so it buys its table discipline by
# starving the tables it keeps. This caps the count directly and lets every survivor carry every
# box it had. A budget of 0 never calls the budgeted binding at all.
_PRUNE_TABLE_BUDGET_DEFAULT = 0
# The arity at or above which a table counts against the bar. Only consulted when a table budget
# or table price is armed, so its value cannot perturb a default fit. Mirrors
# `t_boost_core::prune::DEFAULT_TABLE_MIN_ARITY`.
_PRUNE_TABLE_MIN_ARITY_DEFAULT = 3
# Positivity floor of the exposure-marginal reference measure (2026-09-06): the total floor
# mass as a fraction of the empirical mass, spread evenly over each axis's cells.
_MEASURE_FLOOR_DEFAULT = 1e-3
# `_PRUNE_LAMBDA_TABLES_DEFAULT = 0.0` — SHIPPED OFF, and inert to the last bit when off.
#
# The SELECTION-time analogue of `prune_table_budget`, exactly as `prune_lambda_boxes` is the
# selection-time analogue of `prune_box_budget`: the waypoint objective gains
# `+ prune_lambda_tables * (kept tables of arity >= prune_table_min_arity)`.
#
# KNOWN STRUCTURAL LIMIT, stated here because it decides which knob you should reach for. Like
# `prune_lambda_boxes`, this re-picks a waypoint on an already-fixed deviance-greedy backward
# path; it cannot reorder the walk, so it cannot surrender a 3-way table while holding a pair.
# Walking further back sheds mains and pairs too. For the explainability bar `prune_table_budget`
# is the sharper instrument — it touches ONLY supports at or above the arity floor. This exists
# so selection can SEE table count (the gap the av38 battery named), and so a tuned search has a
# continuous dial as well as a cliff.
_PRUNE_LAMBDA_TABLES_DEFAULT = 0.0
_PRUNE_DROP_Z_DEFAULT: float | None = 2.0
_PRUNE_KEEP_BUDGET_DEFAULT = 32

# PINNED-BANK FOLD FIDELITY (`prune_fold_fidelity`). `False` = the shipped fidelity and a
# byte-identical fit.
#
# THE DEFECT IT ATTACKS. The av37 gate judges every candidate support the DEPLOY bank realized,
# but scores them with `prune_n_folds` cheap fold refits that each search their OWN structure. A
# candidate no fold model ever built produces no fold score at all, so `aggregate_prune_selection`
# sees `gain_n == 0`, imputes `mean_gain = 0.0`, and the legacy rule (`mean_gain > 0`) drops it —
# not on evidence but on ABSENCE. Measured on credit_g (5 tuned splits, 555 candidates/fit):
# 443.6 candidates carry no fold evidence, 344.4 of them order >= 3. Prune-off scores +0.0069
# over av36 where the best keep-budget scores +0.0040, so the unjudged residue is worth ~+0.003.
#
# WHAT IT DOES. After each fold model is fit, the candidate supports its bank is MISSING are
# given per-fold values by an adaptive ridge on the purified cell basis, fit on that fold's TRAIN
# rows only (held-out rows carry IRLS weight 0). The fold bank then covers the candidate set and
# `prune_bank` returns a real drop-one gain for every candidate, so the paired statistic the gate
# needs exists everywhere instead of nowhere. No extra boosting fit — one backfit solve per fold
# on top of the fold fit that already runs. Contrast `prune_fold_bags` (the other fidelity design,
# measured on the `fold-fidelity` branch), which buys PARTIAL coverage for 2.05-2.33x the fit.
#
# LEAKAGE, STATED PLAINLY. The candidate SUPPORT SET is chosen by the full-data fit, which saw
# every fold's held-out rows; only the VALUES are honest. The augmented evidence is therefore
# selection-biased toward keep by roughly the 1/k share of the structure signal, which is why the
# verdict is an OUTER-test battery number and never an inner-evidence one.
#
# Single-output path only: the K>=3 multiclass prune has no fold refits at all (it uses one
# train/select split), so a non-default value there raises instead of silently doing nothing.
#
# WHY IT IS NOT THE DEFAULT: THE RESIDUE IS SIZE, NOT EVIDENCE.
# It does exactly what it claims. On credit_g the candidates carrying a paired per-fold z go
# from 69.9/175.2 (order-2) and 13.4/345.7 (order-3) to 175.2/175.2 and 345.7/345.7 — complete
# coverage, every fold, at 0.93-1.51x the deploy fit (credit_g 0.82s -> 1.24s; fremotor_prem at
# depth 6 / order 4, 800-table fold banks, 119s -> 123s). And then the completed evidence votes
# to DROP. Against the shipped b160 control, 30 splits x 2 seeds, paired and split-clustered:
#
#   arm                       skill delta   t       deployed tables   candidates with a z
#   ff      (b160)            -0.00376    -3.38      73.9  (ctrl 99.1)   541/541 (ctrl 103/541)
#   ff96                      -0.00405    -3.54      67.0
#   ff256   (SIZE-MATCHED)    -0.00351    -3.24      92.9
#   ffunl   (no budget)       +0.00296    +5.17     186.3
#   bunl    (no budget, OFF)  +0.00014    +0.45     104.9
#   noprune (ceiling)         +0.00315    +5.21       -
#
# Read the last three rows together. Fidelity plus an unbounded budget REACHES the prune-off
# ceiling (+0.00296 vs +0.00315, indistinguishable) and the budget alone cannot (+0.00014) —
# so complete evidence is what lets the bank grow. But growth is the whole effect: a pooled
# within-cell regression of skill delta on (deployed-table delta, fidelity indicator) over 300
# split-clustered arm-cells gives +0.000057 per deployed table (t=+3.89) and -0.00224 for
# fidelity ON (t=-2.03). At MATCHED deployed size the completed evidence picks a WORSE bank.
#
# The mechanism is visible in the arity census: on credit_g NO order-3 table ever survives
# purification into the deployed bank (deployed order-3 = 0.0 on every arm). Kept triples are a
# CLOSURE PUMP — they exist only to drag their pairs in by downward closure. The control keeps
# 19.5 triples and deploys 79.1 pairs; fidelity keeps 5.7 and deploys 53.9. Honest evidence
# correctly judges those triples worthless and, in doing so, throws away the pairs that were
# paying. That is a heredity/closure-policy defect, not an evidence defect, and no amount of
# fold fidelity addresses it.
#
# COLLATERAL (2-4 splits x 2 seeds, same control): swautoins and swautoins_pp are a STRICT
# no-op (banks of 14 candidates are already fully covered — 8/8 byte-identical artifacts);
# autobi_us +0.00436 (t=+0.97, deployed 27.8 -> 56.8), ohlsson_pp +0.00049 (t=+1.16),
# brautocoll -0.00044 (t=-1.32), freclaimdam -0.00161 (t=-1.01). Nothing significant either
# way off credit_g, and every cell that moves moves with its deployed size.
#
# NOT A SHAPE ARTIFACT. Swept over 9 (cell_cap, ridge_rows) combinations in {4,8,16} x {5,20,80}
# on 6 splits: `ff` is negative in all 9 and `ffunl` positive in all 9.
#
# WHAT IT IS STILL GOOD FOR. `>=4-way` coverage: the axis-diversity battery stalled because
# almost no high-arity candidate carried any held-out z. On fremotor_prem (depth 6, order 4)
# fidelity takes order-2 from 162/168 to 168/168 and order-3 from 472/836 to 836/836, but
# order-4 only from 68/1184 to 180/1184 — the rest are declined by the factored-effect guard in
# `prune::augment_bank_to_candidates` (a candidate whose order-3 subset went factored would be
# duplicated in `prune_bank`'s identity space). Lifting THAT guard is the prerequisite for
# "generate freely, prune the noise" at order 4+; the evidence machinery itself is no longer
# the blocker below order 4. Full battery: insur-arena scratchpad/fold_fidelity_battery.
_PRUNE_FOLD_FIDELITY_DEFAULT = False

# Fold-stability bar in `aggregate_prune_selection`'s keep rule (exposed 2026-09-04 as
# `prune_min_stability`; 0.5 is the value it was hardcoded at since 15d57d3). A table is kept
# when `mean_gain > min_mean_gain AND (kept_rate >= s OR positive_rate >= s)`, both rates over
# the prune folds — so `s` asks "in how many folds must this table show signal". `1.0` demands
# every fold. The rates quantise to 1/prune_n_folds, so at the default 5 folds this is a
# 3-position dial: (0.4,0.6] = 3/5, (0.6,0.8] = 4/5, (0.8,1.0] = 5/5.
_PRUNE_MIN_STABILITY_DEFAULT = 0.5

# Gain floor in that same rule (exposed 2026-09-04 as `prune_min_mean_gain`). `0.0` is the
# historical `mean_gain > 0.0` test, bit-for-bit — a table kept on a mean drop-gain of 1e-9 is
# kept on noise, and this is the knob that says how much held-out gain a table must actually
# earn. The native report has always carried a `min_mean_gain` field and always written a
# literal 0.0 into it; it now reports the threshold that was used.
_PRUNE_MIN_MEAN_GAIN_DEFAULT = 0.0

# Keep-set selector (2026-09-26). "ranked_path": rank every interaction table the soup built by its
# purified variance, admit them in that order subject to heredity (a k-way table only once all its
# (k-1)-way subsets are in), score heredity-closed prefixes on the soup's out-of-bag jury in ONE pass,
# and deploy the prefix with the lowest out-of-bag deviance. No prune-CV refits, no stability vote,
# no evidence gate/keep budget, no guard. Measured on 35 arena datasets (one split each, the arena's
# tuned hp, real deploys): pairs -27%, >=3-way tables -72%, mean test deviance +0.016%, median
# -0.011%, vs the fold vote + guard. The old guard existed because tables judged one at a time miss
# jointly-important sets and could only be rescued in all-or-nothing chunks (catelematic13: 930 kept,
# 7,554 re-admitted). "fold_vote" is the previous selector, kept for ablation.
_PRUNE_SELECTOR_DEFAULT = "ranked_path"
_PRUNE_PATH_STEPS_DEFAULT = 32
# Stop at the smallest prefix capturing this fraction of the out-of-bag improvement the path offers
# over the mains-only model (1.0 = the minimum). 0.995 measured (35 datasets, additive): Elo 1428 ->
# 1424, pairs -31% and >=3-way tables -30% vs stopping at the minimum.
_PRUNE_PATH_FRACTION_DEFAULT = 0.995
# ...and at least the smallest prefix whose out-of-bag deviance is within this fraction of the best
# (the deploy takes the LARGER of the two prefixes). The gain fraction alone concentrates losses
# where interactions matter most — 0.5% of a large improvement is a large deviance loss (homesite
# +0.78%); this bound caps that at 0.1% of deviance, the same figure as banding's deviance cap.
# Measured offline (35 datasets): Elo 1427 vs 1424 (fraction alone), 1428 (the minimum).
_PRUNE_PATH_TOLERANCE_DEFAULT = 0.001
# Banding (2026-09-26): after the prune, every interaction table becomes a small product grid of
# bands whose change to predictions is held within (tolerance * sigma)^2, sigma = the soup's bag
# noise (see `t_boost_core::banding`). 0.75 measured (arena, 31 datasets): 3-way cells 274M -> 84k,
# largest 3-way 10.8k cells, pooled test deviance +0.015%. None = no banding.
_BAND_TOLERANCE_DEFAULT: float | None = 0.75
# ...and never more than this fraction of the model's own training deviance (the curvature-weighted
# squared move is the second-order estimate of the deviance it costs). The noise tolerance alone
# lets a very noisy model move a long way systematically; this caps what that can cost. 0.001 = 0.1%.
_BAND_DEVIANCE_CAP_DEFAULT = 0.001
_PRUNE_GUARD_Z_DEFAULT = 0.0
_PRUNE_GUARD_Z_DN_DEFAULT = 2.0
_PRUNE_GUARD_TOL_FLOOR_DEFAULT = 0.005
# K>=3 guard floor (2026-09-07): 0.2% of the honest `dev_null - dev_full` deviance improvement.
# On the InsurArena multiclass panels this is 0.0001-0.0013 in log-loss units — below the paired
# 2-SE term wherever the evidence is coarse (pg16, prudential) and just above it where the carve
# is sharp (fremotor), where it buys a 30-40% smaller bank for ~0.0002 skill.
_MULTICLASS_GUARD_FLOOR_DEFAULT = 0.002


def _guard_tol_effective(
    tol: float, gap_se: float, z_up: float, z_dn: float, tol_floor: float
) -> float:
    """The bar the set-level prune guard actually uses — trigger AND ladder stopping rule.

    Two SE-aware corrections to the fixed relative tolerance `tol`, in this order:

    * UPWARD (`z_up` = `prune_guard_z`, av37, shipped OFF at 0.0): `max(tol, z_up * gap_se)` —
      only re-admit on a statistically real gap. Falsified; see `_PRUNE_GUARD_Z_DEFAULT`.
    * DOWNWARD (`z_dn` = `prune_guard_z_dn`, av39): `min(tol, max(tol_floor, z_dn * gap_se))` —
      where the jury is sharp, do not license 20 SE of measurable damage. See
      `_PRUNE_GUARD_Z_DN_DEFAULT`.

    Each correction is skipped when its multiplier is 0 or `gap_se` is not finite, so
    `z_up = z_dn = 0` returns `tol` unchanged and reproduces the pre-av39 guard exactly. The
    downward step is applied to the upward step's OUTPUT, so the two are composable and neither
    silently disables the other: with both armed the bar is
    `min(max(tol, z_up*se), max(tol_floor, z_dn*se))` clipped from below by nothing else.
    """
    tol_eff = float(tol)
    finite = bool(np.isfinite(gap_se))
    if float(z_up) > 0.0 and finite:
        tol_eff = max(tol_eff, float(z_up) * float(gap_se))
    if float(z_dn) > 0.0 and finite:
        tol_eff = min(tol_eff, max(float(tol_floor), float(z_dn) * float(gap_se)))
    return float(tol_eff)


def _guard_mu(obj: str, raw: "np.ndarray") -> "np.ndarray":
    """Response-scale mean for link-scale `raw` — the guard's inverse link, matching the
    native losses (log for poisson/gamma/tweedie, logit for logistic, identity otherwise)."""
    raw = np.asarray(raw, dtype=np.float64)
    mu: np.ndarray
    if obj in {"poisson", "gamma", "tweedie"}:
        mu = np.exp(np.clip(raw, -30.0, 30.0))
    elif obj == "logistic":
        mu = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    else:
        mu = raw
    return mu


def _heredity_sequence(mains: list[Any], inter: list[Any]) -> list[Any]:
    """The ranked path's admission order: `inter` (supports ranked best first) enter one by one,
    a k-way table only once all its (k-1)-way subsets are in (mains are sticky, so pairs are
    unconditioned), and a deferred table enters as soon as it qualifies.

    Reproduces, in O(n log n), the order of the original pass loop -- append each support to a
    pending list, then repeat full passes over the pending list in order, admitting every table
    whose subsets are all in, until a pass admits nothing: a table readied by an admission earlier
    in the pending list than itself enters in the same pass, one readied by a later table waits
    for the next pass.
    """
    import heapq

    admitted = set(mains)
    missing: dict[int, int] = {}
    waiters: dict[Any, list[int]] = {}
    seq: list[Any] = []
    for pos, u in enumerate(inter):
        need = 0
        if len(u) > 2:
            for i in range(len(u)):
                sub = u[:i] + u[i + 1:]
                if sub not in admitted:
                    need += 1
                    waiters.setdefault(sub, []).append(pos)
        missing[pos] = need
        ready = [pos] if need == 0 else []
        while ready:  # one pass per iteration; `ready` holds positions readied behind the cursor
            this_pass = ready
            heapq.heapify(this_pass)
            ready = []
            while this_pass:
                cur = heapq.heappop(this_pass)
                v = inter[cur]
                admitted.add(v)
                seq.append(v)
                for q in waiters.pop(v, ()):
                    missing[q] -= 1
                    if missing[q] == 0:
                        if q > cur:
                            heapq.heappush(this_pass, q)
                        else:
                            ready.append(q)
    return seq


def _band_curvature(
    obj: str, raw: "np.ndarray", w: "np.ndarray", tweedie_rho: float | None
) -> "np.ndarray":
    """Expected loss curvature per row in the link (`raw`, offset included) — the banding weights."""
    mu = _guard_mu(obj, raw)
    h: np.ndarray
    if obj == "poisson":
        h = w * mu
    elif obj == "logistic":
        h = w * mu * (1.0 - mu)
    elif obj == "tweedie":
        p = float(tweedie_rho if tweedie_rho is not None else 1.5)
        h = w * mu ** (2.0 - p)
    else:
        h = np.asarray(w, dtype=np.float64)
    return h


def _guard_reanchor(
    obj: str, raw: "np.ndarray", y: "np.ndarray", w: "np.ndarray", enabled: bool
) -> "np.ndarray":
    """Apply the DEPLOY re-anchor to a raw-bank score before the guard measures it.

    `apply_keepset` shifts the shipped bank's intercept by `ln(Σwy / Σwμ)` whenever
    `reanchor` is on (native `prune::reanchor_shift`, mirrored here exactly). The per-bag banks
    the out-of-bag evidence sums have NOT had that shift, and the shift is NOT second-order:
    purification centres each table under the REFERENCE measure (Laplace-smoothed product of
    marginals), not the empirical one, so dropping a large share of the tables moves the
    empirical level materially. Measured on allstate_sev split 0 — 76% of tables dropped — the
    guard saw a +15.8% gap of which ~13.1 points was this missing shift alone, firing where the
    artifact-scoring av33 guard correctly stayed silent at +3.9%.

    Each arm gets its OWN one-parameter shift on the SAME rows, so this can only remove a level
    artifact common to the comparison — it cannot manufacture a difference between the arms.
    `enabled=False` (the estimator's resolved `reanchor`) returns `raw` untouched, matching a
    deploy that does not re-anchor either.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if not enabled:
        return raw
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    sum_wy = float((w * y).sum())
    sum_wmu = float((w * _guard_mu(obj, raw)).sum())
    if sum_wmu > 0.0 and sum_wy > 0.0:
        return raw + math.log(sum_wy / sum_wmu)
    return raw


# --- Post-prune OOB slope re-anchor (2026-08-22) -----------------------------------------
# Log-link objectives only. `apply_keepset` re-solves the deployed bank's INTERCEPT but never
# its SCALE, and dropping ~76% of the tables compresses the link-scale risk spread by ~10%
# (measured on allstate_sev: sd(log mu) pruned/unpruned = 0.904-0.914). Under gamma the
# flattened score over-predicts the low-severity rows and the deviance punishes that
# asymmetrically — test-y decile 0 alone carries 93-132% of the whole prune cost.
#
# `|b - 1|` epsilon below which the correction is declared a no-op and the artifact is left
# BYTE-IDENTICAL. This is the MATERIALITY gate — "is the rescale big enough to bother with" —
# and it is deliberately not the one carrying the safety argument (see `_SLOPE_MIN_Z`). It
# earns its keep at the two ends the z gate cannot reach: a guard that grew the keep-set back
# to the full bank makes the correction an exact algebraic identity (allstate_sev split 3
# measures b = 1.0000 on a 10,307-table bank), and a fit big enough to make a 0.3% wobble
# "significant" should still not be rewritten for 0.3%.
_SLOPE_EPS = 0.01
# ...and the statistical gate, which is the one that actually decides. `|b - 1|` says the score
# LOOKS mis-scaled; `z = |b - 1| / se(b)` says whether that is distinguishable from the sampling
# noise of a thin out-of-bag jury. The paired 7-dataset / 45-split battery (2026-08-22) puts an
# EMPTY BAND between the two populations:
#
#   z = 3.87 - 22.3   7 allstate_sev splits, sd ratio 0.90-0.97, b 1.016-1.098.
#                     Every one a test-skill GAIN, worst +0.00092, mean +0.0066.
#   z = 0.00 -  1.83  the other 38 splits, across 6 datasets. Firing on these only ever
#                     cost skill: freclaimdam -0.0047, ohlsson_pp -0.0398, beMTPL16 -0.0006.
#
# `|b - 1|` alone does NOT separate them — the harmful population fits b up to 1.12 — which is
# why the epsilon above cannot be the safety gate. 3.0 is the middle of the gap, not a tuned
# value; anything in 2..3.8 selects the identical 7 splits. The low end is held down by a
# separate probe: a correctly-fitted BINOMIAL slope on homesite_conv (not shipped, see
# `_SLOPE_OBJECTIVES`) reached z = 2.50 while costing -0.0135, so 2.0 would not have been safe
# had logistic ever been admitted.
_SLOPE_MIN_Z = 3.0
# Objectives the correction is defined for: the LOG-LINK family, where the prune-induced
# compression was measured and where `a + b*F` is a rescale of a multiplicative risk score.
#
# `logistic` is absent ON EVIDENCE, not merely for want of it. An affine rescale of a logit is
# principled in the abstract — it is exactly Platt scaling — so homesite_conv (261k rows, the
# battery's classifier) was probed with a correctly-fitted BINOMIAL IRLS and the same
# b_keep/b_full ratio. It HARMS: split 0 fits b = 1.030 and costs -0.0135 test skill, split 1
# b = 0.991 for -0.0014. That is the expected shape — the mechanism is gamma's asymmetric y/mu
# penalty on a compressed score, and a symmetric logit deviance has no such exposure — but it is
# now measured rather than assumed. Widening this set means bringing that link's own IRLS too;
# `_slope_variance_power` raises rather than let a log-link solve answer for a logit.
_SLOPE_OBJECTIVES = frozenset({"poisson", "gamma", "tweedie"})
# IRLS budget. Started at the shipped score (see `_fit_link_slope`), the 2-parameter log-link
# GLM below converges in a handful of Newton steps on every battery cell; the cap only bounds a
# pathological input, and the best-iterate tracking makes an overshoot harmless anyway.
_SLOPE_MAX_ITER = 50
# Rows scored on both banks to verify the fold before it is adopted (see
# `_slope_reanchor_post_prune`). A strided sample of the fit design — enough to catch any
# value-carrying field the fold missed, cheap enough to be invisible next to the fit.
_SLOPE_VERIFY_ROWS = 4096


def _slope_variance_power(obj: str, tweedie_rho: float | None) -> float:
    """The GLM variance function's power `p` in `V(mu) = mu^p` for a log-link objective.

    Raises for anything else. The solve below is log-link ALL THE WAY DOWN (`mu = exp(eta)`,
    working response `eta + (y - mu)/mu`), so handing it a logit — or an identity link — does
    not merely lose accuracy, it silently converges to the identity and reports "no correction
    needed" for a score that genuinely needs one. Verified: on a logit compressed by a true
    factor of 1.3 the log-link solve returns exactly `b = 1`. That failure is invisible in the
    report, so it is made loud here instead: extending `_SLOPE_OBJECTIVES` to a new link must
    force the author to supply that link's IRLS, not inherit a wrong one.
    """
    if obj == "poisson":
        return 1.0
    if obj == "gamma":
        return 2.0
    if obj == "tweedie":
        return float(tweedie_rho if tweedie_rho is not None else 1.5)
    raise ValueError(
        f"the post-prune slope solve is log-link only; objective {obj!r} needs its own IRLS "
        "(a logit would be Platt scaling, with its own working weights)"
    )


def _fit_link_slope(
    obj: str, raw: "np.ndarray", y: "np.ndarray", w: "np.ndarray", tweedie_rho: float | None
) -> "tuple[float, float, float]":
    """Deviance-minimizing `(a, b)` for `eta = a + b * raw` under the objective's own GLM.

    This is the canonical one-covariate log-link GLM fit — Fisher scoring (IRLS) on the 2-column
    design `[1, raw]`, which for the log link is exactly Newton on the deviance — NOT a
    least-squares fit of `log y` on `raw`. The distinction matters: the objective's variance
    function is what makes the correction deviance-optimal for the loss the model is scored on,
    and on gamma the two disagree materially (an OLS slope is pulled by the small-`y` tail that
    the gamma weighting deliberately discounts).

    `w` is the exposure-weight the guard measures on (the guard runs only when there is no
    native exposure offset, so exposure lives in `w` and the fit is exposure-weighted exactly as
    the objective demands). `raw` is centered before the solve purely for conditioning; the
    returned pair is on the original scale.

    Returns `(a, b, se_b)`, where `se_b` is the slope's GLM standard error (see below) — the
    scale against which `|b - 1|` has to be judged. Returns `(0.0, 1.0, inf)` — the exact
    identity, with no evidence — for any input that is degenerate or that fails to produce a
    finite improving step, so a caller that folds it in is always a no-op.
    """
    raw = np.asarray(raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(raw) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if ok.sum() < 2:
        return (0.0, 1.0, float("inf"))
    raw, y, w = raw[ok], y[ok], w[ok]
    sw = float(w.sum())
    if not (sw > 0.0):
        return (0.0, 1.0, float("inf"))
    # Center for conditioning: solve in `t = raw - m`, then shift the intercept back.
    m = float(np.dot(w, raw) / sw)
    t = raw - m
    if not np.isfinite(m) or float(np.dot(w, t * t)) <= 0.0:
        return (0.0, 1.0, float("inf"))  # a constant score has no slope to fit
    p = _slope_variance_power(obj, tweedie_rho)
    # Start AT the shipped artifact: in the centered parametrisation `eta = a + b*t` with
    # `t = raw - m`, the identity `eta = raw` is `(a, b) = (m, 1)` — NOT `(0, 1)`, which would
    # start the solve at a score deflated by `e^-m` (a factor of ~3000 on a severity bank) and
    # let gamma's IRLS oscillate between the two exp clips instead of converging.
    a, b = m, 1.0
    best = (m, 1.0)
    best_dev = _guard_mean_deviance(obj, y, a + b * t, w, tweedie_rho)
    for _ in range(_SLOPE_MAX_ITER):
        eta = a + b * t
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        # Fisher weights `w * mu^(2-p)` and working response `eta + (y - mu)/mu`: the standard
        # log-link IRLS for `V(mu) = mu^p`, which covers poisson (p=1), gamma (p=2) and
        # tweedie (p=rho) in one expression.
        ww = w * np.power(mu, 2.0 - p)
        z = eta + (y - mu) / mu
        if not (np.isfinite(ww).all() and np.isfinite(z).all()):
            break
        s0 = float(ww.sum())
        s1 = float(np.dot(ww, t))
        s2 = float(np.dot(ww, t * t))
        r0 = float(np.dot(ww, z))
        r1 = float(np.dot(ww, z * t))
        det = s0 * s2 - s1 * s1
        if not np.isfinite(det) or det <= 0.0:
            break
        a_new = (s2 * r0 - s1 * r1) / det
        b_new = (s0 * r1 - s1 * r0) / det
        if not (np.isfinite(a_new) and np.isfinite(b_new)):
            break
        step = max(abs(a_new - a), abs(b_new - b))
        a, b = a_new, b_new
        # Keep the best iterate by the objective's OWN deviance rather than trusting the last
        # one. Fisher scoring on a well-posed 2-parameter problem converges monotonically here,
        # but this makes the helper total: any oscillation or overshoot degrades to the best
        # step actually seen, and a solve that never improves returns the exact identity.
        dev_now = _guard_mean_deviance(obj, y, a + b * t, w, tweedie_rho)
        if np.isfinite(dev_now) and dev_now < best_dev:
            best_dev, best = dev_now, (a, b)
        if step < 1e-12:
            break
    a, b = best
    # Undo the centering: `a + b*(raw - m)` == `(a - b*m) + b*raw`.
    a_out = a - b * m
    if not (np.isfinite(a_out) and np.isfinite(b) and b > 0.0):
        return (0.0, 1.0, float("inf"))
    # --- standard error of the slope, at the solution -------------------------------------
    # The textbook GLM result: `Var(beta) = phi * (X'WX)^-1`, so the slope's variance is
    # `phi * s0/det` on the centered design. `phi` is estimated by the Pearson statistic over
    # residual df rather than assumed 1 — poisson counts here are routinely over-dispersed, and
    # gamma/tweedie have a free dispersion by construction. Centering is what makes this the
    # variance of the SLOPE alone, uncorrelated with the level.
    #
    # It is the honest gate: `|b - 1|` says the score looks mis-scaled, `se` says whether that
    # is distinguishable from the sampling noise of a thin out-of-bag jury on a small portfolio.
    # A row-count floor is a crude proxy for the same thing; this is the quantity itself.
    eta = a + b * t
    mu = np.exp(np.clip(eta, -30.0, 30.0))
    ww = w * np.power(mu, 2.0 - p)
    s0 = float(ww.sum())
    s1 = float(np.dot(ww, t))
    s2 = float(np.dot(ww, t * t))
    det = s0 * s2 - s1 * s1
    n_eff = float(raw.size)
    if not (np.isfinite(det) and det > 0.0 and n_eff > 2.0):
        return (float(a_out), float(b), float("inf"))
    pearson = float(np.sum(w * (y - mu) ** 2 / np.power(mu, p)))
    phi = pearson / (n_eff - 2.0)
    var_b = phi * s0 / det
    se_b = math.sqrt(var_b) if np.isfinite(var_b) and var_b > 0.0 else float("inf")
    return (float(a_out), float(b), float(se_b))


def _scale_tensor_json(tensor: "dict[str, Any]", mult: float) -> None:
    """Multiply a serialized `Tensor`'s cells in place. Handles both `TensorData` variants."""
    data = tensor.get("data")
    if not isinstance(data, dict):
        return
    if "Dense" in data:
        data["Dense"] = [float(v) * mult for v in data["Dense"]]
    elif "Sparse" in data:
        for entry in data["Sparse"]:
            entry["value"] = float(entry["value"]) * mult


def _fold_slope_into_bank_json(doc: "dict[str, Any]", a: float, b: float) -> None:
    """Fold `F -> a + b*F` into a serialized TableModel, in place.

    The score is `f0 + Σ_u f_u(x)`, so an affine transform of it is EXACTLY representable in the
    artifact's own algebra: `a + b*(f0 + Σ f_u) = (a + b*f0) + Σ (b * f_u)`. Table structure,
    count, supports, axes and grids are untouched — only cell VALUES and the intercept move, so
    the deployed bank stays the same fANOVA decomposition with the same tables.

    Purification survives multiplication exactly: purity is the statement that every axis-slice
    of every table has `w`-weighted mean zero, and `Σ w·(b·v) = b·Σ w·v = 0` for any `b`. Mass
    conservation likewise re-centers on the scaled intercept. The cached second moments are the
    only things that do NOT ride along for free — `variance` is `σ²(f_u)`, which scales by `b²`,
    and the display-only `se_band` is a link-scale standard error, which scales by `|b|` — so
    both are rewritten here rather than left stale (a stale `variance` keeps Sobol SHARES right,
    since every table is off by the same factor, but reports the absolute figure wrong and would
    break the I2 variance-sum identity).
    """
    bank = doc["model"]["bank"]
    bank["f0"] = a + b * float(bank["f0"])
    for tab in bank.get("tables") or []:
        _scale_tensor_json(tab["values"], b)
        tab["variance"] = float(tab["variance"]) * b * b
        se = tab.get("se_band")
        if isinstance(se, dict) and isinstance(se.get("per_cell"), dict):
            _scale_tensor_json(se["per_cell"], abs(b))
        # `support` is per-cell training mass, not a link-scale value — deliberately untouched.
    for fac in bank.get("factored") or []:
        # Over-budget order-3 effects in per-tree box form. Each box's 8 purified octant values
        # are link-scale contributions and scale like any other cell; `per_axis_w` is a
        # normalized cell measure (Σ = 1) and `low` is a routing mask — neither is scaled.
        for box in fac.get("boxes") or []:
            box["p"] = [float(v) * b for v in box["p"]]
        fac["variance"] = float(fac["variance"]) * b * b


def _guard_unit_deviance(
    obj: str, y: "np.ndarray", raw: "np.ndarray", w: "np.ndarray", tweedie_rho: float | None
) -> "np.ndarray":
    """PER-ROW unit deviance of link-scale predictions `raw` — the array `_guard_mean_deviance`
    reduces. Split out so the guard can measure the SPREAD of its own evidence, not only its
    mean (see `_guard_gap_se`)."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    mu = _guard_mu(obj, raw)
    if obj == "poisson":
        ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        dev = 2.0 * (ylogy - (y - mu))
    elif obj == "gamma":
        dev = 2.0 * (-np.log(np.where(y > 0, y, 1.0) / mu) + (y - mu) / mu)
    elif obj == "tweedie":
        p = float(tweedie_rho or 1.5)
        yp = np.where(y > 0, y, 1.0)
        dev = 2.0 * (
            np.where(y > 0, np.power(yp, 2.0 - p), 0.0) / ((1.0 - p) * (2.0 - p))
            - y * np.power(mu, 1.0 - p) / (1.0 - p)
            + np.power(mu, 2.0 - p) / (2.0 - p)
        )
    elif obj == "logistic":
        eps = 1e-12
        mu = np.clip(mu, eps, 1.0 - eps)
        dev = -2.0 * (y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu))
    else:
        dev = (y - mu) ** 2
    return np.asarray(dev, dtype=np.float64)


def _guard_mean_deviance(
    obj: str, y: "np.ndarray", raw: "np.ndarray", w: "np.ndarray", tweedie_rho: float | None
) -> float:
    """Weighted mean unit deviance of link-scale predictions `raw` — the prune guard's metric.

    Mirrors the native losses' unit deviance on the response scale (log link for
    poisson/gamma/tweedie, logit for logistic, identity otherwise). Only the RATIO of two
    values on the same rows is ever used, so the normalization constant is irrelevant.
    """
    w = np.asarray(w, dtype=np.float64)
    dev = _guard_unit_deviance(obj, y, raw, w, tweedie_rho)
    ws = float(w.sum())
    return float((dev * w).sum() / ws) if ws > 0 else float("nan")


def _guard_gap_se(
    dev_sel: "np.ndarray", dev_full: "np.ndarray", w: "np.ndarray"
) -> float:
    """Standard error of the RELATIVE deviance gap `(D_sel - D_full) / D_full`.

    `D_x` is the weighted mean unit deviance the guard compares; both arms are measured on the
    SAME evidence rows, so the per-row difference is paired and its spread is the sampling noise
    of the comparison itself — the quantity the fixed 5% relative tolerance has no way to see.
    Only the NUMERATOR is random here. The test the guard runs is
    `D_sel - D_full > tol * D_full`, and `D_full` appears on both sides: it is the common scale
    the comparison is expressed in, not a source of disagreement between the two arms. So the
    SE is that of the paired per-row DIFFERENCE, divided by the full-bank level —
    `sd(w_i * (dev_sel_i - dev_full_i)) * sqrt(n) / (sum(w) * D_full)`.

    THIS IS NOT A SIMPLIFICATION, IT IS THE MEASURED FIX. An earlier revision delta-methoded the
    whole ratio, carrying the LEVEL's variance (`r^2 * Var(D_full)`) into the bar. On a
    heavy-tailed objective that term is the tail of the deviance itself and it swamps everything:
    ohlsson_pp (tweedie pure premium, 34,626 evidence rows) split 6 measured a +10.1% selected-
    vs-full gap — a real detonation the av36 guard caught, re-admitting 13 tables to 38 — against
    a ratio-SE of 7.5%, so a z=3 bar sat at 22.5% and stayed SILENT. That split's test skill went
    from -0.128 to -1.799. The paired difference carries no such tail: the two arms differ only
    by the dropped tables' per-row contributions.

    ONE HONEST CAVEAT, recorded rather than corrected: on the out-of-bag path the rows are not
    independent draws — each row's score is a mean over the ~1.9 bags that never drew it, and
    neighbouring rows share bags. Ignoring that correlation UNDERSTATES this SE, i.e. errs
    toward the guard still firing, which is the safe direction for a no-harm guard. Returns
    `inf` when there is nothing to measure (fewer than two rows, or a non-positive full-bank
    deviance), which makes the caller fall back to the fixed tolerance alone.
    """
    w = np.asarray(w, dtype=np.float64)
    n = int(w.size)
    sw = float(w.sum())
    if n < 2 or not (sw > 0.0):
        return float("inf")
    d = w * (np.asarray(dev_sel, float) - np.asarray(dev_full, float))
    lvl = float((w * np.asarray(dev_full, float)).sum() / sw)
    if not (np.isfinite(lvl) and lvl > 0.0):
        return float("inf")
    sd = float(np.std(d, ddof=1))
    if not np.isfinite(sd):
        return float("inf")
    # sum(d) / sum(w) is the absolute gap; its SE is sd(d) * sqrt(n) / sum(w).
    se_abs = sd * math.sqrt(n) / sw
    se_rel = se_abs / lvl
    return float(se_rel) if np.isfinite(se_rel) else float("inf")

# ES patience for the prune-CV FOLD fits only (deploy fits keep the estimator's own
# early_stopping_rounds). Promoted from a study hook to the default 2026-07-16: fold fits are
# table-survival votes, ~48% of their rounds were post-best patience tail at the 500 cap, and
# 250 left every studied keep-set (brvehins1, euhealth, 406k MTPL) bit-identical with the MTPL
# CV scan 19.4% faster.
_PRUNE_FOLD_ES_PATIENCE = 250
# K>=3 fold-fit patience (2026-09-07): measured byte-identical to 250 on the arena panels.
_MULTICLASS_PRUNE_FOLD_ES_PATIENCE = 100


def _estimator_metadata(est: "_BaseTBoost") -> dict[str, Any]:
    # Distinguish the pruned tables-only multiclass container (TBMT) from the full per-tree one
    # (TBMC) — both hang off `_multi_model`, but they decode through different Rust classes, so a
    # single "multi" kind here misroutes TBMT bytes to the TBMC decoder on load (H8).
    multi = getattr(est, "_multi_model", None)
    if isinstance(multi, _MultiClassTableModel):
        kind = "multi_tables"
    elif multi is not None:
        kind = "multi"
    else:
        kind = "single"
    md: dict[str, Any] = {"kind": kind, "objective": est.objective, "tweedie_rho": est.tweedie_rho}
    cat_idx = getattr(est, "_cat_indices_", None)
    if cat_idx:
        md["cat_indices"] = [int(i) for i in cat_idx]
    names = getattr(est, "feature_names_in_", None)
    if names is not None:
        md["feature_names_in_"] = [str(v) for v in names]
    classes = getattr(est, "classes_", None)
    if classes is not None:
        md["classes_"] = np.asarray(classes).tolist()
    # The INPUT column count, which `_attach_model` cannot recover from the model alone: its
    # `n_features` is AXIS-indexed, and a multi-channel categorical (`cat_channels`) owns more
    # than one axis. Without this a reloaded multi-channel model rejects the very frame it was
    # fit on ("X has 2 features but model expects 3").
    #
    # Written ONLY when the two genuinely differ — i.e. some categorical really did emit more
    # than one channel axis. Every single-channel fit (all of them before `cat_channels`
    # existed, and every fit that leaves it at its default) therefore keeps a byte-identical
    # envelope, so this fix cannot perturb an existing blob. Older blobs simply lack the key
    # and keep the old (axis-count) behavior, which was already correct for them.
    n_in = getattr(est, "n_features_in_", None)
    fitted = multi if multi is not None else getattr(est, "_model", None)
    n_axes = getattr(fitted, "n_features", None) if fitted is not None else None
    if n_in is not None and n_axes is not None and int(n_axes) != int(n_in):
        md["n_features_in_"] = int(n_in)
    for key in ("_ae_requires_weight_", "_ae_requires_exposure_"):
        if getattr(est, key, False):
            md[key] = True
    return md


def _restore_metadata(est: "_BaseTBoost", md: dict[str, Any]) -> None:
    if "objective" in md:
        est.objective = str(md["objective"])
    if "tweedie_rho" in md:
        est.tweedie_rho = float(md["tweedie_rho"])
    for key in ("_ae_requires_weight_", "_ae_requires_exposure_"):
        setattr(est, key, bool(md.get(key, False)))
    if md.get("cat_indices") is not None:
        est._cat_indices_ = [int(i) for i in md["cat_indices"]]
    if md.get("feature_names_in_") is not None:
        est.feature_names_in_ = np.asarray(md["feature_names_in_"], dtype=object)
    if md.get("classes_") is not None:
        est.classes_ = np.asarray(md["classes_"])
    # Restore the INPUT column count over `_attach_model`'s axis count (see
    # `_estimator_metadata`); older blobs without the key keep whatever `_attach_model` set.
    if md.get("n_features_in_") is not None:
        est.n_features_in_ = int(md["n_features_in_"])


def _pack_bytes(est: "_BaseTBoost", inner: bytes) -> bytes:
    md = _estimator_metadata(est)
    if (not md.get("cat_indices") and md.get("classes_") is None
            and not md.get("_ae_requires_weight_") and not md.get("_ae_requires_exposure_")):
        return inner  # numeric regressor: unchanged raw wire format
    header = json.dumps(md).encode("utf-8")
    return _ESTIMATOR_MAGIC + len(header).to_bytes(4, "big") + header + inner


def _unpack_bytes(data: bytes) -> tuple[dict[str, Any], bytes]:
    if data[:4] == _ESTIMATOR_MAGIC:
        hlen = int.from_bytes(data[4:8], "big")
        md = json.loads(data[8 : 8 + hlen].decode("utf-8"))
        inner = data[8 + hlen :]
        # The inner payload's own magic is authoritative over a stale/buggy envelope kind
        # (mirrors __setstate__'s sniffing): blobs saved before the multi_tables kind existed
        # (H8) carry kind='multi' even when inner is actually a TBMT tables container.
        if inner[:4] == _MULTICLASS_TABLES_MAGIC:
            md["kind"] = "multi_tables"
        elif inner[:4] == _MULTICLASS_MAGIC:
            md["kind"] = "multi"
        return md, inner
    if data[:4] == _MULTICLASS_MAGIC:
        return {"kind": "multi"}, data
    if data[:4] == _MULTICLASS_TABLES_MAGIC:
        return {"kind": "multi_tables"}, data
    return {"kind": "single"}, data


def _pack_json(est: "_BaseTBoost", inner_json: str) -> str:
    md = _estimator_metadata(est)
    if (not md.get("cat_indices") and md.get("classes_") is None
            and not md.get("_ae_requires_weight_") and not md.get("_ae_requires_exposure_")):
        return inner_json  # numeric regressor: unchanged raw model JSON
    md["__tri_estimator__"] = 1
    md["model"] = inner_json
    return json.dumps(md)


def _unpack_json(text: str) -> tuple[dict[str, Any], str]:
    obj = json.loads(text)
    if isinstance(obj, dict) and obj.get("__tri_estimator__"):
        inner = str(obj["model"])
        # As in `_unpack_bytes`: trust the inner JSON's own kind string over a stale envelope
        # kind, so old 'multi'-labeled tables blobs still dispatch correctly (H8).
        if '"t-boost-multiclass-tables"' in inner:
            obj["kind"] = "multi_tables"
        elif '"t-boost-multiclass"' in inner:
            obj["kind"] = "multi"
        return obj, inner
    if '"t-boost-multiclass-tables"' in text:
        kind = "multi_tables"
    elif '"t-boost-multiclass"' in text:
        kind = "multi"
    else:
        kind = "single"
    return {"kind": kind}, text

__all__ = [
    "PrecisionWarning",
    "TBoostRegressor",
    "TBoostClassifier",
    "recommended_recipe",
]


class PrecisionWarning(UserWarning):
    """Input was copied to float32 before entering the Rust core."""


def _feature_names_from_x(x: Any) -> list[str] | None:
    columns = getattr(x, "columns", None)
    if columns is None:
        return None
    return [str(c) for c in columns]


def _as_float32_2d(x: Any, *, warn: bool) -> np.ndarray:
    arr0 = np.asarray(x)
    if arr0.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={arr0.ndim}")
    if warn and arr0.dtype != np.float32:
        warnings.warn(
            "t-boost converts input features to float32 before fitting/scoring",
            PrecisionWarning,
            stacklevel=3,
        )
    # Preserve the input's own memory order rather than forcing C: the native layer
    # (raw_columns_from_array, crates/t-boost-py/src/lib.rs) ingests F-contiguous float32
    # via one memcpy per column, no transpose -- forcing order="C" here would silently defeat
    # that path for every estimator caller by copying F-order input into C-order before it
    # ever reaches Rust. order="A" keeps an already-contiguous array's own C/F layout and,
    # critically, returns the SAME object with no copy at all once the dtype already matches
    # (the common already-float32 case, either order). A genuinely non-contiguous input
    # (neither C nor F -- e.g. a strided slice) is NOT made contiguous by order="A" alone when
    # its dtype already matches float32, since no copy is triggered in that case either; force
    # it into a contiguous layout explicitly below for that one remaining case -- there is no
    # "natural" order to preserve for a non-contiguous array, and the native layer requires one.
    out = np.asarray(arr0, dtype=np.float32, order="A")
    if not (out.flags["C_CONTIGUOUS"] or out.flags["F_CONTIGUOUS"]):
        out = np.ascontiguousarray(out)
    return out


def _as_float32_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32, order="C")
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _n_columns(x: Any) -> int:
    shape = getattr(x, "shape", None)
    if shape is None:
        shape = np.asarray(x).shape
    if len(shape) != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={len(shape)}")
    return int(shape[1])


def _column_as_str_list(x: Any, j: int) -> list[str]:
    if hasattr(x, "iloc"):
        return _series_as_str_list(x.iloc[:, j])
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={arr.ndim}")
    return [_cat_level(v) for v in arr[:, j].tolist()]


# Below this many rows the per-value loop beats factorize's fixed cost; both give the same list.
_FACTORIZE_MIN_ROWS = 64


def _factorize_safe(col: Any) -> tuple[np.ndarray, Any] | None:
    """``(codes, uniques)`` for a pandas Series (code ``-1`` = missing), or ``None`` when
    factorizing could change a label.

    Factorize merges values by EQUALITY, so ``1``/``1.0``/``True`` in an object column, or
    ``-0.0``/``0.0`` in a float column, would collapse to one level although ``_cat_level``
    stringifies them differently. It is therefore used only where equal values have equal
    strings: categorical, pandas string, numpy integer/bool, and object columns whose every level
    is exactly ``str``. Below ``_FACTORIZE_MIN_ROWS`` the per-value path is cheaper.
    """
    if len(col) < _FACTORIZE_MIN_ROWS:
        return None
    import pandas as pd

    dt = col.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return np.asarray(col.cat.codes), dt.categories
    if dt == object or isinstance(dt, pd.StringDtype):
        codes, uniques = pd.factorize(col, use_na_sentinel=True)
        if dt == object and not all(type(u) is str for u in uniques):
            return None
        return codes, uniques
    if isinstance(dt, np.dtype) and dt.kind in "iub":
        codes, uniques = pd.factorize(col, use_na_sentinel=True)
        return codes, uniques
    return None


def _factorized_labels(uniques: Any) -> list[str]:
    """``_cat_level`` of each factorized level, plus ``_CAT_MISSING`` last (factorize's ``-1``)."""
    return [_cat_level(u) for u in uniques] + [_CAT_MISSING]


def _factorized_levels(col: Any) -> list[str] | None:
    """``[_cat_level(v) for v in col]`` for a pandas Series, computed once per DISTINCT level.

    The per-value loop runs ``_cat_level`` (and its ``pd.isna`` probe) on every row — 0.56 s of a
    1M-row predict on freMTPL2freq's four categoricals, more than the whole native scoring pass.
    Factorizing first, stringifying the K levels and gathering by code is 3-4x faster and gives
    the identical list where ``_factorize_safe`` allows it; ``None`` otherwise.
    """
    fz = _factorize_safe(col)
    if fz is None:
        return None
    codes, uniques = fz
    lookup = np.empty(len(uniques) + 1, dtype=object)
    lookup[:] = _factorized_labels(uniques)
    out: list[str] = lookup[np.asarray(codes)].tolist()  # code -1 -> last slot, _CAT_MISSING
    return out


def _column_as_codes(x: Any, j: int) -> tuple[np.ndarray, list[str]]:
    """Categorical column ``j`` as ``(codes, labels)``: row ``r``'s label is
    ``labels[codes[r]]``, exactly the label ``_column_as_str_list`` gives it. The native serve
    path then encodes and bins each distinct label once instead of hashing a string per row, and
    no per-row string crosses into the extension (``cat_codes`` on the table models)."""
    if hasattr(x, "iloc"):
        col = x.iloc[:, j]  # once: pandas column access dominates a small predict
        fz = _factorize_safe(col)
        if fz is not None:
            codes, uniques = fz
            labels = _factorized_labels(uniques)
            codes = np.asarray(codes, dtype=np.int64)
            codes = np.where(codes < 0, len(labels) - 1, codes).astype(np.uint32)
            return codes, labels
        labels_per_row = [_cat_level(v) for v in col.to_numpy()]
    else:
        labels_per_row = _column_as_str_list(x, j)
    # Small or factorize-unsafe columns: one label per row (labels may repeat), which costs what
    # the per-row label path always did — deduplicating here would only add work.
    return np.arange(len(labels_per_row), dtype=np.uint32), labels_per_row


def _series_as_str_list(col: Any) -> list[str]:
    """A pandas Series' per-row category labels (see `_factorized_levels`)."""
    levels = _factorized_levels(col)
    if levels is not None:
        return levels
    return [_cat_level(v) for v in col.to_numpy()]


def _numeric_subset(x: Any, idx: list[int]) -> Any:
    if hasattr(x, "iloc"):
        return x.iloc[:, idx]
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={arr.ndim}")
    return arr[:, idx]



def _ctor_defaults(cls: type) -> dict[str, Any]:
    """Constructor defaults for `cls` (see `_ctor_defaults_cached`)."""
    return _ctor_defaults_cached(cls)


@_lru_cache(maxsize=None)
def _ctor_defaults_cached(cls: Any) -> dict[str, Any]:
    """Constructor defaults for `cls`, so a deprecation warning fires only on a value the user
    actually set. Cached: the signature never changes at runtime."""
    import inspect as _inspect

    return {
        name: p.default
        for name, p in _inspect.signature(cls.__init__).parameters.items()
        if name != "self" and p.default is not _inspect.Parameter.empty
    }

class _BaseTBoost(BaseEstimator):  # type: ignore[misc]  # sklearn is untyped (no py.typed)
    # Fitted state (created only by fit/attach, not __init__): a single-output `_model` (regression
    # / binary), or the multiclass `_multi_model` (K>=3 softmax). Annotated for the type checker.
    _model: _Model | _TableModel
    _multi_model: _MultiClassModel | _MultiClassTableModel | None

    # NOTE: constructor defaults deliberately embed the recommended recipe so a bare estimator
    # performs well without tuning (see `recommended_recipe`) — and, since 2026-07-15, they match
    # the insur-arena benchmark deployment exactly (Ralph: "i want the library to match what was
    # benchmarked"): adaptive early-stopping patience (early_stopping_adaptive=1.5) and post-fit
    # table pruning (prune=True) are ON by default, mirroring what the arena adapter set on every
    # scored cell. Early stopping is ON
    # (validation_fraction=0.1, early_stopping_rounds=500) against a large n_trees=4000 cap, with
    # leaf refinement (leaf_refine_steps=4), per-tree column sampling (colsample_bytree=0.8), and
    # outer bagging (n_bags=8 — spec §09's stronger-variance-reduction setting; the 2026-07-10
    # ohlsson-gap study measured row-bag + column diversity as the two stacking ranking levers:
    # +0.0035 gini / -0.0002 deviance over 4 bags + full columns, replicated across 9 datasets).
    # n_bags=8 is ~8x the train/predict cost of a single fit — accuracy is the product priority.
    # bag_subsample=0.8 (subagging, WITHOUT replacement) rather than 1.0 (bootstrap): a bootstrap
    # bag duplicates rows, so the per-bag early-stop validation carve overlaps the bag's training
    # multiset — a train/val leak that makes val deviance improve forever and defeats early stopping
    # (measured on French MTPL: bags ran to the 4000-tree cap and over-dug the low-rate tail;
    # subagging stops at ~650 trees/bag, heals the tail, and beats EBM). Set 1.0 to restore bootstrap.
    # These are PRODUCT defaults; the native `_Booster`/Rust `Config::default()` stay neutral
    # (single booster, no early stopping) as the engine primitive that tests build on.
    #
    # Binary classification (K=2): the internal early-stopping holdout carve is stratified by
    # class in the core (`holdout_mask`, crates/t-boost-core/src/engine/boost.rs), for every
    # `n_bags` configuration — every bag's own carve routes through `fit_single`, which derives
    # the stratum from `y` itself (logistic objective) before carving, so bagging's per-bag row
    # subsampling doesn't bypass it.
    #
    # Multiclass (K>=3): the early-stopping holdout carve IS class-stratified (the class labels
    # are the strata — `engine::boost::fit_multiclass`'s carve passes `Some(&labels)`), so rare
    # classes keep holdout representation.
    #
    # Every path (regression, binary, multiclass), grouped panel data: pass `groups=` to `fit`
    # — the row-level carve leaks near-duplicate rows of the same entity into validation and
    # defeats early stopping (fits run to the tree cap); the group-aware carve assigns whole
    # groups to one side instead (see `_carve_group_holdout`/`_carve_group_folds`).
    # A DEGENERATE grouping (every group exactly one row — an anonymised per-row policy id
    # framed as a panel key) is normalized away at ingest and fits exactly as `groups=None`
    # would: at group size 1 the group-honest disciplines are vacuous, so the ~10%-of-rows
    # shared ES carve they cost is pure loss (see `_is_degenerate_grouping`).
    def __init__(
        self,
        n_trees: int = 4000,
        learning_rate: float = 0.05,
        lambda_: float = 1.0,
        # --- EVERYTHING BELOW IS KEYWORD-ONLY (2026-09-04) -------------------------------------
        # `n_trees`, `learning_rate` and `lambda_` keep their positions — they are the only three
        # anyone plausibly passes positionally. The remaining ~86 become keyword-only, which makes
        # any FUTURE removal or reordering safe: in a positional signature, deleting a parameter
        # silently re-aims every argument after it onto the wrong name, and on an 89-parameter
        # constructor that is a defect waiting to happen.
        #
        # Verified free before landing: there is no positional construction of either estimator
        # anywhere in this repo or in insur-arena — the sole programmatic path is
        # `recommended_recipe`, already keyword-only past `objective`.
        #
        # And it is safe for sklearn: both real sklearn and the `_compat` shim build
        # `_get_param_names` by excluding VAR_POSITIONAL and VAR_KEYWORD only, so KEYWORD_ONLY
        # parameters survive `get_params` / `set_params` / `clone` intact. A `**kwargs` tail would
        # NOT — it vanishes from `get_params`, so `clone` would silently drop it. That is why the
        # long tail is moved behind `*` rather than into `**kwargs`.
        *,
        lambda_scale_invariant: bool = False,
        l1_leaf: float = 0.0,
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_delta_step_gated: Any = None,
        max_bin: int = 254,
        objective: str = "squared_error",
        tweedie_rho: float = 1.5,
        min_data_in_leaf: int | None = None,
        min_sum_hessian_in_leaf: float = 0.0,
        min_weight_sum_in_leaf: float = 0.0,
        path_smooth: float | None = None,
        subsample: float | None = None,
        colsample_bytree: float = 0.8,
        learning_rate_decay: float = 0.0,
        validation_fraction: float | None = 0.1,
        early_stopping_rounds: int = 500,
        early_stopping_adaptive: float | None = _ES_ADAPTIVE_DEFAULT,
        early_stopping_min_delta: float = _ES_MIN_DELTA_DEFAULT,
        interaction_gain_hurdle: float = _GAIN_HURDLE_DEFAULT,
        interaction_gain_hurdle_mode: str = _GAIN_HURDLE_MODE_DEFAULT,
        graduate: bool | None = None,
        graduation_alpha: float | None = None,
        graduation_high_order_alpha: float = 0.0,
        leaf_refine_steps: int = 4,
        leaf_refine_backtracks: int = 4,
        refine_closed_form_tier2: bool = True,
        incremental_mu: bool = False,
        mvs_min_rows: int = 1,
        hist_precision: str | None = None,
        n_bags: int = 8,
        bag_subsample: float = 0.8,
        cell_refit_base: float | None = None,
        cell_refit_gamma: float = 2.0,
        ridge_refit_l2: float | None = None,
        ridge_refit_max_iter: int = 5,
        nesterov: bool = False,
        dart_drop_rate: float | None = None,
        random_strength: float = 0.0,
        reanchor: bool | None = None,
        reanchor_slope: bool | None = None,
        max_interaction_order: int = 3,
        max_depth: int = 3,
        table_budget_cells: int | None = None,
        table_budget_order_shrink: float = _TABLE_BUDGET_ORDER_SHRINK_DEFAULT,
        seed: int = 0,
        n_jobs: int | None = None,
        monotone_constraints: Any = None,
        categorical_features: Any = None,
        cat_smooth: float | None = None,
        cat_target: str | None = None,
        cat_leakage: str | None = None,
        cat_n_perms: int = 1,
        cat_k: int = 5,
        cat_min_data_per_group: float = 10.0,
        cat_direct_max_levels: int = 16,
        cat_channels: list[str] | None = None,
        cat_count_min_levels: int = 20,
        cat_class_freq_min_levels: int = 3,
        prune: bool = _PRUNE_DEFAULT,
        prune_validation_fraction: float = 0.15,
        prune_se_rule: float = 0.0,
        prune_n_folds: int = 5,
        prune_refit_full: bool = False,
        prune_rebalance: bool = True,
        ref_measure: str | None = None,
        measure_floor: float = _MEASURE_FLOOR_DEFAULT,
        prune_guard: bool = True,
        prune_guard_tol: float = 0.05,
        prune_drop_z: float | None = _PRUNE_DROP_Z_DEFAULT,
        prune_keep_budget: int = _PRUNE_KEEP_BUDGET_DEFAULT,
        prune_fold_fidelity: bool = _PRUNE_FOLD_FIDELITY_DEFAULT,
        prune_guard_z: float = _PRUNE_GUARD_Z_DEFAULT,
        prune_guard_z_dn: float = _PRUNE_GUARD_Z_DN_DEFAULT,
        prune_guard_tol_floor: float = _PRUNE_GUARD_TOL_FLOOR_DEFAULT,
        multiclass_prune_guard: bool = True,
        multiclass_prune_sel_bags: int = 1,
        multiclass_prune_cv: bool = True,
        multiclass_prune_guard_floor: float = _MULTICLASS_GUARD_FLOOR_DEFAULT,
        prune_box_budget: int = _PRUNE_BOX_BUDGET_DEFAULT,
        prune_lambda_boxes: float = _PRUNE_LAMBDA_BOXES_DEFAULT,
        prune_table_budget: int = _PRUNE_TABLE_BUDGET_DEFAULT,
        prune_lambda_tables: float = _PRUNE_LAMBDA_TABLES_DEFAULT,
        prune_table_min_arity: int = _PRUNE_TABLE_MIN_ARITY_DEFAULT,
        prune_min_stability: float = _PRUNE_MIN_STABILITY_DEFAULT,
        prune_min_mean_gain: float = _PRUNE_MIN_MEAN_GAIN_DEFAULT,
        prune_selector: str = _PRUNE_SELECTOR_DEFAULT,
        prune_path_steps: int = _PRUNE_PATH_STEPS_DEFAULT,
        prune_path_fraction: float = _PRUNE_PATH_FRACTION_DEFAULT,
        prune_path_tolerance: float = _PRUNE_PATH_TOLERANCE_DEFAULT,
        band_tolerance: float | None = _BAND_TOLERANCE_DEFAULT,
        band_deviance_cap: float = _BAND_DEVIANCE_CAP_DEFAULT,
        prune_fold_min_rows: int = _MIN_PRUNE_FOLD_ROWS,
        prune_fold_es_patience: int | None = None,
        prune_guard_min_rows: int = _PRUNE_GUARD_MIN_ROWS,
        prune_slope_eps: float = _SLOPE_EPS,
        prune_slope_min_z: float = _SLOPE_MIN_Z,
        prune_size_penalty: float | None = None,
        early_stopping: int | float | None = None,
    ) -> None:
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        # Scale-invariant lambda (prototype, default OFF = bit-identical): rescales `lambda_` by
        # the round's mean per-row hessian, turning it into a live knob on exposure-weighted
        # objectives where the raw `H+lambda` denominator otherwise makes it a dead no-op. See the
        # native `Config::lambda_scale_invariant` doc for the mechanism.
        self.lambda_scale_invariant = lambda_scale_invariant
        self.l1_leaf = l1_leaf
        self.min_split_gain = min_split_gain
        self.max_delta_step = max_delta_step
        self.max_delta_step_gated = max_delta_step_gated
        self.max_bin = max_bin
        self.objective = objective
        self.tweedie_rho = tweedie_rho
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.min_weight_sum_in_leaf = min_weight_sum_in_leaf
        self.path_smooth = path_smooth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.learning_rate_decay = learning_rate_decay
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        # Opt-in adaptive early-stopping patience (None = fixed `early_stopping_rounds`, the
        # bit-identical default). A ratio r>0 grows the effective patience with progress:
        # clamp(ceil(r * best_round), 50, early_stopping_rounds). See the native `Config`.
        self.early_stopping_adaptive = early_stopping_adaptive
        # Relative early-stopping improvement tolerance: a round only becomes the new best (resets the
        # patience window and sets the deployed truncation point) when the validation deviance clears
        # `best * (1 - early_stopping_min_delta)`. The shipped default `_ES_MIN_DELTA_DEFAULT` ignores
        # epsilon-level validation noise so early stopping terminates on large low-signal data; 0.0 is
        # the legacy any-improvement rule. Must be finite and in [0.0, 1.0). See the native `Config`.
        self.early_stopping_min_delta = early_stopping_min_delta
        self.interaction_gain_hurdle = interaction_gain_hurdle
        self.interaction_gain_hurdle_mode = interaction_gain_hurdle_mode
        self.graduate = graduate
        self.graduation_alpha = graduation_alpha
        self.graduation_high_order_alpha = graduation_high_order_alpha
        self.leaf_refine_steps = leaf_refine_steps
        self.leaf_refine_backtracks = leaf_refine_backtracks
        # Tier-2 log-link closed-form leaf refinement (default ON): derive the per-step Newton deltas
        # from the closed form, dropping the remaining per-row passes for log-link (Poisson) refine.
        # ~1e-7 leaf drift vs the exact per-row path (not bit-identical); set False for the
        # byte-identical Tier-1 path. See the native `Config`.
        self.refine_closed_form_tier2 = refine_closed_form_tier2
        # Incremental mu=exp(F) cache for log-link (Poisson) fits: maintain mu multiplicatively across
        # rounds instead of an O(N) exp per round in grad_hess (a speed knob). ~1e-9..1e-6 drift from
        # the exact path (not bit-identical); default OFF. See the native `Config`.
        self.incremental_mu = incremental_mu
        self.mvs_min_rows = mvs_min_rows
        self.hist_precision = hist_precision
        self.n_bags = n_bags
        self.bag_subsample = bag_subsample
        self.cell_refit_base = cell_refit_base
        self.cell_refit_gamma = cell_refit_gamma
        self.ridge_refit_l2 = ridge_refit_l2
        self.ridge_refit_max_iter = ridge_refit_max_iter
        self.nesterov = nesterov
        self.dart_drop_rate = dart_drop_rate
        self.random_strength = random_strength
        self.reanchor = reanchor
        self.reanchor_slope = reanchor_slope
        self.max_interaction_order = max_interaction_order
        self.max_depth = max_depth
        self.table_budget_cells = table_budget_cells
        self.table_budget_order_shrink = table_budget_order_shrink
        self.seed = seed
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.categorical_features = categorical_features
        self.cat_smooth = cat_smooth
        self.cat_target = cat_target
        self.cat_leakage = cat_leakage
        self.cat_n_perms = cat_n_perms
        self.cat_k = cat_k
        self.cat_min_data_per_group = cat_min_data_per_group
        self.cat_direct_max_levels = cat_direct_max_levels
        # P1 multi-channel (default None = mean-only, bit-identical): `["mean", "count"]` adds a
        # target-free count/rarity axis alongside the mean-TS channel on every native
        # categorical (design/multichannel-categoricals.md in the native repo). Improves
        # predict/score accuracy on high-cardinality categoricals. `.tables()`/`.explain()`/
        # pruning fully support multi-channel now (Stage B, 2026-07-24): the channels collapse
        # LOSSLESSLY to one relativity per categorical (one table per raw feature, exact — the
        # pruned tables-only model reproduces the ensemble to <1e-6), and the export is now
        # keyed by actual category level (Piece B, 2026-07-24): every axis lists its levels'
        # labels alongside the numeric borders, single- or multi-channel alike.
        self.cat_channels = cat_channels
        # Count-channel cardinality gate (2026-07-24): the measured win came from adding the
        # count channel to HIGH-cardinality categoricals specifically, so the count axis is
        # only fit for a feature whose post-rare-pooling distinct level count is >= this
        # threshold; a feature below it behaves exactly as mean-only. Inert when
        # `cat_channels` doesn't request `"count"`. `0` disables the gate (every categorical
        # gets the count channel whenever `cat_channels` requests it).
        self.cat_count_min_levels = cat_count_min_levels
        # P3 per-class-channel cardinality gate (only relevant when `cat_channels` includes
        # "class_freq"). The default 3 is the encoding-invariance floor: below 3 post-rare-pooling
        # levels a categorical axis admits exactly one partition however its levels are ordered,
        # so per-class channels cannot add structure. Raise it (e.g. to 20, matching
        # `cat_count_min_levels`) to restrict the per-class channels to high-cardinality features.
        self.cat_class_freq_min_levels = cat_class_freq_min_levels
        # Post-fit table pruning (default off): drop fANOVA tables that don't improve held-out
        # deviance, yielding a smaller, more explainable tables-only model. See `_fit_and_prune`.
        # `prune_se_rule=0.0` selects the held-out-deviance minimum (improves or is neutral on every
        # benchmark tried); a larger value (e.g. 0.5 / 1.0, the classic 1-SE rule) prunes harder for
        # parsimony but MONOTONICALLY degrades the metric — a pure explainability↔accuracy dial, opt in
        # explicitly for smaller models. Honored on both the single-output (regressor/binary, K-fold
        # CV) and multiclass (K>=3, single split) paths.
        # `prune_validation_fraction` only governs the multiclass (K>=3) path's single train/select
        # split; the single-output path uses `prune_n_folds`-fold CV instead (a fraction has no
        # analogous role there). No silent no-ops: fitting with `prune=True` and a non-default
        # `prune_validation_fraction` on the single-output path raises ValueError (see
        # `_validate_dead_prune_params`) rather than quietly ignoring the request. (2026-09-09
        # to 2026-09-20 this was FALSE: it also sized a graduation holdout carved out of every
        # single-output fit, so the default value silently drove a 15% training-data cut on a
        # path that rejected an explicit value. That carve is gone -- see `_fit_and_prune`.)
        # `prune_refit_full` is DEPRECATED and has no effect on ANY path: contribution-stability
        # pruning (see `_fit_and_prune`) always fits the deployed model on all rows -- restored
        # 2026-09-20, after the graduation holdout briefly made this claim false -- and applies the
        # keep-set to that same fit — there is no separate "subset-pruned" fit to conditionally
        # refit, and neither native prune entry point (`fit_prune_selection`,
        # `fit_multiclass_pruned`) takes a parameter it could map to. Setting it True raises
        # ValueError at fit time (see `_validate_dead_prune_params`). Kept only for backward
        # constructor compatibility (`get_params`/`set_params`/`clone` still round-trip it).
        self.prune = prune
        self.prune_validation_fraction = prune_validation_fraction
        self.prune_se_rule = prune_se_rule
        self.prune_n_folds = prune_n_folds
        self.prune_refit_full = prune_refit_full
        # Post-prune re-solve ("rebalance on means", default on): after the keep-set is chosen,
        # recalibrate the surviving tables' cell values to the deviance-optimum of the reduced
        # structure (a single IRLS ridge cell-refit step), re-purified to preserve the 1-/2-/3-way
        # order separation. Held-out no-harm guarded — adopted only if it lowers held-out deviance,
        # so it can only help or be neutral. Set False to deploy the drop-only pruned model.
        self.prune_rebalance = prune_rebalance
        self.ref_measure = ref_measure
        self.measure_floor = measure_floor
        # SET-LEVEL prune no-harm guard (2026-08-04). The keep-set is chosen from per-table
        # leave-one-out held-out gains, which are structurally blind to mass carried JOINTLY by
        # correlated tables (each table's marginal drop-gain is ~0 when siblings absorb it; the
        # set's joint value is not the sum of marginals). Measured failures: allstate_sev fold-CV
        # selections detonating +23% test deviance on some splits while a keep-set from another
        # split scored normally, and the av32 mean+count collapsed-channel pair being dropped
        # together on homesite_conv (+7.7%). The guard compares the DEPLOYED selected bank against
        # the full bank on honest rows — each bag's own OUT-OF-BAG rows for an ungrouped fit
        # (zero data cost, every size), the shared group-honest ES holdout for a grouped one —
        # and, on a breach beyond `prune_guard_tol` (relative deviance), re-admits dropped tables
        # in report-ranked order until the breach clears. Binary table dropping is preserved;
        # only the keep-set grows. Silent (no breach) ⇒ the deployed model is exactly the
        # unguarded one, and the FIT is byte-identical to a guard-off fit either way. See the
        # guard block in `_fit_and_prune` for the evidence design and its caveats.
        self.prune_guard = prune_guard
        self.prune_guard_tol = prune_guard_tol
        # --- EVIDENCE-GATED PRUNING (av37) ------------------------------------------------
        # `prune_drop_z=None` + `prune_guard_z=0.0` reproduce the av36 selection and guard
        # byte-for-byte; both defaults above turn the gate on. Neither touches the FIT — this
        # is post-fit selection only, exactly like `prune_guard` itself.
        #
        # `prune_drop_z`: a table with paired per-fold CV evidence is DROPPED only when its mean
        # drop-gain clears this many standard errors of that mean over the prune folds. No
        # evidence => keep. The av36 rule dropped on the SIGN of a fold-mean whose scatter
        # dwarfed it, which on small banks is a seed coin flip (measured: swautoins keeps
        # 11.0 +/- 1.9 of 14 candidate tables across 90 fits of identical data).
        # `prune_keep_budget`: the explainability budget. Ambiguous admissions are ranked by
        # mean gain and admitted only while the pre-cascade keep-set stays within
        # max(prune_keep_budget, the av36 keep-set size) — so a bank that was already larger
        # than the budget is left exactly as av36 chose it, and a small candidate bank (the
        # regime where the fold evidence cannot resolve anything) comes back whole.
        # `prune_guard_z`: the set-level no-harm guard breaches on
        # gap > max(prune_guard_tol, prune_guard_z * SE(gap)) rather than on the fixed relative
        # tolerance alone. SHIPPED AT 0.0 — i.e. the fixed tolerance — because the measurement
        # refused the idea: the guard's row-level SE on a compound target cannot resolve even a
        # +10% set-level gap (see the `_PRUNE_GUARD_Z_DEFAULT` note for the ohlsson_pp case that
        # settles it). `gap_se` and `tol_effective` are still reported on every fit.
        # `prune_box_budget`: the DEPLOYED-BOX budget (av38 readability gate). The av37
        # `prune_keep_budget` counts TABLES, which is the right unit for a dense bank and the
        # wrong one for a factored one — a factored effect deploys one rank-1 region box per
        # realized tree-region (merged across trees only where the per-axis masks agree
        # exactly), and the rating export emits one row per box. A depth-3 tree contributes
        # ONE box; a depth-6 tree contributes up to `Pi k_i` of them. Lifting `max_depth`
        # past 3 therefore barely moves the table count and
        # multiplies the boxes-per-effect instead (fremotor_payfreq split 0: 101 -> 145
        # order-3 effects but 7,358 -> 165,809 boxes). This budget caps the deployed box TOTAL
        # and spends it on the best held-out drop-gain per box first; the remainder is dropped
        # binarily, exactly as the prune already drops. `0` (the shipped default) disables it
        # and every fit is bit-identical to av37/order-lift. See `prune.rs::apply_box_budget`.
        self.prune_drop_z = prune_drop_z
        self.prune_keep_budget = prune_keep_budget
        # Pin each prune fold's bank to the deploy fit's candidate supports (see
        # `_PRUNE_FOLD_FIDELITY_DEFAULT`). False reproduces every pre-existing fit byte-for-byte.
        self.prune_fold_fidelity = prune_fold_fidelity
        self.prune_guard_z = prune_guard_z
        # --- av39 DOWNWARD guard recalibration --------------------------------------------
        # `prune_guard_z_dn` / `prune_guard_tol_floor` tighten the guard's bar to
        # min(tol, max(tol_floor, z_dn * SE(gap))) — the same bar for the breach TRIGGER and the
        # re-admission ladder's stopping rule. Where the jury is sharp the fixed 5% tolerance is
        # a ~20-SE bar (homesite_conv: SE 0.21%-0.27%) that licenses damage the guard measures
        # perfectly well; the min() means a blunt jury (ohlsson_pp: SE 9.56%) is pinned back at
        # `tol` and bit-identical, so this knob can only ever make the guard fire MORE, never
        # less. `prune_guard_z_dn = 0.0` restores av37/av38 byte-for-byte. See
        # `_PRUNE_GUARD_Z_DN_DEFAULT` for the forensic that sized it.
        self.prune_guard_z_dn = prune_guard_z_dn
        self.prune_guard_tol_floor = prune_guard_tol_floor
        # The K>=3 analogue of `prune_guard`, on out-of-bag evidence from the deploy soup's
        # per-class bag banks (`prune::multiclass_prune_guard`). A SEPARATE flag, and OFF by
        # default, because it is new: `prune_guard`'s default-on state carries the single-output
        # battery behind it, and flipping this one silently would change every shipped K>=3
        # artifact. It reuses `prune_guard_tol`. Ignored entirely for K<=2 and regression.
        self.multiclass_prune_guard = multiclass_prune_guard
        # Bag count of the K>=3 prune's SELECTION fit — the multiclass analogue of the
        # single-output path's `prune_fold_bags`, landing on a different seam because the two
        # prune designs differ. The single-output prune CV fits one model PER FOLD, so fidelity
        # there means the fold model's bag count; the K>=3 prune has NO fold refits at all (it
        # round-robins the selection rows and re-scores ONE selection model's per-class banks
        # per group), so its only fidelity seam is the selection fit itself.
        #
        # The defect is the same: the deploy fit is an `n_bags` soup whose per-class bank is the
        # UNION of what its bags built, the candidate id set comes from the SELECTION model's
        # banks, and the keep-set is applied retain-only — so a table the deploy fit grew and
        # the selection fit never did is dropped WITHOUT EVER BEING SCORED. Raising this gives
        # the selection fit the deployed bagged shape so its bank covers the deploy bank.
        #
        # Default 1 = the shipped unbagged selection fit, byte-identical to every earlier
        # release. OFF by default for the same reason `prune_fold_bags` is: on the single-output
        # path the recovered coverage measured as bank SIZE, not skill. Costs `sel_bags`x the
        # selection fit. Ignored entirely for K<=2 and regression.
        self.multiclass_prune_sel_bags = multiclass_prune_sel_bags
        # K>=3 selection regime (2026-09-07). True = the single-output prune's design ported to
        # multiclass: `prune_n_folds` fold refits, each scored on its held-out fold, voted by
        # `aggregate_prune_selection` with the `prune_drop_z` gate and `prune_keep_budget`, and
        # the `multiclass_prune_guard` measuring the result with an SE-aware bar. False = the
        # historical single train/select split walk (`prune_validation_fraction`), which was
        # measured to drop tables on noise (design/multiclass-design.md §9g).
        self.multiclass_prune_cv = multiclass_prune_cv
        # Floor of the K>=3 guard's bar, as a fraction of the model's honest deviance improvement
        # over the class prior (`dev_null - dev_full` on the evidence rows). The SE term alone
        # demands statistical indistinguishability, which on a sharp instrument means near-zero
        # pruning (fremotor_payfreq 2026-09-07: a 2-SE bar of 0.0001 re-admitted 96% of the
        # bank); this names the price a simpler model is allowed to pay. 0 = SE term only.
        self.multiclass_prune_guard_floor = multiclass_prune_guard_floor
        self.prune_box_budget = prune_box_budget
        self.prune_lambda_boxes = prune_lambda_boxes
        self.prune_table_budget = prune_table_budget
        self.prune_lambda_tables = prune_lambda_tables
        self.prune_table_min_arity = prune_table_min_arity
        # --- PREVIOUSLY-HARDCODED PRUNE INTERNALS (2026-09-04) --------------------------
        # Each of these was a module constant or a function-local literal, so the
        # explainability/accuracy trade-off of the shipped model was governed by numbers no
        # caller could reach. Every default below IS the value that was hardcoded, so a
        # default-constructed estimator is bit-identical to before they were exposed.
        #
        # `prune_min_stability` is the one that matters: the aggregator keeps a table when
        # `mean_gain > prune_min_mean_gain and (kept_rate >= s or positive_rate >= s)`, so `s`
        # is "in how many prune folds must this table show signal". Measured 2026-09-03 on
        # French MTPL frequency (3 seeds, n_bags=8): 0.5 -> 1.0 takes the deployed bank from
        # 58.3 to 37.0 tables and 1216 to 389 boxes for -0.49% +/- 0.66 of test D2 — the best
        # explainability-per-accuracy-point of any prune knob measured, and it halves the
        # seed-to-seed spread in bank size. NOT promoted to the default: one book, three seeds,
        # an effect whose sign moves with the seed. At `prune_n_folds=5` the fold rates quantise
        # to fifths, so it is effectively a 3-position dial.
        self.prune_min_stability = prune_min_stability
        # The gain floor in that same rule — the threshold-parameterised counterpart to
        # `prune_table_budget`'s count cap. `0.0` is the historical `mean_gain > 0.0` test. The
        # native report has always CLAIMED this field and always written a literal 0.0; it now
        # reports what was actually used.
        self.prune_min_mean_gain = prune_min_mean_gain
        # See `_PRUNE_SELECTOR_DEFAULT`.
        self.prune_selector = prune_selector
        self.prune_path_steps = prune_path_steps
        self.prune_path_fraction = prune_path_fraction
        self.prune_path_tolerance = prune_path_tolerance
        # See `_BAND_TOLERANCE_DEFAULT`.
        self.band_tolerance = band_tolerance
        self.band_deviance_cap = band_deviance_cap
        # Rows-per-fold floor that sizes the prune CV: k = max(2, min(prune_n_folds,
        # n // prune_fold_min_rows, n)), so on small data the fold count adapts DOWN.
        self.prune_fold_min_rows = prune_fold_min_rows
        # ES patience for the prune-CV FOLD fits only (deploy fits keep `early_stopping_rounds`).
        # `None` resolves to the shipped 250; an explicit int overrides it.
        self.prune_fold_es_patience = prune_fold_es_patience
        # Minimum honest evidence rows before the set-level no-harm guard will judge at all.
        self.prune_guard_min_rows = prune_guard_min_rows
        # Post-prune slope re-anchor gates: the dead band on |b-1|, and the significance bar the
        # correction must clear before it is applied.
        self.prune_slope_eps = prune_slope_eps
        self.prune_slope_min_z = prune_slope_min_z
        # --- MERGED SPELLINGS (2026-09-04) -----------------------------------------------------
        # Stored verbatim so `get_params`/`clone` round-trip them, then RESOLVED onto the legacy
        # attributes below. Resolution — not reimplementation — is what makes this free: every
        # downstream read, and the entire Rust path, still sees only the old attributes, so a fit
        # is byte-identical whichever spelling the caller used. See `_DEPRECATED`.
        self.prune_size_penalty = prune_size_penalty
        self.early_stopping = early_stopping
        # The conflict test is DISAGREEMENT, not mere presence, and that is load-bearing:
        # `sklearn.clone` reconstructs from `get_params()`, which returns BOTH the new spelling
        # and the legacy attribute this resolution just wrote. A presence test would therefore
        # raise on every clone (i.e. inside any cross-validation), while a disagreement test makes
        # the resolution idempotent — cloning a resolved estimator reproduces it exactly.
        if prune_size_penalty is not None:
            if prune_lambda_tables and float(prune_lambda_tables) != float(prune_size_penalty):
                raise ValueError(
                    f"`prune_size_penalty`={prune_size_penalty} conflicts with "
                    f"`prune_lambda_tables`={prune_lambda_tables} — they are the same lever "
                    f"(`prune_lambda_boxes` is the third spelling). Set one."
                )
            self.prune_lambda_tables = float(prune_size_penalty)
        if early_stopping is not None:
            _es_int = isinstance(early_stopping, int) and not isinstance(early_stopping, bool)
            _clash = (
                (_es_int and early_stopping_rounds not in (500, int(early_stopping)))
                or (not _es_int and early_stopping_adaptive not in (1.5, float(early_stopping)))
            )
            if _clash:
                raise ValueError(
                    f"`early_stopping`={early_stopping} conflicts with the legacy "
                    f"`early_stopping_rounds`={early_stopping_rounds} / "
                    f"`early_stopping_adaptive`={early_stopping_adaptive} pair. Set one."
                )
            # int -> fixed patience; float -> the adaptive ratio. The engine clamps to
            # `clamp(ceil(adaptive * best_round), 50, rounds)`, so the two are one setting.
            if isinstance(early_stopping, bool):
                raise ValueError("`early_stopping` takes an int patience or a float ratio")
            if isinstance(early_stopping, int):
                self.early_stopping_rounds = int(early_stopping)
            else:
                self.early_stopping_adaptive = float(early_stopping)

    def set_params(self, **params: Any) -> "_BaseTBoost":
        result: _BaseTBoost = super().set_params(**params)
        for name in (
            "_model",
            "_multi_model",
            "_precision_warning_emitted_",
            "_multiclass_bagging_warning_emitted_",
            "_multiclass_leaf_refine_warning_emitted_",
            "_cat_indices_",
            "n_features_in_",
            "feature_names_in_",
            "classes_",
            "pruning_report_",
            "cell_refit_report_",
            "graduation_report_",
            "_fit_sample_weight_",
            "_fit_exposure_",
            "_ae_requires_weight_",
            "_ae_requires_exposure_",
            "graduation_validation_",
        ):
            if hasattr(self, name):
                delattr(self, name)
        return result

    def _clear_stale_fit_state(self) -> None:
        """Delete fitted-state attributes a new fit may not re-set, so a direct refit (bypassing
        `set_params`, e.g. calling `.fit()` again on an already-fitted estimator) never leaves
        `feature_names_in_`/`pruning_report_`/`graduation_report_` describing the PREVIOUS fit —
        e.g. a `prune=True` fit followed by `set_params(prune=False)` and a refit used to leave
        the old `pruning_report_` attached, and refitting a DataFrame-fitted estimator on a plain
        ndarray used to leave the old `feature_names_in_`. Called at the top of `_fit_model` /
        `_fit_multiclass`, before either conditionally re-sets any of these. Mirrors (and is also
        invoked independently of) the clear-list in `set_params`.
        """
        for name in (
            "feature_names_in_",
            "pruning_report_",
            "cell_refit_report_",
            "graduation_report_",
            "_fit_sample_weight_",
            "_fit_exposure_",
            "_ae_requires_weight_",
            "_ae_requires_exposure_",
            "graduation_validation_",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _validate_box_budget_reachable(self) -> None:
        """No silent no-op: the deployed-box budget lives inside the prune's keep-set
        application, so it is uninterpretable on an unpruned fit — an unpruned model has no
        keep-set to trim and deploys the whole ensemble. Raise rather than ignore."""
        if int(getattr(self, "prune_box_budget", 0) or 0) > 0 and not self.prune:
            raise ValueError(
                "prune_box_budget has no effect with prune=False and cannot be honored: the "
                "deployed-box budget trims the prune's keep-set, and an unpruned fit has no "
                "keep-set. Enable prune, or leave prune_box_budget at its default (0)."
            )
        if int(getattr(self, "prune_table_budget", 0) or 0) > 0 and not self.prune:
            raise ValueError(
                "prune_table_budget has no effect with prune=False and cannot be honored: the "
                "deployed multi-way table budget trims the prune's keep-set, and an unpruned fit "
                "has no keep-set. Enable prune, or leave prune_table_budget at its default (0)."
            )

    # ---------------------------------------------------------------- deprecation registry ----
    # Old name -> (new name, one-line reason). A deprecated parameter STILL WORKS and still does
    # exactly what it always did: it is forwarded untouched, so every fit is byte-identical and no
    # cached benchmark cell moves. What changes is that using it warns, once, at FIT.
    #
    # WHY AT FIT AND NOT `__init__`: sklearn forbids work in `__init__` (it must only assign), and
    # `clone`/`get_params` re-enter the constructor constantly — warning there would fire on
    # cross-validation machinery rather than on the user's own call.
    #
    # WHY WARN RATHER THAN DELETE: deleting a parameter turns a specific, explanatory error into a
    # bare `TypeError: unexpected keyword argument`. The raises in `_validate_dead_prune_params`
    # below are this library's best property and a deletion would trade one away. A RAISE ALWAYS
    # OUTRANKS A WARN — `_validate_dead_prune_params` runs first, so an unhonourable value still
    # fails loudly rather than being softened into a deprecation notice.
    #
    # Removal is a 1.0 decision, not a 0.x one; until then the alias is kept and the `*`
    # keyword-only barrier in `__init__` is what makes the eventual removal safe.
    _DEPRECATED: dict[str, tuple[str, str]] = {
        "prune_lambda_boxes": (
            "prune_size_penalty",
            "prices boxes; prune_lambda_tables prices tables; both only ever re-pick a waypoint "
            "on one deviance-ordered path, so they are one degree of freedom",
        ),
        # NOT `early_stopping_rounds`. The consensus review proposed merging the pair, and the
        # mechanism is sound — `effective_patience = clamp(ceil(adaptive * best_round), 50,
        # rounds)`, so they are one setting — but `early_stopping_rounds` is the ECOSYSTEM-
        # STANDARD spelling (LightGBM, XGBoost and CatBoost all use it) and is the clamp's own
        # ceiling. Deprecating the familiar half to promote a t-boost-specific merged name is a
        # worse trade than leaving it. Only the t-boost-specific half is deprecated; the merged
        # `early_stopping` spelling exists for callers who want to set one thing.
        "early_stopping_adaptive": (
            "early_stopping",
            "a float `early_stopping` IS the adaptive ratio; the int form sets the rounds "
            "ceiling, and the engine already reconciles them as "
            "clamp(ceil(adaptive * best_round), 50, rounds)",
        ),
        "cat_k": ("cat_leakage", "fold count belongs to the scheme it parameterises, e.g. 'kfold:5'"),
        "cat_n_perms": (
            "cat_leakage",
            "permutation count belongs to the scheme it parameterises, e.g. 'ordered:3'; today "
            "cat_n_perms=0 under kfold is accepted and silently ignored",
        ),
    }

    # ------------------------------------------------------------------- binding report -------
    # Deployed-table count, and the ONLY sanctioned one. `pruning_report_["kept"]` is the
    # PRE-BUDGET keep-set (it does not shrink when `prune_table_budget` trims the artifact) and
    # `pruning_report_["deployed"]` counts only the DENSE tables — the whole order-3 bank is
    # stored as `factored` and is absent from it, understating a large bank by up to ~5.7x. Both
    # are selection intermediates and neither is a size. The artifact is the truth.
    def _deployed_table_count(self) -> int | None:
        import json as _json

        tm = getattr(self, "_model", None) or getattr(self, "_multi_model", None)
        if tm is None:
            return None
        # A table model counts its own bank natively (no JSON round-trip of the whole bank).
        counter = getattr(tm, "deployed_table_count", None)
        if counter is not None:
            return int(counter())
        try:
            j = _json.loads(tm.to_json())["model"]
        except Exception:
            return None
        # An UNPRUNED fit serialises a full ensemble, not a table bank, so there is no deployed
        # table count to report — say None rather than guessing zero.
        try:
            banks = [c["bank"] for c in j["classes"]] if "classes" in j else [j["bank"]]
        except (KeyError, TypeError):
            return None
        return sum(len(b.get("tables") or []) + len(b.get("factored") or []) for b in banks)

    @property
    def binding_report_(self) -> list[dict[str, Any]]:
        """Which parameters you set actually BOUND on this fit.

        One row per parameter set away from its default:

            status = HONOURED    it did what it says
                     INERT       it was reachable but had no effect on this fit
                     OVERRIDDEN  another knob's value voided it — named in `overridden_by`

        This exists because ~43% of this library's surface is CONDITIONALLY live: whether a knob
        does anything depends on the task, the objective, the prune path and on other knobs, and
        none of that is visible in the signature. The failure it is built to stop is real and
        shipped: `prune_min_stability=1.0` is silently INERT whenever `prune_keep_budget` is
        unbounded, because the av37 evidence path re-admits every table the stability gate
        refused — on one benchmark dataset, 405 of 417 of them.

        Deliberately NOT reported as sizes: `kept`, `legacy_selected`, `evidence_admitted`. They
        are pre-budget selection intermediates; they appear only inside a `reason` string. The one
        size field is `deployed_tables`, read off the artifact.
        """
        rows: list[dict[str, Any]] = []
        defaults = _ctor_defaults(type(self))
        rep = getattr(self, "pruning_report_", None) or {}
        pruning_on = bool(getattr(self, "prune", False)) and bool(rep)

        def add(name: str, status: str, reason: str, overridden_by: str | None = None) -> None:
            rows.append({
                "param": name, "value": getattr(self, name, None), "status": status,
                "reason": reason, "overridden_by": overridden_by,
            })

        # the (min_stability, min_mean_gain) x (drop_z, keep_budget) antagonism
        admitted = len(rep.get("evidence_admitted") or [])
        legacy_n = rep.get("legacy_selected")
        drop_z, keep_budget = rep.get("drop_z"), rep.get("keep_budget")
        # FULLY VOIDED is the condition worth interrupting for, and the library already computes
        # the signal: `evidence_budget_bound` is False exactly when the admission budget never
        # constrained the evidence path — i.e. every ambiguous table the stability/mean-gain gate
        # refused was taken back. Combined with `admitted > 0` that means the gate's refusals were
        # all reversed and the deployed bank is what it would have been with the gate untouched.
        # (Verified on a 25-level fixture: at BOTH keep_budget 32 and 1e6, `min_stability=1.0`
        # deploys 7 tables — identical to not setting it at all.) When the budget DOES bind the
        # gate had partial effect, which is not a no-op and is not worth a warning.
        evidence_open = (
            drop_z is not None
            and rep.get("evidence_budget_bound") is False
        )
        for gate in ("prune_min_stability", "prune_min_mean_gain"):
            if gate not in defaults or getattr(self, gate, None) == defaults[gate]:
                continue
            if not pruning_on:
                add(gate, "INERT", "prune=False, so no table selection ran", "prune")
            elif evidence_open and admitted:
                add(gate, "OVERRIDDEN",
                    f"the evidence path re-admitted {admitted} table(s) this gate refused; its "
                    f"admission budget ({keep_budget}) exceeds the gate's own keep-set "
                    f"({legacy_n}), so tightening the gate cannot remove a table",
                    "prune_keep_budget")
            else:
                add(gate, "HONOURED", "the gate decided the keep-set")

        # every other prune_* knob is inert when pruning is off
        if not pruning_on:
            for name in defaults:
                if name.startswith("prune_") and getattr(self, name, None) != defaults[name] \
                        and not any(r["param"] == name for r in rows):
                    add(name, "INERT", "prune=False, so the prune stage did not run", "prune")

        # size caps: a cap above the bank never binds (the count is only read when a cap is set)
        for cap, what in (("prune_table_budget", "tables"), ("prune_box_budget", "boxes")):
            v = getattr(self, cap, None)
            if cap in defaults and v not in (None, defaults[cap]) and pruning_on and v:
                deployed = self._deployed_table_count() if what == "tables" else None
                if what == "tables" and deployed is not None and deployed < int(v):
                    add(cap, "INERT",
                        f"the deployed bank holds {deployed} table(s), below the cap of {v}")
                else:
                    add(cap, "HONOURED", f"cap applied to deployed {what}")

        # monotone: disclose WHICH axes each sign was expanded onto (the fix of 2026-09-04 binds
        # every axis a categorical emits, which is a policy choice the caller should see)
        mono = getattr(self, "monotone_constraints", None)
        if mono:
            _fn = getattr(self, "feature_names_in_", None)      # a numpy array; `or []` is ambiguous
            names = [] if _fn is None else list(_fn)
            add("monotone_constraints", "HONOURED",
                f"signs expanded onto every axis of each named feature"
                + (f"; input features: {names}" if names else ""))

        # deprecated spellings still work, but say so here as well as in the warning
        for old, (new, _why) in self._DEPRECATED.items():
            if old in defaults and getattr(self, old, None) != defaults[old]:
                add(old, "HONOURED", f"deprecated spelling; prefer `{new}`")

        return rows

    def _warn_voided_params(self) -> None:
        """Warn when a parameter you set was VOIDED by another parameter's value.

        This is the proactive half of `binding_report_`, and it exists because the report alone
        only helps someone who already suspects a problem. The failure being prevented is a user
        spending an afternoon tuning a knob that cannot do anything: set `prune_min_stability` to
        1.0 with an unbounded `prune_keep_budget` and the model is byte-identical to not setting
        it at all, because the evidence path re-admits every table the gate refused.

        ONLY the `OVERRIDDEN` class warns, deliberately. `INERT` is the weaker finding — a knob
        that simply did not bite on this data, most often benignly (a size cap set above the bank
        did nothing, and that is fine). Warning on it would fire on ordinary fits: measured on the
        insur-arena campaign, the depth gate arms `prune_table_budget=100` on a lifted candidate
        whose bank holds 24 tables, which is INERT and entirely harmless. `OVERRIDDEN` is the
        class where another knob is actively cancelling yours, which is never benign and is always
        worth interrupting for. `INERT` stays in `binding_report_` and in `check_bindings()`.
        """
        import warnings as _warnings

        for row in self.binding_report_:
            if row["status"] != "OVERRIDDEN":
                continue
            _warnings.warn(
                f"`{row['param']}={row['value']!r}` had NO EFFECT on this fit: "
                f"{row['reason']}. Set `{row['overridden_by']}` differently, or drop "
                f"`{row['param']}`. See `binding_report_` for the full picture.",
                RuntimeWarning, stacklevel=3,
            )

    def check_bindings(self) -> None:
        """Raise if anything you set was INERT or OVERRIDDEN on this fit.

        The strict counterpart of `binding_report_`, for a fit whose parameters are part of a
        filed or otherwise defended model, where a silently ignored setting is a defect rather
        than a curiosity. A benchmark sweep would not call this; a deployment should.
        """
        bad = [r for r in self.binding_report_ if r["status"] in ("INERT", "OVERRIDDEN")]
        if bad:
            detail = "; ".join(
                f"{r['param']}={r['value']!r} {r['status']}"
                + (f" (voided by {r['overridden_by']})" if r["overridden_by"] else "")
                + f": {r['reason']}"
                for r in bad
            )
            raise ValueError(f"{len(bad)} parameter(s) did not bind on this fit — {detail}")

    def _resolve_cat_leakage(self) -> dict[str, Any]:
        """Resolve `cat_leakage` into the `(scheme, cat_n_perms, cat_k)` triple the engine takes.

        The scheme may now carry its own parameter — `"kfold:5"`, `"ordered:3"` — which is the
        merged spelling of what used to be three independent knobs. They were never independent:
        `cat_k` is read ONLY by the kfold arm and `cat_n_perms` ONLY by the ordered arm of
        `parse_cat_leakage`, and each is validated only inside its own arm, so today
        `cat_n_perms=0` under kfold is accepted and silently ignored. Attaching the number to the
        scheme makes the invalid combinations unrepresentable rather than merely unread.

        The legacy spelling still works and is still exact — this returns the same triple either
        way, so a fit is byte-identical.
        """
        raw = self.cat_leakage
        n_perms, k = int(self.cat_n_perms), int(self.cat_k)
        if isinstance(raw, str) and ":" in raw:
            scheme, _, arg = raw.partition(":")
            scheme, arg = scheme.strip(), arg.strip()
            try:
                val = int(arg)
            except ValueError:
                raise ValueError(
                    f"cat_leakage={raw!r}: the part after ':' must be an integer, e.g. 'kfold:5'"
                ) from None
            if scheme == "kfold":
                k = val
            elif scheme == "ordered":
                n_perms = val
            else:
                raise ValueError(
                    f"cat_leakage={raw!r}: only 'kfold' and 'ordered' take a parameter "
                    f"(got scheme {scheme!r}); 'loo' takes none."
                )
            # An explicit legacy value that DISAGREES with the tag is a genuine conflict; one that
            # agrees is what `clone` reproduces, so only disagreement raises.
            legacy = self.cat_k if scheme == "kfold" else self.cat_n_perms
            default = 5 if scheme == "kfold" else 1
            if int(legacy) not in (default, val):
                raise ValueError(
                    f"cat_leakage={raw!r} conflicts with "
                    f"{'cat_k' if scheme == 'kfold' else 'cat_n_perms'}={legacy}. Set one."
                )
            raw = scheme
        return {"cat_leakage": raw, "cat_n_perms": n_perms, "cat_k": k}

    def _warn_deprecated_params(self) -> None:
        """Warn once per fit for each deprecated parameter set away from its default."""
        import warnings as _warnings

        defaults = _ctor_defaults(type(self))
        for old, (new, why) in self._DEPRECATED.items():
            if old not in defaults:
                continue
            cur = getattr(self, old, None)
            if cur is None or cur == defaults[old]:
                continue                      # untouched: say nothing
            _warnings.warn(
                f"`{old}` is deprecated and will be removed in 1.0; use `{new}` instead "
                f"({why}). Your value is still being honoured exactly as before.",
                DeprecationWarning, stacklevel=3,
            )

    def _validate_dead_prune_params(self, *, multiclass: bool) -> None:
        """No silent parameter no-ops: raise rather than ignore an explicit value the fit cannot
        actually honor. `prune_refit_full` is architecturally dead on EVERY prune path —
        contribution-stability pruning (see `_fit_and_prune`) always fits the deployed model on
        all rows and applies the keep-set to that same fit, so there is no separate
        subset-pruned fit left to conditionally refit; `fit_prune_selection` /
        `fit_multiclass_pruned` (see `_t_boost.pyi`) take no parameter it could map to.
        `prune_validation_fraction` only has a target on the multiclass (K>=3) path
        (`fit_multiclass_pruned`'s single train/select split, honored below in
        `_fit_multiclass`); the single-output (regressor/binary) path uses K-fold CV via
        `fit_prune_selection`'s `k_folds`/`fold_of` instead — a fraction has no analogous
        parameter there either. `prune_se_rule` is the same shape one level down: it reaches the
        single-output path's native call but not its OUTCOME, because the per-fold selections it
        steers are aggregated by a rule that ORs `kept_rate` against `positive_rate` (see the
        long note at that check). Called only when `self.prune` is True (both callers gate on
        it).
        """
        if int(getattr(self, "prune_box_budget", 0) or 0) < 0:
            raise ValueError(
                "prune_box_budget must be >= 0 (0 disables the deployed-box budget)."
            )
        lam = float(getattr(self, "prune_lambda_boxes", 0.0) or 0.0)
        if not math.isfinite(lam) or lam < 0.0:
            raise ValueError(
                "prune_lambda_boxes must be a finite value >= 0 (0.0 disables the "
                "selection-time box price); got "
                f"{getattr(self, 'prune_lambda_boxes', None)!r}."
            )
        if int(getattr(self, "prune_table_budget", 0) or 0) < 0:
            raise ValueError(
                "prune_table_budget must be >= 0 (0 disables the deployed multi-way table "
                "budget)."
            )
        lam_t = float(getattr(self, "prune_lambda_tables", 0.0) or 0.0)
        if not math.isfinite(lam_t) or lam_t < 0.0:
            raise ValueError(
                "prune_lambda_tables must be a finite value >= 0 (0.0 disables the "
                "selection-time table price); got "
                f"{getattr(self, 'prune_lambda_tables', None)!r}."
            )
        # The floor is validated even when both table knobs are off, so a typo cannot lie dormant
        # until the one run that arms them. `MAX_ORDER` is 8; a floor above it would price
        # nothing and a floor below 1 would price the whole bank rather than the multi-way part.
        arity = int(getattr(self, "prune_table_min_arity", _PRUNE_TABLE_MIN_ARITY_DEFAULT) or 0)
        if not 1 <= arity <= _MAX_ORDER:
            raise ValueError(
                f"prune_table_min_arity must be in 1..{_MAX_ORDER} (the explainability bar is "
                f"{_PRUNE_TABLE_MIN_ARITY_DEFAULT}); got "
                f"{getattr(self, 'prune_table_min_arity', None)!r}."
            )
        # --- the prune internals exposed 2026-09-04 -------------------------------------
        stab = float(getattr(self, "prune_min_stability", _PRUNE_MIN_STABILITY_DEFAULT))
        if not (math.isfinite(stab) and 0.0 < stab <= 1.0):
            raise ValueError(
                "prune_min_stability must be a finite value in (0, 1] — the fraction of prune "
                "folds a table must show signal in. The fold rates quantise to "
                "1/prune_n_folds, so at the default 5 folds the distinct settings are "
                f"(0.4, 0.6] = 3/5, (0.6, 0.8] = 4/5 and (0.8, 1.0] = 5/5; got {stab!r}."
            )
        gain_floor = float(getattr(self, "prune_min_mean_gain", _PRUNE_MIN_MEAN_GAIN_DEFAULT))
        if not (math.isfinite(gain_floor) and gain_floor >= 0.0):
            raise ValueError(
                "prune_min_mean_gain must be a finite value >= 0 (0.0 is the historical "
                f"`mean_gain > 0` rule); got {gain_floor!r}."
            )
        if int(getattr(self, "prune_fold_min_rows", _MIN_PRUNE_FOLD_ROWS)) < 1:
            raise ValueError(
                "prune_fold_min_rows must be >= 1: it is the rows-per-fold floor that sizes "
                "the prune CV (k = max(2, min(prune_n_folds, n // this, n)))."
            )
        patience = getattr(self, "prune_fold_es_patience", None)
        if patience is not None and int(patience) < 1:
            raise ValueError(
                "prune_fold_es_patience must be >= 1, or None to resolve to the shipped 250."
            )
        if int(getattr(self, "prune_guard_min_rows", _PRUNE_GUARD_MIN_ROWS)) < 1:
            raise ValueError(
                "prune_guard_min_rows must be >= 1: it is the honest-evidence-row floor below "
                "which the set-level no-harm guard declines to judge."
            )
        eps = float(getattr(self, "prune_slope_eps", _SLOPE_EPS))
        if not (math.isfinite(eps) and eps >= 0.0):
            raise ValueError(
                f"prune_slope_eps must be a finite value >= 0 (the |b-1| dead band); got {eps!r}."
            )
        min_z = float(getattr(self, "prune_slope_min_z", _SLOPE_MIN_Z))
        if not (math.isfinite(min_z) and min_z >= 0.0):
            raise ValueError(
                "prune_slope_min_z must be a finite value >= 0 (the significance bar the slope "
                f"correction must clear); got {min_z!r}."
            )
        # Both of these reach only `aggregate_prune_selection`, which the K>=3 path does not
        # use — it selects on one train/select split with no per-fold vote to threshold. Same
        # discipline as `prune_fold_fidelity` below: raise rather than accept a value that
        # cannot be honored.
        legacy_split = multiclass and not bool(getattr(self, "multiclass_prune_cv", True))
        if legacy_split and stab != _PRUNE_MIN_STABILITY_DEFAULT:
            raise ValueError(
                "prune_min_stability has no effect on the multiclass (K>=3) prune path and "
                "cannot be honored: it is the per-fold stability bar in the single-output "
                "aggregator, and that path selects on ONE train/select split with no fold vote "
                f"to threshold. Leave it at its default ({_PRUNE_MIN_STABILITY_DEFAULT})."
            )
        if legacy_split and gain_floor != _PRUNE_MIN_MEAN_GAIN_DEFAULT:
            raise ValueError(
                "prune_min_mean_gain has no effect on the multiclass (K>=3) prune path and "
                "cannot be honored: it is the gain floor in the single-output aggregator's "
                f"keep rule. Leave it at its default ({_PRUNE_MIN_MEAN_GAIN_DEFAULT})."
            )
        if self.prune_refit_full:
            raise ValueError(
                "prune_refit_full has no effect and cannot be honored: contribution-stability "
                "pruning always fits the deployed model on all rows and applies the keep-set to "
                "that same fit — there is no separate subset-pruned fit to conditionally refit, "
                "and the native prune entry points take no parameter it could map to. Leave this "
                "at its default (False); the parameter will be removed in a future release."
            )
        # `prune_se_rule` has a target on this path — it is passed to `fit_prune_selection` and
        # each prune fold's backward walk really does apply the SE band (`prune.rs`'s
        # `threshold = min_obj + cfg.se_rule * path[min_i].se`). What it does NOT have is a route
        # to the DEPLOYED keep-set. The fold walks reach the aggregator only as `kept_rate`, and
        # `aggregate_prune_selection`'s rule is
        #     mean_gain > 0.0 && (kept_rate >= min_stability || positive_rate >= min_stability)
        # so `kept_rate` is OR'd against `positive_rate` behind an se_rule-independent
        # `mean_gain > 0` precondition. Since the walk drops greedily on deviance, it keeps
        # exactly the tables whose removal hurts — i.e. the folds where `kept_rate` counts are
        # very nearly the folds where `positive_rate` counts, and the disjunction is already
        # satisfied without it.
        #
        # MEASURED (2026-09-03, French MTPL frequency, single-bag): `prune_se_rule` in
        # {0, 0.5, 1, 2, 10, 100} returns bit-identical banks and identical test deviance at 8k,
        # 25k, 80k and 406k training rows, with `prune_drop_z` on and off. se_rule = 100 collapses
        # every fold's walk to its simplest waypoint and the deployed bank does not move.
        #
        # This is the same finding the 2026-07-11 review raised (MEDIUM: "prune_se_rule ... is
        # hardcoded to se_rule=0.0 in the single-output call"). The remediation wired the value
        # through, but `e7e4945` had already changed the keep rule from
        # `kept_rate >= min_stability || (mean_gain > 0.0 && positive_rate >= min_stability)` —
        # where `kept_rate` alone was decisive — to the form above, so the wiring restored the
        # plumbing into a path whose outcome no longer depends on it. Raising here is what the
        # review actually asked for: no silent no-op.
        if not multiclass and float(self.prune_se_rule) != 0.0:
            raise ValueError(
                "prune_se_rule has no effect on the regressor/binary-classifier prune path and "
                "cannot be honored: this path aggregates per-fold selections, and the SE band "
                "reaches that aggregation only through `kept_rate`, which the keep rule ORs "
                "against `positive_rate` (measured: se_rule in {0, 0.5, 1, 2, 10, 100} gives "
                "bit-identical banks). It IS honored on the multiclass (K>=3) path, which "
                "selects on one train/select split. For a smaller bank here, use "
                "prune_table_budget (a cap on deployed multi-way tables, ranked by held-out "
                "drop-gain) or prune_box_budget; leave prune_se_rule at its default (0.0)."
            )
        if multiclass and not legacy_split and self.prune_validation_fraction != 0.15:
            raise ValueError(
                "prune_validation_fraction has no effect on the multiclass (K>=3) CV prune path "
                "and cannot be honored: selection now runs on prune_n_folds fold refits "
                "(multiclass_prune_cv=True). Set prune_n_folds instead, or pass "
                "multiclass_prune_cv=False for the legacy single train/select split."
            )
        if not multiclass and self.prune_validation_fraction != 0.15:
            raise ValueError(
                "prune_validation_fraction has no effect on the regressor/binary-classifier "
                "prune path and cannot be honored: it only governs the multiclass (K>=3) path's "
                "single train/select split. This path uses K-fold CV instead, sized by "
                "prune_n_folds — set that instead, or leave prune_validation_fraction at its "
                "default (0.15)."
            )
        floor = float(getattr(self, "multiclass_prune_guard_floor", _MULTICLASS_GUARD_FLOOR_DEFAULT))
        if not (math.isfinite(floor) and floor >= 0.0):
            raise ValueError(
                f"multiclass_prune_guard_floor must be a finite fraction >= 0 (got {floor!r})."
            )
        if int(self.multiclass_prune_sel_bags) < 1:
            raise ValueError(
                f"multiclass_prune_sel_bags must be >= 1 (got "
                f"{self.multiclass_prune_sel_bags!r}): it is the bag count of the K>=3 prune's "
                "selection fit, and 1 is the shipped unbagged fidelity."
            )
        if not multiclass and int(self.multiclass_prune_sel_bags) != 1:
            raise ValueError(
                "multiclass_prune_sel_bags has no effect on the regressor/binary-classifier "
                "prune path and cannot be honored: it is the bag count of the multiclass "
                "(K>=3) prune's SELECTION fit, and this path has per-fold refits instead. "
                "Leave it at its default (1)."
            )
        if multiclass and bool(getattr(self, "prune_fold_fidelity", False)):
            raise ValueError(
                "prune_fold_fidelity has no effect on the multiclass (K>=3) prune path and "
                "cannot be honored: that path selects on ONE train/select split "
                "(`fit_multiclass_pruned`) and has no per-fold refits to pin a candidate bank "
                "into. Leave it at its default (False)."
            )

    def _validate_cat_channels(self, *, multiclass: bool) -> None:
        """Reject a ``cat_channels`` request this fit cannot honor (no silent no-ops).

        ``"class_freq"`` (P3 per-class frequency channels) exists to replace the ORDINAL
        label-mean categorical encoding that only the K>=3 softmax path produces — it has no
        meaning anywhere else:

        * regression: there are no classes at all, so the request is uninterpretable -> raise.
        * binary (K=2): the mean-TS channel is already fit against the 0/1 class indicator, so
          it IS the class-1 frequency channel exactly (and a class-0 channel would be its
          complement, carrying no extra partition). The request is therefore SATISFIED by the
          default behavior rather than ignored, so it does not raise.

        Called from ``_fit_model`` (regressor/binary) and ``_fit_multiclass`` (K>=3).
        """
        channels = self.cat_channels
        if channels is None:
            return
        names = {str(c).strip().lower() for c in channels}
        unknown = names - {"mean", "count", "class_freq", "classfreq", "class"}
        if unknown:
            raise ValueError(
                "cat_channels entries must be 'mean', 'count' or 'class_freq'; got "
                f"{sorted(unknown)!r}"
            )
        wants_class_freq = bool(names & {"class_freq", "classfreq", "class"})
        if wants_class_freq and not multiclass and isinstance(self, RegressorMixin):
            raise ValueError(
                "cat_channels='class_freq' has no effect on a regression fit and cannot be "
                "honored: the per-class frequency channels replace the ORDINAL label-mean "
                "categorical encoding, which only exists on the multiclass (K>=3) softmax "
                "path. Use ['mean'] or ['mean','count'] here."
            )

    def _validate_multiclass_inert_params(self) -> None:
        """No silent parameter no-ops: the native multiclass (K>=3) path never applies the
        fully-corrective ridge leaf refit, DART, intercept/slope re-anchoring, or the
        scale-invariant lambda rescale, regardless of these constructor
        settings (`engine::boost::fit_multiclass`'s own doc, and this class's "Honesty notes").

        The raise-family params (`ridge_refit_l2`/`ridge_refit_max_iter`, `dart_drop_rate`,
        `reanchor`/`reanchor_slope`, `lambda_scale_invariant`) default to their own no-op value
        (``None``/the identity constant/``False``) — reaching this method with a non-default
        value means the caller explicitly asked for something that cannot be honored, so it
        RAISES: there is no ordinary multiclass fit this would break.

        `n_bags`/`bag_subsample` (outer bags souped per class) and
        `leaf_refine_steps`/`leaf_refine_backtracks` (per-round Jacobi re-linearization with a
        joint-deviance backtrack) are HONORED for K>=3 as of 2026-07-14,
        `subsample`/`mvs_min_rows` (§06.5 MVS row sampling) as of 2026-08-23, and
        `cell_refit_base`/`cell_refit_gamma` (the §G1 OOB cell refit, as K decoupled
        diagonal-Hessian cell solves accepted by ONE joint backtrack on the multinomial loss —
        `engine::boost::attach_multiclass_cell_correction`) as of 2026-08-25; none of them are
        validated here. As on the single-output path the refit needs a bag partition, so
        `cell_refit_base` with `n_bags=1` is silently inert on BOTH paths alike.

        Called once, near the top of `_fit_multiclass`, before either the pruned or unpruned
        branch (both are equally affected).
        """

        if self.graduation_high_order_alpha != 0:
            raise ValueError("higher-order smoothing does not support multiclass (K>=3)")

        def _raise(name: str, mechanism: str) -> None:
            raise ValueError(
                f"{name} has no effect on multiclass (K>=3) fits: the native multiclass path "
                f"is v1's base boosting loop only and does not apply {mechanism}. Leave "
                f"{name} at its default, or use a binary (K=2) classifier or "
                "TBoostRegressor, where it is honored."
            )

        if self.ridge_refit_l2 is not None:
            _raise("ridge_refit_l2", "the fully-corrective ridge leaf refit")
        if self.ridge_refit_max_iter != 5:
            _raise("ridge_refit_max_iter", "the fully-corrective ridge leaf refit")
        if self.dart_drop_rate is not None:
            _raise("dart_drop_rate", "DART")
        if self.reanchor is not None:
            _raise("reanchor", "intercept re-anchoring")
        if self.reanchor_slope is not None:
            _raise("reanchor_slope", "affine slope recalibration")
        # `subsample`/`mvs_min_rows` LEFT the raise family (2026-08-23): §06.5 MVS row sampling
        # is now wired for K>=3 (`engine::boost::mc_sample_rows`). One row draw per round is
        # shared by all K class trees, weighted by the JOINT gradient/hessian norm over classes
        # — the softmax couples the classes to the same round-start state, so per-class
        # independent draws would give one round's K trees disjoint views of the data. The
        # estimator, seed stream and 1/p_i correction are the single-output sampler's verbatim.
        # `subsample=None` (the default) is `Sampling::Full` ⇒ byte-identical to before.
        if self.lambda_scale_invariant:
            _raise("lambda_scale_invariant", "the scale-invariant lambda rescale")

        # n_bags/bag_subsample ARE honored for K>=3 as of 2026-07-14: the multiclass path
        # dispatches on `EnsembleSpec` exactly like the single-output path (subagged per-bag
        # fits souped per class — `engine::boost::fit_multiclass_bagged`), so no warning.
        # leaf_refine_steps has no effect for K>=3: a Jacobi re-linearization measured
        # 0-wins/2-ties/3-losses as a drop-in (2026-07-14 battery) and was removed, so no
        # warning fires for the shipped leaf_refine_steps=4.

    def _resolve_n_jobs(self) -> int | None:
        """Resolve n_jobs per the sklearn convention into a positive thread count.

        ``None`` keeps the native default (all cores); ``-1`` means all cores, and a
        negative ``n`` means ``cpu_count + 1 + n`` (so ``-1`` is all, ``-2`` all-but-one).
        Negative values must be translated here because the native layer takes an
        unsigned count and would otherwise raise ``OverflowError``.
        """
        n = self.n_jobs
        if n is None:
            return None
        n = int(n)
        if n < 0:
            import os

            cores = os.cpu_count() or 1
            n = max(1, cores + 1 + n)
        return n

    def _resolve_monotone(
        self, n_features: int, feature_names: list[str] | None
    ) -> list[int] | None:
        """Resolve ``monotone_constraints`` into a positional sign vector (-1/0/+1).

        Accepts a length-``n_features`` sequence (positional) or a dict keyed by feature
        name (matched against ``feature_names`` or the canonical ``f{i}``) or integer
        index. Returns ``None`` when no constraint is active.
        """
        mc = self.monotone_constraints
        if mc is None:
            return None
        signs = [0] * n_features
        if isinstance(mc, dict):
            name_to_idx = (
                {name: i for i, name in enumerate(feature_names)}
                if feature_names is not None
                else {}
            )
            for key, val in mc.items():
                if isinstance(key, str):
                    if key in name_to_idx:
                        idx = name_to_idx[key]
                    elif key.startswith("f") and key[1:].isdigit():
                        idx = int(key[1:])
                    else:
                        raise ValueError(f"unknown monotone feature {key!r}")
                else:
                    idx = int(key)
                if not 0 <= idx < n_features:
                    raise ValueError(f"monotone feature index {idx} out of range")
                signs[idx] = int(val)
        else:
            seq = list(mc)
            if len(seq) != n_features:
                raise ValueError(
                    f"monotone_constraints length {len(seq)} != n_features {n_features}"
                )
            signs = [int(v) for v in seq]
        for s in signs:
            if s not in (-1, 0, 1):
                raise ValueError(f"monotone sign {s} must be -1, 0, or 1")
        return signs if any(signs) else None

    def _resolve_categorical(
        self, n_features: int, feature_names: list[str] | None
    ) -> list[int]:
        cats = self.categorical_features
        if cats is None:
            return []
        if isinstance(cats, (str, int, np.integer)):
            seq: list[Any] = [cats]
        else:
            seq = list(cats)
        if all(isinstance(v, (bool, np.bool_)) for v in seq):
            if len(seq) != n_features:
                raise ValueError(
                    f"categorical_features mask length {len(seq)} != n_features {n_features}"
                )
            return [i for i, flag in enumerate(seq) if bool(flag)]

        name_to_idx = (
            {name: i for i, name in enumerate(feature_names)}
            if feature_names is not None
            else {}
        )
        idxs: list[int] = []
        for key in seq:
            if isinstance(key, str):
                if key in name_to_idx:
                    idx = name_to_idx[key]
                elif key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                else:
                    raise ValueError(f"unknown categorical feature {key!r}")
            else:
                idx = int(key)
            if not 0 <= idx < n_features:
                raise ValueError(f"categorical feature index {idx} out of range")
            idxs.append(idx)
        return sorted(set(idxs))

    def _split_columns(
        self, X: Any, cat_idx: list[int], feature_names: list[str] | None, *, coded: bool = False
    ) -> tuple[np.ndarray, Any, list[str] | None]:
        if is_polars_eager(X):
            # Polars-native split: numeric block lands F-contiguous float32 (polars'
            # to_numpy default order — the native layer's cheap no-transpose ingest),
            # categorical columns become string lists with nulls -> _CAT_MISSING.
            numeric_x, cat_x, needs_warn = split_polars_columns(X, cat_idx, coded=coded)
            if needs_warn and not getattr(self, "_precision_warning_emitted_", False):
                warnings.warn(
                    "t-boost converts input features to float32 before fitting/scoring",
                    PrecisionWarning,
                    stacklevel=3,
                )
                self._precision_warning_emitted_ = True
            axis_names = feature_names
            if feature_names is not None and cat_idx:
                cat_set = set(cat_idx)
                numeric_idx = [i for i in range(len(feature_names)) if i not in cat_set]
                axis_names = [feature_names[i] for i in numeric_idx] + [
                    feature_names[j] for j in cat_idx
                ]
            return numeric_x, cat_x, axis_names

        if not cat_idx:
            return self._as_float32_2d_once(X), None, feature_names

        n_features = _n_columns(X)
        cat_set = set(cat_idx)
        numeric_idx = [i for i in range(n_features) if i not in cat_set]
        numeric_x = self._as_float32_2d_once(_numeric_subset(X, numeric_idx))
        cat_x = [
            _column_as_codes(X, j) if coded else _column_as_str_list(X, j) for j in cat_idx
        ]
        for pos, col in zip(cat_idx, cat_x):
            if len(col[0] if coded else col) != numeric_x.shape[0]:
                raise ValueError(
                    f"categorical feature {pos} has {len(col)} rows but X has {numeric_x.shape[0]}"
                )
        axis_names = None
        if feature_names is not None:
            axis_names = [feature_names[i] for i in numeric_idx] + [
                feature_names[j] for j in cat_idx
            ]
        return numeric_x, cat_x, axis_names

    def _resolve_fit_vectors(
        self, X: Any, vectors: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Resolve fit-time per-row arguments against a polars ``X`` (rustystats-style).

        Collects a LazyFrame, resolves each vector given as a column-name string (or polars
        Series) to numpy, and drops the consumed columns from the feature frame — naming
        ``y="ClaimCount", exposure="Exposure"`` makes every REMAINING column a feature.
        Non-polars ``X`` passes through untouched (a column-name string then raises).
        """
        X = collect_frame(X)
        frame = X if is_polars_eager(X) else None
        consumed: list[str] = []
        resolved = {
            name: resolve_vector(frame, arg, name, consumed)
            for name, arg in vectors.items()
        }
        if consumed:
            X = X.drop(list(dict.fromkeys(consumed)))
        return X, resolved

    def _resolve_explain_inputs(
        self, X: Any, sample_weight: Any, exposure: Any
    ) -> tuple[Any, Any, Any]:
        """Resolve `tables()`-time polars inputs: a LazyFrame collects only the model's
        feature columns plus any column-name mass arguments, then string/Series
        ``sample_weight``/``exposure`` resolve against the collected frame. Non-polars
        input passes through untouched."""
        if is_polars_frame(X):
            names = getattr(self, "feature_names_in_", None)
            needed = None
            if names is not None:
                needed = [str(n) for n in names] + [
                    v for v in (sample_weight, exposure) if isinstance(v, str)
                ]
            X = collect_frame(X, needed)
        frame = X if is_polars_eager(X) else None
        sample_weight = resolve_vector(frame, sample_weight, "sample_weight")
        exposure = resolve_vector(frame, exposure, "exposure")
        return X, sample_weight, exposure

    def _serve_kwargs(self, X: Any, model: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """The design and categorical keyword for a native predict on ``model``. Table models
        take categoricals as codes into distinct labels (``cat_codes``), the cheap form; tree
        models and polars input keep per-row labels (``cat_x``)."""
        coded = isinstance(model, (_TableModel, _MultiClassTableModel))
        x32, cat = self._serve_design(X, coded=coded)
        if cat and isinstance(cat[0], tuple):
            return x32, {"cat_codes": cat}
        return x32, {"cat_x": cat}

    def _serve_design(self, X: Any, *, coded: bool = False) -> tuple[np.ndarray, Any]:
        _reject_sparse(X)
        if is_polars_frame(X):
            # Name-based serve (rustystats semantics): pick the model's feature columns in
            # their fit-time order — extra columns are ignored, input column order is
            # irrelevant, missing columns raise. A LazyFrame collects only these columns.
            # A model fitted without names (plain ndarray) falls back to the positional
            # contract below, after a full collect.
            names = getattr(self, "feature_names_in_", None)
            if names is not None:
                X = select_feature_columns(X, [str(n) for n in names])
            else:
                X = collect_frame(X)
        n_features = _n_columns(X)
        expected = int(getattr(self, "n_features_in_"))
        if n_features != expected:
            raise ValueError(f"X has {n_features} features but model expects {expected}")

        cat_idx = getattr(self, "_cat_indices_", None)
        if cat_idx is None:
            cat_idx = self._resolve_categorical(n_features, _feature_names_from_x(X))
            self._cat_indices_ = cat_idx
        numeric_x, cat_x, _ = self._split_columns(
            X, list(cat_idx), _feature_names_from_x(X), coded=coded
        )
        return numeric_x, cat_x

    def _default_fit_pool(self, n_features: int) -> int | None:
        """Fit-pool width when ``n_jobs=None`` (P1.1 diagnosis, plan/speed-campaign-2.md).

        The boost round loop's parallel tasks are SMALL (a 200k-row round splits its
        histogram work into ~180us chunks), so wide pools lose to per-round fork-join
        wake/sync churn: a single bag PEAKS at ~4 threads and is no faster at 22 than at
        1; an 8-bag 406k MTPL fit runs 18% faster on a 12-thread pool than on all 22
        cores (full default fit 101.2s -> 88.0s), brvehins1 29% faster at 8 threads.
        Width therefore scales with the OUTER parallelism (bags) and the per-round task
        size (features) — wide-feature shapes (allstate, 130 cols) saturate to all cores
        and are unharmed. Explicit ``n_jobs`` always wins; thread count never changes
        results (the determinism guarantee), so this is a pure scheduling choice.
        """
        if self.n_jobs is not None:
            return self._resolve_n_jobs()
        import os

        cores = os.cpu_count() or 1
        bags = max(int(self.n_bags), 1)
        return min(cores, max(4, -(-3 * bags // 2), int(n_features)))

    def _measure_kwargs(self) -> dict[str, Any]:
        """The purification measure this estimator fits under, as binding keyword arguments.

        ``ref_measure=None`` resolves to ``"exposure"`` here, in ONE place, so every binding
        that builds a bank at fit time (the prune-CV folds, the deploy keep-set, the guard's
        out-of-bag evidence, the multiclass prune) sees the same explicit name — the binding's
        own ``None`` default is the legacy ``product_marginals`` and must never be relied on.
        """
        name = self.ref_measure if self.ref_measure is not None else "exposure"
        floor = float(self.measure_floor)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError(f"measure_floor must be finite and > 0, got {self.measure_floor!r}")
        return {"ref_measure": str(name), "laplace": 1.0, "measure_floor": floor}

    def actual_vs_expected(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        exposure: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Actual versus expected by rating-factor level, for every feature (2026-09-06).

        This shows where observed and predicted totals differ. Exact factor-level balance
        is not a general property of boosted models or of every GLM/link/penalty combination. One entry per raw feature, aggregated over the merged-grid
        cells the deployed tables actually use (cell 0 is the missing-value cell), with
        ``actual`` = Σ weight·y, ``expected`` = Σ weight·prediction (the prediction already
        carries ``exposure`` for log-link fits), ``mass`` = Σ weight·exposure and ``rows``
        per cell. ``ae`` is ``actual / expected`` (``nan`` where expected is zero).

        Only available on a pruned fit (``prune=True``, the default), whose deployed model
        is a set of tables with a fixed grid. Pass ``sample_weight`` and ``exposure`` explicitly
        when the fit used them, including on the training data. Row count does not establish
        alignment: fitted vectors are never silently reused. Use explicit unit vectors when
        an unweighted or unit-exposure evaluation is intended. ``y`` may name a polars column.
        """
        model = getattr(self, "_model", None)
        if model is None or not hasattr(model, "cell_indices"):
            raise ValueError(
                "actual_vs_expected needs a pruned tables-only fit (prune=True); the tree "
                "ensemble of an unpruned fit has no fixed cell grid to aggregate over."
            )
        X, vectors = self._resolve_fit_vectors(
            X, {"y": y, "sample_weight": sample_weight, "exposure": exposure}
        )
        y, sample_weight, exposure = (vectors[k] for k in ("y", "sample_weight", "exposure"))
        for name, value, flag in (
            ("sample_weight", sample_weight, "_ae_requires_weight_"),
            ("exposure", exposure, "_ae_requires_exposure_"),
        ):
            if value is None and getattr(self, flag, False):
                raise ValueError(f"actual_vs_expected requires explicit {name} aligned to X")
        x32, cat_x = self._serve_design(X)
        yv = _as_float32_1d(y, "y").astype(np.float64)
        n = int(x32.shape[0])
        if yv.shape[0] != n:
            raise ValueError(f"y has {yv.shape[0]} rows but X has {n}")
        w = (
            _as_float32_1d(sample_weight, "sample_weight").astype(np.float64)
            if sample_weight is not None
            else np.ones(n, dtype=np.float64)
        )
        e = (
            _as_float32_1d(exposure, "exposure").astype(np.float64)
            if exposure is not None
            else np.ones(n, dtype=np.float64)
        )
        for name, values in (("y", yv), ("sample_weight", w), ("exposure", e)):
            if values.shape != (n,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain {n} finite values")
            if name != "y" and np.any(values < 0):
                raise ValueError(f"{name} must be non-negative")
        pred = np.asarray(self._expected_response(X, exposure), dtype=np.float64)
        cells = np.asarray(model.cell_indices(x32, cat_x=cat_x), dtype=np.int64)
        names = list(model.raw_feature_names())
        out: list[dict[str, Any]] = []
        for j, name in enumerate(names):
            c = cells[:, j]
            k = int(c.max()) + 1 if c.size else 0
            actual = np.bincount(c, weights=w * yv, minlength=k)
            expected = np.bincount(c, weights=w * pred, minlength=k)
            mass = np.bincount(c, weights=w * e, minlength=k)
            rows = np.bincount(c, minlength=k)
            with np.errstate(divide="ignore", invalid="ignore"):
                ae = np.where(expected > 0, actual / expected, np.nan)
            out.append({
                "feature": name,
                "raw": j,
                "actual": actual.tolist(),
                "expected": expected.tolist(),
                "mass": mass.tolist(),
                "rows": rows.tolist(),
                "ae": ae.tolist(),
            })
        return out

    def pricing_report(
        self, X: Any, y: Any, *, sample_weight: Any | None = None,
        exposure: Any | None = None, ref_measure: str | None = None,
    ) -> dict[str, Any]:
        """Return rating tables, labelled A/E cells and stage diagnostics for review.

        Pass aligned weights/exposure explicitly, as for ``actual_vs_expected``.
        The report describes the supplied evaluation data; it is not evidence that
        those data were held out. Save it beside the serialized model.
        """
        ae = self.actual_vs_expected(X, y, sample_weight=sample_weight, exposure=exposure)
        design, vectors = self._resolve_fit_vectors(
            X, {"y": y, "sample_weight": sample_weight, "exposure": exposure}
        )
        bank = json.loads(self.tables(
            design, ref_measure=ref_measure, sample_weight=vectors["sample_weight"],
            exposure=vectors["exposure"],
        ))
        axes = {int(a["raw"]): a for t in bank["tables"] for a in t["axes"]}
        for factor in ae:
            # Undefined ratios have an explicit JSON null in the portable report.
            factor["ae"] = [float(v) if np.isfinite(v) else None for v in factor["ae"]]
            factor["axis"] = axes.get(int(factor["raw"]))
        return {
            "report_version": 1, "prediction_units": (
                "per unit exposure" if getattr(self, "_ae_requires_exposure_", False)
                else "target units (including any rate units defined by the caller)"
            ),
            "reference_normalization": "zero mean on the link scale; not arithmetic mean rate",
            "band_interpretation": "bag stability; not claim prediction or credibility intervals",
            "tables": bank, "actual_vs_expected": ae,
            "pruning": getattr(self, "pruning_report_", None),
            "graduation": getattr(self, "graduation_validation_", None),
        }

    def _expected_response(self, X: Any, exposure: Any | None) -> "np.ndarray":
        """The fitted mean response per row on the scale of ``y`` (subclasses)."""
        raise NotImplementedError

    def _export_measure_kwargs(
        self, ref_measure: str | None, laplace: float, measure_floor: float | None
    ) -> dict[str, Any]:
        """Resolve ``tables()``'s measure arguments against the fit-time measure.

        ``ref_measure=None`` means "the ledger this estimator fitted under": for a pruned fit
        the frozen deployed bank as is, for an unpruned fit a fresh purification under the
        fit-time measure. Any explicit name re-expresses the same function under that
        measure (predictions never change). ``measure_floor=None`` inherits the fit-time
        floor.
        """
        fit = self._measure_kwargs()
        name = fit["ref_measure"] if ref_measure is None else str(ref_measure)
        floor = fit["measure_floor"] if measure_floor is None else float(measure_floor)
        return {"ref_measure": name, "laplace": float(laplace), "measure_floor": floor}

    def _new_booster(
        self,
        n_bags: int | None = None,
        n_trees: int | None = None,
        disable_cell_refit: bool = False,
        early_stopping_rounds_override: int | None = None,
        fit_pool_width: int | None = None,
    ) -> _Booster:
        # `n_bags` override lets the v2-prune CV fit cheap single-bag fold models (structure, not
        # variance-reduction, drives table selection) while the deployed model keeps its full bagging.
        # `fit_pool_width` (from `_default_fit_pool`) sizes the fit's rayon pool when the user left
        # n_jobs=None — see that method for the measured rationale; None falls back to the old
        # ambient-pool behavior (used by serve-time paths, which want full width for row-parallel
        # scoring).
        # `disable_cell_refit` accompanies that override: the OOB cell refit REQUIRES n_bags >= 2
        # (it refits toward the bags' OOB residual), so a single-bag fold fit must drop it or the
        # native config validation rejects the combination — the deployed fit keeps the user's
        # cell_refit_base untouched.
        # `early_stopping_rounds_override` replaces `self.early_stopping_rounds` when given (None
        # = untouched). `_fit_and_prune` uses it to run the prune-CV fold fits at a tighter
        # patience than the deployed model's own early stopping.
        # Reanchor (exact 1-D intercept re-solve) defaults ON for log-link objectives
        # (Gamma/Tweedie gain; Poisson is a no-op) AND, since 2026-07-23, for binary logistic:
        # the historical "off for logistic" default was measured while the post-prune path
        # silently DROPPED the logit correction (fixed 2026-07-22); re-measured post-fix on a
        # 6-set clf battery it is no-harm where balance is already right (−0.004%) and repairs
        # it where off (tic2000 balance .963→.996, brautocoll −1.5..−4.1%). This resolution
        # site is on the single-output path only, so K>=3 multiclass is untouched (its deployed
        # pruned artifact gets the IPF intercept re-anchor instead).
        # Reanchor_slope (evidence-gated affine holdout recalibration) keeps the log-link-only
        # default: validated there, unmeasured for logistic.
        # `None` = link-aware default; an explicit bool always wins.
        obj = str(self.objective).replace("-", "_").lower()
        reanchor = self.reanchor
        if reanchor is None:
            reanchor = obj in {"poisson", "gamma", "tweedie", "logistic"}
        reanchor_slope = self.reanchor_slope
        if reanchor_slope is None:
            reanchor_slope = obj in {"poisson", "gamma", "tweedie"}
        # path_smooth `None` resolves objective-aware (2026-07-23): heavy-tail gamma/tweedie
        # deviance is dominated by sparse-evidence cell variance, and the parent-shrinkage
        # credibility blend is the zero-cost smoother for exactly that (6-set screen:
        # freclaimdam -1.2%, ohlsson_pp -0.56%, ausautoBI -0.20%, worst case +0.05%). All
        # other objectives resolve to 0.0 (off) — bit-identical to the old default.
        path_smooth = self.path_smooth
        if path_smooth is None:
            path_smooth = 10.0 if obj in {"gamma", "tweedie"} else 0.0
        max_depth = int(self.max_depth)
        if not _LEGACY_MAX_DEPTH <= max_depth <= _MAX_DEPTH:
            raise ValueError(
                f"max_depth must be in {_LEGACY_MAX_DEPTH}..{_MAX_DEPTH}, got {max_depth}"
            )
        if max_depth > _LEGACY_MAX_DEPTH and self.ridge_refit_l2 is not None:
            raise ValueError(
                "ridge_refit_l2 is not supported with max_depth > "
                f"{_LEGACY_MAX_DEPTH}: the fully-corrective refit assumes a fixed "
                "8-leaf-column stride per tree and would be 64x larger and "
                "rank-deficient at depth 6"
            )
        max_order = int(self.max_interaction_order)
        if not 1 <= max_order <= _MAX_ORDER:
            raise ValueError(
                f"max_interaction_order must be in 1..{_MAX_ORDER}, got {max_order}"
            )
        if max_order > max_depth:
            raise ValueError(
                f"max_interaction_order {max_order} exceeds max_depth {max_depth}: an "
                "oblivious tree needs one level per distinct raw feature, so order "
                f"{max_order} is unreachable at depth {max_depth}. Raise max_depth."
            )
        min_data_in_leaf = _resolve_min_data_in_leaf(self.min_data_in_leaf, max_depth)
        table_budget_cells = _resolve_table_budget_cells(
            self.table_budget_cells, max_depth, max_order
        )
        return _Booster(
            n_trees=int(self.n_trees if n_trees is None else n_trees),
            learning_rate=float(self.learning_rate),
            lambda_=float(self.lambda_),
            lambda_scale_invariant=bool(self.lambda_scale_invariant),
            l1_leaf=float(self.l1_leaf),
            min_split_gain=float(self.min_split_gain),
            max_delta_step=self.max_delta_step,
            max_delta_step_gated=self.max_delta_step_gated,
            max_bin=int(self.max_bin),
            objective=self.objective,
            tweedie_rho=float(self.tweedie_rho),
            min_data_in_leaf=min_data_in_leaf,
            min_sum_hessian_in_leaf=float(self.min_sum_hessian_in_leaf),
            min_weight_sum_in_leaf=float(self.min_weight_sum_in_leaf),
            path_smooth=float(path_smooth),
            subsample=None if self.subsample is None else float(self.subsample),
            colsample_bytree=float(self.colsample_bytree),
            learning_rate_decay=float(self.learning_rate_decay),
            validation_fraction=(
                None if self.validation_fraction is None else float(self.validation_fraction)
            ),
            early_stopping_rounds=(
                int(self.early_stopping_rounds)
                if early_stopping_rounds_override is None
                else int(early_stopping_rounds_override)
            ),
            early_stopping_adaptive=(
                None if self.early_stopping_adaptive is None else float(self.early_stopping_adaptive)
            ),
            early_stopping_min_delta=float(self.early_stopping_min_delta),
            interaction_gain_hurdle=float(self.interaction_gain_hurdle),
            interaction_gain_hurdle_mode=self.interaction_gain_hurdle_mode,
            leaf_refine_steps=int(self.leaf_refine_steps),
            leaf_refine_backtracks=int(self.leaf_refine_backtracks),
            refine_closed_form_tier2=bool(self.refine_closed_form_tier2),
            incremental_mu=bool(self.incremental_mu),
            mvs_min_rows=int(self.mvs_min_rows),
            hist_precision=self.hist_precision,
            n_bags=int(self.n_bags if n_bags is None else n_bags),
            bag_subsample=float(self.bag_subsample),
            cell_refit_base=(
                None
                if (disable_cell_refit or self.cell_refit_base is None)
                else float(self.cell_refit_base)
            ),
            cell_refit_gamma=float(self.cell_refit_gamma),
            ridge_refit_l2=None if self.ridge_refit_l2 is None else float(self.ridge_refit_l2),
            ridge_refit_max_iter=int(self.ridge_refit_max_iter),
            nesterov=bool(self.nesterov),
            dart_drop_rate=None if self.dart_drop_rate is None else float(self.dart_drop_rate),
            random_strength=float(self.random_strength),
            reanchor=bool(reanchor),
            reanchor_slope=bool(reanchor_slope),
            max_interaction_order=int(self.max_interaction_order),
            max_depth=max_depth,
            table_budget_cells=table_budget_cells,
            table_budget_order_shrink=float(self.table_budget_order_shrink),
            cat_smooth=None if self.cat_smooth is None else float(self.cat_smooth),
            cat_target=self.cat_target,
            **self._resolve_cat_leakage(),
            cat_min_data_per_group=float(self.cat_min_data_per_group),
            cat_direct_max_levels=int(self.cat_direct_max_levels),
            cat_channels=self.cat_channels,
            cat_count_min_levels=int(self.cat_count_min_levels),
            cat_class_freq_min_levels=int(self.cat_class_freq_min_levels),
            seed=int(self.seed),
            # `n_jobs` is always the user's TRUE thread budget (never the heuristic): the native
            # layer caps the process-GLOBAL rayon pool to it once per process, so substituting
            # `fit_pool_width` here (as this line used to) would permanently pin every later
            # ambient-pool serve call (predict/tables/...) at the fit's heuristic width instead of
            # the user's real n_jobs. `fit_pool_width` is passed separately below and only sizes
            # this ONE fit's own local pool.
            n_jobs=self._resolve_n_jobs(),
            fit_pool_width=fit_pool_width,
        )

    def _fit_model(
        self,
        x: Any,
        y: Any,
        *,
        sample_weight: Any | None,
        exposure: Any | None,
        class_labels: list[str] | None,
        groups: Any | None = None,
    ) -> _Model | _TableModel:
        self._clear_stale_fit_state()
        high_alpha = self.graduation_high_order_alpha
        if (isinstance(high_alpha, (bool, np.bool_))
                or not isinstance(high_alpha, (int, float, np.integer, np.floating))
                or not np.isfinite(high_alpha) or not 0.0 <= high_alpha <= 1.0):
            raise ValueError("graduation_high_order_alpha must be finite and in [0, 1]")
        if high_alpha > 0 and (not self.prune or self.graduate is False):
            raise ValueError("graduation_high_order_alpha requires prune=True and graduation enabled")
        # No silent no-op: Whittaker-Henderson graduation only exists inside `_fit_and_prune`
        # (see below) — an explicit graduate=True with prune=False is currently un-satisfiable,
        # not a quiet pass-through. The auto-default (graduate=None, resolved to True for
        # poisson/gamma) stays silent here: it never reaches this branch when prune is off
        # (the resolution itself lives inside `_fit_and_prune`), so it truthfully has no effect
        # for non-pruned fits without needing a special case.
        if self.graduate and not self.prune:
            raise ValueError(
                "graduate=True has no effect without prune=True: Whittaker-Henderson "
                "graduation only applies to the pruned tables-only artifact. Set prune=True, "
                "or leave graduate at its default (None)."
            )
        if BUILD_PROFILE == "debug":
            warnings.warn(
                "the t_boost extension is a DEBUG build — fit timings are unrepresentative"
                " (typically 5-30x slower); rebuild with `maturin develop --release`",
                RuntimeWarning,
                stacklevel=2,
            )
        self._validate_cat_channels(multiclass=False)
        _reject_sparse(x)
        x = collect_frame(x)  # LazyFrame: full collect (every non-consumed column is a feature)
        feature_names = _feature_names_from_x(x)
        n_features = _n_columns(x)
        if n_features == 0:
            raise ValueError("X has no features (0 columns)")
        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length {len(feature_names)} != n_features {n_features}"
            )
        # Declared categoricals union with polars dtype-detected ones (String/Categorical/Enum
        # columns cannot be numeric axes, so they need no declaration).
        cat_idx = sorted(
            set(self._resolve_categorical(n_features, feature_names))
            | set(auto_categorical_idx(x))
        )
        x32, cat_x, axis_names = self._split_columns(x, cat_idx, feature_names)
        y32 = _as_float32_1d(y, "y")
        if x32.shape[0] != y32.shape[0]:
            raise ValueError(f"X has {x32.shape[0]} rows but y has {y32.shape[0]}")
        if x32.shape[0] == 0:
            raise ValueError("X has 0 samples; need at least 1 row to fit")
        weight32 = None
        if sample_weight is not None:
            weight32 = _as_float32_1d(sample_weight, "sample_weight")
            if weight32.shape[0] != y32.shape[0]:
                raise ValueError(
                    f"sample_weight has {weight32.shape[0]} rows but y has {y32.shape[0]}"
                )
            if float(np.sum(weight32)) <= 0.0:
                raise ValueError("sample_weight sums to zero; no effective samples to fit")
        exposure32 = None
        if exposure is not None:
            exposure32 = _as_float32_1d(exposure, "exposure")
            if exposure32.shape[0] != y32.shape[0]:
                raise ValueError(
                    f"exposure has {exposure32.shape[0]} rows but y has {y32.shape[0]}"
                )
        groups_arr: np.ndarray | None = None
        if groups is not None:
            groups_arr = np.asarray(groups).ravel()
            if groups_arr.shape[0] != y32.shape[0]:
                raise ValueError(
                    f"groups has {groups_arr.shape[0]} rows but y has {y32.shape[0]}"
                )
            # DEGENERATE grouping (every group a singleton) == no grouping: drop it here so the
            # whole fit takes the ungrouped path byte-for-byte, recovering the ~10%-of-rows ES
            # carve those fits were paying for a vacuous guarantee (see `_is_degenerate_grouping`
            # for the identity argument). Validation above runs FIRST — a length mismatch is
            # still an error, degenerate or not. A `str` groups column is still consumed out of
            # the feature set by `_resolve_fit_vectors`; the identity is over the group VALUES.
            if _is_degenerate_grouping(groups_arr):
                groups_arr = None
        # GROUP-AWARE outer bags (2026-09-07): dense group codes make every bag a subsample of
        # whole groups, so the out-of-bag rows the cell refit and the prune guard read are honest
        # on panel data (a policy's other years never trained the jury that scores it).
        bag_codes = (
            np.ascontiguousarray(np.unique(groups_arr, return_inverse=True)[1], dtype=np.uint32)
            if groups_arr is not None
            else None
        )
        # Stash the validated fit-time row masses so `tables()` can default its explain-time
        # weighting to them (explicit call-time arguments override; cleared by set_params /
        # _clear_stale_fit_state so they never leak across a refit).
        self._ae_requires_weight_ = weight32 is not None
        self._ae_requires_exposure_ = exposure32 is not None
        if weight32 is not None:
            self._fit_sample_weight_ = weight32
        if exposure32 is not None:
            self._fit_exposure_ = exposure32
        # Monotone signs are positional over the ORIGINAL features; native categoricals
        # reorder the design (numeric axes first, then categoricals, per `_split_columns`),
        # so remap the sign vector to that axis order before handing it to the core.
        monotone = self._resolve_monotone(n_features, feature_names)
        if self.graduation_high_order_alpha > 0 and monotone and any(monotone):
            raise ValueError("higher-order smoothing does not support monotone constraints")
        if monotone is not None and cat_idx:
            cat_set = set(cat_idx)
            numeric_idx = [i for i in range(n_features) if i not in cat_set]
            monotone = [monotone[i] for i in numeric_idx] + [monotone[j] for j in cat_idx]
        if monotone is not None and any(monotone) and self.cell_refit_base is not None:
            raise ValueError("cell_refit_base cannot preserve monotone_constraints; leave it unset")
        booster = self._new_booster(fit_pool_width=self._default_fit_pool(n_features))
        model: _Model | _TableModel
        self._validate_box_budget_reachable()
        if self.prune:
            model = self._fit_and_prune(
                booster,
                x32,
                y32,
                weight32,
                exposure32,
                axis_names,
                class_labels,
                monotone,
                cat_x,
                groups_arr,
            )
        else:
            es_holdout: list[bool] | None = None
            if groups_arr is not None and self.validation_fraction is not None:
                seed = int(self.seed) if getattr(self, "seed", None) is not None else 0
                es_holdout = _carve_group_holdout(
                    groups_arr, float(self.validation_fraction), [seed, 1]
                ).tolist()
            model = booster.fit(
                x32,
                y32,
                weight=weight32,
                exposure=exposure32,
                feature_names=axis_names,
                class_labels=class_labels,
                monotone=monotone,
                cat_x=cat_x,
                es_holdout=es_holdout,
                bag_groups=bag_codes,
            )
        if not self.prune and isinstance(model, _Model):
            self.delta_step_gate_ = model.delta_step_gate
        self._model = model
        self._cat_indices_ = cat_idx
        self.n_features_in_ = n_features
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        return model

    def _fit_and_prune(
        self, booster: _Booster, x32: np.ndarray, y32: np.ndarray,
        weight32: np.ndarray | None, exposure32: np.ndarray | None,
        axis_names: list[str] | None, class_labels: list[str] | None,
        monotone: list[int] | None, cat_x: list[list[str]] | None,
        groups_arr: np.ndarray | None = None,
    ) -> _TableModel:
        """Fit and prune on EVERY row, then graduate the deployed tables.

        GRADUATION IS A USER SWITCH (2026-09-20), not a call the library makes for you.

        The previous design reserved an independent holdout (`prune_validation_fraction` of
        the rows) purely to score ONE accept/reject bit, and never gave those rows back: the
        boost, the prune, the keep-set vote, the refits and the recalibration all ran on the
        remainder. Measured on the arena catalog (40 sets, default variant) that cost +1.29%
        mean test deviance -- +6.05% swautoins, +2.85% freclaimdam, +2.45% swautoins_pp,
        +3.6% catelematic13 -- to buy a smoothing adopted on 30% of fits for a median
        +0.045% ON THE GATING STATISTIC ITSELF (the holdout deviance that decided it, so
        biased high). Applying the same smoothing to a full-data fit with NO gate measures
        -0.015% mean against never graduating: free, inside split noise.

        So the fit keeps every row and `graduate` decides. Whether a smoothed tariff is worth
        having is a property of what the model is FOR -- a filing, a quote engine, a
        monitoring baseline -- which the caller knows and a deviance comparison does not.

        `graduate=None` resolves to ON: the smoothing is free on the measured catalog and a
        graduated bank is the better artifact to print. `graduate=False` skips it entirely
        (and `graduation_report_` is then never set). Monotone fits keep the full bank
        verbatim (see `_fit_and_prune_core`) and never graduate. Multiclass (K>=3) has no
        graduation path at all and never reaches this method.
        """
        n = len(y32)
        do_graduate = self.graduate is not False and not (monotone and any(monotone))
        if do_graduate:
            self.graduation_report_: list[dict[str, Any]] = []
        # `group_disjoint`/`selection_independent` are gone with the holdout they described:
        # there is no evidence set to be disjoint from or independent of. `holdout_rows` stays,
        # pinned at 0, so a reader (and the arena's evidence) can see that nothing was withheld
        # rather than having to infer it from a missing key.
        self.graduation_validation_: dict[str, Any] = {
            "evidence": "none", "training_rows": n, "holdout_rows": 0, "adopted": False,
        }
        model = self._fit_and_prune_core(
            booster, x32, y32, weight32, exposure32, axis_names, class_labels,
            monotone, cat_x, groups_arr,
        )
        if do_graduate:
            w = weight32 if weight32 is not None else np.ones(n, dtype=np.float32)
            model = self._graduate_table_model(
                model, x32, y32, np.ascontiguousarray(w), exposure32, cat_x,
            )
        else:
            self.graduation_validation_["skipped"] = (
                "disabled" if self.graduate is False else "monotone_constraints"
            )
        self.pruning_report_["graduation_validation"] = self.graduation_validation_
        return model

    def _ranked_path_select(
        self, full_model: Any, full_supports: list[Any], share: dict[Any, Any], x32: np.ndarray,
        y32: np.ndarray, w_full: np.ndarray, expo_full: np.ndarray | None, cat_x: Any, obj: str,
    ) -> tuple[list[Any], dict[str, Any]] | None:
        """Ranked-path keep-set (see `_PRUNE_SELECTOR_DEFAULT`). `share` maps each support to its
        purified variance in the full bank (`_Model.table_variances`). Returns `(keep, report)`,
        or None when the out-of-bag jury has fewer than `prune_guard_min_rows` rows."""
        n_jobs = self._resolve_n_jobs()
        supports = sorted({tuple(sorted(int(i) for i in u)) for u in full_supports})
        mains = [u for u in supports if len(u) == 1]
        inter = sorted((u for u in supports if len(u) > 1), key=lambda u: (-share.get(u, 0.0), u))
        seq = _heredity_sequence(mains, inter)
        n = len(seq)
        steps = max(2, int(getattr(self, "prune_path_steps", _PRUNE_PATH_STEPS_DEFAULT)))
        sizes = sorted({0, n} | {int(round(x)) for x in np.geomspace(1, max(n, 1), steps)})
        groups = [[list(u) for u in mains]]
        for a, b in zip(sizes[:-1], sizes[1:]):
            groups.append([list(u) for u in seq[a:b]])
        # The path reads the intercept and group sums only: skip scoring each bag's full bank.
        _full_sum, f0_sum, group_sums, counts = full_model.bag_oob_group_raw(
            x32, groups, cat_x=cat_x, n_jobs=n_jobs,
            weight=np.ascontiguousarray(w_full), exposure=expo_full, **self._measure_kwargs(),
            with_full=False,
        )
        counts = np.asarray(counts, dtype=np.int64)
        ev = counts > 0
        n_val = int(ev.sum())
        if n_val < int(getattr(self, "prune_guard_min_rows", _PRUNE_GUARD_MIN_ROWS)):
            return None
        cnt = counts[ev].astype(np.float64)
        yv = np.asarray(y32, dtype=np.float64)[ev]
        wv = np.asarray(w_full, dtype=np.float64)[ev]
        offset = (np.log(np.asarray(expo_full, dtype=np.float64)[ev])
                  if expo_full is not None else 0.0)
        gs = np.asarray(group_sums, dtype=np.float64)
        raw = np.asarray(f0_sum, dtype=np.float64)[ev] / cnt + offset
        devs = []  # group g adds the tables of prefix sizes[g]; group 0 is the mains
        for g in range(gs.shape[0]):
            raw = raw + gs[g][ev] / cnt
            devs.append(_guard_mean_deviance(obj, yv, raw, wv, self.tweedie_rho))
        best = int(np.argmin(devs))
        frac = float(getattr(self, "prune_path_fraction", _PRUNE_PATH_FRACTION_DEFAULT))
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"prune_path_fraction must be in (0, 1], got {frac!r}")
        tol = float(getattr(self, "prune_path_tolerance", _PRUNE_PATH_TOLERANCE_DEFAULT))
        if not tol >= 0.0:
            raise ValueError(f"prune_path_tolerance must be >= 0, got {tol!r}")
        gain = devs[0] - devs[best]
        if gain > 0.0 and frac < 1.0:
            i_frac = next(i for i in range(len(devs)) if devs[0] - devs[i] >= frac * gain)
            i_tol = next(i for i in range(len(devs)) if devs[i] <= devs[best] * (1.0 + tol))
            best = max(i_frac, i_tol)
        keep_t = mains + seq[: sizes[best]]
        keep = [list(u) for u in keep_t]
        kept_set = set(keep_t)
        report = {
            "selector": "ranked_path",
            "kept": keep,
            "effective_order": max((len(u) for u in keep_t), default=0),
            "main_effect_policy": "sticky",
            "n_candidates": len(supports),
            "path": {"sizes": sizes, "oob_deviance": devs, "best": best, "oob_rows": n_val,
                     "fraction": frac, "tolerance": tol},
            "table_scores": [
                {"u": list(u), "order": len(u), "mean_gain": share.get(u, 0.0),
                 "variance_share": share.get(u, 0.0), "selected": u in kept_set,
                 "sticky": len(u) == 1}
                for u in supports
            ],
            "guard": {"enabled": False, "skipped": "ranked_path selector"},
        }
        return keep, report

    def _band_deployed(
        self, table_model: Any, full_model: Any, keep: list[Any], x32: np.ndarray, y32: np.ndarray,
        w_full: np.ndarray, expo_full: np.ndarray | None, cat_x: Any, obj: str, tolerance: float,
    ) -> Any:
        """Band the deployed tables (see `_BAND_TOLERANCE_DEFAULT`); records the report."""
        n_jobs = self._resolve_n_jobs()
        target = np.asarray(table_model.predict_raw(x32, cat_x=cat_x), dtype=np.float64)
        w = np.asarray(w_full, dtype=np.float64)
        offset = (np.log(np.asarray(expo_full, dtype=np.float64))
                  if expo_full is not None else 0.0)
        h = _band_curvature(obj, target + offset, w, self.tweedie_rho)
        mass = w * (np.asarray(expo_full, dtype=np.float64) if expo_full is not None else 1.0)
        try:
            # Each bag's bank is purified under the same row mass (weight x exposure) as every
            # other fit-time bank, so the noise is measured on the tables that actually ship.
            var, n_bags = full_model.bag_score_variance(
                x32, keep=[list(u) for u in keep], cat_x=cat_x, n_jobs=n_jobs,
                weight=np.ascontiguousarray(w_full), exposure=expo_full,
                **self._measure_kwargs(),
            )
        except ValueError as exc:
            # No per-bag replicates to measure noise against (e.g. a §G1 cell-corrected soup):
            # ship the unbanded tables and say so.
            if isinstance(getattr(self, "pruning_report_", None), dict):
                self.pruning_report_["banding"] = {"skipped": f"no bag noise estimate: {exc}"}
            return table_model
        s2 = np.asarray(var, dtype=np.float64) / max(int(n_bags), 1)
        sigma = float(np.sqrt(np.sum(h * s2) / max(np.sum(h), 1e-300)))
        seed = int(self.seed) if getattr(self, "seed", None) is not None else 0
        # expected deviance increase of a move δ is Σ h δ² / Σ w (deviance = 2·NLL, h = NLL
        # curvature), so capping it at `band_deviance_cap` of the mean deviance caps the MSE at:
        dev_mean = _guard_mean_deviance(obj, y32, target + offset, w, self.tweedie_rho)
        cap_frac = float(getattr(self, "band_deviance_cap", _BAND_DEVIANCE_CAP_DEFAULT))
        mse_cap = (cap_frac * dev_mean * float(np.sum(w)) / max(float(np.sum(h)), 1e-300)
                   if np.isfinite(dev_mean) and cap_frac > 0 else float("inf"))
        banded, report_json = table_model.band(
            x32, np.ascontiguousarray(h), np.ascontiguousarray(mass), np.ascontiguousarray(target),
            sigma, tolerance=tolerance, cat_x=cat_x, seed=seed, n_jobs=n_jobs, mse_cap=mse_cap,
        )
        if isinstance(getattr(self, "pruning_report_", None), dict):
            self.pruning_report_["banding"] = json.loads(report_json)
        return banded

    def _fit_and_prune_core(
        self,
        booster: _Booster,
        x32: np.ndarray,
        y32: np.ndarray,
        weight32: np.ndarray | None,
        exposure32: np.ndarray | None,
        axis_names: list[str] | None,
        class_labels: list[str] | None,
        monotone: list[int] | None,
        cat_x: list[list[str]] | None,
        groups_arr: np.ndarray | None = None,
    ) -> _TableModel:
        # Contribution-stability pruning: fit the deployed model on all rows, fit honest fold
        # models to estimate which table supports survive held-out deviance pruning, aggregate those
        # supports by stability/contribution, then apply the selected keep-set to the full-data fit.
        # Main effects are sticky by design; interactions must earn survival on validation folds.
        self._validate_dead_prune_params(multiclass=False)
        import os as _os
        import time as _time
        import math

        _prof = _os.environ.get("TBOOST_PROFILE")
        _pt = _time.perf_counter()

        def _lap(label: str) -> None:
            nonlocal _pt
            if _prof:
                import sys as _sys

                print(f"[pyprune] {label} {_time.perf_counter() - _pt:.2f}s", file=_sys.stderr, flush=True)
            _pt = _time.perf_counter()

        n = int(x32.shape[0])
        seed = int(self.seed) if getattr(self, "seed", None) is not None else 0
        # `obj` is bound UNCONDITIONALLY: three independent decisions below read it (the
        # `reanchor` default, the ES-strata group carve, the `graduate` default), so binding it
        # inside the `reanchor is None` branch left it unbound whenever a caller passed
        # `reanchor` explicitly — which the tuned recipe does for every log-link objective
        # (`recommended_recipe`, `params["reanchor"] = True`). That crashed every grouped tuned
        # deploy fit with UnboundLocalError (insur-arena fremotor_prem, 9/9 splits NaN).
        obj = str(self.objective).replace("-", "_").lower()
        reanchor = self.reanchor
        if reanchor is None:
            # keep in lockstep with `_new_booster`'s resolution (logistic added 2026-07-23,
            # single-output path only — see the comment there).
            reanchor = obj in {"poisson", "gamma", "tweedie", "logistic"}
        reanchor = bool(reanchor)
        w_full = weight32 if weight32 is not None else np.ones(n, dtype=np.float32)
        expo_full = np.ascontiguousarray(exposure32) if exposure32 is not None else None

        # Group-honest carves (panel data): the deploy fit's ES holdout AND the prune-CV fold
        # fits' own nested ES holdout (below, threaded through `fit_prune_selection` and
        # gathered per fold in Rust) assign WHOLE groups to one side instead of rows — the
        # row-level carve otherwise leaks near-duplicate rows of the same entity into
        # validation, which defeats early stopping (fits run to the tree cap) and biases prune
        # selection (see `_carve_group_holdout`'s doc). Distinct deterministic streams
        # ([seed, 1]/[seed, 2]) keep the two draws independent; [seed, 0] is reserved for the
        # group-aware fold assignment below, mirroring the multiclass carve's key numbering.
        # UNGROUPED fits carve NOTHING here — every bag keeps its own internal ES slice, exactly
        # as it did before the guard existed. av33 briefly gave large ungrouped pruned fits a
        # shared row-level carve so the guard would have an honest slice to score on; the OOB
        # evidence path (see the guard block below) gets that honesty from the bags' own
        # out-of-bag rows instead, at zero cost to the fit, so the carve — and the 100k size gate
        # that rationed its ~10%-of-rows cost — are both gone. Ungrouped pruned fits are now
        # byte-identical to guard-off fits at EVERY size.
        deploy_es_holdout: list[bool] | None = None
        fold_es_holdout: list[bool] | None = None
        if self.validation_fraction is not None and groups_arr is not None:
            # ES-runaway fix #2 (2026-07-20): on high-zero-mass Poisson/Tweedie panels an
            # unstratified group-honest carve can land almost entirely on zero-event groups —
            # the group-carve analog of the row-level bug es_strata_for_loss fixes (native
            # core). Gamma is y > 0 by construction (nothing to stratify against) and
            # SquaredError has no zero-mass semantics — both stay unstratified, unaffected.
            # `logistic` joins them (2026-07-21) so this mirrors `es_strata_for_loss` exactly:
            # the native row-level carve already stratifies a rare positive CLASS the same way,
            # and a grouped binary panel would otherwise be the one path that silently didn't.
            es_strata = (
                (np.asarray(y32) > (0.5 if obj == "logistic" else 0.0)).astype(np.int64)
                if obj in {"poisson", "tweedie", "logistic"}
                else None
            )
            deploy_es_holdout = _carve_group_holdout(
                groups_arr, float(self.validation_fraction), [seed, 1], strata=es_strata
            ).tolist()
            fold_es_holdout = _carve_group_holdout(
                groups_arr, float(self.validation_fraction), [seed, 2], strata=es_strata
            ).tolist()

        # 1. Deployed model: fit on ALL rows (full-data table estimates).
        # x32 is passed as-is (not re-forced to C-order): it is already contiguous in whichever
        # order _as_float32_2d preserved, and the native fit() ingests either layout directly
        # (raw_columns_from_array, crates/t-boost-py/src/lib.rs) -- wrapping it here again
        # would silently undo an F-contiguous caller's cheap ingest. y/weight/exposure stay
        # C-contiguous: the native 1-D `as_slice` path requires it.
        # GROUP-AWARE outer bags for the deployed fit (see `_fit_model`): dense group codes make
        # every bag a subsample of whole groups, so the out-of-bag rows below are honest.
        bag_codes = (
            np.ascontiguousarray(np.unique(groups_arr, return_inverse=True)[1], dtype=np.uint32)
            if groups_arr is not None
            else None
        )
        full_model = booster.fit(
            x32,
            np.ascontiguousarray(y32),
            weight=(np.ascontiguousarray(weight32) if weight32 is not None else None),
            exposure=(np.ascontiguousarray(exposure32) if exposure32 is not None else None),
            feature_names=axis_names,
            class_labels=class_labels,
            monotone=monotone,
            cat_x=cat_x,
            es_holdout=deploy_es_holdout,
            bag_groups=bag_codes,
        )
        # §05.6-addendum rate-collapse gate report for the DEPLOYED fit (the pruning fold
        # models have their own, uninteresting, gates). Recorded here rather than after the
        # prune because pruning replaces the `_Model` with a `_TableModel`, which carries no
        # fit-time diagnostics.
        self.delta_step_gate_ = full_model.delta_step_gate
        _lap(f"(1) deployed {int(self.n_bags)}-bag fit")

        # The full bank's per-table purified variances: the ranked path's ranking key, and its keys
        # are the bank's supports (the table SET does not depend on the measure), so one bank build
        # serves both instead of a separate `table_supports` pass.
        table_share = {
            tuple(sorted(int(i) for i in u)): float(v)
            for u, v in full_model.table_variances(
                x32, np.ascontiguousarray(w_full), cat_x=cat_x, exposure=expo_full,
                n_jobs=self._resolve_n_jobs(), **self._measure_kwargs(),
            )
        }
        full_supports = sorted(table_share)
        if monotone is not None and any(monotone):
            if self.prune_box_budget or self.prune_table_budget:
                raise ValueError("monotone_constraints cannot be combined with table/box budgets")
            self.pruning_report_: dict[str, Any] = {
                "kept": [list(u) for u in full_supports],
                "constraints": "full table bank retained; pruning/refits/graduation disabled",
                "guard": {"enabled": False, "skipped": "full monotone bank retained"},
            }
            return full_model.apply_keepset(
                x32, np.ascontiguousarray(y32), np.ascontiguousarray(w_full),
                [list(u) for u in full_supports], reanchor=reanchor, rebalance=False,
                n_jobs=self._resolve_n_jobs(), exposure=expo_full, cat_x=cat_x,
                **self._measure_kwargs(),
            )
        full_tree_count = int(getattr(full_model, "n_trees", 0))
        trees_per_bag = max(1, math.ceil(full_tree_count / max(1, int(self.n_bags))))
        fold_tree_cap = min(int(self.n_trees), max(64, int(math.ceil(2.0 * trees_per_bag))))
        _lap(
            f"(1b) support scan ({len(full_supports)} tables, "
            f"full_trees={full_tree_count}, prune_fold_cap={fold_tree_cap})"
        )

        # 2. Keep-set selection. Default: the ranked path on the soup's out-of-bag jury (see
        #    `_PRUNE_SELECTOR_DEFAULT`). It needs honest out-of-bag rows; a fit without them (a single
        #    bag) falls back to the fold vote below.
        use_path = False
        if str(getattr(self, "prune_selector", _PRUNE_SELECTOR_DEFAULT)) == "ranked_path":
            if str(self.prune_selector) not in ("ranked_path", "fold_vote"):
                raise ValueError(f"prune_selector must be 'ranked_path' or 'fold_vote', got {self.prune_selector!r}")
            oob_honest_path = (
                groups_arr is None or _is_degenerate_grouping(groups_arr) or bag_codes is not None
            )
            if oob_honest_path and bool(full_model.bag_oob_available()):
                sel = self._ranked_path_select(
                    full_model, full_supports, table_share, x32, y32, w_full, expo_full, cat_x, obj
                )
                if sel is not None:
                    keep, self.pruning_report_ = sel
                    use_path = True
                    _lap(f"(2) ranked path ({len(keep)}/{len(full_supports)} tables)")
        elif str(self.prune_selector) != "fold_vote":
            raise ValueError(f"prune_selector must be 'ranked_path' or 'fold_vote', got {self.prune_selector!r}")
        if not use_path:
            # 2. Honest K-fold CV over all rows. Sparse positive rows are spread across folds first so
            #    rare claim signal can participate in table selection rather than vanish into one fold.
            #    The fold count adapts down at small n so each fold's scoring slice keeps at least
            #    _MIN_PRUNE_FOLD_ROWS rows (see the constant's comment); prune_n_folds is the cap.
            k_folds = max(2, min(int(self.prune_n_folds), n // int(self.prune_fold_min_rows), n))
            if groups_arr is not None:
                # Group-aware fold assignment (see `_carve_group_folds`): every row of the same
                # entity lands in the SAME fold, so a group can never straddle a fold's train side
                # and its held-out scoring side. This replaces the event/non-event row-level
                # stratification below — a group can mix event/non-event rows, so the two disciplines
                # cannot compose without risking a split group; grouped panels take group honesty.
                fold_of = _carve_group_folds(groups_arr, k_folds, [seed, 0])
            else:
                rng = np.random.default_rng(seed)
                fold_of = np.empty(n, dtype=np.int64)
                event = np.asarray(y32) > 0
                for idx in (np.flatnonzero(event), np.flatnonzero(~event)):
                    if idx.size == 0:
                        continue
                    shuffled = rng.permutation(idx)
                    fold_of[shuffled] = np.arange(shuffled.size, dtype=np.int64) % k_folds
            # Prune-CV FOLD fits run at a tighter ES patience than the deploy fit
            # (_PRUNE_FOLD_ES_PATIENCE = 250 vs the estimator's 500 default; Ralph-promoted
            # 2026-07-16): the fold fits exist only to vote on table survival, and ~48% of their
            # rounds were post-best patience tail. Keep-set-identity study: selections IDENTICAL
            # and test-deviance deltas exactly 0.0 on brvehins1, euhealth AND the 406k MTPL fit,
            # with the MTPL CV scan -19.4% (44.3s -> 35.7s). The deploy fit's patience is untouched.
            # `prune_fold_es_patience` (exposed 2026-09-04) overrides it; `None` keeps the shipped 250.
            _explicit_patience = getattr(self, "prune_fold_es_patience", None)
            _fold_patience_override = (
                int(_explicit_patience) if _explicit_patience is not None else _PRUNE_FOLD_ES_PATIENCE
            )
            n_feats_total = int(x32.shape[1]) + (len(cat_x) if cat_x else 0)
            fold_booster = self._new_booster(
                n_bags=1,
                n_trees=fold_tree_cap,
                disable_cell_refit=True,
                early_stopping_rounds_override=_fold_patience_override,
                # same pool width as the deploy fit: the 5 folds provide the outer parallelism
                # here, exactly like bags do there (validated by the end-to-end n_jobs A/B).
                fit_pool_width=self._default_fit_pool(n_feats_total),
            )
            min_stability = float(self.prune_min_stability)
            keep, report_json = fold_booster.fit_prune_selection(
                x32,  # see fit() call above
                np.ascontiguousarray(y32),
                np.ascontiguousarray(fold_of),
                int(k_folds),
                [list(fs) for fs in full_supports],
                weight=(np.ascontiguousarray(weight32) if weight32 is not None else None),
                exposure=(np.ascontiguousarray(exposure32) if exposure32 is not None else None),
                feature_names=axis_names,
                class_labels=class_labels,
                monotone=monotone,
                cat_x=cat_x,
                **self._measure_kwargs(),
                se_rule=float(self.prune_se_rule),
                lambda_boxes=float(getattr(self, "prune_lambda_boxes", 0.0) or 0.0),
                # `n_folds` here sub-divides EACH outer honest-CV fold's own held-out rows for a
                # within-fold 1-SE band (prune.rs `mean_and_se`); at n_folds=1 that band's SE is
                # always 0.0 (a single sample has no variance), so `se_rule` — however wired above —
                # multiplies against zero and can never move the selection off the bare deviance
                # minimum. n_folds=2 is the smallest value that gives `se_rule` any effect at all;
                # verified against the default se_rule=0.0 (six datasets/seeds, `n_folds` in {1, 2}):
                # the selected keep-set is unchanged there, since a zero multiplier already made the
                # SE irrelevant — this only activates the opt-in se_rule>0 parsimony dial.
                n_folds=2,
                reanchor=reanchor,
                min_stability=min_stability,
                min_mean_gain=float(self.prune_min_mean_gain),
                es_holdout=fold_es_holdout,
                # av37 evidence gate — `None` reproduces the av36 aggregator byte-for-byte (the
                # binding defaults to None, so an older compiled module would silently ship av36
                # behavior; it is passed EXPLICITLY and positionally-named so a stale build raises
                # TypeError instead, the same discipline `bag_oob_available` uses below).
                drop_z=(None if self.prune_drop_z is None else float(self.prune_drop_z)),
                keep_budget=max(0, int(self.prune_keep_budget)),
                # Selection-time multi-way TABLE price (2026-08-27). 0.0 is inert to the last bit;
                # passed explicitly for the same stale-build reason `drop_z` is.
                lambda_tables=float(getattr(self, "prune_lambda_tables", 0.0) or 0.0),
                table_min_arity=int(
                    getattr(self, "prune_table_min_arity", _PRUNE_TABLE_MIN_ARITY_DEFAULT)
                ),
                # Pinned-bank fold fidelity. False is inert to the last bit; passed explicitly for
                # the same stale-build reason `drop_z` is.
                fold_fidelity=bool(
                    getattr(self, "prune_fold_fidelity", _PRUNE_FOLD_FIDELITY_DEFAULT)
                ),
            )
            self.pruning_report_ = json.loads(report_json)
            _lap(f"(2) CV contribution scan ({self.pruning_report_['cv_folds']}/{k_folds} folds)")
        # Graduation is handled after this complete pipeline, on a separate holdout
        # excluded from every fit, encoder, keep-set vote, refit and recalibration.
        graduate = False
        grad_eval_rows = None

        # --- DEPLOYED-BOX BUDGET (av38) --------------------------------------------------
        # The budget spends on the SAME held-out evidence the av37 gate ranks by — the
        # aggregated per-support mean drop-gain — so a support the folds paid most to keep is
        # the last one a tight budget gives up. Supports the folds never scored rank at 0.0,
        # below every positively-evidenced support and above every harmful one. It is applied
        # natively, inside the keep-set application, because that is the only place the
        # deployed bank's per-support BOX cost is known exactly; a budget of 0 never calls the
        # budgeted binding at all, so an unbudgeted fit cannot be perturbed by this code.
        box_budget = max(0, int(getattr(self, "prune_box_budget", 0) or 0))
        # --- DEPLOYED MULTI-WAY TABLE BUDGET (2026-08-27) --------------------------------
        # Ralph's revised bar: cells within a table are free, the COUNT of >=3-way tables is
        # not. Same evidence ranking and the same binary dropping as the box budget, applied in
        # the same place and composed after it; a budget of 0 never calls the budgeted binding.
        table_budget = max(0, int(getattr(self, "prune_table_budget", 0) or 0))
        table_min_arity = int(
            getattr(self, "prune_table_min_arity", _PRUNE_TABLE_MIN_ARITY_DEFAULT)
        )
        box_rank: list[tuple[list[int], float]] = []
        if box_budget > 0 or table_budget > 0:
            for row in self.pruning_report_.get("table_scores") or []:
                try:
                    box_rank.append(
                        ([int(i) for i in row["u"]], float(row.get("mean_gain") or 0.0))
                    )
                except (KeyError, TypeError, ValueError):
                    continue

        def _assemble_deploy(keep_now: list[Any]) -> "_TableModel":
            if box_budget > 0 or table_budget > 0:
                tm, box_report, table_report = full_model.apply_keepset_budgeted(
                    x32,  # see fit() call above
                    np.ascontiguousarray(y32),
                    np.ascontiguousarray(w_full),
                    keep_now,
                    box_budget,
                    box_rank,
                    reanchor=reanchor,
                    rebalance=bool(self.prune_rebalance),
                    n_jobs=self._resolve_n_jobs(),
                    exposure=expo_full,
                    cat_x=cat_x,
                    table_budget=table_budget,
                    table_min_arity=table_min_arity,
                    **self._measure_kwargs(),
                )
                # Last assembly wins: the guard's re-admission ladder re-enters here, and the
                # bank the caller ends up holding is the one this report must describe.
                if box_budget > 0:
                    self.pruning_report_["box_budget"] = json.loads(box_report)
                if table_budget > 0:
                    self.pruning_report_["table_budget"] = json.loads(table_report)
            else:
                tm = full_model.apply_keepset(
                    x32,  # see fit() call above
                    np.ascontiguousarray(y32),
                    np.ascontiguousarray(w_full),
                    keep_now,
                    reanchor=reanchor,
                    rebalance=bool(self.prune_rebalance),
                    n_jobs=self._resolve_n_jobs(),
                    exposure=expo_full,
                    cat_x=cat_x,
                    **self._measure_kwargs(),
                )
            if graduate:
                # Once per assembled artifact (the OOB guard assembles a single time after its
                # ladder; only the panel carve path assembles per rung, on the cheap legacy
                # gate). Runs BEFORE the slope re-anchor, which is the last affine step.
                tm = self._graduate_table_model(
                    tm, x32, y32, w_full, expo_full, cat_x, eval_rows=grad_eval_rows
                )
            return tm

        table_model = _assemble_deploy(keep)
        band_tol = getattr(self, "band_tolerance", _BAND_TOLERANCE_DEFAULT)
        if band_tol is not None and bool(full_model.bag_oob_available()):
            table_model = self._band_deployed(
                table_model, full_model, keep, x32, y32, w_full, expo_full, cat_x, obj,
                float(band_tol),
            )
            _lap(f"(3b) banding (tolerance {band_tol})")
        _lap(
            f"(3) deploy contribution keep-set ({len(keep)}/{len(full_supports)} tables"
            + (", graduated)" if graduate else ")")
        )

        # --- Set-level prune no-harm guard (2026-08-04; OOB evidence 2026-08-21) -----------
        # The keep-set above is chosen from PER-TABLE leave-one-out held-out gains, which are
        # structurally blind to mass carried JOINTLY by correlated tables: each table's
        # marginal drop-gain is ~0 when its siblings absorb the signal, so a whole correlated
        # family can be dropped while the fold evidence claims the total cost is ~0 (measured:
        # allstate_sev split-7 selection detonated +23% outer test deviance while a keep-set
        # from another split scored normally; homesite_conv's av32 collapsed mean+count channel
        # pair was dropped together, +7.7%). This guard evaluates the DEPLOYED artifact —
        # selected bank vs FULL bank — on honest rows, and on a breach re-admits dropped tables
        # in report-ranked order (mean_gain desc) until the selected bank is within
        # `prune_guard_tol` of the full bank. Binary table dropping is preserved; only the
        # keep-set grows. Silent (no breach) => the deployed model is byte-identical to the
        # unguarded one, and the FIT is byte-identical to a guard-off fit either way (the guard
        # is post-fit selection only).
        #
        # WHERE THE HONEST ROWS COME FROM — two evidence paths, by fit shape:
        #
        # * "oob" (UNGROUPED bagged fits — the default, every size). Each bag trained on a
        #   seeded subsample of the rows; the rows it did NOT draw are out of bag for it, and
        #   the engine records that membership (`_Model.bag_in_bag_mask`, drawn not replayed).
        #   For row r, the bags that never saw r are its jury, and the mean of their banks is
        #   an honest estimate of the deployed score at r — the mean over ALL bags reproduces
        #   the soup's deployed bank exactly (purify is linear at a fixed grid/measure), so the
        #   jury mean is that same quantity, unbiased, computed blind. This is the estimator
        #   the §G1 cell-refit's own no-harm guard already runs on tree scores (and a fit that
        #   carries such a correction keeps the guard: the correction is soup-level, so the
        #   evidence pass shares it whole across the bags — see `bag_member_model`). It costs the
        #   fit NOTHING: no carve, no size gate, and at the shipped `n_bags=8` recipe about
        #   83% of rows get a jury (measured on beMTPL16/euhealth: 83% coverage, mean jury 1.92
        #   bags).
        #   Three honest caveats, all reported in the `guard` block so they stay falsifiable:
        #     (i) a ~1.9-bag jury is noisier than the 8-bag deployed soup, and that extra
        #         variance sits in BOTH arms' deviance, so the measured ratio is diluted toward
        #         1 — this direction errs toward silence. `oob_mean_jury` records the jury size.
        #    (ii) both arms carry the deploy RE-ANCHOR before they are measured
        #         (`_guard_reanchor`). An earlier revision skipped it, on the argument that
        #         purified tables are reference-measure CENTERED so dropping them is
        #         level-neutral. THAT ARGUMENT IS WRONG in the regime the guard operates in:
        #         purification centres under the REFERENCE measure (Laplace-smoothed product of
        #         marginals), not the empirical one, and dropping most of the tables moves the
        #         empirical level a lot. Measured: allstate_sev s0 dropped 76% of its tables and
        #         showed a level_shift of 0.199 on the gamma log link, which is +13.1 points of
        #         the +15.8% gap the guard saw — it fired where the artifact-scoring av33 guard
        #         correctly stayed silent at +3.9%. s7 was worse still (0.482 shift, +70.7 of
        #         +83.4 points). `level_shift` keeps reporting the PRE-anchor gap so the
        #         artifact stays visible; `reanchored` records whether the correction applied.
        #   (iii) the arms are also raw banks in the sense that neither carries the deploy
        #         REBALANCE (`prune_rebalance`, on by default) or GRADUATION (`graduate`, on by
        #         default for poisson/gamma). Both of those improve the SELECTED arm only —
        #         rebalance re-solves the surviving cells to the pruned structure's optimum and
        #         is itself held-out-gated — so omitting them overstates the selected arm's
        #         deviance and biases the guard toward FIRING, not toward silence. The carve
        #         path scores the assembled artifact instead and has no such bias, so the two
        #         paths' `dev_selected_*` are not interchangeable. Measured on the 6-split
        #         beMTPL16/euhealth probe the total selected-vs-full gap was 0.001%-0.18% —
        #         two to three orders below the 5% default tolerance — so the bias has room to
        #         be wrong by a lot before it can move a decision. Pricing every rung through
        #         `_assemble_deploy` would remove it, at the cost of the one-pass ladder (up to
        #         ~14 full bank rebuilds per fire); the cheap version is the one that ships.
        # * "carve" (PANEL fits). Panel fits already hold a shared group-honest ES slice out of
        #   every bag (`deploy_es_holdout`), and it stays the evidence: a panel fit's OOB rows
        #   are NOT honest, because an out-of-bag row's panel-mates are almost surely in bag,
        #   which is exactly the memorization `_carve_group_holdout` exists to stop. Bags are
        #   drawn at ROW granularity, so nothing stops a group from straddling the in-bag /
        #   out-of-bag boundary. OOB is therefore not a superset here and panel fits keep their
        #   battle-tested semantics unchanged. (Their carve is an EARLY-STOPPING carve that
        #   predates the guard; nothing about it is guard-driven, and this change does not
        #   touch it either way.)
        #
        #   DEGENERATE grouping is the exception: if every group is a SINGLETON (max group size
        #   1 — e.g. an anonymised per-row policy id, which is what `euhealth`'s `id_anon`
        #   actually is despite the panel framing), no group can straddle the boundary, so OOB
        #   evidence is exactly as honest as it is on an ungrouped fit AND strictly better than
        #   the carve: ~75% of rows get a jury (the ~83% ungrouped coverage, less the shared
        #   holdout that is in bag for every bag) against the carve's ~10% single draw. Measured
        #   on euhealth: 87,952 evidence rows against the carve's 11,735. Those fits
        #   take the OOB path. Their fit is untouched — the shared ES carve is still built and
        #   still early-stops the bags; only the guard's evidence moves.
        #
        # HOW THE TWO ESTIMATORS COMPARE, measured (deploy-fidelity arena fits, av34 battery).
        # They agree in DIRECTION everywhere probed but disagree in magnitude by a few points,
        # because a ~1.9-bag jury over ~75-83% of the rows and a single ~10% carve are
        # different measurements of the same thing:
        #     homesite_conv s0  carve +7.7%  oob +8.7%   both fire      oob -0.06% test dev
        #                   s1  carve +5.6%  oob +4.8%   OOB MISSES     oob +3.93% test dev
        #                   s2  carve +5.6%  oob +8.7%   oob fires harder, -4.84% test dev
        #     allstate_sev  s0  carve +3.9%  oob +4.6%   both silent    oob -0.21% test dev
        #                   s7  carve +26%   oob +23%    both fire      oob -0.03% test dev
        # s1 is a NEAR-MISS against the 5% tolerance (4.8 vs 5.6), not a direction error — but
        # it is a real +3.9% regression on that split, and it is the reason `prune_guard_tol`
        # should be re-calibrated for this estimator rather than inherited from the carve.
        # Across those five splits the paired mean is -0.24%, and the six neutrality splits
        # (beMTPL16, euhealth) are bit-identical.
        #
        # Neither path is honest about the SELECTION itself — `fit_prune_selection` votes over
        # all n rows, so the keep-set has seen the evidence rows. That was equally true of the
        # av33 carve; the guard's job is to catch a catastrophic set-level miss, not to be an
        # unbiased generalization estimate.
        #
        # Skipped (reported) when disabled or no evidence rows are available (n_bags=1
        # has no OOB rows at all, and an ungrouped single-bag fit gets no guard — the pre-av33
        # behavior); or fewer than `_PRUNE_GUARD_MIN_ROWS` evidence rows survive.
        guard_info: dict[str, Any] = {
            "enabled": bool(self.prune_guard) and not use_path, "fired": False,
            "selection_independent": False,
            "exposure_offset": expo_full is not None,
        }
        if use_path:
            guard_info["skipped"] = "ranked_path selector"
        if self.prune_guard and not use_path:
            n_jobs = self._resolve_n_jobs()
            tol = float(self.prune_guard_tol)
            # The re-admission ladder is fixed BEFORE any evidence is scored: report-ranked
            # candidates, chunked by the doubling rule. Pinning it up front is what lets the
            # OOB path price every rung in a single pass over the per-bag banks (rung k is the
            # prefix union of chunks 0..k, and table scores are additive).
            kept_set = {tuple(sorted(int(i) for i in u)) for u in keep}
            _seen: set[tuple[int, ...]] = set()
            ranked: list[tuple[int, ...]] = []
            for t in sorted(
                self.pruning_report_.get("table_scores") or [],
                key=lambda t: -(t.get("mean_gain") or 0.0),
            ):
                u = tuple(sorted(int(i) for i in t["u"]))
                if u not in kept_set and u not in _seen:
                    _seen.add(u)
                    ranked.append(u)
            chunks: list[list[tuple[int, ...]]] = []
            _grown, _rest = len(keep), list(ranked)
            while _rest:
                take = max(1, min(len(_rest), _grown))
                chunks.append(_rest[:take])
                _rest = _rest[take:]
                _grown += take

            def _dev(raw: "np.ndarray", yv: "np.ndarray", wv: "np.ndarray") -> float:
                return _guard_mean_deviance(obj, yv, raw, wv, self.tweedie_rho)

            evidence: str | None = None
            skipped: str | None = None
            # OOB rows are honest exactly when no group can straddle the in-bag/out-of-bag
            # boundary — i.e. when there are no groups, or every group is a singleton (see the
            # DEGENERATE-grouping note above). Bags are drawn at row granularity, so any real
            # panel structure disqualifies them.
            #
            # Since av35 the second disjunct is UNREACHABLE from `fit()`: `_fit_model` normalizes
            # a degenerate grouping to `None` at ingest, so such a fit arrives here already
            # ungrouped (and got the ungrouped fit too — the carve it used to pay for is gone).
            # It is kept as a fallback for a direct `_fit_and_prune` call with raw groups, and
            # because it is the same predicate the normalizer uses (one shared helper, so the
            # two can never drift into disagreeing about what "degenerate" means).
            # With group-aware bags (`bag_codes`, 2026-09-07) a grouped fit's out-of-bag rows are
            # honest: no group straddles a bag boundary, so the jury never trained on the row's
            # group. Measured on fremotor_prem (gamma severity): the out-of-bag guard shipped
            # 0.1266 vs the carve guard's 0.1222 on split 0 and tied on split 1, on 25.8k
            # evidence rows against the carve's 3.5k. (A leaky row-bag out-of-bag jury must never
            # be read here: it re-admits the whole bank on memorised rows.)
            oob_honest = (
                groups_arr is None or _is_degenerate_grouping(groups_arr) or bag_codes is not None
            )
            # ASK, don't catch: `bag_oob_available` is false exactly for a fit with no bag
            # partition/membership — in practice `n_bags=1`. Anything the evidence
            # pass itself raises — a mismatched design, an allocation failure — is a real bug
            # and must surface, not be silently reported as "no out-of-bag rows".
            # Called DIRECTLY, never through a `getattr(..., default)`: a missing attribute
            # means the compiled module is older than this file (a stale `maturin develop`),
            # and a default of False would silently disable the guard on exactly the large
            # fits it exists for — which is what happened once during this feature's own
            # validation probe. Let the AttributeError surface instead.
            oob_ready = bool(full_model.bag_oob_available())

            def _run_ladder(
                dev_of_rung: Any, dev_full: float, n_val: int, tag: str, extra: dict[str, Any],
                assemble_each_rung: bool, gap_se: float = float("inf"),
            ) -> None:
                """Breach test + report-ranked re-admission ladder, shared by both evidence
                paths. `dev_of_rung(k)` is rung k's deviance (k=0 is the deployed keep-set,
                k>=1 adds `chunks[k-1]`); it may assemble the deploy model as a side effect
                (the carve path scores the artifact itself), which is what
                `assemble_each_rung` records. Only the keep-set ever grows — binary table
                dropping is preserved.
                """
                nonlocal keep, table_model
                dev_sel = dev_of_rung(0)
                # --- SE-AWARE BREACH TEST (av37) ------------------------------------------
                # av36 breached on a FIXED 5% relative gap, whatever the jury's own spread. The
                # jury is a ~1.9-bag mean over out-of-bag rows and its verdict correlates
                # -0.004 / -0.267 / +0.092 with the eventual test verdict on the trio, so on a
                # small or heavy-tailed portfolio a 5% gap is routinely inside the noise — and
                # every spurious fire GROWS the deployed bank, which is the one thing the
                # product cannot absorb. The bar is now max(tol, z * SE(gap)): the guard may
                # only re-admit on evidence that is statistically real. Where the jury is sharp
                # (large sets: SE well under a point) z*SE sits below `tol` and the test is
                # exactly av36's. `prune_guard_z = 0.0` restores av36 unconditionally.
                #
                # The SAME bar terminates the re-admission ladder, so a fire that clears it
                # stops as soon as the residual gap is no longer distinguishable — that is what
                # keeps a real detonation's re-admission proportionate instead of shipping the
                # whole bank.
                #
                # --- SE-AWARE DOWNWARD RECALIBRATION (av39) --------------------------------
                # The upward bar above could only ever make the guard fire LESS. The homesite
                # forensic found the opposite failure — a 0.21%-0.27% SE against a 5% bar — in
                # BOTH shapes: a silent +4.80% trigger miss and a ladder that fired and then
                # stopped one rung in at a 4.48% residual. `_guard_tol_effective` composes the
                # two corrections; `min(tol, .)` means the recalibration can only tighten, so a
                # blunt jury (ohlsson_pp, SE 9.56%) is pinned back at `tol` and bit-identical.
                # BOTH the breach test and the `while` below read `tol_eff` through
                # `threshold` — the stopping rule is as load-bearing as the trigger.
                z = float(self.prune_guard_z)
                z_dn = float(self.prune_guard_z_dn)
                tol_floor = float(self.prune_guard_tol_floor)
                tol_eff = _guard_tol_effective(tol, gap_se, z, z_dn, tol_floor)
                guard_info.update(
                    evidence=tag, holdout_rows=n_val, tol=tol, dev_full=float(dev_full),
                    dev_selected_initial=float(dev_sel), kept_initial=len(keep),
                    guard_z=z, gap_se=(float(gap_se) if np.isfinite(gap_se) else None),
                    guard_z_dn=z_dn, tol_floor=tol_floor, n_chunks=len(chunks),
                    tol_effective=float(tol_eff), **extra,
                )
                threshold = dev_full * (1.0 + tol_eff)
                if not (np.isfinite(dev_sel) and np.isfinite(dev_full) and dev_sel > threshold):
                    _lap(f"(6) prune guard [{tag}] (silent)")
                    return
                steps = 0
                while dev_sel > threshold and steps < len(chunks):
                    steps += 1
                    dev_sel = dev_of_rung(steps)
                keep_now = [sorted(int(i) for i in u) for u in keep]
                for c in chunks[:steps]:
                    keep_now += [list(u) for u in c]
                if not assemble_each_rung:
                    table_model = _assemble_deploy(keep_now)
                guard_info.update(
                    fired=True, steps=steps, dev_selected_final=float(dev_sel),
                    kept_final=len(keep_now),
                )
                keep = keep_now  # the reconciliation below ships what we deployed
                self.pruning_report_["kept"] = [sorted(int(i) for i in u) for u in keep_now]
                _lap(
                    f"(6) prune guard FIRED [{tag}] ({steps} step(s), kept "
                    f"{guard_info['kept_initial']}->{len(keep_now)}, evidence dev "
                    f"{guard_info['dev_selected_initial']:.5f}->{dev_sel:.5f} vs full "
                    f"{dev_full:.5f})"
                )

            # HONESTY FLAG, not a control-flow change. The OOB rung path's rungs are raw
            # per-support bank sums over `keep` — the SELECTION keep-set — assembled from the
            # full model. What ships is `_assemble_deploy(keep)`, which then spends
            # `prune_box_budget` / `prune_table_budget`. When a budget actually ENGAGES, rung 0
            # therefore describes a bank that is not deployed: the breach test can read silent
            # while the shipped bank is worse than the full one, and a ladder that fires
            # re-admits tables the budget immediately re-caps, reporting a `kept_final` the
            # artifact does not carry.
            #
            # It is flagged rather than skipped, deliberately. Skipping was tried and it BREAKS
            # the knob's own no-op guarantee: at n=1200 the guard is eligible
            # (`_PRUNE_GUARD_MIN_ROWS` is 500), so a NON-binding cap — one at or above the
            # bank's own count, which must be a verbatim no-op — would have suppressed a guard
            # fire and shipped 4 three-way tables where the unbudgeted fit ships 10. Changing
            # control flow here also silently moves every shipped `prune_box_budget` cell.
            #
            # The CARVE path below is already honest: it scores `table_model`, the assembled and
            # budgeted deploy. The real repair for the OOB path is to assemble each rung as the
            # carve path does, paying one deploy assembly per step. That is the follow-up; until
            # then the report says so out loud.
            guard_min_rows = int(getattr(self, "prune_guard_min_rows", _PRUNE_GUARD_MIN_ROWS))
            if oob_honest and oob_ready:
                # --- OOB path (ungrouped, or singleton groups) ----------------------------
                _t_ev = _time.perf_counter()
                group_arg = [[sorted(int(i) for i in u) for u in keep]]
                group_arg += [[list(u) for u in c] for c in chunks]
                full_sum, f0_sum, group_sums, counts = full_model.bag_oob_group_raw(
                    x32,  # see fit() call above — MUST be the fit design, rows in fit order
                    group_arg,
                    cat_x=cat_x,
                    n_jobs=n_jobs,
                    # Same measure AND same per-row mass as the deployed bank, or the OOB jury
                    # would score a different ledger than the one that ships.
                    weight=np.ascontiguousarray(w_full),
                    exposure=expo_full,
                    **self._measure_kwargs(),
                )
                counts = np.asarray(counts, dtype=np.int64)
                ev = counts > 0
                n_val = int(ev.sum())
                if n_val < guard_min_rows:
                    skipped = f"out-of-bag rows {n_val} < {guard_min_rows}"
                else:
                    evidence = "oob"
                    cnt = counts[ev].astype(np.float64)
                    yv = np.asarray(y32, dtype=np.float64)[ev]
                    wv = np.asarray(w_full, dtype=np.float64)[ev]
                    offset_ev = (np.log(np.asarray(expo_full, dtype=np.float64)[ev])
                                 if expo_full is not None else np.zeros(n_val))
                    raw_full = np.asarray(full_sum, dtype=np.float64)[ev] / cnt + offset_ev
                    # Rung 0 is the deployed keep-set; each further rung adds one chunk. Every
                    # rung is the running PREFIX union, so the intercept enters once and each
                    # group's tables accumulate on top of it.
                    group_sums = np.asarray(group_sums, dtype=np.float64)
                    acc = np.asarray(f0_sum, dtype=np.float64)[ev] / cnt + offset_ev
                    rungs = []
                    for g in range(len(group_arg)):
                        acc = acc + group_sums[g][ev] / cnt
                        rungs.append(acc)
                    # Both arms carry the deploy re-anchor before they are measured (see
                    # `_guard_reanchor`): without it the guard tests a level artifact that the
                    # shipped model does not have. `level_shift` below reports the RAW (pre-
                    # anchor) gap, so the artifact stays visible in the report.
                    raw_gap = float(np.average(rungs[0] - raw_full, weights=wv))
                    raw_full = _guard_reanchor(obj, raw_full, yv, wv, reanchor)
                    dev_full = _dev(raw_full, yv, wv)
                    # Paired per-row deviances of the two arms the breach test compares — the
                    # spread of THAT difference is what sizes the SE-aware bar (see
                    # `_guard_gap_se` and the note in `_run_ladder`).
                    gap_se = _guard_gap_se(
                        _guard_unit_deviance(
                            obj, yv, _guard_reanchor(obj, rungs[0], yv, wv, reanchor), wv,
                            self.tweedie_rho,
                        ),
                        _guard_unit_deviance(obj, yv, raw_full, wv, self.tweedie_rho),
                        wv,
                    )
                    _run_ladder(
                        lambda k: _dev(
                            _guard_reanchor(obj, rungs[k], yv, wv, reanchor), yv, wv
                        ),
                        dev_full,
                        n_val,
                        "oob",
                        {
                            "oob_rows": n_val,
                            "oob_rows_uncovered": int(counts.size - n_val),
                            "oob_mean_jury": float(cnt.mean()),
                            "n_bags": int(self.n_bags),
                            "level_shift": raw_gap,
                            "reanchored": bool(reanchor),
                            "evidence_seconds": float(_time.perf_counter() - _t_ev),
                        },
                        assemble_each_rung=False,
                        gap_se=gap_se,
                    )
                    # --- post-prune slope re-anchor (log-link objectives only) -------------
                    # The guard has settled the FINAL keep-set, so `rungs[final_k]` is the
                    # out-of-bag link-scale score of exactly the bank that is about to ship —
                    # the one array in this method on which the correction is honest. Fitting
                    # it on any EARLIER rung and applying it to a bank the ladder later grew is
                    # catastrophic, not merely wasteful: measured on allstate_sev split 3, the
                    # pre-guard rung's b = 1.23 (it is describing a 1,934-table bank) applied to
                    # the 10,307-table bank the guard actually deployed took the split from
                    # +0.0926 to +0.0091. `final_k` is read from the ladder's own outcome for
                    # that reason, never assumed to be 0.
                    if obj in _SLOPE_OBJECTIVES and expo_full is None:
                        final_k = (
                            int(guard_info.get("steps") or 0)
                            if guard_info.get("fired")
                            else 0
                        )
                        table_model = self._slope_reanchor_post_prune(
                            table_model, obj, rungs[final_k], yv, wv, reanchor,
                            final_k, x32, cat_x, n_jobs, _lap, raw_full,
                        )
                    elif expo_full is not None:
                        self.pruning_report_["slope"] = {
                            "skipped": "native exposure offset: fixed offset is not rescaled"
                        }
            if evidence is None and deploy_es_holdout is not None:
                # --- Shared-carve path — av33 semantics, unchanged -------------------------
                # Reached by every real panel fit, and as a FALLBACK when a singleton-grouped
                # fit cannot supply out-of-bag evidence (n_bags=1, a §G1 cell correction, or
                # too few covered rows): its carve is already held out of the fit, so using it
                # is strictly better than dropping the guard.
                _t_ev = _time.perf_counter()
                gmask = np.asarray(deploy_es_holdout, dtype=bool)
                n_val = int(gmask.sum())
                if n_val < guard_min_rows:
                    skipped = f"holdout_rows {n_val} < {guard_min_rows}"
                else:
                    evidence = "carve"
                    skipped = None
                    xv = np.ascontiguousarray(np.asarray(x32)[gmask])
                    cat_xv = (
                        [list(np.asarray(col, dtype=object)[gmask]) for col in cat_x]
                        if cat_x
                        else None
                    )
                    yv = np.asarray(y32, dtype=np.float64)[gmask]
                    wv = np.asarray(w_full, dtype=np.float64)[gmask]
                    raw_full = np.asarray(
                        full_model.predict_raw(xv, cat_x=cat_xv, n_jobs=n_jobs), dtype=np.float64
                    )
                    offset_val = (np.log(np.asarray(expo_full, dtype=np.float64)[gmask])
                                  if expo_full is not None else np.zeros(n_val))
                    raw_full = raw_full + offset_val
                    dev_full = _dev(raw_full, yv, wv)
                    # Same SE-aware bar as the OOB path, measured on the artifact this path
                    # scores (rung 0 is the deployed keep-set as assembled above).
                    gap_se = _guard_gap_se(
                        _guard_unit_deviance(
                            obj, yv,
                            np.asarray(
                                table_model.predict_raw(xv, cat_x=cat_xv, n_jobs=n_jobs),
                                dtype=np.float64,
                            ) + offset_val,
                            wv, self.tweedie_rho,
                        ),
                        _guard_unit_deviance(obj, yv, raw_full, wv, self.tweedie_rho),
                        wv,
                    )

                    def _carve_rung(k: int) -> float:
                        # The carve path scores the ARTIFACT: rung k is re-assembled with
                        # reanchor/rebalance/graduation before it is measured.
                        nonlocal table_model
                        if k:
                            keep_k = [list(u) for u in keep]
                            for c in chunks[:k]:
                                keep_k += [list(u) for u in c]
                            table_model = _assemble_deploy(keep_k)
                        raw = np.asarray(
                            table_model.predict_raw(xv, cat_x=cat_xv, n_jobs=n_jobs),
                            dtype=np.float64,
                        )
                        return _dev(raw + offset_val, yv, wv)

                    _run_ladder(
                        _carve_rung, dev_full, n_val, "carve",
                        {"evidence_seconds": float(_time.perf_counter() - _t_ev)},
                        assemble_each_rung=True, gap_se=gap_se,
                    )
            if evidence is None and skipped is None:
                skipped = (
                    "no honest evidence: a panel fit with validation_fraction unset"
                    if not oob_honest
                    else "no out-of-bag evidence (single-bag fit)"
                )
            if evidence is None and skipped is not None:
                guard_info["skipped"] = skipped
        # See the note above the OOB branch: that path's rungs describe the pre-budget keep-set,
        # so once a budget actually ENGAGES its verdict is about a bank that did not ship. Said
        # in the report rather than silently, and only when a budget really bound — a budget
        # that stayed idle left the deployed bank equal to the keep-set, so the verdict holds.
        if guard_info.get("evidence") == "oob" and any(
            (self.pruning_report_.get(k) or {}).get("engaged")
            for k in ("box_budget", "table_budget")
        ):
            guard_info["budget_blind"] = True
        self.pruning_report_["guard"] = guard_info

        # Reconcile the selection report with the shipped artifact: keep-set members whose tables
        # purify to nothing (typically order-3 candidates on narrow data) never reach the deployed
        # bank, and the report used to silently overstate the model.
        deployed = sorted(tuple(u) for u in table_model.deployed_supports())
        kept_keys = {tuple(sorted(int(i) for i in u)) for u in self.pruning_report_.get("kept", [])}
        self.pruning_report_["deployed"] = [list(u) for u in deployed]
        self.pruning_report_["kept_not_deployed"] = [
            list(u) for u in sorted(kept_keys - set(deployed))
        ]
        return table_model

    # --- Whittaker-Henderson graduation of the deployed fANOVA tables (post-fit) -------------
    # The classical actuarial smoother applied to the glassbox artifact: each dense 1-D/2-D
    # table's cells v (link scale) with per-cell training support s are replaced by
    #   v' = argmin  sum_c s_c (v'_c - v_c)^2 + alpha * ||D2 v'||^2
    # (second differences along each ordinal axis; the leading zero-support missing slot is
    # excluded from the chain — missing is not ordinally adjacent). alpha is chosen PER TABLE by
    # generalized cross-validation — the canonical WH lambda selection — with alpha = 0
    # admissible, so a table whose steps are real (aggregated tariff cells) is left untouched
    # while a noisy small-portfolio shape borrows strength from its neighbours. Factored
    # (sparse 3-D) tables and very large grids are left alone. After smoothing, f0 is
    # re-anchored on the training data so the aggregate balance holds.
    #
    # ORDINAL AXES ONLY: categorical-TS axes are excluded from the difference chain — their
    # cell order is the (noisy) risk-sorted encoding, not an ordinal scale, and smoothing
    # along it irons out real discrete level differences (measured: euhealth relation×gender
    # carried +3.0e-4 of a +3.6e-4 all-splits test-deviance regression, 2026-07-10). Pure-cat
    # tables are left verbatim; mixed 2-D tables smooth along the numeric direction only.
    _GRADUATION_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4)
    # Two-tier size budget (2026-07-16). `graduation_tables` (Rust) is asked for the generous
    # `_GRADUATION_MAX_CELLS_RAW` ceiling because it runs BEFORE axis kinds are resolved, so it
    # cannot yet tell a cheap SEPARABLE table from an expensive JOINT one. The real budget is then
    # enforced here in `_graduate_one`, per table, once row_smooth/col_smooth are known:
    #   - JOINT path (both directions smoothed, or a 1-D table): `_GRADUATION_MAX_CELLS` cells,
    #     since the cost is one dense O((m*n)^3) eigendecomposition over the flattened grid.
    #   - SEPARABLE path (exactly one direction smoothed): the cost is n_levels independent
    #     O(chain_len^3) eigendecompositions (see `_dr_family_separable`) — cheap even at large
    #     `m*n` because the levels never couple — so it is gated on `chain_len` alone, not cells.
    # Without this split, a cheap separable table (e.g. a [245, 12] numeric x categorical grid,
    # 0.12s post-separable-rewrite) was wrongly excluded by the joint-cost cap before it ever
    # reached Python.
    _GRADUATION_MAX_CELLS = 2500
    _GRADUATION_MAX_CELLS_RAW = 32768
    _GRADUATION_MAX_SEPARABLE_CHAIN = 512

    @staticmethod
    def _d2(k: int) -> "np.ndarray":
        d = np.zeros((max(k - 2, 0), k))
        for i in range(k - 2):
            d[i, i], d[i, i + 1], d[i, i + 2] = 1.0, -2.0, 1.0
        return d

    @staticmethod
    def _dr_family(
        v: "np.ndarray", s: "np.ndarray", penalty: "np.ndarray"
    ) -> "Callable[[float], tuple[np.ndarray, float]] | None":
        """Factor the ridge family alpha -> ((W+aP)^-1 W v, df) once (Demmler-Reinsch).

        Exactly the per-alpha dense solve, restructured: one symmetric eigendecomposition
        per table instead of a k-RHS k x k solve per alpha (the O(k^3) x |grid| cost that
        dominated whole fits on near-cap tables). Zero-support cells are eliminated
        EXACTLY first — their stationarity `P00 v0' = -P0+ v+'` is alpha-free, so the
        observed block sees the alpha-independent Schur-complement penalty and
        `df = tr[(W+aP)^-1 W]` reduces to the observed block by the standard Schur
        identity.

        The minimum-norm fill `x0 = pinv(P00, hermitian=True) @ P0+` is used
        UNCONDITIONALLY (2026-07-16, Ralph-ratified) rather than only as a fallback when
        a Cholesky factorization of `P00` raises: on exactly-singular blocks (sparse
        2-way surfaces whose empty region has flat difference-penalty directions),
        whether Cholesky raises is a BLAS-rounding- and thread-count-dependent coin
        flip, not a property of the table — the review panel measured it flipping
        outcome on 6/14 such blocks between 1 and 22 OpenBLAS threads on the same
        machine, which made the zero-support fill non-reproducible across environments.
        That divergence is invisible to the model's OWN guarantees, though: for any null
        vector z of the PSD `P00` (i.e. any direction the Cholesky/pinv choice could
        disagree on), `P0+ @ z = 0`, so the generalized Schur complement — and hence
        every OBSERVED-cell result (smoother, df, chosen alpha) — is provably
        branch-invariant. Always taking the pinv path removes the environment-dependent
        divergence entirely at zero extra cost (pinv was already the fallback compute
        path); observed-cell smoother/df/alphas are unchanged by the switch.
        Returns a closure `alpha -> (v_s, df)`, or None only on genuine LAPACK failure
        (the dense per-alpha fallback remains as the last resort).
        """
        pos = s > 0
        try:
            if pos.all():
                d = np.sqrt(s)
                x0 = None
            else:
                p00 = penalty[np.ix_(~pos, ~pos)]
                p0p = penalty[np.ix_(~pos, pos)]
                # Always the minimum-norm fill (see docstring) — no Cholesky-first branch.
                # x0 = pinv(P00) @ P0+ ; v0'(alpha) = -x0 @ v+' for every alpha
                x0 = np.linalg.pinv(p00, hermitian=True) @ p0p
                penalty = penalty[np.ix_(pos, pos)] - penalty[np.ix_(pos, ~pos)] @ x0
                d = np.sqrt(s[pos])
            m = penalty / d[:, None] / d[None, :]
            lam, q = np.linalg.eigh((m + m.T) / 2.0)
            lam = np.clip(lam, 0.0, None)
            u = q.T @ (d * (v[pos] if x0 is not None else v))
        except np.linalg.LinAlgError:
            return None

        def family(alpha: float) -> tuple["np.ndarray", float]:
            g = 1.0 / (1.0 + alpha * lam)
            v_pos = (q @ (g * u)) / d
            if x0 is None:
                return v_pos, float(g.sum())
            out = np.empty_like(v)
            out[pos] = v_pos
            out[~pos] = -(x0 @ v_pos)
            return out, float(g.sum())

        return family

    @classmethod
    def _dr_family_separable(
        cls, v2: "np.ndarray", s2: "np.ndarray"
    ) -> "Callable[[float], tuple[np.ndarray, float]] | None":
        """Demmler-Reinsch family for a 2-D block smoothed along axis 0 ONLY.

        The penalty ``kron(D'D, I)`` has no coupling across axis-1 levels, so ``(W + aP)``
        is block-diagonal per level and the ridge family factors EXACTLY into independent
        per-level chains: n eigendecompositions of size m instead of one of size m*n (the
        81x21 MTPL table: 21 x O(80^3) vs O(1680^3) — measured 79% of the graduation wall
        sat in these single-direction tables, 2026-07-16). ``v_s`` blocks concatenate and
        ``df`` sums, both by the block-diagonal structure. A never-observed level gets the
        minimum-norm stationary value — the zero vector — matching what the joint pinv
        path assigns (its ``x0`` columns for a decoupled empty level are zero).
        """
        m, n = v2.shape
        pen = None
        levels: list[tuple[str, Any]] = []
        for j in range(n):
            vj, sj = v2[:, j], s2[:, j]
            if not (sj > 0).any():
                levels.append(("empty", None))
                continue
            if pen is None:
                d = cls._d2(m)
                pen = d.T @ d
            fam = cls._dr_family(np.ascontiguousarray(vj), np.ascontiguousarray(sj), pen)
            if fam is None:
                return None  # genuine LAPACK failure on a level: joint path decides
            levels.append(("fam", fam))

        def family(alpha: float) -> tuple["np.ndarray", float]:
            out = np.empty((m, n))
            df = 0.0
            for j, (kind, fam) in enumerate(levels):
                if kind == "empty":
                    out[:, j] = 0.0
                    continue
                v_s, dfj = fam(alpha)
                out[:, j] = v_s
                df += dfj
            return out.ravel(), float(df)

        return family

    @classmethod
    def _gcv_smooth(cls, v: "np.ndarray", s: "np.ndarray", penalty: "np.ndarray | None",
                    fixed_alpha: float | None,
                    family: "Callable[[float], tuple[np.ndarray, float]] | None" = None,
                    ) -> tuple["np.ndarray", float]:
        """Return (smoothed cells, chosen alpha) for one flattened table block.

        ``family`` lets the caller supply a prebuilt ridge family (the separable
        block-diagonal one); ``penalty`` may then be None and the dense fallback is
        unreachable. Without it, the family is built from ``penalty`` as before.
        """
        n_data = int(np.sum(s > 0))
        if n_data < 3 or (penalty is not None and penalty.shape[0] != v.shape[0]):
            return v, 0.0
        if fixed_alpha is not None:
            alpha = float(fixed_alpha)
            if alpha <= 0.0:
                return v, 0.0
            if family is not None:
                v_s, _ = family(alpha)
                return (v_s, alpha) if np.all(np.isfinite(v_s)) else (v, 0.0)
            # A fixed alpha needs one penalized estimate, not an eigendecomposition of the whole
            # smoothing family or its effective DoF. Solve the one RHS directly.
            if penalty is None:  # no family and no penalty: nothing to smooth with
                return v, 0.0
            try:
                v_s = np.linalg.solve(np.diag(s) + alpha * penalty, s * v)
            except np.linalg.LinAlgError:
                return v, 0.0
            return (v_s, alpha) if np.all(np.isfinite(v_s)) else (v, 0.0)
        # alpha=0 keeps the table verbatim; its GCV is 0/0-degenerate, so treat it as the
        # incumbent and require a strictly finite, lower-GCV smoother to displace it via the
        # 1-SE-free classic rule: minimize GCV over positive alphas, adopt only if the winner's
        # GCV beats the smallest positive-alpha baseline trend (standard WH-GCV practice).
        candidates = cls._GRADUATION_ALPHAS
        if family is None:
            if penalty is None:  # no family and no penalty: nothing to smooth with
                return v, 0.0
            family = cls._dr_family(v, s, penalty)
        gcvs = []
        for alpha in candidates:
            if alpha is None or alpha <= 0:
                continue
            if family is not None:
                v_s, df = family(float(alpha))
            elif penalty is None:  # unreachable: a missing family is built from `penalty` above
                continue
            else:
                # Dense per-alpha fallback (the Demmler-Reinsch factorization above declined a
                # singular zero-support block). `w_mat` is local to this branch — computed
                # fresh from `s` rather than threaded in as an Optional from outside — so its
                # type is a plain ndarray, not `ndarray | None`, matching what `solve` expects.
                w_mat = np.diag(s)
                a_mat = w_mat + float(alpha) * penalty
                try:
                    x_mat = np.linalg.solve(a_mat, w_mat)
                except np.linalg.LinAlgError:
                    continue
                v_s = x_mat @ v
                df = float(np.trace(x_mat))
            denom = 1.0 - df / n_data
            if denom <= 1e-6 or not np.all(np.isfinite(v_s)):
                continue
            gcv = float(np.sum(s * (v - v_s) ** 2)) / (denom * denom) / n_data
            gcvs.append((gcv, float(alpha), v_s))
        if not gcvs:
            return v, 0.0
        gcvs.sort(key=lambda t: (t[0], t[1]))
        # adopt the GCV minimizer only if it beats the LEAST-smoothed candidate's GCV — the
        # alpha->0+ limit of the curve stands in for the (degenerate) alpha=0 incumbent.
        g_min, alpha, v_s = gcvs[0]
        g_smallest_alpha = min(gcvs, key=lambda t: t[1])[0]
        if g_min < g_smallest_alpha:
            return v_s, alpha
        return v, 0.0

    @staticmethod
    def _clamp_extrapolated_fill(vals: "np.ndarray", sup: "np.ndarray") -> "np.ndarray":
        """Clamp zero-support (unobserved) cells to the span the observed cells justify.

        The zero-support stationary fill `v0' = -(x0 @ v+')` (see `_dr_family`) is an
        UNBOUNDED affine extrapolation on the smoothed (often log) scale: a categorical
        level with exactly 2 observed bins gets a straight line through 2 noisy points,
        projected across the whole chain. Neither GCV (support-weighted) nor the
        train-deviance adoption guard in `apply_graduation` (zero-weight rows) can see
        this — both are blind to unobserved cells by construction. Bound every
        unobserved cell to `[min(observed, 0), max(observed, 0)]`; including 0 keeps the
        benign "no effect" fill admissible even when every observed cell sits on one
        side of it. Returns `vals` unchanged when there is nothing to clamp (no
        zero-support cells, or no observed cells to bound against).
        """
        observed = sup > 0
        if not observed.any() or observed.all():
            return vals
        obs_vals = vals[observed]
        lo = min(float(obs_vals.min()), 0.0)
        hi = max(float(obs_vals.max()), 0.0)
        vals = vals.copy()
        vals[~observed] = np.clip(vals[~observed], lo, hi)
        return vals

    @classmethod
    def _graduate_one(
        cls, item: tuple[Any, ...], fixed: float | None, w_scale: float
    ) -> "tuple[int, Any, np.ndarray, float] | None":
        """Graduate ONE table (the former serial loop body of `_graduate_table_model`).

        Returns (table_idx, features, smoothed cells, alpha), or None when the table has no
        ordinal direction to smooth along (pure-cat / too small / no penalty) — the old
        `continue` cases, which produce neither an update nor a report entry. Pure function of
        its arguments plus class constants, so the caller may fan tables out on a thread pool;
        each call is exactly the serial computation (same BLAS threading, same op order)."""
        table_idx, features, shape, values, support, axis_cat = item
        v = np.asarray(values, dtype=float)
        sup = np.asarray(support, dtype=float) * w_scale
        if len(shape) == 1:
            if axis_cat and axis_cat[0]:
                return None  # pure-cat table: no ordinal direction to smooth along
            # Safety guard: `graduation_tables()` is now asked for the generous
            # `_GRADUATION_MAX_CELLS_RAW` ceiling (see the two-tier cap comment on
            # `_GRADUATION_MAX_CELLS`), which no longer enforces the joint budget for 1-D
            # tables upstream — enforce it here instead.
            if v.size > cls._GRADUATION_MAX_CELLS:
                return None
            # Bin 0 is the RESERVED missing slot on a numeric axis (core bin.rs) whether or
            # not this particular training set actually populated it — missing is not
            # ordinally adjacent to bin 1, so it never enters the D2 chain, regardless of
            # its support (a populated missing bin must not be smoothed toward its
            # numeric neighbors either).
            start = 1 if v.size > 1 else 0
            pen_d = cls._d2(v.size - start)
            pen = pen_d.T @ pen_d
            v_s, alpha = cls._gcv_smooth(v[start:], sup[start:], pen, fixed)
            if alpha > 0.0:
                v_s = cls._clamp_extrapolated_fill(v_s, sup[start:])
            out = v.copy(); out[start:] = v_s
        elif len(shape) == 2:
            if len(axis_cat) == 2 and all(axis_cat):
                return None  # cat x cat: no ordinal direction to smooth along
            v2, s2 = v.reshape(shape), sup.reshape(shape)
            # Same reserved-missing-bin exclusion as the 1-D case, per numeric axis —
            # unconditional on support, and only for axes that ARE numeric (a categorical
            # axis has no reserved bin-0 convention; its own D2 contribution is already
            # gated off below by axis_cat).
            r0 = 1 if (shape[0] > 1 and not (axis_cat and axis_cat[0])) else 0
            c0 = 1 if (shape[1] > 1 and not (len(axis_cat) == 2 and axis_cat[1])) else 0
            vv, ss = v2[r0:, c0:], s2[r0:, c0:]
            m, n = vv.shape
            if m * n < 4 or (m < 3 and n < 3):
                return None
            row_smooth = m >= 3 and not (axis_cat and axis_cat[0])
            col_smooth = n >= 3 and not (len(axis_cat) == 2 and axis_cat[1])
            if not (row_smooth or col_smooth):
                return None  # the only smoothable direction was categorical or too short
            # Two-tier size budget (see the `_GRADUATION_MAX_CELLS` comment): a JOINT table
            # (both directions smoothed) pays for one dense O((m*n)^3) eigendecomposition, so
            # it stays capped on total cells. A SEPARABLE table (exactly one direction
            # smoothed) factors into independent per-level chains — its cost is n_levels x
            # O(chain_len^3) — so it is capped on the smoothed chain length instead, which lets
            # it stay eligible even when m*n alone would exceed the joint cap.
            if row_smooth and col_smooth:
                if m * n > cls._GRADUATION_MAX_CELLS:
                    return None
            else:
                chain_len = m if row_smooth else n
                if chain_len > cls._GRADUATION_MAX_SEPARABLE_CHAIN:
                    return None
            family = None
            pen = None
            if row_smooth != col_smooth:
                # Exactly one smoothed direction (numeric x cat, or one axis too short):
                # kron(D'D, I) has no coupling across the unsmoothed axis, so the ridge
                # family separates EXACTLY into per-level chains — measured 79% of the
                # graduation wall sat in these tables (2026-07-16).
                v_o = np.ascontiguousarray(vv if row_smooth else vv.T)
                s_o = np.ascontiguousarray(ss if row_smooth else ss.T)
                sep = cls._dr_family_separable(v_o, s_o)
                if sep is not None:
                    if row_smooth:
                        family = sep
                    else:
                        def family(
                            alpha: float, _sep: Any = sep, _shape: tuple[int, ...] = v_o.shape
                        ) -> tuple[np.ndarray, float]:
                            v_s, df = _sep(alpha)
                            return v_s.reshape(_shape).T.ravel(), df
            if family is None:
                pen = np.zeros((m * n, m * n))
                # kron(D, I)'kron(D, I) == kron(D'D, I) (and the I-first twin): assemble the
                # penalty from the small m x m / n x n Gram products instead of materializing
                # the ((m-2)n x mn) difference operator and paying an O((mn)^2 (m-2)n) dgemm
                # per direction (~15 Gflop per near-cap table). Bit-identical: the big dgemm's
                # extra summands are exact +0.0 terms, so each entry reduces to the same <=3
                # products added in the same k-order.
                if row_smooth:
                    d = cls._d2(m)
                    pen += np.kron(d.T @ d, np.eye(n))
                if col_smooth:
                    d = cls._d2(n)
                    pen += np.kron(np.eye(m), d.T @ d)
            v_s, alpha = cls._gcv_smooth(vv.ravel(), ss.ravel(), pen, fixed, family=family)
            if alpha > 0.0:
                # Single clamp point for BOTH the joint and separable paths: both funnel
                # through this one `_gcv_smooth` call, so one call here covers either.
                v_s = cls._clamp_extrapolated_fill(v_s, ss.ravel())
            out2 = v2.copy(); out2[r0:, c0:] = v_s.reshape(m, n)
            out = out2.ravel()
        else:
            return None
        return int(table_idx), features, out, alpha

    # --- Post-prune slope re-anchor ----------------------------------------------------------
    # Deliberately SELF-GATING rather than parameterised: there is no new estimator knob. The
    # correction is fitted on out-of-bag evidence the guard has already paid for, it is reported
    # in full under `pruning_report_["slope"]` whether or not it fires, and it is applied only
    # when the fitted slope is far enough from 1 that the bank demonstrably needs rescaling. A
    # fit that does not need it keeps a byte-identical artifact.
    def _slope_reanchor_post_prune(
        self, table_model: "_TableModel", obj: str, rung_raw: "np.ndarray",
        yv: "np.ndarray", wv: "np.ndarray", reanchor: bool, rung: int,
        x32: "np.ndarray", cat_x: "list[list[str]] | None", n_jobs: int | None,
        lap: "Callable[[str], None]", raw_full: "np.ndarray",
    ) -> "_TableModel":
        """Fit `a + b*F` on the FINAL keep-set's OOB scores and fold it into the shipped bank.

        Returns the rescaled model, or `table_model` UNCHANGED (and byte-identical) when the
        correction is a no-op, is not trustworthy, or does not verify. Every outcome is recorded
        in `pruning_report_["slope"]`.
        """
        slope_eps = float(getattr(self, "prune_slope_eps", _SLOPE_EPS))
        slope_min_z = float(getattr(self, "prune_slope_min_z", _SLOPE_MIN_Z))
        info: dict[str, Any] = {"rung": int(rung), "evidence": "oob", "eps": slope_eps}
        # Measure on the same re-anchored footing the guard used, so `dev_before` is comparable
        # with `guard["dev_selected_final"]` and the reported gain is the correction's alone.
        raw = _guard_reanchor(obj, rung_raw, yv, wv, reanchor)
        rawf = np.asarray(raw_full, dtype=np.float64)
        # --- the estimator: the slope of the keep-set RELATIVE to the full bank ---------------
        # Fitting `b` on the keep-set's out-of-bag score alone measures TWO things at once, and
        # only one of them is the prune's fault. An out-of-bag score is an average over the bags
        # that did NOT see the row, so it is a systematically weaker (under-fit) predictor than
        # the full-data artifact that actually ships, and it reads b > 1 EVEN WITH NO PRUNING AT
        # ALL. Measured: on the healthy 8k guard fixture the FULL support set — nothing pruned,
        # sd ratio exactly 1.000 — still fits b = 1.040; on brvehins1_pp split 0, sd ratio 0.999
        # and b = 1.093, and applying it cost -0.0007 test skill.
        #
        # The full bank's own out-of-bag slope is that bias, measured on THE SAME ROWS with THE
        # SAME jury and the same y. Dividing it out leaves the part attributable to the prune:
        #
        #     b_prune = b_keep / b_full
        #
        # On allstate_sev, where the correction was discovered, b_full is 0.9964-1.0039 and the
        # ratio barely moves the number — the bias is negligible with 8 bags over 104k rows,
        # which is exactly why fitting `b_keep` alone looked sound there. It is not sound in
        # general, and the ratio costs one extra 2-parameter solve on an array already in hand.
        b_keep, se_keep = _fit_link_slope(obj, raw, yv, wv, self.tweedie_rho)[1:]
        b_full, se_full = _fit_link_slope(obj, rawf, yv, wv, self.tweedie_rho)[1:]
        b = b_keep / b_full if np.isfinite(b_full) and b_full > 0.0 else 1.0
        # Conservative standard error for the ratio: the two slopes are strongly POSITIVELY
        # correlated (same rows, same y, nested banks), so their difference has a much smaller
        # variance than this sum — erring toward silence is the right direction for a
        # correction that can only be judged after the fact.
        se_b = float(math.sqrt(se_keep * se_keep + se_full * se_full))
        # How far from 1 the ratio is IN ITS OWN STANDARD ERRORS. This is the statistic the
        # apply-gate is built on; `b` alone cannot tell a real 6% compression on 104k out-of-bag
        # rows from a 5% wobble on 2k.
        z_b = abs(b - 1.0) / se_b if np.isfinite(se_b) and se_b > 0.0 else 0.0
        # The intercept is not fitted jointly with the ratio: once the scale is set, the level is
        # whatever restores the aggregate balance the deploy re-anchor promises. `_guard_reanchor`
        # IS that solve (and is a no-op when the estimator's `reanchor` is off, so a deploy that
        # does not re-anchor does not acquire a level shift here either).
        scaled_raw = _guard_reanchor(obj, b * raw, yv, wv, reanchor)
        a = float(np.average(scaled_raw - b * raw, weights=wv)) if raw.size else 0.0
        dev_before = _guard_mean_deviance(obj, yv, raw, wv, self.tweedie_rho)
        dev_after = _guard_mean_deviance(obj, yv, a + b * raw, wv, self.tweedie_rho)
        # The relative deviance the correction buys on its own evidence.
        rel_gain = (
            float((dev_before - dev_after) / dev_before)
            if np.isfinite(dev_before) and dev_before > 0.0 and np.isfinite(dev_after)
            else float("nan")
        )
        # The mechanism variable itself: how much of the full bank's link-scale spread the
        # deployed keep-set still carries. Pruning compresses it (allstate_sev measured 0.90-0.91
        # on the splits where the correction pays); a ratio at 1.0 means nothing was compressed
        # and there is nothing for a slope to undo.
        sd_keep = float(np.sqrt(np.cov(raw, aweights=wv, ddof=0))) if raw.size > 1 else 0.0
        sd_full = float(np.sqrt(np.cov(rawf, aweights=wv, ddof=0))) if rawf.size > 1 else 0.0
        info.update(
            a=float(a), b=float(b), se_b=float(se_b), z_b=float(z_b),
            b_keep=float(b_keep), b_full=float(b_full),
            se_keep=float(se_keep), se_full=float(se_full),
            rows=int(np.asarray(yv).size), sd_keep=sd_keep, sd_full=sd_full,
            sd_ratio=(sd_keep / sd_full if sd_full > 0.0 else float("nan")),
            dev_before=float(dev_before), dev_after=float(dev_after), rel_gain=rel_gain,
        )
        info["min_z"] = slope_min_z
        if abs(b - 1.0) <= slope_eps:
            # The overwhelmingly common case on a healthy or guard-relaxed fit. Bail BEFORE
            # touching the artifact so the model is byte-identical to the pre-slope build.
            info.update(applied=False, skipped=f"|b-1| {abs(b - 1.0):.4g} <= {slope_eps}")
            self.pruning_report_["slope"] = info
            return table_model
        if not (np.isfinite(z_b) and z_b >= slope_min_z):
            # The gate that does the real work. A slope 5% from 1 on a 2k-row portfolio with a
            # 1.9-bag jury is noise, and folding it in costs real skill (freclaimdam, beMTPL16,
            # brvehins1_pp all measured harm from exactly this). Silence is the safe default:
            # the un-corrected bank is the shipped av35 artifact.
            info.update(applied=False, skipped=f"z {z_b:.3g} < {slope_min_z}")
            self.pruning_report_["slope"] = info
            return table_model
        doc = json.loads(table_model.to_json())
        _fold_slope_into_bank_json(doc, a, b)
        scaled = _TableModel.from_json(json.dumps(doc))
        # VERIFY, then adopt. The fold is an exact identity on paper, but it is written against a
        # serialization schema, and a schema that grew a new value-carrying field would corrupt
        # the artifact silently. Scoring both banks on real rows is the check that cannot be
        # fooled by a field this code does not know about: if anything failed to scale, the
        # affine relation breaks and we ship the untouched model.
        rows = np.asarray(x32)
        stride = max(1, rows.shape[0] // _SLOPE_VERIFY_ROWS)
        idx = np.arange(0, rows.shape[0], stride)[:_SLOPE_VERIFY_ROWS]
        xs = np.ascontiguousarray(rows[idx])
        cat_xs = (
            [list(np.asarray(col, dtype=object)[idx]) for col in cat_x] if cat_x else None
        )
        old = np.asarray(
            table_model.predict_raw(xs, cat_x=cat_xs, n_jobs=n_jobs), dtype=np.float64
        )
        new = np.asarray(
            scaled.predict_raw(xs, cat_x=cat_xs, n_jobs=n_jobs), dtype=np.float64
        )
        want = a + b * old
        # f32 scoring accumulation: compare on the scale of the score itself.
        scale = float(np.max(np.abs(want))) if want.size else 0.0
        err = float(np.max(np.abs(new - want))) if want.size else 0.0
        info["verify_max_abs_err"] = err
        if not (np.isfinite(new).all() and err <= 1e-4 * max(scale, 1.0)):
            info.update(applied=False, skipped=f"affine verification failed (max err {err:.3g})")
            self.pruning_report_["slope"] = info
            return table_model
        info["applied"] = True
        self.pruning_report_["slope"] = info
        lap(
            f"(7) post-prune slope re-anchor APPLIED (rung {rung}, a={a:+.4f} b={b:.4f}, "
            f"OOB dev {dev_before:.5f}->{dev_after:.5f})"
        )
        return scaled

    def _graduate_table_model(self, table_model: "_TableModel", x32: "np.ndarray",
                              y32: "np.ndarray", w_full: "np.ndarray",
                              exposure: "np.ndarray | None",
                              cat_x: "list[list[str]] | None",
                              eval_rows: "np.ndarray | None" = None) -> "_TableModel":
        if eval_rows is not None:
            raise ValueError("eval_rows from the fit design are not an independent holdout")
        report: list[dict[str, Any]] = []
        updates: list[tuple[int, list[float]]] = []
        fixed = self.graduation_alpha
        # Fidelity weights: the bank's `support` is BINNED ROW COUNTS, but the GCV noise model
        # needs the cell's information mass. On aggregated portfolios one row can carry
        # thousands of policy-years (swautoins: mean weight ~1200), so row-count GCV
        # over-smooths by that factor. First-order fix: scale support by the mean training
        # weight — exact for ~uniform weights, and a no-op (scale ~ 1) on individual-policy
        # data. (Per-cell weight mass is not stored in the wire format; revisit if it ever is.)
        w_scale = float(np.sum(w_full)) / float(max(len(w_full), 1))
        if self._measure_kwargs()["ref_measure"] == "exposure":
            # Under the exposure measure `support` already IS the per-cell effective mass
            # (spec §08.7), so the row-count rescale above would double count it.
            w_scale = 1.0
        # Rust exposes only eligible dense table payloads and resolves categorical provenance,
        # avoiding a complete model JSON round-trip merely to inspect these tensors. The raw
        # ceiling passed here (`_GRADUATION_MAX_CELLS_RAW`) is deliberately generous: Rust runs
        # before axis kinds are resolved, so it cannot yet tell a cheap separable table from an
        # expensive joint one — the real per-table budget is enforced inside `_graduate_one`
        # (see the two-tier cap comment on `_GRADUATION_MAX_CELLS`).
        items = list(table_model.graduation_tables(self._GRADUATION_MAX_CELLS_RAW))
        jobs = self._resolve_n_jobs()
        jobs = max(1, min(int(jobs), len(items))) if (jobs and items) else 1
        # Tables are independent and the loop can fan out on a thread pool sized to n_jobs —
        # LAPACK releases the GIL. Each table's computation (and its BLAS threading) is exactly
        # the serial call, and results are reassembled in iteration order, so `updates`/`report`
        # are byte-identical to the serial path.
        #
        # The solver loop below runs under a single-threaded BLAS cap
        # (`threadpoolctl.threadpool_limits(limits=1, user_api="blas")`). Measured 2026-07-16: a
        # small `eigh` (k=80, a typical table chain length) costs 158ms under OpenBLAS's default
        # 22-thread pool vs 0.4ms capped to 1 thread — wake/barrier overhead per BLAS-2 call that
        # swamps the actual FLOPs on matrices this small. Over a whole table-loop that is 20.8s
        # multithreaded vs 0.32s capped (20-88s vs 0.3-0.7s, load-dependent, on the wider battery).
        # This REPLACES an earlier, now-refuted claim that "numpy's OpenBLAS already runs each
        # factorization at all cores, so a table-level pool on top only adds contention" — that
        # comparison pitted two OVERSUBSCRIBED configurations against each other (plain vs
        # thread-pool-fanned-out), never against a capped baseline, so it never saw the real cost.
        # The cap also makes the Cholesky/pinv branch choice and `eigh` results
        # thread-count-deterministic (see `_dr_family`'s always-pinv fix, which independently
        # closes the reproducibility gap that branch left even before this cap). A table-level
        # thread pool (the `n_jobs` fan-out below) is STILL pointless — now because the capped
        # loop is sub-second, not because BLAS was already saturating the cores.
        try:
            import threadpoolctl

            _blas_cap = threadpoolctl.threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            import contextlib

            _blas_cap = contextlib.nullcontext()
        with _blas_cap:
            if jobs > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    results = list(pool.map(
                        lambda it: self._graduate_one(it, fixed, w_scale), items
                    ))
            else:
                results = [self._graduate_one(it, fixed, w_scale) for it in items]
        for res in results:
            if res is None:
                continue
            table_idx, features, out, alpha = res
            if alpha > 0.0:
                updates.append((int(table_idx), out.tolist()))
            report.append({"features": features, "alpha": alpha, "table_idx": int(table_idx)})
        if not updates and self.graduation_high_order_alpha == 0:
            self.graduation_report_ = report
            return table_model

        # Candidate construction and portfolio re-anchoring use the fit rows -- which are now
        # ALL the rows (see `_fit_and_prune`).
        args = (x32, np.ascontiguousarray(y32), np.ascontiguousarray(w_full), updates)
        kwargs: dict[str, Any] = dict(
            exposure=(np.ascontiguousarray(exposure) if exposure is not None else None),
            cat_x=cat_x, n_jobs=self._resolve_n_jobs(),
        )
        if self.graduation_high_order_alpha > 0:
            candidate, train_adopted, high_report = table_model.apply_high_order_graduation(
                *args, alpha=float(self.graduation_high_order_alpha),
                box_budget=max(0, int(self.prune_box_budget or 0)), **kwargs,
            )
            for entry in json.loads(high_report):
                report.append({**entry, "method": "reference_diffusion",
                               "requested_alpha": entry["alpha"],
                               "alpha": entry["alpha"] if entry["applied"] else 0.0})
        else:
            candidate, train_adopted = table_model.apply_graduation(*args, **kwargs)
        if not updates and not any(r.get("applied", False) for r in report):
            self.graduation_report_ = report
            self.graduation_validation_.update(
                adopted=False, candidate_count=0, skipped="no eligible smoothing updates",
            )
            return table_model
        # NO ACCEPT/REJECT GATE (2026-09-20): see `_fit_and_prune` for the measurement that
        # retired it. `train_adopted` is the native layer's own consistency check on the
        # re-anchored candidate -- it declines when the smoothed bank cannot be balanced back
        # onto the fit rows -- NOT a generalization test. Nothing here scores the candidate
        # against held-out outcomes, because nothing is held out any more.
        adopted = bool(train_adopted)
        self.graduation_validation_.update(
            adopted=adopted, candidate_count=1, n_smoothed=len(updates),
        )
        self.graduation_report_ = [
            {**r, "rejected": not adopted, "alpha": r["alpha"] if adopted else 0.0}
            for r in report
        ]
        return candidate if adopted else table_model

    def _as_float32_2d_once(self, x: Any) -> np.ndarray:
        arr0 = np.asarray(x)
        warn = arr0.dtype != np.float32 and not getattr(
            self, "_precision_warning_emitted_", False
        )
        out = _as_float32_2d(x, warn=warn)
        if warn:
            self._precision_warning_emitted_ = True
        return out

    def _attach_model(self, model: _Model | _TableModel) -> None:
        # Native legacy blobs already carry objective metadata in their schema.
        # Restore it even when there is no Python estimator envelope.
        native = json.loads(model.to_json())["model"]
        objective = native["schema"]["objective"]
        self.objective = {
            "SquaredError": "squared_error", "Poisson": "poisson", "Gamma": "gamma",
            "Tweedie": "tweedie", "Logistic": "logistic",
        }[objective["loss"]]
        if objective.get("tweedie_rho") is not None:
            self.tweedie_rho = float(objective["tweedie_rho"])
        self._model = model
        self.n_features_in_ = model.n_features
        names = model.feature_names
        if names:
            self.feature_names_in_ = np.asarray(names, dtype=object)

    def __sklearn_is_fitted__(self) -> bool:
        # Explicit fitted check: a single `_model` OR a multiclass `_multi_model`. Also avoids the
        # false-positive where `lambda_`'s trailing underscore makes sklearn's default
        # `check_is_fitted` treat an unfitted estimator as fitted.
        return (
            getattr(self, "_model", None) is not None
            or getattr(self, "_multi_model", None) is not None
        )

    def __sklearn_tags__(self) -> Any:
        # sklearn >= 1.6 tags. t-boost handles NaN in numeric features natively (routed to the
        # reserved missing bin, like LightGBM), so declare it rather than have checks expect a raise.
        tags = super().__sklearn_tags__()
        try:
            tags.input_tags.allow_nan = True
        except AttributeError:
            pass
        return tags

    def __getstate__(self) -> dict[str, Any]:
        # The Rust model handles (`_Model` / `_MultiClassModel`) are `@final` pyo3 classes that
        # cannot be pickled directly, so serialize them to bytes here (round-trips via the versioned
        # wire format). Everything else on __dict__ (params, classes_, n_features_in_, …) pickles
        # normally. This is what lets joblib.Memory / pickle cache a fitted estimator.
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        multi = state.pop("_multi_model", None)
        if multi is not None:
            state["__tri_multi_bytes__"] = multi.to_bytes()
        elif model is not None:
            state["__tri_model_bytes__"] = model.to_bytes()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        multi_bytes = state.pop("__tri_multi_bytes__", None)
        model_bytes = state.pop("__tri_model_bytes__", None)
        self.__dict__.update(state)
        if multi_bytes is not None:
            if multi_bytes[:4] == _MULTICLASS_TABLES_MAGIC:
                self._multi_model = _MultiClassTableModel.from_bytes(multi_bytes)
            else:
                self._multi_model = _MultiClassModel.from_bytes(multi_bytes)
        elif model_bytes is not None:
            if model_bytes[:4] == _TABLES_MAGIC:
                self._model = _TableModel.from_bytes(model_bytes)
            else:
                self._model = _Model.from_bytes(model_bytes)
            self._multi_model = None


class TBoostRegressor(RegressorMixin, _BaseTBoost):  # type: ignore[misc]
    """Exact, depth-3 oblivious-tree gradient boosting regressor.

    Every tree is a symmetric (oblivious) depth-3 tree that touches at most three distinct
    raw features, so the fitted ensemble decomposes exactly into a sum of at-most-3-way
    fANOVA effect tables (see :meth:`tables`) with no approximation: the score any row
    receives is exactly the sum of the table lookups that explain it. Supports squared-error
    regression and the Poisson/Gamma/Tweedie GLM families via ``objective``.

    The defaults include a complete fitting pipeline. The September 2026 benchmark
    describes an earlier pipeline; current graduation reserves a separate holdout. early stopping is on
    by default (``validation_fraction=0.1``, ``early_stopping_rounds=500`` with adaptive
    patience ``early_stopping_adaptive=1.5``, against a large ``n_trees=4000`` cap), outer
    bagging is on by default (``n_bags=8``), per-tree column sampling is on by default
    (``colsample_bytree=0.8``), and post-fit table pruning is on by default (``prune=True`` —
    the fitted artifact is the CV-pruned tables-only model). Pass
    ``n_bags=1, validation_fraction=None, prune=False`` for the cheapest single-fit baseline.

    polars DataFrames/LazyFrames are first-class ``X`` input (their extracted numeric block
    is already F-contiguous float32, the cheapest ingest layout; String/Categorical/Enum
    columns are categorical automatically, and ``y``/``sample_weight``/``exposure`` may name
    frame columns). For plain arrays, pass
    F-contiguous ``float32`` arrays (``numpy.asfortranarray``) for ``X`` for the cheapest ingest
    — its columns copy in one pass each with no transpose, unlike the C-contiguous default.

    Parameters
    ----------
    n_trees : int, default=4000
        Maximum number of boosting rounds (an upper bound; a round also stops early if no
        candidate split clears ``min_split_gain``). The large default cap is meant to be
        governed by early stopping (see ``validation_fraction``), not hit directly.
    learning_rate : float, default=0.05
        Shrinkage factor applied to every tree's leaf values before they are added to the
        running score.
    lambda_ : float, default=1.0
        L2 regularization on leaf values: the ridge penalty ``lambda`` in the Newton
        leaf-value solve ``w* = -G / (H + lambda)`` and in the split gain.
    lambda_scale_invariant : bool, default=False
        Rescale ``lambda_`` by the fit's mean per-row hessian instead of using it literally in
        ``H + lambda``. On exposure-weighted objectives (Poisson with large ``sample_weight``,
        where ``H = weight * mean``) ``H`` can dwarf any ``lambda_`` in the usual ``{0.1, 1,
        10}`` range, making the raw denominator a dead no-op exactly where regularization
        matters most; rescaling turns ``lambda_`` into a scale-free "pseudo-count of prior
        rows" that stays a live knob regardless of the weight/exposure magnitude. Resolved once
        per boosting round and shared by every split-gain and leaf-value evaluation that round.
        ``False`` (the default) is bit-identical to the plain ``H + lambda`` behavior.
    l1_leaf : float, default=0.0
        L1 regularization on leaf values, applied by soft-thresholding the aggregated
        gradient before the Newton solve.
    min_split_gain : float, default=0.0
        Minimum gain a candidate split must clear to be taken. A level whose best gain does
        not exceed this floor stops growing and its node stays a leaf.
    max_delta_step : float or None, default=None
        Caps the magnitude of a leaf's Newton update, applied to the full-precision
        aggregated step before ``learning_rate``. ``None`` defers to the objective's own
        default: the log-link objectives (``poisson``, ``gamma``, ``tweedie``) fall back to
        ``0.7`` to keep leaf updates finite on sparse/zero-heavy targets, while
        ``squared_error`` falls back to uncapped. An explicit value always overrides the
        objective default.
    max_delta_step_gated : None, bool, str, tuple or dict, default=None
        In-fit rate-collapse gate on ``max_delta_step`` (spec §05.6 addendum). On log-link
        Tweedie fits over small aggregated-exposure portfolios, boosting can drive a small
        subgroup's predicted rate ``mu`` toward zero; the ``rho ~ 1.7`` deviance's
        ``-y * mu**(1-rho)`` term then diverges like ``mu**-0.7``, letting a single high-``y``
        test row carry a fifth of the total deviance. The gate watches, at the top of every
        boosting round, the smallest ``ln(mu_i / weighted-mean-rate)`` over the TRAINING rows
        (exactly ``raw_i - offset_i - f0``, so it costs two subtractions per row and no
        ``exp``). The first round that minimum falls below ``ln(collapse_threshold)``, the
        leaf-step clamp tightens to ``capped_step`` FOR THAT ROUND AND EVERY LATER ROUND of
        that fit; already-grown trees are untouched, and the gate never disengages. Under
        bagging each bag gates independently.

        ``None`` (the default) defers to the objective: ``tweedie`` ships
        ``collapse_threshold=1e-3, capped_step=0.3`` — measured to recover +0.02 to +0.03 skill on the
        splits that actually collapse — and every other objective ships no gate at all
        (``gamma`` was measured HARMFUL under a blanket 0.3, ``poisson`` is untested).
        ``False`` disables the gate; a ``(collapse_threshold, capped_step)`` pair or a dict
        with those keys sets it explicitly (and arms it on objectives that ship none).
        An explicit ``max_delta_step`` outranks all of this — a named cap is never gated.

        A fit whose detector never trips is BIT-IDENTICAL to one built with the gate off, so
        this parameter is inert everywhere the collapse does not happen. After ``fit``, the
        attribute ``delta_step_gate_`` reports what the detector saw.
    max_bin : int, default=254
        Maximum number of histogram bins per numeric feature.
    objective : {"squared_error", "poisson", "gamma", "tweedie"}, default="squared_error"
        Loss function to fit: ``"squared_error"`` for ordinary regression, or
        ``"poisson"``/``"gamma"``/``"tweedie"`` for the corresponding GLM-family deviance on
        a log link (use :class:`TBoostClassifier` for binary classification).
    tweedie_rho : float, default=1.5
        Tweedie power parameter (variance proportional to ``mean ** rho``), used only when
        ``objective="tweedie"``. The compound Poisson-Gamma regime typical of insurance
        pure-premium targets needs a value in ``(1, 2)``.
    min_data_in_leaf : int or None, default=None
        Minimum number of training rows a leaf must retain; a split that would leave either
        child below this count is rejected. ``None`` resolves **depth-aware**: ``0`` (inert,
        the historical default) at every depth: P-D2 measured a depth-coupled floor across
        the regression-stress battery and it was never better and sometimes much worse (see
        ``_MIN_DATA_PER_LEAF_AT_DEPTH``). **Set this explicitly for a model destined for
        filing**: it is the only guard that stops a displayed relativity from standing on a
        handful of policies, and a lifted (``max_depth > 3``) fit puts ~3.4x fewer rows in
        each cell.
    min_sum_hessian_in_leaf : float, default=0.0
        Minimum total Hessian a leaf must retain; a split that would leave either child
        below this is rejected. Complements ``min_data_in_leaf`` for weighted or
        non-constant-Hessian objectives, where row count alone under- or over-states a
        leaf's information mass.
    min_weight_sum_in_leaf : float, default=0.0
        Minimum total ``sample_weight`` a leaf must retain; a split that would leave either
        child below this is rejected.
    path_smooth : float or None, default=None
        LightGBM-style path smoothing: blends a leaf's own Newton estimate with its
        parent's with credibility weight ``Z = n/(n + path_smooth)``, shrinking
        thin-evidence leaves toward the ancestor path. ``None`` (the default) resolves
        objective-aware: ``10.0`` for ``gamma``/``tweedie`` — heavy-tail deviance is
        dominated by sparse-evidence cell variance, and this is its zero-cost smoother
        (2026-07-23 six-set screen: freclaimdam −1.2%, ohlsson_pp −0.56%, worst +0.05%) —
        and ``0.0`` (off) for every other objective. An explicit float always wins;
        ``0.0`` disables it.
    subsample : float or None, default=None
        Row-sampling rate for split search. ``None`` scans every row. A value in ``(0, 1)``
        switches to minimal-variance-style (MVS) sampling: a probability-proportional-to-
        gradient row subset chooses the split structure, with ``mvs_min_rows`` as a floor on
        the sampled count; leaf values are still refit from all rows regardless.
    colsample_bytree : float, default=0.8
        Fraction of features randomly sampled per tree.
    learning_rate_decay : float, default=0.0
        Per-round multiplicative decay applied to ``learning_rate``. ``0.0`` is a constant
        rate.
    validation_fraction : float or None, default=0.1
        Fraction of training rows carved out as an internal early-stopping holdout. ``None``
        disables internal early stopping entirely, so every fit runs the full ``n_trees``.
        Paired with ``early_stopping_rounds=500`` against the ``n_trees=4000`` cap, this is
        the tuned default: early stopping decides when a fit actually stops.
    early_stopping_rounds : int, default=500
        Patience: the fit stops once this many rounds pass without a validation-deviance
        improvement (see ``early_stopping_min_delta`` for what counts as an improvement).
        Ignored when ``validation_fraction`` is ``None``.
    early_stopping_adaptive : float or None, default=1.5
        Adaptive patience ratio. A ratio ``r > 0`` grows the effective patience with
        progress: ``clamp(ceil(r * best_round), 50, early_stopping_rounds)`` — a fit that
        peaks early stops sooner. The ``1.5`` default matches every insur-arena benchmark
        run (previously the recipe factory set it while the bare constructor kept fixed
        patience). ``None`` restores the fixed ``early_stopping_rounds`` patience.
    early_stopping_min_delta : float, default=1e-4
        Relative improvement tolerance: a round only becomes the new best (resetting the
        patience window and moving the deployed truncation point) once validation deviance
        clears ``best * (1 - early_stopping_min_delta)``. ``0.0`` is the legacy
        any-improvement rule, under which epsilon-level validation noise can reset patience
        indefinitely; the shipped default ignores that noise so early stopping actually
        terminates on large, low-signal data. Must be finite and in ``[0.0, 1.0)``.
    interaction_gain_hurdle : float, default=2.0
        Interaction-admission hurdle (soft heredity). A split that would introduce a new raw
        feature — raising the tree's fANOVA interaction order — must clear this scale-free
        hurdle against the tree's level-1 (main-effect) gain; the new feature also competes
        against the best already-used raw feature under the split-ranking score, with ties
        won by the lower-order (reused-feature) split. ``0.0`` restores pure greedy
        admission, where a fresh feature is used only when no reused feature is available.
        Must be finite and ``>= 0``.
    interaction_gain_hurdle_mode : {"adaptive", "fixed"}, default="adaptive"
        How ``interaction_gain_hurdle`` is applied. ``"fixed"`` uses the scalar hurdle
        exactly as supplied. ``"adaptive"`` (the default) decays the effective hurdle from
        its full early-tree value toward a lenient late-tree value as main-effect gains
        fade, and additionally requires 3-way admissions to clear a stricter multiplier than
        2-way admissions.
    graduate : bool or None, default=None
        Whittaker-Henderson smoothing of the deployed fANOVA tables after pruning.
        ``None`` (the default) and ``True`` both enable it; ``False`` disables it and
        ``graduation_report_`` is then never set. NO ROWS ARE WITHHELD: the fit, the
        prune and the smoothing all see every row, and the smoothed bank is adopted
        whenever the native re-anchor accepts it (``graduation_validation_["adopted"]``).
        There is deliberately NO accuracy gate -- between 2026-09-09 and 2026-09-20 one
        existed, paid for by a ``prune_validation_fraction`` holdout carved out of every
        fit, and it cost +1.29% mean test deviance on the arena catalog to decide a bit
        worth -0.015%. Whether a smoothed tariff is what you want is your call, not a
        deviance comparison's: set ``graduate=False`` for the raw pruned bank.
        Monotone fits retain the full bank without smoothing. Explicit ``True`` requires
        ``prune=True``. Multiclass (K>=3) graduation is unsupported.
    graduation_alpha : float or None, default=None
        Fixed Whittaker-Henderson smoothing strength, overriding the per-table generalized
        cross-validation (GCV) alpha selection that ``graduate`` normally uses. ``None`` (the
        default) lets each table pick its own GCV-optimal alpha independently, including
        ``alpha=0`` (left untouched) for a table whose apparent roughness is real,
        well-supported shape rather than noise. Only relevant when ``graduate`` resolves to
        ``True``.
    graduation_high_order_alpha : float, default=0.0
        Opt-in strength in ``[0, 1]`` for one reference-weighted neighbour diffusion
        step on factored interactions of orders 3 through 8. Requires ``prune=True``
        and graduation enabled; unsupported for monotone or multiclass fits. Numeric
        axes use finite cell order; categorical axes and missing-to-finite edges are
        excluded. A missing slice may still change along other numeric axes.
        Preserves product-reference zero marginal means and recomputes variances.
        This is a first-difference smoother, separate from dense WH/GCV graduation;
        its strength is fixed, not selected by GCV. Both steps form one candidate,
        adopted whenever the native re-anchor accepts it. Effects exceeding
        1,024 stored boxes per effect, 4,096 added boxes in total, or 2,000,000 added
        mask cells are skipped, with details in ``graduation_report_``. A configured
        ``prune_box_budget`` further limits total stored boxes after smoothing.
        Set ``graduation_alpha=0`` to apply only higher-order smoothing.
    leaf_refine_steps : int, default=4
        Extra per-tree leaf Newton refinement steps applied after a tree's structure is
        fixed.
    leaf_refine_backtracks : int, default=4
        Maximum backtracking attempts per leaf-refinement step (halving the step on a
        non-improving move).
    refine_closed_form_tier2 : bool, default=True
        Tier-2 log-link closed-form leaf refinement (currently applies to Poisson). ``True``
        (the default) derives each per-step leaf Newton delta directly from the per-leaf
        closed form, dropping the remaining per-row passes an exact line search would
        otherwise take — a speed win with roughly 1e-7 leaf-value drift versus the exact path
        (not bit-identical). Set ``False`` for the byte-identical Tier-1 path.
    incremental_mu : bool, default=False
        Incremental ``mu = exp(F)`` cache for log-link (Poisson) fits: maintains ``mu``
        multiplicatively across rounds instead of recomputing an O(N) ``exp`` every round —
        a speed knob with roughly 1e-9 to 1e-6 drift from the exact path (not bit-identical).
    mvs_min_rows : int, default=1
        Floor on the sampled row count when ``subsample`` enables minimal-variance-style
        (MVS) row sampling. Has no effect when ``subsample`` is ``None``.
    hist_precision : {"full", "quantized"} or None, default=None
        Histogram accumulator precision. ``None``/``"full"`` uses the exact full-precision
        accumulator; ``"quantized"`` uses a faster quantized-integer accumulator.
    n_bags : int, default=8
        Number of independent bags trained and averaged (exact ensemble averaging over
        fANOVA banks). Replicated benchmarking found row-bag plus column diversity to be the
        two levers that most move stacking accuracy, at a cost of roughly ``n_bags`` times
        the train/predict time of a single fit — accuracy is the priority of the shipped
        default. ``n_bags=1`` disables bagging.
    bag_subsample : float, default=0.8
        Per-bag row-sampling fraction for outer bagging. ``0.8`` (the default) is subagging
        — sampling without replacement — rather than a ``1.0`` bootstrap: a bootstrap bag
        duplicates rows, which makes the per-bag early-stopping validation carve overlap the
        bag's own training multiset, a train/validation leak that lets validation deviance
        improve indefinitely and defeats early stopping. Set to ``1.0`` to restore classic
        bootstrap bagging. Only relevant when ``n_bags > 1``.
    cell_refit_base : float or None, default=None
        Base ridge penalty for the out-of-bag (OOB) fANOVA cell refit: after bagging, each
        cell coefficient in the averaged bank is re-fit toward its bags' OOB residual under a
        ridge penalty shaped by ``cell_refit_gamma``, then re-purified. ``None`` (the
        default) disables it. Requires real bagging (an OOB partition, i.e. ``n_bags >= 1``)
        — setting it without bagging raises ``ValueError`` rather than silently no-opping.
        Held-out no-harm guarded: adopted only if it improves held-out deviance.
    cell_refit_gamma : float, default=2.0
        Adaptive exponent for the OOB cell refit's penalty (``0`` = flat ridge across all
        cells; larger values penalize high-signal cells less). Only relevant when
        ``cell_refit_base`` is set.
    ridge_refit_l2 : float or None, default=None
        L2 penalty for a fully-corrective ridge leaf refit: after stagewise tree growth, all
        leaf values are jointly re-solved by regularized IRLS over the frozen tree structure.
        ``None`` (the default) disables it; a float enables it at that penalty strength.
    ridge_refit_max_iter : int, default=5
        Maximum IRLS passes for the fully-corrective ridge leaf refit. Only relevant when
        ``ridge_refit_l2`` is set.
    nesterov : bool, default=False
        Reserved for AGBM-style Nesterov look-ahead acceleration. **Currently unsupported**:
        setting this ``True`` always raises ``ValueError`` at fit time (the momentum-
        correction step needed for stable convergence is not yet implemented) rather than
        silently producing a diverging model. Leave at the default.
    dart_drop_rate : float or None, default=None
        Enables DART (Dropout Additive Regression Trees) when set: on each round, previously
        grown trees are dropped with this probability before the new tree is fit, and DART's
        standard alpha normalization is folded into the tree weights. ``None`` (the default)
        disables DART. Must be in ``[0, 1)``. Mutually exclusive with ``nesterov``.
    random_strength : float, default=0.0
        Deterministic split-score noise strength, seeded by ``seed`` so it stays
        reproducible; perturbs split-gain ranking to diversify tree structure. ``0.0`` (the
        default) is inert.
    reanchor : bool or None, default=None
        Whether to re-solve the model's global intercept exactly against the training data
        after fitting, removing post-shrinkage aggregate bias. ``None`` uses a link-aware
        default: on for the log-link objectives ``gamma``/``tweedie`` where shrinkage bias
        is largest (a no-op for ``poisson``), off otherwise. An explicit ``True``/``False``
        always overrides the default.
    reanchor_slope : bool or None, default=None
        Whether to fit an honest affine link-scale recalibration (``F' = b + a * F``) on the
        fit's own internal validation holdout after early stopping, correcting the uniform
        score-scale compression that shrinkage and bulk-dominated early stopping leave
        behind, without touching ranking or the fANOVA decomposition. Evidence-gated: the
        fit is ridge-shrunk toward the identity and only applied when its holdout deviance
        gain clears the prior penalty, so thin or uninformative holdouts stay at
        (near-)identity — this can only help or be neutral, never hurt. Inert without a
        validation holdout (i.e. when ``validation_fraction`` is ``None``). ``None`` follows
        the same link-aware default as ``reanchor``; an explicit ``True``/``False`` always
        wins.
    max_interaction_order : int, default=3
        Whole-tree cap on the number of distinct raw features a tree may use (1 to 4) —
        caps interaction ORDER, which is the number of axes an exported table has. It does
        not cap tree depth; see ``max_depth`` for that. Must be ``<= max_depth`` (a tree
        needs one level per distinct feature), and values above 3 are opt-in.

        Order 4 is not an exactness relaxation. The fANOVA purification cascade is
        n-dimensional, so a 4-way effect is centred, shed and reconstructed by exactly the
        same algebra a 3-way is, and all five decomposability checks pass unchanged. It is a
        READABILITY relaxation, priced in three places: the interaction-gain hurdle doubles
        again at the 3->4 transition, ``table_budget_order_shrink`` halves the table-size
        prior's allowance, and the evidence-gated prune drops any 4-way the held-out folds
        cannot pay for. Measured: on a target whose highest true interaction is a pair,
        raising the cap from 3 to 4 leaves the deployed table set and the fit IDENTICAL.

        **Prefer ``max_depth == max_interaction_order`` at order 4.** A 4-way effect is
        exported as a set of rank-1 region boxes (never a dense cube — see
        ``table_budget_cells``), and at ``max_depth == 4`` each tree contributes exactly one,
        so the effect stays a short readable stack of case expressions (~110 boxes measured
        on a real portfolio). At ``max_depth == 6`` the same tree contributes up to 36
        region boxes and they do not merge across trees, which measured ~5,600 boxes for a
        single effect — still exact, no longer a filing artifact.
    table_budget_cells : int or None, default=None
        Soft cell budget for the split finder's table-size prior: a candidate support's
        ranking score is multiplied by ``(budget / max(budget, projected_cells)) ** 0.5``,
        where ``projected_cells`` is the product of the realized extents of its distinct
        raw features. A ranking steer, never a hard reject, and never a cap on an exported
        table — the hard firewall is separate. ``None`` resolves ``2_000_000`` (a memory
        budget that effectively never binds) at ``max_depth <= 3``, and ``4096`` above it:
        a lifted order-3 effect is exported as a dense cube, and a cube is cubic in the
        per-axis extent, so without the tighter budget the bank overflows its firewall.
        4096 lands at roughly 16 cells per axis for a 3-way table and 64x64 for a pair.
    max_depth : int, default=3
        Whole-tree cap on the number of split LEVELS (3 to 6). Orthogonal to
        ``max_interaction_order``: depth caps *resolution*, order caps *interaction*.

        Once a tree holds ``max_interaction_order`` distinct raw features, deeper levels
        can only REUSE a feature already on the tree with a refined threshold, which is a
        valid lower-order refinement — so a depth-6 tree on 3 features is still exactly a
        3rd-order fANOVA function. Tables get finer grids, never more axes, and the
        exported grid is hard-capped at ``max_bin`` cells per axis at any depth.

        What depth buys is estimation efficiency, not expressiveness: a depth-3 *ensemble*
        already spans the whole ≤3rd-order space. A deeper tree fits its finer partition
        jointly, in one Newton step under one shrinkage and one ``lambda_``, where the
        depth-3 ensemble reaches the same function only through a sequence of
        independently-shrunk 8-cell fits. Per-tree resolution is 8 / 12 / 18 / 27 realized
        cells at depth 3 / 4 / 5 / 6 (a 3.4x lift at depth 6, not 8x — most of the
        ``2**depth`` leaf array is unreachable). Depth 6 at order 1 is a genuinely new
        capability: a 7-step main effect fit in a single boosting round.

        **The default is 3 and should stay 3 unless a held-out probe says otherwise.**
        Deeper trees put ~3.4x fewer rows in each cell, and raising depth automatically
        raises the ``min_data_in_leaf`` floor to compensate (see that parameter). Not
        supported together with ``ridge_refit_l2``, which errors.
    seed : int, default=0
        Seed for every deterministic source of randomness in the fit (row/column sampling,
        bagging, DART dropout, ``random_strength`` noise). The same seed and inputs
        reproduce a bit-identical model.
    n_jobs : int or None, default=None
        Number of threads used for fitting and prediction, following the joblib/sklearn
        convention: ``None`` uses the native default (all cores), ``-1`` means all cores,
        and a negative ``n`` means ``cpu_count + 1 + n`` (so ``-2`` is all-but-one core).
    monotone_constraints : sequence, dict, or None, default=None
        Per-feature monotonicity constraints on the fitted function. ``None`` (the default)
        applies no constraints. Accepts either a length-``n_features`` positional sequence of
        ``{-1, 0, 1}`` (``1`` = increasing, ``-1`` = decreasing, ``0`` = unconstrained), or a
        dict keyed by feature name (matched against DataFrame column names, or the canonical
        ``f{i}``) or integer index, valued the same way.
    categorical_features : sequence, str, int, or None, default=None
        Which features to treat as categorical (target-statistic encoded, see the ``cat_*``
        parameters) rather than numeric. Accepts a single name/index, a sequence of
        names/indices, or a boolean mask the length of ``n_features``. For a polars ``X``,
        String/Categorical/Enum columns are treated as categorical automatically (they have
        no numeric cast) — declare a feature here only to force a NUMERIC-dtyped column onto
        the categorical path. ``None`` (the default) means dtype-detection only: every
        feature of a plain array (and every numeric polars column) is numeric.
    cat_smooth : float or None, default=None
        Categorical target-statistic shrinkage strength. ``None`` (the default) uses an
        empirical-Bayes credibility estimate fit per feature from the data; a float fixes the
        shrinkage to that pseudo-count ``m`` in ``(n_c * mean_c + m * base) / (n_c + m)``
        instead.
    cat_target : {"mean", "log_mean"} or None, default=None
        Target transform used before categorical target-statistic shrinkage. ``None``/
        ``"mean"`` (the default) encodes the exposure-weighted mean target; ``"log_mean"``
        encodes the weighted mean of ``log(y)``, useful for positive severity targets.
    cat_leakage : {"kfold", "ordered", "loo"} or None, default=None
        Leakage-avoidance scheme for training-time categorical encodings (serve-time
        encoding always uses the frozen full-data map regardless of this setting). ``None``/
        ``"kfold"`` (the default) uses ``cat_k``-fold cross-fit target statistics;
        ``"ordered"`` uses ``cat_n_perms`` seeded permutations of ordered target statistics;
        ``"loo"`` uses leave-one-out statistics (deterministic and knob-free, but not the
        default — it reintroduces a per-row dependence on the row's own target that measures
        worse than k-fold cross-fitting).
    cat_n_perms : int, default=1
        Number of seeded permutations for ordered target-statistic encoding. Only relevant
        when ``cat_leakage="ordered"``.
    cat_k : int, default=5
        Number of folds for k-fold cross-fit target-statistic encoding. Only relevant when
        ``cat_leakage`` is ``None``/``"kfold"`` (the default scheme).
    cat_min_data_per_group : float, default=10.0
        Weighted-count floor below which a categorical level collapses into a shared rare
        bucket before encoding.
    cat_direct_max_levels : int, default=16
        Low-cardinality bypass threshold: a categorical feature with 3 to this many distinct
        (post-rare-pooling) levels skips cross-fit encoding and shrinkage entirely — each
        level keeps its raw full-fit target statistic and gets its own histogram bin ordered
        by that statistic. Binary features are always exempt from this bypass regardless of
        the threshold. ``0`` disables the bypass for every feature.
    cat_channels : list of str, or None, default=None
        Which target-statistic channels to fit per native categorical feature. ``None`` (the
        default) fits only the mean-TS channel described above — byte-identical to every
        prior release. ``["mean", "count"]`` additionally fits a SECOND, target-free channel
        per categorical: a count/rarity statistic (``log1p`` of each level's exposure share)
        that lets the trees separate rare-but-informative levels from common-but-mild ones, a
        signal the mean channel alone collapses together. This can materially improve
        ``predict``/``score`` accuracy on high-cardinality categoricals with rare, informative
        levels. Table export (``tables()``, ``explain()``, ``pruning_report_``) FULLY supports
        multi-channel categoricals: the channels collapse LOSSLESSLY to exactly one relativity
        per raw categorical (never a column per channel), preserving the exact fANOVA
        decomposition — the pruned tables-only model reproduces the tree ensemble to <1e-6, and
        all five I2 exactness gates pass on multi-channel fits. The export is keyed by actual
        category level (Piece B): every categorical axis — single- or multi-channel — lists
        each level's label alongside the numeric borders/cell it lands in, so
        ``tables()``/``explain()`` read as e.g. ``Ford Focus 1.6 → 1.12`` rather than only the
        raw encoded value; levels collapsed by rare-pooling appear as one ``"<rare>"`` entry.
        ``"class_freq"`` (multiclass K>=3 only) is not valid here and raises ``ValueError``:
        it replaces the ordinal label-mean encoding that only the softmax path produces.
    cat_count_min_levels : int, default=20
        Cardinality gate for the count channel (only relevant when ``cat_channels`` includes
        ``"count"``): a categorical feature's count axis is fit only when its post-rare-pooling
        distinct level count is at least this many — the measured accuracy win came from
        high-cardinality categoricals specifically, so a low-cardinality one below the
        threshold behaves exactly as mean-only (bit-identical to leaving ``"count"`` off, for
        that feature). ``0`` disables the gate: every categorical gets the count channel
        whenever ``cat_channels`` requests it, regardless of cardinality.
    cat_class_freq_min_levels : int, default=3
        Cardinality gate for the per-class frequency channels (only relevant when
        ``cat_channels`` includes ``"class_freq"``, i.e. multiclass K>=3). A categorical gets
        its per-class axes only when its post-rare-pooling distinct level count is at least
        this many. The default ``3`` is the encoding-invariance floor — below it a categorical
        axis admits exactly one partition however its levels are ordered, so per-class channels
        can only add axes. Raise it (e.g. to ``20``, matching ``cat_count_min_levels``) to
        restrict the channels to high-cardinality features.
    prune : bool, default=True
        Whether to run post-fit table pruning: drop fANOVA tables that don't improve
        cross-validated held-out deviance, and deploy the smaller, more explainable
        tables-only model (see ``pruning_report_``). ON by default — every insur-arena
        benchmark cell deployed the pruned artifact, and the selection is held-out-guarded
        (neutral-or-better by construction) — at the cost of a handful of extra single-bag
        fold fits. Pass ``False`` for the cheaper full-ensemble artifact; note a pruned
        model's ``tables()`` are frozen at fit (no explain-time re-weighting).
    prune_validation_fraction : float, default=0.15
        Train/select split fraction of the LEGACY multiclass (K>=3) prune
        (``multiclass_prune_cv=False``). Every other configuration selects on K-fold CV sized
        by ``prune_n_folds`` — a non-default value there raises ``ValueError`` rather than
        silently no-opping.
    multiclass_prune_cv : bool, default=True
        Multiclass (K>=3) prune selection regime; see :class:`TBoostClassifier`.
    multiclass_prune_guard_floor : float, default=0.002
        Floor of the multiclass (K>=3) prune guard's bar; see :class:`TBoostClassifier`.
    prune_se_rule : float, default=0.0
        Pruning selection rule, in standard errors of the cross-validated deviance estimate.
        ``0.0`` (the default) selects the held-out-deviance minimum, which improves or is
        neutral versus the unpruned model on every benchmark tried. A larger value (e.g.
        ``0.5`` or the classic ``1.0`` one-standard-error rule) prunes more aggressively for
        parsimony but monotonically degrades the deviance metric — a deliberate
        explainability/accuracy trade-off, opt in explicitly for smaller models.

        **MULTICLASS (K>=3) ONLY.** That path selects on a single train/select split, so the
        band decides the deployed keep-set directly. The regressor/binary path instead
        AGGREGATES per-fold selections, and the band reaches that aggregation only through
        ``kept_rate``, which the keep rule ORs against ``positive_rate`` — measured
        bit-identical banks for ``prune_se_rule`` in ``{0, 0.5, 1, 2, 10, 100}`` at four
        dataset sizes. A non-default value therefore raises ``ValueError`` there rather than
        silently no-opping; use ``prune_table_budget`` / ``prune_box_budget`` for a smaller
        bank on that path. Only has an effect together with ``prune=True``.
    prune_n_folds : int, default=5
        Number of cross-validation folds used to estimate each candidate table's
        contribution during pruning. Only has an effect together with ``prune=True``.
    prune_drop_z : float or None, default=2.0
        Evidence bar for DROPPING a table (av37). With paired per-fold CV evidence in hand, a
        candidate interaction table is dropped only when its mean drop-gain clears this many
        standard errors of that mean over the prune folds; no evidence means keep. ``None``
        restores the pre-av37 rule, which kept a table on the SIGN of its fold-mean and so
        dropped on evidence that could not resolve the question (measured: 11.0 +/- 1.9 of 14
        tables kept across 90 fits of identical data, seed the only difference). The gate is
        monotone — it can only ever keep MORE tables than the old rule, never fewer. Tables
        scored by fewer than two folds keep the old verdict: absence from a fold's bank is a
        structural fact, not an ambiguous measurement. Only has an effect with ``prune=True``.
    prune_keep_budget : int, default=32
        Explainability budget for ``prune_drop_z`` admissions. Ambiguous tables are ranked by
        mean gain and admitted only while the pre-cascade keep-set stays within
        ``max(prune_keep_budget, <the pre-av37 keep-set size>)``. A bank that was already
        larger than the budget is therefore left exactly as the old rule chose it, while a
        small candidate bank — the regime where the fold evidence cannot resolve anything —
        comes back whole.
    prune_box_budget : int, default=0
        DEPLOYED-BOX budget: the total rank-1 region boxes the deployed bank may carry. ``0``
        disables it, and the fit is then bit-identical to one without the parameter.

        ``prune_keep_budget`` counts TABLES, which is the right unit for a dense bank and the
        wrong one for a factored one: a factored effect (every order-3 and order-4 effect)
        deploys one box per realized tree-region and the rating export emits one row per box.
        A depth-3 tree contributes ONE box; a depth-6 tree contributes up to ``k0*k1*k2``, so
        lifting ``max_depth`` barely moves the table count and multiplies the boxes instead
        (measured: fremotor_payfreq split 0 goes 7,358 -> 165,809 boxes for 101 -> 145 effects).

        The budget is spent on the best held-out drop-gain PER BOX first — ties broken on
        purified variance per box, then on feature ids — and everything it cannot afford is
        dropped binarily, with the heredity cascade re-run so nothing deploys over a dropped
        subset. Dense tables cost zero boxes and are never touched. A budget at or above the
        bank's own box total is a verbatim no-op. Requires ``prune=True`` (raises otherwise).
    prune_lambda_boxes : float, default=0.0
        SELECTION-time price of one deployed box, in held-out-deviance units. ``0.0``
        disables it and the fit is bit-identical to one without the parameter.

        The selection-time analogue of ``prune_box_budget``, and the piece the av38 depth
        battery named as missing: selection could not see bank size, so it could not REFUSE a
        diffuse config. The backward prune walk optimizes held-out deviance alone and the SE
        rule then breaks TIES toward fewer tables — but a config that buys a real gain at 22x
        the boxes is not a tie, so nothing declined it. With this armed the waypoint objective
        becomes ``mean_deviance + prune_lambda_boxes * n_boxes``, so a diffuse rung has to
        out-earn its own size. The budget still applies afterwards; the two compose.

        Units are deviance PER BOX on the dataset's own per-unit-weight scale, deliberately a
        price rather than a ratio. Calibrate from ``pruning_report_["path"]``, whose points now
        always carry ``n_boxes``: "one SE across the whole bank" is
        ``path[0]["se"] / path[0]["n_boxes"]``, and "1% of full deviance across the whole
        bank" is ``0.01 * path[0]["mean_deviance"] / path[0]["n_boxes"]``. Dense tables cost
        zero boxes and are never priced.
    prune_table_budget : int, default=0
        DEPLOYED MULTI-WAY TABLE budget: the most tables of arity ``>= prune_table_min_arity``
        the deployed bank may carry. ``0`` disables it, and the fit is then bit-identical to
        one without the parameter.

        This is Ralph's 2026-08-27 explainability bar written as a number: *cells within a
        table are free, the COUNT of multi-way tables is what must stay small*.
        ``prune_box_budget`` prices the OTHER quantity — resolution — and bounds table count
        only as a side effect of dropping supports binarily, so it buys its table discipline by
        starving the tables it keeps. This caps the count directly and every survivor keeps
        every box it had.

        Spent on the best held-out drop-gain first (ties on purified variance, then feature
        ids); everything past the cap is dropped binarily with the heredity cascade re-run.
        Because every priced support costs exactly one, the surviving set is MONOTONE in the
        cap — unlike the box budget, where a larger budget can swallow a big effect and squeeze
        out a small one. Supports below the arity floor are never counted and never dropped: a
        main effect or a pair is what a filing reads. A cap at or above the bank's own count is
        a verbatim no-op. Requires ``prune=True`` (raises otherwise).

        Measured on fremotor_payfreq (4-class, budget-300 lane, 2 paired splits) against the
        box budget at a matched >=3-way count: the table budget returns ~15% more of the
        depth-6 gain, paid for in cells the bar now grants.

        For a K-class fit the cap counts SUPPORTS in the shared keep-set — one entry however
        many classes carry a copy — while an arity census that sums per-class copies reads
        about K times higher. Both numbers describe the same bank.
    prune_lambda_tables : float, default=0.0
        SELECTION-time price of one kept table of arity ``>= prune_table_min_arity``, in
        held-out-deviance units. ``0.0`` disables it and the fit is bit-identical to one
        without the parameter. The waypoint objective gains
        ``+ prune_lambda_tables * (kept tables at or above the floor)``.

        KNOWN LIMIT, and the reason this is not the knob to reach for if you want the bar met.
        Like ``prune_lambda_boxes`` it only re-picks a waypoint on an already-fixed
        deviance-greedy backward path; it cannot reorder the walk, so it cannot surrender a
        3-way table while holding a pair, and on the aggregated (regressor/binary) path a
        per-fold vote further damps it. Measured on a 9-feature depth-6 order-3 deploy fit with
        33 three-way tables: the price moves the count to 31, then 29, and SATURATES there
        across four more decades of lambda, while ``prune_table_budget`` lands on 8, 4 or 2
        exactly on request. Use this to let SELECTION see table count — the gap the av38
        battery named — not to hit a number.
    prune_table_min_arity : int, default=3
        Lowest interaction order that ``prune_table_budget`` and ``prune_lambda_tables`` count.
        Consulted only when one of them is armed, so its value cannot perturb a default fit.
        ``3`` is the product bar: mains and pairs are what a filing reads, three-way tables are
        what inflates it. Must be in ``1..8``.
    prune_guard_z : float, default=0.0
        SE multiplier for the set-level no-harm guard's breach test: it breaches, and the
        re-admission ladder stops, on ``gap > max(prune_guard_tol, prune_guard_z * SE)`` where
        ``SE`` is the standard error of the relative deviance gap on the guard's own evidence
        rows (reported as ``gap_se``, with the resulting bar as ``tol_effective``). The shipped
        ``0.0`` is the fixed-relative-tolerance test. Measured and deliberately off: on a
        zero-inflated compound target the row-level SE is dominated by a few large claims and
        cannot resolve even a real set-level gap — see the ``_PRUNE_GUARD_Z_DEFAULT`` note.
    prune_guard_z_dn : float, default=2.0
        SE multiplier for the DOWNWARD recalibration of the same bar (av39). The effective
        tolerance becomes ``min(tol_up, max(prune_guard_tol_floor, prune_guard_z_dn * SE))``,
        where ``tol_up`` is the ``prune_guard_z`` result above — so where the guard's jury is
        sharp the bar tightens to a few SE instead of a fixed 5%, and where it is blunt the
        ``min`` pins the bar back at ``prune_guard_tol`` and the fit is bit-identical. This knob
        can therefore only ever make the guard fire MORE, never less. It governs BOTH the breach
        trigger and the ladder's stopping rule. ``0.0`` restores the pre-av39 guard exactly.
    prune_guard_tol_floor : float, default=0.005
        Lower clamp on the downward recalibration, so an arbitrarily sharp jury cannot drive the
        bar to zero and re-admit the whole bank on immeasurable noise. Ignored when
        ``prune_guard_z_dn = 0``.
    prune_min_stability : float, default=0.5
        Fold-stability bar for keeping a table: the aggregator keeps one when
        ``mean_gain > prune_min_mean_gain`` **and** (it survived at least this fraction of the
        prune folds' own selections, **or** scored a positive drop-gain in at least this
        fraction of them). In plain terms: *in how many folds must this table show signal.*
        Previously a hardcoded ``0.5`` — this is the knob that actually governs the deployed
        bank's size on the regressor/binary path, where ``prune_se_rule`` cannot reach.

        The fold rates quantise to ``1/prune_n_folds``, so at the default 5 folds this is a
        3-position dial: ``(0.4, 0.6]`` = 3/5, ``(0.6, 0.8]`` = 4/5, ``(0.8, 1.0]`` = 5/5.
        Measured on French MTPL frequency (3 seeds, ``n_bags=8``): ``1.0`` takes the deployed
        bank from 58.3 to 37.0 tables and 1216 to 389 boxes for −0.49% ± 0.66 of test D², the
        best explainability-per-accuracy-point of any pruning knob, and it halves the
        seed-to-seed spread in bank size. The default stays ``0.5`` pending a multi-dataset
        battery. Multiclass (K>=3) has no fold vote to threshold, so a non-default value raises
        there. Only has an effect together with ``prune=True``.
    prune_min_mean_gain : float, default=0.0
        Minimum mean held-out drop-gain a table must earn to be kept, in deviance units on the
        dataset's own per-unit-weight scale. ``0.0`` (the default) is the historical
        ``mean_gain > 0`` test, bit-for-bit — under it a table kept on a mean gain of ``1e-9``
        is kept on noise. This is the threshold-parameterised counterpart to
        ``prune_table_budget``'s count cap: drop everything whose evidence is weaker than a
        stated bar, rather than keeping the best-ranked N. The native pruning report has always
        carried a ``min_mean_gain`` field and always written a literal ``0.0``; it now reports
        the threshold actually used. Multiclass (K>=3) raises on a non-default value, as for
        ``prune_min_stability``. Only has an effect together with ``prune=True``.

        **It interacts with ``prune_drop_z``, and not in the obvious direction.** The floor
        applies to the LEGACY keep rule only. With the evidence gate armed (the default
        ``prune_drop_z=2.0``), a table the floor rejects does not simply leave — it becomes an
        *ambiguous candidate* for the gate, whose own rule is "keep unless the fold evidence is
        significantly negative", so it is very likely re-admitted. Raising the floor can
        therefore GROW the deployed bank. Measured on an 8-feature synthetic Poisson fit: with
        the gate on, ``0.0 -> 1e-3`` takes ``legacy_selected`` 18 -> 9 but ``kept`` 26 -> 28;
        with ``prune_drop_z=None`` the same sweep takes ``kept`` 13 -> 9, monotone as intended.
        Pair this with ``prune_drop_z=None`` when using it as a parsimony bar.

        Where the gate admits nothing it needs no such pairing: on French MTPL frequency
        (single-bag, seed 0) the gate re-admits 0 of 27 candidates, and ``0 -> 1e-5 -> 1e-4``
        takes the deployed bank 54 -> 46 -> 31 tables, with ``1e-5`` landing 15% fewer tables at
        a slightly BETTER test deviance (0.55911 vs 0.55915) than the default.
    prune_fold_min_rows : int, default=125
        Rows-per-fold floor that sizes the prune CV: the fold count is
        ``max(2, min(prune_n_folds, n // prune_fold_min_rows, n))``, so on small data the number
        of folds adapts DOWN rather than producing folds too thin to score a table. Raising it
        makes that adaptation kick in earlier. Only has an effect together with ``prune=True``.
    prune_fold_es_patience : int or None, default=None
        Early-stopping patience for the prune-CV FOLD fits only; the deploy fit always keeps the
        estimator's own ``early_stopping_rounds``. ``None`` resolves to the shipped ``250``; an
        explicit integer overrides it. The fold fits exist only to vote on table survival, which is why
        their patience is tighter than the deploy fit's — 250 left every studied keep-set
        bit-identical with the MTPL CV scan 19.4% faster. Only has an effect together with
        ``prune=True``.
    prune_guard_min_rows : int, default=500
        Minimum honest evidence rows before the set-level no-harm guard will judge at all —
        below this the pruned-vs-full deviance ratio is noise and the guard reports itself
        skipped rather than acting on it. Counts out-of-bag-covered rows on the OOB path and
        shared-holdout rows on the grouped carve path. Only has an effect together with
        ``prune=True`` and ``prune_guard=True``.
    prune_slope_eps : float, default=0.01
        Dead band on the post-prune slope re-anchor: the correction is not applied unless the
        out-of-bag scale ``b`` differs from 1 by more than this, so a bank that was not
        materially compressed keeps a byte-identical artifact. See ``pruning_report_["slope"]``.
    prune_slope_min_z : float, default=3.0
        Significance bar for that same correction: ``b`` must also be off by at least this many
        standard errors (``z_b``) before it is applied. Together with ``prune_slope_eps`` this
        is the "materially AND significantly off" test — lowering either makes the re-anchor
        fire more readily. See ``pruning_report_["slope"]``.
    prune_refit_full : bool, default=False
        Deprecated and without effect on any path: contribution-stability pruning always
        fits the deployed model on all rows and applies the keep-set to that same fit, so
        there is no separate subset-pruned fit to conditionally refit. Setting this ``True``
        raises ``ValueError`` at fit time rather than silently ignoring it. Kept only for
        backward constructor compatibility (``get_params``/``set_params``/``clone`` still
        round-trip it).
    prune_rebalance : bool, default=True
        Whether to re-solve the surviving tables' cell values to the deviance-optimum of the
        reduced structure after pruning drops tables ("rebalance on means": a single IRLS
        ridge cell-refit step, re-purified to preserve the 1-/2-/3-way order separation).
        Held-out no-harm guarded: adopted only if it lowers held-out deviance. Set ``False``
        to deploy the drop-only pruned model. Only has an effect together with ``prune=True``.
    ref_measure : {"exposure", "product_marginals", "uniform"} or None, default=None
        The reference measure the fitted tables are purified against, i.e. the weighting
        that decides which table owns the part of the score that could sit in more than one
        place. ``None`` resolves to ``"exposure"`` (2026-09-06): each axis is weighted by
        its exposure-weighted empirical marginal (``sample_weight * exposure`` per row) plus
        a small positivity floor (``measure_floor``), so the prune, the per-bag banks and
        the export all read the fit-time effective mass. ``"product_marginals"`` is the
        pre-2026-09-06 measure: a flat ROW count blended half-and-half with a uniform
        component (``laplace=1``), which gave empty and thin cells half the weight on every
        axis; it is kept so earlier boards stay byte-reproducible. ``"uniform"`` weights
        every cell equally. Changing the measure changes which tables the prune keeps and
        therefore the deployed model, not just the ledger; use ``tables(ref_measure=...)``
        to re-express a fitted model under another measure without touching predictions.
    measure_floor : float, default=1e-3
        Under ``ref_measure="exposure"``: the total floor mass, as a fraction of the
        empirical mass, spread evenly over each axis's cells so every weight stays
        strictly positive. Must be finite and > 0. Inert under the other measures.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``. Only set when ``X`` was a DataFrame-like
        object exposing column names — check with
        ``getattr(est, "feature_names_in_", None)``, per the sklearn convention, rather than
        assuming it is always present.
    pruning_report_ : dict
        Diagnostic report from post-fit table pruning: which fANOVA effects were scanned,
        kept, and deployed, and their cross-validated contributions. Only set when
        ``prune=True``.

        ``["guard"]`` records the set-level no-harm guard, and ``["slope"]`` the post-prune
        slope re-anchor: the deployed bank's out-of-bag scale ``b`` relative to the full
        bank's (``b_keep``/``b_full``), its standard error ``se_b`` and z-statistic ``z_b``,
        the ``sd_ratio`` measuring how much link-scale spread the prune removed, and
        ``applied`` — with ``skipped`` naming the gate when it did not fire. Present only on
        a log-link fit whose guard ran on out-of-bag evidence; the correction is applied only
        when the scale is both materially and *significantly* off, so a fit that does not
        need it keeps a byte-identical artifact.

        The av37 evidence gate adds ``["drop_z"]`` / ``["keep_budget"]`` (the knobs in force),
        ``["legacy_selected"]`` (what the pre-av37 rule would have kept), ``["evidence_budget"]``
        = ``max(keep_budget, legacy_selected)``, ``["evidence_candidates"]`` (tables the gate
        wanted that the old rule refused), ``["evidence_admitted"]`` (those it could afford, in
        rank order), ``["evidence_budget_bound"]`` (whether the budget truncated them) and
        ``["evidence_scores"]`` (per table: fold count, mean gain, SE and t). ``drop_z: null``
        means the gate was off and every other field is inert. ``["guard"]`` likewise gains
        ``guard_z``, ``gap_se`` (the SE of the relative deviance gap on the evidence rows) and
        ``tol_effective`` — the bar actually used for BOTH the breach trigger and the
        ladder's stopping rule — plus the av39 ``guard_z_dn`` / ``tol_floor`` knobs and
        ``n_chunks`` (rungs the ladder had available against ``steps``, the rungs climbed).
        ``tol_effective`` = ``min(max(prune_guard_tol, guard_z*gap_se),
        max(tol_floor, guard_z_dn*gap_se))``, each term skipped when its z is 0.

        ``["box_budget"]`` appears only when ``prune_box_budget > 0``: ``max_boxes`` (the
        budget), ``boxes_before``/``boxes_after`` and ``effects_before``/``effects_after`` (the
        deployed rank-1 box and factored-effect counts either side of it), ``engaged`` (false
        when the bank already fit, in which case the model is bit-identical to an unbudgeted
        one), ``dropped`` (the supports the greedy could not afford, best evidence-per-box
        first, so the tail is the weakest) and ``cascade_dropped`` (those removed after, each
        properly containing a dropped support).
    graduation_report_ : list of dict
        Per-effect graduation diagnostics. Dense entries contain the WH/GCV strength;
        higher-order entries contain ``method="reference_diffusion"``, requested strength,
        candidate box counts, and any skip reason. ``applied`` describes candidate
        construction; ``rejected`` records holdout/training rejection and ``alpha`` is
        zero when skipped or rejected. Only available when graduation is enabled.

    Examples
    --------
    >>> import numpy as np
    >>> from t_boost import TBoostRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3)).astype(np.float32)
    >>> y = X[:, 0] + 0.5 * X[:, 1] * X[:, 2]
    >>> est = TBoostRegressor(n_trees=100, validation_fraction=None, n_bags=1)
    >>> est.fit(X, y).predict(X).shape
    (200,)
    """

    @classmethod
    def from_bytes(cls, data: bytes) -> "TBoostRegressor":
        """Deserialize a fitted estimator from the wire format written by :meth:`to_bytes`.

        Parameters
        ----------
        data : bytes
            A blob previously produced by :meth:`to_bytes` (either this or an older
            release — the envelope is versioned and back-compatible).

        Returns
        -------
        TBoostRegressor
            A new, already-fitted estimator instance.
        """
        est = cls()
        md, inner = _unpack_bytes(data)
        if inner[:4] == _TABLES_MAGIC:
            est._attach_model(_TableModel.from_bytes(inner))
        else:
            est._attach_model(_Model.from_bytes(inner))
        _restore_metadata(est, md)
        return est

    @classmethod
    def from_json(cls, data: str) -> "TBoostRegressor":
        """Deserialize a fitted estimator from the wire format written by :meth:`to_json`.

        Parameters
        ----------
        data : str
            A JSON document previously produced by :meth:`to_json`.

        Returns
        -------
        TBoostRegressor
            A new, already-fitted estimator instance.
        """
        est = cls()
        md, inner = _unpack_json(data)
        if '"t-boost-tables"' in inner:
            est._attach_model(_TableModel.from_json(inner))
        else:
            est._attach_model(_Model.from_json(inner))
        _restore_metadata(est, md)
        return est

    def _expected_response(self, X: Any, exposure: Any | None) -> "np.ndarray":
        # `predict` returns the mean per unit exposure for the log-link objectives (exposure
        # entered the fit as an offset), so the expected total is prediction × exposure.
        pred = np.asarray(self.predict(X), dtype=np.float64)
        if exposure is not None and self.objective in {"poisson", "gamma", "tweedie"}:
            pred = pred * _as_float32_1d(exposure, "exposure").astype(np.float64)
        return pred

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
        exposure: Any | None = None,
        groups: Any | None = None,
    ) -> "TBoostRegressor":
        """Fit the boosted ensemble.

        ``groups``: per-row entity ids for panel data; a ``str`` names a column of the polars
        ``X`` (excluded from the features). When given, the INTERNAL early-stopping holdout
        AND the prune-CV fold carve (train/select split when ``prune=True``) assign WHOLE
        groups to one side (see ``_carve_group_holdout``/``_carve_group_folds``) instead of
        rows — on panels (the same entity contributing near-duplicate rows across periods) the
        row-level carve leaks near-duplicate rows into validation, which defeats early stopping
        (fits run to the tree cap) and biases prune selection.

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Training input. A polars ``DataFrame`` (or ``LazyFrame``, collected at fit) is
            the first-class frame type: column names set ``feature_names_in_``,
            String/Categorical/Enum columns are target-statistic encoded natively as
            categoricals with no declaration needed (nulls collapse to the reserved missing
            level), numeric columns are cast to ``float32`` (nulls become NaN, routed to the
            missing bin), and any column consumed by a column-name ``y``/``sample_weight``/
            ``exposure``/``groups`` below is dropped from the feature set. Other DataFrame-likes
            with column names keep the declared-``categorical_features``-only behavior.
        y : array-like of shape (n_samples,), or str
            Target values, on the natural (not link) scale. A ``str`` names a column of the
            polars ``X`` to use (and exclude from the features), rustystats-style.
        sample_weight : array-like of shape (n_samples,), or str, optional
            Per-row weights; a ``str`` names a column of the polars ``X``. ``None`` weights
            every row equally.
        exposure : array-like of shape (n_samples,), or str, optional
            Per-row exposure (offset) for rate/frequency modeling with the Poisson/Tweedie
            objectives — the fitted rate is multiplied by exposure before comparison to
            ``y``; a ``str`` names a column of the polars ``X``. ``None`` is equivalent to
            an exposure of 1 for every row.
        groups : array-like of shape (n_samples,), or str, optional
            Per-row entity ids for panel/grouped data; a ``str`` names a column of the polars
            ``X``. See the class-level note above. Besides the group-honest internal carves,
            the outer bags are drawn at GROUP granularity (since 2026-09-07), so the
            out-of-bag rows the cell refit and the prune guard read never contain a row
            whose group-mates were in bag. ``None`` (the default) keeps the ordinary
            row-level internal carves — as does an ALL-SINGLETON grouping, which is normalized
            to ``None`` at ingest because at group size 1 it means the same thing (a ``str``
            column is still consumed out of the feature set either way).

        Returns
        -------
        TBoostRegressor
            ``self``, now fitted.
        """
        # Deprecation notices fire HERE, at the top of the public fit, so they are
        # independent of `prune` (most deprecated names are not prune knobs) and never
        # fire from `__init__`, which sklearn requires to be assignment-only and which
        # `clone`/`get_params` re-enter constantly. The unhonourable-parameter RAISES run
        # later on their own paths and outrank these warnings.
        self._warn_deprecated_params()
        X, resolved = self._resolve_fit_vectors(
            X,
            {
                "y": y,
                "sample_weight": sample_weight,
                "exposure": exposure,
                "groups": groups,
            },
        )
        self._fit_model(
            X,
            resolved["y"],
            sample_weight=resolved["sample_weight"],
            exposure=resolved["exposure"],
            class_labels=None,
            groups=resolved["groups"],
        )
        # Proactive half of `binding_report_`: interrupt when a parameter the caller set
        # was actively CANCELLED by another. Fires after the fit because the verdict needs
        # the prune report and the deployed artifact. OVERRIDDEN only — see the method.
        self._warn_voided_params()
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict on the response scale, per unit exposure for log-link fits.

        With Poisson/Gamma/Tweedie exposure offsets, this returns a rate, not a row
        total. Expected totals are ``predict(X) * exposure``; extra exposure columns
        in ``X`` are ignored by prediction.

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Samples to score. A polars frame is matched by NAME to the fit-time feature
            columns: extra columns are ignored, column order is irrelevant, and a
            ``LazyFrame`` collects only the needed columns. Array-likes (and a model fitted
            without column names) must instead have the same number and layout of features
            as the data passed to ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values as ``float64`` (the underlying scores are computed in
            ``float32``; the wider return type follows sklearn's convention and avoids
            ``float32`` accumulation error in downstream metrics).
        """
        check_is_fitted(self, "_model")
        x32, cat_kw = self._serve_kwargs(X, self._model)
        # Return float64 (the core scores in f32): sklearn's convention, and it avoids f32
        # accumulation error in downstream metrics.
        # n_jobs bounds the pruned-table score par_iter to the estimator's thread budget instead
        # of the global pool grabbing every core (which oversubscribes a box running many fits).
        return np.asarray(
            self._model.predict(x32, **cat_kw, n_jobs=self._resolve_n_jobs()), dtype=np.float64
        )

    def predict_raw(self, X: Any) -> np.ndarray:
        """Predict on the link scale, before the objective's inverse link is applied.

        For ``objective="squared_error"`` this is identical to :meth:`predict`. For the
        log-link objectives (``poisson``/``gamma``/``tweedie``) this returns the linear
        predictor (log of the mean) rather than the mean itself.

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Samples to score. A polars frame is matched by NAME to the fit-time feature
            columns: extra columns are ignored, column order is irrelevant, and a
            ``LazyFrame`` collects only the needed columns. Array-likes (and a model fitted
            without column names) must instead have the same number and layout of features
            as the data passed to ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Link-scale predictions as ``float64``.
        """
        check_is_fitted(self, "_model")
        x32, cat_kw = self._serve_kwargs(X, self._model)
        return np.asarray(
            self._model.predict_raw(x32, **cat_kw, n_jobs=self._resolve_n_jobs()),
            dtype=np.float64,
        )

    def to_bytes(self) -> bytes:
        """Serialize the fitted estimator to a compact binary wire format.

        Returns
        -------
        bytes
            A self-describing blob (parameters, fitted metadata, and the model) that
            round-trips through :meth:`from_bytes`.
        """
        check_is_fitted(self, "_model")
        return _pack_bytes(self, self._model.to_bytes())

    def to_json(self) -> str:
        """Serialize the fitted estimator to a human-readable JSON wire format.

        Returns
        -------
        str
            A self-describing JSON document (parameters, fitted metadata, and the model)
            that round-trips through :meth:`from_json`. Larger than :meth:`to_bytes` but
            diffable and inspectable without this library.
        """
        check_is_fitted(self, "_model")
        return _pack_json(self, self._model.to_json())

    def tables(
        self,
        X: Any,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        basis_json: str | None = None,
        measure_floor: float | None = None,
        overflow: str | None = None,
        sample_weight: Any | None = None,
        exposure: Any | None = None,
    ) -> str:
        """Export the exact fANOVA effect-table decomposition as JSON.

        The returned tables are the glassbox explanation of the model: every table lookup
        the served rows in ``X`` actually touch, such that their sum (plus the intercept)
        exactly reproduces :meth:`predict_raw` — no approximation, no surrogate fit.

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Rows used to determine which table cells are populated/reported. A polars frame
            is matched by NAME to the fit-time feature columns (extra columns ignored, any
            order; a ``LazyFrame`` collects only the needed columns). Array-likes (and a
            model fitted without column names) must instead have the same number and layout
            of features as the data passed to ``fit``.
        ref_measure : {"exposure", "product_marginals", "uniform", "joint"} or None, default=None
            Reference measure the tables are purified against. ``None`` means the ledger
            this estimator fitted under (the constructor's ``ref_measure``, ``"exposure"``
            by default): for a pruned fit the deployed bank exactly as stored, for an
            unpruned fit a fresh purification under that measure. An explicit name
            re-expresses the SAME function under that measure — the table sum on every cell,
            and so every prediction, is unchanged; only which table owns shared mass moves.
            On a pruned fit this runs on the bank's own stored support, so no rows are read.
            ``"exposure"`` weights each axis by its exposure-weighted empirical marginal plus
            the ``measure_floor`` positivity floor; ``"product_marginals"`` is the legacy
            half-uniform row-count blend (``laplace``); ``"uniform"`` weights every cell
            equally. ``"joint"`` (export-only, 2026-09-06) re-expresses orders one and two
            using a regularized pairwise joint-exposure reallocation (ridge 0.001,
            positivity floor 1e-6). It preserves the total score, but the ridge prevents
            exact conditional purity and the floor assigns some mass to unsupported cells.
            The export reports residual conditional means on observed support. Main effects
            are not observed marginal A/E or causal effects. Orders three and above stay
            product-purified (a hybrid), variance shares no longer add to one, and the
            equal-split attributions are no longer Shapley values.
        laplace : float, default=1.0
            Laplace smoothing constant for the ``"product_marginals"`` reference measure.
            Unused for the other measures.
        measure_floor : float or None, default=None
            Positivity floor for ``"exposure"``; ``None`` inherits the fit-time
            ``measure_floor``. Unused for the other measures.
        basis_json : str or None, default=None
            Optional JSON-encoded custom basis/grid to export the tables against, in place
            of the model's native training-time bin borders. ``None`` uses the native basis.
        overflow : {"factored", "error", "sparse"} or None, default=None
            Table-budget overflow policy for effects whose merged grid exceeds the dense
            cell budget. ``None``/``"factored"`` (the default) keeps an over-budget effect
            exactly as a factored per-tree-box sum rather than materializing the dense
            table; ``"error"`` raises instead; ``"sparse"`` stores it as an exact sparse
            tensor. All three policies are exactness-preserving.
        sample_weight : array-like of shape (n_samples,) or str, optional
            Per-row weight for ``X``, aligned to ``X``'s rows; a ``str`` names a column of
            the polars ``X``. When present (explicitly, or
            via the fit-time default below), each table's ``support`` — and, under
            ``ref_measure="product_marginals"``, the empirical reference measure itself — is
            computed from the effective row mass (``sample_weight * exposure``, elementwise;
            either alone is used as-is) instead of a flat row count, so a thin-looking cell
            backed by a few highly-weighted rows reports its true mass (spec §08.7). This
            changes ``values``/importances/SE bands too, not just ``support`` — it is a
            different, equally valid decomposition against a different reference measure,
            not a display-only reweighting. **Default:** when ``None``, falls back to the
            ``sample_weight`` the estimator was fit with (if any) — the honest default for
            exporting a weighted/exposure fit's own tables. That fallback only makes sense
            when ``X`` is the fit-time data: pass an explicit array (or ``numpy.ones``, to
            force the unweighted export) whenever ``X`` is different rows, since a different
            ``X`` with a coincidentally matching row count would silently misapply the
            fit-time mass. Cleared by ``set_params`` and refits, never leaks across fits.
            No effect if ``self`` is a pruned (``prune=True``) model: its tables are already
            frozen from fit time.
        exposure : array-like of shape (n_samples,) or str, optional
            Per-row exposure for ``X``, aligned to ``X``'s rows; a ``str`` names a column of
            the polars ``X``. See ``sample_weight`` — the
            two combine multiplicatively into the same effective row mass, and each defaults
            independently to its fit-time value when ``None``.

        Returns
        -------
        str
            A JSON document describing every exported effect table (features, shape,
            values, per-cell support, and standard-error band). Each categorical axis
            (single- or multi-channel) additionally carries a ``"levels"`` list of
            ``{"label": ..., "cell": ...}`` entries — one per post-rare-pooling category
            level (the reserved rare bucket shown as ``"<rare>"``) — so a level's actual
            relativity is ``values``/``relativities`` at that ``cell`` index; levels the
            model cannot distinguish correctly share one ``cell``. ``None`` for a purely
            numeric axis.
        """
        check_is_fitted(self, "_model")
        X, sample_weight, exposure = self._resolve_explain_inputs(X, sample_weight, exposure)
        x32, cat_x = self._serve_design(X)
        # Explicit call-time mass wins; otherwise default to the fit-time stash (each of the
        # two components independently). See the docstring caveat about non-fit-time X.
        weight32 = (
            _as_float32_1d(sample_weight, "sample_weight")
            if sample_weight is not None
            else getattr(self, "_fit_sample_weight_", None)
        )
        exposure32 = (
            _as_float32_1d(exposure, "exposure")
            if exposure is not None
            else getattr(self, "_fit_exposure_", None)
        )
        payload = self._model.tables(
            x32,
            **self._export_measure_kwargs(ref_measure, laplace, measure_floor),
            basis_json=basis_json,
            cat_x=cat_x,
            overflow=overflow,
            weight=weight32,
            exposure=exposure32,
        )
        return annotate_joint_export(payload, ref_measure)


class TBoostClassifier(ClassifierMixin, _BaseTBoost):  # type: ignore[misc]
    """Exact, depth-3 oblivious-tree gradient boosting classifier.

    Binary (K=2) fits stay on an exact single-model logistic path. Fits with three or more
    classes (K>=3) use a native multinomial-softmax path with one exact fANOVA table bank
    per class. Every tree is a symmetric (oblivious) depth-3 tree touching at most three
    distinct raw features, so each class's fitted ensemble decomposes exactly into a sum of
    at-most-3-way fANOVA effect tables (see :meth:`tables`) with no approximation.

    The defaults include a complete fitting pipeline. The September 2026 benchmark
    describes an earlier pipeline; current graduation reserves a separate holdout. early stopping is on
    by default (``validation_fraction=0.1``, ``early_stopping_rounds=500`` with adaptive
    patience ``early_stopping_adaptive=1.5``, against a large ``n_trees=4000`` cap), outer
    bagging is on by default (``n_bags=8``), per-tree column sampling is on by default
    (``colsample_bytree=0.8``), and post-fit table pruning is on by default (``prune=True`` —
    the fitted artifact is the CV-pruned tables-only model). Pass
    ``n_bags=1, validation_fraction=None, prune=False`` for the cheapest single-fit baseline.

    polars DataFrames/LazyFrames are first-class ``X`` input (their extracted numeric block
    is already F-contiguous float32, the cheapest ingest layout; String/Categorical/Enum
    columns are categorical automatically, and ``y``/``sample_weight``/``exposure`` may name
    frame columns). For plain arrays, pass
    F-contiguous ``float32`` arrays (``numpy.asfortranarray``) for ``X`` for the cheapest ingest
    — its columns copy in one pass each with no transpose, unlike the C-contiguous default.

    **Honesty notes.** With ``groups=`` on :meth:`fit`, the outer bags are drawn at GROUP
    granularity (since 2026-09-07): no group straddles a bag's in-bag/out-of-bag boundary, so
    the out-of-bag rows the cell refit and the prune guard read are honest on panel data.
    The internal early-stopping holdout carve is stratified by class in
    the native core for every ``n_bags`` configuration — for binary (K=2) AND multiclass
    (K>=3) fits alike (the per-class labels are the strata). Multiclass fits also have no ``graduate``/Whittaker-Henderson graduation path at
    all (an explicit ``graduate=True`` raises rather than silently no-opping), and no
    ``exposure`` support (:meth:`fit` raises if both are given for K>=3).

    The multiclass (K>=3) native-softmax path HONORS: ``n_bags``/``bag_subsample`` (outer bags
    souped per class, since 2026-07-14), ``subsample``/``mvs_min_rows`` (§06.5 MVS row sampling,
    since 2026-08-23 — one row draw per round shared by the K class trees, weighted by the joint
    gradient/hessian norm across classes), the post-prune per-class intercept re-anchor (an
    intercept-only multinomial IPF, applied unconditionally inside the prune — it is NOT the
    ``reanchor`` knob), ``multiclass_prune_guard`` (the K>=3 set-level prune no-harm guard,
    ON by default since 2026-09-07 with an SE-aware bar; see ``multiclass_prune_cv``), and ``cell_refit_base``/``cell_refit_gamma`` (the §G1 out-of-bag fANOVA
    cell refit, since 2026-08-25 — K decoupled diagonal-Hessian cell solves on the softmax
    working residual, accepted or shrunk by ONE joint backtrack on the true multinomial loss).
    ``leaf_refine_steps``/``leaf_refine_backtracks`` have no effect for K>=3 (a Jacobi
    refinement measured harmful as a drop-in), so a default fit is byte-identical to
    ``leaf_refine_steps=0``.

    It does NOT apply
    ``ridge_refit_l2``/``ridge_refit_max_iter`` (no fully-corrective refit),
    ``dart_drop_rate`` (no DART), ``reanchor``/``reanchor_slope`` (no constructor-driven
    intercept/slope recalibration), or ``lambda_scale_invariant`` (no scale-invariant lambda
    rescale) — each is honored only on the binary (K=2) / regression path. No silent no-ops:
    setting any of those away from its own no-op default raises ``ValueError`` at multiclass fit
    time, since there is a real explicit request there and no way to honor it.
    ``leaf_refine_steps`` is the one exception — its shipped recipe default (``4``) is already
    non-inert, so raising would break ordinary default multiclass fits; it (with
    ``leaf_refine_backtracks``) warns once per estimator instance instead. Each affected
    parameter's entry below repeats this scoping for discoverability.

    Parameters
    ----------
    n_trees : int, default=4000
        Maximum number of boosting rounds (an upper bound; a round also stops early if no
        candidate split clears ``min_split_gain``). The large default cap is meant to be
        governed by early stopping (see ``validation_fraction``), not hit directly.
    learning_rate : float, default=0.05
        Shrinkage factor applied to every tree's leaf values before they are added to the
        running score.
    lambda_ : float, default=1.0
        L2 regularization on leaf values: the ridge penalty ``lambda`` in the Newton
        leaf-value solve ``w* = -G / (H + lambda)`` and in the split gain.
    lambda_scale_invariant : bool, default=False
        Rescale ``lambda_`` by the fit's mean per-row hessian instead of using it literally in
        ``H + lambda``, so it stays a live knob regardless of ``sample_weight`` magnitude (see
        :class:`TBoostRegressor`'s parameter doc for the mechanism). Binary (K=2) path only —
        the native multiclass (K>=3) path does not yet resolve it: setting this to ``True``
        raises ``ValueError`` at multiclass fit time rather than silently no-opping (see the
        class-level honesty note), the same convention as ``subsample``/``reanchor`` above.
    l1_leaf : float, default=0.0
        L1 regularization on leaf values, applied by soft-thresholding the aggregated
        gradient before the Newton solve.
    min_split_gain : float, default=0.0
        Minimum gain a candidate split must clear to be taken. A level whose best gain does
        not exceed this floor stops growing and its node stays a leaf.
    max_delta_step : float or None, default=None
        Caps the magnitude of a leaf's Newton update, applied to the full-precision
        aggregated step before ``learning_rate``. ``None`` defers to the objective's own
        default; for ``objective="logistic"`` (the only supported value) that default is
        uncapped. An explicit value always overrides it.
    max_delta_step_gated : None, bool, str, tuple or dict, default=None
        In-fit rate-collapse gate on ``max_delta_step`` (spec §05.6 addendum). Inert here:
        the gate is an objective-aware default that only ``tweedie`` ships, and
        ``objective="logistic"`` is the only supported value on this estimator. Accepted for
        parameter-surface parity with :class:`TBoostRegressor`, where it is documented in
        full. An explicit ``max_delta_step`` outranks it in any case.
    max_bin : int, default=254
        Maximum number of histogram bins per numeric feature.
    objective : {"logistic"}, default="logistic"
        Loss function to fit. Currently only ``"logistic"`` is supported — :meth:`fit`
        raises ``ValueError`` for any other value. Kept as a constructor parameter (rather
        than hard-coded) for symmetry with :class:`TBoostRegressor` and forward
        compatibility.
    tweedie_rho : float, default=1.5
        Unused by this classifier (relevant only to :class:`TBoostRegressor`'s
        ``objective="tweedie"``). Kept for constructor symmetry.
    min_data_in_leaf : int or None, default=None
        Minimum number of training rows a leaf must retain; a split that would leave either
        child below this count is rejected. ``None`` resolves **depth-aware**: ``0`` (inert,
        the historical default) at every depth: P-D2 measured a depth-coupled floor across
        the regression-stress battery and it was never better and sometimes much worse (see
        ``_MIN_DATA_PER_LEAF_AT_DEPTH``). **Set this explicitly for a model destined for
        filing**: it is the only guard that stops a displayed relativity from standing on a
        handful of policies, and a lifted (``max_depth > 3``) fit puts ~3.4x fewer rows in
        each cell.
    min_sum_hessian_in_leaf : float, default=0.0
        Minimum total Hessian a leaf must retain; a split that would leave either child
        below this is rejected. Complements ``min_data_in_leaf`` for weighted or
        non-constant-Hessian objectives, where row count alone under- or over-states a
        leaf's information mass.
    min_weight_sum_in_leaf : float, default=0.0
        Minimum total ``sample_weight`` a leaf must retain; a split that would leave either
        child below this is rejected.
    path_smooth : float or None, default=None
        LightGBM-style path smoothing: blends a leaf's own Newton estimate with its
        parent's with credibility weight ``Z = n/(n + path_smooth)``, shrinking
        thin-evidence leaves toward the ancestor path. ``None`` (the default) resolves
        objective-aware: ``10.0`` for ``gamma``/``tweedie`` — heavy-tail deviance is
        dominated by sparse-evidence cell variance, and this is its zero-cost smoother
        (2026-07-23 six-set screen: freclaimdam −1.2%, ohlsson_pp −0.56%, worst +0.05%) —
        and ``0.0`` (off) for every other objective. An explicit float always wins;
        ``0.0`` disables it.
    subsample : float or None, default=None
        Row-sampling rate for split search. ``None`` scans every row. A value in ``(0, 1)``
        switches to minimal-variance-style (MVS) sampling: a probability-proportional-to-
        gradient row subset chooses the split structure, with ``mvs_min_rows`` as a floor on
        the sampled count; leaf values are still refit from all rows regardless. Binary
        (K=2) path only — the multiclass (K>=3) path always scans every row: setting this
        away from its default (``None``) raises ``ValueError`` at multiclass fit time rather
        than silently ignoring it (see the class-level honesty note).
    colsample_bytree : float, default=0.8
        Fraction of features randomly sampled per tree.
    learning_rate_decay : float, default=0.0
        Per-round multiplicative decay applied to ``learning_rate``. ``0.0`` is a constant
        rate.
    validation_fraction : float or None, default=0.1
        Fraction of training rows carved out as an internal early-stopping holdout. ``None``
        disables internal early stopping entirely, so every fit runs the full ``n_trees``.
        See the class-level honesty note above regarding class-stratification of this carve
        for binary versus multiclass fits.
    early_stopping_rounds : int, default=500
        Patience: the fit stops once this many rounds pass without a validation-deviance
        improvement (see ``early_stopping_min_delta`` for what counts as an improvement).
        Ignored when ``validation_fraction`` is ``None``.
    early_stopping_adaptive : float or None, default=1.5
        Adaptive patience ratio. A ratio ``r > 0`` grows the effective patience with
        progress: ``clamp(ceil(r * best_round), 50, early_stopping_rounds)`` — a fit that
        peaks early stops sooner. The ``1.5`` default matches every insur-arena benchmark
        run (previously the recipe factory set it while the bare constructor kept fixed
        patience). ``None`` restores the fixed ``early_stopping_rounds`` patience.
    early_stopping_min_delta : float, default=1e-4
        Relative improvement tolerance: a round only becomes the new best (resetting the
        patience window and moving the deployed truncation point) once validation deviance
        clears ``best * (1 - early_stopping_min_delta)``. ``0.0`` is the legacy
        any-improvement rule, under which epsilon-level validation noise can reset patience
        indefinitely; the shipped default ignores that noise so early stopping actually
        terminates on large, low-signal data. Must be finite and in ``[0.0, 1.0)``.
    interaction_gain_hurdle : float, default=2.0
        Interaction-admission hurdle (soft heredity). A split that would introduce a new raw
        feature — raising the tree's fANOVA interaction order — must clear this scale-free
        hurdle against the tree's level-1 (main-effect) gain; the new feature also competes
        against the best already-used raw feature under the split-ranking score, with ties
        won by the lower-order (reused-feature) split. ``0.0`` restores pure greedy
        admission, where a fresh feature is used only when no reused feature is available.
        Must be finite and ``>= 0``.
    interaction_gain_hurdle_mode : {"adaptive", "fixed"}, default="adaptive"
        How ``interaction_gain_hurdle`` is applied. ``"fixed"`` uses the scalar hurdle
        exactly as supplied. ``"adaptive"`` (the default) decays the effective hurdle from
        its full early-tree value toward a lenient late-tree value as main-effect gains
        fade, and additionally requires 3-way admissions to clear a stricter multiplier than
        2-way admissions.
    graduate : bool or None, default=None
        Binary (K=2) fits use the regressor's graduation policy -- smoothing on the
        full fit, no holdout, no accuracy gate (see :class:`TBoostRegressor`).
        Multiclass (K>=3) graduation is unavailable; explicit ``True`` raises.
    graduation_alpha : float or None, default=None
        Fixed smoothing strength on binary fits; unused for multiclass.
    graduation_high_order_alpha : float, default=0.0
        Opt-in strength in ``[0, 1]`` for one reference-weighted neighbour diffusion
        step on factored interactions of orders 3 through 8. Requires ``prune=True``
        and graduation enabled; unsupported for monotone or multiclass fits. Numeric
        axes use finite cell order; categorical axes and missing-to-finite edges are
        excluded. A missing slice may still change along other numeric axes.
        Preserves product-reference zero marginal means and recomputes variances.
        This is a first-difference smoother, separate from dense WH/GCV graduation;
        its strength is fixed, not selected by GCV. Both steps form one candidate,
        adopted whenever the native re-anchor accepts it. Effects exceeding
        1,024 stored boxes per effect, 4,096 added boxes in total, or 2,000,000 added
        mask cells are skipped, with details in ``graduation_report_``. A configured
        ``prune_box_budget`` further limits total stored boxes after smoothing.
        Set ``graduation_alpha=0`` to apply only higher-order smoothing.
    leaf_refine_steps : int, default=4
        Extra per-tree leaf Newton refinement steps applied after a tree's structure is
        fixed. Binary (K=2) path only — the multiclass (K>=3) native-softmax path does not
        yet call this refinement pass. The shipped default (``4``) is already non-zero, so
        raising here would break ordinary default-configured multiclass fits; a multiclass
        fit instead warns once per estimator instance whenever this is nonzero — including
        at its own default — rather than raising (see the class-level honesty note).
    leaf_refine_backtracks : int, default=4
        Maximum backtracking attempts per leaf-refinement step (halving the step on a
        non-improving move). Binary (K=2) path only, for the same reason as
        ``leaf_refine_steps`` — and warned about (not raised) alongside it, for the same
        non-inert-default reason.
    refine_closed_form_tier2 : bool, default=True
        Tier-2 log-link closed-form leaf refinement. Not applicable to the logistic
        objective; kept for constructor symmetry with :class:`TBoostRegressor`.
    incremental_mu : bool, default=False
        Incremental ``mu = exp(F)`` cache for log-link fits. Not applicable to the logistic
        objective; kept for constructor symmetry with :class:`TBoostRegressor`.
    mvs_min_rows : int, default=1
        Floor on the sampled row count when ``subsample`` enables minimal-variance-style
        (MVS) row sampling. Has no effect when ``subsample`` is ``None``, and no effect at
        all on the multiclass (K>=3) path, which does not row-sample: setting this away from
        its default (``1``) independently raises ``ValueError`` at multiclass fit time (see
        ``subsample``).
    hist_precision : {"full", "quantized"} or None, default=None
        Histogram accumulator precision. ``None``/``"full"`` uses the exact full-precision
        accumulator; ``"quantized"`` uses a faster quantized-integer accumulator.
    n_bags : int, default=8
        Number of independent bags trained and averaged (exact ensemble averaging over
        fANOVA banks). Replicated benchmarking found row-bag plus column diversity to be the
        two levers that most move stacking accuracy, at a cost of roughly ``n_bags`` times
        the train/predict time of a single fit — accuracy is the priority of the shipped
        default. ``n_bags=1`` disables bagging. Binary (K=2) path only: the multiclass
        (K>=3) native-softmax path fits a single unbagged model regardless of this setting —
        every multiclass fit costs roughly what a single-bag binary fit would, not ``n_bags``
        times that. The shipped default (``8``) is already non-inert, so a multiclass fit
        warns once per estimator instance whenever this is greater than ``1`` — including at
        its own default — rather than raising (see the class-level honesty note); pass
        ``n_bags=1`` to silence the warning.
    bag_subsample : float, default=0.8
        Per-bag row-sampling fraction for outer bagging. ``0.8`` (the default) is subagging
        — sampling without replacement — rather than a ``1.0`` bootstrap: a bootstrap bag
        duplicates rows, which makes the per-bag early-stopping validation carve overlap the
        bag's own training multiset, a train/validation leak that lets validation deviance
        improve indefinitely and defeats early stopping. Set to ``1.0`` to restore classic
        bootstrap bagging. Only relevant when ``n_bags > 1``, which in turn only has an
        effect on the binary (K=2) path, and is warned about (not raised) on multiclass fits
        alongside ``n_bags`` (see ``n_bags``).
    cell_refit_base : float or None, default=None
        Base ridge penalty for the out-of-bag (OOB) fANOVA cell refit: after bagging, each
        cell coefficient in the averaged bank is re-fit toward its bags' OOB residual under a
        ridge penalty shaped by ``cell_refit_gamma``, then re-purified. ``None`` (the
        default) disables it. Requires real bagging (an OOB partition, i.e. ``n_bags >= 1``)
        — setting it without bagging raises ``ValueError`` rather than silently no-opping.
        Held-out no-harm guarded: adopted only if it improves held-out deviance. Honored on
        the multiclass (K>=3) path too since 2026-08-25: there the softmax Hessian block
        ``diag(p) - p p^T`` is replaced by its diagonal for the solve (K independent cell
        solves, one per class logit), and the resulting K-class step is then accepted, shrunk
        or declined by ONE joint backtrack on the true multinomial deviance of the held-out
        out-of-bag slice — never K independent per-class guards. ``lambda = 0`` drops the whole
        K-class correction.
    cell_refit_gamma : float, default=2.0
        Adaptive exponent for the OOB cell refit's penalty (``0`` = flat ridge across all
        cells; larger values penalize high-signal cells less). Only relevant when
        ``cell_refit_base`` is set.
    ridge_refit_l2 : float or None, default=None
        L2 penalty for a fully-corrective ridge leaf refit: after stagewise tree growth, all
        leaf values are jointly re-solved by regularized IRLS over the frozen tree structure.
        ``None`` (the default) disables it; a float enables it at that penalty strength.
        Binary (K=2) path only — the multiclass (K>=3) path does not apply this refit:
        setting this away from its default (``None``) independently raises ``ValueError`` at
        multiclass fit time (see the class-level honesty note).
    ridge_refit_max_iter : int, default=5
        Maximum IRLS passes for the fully-corrective ridge leaf refit. Only relevant when
        ``ridge_refit_l2`` is set, which in turn only has an effect on the binary (K=2)
        path; setting this away from its default (``5``) independently raises
        ``ValueError`` at multiclass fit time.
    nesterov : bool, default=False
        Reserved for AGBM-style Nesterov look-ahead acceleration. **Currently unsupported**:
        setting this ``True`` always raises ``ValueError`` at fit time, for both the binary
        and multiclass paths (the momentum-correction step needed for stable convergence is
        not yet implemented) rather than silently producing a diverging model. Leave at the
        default.
    dart_drop_rate : float or None, default=None
        Enables DART (Dropout Additive Regression Trees) when set: on each round, previously
        grown trees are dropped with this probability before the new tree is fit, and DART's
        standard alpha normalization is folded into the tree weights. ``None`` (the default)
        disables DART. Must be in ``[0, 1)``. Mutually exclusive with ``nesterov``. Binary
        (K=2) path only — the multiclass (K>=3) path does not apply DART: setting this away
        from its default (``None``) raises ``ValueError`` at multiclass fit time (see the
        class-level honesty note).
    random_strength : float, default=0.0
        Deterministic split-score noise strength, seeded by ``seed`` so it stays
        reproducible; perturbs split-gain ranking to diversify tree structure. ``0.0`` (the
        default) is inert. Applies to both the binary and multiclass paths.
    reanchor : bool or None, default=None
        Whether to re-solve the model's global intercept exactly against the training data
        after fitting, removing post-shrinkage aggregate bias. ``None`` uses a link-aware
        default which resolves to off for the logistic objective (the link-aware "on" case
        only applies to the log-link regression objectives). An explicit ``True``/``False``
        always overrides the default on the binary (K=2) path; the multiclass (K>=3) path
        does not apply intercept re-anchoring at all regardless of this setting — an
        explicit ``True``/``False`` there raises ``ValueError`` rather than silently
        no-opping (see the class-level honesty note).
    reanchor_slope : bool or None, default=None
        Whether to fit an honest affine link-scale recalibration (``F' = b + a * F``) on the
        fit's own internal validation holdout after early stopping, correcting the uniform
        score-scale compression that shrinkage and bulk-dominated early stopping leave
        behind, without touching ranking or the fANOVA decomposition. Evidence-gated: the
        fit is ridge-shrunk toward the identity and only applied when its holdout deviance
        gain clears the prior penalty, so thin or uninformative holdouts stay at
        (near-)identity — this can only help or be neutral, never hurt. Inert without a
        validation holdout (i.e. when ``validation_fraction`` is ``None``). ``None`` follows
        the same link-aware default as ``reanchor`` (off for the logistic objective); an
        explicit ``True``/``False`` always wins on the binary (K=2) path. Like ``reanchor``,
        the multiclass (K>=3) path does not apply this recalibration regardless of the
        setting — an explicit ``True``/``False`` there raises ``ValueError`` rather than
        silently no-opping.
    max_interaction_order : int, default=3
        Whole-tree cap on the number of distinct raw features a tree may use (1 to 4) —
        caps interaction ORDER, which is the number of axes an exported table has. It does
        not cap tree depth; see ``max_depth`` for that. Must be ``<= max_depth`` (a tree
        needs one level per distinct feature), and values above 3 are opt-in.

        Order 4 is not an exactness relaxation. The fANOVA purification cascade is
        n-dimensional, so a 4-way effect is centred, shed and reconstructed by exactly the
        same algebra a 3-way is, and all five decomposability checks pass unchanged. It is a
        READABILITY relaxation, priced in three places: the interaction-gain hurdle doubles
        again at the 3->4 transition, ``table_budget_order_shrink`` halves the table-size
        prior's allowance, and the evidence-gated prune drops any 4-way the held-out folds
        cannot pay for. Measured: on a target whose highest true interaction is a pair,
        raising the cap from 3 to 4 leaves the deployed table set and the fit IDENTICAL.

        **Prefer ``max_depth == max_interaction_order`` at order 4.** A 4-way effect is
        exported as a set of rank-1 region boxes (never a dense cube — see
        ``table_budget_cells``), and at ``max_depth == 4`` each tree contributes exactly one,
        so the effect stays a short readable stack of case expressions (~110 boxes measured
        on a real portfolio). At ``max_depth == 6`` the same tree contributes up to 36
        region boxes and they do not merge across trees, which measured ~5,600 boxes for a
        single effect — still exact, no longer a filing artifact.
    table_budget_cells : int or None, default=None
        Soft cell budget for the split finder's table-size prior: a candidate support's
        ranking score is multiplied by ``(budget / max(budget, projected_cells)) ** 0.5``,
        where ``projected_cells`` is the product of the realized extents of its distinct
        raw features. A ranking steer, never a hard reject, and never a cap on an exported
        table — the hard firewall is separate. ``None`` resolves ``2_000_000`` (a memory
        budget that effectively never binds) at ``max_depth <= 3``, and ``4096`` above it:
        a lifted order-3 effect is exported as a dense cube, and a cube is cubic in the
        per-axis extent, so without the tighter budget the bank overflows its firewall.
        4096 lands at roughly 16 cells per axis for a 3-way table and 64x64 for a pair.
    max_depth : int, default=3
        Whole-tree cap on the number of split LEVELS (3 to 6). Orthogonal to
        ``max_interaction_order``: depth caps *resolution*, order caps *interaction*.

        Once a tree holds ``max_interaction_order`` distinct raw features, deeper levels
        can only REUSE a feature already on the tree with a refined threshold, which is a
        valid lower-order refinement — so a depth-6 tree on 3 features is still exactly a
        3rd-order fANOVA function. Tables get finer grids, never more axes, and the
        exported grid is hard-capped at ``max_bin`` cells per axis at any depth.

        What depth buys is estimation efficiency, not expressiveness: a depth-3 *ensemble*
        already spans the whole ≤3rd-order space. A deeper tree fits its finer partition
        jointly, in one Newton step under one shrinkage and one ``lambda_``, where the
        depth-3 ensemble reaches the same function only through a sequence of
        independently-shrunk 8-cell fits. Per-tree resolution is 8 / 12 / 18 / 27 realized
        cells at depth 3 / 4 / 5 / 6 (a 3.4x lift at depth 6, not 8x — most of the
        ``2**depth`` leaf array is unreachable). Depth 6 at order 1 is a genuinely new
        capability: a 7-step main effect fit in a single boosting round.

        **The default is 3 and should stay 3 unless a held-out probe says otherwise.**
        Deeper trees put ~3.4x fewer rows in each cell, and raising depth automatically
        raises the ``min_data_in_leaf`` floor to compensate (see that parameter). Not
        supported together with ``ridge_refit_l2``, which errors.
    seed : int, default=0
        Seed for every deterministic source of randomness in the fit (row/column sampling,
        bagging, DART dropout, ``random_strength`` noise). The same seed and inputs
        reproduce a bit-identical model.
    n_jobs : int or None, default=None
        Number of threads used for fitting and prediction, following the joblib/sklearn
        convention: ``None`` uses the native default (all cores), ``-1`` means all cores,
        and a negative ``n`` means ``cpu_count + 1 + n`` (so ``-2`` is all-but-one core).
    monotone_constraints : sequence, dict, or None, default=None
        Per-feature monotonicity constraints on the fitted function. ``None`` (the default)
        applies no constraints. Accepts either a length-``n_features`` positional sequence of
        ``{-1, 0, 1}`` (``1`` = increasing, ``-1`` = decreasing, ``0`` = unconstrained), or a
        dict keyed by feature name (matched against DataFrame column names, or the canonical
        ``f{i}``) or integer index, valued the same way.
    categorical_features : sequence, str, int, or None, default=None
        Which features to treat as categorical (target-statistic encoded, see the ``cat_*``
        parameters) rather than numeric. Accepts a single name/index, a sequence of
        names/indices, or a boolean mask the length of ``n_features``. For a polars ``X``,
        String/Categorical/Enum columns are treated as categorical automatically (they have
        no numeric cast) — declare a feature here only to force a NUMERIC-dtyped column onto
        the categorical path. ``None`` (the default) means dtype-detection only: every
        feature of a plain array (and every numeric polars column) is numeric.
    cat_smooth : float or None, default=None
        Categorical target-statistic shrinkage strength. ``None`` (the default) uses an
        empirical-Bayes credibility estimate fit per feature from the data; a float fixes the
        shrinkage to that pseudo-count ``m`` in ``(n_c * mean_c + m * base) / (n_c + m)``
        instead.
    cat_target : {"mean", "log_mean"} or None, default=None
        Target transform used before categorical target-statistic shrinkage. ``None``/
        ``"mean"`` (the default) encodes the exposure-weighted mean target; ``"log_mean"``
        encodes the weighted mean of ``log(y)``. Classification targets are 0/1-valued, so
        ``"mean"`` is the natural choice.
    cat_leakage : {"kfold", "ordered", "loo"} or None, default=None
        Leakage-avoidance scheme for training-time categorical encodings (serve-time
        encoding always uses the frozen full-data map regardless of this setting). ``None``/
        ``"kfold"`` (the default) uses ``cat_k``-fold cross-fit target statistics;
        ``"ordered"`` uses ``cat_n_perms`` seeded permutations of ordered target statistics;
        ``"loo"`` uses leave-one-out statistics (deterministic and knob-free, but not the
        default — it reintroduces a per-row dependence on the row's own target that measures
        worse than k-fold cross-fitting).
    cat_n_perms : int, default=1
        Number of seeded permutations for ordered target-statistic encoding. Only relevant
        when ``cat_leakage="ordered"``.
    cat_k : int, default=5
        Number of folds for k-fold cross-fit target-statistic encoding. Only relevant when
        ``cat_leakage`` is ``None``/``"kfold"`` (the default scheme).
    cat_min_data_per_group : float, default=10.0
        Weighted-count floor below which a categorical level collapses into a shared rare
        bucket before encoding.
    cat_direct_max_levels : int, default=16
        Low-cardinality bypass threshold: a categorical feature with 3 to this many distinct
        (post-rare-pooling) levels skips cross-fit encoding and shrinkage entirely — each
        level keeps its raw full-fit target statistic and gets its own histogram bin ordered
        by that statistic. Binary features are always exempt from this bypass regardless of
        the threshold. ``0`` disables the bypass for every feature.
    cat_channels : list of str, or None, default=None
        Which target-statistic channels to fit per native categorical feature. ``None`` (the
        default) fits only the mean-TS channel described above — byte-identical to every
        prior release. ``["mean", "count"]`` additionally fits a SECOND, target-free channel
        per categorical: a count/rarity statistic (``log1p`` of each level's exposure share)
        that lets the trees separate rare-but-informative levels from common-but-mild ones, a
        signal the mean channel alone collapses together. This can materially improve
        ``predict``/``score`` accuracy on high-cardinality categoricals with rare, informative
        levels. Table export (``tables()``, ``explain()``, ``pruning_report_``) FULLY supports
        multi-channel categoricals: the channels collapse LOSSLESSLY to exactly one relativity
        per raw categorical (never a column per channel), preserving the exact fANOVA
        decomposition — the pruned tables-only model reproduces the tree ensemble to <1e-6, and
        all five I2 exactness gates pass on multi-channel fits. The export is keyed by actual
        category level (Piece B): every categorical axis — single- or multi-channel — lists
        each level's label alongside the numeric borders/cell it lands in, so
        ``tables()``/``explain()`` read as e.g. ``Ford Focus 1.6 → 1.12`` rather than only the
        raw encoded value; levels collapsed by rare-pooling appear as one ``"<rare>"`` entry.
        ``["class_freq"]`` (multiclass K>=3 only) fits ONE channel per class instead of the
        mean channel: each carries that level's frequency of that class,
        ``E[1[y == k] | level]``, cross-fit and smoothed by the same machinery the mean
        channel uses. This replaces the K>=3 path's ORDINAL label-mean target statistic
        (``E[class_index | level]``, which treats nominal classes as though they were ordered
        — the documented multiclass categorical weakness) with the full per-level class
        distribution. List ``"mean"`` alongside it (``["mean", "class_freq"]``) to keep the
        label-mean channel as well. Per-class channels are emitted only for categoricals with
        at least 3 post-rare-pooling levels (below that the axis partition is
        encoding-invariant, so extra channels can add nothing) and cost one extra binned axis
        per class on those features. Replacement is decided PER FEATURE: a categorical the gate
        skips — and every categorical on a K=2 fit, where the channels are inert — keeps its
        mean channel, so no feature is ever left without a target statistic. For K=2 the request
        is therefore already satisfied by the default (the mean channel there IS the class-1
        frequency), and ``["count","class_freq"]`` reduces exactly to ``["mean","count"]``.
        Regression raises.
    cat_count_min_levels : int, default=20
        Cardinality gate for the count channel (only relevant when ``cat_channels`` includes
        ``"count"``): a categorical feature's count axis is fit only when its post-rare-pooling
        distinct level count is at least this many — the measured accuracy win came from
        high-cardinality categoricals specifically, so a low-cardinality one below the
        threshold behaves exactly as mean-only (bit-identical to leaving ``"count"`` off, for
        that feature). ``0`` disables the gate: every categorical gets the count channel
        whenever ``cat_channels`` requests it, regardless of cardinality.
    cat_class_freq_min_levels : int, default=3
        Cardinality gate for the per-class frequency channels (only relevant when
        ``cat_channels`` includes ``"class_freq"``, i.e. multiclass K>=3). A categorical gets
        its per-class axes only when its post-rare-pooling distinct level count is at least
        this many. The default ``3`` is the encoding-invariance floor — below it a categorical
        axis admits exactly one partition however its levels are ordered, so per-class channels
        can only add axes. Raise it (e.g. to ``20``, matching ``cat_count_min_levels``) to
        restrict the channels to high-cardinality features.
    prune : bool, default=True
        Whether to run post-fit table pruning: drop fANOVA tables that don't improve
        cross-validated held-out deviance, and deploy the smaller, more explainable
        tables-only model (see ``pruning_report_``). ON by default — every insur-arena
        benchmark cell deployed the pruned artifact, and the selection is held-out-guarded
        (neutral-or-better by construction). Pass ``False`` for the cheaper full-ensemble
        artifact. Available on both the binary and multiclass (K>=3) paths.
    prune_validation_fraction : float, default=0.15
        Train/select split fraction of the LEGACY multiclass (K>=3) prune
        (``multiclass_prune_cv=False``). Every other configuration selects on K-fold CV sized
        by ``prune_n_folds`` — a non-default value there raises ``ValueError`` rather than
        silently no-opping.
    multiclass_prune_cv : bool, default=True
        Multiclass (K>=3) prune selection regime. ``True`` (the default since 2026-09-07) is
        the single-output prune's design: ``prune_n_folds`` fold refits scored on their
        held-out fold, voted with ``prune_min_stability``, gated by ``prune_drop_z`` and
        ``prune_keep_budget``, then measured by ``multiclass_prune_guard`` with an SE-aware bar.
        ``False`` is the historical single train/select split walk, which was measured to drop
        tables on selection noise and to lose 0.003-0.006 skill on the InsurArena multiclass
        panels.
    multiclass_prune_guard_floor : float, default=0.002
        Floor of the multiclass (K>=3) prune guard's bar, as a fraction of the deploy model's
        honest deviance improvement over the class prior. The guard re-admits tables until the
        pruned model is within ``max(prune_guard_z_dn * SE, floor * (dev_null - dev_full))`` of
        the full bank on honest rows; the floor is the price a simpler model is allowed to pay.
        ``0.0`` keeps the SE term only.
    prune_se_rule : float, default=0.0
        Pruning selection rule, in standard errors of the cross-validated deviance estimate.
        ``0.0`` (the default) selects the held-out-deviance minimum, which improves or is
        neutral versus the unpruned model on every benchmark tried. A larger value (e.g.
        ``0.5`` or the classic ``1.0`` one-standard-error rule) prunes more aggressively for
        parsimony but monotonically degrades the deviance metric — a deliberate
        explainability/accuracy trade-off, opt in explicitly for smaller models.

        **MULTICLASS (K>=3) ONLY.** That path selects on a single train/select split, so the
        band decides the deployed keep-set directly. The regressor/binary path instead
        AGGREGATES per-fold selections, and the band reaches that aggregation only through
        ``kept_rate``, which the keep rule ORs against ``positive_rate`` — measured
        bit-identical banks for ``prune_se_rule`` in ``{0, 0.5, 1, 2, 10, 100}`` at four
        dataset sizes. A non-default value therefore raises ``ValueError`` there rather than
        silently no-opping; use ``prune_table_budget`` / ``prune_box_budget`` for a smaller
        bank on that path. Only has an effect together with ``prune=True``.
    prune_n_folds : int, default=5
        Number of cross-validation folds used to estimate each candidate table's
        contribution during pruning on the binary (K=2) path. Unused on the multiclass path
        (see ``prune_validation_fraction``).
    prune_drop_z : float or None, default=2.0
        Evidence bar for DROPPING a table (av37). With paired per-fold CV evidence in hand, a
        candidate interaction table is dropped only when its mean drop-gain clears this many
        standard errors of that mean over the prune folds; no evidence means keep. ``None``
        restores the pre-av37 rule, which kept a table on the SIGN of its fold-mean and so
        dropped on evidence that could not resolve the question (measured: 11.0 +/- 1.9 of 14
        tables kept across 90 fits of identical data, seed the only difference). The gate is
        monotone — it can only ever keep MORE tables than the old rule, never fewer. Tables
        scored by fewer than two folds keep the old verdict: absence from a fold's bank is a
        structural fact, not an ambiguous measurement. Only has an effect with ``prune=True``.
    prune_keep_budget : int, default=32
        Explainability budget for ``prune_drop_z`` admissions. Ambiguous tables are ranked by
        mean gain and admitted only while the pre-cascade keep-set stays within
        ``max(prune_keep_budget, <the pre-av37 keep-set size>)``. A bank that was already
        larger than the budget is therefore left exactly as the old rule chose it, while a
        small candidate bank — the regime where the fold evidence cannot resolve anything —
        comes back whole.
    prune_box_budget : int, default=0
        DEPLOYED-BOX budget: the total rank-1 region boxes the deployed bank may carry. ``0``
        disables it, and the fit is then bit-identical to one without the parameter.

        ``prune_keep_budget`` counts TABLES, which is the right unit for a dense bank and the
        wrong one for a factored one: a factored effect (every order-3 and order-4 effect)
        deploys one box per realized tree-region and the rating export emits one row per box.
        A depth-3 tree contributes ONE box; a depth-6 tree contributes up to ``k0*k1*k2``, so
        lifting ``max_depth`` barely moves the table count and multiplies the boxes instead
        (measured: fremotor_payfreq split 0 goes 7,358 -> 165,809 boxes for 101 -> 145 effects).

        The budget is spent on the best held-out drop-gain PER BOX first — ties broken on
        purified variance per box, then on feature ids — and everything it cannot afford is
        dropped binarily, with the heredity cascade re-run so nothing deploys over a dropped
        subset. Dense tables cost zero boxes and are never touched. A budget at or above the
        bank's own box total is a verbatim no-op. Requires ``prune=True`` (raises otherwise).
    prune_lambda_boxes : float, default=0.0
        SELECTION-time price of one deployed box, in held-out-deviance units. ``0.0``
        disables it and the fit is bit-identical to one without the parameter.

        The selection-time analogue of ``prune_box_budget``, and the piece the av38 depth
        battery named as missing: selection could not see bank size, so it could not REFUSE a
        diffuse config. The backward prune walk optimizes held-out deviance alone and the SE
        rule then breaks TIES toward fewer tables — but a config that buys a real gain at 22x
        the boxes is not a tie, so nothing declined it. With this armed the waypoint objective
        becomes ``mean_deviance + prune_lambda_boxes * n_boxes``, so a diffuse rung has to
        out-earn its own size. The budget still applies afterwards; the two compose.

        Units are deviance PER BOX on the dataset's own per-unit-weight scale, deliberately a
        price rather than a ratio. Calibrate from ``pruning_report_["path"]``, whose points now
        always carry ``n_boxes``: "one SE across the whole bank" is
        ``path[0]["se"] / path[0]["n_boxes"]``, and "1% of full deviance across the whole
        bank" is ``0.01 * path[0]["mean_deviance"] / path[0]["n_boxes"]``. Dense tables cost
        zero boxes and are never priced.
    prune_table_budget : int, default=0
        DEPLOYED MULTI-WAY TABLE budget: the most tables of arity ``>= prune_table_min_arity``
        the deployed bank may carry. ``0`` disables it, and the fit is then bit-identical to
        one without the parameter.

        This is Ralph's 2026-08-27 explainability bar written as a number: *cells within a
        table are free, the COUNT of multi-way tables is what must stay small*.
        ``prune_box_budget`` prices the OTHER quantity — resolution — and bounds table count
        only as a side effect of dropping supports binarily, so it buys its table discipline by
        starving the tables it keeps. This caps the count directly and every survivor keeps
        every box it had.

        Spent on the best held-out drop-gain first (ties on purified variance, then feature
        ids); everything past the cap is dropped binarily with the heredity cascade re-run.
        Because every priced support costs exactly one, the surviving set is MONOTONE in the
        cap — unlike the box budget, where a larger budget can swallow a big effect and squeeze
        out a small one. Supports below the arity floor are never counted and never dropped: a
        main effect or a pair is what a filing reads. A cap at or above the bank's own count is
        a verbatim no-op. Requires ``prune=True`` (raises otherwise).

        Measured on fremotor_payfreq (4-class, budget-300 lane, 2 paired splits) against the
        box budget at a matched >=3-way count: the table budget returns ~15% more of the
        depth-6 gain, paid for in cells the bar now grants.

        For a K-class fit the cap counts SUPPORTS in the shared keep-set — one entry however
        many classes carry a copy — while an arity census that sums per-class copies reads
        about K times higher. Both numbers describe the same bank.
    prune_lambda_tables : float, default=0.0
        SELECTION-time price of one kept table of arity ``>= prune_table_min_arity``, in
        held-out-deviance units. ``0.0`` disables it and the fit is bit-identical to one
        without the parameter. The waypoint objective gains
        ``+ prune_lambda_tables * (kept tables at or above the floor)``.

        KNOWN LIMIT, and the reason this is not the knob to reach for if you want the bar met.
        Like ``prune_lambda_boxes`` it only re-picks a waypoint on an already-fixed
        deviance-greedy backward path; it cannot reorder the walk, so it cannot surrender a
        3-way table while holding a pair, and on the aggregated (regressor/binary) path a
        per-fold vote further damps it. Measured on a 9-feature depth-6 order-3 deploy fit with
        33 three-way tables: the price moves the count to 31, then 29, and SATURATES there
        across four more decades of lambda, while ``prune_table_budget`` lands on 8, 4 or 2
        exactly on request. Use this to let SELECTION see table count — the gap the av38
        battery named — not to hit a number.
    prune_table_min_arity : int, default=3
        Lowest interaction order that ``prune_table_budget`` and ``prune_lambda_tables`` count.
        Consulted only when one of them is armed, so its value cannot perturb a default fit.
        ``3`` is the product bar: mains and pairs are what a filing reads, three-way tables are
        what inflates it. Must be in ``1..8``.
    prune_guard_z : float, default=0.0
        SE multiplier for the set-level no-harm guard's breach test: it breaches, and the
        re-admission ladder stops, on ``gap > max(prune_guard_tol, prune_guard_z * SE)`` where
        ``SE`` is the standard error of the relative deviance gap on the guard's own evidence
        rows (reported as ``gap_se``, with the resulting bar as ``tol_effective``). The shipped
        ``0.0`` is the fixed-relative-tolerance test. Measured and deliberately off: on a
        zero-inflated compound target the row-level SE is dominated by a few large claims and
        cannot resolve even a real set-level gap — see the ``_PRUNE_GUARD_Z_DEFAULT`` note.
    prune_guard_z_dn : float, default=2.0
        SE multiplier for the DOWNWARD recalibration of the same bar (av39). The effective
        tolerance becomes ``min(tol_up, max(prune_guard_tol_floor, prune_guard_z_dn * SE))``,
        where ``tol_up`` is the ``prune_guard_z`` result above — so where the guard's jury is
        sharp the bar tightens to a few SE instead of a fixed 5%, and where it is blunt the
        ``min`` pins the bar back at ``prune_guard_tol`` and the fit is bit-identical. This knob
        can therefore only ever make the guard fire MORE, never less. It governs BOTH the breach
        trigger and the ladder's stopping rule. ``0.0`` restores the pre-av39 guard exactly.
    prune_guard_tol_floor : float, default=0.005
        Lower clamp on the downward recalibration, so an arbitrarily sharp jury cannot drive the
        bar to zero and re-admit the whole bank on immeasurable noise. Ignored when
        ``prune_guard_z_dn = 0``.
    prune_min_stability : float, default=0.5
        Fold-stability bar for keeping a table: the aggregator keeps one when
        ``mean_gain > prune_min_mean_gain`` **and** (it survived at least this fraction of the
        prune folds' own selections, **or** scored a positive drop-gain in at least this
        fraction of them). In plain terms: *in how many folds must this table show signal.*
        Previously a hardcoded ``0.5`` — this is the knob that actually governs the deployed
        bank's size on the regressor/binary path, where ``prune_se_rule`` cannot reach.

        The fold rates quantise to ``1/prune_n_folds``, so at the default 5 folds this is a
        3-position dial: ``(0.4, 0.6]`` = 3/5, ``(0.6, 0.8]`` = 4/5, ``(0.8, 1.0]`` = 5/5.
        Measured on French MTPL frequency (3 seeds, ``n_bags=8``): ``1.0`` takes the deployed
        bank from 58.3 to 37.0 tables and 1216 to 389 boxes for −0.49% ± 0.66 of test D², the
        best explainability-per-accuracy-point of any pruning knob, and it halves the
        seed-to-seed spread in bank size. The default stays ``0.5`` pending a multi-dataset
        battery. Multiclass (K>=3) has no fold vote to threshold, so a non-default value raises
        there. Only has an effect together with ``prune=True``.
    prune_min_mean_gain : float, default=0.0
        Minimum mean held-out drop-gain a table must earn to be kept, in deviance units on the
        dataset's own per-unit-weight scale. ``0.0`` (the default) is the historical
        ``mean_gain > 0`` test, bit-for-bit — under it a table kept on a mean gain of ``1e-9``
        is kept on noise. This is the threshold-parameterised counterpart to
        ``prune_table_budget``'s count cap: drop everything whose evidence is weaker than a
        stated bar, rather than keeping the best-ranked N. The native pruning report has always
        carried a ``min_mean_gain`` field and always written a literal ``0.0``; it now reports
        the threshold actually used. Multiclass (K>=3) raises on a non-default value, as for
        ``prune_min_stability``. Only has an effect together with ``prune=True``.

        **It interacts with ``prune_drop_z``, and not in the obvious direction.** The floor
        applies to the LEGACY keep rule only. With the evidence gate armed (the default
        ``prune_drop_z=2.0``), a table the floor rejects does not simply leave — it becomes an
        *ambiguous candidate* for the gate, whose own rule is "keep unless the fold evidence is
        significantly negative", so it is very likely re-admitted. Raising the floor can
        therefore GROW the deployed bank. Measured on an 8-feature synthetic Poisson fit: with
        the gate on, ``0.0 -> 1e-3`` takes ``legacy_selected`` 18 -> 9 but ``kept`` 26 -> 28;
        with ``prune_drop_z=None`` the same sweep takes ``kept`` 13 -> 9, monotone as intended.
        Pair this with ``prune_drop_z=None`` when using it as a parsimony bar.

        Where the gate admits nothing it needs no such pairing: on French MTPL frequency
        (single-bag, seed 0) the gate re-admits 0 of 27 candidates, and ``0 -> 1e-5 -> 1e-4``
        takes the deployed bank 54 -> 46 -> 31 tables, with ``1e-5`` landing 15% fewer tables at
        a slightly BETTER test deviance (0.55911 vs 0.55915) than the default.
    prune_fold_min_rows : int, default=125
        Rows-per-fold floor that sizes the prune CV: the fold count is
        ``max(2, min(prune_n_folds, n // prune_fold_min_rows, n))``, so on small data the number
        of folds adapts DOWN rather than producing folds too thin to score a table. Raising it
        makes that adaptation kick in earlier. Only has an effect together with ``prune=True``.
    prune_fold_es_patience : int or None, default=None
        Early-stopping patience for the prune-CV FOLD fits only; the deploy fit always keeps the
        estimator's own ``early_stopping_rounds``. ``None`` resolves to the shipped ``250``; an
        explicit integer overrides it. The fold fits exist only to vote on table survival, which is why
        their patience is tighter than the deploy fit's — 250 left every studied keep-set
        bit-identical with the MTPL CV scan 19.4% faster. Only has an effect together with
        ``prune=True``.
    prune_guard_min_rows : int, default=500
        Minimum honest evidence rows before the set-level no-harm guard will judge at all —
        below this the pruned-vs-full deviance ratio is noise and the guard reports itself
        skipped rather than acting on it. Counts out-of-bag-covered rows on the OOB path and
        shared-holdout rows on the grouped carve path. Only has an effect together with
        ``prune=True`` and ``prune_guard=True``.
    prune_slope_eps : float, default=0.01
        Dead band on the post-prune slope re-anchor: the correction is not applied unless the
        out-of-bag scale ``b`` differs from 1 by more than this, so a bank that was not
        materially compressed keeps a byte-identical artifact. See ``pruning_report_["slope"]``.
    prune_slope_min_z : float, default=3.0
        Significance bar for that same correction: ``b`` must also be off by at least this many
        standard errors (``z_b``) before it is applied. Together with ``prune_slope_eps`` this
        is the "materially AND significantly off" test — lowering either makes the re-anchor
        fire more readily. See ``pruning_report_["slope"]``.
    prune_refit_full : bool, default=False
        Deprecated and without effect on any path: contribution-stability pruning always
        fits the deployed model on all rows and applies the keep-set to that same fit, so
        there is no separate subset-pruned fit to conditionally refit. Setting this ``True``
        raises ``ValueError`` at fit time rather than silently ignoring it. Kept only for
        backward constructor compatibility (``get_params``/``set_params``/``clone`` still
        round-trip it).
    prune_rebalance : bool, default=True
        Whether to re-solve the surviving tables' cell values to the deviance-optimum of the
        reduced structure after pruning drops tables ("rebalance on means": a single IRLS
        ridge cell-refit step, re-purified to preserve the 1-/2-/3-way order separation).
        Held-out no-harm guarded: adopted only if it lowers held-out deviance. Set ``False``
        to deploy the drop-only pruned model. Binary (K=2) path only. Only has an effect
        together with ``prune=True``.
    ref_measure : {"exposure", "product_marginals", "uniform"} or None, default=None
        The reference measure the fitted tables are purified against, i.e. the weighting
        that decides which table owns the part of the score that could sit in more than one
        place. ``None`` resolves to ``"exposure"`` (2026-09-06): each axis is weighted by
        its exposure-weighted empirical marginal (``sample_weight * exposure`` per row) plus
        a small positivity floor (``measure_floor``), so the prune, the per-bag banks and
        the export all read the fit-time effective mass. ``"product_marginals"`` is the
        pre-2026-09-06 measure: a flat ROW count blended half-and-half with a uniform
        component (``laplace=1``), which gave empty and thin cells half the weight on every
        axis; it is kept so earlier boards stay byte-reproducible. ``"uniform"`` weights
        every cell equally. Changing the measure changes which tables the prune keeps and
        therefore the deployed model, not just the ledger; use ``tables(ref_measure=...)``
        to re-express a fitted model under another measure without touching predictions.
    measure_floor : float, default=1e-3
        Under ``ref_measure="exposure"``: the total floor mass, as a fraction of the
        empirical mass, spread evenly over each axis's cells so every weight stays
        strictly positive. Must be finite and > 0. Inert under the other measures.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``, sorted (``numpy.unique`` order). ``predict``
        indexes back into this array.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``. Only set when ``X`` was a DataFrame-like
        object exposing column names — check with
        ``getattr(est, "feature_names_in_", None)``, per the sklearn convention, rather than
        assuming it is always present.
    pruning_report_ : dict
        Diagnostic report from post-fit table pruning: which fANOVA effects were scanned,
        kept, and deployed, and their cross-validated contributions. Only set when
        ``prune=True``.

        ``["guard"]`` records the set-level no-harm guard, and ``["slope"]`` the post-prune
        slope re-anchor: the deployed bank's out-of-bag scale ``b`` relative to the full
        bank's (``b_keep``/``b_full``), its standard error ``se_b`` and z-statistic ``z_b``,
        the ``sd_ratio`` measuring how much link-scale spread the prune removed, and
        ``applied`` — with ``skipped`` naming the gate when it did not fire. Present only on
        a log-link fit whose guard ran on out-of-bag evidence; the correction is applied only
        when the scale is both materially and *significantly* off, so a fit that does not
        need it keeps a byte-identical artifact.

        The av37 evidence gate adds ``["drop_z"]`` / ``["keep_budget"]`` (the knobs in force),
        ``["legacy_selected"]`` (what the pre-av37 rule would have kept), ``["evidence_budget"]``
        = ``max(keep_budget, legacy_selected)``, ``["evidence_candidates"]`` (tables the gate
        wanted that the old rule refused), ``["evidence_admitted"]`` (those it could afford, in
        rank order), ``["evidence_budget_bound"]`` (whether the budget truncated them) and
        ``["evidence_scores"]`` (per table: fold count, mean gain, SE and t). ``drop_z: null``
        means the gate was off and every other field is inert. ``["guard"]`` likewise gains
        ``guard_z``, ``gap_se`` (the SE of the relative deviance gap on the evidence rows) and
        ``tol_effective`` — the bar actually used for BOTH the breach trigger and the
        ladder's stopping rule — plus the av39 ``guard_z_dn`` / ``tol_floor`` knobs and
        ``n_chunks`` (rungs the ladder had available against ``steps``, the rungs climbed).
        ``tol_effective`` = ``min(max(prune_guard_tol, guard_z*gap_se),
        max(tol_floor, guard_z_dn*gap_se))``, each term skipped when its z is 0.

        ``["box_budget"]`` appears only when ``prune_box_budget > 0``: ``max_boxes`` (the
        budget), ``boxes_before``/``boxes_after`` and ``effects_before``/``effects_after`` (the
        deployed rank-1 box and factored-effect counts either side of it), ``engaged`` (false
        when the bank already fit, in which case the model is bit-identical to an unbudgeted
        one), ``dropped`` (the supports the greedy could not afford, best evidence-per-box
        first, so the tail is the weakest) and ``cascade_dropped`` (those removed after, each
        properly containing a dropped support).
    graduation_report_ : list of dict
        One entry per eligible table on a BINARY (K=2) fit -- the binary bank graduates
        through the same path as the regressor's (see ``graduate``). Not set for
        multiclass (K>=3): there is no Whittaker-Henderson path for the per-class banks,
        and an explicit ``graduate=True`` raises there rather than no-opping.

    Examples
    --------
    >>> import numpy as np
    >>> from t_boost import TBoostClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3)).astype(np.float32)
    >>> y = (X[:, 0] + 0.5 * X[:, 1] * X[:, 2] > 0).astype(int)
    >>> est = TBoostClassifier(n_trees=100, validation_fraction=None, n_bags=1)
    >>> est.fit(X, y).predict(X).shape
    (200,)
    """

    @classmethod
    def from_bytes(cls, data: bytes) -> "TBoostClassifier":
        """Deserialize a fitted estimator from the wire format written by :meth:`to_bytes`.

        Parameters
        ----------
        data : bytes
            A blob previously produced by :meth:`to_bytes` (either this or an older
            release — the envelope is versioned and back-compatible).

        Returns
        -------
        TBoostClassifier
            A new, already-fitted estimator instance.
        """
        est = cls()
        # Optional TBP1 envelope carries the categorical layout; inner is a binary Model blob or a
        # multiclass container (TBMC magic). Numeric models have no envelope (back-compat).
        md, inner = _unpack_bytes(data)
        if md.get("kind") == "multi":
            est._attach_multiclass_model(_MultiClassModel.from_bytes(inner))
        elif md.get("kind") == "multi_tables":
            est._attach_multiclass_model(_MultiClassTableModel.from_bytes(inner))
        elif inner[:4] == _TABLES_MAGIC:
            est._attach_classifier_model(_TableModel.from_bytes(inner))
        else:
            est._attach_classifier_model(_Model.from_bytes(inner))
        _restore_metadata(est, md)
        return est

    @classmethod
    def from_json(cls, data: str) -> "TBoostClassifier":
        """Deserialize a fitted estimator from the wire format written by :meth:`to_json`.

        Parameters
        ----------
        data : str
            A JSON document previously produced by :meth:`to_json`.

        Returns
        -------
        TBoostClassifier
            A new, already-fitted estimator instance.
        """
        est = cls()
        md, inner = _unpack_json(data)
        if md.get("kind") == "multi":
            est._attach_multiclass_model(_MultiClassModel.from_json(inner))
        elif md.get("kind") == "multi_tables":
            est._attach_multiclass_model(_MultiClassTableModel.from_json(inner))
        elif '"t-boost-tables"' in inner:
            est._attach_classifier_model(_TableModel.from_json(inner))
        else:
            est._attach_classifier_model(_Model.from_json(inner))
        _restore_metadata(est, md)
        return est

    # Defaults mirror `_BaseTBoost.__init__` (the recommended recipe); see the note there.
    def __init__(
        self,
        n_trees: int = 4000,
        learning_rate: float = 0.05,
        lambda_: float = 1.0,
        # --- EVERYTHING BELOW IS KEYWORD-ONLY (2026-09-04) -------------------------------------
        # `n_trees`, `learning_rate` and `lambda_` keep their positions — they are the only three
        # anyone plausibly passes positionally. The remaining ~86 become keyword-only, which makes
        # any FUTURE removal or reordering safe: in a positional signature, deleting a parameter
        # silently re-aims every argument after it onto the wrong name, and on an 89-parameter
        # constructor that is a defect waiting to happen.
        #
        # Verified free before landing: there is no positional construction of either estimator
        # anywhere in this repo or in insur-arena — the sole programmatic path is
        # `recommended_recipe`, already keyword-only past `objective`.
        #
        # And it is safe for sklearn: both real sklearn and the `_compat` shim build
        # `_get_param_names` by excluding VAR_POSITIONAL and VAR_KEYWORD only, so KEYWORD_ONLY
        # parameters survive `get_params` / `set_params` / `clone` intact. A `**kwargs` tail would
        # NOT — it vanishes from `get_params`, so `clone` would silently drop it. That is why the
        # long tail is moved behind `*` rather than into `**kwargs`.
        *,
        lambda_scale_invariant: bool = False,
        l1_leaf: float = 0.0,
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_delta_step_gated: Any = None,
        max_bin: int = 254,
        objective: str = "logistic",
        tweedie_rho: float = 1.5,
        min_data_in_leaf: int | None = None,
        min_sum_hessian_in_leaf: float = 0.0,
        min_weight_sum_in_leaf: float = 0.0,
        path_smooth: float | None = None,
        subsample: float | None = None,
        colsample_bytree: float = 0.8,
        learning_rate_decay: float = 0.0,
        validation_fraction: float | None = 0.1,
        early_stopping_rounds: int = 500,
        early_stopping_adaptive: float | None = _ES_ADAPTIVE_DEFAULT,
        early_stopping_min_delta: float = _ES_MIN_DELTA_DEFAULT,
        interaction_gain_hurdle: float = _GAIN_HURDLE_DEFAULT,
        interaction_gain_hurdle_mode: str = _GAIN_HURDLE_MODE_DEFAULT,
        graduate: bool | None = None,
        graduation_alpha: float | None = None,
        graduation_high_order_alpha: float = 0.0,
        leaf_refine_steps: int = 4,
        leaf_refine_backtracks: int = 4,
        refine_closed_form_tier2: bool = True,
        incremental_mu: bool = False,
        mvs_min_rows: int = 1,
        hist_precision: str | None = None,
        n_bags: int = 8,
        bag_subsample: float = 0.8,
        cell_refit_base: float | None = None,
        cell_refit_gamma: float = 2.0,
        ridge_refit_l2: float | None = None,
        ridge_refit_max_iter: int = 5,
        nesterov: bool = False,
        dart_drop_rate: float | None = None,
        random_strength: float = 0.0,
        reanchor: bool | None = None,
        reanchor_slope: bool | None = None,
        max_interaction_order: int = 3,
        max_depth: int = 3,
        table_budget_cells: int | None = None,
        table_budget_order_shrink: float = _TABLE_BUDGET_ORDER_SHRINK_DEFAULT,
        seed: int = 0,
        n_jobs: int | None = None,
        monotone_constraints: Any = None,
        categorical_features: Any = None,
        cat_smooth: float | None = None,
        cat_target: str | None = None,
        cat_leakage: str | None = None,
        cat_n_perms: int = 1,
        cat_k: int = 5,
        cat_min_data_per_group: float = 10.0,
        cat_direct_max_levels: int = 16,
        cat_channels: list[str] | None = None,
        cat_count_min_levels: int = 20,
        cat_class_freq_min_levels: int = 3,
        prune: bool = _PRUNE_DEFAULT,
        prune_validation_fraction: float = 0.15,
        prune_se_rule: float = 0.0,
        prune_n_folds: int = 5,
        prune_refit_full: bool = False,
        prune_rebalance: bool = True,
        ref_measure: str | None = None,
        measure_floor: float = _MEASURE_FLOOR_DEFAULT,
        prune_guard: bool = True,
        prune_guard_tol: float = 0.05,
        prune_drop_z: float | None = _PRUNE_DROP_Z_DEFAULT,
        prune_keep_budget: int = _PRUNE_KEEP_BUDGET_DEFAULT,
        prune_fold_fidelity: bool = _PRUNE_FOLD_FIDELITY_DEFAULT,
        prune_guard_z: float = _PRUNE_GUARD_Z_DEFAULT,
        prune_guard_z_dn: float = _PRUNE_GUARD_Z_DN_DEFAULT,
        prune_guard_tol_floor: float = _PRUNE_GUARD_TOL_FLOOR_DEFAULT,
        multiclass_prune_guard: bool = True,
        multiclass_prune_sel_bags: int = 1,
        multiclass_prune_cv: bool = True,
        multiclass_prune_guard_floor: float = _MULTICLASS_GUARD_FLOOR_DEFAULT,
        prune_box_budget: int = _PRUNE_BOX_BUDGET_DEFAULT,
        prune_lambda_boxes: float = _PRUNE_LAMBDA_BOXES_DEFAULT,
        prune_table_budget: int = _PRUNE_TABLE_BUDGET_DEFAULT,
        prune_lambda_tables: float = _PRUNE_LAMBDA_TABLES_DEFAULT,
        prune_table_min_arity: int = _PRUNE_TABLE_MIN_ARITY_DEFAULT,
        prune_min_stability: float = _PRUNE_MIN_STABILITY_DEFAULT,
        prune_min_mean_gain: float = _PRUNE_MIN_MEAN_GAIN_DEFAULT,
        prune_selector: str = _PRUNE_SELECTOR_DEFAULT,
        prune_path_steps: int = _PRUNE_PATH_STEPS_DEFAULT,
        prune_path_fraction: float = _PRUNE_PATH_FRACTION_DEFAULT,
        prune_path_tolerance: float = _PRUNE_PATH_TOLERANCE_DEFAULT,
        band_tolerance: float | None = _BAND_TOLERANCE_DEFAULT,
        band_deviance_cap: float = _BAND_DEVIANCE_CAP_DEFAULT,
        prune_fold_min_rows: int = _MIN_PRUNE_FOLD_ROWS,
        prune_fold_es_patience: int | None = None,
        prune_guard_min_rows: int = _PRUNE_GUARD_MIN_ROWS,
        prune_slope_eps: float = _SLOPE_EPS,
        prune_slope_min_z: float = _SLOPE_MIN_Z,
        prune_size_penalty: float | None = None,
        early_stopping: int | float | None = None,
    ) -> None:
        super().__init__(
            n_trees=n_trees,
            learning_rate=learning_rate,
            lambda_=lambda_,
            lambda_scale_invariant=lambda_scale_invariant,
            l1_leaf=l1_leaf,
            min_split_gain=min_split_gain,
            max_delta_step=max_delta_step,
            max_delta_step_gated=max_delta_step_gated,
            max_bin=max_bin,
            objective=objective,
            tweedie_rho=tweedie_rho,
            min_data_in_leaf=min_data_in_leaf,
            min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
            min_weight_sum_in_leaf=min_weight_sum_in_leaf,
            path_smooth=path_smooth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            learning_rate_decay=learning_rate_decay,
            validation_fraction=validation_fraction,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_adaptive=early_stopping_adaptive,
            prune_size_penalty=prune_size_penalty,
            early_stopping=early_stopping,
            early_stopping_min_delta=early_stopping_min_delta,
            interaction_gain_hurdle=interaction_gain_hurdle,
            interaction_gain_hurdle_mode=interaction_gain_hurdle_mode,
            graduate=graduate,
            graduation_alpha=graduation_alpha,
            graduation_high_order_alpha=graduation_high_order_alpha,
            leaf_refine_steps=leaf_refine_steps,
            leaf_refine_backtracks=leaf_refine_backtracks,
            refine_closed_form_tier2=refine_closed_form_tier2,
            incremental_mu=incremental_mu,
            mvs_min_rows=mvs_min_rows,
            hist_precision=hist_precision,
            n_bags=n_bags,
            bag_subsample=bag_subsample,
            cell_refit_base=cell_refit_base,
            cell_refit_gamma=cell_refit_gamma,
            ridge_refit_l2=ridge_refit_l2,
            ridge_refit_max_iter=ridge_refit_max_iter,
            nesterov=nesterov,
            dart_drop_rate=dart_drop_rate,
            random_strength=random_strength,
            reanchor=reanchor,
            reanchor_slope=reanchor_slope,
            max_interaction_order=max_interaction_order,
            max_depth=max_depth,
            table_budget_cells=table_budget_cells,
            table_budget_order_shrink=table_budget_order_shrink,
            seed=seed,
            n_jobs=n_jobs,
            monotone_constraints=monotone_constraints,
            categorical_features=categorical_features,
            cat_smooth=cat_smooth,
            cat_target=cat_target,
            cat_leakage=cat_leakage,
            cat_n_perms=cat_n_perms,
            cat_k=cat_k,
            cat_min_data_per_group=cat_min_data_per_group,
            cat_direct_max_levels=cat_direct_max_levels,
            cat_channels=cat_channels,
            cat_count_min_levels=cat_count_min_levels,
            cat_class_freq_min_levels=cat_class_freq_min_levels,
            prune=prune,
            prune_validation_fraction=prune_validation_fraction,
            prune_se_rule=prune_se_rule,
            prune_n_folds=prune_n_folds,
            prune_refit_full=prune_refit_full,
            prune_rebalance=prune_rebalance,
            ref_measure=ref_measure,
            measure_floor=measure_floor,
            prune_guard=prune_guard,
            prune_guard_tol=prune_guard_tol,
            prune_drop_z=prune_drop_z,
            prune_keep_budget=prune_keep_budget,
            prune_fold_fidelity=prune_fold_fidelity,
            prune_guard_z=prune_guard_z,
            prune_guard_z_dn=prune_guard_z_dn,
            prune_guard_tol_floor=prune_guard_tol_floor,
            multiclass_prune_guard=multiclass_prune_guard,
            multiclass_prune_sel_bags=multiclass_prune_sel_bags,
            multiclass_prune_cv=multiclass_prune_cv,
            multiclass_prune_guard_floor=multiclass_prune_guard_floor,
            prune_box_budget=prune_box_budget,
            prune_lambda_boxes=prune_lambda_boxes,
            prune_table_budget=prune_table_budget,
            prune_lambda_tables=prune_lambda_tables,
            prune_table_min_arity=prune_table_min_arity,
            prune_min_stability=prune_min_stability,
            prune_min_mean_gain=prune_min_mean_gain,
            prune_selector=prune_selector,
            prune_path_steps=prune_path_steps,
            prune_path_fraction=prune_path_fraction,
            prune_path_tolerance=prune_path_tolerance,
            band_tolerance=band_tolerance,
            band_deviance_cap=band_deviance_cap,
            prune_fold_min_rows=prune_fold_min_rows,
            prune_fold_es_patience=prune_fold_es_patience,
            prune_guard_min_rows=prune_guard_min_rows,
            prune_slope_eps=prune_slope_eps,
            prune_slope_min_z=prune_slope_min_z,
        )

    def _expected_response(self, X: Any, exposure: Any | None) -> "np.ndarray":
        proba = np.asarray(self.predict_proba(X), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        raise ValueError("actual_vs_expected is defined for regression and binary fits only")

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
        exposure: Any | None = None,
        groups: Any | None = None,
    ) -> "TBoostClassifier":
        """Fit the boosted ensemble.

        Dispatches on the number of distinct classes found in ``y``: exactly two classes
        fit the exact single-model logistic path; three or more fit the native-softmax
        multiclass path (one exact fANOVA bank per class).

        ``groups``: per-row entity ids for panel data; a ``str`` names a column of the polars
        ``X`` (excluded from the features). When given, the INTERNAL early-stopping holdout
        and the prune-CV carve (train/select split for K>=3, K-fold assignment for binary/
        regression) assign WHOLE groups to one side (see ``_carve_group_holdout``/
        ``_carve_group_folds``) instead of rows — on panels the row-level carve leaks
        near-duplicate rows into validation, which defeats early stopping (fits run to the
        tree cap) and biases prune selection. Honored for BOTH the binary (K=2) and multiclass
        (K>=3) paths. An ALL-SINGLETON grouping is normalized to ``None`` at ingest and fits
        exactly as an ungrouped fit does — at group size 1 the two mean the same thing (see
        ``_is_degenerate_grouping``).

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Training input. A polars ``DataFrame`` (or ``LazyFrame``, collected at fit) is
            the first-class frame type: column names set ``feature_names_in_``,
            String/Categorical/Enum columns are target-statistic encoded natively as
            categoricals with no declaration needed (nulls collapse to the reserved missing
            level), numeric columns are cast to ``float32`` (nulls become NaN, routed to the
            missing bin), and any column consumed by a column-name ``y``/``sample_weight``/
            ``exposure``/``groups`` below is dropped from the feature set. Other
            DataFrame-likes with column names keep the declared-``categorical_features``-only
            behavior.
        y : array-like of shape (n_samples,), or str
            Discrete class labels — at least two distinct values, and not a continuous
            target (checked via ``sklearn.utils.multiclass.type_of_target``). A ``str``
            names a column of the polars ``X`` to use (and exclude from the features),
            rustystats-style.
        sample_weight : array-like of shape (n_samples,), or str, optional
            Per-row weights; a ``str`` names a column of the polars ``X``. ``None`` weights
            every row equally.
        exposure : array-like of shape (n_samples,), or str, optional
            Per-row exposure (offset); a ``str`` names a column of the polars ``X``.
            Supported only for binary (K=2) fits; passing a non-``None`` value together
            with three or more classes raises ``ValueError``.

        Returns
        -------
        TBoostClassifier
            ``self``, now fitted, with ``classes_`` set to the sorted distinct labels
            found in ``y``.
        """
        # Deprecation notices fire HERE, at the top of the public fit, so they are
        # independent of `prune` (most deprecated names are not prune knobs) and never
        # fire from `__init__`, which sklearn requires to be assignment-only and which
        # `clone`/`get_params` re-enter constantly. The unhonourable-parameter RAISES run
        # later on their own paths and outrank these warnings.
        self._warn_deprecated_params()
        if str(self.objective).replace("-", "_").lower() != "logistic":
            raise ValueError("TBoostClassifier currently requires objective='logistic'")
        X, resolved = self._resolve_fit_vectors(
            X,
            {
                "y": y,
                "sample_weight": sample_weight,
                "exposure": exposure,
                "groups": groups,
            },
        )
        y = resolved["y"]
        sample_weight = resolved["sample_weight"]
        exposure = resolved["exposure"]
        groups = resolved["groups"]
        y_arr = np.asarray(y)
        target_type = type_of_target(y_arr)
        if target_type in ("continuous", "continuous-multioutput"):
            raise ValueError(
                f"Unknown label type: {target_type!r} — TBoostClassifier needs discrete class "
                "labels, not a continuous target"
            )
        classes = np.unique(y_arr)
        n_classes = int(classes.shape[0])
        if n_classes < 2:
            raise ValueError("TBoostClassifier requires at least two classes")
        self.classes_ = classes
        if n_classes == 2:
            # Binary stays on the exact logistic single-model path (unchanged wire format/behavior).
            self._multi_model = None
            y01 = (y_arr == classes[1]).astype(np.float32, copy=False)
            self._fit_model(
                X,
                y01,
                sample_weight=sample_weight,
                exposure=exposure,
                class_labels=[str(c) for c in classes],
                groups=groups,
            )
            return self
        # K >= 3: native-softmax (multinomial) multiclass path.
        if exposure is not None:
            raise ValueError("exposure is not supported for multiclass (K >= 3) classification")
        self._fit_multiclass(X, y_arr, classes, sample_weight=sample_weight, groups=groups)
        # Proactive half of `binding_report_`: interrupt when a parameter the caller set
        # was actively CANCELLED by another. Fires after the fit because the verdict needs
        # the prune report and the deployed artifact. OVERRIDDEN only — see the method.
        self._warn_voided_params()
        return self

    def _fit_multiclass(
        self,
        X: Any,
        y_arr: np.ndarray,
        classes: np.ndarray,
        *,
        sample_weight: Any | None,
        groups: Any | None = None,
    ) -> _MultiClassModel | _MultiClassTableModel:
        self._clear_stale_fit_state()
        # No silent no-op: there is no graduation path for multiclass (K>=3) at all, prune or
        # not, so an explicit graduate=True can never be honored here.
        if self.graduate:
            raise ValueError(
                "graduate=True has no effect for multiclass (K>=3) fits: there is no "
                "Whittaker-Henderson graduation path for the multiclass per-class banks. "
                "Leave graduate at its default (None)."
            )
        self._validate_multiclass_inert_params()
        self._validate_cat_channels(multiclass=True)
        _reject_sparse(X)
        X = collect_frame(X)  # LazyFrame: full collect (every non-consumed column is a feature)
        feature_names = _feature_names_from_x(X)
        n_features = _n_columns(X)
        if n_features == 0:
            raise ValueError("X has no features (0 columns)")
        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length {len(feature_names)} != n_features {n_features}"
            )
        # Same declared-union-dtype categorical resolution as `_fit_model`.
        cat_idx = sorted(
            set(self._resolve_categorical(n_features, feature_names))
            | set(auto_categorical_idx(X))
        )
        x32, cat_x, axis_names = self._split_columns(X, cat_idx, feature_names)
        # Integer class labels 0..K-1 (np.unique returns sorted classes, so searchsorted is exact).
        y_labels = np.searchsorted(classes, y_arr).astype(np.float32, copy=False)
        if x32.shape[0] != y_labels.shape[0]:
            raise ValueError(f"X has {x32.shape[0]} rows but y has {y_labels.shape[0]}")
        if x32.shape[0] == 0:
            raise ValueError("X has 0 samples; need at least 1 row to fit")
        groups_arr: np.ndarray | None = None
        if groups is not None:
            groups_arr = np.asarray(groups).ravel()
            if groups_arr.shape[0] != y_labels.shape[0]:
                raise ValueError(
                    f"groups has {groups_arr.shape[0]} rows but y has {y_labels.shape[0]}"
                )
            # Same degenerate-grouping normalization as `_fit_model` (see the note there and
            # `_is_degenerate_grouping`): all-singleton groups ARE no groups, so the multiclass
            # fit takes the ungrouped path instead of paying for group-honest carves.
            if _is_degenerate_grouping(groups_arr):
                groups_arr = None
        # GROUP-AWARE outer bags (2026-09-07): dense group codes make every bag a subsample of
        # whole groups, so the out-of-bag rows the cell refit and the prune guard read are honest
        # on panel data (a policy's other years never trained the jury that scores it).
        bag_codes = (
            np.ascontiguousarray(np.unique(groups_arr, return_inverse=True)[1], dtype=np.uint32)
            if groups_arr is not None
            else None
        )
        weight32 = None
        if sample_weight is not None:
            weight32 = _as_float32_1d(sample_weight, "sample_weight")
            if weight32.shape[0] != y_labels.shape[0]:
                raise ValueError(
                    f"sample_weight has {weight32.shape[0]} rows but y has {y_labels.shape[0]}"
                )
            if float(np.sum(weight32)) <= 0.0:
                raise ValueError("sample_weight sums to zero; no effective samples to fit")
        # Same fit-time-mass stash as `_fit_model` (multiclass fit() rejects `exposure`, so only
        # sample_weight can exist here); `tables()` defaults to it, call-time args override.
        if weight32 is not None:
            self._fit_sample_weight_ = weight32
        monotone = self._resolve_monotone(n_features, feature_names)
        if self.graduation_high_order_alpha > 0 and monotone and any(monotone):
            raise ValueError("higher-order smoothing does not support monotone constraints")
        if monotone is not None and cat_idx:
            cat_set = set(cat_idx)
            numeric_idx = [i for i in range(n_features) if i not in cat_set]
            monotone = [monotone[i] for i in numeric_idx] + [monotone[j] for j in cat_idx]
        if monotone is not None and any(monotone):
            raise ValueError(
                "monotone_constraints do not guarantee monotone softmax probabilities; "
                "multiclass constrained fits are not supported"
            )
        booster = self._new_booster(fit_pool_width=self._default_fit_pool(n_features))
        class_label_strs = [str(c) for c in classes]
        k = int(classes.shape[0])
        model: _MultiClassModel | _MultiClassTableModel
        self._validate_box_budget_reachable()
        if self.prune:
            self._validate_dead_prune_params(multiclass=True)
            n = int(x32.shape[0])
            seed = int(self.seed) if getattr(self, "seed", None) is not None else 0
            common: dict[str, Any] = dict(
                weight=weight32,
                feature_names=axis_names,
                monotone=monotone,
                cat_x=cat_x,
                **self._measure_kwargs(),
                se_rule=float(self.prune_se_rule),
                lambda_boxes=float(getattr(self, "prune_lambda_boxes", 0.0) or 0.0),
                sel_bags=int(self.multiclass_prune_sel_bags),
                prune_guard=bool(self.multiclass_prune_guard),
                prune_guard_tol=float(self.prune_guard_tol),
                prune_guard_min_rows=int(self.prune_guard_min_rows),
                # Group-aware bags make a grouped fit's out-of-bag rows honest (see `_fit_model`).
                guard_oob_honest=True,
                bag_groups=bag_codes,
                # The SE-aware bar (2026-09-07): the raw relative tolerance alone never fires on
                # a multiclass log-loss (irreducible class entropy dwarfs any prune harm), so the
                # single-output guard's downward tightening `prune_guard_z_dn` is the live bar.
                guard_z=float(self.prune_guard_z_dn),
                guard_floor=float(self.multiclass_prune_guard_floor),
                box_budget=max(0, int(getattr(self, "prune_box_budget", 0) or 0)),
                lambda_tables=float(getattr(self, "prune_lambda_tables", 0.0) or 0.0),
                table_budget=max(0, int(getattr(self, "prune_table_budget", 0) or 0)),
                table_min_arity=int(
                    getattr(self, "prune_table_min_arity", _PRUNE_TABLE_MIN_ARITY_DEFAULT)
                ),
                # See `_PRUNE_SELECTOR_DEFAULT`: the ranked path replaces the fold CV and the guard.
                ranked_path=str(getattr(self, "prune_selector", _PRUNE_SELECTOR_DEFAULT)) == "ranked_path",
                path_steps=int(getattr(self, "prune_path_steps", _PRUNE_PATH_STEPS_DEFAULT)),
                path_fraction=float(getattr(self, "prune_path_fraction", _PRUNE_PATH_FRACTION_DEFAULT)),
                path_tolerance=float(getattr(self, "prune_path_tolerance", _PRUNE_PATH_TOLERANCE_DEFAULT)),
                # See `_BAND_TOLERANCE_DEFAULT`: every class bank is banded after the keep-set.
                band_tolerance=(
                    None if (bt := getattr(self, "band_tolerance", _BAND_TOLERANCE_DEFAULT)) is None
                    else float(bt)
                ),
                band_deviance_cap=float(getattr(self, "band_deviance_cap", _BAND_DEVIANCE_CAP_DEFAULT)),
            )
            if bool(self.multiclass_prune_cv):
                # The single-output prune's design (see `_fit_model`): group-honest or
                # class-stratified folds, one fold refit per fold scored on its held-out rows,
                # aggregated by `aggregate_prune_selection`. Carve streams mirror `_fit_model`:
                # [seed, 0] folds, [seed, 1] the deploy fit's ES holdout (the SAME stream the
                # unpruned `fit_multiclass` uses, so pruned and unpruned deploys share a carve),
                # [seed, 2] the fold fits' shared ES holdout.
                k_folds = max(
                    2, min(int(self.prune_n_folds), n // int(self.prune_fold_min_rows), n)
                )
                if groups_arr is not None:
                    fold_of = _carve_group_folds(groups_arr, k_folds, [seed, 0])
                else:
                    rng = np.random.default_rng(seed)
                    fold_of = np.empty(n, dtype=np.int64)
                    for cls in range(k):
                        idx = np.flatnonzero(y_labels == cls)
                        if idx.size == 0:
                            continue
                        shuffled = rng.permutation(idx)
                        fold_of[shuffled] = np.arange(shuffled.size, dtype=np.int64) % k_folds
                deploy_es_holdout: list[bool] | None = None
                fold_es_holdout: list[bool] | None = None
                if groups_arr is not None and self.validation_fraction is not None:
                    deploy_es_holdout = _carve_group_holdout(
                        groups_arr, float(self.validation_fraction), [seed, 1]
                    ).tolist()
                    fold_es_holdout = _carve_group_holdout(
                        groups_arr, float(self.validation_fraction), [seed, 2]
                    ).tolist()
                _patience = getattr(self, "prune_fold_es_patience", None)
                table_model, report_json = booster.fit_multiclass_pruned(
                    x32,
                    y_labels,
                    k,
                    class_label_strs,
                    [],
                    fold_of=np.ascontiguousarray(fold_of, dtype=np.int64),
                    k_folds=int(k_folds),
                    fold_es_holdout=fold_es_holdout,
                    deploy_es_holdout=deploy_es_holdout,
                    # 100 rather than the single-output path's 250 (2026-09-07): the K>=3 fold
                    # models only feed the vote, and 100 measured byte-identical keep-sets and
                    # predictions on pg16/fremotor at 15% less fold time.
                    fold_es_patience=(
                        int(_patience)
                        if _patience is not None
                        else _MULTICLASS_PRUNE_FOLD_ES_PATIENCE
                    ),
                    n_folds=2,
                    min_stability=float(self.prune_min_stability),
                    min_mean_gain=float(self.prune_min_mean_gain),
                    drop_z=(None if self.prune_drop_z is None else float(self.prune_drop_z)),
                    keep_budget=max(0, int(self.prune_keep_budget)),
                    **common,
                )
            else:
                # Legacy single train/select split (`multiclass_prune_cv=False`).
                es_holdout: list[bool] | None = None
                deploy_es_holdout_legacy: list[bool] | None = None
                if groups_arr is not None:
                    sel_mask = _carve_group_holdout(
                        groups_arr, float(self.prune_validation_fraction), [seed, 0]
                    )
                    sel_idx = np.flatnonzero(sel_mask)
                    if self.validation_fraction is not None:
                        comp_idx = np.flatnonzero(~sel_mask)
                        es_holdout = _carve_group_holdout(
                            groups_arr[comp_idx], float(self.validation_fraction), [seed, 1]
                        ).tolist()
                        deploy_es_holdout_legacy = _carve_group_holdout(
                            groups_arr, float(self.validation_fraction), [seed, 2]
                        ).tolist()
                else:
                    rng = np.random.default_rng(seed)
                    perm = rng.permutation(n)
                    n_sel = max(
                        1, min(int(round(n * float(self.prune_validation_fraction))), n - 1)
                    )
                    sel_idx = np.sort(perm[:n_sel])
                table_model, report_json = booster.fit_multiclass_pruned(
                    x32,
                    y_labels,
                    k,
                    class_label_strs,
                    sel_idx.tolist(),
                    n_folds=int(self.prune_n_folds),
                    es_holdout=es_holdout,
                    deploy_es_holdout=deploy_es_holdout_legacy,
                    **common,
                )
            self.pruning_report_ = json.loads(report_json)
            # §G1 K>=3 OOB cell-refit diagnostics for the DEPLOYED fit — the multiclass analogue
            # of `delta_step_gate_`. Pruning replaces the `_MultiClassModel` with a
            # `_MultiClassTableModel`, which carries no fit-time diagnostics, so the native side
            # folds the report into the prune report JSON and it is lifted back out here. `None`
            # whenever `cell_refit_base` was not set (the default), so an unrefitted fit reports
            # exactly what it always did.
            self.cell_refit_report_ = self.pruning_report_.get("cell_refit")
            model = table_model
        else:
            es_holdout_full: list[bool] | None = None
            if groups_arr is not None and self.validation_fraction is not None:
                seed = int(self.seed) if getattr(self, "seed", None) is not None else 0
                es_holdout_full = _carve_group_holdout(
                    groups_arr, float(self.validation_fraction), [seed, 1]
                ).tolist()
            model = booster.fit_multiclass(
                x32,
                y_labels,
                k,
                class_label_strs,
                weight=weight32,
                feature_names=axis_names,
                monotone=monotone,
                cat_x=cat_x,
                es_holdout=es_holdout_full,
                bag_groups=bag_codes,
            )
            self.cell_refit_report_ = model.cell_refit_report
        self._multi_model = model
        self._cat_indices_ = cat_idx
        self.n_features_in_ = n_features
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        return model

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Samples to score. A polars frame is matched by NAME to the fit-time feature
            columns: extra columns are ignored, column order is irrelevant, and a
            ``LazyFrame`` collects only the needed columns. Array-likes (and a model fitted
            without column names) must instead have the same number and layout of features
            as the data passed to ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, 2) or (n_samples, n_classes)
            For a binary (K=2) fit, columns are ``[P(classes_[0]), P(classes_[1])]``. For a
            multiclass (K>=3) fit, column ``j`` is ``P(classes_[j])``. Always ``float64``
            (the underlying scores are computed in ``float32``; the wider return type
            follows sklearn's convention and avoids ``float32`` accumulation error in
            downstream log-loss/Brier/AUROC metrics).
        """
        check_is_fitted(self)
        mm = self._multi_model
        x32, cat_kw = self._serve_kwargs(X, mm if mm is not None else self._model)
        # float64 outputs (the core scores in f32): sklearn convention; avoids f32 error in
        # downstream log-loss / Brier / AUROC.
        if mm is not None:
            return np.asarray(mm.predict_proba(x32, **cat_kw), dtype=np.float64)
        return np.asarray(
            self._model.predict_proba(x32, **cat_kw, n_jobs=self._resolve_n_jobs()),
            dtype=np.float64,
        )

    def decision_function(self, X: Any) -> np.ndarray:
        """Predict on the link (logit) scale, before the sigmoid/softmax is applied.

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Samples to score. A polars frame is matched by NAME to the fit-time feature
            columns: extra columns are ignored, column order is irrelevant, and a
            ``LazyFrame`` collects only the needed columns. Array-likes (and a model fitted
            without column names) must instead have the same number and layout of features
            as the data passed to ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_classes)
            For a binary (K=2) fit, the 1-D logit of ``P(classes_[1])``. For a multiclass
            (K>=3) fit, an ``(n_samples, n_classes)`` array of raw per-class logits
            (sklearn's one-vs-rest-shaped convention for ``decision_function``, though the
            underlying fit is a joint softmax, not independent one-vs-rest models). Always
            ``float64``.
        """
        check_is_fitted(self)
        mm = self._multi_model
        x32, cat_kw = self._serve_kwargs(X, mm if mm is not None else self._model)
        if mm is not None:
            # Multiclass: (n_samples, n_classes) raw logits, per sklearn's OvR-shaped convention.
            return np.asarray(mm.predict_raw(x32, **cat_kw), dtype=np.float64)
        return np.asarray(
            self._model.predict_raw(x32, **cat_kw, n_jobs=self._resolve_n_jobs()),
            dtype=np.float64,
        )

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Samples to score. A polars frame is matched by NAME to the fit-time feature
            columns: extra columns are ignored, column order is irrelevant, and a
            ``LazyFrame`` collects only the needed columns. Array-likes (and a model fitted
            without column names) must instead have the same number and layout of features
            as the data passed to ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels drawn from ``classes_``: for binary (K=2) fits, ``classes_[1]``
            wherever ``predict_proba(X)[:, 1] >= 0.5`` else ``classes_[0]``; for multiclass
            (K>=3) fits, the highest-probability class.
        """
        proba = self.predict_proba(X)
        if self._multi_model is not None:
            return np.asarray(self.classes_[np.argmax(proba, axis=1)])
        return self.classes_[(proba[:, 1] >= 0.5).astype(np.intp)]

    def to_bytes(self) -> bytes:
        """Serialize the fitted estimator to a compact binary wire format.

        Returns
        -------
        bytes
            A self-describing blob (parameters, fitted metadata including ``classes_``,
            and the model — binary or multiclass) that round-trips through
            :meth:`from_bytes`.
        """
        check_is_fitted(self)
        mm = self._multi_model
        inner = mm.to_bytes() if mm is not None else self._model.to_bytes()
        return _pack_bytes(self, inner)

    def to_json(self) -> str:
        """Serialize the fitted estimator to a human-readable JSON wire format.

        Returns
        -------
        str
            A self-describing JSON document (parameters, fitted metadata including
            ``classes_``, and the model — binary or multiclass) that round-trips through
            :meth:`from_json`. Larger than :meth:`to_bytes` but diffable and inspectable
            without this library.
        """
        check_is_fitted(self)
        mm = self._multi_model
        inner = mm.to_json() if mm is not None else self._model.to_json()
        return _pack_json(self, inner)

    def tables(
        self,
        X: Any,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        basis_json: str | None = None,
        measure_floor: float | None = None,
        overflow: str | None = None,
        sample_weight: Any | None = None,
        exposure: Any | None = None,
    ) -> str:
        """Export the exact fANOVA effect-table decomposition as JSON.

        The returned tables are the glassbox explanation of the model: every table lookup
        the served rows in ``X`` actually touch, such that their sum (plus the intercept)
        exactly reproduces :meth:`decision_function` — no approximation, no surrogate fit.
        For a multiclass (K>=3) fit, each class has its own independent table bank.

        Parameters
        ----------
        X : array-like, polars DataFrame/LazyFrame, of shape (n_samples, n_features)
            Rows used to determine which table cells are populated/reported. A polars frame
            is matched by NAME to the fit-time feature columns (extra columns ignored, any
            order; a ``LazyFrame`` collects only the needed columns). Array-likes (and a
            model fitted without column names) must instead have the same number and layout
            of features as the data passed to ``fit``.
        ref_measure : {"exposure", "product_marginals", "uniform", "joint"} or None, default=None
            Reference measure the tables are purified against. ``None`` means the ledger
            this estimator fitted under (the constructor's ``ref_measure``, ``"exposure"``
            by default): for a pruned fit the deployed bank exactly as stored, for an
            unpruned fit a fresh purification under that measure. An explicit name
            re-expresses the SAME function under that measure — the table sum on every cell,
            and so every prediction, is unchanged; only which table owns shared mass moves.
            On a pruned fit this runs on the bank's own stored support, so no rows are read.
            ``"exposure"`` weights each axis by its exposure-weighted empirical marginal plus
            the ``measure_floor`` positivity floor; ``"product_marginals"`` is the legacy
            half-uniform row-count blend (``laplace``); ``"uniform"`` weights every cell
            equally. ``"joint"`` (export-only, 2026-09-06) re-expresses orders one and two
            using a regularized pairwise joint-exposure reallocation (ridge 0.001,
            positivity floor 1e-6). It preserves the total score, but the ridge prevents
            exact conditional purity and the floor assigns some mass to unsupported cells.
            The export reports residual conditional means on observed support. Main effects
            are not observed marginal A/E or causal effects. Orders three and above stay
            product-purified (a hybrid), variance shares no longer add to one, and the
            equal-split attributions are no longer Shapley values.
        laplace : float, default=1.0
            Laplace smoothing constant for the ``"product_marginals"`` reference measure.
            Unused for the other measures.
        measure_floor : float or None, default=None
            Positivity floor for ``"exposure"``; ``None`` inherits the fit-time
            ``measure_floor``. Unused for the other measures.
        basis_json : str or None, default=None
            Optional JSON-encoded custom basis/grid to export the tables against, in place
            of the model's native training-time bin borders. ``None`` uses the native basis.
        overflow : {"factored", "error", "sparse"} or None, default=None
            Table-budget overflow policy for effects whose merged grid exceeds the dense
            cell budget. ``None``/``"factored"`` (the default) keeps an over-budget effect
            exactly as a factored per-tree-box sum rather than materializing the dense
            table; ``"error"`` raises instead; ``"sparse"`` stores it as an exact sparse
            tensor. All three policies are exactness-preserving.
        sample_weight : array-like of shape (n_samples,) or str, optional
            Per-row weight for ``X``, aligned to ``X``'s rows; a ``str`` names a column of
            the polars ``X``. When present (explicitly, or
            via the fit-time default below), each table's ``support`` — and, under
            ``ref_measure="product_marginals"``, the empirical reference measure itself — is
            computed from the effective row mass (``sample_weight * exposure``, elementwise;
            either alone is used as-is) instead of a flat row count, so a thin-looking cell
            backed by a few highly-weighted rows reports its true mass (spec §08.7). This
            changes ``values``/importances/SE bands too, not just ``support`` — it is a
            different, equally valid decomposition against a different reference measure,
            not a display-only reweighting. Applies per class for a multiclass (K>=3) fit
            (the same row mass is used for every class's bank). **Default:** when ``None``,
            falls back to the ``sample_weight`` the estimator was fit with (if any) — the
            honest default for exporting a weighted/exposure fit's own tables. That fallback
            only makes sense when ``X`` is the fit-time data: pass an explicit array (or
            ``numpy.ones``, to force the unweighted export) whenever ``X`` is different rows,
            since a different ``X`` with a coincidentally matching row count would silently
            misapply the fit-time mass. Cleared by ``set_params`` and refits, never leaks
            across fits. No effect if ``self`` is a pruned (``prune=True``) model: its tables
            are already frozen from fit time.
        exposure : array-like of shape (n_samples,) or str, optional
            Per-row exposure for ``X``, aligned to ``X``'s rows; a ``str`` names a column of
            the polars ``X``. See ``sample_weight`` — the
            two combine multiplicatively into the same effective row mass, and each defaults
            independently to its fit-time value when ``None``. Unlike ``fit`` (where exposure
            is rejected for multiclass), this explain-time exposure is accepted for
            multiclass tables too — it is a support-weighting mass, not the fit-time GLM
            offset — though a multiclass fit never has a fit-time exposure to default to
            (``fit`` rejects ``exposure`` for K>=3 before it would be stashed), so ``None``
            there is always the plain unweighted-on-exposure export unless overridden.

        Returns
        -------
        str
            A JSON document describing every exported effect table (features, shape,
            values, per-cell support, and standard-error band), one bank per class for
            multiclass fits.
        """
        check_is_fitted(self)
        X, sample_weight, exposure = self._resolve_explain_inputs(X, sample_weight, exposure)
        x32, cat_x = self._serve_design(X)
        mm = self._multi_model
        target: _MultiClassModel | _MultiClassTableModel | _Model | _TableModel = (
            mm if mm is not None else self._model
        )
        # Explicit call-time mass wins; otherwise default to the fit-time stash (multiclass
        # fits never stash exposure — fit() rejects it for K>=3 — so only sample_weight can
        # default there). See the regressor docstring's non-fit-time-X caveat.
        weight32 = (
            _as_float32_1d(sample_weight, "sample_weight")
            if sample_weight is not None
            else getattr(self, "_fit_sample_weight_", None)
        )
        exposure32 = (
            _as_float32_1d(exposure, "exposure")
            if exposure is not None
            else getattr(self, "_fit_exposure_", None)
        )
        payload = target.tables(
            x32,
            **self._export_measure_kwargs(ref_measure, laplace, measure_floor),
            basis_json=basis_json,
            cat_x=cat_x,
            overflow=overflow,
            weight=weight32,
            exposure=exposure32,
        )
        return annotate_joint_export(payload, ref_measure)

    def _attach_classifier_model(self, model: _Model | _TableModel) -> None:
        labels = model.class_labels
        if labels is None or len(labels) != 2:
            raise ValueError("serialized classifier model must carry exactly two class labels")
        self.classes_ = np.asarray(labels, dtype=object)
        self._multi_model = None
        self._attach_model(model)

    def _attach_multiclass_model(self, model: _MultiClassModel | _MultiClassTableModel) -> None:
        self.classes_ = np.asarray(model.class_labels, dtype=object)
        self._multi_model = model
        self.n_features_in_ = model.n_features
        names = model.feature_names
        if names:
            self.feature_names_in_ = np.asarray(names, dtype=object)


_CLASSIFIER_OBJECTIVES = {"logistic", "classifier", "binary", "multiclass", "softmax", "log_loss"}


def recommended_recipe(
    objective: str = "squared_error",
    *,
    budget: int = 4000,
    n_jobs: int | None = 4,
    tuned: bool = True,
    seed: int = 0,
    **overrides: Any,
) -> "_BaseTBoost":
    """Return a t-boost estimator configured with the benchmark recipe.

    The estimator constructors now default to this recipe (early stopping with adaptive
    patience, leaf refinement, outer bagging, post-fit table pruning — the exact insur-arena
    deployment since 2026-07-15), so a bare ``TBoostRegressor()`` / ``TBoostClassifier()``
    already IS the benchmarked configuration. This factory remains the explicit,
    benchmark-facing entry point: it caps ``n_trees`` via ``budget`` (early stopping decides
    the real length), sets ``n_jobs``, and — for log-link objectives with ``tuned=True`` —
    additionally sets the Gamma log-mean target-statistic variant and explicit categorical
    params, on top of the link-aware ``reanchor`` and k-fold mean target-statistic encoding
    the plain defaults already include. (A short-lived gamma/tweedie ``n_bags=16`` shipped and
    was reverted on 2026-07-23: event-stratified subagging proved to be the real variance fix,
    and the constructor's objective-aware ``path_smooth`` default covers the remainder free.)
    All of these levers are exactness-preserving.

    ``objective`` selects a ``TBoostRegressor`` (``squared_error`` / ``poisson`` / ``gamma`` /
    ``tweedie``) or a ``TBoostClassifier`` (``logistic`` — which auto-uses native softmax for ≥3
    classes at fit). ``budget`` is the ``n_trees`` cap (early stopping decides the real length).
    Any keyword in ``overrides`` wins over the recipe. Log-link objectives also default to the
    evidence-gated ``reanchor_slope`` holdout recalibration (an estimator default, not recipe magic).

    Note: for multiclass (K≥3), ``n_bags``/``bag_subsample`` ARE honored (outer bags souped per
    class, 2026-07-14) and so is ``subsample``/``mvs_min_rows`` (§06.5 MVS, 2026-08-23);
    ``leaf_refine_steps`` has no effect (a K>=3 refinement measured harmful as a drop-in and
    was removed).
    """
    obj = str(objective).replace("-", "_").lower()
    is_classifier = obj in _CLASSIFIER_OBJECTIVES

    params: dict[str, Any] = {
        "n_trees": int(budget),
        "learning_rate": 0.05,
        "lambda_": 1.0,
        "max_bin": 254,
        "leaf_refine_steps": 4,
        "validation_fraction": 0.1,
        "early_stopping_rounds": 500,  # load-bearing knob (also the estimator default now)
        # early_stopping_adaptive=1.5 and prune=True are inherited from the constructor defaults
        # (benchmark parity, 2026-07-15) — no longer recipe-only. Overridable via **overrides.
        "interaction_gain_hurdle_mode": _GAIN_HURDLE_MODE_DEFAULT,
        "max_interaction_order": 3,
        "seed": int(seed),
        "n_jobs": n_jobs,
    }
    if tuned:
        params["n_bags"] = 8
        params["colsample_bytree"] = 0.8
        if obj in {"poisson", "gamma", "tweedie"}:
            # Log-link objectives: reanchor removes post-shrinkage aggregate bias; leakage-free
            # K-fold target-statistic categorical encoding (log-mean for the heavy-tailed Gamma).
            params["reanchor"] = True
            params["cat_target"] = "log_mean" if obj == "gamma" else "mean"
            params["cat_leakage"] = "kfold"
            params["cat_k"] = 5
        # 2026-07-23: the short-lived gamma/tweedie n_bags=16 (2026-07-22) is reverted — the
        # event-stratified subagging that shipped alongside it turned out to be the actual
        # variance fix (post-stratification 6-set screen: 16 no longer beats 8 anywhere;
        # ohlsson_pp now PREFERS 8 by 1.2%). The remaining tail-cell variance is handled by
        # the constructor's objective-aware `path_smooth` default (10.0 for gamma/tweedie),
        # which costs nothing. Net: cheaper than both the 16-bag state and the pre-batch state.
    params.update(overrides)

    if is_classifier:
        params.pop("objective", None)
        return TBoostClassifier(objective="logistic", **params)
    return TBoostRegressor(objective=objective, **params)
