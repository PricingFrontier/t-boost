from __future__ import annotations

from typing import Any, Sequence, final

import numpy as np

__all__ = [
    "BUILD_PROFILE",
    "TBoostError",
    "InvariantError",
    "ExactnessError",
    "SerializationError",
    "InternalError",
    "TBoostValueError",
    "TBoostTypeError",
    "_Booster",
    "_Model",
    "_MultiClassModel",
    "_MultiClassTableModel",
    "_TableBank",
    "_TableModel",
]

BUILD_PROFILE: str
"""`"debug"` or `"release"` — the cargo profile the extension was compiled with."""


class TBoostError(Exception): ...
class InvariantError(TBoostError): ...
class ExactnessError(TBoostError): ...
class SerializationError(TBoostError): ...
class InternalError(TBoostError): ...

# Built via genuine multiple inheritance (spec §12.7): PbError::InvalidInput/ShapeMismatch/
# InvalidConfig and ::DtypeMismatch respectively raise these, so `except ValueError`/`except
# TypeError` (the spec-promised builtin) AND `except TBoostError` (the pre-existing
# catch-all) both match the same raised instance.
class TBoostValueError(TBoostError, ValueError): ...
class TBoostTypeError(TBoostError, TypeError): ...


@final
class _Booster:
    def __new__(
        cls,
        n_trees: int = 1000,
        learning_rate: float = 0.05,
        lambda_: float = 1.0,
        lambda_scale_invariant: bool = False,
        l1_leaf: float = 0.0,
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_delta_step_gated: Any = None,
        max_bin: int = 254,
        objective: str | None = None,
        tweedie_rho: float = 1.5,
        min_data_in_leaf: int = 0,
        min_sum_hessian_in_leaf: float = 0.0,
        min_weight_sum_in_leaf: float = 0.0,
        path_smooth: float = 0.0,
        subsample: float | None = None,
        colsample_bytree: float = 1.0,
        learning_rate_decay: float = 0.0,
        validation_fraction: float | None = None,
        early_stopping_rounds: int = 50,
        early_stopping_adaptive: float | None = None,
        early_stopping_min_delta: float = 0.0,
        interaction_gain_hurdle: float = 0.0,
        interaction_gain_hurdle_mode: str | None = None,
        leaf_refine_steps: int = 0,
        leaf_refine_backtracks: int = 4,
        refine_closed_form_tier2: bool = True,
        incremental_mu: bool = False,
        mvs_min_rows: int = 1,
        hist_precision: str | None = None,
        n_bags: int = 0,
        bag_subsample: float = 1.0,
        ridge_refit_l2: float | None = None,
        ridge_refit_max_iter: int = 5,
        nesterov: bool = False,
        dart_drop_rate: float | None = None,
        random_strength: float = 0.0,
        reanchor: bool = False,
        reanchor_slope: bool = False,
        max_interaction_order: int = 3,
        max_depth: int = 3,
        table_budget_cells: int = 2_000_000,
        table_budget_order_shrink: float = 2.0,
        cat_smooth: float | None = None,
        cat_target: str | None = None,
        cat_leakage: str | None = None,
        cat_n_perms: int = 1,
        cat_k: int = 5,
        cat_min_data_per_group: float = 10.0,
        cat_direct_max_levels: int = 16,
        cat_channels: Sequence[str] | None = None,
        cat_count_min_levels: int = 20,
        cat_class_freq_min_levels: int = 3,
        cell_refit_base: float | None = None,
        cell_refit_gamma: float = 2.0,
        seed: int = 0,
        n_jobs: int | None = None,
        fit_pool_width: int | None = None,
    ) -> _Booster: ...

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        class_labels: Sequence[str] | None = None,
        monotone: Sequence[int] | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        es_holdout: Sequence[bool] | None = None,
        bag_groups: np.ndarray | None = None,
    ) -> _Model: ...

    def fit_multiclass(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        class_labels: Sequence[str],
        weight: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        monotone: Sequence[int] | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        es_holdout: Sequence[bool] | None = None,
        bag_groups: np.ndarray | None = None,
    ) -> _MultiClassModel: ...

    def fit_multiclass_pruned(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        class_labels: Sequence[str],
        sel_rows: Sequence[int],
        weight: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        monotone: Sequence[int] | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        se_rule: float = 0.0,
        lambda_boxes: float = 0.0,
        n_folds: int = 5,
        es_holdout: Sequence[bool] | None = None,
        deploy_es_holdout: Sequence[bool] | None = None,
        prune_guard: bool = False,
        prune_guard_tol: float = 0.05,
        prune_guard_min_rows: int = 500,
        guard_oob_honest: bool = True,
        sel_bags: int = 1,
        box_budget: int = 0,
        lambda_tables: float = 0.0,
        table_budget: int = 0,
        table_min_arity: int = 3,
        fold_of: np.ndarray | None = None,
        k_folds: int = 0,
        fold_es_holdout: Sequence[bool] | None = None,
        fold_es_patience: int | None = None,
        min_stability: float = 0.5,
        min_mean_gain: float = 0.0,
        drop_z: float | None = None,
        keep_budget: int = 0,
        guard_z: float = 0.0,
        guard_floor: float = 0.0,
        bag_groups: np.ndarray | None = None,
        ranked_path: bool = False,
        path_steps: int = 32,
        path_fraction: float = 1.0,
        band_tolerance: float | None = None,
        band_deviance_cap: float = 0.001,
        path_tolerance: float = 0.0,
    ) -> tuple[_MultiClassTableModel, str]: ...

    def fit_prune_folds(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fold_of: np.ndarray,
        k_folds: int,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        class_labels: Sequence[str] | None = None,
        monotone: Sequence[int] | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        se_rule: float = 0.0,
        lambda_boxes: float = 0.0,
        n_folds: int = 1,
        reanchor: bool = True,
        es_holdout: Sequence[bool] | None = None,
        lambda_tables: float = 0.0,
        table_min_arity: int = 3,
    ) -> list[str]: ...

    def fit_prune_selection(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fold_of: np.ndarray,
        k_folds: int,
        full_supports: Sequence[Sequence[int]],
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        class_labels: Sequence[str] | None = None,
        monotone: Sequence[int] | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        se_rule: float = 0.0,
        lambda_boxes: float = 0.0,
        n_folds: int = 1,
        reanchor: bool = True,
        min_stability: float = 0.5,
        min_mean_gain: float = 0.0,
        es_holdout: Sequence[bool] | None = None,
        drop_z: float | None = None,
        keep_budget: int = 0,
        lambda_tables: float = 0.0,
        table_min_arity: int = 3,
        fold_fidelity: bool = False,
    ) -> tuple[list[list[int]], str]: ...


@final
class _Model:
    @staticmethod
    def from_json(s: str) -> _Model: ...

    @staticmethod
    def from_bytes(bytes: bytes) -> _Model: ...

    @property
    def n_features(self) -> int: ...

    @property
    def feature_names(self) -> list[str]: ...

    @property
    def class_labels(self) -> list[str] | None: ...

    @property
    def n_trees(self) -> int: ...

    def predict(
        self,
        x: np.ndarray,
        out: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def predict_raw(
        self,
        x: np.ndarray,
        out: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def predict_proba(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def explain(
        self,
        x: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        cat_x: Sequence[Sequence[str]] | None = None,
        overflow: str | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> _TableBank: ...

    def tables(
        self,
        x: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        basis_json: str | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        overflow: str | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> str: ...

    def table_supports(
        self,
        x: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        cat_x: Sequence[Sequence[str]] | None = None,
        overflow: str | None = None,
    ) -> list[list[int]]: ...

    def table_variances(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        cat_x: Sequence[Sequence[str]] | None = None,
        exposure: np.ndarray | None = None,
        n_jobs: int | None = None,
    ) -> list[tuple[list[int], float]]: ...

    def bag_score_variance(
        self,
        x: np.ndarray,
        keep: Sequence[Sequence[int]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]: ...

    def bag_in_bag_mask(self) -> np.ndarray: ...
    def bag_oob_available(self) -> bool: ...

    def bag_raw_scores(
        self,
        x: np.ndarray,
        keep: Sequence[Sequence[int]] | None = None,
        rows: Sequence[int] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def bag_oob_group_raw(
        self,
        x: np.ndarray,
        groups: Sequence[Sequence[Sequence[int]]],
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
        with_full: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

    @property
    def delta_step_gate(self) -> dict[str, Any] | None: ...

    def to_json(self) -> str: ...

    def to_bytes(self) -> bytes: ...

    def prune_to_tables(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        sel_rows: Sequence[int],
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        se_rule: float = 1.0,
        lambda_boxes: float = 0.0,
        n_folds: int = 5,
        reanchor: bool = True,
        exposure: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
    ) -> tuple[_TableModel, str]: ...

    def apply_keepset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        keep: Sequence[Sequence[int]],
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        reanchor: bool = True,
        rebalance: bool = False,
        n_jobs: int | None = None,
        exposure: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
    ) -> _TableModel: ...

    def apply_keepset_budgeted(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        keep: Sequence[Sequence[int]],
        box_budget: int,
        box_rank: Sequence[tuple[Sequence[int], float]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        reanchor: bool = True,
        rebalance: bool = False,
        n_jobs: int | None = None,
        exposure: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        table_budget: int = 0,
        table_min_arity: int = 3,
    ) -> tuple[_TableModel, str, str]: ...

    def bag_bank_jsons(
        self,
        x: np.ndarray,
        keep: Sequence[Sequence[int]],
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> list[str]: ...


@final
class _TableModel:
    def deployed_table_count(self) -> int: ...
    def deployed_census(self) -> dict[str, Any]: ...
    def deployed_supports(self) -> list[list[int]]: ...
    def cell_indices(
        self, x: np.ndarray, cat_x: Sequence[Sequence[str]] | None = None
    ) -> np.ndarray: ...
    def raw_feature_names(self) -> list[str]: ...
    def effect_contributions(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        cat_codes: Sequence[tuple[np.ndarray, Sequence[str]]] | None = None,
        n_jobs: int | None = None,
    ) -> tuple[float, np.ndarray, list[list[int]]]: ...
    def sobol(self) -> list[tuple[list[int], float]]: ...
    def band(
        self,
        x: np.ndarray,
        h: np.ndarray,
        mass: np.ndarray,
        target: np.ndarray,
        sigma: float,
        tolerance: float = 0.75,
        cat_x: Sequence[Sequence[str]] | None = None,
        max_select_rows: int = 60000,
        pseudo_rows: int = 20000,
        seed: int = 0,
        n_jobs: int | None = None,
        mse_cap: float = ...,
    ) -> tuple[_TableModel, str]: ...
    def graduation_tables(
        self, max_cells: int = 2500
    ) -> list[tuple[int, list[int], list[int], list[float], list[float], list[bool]]]: ...

    def apply_high_order_graduation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        updates: Sequence[tuple[int, Sequence[float]]],
        alpha: float,
        exposure: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
        box_budget: int = 0,
    ) -> tuple[_TableModel, bool, str]: ...

    def apply_graduation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        updates: Sequence[tuple[int, Sequence[float]]],
        exposure: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
        eval_rows: Sequence[int] | None = None,
    ) -> tuple[_TableModel, bool]: ...

    @staticmethod
    def from_json(s: str) -> _TableModel: ...

    @staticmethod
    def from_bytes(bytes: bytes) -> _TableModel: ...

    @property
    def n_features(self) -> int: ...

    @property
    def feature_names(self) -> list[str]: ...

    @property
    def class_labels(self) -> list[str] | None: ...

    def predict(
        self,
        x: np.ndarray,
        out: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        cat_codes: Sequence[tuple[np.ndarray, Sequence[str]]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def predict_raw(
        self,
        x: np.ndarray,
        out: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        cat_codes: Sequence[tuple[np.ndarray, Sequence[str]]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def predict_proba(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        cat_codes: Sequence[tuple[np.ndarray, Sequence[str]]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def tables(
        self,
        x: np.ndarray | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        basis_json: str | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        overflow: str | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> str: ...

    def to_json(self) -> str: ...

    def to_bytes(self) -> bytes: ...


@final
class _MultiClassModel:
    @staticmethod
    def from_json(s: str) -> _MultiClassModel: ...

    @staticmethod
    def from_bytes(bytes: bytes) -> _MultiClassModel: ...

    @property
    def cell_refit_report(self) -> dict[str, Any] | None: ...

    @property
    def n_features(self) -> int: ...

    @property
    def n_classes(self) -> int: ...

    @property
    def class_labels(self) -> list[str]: ...

    @property
    def feature_names(self) -> list[str]: ...

    def predict_proba(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def predict_raw(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def mc_supports(
        self,
        x: np.ndarray,
        weight: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
    ) -> list[tuple[list[int], float]]: ...

    def to_tables(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
    ) -> _MultiClassTableModel: ...
    def mc_apply_keepset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        keep: Sequence[Sequence[int]],
        cat_x: Sequence[Sequence[str]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
    ) -> _MultiClassTableModel: ...

    def mc_carve_arms(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        groups: Sequence[Sequence[Sequence[int]]],
        mask: Sequence[bool],
        cat_x: Sequence[Sequence[str]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]: ...

    def mc_oob_arms(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        groups: Sequence[Sequence[Sequence[int]]],
        cat_x: Sequence[Sequence[str]] | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]: ...

    def tables(
        self,
        x: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        basis_json: str | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        overflow: str | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> str: ...

    def to_json(self) -> str: ...

    def to_bytes(self) -> bytes: ...

    def prune_to_tables(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        weight: np.ndarray,
        sel_rows: Sequence[int],
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        se_rule: float = 0.0,
        lambda_boxes: float = 0.0,
        n_folds: int = 5,
        cat_x: Sequence[Sequence[str]] | None = None,
        n_jobs: int | None = None,
    ) -> tuple[_MultiClassTableModel, str]: ...


@final
class _MultiClassTableModel:
    @staticmethod
    def from_json(s: str) -> _MultiClassTableModel: ...

    @staticmethod
    def from_bytes(bytes: bytes) -> _MultiClassTableModel: ...

    @property
    def n_features(self) -> int: ...

    @property
    def n_classes(self) -> int: ...

    def deployed_table_count(self) -> int: ...
    def deployed_census(self) -> dict[str, Any]: ...

    @property
    def class_labels(self) -> list[str]: ...

    @property
    def feature_names(self) -> list[str]: ...

    def predict_proba(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        cat_codes: Sequence[tuple[np.ndarray, Sequence[str]]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...

    def predict_raw(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        cat_codes: Sequence[tuple[np.ndarray, Sequence[str]]] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray: ...
    def effect_contributions(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
        cat_codes: Sequence[tuple[np.ndarray, Sequence[str]]] | None = None,
        n_jobs: int | None = None,
    ) -> list[tuple[float, np.ndarray, list[list[int]]]]: ...
    def sobol(self) -> list[list[tuple[list[int], float]]]: ...

    def tables(
        self,
        x: np.ndarray | None = None,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        measure_floor: float = 0.001,
        basis_json: str | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
        overflow: str | None = None,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> str: ...

    def to_json(self) -> str: ...

    def to_bytes(self) -> bytes: ...


@final
class _TableBank:
    @property
    def f0(self) -> float: ...

    @property
    def n_tables(self) -> int: ...

    def score_cells(self, cells: Sequence[int]) -> float: ...

    def sobol(self) -> list[tuple[list[int], float]]: ...
