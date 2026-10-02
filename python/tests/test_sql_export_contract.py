"""Independent SQLite scoring of dense numeric tables and default third-order boxes."""
import json
import sqlite3
import pytest

import numpy as np

from t_boost import TBoostRegressor


@pytest.mark.parametrize("depth,order", [(3, 3), (4, 3), (5, 4)])
def test_sql_reconstructs_numeric_boundaries_missing_and_boxes(depth, order):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 4)).astype(np.float32)
    y = (1.2 * (x[:, 0] > 0) - .8 * (x[:, 1] > 0)
         + 1.5 * ((x[:, 0] > 0) & (x[:, 1] > 0) & (x[:, 2] > 0))).astype(np.float32)
    if order == 4:
        y += 2 * np.all(x > 0, axis=1)
    model = TBoostRegressor(n_trees=12, n_bags=1, n_jobs=2, validation_fraction=None,
                           max_depth=depth, max_interaction_order=order,
                           prune=False, interaction_gain_hurdle=0,
                           colsample_bytree=1, leaf_refine_steps=0).fit(x, y)
    bank = json.loads(model.tables(x))
    assert bank['factored'], 'fixture must exercise boxes, not only dense tables'
    terms = [repr(bank['f0'])]
    audit = [row.astype(float) for row in x[:20]]
    for table in bank['tables']:
        indices = []
        for axis in table['axes']:
            j = axis['raw']
            # Promote the exact float32 border to SQL double, as well as normalizing x.
            borders = [float(np.float32(b)) for b in axis['borders']]
            cell = f'CASE WHEN x{j} IS NULL THEN 0 ELSE 1'
            for border in borders:
                cell += f' + CASE WHEN f32(x{j}) > {border!r} THEN 1 ELSE 0 END'
                for value in (np.nextafter(np.float32(border), np.float32(-np.inf)), border,
                              np.nextafter(np.float32(border), np.float32(np.inf))):
                    row = np.zeros(4)
                    row[j] = value
                    audit.append(row)
            indices.append(cell + ' END')
        flat = indices[0]
        for cell, size in zip(indices[1:], table['shape'][1:]):
            flat = f'(({flat}) * {size} + ({cell}))'
        terms.append('(CASE (' + flat + ') ' + ' '.join(
            f'WHEN {i} THEN {float(v)!r}' for i, v in enumerate(table['values'])) + ' END)')
    for effect in bank['factored']:
        for box in effect['boxes']:
            sides = []
            for level, raw in enumerate(effect['feature_set']):
                border = float(np.float32(box['thresholds'][level]))
                missing = int(box['missing_left'][level])
                sides.append(f'((CASE WHEN x{raw} IS NULL THEN {missing} '
                             f'WHEN f32(x{raw}) <= {border!r} THEN 1 ELSE 0 END) * {1 << level})')
            leaf = ' + '.join(sides)
            terms.append('(CASE (' + leaf + ') ' + ' '.join(
                f'WHEN {i} THEN {float(v)!r}' for i, v in enumerate(box['octants'])) + ' END)')
    audit.extend([np.full(4, np.nan), np.full(4, -1e20), np.full(4, 1e20)])
    for feature in range(4):
        row = x[0].astype(float).copy()
        row[feature] = np.nan
        audit.append(row)
    db = sqlite3.connect(':memory:')
    db.create_function('f32', 1, lambda v: float(np.float32(v)) if v is not None else None)
    db.execute('CREATE TABLE risks (x0 REAL, x1 REAL, x2 REAL, x3 REAL)')
    db.executemany('INSERT INTO risks VALUES (?, ?, ?, ?)',
                   [[None if np.isnan(v) else float(v) for v in row] for row in audit])
    sql_scores = np.array([row[0] for row in db.execute('SELECT ' + ' + '.join(terms) + ' FROM risks')])
    native = model.predict_raw(np.asarray(audit, dtype=np.float32))
    np.testing.assert_allclose(sql_scores, native, rtol=0, atol=4 * 12 * np.finfo(np.float32).eps)
    db.close()


def test_categorical_missing_rare_and_unseen_are_not_interchangeable():
    import polars as pl
    x = pl.DataFrame({'cat': ['a'] * 100 + ['b'] * 100 + ['rare'] * 2 + [None] * 20})
    y = np.array([0] * 100 + [2] * 100 + [4] * 2 + [6] * 20, dtype=np.float32)
    model = TBoostRegressor(n_trees=30, n_bags=1, graduate=False).fit(x, y)
    bank = json.loads(model.tables(x))
    levels = {v['label']: v['cell'] for v in bank['tables'][0]['axes'][0]['levels']}
    nx, cats = model._serve_design(pl.DataFrame({'cat': ['rare', 'never-seen', None]}))
    cells = np.asarray(model._model.cell_indices(nx, cat_x=cats))[:, 0]
    assert cells[0] == levels['<rare>']
    assert cells[2] == levels['__t_boost_missing__'] and cells[2] != 0
    assert cells[1] != cells[0], 'unseen uses encoder base, not the rare bucket'
