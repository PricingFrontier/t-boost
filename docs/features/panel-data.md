# Panel data

In panel data the same entity contributes several similar objects: for example, a policy that
renews every year. A split of such data by object puts near-duplicate objects of the same
entity on both sides. The validation dataset then looks like the training data, the overfitting
detector does not trigger, and pruning keeps tables it should drop.

Pass the entity of each object in the `groups` parameter of `fit`:

```python
model = TBoostRegressor(objective="poisson")
model.fit(train_data, "ClaimCount", exposure="Exposure", groups="PolicyID")
```

Whole groups are then assigned to one side of every internal split: the validation objects of
the overfitting detector, the bag samples and their out-of-bag objects, and the pruning folds.
The `groups` column is not used as a feature. A grouping in which every group has one object is
the same as no grouping.
