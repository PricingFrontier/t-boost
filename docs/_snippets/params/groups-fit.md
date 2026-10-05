### groups

#### Description

The entity each object belongs to, for panel data where the same entity (for example, a policy
renewed each year) contributes several similar objects. A string names a column of a polars `X`.

When it is given, whole groups are assigned to one side of every internal split instead of
single objects: the validation objects of early stopping, the bag samples and the pruning
folds. Otherwise near-duplicate objects of the same entity leak across these splits, early
stopping does not trigger, and pruning is biased. A grouping in which every
group has one object is the same as no grouping.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None (every object is its own group)
