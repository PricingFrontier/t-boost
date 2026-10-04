!!! note

    The model prediction results will be correct only if the `X` parameter with feature values
    contains all the features used in the model. For a polars `DataFrame` or `LazyFrame`, the
    features are matched by name: extra columns are ignored and the order of the columns does
    not matter (a `LazyFrame` collects only the columns the model needs). For other types, and
    for a model trained without feature names, the features must be in the same number and
    order as the columns provided during the training.
