### callbacks

#### Description

Functions called after every iteration of the training of the deployed model, each with a
dictionary:

- `phase` — `"fit"`.
- `bag`, `n_bags` — The bag and the number of bags.
- `round` — The iteration, starting at 1.
- `n_trees` — The maximum number of trees (the `n_trees` parameter).
- `train_deviance`, `eval_deviance` — The mean deviance on the bag's training objects and on
  its validation objects (`None` without validation objects).

The bags are trained in parallel, so calls for different bags interleave. Each bag's iterations
arrive in order, all on the calling thread. A true return value stops every bag after its current
iteration (`stopping_reason_` is then `"callback"`); an exception stops the training and is
raised from `fit`. The deviances of every iteration are kept in `evals_result_`.

**Possible types**

- callable
- list of callables

**Default value**

None
