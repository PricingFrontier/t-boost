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
arrive in order, all on the calling thread. A true return value stops that bag (`bag` in the
dictionary) after the iteration, just as early stopping stops each bag at its own iteration
(`stopping_reason_per_bag_` is then `"callback"`). To stop the whole training, return a true
value for every bag. A function that decides from the dictionary alone stops the same bags at the
same iterations on every run. An exception stops the training and is raised from `fit`. Both
deviances of every iteration are kept in `evals_result_`. The deviance on the training objects
takes an extra pass per iteration, 5–11% of the training time.

**Possible types**

- callable
- list of callables

**Default value**

None
