# Bagging

t-boost trains [`n_bags`](../training-parameters/bagging.md#n_bags) bags (8 by default) and
averages them into one model. Bagging affects the following important aspects of the training:

- **Regularization**

    Each bag sees a different sample of the objects and of the features, so averaging the bags
    reduces the variance of the model.

- **Honest evidence**

    The objects a bag did not see are its out-of-bag objects. Every object that is out of bag
    for at least one bag can be scored by the bags that did not see it, which gives an estimate
    of the model's performance on unseen data at no extra cost. Pruning, banding and the
    out-of-bag cell refit use this evidence.

## Sampling the objects of a bag {#sampling}

Each bag is trained on a sample of [`bag_subsample`](../training-parameters/bagging.md#bag_subsample)
of the objects (80% by default), drawn without replacement (subagging). A value of 1 or more
draws a full-size bootstrap sample with replacement instead.

The sample is stratified in the same way as the validation objects of the overfitting detector:
by class for classification, and by zero versus non-zero target for the `poisson` and `tweedie`
objectives, so rare classes and rare events are represented in every bag. When `groups` is passed
to `fit`, whole groups are sampled, so the out-of-bag objects never include an object whose group
was in the bag.

Each bag then sets aside its own validation objects for the
[overfitting detector](overfitting-detector.md), unless `groups` or an `eval_set` is given (the
bags then share one validation dataset), and stops at its own best iteration.

## Averaging the bags {#averaging}

The bags are averaged exactly: the score of the averaged model is the mean of the bags' scores,
and its rating tables are the means of the bags' tables. The deployed model is therefore one set
of tables, whatever the number of bags: a prediction looks up those tables once instead of
evaluating each bag.

The spread between the bags measures the model's noise, which [banding](readable-tables.md#banding)
uses to decide how coarse the bands can be.
