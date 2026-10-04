# TBoostClassifier

--8<-- "_snippets/tboostclassifier_signature.md"

## Purpose {#purpose}

Implementation of [the scikit-learn estimator API](https://scikit-learn.org/stable/developers/develop.html)
for t-boost classification. scikit-learn itself is optional: without it, the class runs
standalone with the same methods.

Supports model training, inference and auxiliary calculations like rating tables, prediction
contributions and feature importance.

The model depends on the number of classes in the target:

- Two classes — A logistic model (see
  [Classification: objectives and metrics](../../objectives/classification.md)).
- Three or more classes — A softmax model with one set of rating tables per class (see
  [Multiclassification: objectives and metrics](../../objectives/multiclassification.md)).

## Parameters {#parameters}

See [Training parameters](../../training-parameters/index.md) for the full list of parameters.
`n_trees`, `learning_rate` and `lambda_` can be passed by position; every other parameter is
keyword-only. For multiclassification, see also
[Multiclassification settings](../../training-parameters/multiclassification.md).

## Attributes {#attributes}

### [classes_](attributes.md#classes_)

The class labels seen during `fit`.

### [feature_importances_](attributes.md#feature_importances_)

Return the importance of each input feature, in the order of the input columns.

### [required_columns](attributes.md#required_columns)

The input columns needed to apply the model.

### [n_features_in_](attributes.md#n_features_in_), [feature_names_in_](attributes.md#feature_names_in_)

The number and the names of the features seen during `fit`.

### [categories_](attributes.md#categories_)

The levels of each categorical feature seen during `fit`.

### [link](attributes.md#link)

The link between the raw score and the prediction: `logit` or `softmax`.

### [n_trees_](attributes.md#n_trees_), [n_trees_per_bag_](attributes.md#n_trees_per_bag_)

The number of trees in the model, overall and per bag (binary classification).

### [stopping_reason_](attributes.md#stopping_reason_), [stopping_reason_per_bag_](attributes.md#stopping_reason_per_bag_)

Why the training stopped, overall and per bag (binary classification).

### [evals_result_](attributes.md#evals_result_)

Return the values of metrics calculated during the training.

### [pruning_report_](attributes.md#pruning_report_), [graduation_report_](attributes.md#graduation_report_)

The records of the table selection and of graduation.

### [binding_report_](attributes.md#binding_report_)

Which of the parameters set away from their defaults took effect on the fit.

### [metadata](attributes.md#metadata)

Your own JSON metadata, saved with the model.

## Methods {#methods}

### [fit](fit.md)

Train a model.

### [predict](predict.md)

Apply the model to the given dataset to predict the class labels.

### [predict_proba](predict_proba.md)

Apply the model to the given dataset to predict the probability that the object belongs to the
given classes.

### [decision_function](decision_function.md)

Apply the model to the given dataset and return the raw score on the logit scale.

### [predict_contributions](predict_contributions.md)

Calculate the contribution of each rating table to the prediction for every object.

### [tables](tables.md)

Export the exact rating tables of the model as a JSON document.

### [cell_indices](cell_indices.md)

Return the rating-table cell every object falls in (binary classification).

### [actual_vs_expected](actual_vs_expected.md)

Calculate actual versus expected totals by rating-factor level (binary classification).

### [pricing_report](pricing_report.md)

Return the rating tables, the actual versus expected cells and the stage diagnostics for review
(binary classification).

### [unseen_values](unseen_values.md)

Count the categorical values in the dataset that are absent from the training data.

### [score](score.md)

Calculate the Accuracy metric for the objects in the given dataset.

### [check_bindings](check_bindings.md)

Raise an error if a parameter set away from its default had no effect on the fit.

### [get_params](get_params.md)

Return the values of all training parameters.

### [set_params](set_params.md)

Set the training parameters.

### [to_bytes](to_bytes.md)

Serialize the trained model to a compact binary format.

### [to_json](to_json.md)

Serialize the trained model to a JSON document.

### [from_bytes](from_bytes.md)

Load a model from the binary format written by `to_bytes`.

### [from_json](from_json.md)

Load a model from the JSON document written by `to_json`.
