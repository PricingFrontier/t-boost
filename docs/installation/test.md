# Test t-boost

Use the following example to test t-boost:

```python
import numpy
from t_boost import TBoostRegressor

dataset = numpy.array([[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60], [20, 15, 85, 60]],
                      dtype=numpy.float32)
train_labels = [1.2, 3.4, 9.5, 24.5]
model = TBoostRegressor(learning_rate=1, max_depth=3, objective="squared_error")
fit_model = model.fit(dataset, train_labels)

print(fit_model.predict(dataset))
```
