# TODO

## 0. Fixed issues made by refactoring

### Write example to check if there is any issues and show how to use

## 1. Setup of pip
   # . Build the wheel:

    ```bash 
   python setup.py sdist
   ```

   # . Check the wheel:

    ```bash
   twine check dist/*
   ```

   # . Upload the wheel:

   ```bash
   twine upload dist/*
   ```

## 2. Estimator 
    Estimator would be a class like scikit-learn, and user can use them in a easy way, like the following:

    ```python
    import numpy as np
    import torecsys as trs

    # create training data
    x_train = np.array(...) # or even a pandas.DataFrame
    y_train = np.array(...)

    # initialize a model with the specific arguments
    model = trs.estimator.WideAndDeepEstimator(...)

    # fit the model by numpy.array / pandas.DataFrame / torch.tensor
    model.fit(x_train, y_train)

    # make inference
    y_pred = model.predict(x_pred)
    ```

## 3. README, icon, documentations, ...
