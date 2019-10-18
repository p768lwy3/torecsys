# TODO

## -3. Build another new sample dataset for debugging and testing

but what should i build?

## -2. Planning of `Estimator` and `Trainer`

### 1. Estimator would be a class like scikit-learn, and user can use them in a easy way, like the following:

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

### 2. Trainer would be a class more flexible, and user can modify the model in their own way, like the following:

    ```python

    import torecsys as trs

    # initialize a dataloadaer to iterate training data
    dataloader = trs.data.build_dataloader(x=x_train, y=y_train, ...)

    # initialize a inputs class with a schema and a model class with a structure
    inputs = trs.inputs.InputWrapper(...)
    model = trs.models.WideAndDeep(...)

    # initialize a trainer class with the inputs class and the model class
    trainer = trs.Trainer(inputs=inputs, model=model)

    # train the model by the dataloader
    trainer.train(dataloader)

    # make inference
    y_pred = trainer.predict(x_pred)
    ```

## -1. Update from PyTorch 1.2 to PyTorch 1.3

1. `Named Tensor` is a extremely cool feature to make the code much more readable and debuggable. Planning to change the parts which can be written in `Named Tensor`. And Hope everything can be written in `Named Tensor` in the near future!

## 0. Project

1. Finish the module
2. Examples to use the module
3. Setup of pypl
4. README, icon, documentations, ...

## 1. Module

### torecsys

1. .layers.* -> .models.\*.layers.\*
2. .estimators.* -> .models.\*.estimators.\*
3. .losses.* -> .models.\*.losses.\*

### torecsys.data

1. .sampledata
2. .dataset
3. .dataloader
4. .negsampling
5. .subsampling

### torecsys.models

1. .inputs
    * .AudioInputs
    * .EmbeddingDict
    * .FieldAwareIndexEmbedding
    * .ImageInputs
    * .ImagesListInputs
    * .ListIndexEmbedding
    * .PretrainedImageEmbedding
    * .PretrainedTextEmbedding
    * .SequenceIndexEmbedding
    * .SingleIndexEmbedding
    * .StackedInputs
    * .ValueInputs

2. .ctr (Click Through Rate Prediction)
    1. .base
        * _CtrModel
    2. .estimators
        * NeuralCollaborativeFiltering: https://arxiv.org/pdf/1708.05031.pdf
        * NeuralFactorizationMachine : https://arxiv.org/pdf/1708.05027.pdf
        * ProductNeuralNetwork : https://arxiv.org/abs/1611.00144
        * WideAndDeep : https://arxiv.org/abs/1606.07792
        * xDeepFM : https://arxiv.org/abs/1803.05170

3. .emb (Embedding)
    1. .base
        * _EmbModel
    2. .layers
        * MatrixFactorization
        * SingularValueDecomposition
        * CBOW
        * SkipGram
        * Starspace
        * Node2Vec
    3. .estimators
        * MatrixFactorization
        * SingularValueDecomposition
        * CBOW
        * SkipGram
        * Starspace
        * Node2Vec
        * AttentionWalk

4. .ltr (Learning to Rank)
    1. base
        * _LtrModel
    2. losses
        * BPR
        * Triplet
    3. layers
        * pointwise
        * pairwise
        * listwise
    4. estimators
        * pointwise
        * pairwise
        * listwise

5. .metrics
    1. .accuracy
    2. .recall
    3. .precision
    4. .nauc

### torecsys.utils

1. .logging.tqdm
    * .TqdmHandler
2. .cli
    * .train
        * ```torecsys --mode="train" --filepath="./dataset/example.csv" --model="./model/modeldir"```
    * .eval
        * ```torecsys --model="eval" --filepath="./dataset/example.csv" --model="./model/modeldir"```
    * .test
        * ```torecsys --model="test" --filepath="./dataset/example.csv" --model="./model/modeldir"```
    * .serve
        * ```torecsys --model="serve" --filepath="./dataset/example.csv" --model="./model/modeldir" --host="localhost" --port=8080```
