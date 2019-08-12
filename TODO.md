# TODO:

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
    2. .layers
        * AttentionFactorizationMachine 
        * BiInteraction -> torecsys.layers.FactorizationMachine
        * CIN
        * CrossNetwork
        * Dense
        * FactorizationMachine
        * FieldAwareFactorizationMachine
        * GeneralizedMatrixFactorization -> torecsys.layers.MatrixFactorization
        * InnerProduct
        * Linear
        * MultiLayerPerceptron -> torecsys.layers.Dense
        * OuterProduct
    3. .estimators
        * AttentionFactorizationMachine : https://arxiv.org/abs/1708.04617
        * DeepAndCross : https://arxiv.org/abs/1708.05123
        * DeepMCP : https://arxiv.org/abs/1906.04365
        * DeepFFM : https://arxiv.org/abs/1905.06336
        * DeepFM  : https://arxiv.org/abs/1703.04247
        * FactorizationMachine : https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
        * FactorizationSupportedNeuralNetwork : https://arxiv.org/abs/1601.02376
        * FATDeepFFM (use conv1d kernel 1x1 to compose kxn field matrix to 1xn vector) : https://arxiv.org/abs/1905.06336
        * FieldAwareFactorizationMachine : https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
        * FieldAwareNeuralFactorizationMachine : https://arxiv.org/abs/1902.09096
        * LogisticRegression
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
    