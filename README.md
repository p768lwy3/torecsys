# ToR[e]cSys

--------------------------------------------------------------------------------

## News

It is happy to know the new package of [Tensorflow Recommenders](https://www.tensorflow.org/recommenders).

--------------------------------------------------------------------------------

ToR[e]cSys is a PyTorch Framework to implement recommendation system algorithms, including but not limited to click-through-rate (CTR) prediction, learning-to-ranking (LTR), and Matrix/Tensor Embedding. The project objective is to develop a ecosystem to experiment, share, reproduce, and deploy in real world in a smooth and easy way (Hope it can be done).

- [Installation](#installation)
- [Implemented Models](#implemented-models)
- [Documentation](#documentation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Authors](#authors)
- [License](#license)

## Installation

TBU

## Documentation

The complete documentation for ToR[e]cSys is available via [ReadTheDocs website](https://torecsys.readthedocs.io/en/latest/). \
Thank you for ReadTheDocs! You are the best!

## Implemented Models

### 1. Subsampling

| Model Name | Research Paper | Year |
| ---------- | -------------- | ---- |
| Word2Vec   | [Omer Levy et al, 2015. Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://levyomer.files.wordpress.com/2015/03/improving-distributional-similarity-tacl-2015.pdf) | 2015 |

### 2. Negative Sampling

| Model Name | Research Paper | Year |
| ---------- | -------------- | ---- |
| TBU        |                |      |

### 3. Click through Rate (CTR) Model

| Model Name | Research Paper | Year |
| ---------- | -------------- | ---- |
| [Logistic Regression](torecsys/models/ctr/logistic_regression.py) | / | / |
| [Factorization Machine](torecsys/models/ctr/factorization_machine.py) | [Steffen Rendle, 2010. Factorization Machine](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | 2010 |
| [Factorization Machine Support Neural Network](torecsys/models/ctr/factorization_machine_supported_neural_network.py) | [Weinan Zhang et al, 2016. Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) | 2016 |
| [Field-Aware Factorization Machine](torecsys/models/ctr/field_aware_factorization_machine.py) | [Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) | 2016 |
| [Product Neural Network](torecsys/models/ctr/product_neural_network.py) | [Yanru QU et al, 2016. Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) | 2016 |
| [Attentional Factorization Machine](torecsys/models/ctr/attentional_factorization_machine.py) | [Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/abs/1708.04617) | 2017 |
| [Deep and Cross Network](torecsys/models/ctr/deep_and_cross_network.py) | [Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | 2017 |
| [Deep Factorization Machine](torecsys/models/ctr/deep_fm.py) | [Huifeng Guo et al, 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | 2017 |
| [Neural Collaborative Filtering](torecsys/models/ctr/neural_collaborative_filtering.py) | [Xiangnan He et al, 2017. Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) | 2017 |
| [Neural Factorization Machine](torecsys/models/ctr/neural_factorization_machine.py) | [Xiangnan He et al, 2017. Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027) | 2017 |
| [eXtreme Deep Factorization Machine](torecsys/models/ctr/xdeep_fm.py) | [Jianxun Lian et al, 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170.pdf) | 2018 |
| [Deep Field-Aware Factorization Machine](torecsys/models/ctr/deep_ffm.py) | [Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine](https://arxiv.org/abs/1905.06336) | 2019 |
| [Deep Matching Correlation Prediction](torecsys/models/ctr/deep_mcp.py) | [Wentao Ouyang et al, 2019. Representation Learning-Assisted Click-Through Rate Prediction](https://arxiv.org/pdf/1906.04365.pdf) | 2019 |
| [Deep Session Interest Network](torecsys/models/ctr/deep_session_interest_network.py) | [Yufei Feng et al, 2019. Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1905.06482) | 2019 |
| [Elaborated Entire Space Supervised Multi Task Model](torecsys/models/ctr/elaborated_entire_space_supervised_multi_task.py) | [Hong Wen et al, 2019. Conversion Rate Prediction via Post-Click Behaviour Modeling](https://arxiv.org/abs/1910.07099) | 2019 |
| [Entire Space Multi Task Model](torecsys/models/ctr/entire_space_multi_task.py) | [Xiao Ma et al, 2019. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931) | 2019 |
| [Field Attentive Deep Field Aware Factorization Machine](torecsys/models/ctr/fat_deep_ffm.py) | [Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine](https://arxiv.org/abs/1905.06336)  | 2019 |
| [Position-bias aware learning framework](torecsys/models/ctr/position_bias_aware_learning_framework.py) | [Huifeng Guo et al, 2019. PAL: a position-bias aware learning framework for CTR prediction in live recommender systems](https://dl.acm.org/citation.cfm?id=3347033&dl=ACM&coll=DL) | 2019 |

### 4. Embedding Model

| Model Name | Research Paper | Year |
| ---------- | -------------- | ---- |
| [Matrix Factorization](torecsys/models/emb/matrix_factorization.py) | / | / |
| [Starspace](torecsys/models/emb/starspace.py)| [Ledell Wu et al, 2017 StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856) | 2017 |

### 5. Learning-to-Rank (LTR) Model

| Model Name | Research Paper | Year |
| ---------- | -------------- | ---- |
| [Personalized Re-ranking Model](torecsys/models/ltr/personalized_reranking.py) | [Changhua Pei et al, 2019. Personalized Re-ranking for Recommendation](https://arxiv.org/abs/1904.06813) | 2019 |

## Getting Started

There are several ways using ToR[e]cSys to develop a Recommendation System. Before talking about them, we first need to discuss about components of ToR[e]cSys.

A model in ToR[e]cSys is constructed by two parts mainly: inputs and model, and they will be wrapped into a sequential module ([torecsys.models.sequential](https://github.com/p768lwy3/torecsys/blob/master/torecsys/models/sequential.py)) to be trained by Trainer ([torecsys.trainer.Trainer](https://github.com/p768lwy3/torecsys/blob/master/torecsys/trainer/trainer.py)). \

For inputs module ([torecsys.inputs](https://github.com/p768lwy3/torecsys/tree/master/torecsys/inputs)), it will handle most kinds of inputs in recommendation system, like categorical features, images, etc, with several kinds of methods, including token embedding, pre-trained image models, etc.

For models module ([torecsys.models](https://github.com/p768lwy3/torecsys/tree/master/torecsys/models)), it will implement some famous models in recommendation system, like Factorization Machine family. I hope I can make the library rich. To construct a model in the module, in addition to the modules implemented in [PyTorch](https://pytorch.org/docs/stable/nn.html), I will also implement some layers in [torecsys.layers](https://github.com/p768lwy3/torecsys/tree/master/torecsys/layers) which are called by models usually.

After the explanation of ToR[e]cSys, let's move on to the `Getting Started`. We can use ToR[e]cSys in the following ways:

1. Run by command-line (In development)

    ```bash
> torecsys build --inputs_config='{}' \
--model_config='{"method":"FM", "embed_size": 8, "num_fields": 2}' \
--regularizer_config='{"weight_decay": 0.1}' \
--criterion_config='{"method": "MSELoss"}' \
--optimizer_config='{"method": "SGD", "lr": "0.01"}' \
...
    ```

2. Run by class method

    ```python
import torecsys as trs

# build trainer by class method
trainer = trs.trainer.Trainer() \
    .bind_objective("CTR") \
    .set_inputs() \
    .set_model(method="FM", embed_size=8, num_fields=2) \
    .set_sequential() \
    .set_regularizer(weight_decay=0.1) \
    .build_criterion(method="MSELoss") \
    .build_optimizer(method="SGD", lr="0.01") \
    .build_loader(name="train", ...) \
    .build_loader(name="eval", ...) \
    .set_targets_name("labels") \
    .set_max_num_epochs(10) \
    .use_cuda()

# start to fit the model
trainer.fit()
    ```

3. Run like PyTorch Module

    ```python
import torch
import torch.nn as nn
import torecsys as trs

# some codes here
inputs = trs.inputs.InputsWrapper(schema=schema)
model = trs.models.FactorizationMachineModel(embed_size=8, num_fields=2)

for i in range(epochs):
    optimizer.zero_grad()
    outputs = model(**inputs(batches))
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    ```

(In development) You can anyone you like to train a Recommender System and serve it in the following ways:

1. Run by command-line

    ```bash
    > torecsys serve --load_from='{}'
    ```

2. Run by class method

    ```python
import torecsys as trs

serving = trs.serving.Model() \
    .load_from(filepath=filepath)
    .run()
    ```

3. Serve it yourself

    ```python
from flask import Flask, request
import torecsys as trs

model = trs.serving.Model() \
    .load_from(filepath=filepath)

@app.route("/predict")
def predict():
    args = request.json
    inference = model.predict(args)
    return inference, 200

if __name__ == "__main__":
    app.run()
    ```

For further details, please refer to the [example](https://github.com/p768lwy3/torecsys/tree/master/example) in repository or read the [documentation](https://torecsys.readthedocs.io/en/latest/). Hope you enjoy~

## Examples

TBU

### Sample Codes

TBU

### Sample of Experiments

TBU

## Authors

- [Jasper Li](https://github.com/p768lwy3) - Developer

## License

ToR[e]cSys is MIT-style licensed, as found in the LICENSE file.
