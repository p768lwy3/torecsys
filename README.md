# ToR[e]cSys

--------------------------------------------------------------------------------

ToR[e]cSys is a Python package which implementing famous recommendation system \
algorithm in PyTorch, including Click-through-rate prediction, Learning-to-ranking, \
and Items Embedding.

- [Installation](#installation)
- [Implemented Models](#implemented-models)
- [Documentation](#documentation)
- [More About ToR[e]cSys](#more-about-torecsys)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Authors](#authors)
- [License](#license)

## Installation

### By pip package

```bash
pip install torecsys
```

### From source

```bash
git clone https://github.com/p768lwy3/torecsys.git
python setup.py build
python setup.py install
```

### Build Documentation

```bash
git clone https://github.com/p768lwy3/torecsys.git
cd ./torecsys/doc
./make html
```

## Documentation

The complete documentation for ToR[e]cSys is avaiable via [ReadTheDocs website](https://torecsys.readthedocs.io/en/latest/).  
Thank you for ReadTheDocs!!!

## Implemented Models

| Model Name | Research Paper | Type |
| ---------- | -------------- | ---- |
| [Attentional Factorization Machine](torecsys/models/ctr/attentional_factorization_machine.py) | [Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/abs/1708.04617) | Click Through Rate |
| [Deep and Cross Network](torecsys/models/ctr/deep_and_cross_network.py) | [Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | Click Through Rate |
| [Deep Field-Aware Factorization Machine](torecsys/models/ctr/deep_ffm.py) | [Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine](https://arxiv.org/abs/1905.06336) | Click Through Rate |
| [Deep Factorization Machine](torecsys/models/ctr/deep_fm.py) | [Huifeng Guo et al, 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | Click Through Rate |
| [Deep Matching Correlation Prediction](torecsys/models/ctr/deep_mcp.py) | [Wentao Ouyang et al, 2019. Representation Learning-Assisted Click-Through Rate Prediction](https://arxiv.org/pdf/1906.04365.pdf) | Click Through Rate |
| [Elaborated Entire Space Supervised Multi Task Model](torecsys/models/ctr/elaborated_entire_space_supervised_multi_task.py) | [Hong Wen et al, 2019. Conversion Rate Prediction via Post-Click Behaviour Modeling](https://arxiv.org/abs/1910.07099) | Click Through Rate |
| [Entire Space Multi Task Model](torecsys/models/ctr/entire_space_multi_task.py) | [Xiao Ma et al, 2019. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931) | Click Through Rate |
| [Factorization Machine](torecsys/models/ctr/factorization_machine.py) | [Steffen Rendle, 2010. Factorization Machine](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | Click Through Rate |
| [Factorization Machine Support Neural Network](torecsys/models/ctr/factorization_machine_supported_neural_network.py) | [Weinan Zhang et al, 2016. Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) | Click Through Rate |
| [Field Attentive Deep Field Aware Factorization Machine](torecsys/models/ctr/fat_deep_ffm.py) | [Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine](https://arxiv.org/abs/1905.06336)  | Click Through Rate |
| [Field-Aware Factorization Machine](torecsys/models/ctr/field_aware_factorization_machine.py) | [Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) | Click Through Rate |
| [Logistic Regression](torecsys/models/ctr/logistic_regression.py) | / | Click Through Rate |
| [Neural Collaborative Filtering](torecsys/models/ctr/neural_collaborative_filtering.py) | [Xiangnan He, 2017. Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) | Click Through Rate |
| [Neural Factorization Machine](torecsys/models/ctr/neural_factorization_machine.py) | [Xiangnan He et al, 2017. Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027) | Click Through Rate |
| [Product Neural Network](torecsys/models/ctr/product_neural_network.py) | [Yanru QU, 2016. Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) | Click Through Rate |
| [eXtreme Deep Factorization Machine](torecsys/models/ctr/xdeep_fm.py) | [Jianxun Lian et al, 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170.pdf) | Click Through Rate |
| [Deep Session Interest Network](torecsys/models/ctr/deep_session_interest_network.py) | [Yufei Feng, 2019. Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1905.06482) | Click Through Rate |
| [Positon-bias aware learning framework](torecsys/models/ctr/position_bias_aware_learning_framework.py) | [PAL: a position-bias aware learning framework for CTR prediction in live recommender systems](https://dl.acm.org/citation.cfm?id=3347033&dl=ACM&coll=DL) | Click Through Rate |
| [Matrix Factorization](torecsys/models/emb/matrix_factorization.py) | / | Embedding |
| [Starspace](torecsys/models/emb/starspace.py)| [Ledell Wu et al, 2017 StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856) | Embedding |
| [Personalized Re-ranking Model](torecsys/models/ltr/personalized_reranking.py) | [Personalized Re-ranking for Recommendation](https://arxiv.org/abs/1904.06813) | Learning to Rank |

## More About ToR[e]cSys

| Component | Description |
| --------- | ----------- |
| [**torecsys.data**] | download sample data, build dataloader, and other functions for convenience |
| [**torecsys.estimators**] | models with embedding, which can be trained with ```.fit(dataloader)``` directly |
| [**torecsys.functional**] | functions used in recommendation system |
| [**torecsys.inputs**] | inputs' functions, including embedding, image transformations |
| [**torecsys.layers**] | layers-level implementation of algorithms |
| [**torecsys.losses**] | loss functions used in recommendation system |
| [**torecsys.metrics**] | metrics to evaluate recommendation system |
| [**torecsys.models**] | whole-architecture of models which can be trained by **torecsys.base.trainer** |
| [**torecsys.utils**] | little tools used in torecsys |

(!!! To be confirmed)

### torecsys.models

```torecsys.models``` is a part of model excluding embedding part, so you can choose \
a suitable embedding method for your model with the following codes:

### torecsys.estimators

```torecsys.estimators``` is another type of model to be used directly if the input \
fields and features implemented in the papers are suitable for you:

## Getting Started

(!!! To be confirmed)

### Load Sample data

load the movielens dataset, for example:

### Build Dataset and DataLoader with Sample data

### Use Estimators to train a model

### Make prediction with estimators

## Examples

## Authors

- [Jasper Li](https://github.com) - Developer

## License

ToR[e]cSys is MIT-style licensed, as found in the LICENSE file.
