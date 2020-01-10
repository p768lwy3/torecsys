# ToR[e]cSys

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

The complete documentation for ToR[e]cSys is avaiable via [ReadTheDocs website](https://torecsys.readthedocs.io/en/latest/). \
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
| [Positon-bias aware learning framework](torecsys/models/ctr/position_bias_aware_learning_framework.py) | [Huifeng Guo et al, 2019. PAL: a position-bias aware learning framework for CTR prediction in live recommender systems](https://dl.acm.org/citation.cfm?id=3347033&dl=ACM&coll=DL) | 2019 |

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

TBU

## Examples

### Sample Codes

TBU

### Sample of Experiments

TBU

## Authors

- [Jasper Li](https://github.com) - Developer

## License

ToR[e]cSys is MIT-style licensed, as found in the LICENSE file.
