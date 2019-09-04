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
| [Factorization Machine](torecsys/models/ctr/factorization_machine.py) | [Steffen Rendle, 2010. Factorization Machine](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | Click Through Rate |
| [Factorization Machine Support Neural Network](torecsys/models/ctr/factorization_machine_supported_neural_network.py) | [Weinan Zhang et al, 2016. Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) | Click Through Rate |
| [Field-Aware Factorization Machine](torecsys/models/ctr/field_aware_factorization_machine.py) | [Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) | Click Through Rate |
| [Field-Aware Neural Factorization Machine](torecsys/models/ctr/field_aware_neural_factorization_machine.py) | [Li Zhang et al, 2019. Field-aware Neural Factorization Machine for Click-Through Rate Prediction](https://arxiv.org/abs/1902.09096) | Click Through Rate |

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
