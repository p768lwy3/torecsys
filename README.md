# ToR[e]cSys

--------------------------------------------------------------------------------

ToR[e]cSys is a Python package which implementing famous recommendation system
algorithm in PyTorch, including Click-through-rate prediction, Learning-to-ranking, 
and Items Embedding.

- [Installation](#installation)
- [Documentation](#documentation)
- [More About ToR[e]cSys](#more-about-torecsys)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Authors](#authors)
- [License](#license)


## Installation
--------------------------------------------------------------------------------
## (!!!Planning)

### by pip package
```bash
# on planning
$ pip install torecsys 
```

### from source
```
git clone https://...
python setup.py build
python setup.py install
```


## Documentation
--------------------------------------------------------------------------------
## (!!!Developing)
The complete documentation for ToR[e]cSys is avaiable via [ReadTheDocs website].  
Thank you for ReadTheDocs!


## More About ToR[e]cSys
--------------------------------------------------------------------------------
## (!!!Planning)
| Component | Description |
| --------- | ----------- |
| [**torecsys.data**] | download sample data, build dataloader, and other functions for convenience |
| [**torecsys.estimators**] | models with embedding, which can be trained with ```.fit(dataloader)``` directly |
| [**torecsys.functional**] | functions used in recommendation system |
| [**torecsys.inputs] | inputs' functions, including embedding, image transformations |
| [**torecsys.layers**] | layers-level implementation of algorithms |
| [**torecsys.losses**] | loss functions used in recommendation system |
| [**torecsys.metrics**] | metrics to evaluate recommendation system |
| [**torecsys.models**] | whole-architecture of models which can be trained by **torecsys.base.trainer** |
| [**torecsys.utils**] | little tools used in torecsys |


### torecsys.models
```torecsys.models``` is a part of model excluding embedding part, so you can choose 
a suitable embedding method for your model with the following codes:

```python
import torecsys as trs

emb = trs.inputs.EmbeddingDict(...)
model = trs.models.WideAndDeepModule(...)
trainer = trs.Trainer(emb, model, ...)
trainer.fit(dataloader)
trainer.predict(test_data)
```

### torecsys.estimators
```torecsys.estimators``` is another type of model to be used directly if the input 
fields and features implemented in the papers are suitable for you:

```python
import torecsys as trs

est = trs.estimators.MatrixFactorization(...)
est.fit(dataloader)
est.predict(test_data)
```


## Getting Started
--------------------------------------------------------------------------------
### Load Sample data
load the movielens dataset, for example:
```python
import torecsys as trs

# Load the movielens dataset
trs.data.download_ml_data(size="latest-small")
ratings, _ = trs.data.load_ml_data(size="latest-small")

```

### Build Dataset and DataLoader with Sample data
```python
import torch
import torch.data.utils
from torecsys.data.dataset import DataFrameToDataset
from torecsys.data.dataloader import trs_collate_fn

# build dataset
dataset = DataFrameToDataset(ratings, ["userId", "movieId"])

# build dataloader
collate_fn = trs_collate_fn(...)
dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn)

for batch in dataloader:
    print(batch) # RETUNRS: ...

```

### Use Estimators to train a model
```python
est = trs.estimators.MatrixFactorizationEstimator(...)
est.fit(dataloader)


```

### Make prediction with estimators
```python
print(est.predict(torch.Tensor([...]))) # RETURNS: 
print(ytrue)

```


## Examples
--------------------------------------------------------------------------------
* [Example X. Building a Matrix Factorization Recommendation System](https://github.com)
* [Example X. Using Factorization-machine-type estimators to make Click Through Rate Prediction](https://github.com)
* [Example X. Building your own model with Modules and Trainer](https://github.com)
* [Example X. StarSpace: Embed All the Things!](https://github.com)


## Authors
--------------------------------------------------------------------------------
* [Jasper Li](https://github.com) - Developer


## License
--------------------------------------------------------------------------------
ToR[e]cSys is MIT-style licensed, as found in the LICENSE file.
