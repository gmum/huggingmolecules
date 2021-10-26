# HuggingMolecules

<p align="center">
  <img src="https://user-images.githubusercontent.com/38813586/117339001-85a2bc80-ae9f-11eb-9adb-7c4477cfc5bb.png" width="200"/>
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

We envision models that are pre-trained on a vast range of domain-relevant tasks to become key for molecule property
prediction. This repository aims to give easy access to state-of-the-art pre-trained models.

## Quick tour

To quickly fine-tune a model on a dataset using the pytorch lightning package follow the below example based on the MAT
model and the freesolv dataset:

```python
from huggingmolecules import MatModel, MatFeaturizer

# The following import works only from the source code directory:
from experiments.src import TrainingModule, get_data_loaders

from torch.nn import MSELoss
from torch.optim import Adam

from pytorch_lightning import Trainer
from pytorch_lightning.metrics import MeanSquaredError

# Build and load the pre-trained model and the appropriate featurizer:
model = MatModel.from_pretrained('mat_masking_20M')
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')

# Build the pytorch lightning training module:
pl_module = TrainingModule(model,
                           loss_fn=MSELoss(),
                           metric_cls=MeanSquaredError,
                           optimizer=Adam(model.parameters()))

# Build the data loader for the freesolv dataset:
train_dataloader, _, _ = get_data_loaders(featurizer,
                                          batch_size=32,
                                          task_name='ADME',
                                          dataset_name='hydrationfreeenergy_freesolv')

# Build the pytorch lightning trainer and fine-tune the module on the train dataset:
trainer = Trainer(max_epochs=100)
trainer.fit(pl_module, train_dataloader=train_dataloader)

# Make the prediction for the batch of SMILES strings:
batch = featurizer(['C/C=C/C', '[C]=O'])
output = pl_module.model(batch)
```

## Installation

Create your conda environment and install the rdkit package:

```
conda create -n huggingmolecules python=3.8.5
conda activate huggingmolecules
conda install -c conda-forge rdkit==2020.09.1
```

Then install huggingmolecules from the cloned directory:

```
conda activate huggingmolecules
pip install -e ./src
```

Huggingmolecules caches weights and configs of the models. To avoid issues with incompatibility of different package versions, it is recommended to clean up the cache directory after every package update:
```
python -m src.clean_cache --all
```

## Project Structure

The project consists of two main modules: `src/` and `experiments/` modules:

* The `src/` module contains abstract interfaces for pre-trained models along with their implementations based on the
  pytorch library. This module makes configuring, downloading and running existing models easy and out-of-the-box.
* The `experiments/` module makes use of abstract interfaces defined in the `src/` module and implements scripts based
  on the pytorch lightning package for running various experiments. This module makes training, benchmarking and
  hyper-tuning of models flawless and easily extensible.

## Supported models architectures

Huggingmolecules currently provides the following models architectures:

* [MAT](https://github.com/ardigen/MAT)
* [GROVER](https://github.com/tencent-ailab/grover)
* [R-MAT](https://arxiv.org/abs/2110.05841) (weights were obtained by joint efforts with Nvidia)

For ease of benchmarking, we also include wrappers in the `experiments/` module for three other models architectures:

* [chemprop](https://github.com/chemprop/chemprop)
* [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry)
* [MolBERT](https://github.com/BenevolentAI/MolBERT)

## The src/ module

The implementations of the models in the `src/` module are divided into three modules: configuration, featurization and
models module. The relation between these modules is shown on the following examples based on the MAT model:

### Configuration examples

```python
from huggingmolecules import MatConfig

# Build the config with default parameters values, 
# except 'd_model' parameter, which is set to 1200:
config = MatConfig(d_model=1200)

# Build the pre-defined config:
config = MatConfig.from_pretrained('mat_masking_20M')

# Build the pre-defined config with 'init_type' parameter set to 'normal':
config = MatConfig.from_pretrained('mat_masking_20M', init_type='normal')

# Save the pre-defined config with the previous modification:
config.save_to_cache('mat_masking_20M_normal.json')

# Restore the previously saved config:
config = MatConfig.from_pretrained('mat_masking_20M_normal.json')
```

### Featurization examples

```python
from huggingmolecules import MatConfig, MatFeaturizer

# Build the featurizer with pre-defined config:
config = MatConfig.from_pretrained('mat_masking_20M')
featurizer = MatFeaturizer(config)

# Build the featurizer in one line:
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')

# Encode (featurize) the batch of two SMILES strings: 
batch = featurizer(['C/C=C/C', '[C]=O'])
```

### Models examples

```python
from huggingmolecules import MatConfig, MatFeaturizer, MatModel

# Build the model with the pre-defined config:
config = MatConfig.from_pretrained('mat_masking_20M')
model = MatModel(config)

# Load the pre-trained weights 
# (which do not include the last layer of the model)
model.load_weights('mat_masking_20M')

# Build the model and load the pre-trained weights in one line:
model = MatModel.from_pretrained('mat_masking_20M')

# Encode (featurize) the batch of two SMILES strings: 
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')
batch = featurizer(['C/C=C/C', '[C]=O'])

# Feed the model with the encoded batch:
output = model(batch)

# Save the weights of the model (usually after the fine-tuning process):
model.save_weights('tuned_mat_masking_20M.pt')

# Load the previously saved weights
# (which now includes all layers of the model):
model.load_weights('tuned_mat_masking_20M.pt')

# Load the previously saved weights, but without 
# the last layer of the model ('generator' in the case of the 'MatModel')
model.load_weights('tuned_mat_masking_20M.pt', excluded=['generator'])

# Build the model and load the previously saved weights:
config = MatConfig.from_pretrained('mat_masking_20M')
model = MatModel.from_pretrained('tuned_mat_masking_20M.pt',
                                 excluded=['generator'],
                                 config=config)
```

### Running tests

To run base tests for `src/` module, type:

```
pytest src/ --ignore=src/tests/downloading/
```

To additionally run tests for downloading module (which will download all models to your local computer and therefore
may be slow), type:

```
pytest src/tests/downloading
```

## The experiments/ module

### Requirements

In addition to dependencies defined in the `src/` module, the `experiments/` module goes along with few others. To
install them, run:

```pip install -r experiments/requirements.txt```

The following packages are crucial for functioning of the `experiments/` module:

* [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [optuna](https://github.com/optuna/optuna)
* [gin-config](https://github.com/google/gin-config)
* [TDC](https://github.com/mims-harvard/TDC)

### Neptune.ai

In addition, we recommend installing the neptune.ai package:

1. Sign up to neptune.ai at https://neptune.ai/.

1. Get your Neptune API token (see
   [getting-started](https://docs.neptune.ai/getting-started/quick-starts/hello-world.html) for help).
1. Export your Neptune API token to ```NEPTUNE_API_TOKEN``` environment variable.
1. Install neptune-client:
   ```pip install neptune-client```.
1. Enable neptune.ai in the ```experiments/configs/setup.gin``` file.
1. Update ```neptune.project_name``` parameters in ```experiments/configs/bases/*.gin``` files.

### Running scripts:

We recommend running experiments scripts from the source code. For the moment there are three scripts implemented:

* ```experiments/scripts/train.py``` - for training with the pytorch lightning package
* ```experiments/scripts/tune_hyper.py``` - for hyper-parameters tuning with the optuna package
* ```experiments/scripts/benchmark.py``` - for benchmarking based on the hyper-parameters tuning (grid-search)

In general running scripts can be done with the following syntax:

```
python -m experiments.scripts.<script_name> /
       -d <dataset_name> / 
       -m <model_name> /
       -b <parameters_bindings>
```

Then the script ```<script_name>.py``` runs with functions/methods parameters values defined in the following gin-config
files:

1. ```experiments/configs/bases/<script_name>.gin```
2. ```experiments/configs/datasets/<dataset_name>.gin```
3. ```experiments/configs/models/<model_name>.gin```

If the binding flag ```-b``` is used, then bindings defined in ```<parameters_binding>``` overrides corresponding
bindings defined in above gin-config files.

So for instance, to fine-tune the MAT model (pre-trained on masking_20M task) on the freesolv dataset using GPU 1,
simply run:

```
python -m experiments.scripts.train /
       -d freesolv / 
       -m mat /
       -b model.pretrained_name=\"mat_masking_20M\"#train.gpus=[1]
```

or equivalently:

```
python -m experiments.scripts.train /
       -d freesolv / 
       -m mat /
       --model.pretrained_name mat_masking_20M /
       --train.gpus [1]
```

### Local dataset

To use a local dataset, create an appropriate gin-config file in the ```experiments/configs/datasets``` directory and
specify the ```data.data_path``` parameter within. For details see
the [get_data_split](https://github.com/panpiort8/huggingmolecules/blob/3a06204fa23ec554f4c633bf4b1751743e1fdeaf/experiments/src/training/training_utils.py#L217)
implementation.

### Benchmarking

For the moment there is one benchmark available. It works as follows:

* ```experiments/scripts/benchmark.py```: on the given dataset we fine-tune the given model on 10 learning rates and 6
  seeded data splits (60 fine-tunings in total). Then we choose that learning rate that minimizes an averaged (on 6 data
  splits) validation metric (metric computed on the validation dataset, e.g. RMSE). The result is the averaged value of
  test metric for the chosen learning rate.

Running a benchmark is essentially the same as running any other script from the `experiments/` module. So for instance
to benchmark the vanilla MAT model (without pre-training) on the Caco-2 dataset using GPU 0, simply run:

```
python -m experiments.scripts.benchmark /
       -d caco2 / 
       -m mat /
       --model.pretrained_name None /
       --train.gpus [0]
```

However, the above script will only perform 60 fine-tunings. It won't compute the final benchmark result. To do that wee
need to run:

```
python -m experiments.scripts.benchmark --results_only /
       -d caco2 / 
       -m mat
```

The above script won't perform any fine-tuning, but will only compute the benchmark result. If we had neptune enabled
in ```experiments/configs/setup.gin```, all data necessary to compute the result will be fetched from the neptune
server.

## Benchmark results

We performed the benchmark described in [Benchmarking](#Benchmarking) as ```experiments/scripts/benchmark.py``` for
various models architectures and pre-training tasks.

### Summary

We report mean/median ranks of tested models across all datasets (both regression and classification ones). For detailed
results see [Regression](#Regression) and [Classification](#Classification) sections.

model | mean rank | rank std |
--- | :---: | :---:
MAT 200k     |  5.6 |  3.5 |
MAT 2M       |  5.3 |  3.4 |
MAT 20M      |  4.1 |  2.2 |
GROVER Base  | 3.8 |  2.7 |
GROVER Large |  **3.6** |  2.4 |
ChemBERTa    |  7.4 |  2.8 |
MolBERT      |  5.9 |  2.9 |
D-MPNN       |  6.3 |  2.3 |
D-MPNN 2d    |  6.4 |  2.0 |
D-MPNN mc    |  5.3 |  2.1 |

### Regression

As the metric we used MAE for QM7 and RMSE for the rest of datasets.

model | FreeSolv | Caco-2 | Clearance | QM7 | Mean rank
--- | :---: | :---: | :---: | :---: | :---:
MAT 200k      | 0.913 ± 0.196 | **0.405 ± 0.030** | 0.649 ± 0.341 | 87.578 ± 15.375 | 5.25
MAT 2M        | 0.898 ± 0.165 | 0.471 ± 0.070 | 0.655 ± 0.327 | 81.557 ± 5.088 | 6.75
MAT 20M       | **0.854 ± 0.197** | 0.432 ± 0.034 | 0.640 ± 0.335 | 81.797 ± 4.176 | 5.0
Grover Base   | 0.917 ± 0.195 | 0.419 ± 0.029 | 0.629 ± 0.335 | **62.266 ± 3.578** | 3.25
Grover Large  | 0.950 ± 0.202 | 0.414 ± 0.041 | **0.627 ± 0.340** | 64.941 ± 3.616 | **2.5**
ChemBERTa     | 1.218 ± 0.245 | 0.430 ± 0.013 | 0.647 ± 0.314 | 177.242 ± 1.819 | 8.0
MolBERT       | 1.027 ± 0.244 | 0.483 ± 0.056 | 0.633 ± 0.332 | 177.117 ± 1.799 | 8.0
Chemprop      | 1.061 ± 0.168 | 0.446 ± 0.064 | 0.628 ± 0.339 | 74.831 ± 4.792 | 5.5
Chemprop 2d <sup>1</sup>  | 1.038 ± 0.235 | 0.454 ± 0.049 | 0.628 ± 0.336 | 77.912 ± 10.231 | 6.0
Chemprop mc <sup>2</sup> | 0.995 ± 0.136 | 0.438 ± 0.053 | **0.627 ± 0.337** | 75.575 ± 4.683 | 4.25

<sup>1</sup> chemprop with additional *rdkit_2d_normalized* features generator  
<sup>2</sup> chemprop with additional *morgan_count* features generator

### Classification

We used ROC AUC as the metric.

model | HIA | Bioavailability | PPBR | Tox21 (NR-AR) | BBBP | Mean rank
--- | :---: | :---: | :---: | :---: | :---: | :---:
MAT 200k      | **0.943 ± 0.015** |0.660 ± 0.052 | 0.896 ± 0.027 | 0.775 ± 0.035 | 0.709 ± 0.022 | 5.8
MAT 2M        | 0.941 ± 0.013 | 0.712 ± 0.076 | **0.905 ± 0.019** | **0.779 ± 0.056** | 0.713 ± 0.022 | 4.2
MAT 20M       | 0.935 ± 0.017 | 0.732 ± 0.082 | 0.891 ± 0.019 | **0.779 ± 0.056** | 0.735 ± 0.006 | **3.4**
Grover Base   | 0.931 ± 0.021| **0.750 ± 0.037** | 0.901 ± 0.036 | 0.750 ± 0.085 | 0.735 ± 0.006 | 4.0
Grover Large  | 0.932 ± 0.023 | 0.747 ± 0.062 | 0.901 ± 0.033 | 0.757 ± 0.057 | 0.757 ± 0.057 | 4.2
ChemBERTa     | 0.923 ± 0.032 | 0.666 ± 0.041 | 0.869 ± 0.032 | **0.779 ± 0.044** | 0.717 ± 0.009 | 7.0
MolBERT       | 0.942 ± 0.011 | 0.737 ± 0.085 | 0.889 ± 0.039 | 0.761 ± 0.058 | **0.742 ± 0.020** | 4.6
Chemprop      | 0.924 ± 0.069 | 0.724 ± 0.064 | 0.847 ± 0.052 | 0.766 ± 0.040 | 0.726 ± 0.008 | 7.0
Chemprop 2d   | 0.923 ± 0.015 | 0.712 ± 0.067 | 0.874 ± 0.030 | 0.775 ± 0.041 | 0.724 ± 0.006 | 6.8
Chemprop mc | 0.924 ± 0.082 | 0.740 ± 0.060 | 0.869 ± 0.033 | 0.772 ± 0.041 | 0.722 ± 0.008 | 6.2
