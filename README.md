# Huggingmolecules

We envision models that are pre-trained on a vast range of domain-relevant tasks to become the backbone of the drug discovery
and, in the long run, other applications of predictive modeling to chemistry. This repository aims to further this vision
and give easy access to state-of-the-art pre-trained models.

## Quick tour

To quickly fine-tune a model on a dataset using pytorch lightning package follow the below example based on the MAT
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

# Build the data loader for freesolv dataset:
train_dataloader, _, _ = get_data_loaders(featurizer,
                                          batch_size=32,
                                          task_name='ADME',
                                          dataset_name='hydrationfreeenergy_freesolv')

# Build the pytorch lightning trainer and fine-tune the module on train dataset:
trainer = Trainer(max_epochs=100)
trainer.fit(pl_module, train_dataloader=train_dataloader)

# Make the prediction for the batch of SMILES strings:
batch = featurizer(['C/C=C/C', '[C]=O'])
output = pl_module.model(batch)
```

## Installation

Create your conda environment first:

```conda create -y -n -q huggingmolecules python=3.8.5 -c rdkit rdkit=2020.09.1```

Then install the package by running the following commands from the cloned directory:

```
conda activate huggingmolecules
pip install -e ./src
```

## Project Structure

The project consists of two main modules: src/ and experiments/ modules:

* The src/ module contains abstract interfaces for pre-trained models along with their implementations based on pytorch
  library. This module makes configuring, downloading and running existing models easy and out-of-the-box.
* The experiments/ module makes use of abstract interfaces defined in the src/ module and implements scripts based on
  pytorch lightning for running various experiments. This module makes training, benchmarking and hyper-tuning of models
  flawless and easily extensible.

## Supported models architectures

Huggingmolecules currently provides the following models architectures:

* [MAT](https://github.com/ardigen/MAT)
* [MAT++]()
* [GROVER](https://github.com/tencent-ailab/grover)

For ease of benchmarking, we also include wrappers in the experiments/ module for three other models architectures:

* [chemprop](https://github.com/chemprop/chemprop)
* [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry)
* [MolBERT](https://github.com/BenevolentAI/MolBERT)


## The src/ module

The implementations of the models in the src/ module are divided into three modules: configuration, featurization and models
module. The relation between these modules is shown on the following examples based on the MAT model:

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

## The experiments/ module

### Requirements

In addition to dependencies defined in the src/ module, the experiments/ module goes along with few others. To install them, run:

```pip install -r experiments/requirements.txt```

The following packages are crucial for functioning of the experiments/ module:
* [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [optuna](https://github.com/optuna/optuna)
* [gin-config](https://github.com/google/gin-config)
* [TDC](https://github.com/mims-harvard/TDC)

In addition, we recommend installing neptune package and enable using neptune option it in ```experiments/configs/setup.gin``` file.

### Running scripts:

We recommend running experiments scripts from the source code. 
For the moment there are three scripts implemented:

* ```experiments/train.py``` - for training with pytorch lightning package
* ```experiments/tune_hyper.py``` - for hyper-parameters tuning with optuna package
* ```experiments/benchmark_1.py``` - for benchmarking based on hyper-parameters tuning

In general running scripts can be done with the following syntax:

```
python -m experiments.<script_name> /
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

So for instance, if we want to fine-tune MAT model (pretrained on masking_20M task) on freesolv dataset using GPU 1, we
simply type:

```
python -m experiments.train /
       -d freesolv / 
       -m mat /
       -b model.pretrained_name=\"mat_masking_20M\"#train.gpus=[1]
```

or equivalently:

```
python -m experiments.train /
       -d freesolv / 
       -m mat /
       --model.pretrained_name mat_masking_20M /
       --train.gpus [1]
```

### Benchmarking

For the moment there is one benchmark available. It works as follows:

* ```experiments/benchmark_1.py```: on the given dataset we fine-tune the given model on 7 learning rates and 6 seeded
  data splits (42 fine-tunings in total). Then we choose that learning rate that minimizes an averaged (on 6 data
  splits) validation loss. The result is the averaged value of test metric (metric computed on the test dataset, e.g.
  test loss) for the chosen learning rate.

Running a benchmark is essentially the same as running any other script form experiments/ module. So for instance to
benchmark vanilla MAT model (without pretraining) on BBB dataset using GPU 0, we simply type:

```
python -m experiments.benchmark_1 /
       -d bbb / 
       -m mat /
       --model.pretrained_name None /
       --train.gpus [0]
```

However, the above script will only perform 42 fine-tunings without computing the final benchmark result. To compute it
we need to type:

```
python -m experiments.benchmark_1 --results_only /
       -d bbb / 
       -m mat
```

The above script won't perform any fine-tuning, but will only compute the benchmark result. If we had neptune enabled
in ```experiments/configs/setup.gin```, all data necessary to compute the result will be
fetched from the neptune server.

## Benchmark results

We performed the benchmark described in [Benchmarking](#benchmarking) as ```experiments/benchmark_1.py``` for various models
architectures and pretraining tasks. Here are our results:

// TODO
