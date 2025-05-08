# DLNLP_assignment_25

## Repository Overview
This repository contains the implementation of a Transformer-based model for machine translation. The project is organized into the following key components:
- **`main.py`**: The main script for training and evaluating the Transformer model.
- **`src/`**: Contains custom modules for preprocessing, model definition, training, and testing.
  - `preprocessor.py`: Handles data preprocessing tasks such as tokenization, vocabulary building, and numericalization.
  - `models.py`: Defines the Transformer and MultiSourceTransformer models.
  - `model_trainer.py`: Implements the training loop for the model.
  - `model_tester.py`: Handles model evaluation.
- **`dataset.py`**: Defines custom dataset classes and collate functions for data loading.
- **`utils/`**: Contains utility modules for logging and configuration management.
  - `logger.py`: Handles logging of training and evaluation results.
  - `config.py`: Stores configuration parameters for the project.
- **`env/requirements.txt`**: Lists all the required Python packages and their versions.

## Main Packages Required
The following Python packages are required to run `main.py`:
- `torch` (PyTorch)
- `torchtext`
- `scikit-learn`
- `json`
- `functools`
- `datasets`
- `spacy`
- `transformers`
- `numpy`

For a complete list of dependencies, refer to the [`env/requirements.txt`](env/requirements.txt) file.

## Setting Up the Environment
Follow these steps to set up the environment and run the project:

### 1. Create a Virtual Environment
Run the following commands to create and activate a virtual environment, and run the setup script to download other necessay dependencies:
```bash
conda env create -f env/environment.yml
conda activate nlp
```

### 2. Run shell script
Run the following commands to download other necessary dependencies:
```bash
chmod +x setup.sh
./setup.sh
```
**NOTE** Will need to use Git Bash or another terminal that can use Unix commands to run these commands. The python path will also need to be changed to your local python path in [setup.sh](setup.sh) before running,

## Running main.py
The `main.py` runs with a number of CLI arguments. By default it will have the following arguments:

1. `GPU=0`
2. `MODEL='single'`
3. `DROPOUT=0.2`
4. `POSITIONAL_EMBEDDING_TYPE='sequential'`
5. `FUSION_TYPE='single'`
6. `TOKENISATIONS=['word']`
7. `REMOVE_UNDERSCORE=True`
8. `ADDITIONAL_POSITIONAL_EMBEDDING=True`

The default `LOG_PATH` should be change to folder on your local device. If it is not found, it will automatically store in the root of the repository and create a folder called `logs/`

Here is an example of running `main.py` with some arguments:

```bash
python main.py --GPU 0 --MODEL single --TOKENISATIONS word subword --FUSION_TYPE single --DROPOOUT 0.2
```

Please refer to [utils/config.py](utils/config.py) for the full details of what is allowed for each arguments. 

