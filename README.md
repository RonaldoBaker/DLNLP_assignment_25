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
Run the following commands to create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
