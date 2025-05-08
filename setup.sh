#!/bin/bash

# Script to set up spaCy language models for the project
# Explicitly define the correct Python executable
PYTHON_PATH=/home/zceerba/.conda/envs/nlp/bin/python

echo "Using Python at: $PYTHON_PATH"
echo "Downloading spaCy models..."

# Donwload the English and Spanish models
$PYTHON_PATH -m spacy download en_core_web_sm
$PYTHON_PATH -m spacy download es_core_news_sm

echo "spaCy models installed successfully."