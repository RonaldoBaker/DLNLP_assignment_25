"""
This module contains the Logger class, which is responsible for logging
hyperparameters, training metrics, and test results during the training and testing of a model.
"""

# Import necessary libraries
import os
import sys
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom modules
from utils.config import config

class Logger:
    """
    Logger class for logging hyperparameters, training metrics, and test results.
    """
    def __init__(self, trainer, tester, hyperparameters: dict, log_dir: str = None, inner_log_dir: str = "single"):
        self.trainer = trainer
        self.tester = tester
        self.hyperparameters = hyperparameters
        # If the log directory has already been set, use it
        if log_dir is None:
            self.log_dir = Logger.set_log_dir(inner_log_dir, hyperparameters)
        else:
            self.log_dir = log_dir


    @staticmethod
    def set_log_dir(inner_log_dir: str, hyperparameters: dict) -> str:
        """
        Sets the log directory for the current run.

        Args:
            inner_log_dir (str) : The inner log directory name.
            hyperparameters (dict) : The hyperparameters used for the current run.
        
        Returns:
            str: The full path to the log directory.
        """
        # Get current date and time
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y.%m.%d_%H-%M-%S")

        # Add tokenisation to the log directory name
        tokenisations = "_".join(hyperparameters["tokenisations"])

        # Create a new directory for the current run
        inner_log_dirtimestamp_str = f"{inner_log_dir}_{tokenisations}_{timestamp_str}" # single_2023.10.01_12:00:00 for example / multi_word_subword_2023.10.01_12:00:00

        # /home/zceerba/nlp/DLNLP_assignment_25/logs/single/single_2023.10.01_12:00:00
        log_dir = os.path.join(config.LOG_PATH, inner_log_dir, inner_log_dirtimestamp_str)

        if not os.path.exists(log_dir):
            # Create the directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)

        return log_dir


    def log_hyperparameters(self):
        """
        Logs the hyperparameters to a json file.
        """
        hyperparameters_path = os.path.join(self.log_dir, "hyperparameters.json")
        with open(hyperparameters_path, "w") as f:
            json.dump(self.hyperparameters, f, indent=4)


    def log_and_plot_training_metrics(self):
        """
        Logs the training metrics and plots the training and validation losses and BLEU scores.
        """
        train_losses = self.trainer.train_losses
        val_losses = self.trainer.val_losses
        val_bleus = self.trainer.val_bleus
        epochs = range(1, len(train_losses) + 1)

        # Save the curves to png files
        self.plot_losses(train_losses, val_losses, epochs)
        self.plot_bleu(val_bleus, epochs)

        # Save the training metrics to a csv file
        loss_logger_path = os.path.join(self.log_dir, "loss_logger.csv")
        losses = {
            "epoch": epochs,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_bleu": val_bleus
        }
        df = pd.DataFrame(losses)
        df.to_csv(loss_logger_path, index=False)


    def plot_losses(self, train_losses, val_losses, epochs):
        """
        Plots the training and validation losses.

        Args:
            train_losses (list) : List of training losses.
            val_losses (list) : List of validation losses.
            epochs (list) : List of epochs.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.grid()
        plt.title("Cross Entropy Training and Validation Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "loss_curves.png"))


    def plot_bleu(self, val_bleus, epochs):
        """
        Plots the validation BLEU scores.
        
        Args:
            val_bleus (list) : List of validation BLEU scores.
            epochs (list) : List of epochs.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_bleus, label="Validation BLEU")
        plt.grid()
        plt.title("Validation BLEU Score")
        plt.xlabel("Epochs")
        plt.ylabel("BLEU Score")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "bleu_curves.png"))


    def log_test_results(self):
        """
        Logs the test results to a json file.
        """
        # Append the elapsed time to the results
        elapsed_time = self.trainer.elapsed_time
        results = self.tester.results
        results["elapsed_time"] = elapsed_time

        # Save the results to a json file
        test_results_path = os.path.join(self.log_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(self.tester.results, f, indent=4)


    def log_all(self):
        """
        Logs all the information to the log directory.
        """
        self.log_hyperparameters()
        self.log_and_plot_training_metrics()
        self.log_test_results()