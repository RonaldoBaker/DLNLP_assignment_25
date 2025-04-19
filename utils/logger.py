import os
import sys
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import config

class Logger:
    def __init__(self, trainer, tester, hyperparameters, inner_log_dir: str = "base"):
        self.trainer = trainer
        self.tester = tester
        self.hyperparameters = hyperparameters
        self.log_dir = self.set_log_dir(inner_log_dir)


    def set_log_dir(self, inner_log_dir: str):
        # Get current date and time
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y.%m.%d_%H:%M:%S")

        # Add tokenisation to the log directory name
        tokenisations = "_".join(self.hyperparameters["tokenisations"])

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


    def log_model_weights(self):
        best_model_checkpoint = self.trainer.best_model
        last_model_checkpoint = self.trainer.last_model
        torch.save(best_model_checkpoint, os.path.join(self.log_dir, "best_model.pth"))
        torch.save(last_model_checkpoint, os.path.join(self.log_dir, "last_model.pth"))


    def log_all(self):
        """
        Logs all the information to the log directory.
        """
        self.log_hyperparameters()
        self.log_and_plot_training_metrics()
        self.log_test_results()
        self.log_model_weights()