"""
Given hyperparameters, get a portion of the TinyStories data set
(https://arxiv.org/abs/2305.07759), build the Transformer model,
train using Adam, and plot the loss against wall-clock time.
"""

import os
import pickle

from data import Data
from model import Transformer
import numpy as np
from plotting import plot_experiment
import torch
from torchinfo import summary
from train import Train

# Get Path.
path = os.path.dirname(os.getcwd())

# Create folders for experiments and plots.
if not os.path.exists(os.path.join(path, "experiments")):
    os.makedirs(os.path.join(path, "experiments"))
if not os.path.exists(os.path.join(path, "plots")):
    os.makedirs(os.path.join(path, "plots"))

# Set device.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set parameters.
seq_length = 100
n_layer = 3
n_head = 3
embed_dim = 30
head_dim = 10
mlp_dim = 120
n = 100
batch_size = 10
epochs = 100
step_size = .01 # should scale with batch_size
n_plot = 100 # number of points where full loss is computed
plot_only = False  # change to True if you want to plot existing experimental results

# Run experiment.
experiment_id = str(seq_length)+"."+str(n)+"."+str(batch_size)+"."+str(step_size)
if plot_only is False:
    print("Run Experiment")

    # Get data.
    data_obj = Data(seq_length=seq_length, n=n)
    data_obj.get_data()
    vocab_size = data_obj.vocab_size
    print("Vocabulary size: " + str(vocab_size))
    torch.manual_seed(0)  # change torch seed here

    # Get model.
    model_obj = Transformer(
        seq_length=seq_length,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        embed_dim=embed_dim,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
    ).to(device)
    summary(model_obj)

    # Train model.
    print("---Training---")
    train_obj = Train(
        batch_size=batch_size,
        epochs=epochs,
        n_plot=n_plot,
        step_size=step_size,
        model=model_obj,
        training_dataset=data_obj.training_dataset,
        test_dataset=data_obj.test_dataset,
    )
    train_obj.train(device)

    with open(
        os.path.join(path, "experiments", str(experiment_id) + ".pkl"),
        "wb",
    ) as f:
        pickle.dump({"experiment": train_obj}, f)
    f.close()
with open(
    os.path.join(path, "experiments", str(experiment_id) + ".pkl"), "rb"
) as f:
    experiment = pickle.load(f)
f.close()

# Plot results.
print("Plot Experiment")
plot_experiment(experiment["experiment"], path)