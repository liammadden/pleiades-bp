import os

import matplotlib.pyplot as plt
import numpy as np

fontsize_axis = 7
fontsize_ticks = 6

default_height_to_width_ratio = (5.0**0.5 - 1.0) / 2.0

def plot_experiment(experiment, path):

    training_loss_values = experiment.training_losses
    test_loss_values = experiment.test_losses
    times = experiment.times
    epochs = experiment.epochs
    n_plot = experiment.n_plot
    freq = max([round(epochs / n_plot), 1])
    epoch_count = np.arange(0, epochs + freq, freq)[0 : len(training_loss_values)]

    nrows = 2
    ncols = 2
    _, ax = plot_settings(nrows=nrows, ncols=ncols)

    plot_lineplot(
        axis=ax[0, 0],
        xdata=epoch_count,
        ydata=training_loss_values,
        xlabel="Epochs",
        ylabel="Training Loss",
    )

    plot_lineplot(
        axis=ax[1, 0],
        xdata=times,
        ydata=training_loss_values,
        xlabel="Seconds",
        ylabel="Training Loss",
    )

    plot_lineplot(
        axis=ax[0, 1],
        xdata=epoch_count,
        ydata=test_loss_values,
        xlabel="Epochs",
        ylabel="Test Loss",
    )

    plot_lineplot(
        axis=ax[1, 1],
        xdata=times,
        ydata=test_loss_values,
        xlabel="Seconds",
        ylabel="Test Loss",
    )

    plt.savefig(
        os.path.join(path, "plots", "loss-plot" + ".pdf"),
        bbox_inches="tight",
    )
    return


def plot_lineplot(axis, xdata, ydata, xlabel, ylabel):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-4, 4))
    axis.plot(xdata, ydata)


def plot_settings(
    nrows=1, ncols=1, width=6.0, height_to_width_ratio=default_height_to_width_ratio
):
    subplot_width = width / ncols
    subplot_height = height_to_width_ratio * subplot_width
    height = subplot_height * nrows
    figsize = (width, height)

    plt.rcParams.update(
        {
            "axes.labelsize": fontsize_axis,
            "figure.figsize": figsize,
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            "lines.linewidth": 1,
            "lines.marker": "none",
            "xtick.labelsize": fontsize_ticks,
            "ytick.labelsize": fontsize_ticks,
            "figure.dpi": 250,
        }
    )

    return plt.subplots(nrows, ncols, constrained_layout=True)