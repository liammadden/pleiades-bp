"""
Given data set, model, and hyperparameters, train using FoLo.
"""

from dataclasses import dataclass, field
import random
import time

import numpy as np
import torch
import torch.linalg as la
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

@dataclass
class Train:
    batch_size: int = 10
    epochs: int = 100 # number of epochs to take
    n_plot: int = 100 # number of points where full loss is computed
    step_size: any = .01 # should scale with batch_size
    model: any = None # will have its parameters updated in place
    training_dataset: any = None
    test_dataset: any = None
    training_losses: any = field(default_factory=list)
    test_losses: any = field(default_factory=list)
    times: any = field(default_factory=list)

    def train(self, device):
        # Set up loss, optimizer, and data.
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = optim.Adam(self.model.parameters(), lr=self.step_size)
        trainloader = torch_data.DataLoader(
            self.training_dataset, batch_size=self.batch_size, shuffle=False
        )
        training_losses = []
        test_losses = []
        training_loss, test_loss = self.compute_full_loss(device)
        print(f"Epoch: {0}, Training Loss: {training_loss}, Test Loss: {test_loss}")
        training_losses.append(training_loss)
        test_losses.append(test_loss)

        # Set up when to compute full loss.
        n = len(self.training_dataset) # number of samples
        freq = max([round(self.epochs / self.n_plot), 1]) # for computing full loss
        
        times = [0] # wall-clock times between calls to compute_full_loss
        old_time = time.perf_counter() # start timer
        for epoch in range(self.epochs):
            for _, batch in enumerate(trainloader):
                # Take one optimizer step.
                batch = batch.to(device)
                optimizer.zero_grad()
                output = self.model(batch[:, :-1])
                loss = criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    batch[:, 1:].contiguous().view(-1),
                )
                loss.backward()
                optimizer.step()

            # Compute full loss.
            if (epoch + 1) % freq == 0:
                times.append(times[-1] + time.perf_counter() - old_time)
                training_loss, test_loss = self.compute_full_loss(device)
                print(
                    f"Epoch: {epoch + 1}, Training Loss: {training_loss}, Test Loss: {test_loss}"
                )
                training_losses.append(training_loss)
                test_losses.append(test_loss)
                old_time = time.perf_counter() # start timer

        self.training_losses = training_losses
        self.test_losses = test_losses
        self.times = times

    @torch.no_grad()
    def compute_full_loss(self, device):
        n_training = len(self.training_dataset) # number of training samples
        n_test = len(self.test_dataset) # number of training samples
        seq_length = len(self.training_dataset[0]) # length of each sample
        criterion = nn.CrossEntropyLoss(reduction="sum") # divide at end for mean

        # Compute training loss.
        training_loss = 0
        for batch in torch_data.DataLoader(
            dataset=self.training_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        ):
            batch = batch.to(device)
            output = self.model(batch[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                batch[:, 1:].contiguous().view(-1),
            )
            training_loss += loss
        training_loss = training_loss.item()/n_training/(seq_length - 1)

        # Compute test loss.
        test_loss = 0
        for batch in torch_data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        ):
            batch = batch.to(device)
            output = self.model(batch[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                batch[:, 1:].contiguous().view(-1),
            )
            test_loss += loss
        test_loss = test_loss.item()/n_test/(seq_length - 1)

        return training_loss, test_loss