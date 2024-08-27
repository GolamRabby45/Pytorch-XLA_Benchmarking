import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import time

import numpy as np
import pandas as pd

# Setting the device to XLA (e.g., TPU or XLA-accelerated GPU)
device = xm.xla_device()
xm.master_print(f"Running on XLA device: {device}")

# Loading of the MNIST dataset
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)

# Create Distributed Samplers for training and testing datasets
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
)

# Create DataLoader objects for training and testing data
train_loader = DataLoader(train_data, batch_size=10, sampler=train_sampler, num_workers=4, drop_last=True)
test_loader = DataLoader(test_data, batch_size=10, sampler=test_sampler, num_workers=4, drop_last=False)

# Define CNN model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)  # Flatten the tensor
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

# Set random seed for reproducibility
torch.manual_seed(42)
model = ConvolutionalNetwork()

# Move the model to the XLA device
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the training model function
def train_model(model, criterion, optimizer, train_loader, test_loader, device, epochs=5):
    start_time = time.time()

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0

        # Run the training batches
        model.train()
        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1

            # Move data to the XLA device
            X_train, y_train = X_train.to(device), y_train.to(device)

            # Apply the model
            y_pred = model(X_train)  # We don't flatten X_train here
            loss = criterion(y_pred, y_train)

            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)

            # Print interim results
            if b % 600 == 0:
                xm.master_print(f'epoch: {i:2} batch: {b:4} [{10 * b:6}/60000]  loss: {loss.item():10.8f}  '
                                f'accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

        train_losses.append(loss.item())
        train_correct.append(trn_corr.item())

        # Run the testing batches
        model.eval()
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Move data to the XLA device
                X_test, y_test = X_test.to(device), y_test.to(device)

                # Apply the model
                y_val = model(X_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test).sum()

        loss = criterion(y_val, y_test)
        test_losses.append(loss.item())
        test_correct.append(tst_corr.item())

    xm.master_print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # Print the time elapsed

# Define the benchmarking function
def benchmark_model(model, test_loader, device):
    model.eval()
    total_time = 0
    processed_samples = 0

    start_time = time.time()

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Move data to the XLA device
            X_test = X_test.to(device)

            # Time the inference of the batch
            xm.mark_step()  # Sync step before computation
            start_batch = time.time()
            output = model(X_test)  # Forward pass
            xm.mark_step()  # Sync step after computation
            total_time += time.time() - start_batch
            processed_samples += X_test.size(0)

    total_time = time.time() - start_time
    throughput = processed_samples / total_time

    xm.master_print(f'Total inference time: {total_time:.3f} seconds')
    xm.master_print(f'Throughput: {throughput:.2f} samples/second')

    return total_time, throughput

# XLA Multiprocessing function
'''def _mp_fn(rank, flags):
    train_model(model, criterion, optimizer, train_loader, test_loader, device, epochs=5)
    benchmark_model(model, test_loader, device)

FLAGS = {}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')'''
