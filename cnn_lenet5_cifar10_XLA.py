import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu

# Set XLA device
DEVICE = xm.xla_device()
print(f"Running on XLA device: {DEVICE}")

# Hyperparameters and settings
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_CLASSES = 10
GRAYSCALE = False

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)

# CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='data', train=False, transform=transforms.ToTensor())

# XLA-specific DistributedSampler for data loading
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=4, drop_last=False)

# LeNet5 Model Definition
class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        in_channels = 1 if grayscale else 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.Tanh(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.Tanh(),
            nn.Linear(84 * in_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# Initialize the model and optimizer
model = LeNet5(NUM_CLASSES, GRAYSCALE)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function to compute accuracy
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    model.eval()  # Set model to evaluation mode
    para_loader = pl.ParallelLoader(data_loader, [device])
    with torch.no_grad():  # Disable gradient calculation
        for features, targets in para_loader.per_device_loader(device):
            features, targets = features.to(device), targets.to(device)
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100

# Training Loop
def train_model(model, train_loader, optimizer, device, num_epochs):
    start_train_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        para_loader = pl.ParallelLoader(train_loader, [device])
        for batch_idx, (features, targets) in enumerate(para_loader.per_device_loader(device)):
            features, targets = features.to(DEVICE), targets.to(DEVICE)

            # Forward pass and loss calculation
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)

            # Backpropagation and optimization
            optimizer.zero_grad()
            cost.backward()
            xm.optimizer_step(optimizer)  # XLA-specific optimizer step

            # Logging the loss for every 50th batch
            if not batch_idx % 50:
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Cost: {cost:.4f}')

        # Compute and print training accuracy after each epoch
        train_acc = compute_accuracy(model, train_loader, device=DEVICE)
        print(f'Epoch: {epoch+1}/{num_epochs} | Train Accuracy: {train_acc:.2f}%')

    total_train_time = time.time() - start_train_time
    return total_train_time

# Benchmark Function to compute KPIs
def benchmark_model(model, train_loader, test_loader, device, num_epochs):
    # Training
    print("Benchmarking Training Phase:")
    train_time = train_model(model, train_loader, optimizer, device, num_epochs)

    # Inference Time and Throughput
    print("Benchmarking Inference Phase:")
    model.eval()
    total_time_inference = 0
    num_samples = 0
    para_loader = pl.ParallelLoader(test_loader, [device])
    with torch.no_grad():
        for data, targets in para_loader.per_device_loader(device):
            data, targets = data.to(device), targets.to(device)
            batch_size = data.size(0)
            num_samples += batch_size
            start_time = time.time()
            outputs = model(data)
            total_time_inference += time.time() - start_time

    avg_inference_time_per_batch = total_time_inference / len(test_loader)
    avg_inference_time_per_sample = (total_time_inference / num_samples) * 1000  # in milliseconds
    throughput = num_samples / total_time_inference  # samples per second

    # Accuracy
    train_acc = compute_accuracy(model, train_loader, device=DEVICE)
    test_acc = compute_accuracy(model, test_loader, device=DEVICE)

    # Print KPIs
    print(f"KPIs:")
    print(f"Accuracy (Train): {train_acc:.2f}%")
    print(f"Accuracy (Test): {test_acc:.2f}%")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Avg Inference Time (ms/sample): {avg_inference_time_per_sample:.4f} ms")
    print(f"Throughput (samples/second): {throughput:.2f} samples/s")

# Run the benchmarking
benchmark_model(model, train_loader, test_loader, DEVICE, NUM_EPOCHS)
