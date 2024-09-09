import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

# Ensure reproducibility by setting seeds
def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    xm.set_rng_state(seed)

# Set XLA device
DEVICE = xm.xla_device()
print(f"Running on XLA device: {DEVICE}")

# Set random seed and hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
NUM_EPOCHS = 40
NUM_CLASSES = 10
set_all_seeds(RANDOM_SEED)

# Load CIFAR-10 dataset
def get_dataloaders_cifar10(batch_size, num_workers, train_transforms, test_transforms, validation_fraction=0.1):
    # Download and transform CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
    
    # Split the training data into train and validation sets
    num_train_samples = int((1 - validation_fraction) * len(train_dataset))
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [num_train_samples, len(train_dataset) - num_train_samples])
    
    # XLA-specific data samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)

    # DataLoader for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, drop_last=False)
    
    return train_loader, valid_loader, test_loader

train_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                       transforms.RandomCrop((64, 64)),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                      transforms.CenterCrop((64, 64)),
                                      transforms.ToTensor()])

train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
    batch_size=BATCH_SIZE, 
    num_workers=4, 
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    validation_fraction=0.1)

# Define the AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        return logits

# Initialize the model, optimizer, and set it to the XLA device
torch.manual_seed(RANDOM_SEED)
model = AlexNet(NUM_CLASSES)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train_classifier(num_epochs, model, optimizer, device, train_loader, valid_loader):
    criterion = nn.CrossEntropyLoss()
    start_train_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        correct_preds = 0
        total_preds = 0
        running_loss = 0.0

        para_loader = pl.ParallelLoader(train_loader, [device])
        for data, targets in para_loader.per_device_loader(device):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            xm.optimizer_step(optimizer)  # XLA-specific optimizer step

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)

        train_acc = 100 * correct_preds / total_preds
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader.dataset):.4f}, Train Accuracy: {train_acc:.2f}%')

    total_train_time = time.time() - start_train_time
    print(f"Total Training Time: {total_train_time:.2f}s")
    return total_train_time

# Evaluation and benchmarking
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_preds = 0
    total_preds = 0
    para_loader = pl.ParallelLoader(data_loader, [device])
    
    with torch.no_grad():
        for data, targets in para_loader.per_device_loader(device):
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)
    
    accuracy = 100 * correct_preds / total_preds
    return accuracy

def benchmark_model(model, train_loader, test_loader, device):
    # Training
    print("Benchmarking Training Phase:")
    train_time = train_classifier(NUM_EPOCHS, model, optimizer, device, train_loader, test_loader)

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
    train_acc = compute_accuracy(model, train_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)

    # Print KPIs
    print(f"KPIs:")
    print(f"Accuracy (Train): {train_acc:.2f}%")
    print(f"Accuracy (Test): {test_acc:.2f}%")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Avg Inference Time (ms/sample): {avg_inference_time_per_sample:.4f} ms")
    print(f"Throughput (samples/second): {throughput:.2f} samples/s")

# Run the benchmarking
benchmark_model(model, train_loader, test_loader, DEVICE)
