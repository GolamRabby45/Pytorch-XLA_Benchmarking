import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Check if CUDA is available and set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Hyperparameters and settings
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_CLASSES = 10
GRAYSCALE = True

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)

# MNIST Dataset with Resizing to 32x32
resize_transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])

train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=resize_transform,
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=resize_transform)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

# LeNet5 Model Definition
class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        in_channels = 1 if grayscale else 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
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
    model.eval()
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100

# Training Function
def train_model(model, train_loader, optimizer, device, num_epochs):
    start_train_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(DEVICE), targets.to(DEVICE)

            # Forward pass and loss calculation
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)

            # Backpropagation and optimization
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Logging every 50 batches
            if not batch_idx % 50:
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Cost: {cost:.4f}')

        # Compute accuracy after each epoch
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
    with torch.no_grad():
        for data, targets in test_loader:
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
