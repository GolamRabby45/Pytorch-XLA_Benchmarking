import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch_xla.core.xla_model as xm
from torchvision import datasets, transforms
import time

# Check if GPU is available and set the device
device = xm.xla_device()
print(f"Running on XLA device: {device}")

# Loading and transforming the MNIST dataset
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)

# Creating DataLoader objects for training and testing data
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

'''
# Use XLA-specific distributed sampler for data loading
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
)

train_loader = DataLoader(train_data, batch_size=10, sampler=train_sampler, num_workers=4, drop_last=True)
test_loader = DataLoader(test_data, batch_size=10, sampler=test_sampler, num_workers=4, drop_last=False)
'''

# Defining the Convolutional Neural Network model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
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

# Initialize the model, move to the GPU, loss function, and optimizer
torch.manual_seed(42)
model = ConvolutionalNetwork()

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_fn(model, train_loader, optimizer, criterion, device, epochs=5):
    start_time = time.time()
    train_losses = []
    train_correct = []

    for i in range(epochs):
        trn_corr = 0

        # Run the training batches
        model.train()
        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1

            # Move data to the GPU
            X_train, y_train = X_train.to(device), y_train.to(device)

            # Apply the model
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            # Update parameters
            optimizer.zero_grad()
            loss.backward()

            # XLA-specific optimizer step
            xm.optimizer_step(optimizer, barrier=True)
            

            # Print interim results
            if b % 600 == 0:
                print(f'epoch: {i:2} batch: {b:4} [{10*b:6}/60000] loss: {loss.item():10.8f} '
                      f'accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

        train_losses.append(loss)
        train_correct.append(trn_corr)

    duration = time.time() - start_time
    print(f'\nTraining completed in: {duration:.0f} seconds')

# Evaluation function with benchmarking (throughput and total inference time)
def eval_fn(model, test_loader, criterion, device):
    model.eval()
    test_losses = []
    test_correct = []
    tst_corr = 0

    total_time = 0
    processed_samples = 0

    start_time = time.time()

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Move data to the GPU
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Time the inference of the batch
            start_batch = time.time()
            y_val = model(X_test)
            total_time += time.time() - start_batch

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

            processed_samples += X_test.size(0)

        # Calculate loss for the final batch
        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)

    # Calculate total time and throughput
    total_time = time.time() - start_time
    throughput = processed_samples / total_time

    # Print evaluation results
    print(f'Test loss: {loss.item():.4f}, Test accuracy: {tst_corr.item() / len(test_loader.dataset) * 100:.2f}%')
    print(f'Total inference time: {total_time:.3f} seconds')
    print(f'Throughput: {throughput:.2f} samples/second')

# Execute the training and evaluation functions
train_fn(model, train_loader, optimizer, criterion, device, epochs=5)
eval_fn(model, test_loader, criterion, device)
