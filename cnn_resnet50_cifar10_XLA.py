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
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_CLASSES = 10
GRAYSCALE = False  # CIFAR-10 images are RGB

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)

# CIFAR-10 Dataset with Resizing to 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='data', 
                                 train=True, 
                                 transform=transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='data', 
                                train=False, 
                                transform=transform)

# XLA-specific distributed sampler for data loading
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=True)

test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=4, drop_last=False)

# ResNet-50 Model Definition
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # 1x1
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)  # 3x3
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)  # 1x1
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        super(ResNet, self).__init__()
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Layers
        self.layer1 = self._make_layer(block, 64, layers[0])     # layers[0] = 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # layers[1] = 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # layers[2] = 6
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # layers[3] = 3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Using Kaiming He initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: [batch_size, 3, 224, 224]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Output shape: [batch_size, 64, 56, 56]

        x = self.layer1(x)    # Output shape: [batch_size, 256, 56, 56]
        x = self.layer2(x)    # Output shape: [batch_size, 512, 28, 28]
        x = self.layer3(x)    # Output shape: [batch_size, 1024, 14, 14]
        x = self.layer4(x)    # Output shape: [batch_size, 2048, 7, 7]

        x = self.avgpool(x)   # Output shape: [batch_size, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 2048]
        logits = self.fc(x)        # Output shape: [batch_size, num_classes]
        probas = F.softmax(logits, dim=1)
        return logits, probas

def resnet50(num_classes):
    """Constructs a ResNet-50 model."""
    model = ResNet(block=Bottleneck, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=GRAYSCALE)
    return model

# Initialize the model and optimizer
model = resnet50(NUM_CLASSES)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function to compute accuracy
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    model.eval()
    para_loader = pl.ParallelLoader(data_loader, [device])
    with torch.no_grad():
        for features, targets in para_loader.per_device_loader(device):
            features, targets = features.to(device), targets.to(device)
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    accuracy = 100.0 * correct_pred.double() / num_examples
    return accuracy.item()

# Training Function
def train_model(model, train_loader, optimizer, device, num_epochs):
    start_train_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        # Set the sampler epoch for shuffling
        train_loader.sampler.set_epoch(epoch)
        para_loader = pl.ParallelLoader(train_loader, [device])
        for batch_idx, (features, targets) in enumerate(para_loader.per_device_loader(device)):
            features, targets = features.to(device), targets.to(device)

            # Forward pass and loss calculation
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            cost.backward()
            xm.optimizer_step(optimizer)  # XLA-specific optimizer step

            # Logging every 50 batches
            if not batch_idx % 50:
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Cost: {cost.item():.4f}')

        # Compute accuracy after each epoch
        train_acc = compute_accuracy(model, train_loader, device=device)
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

    avg_inference_time_per_sample = (total_time_inference / num_samples) * 1000  # in milliseconds
    throughput = num_samples / total_time_inference  # samples per second

    # Accuracy
    train_acc = compute_accuracy(model, train_loader, device=device)
    test_acc = compute_accuracy(model, test_loader, device=device)

    # Print KPIs
    print(f"KPIs:")
    print(f"Accuracy (Train): {train_acc:.2f}%")
    print(f"Accuracy (Test): {test_acc:.2f}%")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Avg Inference Time (ms/sample): {avg_inference_time_per_sample:.4f} ms")
    print(f"Throughput (samples/second): {throughput:.2f} samples/s")

# Run the benchmarking
benchmark_model(model, train_loader, test_loader, DEVICE, NUM_EPOCHS)
