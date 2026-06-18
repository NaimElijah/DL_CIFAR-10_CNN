"""CIFAR-10 data loading, shared by all stages/files.

Follows the PyTorch CIFAR-10 blitz tutorial: ToTensor + Normalize to [-1, 1]
with mean/std of 0.5 per channel.
"""
import torch
import torchvision
import torchvision.transforms as transforms

CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Normalize each channel to roughly [-1, 1]; (x - 0.5) / 0.5.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def get_loaders(batch_size=4, num_workers=2, data_root="./data", subset_factor=1):
    """Return (trainloader, testloader).

    subset_factor > 1 keeps only every Nth image (the assignment allows a 10x
    reduction to 6,000 training images if compute is limited).
    """
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=_transform)
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_transform)

    if subset_factor > 1:
        train_idx = range(0, len(trainset), subset_factor)
        test_idx = range(0, len(testset), subset_factor)
        trainset = torch.utils.data.Subset(trainset, list(train_idx))
        testset = torch.utils.data.Subset(testset, list(test_idx))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def unnormalize(img):
    """Invert the training normalization for visualization. Accepts a tensor."""
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1)
