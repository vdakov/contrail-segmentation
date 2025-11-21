import torchvision 
import albumentations as A
import torch
from torch.utils.data import Subset, random_split, DataLoader

def load_mnist_data(train, data_augmentations, subset_size=100, num_classes=10, val_split=0.2):
    # Load full train/test datasets
    full_trainset = torchvision.datasets.MNIST(
        root='./data', train=train, download=True, transform=data_augmentations
    )

    
    # Create balanced subset
    samples_per_class = subset_size // num_classes
    balanced_indices = get_balanced_indices_by_targets(full_trainset.targets, samples_per_class)
    balanced_trainset = Subset(full_trainset, balanced_indices)

    # Split into train/val
    val_size = int(len(balanced_trainset) * val_split)
    train_size = len(balanced_trainset) - val_size
    generator = torch.Generator().manual_seed(42)
    trainset, valset = random_split(balanced_trainset, [train_size, val_size], generator=generator)


    # Test set
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=data_augmentations
    )

    # Dataloaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    return trainset, valloader, testset, trainloader, valloader, testloader

    
def get_balanced_indices_by_targets(targets, samples_per_class, n_classes=10):
    indices = []
    for class_label in range(n_classes):
        class_indices = (targets == class_label).nonzero(as_tuple=True)[0]
        selected = class_indices[:samples_per_class]
        indices.extend(selected.tolist())
    return indices