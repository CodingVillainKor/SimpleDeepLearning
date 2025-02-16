import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class MNISTData:
    def __init__(self):
        dataset = MNIST(root='./data', train=True, download=True)
        self.dataset = [(ToTensor()(img), target) for img, target in dataset]
        val_dataset = MNIST(root='./data', train=False, download=True)
        self.val_dataset = [(ToTensor()(img), target) for img, target in val_dataset]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]