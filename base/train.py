import torch
from torch.utils.data import DataLoader

from mnist import MNISTData
from model import Net

class Trainer:
    def __init__(self):
        self.model = Net()
        dataset = MNISTData()
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.val_dataloader = DataLoader(dataset.val_dataset, batch_size=32, shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train(self, epochs):
        for epoch in range(epochs):
            for i, (data, target) in enumerate(self.dataloader, 1):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print(f"\rEpoch {epoch}, Loss {loss.item()}", end='')
        print()

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.val_dataloader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print(f"Accuracy: {correct/total}, {correct=}, {total=}")


def main():
    trainer = Trainer()
    trainer.train(5)
    trainer.test()

if __name__ == "__main__":
    main()