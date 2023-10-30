from __future__ import annotations

from itertools import cycle
from torch.optim import Optimizer
from torch.utils.data import (DataLoader, Dataset)
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import torch.onnx


class MNISTDataset(Dataset):
    def __init__(self, train: bool, path: str, device: torch.device) -> None:
        super().__init__()
        self.path = path
        self.prefix = 'train' if train else 'test'
        self.path_xs = os.path.join(self.path, f'mnist_{self.prefix}_xs.pt')
        self.path_ys = os.path.join(self.path, f'mnist_{self.prefix}_ys.pt')
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.1307, ), (0.3081, ))])

        if not os.path.exists(self.path_xs) or not os.path.exists(self.path_ys):
            set = MNIST(path, train=train, download=True, transform=self.transform)
            loader = DataLoader(set, batch_size=batch_size, shuffle=train)
            n = len(set)

            xs = torch.empty((n, *set[0][0].shape), dtype=torch.float32)
            ys = torch.empty((n, ), dtype=torch.long)
            desc = f'Preparing {self.prefix.capitalize()} Set'
            for i, (x, y) in enumerate(tqdm(loader, desc=desc)):
                xs[i * batch_size:min((i + 1) * batch_size, n)] = x
                ys[i * batch_size:min((i + 1) * batch_size, n)] = y

            torch.save(xs, self.path_xs)
            torch.save(ys, self.path_ys)
        
        self.device = device
        self.xs = torch.load(self.path_xs, map_location=self.device)
        self.ys = torch.load(self.path_ys, map_location=self.device)

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, loader: DataLoader, optimizer: Optimizer, scheduler) -> None:
        self.train()
        batches = iter(cycle(loader))
        x, l = next(batches)
        optimizer.zero_grad(set_to_none=True)
        logits = self(x)
        loss = F.nll_loss(torch.log_softmax(logits, dim=1), l)
        loss.backward()
        optimizer.step()
        scheduler.step()

    @torch.inference_mode()
    def test(self, loader: DataLoader) -> None:
        self.eval()
        loss, acc = 0, 0.0
        for x, l in tqdm(loader, total=len(loader), desc='Testing'):
            logits = self(x)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            loss += F.nll_loss(torch.log_softmax(logits, dim=1), l, reduction='sum').item()
            acc += (preds == l.view_as(preds)).sum().item()
        print()
        print(f'Loss: {loss / len(loader.dataset):.2e}')
        print(f'Accuracy: {acc / len(loader.dataset) * 100:.2f}%')
        print()
        return acc / len(loader.dataset) * 100


if __name__ == '__main__':
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    device = torch.device('cpu')
    epochs = 50
    batch_size = 512
    lr = 1e-2
 

    train_set = MNISTDataset(True, './datasets', device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    
    test_set = MNISTDataset(False, './datasets', device)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MLP().to(device)
    if os.path.exists("mnist.pt"):
        model.load_state_dict(torch.load("mnist.pt"))

    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.7, 0.9)) 
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=int(((len(train_set) - 1) // batch_size + 1) * epochs))

    # model.fit(train_loader, optimizer, scheduler, epochs)
    acc = model.test(test_loader)
    for i in range(epochs):
        for _ in tqdm(range(len(train_loader)), desc='Training'):
            model.fit(train_loader, optimizer, scheduler)
        if(acc < model.test(test_loader)):
           torch.save(model.state_dict(), 'mnist.pt')
           
    torch.onnx.export(
            model.cpu(),
            torch.empty((1, 1, 28, 28), dtype=torch.float32),
            'mnist.onnx',
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            )

    



    
