import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import PrunableMLP, PrunableLinear


def get_sparsity_loss(model):
    loss = 0

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = layer.get_gates()
            loss += torch.sum(gates)

    return loss


def train_model(lambda_val=0.001, epochs=5):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = PrunableMLP().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            classification_loss = criterion(outputs, y)

            sparsity_loss = get_sparsity_loss(model)

            total_loss_final = (
                classification_loss
                + lambda_val * sparsity_loss
            )

            total_loss_final.backward()
            optimizer.step()

            total_loss += total_loss_final.item()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {total_loss:.4f}"
        )

    return model