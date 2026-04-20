import torch
from model import PrunableLinear


def calculate_sparsity(model, threshold=0.05):
    total_weights = 0
    pruned_weights = 0

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = layer.get_gates().detach().cpu()

            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()

    sparsity_percent = (
        100 * pruned_weights / total_weights
    )

    return sparsity_percent


def evaluate(model, test_loader):
    device = next(model.parameters()).device

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total

    return accuracy