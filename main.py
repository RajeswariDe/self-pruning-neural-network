import os
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from train import train_model
from evaluate import evaluate, calculate_sparsity
from model import PrunableLinear


os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


lambda_values = [0.0001, 0.001, 0.01]

results = []


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


for lam in lambda_values:
    print(f"\nTraining with lambda = {lam}")

    model = train_model(
        lambda_val=lam,
        epochs=10
    )

    accuracy = evaluate(
        model,
        test_loader
    )

    sparsity = calculate_sparsity(
        model
    )

    results.append([
        lam,
        accuracy,
        sparsity
    ])

    print(
        f"Lambda: {lam} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"Sparsity: {sparsity:.2f}%"
    )

    all_gates = []

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = layer.get_gates().detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title(f"Gate Distribution (lambda={lam})")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")

    plot_path = f"outputs/plots/gate_distribution_{lam}.png"
    plt.savefig(plot_path)
    plt.close()


df = pd.DataFrame(
    results,
    columns=[
        "Lambda",
        "Test Accuracy",
        "Sparsity %"
    ]
)

df.to_csv(
    "outputs/results.csv",
    index=False
)

print("\nFinal Results:")
print(df)
print("\nresults.csv saved successfully")