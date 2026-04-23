# Self-Pruning Neural Network (Tredence AI Case Study)

## Overview

This project implements a **Self-Pruning Neural Network** where each weight is associated with a learnable gate.
The model automatically learns which connections are important and prunes the rest during training instead of performing pruning after training.



## Problem Statement

Traditional neural networks are usually pruned after training using separate pruning techniques.
In this project, pruning happens during training itself.

Each weight has:

* a normal trainable weight
* a learnable gate score

The gate determines whether the connection should remain active or be suppressed.

This allows the model to learn:

**Which weights are important and which can be removed**

automatically.

---

## Core Idea

Each layer uses:

```python
gates = sigmoid(gate_scores)
pruned_weight = weight * gates
```

This means:

* gate close to 1 → important connection
* gate close to 0 → weak/unnecessary connection

Thus, the network becomes sparse while training.

---

## Custom Layer: PrunableLinear

A custom `PrunableLinear` layer was implemented instead of using `torch.nn.Linear`.

It contains:

* `weight`
* `bias`
* `gate_scores`

The forward pass computes:

```text
effective_weight = weight × sigmoid(gate_scores)
```

and uses it for prediction.

---

## Loss Function

Total loss used during training:

```text
Total Loss = Classification Loss + λ × Sparsity Loss
```

Where:

### Classification Loss

* CrossEntropyLoss

### Sparsity Loss

* L1 norm of all gate values

This encourages many gate values to move toward zero, resulting in pruning.

---

## Why L1 Regularization Creates Sparsity

L1 regularization penalizes all gate values linearly.

This causes:

* small values to move toward zero
* unnecessary connections to disappear
* only important weights to remain active

This helps reduce model complexity while preserving performance.

---

## Model Architecture

Simple MLP architecture used:

```text
Input (CIFAR-10: 32×32×3)
→ Flatten
→ PrunableLinear (512)
→ ReLU
→ PrunableLinear (256)
→ ReLU
→ PrunableLinear (10)
```

This simple architecture was chosen to focus on pruning behavior rather than model complexity.

---

## Dataset Used

### CIFAR-10

Using:

* `torchvision.datasets.CIFAR10`

Contains:

* 10 image classes
* 50,000 training images
* 10,000 testing images

Standard normalization was applied.

---

## Experiments

Three λ values were tested:

* 0.0001
* 0.001
* 0.01

These values control pruning strength.

### Lower λ

* less pruning
* better accuracy

### Higher λ

* more pruning
* slightly lower accuracy

---

## Final Results

| Lambda | Test Accuracy | Sparsity % |
| ------ | ------------: | ---------: |
| 0.0001 |        54.32% |     42.31% |
| 0.001  |        51.77% |     47.10% |
| 0.01   |        49.48% |     47.31% |

---

## Observations

### Higher λ increased sparsity

The model pruned more connections when stronger sparsity regularization was applied.

### Accuracy dropped slightly

Even after removing many weights, the model still maintained reasonable classification accuracy.

This shows successful self-pruning behavior.

---

## Sparsity Threshold Note

The original assignment defines pruning using:

```text
gate < 1e-2
```

For better interpretability of weak connections, an additional practical threshold:

```text
gate < 0.05
```

was also analyzed.

This provided clearer visualization of sparsity patterns.

---

## Gate Distribution Analysis

The histogram of gate values showed:

* strong concentration near zero
* smaller cluster of important active weights

This confirms that the model successfully learned sparse representations.

Plots are available in:

```text
outputs/plots/
```

---

## Project Structure

```text
self-pruning-neural-network/
│
├── model.py
├── train.py
├── evaluate.py
├── main.py
│
├── README.md
├── report.md
├── requirements.txt
│
└── outputs/
    ├── plots/
    ├── models/
    └── results.csv
```

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run training

```bash
python main.py
```

This will:

* train the model
* evaluate accuracy
* calculate sparsity
* generate plots
* save results.csv

---

## Key Contributions

* Built custom `PrunableLinear` layer
* Implemented learnable gating mechanism
* Added sparsity-aware training using L1 regularization
* Compared multiple λ values
* Analyzed tradeoff between accuracy and sparsity
* Generated pruning visualization plots

---

## Author

Rajeswari De
