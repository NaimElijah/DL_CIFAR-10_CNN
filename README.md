# DL_CIFAR-10_CNN

CNN classification, joint reconstruction, and latent-channel analysis on CIFAR-10 — Assignment 4 for *Applied Deep Learning* (Spring 2026).

Full write-up with figures and discussion: [`docs/report.pdf`](docs/report.pdf).

## Overview

The project starts from the standard PyTorch CIFAR-10 tutorial CNN and extends it in three stages, each implemented as its own script in `src/`:

| Task | Script | What it does |
|---|---|---|
| 1 | `cnn_classify.py` | Trains the tutorial CNN on CIFAR-10 and reports train/test accuracy, per-class accuracy, and sample predictions. |
| 2 | `cnn_deconv.py` | Adds a decoder that mirrors the encoder (`DeconvNet`) and trains classification + reconstruction jointly with a combined loss. |
| 3 | `cnn_latent_repr.py` | Loads the Task 2 model and decodes individual latent channels in isolation to visualize what each one has learned. |

## Architecture

**Encoder (shared by all three tasks):**

```
3x32x32 -> Conv(3->6, 5x5) -> ReLU -> MaxPool(2) -> 6x14x14   = z(1)
        -> Conv(6->16, 5x5) -> ReLU -> MaxPool(2) -> 16x5x5   = z(2)
```

**Classifier head:** `z(2)` flattened -> FC(400→120) → FC(120→84) → FC(84→10).

**Decoder head (`DeconvNet` only):** mirrors the encoder in reverse using the pooling indices captured by `MaxPool2d(return_indices=True)`:

```
16x5x5 = z(2) -> MaxUnpool -> ConvTranspose(16->6, 5x5) -> ReLU -> 6x14x14
              -> MaxUnpool -> ConvTranspose(6->3, 5x5)  -> tanh -> 3x32x32 = reconstruction
```

`DeconvNet` is trained end-to-end on the combined loss `L = L_CE(ŷ, y) + λ · L_rec(x̂, x)`, where `L_rec` is the per-channel squared error (Frobenius norm) averaged over channels and batch. λ = 0.01 brings the two loss terms to a comparable scale.

## Project structure

```
.
├── src/
│   ├── data.py              # CIFAR-10 loading, normalization, unnormalize() for plotting
│   ├── model.py              # Net (classifier) and DeconvNet (classifier + decoder)
│   ├── cnn_classify.py       # Train & evaluate Net
│   ├── cnn_deconv.py         # Train & evaluate DeconvNet (joint loss)
│   └── cnn_latent_repr.py    # Per-channel latent decoding
└── docs/
    └── report.pdf            # Write-up with full results and figures
```

Running any script creates a `figures/` directory (saved plots) alongside the data folder; trained weights are saved as `.pth` files in the working directory.

## Setup

```bash
pip install torch torchvision matplotlib
```

CIFAR-10 is downloaded automatically by `torchvision` on first run (into `./data` by default).

## Usage

Run scripts from inside `src/` (they import each other as local modules).

**Task 1 — classification:**
```bash
python cnn_classify.py --epochs 2
python cnn_classify.py --epochs 5 --subset 10   # use a 10x smaller dataset
```
Saves `cifar_net.pth` and `figures/task1_sample_predictions.png`.

**Task 2 — joint classification + reconstruction:**
```bash
python cnn_deconv.py --epochs 15 --batch-size 128 --lam 0.05
python cnn_deconv.py --epochs 10 --subset 10
```
Saves `deconv_net.pth`, `figures/task2_accuracy_curves.png`, and `figures/task2_reconstructions.png`.

**Task 3 — latent channel analysis** (requires a `deconv_net.pth` from Task 2):
```bash
python cnn_latent_repr.py --train-idx 7 --test-idx 3
```
Saves `figures/task3_z1_channels.png` and `figures/task3_z2_channels.png`.

Common flags: `--subset N` keeps every Nth image (e.g. `--subset 10` for a ~6,000-image training set when compute is limited); `--save-model` overrides the output checkpoint path.

## Results (from `docs/report.pdf`)

**Task 1** (2 epochs, batch size 4): 56.71% train / 55.24% test accuracy. Man-made classes (truck, plane, ship) are recognized most reliably; animal classes (bird, cat, deer) are hardest.

**Task 2** (10 epochs, λ = 0.01): 69.34% train / 59.98% test accuracy — on par with Task 1, showing the reconstruction objective doesn't hurt classification. The decoder recovers global color and coarse shape but loses fine detail, since `MaxUnpool` only restores pooled maxima.

**Task 3:** `z(1)` channels behave as low-level edge/luminance/color detectors that still preserve spatial layout (14×14). `z(2)` channels are far coarser (5×5) and encode abstract, distributed figure-ground structure — no single channel reconstructs the image alone, consistent with the usual CNN feature hierarchy.

See the report for full per-class accuracy tables and figures.

## License
This project was done for learning & experimenting purposes.