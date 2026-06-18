""" Joint classification + reconstruction with a deconvolutional decoder.

Trains DeconvNet end-to-end with the combined loss
    L = L_ce(y_hat, y) + lambda * L_rec(x_hat, x)
where L_rec = (1/3) * sum_{i=1..3} ||x_hat_i - x_i||_F^2  (per-channel squared
error, averaged over the 3 channels, then averaged over the batch).

Reports everything the assignment asks for:
  - train/test accuracy curves over epochs (-> figures/)
  - final test accuracy
  - the lambda used
  - 2-3 original/reconstruction pairs (-> figures/)

Usage:
    py -3 cnn_deconv.py --epochs 15 --batch-size 128 --lam 0.05
    py -3 cnn_deconv.py --epochs 15 --subset 10        # smaller dataset
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data import CLASSES, get_loaders, unnormalize
from model import DeconvNet

FIG_DIR = "figures"


def recon_loss(x_hat, x):
    """L_rec = (1/3) * sum_i ||x_hat_i - x_i||_F^2, averaged over the batch.
    Per channel we sum the squared error over all pixels (Frobenius^2), then
    average over the 3 channels and over the batch.
    """
    # x_hat and x have shape [B, 3, 32, 32]; we want a single scalar loss.
    per_channel = ((x_hat - x) ** 2).sum(dim=(2, 3))   # [B, 3]
    return per_channel.mean(dim=1).mean()              # mean over channels, then batch


def evaluate(net, loader, device):
    """Return overall classification accuracy on a loader."""
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            # Move to device, forward pass, compute predictions, update counts
            images, labels = images.to(device), labels.to(device)
            logits, _ = net(images)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def per_class_accuracy(net, loader, device):
    net.eval()
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)
    with torch.no_grad():
        # We want to compute accuracy per class, so we need to update counts separately for each class.
        for images, labels in loader:
            # Move to device, forward pass, compute predictions, update counts
            images, labels = images.to(device), labels.to(device)
            logits, _ = net(images)
            _, predicted = torch.max(logits, 1)
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                class_correct[label] += int(pred == label)
    return {
        CLASSES[i]: (class_correct[i] / class_total[i] if class_total[i] else 0.0)
        for i in range(len(CLASSES))
    }


def train(net, trainloader, testloader, device, epochs, lam, lr, momentum):
    """Train jointly; record per-epoch train/test accuracy and loss components."""
    ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    history = {"train_acc": [], "test_acc": [], "ce": [], "rec": []}

    for epoch in range(epochs):
        # Train one epoch; accumulate total CE and Rec loss to report average per batch at the end of the epoch.
        net.train()
        run_ce = run_rec = n_batches = 0.0
        for images, labels in trainloader:
            # Move to device, zero gradients, forward pass, compute losses, backward pass, update weights
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass returns both logits and reconstruction; we need both for the loss.
            logits, recon = net(images)
            # Compute the two loss components and combine them with the given lambda.
            l_ce = ce(logits, labels)
            l_rec = recon_loss(recon, images)
            loss = l_ce + lam * l_rec
            # Backpropagate and update weights.
            loss.backward()
            optimizer.step()
            # Accumulate the CE and Rec loss for reporting at the end of the epoch.
            run_ce += l_ce.item()
            run_rec += l_rec.item()
            n_batches += 1

        train_acc = evaluate(net, trainloader, device)
        test_acc = evaluate(net, testloader, device)
        
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["ce"].append(run_ce / n_batches)
        history["rec"].append(run_rec / n_batches)
        print(f"[epoch {epoch + 1:2d}/{epochs}] "
              f"CE {run_ce / n_batches:.3f}  Rec {run_rec / n_batches:7.2f}  "
              f"train_acc {100 * train_acc:5.2f}%  test_acc {100 * test_acc:5.2f}%")
    print("Finished training.")
    return history


def save_accuracy_curves(history, lam, path=None):
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, [100 * a for a in history["train_acc"]], "o-", label="train")
    ax1.plot(epochs, [100 * a for a in history["test_acc"]], "s-", label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy (%)")
    ax1.set_title("Classification accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CE (~1) and raw Rec (~64) live on very different scales, so use twin axes.
    ax2.plot(epochs, history["ce"], "o-", color="tab:red", label="CE loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cross-entropy loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.grid(True, alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(epochs, history["rec"], "s-", color="tab:green", label="Rec loss")
    ax2b.set_ylabel("reconstruction loss", color="tab:green")
    ax2b.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_title(f"Loss components (λ = {lam})")
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right")

    fig.tight_layout()
    path = path or os.path.join(FIG_DIR, "task2_accuracy_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy curves to {path}")


def save_reconstructions(net, loader, device, n=4, path=None):
    """Save n original/reconstruction pairs (top row original, bottom recon)."""
    import matplotlib.pyplot as plt
    net.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    with torch.no_grad():
        _, recon = net(images)

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    for i in range(n):
        axes[0, i].imshow(unnormalize(images[i]).permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(unnormalize(recon[i]).permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", rotation=90, size=12)
    axes[1, 0].set_ylabel("reconstruction", rotation=90, size=12)
    # set_ylabel is hidden by axis('off'); add row titles via text instead
    fig.text(0.02, 0.74, "original", rotation=90, va="center", size=13)
    fig.text(0.02, 0.28, "reconstruction", rotation=90, va="center", size=13)

    fig.tight_layout(rect=(0.04, 0, 1, 1))
    path = path or os.path.join(FIG_DIR, "task2_reconstructions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reconstructions to {path}")


def main():
    # Parse command-line arguments, set up data loaders, model, and training loop.

    parser = argparse.ArgumentParser()
    # Keep cnn_classify's regime (batch 4, SGD lr 1e-3, momentum 0.9) but run more
    # epochs so the accuracy curve has real shape. Decoder added, trained jointly.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lam", type=float, default=0.01,
                        help="lambda weighting the reconstruction loss")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--subset", type=int, default=1,
                        help="keep every Nth image (e.g. 10 for the smaller set)")
    parser.add_argument("--save-model", default="deconv_net.pth")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  lambda = {args.lam}")

    trainloader, testloader = get_loaders(
        batch_size=args.batch_size, subset_factor=args.subset)

    net = DeconvNet().to(device)
    history = train(net, trainloader, testloader, device,
                    epochs=args.epochs, lam=args.lam,
                    lr=args.lr, momentum=args.momentum)

    if args.save_model:
        # Save the model's state_dict, which contains the learned parameters. This is a common PyTorch convention for saving models.
        torch.save(net.state_dict(), args.save_model)
        print(f"Saved model to {args.save_model}")

    final_test = history["test_acc"][-1]
    final_train = history["train_acc"][-1]
    per_class = per_class_accuracy(net, testloader, device)

    print("\n=== Results ===")
    print(f"Final training accuracy: {100 * final_train:.2f}%")
    print(f"Final test accuracy:     {100 * final_test:.2f}%")
    print(f"Lambda:                  {args.lam}")
    print("\nPer-class test accuracy:")
    for cls, acc in per_class.items():
        print(f"  {cls:<8} {100 * acc:5.1f}%")

    save_accuracy_curves(history, args.lam)
    save_reconstructions(net, testloader, device)


if __name__ == "__main__":
    main()

