""" CIFAR-10 classification with the tutorial CNN.

Trains the network, then reports everything the assignment asks for:
  - training accuracy
  - test accuracy
  - a figure of sample test images with predicted labels (-> figures/)
  - the per-class accuracy table

Usage:
    py -3 cnn_classify.py --epochs 2
    py -3 cnn_classify.py --epochs 5 --subset 10   # 10x smaller dataset
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data import CLASSES, get_loaders, unnormalize
from model import Net

FIG_DIR = "figures"



def train(net, loader, device, epochs, lr=0.001, momentum=0.9):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader):
            # Move to device, zero gradients, forward pass, compute loss, backward pass, update weights
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 2000 == 0:
                print(f"[epoch {epoch + 1}, batch {i + 1:5d}] " f"loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    print("Finished training.")




def evaluate(net, loader, device):
    """Return (overall_accuracy, per_class_accuracy_dict)."""
    net.eval()
    correct = total = 0
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)
    with torch.no_grad():
        for images, labels in loader:
            # Move to device, forward pass, compute predictions, update counts
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(net(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                class_correct[label] += int(pred == label)
    overall = correct / total
    per_class = {
        CLASSES[i]: (class_correct[i] / class_total[i] if class_total[i] else 0.0)
        for i in range(len(CLASSES))
    }
    return overall, per_class





def save_sample_predictions(net, loader, device, n=8, path=None):
    """Save a grid of test images with their predicted labels."""
    import matplotlib.pyplot as plt
    net.eval()
    # Gather n images across batches (the loader's batch size may be < n).
    imgs, lbls = [], []
    # Loop over the loader until we have at least n images.
    for bx, by in loader:
        imgs.append(bx)
        lbls.append(by)
        if sum(t.size(0) for t in imgs) >= n:
            break
    # Concatenate the images and labels, then take the first n.
    images = torch.cat(imgs)[:n]
    labels = torch.cat(lbls)[:n]
    with torch.no_grad():
        _, preds = torch.max(net(images.to(device)), 1)
    preds = preds.cpu()

    # One subplot per image so each label sits directly under its own picture;
    # green title = correct prediction, red = wrong.
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    # `axes` is either a 2D array (if rows > 1) or a single Axes (if rows == 1), so flatten it to always be a 1D list.
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.9 * rows))
    # `axes` is either a 2D array (if rows > 1) or a single Axes (if rows == 1), so flatten it to always be a 1D list.
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    # Plot the first n images with their predicted and true labels; hide any extra axes if n < rows*cols.
    for i, ax in enumerate(axes):
        if i < n:
            # imshow expects HWC and unnormalize expects CHW, so permute the dimensions to get from CHW to HWC.
            ax.imshow(unnormalize(images[i]).permute(1, 2, 0).numpy())
            correct = preds[i] == labels[i]
            ax.set_title(
                f"true: {CLASSES[labels[i]]}\npred: {CLASSES[preds[i]]}",
                fontsize=10, color=("green" if correct else "red"))
        ax.axis("off")
    fig.tight_layout()
    # Save the figure; default path is figures/classify_sample_predictions.png.
    path = path or os.path.join(FIG_DIR, "classify_sample_predictions.png")
    # Save with a higher DPI and tight bounding box to reduce whitespace around the images.
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample predictions to {path}")



def main():
    # Parse command-line arguments for training configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--subset", type=int, default=1,
                        help="keep every Nth image (e.g. 10 for the smaller set)")
    parser.add_argument("--save-model", default="cifar_net.pth")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_loaders(
        batch_size=args.batch_size, subset_factor=args.subset)

    net = Net().to(device)
    train(net, trainloader, device, epochs=args.epochs)

    if args.save_model:
        # Save the model's state_dict, which contains the learned parameters. This is a common PyTorch convention for saving models.
        torch.save(net.state_dict(), args.save_model)
        print(f"Saved model to {args.save_model}")

    train_acc, _ = evaluate(net, trainloader, device)
    test_acc, per_class = evaluate(net, testloader, device)

    print("\n=== Results ===")
    print(f"Training accuracy: {100 * train_acc:.2f}%")
    print(f"Test accuracy:     {100 * test_acc:.2f}%")
    print("\nPer-class accuracy:")
    print(f"  {'class':<8} accuracy")
    for cls, acc in per_class.items():
        print(f"  {cls:<8} {100 * acc:5.1f}%")

    save_sample_predictions(net, testloader, device)




if __name__ == "__main__":
    main()
