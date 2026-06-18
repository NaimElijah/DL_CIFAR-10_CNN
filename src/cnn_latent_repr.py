""" Latent representation analysis.

For one train image and one test image we isolate individual channels of the
latents and decode them, to see what each channel of the learned representation
captures:

  - z^(1): the conv-1 latent (6 channels, 6x14x14). We keep ONE channel at a
    time, zero the other five, and decode via the conv1-block inverse.
  - z^(2): the conv-2 latent (16 channels, 16x5x5). We pick 3 channels and,
    for each, keep only that channel (zero the other 15) and decode the full path.

Loads the model trained in cnn_deconv (deconv_net.pth).

Usage:
    py -3 cnn_latent_repr.py --train-idx 7 --test-idx 3
"""
import argparse
import os

import torch

from data import CLASSES, get_loaders, unnormalize
from model import DeconvNet

FIG_DIR = "figures"



def isolate_channel(z, c):
    """Return a copy of z with every channel zeroed except channel c."""
    masked = torch.zeros_like(z)
    masked[:, c] = z[:, c]
    return masked



def to_img(t):
    """Tensor (1,3,32,32) or (3,32,32) -> HxWx3 numpy in [0,1] for imshow."""
    # If the input tensor has a batch dimension (1,3,32,32), we remove it to get (3,32,32).
    if t.dim() == 4:
        t = t[0]
    # Unnormalize the tensor to bring pixel values back to the [0,1] range and convert it to a numpy array with shape (H,W,3) for visualization.
    return unnormalize(t).permute(1, 2, 0).numpy()



def main():
    # Parse command-line arguments for train/test image indices and model path.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-idx", type=int, default=7)
    parser.add_argument("--test-idx", type=int, default=3)
    parser.add_argument("--model", default="deconv_net.pth")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    # Create the figures directory if it doesn't exist.
    os.makedirs(FIG_DIR, exist_ok=True)
    # Set the device to GPU if available, otherwise use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DeconvNet().to(device)
    # Load the trained model's state_dict from the specified path and set the model to evaluation mode.
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()

    # Deterministic single images straight from the underlying datasets.
    trainloader, testloader = get_loaders(batch_size=1)
    train_img, train_lbl = trainloader.dataset[args.train_idx]
    test_img, test_lbl = testloader.dataset[args.test_idx]
    # Prepare samples for analysis.
    samples = [
        ("train", train_img, train_lbl),
        ("test", test_img, test_lbl),
    ]

    # Encode each image once; keep z1, z2 and the decode cache (pool indices).
    # decode cache is a dictionary containing the pooling indices and sizes needed for unpooling during decoding.
    encoded = {}
    with torch.no_grad():
        # Iterate over the samples (train and test images) to encode them and store their latent representations and reconstruction information.
        for name, img, lbl in samples:
            # Add a batch dimension to the image tensor and move it to the specified device (GPU or CPU).
            x = img.unsqueeze(0).to(device)
            z1, z2, cache = net.encode(x)
            recon_full = net.decode(z2, cache, level=2)
            encoded[name] = {"x": x, "lbl": lbl, "z1": z1, "z2": z2,
                             "cache": cache, "recon": recon_full}

    n_ch1 = encoded["train"]["z1"].size(1)   # 6
    n_ch2 = encoded["train"]["z2"].size(1)    # 16

    # Pick the 3 most-active z^(2) channels (averaged over both images), and use
    # the SAME indices for both so each column is comparable across images.
    act = sum(encoded[n]["z2"].abs().mean(dim=(0, 2, 3)) for n in encoded) / 2
    top3 = torch.topk(act, 3).indices.tolist()
    print(f"z^(1): {n_ch1} channels  |  z^(2): {n_ch2} channels, "
          f"showing top-3 by activation = {top3}")



    # ---- Figure 1: z^(1) single-channel reconstructions (all 6 channels) ----
    cols = 2 + n_ch1   # original, full recon, then each channel
    fig, axes = plt.subplots(2, cols, figsize=(1.7 * cols, 4.2))
    for r, (name, _, _) in enumerate(samples):
        e = encoded[name]
        axes[r, 0].imshow(to_img(e["x"]))
        axes[r, 1].imshow(to_img(e["recon"]))
        with torch.no_grad():
            # For each channel in z^(1), isolate that channel and decode it using the conv1-block inverse. This allows us to visualize what each individual channel of the first convolutional layer captures in terms of image features.
            for c in range(n_ch1):
                # Isolate channel c of z^(1) and decode it using the conv1-block inverse. The decode function takes the isolated latent representation and reconstructs an image from it, allowing us to see what features that specific channel is capturing.
                dec = net.decode(isolate_channel(e["z1"], c), e["cache"], level=1)
                axes[r, 2 + c].imshow(to_img(dec))
        # Hide the axes ticks for a cleaner visualization and set the y-axis label to indicate whether the image is from the training or test set, along with its true class label.
        for ax in axes[r]:
            # Hide the axes ticks for a cleaner visualization.
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"{name}\n({CLASSES[e['lbl']]})", size=11)
    titles = ["original", "full recon"] + [f"z¹ ch {c}" for c in range(n_ch1)]
    # Set the titles for each column in the figure to indicate what is being displayed: the original image, the full reconstruction, and the individual channel reconstructions for z^(1).
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, size=10)
    fig.suptitle("z⁽¹⁾ (conv-1): single-channel reconstructions", size=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = os.path.join(FIG_DIR, "task3_z1_channels.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")



    # ---- Figure 2: z^(2) single-channel reconstructions (3 channels) ----
    cols = 2 + len(top3)
    fig, axes = plt.subplots(2, cols, figsize=(1.9 * cols, 4.2))
    # For each sample (train and test images), we isolate the top 3 most active channels of z^(2) and decode them to visualize what features these channels capture in the latent representation. The figure will show the original image, the full reconstruction, and the reconstructions from each of the top 3 channels of z^(2).
    for r, (name, _, _) in enumerate(samples):
        # Retrieve the encoded representation for the current sample (train or test image) from the encoded dictionary. This includes the original image, its label, the latent representations z1 and z2, the cache for decoding, and the full reconstruction.
        e = encoded[name]
        axes[r, 0].imshow(to_img(e["x"]))
        axes[r, 1].imshow(to_img(e["recon"]))
        with torch.no_grad():
            # For each of the top 3 most active channels in z^(2), we isolate that channel and decode it using the full decoder path. This allows us to visualize what features each of these channels captures in the latent representation, providing insight into how the model processes and represents the input images.
            for j, c in enumerate(top3):
                # Isolate channel c of z^(2) and decode it using the full decoder path. The decode function takes the isolated latent representation and reconstructs an image from it, allowing us to see what features that specific channel is capturing in terms of image features.
                dec = net.decode(isolate_channel(e["z2"], c), e["cache"], level=2)
                axes[r, 2 + j].imshow(to_img(dec))
        # Hide the axes ticks for a cleaner visualization and set the y-axis label to indicate whether the image is from the training or test set, along with its true class label.
        for ax in axes[r]:
            # Hide the axes ticks for a cleaner visualization.
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"{name}\n({CLASSES[e['lbl']]})", size=11)
    titles = ["original", "full recon"] + [f"z² ch {c}" for c in top3]
    # Set the titles for each column in the figure to indicate what is being displayed: the original image, the full reconstruction, and the individual channel reconstructions for z^(2).
    for c, t in enumerate(titles):
        # Set the title for each column in the figure to indicate what is being displayed: the original image, the full reconstruction, and the individual channel reconstructions for z^(2).
        axes[0, c].set_title(t, size=10)
    fig.suptitle("z⁽²⁾ (conv-2): single-channel reconstructions", size=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p2 = os.path.join(FIG_DIR, "task3_z2_channels.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p2}")

    print(f"\nTrain image: idx {args.train_idx} ({CLASSES[train_lbl]})  |  "
          f"Test image: idx {args.test_idx} ({CLASSES[test_lbl]})")


if __name__ == "__main__":
    main()
