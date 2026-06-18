"""CNNs for CIFAR-10.

`Net` is the tutorial classifier:
Conv(3->6,5) -> ReLU -> Pool(2) -> Conv(6->16,5) -> ReLU -> Pool(2)
-> FC(400->120) -> FC(120->84) -> FC(84->10).

`DeconvNet` keeps that exact encoder + classifier head and adds a
decoder head that mirrors the encoder to reconstruct the input image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 3x32x32 -> 6x14x14
        x = self.pool(F.relu(self.conv2(x)))   # 6x14x14 -> 16x5x5
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x







# this is the deconvolutional network, which adds a decoder to the classifier
class DeconvNet(nn.Module):
    """Encoder + classifier head (as in Net) + a mirrored decoder head.

    Encoder block = Conv -> ReLU -> MaxPool. The decoder inverts each block in
    reverse order as MaxUnpool -> ConvTranspose -> nonlinearity, reusing the
    pooling indices so MaxUnpool can place values back where MaxPool took them.

    Shapes:  3x32x32 -conv1-> 6x28x28 -pool-> 6x14x14 (z1)
                       -conv2-> 16x10x10 -pool-> 16x5x5 (z2)
    Decoding z2 mirrors this back to 3x32x32.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # --- encoder (identical to Net) ---
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # --- classifier head (almost identical to Net) ---
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        # --- decoder head (mirror of the encoder) ---
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5)   # inverts conv2 (10->14 after unpool)
        self.deconv1 = nn.ConvTranspose2d(6, 3, 5)    # inverts conv1 (28->32)


    def encode(self, x):
        """Run the encoder; return (z1, z2, cache) where cache holds the pool
        indices and pre-pool sizes the decoder needs."""
        x = F.relu(self.conv1(x))            # 6x28x28
        size1 = x.size()
        z1, idx1 = self.pool(x)              # 6x14x14
        x = F.relu(self.conv2(z1))           # 16x10x10
        size2 = x.size()
        z2, idx2 = self.pool(x)              # 16x5x5
        cache = {"idx1": idx1, "idx2": idx2, "size1": size1, "size2": size2}
        return z1, z2, cache


    def classify(self, z2):
        """Return logits from the classifier head, starting from z2."""
        x = z2.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


    def decode(self, z, cache, level=2):
        """Decode back to a 3x32x32 image, starting from z at the given level.

        level=2: z is z2 (16x5x5) -> full decode path.
        level=1: z is z1 (6x14x14) -> only the conv1 block's inverse.
        """
        if level == 2:
            # decode one layer backwards
            x = self.unpool(z, cache["idx2"], output_size=cache["size2"])  # 16x10x10
            x = F.relu(self.deconv2(x))                                    # 6x14x14
            z1 = x
        else:
            z1 = z
        x = self.unpool(z1, cache["idx1"], output_size=cache["size1"])     # 6x28x28
        x = torch.tanh(self.deconv1(x))                                    # 3x32x32
        return x



    def forward(self, x):
        """Return (logits, reconstruction). The reconstruction is always decoded from z2, the output of the last encoder block."""
        _, z2, cache = self.encode(x)
        logits = self.classify(z2)
        recon = self.decode(z2, cache, level=2)
        return logits, recon

