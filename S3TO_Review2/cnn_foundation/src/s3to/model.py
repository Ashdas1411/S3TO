"""CNN model for the S3TO federated-learning experiment."""

import torch
from torch import nn


class MNISTCNN(nn.Module):
    """Small CNN that classifies MNIST digits from 0 to 9."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN().to(device)

    sample_images = torch.randn(8, 1, 28, 28, device=device)

    with torch.no_grad():
        predictions = model(sample_images)

    assert predictions.shape == (8, 10)

    print(f"Device: {device}")
    print(f"Input shape: {tuple(sample_images.shape)}")
    print(f"Output shape: {tuple(predictions.shape)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("CNN forward-pass validation: PASSED")


if __name__ == "__main__":
    main()
