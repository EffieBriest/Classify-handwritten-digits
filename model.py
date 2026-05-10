import torch
from torch import nn
from pathlib import Path

class DigitRecognizitonModel(nn.Module):
    def __init__(self, output_shape, hidden_dim=64):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = hidden_dim,
                out_channels = hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


        self.conv_block2=nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = hidden_dim,
                out_channels = hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_dim * 7 * 7,
                out_features=output_shape
            )
        )


    def forward(self, x):
        return self.classifier(self.conv_block2(self.conv_block1(x)))
    
    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path)

    def load_model(model_class, path="models/mnist_cnn.pth", output_shape=10, device="cpu"):
        model = model_class(output_shape=output_shape)

        model.load_state_dict(torch.load(path, map_location=device))

        model.to(device)
        model.eval()

        print(f"Model loaded from: {path}")

        return model



