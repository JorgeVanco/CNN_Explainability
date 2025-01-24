import torch
import torch.nn as nn


class MyModel(nn.Module):
    """
    This is the class to construct the model. Only layers defined in
    this script can be used.
    """

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple[int, ...]
    ) -> None:
        """
        This method is the constructor of the model.

        Args:
            input_size: size of the input
            output_size: size of the output
            hidden_sizes: three hidden sizes of the model
        """

        super().__init__()

        self.model: nn.Sequential = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            *[
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(len(hidden_sizes) - 1)
            ],
            nn.Linear(hidden_sizes[-1], output_size)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: input tensor, Dimensions: [batch, channels, height,
                width].

        Returns:
            outputs of the model. Dimensions: [batch, 1].
        """

        return self.model(inputs.view(inputs.shape[0], -1))


class Cifar10ConvModel(nn.Module):
    def __init__(self, jitter=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout2d(p=jitter),  # jitter described in the paper
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            # nn.Dropout2d(p=0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            # nn.Dropout2d(p=0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            # nn.Dropout2d(p=0.25),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
