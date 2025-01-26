# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader

# other libraries
from typing import Final

# own modules
from src.utils import (
    test_step,
    load_data,
    show_saliency_map_grid,
    compute_gradients_input,
    freeze_model,
)

# static variables
DATA_PATH: Final[str] = "data"
batch_size: int = 16

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def calculate_saliency_map(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> None:
    # Freeze parameters
    freeze_model(model)

    # get a batch
    inputs, targets = next(iter(dataloader))

    max_indices = compute_gradients_input(model, inputs, device)

    show_saliency_map_grid(
        inputs.detach().cpu(), inputs.grad, targets, max_indices, ncol=4
    )


def main() -> None:
    train_data, val_data, test_data = load_data(DATA_PATH, batch_size=batch_size)
    name: str = "scheduler_run"

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device)
    print(f"Model test accuracy: {accuracy}")

    # calculate saliency map
    calculate_saliency_map(model, train_data)


if __name__ == "__main__":
    main()
