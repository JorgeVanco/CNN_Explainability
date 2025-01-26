# deep learning libraries
import os
from matplotlib import pyplot as plt
import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader

# other libraries
from typing import Final

from tqdm import tqdm

# own modules
from src.utils import (
    test_step,
    load_data,
    show_saliency_map_grid,
    compute_gradients_input,
    freeze_model,
    calc_mean_image,
    show_class_model_visualization_grid,
)

# static variables
DATA_PATH: Final[str] = "data"
batch_size: int = 16
NUM_CLASSES = 10

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

    saliency_map_grid = show_saliency_map_grid(
        inputs.detach().cpu(), inputs.grad, targets, max_indices, ncol=4
    )
    return saliency_map_grid


def calculate_class_model_visualization(
    model: torch.nn.Module, dataloader: DataLoader
) -> None:
    batch, _ = next(iter(dataloader))
    input = torch.zeros(
        (NUM_CLASSES, *batch.shape[1:]), requires_grad=True, device=device
    )

    freeze_model(model)

    optimizer = torch.optim.Adam([input], lr=0.05, weight_decay=0.05)
    for _ in tqdm(range(15000), desc="Calculating Class model visualization"):
        optimizer.zero_grad()
        output = model(input)
        loss = -output[torch.arange(10), torch.arange(10)].mean()
        loss.backward()
        optimizer.step()

    mean_image = calc_mean_image(dataloader).to(device)

    result = input + mean_image

    class_model_visualization_grid = show_class_model_visualization_grid(
        result.detach().cpu()
    )
    return class_model_visualization_grid


def main() -> None:
    train_data, val_data, test_data = load_data(DATA_PATH, batch_size=batch_size)
    name: str = "scheduler_run"

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    dir_path = os.path.join("explain", name)
    os.makedirs(dir_path, exist_ok=True)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device)
    print(f"Model test accuracy: {accuracy}")

    # calculate saliency map
    saliency_map = calculate_saliency_map(model, train_data)
    saliency_map.savefig(f"{dir_path}/saliency_map.png")

    # calculate class model visualization
    class_model_visualization = calculate_class_model_visualization(model, train_data)
    class_model_visualization.savefig(f"{dir_path}/class_model_visualization.png")


if __name__ == "__main__":
    main()
