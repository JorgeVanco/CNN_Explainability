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
    save_pdf,
)

# static variables
name: str = "run"
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10
batch_size: Final[int] = 16

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def calculate_saliency_map(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> plt.Figure:
    # Freeze parameters
    freeze_model(model)

    # get a batch
    inputs, targets = next(iter(dataloader))

    max_indices: torch.Tensor = compute_gradients_input(model, inputs, device)

    saliency_map_grid: plt.Figure = show_saliency_map_grid(
        inputs.detach().cpu(), inputs.grad, targets, max_indices, ncol=4
    )
    return saliency_map_grid


def calculate_class_model_visualization(
    model: torch.nn.Module, dataloader: DataLoader
) -> plt.Figure:
    batch, _ = next(iter(dataloader))
    input = torch.zeros(
        (NUM_CLASSES, *batch.shape[1:]), requires_grad=True, device=device
    )

    freeze_model(model)

    optimizer = torch.optim.Adam([input], lr=0.1, weight_decay=0.001)
    for _ in tqdm(range(15000), desc="Calculating Class model visualization"):
        optimizer.zero_grad()
        output = model(input)
        loss = -output[torch.arange(10), torch.arange(10)].mean()
        loss.backward()
        optimizer.step()

    mean_image = calc_mean_image(dataloader).to(device)

    result = input + mean_image

    class_model_visualization_grid: plt.Figure = show_class_model_visualization_grid(
        result.detach().cpu()
    )
    return class_model_visualization_grid


def main() -> None:
    train_data: DataLoader
    val_data: DataLoader
    test_data: DataLoader
    train_data, val_data, test_data = load_data(DATA_PATH, batch_size=batch_size)

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    dir_path = os.path.join("explain", name)
    os.makedirs(dir_path, exist_ok=True)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device)
    print(f"Model test accuracy: {accuracy}")

    # calculate saliency map
    saliency_map: plt.Figure = calculate_saliency_map(model, test_data)
    saliency_map.savefig(f"{dir_path}/saliency_map.png")

    # calculate class model visualization
    class_model_visualization: plt.Figure = calculate_class_model_visualization(
        model, train_data
    )
    class_model_visualization.savefig(f"{dir_path}/class_model_visualization.png")

    save_pdf(
        [saliency_map, class_model_visualization], f"{dir_path}/explainability.pdf"
    )


if __name__ == "__main__":
    main()
